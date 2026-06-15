use gl::types::*;
use glfw::{Action, Context, InitHint, Key, MouseButton, WindowEvent};
use nalgebra_glm as glm;
use std::collections::HashMap;
use std::{
    ffi::CStr,
    fs::File,
    io::{BufRead, BufReader},
    mem::size_of,
    path::Path,
};

pub struct Orientation {
    pub u: glm::Vec3,
    pub v: glm::Vec3,
    pub w: glm::Vec3,
    pub pos: glm::Vec3,
}

impl Orientation {
    pub fn cam() -> Self {
        Orientation {
            u: glm::Vec3::new(1.0, 0.0, 0.0),
            v: glm::Vec3::new(0.0, 1.0, 0.0),
            w: glm::Vec3::new(0.0, 0.0, 1.0),
            pos: glm::Vec3::new(0.0, 0.0, 0.0),
        }
    }
    pub fn rotate(&mut self, angles: glm::Vec3) {
        if angles.x != 0.0 {
            let rotation_quat = glm::quat_angle_axis(angles.x, &glm::Vec3::new(0.0, 1.0, 0.0));
            self.w = glm::quat_rotate_vec3(&rotation_quat, &self.w);
            self.u = glm::quat_rotate_vec3(&rotation_quat, &self.u);
            self.v = glm::quat_rotate_vec3(&rotation_quat, &self.v);
        }
        if angles.y != 0.0 {
            let rotation_quat = glm::quat_angle_axis(angles.y, &self.u);
            self.w = glm::quat_rotate_vec3(&rotation_quat, &self.w);
            self.v = glm::quat_rotate_vec3(&rotation_quat, &self.v);
        }
        if angles.z != 0.0 {
            let rotation_quat = glm::quat_angle_axis(angles.z, &self.w);
            self.u = glm::quat_rotate_vec3(&rotation_quat, &self.u);
            self.v = glm::quat_rotate_vec3(&rotation_quat, &self.v);
        }
        self.w = glm::normalize(&self.w);
        self.u = glm::normalize(&glm::cross(&self.v, &self.w));
        self.v = glm::normalize(&glm::cross(&self.w, &self.u));
    }
}

pub struct GlState {
    pub window_dims: (i32, i32),
    pub window_pos: (i32, i32),
    pub frame_dims: (i32, i32),
    pub window_name: String,
    pub cam_state: Orientation,
    pub window: glfw::PWindow,
    pub events: glfw::GlfwReceiver<(f64, glfw::WindowEvent)>,
    pub wasdsp_pressed: [bool; 5],
    pub left_click: bool,
    pub last_cursor_pos: (f32, f32),
    pub def_vshader_id: GLuint,
    pub def_fshader_id: GLuint,
    pub def_pipeline: GLuint,
    pub offbo: GLuint,
    pub offtex: GLuint,
    pub blurtex: GLuint,
    pub off_depth_tex: GLuint,
    pub fovy: f32,
    pub near: f32,
    pub far: f32,
    pub clear_color: [GLfloat; 4],
    pub clear_depth: GLfloat,
}

pub fn setup_glfw() -> (glfw::Glfw, glfw::PWindow, glfw::GlfwReceiver<(f64, glfw::WindowEvent)>) {
    // Initialize glfw
    glfw::init_hint(InitHint::Platform(glfw::Platform::Win32));
    let mut glfw = glfw::init(glfw_error_callback).unwrap();

    // Specify OpenGL version
    glfw.window_hint(glfw::WindowHint::OpenGlProfile(glfw::OpenGlProfileHint::Core));
    glfw.window_hint(glfw::WindowHint::ContextVersionMajor(4));
    glfw.window_hint(glfw::WindowHint::ContextVersionMinor(6));
    glfw.window_hint(glfw::WindowHint::OpenGlDebugContext(true));

    // Request debug context

    let (mut window, events) = glfw
        .create_window(500, 500, "water", glfw::WindowMode::Windowed)
        .expect("Failed to create GLFW window.");

    window.make_current();
    window.set_key_polling(true);
    window.set_framebuffer_size_polling(true);
    window.set_mouse_button_polling(true);
    window.set_cursor_pos_polling(true);
    // Load OpenGL functions
    gl::load_with(|symbol| window.get_proc_address(symbol).map_or(std::ptr::null(), |p| p as *const _));
    // Set debug context
    unsafe {
        // Callback is called immediately when an error happens
        gl::Enable(gl::DEBUG_OUTPUT_SYNCHRONOUS);
        gl::DebugMessageCallback(Some(gl_debug_output), std::ptr::null());
        gl::DebugMessageControl(
            gl::DONT_CARE,
            gl::DONT_CARE,
            gl::DONT_CARE,
            0,
            std::ptr::null(),
            gl::TRUE,
        );
    }

    (glfw, window, events)
}

pub struct Mesh {
    pub vao_id: GLuint,
    pub nindices: GLsizei,
}

pub fn init_mesh(path: &str) -> Mesh {
    let file = File::open(path).expect(&format!("Could not open {}", path));
    let reader = BufReader::new(file);
    let mut v: Vec<(f32, f32, f32)> = Vec::new();
    let mut vn: Vec<(f32, f32, f32)> = Vec::new();
    let mut vt: Vec<(f32, f32)> = Vec::new();
    let mut indices: Vec<(i32, i32, i32)> = Vec::new();
    let mut element_indices: Vec<i32> = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let mut itr = line.split(' ');
        itr.next();
        if line.starts_with("vt") {
            let u: f32 = itr.next().unwrap().parse().unwrap();
            let v_val: f32 = itr.next().unwrap().parse().unwrap();
            vt.push((u, v_val));
        } else if line.starts_with("vn") {
            let x: f32 = itr.next().unwrap().parse().unwrap();
            let y: f32 = itr.next().unwrap().parse().unwrap();
            let z: f32 = itr.next().unwrap().parse().unwrap();
            vn.push((x, y, z));
        } else if line.starts_with("v") {
            let x: f32 = itr.next().unwrap().parse().unwrap();
            let y: f32 = itr.next().unwrap().parse().unwrap();
            let z: f32 = itr.next().unwrap().parse().unwrap();
            v.push((x, y, z));
        } else if line.starts_with("f") {
            for _ in 0..3 {
                let face_data = itr.next().unwrap();
                let mut face_itr = face_data.split('/');
                let vi: i32 = face_itr.next().unwrap().parse().unwrap();
                let vti: i32 = face_itr.next().unwrap().parse().unwrap();
                let vni: i32 = face_itr.next().unwrap().parse().unwrap();
                indices.push((vi, vti, vni));
            }
        }
    }

    let mut vertices: Vec<f32> = Vec::new();
    let mut triple_to_index: HashMap<(i32, i32, i32), i32> = HashMap::new();
    let mut next_index: i32 = 0;

    for (vi, vti, vni) in indices {
        let triple = (vi, vti, vni);
        let idx = if let Some(&idx) = triple_to_index.get(&triple) {
            idx
        } else {
            let pos = v[(vi - 1) as usize];
            let normal = vn[(vni - 1) as usize];
            let uv = vt[(vti - 1) as usize];
            vertices.push(pos.0);
            vertices.push(pos.1);
            vertices.push(pos.2);
            vertices.push(normal.0);
            vertices.push(normal.1);
            vertices.push(normal.2);
            vertices.push(uv.0);
            vertices.push(uv.1);
            triple_to_index.insert(triple, next_index);
            let assigned = next_index;
            next_index += 1;
            assigned
        };
        element_indices.push(idx);
    }

    let (mut vao_id, mut vbo_id, mut veo_id): (GLuint, GLuint, GLuint) = (0, 0, 0);
    unsafe {
        gl::CreateVertexArrays(1, &mut vao_id);
        gl::CreateBuffers(1, &mut vbo_id);
        gl::CreateBuffers(1, &mut veo_id);

        // Load indices data veo
        gl::NamedBufferData(
            veo_id,
            (element_indices.len() * size_of::<f32>()) as GLsizeiptr,
            element_indices.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW,
        );
        gl::NamedBufferData(
            vbo_id,
            (vertices.len() * size_of::<f32>()) as GLsizeiptr,
            vertices.as_ptr() as *const GLvoid,
            gl::STATIC_DRAW,
        );

        // Interleaved
        gl::VertexArrayVertexBuffer(vao_id, 0, vbo_id, 0, (size_of::<f32>() * 8) as GLsizei);
        gl::EnableVertexArrayAttrib(vao_id, 0);
        gl::VertexArrayAttribFormat(vao_id, 0, 3, gl::FLOAT, gl::FALSE, 0);
        // Normal (tightly packed vec3)
        gl::VertexArrayVertexBuffer(vao_id, 1, vbo_id, 0, (size_of::<f32>() * 8) as GLsizei);
        gl::EnableVertexArrayAttrib(vao_id, 1);
        gl::VertexArrayAttribFormat(vao_id, 1, 3, gl::FLOAT, gl::FALSE, (size_of::<f32>() * 3) as GLuint);
        // UV
        gl::VertexArrayVertexBuffer(vao_id, 2, vbo_id, 0, (size_of::<f32>() * 8) as GLsizei);
        gl::EnableVertexArrayAttrib(vao_id, 2);
        gl::VertexArrayAttribFormat(vao_id, 2, 2, gl::FLOAT, gl::FALSE, (size_of::<f32>() * 6) as GLuint);

        gl::VertexArrayAttribBinding(vao_id, 0, 0);
        gl::VertexArrayAttribBinding(vao_id, 1, 1);
        gl::VertexArrayAttribBinding(vao_id, 2, 2);

        // Attach element array buffer to vao
        gl::VertexArrayElementBuffer(vao_id, veo_id);

        // Label for testing
        gl::ObjectLabel(gl::BUFFER, vbo_id, 3, "vbo".as_ptr() as *const i8);
        gl::ObjectLabel(gl::BUFFER, veo_id, 3, "veo".as_ptr() as *const i8);
        gl::ObjectLabel(gl::VERTEX_ARRAY, vao_id, 3, "vao".as_ptr() as *const i8);
    }
    Mesh {
        vao_id: vao_id,
        nindices: element_indices.len() as GLsizei,
    }
}

pub fn load_shader(path: &str) -> GLuint {
    let src = std::fs::read_to_string(path).expect(&format!("Failed to read shader: {}", path));

    unsafe {
        let shader_type = if path.ends_with(".vert") {
            gl::VERTEX_SHADER
        } else if path.ends_with(".frag") {
            gl::FRAGMENT_SHADER
        } else {
            panic!("Unknown shader type for path: {}", path);
        };

        let shader_id = gl::CreateShader(shader_type);
        let ptr = src.as_ptr() as *const i8;
        let len = src.len() as GLint;
        gl::ShaderSource(shader_id, 1, &ptr, &len);
        gl::CompileShader(shader_id);

        let mut is_compiled: GLint = gl::FALSE as GLint;
        gl::GetShaderiv(shader_id, gl::COMPILE_STATUS, &mut is_compiled);
        if is_compiled == gl::FALSE as GLint {
            eprintln!("Unable to compile shader \"{}\"", path);

            let mut err_len: GLint = 0;
            gl::GetShaderiv(shader_id, gl::INFO_LOG_LENGTH, &mut err_len);
            let mut err_log: Vec<i8> = vec![0; err_len as usize + 1];
            gl::GetShaderInfoLog(shader_id, err_len, &mut err_len, err_log.as_mut_ptr());

            let err_str = CStr::from_ptr(err_log.as_ptr()).to_string_lossy();
            eprintln!("{}", err_str);

            std::process::exit(1);
        }

        let program_id = gl::CreateProgram();
        gl::ProgramParameteri(program_id, gl::PROGRAM_SEPARABLE, gl::TRUE as GLint);
        gl::AttachShader(program_id, shader_id);
        gl::LinkProgram(program_id);

        let mut is_linked: GLint = gl::FALSE as GLint;
        gl::GetProgramiv(program_id, gl::LINK_STATUS, &mut is_linked);
        if is_linked == gl::FALSE as GLint {
            eprintln!("Unable to link shader \"{}\"", path);

            let mut err_len: GLint = 0;
            gl::GetProgramiv(program_id, gl::INFO_LOG_LENGTH, &mut err_len);
            let mut err_log: Vec<i8> = vec![0; err_len as usize + 1];
            gl::GetProgramInfoLog(program_id, err_len, &mut err_len, err_log.as_mut_ptr());

            let err_str = CStr::from_ptr(err_log.as_ptr()).to_string_lossy();
            eprintln!("{}", err_str);

            std::process::exit(1);
        }

        gl::DetachShader(program_id, shader_id);
        gl::DeleteShader(shader_id);

        program_id
    }
}

pub fn callbacks(state: &mut GlState) {
    for (_, event) in glfw::flush_messages(&state.events) {
        match event {
            WindowEvent::FramebufferSize(x, y) => {
                state.frame_dims = (x, y);
                unsafe {
                    gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.offtex);
                    gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.off_depth_tex);
                    gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.blurtex);

                    gl::TextureParameteri(state.offtex, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
                    gl::TextureParameteri(state.offtex, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
                    gl::TextureParameteri(state.offtex, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
                    gl::TextureParameteri(state.offtex, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

                    gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.off_depth_tex);
                    gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
                    gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
                    gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
                    gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

                    gl::TextureStorage2D(state.offtex, 1, gl::RGBA32F, state.frame_dims.0, state.frame_dims.1);
                    gl::TextureStorage2D(
                        state.off_depth_tex,
                        1,
                        gl::DEPTH_COMPONENT32F,
                        state.frame_dims.0,
                        state.frame_dims.1,
                    );
                    gl::TextureStorage2D(state.blurtex, 1, gl::RGBA32F, state.frame_dims.0, state.frame_dims.1);

                    gl::NamedFramebufferTexture(state.offbo, gl::COLOR_ATTACHMENT0, state.offtex, 0);
                    gl::NamedFramebufferTexture(state.offbo, gl::DEPTH_ATTACHMENT, state.off_depth_tex, 0);
                }
            }
            WindowEvent::Key(Key::W, _, Action::Press, _) => {
                state.wasdsp_pressed[0] = true;
            }
            WindowEvent::Key(Key::W, _, Action::Release, _) => {
                state.wasdsp_pressed[0] = false;
            }
            WindowEvent::Key(Key::A, _, Action::Press, _) => {
                state.wasdsp_pressed[1] = true;
            }
            WindowEvent::Key(Key::A, _, Action::Release, _) => {
                state.wasdsp_pressed[1] = false;
            }
            WindowEvent::Key(Key::S, _, Action::Press, _) => {
                state.wasdsp_pressed[2] = true;
            }
            WindowEvent::Key(Key::S, _, Action::Release, _) => {
                state.wasdsp_pressed[2] = false;
            }
            WindowEvent::Key(Key::D, _, Action::Press, _) => {
                state.wasdsp_pressed[3] = true;
            }
            WindowEvent::Key(Key::D, _, Action::Release, _) => {
                state.wasdsp_pressed[3] = false;
            }
            WindowEvent::Key(Key::Space, _, Action::Press, _) => {
                state.wasdsp_pressed[4] = true;
            }
            WindowEvent::Key(Key::Space, _, Action::Release, _) => {
                state.wasdsp_pressed[4] = false;
            }
            WindowEvent::CursorPos(x, y) => {
                let rot_rate = 0.01;
                let (dx, dy) = (state.last_cursor_pos.0 - x as f32, state.last_cursor_pos.1 - y as f32);
                state.last_cursor_pos = (x as f32, y as f32);
                if state.left_click {
                    state
                        .cam_state
                        .rotate(glm::Vec3::new(dx * rot_rate, dy * rot_rate, 0.0));
                }
            }
            WindowEvent::MouseButton(MouseButton::Button1, Action::Press, _) => {
                state.left_click = true;
            }
            WindowEvent::MouseButton(MouseButton::Button1, Action::Release, _) => {
                state.left_click = false;
            }
            _ => {}
        }
    }
}

pub fn move_cam(state: &mut GlState) {
    let move_rate = 0.08;
    if state.wasdsp_pressed[0] {
        state.cam_state.pos -= state.cam_state.w * move_rate;
    }
    if state.wasdsp_pressed[1] {
        state.cam_state.pos -= state.cam_state.u * move_rate;
    }
    if state.wasdsp_pressed[2] {
        state.cam_state.pos += state.cam_state.w * move_rate;
    }
    if state.wasdsp_pressed[3] {
        state.cam_state.pos += state.cam_state.u * move_rate;
    }
    if state.wasdsp_pressed[4] {
        state.cam_state.pos += glm::Vec3::new(0.0, 1.0, 0.0) * move_rate;
    }
}

pub trait Render {
    fn draw(&mut self, state: &GlState);
}

fn glfw_error_callback(err: glfw::Error, description: String) {
    println!("GLFW error {:?}: {:?}", err, description);
}

extern "system" fn gl_debug_output(
    source: GLenum,
    gltype: GLenum,
    id: GLuint,
    severity: GLenum,
    _length: GLsizei,
    message: *const GLchar,
    _user_param: *mut std::ffi::c_void,
) {
    // Ignore non-significant error/warning codes
    // TODO: change these magic numbers
    if id == 131169 || id == 131185 || id == 131218 || id == 131204 {
        return;
    }

    let message_str = unsafe { CStr::from_ptr(message).to_string_lossy() };

    println!("---------------");
    println!("Debug message ({}): {}", id, message_str);

    match source {
        gl::DEBUG_SOURCE_API => print!("Source: API"),
        gl::DEBUG_SOURCE_WINDOW_SYSTEM => print!("Source: Window System"),
        gl::DEBUG_SOURCE_SHADER_COMPILER => print!("Source: Shader Compiler"),
        gl::DEBUG_SOURCE_THIRD_PARTY => print!("Source: Third Party"),
        gl::DEBUG_SOURCE_APPLICATION => print!("Source: Application"),
        gl::DEBUG_SOURCE_OTHER => print!("Source: Other"),
        _ => print!("Source: Unknown"),
    }
    println!();

    match gltype {
        gl::DEBUG_TYPE_ERROR => print!("Type: Error"),
        gl::DEBUG_TYPE_DEPRECATED_BEHAVIOR => print!("Type: Deprecated Behaviour"),
        gl::DEBUG_TYPE_UNDEFINED_BEHAVIOR => print!("Type: Undefined Behaviour"),
        gl::DEBUG_TYPE_PORTABILITY => print!("Type: Portability"),
        gl::DEBUG_TYPE_PERFORMANCE => print!("Type: Performance"),
        gl::DEBUG_TYPE_MARKER => print!("Type: Marker"),
        gl::DEBUG_TYPE_PUSH_GROUP => print!("Type: Push Group"),
        gl::DEBUG_TYPE_POP_GROUP => print!("Type: Pop Group"),
        gl::DEBUG_TYPE_OTHER => print!("Type: Other"),
        _ => print!("Type: Unknown"),
    }
    println!();

    match severity {
        gl::DEBUG_SEVERITY_HIGH => println!("Severity: high"),
        gl::DEBUG_SEVERITY_MEDIUM => println!("Severity: medium"),
        gl::DEBUG_SEVERITY_LOW => println!("Severity: low"),
        gl::DEBUG_SEVERITY_NOTIFICATION => println!("Severity: notification"),
        _ => println!("Severity: unknown"),
    }
}
