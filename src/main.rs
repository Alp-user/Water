mod util;
use gl::types::*;
use glfw::{Action, Context, Key};
use nalgebra_glm as glm;

fn main() {
    let (mut glfw, window, events) = util::setup_glfw();
    let clear_color: [GLfloat; 4] = [1.0, 1.0, 1.0, 1.0];
    let clear_depth: GLfloat = 1.0;
    let tri_mesh = util::init_mesh("meshes/sphere.obj");
    let mut state = util::GlState {
        window_dims: (500, 500),
        frame_dims: (500, 500),
        window_pos: (0, 0),
        window_name: String::from("water"),
        cam_state: util::Orientation::cam(),
        window: window,
        events: events,
        wasdsp_pressed: [false; 5],
        left_click: false,
        last_cursor_pos: (0.0, 0.0),
    };
    state.cam_state.pos = glm::Vec3::new(0.0, 0.0, 10.0);
    let cursor_pos = state.window.get_cursor_pos();
    state.last_cursor_pos = (cursor_pos.0 as f32, cursor_pos.1 as f32);

    // Program Pipeline
    let mut pipeline: GLuint = 0;
    let (vshader_id, fshader_id): (GLuint, GLuint) = (
        util::load_shader("shaders/regular.vert"),
        util::load_shader("shaders/regular.frag"),
    );
    unsafe {
        // Create and bind pipeline
        gl::CreateProgramPipelines(1, &mut pipeline as *mut u32);
        gl::BindProgramPipeline(pipeline);
        gl::UseProgramStages(pipeline, gl::VERTEX_SHADER_BIT, vshader_id);
        gl::UseProgramStages(pipeline, gl::FRAGMENT_SHADER_BIT, fshader_id);
        // Set uniforms
        gl::Disable(gl::CULL_FACE);
        gl::FrontFace(gl::CCW);
    }
    while !state.window.should_close() {
        unsafe {
            // Clear color attachment of framebuffer
            gl::ClearNamedFramebufferfv(0, gl::COLOR, 0, clear_color.as_ptr());
            // Clear depth attachment of framebuffer
            gl::ClearNamedFramebufferfv(0, gl::DEPTH, 0, &clear_depth);
            // Callbacks
            util::callbacks(&mut state);
            util::move_cam(&mut state);
            // Update matrices
            let model: glm::Mat4x4 = glm::scale(&glm::identity(), &glm::Vec3::new(1.0, 1.0, 1.0));
            let view: glm::Mat4x4 = glm::look_at(
                &state.cam_state.pos,
                &(state.cam_state.pos - state.cam_state.w),
                &state.cam_state.v,
            );
            let aspect_ratio = (state.frame_dims.0 as f32) / (state.frame_dims.1 as f32);
            let projection = glm::perspective(aspect_ratio, 50.0_f32.to_radians(), 0.1, 100.0);
            let normal: glm::Mat3 = glm::inverse_transpose(glm::mat4_to_mat3(&view));

            gl::ProgramUniformMatrix4fv(
                vshader_id,
                0,
                1,
                gl::FALSE,
                glm::value_ptr(&model).as_ptr(),
            );
            gl::ProgramUniformMatrix4fv(
                vshader_id,
                1,
                1,
                gl::FALSE,
                glm::value_ptr(&view).as_ptr(),
            );
            gl::ProgramUniformMatrix4fv(
                vshader_id,
                2,
                1,
                gl::FALSE,
                glm::value_ptr(&projection).as_ptr(),
            );
            gl::ProgramUniformMatrix3fv(
                vshader_id,
                3,
                1,
                gl::FALSE,
                glm::value_ptr(&normal).as_ptr(),
            );
            gl::ProgramUniform3f(
                fshader_id,
                0,
                state.cam_state.pos.x,
                state.cam_state.pos.y,
                state.cam_state.pos.z,
            );

            // Set viewport
            gl::Viewport(0, 0, state.frame_dims.0, state.frame_dims.1);
            gl::BindVertexArray(tri_mesh.vao_id);
            gl::UseProgramStages(pipeline, gl::VERTEX_SHADER_BIT, vshader_id);
            gl::UseProgramStages(pipeline, gl::FRAGMENT_SHADER_BIT, fshader_id);
            gl::DrawElements(
                gl::TRIANGLES,
                tri_mesh.nindices,
                gl::UNSIGNED_INT,
                std::ptr::null(),
            );

            glfw.poll_events();
            state.window.swap_buffers();
        }
    }
}
