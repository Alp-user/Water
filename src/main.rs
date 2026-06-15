mod util;
mod water;
use gl::types::*;
use glfw::{Action, Context, Key};
use nalgebra_glm as glm;
use std::thread;
use std::time::{Duration, Instant};
use util::Render;
use water::Water;

fn main() {
    let (mut glfw, window, events) = util::setup_glfw();
    let tri_mesh = util::init_mesh("meshes/sphere.obj");
    let (vshader_id, fshader_id): (GLuint, GLuint) = (
        util::load_shader("shaders/regular.vert"),
        util::load_shader("shaders/regular.frag"),
    );
    // Program Pipeline
    let mut pipeline: GLuint = 0;
    unsafe {
        // Create and bind pipeline
        gl::CreateProgramPipelines(1, &mut pipeline as *mut u32);
        gl::BindProgramPipeline(pipeline);
        gl::UseProgramStages(pipeline, gl::VERTEX_SHADER_BIT, vshader_id);
        gl::UseProgramStages(pipeline, gl::FRAGMENT_SHADER_BIT, fshader_id);
        // Set uniforms
        gl::Disable(gl::CULL_FACE);
        gl::Enable(gl::DEPTH_TEST);
        gl::Enable(gl::BLEND);
        gl::BlendFunc(gl::SRC_ALPHA, gl::ONE_MINUS_SRC_ALPHA);
        gl::FrontFace(gl::CCW);
    }

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
        def_vshader_id: vshader_id,
        def_fshader_id: fshader_id,
        def_pipeline: pipeline,
        offbo: 0,
        offtex: 0,
        blurtex: 0,
        off_depth_tex: 0,
        fovy: 50.0_f32.to_radians(),
        near: 0.1,
        far: 100.0,
        clear_color: [1.0, 1.0, 1.0, 1.0],
        clear_depth: 1.0,
    };
    state.cam_state.pos = glm::Vec3::new(0.0, 0.0, 10.0);
    let cursor_pos = state.window.get_cursor_pos();
    state.last_cursor_pos = (cursor_pos.0 as f32, cursor_pos.1 as f32);

    unsafe {
        // Create offscreen framebuffer
        gl::CreateFramebuffers(1, &mut state.offbo);
        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.offtex);
        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.blurtex);

        gl::TextureParameteri(state.offtex, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TextureParameteri(state.offtex, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TextureParameteri(state.offtex, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
        gl::TextureParameteri(state.offtex, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

        gl::TextureParameteri(state.blurtex, gl::TEXTURE_MIN_FILTER, gl::LINEAR as i32);
        gl::TextureParameteri(state.blurtex, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TextureParameteri(state.blurtex, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
        gl::TextureParameteri(state.blurtex, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.off_depth_tex);
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_MIN_FILTER, gl::NEAREST as i32);
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_MAG_FILTER, gl::NEAREST as i32);
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_WRAP_S, gl::REPEAT as i32);
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_WRAP_T, gl::REPEAT as i32);

        gl::TextureStorage2D(
            state.offtex,
            1, // 1 level (no mipmaps)
            gl::RGBA32F,
            state.frame_dims.0,
            state.frame_dims.1,
        );
        gl::TextureStorage2D(
            state.blurtex,
            1, // 1 level (no mipmaps)
            gl::RGBA32F,
            state.frame_dims.0,
            state.frame_dims.1,
        );
        gl::TextureStorage2D(
            state.off_depth_tex,
            1, // 1 level
            gl::DEPTH_COMPONENT32F,
            state.frame_dims.0,
            state.frame_dims.1,
        );

        gl::NamedFramebufferTexture(state.offbo, gl::COLOR_ATTACHMENT0, state.offtex, 0);
        gl::NamedFramebufferTexture(state.offbo, gl::DEPTH_ATTACHMENT, state.off_depth_tex, 0);
        gl::ObjectLabel(gl::TEXTURE, state.offtex, 3, "tex".as_ptr() as *const i8);
        gl::ObjectLabel(gl::TEXTURE, state.off_depth_tex, 3, "dep".as_ptr() as *const i8);
    }

    let mut time = Instant::now();
    let mut accumulator = 0.0;
    let mut owater = Water::new((5, 70, 5), 0.1);
    while !state.window.should_close() {
        let time_now = Instant::now();
        let time_elapsed = (time_now - time).as_secs_f32();
        time = time_now;
        accumulator += time_elapsed;
        while accumulator > owater.dt {
            accumulator -= owater.dt;
            owater.init_simulation();
            owater.load_grid();
            owater.load_densities();
            owater.load_gravity();
            owater.load_pressure();
            owater.load_viscosity();
            owater.load_surface_tension();
            owater.simulate();
            owater.load_offsets();
        }
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            owater.draw(&state);

            gl::Disable(gl::DEPTH_TEST);
            gl::BindFramebuffer(gl::FRAMEBUFFER, state.offbo);
            gl::NamedFramebufferTexture(state.offbo, gl::COLOR_ATTACHMENT0, state.blurtex, 0);
            gl::NamedFramebufferTexture(state.offbo, gl::DEPTH_ATTACHMENT, state.off_depth_tex, 0);
            gl::ClearNamedFramebufferfv(state.offbo, gl::COLOR, 0, state.clear_color.as_ptr());
            gl::ClearNamedFramebufferfv(state.offbo, gl::DEPTH, 0, &state.clear_depth);
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, state.offtex);
            gl::ProgramUniform1i(state.def_vshader_id, 5, 0);
            gl::ProgramUniform1i(state.def_fshader_id, 1, 1);
            gl::ProgramUniform1f(
                state.def_fshader_id,
                5,
                (state.frame_dims.1) as f32 / (2.0 * f32::tan(state.fovy / 2.0)),
            );
            gl::ProgramUniform1f(state.def_fshader_id, 6, owater.world_blur_radius);
            gl::ProgramUniform2f(
                state.def_fshader_id,
                2,
                1.0 / state.frame_dims.0 as f32,
                1.0 / state.frame_dims.1 as f32,
            );
            gl::DrawArrays(gl::TRIANGLES, 0, 3); // blur x pass

            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, state.blurtex);
            gl::NamedFramebufferTexture(state.offbo, gl::COLOR_ATTACHMENT0, state.offtex, 0);
            gl::NamedFramebufferTexture(state.offbo, gl::DEPTH_ATTACHMENT, state.off_depth_tex, 0);
            gl::ClearNamedFramebufferfv(state.offbo, gl::COLOR, 0, state.clear_color.as_ptr());
            gl::ClearNamedFramebufferfv(state.offbo, gl::DEPTH, 0, &state.clear_depth);
            gl::ProgramUniform1i(state.def_fshader_id, 1, 2);
            gl::DrawArrays(gl::TRIANGLES, 0, 3); // blur y pass

            let aspect_ratio = (state.frame_dims.0 as f32) / (state.frame_dims.1 as f32);
            let projection = glm::perspective(aspect_ratio, state.fovy, state.near, state.far);
            let view: glm::Mat4x4 = glm::look_at(
                &state.cam_state.pos,
                &(state.cam_state.pos - state.cam_state.w),
                &state.cam_state.v,
            );
            let mut light_dir = glm::Vec4::new(0.0, 1.0, 0.0, 1.0);
            light_dir = glm::normalize(&(view * light_dir));
            gl::ActiveTexture(gl::TEXTURE0);
            gl::BindTexture(gl::TEXTURE_2D, state.offtex);
            gl::BindFramebuffer(gl::FRAMEBUFFER, 0);
            gl::ClearNamedFramebufferfv(0, gl::COLOR, 0, state.clear_color.as_ptr());
            gl::ClearNamedFramebufferfv(0, gl::DEPTH, 0, &state.clear_depth);
            gl::ProgramUniform1i(state.def_vshader_id, 5, 3);
            gl::ProgramUniform1i(state.def_fshader_id, 1, 4);
            gl::ProgramUniform3f(state.def_fshader_id, 7, light_dir.x, light_dir.y, light_dir.z);
            gl::ProgramUniformMatrix4fv(
                state.def_fshader_id,
                8,
                1,
                gl::FALSE,
                glm::value_ptr(&glm::inverse(&view)).as_ptr(),
            );
            gl::ProgramUniformMatrix4fv(
                state.def_vshader_id,
                7,
                1,
                gl::FALSE,
                glm::value_ptr(&glm::inverse(&projection)).as_ptr(),
            );
            gl::DrawArrays(gl::TRIANGLES, 0, 3);
            // Callbacks

            util::callbacks(&mut state);
            util::move_cam(&mut state);

            glfw.poll_events();
            state.window.swap_buffers();
        }
    }
}
