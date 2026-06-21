mod hash;
mod util;
mod water;
use gl::types::*;
use glfw::Context;
use nalgebra_glm as glm;
use std::ffi::c_void;
use std::time::Instant;
use util::Render;
use water::Water;

const LOCAL_WORKGROUP_SIZE: i32 = 64;
const MAX_PARTICLES: usize = 16384;
const MAX_GHOST_PARTICLES: usize = 16384;
const MAX_CELLS: usize = 32768;

const COMP_POSITIONS_OFFSET: usize = 0;
const COMP_GHOST_POSITIONS_OFFSET: usize = COMP_POSITIONS_OFFSET + COMP_POSITIONS_SIZE;
const COMP_VELOCITIES_OFFSET: usize = COMP_GHOST_POSITIONS_OFFSET + COMP_GHOST_POSITIONS_SIZE;
const COMP_FORCES_OFFSET: usize = COMP_VELOCITIES_OFFSET + COMP_VELOCITIES_SIZE;
const COMP_CELLS_OFFSET: usize = COMP_FORCES_OFFSET + COMP_FORCES_SIZE;
const COMP_GHOST_CELLS_OFFSET: usize = COMP_CELLS_OFFSET + COMP_CELLS_SIZE;
const COMP_DENSITY_OFFSET: usize = COMP_GHOST_CELLS_OFFSET + COMP_GHOST_CELLS_SIZE;
const COMP_GHOST_DENSITY_OFFSET: usize = COMP_DENSITY_OFFSET + COMP_DENSITY_SIZE;
const COMP_INDICES_OFFSET: usize = COMP_GHOST_DENSITY_OFFSET + COMP_GHOST_DENSITY_SIZE;
const COMP_GHOST_INDICES_OFFSET: usize = COMP_INDICES_OFFSET + COMP_INDICES_SIZE;

const COMP_POSITIONS_SIZE: usize = size_of::<glm::Vec4>() * MAX_PARTICLES;
const COMP_GHOST_POSITIONS_SIZE: usize = size_of::<glm::Vec4>() * MAX_GHOST_PARTICLES;
const COMP_VELOCITIES_SIZE: usize = size_of::<glm::Vec4>() * MAX_PARTICLES;
const COMP_FORCES_SIZE: usize = size_of::<glm::Vec4>() * MAX_PARTICLES;
const COMP_CELLS_SIZE: usize = size_of::<glm::UVec2>() * MAX_CELLS;
const COMP_GHOST_CELLS_SIZE: usize = size_of::<glm::UVec2>() * MAX_CELLS;
const COMP_DENSITY_SIZE: usize = size_of::<f32>() * MAX_PARTICLES;
const COMP_GHOST_DENSITY_SIZE: usize = size_of::<f32>() * MAX_GHOST_PARTICLES;
const COMP_INDICES_SIZE: usize = size_of::<u32>() * MAX_PARTICLES;
const COMP_GHOST_INDICES_SIZE: usize = size_of::<u32>() * MAX_GHOST_PARTICLES;

fn main() {
    let (mut glfw, window, events) = util::setup_glfw();
    let (vshader_id, fshader_id, cshader_id): (GLuint, GLuint, GLuint) = (
        util::load_shader("shaders/regular.vert"),
        util::load_shader("shaders/regular.frag"),
        util::load_shader("shaders/regular.comp"),
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
        window_dims: (1000, 1000),
        frame_dims: (1000, 1000),
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
        def_cshader_id: cshader_id,
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
    state.last_cursor_pos = (f32!(cursor_pos.0), f32!(cursor_pos.1));

    unsafe {
        // Create offscreen framebuffer
        gl::CreateFramebuffers(1, &mut state.offbo);
        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.offtex);
        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.blurtex);

        gl::TextureParameteri(state.offtex, gl::TEXTURE_MIN_FILTER, i32!(gl::LINEAR));
        gl::TextureParameteri(state.offtex, gl::TEXTURE_MAG_FILTER, i32!(gl::NEAREST));
        gl::TextureParameteri(state.offtex, gl::TEXTURE_WRAP_S, i32!(gl::REPEAT));
        gl::TextureParameteri(state.offtex, gl::TEXTURE_WRAP_T, i32!(gl::REPEAT));

        gl::TextureParameteri(state.blurtex, gl::TEXTURE_MIN_FILTER, i32!(gl::LINEAR));
        gl::TextureParameteri(state.blurtex, gl::TEXTURE_MAG_FILTER, i32!(gl::NEAREST));
        gl::TextureParameteri(state.blurtex, gl::TEXTURE_WRAP_S, i32!(gl::REPEAT));
        gl::TextureParameteri(state.blurtex, gl::TEXTURE_WRAP_T, i32!(gl::REPEAT));

        gl::CreateTextures(gl::TEXTURE_2D, 1, &mut state.off_depth_tex);
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_MIN_FILTER, i32!(gl::NEAREST));
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_MAG_FILTER, i32!(gl::NEAREST));
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_WRAP_S, i32!(gl::REPEAT));
        gl::TextureParameteri(state.off_depth_tex, gl::TEXTURE_WRAP_T, i32!(gl::REPEAT));

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
    let mut owater = Water::new((2, 5, 2), (2, 2, 2), (2, 2, 2), 0.1);
    println!(
        "Particles:{}, GhostParticles:{}, Cells:{}",
        owater.positions.len(),
        owater.ghost_positions.len(),
        owater.grid_lens.0 * owater.grid_lens.1 * owater.grid_lens.2
    );
    while !state.window.should_close() {
        let time_now = Instant::now();
        let time_elapsed = (time_now - time).as_secs_f32();
        time = time_now;
        accumulator += time_elapsed;
        while accumulator > owater.dt {
            unsafe {
                gl::BindProgramPipeline(state.def_pipeline);
                gl::UseProgramStages(state.def_pipeline, gl::VERTEX_SHADER_BIT, 0);
                gl::UseProgramStages(state.def_pipeline, gl::FRAGMENT_SHADER_BIT, 0);
                gl::UseProgramStages(state.def_pipeline, gl::COMPUTE_SHADER_BIT, state.def_cshader_id);
                owater.water_ubo.particle_count = i32!(owater.positions.len());
                owater.water_ubo.ghost_particle_count = i32!(owater.ghost_positions.len());
                let workgroup_count =
                    (owater.water_ubo.particle_count + LOCAL_WORKGROUP_SIZE - 1) / LOCAL_WORKGROUP_SIZE;


                // Upload positions
                gl::NamedBufferSubData(
                    owater.ssbo,
                    COMP_POSITIONS_OFFSET as isize,
                    (1 * size_of::<glm::Vec4>() * usize!(owater.water_ubo.particle_count)) as isize,
                    owater.positions.as_ptr() as *const c_void,
                );

                owater.load_grid();
                owater.load_grid_ssbo();
                owater.init_simulation();

                owater.set_calculation_type_update_ubo(water::CalculationType::InitSimulationStep);
                gl::DispatchCompute(u32!(workgroup_count), 1, 1);
                gl::MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT);
                gl::GetNamedBufferSubData(
                    owater.ssbo,
                    COMP_FORCES_OFFSET as isize,
                    (1 * size_of::<glm::Vec4>() * usize!(owater.water_ubo.particle_count)) as isize,
                    owater.forces.as_mut_ptr() as *mut c_void,
                );

                // owater.set_calculation_type_update_ubo(water::CalculationType::CalculateDensities);
                // gl::DispatchCompute(u32!(workgroup_count), 1, 1);
                // gl::MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT);

                // owater.set_calculation_type_update_ubo(water::CalculationType::CalculateGhostDensities);
                // gl::DispatchCompute(u32!(workgroup_count), 1, 1);
                // gl::MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT);
                // gl::GetNamedBufferSubData(
                //     owater.ssbo,
                //     COMP_DENSITY_OFFSET as isize,
                //     (1 * size_of::<f32>() * usize!(owater.water_ubo.particle_count)) as isize,
                //     owater.densities.as_mut_ptr() as *mut c_void,
                // );
                // gl::GetNamedBufferSubData(
                //     owater.ssbo,
                //     COMP_GHOST_DENSITY_OFFSET as isize,
                //     (1 * size_of::<f32>() * usize!(owater.water_ubo.ghost_particle_count)) as isize,
                //     owater.ghost_densities.as_mut_ptr() as *mut c_void,
                // );

                owater.load_densities();
                println!("{:?}", owater.densities);
                // owater.load_gravity();

                owater.set_calculation_type_update_ubo(water::CalculationType::CalculateGravity);
                gl::DispatchCompute(u32!(workgroup_count), 1, 1);
                gl::MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT);

                // owater.set_calculation_type_update_ubo(water::CalculationType::CalculatePressure);
                // gl::DispatchCompute(u32!(workgroup_count), 1, 1);
                // gl::MemoryBarrier(gl::SHADER_STORAGE_BARRIER_BIT);
                gl::GetNamedBufferSubData(
                    owater.ssbo,
                    COMP_FORCES_OFFSET as isize,
                    (1 * size_of::<glm::Vec4>() * usize!(owater.water_ubo.particle_count)) as isize,
                    owater.forces.as_mut_ptr() as *mut c_void,
                );

                owater.load_pressure();
                owater.load_viscosity();
                owater.load_surface_tension();
                owater.simulate();
                owater.load_offsets();

                gl::UseProgramStages(state.def_pipeline, gl::VERTEX_SHADER_BIT, state.def_vshader_id);
                gl::UseProgramStages(state.def_pipeline, gl::FRAGMENT_SHADER_BIT, state.def_fshader_id);
            }
            accumulator -= owater.dt;
        }
        unsafe {
            gl::Enable(gl::DEPTH_TEST);
            owater.draw(&state);

            gl::Disable(gl::DEPTH_TEST);
            // gl::BindFramebuffer(gl::FRAMEBUFFER, state.offbo);
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
                (f32!(state.frame_dims.1)) / (2.0 * f32::tan(state.fovy / 2.0)),
            );
            gl::ProgramUniform1f(state.def_fshader_id, 6, owater.world_blur_radius);
            gl::ProgramUniform2f(
                state.def_fshader_id,
                2,
                1.0 / f32!(state.frame_dims.0),
                1.0 / f32!(state.frame_dims.1),
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

            let aspect_ratio = (f32!(state.frame_dims.0)) / (f32!(state.frame_dims.1));
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
