use super::util::{self, GlState, Render};
use gl::types::*;
use nalgebra_glm as glm;
use rayon::prelude::*;
use std::collections::HashMap;

pub struct Water {
    pub mesh: util::Mesh,
    pub instance_vbo_id: GLuint,
    pub ghost_positions: Vec<glm::Vec3>,
    pub positions: Vec<glm::Vec3>,
    pub velocities: Vec<glm::Vec3>,
    pub forces: Vec<glm::Vec3>,
    pub densities: Vec<f32>,
    pub ghost_rate: i32,
    pub particle_partition: (i32, i32, i32),
    pub grid_dimensions: (f32, f32, f32),
    pub particle_count: i32,
    pub particle_mass: f32,
    pub grid: HashMap<(i32, i32, i32), Vec<usize>>,
    pub ghost_grid: HashMap<(i32, i32, i32), Vec<usize>>,
    pub patch_size: f32,
    pub lbox: i32, // Box for current particle to consider others
    pub rest_density: f32,
    pub ghost_density: f32,
    pub gas_constant: f32,
    pub viscosity_constant: f32,
    pub gconstant: f32,
    pub tension_constant: f32,
    pub bouncing_constant: f32,
    pub dt: f32,
    pub radius: f32,
    pub aligned_box: AlignedBox,
    pub world_blur_radius: f32,
}

fn coord_to_grid(pos: glm::Vec3, patch_size: f32) -> (i32, i32, i32) {
    let x = pos.x / patch_size;
    let y = pos.y / patch_size;
    let z = pos.z / patch_size;
    (x as i32, y as i32, z as i32)
}
#[derive(Debug)]
pub struct AlignedBox {
    miny: f32,
    maxy: f32,
    minx: f32,
    maxx: f32,
    minz: f32,
    maxz: f32,
}

impl Water {
    pub fn new(particle_partition: (i32, i32, i32), radius: f32) -> Self {
        let mut owater = Water {
            mesh: util::init_mesh("meshes/sphere.obj"),
            instance_vbo_id: 0,
            ghost_positions: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            forces: Vec::new(),
            densities: Vec::new(),
            particle_partition,
            particle_count: particle_partition.0 * particle_partition.1 * particle_partition.2, // Default
            grid_dimensions: (0.0, 0.0, 0.0),
            particle_mass: 0.02,
            grid: HashMap::new(),
            ghost_grid: HashMap::new(),
            patch_size: 2.0 * radius,
            lbox: 2,
            ghost_rate: 2,
            rest_density: 0.0,
            ghost_density: 0.0,
            gas_constant: 0.16,
            viscosity_constant: 0.01,
            gconstant: 9.8,
            tension_constant: 0.000728,
            bouncing_constant: 0.13,
            dt: 1.0 / 100.0,
            radius,
            aligned_box: AlignedBox {
                miny: 0.0,
                maxy: particle_partition.1 as f32 * 4.0 * radius,
                minx: 0.0,
                maxx: particle_partition.0 as f32 * 4.0 * radius,
                minz: 0.0,
                maxz: particle_partition.2 as f32 * 4.0 * radius,
            },
            world_blur_radius: radius * 32.0,
        };
        owater.load_ghosts();
        owater.load_ghost_grid();
        owater.grid_dimensions = (
            owater.patch_size * owater.particle_partition.0 as f32,
            owater.patch_size * owater.particle_partition.1 as f32,
            owater.patch_size * owater.particle_partition.2 as f32,
        );
        for i in 0..particle_partition.0 {
            let ipos = i as f32 * 2.0 * owater.radius;
            for j in 0..particle_partition.1 {
                let jpos = j as f32 * 2.0 * owater.radius;
                for k in 0..particle_partition.2 {
                    owater
                        .positions
                        .push(glm::Vec3::new(ipos, jpos, k as f32 * 2.0 * owater.radius));
                    owater.velocities.push(glm::Vec3::new(0.0, 0.0, 0.0));
                    owater.forces.push(glm::Vec3::new(0.0, 0.0, 0.0));
                    owater.densities.push(0.0);
                }
            }
        }
        owater.rest_density = owater.measure_rest_density() * 0.80;
        owater.ghost_density = owater.rest_density;
        unsafe {
            gl::CreateBuffers(1, &mut owater.instance_vbo_id);
            gl::VertexArrayVertexBuffer(
                owater.mesh.vao_id,
                3,
                owater.instance_vbo_id,
                0,
                (size_of::<f32>() * 3) as GLsizei,
            );
            gl::EnableVertexArrayAttrib(owater.mesh.vao_id, 3);
            gl::VertexArrayAttribFormat(owater.mesh.vao_id, 3, 3, gl::FLOAT, gl::FALSE, 0);
            gl::VertexArrayBindingDivisor(owater.mesh.vao_id, 3, 1);
            gl::VertexArrayAttribBinding(owater.mesh.vao_id, 3, 3);
        }
        owater.load_offsets();
        owater
    }
    pub fn measure_rest_density(&mut self) -> f32 {
        self.load_grid();
        self.load_densities();
        let total: f32 = self.densities.iter().sum();
        total / self.particle_count as f32
    }
    fn load_ghosts(&mut self) {
        let increment = self.radius * 2.0 / self.ghost_rate as f32;
        let padding = self.patch_size * 5.0;
        // let ndepth_layers = self.lbox;
        let ndepth_layer = 4;
        let mut i = self.aligned_box.minx;
        let mut j;
        while i < self.aligned_box.maxx + padding {
            j = self.aligned_box.minz;
            while j < self.aligned_box.maxz + padding {
                for d in 1..ndepth_layer {
                    self.ghost_positions
                        .push(glm::Vec3::new(i, self.aligned_box.miny - d as f32 * increment, j));
                    self.ghost_positions
                        .push(glm::Vec3::new(i, self.aligned_box.maxy + d as f32 * increment, j));
                }
                j += increment;
            }
            i += increment;
        }
        i = self.aligned_box.minx;
        while i < self.aligned_box.maxx + padding {
            j = self.aligned_box.miny;
            while j < self.aligned_box.maxy + padding {
                for d in 1..ndepth_layer {
                    self.ghost_positions
                        .push(glm::Vec3::new(i, j, self.aligned_box.minz - d as f32 * increment));
                    self.ghost_positions
                        .push(glm::Vec3::new(i, j, self.aligned_box.maxz + d as f32 * increment));
                }
                j += increment;
            }
            i += increment;
        }
        i = self.aligned_box.miny;
        while i < self.aligned_box.maxy + padding {
            j = self.aligned_box.minz;
            while j < self.aligned_box.maxz + padding {
                for d in 1..ndepth_layer {
                    self.ghost_positions
                        .push(glm::Vec3::new(self.aligned_box.minx - d as f32 * increment, i, j));
                    self.ghost_positions
                        .push(glm::Vec3::new(self.aligned_box.maxx + d as f32 * increment, i, j));
                }
                j += increment;
            }
            i += increment;
        }
    }
    pub fn load_offsets(&mut self) {
        unsafe {
            gl::NamedBufferData(
                self.instance_vbo_id,
                (self.particle_count * 3 * (std::mem::size_of::<f32>() as i32)) as GLsizeiptr,
                self.positions.as_ptr() as *const GLvoid,
                gl::DYNAMIC_DRAW,
            );
        }
    }
    pub fn load_grid(&mut self) {
        self.grid.clear();
        for i in 0..(self.particle_count as usize) {
            let par_coords = coord_to_grid(self.positions[i], self.patch_size);
            if let Some(uvec) = self.grid.get_mut(&par_coords) {
                uvec.push(i);
            } else {
                let mut new_vec = Vec::new();
                new_vec.push(i);
                self.grid.insert(par_coords, new_vec);
            }
        }
    }
    pub fn load_ghost_grid(&mut self) {
        self.ghost_grid.clear();
        for i in 0..self.ghost_positions.len() {
            let par_coords = coord_to_grid(self.ghost_positions[i], self.patch_size);
            if let Some(uvec) = self.ghost_grid.get_mut(&par_coords) {
                uvec.push(i);
            } else {
                let mut new_vec = Vec::new();
                new_vec.push(i);
                self.ghost_grid.insert(par_coords, new_vec);
            }
        }
    }
    pub fn h_value(&self) -> f32 {
        let unit_voxel_length = self.patch_size * self.lbox as f32;
        glm::length(&glm::Vec3::new(unit_voxel_length, unit_voxel_length, unit_voxel_length))
    }
    pub fn init_simulation(&mut self) {
        self.forces.par_iter_mut().for_each(|oforce| {
            *oforce = glm::Vec3::new(0.0, 0.0, 0.0);
        })
    }
    pub fn load_densities(&mut self) {
        let kernel = Kernels::new(self.h_value());
        let positions = &self.positions;
        let ghost_positions = &self.ghost_positions;
        let grid = &self.grid;
        let ghost_grid = &self.ghost_grid;
        let lbox = self.lbox;

        self.densities.par_iter_mut().enumerate().for_each(|(i, odensity)| {
            let par_coords = coord_to_grid(positions[i], self.patch_size);
            let mut density = 0.0;
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord).clone() {
                            for &index in indices {
                                let inside_result = kernel.poly6(&(positions[i] - positions[index]));
                                density += inside_result;
                            }
                            continue;
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord).clone() {
                            for &index in indices {
                                let inside_result = kernel.poly6(&(positions[i] - ghost_positions[index]));
                                density += inside_result;
                            }
                        }
                    }
                }
            }
            density *= self.particle_mass;
            *odensity = density;
        });
    }

    // NOTE: talk about max(0.0) liquid not having negative pressure
    pub fn load_pressure(&mut self) {
        let kernel = Kernels::new(self.h_value());
        let positions = &self.positions;
        let grid = &self.grid;
        let densities = &self.densities;
        let ghost_positions = &self.ghost_positions;
        let ghost_grid = &self.ghost_grid;
        let lbox = self.lbox;
        let patch_size = self.patch_size;
        let gas_constant = self.gas_constant;
        let rest_density = self.rest_density;
        let particle_mass = self.particle_mass;

        self.forces.par_iter_mut().enumerate().for_each(|(i, oforce)| {
            let par_coords = coord_to_grid(positions[i], patch_size);
            let mut force = glm::Vec3::new(0.0, 0.0, 0.0);
            let cpressure = gas_constant * (densities[i] - rest_density).max(0.0);
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord).clone() {
                            for &index in indices {
                                let mut inside_result =
                                    particle_mass * kernel.spiky_gradient(&(positions[i] - positions[index]));
                                let opressure = gas_constant * (densities[index] - rest_density).max(0.0);
                                inside_result *= (cpressure + opressure) / (2.0 * densities[index]);
                                force -= inside_result;
                            }
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord).clone() {
                            for &index in indices {
                                let mut inside_result =
                                    particle_mass * kernel.spiky_gradient(&(positions[i] - ghost_positions[index]));
                                let opressure = gas_constant * (self.ghost_density - rest_density).max(0.0);
                                inside_result *= (cpressure + opressure) / (2.0 * self.ghost_density);
                                force -= inside_result;
                            }
                        }
                    }
                }
            }
            *oforce += force;
        })
    }
    pub fn load_viscosity(&mut self) {
        let kernel = Kernels::new(self.h_value());
        let positions = &self.positions;
        let velocities = &self.velocities;
        let densities = &self.densities;
        let grid = &self.grid;
        let ghost_positions = &self.ghost_positions;
        let ghost_grid = &self.ghost_grid;
        let lbox = self.lbox;
        let patch_size = self.patch_size;
        let viscosity_constant = self.viscosity_constant;
        let particle_mass = self.particle_mass;

        self.forces.par_iter_mut().enumerate().for_each(|(i, oforce)| {
            let par_coords = coord_to_grid(positions[i], patch_size);
            let mut force = glm::Vec3::new(0.0, 0.0, 0.0);
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord).clone() {
                            for &index in indices {
                                let inside_result = kernel.viscosity_laplacian(&(positions[i] - positions[index]));
                                let inside_force = (velocities[index] - velocities[i]) / densities[index];
                                force += inside_result * inside_force;
                            }
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord).clone() {
                            for &index in indices {
                                let inside_result =
                                    kernel.viscosity_laplacian(&(positions[i] - ghost_positions[index]));
                                let inside_force = -velocities[i] / self.ghost_density;
                                force += inside_result * inside_force;
                            }
                        }
                    }
                }
            }
            force *= viscosity_constant;
            force *= particle_mass;
            *oforce += force;
        })
    }
    pub fn load_gravity(&mut self) {
        let gravity = glm::Vec3::new(0.0, -1.0, 0.0) * self.gconstant * self.particle_mass;
        self.forces.par_iter_mut().for_each(|oforce| {
            *oforce += gravity;
        })
    }
    pub fn load_surface_tension(&mut self) {
        let kernel = Kernels::new(self.h_value());
        let h = kernel.h;
        let positions = &self.positions;
        let densities = &self.densities;
        let grid = &self.grid;
        let ghost_positions = &self.ghost_positions;
        let ghost_grid = &self.ghost_grid;
        let lbox = self.lbox;
        let patch_size = self.patch_size;
        let particle_mass = self.particle_mass;
        let tension_constant = self.tension_constant;

        self.forces.par_iter_mut().enumerate().for_each(|(i, oforce)| {
            let par_coords = coord_to_grid(positions[i], patch_size);
            let mut surface_normal = glm::Vec3::new(0.0, 0.0, 0.0);
            let mut curvature_term = 0.0;
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord).clone() {
                            for &index in indices {
                                surface_normal += (1.0 / densities[index])
                                    * kernel.poly6_gradient(&(positions[i] - positions[index]));
                                curvature_term += (1.0 / densities[index])
                                    * kernel.poly6_laplacian(&(positions[i] - positions[index]));
                            }
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord).clone() {
                            for &index in indices {
                                surface_normal += (1.0 / self.ghost_density)
                                    * kernel.poly6_gradient(&(positions[i] - ghost_positions[index]));
                                curvature_term += (1.0 / self.ghost_density)
                                    * kernel.poly6_laplacian(&(positions[i] - ghost_positions[index]));
                            }
                        }
                    }
                }
            }
            surface_normal *= particle_mass;
            curvature_term *= particle_mass;
            if glm::length(&surface_normal) < h * 0.1 {
                return;
            }
            let curvature_constant = -curvature_term / glm::length(&surface_normal);
            *oforce += tension_constant * curvature_constant * surface_normal;
        })
    }
    pub fn simulate(&mut self) {
        let forces = &self.forces;
        let particle_mass = self.particle_mass;
        let dt = self.dt;
        let bouncing_constant = self.bouncing_constant;
        let abox = &self.aligned_box;

        self.positions
            .par_iter_mut()
            .zip(self.velocities.par_iter_mut())
            .enumerate()
            .for_each(|(i, (oposition, ovelocity))| {
                let acceleration = forces[i] / particle_mass;
                *oposition += *ovelocity * dt;
                *ovelocity += acceleration * dt;
                // if oposition.y < abox.miny {
                //     oposition.y = abox.miny;
                //     *ovelocity = glm::reflect_vec(&*ovelocity, &glm::Vec3::new(0.0, -1.0, 0.0)) * bouncing_constant;
                // } else if oposition.y > abox.maxy {
                //     oposition.y = abox.maxy;
                //     *ovelocity = glm::reflect_vec(&*ovelocity, &glm::Vec3::new(0.0, 1.0, 0.0)) * bouncing_constant;
                // } else if oposition.x < abox.minx {
                //     oposition.x = abox.minx;
                //     *ovelocity = glm::reflect_vec(&*ovelocity, &glm::Vec3::new(1.0, 0.0, 0.0)) * bouncing_constant;
                // } else if oposition.x > abox.maxx {
                //     oposition.x = abox.maxx;
                //     *ovelocity = glm::reflect_vec(&*ovelocity, &glm::Vec3::new(-1.0, 0.0, 0.0)) * bouncing_constant;
                // } else if oposition.z < abox.minz {
                //     oposition.z = abox.minz;
                //     *ovelocity = glm::reflect_vec(&*ovelocity, &glm::Vec3::new(0.0, 0.0, -1.0)) * bouncing_constant;
                // } else if oposition.z > abox.maxz {
                //     oposition.z = abox.maxz;
                //     *ovelocity = glm::reflect_vec(&*ovelocity, &glm::Vec3::new(0.0, 0.0, 1.0)) * bouncing_constant;
                // }
            })
    }
}

impl Render for Water {
    fn draw(&mut self, state: &GlState) {
        let model: glm::Mat4x4 = glm::scale(&glm::identity(), &glm::Vec3::new(self.radius, self.radius, self.radius));
        let view: glm::Mat4x4 = glm::look_at(
            &state.cam_state.pos,
            &(state.cam_state.pos - state.cam_state.w),
            &state.cam_state.v,
        );
        let aspect_ratio = (state.frame_dims.0 as f32) / (state.frame_dims.1 as f32);
        let projection = glm::perspective(aspect_ratio, state.fovy, state.near, state.far);
        let normal: glm::Mat3 = glm::inverse_transpose(glm::mat4_to_mat3(&model));
        unsafe {
            gl::ProgramUniformMatrix4fv(state.def_vshader_id, 0, 1, gl::FALSE, glm::value_ptr(&model).as_ptr());
            gl::ProgramUniformMatrix4fv(state.def_vshader_id, 1, 1, gl::FALSE, glm::value_ptr(&view).as_ptr());
            gl::ProgramUniformMatrix4fv(
                state.def_vshader_id,
                2,
                1,
                gl::FALSE,
                glm::value_ptr(&projection).as_ptr(),
            );
            gl::ProgramUniformMatrix3fv(state.def_vshader_id, 3, 1, gl::FALSE, glm::value_ptr(&normal).as_ptr());
            gl::ProgramUniformMatrix4fv(
                state.def_fshader_id,
                4,
                1,
                gl::FALSE,
                glm::value_ptr(&projection).as_ptr(),
            );
            gl::ProgramUniform1i(state.def_vshader_id, 4, true as i32);
            gl::ProgramUniform1i(state.def_vshader_id, 5, 1);
            gl::ProgramUniform1f(state.def_vshader_id, 6, self.radius);
            gl::ProgramUniform3f(
                state.def_fshader_id,
                0,
                state.cam_state.pos.x,
                state.cam_state.pos.y,
                state.cam_state.pos.z,
            );
            gl::ProgramUniform1i(state.def_fshader_id, 1, 3);
            gl::ProgramUniform1f(state.def_fshader_id, 3, self.radius);

            // Set viewport
            gl::Viewport(0, 0, state.frame_dims.0, state.frame_dims.1);
            gl::BindVertexArray(self.mesh.vao_id);
            gl::BindFramebuffer(gl::FRAMEBUFFER, state.offbo);
            gl::NamedFramebufferTexture(state.offbo, gl::COLOR_ATTACHMENT0, state.offtex, 0);
            gl::NamedFramebufferTexture(state.offbo, gl::DEPTH_ATTACHMENT, state.off_depth_tex, 0);
            gl::ClearNamedFramebufferfv(state.offbo, gl::COLOR, 0, state.clear_color.as_ptr());
            gl::ClearNamedFramebufferfv(state.offbo, gl::DEPTH, 0, &state.clear_depth);
            gl::UseProgramStages(state.def_pipeline, gl::VERTEX_SHADER_BIT, state.def_vshader_id);
            gl::UseProgramStages(state.def_pipeline, gl::FRAGMENT_SHADER_BIT, state.def_fshader_id);
            gl::DrawArraysInstanced(gl::TRIANGLES, 0, 6, self.particle_count);
        }
    }
}

pub struct Kernels {
    h: f32,
    poly6_coeff: f32,
    poly6_grad_coeff: f32,
    poly6_lap_coeff: f32,
    spiky_grad_coeff: f32,
    visc_lap_coeff: f32,
}

impl Kernels {
    pub fn new(h: f32) -> Self {
        use std::f32::consts::PI;
        Self {
            h,
            poly6_coeff: 315.0 / (64.0 * PI * h.powi(9)),
            poly6_grad_coeff: -6.0 * 315.0 / (64.0 * PI * h.powi(9)),
            poly6_lap_coeff: 6.0 * 315.0 / (64.0 * PI * h.powi(9)),
            spiky_grad_coeff: -45.0 / (PI * h.powi(6)),
            visc_lap_coeff: 45.0 / (PI * h.powi(5)),
        }
    }

    /// Poly6 — value. Used for density.
    pub fn poly6(&self, r: &glm::Vec3) -> f32 {
        let r2 = glm::dot(r, r);
        let h2 = self.h * self.h;
        if r2 > h2 {
            return 0.0;
        }
        let diff = h2 - r2;
        self.poly6_coeff * diff * diff * diff
    }

    /// Poly6 — gradient. Used for surface tension / color field.
    pub fn poly6_gradient(&self, r: &glm::Vec3) -> glm::Vec3 {
        let r2 = glm::dot(r, r);
        let h2 = self.h * self.h;
        if r2 > h2 {
            return glm::zero();
        }
        let diff = h2 - r2;
        r * (self.poly6_grad_coeff * diff * diff)
    }

    /// Poly6 — Laplacian. Used for surface tension curvature.
    pub fn poly6_laplacian(&self, r: &glm::Vec3) -> f32 {
        let r2 = glm::dot(r, r);
        let h2 = self.h * self.h;
        if r2 > h2 {
            return 0.0;
        }
        let diff = h2 - r2;
        self.poly6_lap_coeff * diff * (7.0 * r2 - 3.0 * h2)
    }

    /// Spiky — gradient. Used for pressure.
    pub fn spiky_gradient(&self, r: &glm::Vec3) -> glm::Vec3 {
        let len = glm::length(r);
        if len > self.h || len < 1e-6 {
            return glm::zero();
        }
        let diff = self.h - len;
        r * (self.spiky_grad_coeff * diff * diff / len)
    }

    /// Viscosity — Laplacian. Used for viscosity.
    pub fn viscosity_laplacian(&self, r: &glm::Vec3) -> f32 {
        let len = glm::length(r);
        if len > self.h {
            return 0.0;
        }
        self.visc_lap_coeff * (1.0 - len / self.h)
    }
}
