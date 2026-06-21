use crate::util::{self, GlState, Render};
use crate::*;
use crate::{f32, i32, u32};
use gl::types::*;
use nalgebra_glm as glm;
use rayon::prelude::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::size_of;
use std::ptr::null;

#[repr(i32)]
#[derive(Default, Debug)]
pub enum CalculationType {
    #[default]
    CalculateDensities = 0,
    CalculatePressure = 1,
    CalculateViscosity = 2,
    CalculateSurfaceTension = 3,
    CalculateGravity = 4,
    CalculateGhostDensities = 5,
    InitSimulationStep = 6,
    FinalizeSimulationStep = 7,
    CalculateBomb = 8,
}
pub enum SSBOType {
    Positions,
    GhostPositions,
    Velocities,
    Forces,
    Cells,
    GhostCells,
    Densitie,
    GhostDensitie,
    Indices,
    GhostIndices,
}
#[repr(C, align(16))]
#[derive(Default, Debug)]
pub struct WaterUbo {
    // 16 byte boundaries
    pub grid_lens: glm::IVec3,
    pub patch_size: f32,

    pub bounding_mins: glm::Vec3,
    pub lbox: i32,

    pub bounding_maxs: glm::Vec3,
    pub gas_constant: f32,

    pub rest_density: f32,
    pub particle_mass: f32,
    pub ghost_particle_mass: f32,
    pub viscosity_constant: f32,

    pub gconstant: f32,
    pub tension_constant: f32,
    pub calculation_type: CalculationType,
    pub particle_count: i32,

    pub bomb_position: glm::Vec3,
    pub ghost_particle_count: i32,

    pub box_bomb: i32,
    pub bomb_constant: f32,
    pub _padding0: [i32; 2],
}
pub struct Water {
    pub mesh: util::Mesh,
    pub instance_vbo_id: GLuint,
    pub ssbo: GLuint,
    pub debug_ssbo: GLuint,
    pub ubo: GLuint,
    pub ghost_positions: Vec<glm::Vec4>,
    pub positions: Vec<glm::Vec4>,
    pub velocities: Vec<glm::Vec4>,
    pub forces: Vec<glm::Vec4>,
    pub densities: Vec<f32>,
    pub ghost_densities: Vec<f32>,
    pub ghost_rate: i32,
    pub ghost_margin: (i32, i32, i32),
    pub empty_cells_margin: (i32, i32, i32),
    pub particle_partition: (i32, i32, i32),
    pub ghost_partition: (i32,i32,i32),
    pub empty_partition: (i32,i32,i32),
    pub total_partition: (i32, i32, i32),
    pub particle_count: i32,
    pub particle_mass: f32,
    pub ghost_particle_mass: f32,
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
    pub water_ubo: WaterUbo,
}

fn coord_to_grid(pos: glm::Vec4, patch_size: f32) -> (i32, i32, i32) {
    let x = pos.x / patch_size;
    let y = pos.y / patch_size;
    let z = pos.z / patch_size;
    (i32!(x), i32!(y), i32!(z))
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
    pub fn new(
        particle_partition: (i32, i32, i32),
        ghost_margin: (i32, i32, i32),
        empty_cells_margin: (i32, i32, i32),
        radius: f32,
    ) -> Self {
        let mut owater = Water {
            mesh: util::init_mesh("meshes/sphere.obj"),
            instance_vbo_id: 0,
            ssbo: 0,
            debug_ssbo: 0,
            ubo: 0,
            ghost_positions: Vec::new(),
            positions: Vec::new(),
            velocities: Vec::new(),
            forces: Vec::new(),
            densities: Vec::new(),
            ghost_densities: Vec::new(),
            particle_partition,
            ghost_partition: ghost_margin,
            empty_partition: empty_cells_margin,
            ghost_margin,
            empty_cells_margin,
            particle_count: particle_partition.0 * particle_partition.1 * particle_partition.2, // Default
            total_partition: (0, 0, 0),
            particle_mass: 0.02,
            ghost_particle_mass: 0.10,
            grid: HashMap::new(),
            ghost_grid: HashMap::new(),
            patch_size: 2.0 * radius,
            lbox: 2,
            ghost_rate: 1,
            rest_density: 0.0,
            ghost_density: 0.0,
            gas_constant: 0.26,
            viscosity_constant: 0.02,
            gconstant: 9.8,
            tension_constant: 0.000728,
            bouncing_constant: 0.13,
            dt: 1.0 / 100.0,
            radius,
            aligned_box: AlignedBox {
                miny: 0.0,
                maxy: f32!(particle_partition.1) * 4.0 * radius,
                minx: 0.0,
                maxx: f32!(particle_partition.0) * 4.0 * radius,
                minz: 0.0,
                maxz: f32!(particle_partition.2) * 4.0 * radius,
            },
            world_blur_radius: radius * 32.0,
            water_ubo: WaterUbo {
                grid_lens: glm::IVec3::new(0, 0, 0),
                patch_size: 2.0 * radius,
                lbox: 2,
                bounding_maxs: glm::Vec3::new(0.0, 0.0, 0.0),
                bounding_mins: glm::Vec3::new(0.0, 0.0, 0.0),
                gas_constant: 0.36,
                rest_density: 0.0,
                particle_mass: 0.02,
                ghost_particle_mass: 0.30,
                viscosity_constant: 0.01,
                gconstant: 9.8,
                tension_constant: 0.000728,
                calculation_type: CalculationType::CalculateDensities,
                particle_count: 0,
                ghost_particle_count: 0,
                bomb_position: glm::Vec3::new(0.0,0.0,0.0),
                box_bomb: 7,
                bomb_constant: 30.0,
                _padding0: [0; 2],
            },
        };

        let total_particles_partition: (i32, i32, i32) = (
            particle_partition.0 + ghost_margin.0 * 2 + empty_cells_margin.0 * 2,
            particle_partition.1 + ghost_margin.1 + empty_cells_margin.1, // no ghosts at top
            particle_partition.2 + ghost_margin.2 * 2 + empty_cells_margin.2 * 2,
        );
        // If 2*radius != patch_size this is wrong!
        owater.total_partition = total_particles_partition;
        owater.water_ubo.grid_lens = glm::IVec3::new(owater.total_partition.0, owater.total_partition.1, owater.total_partition.2);

        let ghost_range_x_start: (i32, i32) = (0, ghost_margin.0 - 1);
        let ghost_range_x_end: (i32, i32) = (
            total_particles_partition.0 - ghost_margin.0,
            total_particles_partition.0 - 1,
        );
        let ghost_range_y: (i32, i32) = (0, ghost_margin.1 - 1);
        let ghost_range_z_start: (i32, i32) = (0, ghost_margin.2 - 1);
        let ghost_range_z_end: (i32, i32) = (
            total_particles_partition.2 - ghost_margin.2,
            total_particles_partition.2 - 1,
        );

        let empty_range_x_start: (i32, i32) = (ghost_margin.0, ghost_margin.0 + empty_cells_margin.0 - 1);
        let empty_range_x_end: (i32, i32) = (
            total_particles_partition.0 - ghost_margin.0 - empty_cells_margin.0,
            total_particles_partition.0 - ghost_margin.0 - 1,
        );
        let empty_range_y: (i32, i32) = (ghost_margin.1, ghost_margin.1 + empty_cells_margin.1 - 1);
        let empty_range_z_start: (i32, i32) = (ghost_margin.2, ghost_margin.2 + empty_cells_margin.2 - 1);
        let empty_range_z_end: (i32, i32) = (
            total_particles_partition.2 - ghost_margin.2 - empty_cells_margin.2,
            total_particles_partition.2 - ghost_margin.2 - 1,
        );

        let is_particle_ghost = |(i, j, k): (i32, i32, i32)| {
            j <= ghost_range_y.1
                || i <= ghost_range_x_start.1
                || i >= ghost_range_x_end.0
                || k <= ghost_range_z_start.1
                || k >= ghost_range_z_end.0
        };
        // Must come after is_particle_ghost as if else
        let is_particle_empty = |(i, j, k): (i32, i32, i32)| {
            j <= empty_range_y.1
                || i <= empty_range_x_start.1
                || i >= empty_range_x_end.0
                || k <= empty_range_z_start.1
                || k >= empty_range_z_end.0
        };
        let bomb_x = total_particles_partition.0 as f32 / 2.0 * 2.0 * owater.radius;
        let bomb_z = total_particles_partition.2 as f32 / 2.0 * 2.0 * owater.radius;
        owater.water_ubo.bomb_position = glm::Vec3::new(bomb_x, 0.5, bomb_z);

        let (mut ipos, mut jpos, mut kpos): (f32, f32, f32) = (0.0, 0.0, 0.0);
        for i in 0..total_particles_partition.0 {
            ipos = f32!(i) * 2.0 * owater.radius;
            for j in 0..total_particles_partition.1 {
                jpos = f32!(j) * 2.0 * owater.radius;
                for k in 0..total_particles_partition.2 {
                    kpos = f32!(k) * 2.0 * owater.radius;
                    let new_pos = glm::Vec4::new(ipos, jpos, kpos, 0.0);
                    let index = owater.ghost_positions.len();

                    if is_particle_ghost((i, j, k)) {
                        owater.ghost_positions.push(new_pos);
                        owater.ghost_densities.push(0.0);
                        let grid_coord = coord_to_grid(new_pos, owater.patch_size);
                        if let Some(indices) = owater.ghost_grid.get_mut(&grid_coord) {
                            indices.push(index);
                        } else {
                            owater.ghost_grid.insert(grid_coord, vec![index; 1]);
                        }
                        continue;
                    }
                    if is_particle_empty((i, j, k)) {
                        continue;
                    }

                    owater.positions.push(glm::Vec4::new(ipos, jpos, kpos, 0.0));
                    owater.velocities.push(glm::Vec4::new(0.0, 0.0, 0.0, 0.0));
                    owater.forces.push(glm::Vec4::new(0.0, 0.0, 0.0, 0.0));
                    owater.densities.push(0.0);
                }
            }
        }
        owater.water_ubo.bounding_maxs = glm::Vec3::new(ipos, jpos, kpos);

        owater.water_ubo.particle_count = i32!(owater.positions.len());
        owater.water_ubo.ghost_particle_count = i32!(owater.ghost_positions.len());
        owater.rest_density = owater.measure_rest_density() * 0.80;
        owater.water_ubo.rest_density = owater.rest_density;
        owater.ghost_density = owater.rest_density;
        unsafe {
            gl::CreateBuffers(1, &mut owater.instance_vbo_id);
            gl::VertexArrayVertexBuffer(
                owater.mesh.vao_id,
                3,
                owater.instance_vbo_id,
                0,
                (size_of::<f32>() * 4) as GLsizei,
            );
            gl::EnableVertexArrayAttrib(owater.mesh.vao_id, 3);
            gl::VertexArrayAttribFormat(owater.mesh.vao_id, 3, 3, gl::FLOAT, gl::FALSE, 0);
            gl::VertexArrayBindingDivisor(owater.mesh.vao_id, 3, 1);
            gl::VertexArrayAttribBinding(owater.mesh.vao_id, 3, 3);
        }
        // owater.load_offsets();

        // Ghost grid gpu conversion
        let (gpu_ghost_indices_array, gpu_ghost_cells_array) = owater.transform_hashmap_gpu_grid(&owater.ghost_grid);

        // SSBO
        let mut ssbo: GLuint = 0;
        unsafe {
            gl::CreateBuffers(1, &mut ssbo as *mut GLuint);
            gl::NamedBufferStorage(
                ssbo,
                (3 * MAX_PARTICLES * size_of::<glm::Vec4>()) as isize
                    + (2 * MAX_GHOST_PARTICLES * size_of::<glm::Vec4>()) as isize
                    + (2 * MAX_CELLS * size_of::<glm::UVec2>()) as isize
                    + (1 * MAX_PARTICLES * size_of::<u32>()) as isize
                    + (1 * MAX_GHOST_PARTICLES * size_of::<u32>()) as isize
                    + (1 * MAX_PARTICLES * size_of::<f32>()) as isize,
                null(),
                gl::DYNAMIC_STORAGE_BIT | gl::MAP_READ_BIT,
            );
            gl::NamedBufferSubData(
                ssbo,
                COMP_GHOST_CELLS_OFFSET as isize,
                (size_of::<glm::UVec2>() * gpu_ghost_cells_array.len()) as isize,
                gpu_ghost_cells_array.as_ptr() as *const c_void,
            );
            gl::NamedBufferSubData(
                ssbo,
                COMP_GHOST_INDICES_OFFSET as isize,
                (size_of::<u32>() * gpu_ghost_indices_array.len()) as isize,
                gpu_ghost_indices_array.as_ptr() as *const c_void,
            );
            // Upload ghost positions
            gl::NamedBufferSubData(
                ssbo,
                COMP_GHOST_POSITIONS_OFFSET as isize,
                (1 * size_of::<glm::Vec4>() * usize!(owater.water_ubo.ghost_particle_count)) as isize,
                owater.ghost_positions.as_ptr() as *const c_void,
            );
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, ssbo);
            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 0, ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            gl::ObjectLabel(gl::BUFFER, ssbo, 4, "SSBO".as_ptr() as *const i8);
        }
        // Debug SSBO
        let mut debug_ssbo: GLuint = 0;
        unsafe {
            gl::CreateBuffers(1, &mut debug_ssbo as *mut GLuint);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, debug_ssbo);
            gl::BindBufferBase(gl::SHADER_STORAGE_BUFFER, 1, debug_ssbo);
            gl::BindBuffer(gl::SHADER_STORAGE_BUFFER, 0);
            gl::NamedBufferStorage(
                debug_ssbo,
                (size_of::<glm::Vec4>() * MAX_PARTICLES) as isize,
                null(),
                gl::MAP_READ_BIT,
            );
        }

        // UBO
        let mut ubo: GLuint = 0;
        unsafe {
            gl::CreateBuffers(1, &mut ubo as *mut GLuint);
            gl::BindBufferBase(gl::UNIFORM_BUFFER, 0, ubo);
            gl::NamedBufferStorage(
                ubo,
                size_of::<WaterUbo>() as isize,
                &owater.water_ubo as *const WaterUbo as *const c_void,
                gl::DYNAMIC_STORAGE_BIT,
            );
        }

        owater.ssbo = ssbo;
        owater.ubo = ubo;
        owater.debug_ssbo = debug_ssbo;
        owater.water_ubo.gas_constant = owater.gas_constant;
        owater.water_ubo.gconstant = owater.gconstant;
        owater.water_ubo.ghost_particle_mass = owater.ghost_particle_mass;
        owater.water_ubo.particle_mass = owater.particle_mass;
        owater.water_ubo.viscosity_constant = owater.viscosity_constant;
        owater.water_ubo.tension_constant = owater.tension_constant;
        owater
    }
    pub fn add_particles(&mut self) {
        let jpos = 2.0 * self.radius * (self.total_partition.1 - 1) as f32;
        let starti = self.empty_partition.0 + self.ghost_partition.0;
        let startk = self.empty_partition.2 + self.ghost_partition.2;
        let endi = self.total_partition.0 - starti;
        let endk = self.total_partition.2 - startk;
        for i in starti..endi {
            let ipos = f32!(i) * 2.0 * self.radius;
            for k in startk..endk {
                let kpos = f32!(k) * 2.0 * self.radius;
                // println!("({},{},{})", ipos, jpos, kpos);
                self.positions.push(glm::Vec4::new(ipos, jpos, kpos, 0.0));
                self.velocities.push(glm::Vec4::new(0.0, 0.0, 0.0, 0.0));
                self.forces.push(glm::Vec4::new(0.0, 0.0, 0.0, 0.0));
                self.densities.push(0.0);
            }
        }
        self.particle_count = self.positions.len() as i32;
        self.water_ubo.particle_count = self.particle_count;
    }
    pub fn print_ubo(&self) {
        let mut dummy: WaterUbo = WaterUbo::default();
        unsafe {
            gl::GetNamedBufferSubData(
                self.ubo,
                0,
                (size_of::<WaterUbo>()) as isize,
                &mut dummy as *mut WaterUbo as *mut c_void,
            );
        }
        println!("Ubo: {:?}", dummy);
    }
    pub fn print_debug_ssbo(&self, length: usize) {
        let data: Vec<glm::IVec4> = vec![glm::IVec4::new(0, 0, 0, 0); length];
        unsafe {
            gl::GetNamedBufferSubData(
                self.debug_ssbo,
                0,
                (length * size_of::<glm::IVec4>()) as isize,
                data.as_ptr() as *mut c_void,
            );
        }
        for vec in data {
            println!("{:?}", vec);
        }
    }

    pub fn load_grid_ssbo(&self) {
        // We need to update indices and cells arrays
        let (indices, cells) = self.transform_hashmap_gpu_grid(&self.grid);
        unsafe {
            gl::NamedBufferSubData(
                self.ssbo,
                COMP_CELLS_OFFSET as isize,
                (size_of::<glm::UVec2>() * cells.len()) as isize,
                cells.as_ptr() as *const c_void,
            );
            gl::NamedBufferSubData(
                self.ssbo,
                COMP_INDICES_OFFSET as isize,
                (size_of::<u32>() * indices.len()) as isize,
                indices.as_ptr() as *const c_void,
            );
        }
    }
    pub fn fetch_ssbo_data(&mut self) {
        // Fetch positions to calculate the grid
        unsafe {
            gl::GetNamedBufferSubData(
                self.ssbo,
                0,
                (self.positions.len() * size_of::<glm::Vec4>()) as isize,
                self.positions.as_mut_ptr() as *mut c_void,
            );
        }
    }
    pub fn transform_hashmap_gpu_grid(
        &self,
        grid: &HashMap<(i32, i32, i32), Vec<usize>>,
    ) -> (Vec<u32>, Vec<glm::UVec2>) {
        let mut indices_flattened = Vec::with_capacity(self.positions.len());
        let mut cells = Vec::new();
        for i in 0..self.total_partition.0 {
            for j in 0..self.total_partition.1 {
                for k in 0..self.total_partition.2 {
                    let grid_key = (i, j, k);
                    // let cell_index = k * self.grid_lens.1 * self.grid_lens.0 + j * self.grid_lens.0 + i;

                    let mut new_cell = glm::UVec2::new(u32!(indices_flattened.len()), u32::MAX);
                    if let Some(indices) = grid.get(&grid_key) {
                        for &index in indices {
                            indices_flattened.push(u32!(index));
                        }
                        new_cell.y = u32!(indices_flattened.len() - 1);
                    } else {
                        new_cell.x = u32::MAX;
                    }
                    cells.push(new_cell);
                }
            }
        }
        (indices_flattened, cells)
    }

    pub fn measure_rest_density(&mut self) -> f32 {
        self.load_grid();
        self.load_densities();
        let total: f32 = self.densities.iter().sum();
        total / f32!(self.particle_count)
    }
    pub fn load_offsets(&mut self) {
        unsafe {
            gl::NamedBufferData(
                self.instance_vbo_id,
                (self.particle_count * 4 * (i32!(std::mem::size_of::<f32>()))) as GLsizeiptr,
                self.positions.as_ptr() as *const GLvoid,
                gl::DYNAMIC_DRAW,
            );
        }
    }
    pub fn load_grid(&mut self) {
        self.grid.clear();

        // Add regular cells to the grid
        for i in 0..self.positions.len() {
            let par_coords = coord_to_grid(self.positions[i], self.patch_size);
            if let Some(indices) = self.grid.get_mut(&par_coords) {
                indices.push(i);
            } else {
                let mut new_vec = Vec::new();
                new_vec.push(i);
                self.grid.insert(par_coords, new_vec);
            }
        }
    }
    pub fn h_value(&self) -> f32 {
        let unit_voxel_length = self.patch_size * f32!(self.lbox);
        glm::length(&glm::Vec3::new(unit_voxel_length, unit_voxel_length, unit_voxel_length))
    }
    pub fn init_simulation(&mut self) {
        self.forces.par_iter_mut().for_each(|oforce| {
            *oforce = glm::Vec4::new(0.0, 0.0, 0.0, 0.0);
        })
    }
    pub fn load_densities(&mut self) {
        let kernel = Kernels::new(self.h_value());
        let positions = &self.positions;
        let ghost_positions = &self.ghost_positions;
        let grid = &self.grid;
        let ghost_grid = &self.ghost_grid;
        let lbox = self.lbox;

        // Densities of the regular particles
        self.densities.par_iter_mut().enumerate().for_each(|(i, odensity)| {
            let par_coords = coord_to_grid(positions[i], self.patch_size);
            let mut density = 0.0;
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord) {
                            for &index in indices {
                                let inside_result =
                                    kernel.poly6(&(positions[i] - positions[index])) * self.particle_mass;
                                density += inside_result;
                            }
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord) {
                            for &index in indices {
                                let inside_result =
                                    kernel.poly6(&(positions[i] - ghost_positions[index])) * self.ghost_particle_mass;
                                density += inside_result;
                            }
                        }
                    }
                }
            }
            *odensity = density;
        });
        // Densities of the ghost particles
        self.ghost_densities
            .par_iter_mut()
            .enumerate()
            .for_each(|(i, odensity)| {
                let par_coords = coord_to_grid(ghost_positions[i], self.patch_size);
                let mut density = 0.0;
                for a in (-lbox)..=(lbox) {
                    for b in (-lbox)..=(lbox) {
                        for c in (-lbox)..=(lbox) {
                            let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                            if let Some(indices) = grid.get(&other_coord) {
                                for &index in indices {
                                    let inside_result =
                                        kernel.poly6(&(ghost_positions[i] - positions[index])) * self.particle_mass;
                                    density += inside_result;
                                }
                            }
                            if let Some(indices) = ghost_grid.get(&other_coord) {
                                for &index in indices {
                                    let inside_result = kernel.poly6(&(ghost_positions[i] - ghost_positions[index]))
                                        * self.ghost_particle_mass;
                                    density += inside_result;
                                }
                            }
                        }
                    }
                }
                *odensity = density;
            });
    }

    // NOTE: talk about max(0.0) liquid not having negative pressure
    pub fn load_pressure(&mut self) {
        let kernel = Kernels::new(self.h_value());
        let positions = &self.positions;
        let grid = &self.grid;
        let ghost_grid = &self.ghost_grid;
        let densities = &self.densities;
        let ghost_densities = &self.ghost_densities;
        let ghost_positions = &self.ghost_positions;
        let lbox = self.lbox;
        let patch_size = self.patch_size;
        let gas_constant = self.gas_constant;
        let rest_density = self.rest_density;
        let particle_mass = self.particle_mass;
        let ghost_particle_mass = self.ghost_particle_mass;

        self.forces.par_iter_mut().enumerate().for_each(|(i, oforce)| {
            let par_coords = coord_to_grid(positions[i], patch_size);
            let mut force = glm::Vec4::new(0.0, 0.0, 0.0, 0.0);
            let cpressure = gas_constant * (densities[i] - rest_density).max(0.0);
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord) {
                            for &index in indices {
                                let mut inside_result =
                                    particle_mass * kernel.spiky_gradient(&(positions[i] - positions[index]));
                                let opressure = gas_constant * (densities[index] - rest_density).max(0.0);
                                inside_result *= (cpressure + opressure) / (2.0 * densities[index]);
                                force -= inside_result;
                            }
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord) {
                            for &index in indices {
                                let mut inside_result = ghost_particle_mass
                                    * kernel.spiky_gradient(&(positions[i] - ghost_positions[index]));
                                let opressure = gas_constant * (ghost_densities[index] - rest_density).max(0.0);
                                inside_result *= (cpressure + opressure) / (2.0 * ghost_densities[index]);
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
        let ghost_densities = &self.ghost_densities;
        let grid = &self.grid;
        let ghost_grid = &self.ghost_grid;
        let ghost_positions = &self.ghost_positions;
        let lbox = self.lbox;
        let patch_size = self.patch_size;
        let viscosity_constant = self.viscosity_constant;
        let particle_mass = self.particle_mass;
        let ghost_particle_mass = self.ghost_particle_mass;

        self.forces.par_iter_mut().enumerate().for_each(|(i, oforce)| {
            let par_coords = coord_to_grid(positions[i], patch_size);
            let mut force = glm::Vec4::new(0.0, 0.0, 0.0, 0.0);
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord) {
                            for &index in indices {
                                let inside_result = kernel.viscosity_laplacian(&(positions[i] - positions[index]));
                                let inside_force = (velocities[index] - velocities[i]) / densities[index];
                                force += inside_result * inside_force * particle_mass;
                            }
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord) {
                            for &index in indices {
                                let inside_result =
                                    kernel.viscosity_laplacian(&(positions[i] - ghost_positions[index]));
                                let inside_force = -velocities[i] / ghost_densities[index] * ghost_particle_mass;
                                force += inside_result * inside_force;
                            }
                        }
                    }
                }
            }
            force *= viscosity_constant;
            *oforce += force;
        })
    }
    pub fn load_gravity(&mut self) {
        let gravity = glm::Vec4::new(0.0, -1.0, 0.0, 0.0) * self.gconstant * self.particle_mass;
        self.forces.par_iter_mut().for_each(|oforce| {
            *oforce += gravity;
        })
    }
    pub fn load_surface_tension(&mut self) {
        let kernel = Kernels::new(self.h_value());
        let h = kernel.h;
        let positions = &self.positions;
        let densities = &self.densities;
        let ghost_densities = &self.ghost_densities;
        let grid = &self.grid;
        let ghost_grid = &self.ghost_grid;
        let ghost_positions = &self.ghost_positions;
        let lbox = self.lbox;
        let patch_size = self.patch_size;
        let particle_mass = self.particle_mass;
        let ghost_particle_mass = self.ghost_particle_mass;
        let tension_constant = self.tension_constant;

        self.forces.par_iter_mut().enumerate().for_each(|(i, oforce)| {
            let par_coords = coord_to_grid(positions[i], patch_size);
            let mut surface_normal = glm::Vec4::new(0.0, 0.0, 0.0, 0.0);
            let mut curvature_term = 0.0;
            for a in (-lbox)..=(lbox) {
                for b in (-lbox)..=(lbox) {
                    for c in (-lbox)..=(lbox) {
                        let other_coord = (par_coords.0 + a, par_coords.1 + b, par_coords.2 + c);
                        if let Some(indices) = grid.get(&other_coord) {
                            for &index in indices {
                                surface_normal += particle_mass
                                    * (1.0 / densities[index])
                                    * kernel.poly6_gradient(&(positions[i] - positions[index]));
                                curvature_term += particle_mass
                                    * (1.0 / densities[index])
                                    * kernel.poly6_laplacian(&(positions[i] - positions[index]));
                            }
                        }
                        if let Some(indices) = ghost_grid.get(&other_coord) {
                            for &index in indices {
                                surface_normal += ghost_particle_mass
                                    * (1.0 / ghost_densities[index])
                                    * kernel.poly6_gradient(&(positions[i] - ghost_positions[index]));
                                curvature_term += ghost_particle_mass
                                    * (1.0 / ghost_densities[index])
                                    * kernel.poly6_laplacian(&(positions[i] - ghost_positions[index]));
                            }
                        }
                    }
                }
            }
            if glm::length(&surface_normal) < h * 0.1 {
                return;
            }
            let curvature_constant = -curvature_term / glm::length(&surface_normal);
            *oforce += tension_constant * curvature_constant * surface_normal;
        })
    }
    pub fn set_calculation_type_update_ubo(&mut self, calc_type: CalculationType) {
        self.water_ubo.calculation_type = calc_type;
        unsafe {
            gl::NamedBufferSubData(
                self.ubo,
                0,
                size_of::<WaterUbo>() as isize,
                &self.water_ubo as *const WaterUbo as *const c_void,
            );
        }
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
        let aspect_ratio = (f32!(state.frame_dims.0)) / (f32!(state.frame_dims.1));
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
            gl::ProgramUniform1i(state.def_vshader_id, 4, i32!(true));
            gl::ProgramUniform1i(state.def_vshader_id, 5, 1); // Mode: Billboard Quad
            gl::ProgramUniform1f(state.def_vshader_id, 6, self.radius);
            gl::ProgramUniform3f(
                state.def_fshader_id,
                0,
                state.cam_state.pos.x,
                state.cam_state.pos.y,
                state.cam_state.pos.z,
            );
            gl::ProgramUniform1i(state.def_fshader_id, 1, 3); // Mode: Sphere
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
    pub fn poly6(&self, r: &glm::Vec4) -> f32 {
        let r2 = glm::dot(r, r);
        let h2 = self.h * self.h;
        if r2 > h2 {
            return 0.0;
        }
        let diff = h2 - r2;
        self.poly6_coeff * diff * diff * diff
    }

    /// Poly6 — gradient. Used for surface tension / color field.
    pub fn poly6_gradient(&self, r: &glm::Vec4) -> glm::Vec4 {
        let r2 = glm::dot(r, r);
        let h2 = self.h * self.h;
        if r2 > h2 {
            return glm::zero();
        }
        let diff = h2 - r2;
        r * (self.poly6_grad_coeff * diff * diff)
    }

    /// Poly6 — Laplacian. Used for surface tension curvature.
    pub fn poly6_laplacian(&self, r: &glm::Vec4) -> f32 {
        let r2 = glm::dot(r, r);
        let h2 = self.h * self.h;
        if r2 > h2 {
            return 0.0;
        }
        let diff = h2 - r2;
        self.poly6_lap_coeff * diff * (7.0 * r2 - 3.0 * h2)
    }

    /// Spiky — gradient. Used for pressure.
    pub fn spiky_gradient(&self, r: &glm::Vec4) -> glm::Vec4 {
        let len = glm::length(r);
        if len > self.h || len < 1e-6 {
            return glm::zero();
        }
        let diff = self.h - len;
        r * (self.spiky_grad_coeff * diff * diff / len)
    }

    /// Viscosity — Laplacian. Used for viscosity.
    pub fn viscosity_laplacian(&self, r: &glm::Vec4) -> f32 {
        let len = glm::length(r);
        if len > self.h {
            return 0.0;
        }
        self.visc_lap_coeff * (1.0 - len / self.h)
    }
}
