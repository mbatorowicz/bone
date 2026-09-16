//! Siły O(N²) na GPU. Wzór ten sam co [`crate::sr::backends::exact`].

use super::{dispatch, make_storage, read_buffer, shader, with_gpu};
use crate::sr::state::Field;
use crate::vec3::{vec3, Vec3};

pub struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Particle {
    x: f64,
    y: f64,
    z: f64,
    mass: f64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Out {
    fx: f64,
    fy: f64,
    fz: f64,
    phi: f64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    g: f64,
    eps2: f64,
}

pub fn nbody(positions: &[Vec3], masses: &[f64], g: f64, softening: f64) -> Option<Field> {
    let n = positions.len();
    if n != masses.len() || n < 2 {
        return None;
    }
    with_gpu(|gpu| {
        if !gpu.f64 {
            return None;
        }
        if gpu.nbody.is_none() {
            gpu.nbody = Pipeline::new(&gpu.device);
        }
        let pipe = gpu.nbody.as_ref()?;
        pipe.dispatch(&gpu.device, &gpu.queue, positions, masses, g, softening)
    })
}

impl Pipeline {
    fn new(device: &wgpu::Device) -> Option<Self> {
        let module = shader(device, include_str!("nbody.wgsl"))?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nbody-bgl"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, false),
                storage_entry(2, true),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nbody-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("nbody"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("nbody"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(Self { pipeline, layout })
    }

    fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        positions: &[Vec3],
        masses: &[f64],
        g: f64,
        softening: f64,
    ) -> Option<Field> {
        let n = positions.len() as u32;
        let particles: Vec<Particle> = positions
            .iter()
            .zip(masses.iter())
            .map(|(p, m)| Particle {
                x: p.x,
                y: p.y,
                z: p.z,
                mass: *m,
            })
            .collect();
        let params = Params {
            n,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            g,
            eps2: softening * softening,
        };
        let particle_buf = make_storage(
            device,
            bytemuck::cast_slice(&particles),
            wgpu::BufferUsages::STORAGE,
        );
        let out_size = (n as u64) * std::mem::size_of::<Out>() as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("nbody-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let param_buf = make_storage(
            device,
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::STORAGE,
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("nbody-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });
        let groups = n.div_ceil(64);
        dispatch(device, queue, &self.pipeline, &bind, groups, 1);
        let bytes = read_buffer(device, queue, &out_buf, out_size);
        let out: &[Out] = bytemuck::cast_slice(&bytes);
        let mut force = Vec::with_capacity(positions.len());
        let mut potential = Vec::with_capacity(positions.len());
        for row in out {
            force.push(vec3(row.fx, row.fy, row.fz));
            potential.push(row.phi);
        }
        Some(Field { force, potential })
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
