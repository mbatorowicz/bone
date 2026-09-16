//! Raytracer Schwarzschilda na GPU. Kerr (`a ≠ 0`) zostaje na CPU:
//! Γ mieszane `tφ` to drugi shader, a test cienia przy spinie ma zostać
//! przy tym samym wzorze co [`crate::gr::kerr`].

use super::{dispatch, make_storage, read_buffer, shader, with_gpu};
use crate::gr::raytrace::{Buffer, Config, Hit};

pub struct Pipeline {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RtParams {
    width: u32,
    height: u32,
    max_steps: u32,
    _pad: u32,
    mass: f64,
    cam_r: f64,
    cam_theta: f64,
    cam_phi: f64,
    fov_y: f64,
    r_inner: f64,
    r_outer: f64,
    r_escape: f64,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HitOut {
    tag: u32,
    _pad: u32,
    r: f64,
}

pub fn raytrace_schwarzschild(cfg: Config) -> Option<Buffer> {
    if cfg.spin != 0.0 || cfg.width == 0 || cfg.height == 0 {
        return None;
    }
    with_gpu(|gpu| {
        if !gpu.f64 {
            return None;
        }
        if gpu.trace.is_none() {
            gpu.trace = Pipeline::new(&gpu.device);
        }
        let pipe = gpu.trace.as_ref()?;
        pipe.dispatch(&gpu.device, &gpu.queue, cfg)
    })
}

impl Pipeline {
    fn new(device: &wgpu::Device) -> Option<Self> {
        let module = shader(device, include_str!("raytrace.wgsl"))?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rt-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rt-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raytrace"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("raytrace"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(Self { pipeline, layout })
    }

    fn dispatch(&self, device: &wgpu::Device, queue: &wgpu::Queue, cfg: Config) -> Option<Buffer> {
        let n = (cfg.width as u64)
            .checked_mul(cfg.height as u64)
            .filter(|v| *v > 0)?;
        let params = RtParams {
            width: cfg.width,
            height: cfg.height,
            max_steps: cfg.max_steps.min(u32::MAX as usize) as u32,
            _pad: 0,
            mass: cfg.metric.mass(),
            cam_r: cfg.camera.r(),
            cam_theta: cfg.camera.theta(),
            cam_phi: cfg.camera.phi(),
            fov_y: cfg.camera.fov_y(),
            r_inner: cfg.disk.r_inner,
            r_outer: cfg.disk.r_outer,
            r_escape: cfg.camera.r() * 1.15,
        };
        let param_buf = make_storage(
            device,
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::STORAGE,
        );
        let out_size = n * std::mem::size_of::<HitOut>() as u64;
        let out_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rt-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rt-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: param_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out_buf.as_entire_binding(),
                },
            ],
        });
        let gx = cfg.width.div_ceil(8);
        let gy = cfg.height.div_ceil(8);
        dispatch(device, queue, &self.pipeline, &bind, gx, gy);
        let bytes = read_buffer(device, queue, &out_buf, out_size);
        let rows: &[HitOut] = bytemuck::cast_slice(&bytes);
        let mut hits = Vec::with_capacity(n as usize);
        let mut rgba = vec![0_u8; (n as usize).saturating_mul(4)];
        for (i, row) in rows.iter().enumerate() {
            let hit = match row.tag {
                0 => Hit::Horizon,
                2 => Hit::Disk { r: row.r },
                _ => Hit::Escape,
            };
            let c = hit.rgba();
            let o = i * 4;
            rgba[o] = c[0];
            rgba[o + 1] = c[1];
            rgba[o + 2] = c[2];
            rgba[o + 3] = c[3];
            hits.push(hit);
        }
        Some(Buffer {
            width: cfg.width,
            height: cfg.height,
            rgba,
            hits,
        })
    }
}
