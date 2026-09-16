//! Trójwymiarowa FFT radix-2 na GPU. Rozmiar musi być potęgą dwójki.
//!
//! Siatki 3·2ᵏ (96, 192, 384) zostają na rustfft — mieszane radiksy w shaderze
//! to osobny kawałek, a błąd w osi y/z psuje solver PM ciszej niż O(N²).

use super::{dispatch, make_storage, read_buffer, shader, with_gpu};
use crate::fft::Direction;
use rustfft::num_complex::Complex;

pub struct Pipeline {
    bitrev: wgpu::ComputePipeline,
    butterfly: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FftParams {
    n: u32,
    stride: u32,
    n_lines: u32,
    stage: u32,
    inverse: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub fn fft3(data: &mut [Complex<f32>], ng: usize, dir: Direction) -> bool {
    if ng < 16 || !ng.is_power_of_two() || data.len() != ng * ng * ng {
        return false;
    }
    with_gpu(|gpu| {
        if gpu.fft.is_none() {
            gpu.fft = Pipeline::new(&gpu.device);
        }
        let pipe = gpu.fft.as_ref()?;
        pipe.run(&gpu.device, &gpu.queue, data, ng, dir)
    })
    .is_some()
}

impl Pipeline {
    fn new(device: &wgpu::Device) -> Option<Self> {
        let module = shader(device, include_str!("fft.wgsl"))?;
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fft-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fft-pl"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let bitrev = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft-bitrev"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("bitrev"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let butterfly = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft-butterfly"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("butterfly"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Some(Self {
            bitrev,
            butterfly,
            layout,
        })
    }

    fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &mut [Complex<f32>],
        ng: usize,
        dir: Direction,
    ) -> Option<()> {
        let n = ng as u32;
        let inverse = u32::from(dir == Direction::Inverse);
        let mut packed = Vec::with_capacity(data.len() * 2);
        for z in data.iter() {
            packed.push(z.re);
            packed.push(z.im);
        }
        let buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft-data"),
            size: (packed.len() * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&packed));
        let n_lines = n * n;
        let axes = [(1_u32, n), (n, n), (n * n, n)];
        let logn = n.trailing_zeros();
        for (stride, _) in axes {
            self.pass(
                device,
                queue,
                &buf,
                FftParams {
                    n,
                    stride,
                    n_lines,
                    stage: 0,
                    inverse,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                },
                true,
                n_lines,
            );
            for stage in 0..logn {
                let butterflies = n / 2;
                self.pass(
                    device,
                    queue,
                    &buf,
                    FftParams {
                        n,
                        stride,
                        n_lines,
                        stage,
                        inverse,
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                    false,
                    n_lines * butterflies,
                );
            }
        }
        let bytes = read_buffer(device, queue, &buf, (packed.len() * 4) as u64);
        let out: &[f32] = bytemuck::cast_slice(&bytes);
        for (i, slot) in data.iter_mut().enumerate() {
            *slot = Complex::new(out[2 * i], out[2 * i + 1]);
        }
        Some(())
    }

    fn pass(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &wgpu::Buffer,
        params: FftParams,
        bitrev: bool,
        jobs: u32,
    ) {
        let param_buf = make_storage(
            device,
            bytemuck::bytes_of(&params),
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fft-bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: data.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: param_buf.as_entire_binding(),
                },
            ],
        });
        let pipe = if bitrev {
            &self.bitrev
        } else {
            &self.butterfly
        };
        dispatch(device, queue, pipe, &bind, jobs.div_ceil(64), 1);
    }
}
