//! Liczenie na GPU przez wgpu (Vulkan, f64 gdy karta umie).
//!
//! CUDA nie wchodzi: na tej maszynie nie ma toolkit'u, a okno już stoi na wgpu.
//! Brak karty albo brak `SHADER_F64` wraca na CPU — testy i bieg wsadowy
//! nie mogą zależeć od sterownika.

mod fft;
mod nbody;
mod trace;

use std::sync::{Mutex, OnceLock};

use wgpu::util::DeviceExt;

pub use fft::fft3;
pub use nbody::nbody;
pub use trace::raytrace_schwarzschild;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    f64: bool,
    nbody: Option<nbody::Pipeline>,
    fft: Option<fft::Pipeline>,
    trace: Option<trace::Pipeline>,
}

static GPU: OnceLock<Option<Mutex<Gpu>>> = OnceLock::new();

fn gpu() -> Option<&'static Mutex<Gpu>> {
    GPU.get_or_init(|| Gpu::init().map(Mutex::new)).as_ref()
}

impl Gpu {
    fn init() -> Option<Self> {
        pollster::block_on(Self::init_async())
    }

    async fn init_async() -> Option<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
            ..Default::default()
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await?;
        let f64 = adapter.features().contains(wgpu::Features::SHADER_F64);
        let mut required = wgpu::Features::empty();
        if f64 {
            required |= wgpu::Features::SHADER_F64;
        }
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("bone-gpu"),
                    required_features: required,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .ok()?;
        Some(Self {
            device,
            queue,
            f64,
            nbody: None,
            fft: None,
            trace: None,
        })
    }
}

/// Czy silnik ma urządzenie GPU. Nie mówi, czy f64 działa.
pub fn available() -> bool {
    gpu().is_some()
}

/// Etykieta backendu do `describe`: `GPU f64`, `GPU f32` albo pusto.
pub fn label() -> &'static str {
    match gpu() {
        Some(g) => {
            if g.lock().map(|g| g.f64).unwrap_or(false) {
                "GPU f64"
            } else {
                "GPU f32"
            }
        }
        None => "",
    }
}

fn with_gpu<T>(f: impl FnOnce(&mut Gpu) -> Option<T>) -> Option<T> {
    let lock = gpu()?;
    let mut gpu = lock.lock().ok()?;
    f(&mut gpu)
}

fn make_storage(device: &wgpu::Device, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents,
        usage,
    })
}

fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, size: u64) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bone-gpu-read"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bone-gpu-copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, size);
    queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::Maintain::Wait);
    rx.recv().expect("map_async").expect("map");
    let data = slice.get_mapped_range().to_vec();
    staging.unmap();
    data
}

fn dispatch(device: &wgpu::Device, queue: &wgpu::Queue, pipeline: &wgpu::ComputePipeline, bind: &wgpu::BindGroup, x: u32, y: u32) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("bone-gpu-dispatch"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("bone-gpu-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind, &[]);
        pass.dispatch_workgroups(x.max(1), y.max(1), 1);
    }
    queue.submit(Some(encoder.finish()));
}

fn shader(device: &wgpu::Device, source: &str) -> Option<wgpu::ShaderModule> {
    device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let err = pollster::block_on(device.pop_error_scope());
    if let Some(err) = err {
        eprintln!("bone-gpu shader: {err}");
        return None;
    }
    Some(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sr::backends::exact::{forces_for_rows, Exact};
    use crate::sr::backends::Backend;
    use crate::vec3::vec3;

    #[test]
    fn gpu_nbody_matches_cpu_when_device_exists() {
        let Some(_) = gpu() else {
            return;
        };
        let x: Vec<_> = (0..80)
            .map(|i| {
                let t = i as f64 * 0.41;
                vec3(t.sin() * 3.0, t.cos() * 2.0, (t * 0.6).sin())
            })
            .collect();
        let m: Vec<_> = (0..80).map(|i| 0.7 + (i % 4) as f64 * 0.2).collect();
        let gpu_field = nbody(&x, &m, 2.0, 0.15).expect("GPU ma policzyć");
        let cpu = Exact::cpu_only().compute(&x, &m, 2.0, 0.15);
        let n = x.len() as f64;
        let rms: f64 = gpu_field
            .force
            .iter()
            .zip(cpu.force.iter())
            .map(|(a, b)| (*a - *b).norm_squared())
            .sum::<f64>()
            / n;
        let typical: f64 = cpu.force.iter().map(|f| f.norm_squared()).sum::<f64>() / n;
        assert!(
            rms.sqrt() / typical.sqrt() < 1e-10,
            "RMS {}",
            rms.sqrt() / typical.sqrt()
        );
        let rows = [0usize, 11, 40, 79];
        let subset = forces_for_rows(&x, &m, 2.0, 0.15, &rows);
        for (k, &i) in rows.iter().enumerate() {
            let d = (subset[k] - gpu_field.force[i]).norm();
            assert!(d / gpu_field.force[i].norm() < 1e-10, "wiersz {i}");
        }
    }
}
