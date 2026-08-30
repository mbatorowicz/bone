//! Periodyczny Particle-Mesh: CIC, FFT, jądro Hockneya, gather.

use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::units::G;

pub struct ParticleMesh {
    pub ng: usize,
    pub box_size: f32,
    pub rho_bar: f32,
    density: Vec<f32>,
    force: Vec<[f32; 3]>,
    scratch: Vec<Complex<f32>>,
    planner: FftPlanner<f32>,
}

impl ParticleMesh {
    pub fn new(ng: usize, box_size: f32, rho_bar: f32) -> Self {
        let n = ng * ng * ng;
        Self {
            ng,
            box_size,
            rho_bar,
            density: vec![0.0; n],
            force: vec![[0.0; 3]; n],
            scratch: vec![Complex::new(0.0, 0.0); n],
            planner: FftPlanner::new(),
        }
    }

    pub fn cell(&self) -> f32 {
        self.box_size / self.ng as f32
    }

    pub fn deposit(&mut self, x: &[[f32; 3]], mass: f32) {
        self.density.fill(0.0);
        let ng = self.ng as i32;
        let h = self.cell();
        let inv_h = 1.0 / h;
        let vol = h * h * h;
        for p in x {
            let mut gx = [0.0f32; 3];
            let mut i0 = [0i32; 3];
            for d in 0..3 {
                let u = (p[d] * inv_h).rem_euclid(ng as f32);
                i0[d] = u.floor() as i32;
                gx[d] = u - i0[d] as f32;
            }
            for oz in 0..2 {
                let wz = if oz == 0 { 1.0 - gx[2] } else { gx[2] };
                let iz = (i0[2] + oz).rem_euclid(ng);
                for oy in 0..2 {
                    let wy = if oy == 0 { 1.0 - gx[1] } else { gx[1] };
                    let iy = (i0[1] + oy).rem_euclid(ng);
                    for ox in 0..2 {
                        let wx = if ox == 0 { 1.0 - gx[0] } else { gx[0] };
                        let ix = (i0[0] + ox).rem_euclid(ng);
                        let idx = (iz * ng + iy) * ng + ix;
                        self.density[idx as usize] += mass * wx * wy * wz / vol;
                    }
                }
            }
        }
    }

    /// Φ̃ z ∇²Φ̃ = 4πG ρ̄₀ δ, potem F = −∇Φ̃ w przestrzeni k.
    pub fn solve_forces(&mut self) {
        let ng = self.ng;
        let n = ng * ng * ng;
        let mean: f32 = self.density.iter().sum::<f32>() / n as f32;
        for i in 0..n {
            self.scratch[i] = Complex::new(self.density[i] - mean, 0.0);
        }
        fft3(&mut self.planner, &mut self.scratch, ng, false);

        let l = self.box_size;
        let dk = 2.0 * std::f32::consts::PI / l;
        let green0 = -4.0 * std::f32::consts::PI * G as f32 * self.rho_bar;
        let half = ng as i32 / 2;
        for iz in 0..ng {
            let kz = wave(iz, ng, half) * dk;
            for iy in 0..ng {
                let ky = wave(iy, ng, half) * dk;
                for ix in 0..ng {
                    let kx = wave(ix, ng, half) * dk;
                    let idx = (iz * ng + iy) * ng + ix;
                    let k2 = kx * kx + ky * ky + kz * kz;
                    if k2 < 1e-20 {
                        self.scratch[idx] = Complex::new(0.0, 0.0);
                    } else {
                        // dekonwolucja CIC: sinc⁴
                        let sx = sinc(0.5 * kx * (l / ng as f32));
                        let sy = sinc(0.5 * ky * (l / ng as f32));
                        let sz = sinc(0.5 * kz * (l / ng as f32));
                        let dec = (sx * sy * sz).powi(2).max(1e-6);
                        self.scratch[idx] *= green0 / (k2 * dec);
                    }
                }
            }
        }

        // gradient w k: F_i = −i k_i Φ
        let phi = self.scratch.clone();
        for axis in 0..3 {
            for iz in 0..ng {
                let kz = wave(iz, ng, half) * dk;
                for iy in 0..ng {
                    let ky = wave(iy, ng, half) * dk;
                    for ix in 0..ng {
                        let kx = wave(ix, ng, half) * dk;
                        let k = [kx, ky, kz][axis];
                        let idx = (iz * ng + iy) * ng + ix;
                        // −i k Φ = k * (Im Φ, −Re Φ)  bo −i(a+ib)= b - i a
                        self.scratch[idx] =
                            Complex::new(k * phi[idx].im, -k * phi[idx].re);
                    }
                }
            }
            fft3(&mut self.planner, &mut self.scratch, ng, true);
            let norm = 1.0 / n as f32;
            for i in 0..n {
                self.force[i][axis] = self.scratch[i].re * norm;
            }
        }
    }

    pub fn gather(&self, x: &[[f32; 3]], acc: &mut [[f32; 3]]) {
        let ng = self.ng as i32;
        let inv_h = 1.0 / self.cell();
        acc.par_iter_mut()
            .zip(x.par_iter())
            .for_each(|(out, p)| {
                let mut gx = [0.0f32; 3];
                let mut i0 = [0i32; 3];
                for d in 0..3 {
                    let u = (p[d] * inv_h).rem_euclid(ng as f32);
                    i0[d] = u.floor() as i32;
                    gx[d] = u - i0[d] as f32;
                }
                let mut f = [0.0f32; 3];
                for oz in 0..2 {
                    let wz = if oz == 0 { 1.0 - gx[2] } else { gx[2] };
                    let iz = (i0[2] + oz).rem_euclid(ng);
                    for oy in 0..2 {
                        let wy = if oy == 0 { 1.0 - gx[1] } else { gx[1] };
                        let iy = (i0[1] + oy).rem_euclid(ng);
                        for ox in 0..2 {
                            let w = (if ox == 0 { 1.0 - gx[0] } else { gx[0] }) * wy * wz;
                            let ix = (i0[0] + ox).rem_euclid(ng);
                            let idx = ((iz * ng + iy) * ng + ix) as usize;
                            let g = self.force[idx];
                            f[0] += w * g[0];
                            f[1] += w * g[1];
                            f[2] += w * g[2];
                        }
                    }
                }
                *out = f;
            });
    }

    pub fn density_at(&self, p: [f32; 3]) -> f32 {
        let ng = self.ng as i32;
        let inv_h = 1.0 / self.cell();
        let mut gx = [0.0f32; 3];
        let mut i0 = [0i32; 3];
        for d in 0..3 {
            let u = (p[d] * inv_h).rem_euclid(ng as f32);
            i0[d] = u.floor() as i32;
            gx[d] = u - i0[d] as f32;
        }
        let mut rho = 0.0;
        for oz in 0..2 {
            let wz = if oz == 0 { 1.0 - gx[2] } else { gx[2] };
            let iz = (i0[2] + oz).rem_euclid(ng);
            for oy in 0..2 {
                let wy = if oy == 0 { 1.0 - gx[1] } else { gx[1] };
                let iy = (i0[1] + oy).rem_euclid(ng);
                for ox in 0..2 {
                    let w = (if ox == 0 { 1.0 - gx[0] } else { gx[0] }) * wy * wz;
                    let ix = (i0[0] + ox).rem_euclid(ng);
                    let idx = ((iz * ng + iy) * ng + ix) as usize;
                    rho += w * self.density[idx];
                }
            }
        }
        rho / self.rho_bar - 1.0
    }
}

fn wave(i: usize, ng: usize, half: i32) -> f32 {
    let n = i as i32;
    if n > half {
        (n - ng as i32) as f32
    } else {
        n as f32
    }
}

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        x.sin() / x
    }
}

fn fft3(planner: &mut FftPlanner<f32>, data: &mut [Complex<f32>], ng: usize, inverse: bool) {
    let fft = if inverse {
        planner.plan_fft_inverse(ng)
    } else {
        planner.plan_fft_forward(ng)
    };
    let mut line = vec![Complex::new(0.0, 0.0); ng];
    // x
    for z in 0..ng {
        for y in 0..ng {
            for x in 0..ng {
                line[x] = data[(z * ng + y) * ng + x];
            }
            fft.process(&mut line);
            for x in 0..ng {
                data[(z * ng + y) * ng + x] = line[x];
            }
        }
    }
    // y
    for z in 0..ng {
        for x in 0..ng {
            for y in 0..ng {
                line[y] = data[(z * ng + y) * ng + x];
            }
            fft.process(&mut line);
            for y in 0..ng {
                data[(z * ng + y) * ng + x] = line[y];
            }
        }
    }
    // z
    for y in 0..ng {
        for x in 0..ng {
            for z in 0..ng {
                line[z] = data[(z * ng + y) * ng + x];
            }
            fft.process(&mut line);
            for z in 0..ng {
                data[(z * ng + y) * ng + x] = line[z];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_grid_no_force() {
        let ng = 16;
        let l = 100.0f32;
        let rho = 1.0f32;
        let mut pm = ParticleMesh::new(ng, l, rho);
        let n = ng * ng * ng;
        let h = l / ng as f32;
        let mass = rho * l * l * l / n as f32;
        let x: Vec<[f32; 3]> = (0..ng)
            .flat_map(|iz| {
                (0..ng).flat_map(move |iy| {
                    (0..ng).map(move |ix| {
                        [
                            (ix as f32 + 0.5) * h,
                            (iy as f32 + 0.5) * h,
                            (iz as f32 + 0.5) * h,
                        ]
                    })
                })
            })
            .collect();
        pm.deposit(&x, mass);
        pm.solve_forces();
        let mut acc = vec![[0.0; 3]; x.len()];
        pm.gather(&x, &mut acc);
        let max_f = acc
            .iter()
            .map(|f| f[0].abs() + f[1].abs() + f[2].abs())
            .fold(0.0f32, f32::max);
        assert!(max_f < 1e-3, "max F={max_f}");
    }
}
