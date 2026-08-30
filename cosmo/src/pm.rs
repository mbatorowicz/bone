//! Izolowany Particle-Mesh: Hockney (zero-padding), CIC, bez periodyczności.

use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::units::G;

/// Puste komórki między chmurą a krawędzią siatki (szablon gradientu ±2).
const EDGE_CELLS: i32 = 3;
const MARGIN: f32 = 0.15;
const DECONV_CLAMP: f32 = 4.0;

pub struct ParticleMesh {
    pub ng: usize,
    pub origin: [f32; 3],
    pub h: f32,
    pub rho_bar: f32,
    pub refits: u32,
    density: Vec<f32>,
    force: Vec<[f32; 3]>,
    potential: Vec<f32>,
    scratch: Vec<Complex<f32>>,
    kernel_ft: Vec<Complex<f32>>,
    planner: FftPlanner<f32>,
}

impl ParticleMesh {
    pub fn new(ng: usize, rho_bar: f32) -> Self {
        let ng = ng.max(16);
        let n = ng * ng * ng;
        let p = 2 * ng;
        let pn = p * p * p;
        Self {
            ng,
            origin: [0.0; 3],
            h: 1.0,
            rho_bar,
            refits: 0,
            density: vec![0.0; n],
            force: vec![[0.0; 3]; n],
            potential: vec![0.0; n],
            scratch: vec![Complex::new(0.0, 0.0); pn],
            kernel_ft: vec![Complex::new(0.0, 0.0); pn],
            planner: FftPlanner::new(),
        }
    }

    fn padded(&self) -> usize {
        2 * self.ng
    }

    fn contains(&self, x: &[[f32; 3]]) -> bool {
        let edge = EDGE_CELLS as f32 - 0.5;
        let hi = self.ng as f32 - EDGE_CELLS as f32 - 0.5;
        for p in x {
            for d in 0..3 {
                let u = (p[d] - self.origin[d]) / self.h;
                if u < edge || u > hi {
                    return false;
                }
            }
        }
        true
    }

    fn needed_h(&self, x: &[[f32; 3]]) -> f32 {
        let (span, _) = cloud_span_center(x);
        let usable = (self.ng as i32 - 2 * EDGE_CELLS).max(1) as f32;
        (span * (1.0 + MARGIN)).max(1e-4) / usable
    }

    fn fit(&mut self, x: &[[f32; 3]]) {
        let (span, center) = cloud_span_center(x);
        let usable = (self.ng as i32 - 2 * EDGE_CELLS).max(1) as f32;
        self.h = (span * (1.0 + MARGIN)).max(1e-4) / usable;
        let half = 0.5 * self.ng as f32 * self.h;
        self.origin = [center[0] - half, center[1] - half, center[2] - half];
        self.rebuild_kernel();
        self.refits += 1;
    }

    pub fn ensure_box(&mut self, x: &[[f32; 3]]) {
        let too_coarse = self.refits > 0 && self.h > 1.8 * self.needed_h(x);
        if self.refits == 0 || too_coarse || !self.contains(x) {
            self.fit(x);
        }
    }

    fn rebuild_kernel(&mut self) {
        let p = self.padded();
        let h = self.h;
        let eps2 = h * h;
        let g = G as f32;
        let h3 = h * h * h;
        let half = (p / 2) as i32;
        for iz in 0..p {
            let z = wrap_dist(iz, p, half) * h;
            for iy in 0..p {
                let y = wrap_dist(iy, p, half) * h;
                for ix in 0..p {
                    let x = wrap_dist(ix, p, half) * h;
                    let r2 = x * x + y * y + z * z;
                    let val = if r2 < 1e-20 {
                        0.0
                    } else {
                        -g / (r2 + eps2).sqrt() * h3
                    };
                    self.scratch[(iz * p + iy) * p + ix] = Complex::new(val, 0.0);
                }
            }
        }
        fft3(&mut self.planner, &mut self.scratch, p, false);
        // Dekonwolucja CIC, przycięta przy Nyquiście (jak w bone mesh).
        for iz in 0..p {
            let wz = sinc_pi(wrap_dist(iz, p, half) / p as f32).powi(2);
            for iy in 0..p {
                let wy = sinc_pi(wrap_dist(iy, p, half) / p as f32).powi(2);
                for ix in 0..p {
                    let wx = sinc_pi(wrap_dist(ix, p, half) / p as f32).powi(2);
                    let window = wx * wy * wz;
                    let corr = (1.0 / (window * window).max(1e-12)).min(DECONV_CLAMP);
                    let idx = (iz * p + iy) * p + ix;
                    self.scratch[idx] *= corr;
                }
            }
        }
        self.kernel_ft.copy_from_slice(&self.scratch);
    }

    /// CIC bez zawijania — cząstka poza siatką jest pomijana (po `ensure_box` nie powinno).
    pub fn deposit(&mut self, x: &[[f32; 3]], mass: f32) {
        self.density.fill(0.0);
        let ng = self.ng as i32;
        let inv_h = 1.0 / self.h;
        let vol = self.h * self.h * self.h;
        for p in x {
            let mut gx = [0.0f32; 3];
            let mut i0 = [0i32; 3];
            let mut inside = true;
            for d in 0..3 {
                let u = (p[d] - self.origin[d]) * inv_h;
                if !u.is_finite() {
                    inside = false;
                    break;
                }
                i0[d] = u.floor() as i32;
                gx[d] = u - i0[d] as f32;
                if i0[d] < 0 || i0[d] > ng - 2 {
                    inside = false;
                    break;
                }
            }
            if !inside {
                continue;
            }
            for oz in 0..2 {
                let wz = if oz == 0 { 1.0 - gx[2] } else { gx[2] };
                let iz = i0[2] + oz;
                for oy in 0..2 {
                    let wy = if oy == 0 { 1.0 - gx[1] } else { gx[1] };
                    let iy = i0[1] + oy;
                    for ox in 0..2 {
                        let wx = if ox == 0 { 1.0 - gx[0] } else { gx[0] };
                        let ix = i0[0] + ox;
                        let idx = ((iz * ng + iy) * ng + ix) as usize;
                        self.density[idx] += mass * wx * wy * wz / vol;
                    }
                }
            }
        }
    }

    /// Φ = (−G/r) ∗ ρ, F = −∇Φ. Splot liniowy przez padding Hockneya — bez periodycznych kopii.
    pub fn solve_forces(&mut self) {
        let ng = self.ng;
        let p = self.padded();
        self.scratch.fill(Complex::new(0.0, 0.0));
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let src = (iz * ng + iy) * ng + ix;
                    let dst = (iz * p + iy) * p + ix;
                    self.scratch[dst] = Complex::new(self.density[src], 0.0);
                }
            }
        }
        fft3(&mut self.planner, &mut self.scratch, p, false);
        for i in 0..p * p * p {
            self.scratch[i] *= self.kernel_ft[i];
        }
        fft3(&mut self.planner, &mut self.scratch, p, true);
        let norm = 1.0 / (p * p * p) as f32;
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let src = (iz * p + iy) * p + ix;
                    let dst = (iz * ng + iy) * ng + ix;
                    self.potential[dst] = self.scratch[src].re * norm;
                }
            }
        }
        self.gradient_4th();
    }

    fn phi_at(&self, ix: i32, iy: i32, iz: i32) -> f32 {
        let ng = self.ng as i32;
        let ix = ix.clamp(0, ng - 1) as usize;
        let iy = iy.clamp(0, ng - 1) as usize;
        let iz = iz.clamp(0, ng - 1) as usize;
        self.potential[(iz * self.ng + iy) * self.ng + ix]
    }

    fn gradient_4th(&mut self) {
        let ng = self.ng as i32;
        let scale = 1.0 / (12.0 * self.h);
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let idx = (iz * ng + iy) * ng + ix;
                    let dx = self.phi_at(ix - 2, iy, iz) - 8.0 * self.phi_at(ix - 1, iy, iz)
                        + 8.0 * self.phi_at(ix + 1, iy, iz)
                        - self.phi_at(ix + 2, iy, iz);
                    let dy = self.phi_at(ix, iy - 2, iz) - 8.0 * self.phi_at(ix, iy - 1, iz)
                        + 8.0 * self.phi_at(ix, iy + 1, iz)
                        - self.phi_at(ix, iy + 2, iz);
                    let dz = self.phi_at(ix, iy, iz - 2) - 8.0 * self.phi_at(ix, iy, iz - 1)
                        + 8.0 * self.phi_at(ix, iy, iz + 1)
                        - self.phi_at(ix, iy, iz + 2);
                    self.force[idx as usize] = [-dx * scale, -dy * scale, -dz * scale];
                }
            }
        }
    }

    pub fn update(&mut self, x: &[[f32; 3]], mass: f32) {
        self.ensure_box(x);
        self.deposit(x, mass);
        self.solve_forces();
    }

    pub fn gather(&self, x: &[[f32; 3]], acc: &mut [[f32; 3]]) {
        let ng = self.ng as i32;
        let inv_h = 1.0 / self.h;
        let origin = self.origin;
        let force = &self.force;
        acc.par_iter_mut().zip(x.par_iter()).for_each(|(out, p)| {
            let mut gx = [0.0f32; 3];
            let mut i0 = [0i32; 3];
            let mut inside = true;
            for d in 0..3 {
                let u = (p[d] - origin[d]) * inv_h;
                if !u.is_finite() {
                    inside = false;
                    break;
                }
                i0[d] = u.floor() as i32;
                gx[d] = u - i0[d] as f32;
                if i0[d] < 0 || i0[d] > ng - 2 {
                    inside = false;
                    break;
                }
            }
            if !inside {
                *out = [0.0; 3];
                return;
            }
            let mut f = [0.0f32; 3];
            for oz in 0..2 {
                let wz = if oz == 0 { 1.0 - gx[2] } else { gx[2] };
                let iz = i0[2] + oz;
                for oy in 0..2 {
                    let wy = if oy == 0 { 1.0 - gx[1] } else { gx[1] };
                    let iy = i0[1] + oy;
                    for ox in 0..2 {
                        let w = (if ox == 0 { 1.0 - gx[0] } else { gx[0] }) * wy * wz;
                        let ix = i0[0] + ox;
                        let idx = ((iz * ng + iy) * ng + ix) as usize;
                        let g = force[idx];
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
        let inv_h = 1.0 / self.h;
        let mut gx = [0.0f32; 3];
        let mut i0 = [0i32; 3];
        for d in 0..3 {
            let u = (p[d] - self.origin[d]) * inv_h;
            if !u.is_finite() {
                return -1.0;
            }
            i0[d] = u.floor() as i32;
            gx[d] = u - i0[d] as f32;
            if i0[d] < 0 || i0[d] > ng - 2 {
                return -1.0;
            }
        }
        let mut rho = 0.0;
        for oz in 0..2 {
            let wz = if oz == 0 { 1.0 - gx[2] } else { gx[2] };
            let iz = i0[2] + oz;
            for oy in 0..2 {
                let wy = if oy == 0 { 1.0 - gx[1] } else { gx[1] };
                let iy = i0[1] + oy;
                for ox in 0..2 {
                    let w = (if ox == 0 { 1.0 - gx[0] } else { gx[0] }) * wy * wz;
                    let ix = i0[0] + ox;
                    let idx = ((iz * ng + iy) * ng + ix) as usize;
                    rho += w * self.density[idx];
                }
            }
        }
        rho / self.rho_bar.max(1e-20) - 1.0
    }
}

fn cloud_span_center(x: &[[f32; 3]]) -> (f32, [f32; 3]) {
    let mut lo = [f32::MAX; 3];
    let mut hi = [f32::MIN; 3];
    for q in x {
        for d in 0..3 {
            lo[d] = lo[d].min(q[d]);
            hi[d] = hi[d].max(q[d]);
        }
    }
    if !lo[0].is_finite() {
        return (1.0, [0.0; 3]);
    }
    let center = [
        0.5 * (lo[0] + hi[0]),
        0.5 * (lo[1] + hi[1]),
        0.5 * (lo[2] + hi[2]),
    ];
    let span = (hi[0] - lo[0])
        .max(hi[1] - lo[1])
        .max(hi[2] - lo[2])
        .max(1e-4);
    (span, center)
}

fn wrap_dist(i: usize, n: usize, half: i32) -> f32 {
    let v = i as i32;
    if v > half {
        (v - n as i32) as f32
    } else {
        v as f32
    }
}

/// `sin(πx)/(πx)` — konwencja numpy.sinc, do okna CIC.
fn sinc_pi(x: f32) -> f32 {
    if x.abs() < 1e-6 {
        1.0
    } else {
        let px = x * std::f32::consts::PI;
        px.sin() / px
    }
}

fn fft3(planner: &mut FftPlanner<f32>, data: &mut [Complex<f32>], ng: usize, inverse: bool) {
    let fft = if inverse {
        planner.plan_fft_inverse(ng)
    } else {
        planner.plan_fft_forward(ng)
    };
    let mut line = vec![Complex::new(0.0, 0.0); ng];
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
    fn isolated_not_periodic() {
        let mut pm = ParticleMesh::new(32, 1.0);
        let x = [[-9.0, 0.0, 0.0], [9.0, 0.0, 0.0]];
        pm.update(&x, 1.0);
        let mut acc = vec![[0.0; 3]; 2];
        pm.gather(&x, &mut acc);
        assert!(acc[0][0] > 0.0, "lewa ma iść w prawo, dostałem {}", acc[0][0]);
        assert!(acc[1][0] < 0.0, "prawa ma iść w lewo, dostałem {}", acc[1][0]);
    }

    #[test]
    fn isolated_conserves_momentum() {
        let ng = 16;
        let l = 20.0f32;
        let rho = 1.0f32;
        let mut pm = ParticleMesh::new(ng, rho);
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
        pm.update(&x, mass);
        let mut acc = vec![[0.0; 3]; x.len()];
        pm.gather(&x, &mut acc);
        let mut sum = [0.0f32; 3];
        let mut scale = 0.0f32;
        for f in &acc {
            sum[0] += f[0];
            sum[1] += f[1];
            sum[2] += f[2];
            scale += f[0].abs() + f[1].abs() + f[2].abs();
        }
        let resid = (sum[0].abs() + sum[1].abs() + sum[2].abs()) / scale.max(1e-20);
        assert!(resid < 2e-3, "pęd sił residuum={resid}");
    }

    #[test]
    fn refits_when_cloud_expands() {
        let mut pm = ParticleMesh::new(32, 1.0);
        let x: Vec<[f32; 3]> = (0..64)
            .map(|i| {
                let t = i as f32 * 0.1;
                [t, t * 0.3, -t * 0.2]
            })
            .collect();
        pm.update(&x, 1.0);
        let first = pm.refits;
        let far: Vec<[f32; 3]> = x.iter().map(|q| [q[0] * 40.0, q[1] * 40.0, q[2] * 40.0]).collect();
        pm.update(&far, 1.0);
        assert!(pm.refits > first);
    }

    #[test]
    fn deposit_does_not_wrap() {
        let mut pm = ParticleMesh::new(16, 1.0);
        let inside = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        pm.update(&inside, 1.0);
        let origin = pm.origin;
        let h = pm.h;
        // cząstka daleko poza bieżącą siatką, bez refitu: masa nie wraca na brzeg
        pm.deposit(&[[origin[0] - 50.0 * h, 0.0, 0.0]], 1.0);
        let sum: f32 = pm.density.iter().sum();
        assert!(sum.abs() < 1e-6, "zawinięta masa={sum}");
    }
}
