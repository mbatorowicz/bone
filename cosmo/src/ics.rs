//! Warunki początkowe: Gauss + Zel'dovich (1LPT).

use rand::Rng;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rustfft::{num_complex::Complex, FftPlanner};

use crate::cosmology::Cosmology;
use crate::power::PowerSpectrum;

pub struct InitialState {
    pub x: Vec<[f32; 3]>,
    pub p: Vec<[f32; 3]>,
    pub mass: f32,
    pub box_size: f32,
    pub a: f64,
}

pub fn make_initial_state(
    cosmo: Cosmology,
    box_size: f64,
    n_grid: usize,
    z_start: f64,
    seed: u64,
) -> InitialState {
    let a = 1.0 / (1.0 + z_start);
    let power = PowerSpectrum::eisenstein_hu(cosmo);
    let ng = n_grid;
    let n = ng * ng * ng;
    let l = box_size as f32;
    let volume = box_size.powi(3);
    let rho_bar = cosmo.mean_matter_density();
    let mass = (rho_bar * volume / n as f64) as f32;

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut delta = vec![Complex::new(0.0f32, 0.0); n];
    let mut planner = FftPlanner::<f32>::new();
    let half = ng as i32 / 2;
    let dk = 2.0 * std::f64::consts::PI / box_size;

    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let idx = (iz * ng + iy) * ng + ix;
                if ix == 0 && iy == 0 && iz == 0 {
                    delta[idx] = Complex::new(0.0, 0.0);
                    continue;
                }
                let kx = wave(ix, ng, half) as f64 * dk;
                let ky = wave(iy, ng, half) as f64 * dk;
                let kz = wave(iz, ng, half) as f64 * dk;
                let k = (kx * kx + ky * ky + kz * kz).sqrt();
                // P(k, a_start): amplituda już zawiera D(a)²
                let pk = power.p(k, a);
                let amp = (pk * volume / 2.0).sqrt() as f32;
                let u1 = rng.gen::<f32>().max(1e-8);
                let u2 = rng.gen::<f32>();
                let r = (-2.0 * u1.ln()).sqrt();
                let re = r * (2.0 * std::f32::consts::PI * u2).cos();
                let im = r * (2.0 * std::f32::consts::PI * u2).sin();
                delta[idx] = Complex::new(re, im) * amp;
            }
        }
    }
    enforce_hermitian(&mut delta, ng);

    // φ_k = −δ_k / k² , Ψ = ∇φ
    let mut phi = delta.clone();
    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let idx = (iz * ng + iy) * ng + ix;
                let kx = wave(ix, ng, half) as f32 * dk as f32;
                let ky = wave(iy, ng, half) as f32 * dk as f32;
                let kz = wave(iz, ng, half) as f32 * dk as f32;
                let k2 = kx * kx + ky * ky + kz * kz;
                if k2 < 1e-20 {
                    phi[idx] = Complex::new(0.0, 0.0);
                } else {
                    phi[idx] = delta[idx] * (-1.0 / k2);
                }
            }
        }
    }

    let mut psi = vec![[0.0f32; 3]; n];
    let phi_k = phi.clone();
    for axis in 0..3 {
        let mut field = phi_k.clone();
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let kx = wave(ix, ng, half) as f32 * dk as f32;
                    let ky = wave(iy, ng, half) as f32 * dk as f32;
                    let kz = wave(iz, ng, half) as f32 * dk as f32;
                    let k = [kx, ky, kz][axis];
                    let idx = (iz * ng + iy) * ng + ix;
                    // i k φ
                    field[idx] = Complex::new(-k * phi_k[idx].im, k * phi_k[idx].re);
                }
            }
        }
        fft3_inv(&mut planner, &mut field, ng);
        let norm = 1.0 / n as f32;
        for i in 0..n {
            psi[i][axis] = field[i].re * norm;
        }
    }

    let d = cosmo.growth(a) as f32;
    let f = cosmo.growth_rate(a) as f32;
    let h = cosmo.h_of_a(a) as f32;
    // p = a² ẋ , ẋ = f H Ψ_phys where Ψ_phys = D Ψ_1  (ZA: x = q + D Ψ)
    // ẋ = (dD/dt) Ψ = f D H Ψ
    // p = a² f D H Ψ
    let p_scale = (a * a) as f32 * f * d * h;
    let h_cell = l / ng as f32;

    let mut x = vec![[0.0f32; 3]; n];
    let mut p = vec![[0.0f32; 3]; n];
    let mut i = 0;
    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let q = [
                    (ix as f32 + 0.5) * h_cell,
                    (iy as f32 + 0.5) * h_cell,
                    (iz as f32 + 0.5) * h_cell,
                ];
                let disp = [
                    d * psi[i][0],
                    d * psi[i][1],
                    d * psi[i][2],
                ];
                x[i] = [q[0] + disp[0], q[1] + disp[1], q[2] + disp[2]];
                p[i] = [
                    p_scale * psi[i][0],
                    p_scale * psi[i][1],
                    p_scale * psi[i][2],
                ];
                i += 1;
            }
        }
    }

    InitialState {
        x,
        p,
        mass,
        box_size: l,
        a,
    }
}

fn wave(i: usize, ng: usize, half: i32) -> i32 {
    let n = i as i32;
    if n > half {
        n - ng as i32
    } else {
        n
    }
}

fn enforce_hermitian(field: &mut [Complex<f32>], ng: usize) {
    // Uśrednij F(k) i conj F(−k), żeby pole było rzeczywiste.
    let n = ng;
    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let jx = (n - ix) % n;
                let jy = (n - iy) % n;
                let jz = (n - iz) % n;
                let i = (iz * n + iy) * n + ix;
                let j = (jz * n + jy) * n + jx;
                if i < j {
                    let a = field[i];
                    let b = field[j].conj();
                    let m = Complex::new(0.5 * (a.re + b.re), 0.5 * (a.im + b.im));
                    field[i] = m;
                    field[j] = m.conj();
                } else if i == j {
                    field[i].im = 0.0;
                }
            }
        }
    }
}

fn fft3_inv(planner: &mut FftPlanner<f32>, data: &mut [Complex<f32>], ng: usize) {
    let fft = planner.plan_fft_inverse(ng);
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
    fn ic_finite_no_wrap() {
        let st = make_initial_state(Cosmology::planck18(), 100.0, 16, 49.0, 1);
        assert_eq!(st.x.len(), 16 * 16 * 16);
        assert!(st.mass > 0.0);
        let sum_p: f32 = st.p.iter().map(|p| p[0] + p[1] + p[2]).sum();
        assert!(sum_p.abs() < st.p.len() as f32 * 1e-2);
        for q in &st.x {
            assert!(q[0].is_finite() && q[1].is_finite() && q[2].is_finite());
        }
    }
}
