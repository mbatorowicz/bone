//! Klasyczny stepper Rungego–Kutty 4. rzędu na wektorze stanu.
//!
//! Geodezyjna jeszcze tu nie mieszka: ten plik umie tylko zrobić krok
//! `y ← y + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)` dla dowolnego `y' = f(t, y)`.
//! Bez tego filmu z błędem Eulera vs RK4 nie miałby skąd wziąć liczb.
//!
//! `f` dostaje osobny bufor na pochodną, a [`Scratch`] trzyma k-i między
//! wywołaniami, żeby kolejny krok (orbita) nie alokował czterech wektorów
//! na każdy parametr afiniczny.

/// Bufor roboczy jednego kroku RK4.
///
/// Pola są prywatne, bo mieszanie `k₁` z `y` przez pomyłkę psuje rząd metody
/// w sposób, którego test `y' = y` vs `exp` nie zawsze złapie od razu.
#[derive(Clone, Debug, Default)]
pub struct Scratch {
    k1: Vec<f64>,
    k2: Vec<f64>,
    k3: Vec<f64>,
    k4: Vec<f64>,
    ytmp: Vec<f64>,
}

impl Scratch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_len(n: usize) -> Self {
        Self {
            k1: vec![0.0; n],
            k2: vec![0.0; n],
            k3: vec![0.0; n],
            k4: vec![0.0; n],
            ytmp: vec![0.0; n],
        }
    }

    fn ensure(&mut self, n: usize) {
        if self.k1.len() == n {
            return;
        }
        self.k1.resize(n, 0.0);
        self.k2.resize(n, 0.0);
        self.k3.resize(n, 0.0);
        self.k4.resize(n, 0.0);
        self.ytmp.resize(n, 0.0);
    }
}

/// Jeden krok RK4. Stan `y` jest nadpisywany w miejscu.
///
/// `f(t, y, dy)` zapisuje `y'(t)` do `dy`. Długość `dy` jest tą samą co `y`.
/// Pusty stan to no-op: nie ma składowej, którą można by zepsuć.
pub fn step<F>(t: f64, y: &mut [f64], h: f64, scratch: &mut Scratch, mut f: F)
where
    F: FnMut(f64, &[f64], &mut [f64]),
{
    let n = y.len();
    scratch.ensure(n);
    if n == 0 {
        return;
    }

    let Scratch {
        k1,
        k2,
        k3,
        k4,
        ytmp,
    } = scratch;

    f(t, y, k1);

    for i in 0..n {
        ytmp[i] = y[i] + 0.5 * h * k1[i];
    }
    f(t + 0.5 * h, ytmp, k2);

    for i in 0..n {
        ytmp[i] = y[i] + 0.5 * h * k2[i];
    }
    f(t + 0.5 * h, ytmp, k3);

    for i in 0..n {
        ytmp[i] = y[i] + h * k3[i];
    }
    f(t + h, ytmp, k4);

    let sixth = h / 6.0;
    for i in 0..n {
        y[i] += sixth * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

/// `n` kroków RK4 od `y0` w `t0`. Zwraca stan po `n · h`.
pub fn integrate<F>(t0: f64, y0: &[f64], h: f64, n: usize, mut f: F) -> Vec<f64>
where
    F: FnMut(f64, &[f64], &mut [f64]),
{
    let mut y = y0.to_vec();
    let mut scratch = Scratch::with_len(y.len());
    let mut t = t0;
    for _ in 0..n {
        step(t, &mut y, h, &mut scratch, &mut f);
        t += h;
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp_rhs(_t: f64, y: &[f64], dy: &mut [f64]) {
        dy[0] = y[0];
    }

    fn euler_integrate(y0: f64, h: f64, n: usize) -> f64 {
        let mut y = y0;
        for _ in 0..n {
            y += h * y;
        }
        y
    }

    #[test]
    fn empty_state_is_noop() {
        let mut y: [f64; 0] = [];
        let mut scratch = Scratch::new();
        step(0.0, &mut y, 0.1, &mut scratch, |_t, _y, _dy| unreachable!());
        assert!(y.is_empty());
    }

    #[test]
    fn zero_step_leaves_state() {
        let mut y = [3.0, -1.5];
        let mut scratch = Scratch::with_len(2);
        step(0.0, &mut y, 0.0, &mut scratch, |_t, _y, dy| {
            dy[0] = 99.0;
            dy[1] = -99.0;
        });
        assert_eq!(y, [3.0, -1.5]);
    }

    #[test]
    fn one_step_evaluates_rhs_four_times() {
        let mut y = [1.0];
        let mut scratch = Scratch::new();
        let mut calls = 0;
        step(0.0, &mut y, 0.1, &mut scratch, |_t, y, dy| {
            calls += 1;
            dy[0] = y[0];
        });
        assert_eq!(calls, 4);
    }

    #[test]
    fn exponential_matches_e_at_unit_time() {
        let h = 0.01;
        let n = 100;
        let y = integrate(0.0, &[1.0], h, n, exp_rhs);
        assert!(
            (y[0] - std::f64::consts::E).abs() < 1e-8,
            "RK4 y'=y dało {} zamiast e",
            y[0]
        );
    }

    #[test]
    fn rk4_beats_euler_on_exponential() {
        let h = 0.1;
        let n = 10;
        let rk = integrate(0.0, &[1.0], h, n, exp_rhs)[0];
        let eu = euler_integrate(1.0, h, n);
        let want = std::f64::consts::E;
        assert!(
            (rk - want).abs() < (eu - want).abs(),
            "RK4 |Δ|={} Euler |Δ|={}",
            (rk - want).abs(),
            (eu - want).abs()
        );
    }

    #[test]
    fn constant_derivative_is_exact() {
        let y = integrate(0.0, &[0.0, 1.0], 0.25, 4, |_t, _y, dy| {
            dy[0] = 2.0;
            dy[1] = -3.0;
        });
        assert!((y[0] - 2.0).abs() < 1e-15);
        assert!((y[1] + 2.0).abs() < 1e-15);
    }

    #[test]
    fn harmonic_oscillator_closes_a_half_turn() {
        let t_end = std::f64::consts::PI;
        let n = 80;
        let h = t_end / n as f64;
        let y = integrate(0.0, &[1.0, 0.0], h, n, |_t, y, dy| {
            dy[0] = y[1];
            dy[1] = -y[0];
        });
        assert!((y[0] + 1.0).abs() < 1e-6, "x(π) = {}, want −1", y[0]);
        assert!(y[1].abs() < 1e-6, "v(π) = {}, want 0", y[1]);
    }

    #[test]
    fn scratch_resizes_to_state() {
        let mut scratch = Scratch::new();
        let mut y = [1.0, 2.0, 3.0];
        step(0.0, &mut y, 0.0, &mut scratch, |_t, _y, dy| {
            dy.fill(0.0);
        });
        assert_eq!(scratch.k1.len(), 3);
    }
}
