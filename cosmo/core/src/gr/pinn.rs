//! Residual PDE i maleńka sieć. Bez PyTorcha, bez CUDA, bez Einsteina.
//!
//! PINN na tej fali zgaduje funkcję `u(x, t)`, nie metrykę. Wejście to
//! dwie współrzędne, wyjście to liczba. Nauczycielem jest residual:
//! ciepło `u_t − k u_xx` albo fala `u_tt − c² u_xx`. Pochodne biorą się
//! z różniczki, nie z biblioteki autodiff. Kerr, siatka pola i MPI
//! nie wchodzą.

/// Ukryta warstwa: osiem schodków `tanh`. Wizualizacja rysuje każdy.
pub const HIDDEN: usize = 8;
const N_PARAMS: usize = HIDDEN * 4 + 1;

/// Sieć 2→8→1. Suwaki są `f64`. `tanh` jest schodkiem, nie fizyką.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Net {
    w1: [[f64; 2]; HIDDEN],
    b1: [f64; HIDDEN],
    w2: [f64; HIDDEN],
    b2: f64,
}

impl Net {
    pub fn zeros() -> Self {
        Self {
            w1: [[0.0; 2]; HIDDEN],
            b1: [0.0; HIDDEN],
            w2: [0.0; HIDDEN],
            b2: 0.0,
        }
    }

    /// Nieszczęśliwe suwaki: nie są rozwiązaniem ciepła, więc residual żyje.
    pub fn biased() -> Self {
        let mut net = Self::zeros();
        net.w1[0][0] = 0.4;
        net.w1[0][1] = 0.3;
        net.w2[0] = 0.8;
        net.b2 = 0.1;
        net
    }

    /// `h = tanh(W₁ · (x, t) + b₁)`.
    pub fn hidden(self, x: f64, t: f64) -> [f64; HIDDEN] {
        let mut h = [0.0; HIDDEN];
        for (slot, (w1, b1)) in h.iter_mut().zip(self.w1.iter().zip(self.b1)) {
            *slot = (w1[0] * x + w1[1] * t + b1).tanh();
        }
        h
    }

    /// `u(x, t) = W₂ · tanh(W₁ · (x, t) + b₁) + b₂`.
    pub fn eval(self, x: f64, t: f64) -> f64 {
        let mut acc = self.b2;
        for (w2, h) in self.w2.iter().zip(self.hidden(x, t)) {
            acc += w2 * h;
        }
        acc
    }
}

/// `u = sin(πx) exp(−π² k t)`. Residual ciepła znika.
pub fn heat_exact(k: f64, x: f64, t: f64) -> f64 {
    (std::f64::consts::PI * x).sin() * (-std::f64::consts::PI.powi(2) * k * t).exp()
}

/// `u = sin(πx) cos(c π t)`. Residual fali znika.
pub fn wave_exact(c: f64, x: f64, t: f64) -> f64 {
    (std::f64::consts::PI * x).sin() * (c * std::f64::consts::PI * t).cos()
}

/// `R = u_t − k u_xx`.
pub fn heat_residual<F>(k: f64, x: f64, t: f64, u: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let h = 1.0e-5;
    let u_t = (u(x, t + h) - u(x, t - h)) / (2.0 * h);
    let u_xx = (u(x + h, t) - 2.0 * u(x, t) + u(x - h, t)) / (h * h);
    u_t - k * u_xx
}

/// `R = u_tt − c² u_xx`.
pub fn wave_residual<F>(c: f64, x: f64, t: f64, u: F) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let h = 1.0e-4;
    let u_tt = (u(x, t + h) - 2.0 * u(x, t) + u(x, t - h)) / (h * h);
    let u_xx = (u(x + h, t) - 2.0 * u(x, t) + u(x - h, t)) / (h * h);
    u_tt - c * c * u_xx
}

/// Średnia `R²` ciepła na punktach kolokacji.
pub fn heat_loss(net: Net, k: f64, points: &[(f64, f64)]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }
    let mut s = 0.0;
    for &(x, t) in points {
        let r = heat_residual(k, x, t, |xx, tt| net.eval(xx, tt));
        s += r * r;
    }
    s / points.len() as f64
}

/// Jeden krok spadku po residualu ciepła. Różniczka po suwakach, nie autodiff.
pub fn step_heat(net: &mut Net, k: f64, lr: f64, points: &[(f64, f64)]) {
    let p0 = pack(*net);
    let loss0 = heat_loss(*net, k, points);
    let eps = 1.0e-6;
    let mut p = p0;
    for (slot, dslot) in p.iter_mut().zip(0..N_PARAMS) {
        let mut trial = p0;
        trial[dslot] += eps;
        let d = (heat_loss(unpack(trial), k, points) - loss0) / eps;
        *slot -= lr * d;
    }
    *net = unpack(p);
}

fn pack(net: Net) -> [f64; N_PARAMS] {
    let mut p = [0.0; N_PARAMS];
    let mut i = 0;
    for row in net.w1 {
        p[i] = row[0];
        p[i + 1] = row[1];
        i += 2;
    }
    for b in net.b1 {
        p[i] = b;
        i += 1;
    }
    for w in net.w2 {
        p[i] = w;
        i += 1;
    }
    p[i] = net.b2;
    p
}

fn unpack(p: [f64; N_PARAMS]) -> Net {
    let mut net = Net::zeros();
    let mut i = 0;
    for row in net.w1.iter_mut() {
        row[0] = p[i];
        row[1] = p[i + 1];
        i += 2;
    }
    for b in net.b1.iter_mut() {
        *b = p[i];
        i += 1;
    }
    for w in net.w2.iter_mut() {
        *w = p[i];
        i += 1;
    }
    net.b2 = p[i];
    net
}

#[cfg(test)]
mod tests {
    use super::*;

    fn collocation() -> Vec<(f64, f64)> {
        let mut pts = Vec::new();
        for ix in 1..6 {
            for it in 0..4 {
                pts.push((ix as f64 / 6.0, it as f64 / 20.0));
            }
        }
        pts
    }

    #[test]
    fn zero_net_evaluates_to_zero() {
        let n = Net::zeros();
        assert_eq!(n.eval(0.3, 0.1), 0.0);
        assert_eq!(n.eval(-1.0, 2.0), 0.0);
    }

    #[test]
    fn biased_hidden_matches_tanh_of_first_row() {
        let n = Net::biased();
        let h = n.hidden(1.0, 0.0);
        assert!((h[0] - 0.4_f64.tanh()).abs() < 1e-15);
        assert_eq!(h[1], 0.0);
        assert!((n.eval(1.0, 0.0) - (0.8 * 0.4_f64.tanh() + 0.1)).abs() < 1e-15);
    }

    #[test]
    fn known_weights_evaluate_to_a_finite_number() {
        let mut n = Net::zeros();
        n.w1[0][0] = 0.5;
        n.w2[0] = 2.0;
        n.b2 = -0.25;
        let u = n.eval(0.0, 0.0);
        assert!((u - (2.0 * 0.0_f64.tanh() - 0.25)).abs() < 1.0e-15);
        assert!(n.eval(0.4, 0.2).is_finite());
    }

    #[test]
    fn analytic_heat_residual_vanishes() {
        let k = 0.3;
        for &(x, t) in &[(0.25, 0.05), (0.5, 0.0), (0.8, 0.1)] {
            let r = heat_residual(k, x, t, |xx, tt| heat_exact(k, xx, tt));
            assert!(r.abs() < 2.0e-6, "R={r} at ({x},{t})");
        }
    }

    #[test]
    fn analytic_wave_residual_vanishes() {
        let c = 1.2;
        for &(x, t) in &[(0.25, 0.05), (0.5, 0.1), (0.7, 0.0)] {
            let r = wave_residual(c, x, t, |xx, tt| wave_exact(c, xx, tt));
            assert!(r.abs() < 2.0e-4, "R={r} at ({x},{t})");
        }
    }

    #[test]
    fn heat_loss_of_exact_solution_is_tiny() {
        let k = 0.25;
        let pts = collocation();
        let mut s = 0.0;
        for &(x, t) in &pts {
            let r = heat_residual(k, x, t, |xx, tt| heat_exact(k, xx, tt));
            s += r * r;
        }
        assert!(s / (pts.len() as f64) < 1.0e-11);
    }

    #[test]
    fn heat_step_decreases_loss() {
        let k = 0.4;
        let pts = collocation();
        let mut net = Net::biased();
        let before = heat_loss(net, k, &pts);
        assert!(before > 1.0e-6, "biased net powinien kłamać, loss={before}");
        for _ in 0..8 {
            step_heat(&mut net, k, 0.05, &pts);
        }
        let after = heat_loss(net, k, &pts);
        assert!(after < before, "loss nie spadł: {before} → {after}");
        assert!(after.is_finite());
    }

    #[test]
    fn pack_round_trip_preserves_weights() {
        let n = Net::biased();
        assert_eq!(unpack(pack(n)), n);
    }
}
