//! Jednostki GADGET: Mpc/h, 10¹⁰ M☉/h, km/s.

pub const MPC_KM: f64 = 3.085_677_581_491_367e19;
pub const GYR_S: f64 = 3.155_76e16;
pub const G: f64 = 43.0071;
pub const H100: f64 = 100.0;
pub const GYR_PER_CODE_TIME: f64 = MPC_KM / GYR_S;

pub fn critical_density_0(h: f64) -> f64 {
    assert!(h > 0.0, "h musi być dodatnie, dostałem {h}");
    3.0 * H100 * H100 / (8.0 * std::f64::consts::PI * G)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rho_crit_gadget() {
        let rho = critical_density_0(1.0);
        // 2.7754e1 w 10¹⁰ M☉/h / (Mpc/h)³ = 2.7754e11 M☉/h / Mpc³
        let expected = 27.754;
        assert!((rho - expected).abs() / expected < 1e-4, "rho={rho}");
    }
}
