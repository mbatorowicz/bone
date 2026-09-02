//! Silnik ΛCDM: leapfrog KDK po `ln a`, siły z tego samego solvera PM co tryb SR.
//!
//! Krok jest równomierny w `ln a`, nie w czasie. To nie jest wygoda zapisu: czynniki
//! dryfu i kopnięcia są całkami po `a`, więc w tej zmiennej całkowanie jest dokładne
//! niezależnie od tego, jak szybko zmienia się `H(a)` — a zmienia się o rzędy wielkości
//! między `z = 49` i dziś.
//!
//! # Co to jest, a co nie jest
//!
//! Solver PM ma brzegi IZOLOWANE (metoda Hockneya), a nie periodyczne. Symulowana jest
//! więc odosobniona próbka materii w pustej przestrzeni, a nie kawałek jednorodnego
//! wszechświata z nieskończonym ciągiem kopii. Konsekwencja jest rzeczywista: na brzegu
//! próbki brakuje przyciągania z zewnątrz, więc jej krawędź jest wolniejsza od środka.
//! Za to nic nie zawija się przez ścianę i chmura może swobodnie zapadać się i rozszerzać.

use serde::{Deserialize, Serialize};

use crate::grid::center_span;
use crate::lcdm::cosmology::Cosmology;
use crate::lcdm::ics::make_initial_state;
use crate::lcdm::units::G;
use crate::mesh::Mesh;
use crate::vec3::{Vec3, ZERO};

/// Zapas pudła siatki ponad rozciągłość chmury, jako jej ułamek.
const BOX_MARGIN: f64 = 0.15;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunConfig {
    /// Bok próbki [Mpc/h].
    pub box_size: f64,
    /// Bok siatki warunków początkowych; daje `n_grid³` cząstek.
    pub n_grid: usize,
    /// Bok siatki solvera PM.
    pub pm_grid: usize,
    pub z_start: f64,
    /// Przesunięcie ku czerwieni, przy którym bieg uznaje się za skończony.
    pub z_end: f64,
    /// Bazowy krok w `ln a`; skalowany zależnie od `z`.
    pub dlna: f64,
    pub seed: u64,
}

/// Stan ΛCDM odczytany z checkpointu — argument [`Engine::with_state`].
pub struct Saved {
    pub cosmology: Cosmology,
    pub cfg: RunConfig,
    pub a: f64,
    pub positions: Vec<Vec3>,
    pub momenta: Vec<Vec3>,
    pub mass: f64,
    pub box_size: f64,
    pub step: u64,
    pub initial_contrast: f64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self::structure()
    }
}

impl RunConfig {
    /// Formacja struktur: mała próbka, gęsta siatka — filamenty i halo są widoczne.
    pub fn structure() -> Self {
        Self {
            box_size: 32.0,
            n_grid: 48,
            pm_grid: 48,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0005,
            seed: 42,
        }
    }

    pub fn structure_64() -> Self {
        Self {
            box_size: 40.0,
            n_grid: 64,
            pm_grid: 64,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0004,
            seed: 42,
        }
    }

    /// Duża próbka: skale pozostają liniowe, więc widać sam wzrost amplitudy.
    pub fn linear_growth() -> Self {
        Self {
            box_size: 200.0,
            n_grid: 32,
            pm_grid: 32,
            z_start: 49.0,
            z_end: 0.0,
            dlna: 0.0008,
            seed: 7,
        }
    }

    pub fn n_particles(&self) -> usize {
        self.n_grid.pow(3)
    }
}

/// Krok w `ln a` zależny od `z`.
///
/// Era liniowa (`z > 20`) znosi krok półtora raza większy, bo cząstki poruszają się
/// zgodnie i nic się nie przecina. Przy `z < 5` struktura jest już nieliniowa i ten
/// sam krok dawałby przestrzeliwanie halo — dlatego jest o połowę mniejszy.
pub fn adaptive_dlna(z: f64, base: f64) -> f64 {
    // Funkcja jest całkowita: krok całkowania nie ma prawa wyjść `NaN` nawet przy
    // wejściu bez sensu, bo `NaN` w kroku zatruwa cały dalszy bieg bez żadnego
    // komunikatu, a pierwszym objawem jest pusty ekran.
    let base = if base.is_finite() { base.max(1e-6) } else { 1e-6 };
    let scale = if !z.is_finite() {
        1.0
    } else if z >= 20.0 {
        1.5
    } else if z <= 5.0 {
        0.5
    } else {
        0.5 + (z - 5.0) / 15.0
    };
    (base * scale).clamp(0.00008, 0.05)
}

pub struct Engine {
    pub cosmology: Cosmology,
    pub cfg: RunConfig,
    pub a: f64,
    pub positions: Vec<Vec3>,
    /// Pęd komowy `p = a²ẋ`.
    pub momenta: Vec<Vec3>,
    /// Przyspieszenie komowe `−∇Φ`, gdzie `Φ` jest potencjałem w jednostkach komowych.
    pub accel: Vec<Vec3>,
    pub mass: f64,
    pub box_size: f64,
    pub step: u64,
    /// `σ(δ)` warunków początkowych — patrz [`crate::lcdm::ics::InitialState`].
    pub initial_contrast: f64,
    masses: Vec<f64>,
    mesh: Mesh,
    li: LayzerIrvine,
}

impl Engine {
    pub fn new(cosmology: Cosmology, cfg: RunConfig) -> Self {
        let ic = make_initial_state(cosmology, cfg.box_size, cfg.n_grid, cfg.z_start, cfg.seed);
        let n = ic.positions.len();
        let mut engine = Self {
            cosmology,
            cfg,
            a: ic.a,
            positions: ic.positions,
            momenta: ic.momenta,
            accel: vec![ZERO; n],
            mass: ic.mass,
            box_size: ic.box_size,
            step: 0,
            initial_contrast: ic.delta_rms,
            masses: vec![ic.mass; n],
            mesh: Mesh::new(cfg.pm_grid, BOX_MARGIN),
            li: LayzerIrvine::default(),
        };
        engine.refresh_forces();
        engine.li = LayzerIrvine::start(engine.energies());
        engine
    }

    /// Zbuduj silnik na gotowym stanie — wznowienie z checkpointu.
    ///
    /// Residuum Layzera–Irvine'a startuje od nowa: całka po `ln a` sprzed zapisu
    /// nie jest w pliku, bo nie jest potrzebna do dalszego ruchu. Diagnostyka po
    /// wznowieniu mierzy jakość *dalszego* całkowania.
    pub fn with_state(saved: Saved) -> Self {
        let n = saved.positions.len();
        let mut engine = Self {
            cosmology: saved.cosmology,
            cfg: saved.cfg,
            a: saved.a,
            positions: saved.positions,
            momenta: saved.momenta,
            accel: vec![ZERO; n],
            mass: saved.mass,
            box_size: saved.box_size,
            step: saved.step,
            initial_contrast: saved.initial_contrast,
            masses: vec![saved.mass; n],
            mesh: Mesh::new(saved.cfg.pm_grid, BOX_MARGIN),
            li: LayzerIrvine::default(),
        };
        engine.refresh_forces();
        engine.li = LayzerIrvine::start(engine.energies());
        engine
    }

    pub fn redshift(&self) -> f64 {
        1.0 / self.a - 1.0
    }

    pub fn n(&self) -> usize {
        self.positions.len()
    }

    pub fn age_gyr(&self) -> f64 {
        self.cosmology.age_gyr(self.a)
    }

    pub fn finished(&self) -> bool {
        self.redshift() <= self.cfg.z_end
    }

    /// Środek chmury i jej największa rozciągłość — do ustawienia kamery.
    pub fn center_span(&self) -> (Vec3, f64) {
        center_span(&self.positions)
    }

    fn refresh_forces(&mut self) {
        // Softening zerowy znaczy „tyle, ile daje siatka": `Mesh` podnosi go do
        // rozmiaru oczka, bo poniżej niego nie ma informacji o polu.
        self.mesh.refresh(&self.positions, &self.masses, G, 0.0);
        self.accel = self.mesh.gather_acceleration(&self.positions);
    }

    pub fn advance(&mut self) {
        let a1 = self.a;
        let dlna = adaptive_dlna(self.redshift(), self.cfg.dlna);
        let a2 = a1 * dlna.exp();
        // Punkt środkowy geometryczny, bo krok jest równomierny w `ln a`.
        let a_mid = (a1 * a2).sqrt();

        let kick_in = self.cosmology.kick_factor(a1, a_mid);
        for (p, acc) in self.momenta.iter_mut().zip(self.accel.iter()) {
            *p += *acc * kick_in;
        }

        let drift = self.cosmology.drift_factor(a1, a2);
        for (x, p) in self.positions.iter_mut().zip(self.momenta.iter()) {
            *x += *p * drift;
        }
        self.refresh_forces();

        let kick_out = self.cosmology.kick_factor(a_mid, a2);
        for (p, acc) in self.momenta.iter_mut().zip(self.accel.iter()) {
            *p += *acc * kick_out;
        }

        self.li.accumulate(self.energies(), a1, a2);
        self.a = a2;
        self.step += 1;
    }

    /// Energia kinetyczna i potencjalna w zmiennych komowych.
    ///
    /// `T = Σ p²/(2ma²)` wynika wprost z `p = a²ẋ`.
    ///
    /// `W` jest liczone z twierdzenia wirialnego, a nie z sumowania par: dla potencjału
    /// `1/r` zachodzi `Σ mᵢ xᵢ·aᵢ = U`, gdzie `U = −G Σ_{i<j} mᵢmⱼ/rᵢⱼ`. Dowód jest
    /// jednolinijkowy — po sparowaniu wyrazów `i,j` licznik `xᵢ·(xᵢ−xⱼ) + xⱼ·(xⱼ−xᵢ)`
    /// zwija się do `|xᵢ−xⱼ|²` — i daje wynik BEZ czynnika ½. Czynnik ½, dopisany tu
    /// przez analogię do `U = ½ Σ mᵢφᵢ`, byłby błędem podwójnie policzonego parowania:
    /// residuum Layzera–Irvine'a przestaje wtedy zbiegać do zera i nie mierzy niczego.
    ///
    /// Potencjał właściwy dla zaburzeń to `φ = Φ/a`, więc `W = U/a`. Bez tego dzielenia
    /// bilans domykałby się tylko przy `a ≈ 1`.
    pub fn energies(&self) -> Energies {
        let a2 = self.a * self.a;
        let kinetic: f64 = self
            .momenta
            .iter()
            .map(|p| p.norm_squared() / (2.0 * self.mass * a2))
            .sum();
        let virial: f64 = self
            .positions
            .iter()
            .zip(self.accel.iter())
            .map(|(x, acc)| x.dot(*acc))
            .sum();
        Energies {
            kinetic,
            potential: self.mass * virial / self.a,
        }
    }

    /// Residuum równania Layzera–Irvine'a, znormalizowane do skali energii.
    pub fn layzer_irvine(&self) -> f64 {
        self.li.residual(self.energies())
    }

    /// Kontrast gęstości `1 + δ` w miejscu cząstki, spłaszczony do przedziału [0, 1].
    ///
    /// Skala logarytmiczna, bo `δ` rozciąga się od `−1` w pustkach do setek w halo;
    /// liniowa mapa pokazywałaby wyłącznie halo na czarnym tle, gubiąc filamenty,
    /// czyli dokładnie to, co w tej symulacji jest ciekawe.
    pub fn shade(&self, i: usize) -> f32 {
        let delta = self
            .mesh
            .density_contrast_at(self.positions[i], self.cosmology.mean_matter_density());
        let contrast = (1.0 + delta).max(1e-4);
        ((contrast.log10() * 0.72 + 0.18) as f32).clamp(0.0, 1.0)
    }

    pub fn describe_solver(&self) -> String {
        format!(
            "PM {}³ izolowany, oczko {:.3} Mpc/h",
            self.cfg.pm_grid,
            self.mesh.cell_size().unwrap_or(0.0)
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Energies {
    pub kinetic: f64,
    pub potential: f64,
}

impl Energies {
    pub fn total(self) -> f64 {
        self.kinetic + self.potential
    }

    /// Wyrażenie `2T + W` z równania Layzera–Irvine'a.
    fn virial_combination(self) -> f64 {
        2.0 * self.kinetic + self.potential
    }

    fn scale(self) -> f64 {
        (self.kinetic.abs() + self.potential.abs()).max(1e-30)
    }
}

/// Bilans energii w rozszerzającym się wszechświecie.
///
/// W przestrzeni komowej energia NIE jest zachowana — ekspansja odbiera energię
/// kinetyczną. Zachowana jest kombinacja z równania Layzera–Irvine'a:
/// `d(T + W)/dlna + (2T + W) = 0`. Jej residuum jest tym, czym dla układu izolowanego
/// jest dryf energii: jedyną liczbą mówiącą, czy krok całkowania jest dość mały.
#[derive(Clone, Copy, Debug, Default)]
struct LayzerIrvine {
    initial_total: f64,
    integral: f64,
    last_combination: f64,
}

impl LayzerIrvine {
    fn start(e: Energies) -> Self {
        Self {
            initial_total: e.total(),
            integral: 0.0,
            last_combination: e.virial_combination(),
        }
    }

    /// Całkowanie trapezami po `ln a` — ten sam rząd dokładności co sam leapfrog,
    /// więc residuum mierzy błąd całkowania ruchu, a nie błąd tej kwadratury.
    fn accumulate(&mut self, e: Energies, a1: f64, a2: f64) {
        let combination = e.virial_combination();
        self.integral += 0.5 * (self.last_combination + combination) * (a2.ln() - a1.ln());
        self.last_combination = combination;
    }

    fn residual(&self, e: Energies) -> f64 {
        (e.total() - self.initial_total + self.integral) / e.scale()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(dlna: f64) -> RunConfig {
        RunConfig {
            box_size: 32.0,
            n_grid: 16,
            pm_grid: 16,
            z_start: 49.0,
            z_end: 0.0,
            dlna,
            seed: 1,
        }
    }

    fn engine(dlna: f64) -> Engine {
        Engine::new(Cosmology::planck18(), small(dlna))
    }

    fn largest_move_fraction(before: &[Vec3], after: &[Vec3], span: f64) -> f64 {
        before
            .iter()
            .zip(after.iter())
            .map(|(a, b)| (*b - *a).max_abs_component() / span)
            .fold(0.0, f64::max)
    }

    #[test]
    fn presets_are_internally_consistent() {
        for cfg in [
            RunConfig::structure(),
            RunConfig::structure_64(),
            RunConfig::linear_growth(),
        ] {
            assert!(cfg.box_size > 0.0);
            assert!(cfg.n_grid >= 16 && cfg.pm_grid >= 16);
            assert!(cfg.z_start > cfg.z_end);
            assert!(cfg.dlna > 0.0);
            assert_eq!(cfg.n_particles(), cfg.n_grid.pow(3));
        }
    }

    #[test]
    fn adaptive_step_slows_down_in_the_nonlinear_era() {
        let base = 0.005;
        let early = adaptive_dlna(30.0, base);
        let middle = adaptive_dlna(12.0, base);
        let late = adaptive_dlna(2.0, base);
        assert!(early > middle && middle > late);
        assert!((early - base * 1.5).abs() < 1e-12);
        assert!((late - base * 0.5).abs() < 1e-12);
    }

    #[test]
    fn adaptive_step_is_bounded_for_absurd_input() {
        assert!(adaptive_dlna(1000.0, 10.0) <= 0.05);
        assert!(adaptive_dlna(0.0, 0.0) >= 0.00008);
        assert!(adaptive_dlna(f64::NAN, 0.001).is_finite());
    }

    #[test]
    fn engine_starts_with_forces_and_a_sensible_redshift() {
        let eng = engine(0.0005);
        assert_eq!(eng.n(), 16usize.pow(3));
        assert!((eng.redshift() - 49.0).abs() < 1e-9);
        assert!(eng.accel.iter().any(|a| a.norm() > 0.0), "brak sił");
        assert!(eng.accel.iter().all(|a| a.is_finite()));
        assert!(!eng.finished());
    }

    /// Pierwszy krok nie może przenieść cząstki o zauważalną część chmury; gdyby
    /// przenosił, warunki początkowe albo czynniki kroku byłyby w złych jednostkach.
    #[test]
    fn first_step_moves_particles_by_a_small_fraction() {
        let mut eng = engine(0.0005);
        let before = eng.positions.clone();
        let span = eng.center_span().1;
        eng.advance();
        let moved = largest_move_fraction(&before, &eng.positions, span);
        assert!(moved < 0.05, "największe przesunięcie {moved} rozciągłości");
    }

    /// Przestrzeń jest otwarta: cząstka wyprowadzona za pudło ma tam zostać.
    /// Zawinięcie oznaczałoby periodyczność, której ten solver nie ma.
    #[test]
    fn particles_are_never_wrapped_into_the_box() {
        let mut eng = engine(0.0005);
        eng.positions[0] = crate::vec3::vec3(-1.0, 0.0, 0.0);
        eng.momenta[0] = ZERO;
        eng.accel[0] = ZERO;
        eng.advance();
        assert!(
            eng.positions[0].x < 0.0,
            "pozycja zawinięta: {}",
            eng.positions[0].x
        );
    }

    #[test]
    fn expansion_continues_past_today() {
        let mut eng = engine(0.04);
        for _ in 0..40 {
            eng.advance();
        }
        assert!(eng.redshift() < 20.0, "z = {}", eng.redshift());
        assert!(eng.a > 1.0 / 50.0);
        assert!(eng.age_gyr() > 0.0);
    }

    #[test]
    fn cloud_stays_coherent_over_many_steps() {
        let mut eng = engine(0.04);
        let mut worst = 0.0f64;
        for _ in 0..40 {
            let before = eng.positions.clone();
            let span = eng.center_span().1;
            eng.advance();
            worst = worst.max(largest_move_fraction(&before, &eng.positions, span));
        }
        assert!(worst < 0.15, "największe przesunięcie {worst} rozciągłości");
    }

    /// Energia potencjalna musi być UJEMNA — grawitacja jest przyciągająca.
    /// Znak dodatni znaczyłby, że gradient albo znak jądra jest odwrócony.
    #[test]
    fn potential_energy_is_negative() {
        let e = engine(0.0005).energies();
        assert!(e.kinetic > 0.0, "T = {}", e.kinetic);
        assert!(e.potential < 0.0, "W = {}", e.potential);
    }

    /// Residuum Layzera–Irvine'a musi MALEĆ przy mniejszym kroku. To jest jedyny
    /// sprawdzian, że czynniki dryfu, kopnięcia i wyrażenie na energię opisują ten
    /// sam układ; sama mała wartość residuum mogłaby wynikać z krótkiego biegu.
    #[test]
    fn layzer_irvine_residual_shrinks_with_the_step() {
        let mut residuals = Vec::new();
        for dlna in [0.02, 0.005] {
            let mut eng = engine(dlna);
            let target = eng.a * (0.02f64 * 20.0).exp();
            while eng.a < target {
                eng.advance();
            }
            residuals.push(eng.layzer_irvine().abs());
        }
        assert!(
            residuals[1] < residuals[0],
            "residuum nie zmalało: {:?}",
            residuals
        );
    }

    #[test]
    fn shade_stays_in_range_and_tracks_density() {
        let eng = engine(0.0005);
        let shades: Vec<f32> = (0..eng.n()).map(|i| eng.shade(i)).collect();
        assert!(shades.iter().all(|s| (0.0..=1.0).contains(s)));
        let spread = shades.iter().copied().fold(0.0f32, f32::max)
            - shades.iter().copied().fold(1.0f32, f32::min);
        assert!(spread > 0.0, "wszystkie cząstki mają ten sam odcień");
    }

    #[test]
    fn describe_solver_reports_the_grid() {
        let text = engine(0.0005).describe_solver();
        assert!(text.contains("16³"), "{text}");
    }
}
