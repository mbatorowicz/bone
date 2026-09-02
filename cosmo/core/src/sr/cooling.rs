//! Dyssypacja: odprowadzanie energii z ruchu nieuporządkowanego.
//!
//! To jest jedyny element tego modelu, który nie jest zachowawczy, i dlatego jedyny,
//! który pozwala układowi zrobić coś, czego sama grawitacja zrobić nie może. Bez
//! niego zwirializowana chmura jest zamknięta: ma rozkład prędkości, który
//! podtrzymuje ją przeciw dalszemu zapadaniu, a energii nie ma gdzie podziać. Żadne
//! zagęszczenie w niej nie powstanie, niezależnie od tego, jak długo liczyć.
//!
//! # Co jest tłumione
//!
//! Prędkość każdej cząstki rozkładamy na lokalny przepływ masowy i odchyłkę od
//! niego, a tłumimy WYŁĄCZNIE odchyłkę:
//!
//! ```text
//! v ← v_masowe + (v − v_masowe)·exp(−λΔt)
//! ```
//!
//! To nie jest szczegół implementacyjny, to cała fizyka tego modułu. Chłodzenie
//! radiacyjne w gazie odprowadza energię ruchu termicznego, a nie energię ruchu
//! całego obłoku — wirujący dysk, który się chłodzi, staje się cieńszy i zimniejszy,
//! ale nie przestaje wirować.
//!
//! # Dlaczego nie tarcie
//!
//! Oczywistsze `v ← v·(1−λΔt)` jest błędne i to widowiskowo. Tłumi ono również ruch
//! masowy, więc każda orbita traci moment pędu i cały układ spada do środka w czasie
//! 1/λ, niezależnie od swojej fizyki. Wygląda to jak kolaps grawitacyjny, a jest
//! zwykłym oporem lepkim — i nie da się tego rozpoznać po samym obrazku.
//!
//! # Skala wygładzania
//!
//! Lokalny przepływ liczymy przez siatkę tymi samymi wagami CIC, którymi solver
//! rozkłada masę. Ma to konsekwencję: „lokalny" znaczy „w skali oczka siatki
//! chłodzenia". Zbyt gęsta siatka daje kilka cząstek na komórkę i dyspersję
//! zdominowaną przez szum próbkowania; zbyt zgrubna wlicza do ruchu masowego
//! względny ruch osobnych zgęstek i chłodzenie zaczyna je sztucznie scalać. Dlatego
//! [`Cooling::describe`] raportuje liczbę cząstek na zajętą komórkę — bez tej liczby
//! wynik nie ma jak być oceniony.
//!
//! # Bilans energii
//!
//! Moduł zwraca ilość odprowadzonej energii, a silnik ją sumuje. Dzięki temu dryf
//! energii w diagnostyce dalej mierzy jakość CAŁKOWANIA: sprawdzaną wielkością
//! zachowaną jest `E_tot + E_wypromieniowana`, a nie samo `E_tot`.

use crate::grid::Box;
use crate::sr::config::PhysicsConfig;
use crate::sr::relativity as sr;
use crate::sr::state::State;
use crate::vec3::{vec3, Vec3, ZERO};

/// Górny limit wzmocnienia tempa przez kontrast gęstości.
///
/// Bez niego jedna komórka o gęstości tysiąc razy większej od średniej dostawałaby
/// λΔt rzędu setek i jej dyspersja spadałaby do zera w jednym kroku — a to jest
/// właśnie ta skala, na której gęstość z siatki jest najmniej wiarygodna.
const MAX_DENSITY_BOOST: f64 = 50.0;

/// Poniżej tylu cząstek na zajętą komórkę estymator dyspersji przestaje nim być.
///
/// Przy jednej cząstce w komórce lokalny „przepływ masowy" to jej własna prędkość,
/// odchyłka wychodzi zerowa i chłodzenie po cichu nic nie robi — najgorszy możliwy
/// tryb awarii, bo wygląda jak poprawny bieg.
const MIN_PARTICLES_PER_CELL: f64 = 4.0;

/// Pola rozkładane na siatkę: masa, trzy składowe pędu masowego i masa·|v|².
const FIELDS: usize = 5;

/// Bok siatki dający około ośmiu cząstek na zajętą komórkę.
///
/// Ta wielkość MUSI skalować się z liczbą cząstek i dlatego nie ma sensownej stałej
/// wartości domyślnej. Kula wypełnia około 0,52 objętości pudła, więc zajętych
/// komórek jest ~0,52·ng³; żądanie ośmiu cząstek na komórkę daje ng ≈ (N/4,2)^(1/3).
/// Wychodzi 10 dla czterech tysięcy cząstek i 31 dla stu dwudziestu — czyli
/// dokładnie ten zakres, w którym stała wartość myliłaby się o rząd wielkości.
pub fn auto_grid(n_particles: usize) -> usize {
    let ng = (n_particles.max(1) as f64 / 4.2).cbrt().round() as usize;
    ng.clamp(4, 256)
}

pub struct Cooling {
    /// 0 oznacza „dobierz z liczby cząstek" — patrz [`auto_grid`].
    requested_grid: usize,
    margin: f64,
    grid: Option<usize>,
    cell: Option<f64>,
    occupied: usize,
    particles: usize,
    warned: bool,
    pending_warning: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridTooFine;

impl Cooling {
    /// # Errors
    /// Siatka poniżej czterech komórek nie ma sensu (0 = automat).
    pub fn new(grid: usize, margin: f64) -> Result<Self, GridTooFine> {
        if grid != 0 && grid < 4 {
            return Err(GridTooFine);
        }
        Ok(Self {
            requested_grid: grid,
            margin,
            grid: None,
            cell: None,
            occupied: 0,
            particles: 0,
            warned: false,
            pending_warning: None,
        })
    }

    /// Bok siatki faktycznie użyty; przed pierwszym użyciem wartość zamówiona.
    pub fn grid(&self) -> usize {
        self.grid.unwrap_or(self.requested_grid)
    }

    pub fn particles_per_cell(&self) -> f64 {
        self.particles as f64 / self.occupied.max(1) as f64
    }

    /// Ostrzeżenie do pokazania użytkownikowi, jeśli jakieś czeka.
    pub fn take_warning(&mut self) -> Option<String> {
        self.pending_warning.take()
    }

    pub fn describe(&self) -> String {
        let Some(cell) = self.cell else {
            let how = if self.requested_grid == 0 {
                "automat".to_string()
            } else {
                format!("{}³", self.requested_grid)
            };
            return format!("chłodzenie {how} (jeszcze nie użyte)");
        };
        let auto = if self.requested_grid == 0 {
            " (automat)"
        } else {
            ""
        };
        let ppc = self.particles_per_cell();
        let note = if ppc >= MIN_PARTICLES_PER_CELL {
            ""
        } else {
            "  ⚠ za mało cząstek na komórkę"
        };
        format!(
            "chłodzenie {}³{auto}, oczko {cell:.3}, {ppc:.1} cząstek/komórkę{note}",
            self.grid()
        )
    }

    /// Ochłodź układ o jeden krok. Zwraca ilość odprowadzonej energii (≥ 0).
    ///
    /// Położenia nie są ruszane, więc energia potencjalna się nie zmienia
    /// i odprowadzona energia jest DOKŁADNIE ubytkiem energii kinetycznej — nie
    /// trzeba tu niczego szacować.
    pub fn apply(&mut self, state: &mut State, phys: &PhysicsConfig, dt: f64) -> f64 {
        let rate = phys.cooling_rate;
        if rate <= 0.0 || dt <= 0.0 || state.n() < 2 {
            return 0.0;
        }
        let c = phys.c;
        let n = state.n();
        let before = self.kinetic(state, c);

        let velocities = state.velocities(c);
        let local = self.local_fields(&state.positions, &state.masses, &velocities);

        let total_mass: f64 = state.masses.iter().sum();
        let mean_mass = total_mass / n as f64;
        let reference_density = self.reference_density(&state.masses, &local.density, total_mass);

        let mut cooled = Vec::with_capacity(n);
        for (i, v) in velocities.iter().enumerate() {
            let boost = density_boost(
                local.density[i],
                reference_density,
                phys.cooling_density_power,
            );
            let sigma = local.sigma[i];
            let floor = minimum_dispersion(local.density[i], phys, c, mean_mass);
            let lambda = rate * boost * floor_factor(sigma, floor);
            // Forma wykładnicza, nie (1 − λΔt): jest dokładnym rozwiązaniem równania
            // d(δv)/dt = −λ·δv, więc pozostaje stabilna i dodatnia przy dowolnie dużym
            // λΔt. Wariant liniowy przy λΔt > 1 odwracałby znak odchyłki.
            let decay = (-lambda * dt).exp().max(slowest_decay(sigma, floor));
            let residual = *v - local.bulk[i];
            cooled.push(local.bulk[i] + residual * decay);
        }

        // |v_nowe| < c wynika z wypukłości: v_nowe leży na odcinku między v i
        // v_masowe, a v_masowe jest średnią ważoną prędkości podświetlnych.
        let reference_momentum = state.total_momentum();
        for ((p, mass), target) in state
            .momenta
            .iter_mut()
            .zip(&state.masses)
            .zip(&cooled)
        {
            *p = sr::momentum(*mass, *target, c).unwrap_or(*p);
        }
        restore_total_momentum(&mut state.momenta, reference_momentum, &state.masses);

        let after = self.kinetic(state, c);
        // Nie zwracamy wartości ujemnej: poprawka pędu może w skrajnym przypadku dodać
        // znikomą ilość energii, a ujemne „wypromieniowanie" zepsułoby bilans, którym
        // mierzymy jakość całkowania.
        (before - after).max(0.0)
    }

    fn kinetic(&self, state: &State, c: f64) -> f64 {
        (0..state.n())
            .map(|i| sr::kinetic_energy(state.masses[i], state.momenta[i], c))
            .sum()
    }

    /// Odniesienie dla kontrastu gęstości: średnia ważona masą.
    ///
    /// Odniesieniem jest bieżąca średnia, nie wartość z chwili startu. Dzięki temu
    /// `cooling_rate` zawsze znaczy „tempo przy średniej gęstości układu" i nie trzeba
    /// go przeliczać przy zmianie masy czy promienia. Cena: gdy cały układ się
    /// zagęszcza, tempo przy danej gęstości maleje. Wybór jest świadomy — parametr
    /// o stałym znaczeniu jest wart więcej niż parametr o stałej wartości.
    fn reference_density(&self, masses: &[f64], density: &[f64], total_mass: f64) -> f64 {
        let weighted: f64 = masses
            .iter()
            .zip(density.iter())
            .map(|(m, d)| m * d)
            .sum();
        weighted / total_mass.max(1e-300)
    }

    /// Lokalny przepływ masowy, dyspersja i gęstość — w położeniach cząstek.
    ///
    /// Uwaga na kolejność działań: z siatki odczytywane są SUMY (masa, pęd, masa·|v|²),
    /// a ilorazy liczone dopiero na cząstkach. Odczytanie gotowego ilorazu z siatki
    /// dawałoby średnią nieważoną masą i psuło wynik tam, gdzie masy w sąsiednich
    /// komórkach są różne.
    fn local_fields(&mut self, positions: &[Vec3], masses: &[f64], velocities: &[Vec3]) -> Local {
        let n = positions.len();
        let ng = *self.grid.get_or_insert(if self.requested_grid == 0 {
            auto_grid(n)
        } else {
            self.requested_grid
        });
        // edge = 0: chłodzenie tylko uśrednia w komórce, nie liczy gradientu.
        let box_ = Box::fit(positions, ng, self.margin, 0);

        let mut sums = vec![0.0f64; box_.cells() * FIELDS];
        let mut stencils = Vec::with_capacity(n);
        for i in 0..n {
            let stencil = box_.stencil_clamped(positions[i]);
            if let Some(s) = stencil {
                let m = masses[i];
                let v = velocities[i];
                let payload = [
                    m,
                    m * v.x,
                    m * v.y,
                    m * v.z,
                    m * v.norm_squared(),
                ];
                s.scatter_fields(&mut sums, &payload);
            }
            stencils.push(stencil);
        }

        let occupied = (0..box_.cells())
            .filter(|cell| sums[cell * FIELDS] != 0.0)
            .count();

        let mut bulk = Vec::with_capacity(n);
        let mut sigma = Vec::with_capacity(n);
        let mut density = Vec::with_capacity(n);
        let cell_volume = box_.cell_volume();
        for stencil in &stencils {
            let Some(s) = stencil else {
                bulk.push(ZERO);
                sigma.push(0.0);
                density.push(0.0);
                continue;
            };
            let gathered = s.gather_fields::<FIELDS>(&sums);
            let mass_at = gathered[0].max(1e-300);
            let flow = vec3(gathered[1], gathered[2], gathered[3]) / mass_at;
            let mean_square = gathered[4] / mass_at;
            // Dyspersja z twierdzenia o rozkładzie: ⟨v²⟩ − |⟨v⟩|². Różnica dwóch
            // bliskich liczb, więc przy jednej cząstce w komórce wychodzi drobne ujemne
            // zero — obcięcie jest konieczne, nie kosmetyczne.
            let variance = (mean_square - flow.norm_squared()).max(0.0);
            bulk.push(flow);
            sigma.push(variance.sqrt());
            density.push(mass_at / cell_volume);
        }

        self.cell = Some(box_.h);
        self.occupied = occupied;
        self.particles = n;
        self.warn_if_undersampled();

        Local {
            bulk,
            sigma,
            density,
        }
    }

    /// Powiedz wprost, gdy siatka jest za gęsta na tyle cząstek.
    ///
    /// Tryb awarii jest tu cichy i dlatego groźny: chłodzenie po prostu przestaje
    /// działać, bieg kończy się bez błędu, a wniosek „dyssypacja nic nie zmienia"
    /// jest fałszywy.
    fn warn_if_undersampled(&mut self) {
        if self.warned || self.particles_per_cell() >= MIN_PARTICLES_PER_CELL {
            return;
        }
        self.warned = true;
        self.pending_warning = Some(format!(
            "siatka chłodzenia {}³ daje tylko {:.1} cząstek na zajętą komórkę — lokalna \
             dyspersja jest wtedy szumem próbkowania, a chłodzenie prawie nie działa. \
             Ustaw siatkę chłodzenia na 0 (automat wybrałby {}) albo zwiększ liczbę cząstek.",
            self.grid(),
            self.particles_per_cell(),
            auto_grid(self.particles)
        ));
    }
}

struct Local {
    bulk: Vec<Vec3>,
    sigma: Vec<f64>,
    density: Vec<f64>,
}

/// `(ρ/ρ_odniesienia)^power`, przycięte od góry.
fn density_boost(density: f64, reference: f64, power: f64) -> f64 {
    if power.abs() < 1e-12 {
        return 1.0;
    }
    let ratio = density / reference.max(1e-300);
    ratio.powf(power).min(MAX_DENSITY_BOOST)
}

/// Dyspersja (trójwymiarowa), poniżej której chłodzenie nie ma prawa zejść.
///
/// Dwie podłogi, brana większa.
///
/// FIZYCZNA — `cooling_floor·c`, odpowiednik temperatury, poniżej której ośrodek
/// przestaje promieniować. Stała, więc ustala NAJWIĘKSZĄ skalę fragmentu.
///
/// NUMERYCZNA — z warunku, żeby masa Jeansa liczyła co najmniej N cząstek. Przy
/// `M_J = ρλ³` i `λ = σ₁ᴰ√(π/Gρ)` wychodzi
///
/// ```text
/// M_J = σ₁ᴰ³·π^{3/2}/(G^{3/2}·√ρ)  ⟹  σ₁ᴰ ≥ [N·m̄·G^{3/2}·√ρ/π^{3/2}]^{1/3}
/// ```
///
/// Istotny jest wykładnik przy gęstości: `ρ^{1/6}`. Gęstość pochodzi ze zgrubnej
/// siatki chłodzenia i dla zwartych zgęstek jest zaniżona nawet sześćdziesięciokrotnie,
/// ale `ρ^{1/6}` tłumi ten błąd do czynnika dwa. Wariant z progiem na samą długość
/// Jeansa zależy od gęstości jak `√ρ` i dziedziczyłby go niemal w całości.
///
/// Uwaga na definicję σ: pola z siatki dają dyspersję TRÓJWYMIAROWĄ, a wzory Jeansa
/// operują na jednej składowej, stąd mnożenie przez √3.
fn minimum_dispersion(density: f64, phys: &PhysicsConfig, c: f64, mean_mass: f64) -> f64 {
    let physical = phys.cooling_floor * c;
    let min_particles = phys.cooling_min_particles;
    if min_particles <= 0.0 || phys.g <= 0.0 || mean_mass <= 0.0 {
        return physical;
    }
    let jeans_mass = min_particles * mean_mass;
    let sigma_1d = (jeans_mass * phys.g.powf(1.5) * density.max(0.0).sqrt()
        / std::f64::consts::PI.powf(1.5))
    .cbrt();
    (sigma_1d * 3.0f64.sqrt()).max(physical)
}

/// Wygaszenie tempa przy dyspersji zbliżającej się do progu.
///
/// Czynnik `1 − (σ_min/σ)²` jest gładki, znika dokładnie przy σ = σ_min i dąży do 1
/// przy σ ≫ σ_min, więc nie wprowadza progu skokowego — chłodzenie zwalnia w miarę
/// zbliżania się do podłogi, zamiast wyłączać się nagle.
fn floor_factor(sigma: f64, floor: f64) -> f64 {
    if floor <= 0.0 {
        return 1.0;
    }
    // Obcięcie od dołu progiem, nie zerem: iloraz jest wtedy zawsze ≤ 1, więc kwadrat
    // nie ma jak się przepełnić, a wynik dla σ ≤ σ_min wychodzi dokładnie 0.
    let safe = sigma.max(floor).max(1e-300);
    1.0 - (floor / safe).powi(2)
}

/// Najmniejszy dopuszczalny współczynnik wygaszenia w jednym kroku.
///
/// Odchyłka jest mnożona przez `exp(−λΔt)`, więc dyspersja maleje w tym samym
/// stosunku. Ograniczenie tego stosunku od dołu do `σ_min/σ` sprawia, że krok kończy
/// się DOKŁADNIE na podłodze, a nie pod nią.
///
/// Samo [`floor_factor`] tego nie zapewnia i nie może: zwalnia tempo na podstawie
/// dyspersji z POCZĄTKU kroku, a przy `λΔt ≈ 1` jeden krok przenosi układ daleko poza
/// punkt, w którym tempo miałoby zniknąć. Bez tego ogranicznika przy dużym
/// `cooling_rate` podłoga nie działa wcale — dyspersja zbiega do zera, choć każda
/// pojedyncza wartość λ jest poprawna.
fn slowest_decay(sigma: f64, floor: f64) -> f64 {
    if floor <= 0.0 {
        return 0.0;
    }
    if sigma <= floor {
        return 1.0;
    }
    floor / sigma
}

/// Wymuś, żeby chłodzenie nie zmieniło całkowitego pędu.
///
/// Tłumienie odchyłek od lokalnej średniej powinno zachowywać pęd samo z siebie, ale
/// interpolacja CIC nie jest dokładnie tożsamościowa: średnia odczytana z siatki
/// różni się od średniej prawdziwej o wielkość rzędu błędu interpolacji. Bez tej
/// poprawki resztkowy pęd narastałby przez tysiące kroków i cały układ zaczynałby
/// dryfować z kadru — czyli powstałby dokładnie ten artefakt, przed którym warunek
/// początkowy się zabezpiecza.
fn restore_total_momentum(momenta: &mut [Vec3], reference: Vec3, masses: &[f64]) {
    let drift = momenta.iter().copied().sum::<Vec3>() - reference;
    let total: f64 = masses.iter().sum();
    if total <= 0.0 {
        return;
    }
    for (p, m) in momenta.iter_mut().zip(masses.iter()) {
        *p -= drift * (*m / total);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::sr::state::State;

    fn phys(rate: f64) -> PhysicsConfig {
        PhysicsConfig {
            g: 0.16,
            c: 30.0,
            cooling_rate: rate,
            cooling_density_power: 0.0,
            cooling_floor: 0.0,
            cooling_min_particles: 0.0,
            ..PhysicsConfig::default()
        }
    }

    /// Chmura o zadanej dyspersji i zadanym ruchu masowym.
    ///
    /// Prędkości przechodzą przez to samo przycięcie, co w warunkach początkowych:
    /// rozkład normalny ma ogon powyżej `c`, więc przy małym `c` bez przycięcia sam
    /// SETUP testu byłby niefizyczny — i test wywracałby się na budowaniu stanu,
    /// zanim dotknąłby chłodzenia.
    fn hot_cloud(n: usize, sigma: f64, bulk: Vec3, c: f64) -> State {
        let mut rng = Rng::seeded(42);
        let positions: Vec<Vec3> = (0..n).map(|_| rng.normal_vec(0.0, 2.0)).collect();
        let masses = vec![1.0; n];
        let momenta: Vec<Vec3> = (0..n)
            .map(|_| {
                let (v, _) = sr::clamp_initial_speed(bulk + rng.normal_vec(0.0, sigma), c);
                sr::momentum(1.0, v, c).expect("prędkość po przycięciu jest podświetlna")
            })
            .collect();
        State::new(positions, momenta, masses).unwrap()
    }

    #[test]
    fn zero_rate_is_a_no_op() {
        let ph = phys(0.0);
        let mut state = hot_cloud(500, 1.0, ZERO, ph.c);
        let before = state.momenta.clone();
        let removed = Cooling::new(0, 0.15).unwrap().apply(&mut state, &ph, 0.1);
        assert_eq!(removed, 0.0);
        assert_eq!(state.momenta, before);
    }

    #[test]
    fn cooling_removes_kinetic_energy() {
        let ph = phys(2.0);
        let mut state = hot_cloud(2_000, 2.0, ZERO, ph.c);
        let kinetic = |s: &State| -> f64 {
            (0..s.n())
                .map(|i| sr::kinetic_energy(s.masses[i], s.momenta[i], ph.c))
                .sum::<f64>()
        };
        let before = kinetic(&state);
        let removed = Cooling::new(0, 0.15).unwrap().apply(&mut state, &ph, 0.2);
        let after = kinetic(&state);
        assert!(removed > 0.0, "nic nie odprowadzono");
        assert!(after < before, "energia nie spadła");
        assert!(
            (removed - (before - after)).abs() / removed < 1e-9,
            "bilans nie domyka się"
        );
    }

    /// SEDNO MODUŁU: ruch masowy musi przeżyć chłodzenie. Gdyby moduł tłumił także
    /// jego, byłby oporem lepkim udającym kolaps grawitacyjny.
    #[test]
    fn bulk_motion_survives_cooling() {
        let ph = phys(5.0);
        let bulk = vec3(3.0, 0.0, 0.0);
        let mut state = hot_cloud(4_000, 1.5, bulk, ph.c);
        let mean_velocity = |s: &State| -> Vec3 {
            (0..s.n()).map(|i| s.velocity(i, ph.c)).sum::<Vec3>() / s.n() as f64
        };
        let before = mean_velocity(&state);
        Cooling::new(0, 0.15).unwrap().apply(&mut state, &ph, 0.5);
        let after = mean_velocity(&state);
        assert!(
            (after.x - before.x).abs() / before.x.abs() < 0.05,
            "ruch masowy stłumiony: {} → {}",
            before.x,
            after.x
        );
    }

    /// Dyspersja natomiast MUSI spaść — inaczej moduł nic nie robi.
    #[test]
    fn dispersion_is_damped() {
        let ph = phys(5.0);
        let mut state = hot_cloud(4_000, 2.0, vec3(2.0, 0.0, 0.0), ph.c);
        let dispersion = |s: &State| -> f64 {
            let mean = (0..s.n()).map(|i| s.velocity(i, ph.c)).sum::<Vec3>() / s.n() as f64;
            ((0..s.n())
                .map(|i| (s.velocity(i, ph.c) - mean).norm_squared())
                .sum::<f64>()
                / s.n() as f64)
                .sqrt()
        };
        let before = dispersion(&state);
        Cooling::new(0, 0.15).unwrap().apply(&mut state, &ph, 0.4);
        let after = dispersion(&state);
        assert!(after < 0.9 * before, "dyspersja {before} → {after}");
    }

    #[test]
    fn total_momentum_is_preserved() {
        let ph = phys(3.0);
        let mut state = hot_cloud(2_000, 2.0, vec3(1.0, -0.5, 0.3), ph.c);
        let before = state.total_momentum();
        Cooling::new(0, 0.15).unwrap().apply(&mut state, &ph, 0.3);
        let scale: f64 = state.momenta.iter().map(|p| p.norm()).sum();
        let drift = (state.total_momentum() - before).norm() / scale;
        assert!(drift < 1e-12, "dryf pędu {drift}");
    }

    #[test]
    fn speeds_stay_subluminal() {
        let mut ph = phys(50.0);
        ph.c = 5.0;
        let mut state = hot_cloud(1_000, 2.0, vec3(1.0, 0.0, 0.0), ph.c);
        let mut cooling = Cooling::new(0, 0.15).unwrap();
        for _ in 0..50 {
            cooling.apply(&mut state, &ph, 0.2);
            for i in 0..state.n() {
                assert!(state.speed_over_c(i, ph.c) < 1.0);
            }
        }
    }

    /// Podłoga musi zatrzymać chłodzenie, inaczej dyspersja zbiegłaby do zera
    /// i fragmenty przestałyby być rozdzielone.
    #[test]
    fn floor_stops_the_cooling() {
        let mut ph = phys(20.0);
        ph.cooling_floor = 0.1; // 0,1·c = 3
        let mut state = hot_cloud(3_000, 3.0, ZERO, ph.c);
        let dispersion = |s: &State| -> f64 {
            let mean = (0..s.n()).map(|i| s.velocity(i, ph.c)).sum::<Vec3>() / s.n() as f64;
            ((0..s.n())
                .map(|i| (s.velocity(i, ph.c) - mean).norm_squared())
                .sum::<f64>()
                / s.n() as f64)
                .sqrt()
        };
        let mut cooling = Cooling::new(0, 0.15).unwrap();
        for _ in 0..200 {
            cooling.apply(&mut state, &ph, 0.2);
        }
        let final_sigma = dispersion(&state);
        assert!(final_sigma > 1.5, "zeszło pod podłogę: σ={final_sigma}");
    }

    #[test]
    fn floor_factor_vanishes_at_the_floor_and_saturates_above() {
        assert_eq!(floor_factor(1.0, 1.0), 0.0);
        assert_eq!(floor_factor(0.5, 1.0), 0.0);
        assert!(floor_factor(1000.0, 1.0) > 0.999);
        assert_eq!(floor_factor(0.3, 0.0), 1.0);
    }

    /// Ogranicznik kroku musi trafiać dokładnie w podłogę: ani jej nie przekraczać,
    /// ani nie blokować chłodzenia, gdy podłogi nie ma.
    #[test]
    fn slowest_decay_lands_exactly_on_the_floor() {
        assert_eq!(slowest_decay(1.0, 0.0), 0.0, "bez podłogi bez ograniczeń");
        assert_eq!(slowest_decay(0.5, 1.0), 1.0, "pod podłogą nie chłodzimy");
        assert_eq!(slowest_decay(1.0, 1.0), 1.0);
        let sigma = 10.0;
        let floor = 2.5;
        assert!((sigma * slowest_decay(sigma, floor) - floor).abs() < 1e-12);
    }

    /// Nawet absurdalnie duże `λΔt` nie ma prawa przeskoczyć podłogi. Samo zwalnianie
    /// tempa przez [`floor_factor`] tego nie gwarantuje, bo patrzy na dyspersję
    /// z POCZĄTKU kroku — a jeden krok może wykonać całą drogę i wylądować pod progiem.
    #[test]
    fn a_huge_step_still_respects_the_floor() {
        let mut ph = phys(1e6);
        ph.cooling_floor = 0.1;
        let mut state = hot_cloud(3_000, 3.0, ZERO, ph.c);
        Cooling::new(0, 0.15).unwrap().apply(&mut state, &ph, 10.0);
        let mean = (0..state.n()).map(|i| state.velocity(i, ph.c)).sum::<Vec3>()
            / state.n() as f64;
        let sigma = ((0..state.n())
            .map(|i| (state.velocity(i, ph.c) - mean).norm_squared())
            .sum::<f64>()
            / state.n() as f64)
            .sqrt();
        assert!(sigma > 1.5, "jeden krok zjechał pod podłogę: σ={sigma}");
    }

    #[test]
    fn density_boost_is_bounded() {
        assert_eq!(density_boost(5.0, 1.0, 0.0), 1.0);
        assert!((density_boost(4.0, 2.0, 1.0) - 2.0).abs() < 1e-12);
        assert_eq!(density_boost(1e9, 1.0, 1.0), MAX_DENSITY_BOOST);
    }

    #[test]
    fn auto_grid_scales_with_particle_count() {
        assert_eq!(auto_grid(4_000), 10);
        assert_eq!(auto_grid(120_000), 31);
        assert!(auto_grid(1) >= 4);
        assert!(auto_grid(usize::MAX / 2) <= 256);
    }

    #[test]
    fn rejects_absurdly_fine_grid() {
        assert!(Cooling::new(3, 0.15).is_err());
        assert!(Cooling::new(0, 0.15).is_ok());
        assert!(Cooling::new(4, 0.15).is_ok());
    }

    /// Za gęsta siatka to cicha awaria, więc musi zostać zgłoszona.
    #[test]
    fn undersampled_grid_is_reported() {
        let ph = phys(1.0);
        let mut state = hot_cloud(200, 1.0, ZERO, ph.c);
        let mut cooling = Cooling::new(64, 0.15).unwrap();
        cooling.apply(&mut state, &ph, 0.1);
        assert!(cooling.take_warning().is_some(), "przemilczane");
        assert!(cooling.take_warning().is_none(), "ostrzeżenie się powtarza");
    }

    #[test]
    fn describe_reports_particles_per_cell() {
        let ph = phys(1.0);
        let mut state = hot_cloud(4_000, 1.0, ZERO, ph.c);
        let mut cooling = Cooling::new(0, 0.15).unwrap();
        assert!(cooling.describe().contains("jeszcze nie użyte"));
        cooling.apply(&mut state, &ph, 0.1);
        let text = cooling.describe();
        assert!(text.contains("cząstek/komórkę"), "{text}");
        assert!(text.contains("automat"), "{text}");
    }
}
