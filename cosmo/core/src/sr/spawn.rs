//! Warunki początkowe.
//!
//! Prędkość początkowa jest skalowana do lokalnej prędkości okrężnej wyliczonej
//! z masy zamkniętej wewnątrz promienia, a nie do ułamka `c`. Dzięki temu
//! `rotation = 1` naprawdę oznacza orbitę kołową dla dowolnego G, N i promienia,
//! zamiast liczby, którą trzeba dobierać metodą prób.

use crate::rng::Rng;
use crate::sr::backends::exact::Exact;
use crate::sr::backends::Backend;
use crate::sr::config::{Config, Geometry};
use crate::sr::relativity as sr;
use crate::sr::state::State;
use crate::vec3::{vec3, Vec3, ZERO};

/// Stan początkowy wraz z ostrzeżeniami, których nie wolno przemilczeć.
pub struct Spawned {
    pub state: State,
    /// Sytuacje, w których policzony warunek początkowy RÓŻNI SIĘ od zamówionego.
    ///
    /// Zwracane, a nie wypisywane, bo ten sam kod obsługuje bieg wsadowy i panel —
    /// a ostrzeżenie wypisane na stderr w aplikacji okienkowej nie dociera nigdzie.
    pub warnings: Vec<String>,
}

/// Kula jednorodna o promieniu `radius`.
fn ball(rng: &mut Rng, n: usize, radius: f64) -> Vec<Vec3> {
    (0..n)
        .map(|_| rng.unit_vector() * (rng.unit().cbrt() * radius))
        .collect()
}

/// Powłoka sferyczna o promieniu `radius`.
fn sphere_shell(rng: &mut Rng, n: usize, radius: f64) -> Vec<Vec3> {
    (0..n).map(|_| rng.unit_vector() * radius).collect()
}

/// Kostka jednorodna o połowie boku `radius`, wypełniona losowo.
///
/// Losowo, a nie na siatce, i to jest istotne: rozkład Poissona ma wbudowane
/// fluktuacje gęstości ~1/√N na każdej skali, czyli gotowe zarodki niestabilności
/// grawitacyjnej. Siatka regularna ich nie ma i musiałaby dostać zaburzenie
/// z zewnątrz, żeby cokolwiek się na niej zaczęło.
fn cube(rng: &mut Rng, n: usize, radius: f64) -> Vec<Vec3> {
    (0..n)
        .map(|_| {
            vec3(
                rng.uniform(-radius, radius),
                rng.uniform(-radius, radius),
                rng.uniform(-radius, radius),
            )
        })
        .collect()
}

/// Walec pełny o osi z, promieniu `radius` i wysokości równej średnicy.
///
/// Proporcje ustawia potem spłaszczenie, więc walec nie potrzebuje własnego
/// parametru długości: 0,2 daje krążek, 3 daje pręt. Pierwiastek z liczby losowej
/// jest konieczny — bez niego cząstki skupiłyby się przy osi, bo pole pierścienia
/// rośnie liniowo z promieniem.
fn cylinder(rng: &mut Rng, n: usize, radius: f64) -> Vec<Vec3> {
    (0..n)
        .map(|_| {
            let r = radius * rng.unit().sqrt();
            let a = rng.uniform(0.0, std::f64::consts::TAU);
            vec3(r * a.cos(), r * a.sin(), rng.uniform(-radius, radius))
        })
        .collect()
}

fn disk(rng: &mut Rng, n: usize, radius: f64, thickness: f64) -> Vec<Vec3> {
    let sigma = thickness.max(1e-6) * radius;
    (0..n)
        .map(|_| {
            let r = radius * rng.unit().sqrt();
            let a = rng.uniform(0.0, std::f64::consts::TAU);
            vec3(r * a.cos(), r * a.sin(), rng.normal(0.0, sigma))
        })
        .collect()
}

/// Torus o promieniu wiodącym `radius` i przekroju `thickness·radius`.
///
/// Przy małej grubości jest to zamknięte włókno bez końców i właśnie dlatego jest
/// to jedyny kształt w tym zestawie, który potrafi fragmentować bez dyssypacji.
/// Włókno o swobodnych końcach zapada się od nich do środka, a to zapadanie skaluje
/// się jak 1/grubość — czyli dokładnie tak samo jak wzrost zgęstek, więc pocienianie
/// nie rozdziela tych dwóch modów i globalny zawsze wygrywa. Torus nie ma końców,
/// więc ten mod po prostu nie istnieje, a przy nadanej rotacji nie ma też globalnego
/// zapadania promieniowego. Zostaje wyłącznie fragmentacja podłużna.
fn torus(rng: &mut Rng, n: usize, radius: f64, thickness: f64) -> Vec<Vec3> {
    let minor = thickness.max(1e-6) * radius;
    (0..n)
        .map(|_| {
            let theta = rng.uniform(0.0, std::f64::consts::TAU);
            let phi = rng.uniform(0.0, std::f64::consts::TAU);
            let rr = minor * rng.unit().sqrt();
            vec3(
                (radius + rr * phi.cos()) * theta.cos(),
                (radius + rr * phi.cos()) * theta.sin(),
                rr * phi.sin(),
            )
        })
        .collect()
}

fn gaussian(rng: &mut Rng, n: usize, radius: f64) -> Vec<Vec3> {
    (0..n).map(|_| rng.normal_vec(0.0, radius / 2.5)).collect()
}

/// Walec o długości 2·radius i gaussowskim przekroju o σ = thickness·radius.
///
/// Grubość jest osobnym parametrem, bo to ONA, a nie długość, wyznacza długość fali
/// fragmentacji (λ ≈ 3,6·σ). Przy sztywno wpisanym σ kształt był samopodobny: na
/// całą długość wypadało zawsze ~7 długości Jeansa, czyli za mało, żeby fragmentacja
/// wygrała z globalnym zapadaniem włókna do środka.
fn filament(rng: &mut Rng, n: usize, radius: f64, thickness: f64) -> Vec<Vec3> {
    let sigma = thickness.max(1e-6) * radius;
    (0..n)
        .map(|_| {
            vec3(
                rng.uniform(-radius, radius),
                rng.normal(0.0, sigma),
                rng.normal(0.0, sigma),
            )
        })
        .collect()
}

fn two_clumps(rng: &mut Rng, n: usize, radius: f64) -> Vec<Vec3> {
    let (sep, sigma) = (0.55 * radius, 0.16 * radius);
    let half = n / 2;
    (0..n)
        .map(|i| {
            let center = if i < half {
                vec3(-sep, 0.0, 0.0)
            } else {
                vec3(sep, 0.0, 0.0)
            };
            center + rng.normal_vec(0.0, sigma)
        })
        .collect()
}

/// Sfera Plummera — jedyny z tych rozkładów, który jest samouzgodnionym rozwiązaniem
/// równania Poissona, więc nadaje się na test równowagi.
///
/// Ogon jest ucięty na 6a (96% masy). Solver siatkowy dopasowuje pudło do PEŁNEJ
/// rozciągłości chmury, więc kilka cząstek na r = 30a obniżyłoby rozdzielczość
/// wszystkim pozostałym.
fn plummer(rng: &mut Rng, n: usize, radius: f64) -> Vec<Vec3> {
    let a = radius / 3.0;
    (0..n)
        .map(|_| {
            let x = rng.unit().max(1e-12);
            let r = a / (x.powf(-2.0 / 3.0) - 1.0).max(1e-12).sqrt();
            rng.unit_vector() * r.min(6.0 * a)
        })
        .collect()
}

/// Wylosuj położenia startowe zadanego kształtu.
///
/// `flatten` skaluje wyłącznie współrzędną z i działa na KAŻDY kształt, już po jego
/// wylosowaniu. Jeden mnożnik zamiast parametru proporcji w każdej funkcji osobno:
/// kula staje się plackiem albo cygarem, kostka płytą albo słupem, walec krążkiem
/// albo prętem. Rozdzielenie rozmiaru od proporcji ma tę zaletę, że zmiana kształtu
/// nie zmienia przy okazji skali, więc porównania między biegami pozostają uczciwe.
pub fn sample_positions(
    geometry: Geometry,
    rng: &mut Rng,
    n: usize,
    radius: f64,
    thickness: f64,
    flatten: f64,
) -> Vec<Vec3> {
    let mut positions = match geometry {
        Geometry::Ball => ball(rng, n, radius),
        Geometry::Cube => cube(rng, n, radius),
        Geometry::Cylinder => cylinder(rng, n, radius),
        Geometry::Disk => disk(rng, n, radius, thickness),
        Geometry::Torus => torus(rng, n, radius, thickness),
        Geometry::SphereShell => sphere_shell(rng, n, radius),
        Geometry::Filament => filament(rng, n, radius, thickness),
        Geometry::Gaussian => gaussian(rng, n, radius),
        Geometry::TwoClumps => two_clumps(rng, n, radius),
        Geometry::Plummer => plummer(rng, n, radius),
    };
    if (flatten - 1.0).abs() > 1e-12 {
        for p in &mut positions {
            p.z *= flatten;
        }
    }
    positions
}

/// `v_okrężna(r) = √(G·M(<r)/√(r²+ε²))` wokół środka masy.
///
/// `M(<r)` liczone z posortowanych promieni, więc koszt to O(N log N) i nie zależy od
/// solvera — to tylko warunek początkowy.
pub fn circular_speed(positions: &[Vec3], masses: &[f64], g: f64, softening: f64) -> Vec<f64> {
    let total: f64 = masses.iter().sum();
    let com = if total > 0.0 {
        positions
            .iter()
            .zip(masses.iter())
            .map(|(p, m)| *p * *m)
            .sum::<Vec3>()
            / total
    } else {
        ZERO
    };
    let radii: Vec<f64> = positions.iter().map(|p| (*p - com).norm()).collect();

    let mut order: Vec<usize> = (0..radii.len()).collect();
    order.sort_by(|a, b| radii[*a].total_cmp(&radii[*b]));

    let mut enclosed = vec![0.0f64; radii.len()];
    let mut running = 0.0;
    for &i in &order {
        running += masses[i];
        enclosed[i] = running;
    }

    radii
        .iter()
        .zip(enclosed.iter())
        .map(|(r, m)| (g * m / (r * r + softening * softening).sqrt()).sqrt())
        .collect()
}

/// Ile cząstek wystarcza do oszacowania energii potencjalnej kształtu.
///
/// `|U|` zależy od ROZKŁADU masy, nie od liczby próbek, więc podpróbka nosząca całą
/// masę układu daje tę samą wartość — a koszt O(m²) trzyma się wtedy w ułamku
/// sekundy zamiast rosnąć do minut przy stu tysiącach cząstek.
const VIRIAL_SAMPLE: usize = 3_000;

/// `|U|` kształtu, oszacowane na podpróbce dokładnym sumowaniem po parach.
///
/// Używa tego samego solvera i tej samej konwencji softeningu, którą liczy symulacja,
/// więc wirial zadany na starcie zgadza się z wiriałem, który potem pokazuje
/// diagnostyka. Własna implementacja sumy po parach rozjechałaby się z nią przy
/// pierwszej zmianie konwencji ε.
pub fn potential_energy(positions: &[Vec3], masses: &[f64], g: f64, softening: f64) -> f64 {
    let n = positions.len();
    let (sample_x, sample_m) = if n > VIRIAL_SAMPLE {
        let mut rng = Rng::seeded(0);
        let pick = rng.sample_indices(n, VIRIAL_SAMPLE);
        let total: f64 = masses.iter().sum();
        (
            pick.iter().map(|&i| positions[i]).collect::<Vec<_>>(),
            // Podpróbka musi nieść CAŁĄ masę układu, inaczej oszacowałaby potencjał
            // układu o masie m/N razy mniejszej, czyli za mały (N/m)² razy.
            vec![total / VIRIAL_SAMPLE as f64; VIRIAL_SAMPLE],
        )
    } else {
        (positions.to_vec(), masses.to_vec())
    };
    Exact::new()
        .compute(&sample_x, &sample_m, g, softening)
        .energy(&sample_m)
        .abs()
}

/// Dobierz mnożnik dyspersji tak, by energia kinetyczna trafiła w cel.
///
/// Rozwiązywane numerycznie, a nie wzorem `K = 3/2·Mσ²`, z dwóch powodów: energia
/// kinetyczna jest relatywistyczna ((γ−1)mc², nie ½mv²), a rotacja wnosi część
/// energii, której skalowanie dyspersji nie dotyczy. `K(s)` jest rosnąca, więc
/// bisekcja jest tu i wystarczająca, i odporna — w przeciwieństwie do wzoru
/// nierelatywistycznego, który przy β ≈ 0,5 myli się o kilkanaście procent.
fn scale_to_virial(
    rotation: &[Vec3],
    dispersion: &[Vec3],
    masses: &[f64],
    c: f64,
    target: f64,
) -> f64 {
    let kinetic = |scale: f64| -> f64 {
        rotation
            .iter()
            .zip(dispersion.iter())
            .zip(masses.iter())
            .map(|((rot, disp), m)| {
                let v = *rot + *disp * scale;
                let (v, _) = sr::clamp_initial_speed(v, c);
                match sr::momentum(*m, v, c) {
                    Ok(p) => sr::kinetic_energy(*m, p, c),
                    Err(_) => 0.0,
                }
            })
            .sum()
    };

    if kinetic(0.0) >= target {
        return 0.0;
    }
    let mut hi = 1.0;
    while kinetic(hi) < target && hi < 1e6 {
        hi *= 2.0;
    }
    let mut lo = 0.0;
    for _ in 0..60 {
        let mid = 0.5 * (lo + hi);
        if kinetic(mid) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Złóż stan początkowy z konfiguracji.
pub fn make_state(cfg: &Config) -> Spawned {
    let (sp, ph) = (&cfg.spawn, &cfg.physics);
    let mut rng = Rng::seeded(sp.seed);
    let n = sp.n_particles.max(1);
    let mut warnings = Vec::new();

    let positions = sample_positions(
        sp.geometry,
        &mut rng,
        n,
        sp.radius,
        sp.thickness,
        sp.flatten,
    );

    // Rozrzut mas jest tylko kosmetyczny; suma jest przeskalowana dokładnie do
    // zamawianej masy układu, żeby liczba cząstek nie wpływała na dynamikę.
    let mut masses: Vec<f64> = (0..n)
        .map(|_| rng.normal(1.0, sp.mass_spread).max(0.05))
        .collect();
    let raw_total: f64 = masses.iter().sum();
    let scale = sp.total_mass / raw_total.max(1e-300);
    for m in &mut masses {
        *m *= scale;
    }

    let rotation_part = if sp.rotation.abs() > 1e-9 {
        rotation_velocities(&positions, &masses, sp.rotation, ph.g, ph.softening)
    } else {
        vec![ZERO; n]
    };

    let dispersion_part = if sp.virial > 0.0 {
        // Dyspersja dobrana do energii potencjalnej TEGO kształtu, nie do wpisanej
        // liczby — dopiero wtedy dwa różne kształty można ze sobą porównywać.
        let target = 0.5 * sp.virial * potential_energy(&positions, &masses, ph.g, ph.softening);
        let noise: Vec<Vec3> = (0..n).map(|_| rng.normal_vec(0.0, 1.0)).collect();
        let s = scale_to_virial(&rotation_part, &noise, &masses, ph.c, target);
        noise.into_iter().map(|v| v * s).collect()
    } else if sp.temperature.abs() > 1e-9 {
        let sigma = sp.temperature * ph.c / 3.0f64.sqrt();
        (0..n).map(|_| rng.normal_vec(0.0, sigma)).collect()
    } else {
        vec![ZERO; n]
    };

    // Warunek początkowy musi być fizyczny: |v| < c z zapasem.
    let mut clamped = 0usize;
    let mut momenta = Vec::with_capacity(n);
    for i in 0..n {
        let (v, hit) = sr::clamp_initial_speed(rotation_part[i] + dispersion_part[i], ph.c);
        if hit {
            clamped += 1;
        }
        momenta.push(sr::momentum(masses[i], v, ph.c).unwrap_or(ZERO));
    }
    if clamped > 0 {
        // Przycięcie ratuje całkowanie, ale niszczy zamawiany warunek początkowy:
        // obcięta orbita kołowa nie jest już kołowa, a układ może wystartować
        // z energią dodatnią i po prostu się rozlecieć. Lepiej powiedzieć to wprost
        // niż pozwolić zobaczyć „zepsutą fizykę".
        let share = 100.0 * clamped as f64 / n as f64;
        warnings.push(format!(
            "{share:.1}% cząstek chciało lecieć szybciej niż {:.0}% c i zostało przyciętych \
             — zamawiany warunek początkowy jest nieosiągalny przy G={:.4} i c={:.1}. Zmniejsz G \
             lub rotację, albo podnieś c (patrz gravity_for_beta).",
            100.0 * sr::MAX_INITIAL_BETA,
            ph.g,
            ph.c
        ));
    }

    // Usuń pęd środka masy, żeby układ nie odpływał z kadru.
    let drift = momenta.iter().copied().sum::<Vec3>() / n as f64;
    for p in &mut momenta {
        *p -= drift;
    }

    let state = State::new(positions, momenta, masses)
        .expect("spawn buduje wszystkie trzy tablice o długości n");
    Spawned { state, warnings }
}

/// Prędkości styczne wokół osi z, w ułamku prędkości okrężnej.
fn rotation_velocities(
    positions: &[Vec3],
    masses: &[f64],
    fraction: f64,
    g: f64,
    softening: f64,
) -> Vec<Vec3> {
    let total: f64 = masses.iter().sum();
    let com = if total > 0.0 {
        positions
            .iter()
            .zip(masses.iter())
            .map(|(p, m)| *p * *m)
            .sum::<Vec3>()
            / total
    } else {
        ZERO
    };
    let axis = vec3(0.0, 0.0, 1.0);
    let v_circ = circular_speed(positions, masses, g, softening);
    positions
        .iter()
        .zip(v_circ.iter())
        .map(|(p, v)| {
            let tangent = axis.cross(*p - com);
            let norm = tangent.norm();
            if norm > 1e-12 {
                // Cząstki na osi obrotu nie mają zdefiniowanej stycznej — zostają
                // w spoczynku.
                tangent * (fraction * v / norm)
            } else {
                ZERO
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sr::config::{PhysicsConfig, SpawnConfig};

    fn cfg_for(geometry: Geometry, n: usize) -> Config {
        Config {
            spawn: SpawnConfig {
                geometry,
                n_particles: n,
                radius: 8.0,
                ..SpawnConfig::default()
            },
            ..Config::default()
        }
    }

    #[test]
    fn every_geometry_produces_finite_particles() {
        let mut rng = Rng::seeded(1);
        for geometry in Geometry::ALL {
            let x = sample_positions(geometry, &mut rng, 500, 8.0, 0.1, 1.0);
            assert_eq!(x.len(), 500, "{geometry:?}");
            assert!(x.iter().all(|p| p.is_finite()), "{geometry:?}");
            let span = crate::grid::center_span(&x).1;
            assert!(span > 0.0 && span.is_finite(), "{geometry:?}: span={span}");
        }
    }

    /// Kula ma mieć promień nieprzekraczający zamówionego i wypełniać go — inaczej
    /// „promień" w panelu nie znaczy tego, co jest napisane.
    #[test]
    fn ball_fills_its_radius() {
        let mut rng = Rng::seeded(2);
        let x = sample_positions(Geometry::Ball, &mut rng, 5_000, 8.0, 0.1, 1.0);
        let max_r = x.iter().map(|p| p.norm()).fold(0.0f64, f64::max);
        assert!(max_r <= 8.0 + 1e-9, "max r={max_r}");
        assert!(max_r > 7.0, "kula nie dochodzi do brzegu: {max_r}");
    }

    /// Objętość kuli rośnie jak r³, więc połowa masy leży za r = 2^(−1/3)·R ≈ 0,794R.
    /// Ten test wyłapuje brak pierwiastka trzeciego stopnia w losowaniu promienia,
    /// czyli błąd dający chmurę skupioną w środku.
    #[test]
    fn ball_is_uniform_in_volume() {
        let mut rng = Rng::seeded(3);
        let x = sample_positions(Geometry::Ball, &mut rng, 20_000, 1.0, 0.1, 1.0);
        let inside = x.iter().filter(|p| p.norm() < 0.5).count() as f64 / x.len() as f64;
        // kula o promieniu ½ zajmuje ⅛ objętości
        assert!((inside - 0.125).abs() < 0.01, "udział={inside}");
    }

    #[test]
    fn shell_has_constant_radius() {
        let mut rng = Rng::seeded(4);
        let x = sample_positions(Geometry::SphereShell, &mut rng, 1_000, 5.0, 0.1, 1.0);
        assert!(x.iter().all(|p| (p.norm() - 5.0).abs() < 1e-9));
    }

    #[test]
    fn flatten_squashes_only_z() {
        let mut rng = Rng::seeded(5);
        let round = sample_positions(Geometry::Ball, &mut rng, 4_000, 8.0, 0.1, 1.0);
        let mut rng = Rng::seeded(5);
        let flat = sample_positions(Geometry::Ball, &mut rng, 4_000, 8.0, 0.1, 0.25);
        let extent = |v: &[Vec3], axis: usize| {
            v.iter().map(|p| p.get(axis).abs()).fold(0.0f64, f64::max)
        };
        assert!((extent(&round, 0) - extent(&flat, 0)).abs() < 1e-9, "x zmienione");
        assert!((extent(&round, 1) - extent(&flat, 1)).abs() < 1e-9, "y zmienione");
        assert!((extent(&flat, 2) / extent(&round, 2) - 0.25).abs() < 1e-9);
    }

    #[test]
    fn torus_has_a_hole() {
        let mut rng = Rng::seeded(6);
        let x = sample_positions(Geometry::Torus, &mut rng, 5_000, 8.0, 0.05, 1.0);
        let min_cyl = x
            .iter()
            .map(|p| (p.x * p.x + p.y * p.y).sqrt())
            .fold(f64::INFINITY, f64::min);
        assert!(min_cyl > 6.0, "torus nie ma dziury: min r={min_cyl}");
    }

    #[test]
    fn two_clumps_are_separated() {
        let mut rng = Rng::seeded(7);
        let x = sample_positions(Geometry::TwoClumps, &mut rng, 4_000, 10.0, 0.1, 1.0);
        let left = x.iter().filter(|p| p.x < 0.0).count();
        assert!(left > 1_800 && left < 2_200, "podział {left}/4000");
    }

    #[test]
    fn total_mass_matches_request_regardless_of_count() {
        for n in [100usize, 4_000, 20_000] {
            let mut cfg = cfg_for(Geometry::Ball, n);
            cfg.spawn.total_mass = 12_345.0;
            let s = make_state(&cfg).state;
            assert!(
                (s.total_mass() - 12_345.0).abs() / 12_345.0 < 1e-12,
                "n={n} masa={}",
                s.total_mass()
            );
        }
    }

    /// Bez tego układ odpływałby z kadru — a to artefakt, który wygląda jak fizyka.
    #[test]
    fn total_momentum_starts_at_zero() {
        let mut cfg = cfg_for(Geometry::Disk, 2_000);
        cfg.spawn.rotation = 1.0;
        let s = make_state(&cfg).state;
        let scale: f64 = s.momenta.iter().map(|p| p.norm()).sum();
        assert!(s.total_momentum().norm() / scale < 1e-12);
    }

    #[test]
    fn rotation_produces_angular_momentum() {
        let mut cold = cfg_for(Geometry::Disk, 2_000);
        cold.spawn.rotation = 0.0;
        cold.spawn.temperature = 0.0;
        let still = make_state(&cold).state;
        assert!(still.angular_momentum().norm() < 1e-9);

        let mut spinning = cfg_for(Geometry::Disk, 2_000);
        spinning.spawn.rotation = 1.0;
        let turning = make_state(&spinning).state;
        assert!(turning.angular_momentum().norm() > 0.0);
    }

    /// Zadany wirial musi być OSIĄGNIĘTY, bo to jedyna wielkość pozwalająca uczciwie
    /// porównywać różne kształty. Sprawdzamy przez niezależny pomiar 2K/|U|.
    #[test]
    fn requested_virial_is_reached() {
        for target in [0.5, 1.0, 1.5] {
            let mut cfg = cfg_for(Geometry::Ball, 1_500);
            cfg.spawn.rotation = 0.0;
            cfg.spawn.virial = target;
            let s = make_state(&cfg).state;

            let c = cfg.physics.c;
            let kinetic: f64 = (0..s.n())
                .map(|i| sr::kinetic_energy(s.masses[i], s.momenta[i], c))
                .sum();
            let u = potential_energy(
                &s.positions,
                &s.masses,
                cfg.physics.g,
                cfg.physics.softening,
            );
            let got = 2.0 * kinetic / u;
            assert!(
                (got - target).abs() / target < 0.15,
                "cel {target}, wyszło {got}"
            );
        }
    }

    /// Ta sama dyspersja prędkości daje INNY wirial w każdym kształcie — i to jest
    /// jedyny powód, dla którego parametr `virial` istnieje.
    ///
    /// Test pilnuje dwóch rzeczy naraz. Po pierwsze rozrzutu: jeżeli spadłby do
    /// kilkunastu procent, `virial` byłby zbędną komplikacją, a nastawianie
    /// temperatury — uczciwym sposobem porównywania kształtów. Po drugie kolejności:
    /// kostka ma masę wypchniętą w narożniki, więc jej `|U|` jest najmniejsze i przy
    /// ustalonej temperaturze wychodzi najgorętsza; dwie gromady mają masę skupioną
    /// w dwóch punktach, więc `|U|` jest największe i wychodzą najzimniejsze. Odwrócenie
    /// tej kolejności znaczyłoby błąd znaku albo skali w [`potential_energy`].
    ///
    /// Liczby z tego testu stoją w README jako tabela — dlatego są tu, a nie tylko tam.
    #[test]
    fn the_same_temperature_means_a_different_virial_in_each_shape() {
        let virial_of = |geometry: Geometry| -> f64 {
            let mut cfg = cfg_for(geometry, 3_000);
            cfg.spawn.rotation = 0.0;
            cfg.spawn.virial = 0.0;
            cfg.spawn.temperature = 0.05;
            let s = make_state(&cfg).state;
            let kinetic: f64 = (0..s.n())
                .map(|i| sr::kinetic_energy(s.masses[i], s.momenta[i], cfg.physics.c))
                .sum();
            let u = potential_energy(
                &s.positions,
                &s.masses,
                cfg.physics.g,
                cfg.physics.softening,
            );
            2.0 * kinetic / u
        };

        let cube = virial_of(Geometry::Cube);
        let clumps = virial_of(Geometry::TwoClumps);
        let ball = virial_of(Geometry::Ball);

        assert!(cube > ball, "kostka {cube} nie jest gorętsza od kuli {ball}");
        assert!(
            ball > clumps,
            "kula {ball} nie jest gorętsza od dwóch gromad {clumps}"
        );
        assert!(
            cube / clumps > 1.8,
            "rozrzut między kształtami zmalał do {:.2}× — parametr `virial` traci sens",
            cube / clumps
        );
    }

    /// `virial` musi przejmować kontrolę nad `temperature`, inaczej dwa parametry
    /// opisujące to samo dawałyby wynik zależny od kolejności ich ustawienia.
    #[test]
    fn virial_overrides_temperature() {
        let mut cfg = cfg_for(Geometry::Ball, 800);
        cfg.spawn.rotation = 0.0;
        cfg.spawn.temperature = 0.4;
        cfg.spawn.virial = 1.0;
        let with_virial = make_state(&cfg).state;

        cfg.spawn.virial = 0.0;
        let with_temperature = make_state(&cfg).state;

        let kinetic = |s: &State| -> f64 {
            (0..s.n())
                .map(|i| sr::kinetic_energy(s.masses[i], s.momenta[i], cfg.physics.c))
                .sum()
        };
        assert!(
            (kinetic(&with_virial) - kinetic(&with_temperature)).abs()
                / kinetic(&with_temperature)
                > 0.05,
            "wirial nie zmienił energii kinetycznej"
        );
    }

    /// Nieosiągalny warunek początkowy MUSI być zgłoszony, a nie po cichu przycięty.
    #[test]
    fn unreachable_initial_condition_is_reported() {
        let cfg = Config {
            spawn: SpawnConfig {
                geometry: Geometry::Ball,
                n_particles: 500,
                rotation: 1.0,
                total_mass: 1e6,
                radius: 1.0,
                ..SpawnConfig::default()
            },
            physics: PhysicsConfig {
                g: 10.0,
                c: 1.0,
                ..PhysicsConfig::default()
            },
            ..Config::default()
        };
        let spawned = make_state(&cfg);
        assert!(
            !spawned.warnings.is_empty(),
            "przycięcie prędkości przemilczane"
        );
        assert!(spawned.state.is_finite());
    }

    #[test]
    fn same_seed_reproduces_the_state() {
        let cfg = cfg_for(Geometry::Plummer, 500);
        let a = make_state(&cfg).state;
        let b = make_state(&cfg).state;
        for i in 0..a.n() {
            assert_eq!(a.positions[i], b.positions[i]);
            assert_eq!(a.momenta[i], b.momenta[i]);
        }
    }

    #[test]
    fn different_seed_changes_the_state() {
        let cfg = cfg_for(Geometry::Plummer, 500);
        let a = make_state(&cfg).state;
        let mut other = cfg.clone();
        other.spawn.seed += 1;
        let b = make_state(&other).state;
        assert!(a.positions.iter().zip(b.positions.iter()).any(|(p, q)| p != q));
    }

    /// Prędkość okrężna musi rosnąć z masą zamkniętą i maleć z promieniem —
    /// sprawdzane na układzie, dla którego znamy odpowiedź analitycznie.
    #[test]
    fn circular_speed_follows_enclosed_mass() {
        let x = vec![ZERO, vec3(4.0, 0.0, 0.0)];
        let m = vec![100.0, 1e-9];
        let g = 2.0;
        let v = circular_speed(&x, &m, g, 0.0);
        // Druga cząstka jest praktycznie bezmasowa, więc widzi masę pierwszej;
        // środek masy leży w niej, czyli r = 4.
        let expected = (g * 100.0 / 4.0).sqrt();
        assert!((v[1] - expected).abs() / expected < 1e-6, "v={}", v[1]);
    }

    /// Oszacowanie |U| na podpróbce musi zgadzać się z pełnym rachunkiem, bo na nim
    /// opiera się cały mechanizm zadawania wiriału.
    #[test]
    fn subsampled_potential_matches_full_sum() {
        let mut rng = Rng::seeded(9);
        let n = 6_000;
        let x = sample_positions(Geometry::Ball, &mut rng, n, 8.0, 0.1, 1.0);
        let m = vec![4_000.0 / n as f64; n];
        let sampled = potential_energy(&x, &m, 0.16, 0.25);
        let full = Exact::new().compute(&x, &m, 0.16, 0.25).energy(&m).abs();
        assert!(
            (sampled - full).abs() / full < 0.1,
            "podpróbka {sampled}, pełne {full}"
        );
    }
}
