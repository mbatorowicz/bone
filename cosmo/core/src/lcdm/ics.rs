//! Warunki początkowe: gaussowskie pole gęstości i przybliżenie Zel'dovicha (1LPT).
//!
//! Cząstki startują z siatki regularnej i są z niej PRZESUNIĘTE o `D(a)Ψ`, a ich pędy
//! wynikają z tego samego `Ψ`. Ta spójność jest istotą metody: prędkość niezgodna
//! z przesunięciem znaczyłaby, że układ startuje z rozbieżnym modem zanikającym, który
//! przez pierwszą część biegu zjada strukturę zamiast ją budować.
//!
//! # Znak przesunięcia
//!
//! Przybliżenie Zel'dovicha wymaga `∇·Ψ = −δ`: materia ma SPŁYWAĆ do zagęszczeń.
//! Bierzemy `Ψ = −∇φ` przy `∇²φ = δ`, czyli w przestrzeni Fouriera
//! `φ_k = −δ_k/k²` i `Ψ_k = i k δ_k/k²`. Znak przeciwny daje realizację odpowiadającą
//! polu `−δ`, więc statystycznie tak samo prawdopodobną i na obrazku nierozróżnialną —
//! i właśnie dlatego pilnuje go test, a nie oko.
//!
//! # Symetria hermitowska
//!
//! Pole `δ_k` jest budowane od razu jako hermitowskie: dla każdej pary `(k, −k)`
//! losujemy raz, a dla modów samosprzężonych (`k ≡ −k`) losujemy liczbę rzeczywistą
//! o pełnej wariancji. Wersja, która losuje wszystkie mody niezależnie i dopiero potem
//! symetryzuje przez uśrednienie `F(k)` i `conj F(−k)`, daje pole rzeczywiste, ale
//! z DWUKROTNIE mniejszą wariancją — normalizacja `σ₈` przestaje wtedy cokolwiek
//! znaczyć, a objaw jest niewidoczny, bo obrazek nadal wygląda jak kosmologia.

use rustfft::num_complex::Complex;

use crate::fft::{Direction, Fft3};
use crate::lcdm::cosmology::Cosmology;
use crate::lcdm::power::PowerSpectrum;
use crate::rng::Rng;
use crate::vec3::{vec3, Vec3, ZERO};

pub struct InitialState {
    pub positions: Vec<Vec3>,
    pub momenta: Vec<Vec3>,
    /// Masa jednej cząstki [10¹⁰ M☉/h].
    pub mass: f64,
    pub box_size: f64,
    pub a: f64,
    /// `σ(δ)` pola startowego na skali oczka siatki.
    ///
    /// Warta pokazania, bo jest jedyną liczbą mówiącą, JAK SILNE są zaburzenia,
    /// od których zaczyna się bieg. Wartość rzędu jedności przy `z = 49` znaczy, że
    /// przybliżenie Zel'dovicha już nie obowiązuje i wynik jest fikcją, choćby
    /// wyglądał przekonująco.
    pub delta_rms: f64,
}

/// Liniowe pole gęstości i odpowiadające mu przesunięcie Zel'dovicha.
///
/// Trzymane razem, bo razem są sprawdzalne: `∇·Ψ = −δ` to jedyny warunek wiążący te
/// dwie tablice i jedyny, którego złamanie odwraca sens całej symulacji.
struct LinearField {
    ng: usize,
    box_size: f64,
    /// Kontrast gęstości w przestrzeni rzeczywistej, indeksowany `(z·ng + y)·ng + x`.
    delta: Vec<f64>,
    /// Przesunięcie [Mpc/h] przy `D = 1`.
    psi: Vec<Vec3>,
}

impl LinearField {
    fn generate(power: &PowerSpectrum, a: f64, box_size: f64, ng: usize, seed: u64) -> Self {
        let cells = ng * ng * ng;
        let volume = box_size.powi(3);
        let dk = std::f64::consts::TAU / box_size;
        let mut rng = Rng::seeded(seed);

        // `δ(x) = (1/N) Σ δ_k e^{ikx}`, więc dla `⟨δ²⟩ = (1/V) Σ P(k)` potrzeba
        // `⟨|δ_k|²⟩ = N² P(k)/V`. Ten czynnik `N` jest źródłem najczęstszego błędu
        // normalizacji w tym miejscu: bez niego amplituda zależy od rozmiaru oczka.
        let scale = cells as f64 / volume.sqrt();

        let mut spectrum = vec![Complex::new(0.0f32, 0.0); cells];
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let idx = (iz * ng + iy) * ng + ix;
                    let mirror = ((ng - iz) % ng * ng + (ng - iy) % ng) * ng + (ng - ix) % ng;
                    if mirror < idx {
                        continue;
                    }
                    let k = wave_vector(ix, iy, iz, ng) * dk;
                    let k_norm = k.norm();
                    if k_norm <= 0.0 {
                        continue;
                    }
                    let sigma = scale * power.p(k_norm, a).max(0.0).sqrt();
                    if mirror == idx {
                        // Mod samosprzężony musi być rzeczywisty, ale z pełną wariancją.
                        spectrum[idx] = Complex::new((sigma * rng.standard_normal()) as f32, 0.0);
                    } else {
                        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
                        let re = sigma * inv_sqrt2 * rng.standard_normal();
                        let im = sigma * inv_sqrt2 * rng.standard_normal();
                        spectrum[idx] = Complex::new(re as f32, im as f32);
                        spectrum[mirror] = Complex::new(re as f32, -im as f32);
                    }
                }
            }
        }

        let mut fft = Fft3::new();
        let norm = 1.0 / cells as f32;

        let mut displacement = vec![ZERO; cells];
        for axis in 0..3 {
            let mut field = vec![Complex::new(0.0f32, 0.0); cells];
            for iz in 0..ng {
                for iy in 0..ng {
                    for ix in 0..ng {
                        let idx = (iz * ng + iy) * ng + ix;
                        let k = wave_vector(ix, iy, iz, ng) * dk;
                        let k2 = k.norm_squared();
                        if k2 <= 0.0 {
                            continue;
                        }
                        // `Ψ_k = i k δ_k / k²`
                        let factor = (k.get(axis) / k2) as f32;
                        let d = spectrum[idx];
                        field[idx] = Complex::new(-factor * d.im, factor * d.re);
                    }
                }
            }
            fft.run(&mut field, ng, Direction::Inverse);
            for (slot, value) in displacement.iter_mut().zip(field.iter()) {
                slot.set(axis, (value.re * norm) as f64);
            }
        }

        let mut density = spectrum;
        fft.run(&mut density, ng, Direction::Inverse);
        Self {
            ng,
            box_size,
            delta: density.iter().map(|v| (v.re * norm) as f64).collect(),
            psi: displacement,
        }
    }

    fn cell_size(&self) -> f64 {
        self.box_size / self.ng as f64
    }

    /// `σ(δ)` — odchylenie standardowe kontrastu gęstości na siatce.
    fn delta_rms(&self) -> f64 {
        let n = self.delta.len().max(1) as f64;
        let mean = self.delta.iter().sum::<f64>() / n;
        (self.delta.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt()
    }

    /// `∇·Ψ` różnicą centralną drugiego rzędu, z zawijaniem po siatce.
    ///
    /// Zawijanie jest tu poprawne, a nie wygodne: pole jest z definicji periodyczne
    /// w pudle warunków początkowych, bo powstało z FFT.
    #[cfg(test)]
    fn divergence(&self) -> Vec<f64> {
        let ng = self.ng;
        let h = self.cell_size();
        let at = |ix: usize, iy: usize, iz: usize| -> Vec3 {
            self.psi[((iz % ng) * ng + (iy % ng)) * ng + (ix % ng)]
        };
        let mut out = vec![0.0; ng * ng * ng];
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let dx = at(ix + 1, iy, iz).x - at(ix + ng - 1, iy, iz).x;
                    let dy = at(ix, iy + 1, iz).y - at(ix, iy + ng - 1, iz).y;
                    let dz = at(ix, iy, iz + 1).z - at(ix, iy, iz + ng - 1).z;
                    out[(iz * ng + iy) * ng + ix] = (dx + dy + dz) / (2.0 * h);
                }
            }
        }
        out
    }
}

pub fn make_initial_state(
    cosmology: Cosmology,
    box_size: f64,
    n_grid: usize,
    z_start: f64,
    seed: u64,
) -> InitialState {
    let ng = n_grid.max(2);
    let a = 1.0 / (1.0 + z_start.max(0.0));
    let power = PowerSpectrum::eisenstein_hu(cosmology);
    let field = LinearField::generate(&power, a, box_size, ng, seed);

    let cells = ng * ng * ng;
    let mass = cosmology.mean_matter_density() * box_size.powi(3) / cells as f64;
    let h_cell = field.cell_size();

    // `Ψ` jest już wzięte przy `D = 1`, bo `P(k, a)` zawiera `D(a)²`. Mnożenie go
    // jeszcze raz przez `D(a)` byłoby podniesieniem amplitudy do kwadratu — dlatego
    // przesunięcie jest tu użyte wprost.
    let f = cosmology.growth_rate(a);
    let hubble = cosmology.h_of_a(a);
    // `p = a²ẋ`, a w przybliżeniu Zel'dovicha `ẋ = f H Ψ`.
    let momentum_scale = a * a * f * hubble;

    let mut positions = Vec::with_capacity(cells);
    let mut momenta = Vec::with_capacity(cells);
    for iz in 0..ng {
        for iy in 0..ng {
            for ix in 0..ng {
                let idx = (iz * ng + iy) * ng + ix;
                let lattice = vec3(
                    (ix as f64 + 0.5) * h_cell,
                    (iy as f64 + 0.5) * h_cell,
                    (iz as f64 + 0.5) * h_cell,
                );
                positions.push(lattice + field.psi[idx]);
                momenta.push(field.psi[idx] * momentum_scale);
            }
        }
    }

    InitialState {
        positions,
        momenta,
        mass,
        box_size,
        a,
        delta_rms: field.delta_rms(),
    }
}

/// Wektor falowy w jednostkach `2π/L`, z modami powyżej Nyquista jako ujemnymi.
fn wave_vector(ix: usize, iy: usize, iz: usize, ng: usize) -> Vec3 {
    vec3(
        wave_number(ix, ng),
        wave_number(iy, ng),
        wave_number(iz, ng),
    )
}

fn wave_number(i: usize, ng: usize) -> f64 {
    let half = ng as i64 / 2;
    let v = i as i64;
    if v > half {
        (v - ng as i64) as f64
    } else {
        v as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn field(ng: usize, box_size: f64, seed: u64) -> LinearField {
        let cosmology = Cosmology::planck18();
        let power = PowerSpectrum::eisenstein_hu(cosmology);
        LinearField::generate(&power, 1.0, box_size, ng, seed)
    }

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len() as f64
    }

    fn variance(values: &[f64]) -> f64 {
        let m = mean(values);
        values.iter().map(|v| (v - m).powi(2)).sum::<f64>() / values.len() as f64
    }

    /// Przy `z = 49` zaburzenia muszą być jeszcze liniowe, czyli `σ(δ) ≪ 1`.
    /// Ta liczba jest raportowana w panelu, więc musi być prawdziwa.
    #[test]
    fn initial_contrast_is_still_linear() {
        let st = make_initial_state(Cosmology::planck18(), 100.0, 16, 49.0, 8);
        assert!(st.delta_rms > 0.0, "pole startowe jest puste");
        assert!(st.delta_rms < 0.3, "σ(δ) = {} przy z=49", st.delta_rms);
    }

    #[test]
    fn state_has_one_particle_per_cell_and_finite_numbers() {
        let st = make_initial_state(Cosmology::planck18(), 100.0, 16, 49.0, 1);
        assert_eq!(st.positions.len(), 16 * 16 * 16);
        assert_eq!(st.momenta.len(), st.positions.len());
        assert!(st.mass > 0.0);
        assert!(st.positions.iter().all(|p| p.is_finite()));
        assert!(st.momenta.iter().all(|p| p.is_finite()));
    }

    /// Masa całkowita musi odtwarzać `Ω_m ρ_kryt V`. Gdyby nie odtwarzała, cała
    /// symulacja liczyłaby inny wszechświat niż ten z panelu.
    #[test]
    fn total_mass_matches_the_mean_density() {
        let cosmology = Cosmology::planck18();
        let box_size = 64.0;
        let st = make_initial_state(cosmology, box_size, 12, 49.0, 3);
        let total = st.mass * st.positions.len() as f64;
        let expected = cosmology.mean_matter_density() * box_size.powi(3);
        assert!((total - expected).abs() / expected < 1e-12, "M = {total}");
    }

    /// Pole gęstości musi wyjść RZECZYWISTE — to sprawdzian symetrii hermitowskiej.
    /// Część urojona pojawia się natychmiast, gdy któraś para `(k, −k)` się rozjedzie.
    #[test]
    fn density_field_is_real_and_has_zero_mean() {
        let f = field(16, 100.0, 5);
        assert!(mean(&f.delta).abs() < 1e-4, "średnia δ = {}", mean(&f.delta));
        assert!(f.delta.iter().all(|v| v.is_finite()));
    }

    /// Sedno przybliżenia Zel'dovicha: `∇·Ψ = −δ`. Zły znak odwraca rolę pustek
    /// i zagęszczeń, a zła skala psuje amplitudę struktury.
    #[test]
    fn divergence_of_displacement_is_minus_delta() {
        let f = field(24, 200.0, 11);
        let div = f.divergence();
        let correlation: f64 = div
            .iter()
            .zip(f.delta.iter())
            .map(|(d, delta)| d * delta)
            .sum::<f64>()
            / (variance(&div).sqrt() * variance(&f.delta).sqrt() * div.len() as f64);
        assert!(
            correlation < -0.9,
            "∇·Ψ nie jest −δ, korelacja = {correlation}"
        );
    }

    /// Amplituda: wariancja pola w przestrzeni rzeczywistej musi być równa sumie
    /// `P(k)/V` po modach, które siatka faktycznie reprezentuje. To jedyny test
    /// wiążący normalizację `σ₈` z tym, co naprawdę trafia do symulacji.
    #[test]
    fn variance_matches_the_sum_over_represented_modes() {
        let ng = 20;
        let box_size = 200.0;
        let cosmology = Cosmology::planck18();
        let power = PowerSpectrum::eisenstein_hu(cosmology);
        let f = LinearField::generate(&power, 1.0, box_size, ng, 7);

        let dk = std::f64::consts::TAU / box_size;
        let volume = box_size.powi(3);
        let mut expected = 0.0;
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let k = wave_vector(ix, iy, iz, ng).norm() * dk;
                    if k > 0.0 {
                        expected += power.p(k, 1.0) / volume;
                    }
                }
            }
        }
        let measured = variance(&f.delta);
        assert!(
            (measured / expected - 1.0).abs() < 0.15,
            "wariancja {measured}, oczekiwano {expected}"
        );
    }

    /// Amplituda nie może zależeć od rozdzielczości inaczej niż przez zakres modów.
    /// Zgubiony czynnik `N` objawiłby się tu jako różnica rzędów wielkości.
    #[test]
    fn amplitude_does_not_scale_with_the_cell_size() {
        let coarse = variance(&field(12, 200.0, 2).delta);
        let fine = variance(&field(24, 200.0, 2).delta);
        // Drobniejsza siatka dodaje mody, więc wariancja rośnie — ale nie o rzędy.
        assert!(fine > coarse, "{fine} vs {coarse}");
        assert!(fine < 6.0 * coarse, "{fine} vs {coarse}");
    }

    #[test]
    fn same_seed_reproduces_the_field() {
        let a = field(12, 100.0, 42);
        let b = field(12, 100.0, 42);
        assert_eq!(a.delta, b.delta);
        let c = field(12, 100.0, 43);
        assert_ne!(a.delta, c.delta);
    }

    /// Pędy muszą być zgodne z przesunięciami, bo inaczej start zawiera mod
    /// zanikający. Iloraz jest tą samą stałą dla każdej cząstki.
    #[test]
    fn momenta_are_proportional_to_displacements() {
        let cosmology = Cosmology::planck18();
        let z = 49.0;
        let a = 1.0 / (1.0 + z);
        let st = make_initial_state(cosmology, 100.0, 12, z, 9);
        let h_cell = 100.0 / 12.0;
        let expected = a * a * cosmology.growth_rate(a) * cosmology.h_of_a(a);

        let mut checked = 0;
        for (i, (p, q)) in st.positions.iter().zip(st.momenta.iter()).enumerate() {
            let ix = i % 12;
            let iy = (i / 12) % 12;
            let iz = i / 144;
            let lattice = vec3(
                (ix as f64 + 0.5) * h_cell,
                (iy as f64 + 0.5) * h_cell,
                (iz as f64 + 0.5) * h_cell,
            );
            let psi = *p - lattice;
            if psi.norm() < 1e-9 {
                continue;
            }
            let ratio = q.norm() / psi.norm();
            assert!(
                (ratio / expected - 1.0).abs() < 1e-9,
                "cząstka {i}: iloraz {ratio}, oczekiwano {expected}"
            );
            checked += 1;
        }
        assert!(checked > 1000, "sprawdzono tylko {checked} cząstek");
    }

    /// Przesunięcia przy `z = 49` muszą być małe względem oczka: przybliżenie
    /// Zel'dovicha jest liniowe i traci sens, gdy trajektorie zaczynają się przecinać.
    #[test]
    fn displacements_are_small_at_high_redshift() {
        let ng = 16;
        let box_size = 100.0;
        let h_cell = box_size / ng as f64;
        let st = make_initial_state(Cosmology::planck18(), box_size, ng, 49.0, 4);
        let worst = st
            .positions
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let ix = i % ng;
                let iy = (i / ng) % ng;
                let iz = i / (ng * ng);
                let lattice = vec3(
                    (ix as f64 + 0.5) * h_cell,
                    (iy as f64 + 0.5) * h_cell,
                    (iz as f64 + 0.5) * h_cell,
                );
                (*p - lattice).norm() / h_cell
            })
            .fold(0.0f64, f64::max);
        assert!(worst < 1.0, "największe przesunięcie {worst} oczka");
    }
}
