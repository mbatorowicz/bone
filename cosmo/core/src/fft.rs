//! Trójwymiarowa FFT na siatce sześciennej, w miejscu.
//!
//! `rustfft` transformuje ciągły wektor, a siatka 3D wymaga trzech przebiegów po
//! kolejnych osiach. Oba miejsca, które tego potrzebują — solver PM i warunki
//! początkowe ΛCDM — używają tego jednego modułu. Rozwinięcie przebiegów na miejscu
//! wywołania dawałoby sześć bloków różniących się wyłącznie wzorcem indeksowania,
//! a błąd w kroku jednej osi przechodzi przez większość testów.
//!
//! Konwencja indeksowania: `idx = (z·ng + y)·ng + x`, czyli x jest ciągłe.
//! Wynik transformaty odwrotnej NIE jest normalizowany; skalowanie przez `ng³`
//! należy do wywołującego, bo splot i tak mnoży przez własne czynniki i jedno
//! dodatkowe przejście po tablicy 512³ jest tu wymierną ceną.
//!
//! # Dlaczego wszystkie trzy przebiegi są równoległe
//!
//! To jest najdroższe miejsce w całym programie: solver PM woła transformatę dwa razy
//! na krok, a na siatce 384³ jeden przebieg to 147 tysięcy linii. Wersja szeregowa
//! liczyła krok presetu `fragmentation` **10 sekund**, z czego niemal całość szła na
//! osie y i z.
//!
//! Równoległość wymaga podziału tablicy na kawałki, do których wątki mają WYŁĄCZNY
//! dostęp. Dla każdej osi ten podział wygląda inaczej:
//!
//! - **x** — linia jest ciągła, więc `par_chunks_mut(ng)` wystarcza;
//! - **y** — linie jednego płata `z = const` leżą w ciągłym bloku `ng²`, więc płaty
//!   dzielą się rozłącznie i każdy trafia do jednego wątku;
//! - **z** — linia przecina wszystkie płaty, więc żaden ciągły podział nie działa.
//!   Zamiast tego rozcinamy tablicę na wiersze `(z, y)` po `ng` elementów i grupujemy
//!   te wiersze po `y`. Grupowanie przekłada same referencje, nie dane, a każda grupa
//!   dostaje wyłączne wiersze — więc kompilator sam potwierdza rozłączność i nie ma
//!   tu ani jednego `unsafe`.
//!
//! Kolejność `x → y → z` w każdym wątku jest dobrana pod pamięć podręczną: przebieg po
//! `z` czyta jeden element z każdego z `ng` wierszy, ale zaraz potem sięga po element
//! obok. Cała linia pamięci podręcznej jest więc wykorzystana, mimo że pojedynczy
//! odczyt wygląda na przypadkowy.

use rayon::prelude::*;
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::sync::Arc;

const ZERO: Complex<f32> = Complex { re: 0.0, im: 0.0 };

/// Kierunek transformaty.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Forward,
    Inverse,
}

/// Plan trzech przebiegów.
///
/// Bufory linii i pamięci roboczej `rustfft` NIE są tu trzymane: każdy wątek potrzebuje
/// własnych, więc powstają raz na zadanie rayona (`for_each_init`), a nie raz na linię.
/// Planner zostaje, bo zbudowanie planu dla danej długości jest kosztowne i warto je
/// zamortyzować na wszystkie wywołania.
pub struct Fft3 {
    planner: FftPlanner<f32>,
}

impl Default for Fft3 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fft3 {
    pub fn new() -> Self {
        Self {
            planner: FftPlanner::new(),
        }
    }

    /// Transformata siatki `ng³` w miejscu.
    ///
    /// # Panics
    /// Gdy `data.len() != ng³`.
    pub fn run(&mut self, data: &mut [Complex<f32>], ng: usize, dir: Direction) {
        assert_eq!(
            data.len(),
            ng * ng * ng,
            "siatka {ng}³ nie zgadza się z długością danych {}",
            data.len()
        );
        if ng == 0 {
            return;
        }
        let fft = match dir {
            Direction::Forward => self.planner.plan_fft_forward(ng),
            Direction::Inverse => self.planner.plan_fft_inverse(ng),
        };

        pass_x(&fft, data, ng);
        pass_y(&fft, data, ng);
        pass_z(&fft, data, ng);
    }
}

/// Bufory jednego wątku: linia i pamięć robocza `rustfft`.
fn workspace(fft: &Arc<dyn Fft<f32>>, ng: usize) -> (Vec<Complex<f32>>, Vec<Complex<f32>>) {
    (
        vec![ZERO; ng],
        vec![ZERO; fft.get_inplace_scratch_len()],
    )
}

/// Oś x: linia jest ciągłym kawałkiem tablicy, więc transformujemy w miejscu.
fn pass_x(fft: &Arc<dyn Fft<f32>>, data: &mut [Complex<f32>], ng: usize) {
    data.par_chunks_mut(ng).for_each_init(
        || vec![ZERO; fft.get_inplace_scratch_len()],
        |scratch, line| fft.process_with_scratch(line, scratch),
    );
}

/// Oś y: cały płat `z = const` to ciągły blok `ng²`, a linie leżą wewnątrz niego.
fn pass_y(fft: &Arc<dyn Fft<f32>>, data: &mut [Complex<f32>], ng: usize) {
    data.par_chunks_mut(ng * ng).for_each_init(
        || workspace(fft, ng),
        |(line, scratch), slab| {
            for x in 0..ng {
                for (y, slot) in line.iter_mut().enumerate() {
                    *slot = slab[y * ng + x];
                }
                fft.process_with_scratch(line, scratch);
                for (y, value) in line.iter().enumerate() {
                    slab[y * ng + x] = *value;
                }
            }
        },
    );
}

/// Oś z: linia przecina wszystkie płaty, więc grupujemy wiersze `(z, y)` po `y`.
///
/// `pencils[y]` zbiera wiersz o tym `y` z każdego płata — czyli dokładnie te dane,
/// których potrzebuje przebieg po `z` dla wszystkich `x` naraz. Przekładane są same
/// referencje, dane zostają na miejscu; rozłączność grup wynika z tego, że każdy
/// wiersz trafia do dokładnie jednej z nich, i sprawdza ją kompilator.
fn pass_z(fft: &Arc<dyn Fft<f32>>, data: &mut [Complex<f32>], ng: usize) {
    let mut pencils: Vec<Vec<&mut [Complex<f32>]>> =
        (0..ng).map(|_| Vec::with_capacity(ng)).collect();
    for slab in data.chunks_mut(ng * ng) {
        for (y, row) in slab.chunks_mut(ng).enumerate() {
            pencils[y].push(row);
        }
    }
    pencils.par_iter_mut().for_each_init(
        || workspace(fft, ng),
        |(line, scratch), rows| {
            for x in 0..ng {
                for (z, row) in rows.iter().enumerate() {
                    line[z] = row[x];
                }
                fft.process_with_scratch(line, scratch);
                for (z, row) in rows.iter_mut().enumerate() {
                    row[x] = line[z];
                }
            }
        },
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn delta_grid(ng: usize) -> Vec<Complex<f32>> {
        let mut g = vec![Complex::new(0.0, 0.0); ng * ng * ng];
        g[0] = Complex::new(1.0, 0.0);
        g
    }

    #[test]
    fn delta_transforms_to_flat_spectrum() {
        let ng = 8;
        let mut g = delta_grid(ng);
        Fft3::new().run(&mut g, ng, Direction::Forward);
        for v in &g {
            assert!((v.re - 1.0).abs() < 1e-5 && v.im.abs() < 1e-5, "v={v}");
        }
    }

    #[test]
    fn forward_then_inverse_is_identity() {
        let ng = 8;
        let n = ng * ng * ng;
        let original: Vec<Complex<f32>> = (0..n)
            .map(|i| Complex::new((i % 13) as f32 - 6.0, (i % 7) as f32 - 3.0))
            .collect();
        let mut g = original.clone();
        let mut fft = Fft3::new();
        fft.run(&mut g, ng, Direction::Forward);
        fft.run(&mut g, ng, Direction::Inverse);
        let norm = 1.0 / n as f32;
        for (got, want) in g.iter().zip(original.iter()) {
            assert!((got.re * norm - want.re).abs() < 1e-3, "{got} vs {want}");
            assert!((got.im * norm - want.im).abs() < 1e-3, "{got} vs {want}");
        }
    }

    /// Transformata musi mieszać WSZYSTKIE trzy osie. Gdyby któryś przebieg
    /// operował na złym kroku, sygnał zmienny wyłącznie po z dałby widmo
    /// niezależne od k_z — i taki błąd nie ujawniłby się w teście na delcie.
    #[test]
    fn each_axis_is_transformed() {
        let ng = 8;
        let mut fft = Fft3::new();
        for axis in 0..3 {
            let mut g = vec![Complex::new(0.0, 0.0); ng * ng * ng];
            for z in 0..ng {
                for y in 0..ng {
                    for x in 0..ng {
                        let coord = [x, y, z][axis] as f32;
                        let phase = 2.0 * std::f32::consts::PI * coord / ng as f32;
                        g[(z * ng + y) * ng + x] = Complex::new(phase.cos(), 0.0);
                    }
                }
            }
            fft.run(&mut g, ng, Direction::Forward);
            // Pojedyncza fala kosinusoidalna o k=1 wzdłuż jednej osi daje moc
            // wyłącznie w dwóch prążkach: k=+1 i k=−1 na TEJ osi.
            let expected = 0.5 * (ng * ng * ng) as f32;
            let mut stride = [1, ng, ng * ng][axis];
            if axis == 2 {
                stride = ng * ng;
            }
            assert!(
                (g[stride].norm() - expected).abs() / expected < 1e-3,
                "os {axis}: prążek k=1 ma moc {}, oczekiwano {expected}",
                g[stride].norm()
            );
            assert!(g[0].norm() < 1e-2, "os {axis}: składowa stała niezerowa");
        }
    }
}
