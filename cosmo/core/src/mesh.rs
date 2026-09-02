//! Particle-Mesh z izolowanymi brzegami — grawitacja wszystkich par w O(N + M log M).
//!
//! Koszt zależy od siatki, nie od liczby par, więc milion cząstek liczy się tyle
//! samo co sto tysięcy. Wszystkie pary wchodzą do wyniku; ograniczeniem jest
//! rozdzielczość przestrzenna (oczko siatki), a nie to, ilu sąsiadów zdążyliśmy
//! policzyć.
//!
//! # Izolowane brzegi
//!
//! Zwykły PM na FFT jest periodyczny: cząstka wylatująca jedną ścianą wraca drugą,
//! a grawitacja przyciąga nieskończoną siatkę kopii pudła. To byłyby ściany tylnymi
//! drzwiami, więc stosujemy metodę Hockneya: siatkę mas rozszerzamy zerami do
//! podwójnego boku, a jądro próbkujemy we współrzędnych zawiniętych (indeks > P/2
//! oznacza ujemną odległość). Splot cykliczny na takiej siatce jest równy splotowi
//! liniowemu na obszarze oryginalnym, czyli układ jest naprawdę otwarty — bez kopii
//! i bez odbić.
//!
//! Potencjał liczymy splotem z jądrem Plummera, a przyspieszenie różnicą skończoną
//! czwartego rzędu na siatce. To dwie transformaty na krok zamiast czterech; przy
//! podwojonym boku każda transformata jest droga, więc ta różnica decyduje o tym,
//! czy symulacja jest interaktywna.
//!
//! Transformata jest zespolona po obu stronach, choć gęstość jest rzeczywista. Wariant
//! rzeczywisty (r2c) zmniejszyłby pamięć i czas o połowę, ale wymaga osobnego planu
//! i innego indeksowania połowy widma — a mierzony koszt kroku spadł już siedmiokrotnie
//! po zrównolegleniu samej transformaty (patrz [`crate::fft`]), więc to nie jest
//! wąskie gardło, dopóki siatka nie przekroczy 192.
//!
//! # Dlaczego widmo jądra jest rzeczywiste
//!
//! Jądro `−G/√(r²+ε²)` we współrzędnych zawiniętych jest PARZYSTE po każdej osi:
//! indeks `P−i` odpowiada odległości `−i`, a jądro zależy tylko od `|r|`. Transformata
//! funkcji parzystej i rzeczywistej jest rzeczywista, więc widmo jądra trzymamy jako
//! `f32`, a nie `Complex<f32>`. Przy siatce 192³ rozszerzonej do 384³ to różnica
//! 226 MB — czyli nie mikrooptymalizacja, a warunek zmieszczenia się w pamięci.
//! Dekonwolucja CIC też jest parzysta i rzeczywista, więc nie psuje tej własności.
//!
//! # Precyzja
//!
//! Siatka i transformaty liczą się w `f32`, cząstki w `f64`. Nie jest to
//! niedopatrzenie: `f64` na siatce 384³ podwoiłby zapotrzebowanie na pamięć do
//! ~1,4 GB, a solver siatkowy i tak jest przybliżony — jego błąd jest zdominowany
//! przez rozmiar oczka i rozmycie CIC, nie przez precyzję arytmetyki. Dokładnych
//! liczb (energia, wzorzec błędu) dostarcza solver `exact`, który pracuje w `f64`.

use rayon::prelude::*;
use rustfft::num_complex::Complex;

use crate::fft::{Direction, Fft3};
use crate::grid::{Box, Stencil, CORNERS};
use crate::vec3::{vec3, Vec3, ZERO};

/// Ile pustych komórek zostawić między chmurą a ścianą pudła.
///
/// Szablon różnicowy czwartego rzędu sięga dwóch komórek, a przy ścianie odczyt
/// musiałby być dosunięty — trzecia komórka jest zapasem na wagi CIC cząstki
/// stojącej dokładnie na granicy.
const EDGE_CELLS: usize = 3;

/// Górny limit korekty dekonwolucji CIC.
///
/// Korekta rośnie nieograniczenie przy częstotliwości Nyquista, gdzie i tak nie ma
/// wiarygodnej informacji — bez przycięcia wzmacniałaby wyłącznie szum.
const DECONV_CLAMP: f32 = 4.0;

/// Najmniejsza sensowna siatka. Poniżej tego marginesy zjadają cały obszar.
const MIN_GRID: usize = 16;

pub struct Mesh {
    grid: usize,
    margin: f64,
    box_: Option<Box>,
    /// Rzeczywiste widmo jądra — patrz uwaga w nagłówku modułu.
    kernel_ft: Vec<f32>,
    kernel_key: Option<KernelKey>,
    scratch: Vec<Complex<f32>>,
    density: Vec<f32>,
    potential_grid: Vec<f32>,
    accel_grid: Vec<[f32; 3]>,
    fft: Fft3,
    requested_softening: f64,
    last_softening: Option<f64>,
    pub refits: u32,
}

/// Klucz cache'u jądra. Jądro zależy wyłącznie od tych czterech liczb, więc jego
/// przebudowa przy niezmienionych parametrach byłaby czystą stratą.
#[derive(Clone, Copy, PartialEq, Debug)]
struct KernelKey {
    padded: usize,
    h: f64,
    g: f64,
    softening: f64,
}

impl Mesh {
    pub fn new(grid: usize, margin: f64) -> Self {
        let grid = grid.max(MIN_GRID);
        let cells = grid * grid * grid;
        Self {
            grid,
            margin,
            box_: None,
            kernel_ft: Vec::new(),
            kernel_key: None,
            scratch: Vec::new(),
            density: vec![0.0; cells],
            potential_grid: vec![0.0; cells],
            accel_grid: vec![[0.0; 3]; cells],
            fft: Fft3::new(),
            requested_softening: 0.0,
            last_softening: None,
            refits: 0,
        }
    }

    pub fn grid(&self) -> usize {
        self.grid
    }

    pub fn box_(&self) -> Option<Box> {
        self.box_
    }

    /// Oczko siatki, albo `None` przed pierwszym rozwiązaniem.
    pub fn cell_size(&self) -> Option<f64> {
        self.box_.map(|b| b.h)
    }

    fn padded(&self) -> usize {
        2 * self.grid
    }

    /// Dopasuj pudło, jeśli chmura z niego wyszła albo zrobiła się o wiele mniejsza.
    ///
    /// Warunek „o wiele mniejsza" (oczko grubsze niż 1,8× potrzebne) istnieje, bo bez
    /// niego zapadająca się chmura zostawałaby na siatce dopasowanej do swojego
    /// rozmiaru początkowego i traciłaby rozdzielczość dokładnie wtedy, kiedy jest
    /// najbardziej potrzebna. Współczynnik 1,8 zamiast 1,0 zapobiega przebudowie
    /// jądra w każdym kroku przy powolnym zapadaniu.
    fn ensure_box(&mut self, positions: &[Vec3]) -> Box {
        let needed = Box::fit(positions, self.grid, self.margin, EDGE_CELLS);
        let stale = match self.box_ {
            None => true,
            Some(current) => current.h > 1.8 * needed.h || !current.contains(positions),
        };
        if stale {
            self.box_ = Some(needed);
            self.refits += 1;
        }
        self.box_.expect("pudło jest ustawione")
    }

    fn ensure_kernel(&mut self, box_: Box, g: f64, softening: f64) {
        let key = KernelKey {
            padded: self.padded(),
            h: box_.h,
            g,
            softening,
        };
        if self.kernel_key == Some(key) {
            return;
        }
        let p = self.padded();
        let pn = p * p * p;
        self.scratch.resize(pn, Complex::new(0.0, 0.0));
        build_kernel(&mut self.scratch, p, box_.h, g, softening);
        self.fft.run(&mut self.scratch, p, Direction::Forward);

        self.kernel_ft.resize(pn, 0.0);
        let deconv = CicDeconvolution::new(p);
        for iz in 0..p {
            for iy in 0..p {
                for ix in 0..p {
                    let idx = (iz * p + iy) * p + ix;
                    self.kernel_ft[idx] = self.scratch[idx].re * deconv.at(ix, iy, iz);
                }
            }
        }
        self.kernel_key = Some(key);
    }

    /// Rozłóż masę na siatkę. Cząstka poza obszarem użytecznym jest pomijana —
    /// dosunięcie jej do ściany byłoby wprowadzeniem brzegu, którego ta metoda
    /// z definicji nie ma.
    fn deposit(&mut self, positions: &[Vec3], masses: &[f64], box_: Box) {
        self.density.fill(0.0);
        for (p, m) in positions.iter().zip(masses.iter()) {
            if let Some(s) = box_.stencil(*p) {
                s.scatter_f32(&mut self.density, *m);
            }
        }
    }

    /// `Φ = (−G/r) ∗ ρ` przez splot liniowy (padding Hockneya), potem `a = −∇Φ`.
    fn solve(&mut self, box_: Box) {
        let ng = self.grid;
        let p = self.padded();
        self.scratch.resize(p * p * p, Complex::new(0.0, 0.0));
        self.scratch.fill(Complex::new(0.0, 0.0));
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let src = (ix * ng + iy) * ng + iz;
                    let dst = (iz * p + iy) * p + ix;
                    self.scratch[dst] = Complex::new(self.density[src], 0.0);
                }
            }
        }
        self.fft.run(&mut self.scratch, p, Direction::Forward);
        self.scratch
            .par_iter_mut()
            .zip(self.kernel_ft.par_iter())
            .for_each(|(value, k)| *value *= *k);
        self.fft.run(&mut self.scratch, p, Direction::Inverse);

        let norm = 1.0 / (p * p * p) as f32;
        for iz in 0..ng {
            for iy in 0..ng {
                for ix in 0..ng {
                    let src = (iz * p + iy) * p + ix;
                    let dst = (ix * ng + iy) * ng + iz;
                    self.potential_grid[dst] = self.scratch[src].re * norm;
                }
            }
        }
        self.gradient_fourth_order(box_.h);
    }

    /// `a = −∇Φ` różnicą centralną czwartego rzędu.
    ///
    /// Odczyt poza siatką jest dosuwany do skrajnej komórki, a nie zawijany.
    /// Margines `EDGE_CELLS` gwarantuje, że żadna cząstka nie czyta tych komórek,
    /// więc różnica jest bez znaczenia dla wyniku — ale zawijanie wprowadzałoby
    /// periodyczność właśnie tam, gdzie ta metoda ma jej nie mieć.
    fn gradient_fourth_order(&mut self, h: f64) {
        let ng = self.grid;
        let scale = (1.0 / (12.0 * h)) as f32;
        let phi = &self.potential_grid;
        let at = |ix: i64, iy: i64, iz: i64| -> f32 {
            let top = ng as i64 - 1;
            let ix = ix.clamp(0, top) as usize;
            let iy = iy.clamp(0, top) as usize;
            let iz = iz.clamp(0, top) as usize;
            phi[(ix * ng + iy) * ng + iz]
        };
        self.accel_grid
            .par_iter_mut()
            .enumerate()
            .for_each(|(idx, out)| {
                let iz = (idx % ng) as i64;
                let iy = ((idx / ng) % ng) as i64;
                let ix = (idx / (ng * ng)) as i64;
                let mut a = [0.0f32; 3];
                for (axis, slot) in a.iter_mut().enumerate() {
                    let step = |n: i64| -> (i64, i64, i64) {
                        match axis {
                            0 => (ix + n, iy, iz),
                            1 => (ix, iy + n, iz),
                            _ => (ix, iy, iz + n),
                        }
                    };
                    let (a2, b2, c2) = step(-2);
                    let (a1, b1, c1) = step(-1);
                    let (a3, b3, c3) = step(1);
                    let (a4, b4, c4) = step(2);
                    let d = at(a2, b2, c2) - 8.0 * at(a1, b1, c1) + 8.0 * at(a3, b3, c3)
                        - at(a4, b4, c4);
                    *slot = -d * scale;
                }
                *out = a;
            });
    }

    /// Uaktualnij siatkę dla podanego stanu; zwraca użyty softening.
    pub fn refresh(&mut self, positions: &[Vec3], masses: &[f64], g: f64, softening: f64) -> f64 {
        let box_ = self.ensure_box(positions);
        // Siatka nie rozdzieli skali mniejszej niż oczko — liczymy tym, co jest
        // osiągalne, i mówimy o tym wprost przez `effective_softening`.
        self.requested_softening = softening;
        let eps = softening.max(box_.h);
        self.last_softening = Some(eps);
        self.ensure_kernel(box_, g, eps);
        self.deposit(positions, masses, box_);
        self.solve(box_);
        eps
    }

    /// Przyspieszenie odczytane z siatki tymi samymi wagami, którymi rozłożono masę.
    pub fn gather_acceleration(&self, positions: &[Vec3]) -> Vec<Vec3> {
        let box_ = match self.box_ {
            Some(b) => b,
            None => return vec![ZERO; positions.len()],
        };
        positions
            .par_iter()
            .map(|p| match box_.stencil(*p) {
                None => ZERO,
                Some(s) => s.gather_vec3_f32(&self.accel_grid),
            })
            .collect()
    }

    /// Potencjał właściwy w położeniach cząstek, z odjętym członem własnym.
    ///
    /// Cząstka rozsmarowana na osiem węzłów przyciąga samą siebie. Bez odjęcia tego
    /// członu energia ma nie tylko przesunięcie, ale i SZUM: wagi zmieniają się, gdy
    /// cząstka wędruje wewnątrz komórki, więc człon własny falowałby w czasie
    /// i zanieczyszczał dryf energii — jedyną liczbę, którą mierzymy jakość
    /// całkowania.
    pub fn gather_potential(&self, positions: &[Vec3], masses: &[f64], g: f64) -> Vec<f64> {
        let (box_, eps) = match (self.box_, self.last_softening) {
            (Some(b), Some(e)) => (b, e),
            _ => return vec![0.0; positions.len()],
        };
        let self_kernel = self_kernel(box_.h, g, eps);
        positions
            .par_iter()
            .zip(masses.par_iter())
            .map(|(p, m)| match box_.stencil(*p) {
                None => 0.0,
                Some(s) => s.gather_f32(&self.potential_grid) - m * self_energy(&s, &self_kernel),
            })
            .collect()
    }

    /// Kontrast gęstości `δ = ρ/ρ̄ − 1` w podanym punkcie; `-1` poza siatką.
    pub fn density_contrast_at(&self, position: Vec3, mean_density: f64) -> f64 {
        let Some(box_) = self.box_ else {
            return -1.0;
        };
        let Some(s) = box_.stencil(position) else {
            return -1.0;
        };
        let rho = s.gather_f32(&self.density);
        // `density` trzyma MASĘ w komórce, a nie gęstość — dzielimy przez objętość
        // dopiero tutaj, żeby depozyt pozostał dokładnie zachowujący masę.
        rho / (box_.cell_volume() * mean_density.max(1e-30)) - 1.0
    }

    /// Softening, którym siatka NAPRAWDĘ liczy — nigdy mniejszy od oczka.
    pub fn effective_softening(&self, requested: f64) -> f64 {
        match self.box_ {
            Some(b) => requested.max(b.h),
            None => requested,
        }
    }

    pub fn describe(&self) -> String {
        let head = format!("mesh {}³→{}³", self.grid, 2 * self.grid);
        match (self.box_, self.last_softening) {
            (Some(b), Some(eps)) => {
                let mut out = format!("{head}, oczko {:.3}", b.h);
                if eps > self.requested_softening + 1e-12 {
                    out.push_str(&format!(", ε podniesione do {eps:.3}"));
                }
                out
            }
            _ => head,
        }
    }
}

/// Jądro Plummera we współrzędnych zawiniętych.
fn build_kernel(out: &mut [Complex<f32>], padded: usize, h: f64, g: f64, softening: f64) {
    let eps2 = softening * softening;
    let half = (padded / 2) as i64;
    for iz in 0..padded {
        let z = wrapped_offset(iz, padded, half) * h;
        for iy in 0..padded {
            let y = wrapped_offset(iy, padded, half) * h;
            for ix in 0..padded {
                let x = wrapped_offset(ix, padded, half) * h;
                let r2 = x * x + y * y + z * z;
                let value = -g / (r2 + eps2).sqrt();
                out[(iz * padded + iy) * padded + ix] = Complex::new(value as f32, 0.0);
            }
        }
    }
}

/// Odległość ze znakiem odpowiadająca indeksowi siatki rozszerzonej.
///
/// Indeks powyżej połowy boku odpowiada odległości ujemnej — to właśnie ta
/// konwencja sprawia, że splot cykliczny na siatce rozszerzonej jest splotem
/// liniowym na siatce oryginalnej.
fn wrapped_offset(i: usize, n: usize, half: i64) -> f64 {
    let v = i as i64;
    if v > half {
        (v - n as i64) as f64
    } else {
        v as f64
    }
}

/// Odwrotność kwadratu okna CIC w przestrzeni Fouriera.
///
/// Rozłożenie masy na osiem węzłów, a potem odczyt tymi samymi wagami, mnoży pole
/// przez kwadrat okna `W(k) = Π sinc²(kᵢh/2)`. To rozmycie jest głównym źródłem
/// błędu PM — większym niż sama rozdzielczość siatki, co widać po tym, że błąd nie
/// maleje przy zwiększaniu ε. Podzielenie widma przez `W²` kasuje je i nic nie
/// kosztuje, bo mnoży się przez zapamiętane jądro.
///
/// Okno jest rozdzielne, więc wystarczy tablica jednowymiarowa — wersja trzymająca
/// pełną siatkę 3D korekt zużywałaby tyle pamięci co samo jądro.
struct CicDeconvolution {
    per_axis: Vec<f32>,
}

impl CicDeconvolution {
    fn new(padded: usize) -> Self {
        let half = (padded / 2) as i64;
        let per_axis = (0..padded)
            .map(|i| {
                let f = wrapped_offset(i, padded, half) / padded as f64;
                sinc_pi(f).powi(2) as f32
            })
            .collect();
        Self { per_axis }
    }

    fn at(&self, ix: usize, iy: usize, iz: usize) -> f32 {
        let window = self.per_axis[ix] * self.per_axis[iy] * self.per_axis[iz];
        (1.0 / (window * window).max(1e-12)).min(DECONV_CLAMP)
    }
}

/// `sin(πx)/(πx)` — konwencja `numpy.sinc`.
fn sinc_pi(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        let px = std::f64::consts::PI * x;
        px.sin() / px
    }
}

/// Macierz 8×8 oddziaływań między węzłami jednej komórki CIC.
fn self_kernel(h: f64, g: f64, softening: f64) -> [[f64; CORNERS]; CORNERS] {
    let mut corners = [ZERO; CORNERS];
    let mut k = 0;
    for dx in 0..2 {
        for dy in 0..2 {
            for dz in 0..2 {
                corners[k] = vec3(dx as f64 * h, dy as f64 * h, dz as f64 * h);
                k += 1;
            }
        }
    }
    let eps2 = softening * softening;
    let mut out = [[0.0; CORNERS]; CORNERS];
    for a in 0..CORNERS {
        for b in 0..CORNERS {
            let r2 = (corners[a] - corners[b]).norm_squared();
            out[a][b] = -g / (r2 + eps2).sqrt();
        }
    }
    out
}

/// `wᵀ K w` — potencjał, którym rozsmarowana cząstka działa na samą siebie.
fn self_energy(s: &Stencil, kernel: &[[f64; CORNERS]; CORNERS]) -> f64 {
    let mut acc = 0.0;
    for (row, w_a) in kernel.iter().zip(s.weight) {
        let paired: f64 = row.iter().zip(s.weight).map(|(k, w_b)| k * w_b).sum();
        acc += w_a * paired;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 43.0;

    /// Siły w jednym przebiegu — to, czego potrzebują testy tego modułu.
    fn forces(mesh: &mut Mesh, x: &[Vec3], m: &[f64], softening: f64) -> Vec<Vec3> {
        mesh.refresh(x, m, G, softening);
        mesh.gather_acceleration(x)
            .iter()
            .zip(m.iter())
            .map(|(a, mass)| *a * *mass)
            .collect()
    }

    fn cube_cloud(per_side: usize, size: f64) -> (Vec<Vec3>, Vec<f64>) {
        crate::fixtures::cube_cloud(per_side, size)
    }

    /// Podstawowa własność: brzegi są izolowane, więc dwie cząstki przyciągają się
    /// wzdłuż odcinka między nimi, a nie „na skróty" przez ścianę pudła.
    #[test]
    fn isolated_boundaries_attract_inward() {
        let x = vec![vec3(-9.0, 0.0, 0.0), vec3(9.0, 0.0, 0.0)];
        let m = vec![1.0, 1.0];
        let f = forces(&mut Mesh::new(32, 0.15), &x, &m, 0.5);
        assert!(f[0].x > 0.0, "lewa: {}", f[0].x);
        assert!(f[1].x < 0.0, "prawa: {}", f[1].x);
    }

    /// Siły wewnętrzne muszą się znosić — to test symetrii deposit ↔ gather.
    #[test]
    fn momentum_residual_is_small() {
        let (x, m) = cube_cloud(12, 20.0);
        let f = forces(&mut Mesh::new(32, 0.15), &x, &m, 0.5);
        let total: Vec3 = f.iter().copied().sum();
        let scale: f64 = f.iter().map(|v| v.norm()).sum();
        let residual = total.norm() / scale.max(1e-30);
        assert!(residual < 3e-3, "residuum pędu {residual}");
    }

    #[test]
    fn refits_when_cloud_collapses() {
        let mut mesh = Mesh::new(32, 0.15);
        let (x, m) = cube_cloud(6, 40.0);
        mesh.refresh(&x, &m, G, 0.5);
        let before = mesh.refits;
        let shrunk: Vec<Vec3> = x.iter().map(|p| *p * 0.01).collect();
        mesh.refresh(&shrunk, &m, G, 0.5);
        assert!(mesh.refits > before, "siatka nie została zagęszczona");
    }

    /// Siatka nie może być przebudowywana bez powodu: jądro to dwie transformaty
    /// na siatce o podwojonym boku, czyli najdroższa rzecz w całym solverze.
    #[test]
    fn steady_cloud_does_not_refit() {
        let mut mesh = Mesh::new(32, 0.15);
        let (x, m) = cube_cloud(6, 40.0);
        mesh.refresh(&x, &m, G, 0.5);
        let before = mesh.refits;
        for _ in 0..5 {
            mesh.refresh(&x, &m, G, 0.5);
        }
        assert_eq!(mesh.refits, before);
    }

    #[test]
    fn effective_softening_never_below_cell() {
        let (x, m) = cube_cloud(6, 30.0);
        let mut mesh = Mesh::new(16, 0.15);
        mesh.refresh(&x, &m, G, 1e-6);
        let h = mesh.cell_size().unwrap();
        assert!((mesh.effective_softening(1e-6) - h).abs() < 1e-12);
    }

    /// Masa musi być zachowana przez depozyt: siatka trzyma dokładnie tyle masy,
    /// ile jest w chmurze. Gdyby cząstki wypadały poza obszar, kontrast gęstości
    /// i potencjał byłyby zaniżone bez żadnego widocznego objawu.
    #[test]
    fn deposit_conserves_mass() {
        let (x, m) = cube_cloud(8, 20.0);
        let mut mesh = Mesh::new(32, 0.15);
        mesh.refresh(&x, &m, G, 0.5);
        let on_grid: f64 = mesh.density.iter().map(|v| *v as f64).sum();
        let total: f64 = m.iter().sum();
        assert!(
            (on_grid - total).abs() / total < 1e-5,
            "na siatce {on_grid}, w chmurze {total}"
        );
    }

    /// Jednorodna chmura ma mieć `δ ≈ 0` w środku, a `δ = −1` poza siatką.
    ///
    /// Siatka musi być ZGRUBNA względem odstępu między cząstkami — inaczej pomiar
    /// mierzy nie gęstość, a szum próbkowania: przy oczku mniejszym od odstępu
    /// większość komórek jest pusta i `δ` w losowym punkcie wynosi −1 nawet w środku
    /// idealnie jednorodnej chmury.
    #[test]
    fn uniform_cloud_has_no_density_contrast_inside() {
        let side = 16;
        let size = 16.0;
        let (x, m) = cube_cloud(side, size);
        let mut mesh = Mesh::new(16, 0.15);
        mesh.refresh(&x, &m, G, 0.5);
        let spacing = size / side as f64;
        assert!(
            mesh.cell_size().unwrap() > 2.0 * spacing,
            "oczko drobniejsze od odstępu cząstek — pomiar byłby szumem"
        );

        let mean = m.iter().sum::<f64>() / size.powi(3);
        let center = vec3(0.5 * size, 0.5 * size, 0.5 * size);
        let inside = mesh.density_contrast_at(center, mean);
        assert!(inside.abs() < 0.3, "w środku δ={inside}");

        let outside = mesh.density_contrast_at(vec3(1e6, 0.0, 0.0), mean);
        assert!((outside + 1.0).abs() < 1e-12, "poza siatką δ={outside}");
    }

    /// Zagęszczenie musi dawać `δ` wyraźnie większe niż tło. Bez tego `density_contrast_at`
    /// mogłoby zwracać stałą i cieniowanie nadal „coś" pokazywałoby.
    #[test]
    fn a_clump_stands_out_from_the_background() {
        let (mut x, mut m) = cube_cloud(12, 24.0);
        let center = vec3(12.0, 12.0, 12.0);
        // Druga taka sama masa upchnięta w jedno miejsce.
        for i in 0..x.len() {
            x.push(center + (x[i] - center) * 0.05);
            m.push(1.0);
        }
        let mut mesh = Mesh::new(24, 0.15);
        mesh.refresh(&x, &m, G, 0.5);
        let mean = m.iter().sum::<f64>() / 24.0f64.powi(3);
        let clump = mesh.density_contrast_at(center, mean);
        let background = mesh.density_contrast_at(vec3(4.0, 4.0, 4.0), mean);
        assert!(clump > background, "zgęstek δ={clump}, tło δ={background}");
    }

    /// Przed pierwszym `refresh` nie ma siatki, więc odczyt musi zwracać zera,
    /// a nie sięgać do nieistniejącego pudła.
    #[test]
    fn gathering_before_the_first_solve_returns_zeros() {
        let mesh = Mesh::new(16, 0.15);
        let x = vec![ZERO, vec3(1.0, 0.0, 0.0)];
        assert_eq!(mesh.gather_acceleration(&x), vec![ZERO, ZERO]);
        assert_eq!(mesh.gather_potential(&x, &[1.0, 1.0], G), vec![0.0, 0.0]);
    }

    #[test]
    fn kernel_spectrum_is_real() {
        let p = 16;
        let mut buf = vec![Complex::new(0.0f32, 0.0); p * p * p];
        build_kernel(&mut buf, p, 0.5, G, 0.5);
        Fft3::new().run(&mut buf, p, Direction::Forward);
        let worst = buf
            .iter()
            .map(|v| v.im.abs() / v.re.abs().max(1e-6))
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "widmo jądra nie jest rzeczywiste: {worst}");
    }

    #[test]
    fn self_energy_weights_sum_correctly() {
        let kernel = self_kernel(1.0, G, 0.5);
        // Cząstka dokładnie w węźle ma wagę 1 na jednym narożniku, więc jej człon
        // własny to element diagonalny — czyli −G/ε.
        let s = Stencil {
            index: [0; CORNERS],
            weight: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let expected = -G / 0.5;
        assert!((self_energy(&s, &kernel) - expected).abs() < 1e-12);
    }
}
