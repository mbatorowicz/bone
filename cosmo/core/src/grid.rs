//! Pudło siatki i wagi cloud-in-cell.
//!
//! Wydzielone z solvera, bo nie są jego własnością: dokładnie tego samego
//! dopasowania pudła i tych samych wag potrzebuje chłodzenie, które musi znać
//! lokalny przepływ masowy. Dwie kopie CIC byłyby gwarancją, że któraś kiedyś
//! przestanie być symetryczna — a symetria deposit ↔ gather jest jedyną rzeczą,
//! która utrzymuje zachowanie pędu.
//!
//! Margines `edge` jest parametrem pudła, nie stałą modułu, bo zależy od tego, co
//! się z siatką robi. Solver liczy gradient szablonem czwartego rzędu sięgającym
//! dwóch komórek, więc potrzebuje trzech pustych komórek przy ścianie; chłodzenie
//! tylko uśrednia w komórce i nie potrzebuje żadnej.
//!
//! # Dwa warianty odczytu i dlaczego nie jeden
//!
//! [`Box::stencil`] zwraca `None` dla cząstki poza użytecznym obszarem, a
//! [`Box::stencil_clamped`] dosuwa ją do skrajnej komórki. Różnica nie jest
//! kosmetyczna i dlatego nie da się jej ukryć za jedną funkcją:
//!
//! * Solver PM MUSI odrzucać. Rozkładanie masy na skrajną komórkę byłoby
//!   wprowadzaniem ściany tylnymi drzwiami — dokładnie tego, czego unika metoda
//!   Hockneya. Po dopasowaniu pudła nie powinno się zdarzać, a jeśli się zdarza,
//!   lepiej stracić cząstkę niż zafałszować pole.
//! * Chłodzenie MUSI dosuwać. Pracuje z `edge = 0`, więc przy grubej siatce
//!   (kilka komórek) cząstka z brzegu chmury naprawdę wypada za ostatni węzeł.
//!   Odrzucenie znaczyłoby „ta cząstka nie ma lokalnego przepływu", czyli ciche
//!   wyłączenie chłodzenia na brzegu.

use crate::vec3::{vec3, Vec3, ZERO};

/// Liczba węzłów, na które CIC rozkłada jedną cząstkę.
pub const CORNERS: usize = 8;

/// Sześcienne pudło siatki dopasowane do chmury.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Box {
    /// Lewy dolny róg siatki.
    pub origin: Vec3,
    /// Rozmiar oczka (izotropowy).
    pub h: f64,
    /// Bok siatki w komórkach.
    pub ng: usize,
    /// Ile komórek przy ścianie musi zostać puste.
    pub edge: usize,
}

/// Osiem węzłów i osiem wag sumujących się do jedności.
#[derive(Clone, Copy, Debug)]
pub struct Stencil {
    pub index: [usize; CORNERS],
    pub weight: [f64; CORNERS],
}

impl Box {
    /// Dopasuj pudło do rozciągłości chmury.
    ///
    /// `margin` to zapas ponad rozciągłość, wyrażony jej ułamkiem. `edge` zwęża
    /// obszar użyteczny z obu stron, więc przy tej samej siatce daje drobniejsze
    /// oczko na mniejszym obszarze.
    ///
    /// Obszar użyteczny to `ng − 2·edge − 2` komórek, a nie `ng − 2·edge`. Dwie
    /// komórki odliczone dodatkowo nie są zapasem „na wszelki wypadek": wagi CIC
    /// jednej cząstki sięgają DWÓCH sąsiednich węzłów, więc cząstka stojąca na
    /// ostatnim węźle obszaru użytecznego rozkłada masę na węzeł już poza nim. Bez
    /// tego odliczenia [`Box::contains`] mogłoby uznać pudło za dobre, a
    /// [`Box::stencil`] i tak odrzuciłby skrajną cząstkę — czyli solver po cichu
    /// gubiłby masę zamiast poprawić pudło.
    pub fn fit(positions: &[Vec3], ng: usize, margin: f64, edge: usize) -> Self {
        let (center, span) = center_span(positions);
        let usable = ng.saturating_sub(2 * edge + 2).max(1) as f64;
        // Zapas musi być ściśle dodatni, żeby cząstka z brzegu chmury nie wypadła
        // dokładnie na granicę obszaru — tam rozstrzyga zaokrąglenie, nie geometria.
        let h = span * (1.0 + margin.max(1e-3)) / usable;
        Self {
            origin: center - Vec3::splat(0.5 * ng as f64 * h),
            h,
            ng,
            edge,
        }
    }

    pub fn cells(self) -> usize {
        self.ng * self.ng * self.ng
    }

    pub fn cell_volume(self) -> f64 {
        self.h * self.h * self.h
    }

    /// Czy wszystkie cząstki mieszczą się w obszarze użytecznym.
    ///
    /// Warunek jest postawiony na INDEKSIE dolnego węzła, a nie na współrzędnej —
    /// dokładnie na tej liczbie, którą sprawdza [`Box::stencil`]. Dzięki temu
    /// „pudło jest dobre" znaczy to samo co „każdą cząstkę da się odczytać", i nie ma
    /// zakresu, w którym te dwa zdania się rozchodzą.
    pub fn contains(self, positions: &[Vec3]) -> bool {
        let lo = self.edge as i64;
        let hi = (self.ng as i64 - 1 - self.edge as i64).max(lo);
        positions.iter().all(|p| match self.local(*p) {
            None => false,
            Some((base, _)) => base.iter().all(|b| *b >= lo && *b <= hi),
        })
    }

    /// Wagi CIC; `None`, gdy cząstka wypada poza obszar odczytu.
    pub fn stencil(self, position: Vec3) -> Option<Stencil> {
        let (base, frac) = self.local(position)?;
        let top = self.ng as i64 - 2;
        if base.iter().any(|b| *b < 0 || *b > top) {
            return None;
        }
        Some(self.assemble([base[0] as usize, base[1] as usize, base[2] as usize], frac))
    }

    /// Wagi CIC z dosunięciem do skrajnej komórki — patrz uwaga w nagłówku modułu.
    pub fn stencil_clamped(self, position: Vec3) -> Option<Stencil> {
        let (base, frac) = self.local(position)?;
        let top = (self.ng - 2) as i64;
        let clamped = [
            base[0].clamp(0, top) as usize,
            base[1].clamp(0, top) as usize,
            base[2].clamp(0, top) as usize,
        ];
        Some(self.assemble(clamped, frac))
    }

    /// Współrzędne w komórkach: indeks dolnego węzła i część ułamkowa.
    ///
    /// `origin` jest położeniem węzła zerowego, nie ściany komórki zerowej. Wersja ze
    /// ścianami wymagałaby wszędzie przesunięcia o pół oczka, a to przesunięcie było
    /// źródłem błędów o jeden przy granicach obszaru — bo pojawiało się w `local`,
    /// ale nie w `contains`.
    fn local(self, position: Vec3) -> Option<([i64; 3], Vec3)> {
        let local = (position - self.origin) / self.h;
        if !local.is_finite() {
            return None;
        }
        let base = [
            local.x.floor(),
            local.y.floor(),
            local.z.floor(),
        ];
        // Poza tym zakresem rzutowanie na i64 jest nieokreślone, a taka cząstka
        // i tak jest wynikiem rozbiegnięcia się symulacji, nie stanem fizycznym.
        if base.iter().any(|b| b.abs() > 1e15) {
            return None;
        }
        let frac = vec3(local.x - base[0], local.y - base[1], local.z - base[2]);
        Some(([base[0] as i64, base[1] as i64, base[2] as i64], frac))
    }

    fn assemble(self, base: [usize; 3], frac: Vec3) -> Stencil {
        let ng = self.ng;
        let mut index = [0usize; CORNERS];
        let mut weight = [0.0f64; CORNERS];
        let mut corner = 0;
        for dx in 0..2 {
            let wx = if dx == 0 { 1.0 - frac.x } else { frac.x };
            let ix = base[0] + dx;
            for dy in 0..2 {
                let wy = if dy == 0 { 1.0 - frac.y } else { frac.y };
                let iy = base[1] + dy;
                for dz in 0..2 {
                    let wz = if dz == 0 { 1.0 - frac.z } else { frac.z };
                    let iz = base[2] + dz;
                    index[corner] = (ix * ng + iy) * ng + iz;
                    weight[corner] = wx * wy * wz;
                    corner += 1;
                }
            }
        }
        Stencil { index, weight }
    }
}

impl Stencil {
    /// Rozłóż skalar na siatkę wagami CIC — ta sama pętla, której używa gather.
    pub fn scatter_f32(self, field: &mut [f32], value: f64) {
        for k in 0..CORNERS {
            field[self.index[k]] += (self.weight[k] * value) as f32;
        }
    }

    /// Zbierz skalar z siatki `f32` tymi samymi wagami, którymi go rozłożono.
    pub fn gather_f32(self, field: &[f32]) -> f64 {
        let mut acc = 0.0;
        for k in 0..CORNERS {
            acc += field[self.index[k]] as f64 * self.weight[k];
        }
        acc
    }

    /// Zbierz wektor `[f32; 3]` (przyspieszenie na siatce).
    pub fn gather_vec3_f32(self, field: &[[f32; 3]]) -> Vec3 {
        let mut acc = ZERO;
        for k in 0..CORNERS {
            let g = field[self.index[k]];
            acc += vec3(g[0] as f64, g[1] as f64, g[2] as f64) * self.weight[k];
        }
        acc
    }

    /// Rozłóż N skalarów na przeplataną siatkę (masa, pęd, |v|² w chłodzeniu).
    pub fn scatter_fields<const N: usize>(self, sums: &mut [f64], payload: &[f64; N]) {
        for k in 0..CORNERS {
            let base = self.index[k] * N;
            let w = self.weight[k];
            for (f, value) in payload.iter().enumerate() {
                sums[base + f] += w * value;
            }
        }
    }

    /// Zbierz N skalarów z przeplatanej siatki.
    pub fn gather_fields<const N: usize>(self, sums: &[f64]) -> [f64; N] {
        let mut gathered = [0.0f64; N];
        for k in 0..CORNERS {
            let base = self.index[k] * N;
            let w = self.weight[k];
            for f in 0..N {
                gathered[f] += w * sums[base + f];
            }
        }
        gathered
    }
}

/// Środek i największa rozciągłość chmury.
///
/// Rozciągłość zerowa albo nieskończona jest zastępowana jedynką: pudło o zerowym
/// oczku dawałoby dzielenie przez zero w każdym późniejszym przeliczeniu, a jedna
/// cząstka albo cząstki w jednym punkcie to sytuacja legalna.
pub fn center_span(positions: &[Vec3]) -> (Vec3, f64) {
    let mut lo = Vec3::splat(f64::INFINITY);
    let mut hi = Vec3::splat(f64::NEG_INFINITY);
    for p in positions {
        lo = lo.min_each(*p);
        hi = hi.max_each(*p);
    }
    if !lo.is_finite() || !hi.is_finite() {
        return (crate::vec3::ZERO, 1.0);
    }
    let center = (lo + hi) * 0.5;
    let span = (hi - lo).max_abs_component();
    if !span.is_finite() || span <= 0.0 {
        return (center, 1.0);
    }
    (center, span)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud() -> Vec<Vec3> {
        vec![
            vec3(-3.0, -2.0, -1.0),
            vec3(3.0, 2.0, 1.0),
            vec3(0.5, 0.0, -0.5),
        ]
    }

    #[test]
    fn weights_sum_to_one() {
        let b = Box::fit(&cloud(), 16, 0.15, 0);
        for p in cloud() {
            let s = b.stencil(p).expect("cząstka w pudle");
            let total: f64 = s.weight.iter().sum();
            assert!((total - 1.0).abs() < 1e-12, "suma wag {total}");
        }
    }

    #[test]
    fn fit_contains_the_cloud() {
        let c = cloud();
        for edge in [0usize, 3] {
            let b = Box::fit(&c, 32, 0.15, edge);
            assert!(b.contains(&c), "edge={edge}");
        }
    }

    /// `edge` musi zwężać obszar użyteczny, czyli zagęszczać oczko przy tej samej
    /// siatce. Gdyby był ignorowany, gradient czwartego rzędu czytałby komórki
    /// zawinięte przez FFT.
    #[test]
    fn edge_margin_shrinks_the_cell() {
        let c = cloud();
        let plain = Box::fit(&c, 32, 0.15, 0);
        let with_edge = Box::fit(&c, 32, 0.15, 3);
        assert!(with_edge.h > plain.h);
        let lo = (with_edge.origin.x - c[0].x) / with_edge.h;
        assert!(lo.abs() > 0.0);
    }

    /// `contains` jest używane do decyzji „pudło jest jeszcze dobre". Musi więc
    /// znaczyć dokładnie „każda cząstka da się odczytać", a nie coś słabszego —
    /// inaczej solver uznaje pudło za dobre i gubi masę cząstek przy ścianie.
    #[test]
    fn contains_implies_every_stencil_exists() {
        let c = cloud();
        for edge in [0usize, 1, 3] {
            for ng in [8usize, 16, 32] {
                let b = Box::fit(&c, ng, 0.15, edge);
                assert!(b.contains(&c), "fit nie mieści chmury: ng={ng} edge={edge}");
                for p in &c {
                    assert!(
                        b.stencil(*p).is_some(),
                        "contains przepuściło cząstkę bez odczytu: ng={ng} edge={edge}"
                    );
                }
            }
        }
    }

    #[test]
    fn stencil_rejects_outside_but_clamped_accepts() {
        let c = cloud();
        let b = Box::fit(&c, 16, 0.15, 0);
        let far = b.origin - Vec3::splat(100.0 * b.h);
        assert!(b.stencil(far).is_none());
        let s = b.stencil_clamped(far).expect("dosunięcie zawsze się udaje");
        let total: f64 = s.weight.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        assert!(s.index.iter().all(|i| *i < b.cells()));
    }

    #[test]
    fn indices_stay_inside_the_grid() {
        let c = cloud();
        let b = Box::fit(&c, 8, 0.15, 0);
        for p in &c {
            let s = b.stencil(*p).expect("w pudle");
            assert!(s.index.iter().all(|i| *i < b.cells()));
        }
    }

    #[test]
    fn non_finite_position_is_rejected() {
        let b = Box::fit(&cloud(), 16, 0.15, 0);
        assert!(b.stencil(vec3(f64::NAN, 0.0, 0.0)).is_none());
        assert!(b.stencil_clamped(vec3(f64::INFINITY, 0.0, 0.0)).is_none());
    }

    #[test]
    fn degenerate_cloud_gets_a_usable_box() {
        let same = vec![vec3(1.0, 1.0, 1.0); 4];
        let b = Box::fit(&same, 16, 0.15, 0);
        assert!(b.h > 0.0 && b.h.is_finite());
        assert!(b.contains(&same));
    }

    /// Interpolacja CIC musi być tożsamościowa dla pola stałego: rozłożenie
    /// jedności i odczytanie jej z powrotem daje jedność. To jest test symetrii
    /// deposit ↔ gather, czyli tego, na czym stoi zachowanie pędu.
    #[test]
    fn deposit_gather_round_trip_is_identity_for_constant_field() {
        let c = cloud();
        let b = Box::fit(&c, 16, 0.15, 0);
        let mut field = vec![0.0f64; b.cells()];
        for p in &c {
            let s = b.stencil(*p).unwrap();
            for k in 0..CORNERS {
                field[s.index[k]] += s.weight[k];
            }
        }
        let deposited: f64 = field.iter().sum();
        assert!((deposited - c.len() as f64).abs() < 1e-12, "{deposited}");
    }
}
