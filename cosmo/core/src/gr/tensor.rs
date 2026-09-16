//! Algebra 4D: wektor, kowektor, maszyna (1,1) i waga (0,2).
//!
//! To skrzynki z lekcji o tensorach, nie równania pola. Wektor to „dokąd”
//! (`V^μ`). Kowektor to rachunek (`ω_μ`). [`Tensor11`] to automat
//! `w^μ = A^μ_ν v^ν`. [`Tensor02`] to waga na dwa przesunięcia. [`Tensor20`]
//! to odwrotność wagi — drzwi potrzebne, żeby podnieść wskaźnik z powrotem.
//!
//! Metryka tu to tylko Minkowski. Sygnatura (−,+,+,+), jak w
//! [`super::metric`], nie jak interwał w [`super::lorentz`]:
//!
//! ```text
//! η_μν = diag(−1, +1, +1, +1)
//! η(v, v) = −s²_lorentz
//! ```
//!
//! To dwa słowniki, nie dwa światy. Opuszczenie na etcie odwraca znak czasu:
//! `v_t = −v^t`. `|β| < 1` tu nie mieszka — boost zostaje w `lorentz`.
//! Riemann i Einstein nie wchodzą.

use super::lorentz::Event;

/// Wektor `V^μ`: składowe „dokąd” w kolejności `(t, x, y, z)`.
///
/// Przy `c = 1` składowa `t` to ta sama liczba co `ct` w [`Event`]. Osobny
/// typ, bo zdarzenie pilnuje interwału STW, a tu składowe są kostiumem
/// na osiach, który metryka może opuścić.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vector {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vector {
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self { t, x, y, z }
    }

    pub fn from_components(c: [f64; 4]) -> Self {
        Self {
            t: c[0],
            x: c[1],
            y: c[2],
            z: c[3],
        }
    }

    pub fn components(self) -> [f64; 4] {
        [self.t, self.x, self.y, self.z]
    }

    /// `v^μ ω_μ`. Powtórzony wskaźnik znika: zostaje liczba.
    pub fn contract(self, covector: Covector) -> f64 {
        dot4(self.components(), covector.components())
    }

    /// `v_μ = η_μν v^ν`. Czas zmienia znak, miejsce zostaje.
    pub fn lower(self) -> Covector {
        Tensor02::minkowski().covector(self)
    }
}

impl From<Event> for Vector {
    fn from(event: Event) -> Self {
        Self::new(event.ct, event.x, event.y, event.z)
    }
}

/// Kowektor `ω_μ`: składowe „ile warte”, gdy przyłożysz linijkę.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Covector {
    pub t: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Covector {
    pub fn new(t: f64, x: f64, y: f64, z: f64) -> Self {
        Self { t, x, y, z }
    }

    pub fn from_components(c: [f64; 4]) -> Self {
        Self {
            t: c[0],
            x: c[1],
            y: c[2],
            z: c[3],
        }
    }

    pub fn components(self) -> [f64; 4] {
        [self.t, self.x, self.y, self.z]
    }

    /// `ω_μ v^μ`. Ten sam skurcz co [`Vector::contract`], od strony rachunku.
    pub fn contract(self, vector: Vector) -> f64 {
        vector.contract(self)
    }

    /// `v^μ = η^{μν} v_ν`. Opuść i podnieś — musisz wrócić do startu.
    pub fn raise(self) -> Vector {
        Tensor20::minkowski().vector(self)
    }
}

/// Maszyna `T^μ_ν`: jedne drzwi w górę, jedne w dół.
///
/// `components[μ][ν] = T^μ_ν`. Wiersz to wynik, kolumna to wsad — jak
/// macierz na wektorze kolumnowym. Skurcz po powtórzonym wskaźniku to ślad.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tensor11 {
    components: [[f64; 4]; 4],
}

impl Tensor11 {
    pub fn from_components(components: [[f64; 4]; 4]) -> Self {
        Self { components }
    }

    pub fn components(self) -> [[f64; 4]; 4] {
        self.components
    }

    /// `δ^μ_ν`. Włożyć oś, wyjąć tę samą oś.
    pub fn identity() -> Self {
        Self::from_components(diag4([1.0, 1.0, 1.0, 1.0]))
    }

    /// `w^μ = T^μ_ν v^ν`.
    pub fn apply(self, vector: Vector) -> Vector {
        let v = vector.components();
        let mut w = [0.0; 4];
        for (mu, slot) in w.iter_mut().enumerate() {
            *slot = dot4(self.components[mu], v);
        }
        Vector::from_components(w)
    }

    /// `(AB)^μ_ν = A^μ_λ B^λ_ν`. Najpierw `other`, potem `self`.
    pub fn compose(self, other: Self) -> Self {
        Self::from_components(mul4(self.components, other.components))
    }

    /// `T^μ_μ`. Cztery kratki przekątnej, jedna liczba.
    pub fn trace(self) -> f64 {
        let t = self.components;
        t[0][0] + t[1][1] + t[2][2] + t[3][3]
    }

    /// `T_μν = η_μλ T^λ_ν`. Maszyna staje się wagą na etcie.
    pub fn lower(self) -> Tensor02 {
        Tensor02::from_components(mul4(Tensor02::minkowski().components(), self.components))
    }
}

/// Waga `T_μν`: dwa sloty na strzałki, wynik liczba.
///
/// `components[μ][ν] = T_μν`. Minkowski jest tu przekątną stałą. Schwarzschild
/// zostaje w [`super::metric`] — gotowa linijka jednego wzoru, nie ogólna
/// skrzynka.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tensor02 {
    components: [[f64; 4]; 4],
}

impl Tensor02 {
    pub fn from_components(components: [[f64; 4]; 4]) -> Self {
        Self { components }
    }

    pub fn components(self) -> [[f64; 4]; 4] {
        self.components
    }

    /// `η_μν = diag(−1, +1, +1, +1)`.
    pub fn minkowski() -> Self {
        Self::from_components(diag4([-1.0, 1.0, 1.0, 1.0]))
    }

    /// `T(u, v) = T_μν u^μ v^ν`.
    pub fn on(self, u: Vector, v: Vector) -> f64 {
        self.covector(u).contract(v)
    }

    /// `ω_μ = T_μν v^ν`. Jeden slot zjedzony, zostaje rachunek.
    pub fn covector(self, vector: Vector) -> Covector {
        let v = vector.components();
        let mut w = [0.0; 4];
        for (mu, slot) in w.iter_mut().enumerate() {
            *slot = dot4(self.components[mu], v);
        }
        Covector::from_components(w)
    }

    /// `T^μ_ν = η^{μλ} T_λν`. Waga dostaje jedne drzwi w górę.
    pub fn raise(self) -> Tensor11 {
        Tensor20::minkowski().contract_weight(self)
    }
}

/// Odwrotność wagi `T^{μν}`. Dla ety te same liczby co `η_μν`, inne drzwi.
///
/// Bez tego typu podniesienie indeksu musiałoby udawać, że dół i góra to
/// ta sama tabliczka. Lekcja o przekładni właśnie tego zabrania.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tensor20 {
    components: [[f64; 4]; 4],
}

impl Tensor20 {
    pub fn from_components(components: [[f64; 4]; 4]) -> Self {
        Self { components }
    }

    pub fn components(self) -> [[f64; 4]; 4] {
        self.components
    }

    /// `η^{μν}`. W (−,+,+,+) odwrotność ety jest etą.
    pub fn minkowski() -> Self {
        Self::from_components(Tensor02::minkowski().components())
    }

    /// `v^μ = T^{μν} ω_ν`.
    pub fn vector(self, covector: Covector) -> Vector {
        let w = covector.components();
        let mut v = [0.0; 4];
        for (mu, slot) in v.iter_mut().enumerate() {
            *slot = dot4(self.components[mu], w);
        }
        Vector::from_components(v)
    }

    /// `M^μ_ν = T^{μλ} g_λν`. Dla `η η⁻¹` wynik musi być jedynką, nie „prawie”.
    pub fn contract_weight(self, weight: Tensor02) -> Tensor11 {
        Tensor11::from_components(mul4(self.components, weight.components()))
    }
}

fn dot4(a: [f64; 4], b: [f64; 4]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn diag4(diag: [f64; 4]) -> [[f64; 4]; 4] {
    let mut m = [[0.0; 4]; 4];
    m[0][0] = diag[0];
    m[1][1] = diag[1];
    m[2][2] = diag[2];
    m[3][3] = diag[3];
    m
}

fn mul4(a: [[f64; 4]; 4], b: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    let mut c = [[0.0; 4]; 4];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j] + a[i][3] * b[3][j];
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gr::lorentz::{boost_x, Event};

    fn assert_vec_close(got: Vector, want: Vector, tol: f64) {
        let g = got.components();
        let w = want.components();
        for (a, b) in g.iter().zip(w) {
            assert!((a - b).abs() < tol, "got {got:?} want {want:?}");
        }
    }

    fn rotation_xy(angle: f64) -> Tensor11 {
        let c = angle.cos();
        let s = angle.sin();
        let mut m = Tensor11::identity().components();
        m[1][1] = c;
        m[1][2] = -s;
        m[2][1] = s;
        m[2][2] = c;
        Tensor11::from_components(m)
    }

    fn boost_x_matrix(beta: f64, gamma: f64) -> Tensor11 {
        let mut m = Tensor11::identity().components();
        m[0][0] = gamma;
        m[0][1] = -gamma * beta;
        m[1][0] = -gamma * beta;
        m[1][1] = gamma;
        Tensor11::from_components(m)
    }

    #[test]
    fn contraction_on_known_numbers_is_exact() {
        let v = Vector::new(1.0, 2.0, 3.0, 4.0);
        let w = Covector::new(5.0, 6.0, 7.0, 8.0);
        let want = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert_eq!(v.contract(w), want);
        assert_eq!(w.contract(v), want);
        assert_eq!(want, 70.0);
        assert!(v.contract(w).is_finite());
    }

    #[test]
    fn trace_contracts_the_repeated_index() {
        let m = Tensor11::from_components([
            [1.0, 9.0, 0.0, 0.0],
            [0.0, 2.0, 8.0, 0.0],
            [0.0, 0.0, 3.0, 7.0],
            [4.0, 0.0, 0.0, 4.0],
        ]);
        assert_eq!(m.trace(), 10.0);
        assert_eq!(Tensor11::identity().trace(), 4.0);
    }

    #[test]
    fn eta_times_inverse_is_exactly_identity() {
        let product = Tensor20::minkowski().contract_weight(Tensor02::minkowski());
        assert_eq!(product, Tensor11::identity());
        assert_eq!(product.trace(), 4.0);
    }

    #[test]
    fn minkowski_lowers_by_flipping_time() {
        let v = Vector::new(2.0, 3.0, -1.0, 0.5);
        let w = v.lower();
        assert_eq!(w, Covector::new(-2.0, 3.0, -1.0, 0.5));
        assert_eq!(w.raise(), v);
    }

    #[test]
    fn raise_after_lower_restores_vector() {
        let v = Vector::new(4.0, 1.0, -2.0, 0.5);
        let back = v.lower().raise();
        assert_eq!(back, v);
        let w = Covector::new(-1.5, 0.25, 8.0, -3.0);
        assert_eq!(w.raise().lower(), w);
    }

    #[test]
    fn minkowski_on_event_matches_lorentz_interval_with_sign_dictionary() {
        let event = Event::new(4.0, 1.0, -2.0, 0.5);
        let v = Vector::from(event);
        let eta = Tensor02::minkowski();
        let s2 = event.interval_sq();
        assert!((eta.on(v, v) + s2).abs() < 1e-15);
        assert!((v.lower().contract(v) + s2).abs() < 1e-15);
        assert!((s2 - (event.ct * event.ct - 1.0 - 4.0 - 0.25)).abs() < 1e-15);
    }

    #[test]
    fn identity_machine_returns_the_same_axis() {
        let v = Vector::new(1.5, -0.4, 0.2, -0.7);
        assert_eq!(Tensor11::identity().apply(v), v);
    }

    #[test]
    fn rotation_changes_components_and_keeps_the_interval() {
        let v = Vector::new(0.0, 3.0, 4.0, 0.0);
        let turned = rotation_xy(std::f64::consts::FRAC_PI_2).apply(v);
        assert_vec_close(turned, Vector::new(0.0, -4.0, 3.0, 0.0), 1e-12);
        let eta = Tensor02::minkowski();
        assert!((eta.on(v, v) - eta.on(turned, turned)).abs() < 1e-12);
    }

    #[test]
    fn composed_machines_are_a_machine() {
        let a = rotation_xy(0.3);
        let b = rotation_xy(0.4);
        let v = Vector::new(1.0, 2.0, -1.5, 0.25);
        assert_vec_close(a.compose(b).apply(v), a.apply(b.apply(v)), 1e-12);
        assert_vec_close(a.compose(b).apply(v), rotation_xy(0.7).apply(v), 1e-12);
        assert_eq!(a.compose(Tensor11::identity()), a);
    }

    #[test]
    fn boost_machine_matches_lorentz_on_an_event() {
        let beta = 0.6;
        let gamma = 1.25;
        let event = Event::new(2.0, 1.0, 0.4, -0.3);
        let from_matrix = boost_x_matrix(beta, gamma).apply(Vector::from(event));
        let from_lorentz = Vector::from(boost_x(event, beta).unwrap());
        assert_vec_close(from_matrix, from_lorentz, 1e-15);
    }

    #[test]
    fn weight_on_two_vectors_is_symmetric_for_eta() {
        let u = Vector::new(1.0, 0.5, -0.2, 0.7);
        let v = Vector::new(-0.3, 1.5, 2.0, 0.0);
        let eta = Tensor02::minkowski();
        assert!((eta.on(u, v) - eta.on(v, u)).abs() < 1e-15);
    }

    #[test]
    fn raise_and_lower_round_trip_the_machine() {
        let machine = boost_x_matrix(0.6, 1.25);
        let weight = machine.lower();
        let back = weight.raise();
        for (got, want) in back
            .components()
            .iter()
            .flatten()
            .zip(machine.components().iter().flatten())
        {
            assert!((got - want).abs() < 1e-15);
        }
        let v = Vector::new(2.0, 1.0, 0.0, 0.0);
        assert!((weight.on(v, v) - Tensor02::minkowski().on(machine.apply(v), v)).abs() < 1e-12);
    }

    #[test]
    fn eta_diagonal_is_textbook() {
        let e = Tensor02::minkowski().components();
        assert_eq!(e[0][0], -1.0);
        assert_eq!(e[1][1], 1.0);
        assert_eq!(e[2][2], 1.0);
        assert_eq!(e[3][3], 1.0);
        for (i, row) in e.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                if i != j {
                    assert_eq!(val, 0.0);
                }
            }
        }
        assert_eq!(
            Tensor20::minkowski().components(),
            Tensor02::minkowski().components()
        );
    }
}
