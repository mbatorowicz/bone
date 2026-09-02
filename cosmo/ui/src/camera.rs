//! Kamera orbitalna i rzut na ekran.
//!
//! Kamera nie ma pozycji w przestrzeni symulacji — ma dwa kąty, przybliżenie
//! i przesunięcie rzutu. To, co widzi, jest wyśrodkowane na chmurze i przeskalowane
//! do jej rozciągłości; `pan` tylko przesuwa ten obraz po ekranie. Wynika to z tego,
//! co się symuluje: chmura zapada się o rzędy wielkości albo rozszerza razem ze
//! wszechświatem, więc kamera o ustalonym położeniu pokazywałaby przez większość
//! biegu pustą przestrzeń albo jedną plamę.

use bone_core::vec3::Vec3;

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    /// Przesunięcie rzutu w zrotowanej przestrzeni (przed skalą pikseli).
    pub pan: (f32, f32),
}

/// Pion jest ograniczony, bo za biegunem obrót zaczyna działać na odwrót i widok
/// staje się niesterowalny.
const PITCH_LIMIT: f32 = 1.2;
/// Górny próg ma pozwolić wypełnić kadr pojedynczym zgęstkiem; niżej przybliżenie
/// urywało się, zanim było widać strukturę wewnątrz chmury.
const ZOOM_RANGE: (f32, f32) = (0.15, 80.0);
const PAN_LIMIT: f32 = 3.0;

impl Default for Camera {
    fn default() -> Self {
        Self {
            yaw: 0.7,
            pitch: 0.35,
            zoom: 1.0,
            pan: (0.0, 0.0),
        }
    }
}

impl Camera {
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw += dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    pub fn zoom_by(&mut self, scroll: f32) {
        self.zoom = (self.zoom * (1.0 - scroll * 0.001)).clamp(ZOOM_RANGE.0, ZOOM_RANGE.1);
    }

    /// Przesuń rzut razem z wskaźnikiem: `dx`, `dy` to delta w pikselach.
    pub fn pan_by(&mut self, dx: f32, dy: f32, width: f32, height: f32) {
        let scale = Self::projection_scale(self.zoom, width, height);
        if scale <= 1e-6 {
            return;
        }
        self.pan.0 = (self.pan.0 + dx / scale).clamp(-PAN_LIMIT, PAN_LIMIT);
        self.pan.1 = (self.pan.1 + dy / scale).clamp(-PAN_LIMIT, PAN_LIMIT);
    }

    fn projection_scale(zoom: f32, width: f32, height: f32) -> f32 {
        width.min(height) * 0.42 * zoom
    }

    /// Ustaw rzut dla konkretnej chmury i rozmiaru obrazu.
    pub fn view(&self, center: Vec3, span: f64, width: usize, height: usize) -> View {
        let scale = Self::projection_scale(self.zoom, width as f32, height as f32);
        View {
            center,
            half_span: (0.5 * span).max(1e-9),
            cos_yaw: self.yaw.cos(),
            sin_yaw: self.yaw.sin(),
            cos_pitch: self.pitch.cos(),
            sin_pitch: self.pitch.sin(),
            scale,
            origin: (
                width as f32 * 0.5 + self.pan.0 * scale,
                height as f32 * 0.5 + self.pan.1 * scale,
            ),
            width,
            height,
        }
    }
}

/// Gotowy rzut: obrót i skala policzone raz na klatkę.
///
/// Istnieje osobno od [`Camera`], bo `project` jest wywoływane raz na cząstkę —
/// przy 250 tys. cząstek liczenie sinusów w środku pętli byłoby połową kosztu
/// rysowania.
pub struct View {
    center: Vec3,
    half_span: f64,
    cos_yaw: f32,
    sin_yaw: f32,
    cos_pitch: f32,
    sin_pitch: f32,
    scale: f32,
    origin: (f32, f32),
    width: usize,
    height: usize,
}

impl View {
    /// Piksel, na który pada cząstka; `None`, gdy wypada poza obraz.
    ///
    /// Odrzucane są też piksele DOKŁADNIE na krawędzi, bo rozmycie poświaty sięga
    /// jednego piksela w każdą stronę i inaczej trafiałoby na przeciwległy brzeg.
    pub fn project(&self, position: Vec3) -> Option<(usize, usize)> {
        let local = (position - self.center) / self.half_span;
        if !local.is_finite() {
            return None;
        }
        let (x, y, z) = (local.x as f32, local.y as f32, local.z as f32);
        let x_rot = x * self.cos_yaw + z * self.sin_yaw;
        let z_rot = -x * self.sin_yaw + z * self.cos_yaw;
        let y_rot = y * self.cos_pitch - z_rot * self.sin_pitch;

        let u = self.origin.0 + x_rot * self.scale;
        let v = self.origin.1 - y_rot * self.scale;
        if !(1.0..self.width as f32 - 1.0).contains(&u)
            || !(1.0..self.height as f32 - 1.0).contains(&v)
        {
            return None;
        }
        Some((u as usize, v as usize))
    }

    pub fn size(&self) -> (usize, usize) {
        (self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bone_core::vec3::{vec3, ZERO};

    fn view() -> View {
        Camera {
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
            pan: (0.0, 0.0),
        }
        .view(ZERO, 2.0, 200, 100)
    }

    #[test]
    fn center_lands_in_the_middle() {
        let (u, v) = view().project(ZERO).expect("środek jest widoczny");
        assert_eq!((u, v), (100, 50));
    }

    /// Ekranowe „w górę" musi odpowiadać rosnącemu `y` w symulacji. Odwrócona os
    /// jest błędem, którego nie widać na chmurze symetrycznej — a każda chmura
    /// startowa taka jest.
    #[test]
    fn positive_y_goes_up_on_screen() {
        let v = view();
        let middle = v.project(ZERO).unwrap().1;
        let above = v.project(vec3(0.0, 0.5, 0.0)).unwrap().1;
        assert!(above < middle, "y w górę dało v={above}, środek {middle}");
    }

    #[test]
    fn yaw_rotates_around_the_vertical_axis() {
        let flat = |yaw: f32| {
            Camera {
                yaw,
                pitch: 0.0,
                zoom: 1.0,
                pan: (0.0, 0.0),
            }
            .view(ZERO, 2.0, 200, 200)
        };
        // Po obrocie o 90° os z pada tam, gdzie wcześniej padała os x.
        let from_z = flat(std::f32::consts::FRAC_PI_2).project(vec3(0.0, 0.0, 0.5));
        let from_x = flat(0.0).project(vec3(0.5, 0.0, 0.0));
        assert_eq!(from_z, from_x);
    }

    #[test]
    fn far_particles_fall_outside() {
        assert!(view().project(vec3(100.0, 0.0, 0.0)).is_none());
        assert!(view().project(vec3(f64::NAN, 0.0, 0.0)).is_none());
    }

    /// Widok jest normalizowany rozciągłością chmury, więc ta sama figura w innej
    /// skali musi dać ten sam obraz. To jest cały mechanizm śledzenia chmury.
    #[test]
    fn projection_is_scale_invariant() {
        let camera = Camera::default();
        let small = camera.view(ZERO, 2.0, 200, 200);
        let large = camera.view(ZERO, 2.0e9, 200, 200);
        assert_eq!(
            small.project(vec3(0.3, -0.2, 0.1)),
            large.project(vec3(0.3e9, -0.2e9, 0.1e9))
        );
    }

    #[test]
    fn orbit_and_zoom_stay_within_limits() {
        let mut camera = Camera::default();
        for _ in 0..1000 {
            camera.orbit(50.0, 50.0);
            camera.zoom_by(-100.0);
            camera.pan_by(200.0, -200.0, 200.0, 200.0);
        }
        assert!(camera.pitch <= PITCH_LIMIT);
        assert!(camera.zoom <= ZOOM_RANGE.1);
        assert!(camera.pan.0 <= PAN_LIMIT);
        assert!(camera.pan.1 >= -PAN_LIMIT);
        for _ in 0..1000 {
            camera.orbit(0.0, -50.0);
            camera.zoom_by(100.0);
            camera.pan_by(-200.0, 200.0, 200.0, 200.0);
        }
        assert!(camera.pitch >= -PITCH_LIMIT);
        assert!(camera.zoom >= ZOOM_RANGE.0);
        assert!(camera.pan.0 >= -PAN_LIMIT);
        assert!(camera.pan.1 <= PAN_LIMIT);
    }

    /// Przeciągnięcie o N pikseli ma przesunąć środek chmury o te same N pikseli.
    /// Inna skala złamałaby „chwytanie" sceny: kursor uciekałby spod punktu.
    #[test]
    fn pan_moves_the_projected_center_with_the_pointer() {
        let mut camera = Camera {
            yaw: 0.0,
            pitch: 0.0,
            zoom: 1.0,
            pan: (0.0, 0.0),
        };
        let before = camera.view(ZERO, 2.0, 200, 100).project(ZERO).unwrap();
        camera.pan_by(20.0, 10.0, 200.0, 100.0);
        let after = camera.view(ZERO, 2.0, 200, 100).project(ZERO).unwrap();
        assert_eq!(after.0, before.0 + 20);
        assert_eq!(after.1, before.1 + 10);
    }

    /// Degeneracja chmury (jedna cząstka, zerowa rozciągłość) nie może dawać
    /// dzielenia przez zero — po zapadnięciu układu to jest stan osiągalny.
    #[test]
    fn zero_span_does_not_break_the_projection() {
        let v = Camera::default().view(ZERO, 0.0, 100, 100);
        assert!(v.project(ZERO).is_some());
    }
}
