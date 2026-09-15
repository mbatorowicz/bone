//! Obraz Schwarzschilda: piksel to geodezyjna zerowa liczona wstecz od kamery.
//!
//! UI i wątek tła są w kroku 14. Tu jest tylko bufor: kamera w tetradzie
//! statycznego obserwatora, RK4 z [`super::geodesic`], los `horyzont` /
//! `dysk` / `ucieczka`. Sygnatura (−,+,+,+) jak w [`super::metric`].
//!
//! Domyślny kadr to 320×180 na CPU. Testy biorą 32×18, żeby kończyć się
//! w rozsądnym czasie bez okna.

use std::f64::consts::FRAC_PI_2;
use std::fmt;

use super::geodesic::{self, GeodesicError, GeodesicState, STATE_LEN};
use super::metric::{horizon_radius, MetricError, Schwarzschild};
use super::rk4::Scratch;

/// Szerokość kadru, który UI podłączy później. Testy biorą [`WIDTH_TINY`].
pub const WIDTH_DEFAULT: u32 = 320;
/// Wysokość kadru 16:9 przy [`WIDTH_DEFAULT`].
pub const HEIGHT_DEFAULT: u32 = 180;
/// Mały obraz testowy: dość pikseli, żeby centrum i róg były różnym losem.
pub const WIDTH_TINY: u32 = 32;
pub const HEIGHT_TINY: u32 = 18;
/// Pionowe pole widzenia kamery kursu, radiany.
pub const FOV_Y: f64 = 0.70;
/// Nachylenie: dość od równika, żeby dysk przecinał się jako płaszczyzna, nie
/// jako cała orbita.
pub const INCLINATION: f64 = 5.0 * std::f64::consts::PI / 12.0;
/// Kamera na łące: `30M`.
pub const CAMERA_R_OVER_M: f64 = 30.0;
/// Zewnętrzny brzeg cienkiego dysku w równiku, w jednostkach `M`.
pub const DISK_OUTER_OVER_M: f64 = 20.0;
const MAX_STEPS_DEFAULT: usize = 1600;
const POLE_SIN_MIN: f64 = 1e-8;

/// Kamera, rozmiar albo metryka nie dają obrazu.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RaytraceError {
    Metric(MetricError),
    Geodesic(GeodesicError),
    CameraInside { r: f64, horizon: f64 },
    BadCamera { r: f64, theta: f64, fov_y: f64 },
    BadSize { width: u32, height: u32 },
    BadDisk { r_inner: f64, r_outer: f64 },
}

impl fmt::Display for RaytraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Metric(ref e) => write!(f, "{e}"),
            Self::Geodesic(ref e) => write!(f, "{e}"),
            Self::CameraInside { r, horizon } => {
                write!(
                    f,
                    "kamera r={r} jest na horyzoncie albo pod nim (2M={horizon})"
                )
            }
            Self::BadCamera { r, theta, fov_y } => {
                write!(
                    f,
                    "kamera r={r}, θ={theta}, fov_y={fov_y} nie jest miejscem obserwacji"
                )
            }
            Self::BadSize { width, height } => {
                write!(f, "obraz {width}×{height} musi mieć dodatnie wymiary")
            }
            Self::BadDisk { r_inner, r_outer } => {
                write!(
                    f,
                    "dysk r∈[{r_inner}, {r_outer}] musi być odcinkiem poza zerem"
                )
            }
        }
    }
}

impl std::error::Error for RaytraceError {}

impl From<MetricError> for RaytraceError {
    fn from(value: MetricError) -> Self {
        Self::Metric(value)
    }
}

impl From<GeodesicError> for RaytraceError {
    fn from(value: GeodesicError) -> Self {
        Self::Geodesic(value)
    }
}

/// Statyczny obserwator patrzący na środek. `fov_y` w radianach, `θ` od osi `z`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    r: f64,
    theta: f64,
    phi: f64,
    fov_y: f64,
}

impl Camera {
    pub fn new(r: f64, theta: f64, phi: f64, fov_y: f64) -> Result<Self, RaytraceError> {
        let sin = theta.sin();
        if !r.is_finite()
            || r <= 0.0
            || !theta.is_finite()
            || sin.abs() < 1e-6
            || !phi.is_finite()
            || !fov_y.is_finite()
            || fov_y <= 0.0
            || fov_y >= std::f64::consts::PI
        {
            return Err(RaytraceError::BadCamera { r, theta, fov_y });
        }
        Ok(Self {
            r,
            theta,
            phi,
            fov_y,
        })
    }

    /// Kurs: `r = 30M`, nachylenie ~75°, pole ~40°.
    pub fn course(bh: Schwarzschild) -> Result<Self, RaytraceError> {
        let mass = bh.mass();
        if mass == 0.0 {
            return Err(RaytraceError::BadCamera {
                r: 0.0,
                theta: INCLINATION,
                fov_y: FOV_Y,
            });
        }
        Self::new(CAMERA_R_OVER_M * mass, INCLINATION, 0.0, FOV_Y)
    }

    pub fn r(self) -> f64 {
        self.r
    }

    pub fn theta(self) -> f64 {
        self.theta
    }

    pub fn phi(self) -> f64 {
        self.phi
    }

    pub fn fov_y(self) -> f64 {
        self.fov_y
    }
}

/// Cienki dysk w równiku: pierwsza przecięcie `θ = π/2` w `[r_inner, r_outer]`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Disk {
    pub r_inner: f64,
    pub r_outer: f64,
}

impl Disk {
    pub fn new(r_inner: f64, r_outer: f64) -> Result<Self, RaytraceError> {
        if !r_inner.is_finite() || !r_outer.is_finite() || r_inner <= 0.0 || r_outer <= r_inner {
            return Err(RaytraceError::BadDisk { r_inner, r_outer });
        }
        Ok(Self { r_inner, r_outer })
    }

    /// ISCO do `20M`. Dziura w środku zostawia cień.
    pub fn course(bh: Schwarzschild) -> Result<Self, RaytraceError> {
        let mass = bh.mass();
        if mass == 0.0 {
            return Err(RaytraceError::BadDisk {
                r_inner: 0.0,
                r_outer: 0.0,
            });
        }
        Self::new(bh.isco_radius(), DISK_OUTER_OVER_M * mass)
    }
}

/// Kadr, kamera i dysk. [`Config::course`] to 320×180; testy biorą [`Config::tiny`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Config {
    pub metric: Schwarzschild,
    pub camera: Camera,
    pub disk: Disk,
    pub width: u32,
    pub height: u32,
    pub max_steps: usize,
}

impl Config {
    pub fn course(mass: f64) -> Result<Self, RaytraceError> {
        Self::with_size(mass, WIDTH_DEFAULT, HEIGHT_DEFAULT)
    }

    pub fn tiny(mass: f64) -> Result<Self, RaytraceError> {
        Self::with_size(mass, WIDTH_TINY, HEIGHT_TINY)
    }

    pub fn with_size(mass: f64, width: u32, height: u32) -> Result<Self, RaytraceError> {
        if width == 0 || height == 0 {
            return Err(RaytraceError::BadSize { width, height });
        }
        let metric = Schwarzschild::new(mass)?;
        let camera = Camera::course(metric)?;
        let horizon = metric.horizon_radius();
        if camera.r() <= horizon {
            return Err(RaytraceError::CameraInside {
                r: camera.r(),
                horizon,
            });
        }
        Ok(Self {
            metric,
            camera,
            disk: Disk::course(metric)?,
            width,
            height,
            max_steps: MAX_STEPS_DEFAULT,
        })
    }
}

/// Los promienia: wpada pod horyzont, trafia w dysk albo ucieka.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Hit {
    Horizon,
    Disk { r: f64 },
    Escape,
}

impl Hit {
    pub fn is_horizon(self) -> bool {
        matches!(self, Self::Horizon)
    }

    pub fn rgba(self) -> [u8; 4] {
        match self {
            Self::Horizon => [0, 0, 0, 255],
            Self::Escape => [10, 12, 20, 255],
            Self::Disk { r } => disk_rgba(r),
        }
    }
}

/// Bufor RGBA plus los każdego piksela. Wiersze od góry, `x` od lewej.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
    pub hits: Vec<Hit>,
}

impl Buffer {
    pub fn hit(&self, x: u32, y: u32) -> Hit {
        self.hits[self.index(x, y)]
    }

    pub fn pixel_rgba(&self, x: u32, y: u32) -> [u8; 4] {
        let i = self.index(x, y) * 4;
        [
            self.rgba[i],
            self.rgba[i + 1],
            self.rgba[i + 2],
            self.rgba[i + 3],
        ]
    }

    fn index(&self, x: u32, y: u32) -> usize {
        (y as usize) * (self.width as usize) + (x as usize)
    }
}

/// `b = L/E` z całkowitego momentu, nie tylko z `L_z`.
pub fn impact_parameter(bh: Schwarzschild, s: GeodesicState) -> Result<f64, MetricError> {
    let energy = s.energy(bh)?;
    if !energy.is_finite() || energy.abs() < 1e-18 {
        return Ok(f64::INFINITY);
    }
    let sin = s.theta.sin();
    let l_tot = s.r * s.r * (s.u_theta * s.u_theta + sin * sin * s.u_phi * s.u_phi).sqrt();
    Ok(l_tot / energy.abs())
}

/// Promień dokładnie w stronę masy: `n = (−1, 0, 0)` w tetradzie, `b = 0`.
pub fn seed_look(bh: Schwarzschild, camera: Camera) -> Result<GeodesicState, RaytraceError> {
    seed_direction(bh, camera, [-1.0, 0.0, 0.0])
}

/// Promień zerowy z piksela: do przodu −ê_r, w prawo ê_φ, w górę −ê_θ.
pub fn seed(
    bh: Schwarzschild,
    camera: Camera,
    width: u32,
    height: u32,
    x: u32,
    y: u32,
) -> Result<GeodesicState, RaytraceError> {
    if width == 0 || height == 0 || x >= width || y >= height {
        return Err(RaytraceError::BadSize { width, height });
    }
    let horizon = bh.horizon_radius();
    if camera.r() <= horizon {
        return Err(RaytraceError::CameraInside {
            r: camera.r(),
            horizon,
        });
    }
    let n = pixel_direction(camera, width, height, x, y);
    seed_direction(bh, camera, n)
}

/// Całkuje geodezyjną aż do horyzontu, dysku albo ucieczki.
pub fn trace(
    bh: Schwarzschild,
    start: GeodesicState,
    disk: Disk,
    r_escape: f64,
    max_steps: usize,
    scratch: &mut Scratch,
) -> Hit {
    let mut y = start.to_array();
    let mut prev = start;
    let r_h = bh.horizon_radius();
    let capture_r = r_h + 0.04 * bh.mass().max(1e-12);
    for _ in 0..max_steps {
        if !prev.r.is_finite() || prev.r <= capture_r {
            return Hit::Horizon;
        }
        if prev.theta.sin().abs() < POLE_SIN_MIN {
            return Hit::Escape;
        }
        let h = affine_step(prev.r, bh.mass());
        match geodesic::step(bh, 0.0, &mut y, h, scratch) {
            Ok(()) => {}
            Err(GeodesicError::Metric(MetricError::InvalidAngle { .. })) => return Hit::Escape,
            Err(_) => return Hit::Horizon,
        }
        let next = match GeodesicState::from_slice(&y) {
            Ok(s) => s,
            Err(_) => return Hit::Horizon,
        };
        if !next.r.is_finite() || next.r <= capture_r {
            return Hit::Horizon;
        }
        if let Some(r_cross) = equator_radius(prev, next) {
            if r_cross >= disk.r_inner && r_cross <= disk.r_outer {
                return Hit::Disk { r: r_cross };
            }
        }
        if next.r >= r_escape && next.u_r > 0.0 {
            return Hit::Escape;
        }
        prev = next;
    }
    if prev.r < 4.0 * bh.mass().max(1e-12) {
        Hit::Horizon
    } else {
        Hit::Escape
    }
}

/// Liczy cały kadr. CPU, bez wątku tła.
pub fn render(cfg: Config) -> Result<Buffer, RaytraceError> {
    if cfg.width == 0 || cfg.height == 0 {
        return Err(RaytraceError::BadSize {
            width: cfg.width,
            height: cfg.height,
        });
    }
    let n = (cfg.width as usize)
        .checked_mul(cfg.height as usize)
        .ok_or(RaytraceError::BadSize {
            width: cfg.width,
            height: cfg.height,
        })?;
    let mut rgba = vec![0_u8; n.saturating_mul(4)];
    let mut hits = vec![Hit::Escape; n];
    let mut scratch = Scratch::with_len(STATE_LEN);
    let r_escape = cfg.camera.r() * 1.15;
    for y in 0..cfg.height {
        for x in 0..cfg.width {
            let start = seed(cfg.metric, cfg.camera, cfg.width, cfg.height, x, y)?;
            let hit = trace(
                cfg.metric,
                start,
                cfg.disk,
                r_escape,
                cfg.max_steps,
                &mut scratch,
            );
            let i = (y as usize) * (cfg.width as usize) + (x as usize);
            hits[i] = hit;
            let c = hit.rgba();
            let o = i * 4;
            rgba[o] = c[0];
            rgba[o + 1] = c[1];
            rgba[o + 2] = c[2];
            rgba[o + 3] = c[3];
        }
    }
    Ok(Buffer {
        width: cfg.width,
        height: cfg.height,
        rgba,
        hits,
    })
}

fn pixel_direction(camera: Camera, width: u32, height: u32, x: u32, y: u32) -> [f64; 3] {
    let w = width as f64;
    let h = height as f64;
    let tany = (0.5 * camera.fov_y()).tan();
    let tanx = tany * (w / h);
    let sx = (2.0 * (x as f64 + 0.5) / w - 1.0) * tanx;
    let sy = (1.0 - 2.0 * (y as f64 + 0.5) / h) * tany;
    let len = (sx * sx + sy * sy + 1.0).sqrt();
    // kamera: prawo = ê_φ, góra = −ê_θ, do przodu (w stronę masy) = −ê_r
    [-1.0 / len, -sy / len, sx / len]
}

fn seed_direction(
    bh: Schwarzschild,
    camera: Camera,
    n: [f64; 3],
) -> Result<GeodesicState, RaytraceError> {
    let f = bh.f(camera.r())?;
    if f <= 0.0 {
        return Err(RaytraceError::CameraInside {
            r: camera.r(),
            horizon: horizon_radius(bh.mass()),
        });
    }
    let sin_th = camera.theta().sin();
    let sqrt_f = f.sqrt();
    Ok(GeodesicState {
        t: 0.0,
        r: camera.r(),
        theta: camera.theta(),
        phi: camera.phi(),
        u_t: 1.0 / sqrt_f,
        u_r: n[0] * sqrt_f,
        u_theta: n[1] / camera.r(),
        u_phi: n[2] / (camera.r() * sin_th),
    })
}

fn affine_step(r: f64, mass: f64) -> f64 {
    let m = mass.max(1e-12);
    (0.07 * (r / m)).clamp(0.03, 0.35) * m
}

fn equator_radius(prev: GeodesicState, next: GeodesicState) -> Option<f64> {
    let a = prev.theta - FRAC_PI_2;
    let b = next.theta - FRAC_PI_2;
    if a * b >= 0.0 || !a.is_finite() || !b.is_finite() {
        return None;
    }
    let t = a / (a - b);
    let r = prev.r + t * (next.r - prev.r);
    if r.is_finite() {
        Some(r)
    } else {
        None
    }
}

fn disk_rgba(r: f64) -> [u8; 4] {
    let t = (12.0 / r.max(1.0)).clamp(0.15, 1.0);
    [
        (70.0 + 185.0 * t) as u8,
        (24.0 + 90.0 * t) as u8,
        (12.0 + 18.0 * t) as u8,
        255,
    ]
}

/// Z łąki, w równiku, do środka. Do testów `b` vs `3√3 M`, bez kamery.
pub fn equatorial_inward(
    bh: Schwarzschild,
    r: f64,
    energy: f64,
    impact: f64,
) -> Result<GeodesicState, RaytraceError> {
    let f = bh.f(r)?;
    if f <= 0.0 {
        return Err(RaytraceError::CameraInside {
            r,
            horizon: bh.horizon_radius(),
        });
    }
    let u_t = energy / f;
    let u_phi = impact / (r * r);
    let ur2 = energy * energy - f * impact * impact / (r * r);
    if ur2 < -1e-12 {
        return Err(RaytraceError::BadCamera {
            r,
            theta: FRAC_PI_2,
            fov_y: 0.0,
        });
    }
    Ok(GeodesicState {
        t: 0.0,
        r,
        theta: FRAC_PI_2,
        phi: 0.0,
        u_t,
        u_r: -ur2.max(0.0).sqrt(),
        u_theta: 0.0,
        u_phi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mass_one() -> Schwarzschild {
        Schwarzschild::new(1.0).unwrap()
    }

    fn scratch() -> Scratch {
        Scratch::with_len(STATE_LEN)
    }

    #[test]
    fn course_resolution_is_three_twenty_by_one_eighty() {
        let cfg = Config::course(1.0).unwrap();
        assert_eq!(cfg.width, 320);
        assert_eq!(cfg.height, 180);
        assert_eq!(WIDTH_DEFAULT, 320);
        assert_eq!(HEIGHT_DEFAULT, 180);
    }

    #[test]
    fn seed_is_null_and_center_has_zero_impact() {
        let bh = mass_one();
        let cam = Camera::course(bh).unwrap();
        let look = seed_look(bh, cam).unwrap();
        assert!(
            look.u_sq(bh).unwrap().abs() < 1e-12,
            "κ = {}",
            look.u_sq(bh).unwrap()
        );
        let b = impact_parameter(bh, look).unwrap();
        assert!(b.abs() < 1e-12, "b wektora −ê_r = {b}");
        assert!(look.u_r < 0.0);
        let mid = seed(bh, cam, 32, 18, 16, 9).unwrap();
        assert!(mid.u_sq(bh).unwrap().abs() < 1e-12);
        let b_mid = impact_parameter(bh, mid).unwrap();
        assert!(
            b_mid < 3.0 * 3.0_f64.sqrt(),
            "b piksela centrum = {b_mid}, ma być < 3√3"
        );
        let corner = seed(bh, cam, 32, 18, 0, 0).unwrap();
        assert!(corner.u_sq(bh).unwrap().abs() < 1e-12);
        let b_corner = impact_parameter(bh, corner).unwrap();
        assert!(
            b_corner > 3.0 * 3.0_f64.sqrt(),
            "b rogu = {b_corner}, ma być > 3√3"
        );
    }

    #[test]
    fn tiny_image_center_is_black_horizon() {
        let img = render(Config::tiny(1.0).unwrap()).unwrap();
        assert_eq!(img.width, 32);
        assert_eq!(img.height, 18);
        // 32×18 nie ma piksela w geometrycznym środku; (16, 9) jest najbliżej.
        let cx = 16;
        let cy = 9;
        assert!(
            img.hit(cx, cy).is_horizon(),
            "centrum = {:?}",
            img.hit(cx, cy)
        );
        assert_eq!(img.pixel_rgba(cx, cy), [0, 0, 0, 255]);
        assert!(
            img.hit(15, 8).is_horizon(),
            "centrum (15,8) = {:?}",
            img.hit(15, 8)
        );
        assert!(
            img.hit(15, 9).is_horizon(),
            "centrum (15,9) = {:?}",
            img.hit(15, 9)
        );
        assert!(
            img.hit(16, 8).is_horizon(),
            "centrum (16,8) = {:?}",
            img.hit(16, 8)
        );
    }

    #[test]
    fn large_impact_pixels_are_not_horizon() {
        let img = render(Config::tiny(1.0).unwrap()).unwrap();
        for (x, y) in [(0, 0), (31, 0), (0, 17), (31, 17)] {
            assert!(
                !img.hit(x, y).is_horizon(),
                "róg ({x},{y}) = {:?}",
                img.hit(x, y)
            );
            assert_ne!(img.pixel_rgba(x, y), [0, 0, 0, 255]);
        }
    }

    #[test]
    fn equatorial_below_critical_hits_horizon() {
        let bh = mass_one();
        let start = equatorial_inward(bh, 30.0, 1.0, 4.0).unwrap();
        assert!(start.u_sq(bh).unwrap().abs() < 1e-12);
        let b = impact_parameter(bh, start).unwrap();
        assert!(b < 3.0 * 3.0_f64.sqrt());
        let hit = trace(
            bh,
            start,
            Disk::course(bh).unwrap(),
            35.0,
            2000,
            &mut scratch(),
        );
        assert!(hit.is_horizon(), "b={b} dało {hit:?}");
    }

    #[test]
    fn equatorial_above_critical_is_not_horizon() {
        let bh = mass_one();
        let start = equatorial_inward(bh, 30.0, 1.0, 8.0).unwrap();
        let b = impact_parameter(bh, start).unwrap();
        assert!(b > 3.0 * 3.0_f64.sqrt());
        let hit = trace(
            bh,
            start,
            Disk::course(bh).unwrap(),
            35.0,
            2000,
            &mut scratch(),
        );
        assert!(!hit.is_horizon(), "b={b} dało horyzont");
        assert_eq!(hit, Hit::Escape);
    }

    #[test]
    fn tiny_image_has_a_disk_pixel() {
        let img = render(Config::tiny(1.0).unwrap()).unwrap();
        let disks = img
            .hits
            .iter()
            .filter(|h| matches!(h, Hit::Disk { .. }))
            .count();
        assert!(
            disks > 0,
            "32×18 pod 75° powinno trafić w dysk, dostało 0 (horyzont {}, ucieczka {})",
            img.hits.iter().filter(|h| h.is_horizon()).count(),
            img.hits.iter().filter(|h| matches!(h, Hit::Escape)).count()
        );
    }

    #[test]
    fn rejects_bad_camera_size_disk() {
        assert!(Camera::new(10.0, 0.0, 0.0, 0.7).is_err());
        assert!(Camera::new(10.0, 1.2, 0.0, 0.0).is_err());
        assert!(Camera::course(Schwarzschild::new(0.0).unwrap()).is_err());
        assert!(matches!(
            Config::with_size(1.0, 0, 18),
            Err(RaytraceError::BadSize {
                width: 0,
                height: 18
            })
        ));
        let bh = mass_one();
        let cam = Camera::new(1.5, INCLINATION, 0.0, FOV_Y).unwrap();
        assert!(matches!(
            seed(bh, cam, 8, 8, 0, 0),
            Err(RaytraceError::CameraInside { .. })
        ));
        assert!(Disk::new(8.0, 4.0).is_err());
    }

    #[test]
    fn photon_sphere_critical_impact_matches_geodesic() {
        let bh = mass_one();
        let s = GeodesicState::circular_photon(bh).unwrap();
        let b = impact_parameter(bh, s).unwrap();
        assert!((b - 3.0 * 3.0_f64.sqrt()).abs() < 1e-12);
    }
}
