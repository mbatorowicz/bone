// Geodezyjna zerowa Schwarzschilda, piksel = wątek. Kerr zostaje na CPU.
// Same skalarne f64: Naga nie obiecuje vec3<f64>.

struct RtParams {
    width: u32,
    height: u32,
    max_steps: u32,
    _pad: u32,
    mass: f64,
    cam_r: f64,
    cam_theta: f64,
    cam_phi: f64,
    fov_y: f64,
    r_inner: f64,
    r_outer: f64,
    r_escape: f64,
}

struct HitOut {
    tag: u32,
    _pad: u32,
    r: f64,
}

struct State {
    t: f64,
    r: f64,
    th: f64,
    ph: f64,
    ut: f64,
    ur: f64,
    uth: f64,
    uph: f64,
}

@group(0) @binding(0) var<storage, read> params: RtParams;
@group(0) @binding(1) var<storage, read_write> hits: array<HitOut>;

const TAG_HORIZON: u32 = 0u;
const TAG_ESCAPE: u32 = 1u;
const TAG_DISK: u32 = 2u;
const POLE_SIN_MIN: f64 = 1e-8lf;
const PI_HALF: f64 = 1.5707963267948966lf;

fn finite64(x: f64) -> bool {
    return !(isNan(x) || isInf(x));
}

fn affine_step(r: f64, mass: f64) -> f64 {
    var m = mass;
    if m < 1e-12lf {
        m = 1e-12lf;
    }
    var q = 0.07lf * (r / m);
    if q < 0.03lf {
        q = 0.03lf;
    }
    if q > 0.35lf {
        q = 0.35lf;
    }
    return q * m;
}

fn rhs(mass: f64, y: State, dy: ptr<function, State>) -> u32 {
    let r = y.r;
    let theta = y.th;
    if !finite64(r) || r <= 0.0lf {
        return 1u;
    }
    let f = 1.0lf - 2.0lf * mass / r;
    if abs(f) < 1e-18lf {
        return 1u;
    }
    if !finite64(theta) {
        return 2u;
    }
    let s = sin(theta);
    if abs(s) < 1e-16lf {
        return 2u;
    }
    let c = cos(theta);
    let r2 = r * r;
    let g_t_tr = mass / (r2 * f);
    let g_r_tt = mass * f / r2;
    let g_r_rr = -mass / (r2 * f);
    let g_r_thth = -r * f;
    let g_r_phph = -r * f * s * s;
    let g_th_rth = 1.0lf / r;
    let g_th_phph = -s * c;
    let g_ph_rph = 1.0lf / r;
    let g_ph_thph = c / s;
    (*dy).t = y.ut;
    (*dy).r = y.ur;
    (*dy).th = y.uth;
    (*dy).ph = y.uph;
    (*dy).ut = -2.0lf * g_t_tr * y.ut * y.ur;
    (*dy).ur = -(g_r_tt * y.ut * y.ut + g_r_rr * y.ur * y.ur + g_r_thth * y.uth * y.uth + g_r_phph * y.uph * y.uph);
    (*dy).uth = -(2.0lf * g_th_rth * y.ur * y.uth + g_th_phph * y.uph * y.uph);
    (*dy).uph = -(2.0lf * g_ph_rph * y.ur * y.uph + 2.0lf * g_ph_thph * y.uth * y.uph);
    return 0u;
}

fn add_scaled(a: State, b: State, s: f64) -> State {
    var o: State;
    o.t = a.t + s * b.t;
    o.r = a.r + s * b.r;
    o.th = a.th + s * b.th;
    o.ph = a.ph + s * b.ph;
    o.ut = a.ut + s * b.ut;
    o.ur = a.ur + s * b.ur;
    o.uth = a.uth + s * b.uth;
    o.uph = a.uph + s * b.uph;
    return o;
}

fn rk4(mass: f64, y: ptr<function, State>, h: f64) -> u32 {
    var k1: State;
    var k2: State;
    var k3: State;
    var k4: State;
    var err = rhs(mass, *y, &k1);
    if err != 0u {
        return err;
    }
    var tmp = add_scaled(*y, k1, 0.5lf * h);
    err = rhs(mass, tmp, &k2);
    if err != 0u {
        return err;
    }
    tmp = add_scaled(*y, k2, 0.5lf * h);
    err = rhs(mass, tmp, &k3);
    if err != 0u {
        return err;
    }
    tmp = add_scaled(*y, k3, h);
    err = rhs(mass, tmp, &k4);
    if err != 0u {
        return err;
    }
    (*y).t = (*y).t + (h / 6.0lf) * (k1.t + 2.0lf * k2.t + 2.0lf * k3.t + k4.t);
    (*y).r = (*y).r + (h / 6.0lf) * (k1.r + 2.0lf * k2.r + 2.0lf * k3.r + k4.r);
    (*y).th = (*y).th + (h / 6.0lf) * (k1.th + 2.0lf * k2.th + 2.0lf * k3.th + k4.th);
    (*y).ph = (*y).ph + (h / 6.0lf) * (k1.ph + 2.0lf * k2.ph + 2.0lf * k3.ph + k4.ph);
    (*y).ut = (*y).ut + (h / 6.0lf) * (k1.ut + 2.0lf * k2.ut + 2.0lf * k3.ut + k4.ut);
    (*y).ur = (*y).ur + (h / 6.0lf) * (k1.ur + 2.0lf * k2.ur + 2.0lf * k3.ur + k4.ur);
    (*y).uth = (*y).uth + (h / 6.0lf) * (k1.uth + 2.0lf * k2.uth + 2.0lf * k3.uth + k4.uth);
    (*y).uph = (*y).uph + (h / 6.0lf) * (k1.uph + 2.0lf * k2.uph + 2.0lf * k3.uph + k4.uph);
    return 0u;
}

fn equator_r(prev_th: f64, prev_r: f64, next_th: f64, next_r: f64) -> f64 {
    let a = prev_th - PI_HALF;
    let b = next_th - PI_HALF;
    if a * b >= 0.0lf || !finite64(a) || !finite64(b) {
        return -1.0lf;
    }
    let t = a / (a - b);
    let r = prev_r + t * (next_r - prev_r);
    if finite64(r) {
        return r;
    }
    return -1.0lf;
}

@compute @workgroup_size(8, 8)
fn raytrace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let p = params;
    let x = gid.x;
    let y = gid.y;
    if x >= p.width || y >= p.height {
        return;
    }
    let idx = y * p.width + x;
    let w = f64(p.width);
    let hgt = f64(p.height);
    let tany = tan(0.5lf * p.fov_y);
    let tanx = tany * (w / hgt);
    let sx = (2.0lf * (f64(x) + 0.5lf) / w - 1.0lf) * tanx;
    let sy = (1.0lf - 2.0lf * (f64(y) + 0.5lf) / hgt) * tany;
    let len = sqrt(sx * sx + sy * sy + 1.0lf);
    let nx = -1.0lf / len;
    let ny = -sy / len;
    let nz = sx / len;
    let f = 1.0lf - 2.0lf * p.mass / p.cam_r;
    if f <= 0.0lf {
        hits[idx].tag = TAG_HORIZON;
        hits[idx].r = 0.0lf;
        return;
    }
    let sqrt_f = sqrt(f);
    let sin_th = sin(p.cam_theta);
    var st: State;
    st.t = 0.0lf;
    st.r = p.cam_r;
    st.th = p.cam_theta;
    st.ph = p.cam_phi;
    st.ut = 1.0lf / sqrt_f;
    st.ur = nx * sqrt_f;
    st.uth = ny / p.cam_r;
    st.uph = nz / (p.cam_r * sin_th);

    let r_h = 2.0lf * p.mass;
    var mass_floor = p.mass;
    if mass_floor < 1e-12lf {
        mass_floor = 1e-12lf;
    }
    let capture_r = r_h + 0.04lf * mass_floor;
    var prev_r = st.r;
    var prev_th = st.th;
    var tag = TAG_ESCAPE;
    var hit_r = 0.0lf;
    var done = 0u;
    for (var s = 0u; s < p.max_steps; s++) {
        if !finite64(prev_r) || prev_r <= capture_r {
            tag = TAG_HORIZON;
            done = 1u;
            break;
        }
        if abs(sin(prev_th)) < POLE_SIN_MIN {
            tag = TAG_ESCAPE;
            done = 1u;
            break;
        }
        let step_h = affine_step(prev_r, p.mass);
        let err = rk4(p.mass, &st, step_h);
        if err == 2u {
            tag = TAG_ESCAPE;
            done = 1u;
            break;
        }
        if err != 0u {
            tag = TAG_HORIZON;
            done = 1u;
            break;
        }
        if !finite64(st.r) || st.r <= capture_r {
            tag = TAG_HORIZON;
            done = 1u;
            break;
        }
        let cross = equator_r(prev_th, prev_r, st.th, st.r);
        if cross >= p.r_inner && cross <= p.r_outer {
            tag = TAG_DISK;
            hit_r = cross;
            done = 1u;
            break;
        }
        if st.r >= p.r_escape && st.ur > 0.0lf {
            tag = TAG_ESCAPE;
            done = 1u;
            break;
        }
        prev_r = st.r;
        prev_th = st.th;
    }
    if done == 0u {
        if prev_r < 4.0lf * mass_floor {
            tag = TAG_HORIZON;
        } else {
            tag = TAG_ESCAPE;
        }
    }
    hits[idx].tag = tag;
    hits[idx].r = hit_r;
}
