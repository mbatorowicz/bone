// Siły O(N²) w f64. Ten sam wzór co exact.rs: φ i F z (xᵢ − xⱼ).
// Człon j = i pomijany jawnie.

struct Particle {
    x: f64,
    y: f64,
    z: f64,
    mass: f64,
}

struct Out {
    fx: f64,
    fy: f64,
    fz: f64,
    phi: f64,
}

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    g: f64,
    eps2: f64,
}

@group(0) @binding(0) var<storage, read> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> field: array<Out>;
@group(0) @binding(2) var<storage, read> params: Params;

@compute @workgroup_size(64)
fn nbody(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = params.n;
    if i >= n {
        return;
    }
    let pi = particles[i];
    var pull_x = 0.0lf;
    var pull_y = 0.0lf;
    var pull_z = 0.0lf;
    var phi = 0.0lf;
    for (var j = 0u; j < n; j++) {
        if j == i {
            continue;
        }
        let pj = particles[j];
        let dx = pi.x - pj.x;
        let dy = pi.y - pj.y;
        let dz = pi.z - pj.z;
        let inv_r = 1.0lf / sqrt(dx * dx + dy * dy + dz * dz + params.eps2);
        let m = pj.mass;
        phi += m * inv_r;
        let s = m * inv_r * inv_r * inv_r;
        pull_x += dx * s;
        pull_y += dy * s;
        pull_z += dz * s;
    }
    let scale = -params.g * pi.mass;
    field[i].fx = pull_x * scale;
    field[i].fy = pull_y * scale;
    field[i].fz = pull_z * scale;
    field[i].phi = -params.g * phi;
}
