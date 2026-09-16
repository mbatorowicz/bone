// 1D radix-2 w linii 3D siatki. Konwencja rustfft: do przodu −2π, wstecz +2π,
// bez 1/n. Po bit-reversal butterfly in-place.

struct FftParams {
    n: u32,
    stride: u32,
    n_lines: u32,
    stage: u32,
    inverse: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read_write> data: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> params: FftParams;

fn line_origin(line: u32) -> u32 {
    let n = params.n;
    let stride = params.stride;
    if stride == 1u {
        return line * n;
    }
    if stride == n {
        let z = line / n;
        let x = line % n;
        return (z * n) * n + x;
    }
    let y = line / n;
    let x = line % n;
    return y * n + x;
}

fn bit_reverse(v: u32, logn: u32) -> u32 {
    var x = v;
    var r = 0u;
    for (var i = 0u; i < logn; i++) {
        r = (r << 1u) | (x & 1u);
        x = x >> 1u;
    }
    return r;
}

fn log2_n(n: u32) -> u32 {
    var v = n;
    var logn = 0u;
    while v > 1u {
        v = v >> 1u;
        logn += 1u;
    }
    return logn;
}

@compute @workgroup_size(64)
fn bitrev(@builtin(global_invocation_id) gid: vec3<u32>) {
    let line = gid.x;
    if line >= params.n_lines {
        return;
    }
    let n = params.n;
    let stride = params.stride;
    let origin = line_origin(line);
    let logn = log2_n(n);
    for (var k = 0u; k < n; k++) {
        let r = bit_reverse(k, logn);
        if r > k {
            let i = origin + k * stride;
            let j = origin + r * stride;
            let tmp = data[i];
            data[i] = data[j];
            data[j] = tmp;
        }
    }
}

@compute @workgroup_size(64)
fn butterfly(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n;
    let half = 1u << params.stage;
    let butterflies = n >> 1u;
    let job = gid.x;
    let total = params.n_lines * butterflies;
    if job >= total {
        return;
    }
    let line = job / butterflies;
    let b = job % butterflies;
    let block = b / half;
    let j = b % half;
    let span = half << 1u;
    let origin = line_origin(line);
    let stride = params.stride;
    let i1 = origin + (block * span + j) * stride;
    let i2 = origin + (block * span + j + half) * stride;
    let sign = select(-1.0, 1.0, params.inverse != 0u);
    let angle = sign * 2.0 * 3.14159265358979323846 * f32(j) / f32(span);
    let wr = cos(angle);
    let wi = sin(angle);
    let a = data[i1];
    let c = data[i2];
    let tr = wr * c.x - wi * c.y;
    let ti = wr * c.y + wi * c.x;
    data[i1] = vec2<f32>(a.x + tr, a.y + ti);
    data[i2] = vec2<f32>(a.x - tr, a.y - ti);
}
