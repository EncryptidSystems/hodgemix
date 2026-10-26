// Graded-spike BitLinear for Champ WebGPU.
//
// A nonzero int8 activation is one event. A zero activation is not stored and
// is not read. The integer accumulator is
//   acc = sum_j (code(row, j) - CENTER) * q_j
// which is the sum bitlinear.wgsl reduces with dot4. The scale xs[0] * ws[0]
// is applied once, after the sum, so the stored f32 matches bitlinear.wgsl
// bit for bit when the integer sums match.
//
// Weights are packed the same way as bitlinear.wgsl: uint8 [NP, K], byte k of
// packed row r holds the 2-bit codes of output rows r, r+NP, r+2NP, r+3NP.
// xq is K/4 little-endian int8 words. CENTER is (radix-1)/2: 1 for a ternary
// limb, 0 for radix 2, 1 for radix 4. ACCUM adds into o instead of replacing it,
// so a later limb uses the same event list.
//
// compact runs as one workgroup (dispatch 1,1,1). It writes ev[0..count) and
// evn[0] = count. ev must hold K u32s. apply then uses bitlinear's dispatch,
// ceil(NP/8) workgroups of 256 threads.

override K: u32;               // inputs, a multiple of 4
override NP: u32;              // packed rows = outputs / 4
override ACCUM: bool = false;
override CENTER: i32 = 1;

const ROWS = 8u;
const LANES = 32u;

@group(0) @binding(0) var<storage, read> xq: array<u32>;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<u32>;
@group(0) @binding(3) var<storage, read> ws: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;
@group(0) @binding(5) var<storage, read_write> ev: array<u32>;   // K packed events
@group(0) @binding(6) var<storage, read_write> evn: array<u32>;  // [0] = event count

var<workgroup> cnt: array<u32, 256>;
var<workgroup> off: array<u32, 256>;
var<workgroup> red: array<vec4<i32>, 256>;

@compute @workgroup_size(256)
fn compact(@builtin(local_invocation_index) tid: u32) {
  let kw = K / 4u;
  var c = 0u;
  for (var i = tid; i < kw; i += 256u) {
    let a = xq[i];
    for (var t = 0u; t < 4u; t++) {
      let q = (bitcast<i32>(a << (24u - 8u * t))) >> 24u;
      if (q != 0) {
        c += 1u;
      }
    }
  }
  cnt[tid] = c;
  workgroupBarrier();
  if (tid == 0u) {
    var s = 0u;
    for (var i = 0u; i < 256u; i++) {
      let n = cnt[i];
      off[i] = s;
      s += n;
    }
    evn[0] = s;
  }
  workgroupBarrier();
  var slot = off[tid];
  for (var i = tid; i < kw; i += 256u) {
    let a = xq[i];
    for (var t = 0u; t < 4u; t++) {
      let q = (bitcast<i32>(a << (24u - 8u * t))) >> 24u;
      if (q != 0) {
        let col = i * 4u + t;
        ev[slot] = (col << 8u) | (bitcast<u32>(q) & 0xFFu);
        slot += 1u;
      }
    }
  }
}

@compute @workgroup_size(256)
fn apply(@builtin(workgroup_id) wg: vec3u, @builtin(num_workgroups) nwg: vec3u,
         @builtin(local_invocation_index) tid: u32) {
  let kw = K / 4u;
  let n = evn[0];
  let row = tid / LANES;
  let lane = tid % LANES;
  let r = (wg.y * nwg.x + wg.x) * ROWS + row;
  var acc = vec4<i32>(0);
  if (r < NP) {
    let base = r * kw;
    for (var e = lane; e < n; e += LANES) {
      let packed = ev[e];
      let col = packed >> 8u;
      let q = (bitcast<i32>(packed << 24u)) >> 24u;
      let word = col >> 2u;
      let t = col & 3u;
      let codes = w[base + word] >> (8u * t);
      let d0 = i32(codes & 3u) - CENTER;
      let d1 = i32((codes >> 2u) & 3u) - CENTER;
      let d2 = i32((codes >> 4u) & 3u) - CENTER;
      let d3 = i32((codes >> 6u) & 3u) - CENTER;
      acc += vec4<i32>(q * d0, q * d1, q * d2, q * d3);
    }
  }
  red[tid] = acc;
  workgroupBarrier();
  for (var s = LANES / 2u; s > 0u; s >>= 1u) {
    if (lane < s) {
      red[tid] += red[tid + s];
    }
    workgroupBarrier();
  }
  if (lane == 0u && r < NP) {
    let v = vec4<f32>(red[tid]) * (xs[0] * ws[0]);
    let i0 = r;
    let i1 = NP + r;
    let i2 = 2u * NP + r;
    let i3 = 3u * NP + r;
    if (ACCUM) {
      o[i0] = o[i0] + v[0];
      o[i1] = o[i1] + v[1];
      o[i2] = o[i2] + v[2];
      o[i3] = o[i3] + v[3];
    } else {
      o[i0] = v[0];
      o[i1] = v[1];
      o[i2] = v[2];
      o[i3] = v[3];
    }
  }
}
