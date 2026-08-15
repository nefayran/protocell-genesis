struct Params {
  sigma: f32, epsilon: f32,
  b_hh: f32, b_ht: f32, b_tt: f32,
  k_fene: f32, r_inf: f32,
  k_bend: f32, r_bend: f32,
  wc: f32,
  kT: f32, gamma: f32,
  dt: f32,
  pad0: f32, pad1: f32, pad2: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> probe: vec4<f32>; // x = kind, y = b

fn wca_cut(b: f32) -> f32 { return pow(2.0, 0.16666667) * b; }

fn wca_dv(r: f32, b: f32) -> f32 {
  if (r >= wca_cut(b) || r <= 0.0) { return 0.0; }
  let s6 = pow(b / r, 6.0);
  return -24.0 * P.epsilon / r * (2.0 * s6 * s6 - s6);
}

fn fene_dv(r: f32) -> f32 {
  let x = r / P.r_inf;
  return P.k_fene * r / (1.0 - x * x);
}

fn bend_dv(r: f32) -> f32 { return P.k_bend * (r - P.r_bend); }

fn attr_dv(r: f32) -> f32 {
  let rc = wca_cut(P.b_tt);
  if (r < rc || r > rc + P.wc) { return 0.0; }
  let x = 3.14159265 * (r - rc) / (2.0 * P.wc);
  return P.epsilon * 3.14159265 * sin(2.0 * x) / (2.0 * P.wc);
}

@compute @workgroup_size(64)
fn probe_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&radii)) { return; }
  let r = radii[i];
  let kind = u32(probe.x);
  var v = 0.0;
  if (kind == 0u) { v = wca_dv(r, probe.y); }
  else if (kind == 1u) { v = fene_dv(r); }
  else if (kind == 2u) { v = bend_dv(r); }
  else { v = attr_dv(r); }
  out[i] = v;
}
