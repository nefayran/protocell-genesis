// Task 2: Langevin force kernel for the soup, plus a full 3-axis periodic wrap.
//
// Concatenated (by soup/src/sim.ts, string concat -- WGSL has no #include) AFTER engine/wgsl/
// forces.wgsl, so this file itself declares NO potential formula -- it reuses wca_dv/fene_dv/
// bend_dv/attr_dv/wca_cut and the Params uniform `P`/GridDims uniform `GB`/neighbor arrays
// `cellStart`/`cellIdx` exactly as forces.wgsl declares them, at the SAME @group(1) bindings 0,1,
// 3,4,5. It does not call forces.wgsl's own force_main/bonded_pair/nonbonded: those hardcode the
// fixed 3-bead-per-lipid topology (`i % 3u`), and the soup's topology is dynamic -- carbons grow
// into chains of unknown length at runtime, tracked by `bondSlots` (written by bond.wgsl) rather
// than by bead index arithmetic.
//
// Species asymmetry (the amphiphile-emergence requirement): attr_dv is added between a pair ONLY
// when NEITHER particle is polar (SP.polar, from data/soup.json's `polar` field per monomer) --
// polar heads get wca_dv (repulsion) alone, from every partner, never attr_dv. Nothing here
// special-cases "amphiphile"; a chain that happens to end in a polar head only behaves like one
// once Task 3 goes looking for the pattern.
//
// Periodicity: engine/wgsl/integrate.wgsl's wrap_main and forces.wgsl's mi() both wrap x,y only and
// leave z open, which is correct for a membrane sitting in vacuum but not for a bulk soup with no
// preferred axis -- so this file adds its own mi3() (wraps all three axes) and soup_wrap_main
// (wraps all three axes) rather than reusing those two. Everything else (kick_main, drift_main,
// thermostat_main from integrate.wgsl) is direction-agnostic and IS reused unchanged.

const SOUP_NONE: u32 = 0xFFFFFFFFu;

@group(1) @binding(6) var<storage, read_write> posRW: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read> bondSlotsRO: array<u32>;

struct Species { radius: vec4<f32>, polar: vec4<f32> };
@group(1) @binding(8) var<uniform> SP: Species;

// Fused integrator steps -- pure arithmetic glue around kick_main/drift_main/thermostat_main's
// OWN formulas (engine/wgsl/integrate.wgsl, reused verbatim there for the membrane engine), not a
// new physical model: kick_drift_wrap_main is exactly kick_main+drift_main+one 3-axis wrap done as
// one dispatch instead of three, and kick_thermostat_main is exactly kick_main+thermostat_main as
// one dispatch instead of two. Folded in because interleaving many SEPARATE small dispatches
// measurably dominated this kernel's wall time far more than the compute itself did (see
// task-2-report.md) -- soup/src/sim.ts's step() calls these instead of integrate.wgsl's kernels so
// the soup's 12 dispatches/step become 9, structurally the same 3-axis-periodic reasoning as
// soup_wrap_main above, just merged with the two Verlet half-kicks either side of it.
@group(1) @binding(10) var<storage, read_write> velRW: array<vec4<f32>>;
@group(1) @binding(11) var<storage, read> forceRO: array<vec4<f32>>;
@group(1) @binding(12) var<storage, read_write> intRng: array<u32>;

fn pcgSoup(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn uniform01Soup(v: u32) -> f32 { return f32(v) / 4294967296.0; }

@compute @workgroup_size(64)
fn kick_drift_wrap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&posRW)) { return; }
  var v = velRW[i].xyz + 0.5 * P.dt * forceRO[i].xyz;
  velRW[i] = vec4<f32>(v, 0.0);
  let box = GB.box.xyz;
  var x = posRW[i].xyz + P.dt * v;
  x = x - floor(x / box) * box;
  posRW[i] = vec4<f32>(x, posRW[i].w);
}

@compute @workgroup_size(64)
fn kick_thermostat_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&velRW)) { return; }
  var v = velRW[i].xyz + 0.5 * P.dt * forceRO[i].xyz;
  var s = intRng[i];
  s = pcgSoup(s);
  let u1x = max(uniform01Soup(s), 1e-7);
  s = pcgSoup(s);
  let u2x = uniform01Soup(s);
  s = pcgSoup(s);
  let u1y = max(uniform01Soup(s), 1e-7);
  s = pcgSoup(s);
  let u2y = uniform01Soup(s);
  s = pcgSoup(s);
  let u1z = max(uniform01Soup(s), 1e-7);
  s = pcgSoup(s);
  let u2z = uniform01Soup(s);
  intRng[i] = s;
  let noise = vec3<f32>(
    sqrt(-2.0 * log(u1x)) * cos(6.28318531 * u2x),
    sqrt(-2.0 * log(u1y)) * cos(6.28318531 * u2y),
    sqrt(-2.0 * log(u1z)) * cos(6.28318531 * u2z),
  );
  let sigma_v = sqrt(2.0 * P.gamma * P.kT * P.dt);
  v = v - P.gamma * P.dt * v + sigma_v * noise;
  velRW[i] = vec4<f32>(v, 0.0);
}

fn mi3(d_in: vec3<f32>, box: vec3<f32>) -> vec3<f32> {
  return d_in - round(d_in / box) * box;
}

fn speciesRadius(kind: f32) -> f32 {
  let k = u32(kind);
  if (k == 0u) { return SP.radius.x; }
  else if (k == 1u) { return SP.radius.y; }
  else if (k == 2u) { return SP.radius.z; }
  else { return SP.radius.w; }
}

fn speciesPolar(kind: f32) -> bool {
  let k = u32(kind);
  if (k == 0u) { return SP.polar.x > 0.5; }
  else if (k == 1u) { return SP.polar.y > 0.5; }
  else if (k == 2u) { return SP.polar.z > 0.5; }
  else { return SP.polar.w > 0.5; }
}

// Lorentz-Berthelot-style arithmetic mean, same mixing convention data/params.json's own
// beadSizes already uses implicitly (its three fixed head/tail pairs are exactly this formula
// evaluated on two fixed radii) -- generalised here to whichever two of the soup's species are in
// contact, from their OWN radiusSigma in data/soup.json rather than a fixed lipid pair.
fn pairB(ti: f32, tj: f32) -> f32 { return P.sigma * (speciesRadius(ti) + speciesRadius(tj)) * 0.5; }

// FENE for every occupied bond slot (chain-chain and chain-head bonds use the same spring; only
// the Metropolis energy that governs whether the bond EXISTS differs between them, carried in
// bond.wgsl's BondParams, not here) plus bend for every pair of neighbours that share this
// particle as their common bonded centre -- the generalisation of forces.wgsl's fixed
// head/tail1/tail2 bend triple to a chain of unknown length: a particle with exactly two occupied
// slots is a "middle" the same way tail1 is in the fixed-topology case, and the bend force is
// credited to the two OUTER particles (mirroring forces.wgsl's bonded_pair, which never applies a
// bend force to the middle bead itself), never to this particle acting as the centre.
fn bondedForce(i: u32, xi: vec3<f32>, box: vec3<f32>) -> vec3<f32> {
  var f = vec3<f32>(0.0);
  let base = i * 3u;
  for (var s = 0u; s < 3u; s = s + 1u) {
    let partner = bondSlotsRO[base + s];
    if (partner == SOUP_NONE) { continue; }
    let d = mi3(xi - pos2[partner].xyz, box);
    let r = max(length(d), 1e-6);
    f = f - fene_dv(r) * d / r;
  }
  for (var s = 0u; s < 3u; s = s + 1u) {
    let m = bondSlotsRO[base + s];
    if (m == SOUP_NONE) { continue; }
    let mbase = m * 3u;
    for (var t = 0u; t < 3u; t = t + 1u) {
      let other = bondSlotsRO[mbase + t];
      if (other == SOUP_NONE || other == i) { continue; }
      let d = mi3(xi - pos2[other].xyz, box);
      let r = max(length(d), 1e-6);
      f = f - bend_dv(r) * d / r;
    }
  }
  return f;
}

fn nonbondedSoup(xi: vec3<f32>, xj: vec3<f32>, ti: f32, tj: f32, box: vec3<f32>) -> vec3<f32> {
  var f = vec3<f32>(0.0);
  let d = mi3(xi - xj, box);
  let r = length(d);
  if (r < 1e-6) { return f; }
  let b = pairB(ti, tj);
  if (r < wca_cut(b)) {
    f = f - wca_dv(r, b) * d / r;
  }
  if (!speciesPolar(ti) && !speciesPolar(tj)) {
    f = f - attr_dv(r) * d / r;
  }
  return f;
}

@compute @workgroup_size(64)
fn soup_force_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let dims = vec3<i32>(GB.dims.xyz);
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;

  var f = bondedForce(i, xi, box);

  let c = cell_coord(xi, GB.dims.xyz, box);
  for (var dz = -1; dz <= 1; dz = dz + 1) {
    let cz = wrap_axis(c.z + dz, dims.z);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      let cy = wrap_axis(c.y + dy, dims.y);
      for (var dx = -1; dx <= 1; dx = dx + 1) {
        let cx = wrap_axis(c.x + dx, dims.x);
        let nc = u32(cx) + GB.dims.x * (u32(cy) + GB.dims.y * u32(cz));
        let start = cellStart[nc];
        let end = cellStart[nc + 1u];
        for (var k = start; k < end; k = k + 1u) {
          let j = cellIdx[k];
          if (j == i) { continue; }
          f = f + nonbondedSoup(xi, pos2[j].xyz, ti, pos2[j].w, box);
        }
      }
    }
  }
  outForce[i] = vec4<f32>(f, 0.0);
}

@compute @workgroup_size(64)
fn soup_wrap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&posRW)) { return; }
  let box = GB.box.xyz;
  var x = posRW[i].xyz;
  x = x - floor(x / box) * box;
  posRW[i] = vec4<f32>(x, posRW[i].w);
}
