// Velocity-Verlet (kick-drift-kick) Langevin integrator, split into four small kernels chained
// by engine/src/sim.ts:
//   kick_main  (half-step: v += 0.5*dt*F, called twice per step, once with the old force and
//               once with the force recomputed at the new positions)
//   drift_main (full-step: x += dt*v)
//   wrap_main  (periodic wrap in x,y only; z is left open)
//   thermostat_main (Langevin friction + noise, applied once per full step)
//
// Splitting it this way (rather than a single semi-implicit-Euler update) means that with
// gamma = 0 the thermostat kernel's own effect vanishes (drag term and noise term both zero) and
// what runs is exactly symplectic velocity Verlet on the conservative force — the scheme the
// energy-drift test needs to stay bounded over thousands of steps.
//
// The RNG state is a per-particle buffer (`rng`), not a seed passed in fresh each dispatch: this
// lets sim.ts encode an entire step(n) call as one command buffer with no per-step CPU round
// trip (there is nothing left for the CPU to update between steps).
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
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> force: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> rng: array<u32>;
@group(0) @binding(5) var<uniform> B: vec4<f32>; // box lengths, xyz; periodic in x,y only

fn pcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn uniform01(v: u32) -> f32 { return f32(v) / 4294967296.0; }

@compute @workgroup_size(64)
fn kick_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&vel)) { return; }
  vel[i] = vec4<f32>(vel[i].xyz + 0.5 * P.dt * force[i].xyz, 0.0);
}

@compute @workgroup_size(64)
fn drift_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  pos[i] = vec4<f32>(pos[i].xyz + P.dt * vel[i].xyz, pos[i].w);
}

@compute @workgroup_size(64)
fn wrap_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  var x = pos[i].xyz;
  x.x = x.x - floor(x.x / B.x) * B.x;
  x.y = x.y - floor(x.y / B.y) * B.y;
  pos[i] = vec4<f32>(x, pos[i].w);
}

@compute @workgroup_size(64)
fn thermostat_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&vel)) { return; }
  var s = rng[i];
  s = pcg(s);
  let u1x = max(uniform01(s), 1e-7);
  s = pcg(s);
  let u2x = uniform01(s);
  s = pcg(s);
  let u1y = max(uniform01(s), 1e-7);
  s = pcg(s);
  let u2y = uniform01(s);
  s = pcg(s);
  let u1z = max(uniform01(s), 1e-7);
  s = pcg(s);
  let u2z = uniform01(s);
  rng[i] = s;
  let noise = vec3<f32>(
    sqrt(-2.0 * log(u1x)) * cos(6.28318531 * u2x),
    sqrt(-2.0 * log(u1y)) * cos(6.28318531 * u2y),
    sqrt(-2.0 * log(u1z)) * cos(6.28318531 * u2z),
  );
  let sigma_v = sqrt(2.0 * P.gamma * P.kT * P.dt);
  var v = vel[i].xyz;
  v = v - P.gamma * P.dt * v + sigma_v * noise;
  vel[i] = vec4<f32>(v, 0.0);
}
