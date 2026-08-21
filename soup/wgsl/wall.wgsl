// Task 'confined-parcel' (2026-08-21): the confining wall's force, as its OWN dispatch that
// ACCUMULATES into outForce.
//
// WHY A SEPARATE KERNEL AND NOT A TERM INSIDE nonbondedSoup/soupForceWalk. There are four force
// paths in this engine (soup_force_main, soup_force_main_unsorted, soup_force_list_main,
// soup_force_brute_main) and eight call sites that produce a force (step, box scale, evaporation,
// rehydration, cold-start relax, protonation recompute, two readbacks). A one-body term added inside
// three of the four kernels is a silent physics bug -- exactly the reasoning
// soup/src/soup-integrate.ts's own encodeEsFar comment already gives for the long-range
// electrostatic half, and exactly the trap that would make the brute-force reference disagree with
// the grid one for a reason that has nothing to do with the neighbour list. This kernel is
// dispatched from INSIDE the three encodeSoupForce* functions, so every path and every call site
// gets it, once, by construction.
//
// It accumulates (`+`), so it must follow the kernel that WROTE outForce, in the same compute pass:
// WebGPU orders dispatches within a pass and inserts the barrier between them, which is what the
// existing grid-rebuild-then-force-then-esFar sequence already relies on.
//
// Concatenated (soup/src/soup-pipelines.ts) AFTER step.wgsl: it reads pos2/outForce/GB, all declared
// by engine/wgsl/forces.wgsl, and WGSL has no forward declarations. Declares exactly one new binding
// (WALL) and no potential formula of its own beyond the harmonic penetration term below, whose CPU
// twin (soup/src/soup-confine.ts's wallForceAt) is what tests/soup-confine.test.ts differentiates
// numerically.
//
// The wall is NEUTRAL: this kernel never reads the species slot pos2[i].w, so the same repulsion
// applies to water, heads, tails, catalyst and mineral alike -- no wetting, no nucleation, no
// per-species offset. See soup/src/soup-confine.ts's header for what that choice does and does not
// bias.
//
// Frozen particles (soup/wgsl/step.wgsl's frozenRO) are deliberately NOT special-cased: their force
// is computed and then never consumed by either integrator kernel, exactly as every pair force on
// them already is, and confinement refuses to run with a mineral platelet at all
// (soup/src/sim.ts), so the case cannot arise in a real run.

// WALL: x = the LIVE parcel radius in sigma (0 = no confinement, kernel returns immediately),
// y = the wall's spring constant in epsilon/sigma^2; z,w unused. Written by
// soup/src/soup-buffers.ts from soup/src/soup-confine.ts's wallUniformBytes -- never a WGSL literal,
// and rewritten by the SAME resizeSoupGrid call that rewrites GB, so the radius and the box the
// centre is derived from can never disagree.
@group(1) @binding(31) var<uniform> WALL: vec4<f32>;

@compute @workgroup_size(64)
fn soup_wall_force_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos2)) { return; }
  if (!(WALL.x > 0.0)) { return; }
  // The parcel is centred in the box, always -- soup/src/soup-confine.ts's parcelCentre, and the
  // reason the uniform carries no centre of its own.
  let c = GB.box.xyz * 0.5;
  let d = pos2[i].xyz - c;
  let s = length(d);
  if (s <= WALL.x) { return; }
  // U(s) = 0.5*k*(s-R)^2 for s > R, so F = -dU/ds * (d/s) = -k*(s-R)*(d/s), pointing INWARD.
  let f = (-WALL.y * (s - WALL.x) / max(s, 1e-6)) * d;
  outForce[i] = outForce[i] + vec4<f32>(f, 0.0);
}
