// State-health scan: how many position/velocity components of the live GPU state are NOT finite.
//
// WHY THIS EXISTS AS ITS OWN FILE AND ITS OWN KERNEL. Three separate runs of this project
// (rho_tot = 0.8444 box 30, 0.75 box 30, 0.80 box 54) produced tens to hundreds of thousands of
// non-finite coordinates and NONE of them threw: the campaign printed a normal progress line
// (`stage=monomers`, an aggregate count, a step time) and the divergence was found only afterwards,
// by an offline scan of a checkpoint file. The mechanism is not that nobody looked -- it is that the
// guard that DID look is blind to NaN by construction: soup/src/soup-grid-verlet.ts's
// assertVerletSafety compares `sqrt(maxDriftSq) > skin/2`, and EVERY comparison against NaN is
// false in both IEEE-754 and JS, so once positions go non-finite the drift itself reads NaN and the
// check silently passes forever. A blown-up run is therefore indistinguishable from a clean
// negative ("stage=monomers, 0 amphiphiles") -- the single most expensive failure class in this
// project's history.
//
// Concatenated (soup/src/soup-pipelines.ts, string concat -- WGSL has no #include) AFTER
// engine/wgsl/forces.wgsl and soup/wgsl/step.wgsl, so it declares no buffer of its own except the
// two-slot counter below: `pos2` is forces.wgsl's own read-only position view at @group(1)
// @binding(0) and `velRW` is step.wgsl's own at @binding(10), both reused verbatim. soup/wgsl/
// step.wgsl was at 596 lines against CLAUDE.md's hard 600 limit, so this is a new file by the
// project's own "split first, then add" rule, not a choice of taste.
//
// WHY THE BIT PATTERN AND NOT `x != x`. The NaN self-inequality trick is the usual one and it is the
// one a fast-math-style optimisation is allowed to fold to `false`; the IEEE-754 exponent field is
// not an optimisation target. All-ones exponent means Inf (zero mantissa) or NaN (nonzero mantissa),
// and this scan wants BOTH -- an infinite coordinate is exactly as dead as a NaN one, and in every
// real blow-up here the infinity comes first. The sign bit lands at bit 8 after the shift and is
// masked off by 0xFF, so the test is sign-agnostic.

@group(1) @binding(22) var<storage, read_write> healthRW: array<atomic<u32>>;

fn soupNonFinite(x: f32) -> bool {
  return ((bitcast<u32>(x) >> 23u) & 0xFFu) == 0xFFu;
}

@compute @workgroup_size(1)
fn soup_reset_nonfinite_main() {
  atomicStore(&healthRW[0], 0u);
  atomicStore(&healthRW[1], 0u);
}

// COST, and why the counters are per-particle-batched rather than per-component atomics: on a
// healthy system this kernel performs three exponent tests per coordinate and ZERO atomic
// operations (the atomicAdd is inside `if (bad != 0u)`), i.e. it is a pure O(N) streaming read of
// the two arrays kick_drift_wrap_main already writes every step -- the same shape and the same order
// of cost as soup_max_drift_main, which this project already pays EVERY step. This one is dispatched
// once per step()-chunk instead (soup/src/soup-integrate.ts's STEP_CHUNK = 1000 steps), so its
// amortised cost is ~1/1000 of that; the measured number is in the task report, not guessed here.
//
// The counters count COMPONENTS, not particles, deliberately: that is the unit the three historical
// offline scans reported (60 495 / 366 282 / 68 049), so the guard's own message is directly
// comparable with the numbers already published rather than a different quantity with the same name.
@compute @workgroup_size(64)
fn soup_scan_nonfinite_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos2)) { return; }
  let x = pos2[i];
  var badPos = 0u;
  if (soupNonFinite(x.x)) { badPos = badPos + 1u; }
  if (soupNonFinite(x.y)) { badPos = badPos + 1u; }
  if (soupNonFinite(x.z)) { badPos = badPos + 1u; }
  if (badPos != 0u) { atomicAdd(&healthRW[0], badPos); }
  let v = velRW[i];
  var badVel = 0u;
  if (soupNonFinite(v.x)) { badVel = badVel + 1u; }
  if (soupNonFinite(v.y)) { badVel = badVel + 1u; }
  if (soupNonFinite(v.z)) { badVel = badVel + 1u; }
  if (badVel != 0u) { atomicAdd(&healthRW[1], badVel); }
}
