// Cold-start energy minimisation: ONE displacement-capped steepest-descent iteration.
//
// WHAT IT FIXES. The fresh-creation layout (soup/src/soup-init-state.ts) is a cubic lattice with
// jitterFrac = 0.15, so the smallest gap it can produce is (1 - 2*0.15)*spacing = 0.7*spacing. At a
// total density of 0.8 sigma^-3 the spacing is 0.8^(-1/3) = 1.077 sigma, i.e. a minimum gap of
// 0.754 sigma -- BELOW the catalyst-catalyst WCA contact distance (sigma*(1.2+1.2)/2 = 1.2 sigma) and
// below the water-catalyst one. Those pairs start ~40% deep in a repulsive core, and 1/r^13 at that
// depth is a force no integrator at data/params.json's own dt can absorb: the project has measured this blow up at
// rho_tot = 0.844, 0.75 and 0.80 (silently, three times -- see soup/wgsl/health.wgsl's header) and
// at 0.933 loudly (a 24.58 sigma displacement in 10 steps). It is NOT a property of the physics: pure
// water at 0.8 sigma^-3 runs stably, because its own lattice is placed at exactly its own spacing.
// The cure is the standard MD one and it is older than this project: relax the coordinates to a local
// energy minimum BEFORE starting dynamics.
//
// WHY NORMALISED STEEPEST DESCENT WITH A DISPLACEMENT CAP, and not the two alternatives. A soft-core
// force cap ramped off over the first steps would have to live inside nonbondedSoup -- i.e. it would
// modify the potential the whole project's calibration rests on, and it would be ON during real
// steps, which this task's brief forbids for any step a measurement uses. A smaller initial dt ramped
// up would ALSO be inside the trajectory (the first thousands of steps would be integrated with a
// different dt than every published run) and would still be integrating an overlap whose force is
// ~1e4. Minimisation is outside the trajectory entirely: it runs before step 1, it touches positions
// only (velocities keep the exact Maxwell-Boltzmann draw createSoup made), it consumes no RNG stream,
// it advances no step counter, and it changes no potential, rate, threshold or constant. The
// potential it descends is nonbondedSoup+bondedForce -- literally the same force kernel the real
// steps use, called through the same encodeSoupForceList/encodeSoupForce path.
//
// WHY THE STEP IS NORMALISED (every particle moves exactly RX.x along its own force direction)
// rather than proportional to |F|. Proportional descent at an overlap of ~1e4 in force units would
// need a step size of ~1e-5 to be stable and then would take 1e5 iterations to travel one sigma:
// the very stiffness that is the problem sets the step size. A normalised step is scale-free -- the
// deeper the overlap, the more exactly the direction points out of it, and the displacement is
// bounded by construction, so no iteration can produce a coordinate jump larger than RX.x. It is the
// same reasoning GROMACS's `steep` integrator uses when it caps its own step, minus the adaptive
// step-size search (which needs a global energy reduction this engine has no kernel for). Settling
// is guaranteed instead by RX.x being decayed to zero by the caller (soup/src/soup-relax.ts) over the
// iteration count -- a normalised descent with a fixed step cannot converge, it orbits the minimum at
// radius ~RX.x, and the decay is what closes that orbit.
//
// Concatenated (soup/src/soup-pipelines.ts) AFTER soup/wgsl/health.wgsl, whose soupNonFinite() this
// file calls -- see the guard below. Declares exactly one new binding (RX) and reuses posRW/forceRO/
// frozenRO/GB verbatim from step.wgsl and forces.wgsl. A separate file rather than an addition to
// step.wgsl because step.wgsl stands at 596 lines against CLAUDE.md's hard 600 limit.

// RX: x = the maximum displacement (sigma) THIS iteration may apply to any particle; y,z,w unused.
// Written by soup/src/soup-relax.ts once per iteration -- never a WGSL literal, and derived there
// from data/soup.json's coldStartRelax section times data/params.json's rank-A sigma.
@group(1) @binding(23) var<uniform> RX: vec4<f32>;

@compute @workgroup_size(64)
fn soup_relax_step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&posRW)) { return; }
  // Task 'clay-surface': the platelet is immobile at every stage, minimisation included -- its
  // lattice IS the mineral phase's geometry, and relaxing it would be relaxing away the wall.
  if (frozenRO[i] != 0u) { return; }
  let f = forceRO[i].xyz;
  // Normalise in two stages, through the largest component, so that no intermediate can overflow:
  // dividing by `a` puts every component in [-1,1] whatever the magnitude of f, and length() of that
  // is in [1, sqrt(3)]. A single f/length(f) would produce Inf/Inf = NaN the moment |f| overflows
  // f32, which is exactly what a pair at r -> 0 delivers.
  let a = max(max(abs(f.x), abs(f.y)), abs(f.z));
  // Zero force (nothing to descend), or an already-non-finite force (a pair at exactly r = 0, which
  // no direction can resolve): leave the particle where it is. Leaving it is the honest outcome --
  // soup/src/soup-relax.ts re-scans for non-finite state after the last iteration and THROWS, so an
  // unrelaxable start fails loudly instead of being smeared into a plausible-looking one.
  if (!(a > 0.0) || soupNonFinite(a)) { return; }
  let dir = normalize(f / a);
  let box = GB.box.xyz;
  var x = posRW[i].xyz + dir * RX.x;
  x = x - floor(x / box) * box;
  posRW[i] = vec4<f32>(x, posRW[i].w);
}
