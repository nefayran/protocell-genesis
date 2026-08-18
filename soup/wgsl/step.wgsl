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
// Species asymmetry (the amphiphile-emergence requirement, REVISED by task 'explicit-water',
// 2026-08-18): attr_dv is added between a pair when shouldAttract(ti,tj) below says so --
// water-water or water-head, from each species' own `polar`/`solvent` flags (SP.polar/SP.solvent,
// data/soup.json's Monomer fields) -- not, as before this task, "neither particle is polar". The
// old rule made hydrophobicity a hand-written tail-tail (and, incidentally, any-nonpolar-pair)
// attraction; this one makes it EMERGENT from water excluding tails (see data/soup.json's
// solvent.basis for the full argument and why the old term was removed outright rather than kept
// alongside the new one). Heads still get no attraction from a tail or from another head -- that
// part of the old asymmetry is unchanged. Nothing here special-cases "amphiphile"; a chain that
// happens to end in a polar head only behaves like one once Task 3 goes looking for the pattern.
//
// Periodicity: engine/wgsl/integrate.wgsl's wrap_main and forces.wgsl's mi() both wrap x,y only and
// leave z open, which is correct for a membrane sitting in vacuum but not for a bulk soup with no
// preferred axis -- so this file adds its own mi3() (wraps all three axes) and soup_wrap_main
// (wraps all three axes) rather than reusing those two. Everything else (kick_main, drift_main,
// thermostat_main from integrate.wgsl) is direction-agnostic and IS reused unchanged.

const SOUP_NONE: u32 = 0xFFFFFFFFu;

@group(1) @binding(6) var<storage, read_write> posRW: array<vec4<f32>>;
@group(1) @binding(7) var<storage, read> bondSlotsRO: array<u32>;

// Surface growth / adsorption (task 'adsorption', 2026-08-17, adsorption-report.md): read-only view
// of soup/wgsl/bond.wgsl's centerLink -- soup/src/sim.ts binds the SAME physical buffer into this
// module's own bind groups at this same binding too, so bondedForce below can read which pair (if
// any) the adsorption tether should pull together. Never written from this module: bond.wgsl's own
// atomicCompareExchangeWeak discipline (centerCas) is the only writer, exactly as bondSlotsRO above
// is read-only here while bond.wgsl (via bondSlots) owns every write to IT.
@group(1) @binding(20) var<storage, read> centerLinkRO: array<u32>;

// perf2-report.md, candidate (b): cell-sorted gather of `pos2` (forces.wgsl), rebuilt every grid
// rebuild by soup_gather_sorted_main below, using the SAME permutation `cellIdx` (fill_main,
// engine/wgsl/neighbor.wgsl) already produces. soup_force_main's O(candidates) neighbour walk
// reads posSortedRW[k] directly (k = the walk's own cell-contiguous cursor) instead of
// pos2[cellIdx[k]] (a scattered read at an arbitrary original index) -- read_write because this
// same declaration is both the gather's write target and the force kernel's read source within
// one shader module.
@group(1) @binding(13) var<storage, read_write> posSortedRW: array<vec4<f32>>;

// Task 'explicit-water' (2026-08-18): widened from a single vec4 per field (4 species max) to two
// vec4 slots per field (8 species max) so a 5th species (water, data/soup.json's "W") fits without
// a new binding or a new buffer -- soup/src/sim.ts's packSpeciesSlots() writes 8 floats per field
// regardless of how many monomers data/soup.json actually declares (unused slots are 0), and
// speciesRadius/speciesPolar/speciesSolvent below index by kind/4u (which vec4) and kind%4u (which
// component), a direct generalisation of the old kind==0u/1u/2u/else branches rather than a new
// mechanism. `solvent` is new: the flag data/soup.json's Monomer.solvent uploads, read by
// shouldAttract() below.
struct Species { radius: array<vec4<f32>, 2>, polar: array<vec4<f32>, 2>, solvent: array<vec4<f32>, 2> };
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
  return SP.radius[k / 4u][k % 4u];
}

fn speciesPolar(kind: f32) -> bool {
  let k = u32(kind);
  return SP.polar[k / 4u][k % 4u] > 0.5;
}

// Task 'explicit-water': the solvent flag (data/soup.json's Monomer.solvent, true only for water).
fn speciesSolvent(kind: f32) -> bool {
  let k = u32(kind);
  return SP.solvent[k / 4u][k % 4u] > 0.5;
}

// Task 'explicit-water': water-water and water-head attract (hydrophilic association); water-tail
// does not (this IS the hydrophobic exclusion, now emergent rather than hand-written); head-head
// does not (unchanged from before -- two polar heads never attracted each other under the old
// !polar&&!polar rule either). The old blanket "both nonpolar" rule (tail-tail, but also
// catalyst/donor pairs, since M and H were nonpolar too) is GONE: it is not a special case carved
// out of this formula, it simply is not one of the three disjuncts below -- see
// data/soup.json's solvent.basis for why keeping it alongside water exclusion would double-count
// the same hydrophobic-effect physics.
fn shouldAttract(ti: f32, tj: f32) -> bool {
  let si = speciesSolvent(ti);
  let sj = speciesSolvent(tj);
  let pi = speciesPolar(ti);
  let pj = speciesPolar(tj);
  return (si && sj) || (si && pj) || (pi && sj);
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
  // Surface growth / adsorption (task 'adsorption', 2026-08-17, adsorption-report.md): the
  // adsorption bond itself, a REAL FENE tether (the SAME spring formula every covalent bond above
  // already uses) between whichever pair centerLink links this particle to -- a catalyst's own
  // held chain tip, or a carbon's own owning catalyst. The link is mutual (soup/wgsl/bond.wgsl's
  // centerCas discipline keeps both sides in agreement), so this fires independently from BOTH
  // sides, exactly like the chain loop above fires from each bonded partner's own perspective with
  // no double-counting. This is what physically holds a growing tip at the catalytic surface --
  // see this file's header and adsorption-report.md for the mechanism this replaces (a
  // bookkeeping-only link with nothing pulling the pair together, which measurably let thermal
  // diffusion separate them within a handful of steps and deadlock nearly every centre
  // permanently). No bend contribution here: this task asks only for the FENE tether, not a
  // three-body angle constraint on a pair that is not part of any real chain topology.
  let owner = centerLinkRO[i];
  if (owner != SOUP_NONE) {
    let d = mi3(xi - pos2[owner].xyz, box);
    let r = max(length(d), 1e-6);
    f = f - fene_dv(r) * d / r;
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
  if (shouldAttract(ti, tj)) {
    f = f - attr_dv(r) * d / r;
  }
  return f;
}

// perf2-report.md, candidate (b). Rebuilt every grid rebuild (soup/src/sim.ts's
// encodeGridRebuild), right after fill_main -- see that file's header for why this specific
// ordering matters (posSortedRW must be populated before soup_force_main/bond_form_main run, and
// AFTER cellIdx is final for this step).
@compute @workgroup_size(64)
fn soup_gather_sorted_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let k = gid.x;
  if (k >= arrayLength(&posSortedRW)) { return; }
  posSortedRW[k] = pos2[cellIdx[k]];
}

// perf2-report.md, candidates (a)+(b), factored so both can be A/B-measured independently on ONE
// implementation instead of two hand-maintained copies: `useSorted=true` reads posSortedRW[k]
// (candidate (b)), `useSorted=false` reads pos2[cellIdx[k]] (the pre-(b) scattered read) --
// candidate (a)'s walk radius (GB.dims.w) applies identically either way, since it is orthogonal
// to which position array backs the read. soup_force_main/soup_force_main_unsorted below are thin
// entry-point wrappers so JS picks the pipeline (soup/src/sim.ts reads
// neighborGrid.sortedGather), never a runtime branch inside the hot loop.
fn soupForceWalk(i: u32, useSorted: bool) -> vec3<f32> {
  let box = GB.box.xyz;
  let dims = vec3<i32>(GB.dims.xyz);
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;

  var f = bondedForce(i, xi, box);

  // perf2-report.md, candidate (a): walk radius comes from GB.dims.w (soup/src/sim.ts derives and
  // asserts it from neighborGrid.cellDivisor -- data/soup.json), not a hardcoded ±1. cellDivisor=1
  // reproduces walkRadius=1, the original 3x3x3 walk, bit-identical.
  let R = i32(GB.dims.w);
  let c = cell_coord(xi, GB.dims.xyz, box);
  for (var dz = -R; dz <= R; dz = dz + 1) {
    let cz = wrap_axis(c.z + dz, dims.z);
    for (var dy = -R; dy <= R; dy = dy + 1) {
      let cy = wrap_axis(c.y + dy, dims.y);
      for (var dx = -R; dx <= R; dx = dx + 1) {
        let cx = wrap_axis(c.x + dx, dims.x);
        let nc = u32(cx) + GB.dims.x * (u32(cy) + GB.dims.y * u32(cz));
        let start = cellStart[nc];
        let end = cellStart[nc + 1u];
        for (var k = start; k < end; k = k + 1u) {
          let j = cellIdx[k];
          if (j == i) { continue; }
          var xj: vec3<f32>;
          var tj: f32;
          if (useSorted) {
            // candidate (b): cell-contiguous read, no indirection through `j`.
            xj = posSortedRW[k].xyz;
            tj = posSortedRW[k].w;
          } else {
            // pre-(b): scattered read at an arbitrary original index.
            xj = pos2[j].xyz;
            tj = pos2[j].w;
          }
          f = f + nonbondedSoup(xi, xj, ti, tj, box);
        }
      }
    }
  }
  return f;
}

@compute @workgroup_size(64)
fn soup_force_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos2)) { return; }
  outForce[i] = vec4<f32>(soupForceWalk(i, true), 0.0);
}

@compute @workgroup_size(64)
fn soup_force_main_unsorted(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos2)) { return; }
  outForce[i] = vec4<f32>(soupForceWalk(i, false), 0.0);
}

// perf2-report.md correctness gate: same physics as soup_force_main, without the neighbour grid --
// an O(N^2) pair loop used only to cross-check the grid result, mirroring engine/wgsl/forces.wgsl's
// own force_main/force_brute_main pattern (that file's header: "Same physics as force_main, without
// the neighbor grid"). Reads pos2 directly (not posSortedRW) -- correctness reference, not a
// performance path, so it has no reason to depend on candidate (b)'s gather at all.
@compute @workgroup_size(64)
fn soup_force_brute_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;
  var f = bondedForce(i, xi, box);
  for (var j = 0u; j < n; j = j + 1u) {
    if (j == i) { continue; }
    f = f + nonbondedSoup(xi, pos2[j].xyz, ti, pos2[j].w, box);
  }
  outForce[i] = vec4<f32>(f, 0.0);
}

// perf2-report.md, STEP 1 diagnosis. Runs the SAME cell walk soup_force_main runs (identical
// radius, identical cell coordinates), but instead of accumulating force, counts every candidate
// examined and, of those, how many actually lie within the interaction range that
// nonbondedSoup would apply a nonzero force over (WCA core OR, for a non-polar pair, the
// tail-tail attraction band) -- mirroring nonbondedSoup's own gates exactly rather than
// re-deriving an approximate geometric threshold that could silently drift from the real physics.
// Never called from the real step loop -- see forceCandidateStatsDEBUG in soup/src/sim.ts, the
// only caller. Counts are double (each unordered pair counted once from i's walk and once from
// j's), matching soup_force_main's own real per-step workload; the RATIO reported is unaffected by
// that doubling since numerator and denominator both carry it.
@group(3) @binding(0) var<storage, read_write> statsOut: array<atomic<u32>>;

@compute @workgroup_size(64)
fn soup_force_stats_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let dims = vec3<i32>(GB.dims.xyz);
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;
  let R = i32(GB.dims.w);
  let c = cell_coord(xi, GB.dims.xyz, box);
  let rcAttr = wca_cut(P.b_tt);
  for (var dz = -R; dz <= R; dz = dz + 1) {
    let cz = wrap_axis(c.z + dz, dims.z);
    for (var dy = -R; dy <= R; dy = dy + 1) {
      let cy = wrap_axis(c.y + dy, dims.y);
      for (var dx = -R; dx <= R; dx = dx + 1) {
        let cx = wrap_axis(c.x + dx, dims.x);
        let nc = u32(cx) + GB.dims.x * (u32(cy) + GB.dims.y * u32(cz));
        let start = cellStart[nc];
        let end = cellStart[nc + 1u];
        for (var k = start; k < end; k = k + 1u) {
          let j = cellIdx[k];
          if (j == i) { continue; }
          atomicAdd(&statsOut[0], 1u);
          let tj = pos2[j].w;
          let d = mi3(xi - pos2[j].xyz, box);
          let r = length(d);
          let b = pairB(ti, tj);
          let withinWca = r < wca_cut(b);
          let withinAttr = shouldAttract(ti, tj) && r >= rcAttr && r <= rcAttr + P.wc;
          if (withinWca || withinAttr) {
            atomicAdd(&statsOut[1], 1u);
          }
        }
      }
    }
  }
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

// perf2-report.md, candidate (c): a Verlet neighbour list. Candidate (a) (smaller cells, wider
// walk) was measured a NET LOSS -- ncells grew with the cell-shrink and the grid rebuild's own
// serial prefix-sum cost grew right along with it, cancelling the candidate-count savings almost
// exactly (perf2-report.md has the phase breakdown). Candidate (b) (sorted gather) was measured
// negligible. Both leave the ±walkRadius cell walk running EVERY step, which the STEP 1 diagnosis
// showed visits ~11x more candidates than real neighbours -- still the dominant cost. A Verlet
// list amortises that walk over `verletList.rebuildEvery` real steps instead of paying it every
// step: soup/src/sim.ts derives listRange = interactionRange + skin and the walk radius needed to
// cover it at the ORIGINAL (undivided) cell size, and asserts both the geometric coverage and (at
// system-creation time, from THIS system's own kT) the drift-safety condition
// 2*rebuildEvery*dt*vBound <= skin before ever running -- the same worst-case-outlier-speed
// argument perf-report.md's rejected candidate (a) already used, just to size a list instead of a
// per-step cell size. The empirical guard soup_max_drift_main below backs that analytical bound up
// with a REAL measurement every step, checked by soup/src/sim.ts after every step() call.

// VL: x = listRange (interactionRange + skin, the geometric superset radius the list is built to,
// always >= the radius soupForceWalk/bondFormWalk would need), y = list capacity per particle (as
// f32, cast to u32 here) -- both from data/soup.json's verletList config via soup/src/sim.ts, never
// a WGSL literal.
@group(1) @binding(19) var<uniform> VL: vec4<f32>;
@group(1) @binding(14) var<storage, read_write> verletList: array<u32>;
@group(1) @binding(15) var<storage, read_write> verletCount: array<u32>;
@group(1) @binding(16) var<storage, read_write> verletOverflow: array<atomic<u32>>;
@group(1) @binding(17) var<storage, read_write> posAtRebuildRW: array<vec4<f32>>;
@group(1) @binding(18) var<storage, read_write> maxDriftSqRW: array<atomic<u32>>;

// Walks ±walkRadius cells (GB.dims.w, sized by soup/src/sim.ts to cover VL.x = interactionRange +
// skin at the grid's OWN cell size -- not shrunk the way rejected candidate (a) shrunk it, so this
// walk's own ncells stays at the cheap, unshrunk count) ONCE per rebuild and records every
// candidate within VL.x, not just the ones within the raw interaction range -- the skin margin is
// exactly what lets soup_force_list_main/bond_form_list_main below trust the list for
// `rebuildEvery` steps without re-walking. Overflow (more geometric candidates than VL.y allows)
// sets a flag rather than silently truncating -- soup/src/sim.ts asserts that flag is clear after
// every rebuild; it must NEVER silently drop a real neighbour the way a completeness violation
// would.
@compute @workgroup_size(64)
fn soup_build_verlet_list_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let dims = vec3<i32>(GB.dims.xyz);
  let xi = pos2[i].xyz;
  let listRange = VL.x;
  let cap = u32(VL.y);
  let R = i32(GB.dims.w);
  let c = cell_coord(xi, GB.dims.xyz, box);
  var count = 0u;
  for (var dz = -R; dz <= R; dz = dz + 1) {
    let cz = wrap_axis(c.z + dz, dims.z);
    for (var dy = -R; dy <= R; dy = dy + 1) {
      let cy = wrap_axis(c.y + dy, dims.y);
      for (var dx = -R; dx <= R; dx = dx + 1) {
        let cx = wrap_axis(c.x + dx, dims.x);
        let nc = u32(cx) + GB.dims.x * (u32(cy) + GB.dims.y * u32(cz));
        let start = cellStart[nc];
        let end = cellStart[nc + 1u];
        for (var k = start; k < end; k = k + 1u) {
          let j = cellIdx[k];
          if (j == i) { continue; }
          let d = mi3(xi - pos2[j].xyz, box);
          if (length(d) < listRange) {
            if (count < cap) {
              verletList[i * cap + count] = j;
              count = count + 1u;
            } else {
              atomicStore(&verletOverflow[0], 1u);
            }
          }
        }
      }
    }
  }
  verletCount[i] = count;
}

@compute @workgroup_size(64)
fn soup_snapshot_positions_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&posAtRebuildRW)) { return; }
  posAtRebuildRW[i] = pos2[i];
}

@compute @workgroup_size(1)
fn soup_reset_max_drift_main() {
  atomicStore(&maxDriftSqRW[0], 0u);
}

// Per-step, O(N) (cheap -- one minimum-image distance and one atomic max per particle, the same
// order of cost as kick_drift_wrap_main, not the O(candidates) walk this whole candidate exists to
// avoid): tracks how far each particle has moved since the snapshot soup_snapshot_positions_main
// took at the last rebuild. atomicMax on the BIT PATTERN of a non-negative f32 (driftSq >= 0
// always) is safe because IEEE-754's bit pattern is monotonically increasing over non-negative
// floats -- a standard trick for float atomicMax where the hardware/WGSL has no native
// atomic<f32>. soup/src/sim.ts reads this back (via bitcast<f32>, done on the JS side by
// reinterpreting the u32) once per step() call and asserts sqrt(driftSq) <= skin/2, the exact
// guarantee this candidate's header promises.
@compute @workgroup_size(64)
fn soup_max_drift_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos2)) { return; }
  let box = GB.box.xyz;
  let d = mi3(pos2[i].xyz - posAtRebuildRW[i].xyz, box);
  let driftSq = dot(d, d);
  atomicMax(&maxDriftSqRW[0], bitcast<u32>(driftSq));
}

// Per-step force from the Verlet list instead of any cell walk: O(realNeighbours), not
// O(candidates). Same bonded contribution (bondedForce, untouched) plus nonbondedSoup summed over
// exactly the particles soup_build_verlet_list_main found within VL.x of i at the last rebuild --
// a strict SUPERSET of the ones within the real interaction range (the skin margin), so this
// never drops a force contribution; it may include a few now-just-outside-range candidates for
// which nonbondedSoup itself correctly evaluates to zero (its own r<wca_cut/attraction gates,
// unchanged), which is exactly the point of a skin.
@compute @workgroup_size(64)
fn soup_force_list_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;
  var f = bondedForce(i, xi, box);
  let cap = u32(VL.y);
  let count = verletCount[i];
  for (var s = 0u; s < count; s = s + 1u) {
    let j = verletList[i * cap + s];
    f = f + nonbondedSoup(xi, pos2[j].xyz, ti, pos2[j].w, box);
  }
  outForce[i] = vec4<f32>(f, 0.0);
}
