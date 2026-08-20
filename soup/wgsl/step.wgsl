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

// Water-calibration task (2026-08-19): x = solvent.attractionScale.epsilonScale (data/soup.json,
// rank D -- see that field's own basis) TIMES the per-class ratio table
// solvent.attractionScale.pairEpsilon builds (task 'hydrophobic-asymmetry', 2026-08-19). One row per
// species CLASS (0 = apolar, 1 = polar, 2 = solvent, 3 = mineral, exactly speciesClass() below), column = the
// other particle's class, 4th component unused; symmetric by construction on the JS side
// (soup/src/soup-attraction.ts, which is also what soup/src/soup-potential.ts reads so the CPU
// Metropolis energy and this kernel cannot drift apart). A cell of 0 means that pair simply does
// not attract -- the boolean shouldAttract() this replaced is now derived FROM the table, not
// alongside it. The depths themselves are ratios to the tail-tail pair, whose own cell is exactly
// epsilonScale, so attr_dv keeps its rank-A Cooke & Deserno absolute depth for tail-tail; nothing
// here redefines attr_dv's shape (P.epsilon/P.b_tt/P.wc, rank A, untouched).
// Task 'clay-surface' (2026-08-19): widened 3 -> 4 rows for the MINERAL class (the clay platelet's
// own row/column in data/soup.json's pairEpsilon). soup/src/soup-attraction.ts sizes the uniform
// from its own CLASS_COUNT, so the two sides cannot disagree about the row count without failing
// bind-group validation outright.
struct AttrScaleTable { rows: array<vec4<f32>, 4> };
@group(1) @binding(9) var<uniform> AttrScale: AttrScaleTable;

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
// Task 'clay-surface' (2026-08-19): `mineral` appended (data/soup.json's Monomer.mineral, true only
// for the clay bead). soup/src/soup-plan.ts's packSpeciesSlots writes 8 floats per field regardless
// of how many monomers exist, so this is one more 32-byte field, not a new mechanism. NOTE
// soup/wgsl/bond-common.wgsl declares its OWN, still-3-field Species against the SAME buffer and
// deliberately stays that way: a uniform binding only requires the buffer to be at least as large as
// the struct, and the bond kernels have no use for the mineral flag (no rule mentions the mineral
// species, so a clay bead can never be a bonding partner).
struct Species { radius: array<vec4<f32>, 2>, polar: array<vec4<f32>, 2>, solvent: array<vec4<f32>, 2>, mineral: array<vec4<f32>, 2> };
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

// Task 'clay-surface' (2026-08-19): per-particle immobility. 1 = this particle belongs to the rigid
// mineral platelet (a clay bead, or a catalyst bead immobilised on the platelet as a surface site);
// 0 = ordinary particle. Written once at creation (soup/src/soup-init-state.ts from
// soup/src/soup-clay.ts's layout) and never again by any kernel.
//
// WHY IT IS CHECKED IN THE TWO INTEGRATOR KERNELS AND NOWHERE ELSE. Immobility here means "the
// position never changes", and the ONLY two kernels that write posRW/velRW during a step are the two
// below. Both return early for a frozen particle after zeroing its velocity, so:
//  - its coordinate is not merely damped, it is never assigned at all -- no force, no dt, no
//    thermostat kick can move it, which is what makes this structural rather than "a heavy mass"
//    (a heavy bead still drifts, and drifts more the longer the run);
//  - the force kernels are UNTOUCHED: a frozen bead's own force is still computed and simply never
//    consumed, so every OTHER particle feels the platelet exactly as it feels any other bead. The
//    honest consequence, stated rather than discovered: the reaction force on the platelet is
//    discarded, i.e. the mineral phase is a momentum sink -- which is what a rigid wall is.
@group(1) @binding(21) var<storage, read> frozenRO: array<u32>;

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
  // Task 'clay-surface': the platelet does not move. posRW[i] is left completely untouched.
  if (frozenRO[i] != 0u) { velRW[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0); return; }
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
  // Task 'clay-surface': a frozen particle gets no half-kick and no Langevin noise either -- were it
  // thermostatted, it would carry a velocity that the drift above then refuses to apply, i.e. a
  // nonzero velocity that means nothing. Its RNG stream is also left unconsumed, so the platelet
  // cannot shift the noise other particles receive.
  if (frozenRO[i] != 0u) { velRW[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0); return; }
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

// Task 'clay-surface': the mineral flag (data/soup.json's Monomer.mineral, true only for the clay
// platelet bead). Used ONLY for the interaction class below -- immobility is a per-PARTICLE flag
// (frozenRO), not this, because the platelet also carries beads of the catalyst species as surface
// sites and those must be frozen while their free-floating siblings are not.
fn speciesMineral(kind: f32) -> bool {
  let k = u32(kind);
  return SP.mineral[k / 4u][k % 4u] > 0.5;
}

// Species CLASS, derived from the two per-species flags data/soup.json already uploads -- the GPU
// twin of soup/src/soup-attraction.ts's speciesClassOf(), kept textually parallel on purpose.
fn speciesClass(kind: f32) -> u32 {
  if (speciesMineral(kind)) { return 3u; }
  if (speciesSolvent(kind)) { return 2u; }
  if (speciesPolar(kind)) { return 1u; }
  return 0u;
}

// Task 'hydrophobic-asymmetry' (2026-08-19): the attraction DEPTH multiplier for this pair, read
// from the per-class table above. Replaces the boolean rule task 'explicit-water' wrote here
// (water-water OR water-head only, with the apolar-apolar term structurally excluded). Restored
// apolar-apolar (tail-tail, and equally any two non-polar beads -- London dispersion does not know
// which bead was labelled catalyst), and water-tail is now a WEAK attraction rather than zero.
// Hydrophobic segregation is still emergent: it comes from the SIGN of the exchange energy
// eps_ww + eps_tt - 2*eps_wt built into the ratios, not from a hand-written "tails attract" rule --
// see data/soup.json's solvent.attractionScale.basis, which states the reversal of that earlier
// removal and the literature the ratios were read from.
fn pairAttrScale(ti: f32, tj: f32) -> f32 {
  return AttrScale.rows[speciesClass(ti)][speciesClass(tj)];
}

// Kept as a named predicate because soup_force_stats_main's own candidate-range diagnostic asks the
// same question this way; it is now DERIVED from the table (a zero cell is exactly "does not
// attract"), so there is no second place a pair rule could be stated.
fn shouldAttract(ti: f32, tj: f32) -> bool {
  return pairAttrScale(ti, tj) > 0.0;
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

// Task 'electrostatics' (2026-08-20): `qi`/`qj` are the two beads' own charges (chargeRO, declared
// in soup/wgsl/electrostatics.wgsl, written by the protonation Monte Carlo -- NOT a species
// property). Passed in rather than looked up here so the cell-sorted gather path can read the
// position from posSortedRW[k] while reading the charge at the ORIGINAL index j = cellIdx[k]. On any
// system without electrostatics every charge is 0 and esForce returns the zero vector, so this
// signature change is bit-neutral for every pre-task run.
fn nonbondedSoup(xi: vec3<f32>, xj: vec3<f32>, ti: f32, tj: f32, box: vec3<f32>, qi: f32, qj: f32) -> vec3<f32> {
  var f = vec3<f32>(0.0);
  let d = mi3(xi - xj, box);
  let r = length(d);
  if (r < 1e-6) { return f; }
  let b = pairB(ti, tj);
  if (r < wca_cut(b)) {
    f = f - wca_dv(r, b) * d / r;
  }
  // Task 'hydrophobic-asymmetry' (2026-08-19): the per-class depth multiplier scales this term's
  // MAGNITUDE only -- same rc/wc/epsilon-shaped ramp attr_dv already computes from
  // P.epsilon/P.b_tt/P.wc (rank A, untouched), same well onset and width for every pair. A zero
  // multiplier is the "this pair does not attract" case, so no separate branch states a pair rule.
  let attrScale = pairAttrScale(ti, tj);
  if (attrScale > 0.0) {
    f = f - attrScale * attr_dv(r) * d / r;
  }
  // Task 'electrostatics' (2026-08-20): the screened-Coulomb term, shifted-force truncated at the
  // SAME cutoff the attraction ends at (soup/wgsl/electrostatics.wgsl's ES.z) so no neighbour-walk
  // or Verlet-list coverage guarantee changes. `+`, not `-`: esForce already returns the force ON i
  // (repulsive for like charges points along +d), matching the sign convention the two `-` terms
  // above reach by negating their own dV/dr.
  f = f + esForce(d, r, qi, qj);
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
  let qi = chargeRO[i];

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
          f = f + nonbondedSoup(xi, xj, ti, tj, box, qi, chargeRO[j]);
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
  let qi = chargeRO[i];
  for (var j = 0u; j < n; j = j + 1u) {
    if (j == i) { continue; }
    f = f + nonbondedSoup(xi, pos2[j].xyz, ti, pos2[j].w, box, qi, chargeRO[j]);
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

// The Verlet neighbour list (build, drift guard, list-reading force kernel) used to live here and
// now lives in soup/wgsl/verlet.wgsl -- moved verbatim by task 'electrostatics' (2026-08-20) because
// this file stood at 596 lines against CLAUDE.md's hard 600 limit and the rule is "split first, then
// add". soup/src/soup-pipelines.ts concatenates it right after this file, so the compiled module is
// unchanged. The screened-Coulomb term nonbondedSoup calls lives in soup/wgsl/electrostatics.wgsl,
// concatenated BEFORE this file (WGSL has no forward declarations).
