// Pair interactions of the soup: the species CLASS derivation, the per-class attraction depth
// table, the pair mixing rule and the whole nonbonded pair force nonbondedSoup(). Split out of
// soup/wgsl/step.wgsl by the file-size rule in CLAUDE.md ("split first, then add"), which the same
// file has been split by twice before -- soup/wgsl/verlet.wgsl (task 'electrostatics') and
// soup/wgsl/health.wgsl + relax.wgsl (task 'loud-failure-and-liquid-water'). This is a PURE MOVE of
// the five blocks named below plus ONE new term (the charge-assisted head-head pair, marked with its
// own task banner); nothing else was edited, no binding index moved, no formula was rewritten.
//
// CONCATENATION ORDER IS LOAD-BEARING (WGSL has no forward declarations, see
// soup/src/soup-pipelines.ts): this file declares Species/SP, AttrScaleTable/AttrScale, mi3() and
// nonbondedSoup(), which soup/wgsl/step.wgsl, verlet.wgsl, electrostatics-long.wgsl and wall.wgsl
// all call -- so it must come BEFORE step.wgsl. It itself calls esForceNear (soup/wgsl/
// electrostatics.wgsl) and wca_dv/attr_dv/wca_cut/P (engine/wgsl/forces.wgsl), so it must come
// AFTER both. Position in the module: forces.wgsl, electrostatics.wgsl, THIS, step.wgsl, ...
//
// Moved verbatim from step.wgsl: the AttrScale uniform block, the Species uniform block, mi3(),
// speciesRadius/speciesPolar/speciesSolvent/speciesMineral/speciesClass/pairAttrScale/
// shouldAttract/pairB, and nonbondedSoup().


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
// Task 'acid-soap-pairing' (2026-08-23): ONE more vec4 after the rows -- `pair.x` is the
// charge-assisted head-head depth (acidSoapScale() below), pair.yzw spare. Not a fifth ROW and not a
// cell inside the rows: the rows are indexed by species CLASS and are a pure function of the two
// species, while this depth is a function of the two beads' PROTONATION STATES, which are dynamical
// variables the constant-pH Monte Carlo owns. soup/src/soup-attraction.ts sizes the uniform from
// (CLASS_COUNT + 1) * 4 floats, so the two sides cannot disagree about the size without failing
// bind-group validation outright.
struct AttrScaleTable { rows: array<vec4<f32>, 4>, pair: vec4<f32> };
@group(1) @binding(9) var<uniform> AttrScale: AttrScaleTable;

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

// Task 'acid-soap-pairing' (2026-08-23): THE CHARGE-ASSISTED HEAD-HEAD PAIR.
//
// The only attraction in this model that reads a DYNAMICAL variable rather than two species ids. It
// fires between exactly one PROTONATED head (q == 0, the neutral carboxylic acid) and one
// DEPROTONATED head (q != 0, the carboxylate) -- never between two heads in the same protonation
// state. That asymmetry is the whole point and it is measured, not assumed: the atomic engine
// (atomic/, MACE-OFF23) associated two carboxyl heads at IDENTICAL composition and found the neutral
// acid+acid dimer UNFAVOURABLE in water (-5.51 kcal/mol with 32 waters, i.e. water wins the hydrogen
// bonds) while acid+carboxylate stayed FAVOURABLE (+10.38 kcal/mol with 32 waters, four independent
// starting geometries converging to one minimum). A polarPolar cell of the class table could not
// express that -- it would attract acid+acid and soap+soap identically.
//
// The protonation state is read from `q`, i.e. from chargeRO, which the constant-pH Monte Carlo
// (soup/src/electrostatics.ts, soup/src/soup-protonation.ts) is the only writer of. No new state and
// no new species: a head that titrates changes which pairs it makes on the SAME sweep that changes
// its charge, which is why the pH window this predicts is a prediction and not a construction.
//
// Depth only -- the WELL SHAPE (onset rc = wca_cut(P.b_tt), width P.wc, and the epsilon it is
// measured in) is attr_dv's own, rank A and untouched, exactly as every class-table cell uses it.
// The value in AttrScale.pair.x is a fraction of the measured association ENERGY, not of a free
// energy: the entropy cost of pairing was NOT computed (that needs enhanced sampling the atomic
// engine does not have) and it works AGAINST the pair, so the measured number is an UPPER BOUND.
// See data/soup.json's solvent.attractionScale.acidSoapPair.basis for the mapping and the sweep.
fn acidSoapScale(ti: f32, tj: f32, qi: f32, qj: f32) -> f32 {
  let s = AttrScale.pair.x;
  if (s <= 0.0) { return 0.0; }
  // Both beads must be HEADS (class 1 = polar). Read through speciesClass() rather than comparing
  // kind indices so a data/soup.json that ever declares a second polar species needs no edit here.
  if (speciesClass(ti) != 1u || speciesClass(tj) != 1u) { return 0.0; }
  // Exactly one charged: the XOR that makes acid+soap a pair and acid+acid / soap+soap not.
  if ((qi != 0.0) == (qj != 0.0)) { return 0.0; }
  return s;
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
  // Task 'acid-soap-pairing' (2026-08-23): the class-pair depth PLUS the charge-assisted head-head
  // depth, summed into ONE multiplier on the SAME attr_dv ramp rather than added as a second term
  // with its own shape -- so the pair term cannot introduce a different well onset, a different
  // width or a second cutoff, and every neighbour-walk/Verlet-list coverage guarantee in this engine
  // stays exactly as it was. With data/soup.json's polarPolar at 0 the head-head total IS the
  // acid-soap term; on any pair that is not one protonated head with one deprotonated head
  // acidSoapScale() is identically 0 and this line is bit-identical to the pre-task one.
  let attrScale = pairAttrScale(ti, tj) + acidSoapScale(ti, tj, qi, qj);
  if (attrScale > 0.0) {
    f = f - attrScale * attr_dv(r) * d / r;
  }
  // Task 'electrostatics' (2026-08-20): the screened-Coulomb term, shifted-force truncated at the
  // SAME cutoff the attraction ends at (soup/wgsl/electrostatics.wgsl's ES.z) so no neighbour-walk
  // or Verlet-list coverage guarantee changes. `+`, not `-`: esForce already returns the force ON i
  // (repulsive for like charges points along +d), matching the sign convention the two `-` terms
  // above reach by negating their own dV/dr.
  // Task 'long-range-electrostatics' (2026-08-20): the NEAR half only. The screened-Coulomb cutoff
  // is now a multiple of the Debye length (up to 15.2 sigma), which no cell walk or Verlet list in
  // this engine covers; esForceNear stops at ES2.x = the grid's own interactionRange, and
  // soup_es_force_far_main (soup/wgsl/electrostatics.wgsl) adds exactly the remainder from a
  // dedicated head-only list. Same shift constants in both halves, so their sum is the whole term.
  f = f + esForceNear(d, r, qi, qj);
  return f;
}
