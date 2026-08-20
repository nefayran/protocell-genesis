// Task 'electrostatics' (2026-08-20): SCREENED COULOMB (Debye-Huckel) between charged head beads.
//
// WHY THIS FILE EXISTS. Every predecessor of this task ran with strictly neutral heads, and
// data/soup.json's own `saltPhLimitation` declared ionic strength and pH UNREPRESENTABLE for exactly
// that reason. Real fatty acids are carboxylic acids: near pH 7-9 the head sits in a protonation
// equilibrium (COOH <-> COO- + H+), the deprotonated head carries -e, and charged heads REPEL. That
// repulsion is what sets the effective head area a0 in Israelachvili's packing parameter
// p = v/(a0*l_c), and it is what keeps aggregates from cross-linking into one phase -- the exact
// failure mode `final-campaign-report.md` measured (one percolating network wrapping 3 of 3 axes).
//
// THE FORM, and why this one. Debye-Huckel / screened Coulomb is the standard coarse-grained
// treatment of electrolyte solutions when the ions are not explicit (Debye & Huckel, Phys. Z. 24
// (1923) 185; textbook form in Israelachvili, *Intermolecular and Surface Forces* 3rd ed. ch. 14, and
// the CG-simulation convention in e.g. MARTINI, Marrink et al. JPCB 111 (2007) 7812, which likewise
// screens a Coulomb term and cuts it at ~1.2 nm):
//
//   U(r) = kT * (l_B/sigma) * q_i q_j * exp(-r/lambda_D) / r        [reduced units, r in sigma]
//
// with l_B the Bjerrum length and lambda_D the Debye length set by the IONIC STRENGTH -- so salt
// becomes a real, stated parameter of this model instead of an unrepresentable one. Both lengths, the
// sigma->nm mapping they are divided by, and the RANGE that mapping carries are in
// data/soup.json's `electrostatics` section with their ranks and bases; nothing numeric is written
// here (tests/params.test.ts's literal scanner enforces that over all of soup/wgsl).
//
// TRUNCATION: SHIFTED FORCE, not a plain cut. Every other term in this engine is continuous at its
// own cutoff (WCA reaches zero at 2^(1/6) b; attr_dv's cos^2 ramp reaches zero at rc+wc), and a
// plain-cut Coulomb would be the only discontinuous force in the model -- which shows up as energy
// drift and, worse, would break the CPU-potential-vs-GPU-force agreement
// tests/soup-area-move.test.ts pins. So the standard shifted-force truncation is used
// (Toxvaerd & Dyre, J. Chem. Phys. 134 (2011) 081102, "Communication: Shifted forces in molecular
// dynamics"; Allen & Tildesley, *Computer Simulation of Liquids*, 2nd ed. sec. 5.2):
//
//   F_sf(r) = F(r) - F(rc)          U_sf(r) = U(r) - U(rc) + (r - rc) * F(rc)
//
// both identically zero at r = rc, F continuous there. ES.z carries rc and ES.w carries F(rc); the
// CPU twin (soup/src/electrostatics.ts) computes the same two numbers from the same fields, so the
// two cannot drift.
//
// WHAT CARRIES THE CHARGE. `chargeRO[i]` is a per-particle charge in units of e, written from the CPU
// by the protonation Monte Carlo (soup/src/soup-protonation.ts) and read here. It is NOT a species
// property: two beads of the SAME head species differ in it, because protonation state is a
// dynamical variable in this model, not a label (see soup/src/electrostatics.ts's header for the
// sampling scheme and its detailed-balance argument). Every non-head bead holds exactly 0 and the
// term below vanishes for it without a branch.
//
// WHAT IS NOT REPRESENTED, restated so it cannot go stale: no explicit H+ or Na+/Cl- beads (the
// screening is a mean-field length, not particles), no titration of the medium (pH is a fixed
// external parameter per run, not a conserved proton budget), no Born/self-energy term (absorbed into
// the intrinsic pKa), and no electrostatic contribution to the BOND Metropolis (which, as before this
// task, uses only its rule's own energyKT -- it already ignored WCA and attraction changes too).

// ES: x = coefficient A = kT * l_B/sigma (units epsilon*sigma, since kT is in epsilon),
//     y = kappa = sigma/lambda_D (inverse screening length, sigma^-1),
//     z = rc (the cutoff, in sigma -- the SAME nonbonded interaction cutoff wca_cut(b_tt)+wc the
//         attraction already ends at, so no neighbour-walk coverage guarantee changes),
//     w = F(rc), the shifted-force constant subtracted from every force below.
// Built by soup/src/electrostatics.ts from data/soup.json; zeroed (A = 0) on any system that does not
// ask for electrostatics, which makes this term identically zero and every pre-task run bit-identical.
@group(1) @binding(25) var<uniform> ES: vec4<f32>;
@group(1) @binding(24) var<storage, read> chargeRO: array<f32>;

/** Magnitude of the screened-Coulomb pair force at separation r for the charge product qq, with the
 * shifted-force constant already removed. Positive = repulsive (like charges). Returns 0 outside the
 * cutoff and for any pair where either bead is neutral, so the caller needs no branch. */
fn esForceMag(r: f32, qq: f32) -> f32 {
  if (ES.x == 0.0 || qq == 0.0 || r >= ES.z) { return 0.0; }
  // F(r) = A*qq*exp(-kappa*r)*(1/r + kappa)/r ; then the shift F(rc) (ES.w, itself already carrying
  // A but NOT qq -- it is stored for unit charge product, so it scales with qq here exactly as F does).
  let u = ES.x * exp(-ES.y * r) / r;
  return qq * (u * (1.0 / r + ES.y) - ES.w);
}

/** The NEAR half of the split: everything below ES2.x (the grid's own interactionRange), which is
 * the part nonbondedSoup adds along with WCA and the attraction. See ES2's own header for why the
 * term is split by radius instead of being moved wholesale. */
fn esForceNear(d: vec3<f32>, r: f32, qi: f32, qj: f32) -> vec3<f32> {
  if (r >= ES2.x) { return vec3<f32>(0.0); }
  return esForce(d, r, qi, qj);
}

/** The FAR half: ES2.x <= r < ES.z, added by soup_es_force_far_main from the dedicated head-only
 * list. Same shift constants as the near half, so near + far is exactly the untruncated term. */
fn esForceFar(d: vec3<f32>, r: f32, qi: f32, qj: f32) -> vec3<f32> {
  if (r < ES2.x) { return vec3<f32>(0.0); }
  return esForce(d, r, qi, qj);
}

/** The pair force vector, added by nonbondedSoup (soup/wgsl/step.wgsl) to every pair it already
 * visits. `d` is the minimum-image displacement xi - xj the caller already computed and `r` its
 * length, and the two charges are passed in rather than looked up -- so this file needs neither mi3
 * nor the box nor an index convention, and can therefore be concatenated BEFORE step.wgsl (WGSL has
 * no forward declarations). The caller reads chargeRO (declared above) at ITS own index pair, which
 * is what keeps the cell-sorted gather path correct: that path reads the POSITION from posSortedRW[k]
 * but still knows the original index j = cellIdx[k], and charge must be read at the original index. */
fn esForce(d: vec3<f32>, r: f32, qi: f32, qj: f32) -> vec3<f32> {
  let qq = qi * qj;
  if (qq == 0.0) { return vec3<f32>(0.0); }
  return esForceMag(r, qq) * d / r;
}


// ============================================================================================
// Task 'long-range-electrostatics' (2026-08-20): THE DEDICATED LONG-RANGE PASS.
//
// WHY IT EXISTS. The version above shared the engine's Lennard-Jones nonbonded cutoff
// (2.7224620 sigma) "to add no new constant". Measured consequence: at 10 mM ionic strength
// lambda_D = 3.80 sigma, so the cutoff spanned 0.72 Debye lengths and discarded
// exp(-x)*(1+x) = 84 % of the INTEGRATED interaction -- and the salt dependence of the apparent
// pKa came out -0.124 between 10 and 100 mM against the literature's ~0.7. A Coulomb term cannot
// share a Lennard-Jones cutoff; that is why every MD package gives electrostatics its own.
//
// WHY A REAL-SPACE CUTOFF AND NOT EWALD/PPPM. Mesh methods exist because the lattice sum of 1/r
// converges only conditionally. Here the screening is physical (a mean-field Debye length), the
// sum converges ABSOLUTELY and exponentially, and a real-space cutoff at a fixed multiple of
// lambda_D is accurate to exp(-x)*(1+x) -- 9.2 % at 4 lambda_D -- with no FFT, which this engine
// does not have. A Wolf / damped-shifted-force scheme with damping alpha = kappa is, term for
// term, the shifted-force screened Coulomb already implemented above, plus a constant self-energy
// that produces no force. See data/soup.json's electrostatics.basis item 12.
//
// WHY A SEPARATE, HEAD-ONLY LIST. Extending the SHARED cutoff to 15.2 sigma would put ~4370
// candidates in every particle's Verlet list (against ~449 now), i.e. ~10x the pair work, and
// would need listCapacity 5000 = 3.84 GB against the measured 4.295 GB ceiling. But charge lives
// on the titratable species only -- 9296 beads of 191778, 4.85 % -- and every other bead holds
// exactly 0, so the long range only ever needs head-head pairs. Compacting the head indices and
// building a list over THEM costs N_h^2 = 86.4 M distance tests per rebuild (8.6 M/step at
// rebuildEvery = 10) against the main list's own 743 M/rebuild, and 9296*2500*4 = 93 MB.
//
// WHY THE COMPACTION IS SERIAL. An atomic append is a one-liner but its ORDER varies run to run,
// which would make the force sum's rounding vary and break the "a checkpoint does not change what
// the run computes next" pin. One invocation walking N in index order is deterministic, and it is
// the same shape as neighbor.wgsl's own prefix_main (also workgroup_size(1) over its whole domain).
//
// ES2: x = splitRadius (the grid's interactionRange -- below it nonbondedSoup already sums the term
//          and the existing walk guarantees completeness, so this pass adds only the remainder),
//      y = listRange (rc_es + verletList.skin, the radius this list is built to),
//      z = listCapacity per head, w = the titratable kind index.
// All four from soup/src/electrostatics.ts, never a WGSL literal.
@group(1) @binding(26) var<uniform> ES2: vec4<f32>;
@group(1) @binding(27) var<storage, read_write> esList: array<u32>;
@group(1) @binding(28) var<storage, read_write> esCount: array<u32>;
@group(1) @binding(29) var<storage, read_write> headIdx: array<u32>;
// [0] = number of titratable beads found, [1] = list-overflow flag (a loud throw on the CPU side,
// never a silent truncation -- soup/src/soup-grid-verlet.ts's assertVerletSafety reads it).
@group(1) @binding(30) var<storage, read_write> esMeta: array<atomic<u32>>;
