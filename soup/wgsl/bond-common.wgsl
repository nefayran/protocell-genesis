// Task 2: stochastic bond formation and breaking, catalyst-gated where the rule says so.
//
// Split (2026-08-18, file-size rule in CLAUDE.md) out of the former monolithic bond.wgsl (809
// lines) into five files by responsibility, concatenated back into one shader module in the exact
// same order by soup/src/sim.ts (via soup/src/soup-pipelines.ts's own raw-text join) -- WGSL does
// not care about source-file boundaries, only the concatenated token stream, so this is a pure move:
// no entry-point name, binding index, or struct layout changed. The five files, in concatenation
// order: THIS ONE (shared constants/bindings/helpers), bond-valence.wgsl (atomic claim-and-rollback,
// including the head-terminal condition -- kept as one function, see that file's own header for why),
// bond-adsorption.wgsl (the catalyst<->tip tether and its two desorption paths), bond-dispatch.wgsl
// (the candidate walk and the three grid-walk entry points), bond-verlet.wgsl (the Verlet-list
// variant of the candidate walk and its own entry point).
//
// Concatenated (by soup/src/sim.ts) after engine/wgsl/forces.wgsl, reusing wca_cut() (the contact
// distance a candidate pair -- or a carbon and a catalyst -- must be within to attempt anything)
// and P.sigma (the pairwise-size mixing scale) from there; it declares no potential of its own.
//
// Detailed balance is NOT re-derived here: BondParams.acceptProbForm/acceptProbBreak are uploaded
// directly from soup/src/rules.ts's acceptanceProbability(rule, kT), and attemptProbForm/
// attemptProbBreak directly from its attemptProbability(rule, dt) -- both evaluated ONCE on the
// CPU from the SAME paired bond/break rule (same energyKT, enforced by assertRulesConsistent), so
// this file only ever compares a single uniform-random draw against a number that already came
// out of rules.ts's one Metropolis formula. There is no separate WGSL reimplementation of
// exp(-deltaE/kT) that could silently drift from it.
//
// Valence is enforced by atomic claim-both-or-rollback: tryClaimSlot() (bond-valence.wgsl) is a
// compareExchange, never a blind store, so two threads racing to fill the last free slot on the
// same particle in the same dispatch cannot both succeed; if the SECOND of a pair's two claims
// fails, the first is rolled back via releaseSlot() rather than left dangling (which would grow a
// valence-1 particle to valence 2 with only one side aware of the bond). Breaking releases both
// sides with releasePartnerSlot(), a compareExchange keyed on the expected partner id, for the
// same reason -- never a blind store to a slot whose current occupant is only assumed.
//
// Race-free by construction, not by luck: every (i, j) pair is attempted by exactly one thread --
// the one whose OWN global index is the smaller of the two (`if (j <= i) { continue; }` in the
// formation walk, `if (partner < i) { continue; }` in the break walk) -- so no edge is ever
// double-attempted from both sides in the same dispatch, and formation/breaking themselves run as
// two separate, ordered dispatches (never concurrently) within one step.

const BOND_NONE: u32 = 0xFFFFFFFFu;
const BOND_RULES: u32 = 2u;

// Task 'explicit-water' (2026-08-18): only `radius` is read in this file (bSpeciesRadius below,
// used for the WCA contact distance every claim/reach check here is built on) -- this struct
// deliberately declares just that leading member, not the polar/solvent fields
// soup/wgsl/step.wgsl's own (wider) Species struct also carries at this same physical buffer:
// a WGSL struct only needs to describe the bytes it actually reads, and the buffer is sized (and
// written, soup/src/sim.ts) for the wider struct regardless of which shader module binds it.
// Widened from vec4<f32> (4 species) to array<vec4<f32>,2> (8 species) for the same reason step.wgsl
// widened its own copy: an 8-species cap is enough for the 5th species (water) this task adds.
struct Species { radius: array<vec4<f32>, 2> };
@group(1) @binding(7) var<uniform> SP: Species;

struct BondParams {
  kindA: vec4<f32>,
  kindB: vec4<f32>,
  attemptProbForm: vec4<f32>,
  attemptProbBreak: vec4<f32>,
  acceptProbForm: vec4<f32>,
  acceptProbBreak: vec4<f32>,
  requiresCatalyst: vec4<f32>,
  slotRoleA: vec4<f32>,
  slotRoleB: vec4<f32>,
  catalystKind: f32,
  // data/soup.json's headPlacement.terminalOnly, 1.0/0.0 the same way requiresCatalyst's flags are
  // -- see tryClaimSlot() (bond-valence.wgsl) for what it gates and why.
  headTerminalOnly: f32,
  // data/soup.json's headPlacement.chainCapacity -- how many chain-pool slots (0..cap-1) a head's
  // OWN claim (tryClaimSlot's role==2 branch) may try, uploaded once from soup/src/sim.ts exactly
  // as headTerminalOnly already is. Replaces the former bpPad1 padding float. bpPad2 remains unused.
  headChainCapacity: f32,
  bpPad2: f32,
  // Surface growth / adsorption (bond-adsorption.wgsl's header): x = data/soup.json's
  // adsorption.maxHoldSteps (real integration steps a centre may hold the SAME chain tip without a
  // successful propagation/termination event before desorbTimeout() releases it regardless
  // of distance); y = bondAttemptInterval.steps (how many real steps elapse per bond-dispatch
  // cycle -- what centerHeldSteps is incremented by each cycle, so x and the running total stay in
  // the SAME real-step units, uploaded once from soup/src/sim.ts). z, w unused.
  adsorptionParams: vec4<f32>,
};
@group(2) @binding(0) var<uniform> BP: BondParams;

@group(1) @binding(6) var<storage, read_write> bondSlots: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> bondEvents: array<atomic<u32>>;
@group(1) @binding(9) var<storage, read_write> bondRng: array<u32>;
// Surface growth (bond-adsorption.wgsl's header, "Surface growth"): one slot per particle, crossing
// kinds -- see that header for exactly what it holds for a catalyst vs a carbon.
@group(1) @binding(20) var<storage, read_write> centerLink: array<atomic<u32>>;

// Surface growth / adsorption (bond-adsorption.wgsl's header): per-particle count of consecutive REAL
// steps a centre has held the SAME chain tip without a successful propagation or termination event on
// it -- meaningful only while centerLink[i] != BOND_NONE, reset to 0 by any such event
// (propagateOnCenter/terminateOnCenter) or by either desorption path. Incremented and checked
// once per bond-dispatch cycle for every CATALYST particle by desorbTimeout().
@group(1) @binding(21) var<storage, read_write> centerHeldSteps: array<atomic<u32>>;

// Surface growth / adsorption (bond-adsorption.wgsl's header): [0] = desorption events triggered by
// the adsorption bond stretching past FENE's own bonded range (desorbStretch), [1] = desorption events
// triggered by the hold-time safety valve (desorbTimeout) -- reported alongside the existing
// per-rule bondEvents (SoupSystem.desorbEvents(), soup/src/sim.ts) so a run can show, with numbers,
// that the predecessor task's deadlock is now impossible by construction, not by luck.
@group(1) @binding(22) var<storage, read_write> desorbEvents: array<atomic<u32>>;

// perf2-report.md, candidate (b): the SAME cell-sorted position gather soup/wgsl/step.wgsl's
// soup_force_main reads (populated once per step, before both force and bond attempts run --
// soup/src/sim.ts's encodeOneIntegrationStep order). Read-only here: this module never gathers,
// only reads what the force module's soup_gather_sorted_main already wrote into the same physical
// buffer this step.
@group(1) @binding(13) var<storage, read> posSortedRO: array<vec4<f32>>;

fn vget(v: vec4<f32>, idx: u32) -> f32 {
  if (idx == 0u) { return v.x; }
  else if (idx == 1u) { return v.y; }
  else if (idx == 2u) { return v.z; }
  else { return v.w; }
}

fn bSpeciesRadius(kind: f32) -> f32 {
  let k = u32(kind);
  return SP.radius[k / 4u][k % 4u];
}

fn bPairB(ti: f32, tj: f32) -> f32 { return P.sigma * (bSpeciesRadius(ti) + bSpeciesRadius(tj)) * 0.5; }

fn bMi3(d_in: vec3<f32>, box: vec3<f32>) -> vec3<f32> {
  return d_in - round(d_in / box) * box;
}

// Small non-cryptographic hash, same family as integrate.wgsl's thermostat noise -- an
// independent generator so bond attempts do not consume the thermostat's own random stream (which
// would couple the two in a way neither is designed to tolerate).
fn bondPcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}
fn bondUniform01(v: u32) -> f32 { return f32(v) / 4294967296.0; }

// Which of the BOND_RULES species-pair rules governs (ti, tj), order-independent. Returns -1 if
// neither of this task's two rules (C-C, C-O) involves this pair -- true for every H/M pair and
// for H or M paired with anything, since neither appears as a kindA/kindB in data/soup.json.
fn matchRule(ti: f32, tj: f32) -> i32 {
  for (var r = 0u; r < BOND_RULES; r = r + 1u) {
    let a = vget(BP.kindA, r);
    let b = vget(BP.kindB, r);
    if ((ti == a && tj == b) || (ti == b && tj == a)) { return i32(r); }
  }
  return -1;
}

// Slot role this particle plays in rule `ruleIdx`, given its OWN kind: 0 = chain pool (try slots
// 0,1 -- a carbon's up-to-two chain-growth bonds), 1 = head slot (slot 2 fixed -- a carbon's
// single bond to a polar head), 2 = head's own chain pool (try slots 0..headChainCapacity-1 -- a
// head's own up-to-`chainCapacity` tails, data/soup.json's headPlacement.chainCapacity). Which
// code applies to which side is resolved once on the CPU from data/soup.json's monomer kinds (see
// soup/src/sim.ts) and simply looked up here.
fn roleOf(ruleIdx: u32, kind: f32) -> u32 {
  if (kind == vget(BP.kindA, ruleIdx)) { return u32(vget(BP.slotRoleA, ruleIdx)); }
  return u32(vget(BP.slotRoleB, ruleIdx));
}

fn hasBondTo(i: u32, j: u32) -> bool {
  let base = i * 3u;
  for (var s = 0u; s < 3u; s = s + 1u) {
    if (atomicLoad(&bondSlots[base + s]) == j) { return true; }
  }
  return false;
}
