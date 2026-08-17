// Task 2: stochastic bond formation and breaking, catalyst-gated where the rule says so.
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
// Valence is enforced by atomic claim-both-or-rollback: tryClaimSlot() below is a
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
//
// Head placement (data/soup.json's headPlacement.terminalOnly, valence-and-heads-report.md /
// terminal-heads-report.md): a carboxyl head belongs on a chain END, never mid-chain, even though
// mid-chain is a valid VALENCE state (2 chain bonds + 1 head bond is exactly the cap). tryClaimSlot
// below adds two conditions on WHERE a bond may form, not on the energies: (1) a head may only
// claim its slot on a carbon that currently has at most one chain (C-C) bond, (2) a carbon that
// already carries a head may not accept a further chain bond. Both are implemented the same way
// the valence cap itself is -- claim the slot optimistically with the existing
// atomicCompareExchangeWeak, THEN check the other condition, and roll the claim back if it was
// violated -- never a pre-check-then-claim, because a plain pre-check reading two different atomics
// without an intervening claim is exactly the kind of race the compareExchange pattern exists to
// avoid. Reversibility: breaking a C-O bond off a terminal-head carbon returns the system to a
// state where forming that bond is again allowed (nothing about the break is gated by this rule),
// so detailed balance for co_bond/co_break is undisturbed; a C-C bond this rule forbids can never
// have formed in the first place, so its reverse (a cc_break of a bond that never existed) never
// needs to fire -- there is no realised transition this rule makes irreversible.
//
// Two tails per head (data/soup.json's headPlacement.chainCapacity, "two-tails" task, packing-
// parameter reasoning in that file's own basis): a head's own slot claim (role==2 below) used to
// try ONLY slot 0 -- a hardcoded single-tail cap. It now loops over BP.headChainCapacity slots
// (0..cap-1), the exact same claim-then-rollback idiom the carbon chain pool (role==0, slots 0/1)
// already uses for its own 2-slot cap, just driven by a uniform instead of two hand-written
// branches. Reversibility is unchanged by this generalisation: releasePartnerSlot (below) already
// scans all 3 of a particle's slots for the one naming the breaking partner, so breaking EITHER a
// head's first or second tail finds and clears the right slot the same way; every C-O bond, whichever
// slot it landed in, still has exactly one paired co_break at the same energyKT (assertRulesConsistent),
// so raising the number of slots a head may claim adds newly REACHABLE forward transitions (a second
// tail attaching) without touching the accept/reject ratio of any single bond attempt -- nothing this
// change makes formable is left without its own always-available reverse.
//
// Surface growth (task 'surface-growth', 2026-08-17, surface-growth-report.md, diagnosis in
// kinetic-growth-report.md): that report measured a 9-10 point undershoot between the CONFIGURED
// propagation:termination ratio (data/soup.json's cc_bond/co_bond attemptRate ratio) and the alpha
// actually RECOVERED from the measured chain-length histogram, plus ~53% of carbon sitting in
// length-1 chains. Root cause: cc_bond already required a catalyst (`requiresCatalyst`, the
// `catalystNear`/`catalystId` check below) but co_bond did not -- termination could fire on ANY
// carbon anywhere in the bulk while propagation was throttled to the sparse catalyst population, so
// the realised ratio was not the configured one. Fischer-Tropsch-type chemistry does not do this: a
// growing chain stays ADSORBED on the one catalytic centre that started it, and BOTH propagation and
// termination happen there, on that centre, until the finished amphiphile desorbs and the centre is
// free again.
//
// This is carried by ONE new buffer, `centerLink` (declared below, next to `bondSlots`): a single
// slot per particle, mirroring bondSlots' own mutual-pointer discipline (compareExchange, never a
// blind store) but crossing KINDS instead of joining same-kind neighbours -- for a catalyst it holds
// the carbon id of the chain tip it currently anchors (or BOND_NONE if free); for a carbon it holds
// the catalyst id anchoring it AS THE ACTIVE GROWING TIP (or BOND_NONE if this carbon is not
// currently a tip: it never nucleated, it is the permanently-inert distal end of an already-started
// chain -- real amphiphile chains grow from and are capped at ONE end only, the other stays a plain
// terminus exactly like a real fatty acid's tail -- or its chain has already been terminated/
// desorbed and released, and may be RE-ADSORBED by a different free catalyst later, see
// propagateOnCenter's own re-adsorption comment below for why this model cannot and does not try to
// tell those two owner-less cases apart). See propagateOnCenter/terminateOnCenter below for how
// propagation/termination read and update it, and this task's own basis text in data/soup.json's
// cc_bond/co_bond entries for the physical argument in full.
//
// Adsorption as a FORCE, and desorption (task 'adsorption', 2026-08-17, adsorption-report.md,
// diagnosis in this task's own predecessor, surface-growth-report.md): recording the association
// in centerLink alone was bookkeeping, not physics -- nothing pulled a claimed tip back toward its
// centre, so thermal diffusion separated them within a handful of steps and almost every one of
// the 400 centres deadlocked permanently on its first unfinished chain (0 amphiphiles, both seeds,
// single-bead fraction 90.5%, recovered α 0.36-0.40 against configured 0.9333 -- surface-growth-
// report.md §3). The fix: soup/wgsl/step.wgsl's bondedForce now applies a REAL FENE spring (the
// SAME fene_dv every covalent bond already uses) between a catalyst and the tip its centerLink
// names, in ADDITION to the existing centerLink bookkeeping here -- the tip is now physically held
// at the surface, not merely remembered. This tether lives entirely in centerLink/centerHeldSteps,
// never in bondSlots, so it costs no carbon valence and stays invisible to findAmphiphiles exactly
// as before (this task's own requirement 1).
//
// Two desorption paths make a permanently stuck centre impossible BY CONSTRUCTION, on top of the
// tether itself (requirement 4): desorbStretch releases a pair whose distance has exceeded FENE's
// own bonded range (P.r_inf, data/params.json's fene.rInf) -- the literal "stretched beyond the
// bonded range" case, checked unconditionally every bond-dispatch cycle from the carbon's own side,
// independent of whether any bond-forming candidate happens to be nearby (the exact gap that let a
// fully-drifted-away tip in the predecessor task never get re-examined at all); desorbTimeout
// releases a centre that has held the SAME chain for more than data/soup.json's own
// adsorption.maxHoldSteps real steps without a single successful propagation/termination event on
// it (both reset the running count), regardless of distance -- the "longer than a stated number of
// steps without an event" case. Both are counted separately (desorbEvents) from the existing
// per-rule bondEvents, since neither is a completed amphiphile.
//
// Occupancy stays 1 chain per centre (data/soup.json's adsorption.occupancy, considered and kept --
// see that field's own basis for why): the deadlock's cause was the missing tether, not the single-
// slot design, and centerLink's own single-u32-per-particle layout is what this task's code changes
// below assume throughout.

const BOND_NONE: u32 = 0xFFFFFFFFu;
const BOND_RULES: u32 = 2u;

struct Species { radius: vec4<f32>, polar: vec4<f32> };
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
  // -- see tryClaimSlot() below for what it gates and why.
  headTerminalOnly: f32,
  // data/soup.json's headPlacement.chainCapacity -- how many chain-pool slots (0..cap-1) a head's
  // OWN claim (tryClaimSlot's role==2 branch) may try, uploaded once from soup/src/sim.ts exactly
  // as headTerminalOnly already is. Replaces the former bpPad1 padding float. bpPad2 remains unused.
  headChainCapacity: f32,
  bpPad2: f32,
  // Surface growth / adsorption (this task, adsorption-report.md): x = data/soup.json's
  // adsorption.maxHoldSteps (real integration steps a centre may hold the SAME chain tip without a
  // successful propagation/termination event before desorbTimeout() below releases it regardless
  // of distance); y = bondAttemptInterval.steps (how many real steps elapse per bond-dispatch
  // cycle -- what centerHeldSteps is incremented by each cycle, so x and the running total stay in
  // the SAME real-step units, uploaded once from soup/src/sim.ts). z, w unused.
  adsorptionParams: vec4<f32>,
};
@group(2) @binding(0) var<uniform> BP: BondParams;

@group(1) @binding(6) var<storage, read_write> bondSlots: array<atomic<u32>>;
@group(1) @binding(8) var<storage, read_write> bondEvents: array<atomic<u32>>;
@group(1) @binding(9) var<storage, read_write> bondRng: array<u32>;
// Surface growth (this file's header, "Surface growth"): one slot per particle, crossing kinds --
// see the header for exactly what it holds for a catalyst vs a carbon.
@group(1) @binding(20) var<storage, read_write> centerLink: array<atomic<u32>>;

// Surface growth / adsorption (this file's header): per-particle count of consecutive REAL steps a
// centre has held the SAME chain tip without a successful propagation or termination event on it --
// meaningful only while centerLink[i] != BOND_NONE, reset to 0 by any such event
// (propagateOnCenter/terminateOnCenter) or by either desorption path below. Incremented and checked
// once per bond-dispatch cycle for every CATALYST particle by desorbTimeout().
@group(1) @binding(21) var<storage, read_write> centerHeldSteps: array<atomic<u32>>;

// Surface growth / adsorption (this file's header): [0] = desorption events triggered by the
// adsorption bond stretching past FENE's own bonded range (desorbStretch), [1] = desorption events
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
  if (k == 0u) { return SP.radius.x; }
  else if (k == 1u) { return SP.radius.y; }
  else if (k == 2u) { return SP.radius.z; }
  else { return SP.radius.w; }
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

fn tryClaimSlot(particle: u32, role: u32, partner: u32) -> i32 {
  let base = particle * 3u;
  let headTerminalOnly = BP.headTerminalOnly > 0.5;
  if (role == 0u) {
    // Chain-pool claim (a C-C bond). headPlacement.terminalOnly's condition (2): a carbon that
    // already carries a head (slot 2 occupied) may not accept a further chain bond, since that
    // would bury the head mid-chain. Claim the chain slot first (unconditionally, same as before),
    // then check slot 2 and roll back if this particle turns out to already have a head -- never
    // the other order, see the file header for why a pre-check would race.
    let r0 = atomicCompareExchangeWeak(&bondSlots[base + 0u], BOND_NONE, partner);
    if (r0.exchanged) {
      if (headTerminalOnly && atomicLoad(&bondSlots[base + 2u]) != BOND_NONE) {
        atomicStore(&bondSlots[base + 0u], BOND_NONE);
        return -1;
      }
      return 0;
    }
    let r1 = atomicCompareExchangeWeak(&bondSlots[base + 1u], BOND_NONE, partner);
    if (r1.exchanged) {
      if (headTerminalOnly && atomicLoad(&bondSlots[base + 2u]) != BOND_NONE) {
        atomicStore(&bondSlots[base + 1u], BOND_NONE);
        return -1;
      }
      return 1;
    }
    return -1;
  } else if (role == 1u) {
    // Head-slot claim (the C side of a C-O bond). headPlacement.terminalOnly's condition (1): a
    // head may only end up on a carbon that has at most one existing chain bond -- a chain end (or
    // a lone carbon), not a carbon already using both chain slots. Claim slot 2 first, then check
    // whether both chain slots turned out to already be occupied and roll back if so.
    let r2 = atomicCompareExchangeWeak(&bondSlots[base + 2u], BOND_NONE, partner);
    if (r2.exchanged) {
      if (headTerminalOnly) {
        let s0 = atomicLoad(&bondSlots[base + 0u]);
        let s1 = atomicLoad(&bondSlots[base + 1u]);
        if (s0 != BOND_NONE && s1 != BOND_NONE) {
          atomicStore(&bondSlots[base + 2u], BOND_NONE);
          return -1;
        }
      }
      return 2;
    }
    return -1;
  } else {
    // Head's own claim (the O side of a C-O bond). data/soup.json's headPlacement.chainCapacity:
    // try each of this head's own slots in turn (0..cap-1) until one is free -- the same
    // claim-via-compareExchange discipline the chain pool (role==0 above) already uses for its own
    // fixed 2-slot cap, generalised to a runtime count instead of two hardcoded branches, so a head
    // may end up holding up to `cap` tails (cap=2: two-tailed amphiphile, the packing-parameter
    // geometry data/soup.json's own basis argues for).
    let cap = u32(BP.headChainCapacity);
    for (var s = 0u; s < cap; s = s + 1u) {
      let r = atomicCompareExchangeWeak(&bondSlots[base + s], BOND_NONE, partner);
      if (r.exchanged) { return i32(s); }
    }
    return -1;
  }
}

fn releaseSlot(particle: u32, slot: i32) {
  if (slot < 0) { return; }
  atomicStore(&bondSlots[particle * 3u + u32(slot)], BOND_NONE);
}

// Releases the slot on `partner` that currently points back to `me` -- a compareExchange keyed on
// the expected value (me), never a blind store, so this cannot clobber a DIFFERENT bond `partner`
// formed concurrently in the same dispatch (impossible for THIS edge under the i<j dedupe above,
// but partner may hold up to two other live bonds in its remaining slots and this must only ever
// touch the one slot that names `me`).
fn releasePartnerSlot(partner: u32, me: u32) {
  let base = partner * 3u;
  for (var s = 0u; s < 3u; s = s + 1u) {
    let r = atomicCompareExchangeWeak(&bondSlots[base + s], me, BOND_NONE);
    if (r.exchanged) { return; }
  }
}

// Surface growth (this file's header): centerLink's own compareExchange primitive, the exact same
// discipline tryClaimSlot/releasePartnerSlot already use, generalised to the single-slot
// catalyst<->tip buffer. `particle` is either a catalyst (holding a tip carbon id) or a carbon
// (holding its owning catalyst id) -- the two kinds never collide since a rule never matches
// catalyst-catalyst or catalyst-carbon, so this same function safely serves both directions.
fn centerCas(particle: u32, expected: u32, newVal: u32) -> bool {
  let r = atomicCompareExchangeWeak(&centerLink[particle], expected, newVal);
  return r.exchanged;
}

fn centerOf(particle: u32) -> u32 { return atomicLoad(&centerLink[particle]); }

// Surface growth / adsorption (this file's header): the owner<->tip proximity/geometry test
// propagateOnCenter/terminateOnCenter already needed, factored once so both of them and the
// desorption safety valves below all read one formula instead of hand-copying it three times. Order
// of the two arguments does not matter -- bPairB(ti,tj) sums both radii, so it is symmetric in its
// own two arguments regardless of which particle's own kind gets passed as which.
fn centerWithinReach(a: u32, b: u32) -> bool {
  let d = bMi3(pos2[a].xyz - pos2[b].xyz, GB.box.xyz);
  return length(d) < wca_cut(bPairB(pos2[a].w, BP.catalystKind)) + P.wc;
}

// Propagation (cc_bond): i and j are both carbons, roleI==roleJ==0 (chain pool). Called only AFTER
// tryClaimSlot has already succeeded for both -- slotI/slotJ are its return values, which classify
// this pair for free: this pool always fills slot 0 before slot 1 (tryClaimSlot's role==0 branch),
// so a particle returned slot 0 had ZERO existing chain bonds (a bare monomer) and one returned
// slot 1 had EXACTLY ONE (a chain end -- the active tip, if it has an owner, or a permanently inert
// unanchored end if it does not). Returns whether the centre bookkeeping accepts this bond; on
// false the caller must roll back the two valence claims tryClaimSlot already made, exactly the
// existing slotJ<0 rollback path already does for a plain valence failure.
fn propagateOnCenter(i: u32, j: u32, slotI: i32, slotJ: i32, catalystId: u32) -> bool {
  if (slotI == 0 && slotJ == 0) {
    // Nucleation: both bare. Needs a FREE catalyst this dispatch's own neighbour walk already found
    // near particle i (catalystId, BOND_NONE if none was within range) to adopt the new tip -- the
    // CAS below is simultaneously the "is it free" check and the claim, so a catalyst already
    // holding some OTHER tip simply fails here rather than needing a separate read-then-claim (the
    // same race-free reasoning tryClaimSlot's own atomics already rest on). j is the arbitrary but
    // deterministic choice of which bare carbon becomes the new tip -- which one does not matter
    // physically, only that both sides of this function agree, and they do (same i/j the valence
    // claims above just used).
    if (catalystId == BOND_NONE) { return false; }
    // Surface growth / adsorption (this task): the fresh tether this claim is about to create
    // (soup/wgsl/step.wgsl's bondedForce reads centerLink at the very next force evaluation) must
    // itself start at REACTION-CONTACT distance (wca_cut(bPairB(...)), the SAME reach every other
    // bond in this file forms at, e.g. bondFormWalk's own catalystNear check) -- catalystId was
    // only ever found within that distance of PARTICLE i, not necessarily of j, the arbitrary tip
    // choice here, so this is not automatic. Checking against the FULL FENE divergence radius
    // (P.r_inf) instead of contact distance was tried first and measured to blow up: fene_dv(r)
    // grows steeply as r approaches r_inf (a stiff spring, not a soft one), so a tether allowed to
    // START anywhere up to just under r_inf can begin already deep in that steep region, and one
    // explicit-Euler kick (soup/wgsl/step.wgsl's kick_drift_wrap_main) at that force magnitude can
    // overshoot r_inf outright, at which point fene_dv's own sign flips and the pair accelerates
    // apart instead of together -- an exponential runaway (measured: max particle drift diverged
    // to 1e4-1e9 sigma within a handful of steps of the FIRST such over-close claim). Every
    // covalent bond in this file has always formed at exactly THIS tighter contact distance and
    // has never shown this instability -- mirroring that convention here, not inventing a new one.
    let dNuc = bMi3(pos2[j].xyz - pos2[catalystId].xyz, GB.box.xyz);
    if (length(dNuc) >= wca_cut(bPairB(pos2[j].w, BP.catalystKind))) { return false; }
    if (!centerCas(catalystId, BOND_NONE, j)) { return false; }
    if (!centerCas(j, BOND_NONE, catalystId)) {
      // The tip side's own claim lost a race (some other thread claimed j's centerLink first,
      // impossible for j itself under the i<j dedupe but not for a DIFFERENT rule's event touching
      // j concurrently) -- undo the catalyst claim rather than leave it pointing at a tip that does
      // not point back, the same both-or-neither discipline tryClaimSlot's own rollback embodies.
      centerCas(catalystId, j, BOND_NONE);
      return false;
    }
    // Surface growth / adsorption (this task): a fresh claim is a successful event -- start this
    // centre's hold-timeout clock (desorbTimeout below) at zero.
    atomicStore(&centerHeldSteps[catalystId], 0u);
    return true;
  }
  var tipOld: u32;
  var tipNew: u32;
  if (slotI == 1 && slotJ == 0) { tipOld = i; tipNew = j; }
  else if (slotI == 0 && slotJ == 1) { tipOld = j; tipNew = i; }
  else {
    // Both already existing chain ends (slotI==1 && slotJ==1): two independently-growing chains
    // meeting mid-bulk. Deliberately disallowed, not merely unhandled -- see data/soup.json's
    // cc_bond basis for why (reassigning an entire chain's ownership would need an O(chain length)
    // walk this per-pair kernel has no way to do locally; FTT growth is monomer insertion, not
    // chain-chain coupling, so refusing this is not a physics loss).
    return false;
  }
  var owner = centerOf(tipOld);
  // Surface growth / adsorption (this task): RE-ADSORPTION. tipOld having no owner is ambiguous by
  // construction (this model tracks nothing that would tell "the permanently-inert distal end of a
  // chain that has always grown from its OTHER end" apart from "an end this file's own desorption
  // paths, desorbStretch/desorbTimeout, released mid-growth" -- both are simply one existing chain
  // bond, no current owner). Refusing every owner-less end outright (this task's FIRST version,
  // measured) makes the second case a PERMANENT dead end the instant it is ever desorbed: neither
  // this function's own extension logic nor terminateOnCenter can act on an end with no owner, so a
  // desorbed-but-unfinished chain could never grow OR be capped again -- measured as ~1108-1110
  // timeout desorptions against just 1-6 real terminations over 60000 steps (adsorption-report.md),
  // i.e. the timeout valve was silently killing chains, not merely freeing centres. The physical
  // picture this model already commits to (a growing chain stays ADSORBED until it desorbs) already
  // implies the reverse is possible too -- a desorbed intermediate re-adsorbing onto a DIFFERENT
  // free site -- so this treats every owner-less chain end as re-adsorbable by whichever free
  // catalyst this dispatch's own candidate walk already found (catalystId), not just a bare monomer.
  var reclaiming = false;
  if (owner == BOND_NONE) {
    if (catalystId == BOND_NONE) { return false; }
    owner = catalystId; // tentative: not yet claimed, only used below to evaluate distance/CAS targets
    reclaiming = true;
  }
  // Growth must happen ON that owner specifically, not merely near SOME catalyst -- checked by
  // direct distance to the owner's OWN current position, stronger than the generic
  // catalystNear/catalystId (which only proves *a* catalyst is near, not that it is this chain's
  // own). An ALREADY-tethered pair (not reclaiming) may sit anywhere within its own thermal
  // fluctuation around the FENE spring's equilibrium, so that case uses the wider
  // wca_cut(bPairB(...))+P.wc reach (centerWithinReach) -- but a RECLAIM is itself a FRESH tether
  // (soup/wgsl/step.wgsl's bondedForce will apply fene_dv to it starting next force evaluation),
  // exactly like nucleation's own fresh claim above, so it needs the SAME tighter reaction-contact
  // check for the SAME measured reason (checking the wider reach here first blew up: a fresh tether
  // allowed to start anywhere up to that wider radius can begin deep in fene_dv's own steep region
  // and overshoot past P.r_inf on the very next kick).
  if (reclaiming) {
    let dReclaim = bMi3(pos2[tipOld].xyz - pos2[catalystId].xyz, GB.box.xyz);
    if (length(dReclaim) >= wca_cut(bPairB(pos2[tipOld].w, BP.catalystKind))) { return false; }
  } else {
    if (!centerWithinReach(tipOld, owner)) { return false; }
  }
  // Surface growth / adsorption (this task): same reasoning, and the same measured instability, as
  // the nucleation branch above -- the re-pointed tether (owner<->tipNew) must itself start at
  // REACTION-CONTACT distance, not merely within the wider reach that gates whether this ATTEMPT
  // may proceed at all (checking against P.r_inf here was the version that blew up, see the
  // nucleation branch's own comment for the full mechanism). tipNew is only ever a cc_bond
  // candidate already within reaction-contact distance of tipOld (bondFormWalk's own
  // matchRule/wca_cut check), so this rejects just the rarer case where that contact-close carbon
  // still happens to be too far from the OWNER itself -- retried next cycle, exactly like the reach
  // check just above.
  let dNew = bMi3(pos2[tipNew].xyz - pos2[owner].xyz, GB.box.xyz);
  if (length(dNew) >= wca_cut(bPairB(pos2[tipNew].w, BP.catalystKind))) { return false; }
  // Perform the reclaim itself only now that both geometry checks passed -- nothing to roll back if
  // either failed above, since claiming happens after, not before, both distance checks.
  if (reclaiming) {
    if (!centerCas(owner, BOND_NONE, tipOld)) { return false; }
    if (!centerCas(tipOld, BOND_NONE, owner)) {
      centerCas(owner, tipOld, BOND_NONE);
      return false;
    }
  }
  // Move the tip: both re-points go through the catalyst's OWN slot as the single arbitration
  // point -- whichever of possibly several racing bare monomers gets here first for this exact
  // owner wins (its CAS on centerLink[owner] succeeds), every loser's CAS simply fails and its own
  // bond attempt is rejected, not corrupted. Following a reclaim, `owner` already holds `tipOld`
  // (the CAS just above), so this is the same "move" transition centerCas always performs, not a
  // special case.
  if (!centerCas(owner, tipOld, tipNew)) {
    if (reclaiming) { centerCas(tipOld, owner, BOND_NONE); }
    return false;
  }
  if (!centerCas(tipNew, BOND_NONE, owner)) {
    centerCas(owner, tipNew, tipOld);
    // false must mean no side effect, exactly as every other rejection in this file guarantees --
    // if this was a reclaim, undo it FULLY (both sides), not just the move: leaving owner pointing
    // at tipOld here (a "successful re-adsorption" the return value never reported) would be a
    // silent, unreported side effect on an attempt this function is about to say failed.
    if (reclaiming) {
      centerCas(owner, tipOld, BOND_NONE);
      centerCas(tipOld, owner, BOND_NONE);
    }
    return false;
  }
  centerCas(tipOld, owner, BOND_NONE);
  // Surface growth / adsorption (this task): a successful extension (reclaim or not) is an event --
  // reset this centre's hold-timeout clock to zero (see desorbTimeout below).
  atomicStore(&centerHeldSteps[owner], 0u);
  return true;
}

// Termination (co_bond): `carbon` is the candidate chain end (role 1, the head-slot claim already
// succeeded). May only succeed on the SAME centre that has been holding this carbon as its active
// tip since nucleation -- an un-anchored carbon (never nucleated, or already the inert distal end of
// a longer chain, see propagateOnCenter's own comment) cannot be capped at all, which is exactly the
// fix for kinetic-growth-report.md's "a lone carbon capped before it ever reaches a centre" finding.
// On success the centre is freed for a new chain.
fn terminateOnCenter(carbon: u32) -> bool {
  let owner = centerOf(carbon);
  if (owner == BOND_NONE) { return false; }
  // Same reach as propagateOnCenter's own owner-proximity check -- see centerWithinReach above.
  if (!centerWithinReach(carbon, owner)) { return false; }
  if (!centerCas(carbon, owner, BOND_NONE)) { return false; }
  centerCas(owner, carbon, BOND_NONE);
  // Surface growth / adsorption (this task): the centre is now free -- its hold-timeout clock is
  // moot (desorbTimeout returns immediately once centerOf(owner)==BOND_NONE), but zeroed here too
  // for the same "leave no stale state behind a release" hygiene the desorption paths below follow.
  atomicStore(&centerHeldSteps[owner], 0u);
  return true;
}

// Surface growth / adsorption (this task, adsorption-report.md), desorption path 1 ("the
// adsorption bond is stretched beyond the bonded range" -- the task's own wording): even with a
// real FENE tether holding tip and centre together (soup/wgsl/step.wgsl's bondedForce), an extreme
// thermal fluctuation could in principle separate them past FENE's own divergence radius (P.r_inf,
// engine/wgsl/forces.wgsl's fene_dv) before the spring's own steeply rising restoring force has
// pulled them back -- fene_dv(r) is only well-behaved for r < P.r_inf; beyond it the formula's own
// denominator changes sign and the spring would push the pair APART instead of together, an
// unphysical regime this valve exists to make unreachable in practice. Checked from the CARBON's
// own side, unconditionally, every bond-dispatch cycle (bondFormDecide below), independent of
// whether any bond-forming CANDIDATE was found nearby at all -- the exact gap that let the
// predecessor task's own deadlock happen: a tip that has drifted fully out of contact range with
// everything never triggers a bond-forming candidate, so a check nested only inside candidate
// handling would never re-fire on it.
fn desorbStretch(carbon: u32, owner: u32) {
  let d = bMi3(pos2[carbon].xyz - pos2[owner].xyz, GB.box.xyz);
  if (length(d) >= P.r_inf) {
    if (centerCas(carbon, owner, BOND_NONE)) {
      centerCas(owner, carbon, BOND_NONE);
      atomicStore(&centerHeldSteps[owner], 0u);
      atomicAdd(&desorbEvents[0], 1u);
    }
  }
}

// Surface growth / adsorption (this task), desorption path 2, the deadlock-impossible-by-
// construction guarantee ("a centre has held the same chain for longer than a stated number of
// steps without an event" -- the task's own wording): a centre that has gone longer than
// BP.adsorptionParams.x REAL steps without a single successful propagation or termination event on
// the chain it holds (both reset centerHeldSteps to 0, see propagateOnCenter/terminateOnCenter)
// releases it regardless of distance -- the safety valve that survives even a pathological
// configuration the stretch check above would never trip (the pair staying close enough, just
// never both satisfying valence + Metropolis + a partner arriving at the same instant). Called once
// per bond-dispatch cycle for every CATALYST particle (never a carbon -- the clock belongs to the
// centre, since a centre, not a tip, is the resource a fresh nucleation elsewhere needs freed).
fn desorbTimeout(catalyst: u32) {
  let tip = centerOf(catalyst);
  if (tip == BOND_NONE) { return; }
  let inc = u32(BP.adsorptionParams.y);
  let prev = atomicAdd(&centerHeldSteps[catalyst], inc);
  if (prev + inc > u32(BP.adsorptionParams.x)) {
    if (centerCas(catalyst, tip, BOND_NONE)) {
      centerCas(tip, catalyst, BOND_NONE);
      atomicStore(&centerHeldSteps[catalyst], 0u);
      atomicAdd(&desorbEvents[1], 1u);
    }
  }
}

// Capacity of the deferred-candidate buffer bond_form_main fills during its single neighbour-cell
// walk. A carbon's own valence caps it at 3 live bonds ever, so needing more than this many
// GEOMETRICALLY-eligible candidates within one contact shell in one step to find its next bond is
// already a crowded edge case; a truncation there costs a retry next step (a small kinetic
// slowdown in the most crowded configurations), not a bias in the equilibrium bond count, since
// which candidates get dropped does not depend on outcome.
const CANDIDATE_CAP: u32 = 6u;

struct BondWalkResult {
  catalystNear: u32, // bool as u32: WGSL struct members used across a function return are fine
                      // either way, u32 keeps this struct trivially copyable like its array members.
  // Surface growth (this file's header): the specific id of the (first-found, same policy as
  // catalystNear) catalyst within range of particle i, BOND_NONE if catalystNear==0 -- used ONLY by
  // propagateOnCenter's nucleation branch (which needs a specific catalyst to claim), never as a
  // substitute for the owner-proximity checks propagateOnCenter/terminateOnCenter do for an
  // ALREADY-anchored tip.
  catalystId: u32,
  nCand: u32,
  candJ: array<u32, CANDIDATE_CAP>,
  candRule: array<u32, CANDIDATE_CAP>,
};

// perf2-report.md, candidates (a)+(b), factored the same way soup/wgsl/step.wgsl's soupForceWalk
// is: `useSorted=true` reads posSortedRO[k] (candidate (b)), `useSorted=false` reads
// pos2[cellIdx[k]] (the pre-(b) scattered read), so both can be A/B-measured on ONE
// implementation. Candidate (a)'s walk radius (GB.dims.w) applies identically either way.
fn bondFormWalk(i: u32, xi: vec3<f32>, ti: f32, box: vec3<f32>, dims: vec3<i32>, useSorted: bool) -> BondWalkResult {
  var result: BondWalkResult;
  result.catalystNear = 0u;
  result.catalystId = BOND_NONE;
  result.nCand = 0u;
  let c = cell_coord(xi, GB.dims.xyz, box);
  let R = i32(GB.dims.w);
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
            xj = posSortedRO[k].xyz;
            tj = posSortedRO[k].w;
          } else {
            xj = pos2[j].xyz;
            tj = pos2[j].w;
          }
          if (result.catalystNear == 0u && tj == BP.catalystKind) {
            let dc = bMi3(xi - xj, box);
            if (length(dc) < wca_cut(bPairB(ti, BP.catalystKind))) {
              result.catalystNear = 1u;
              result.catalystId = j;
            }
          }
          if (j > i && result.nCand < CANDIDATE_CAP) {
            let ruleIdx = matchRule(ti, tj);
            if (ruleIdx >= 0 && !hasBondTo(i, j)) {
              let d = bMi3(xi - xj, box);
              if (length(d) < wca_cut(bPairB(ti, tj))) {
                result.candJ[result.nCand] = j;
                result.candRule[result.nCand] = u32(ruleIdx);
                result.nCand = result.nCand + 1u;
              }
            }
          }
        }
      }
    }
  }
  return result;
}

// Deferred decision pass shared by both entry points below: the requiresCatalyst gate and the two
// Metropolis draws only run once catalystNear is fully resolved by bondFormWalk, exactly as
// before -- this function does no neighbour-cell walk of its own, so it needs no useSorted
// parameter.
fn bondFormDecide(i: u32, ti: f32, rngIn: u32, w: BondWalkResult) {
  var rng = rngIn;
  // Surface growth / adsorption (this task): desorption safety valves run unconditionally, every
  // bond-dispatch cycle, for whichever kind this particle is -- BEFORE the candidate loop below,
  // and independent of whether any candidate was found nearby this cycle at all. See
  // desorbStretch/desorbTimeout's own comments for why that independence is the whole point (a
  // fully drifted-away tip never generates a bond-forming candidate for anyone to check against).
  if (ti == BP.catalystKind) {
    desorbTimeout(i);
  } else {
    let owner = centerOf(i);
    if (owner != BOND_NONE) {
      desorbStretch(i, owner);
    }
  }
  for (var ci = 0u; ci < w.nCand; ci = ci + 1u) {
    let j = w.candJ[ci];
    let r = w.candRule[ci];
    let tj = pos2[j].w;
    if (vget(BP.requiresCatalyst, r) > 0.5 && w.catalystNear == 0u) { continue; }
    rng = bondPcg(rng);
    if (bondUniform01(rng) >= vget(BP.attemptProbForm, r)) { continue; }
    rng = bondPcg(rng);
    if (bondUniform01(rng) >= vget(BP.acceptProbForm, r)) { continue; }
    let roleI = roleOf(r, ti);
    let roleJ = roleOf(r, tj);
    let slotI = tryClaimSlot(i, roleI, j);
    if (slotI < 0) { continue; }
    let slotJ = tryClaimSlot(j, roleJ, i);
    if (slotJ < 0) {
      releaseSlot(i, slotI);
      continue;
    }
    // Surface growth (this file's header): valence alone is not enough any more -- cc_bond
    // (propagation, roleI==roleJ==0) and co_bond (termination, one side role 1 the carbon, the
    // other role 2 the head) must also clear the centre-ownership gate below. A failure here rolls
    // back the two valence claims exactly like a plain slotJ<0 valence failure would.
    var centerOk: bool;
    if (roleI == 0u && roleJ == 0u) {
      centerOk = propagateOnCenter(i, j, slotI, slotJ, w.catalystId);
    } else if (roleI == 1u) {
      centerOk = terminateOnCenter(i);
    } else {
      centerOk = terminateOnCenter(j);
    }
    if (!centerOk) {
      releaseSlot(i, slotI);
      releaseSlot(j, slotJ);
      continue;
    }
    atomicAdd(&bondEvents[r * 2u + 0u], 1u);
  }
  bondRng[i] = rng;
}

@compute @workgroup_size(64)
fn bond_form_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let dims = vec3<i32>(GB.dims.xyz);
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;
  let w = bondFormWalk(i, xi, ti, box, dims, true);
  bondFormDecide(i, ti, bondRng[i], w);
}

@compute @workgroup_size(64)
fn bond_form_main_unsorted(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let dims = vec3<i32>(GB.dims.xyz);
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;
  let w = bondFormWalk(i, xi, ti, box, dims, false);
  bondFormDecide(i, ti, bondRng[i], w);
}

@compute @workgroup_size(64)
fn bond_break_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let ti = pos2[i].w;
  var rng = bondRng[i];
  let base = i * 3u;
  for (var s = 0u; s < 3u; s = s + 1u) {
    let partner = atomicLoad(&bondSlots[base + s]);
    if (partner == BOND_NONE) { continue; }
    if (partner < i) { continue; } // dedupe: only the smaller-indexed side breaks an existing bond
    let tj = pos2[partner].w;
    let ruleIdx = matchRule(ti, tj);
    if (ruleIdx < 0) { continue; }
    let r = u32(ruleIdx);
    rng = bondPcg(rng);
    if (bondUniform01(rng) >= vget(BP.attemptProbBreak, r)) { continue; }
    rng = bondPcg(rng);
    // acceptProbBreak came from rules.ts's acceptanceProbability(breakRule, kT) =
    // exp(-energyKT/kT) for the SAME energyKT the paired bond rule used -- never gated on
    // BP.requiresCatalyst, regardless of the rule's own flag: breaking must not depend on
    // catalyst proximity even if a future rule set requiresCatalyst on a break, so that check is
    // simply never written here rather than checked and expected to stay false.
    if (bondUniform01(rng) >= vget(BP.acceptProbBreak, r)) { continue; }
    atomicStore(&bondSlots[base + s], BOND_NONE);
    releasePartnerSlot(partner, i);
    atomicAdd(&bondEvents[r * 2u + 1u], 1u);
  }
  bondRng[i] = rng;
}

// perf2-report.md, candidate (c): the SAME Verlet list soup/wgsl/step.wgsl's soup_force_list_main
// reads (built by soup_build_verlet_list_main, a geometric superset of the raw interaction range
// including the requiresCatalyst check's own reach and every rule's wca_cut(pairB) reach -- both
// bounded by interactionRange, which the list's own VL.x=interactionRange+skin always covers) --
// so bond formation candidate collection can iterate the list instead of re-walking cells too, one
// fewer O(candidates) walk per step, not two.
@group(1) @binding(19) var<uniform> VL: vec4<f32>;
@group(1) @binding(14) var<storage, read> verletList: array<u32>;
@group(1) @binding(15) var<storage, read> verletCount: array<u32>;

fn bondFormWalkList(i: u32, xi: vec3<f32>, ti: f32, box: vec3<f32>) -> BondWalkResult {
  var result: BondWalkResult;
  result.catalystNear = 0u;
  result.catalystId = BOND_NONE;
  result.nCand = 0u;
  let cap = u32(VL.y);
  let count = verletCount[i];
  for (var s = 0u; s < count; s = s + 1u) {
    let j = verletList[i * cap + s];
    let tj = pos2[j].w;
    if (result.catalystNear == 0u && tj == BP.catalystKind) {
      let dc = bMi3(xi - pos2[j].xyz, box);
      if (length(dc) < wca_cut(bPairB(ti, BP.catalystKind))) {
        result.catalystNear = 1u;
        result.catalystId = j;
      }
    }
    if (j > i && result.nCand < CANDIDATE_CAP) {
      let ruleIdx = matchRule(ti, tj);
      if (ruleIdx >= 0 && !hasBondTo(i, j)) {
        let d = bMi3(xi - pos2[j].xyz, box);
        if (length(d) < wca_cut(bPairB(ti, tj))) {
          result.candJ[result.nCand] = j;
          result.candRule[result.nCand] = u32(ruleIdx);
          result.nCand = result.nCand + 1u;
        }
      }
    }
  }
  return result;
}

@compute @workgroup_size(64)
fn bond_form_list_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let box = GB.box.xyz;
  let xi = pos2[i].xyz;
  let ti = pos2[i].w;
  let w = bondFormWalkList(i, xi, ti, box);
  bondFormDecide(i, ti, bondRng[i], w);
}
