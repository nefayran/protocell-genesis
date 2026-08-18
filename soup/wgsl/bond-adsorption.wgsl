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
// This is carried by ONE new buffer, `centerLink` (bond-common.wgsl, next to `bondSlots`): a single
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
// slot design, and centerLink's own single-u32-per-particle layout is what this section's code
// assumes throughout.

// Surface growth (this file's header): centerLink's own compareExchange primitive, the exact same
// discipline tryClaimSlot/releasePartnerSlot (bond-valence.wgsl) already use, generalised to the
// single-slot catalyst<->tip buffer. `particle` is either a catalyst (holding a tip carbon id) or a
// carbon (holding its owning catalyst id) -- the two kinds never collide since a rule never matches
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
// own side, unconditionally, every bond-dispatch cycle (bondFormDecide, bond-dispatch.wgsl),
// independent of whether any bond-forming CANDIDATE was found nearby at all -- the exact gap that let
// the predecessor task's own deadlock happen: a tip that has drifted fully out of contact range with
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
