// Valence: atomic claim-and-rollback, INCLUDING the head-terminal condition in the same functions
// by design, not by omission of a split. See bond-common.wgsl's header for this file's place in the
// bond.wgsl decomposition.
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
// avoid. This is why the head-terminal condition is NOT split into its own file even though the
// project's split-by-responsibility rule would otherwise suggest it: the claim and the terminal
// check share one atomic transaction (claim, then check-and-maybe-roll-back), and separating them
// into two functions across a file boundary would force either a second CAS (re-introducing the
// exact race this design avoids) or passing the claimed slot across an awkward seam for no benefit --
// this is the "hard-won correctness machinery" the splitting task was told not to touch.
// Reversibility: breaking a C-O bond off a terminal-head carbon returns the system to a
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

fn tryClaimSlot(particle: u32, role: u32, partner: u32) -> i32 {
  let base = particle * 3u;
  let headTerminalOnly = BP.headTerminalOnly > 0.5;
  if (role == 0u) {
    // Chain-pool claim (a C-C bond). headPlacement.terminalOnly's condition (2): a carbon that
    // already carries a head (slot 2 occupied) may not accept a further chain bond, since that
    // would bury the head mid-chain. Claim the chain slot first (unconditionally, same as before),
    // then check slot 2 and roll back if this particle turns out to already have a head -- never
    // the other order, see this file's header for why a pre-check would race.
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
