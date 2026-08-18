// Grid-walk candidate collection and the three grid-walk entry points (bond_form_main,
// bond_form_main_unsorted, bond_break_main). See bond-common.wgsl's header for this file's place in
// the bond.wgsl decomposition; the Verlet-list variant of the candidate walk lives in
// bond-verlet.wgsl instead of here.

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
  // Surface growth (bond-adsorption.wgsl's header): the specific id of the (first-found, same policy
  // as catalystNear) catalyst within range of particle i, BOND_NONE if catalystNear==0 -- used ONLY
  // by propagateOnCenter's nucleation branch (which needs a specific catalyst to claim), never as a
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

// Deferred decision pass shared by every entry point below AND by bond-verlet.wgsl's
// bond_form_list_main: the requiresCatalyst gate and the two Metropolis draws only run once
// catalystNear is fully resolved by whichever candidate walk called this, exactly as before -- this
// function does no neighbour-cell walk of its own, so it needs no useSorted parameter.
fn bondFormDecide(i: u32, ti: f32, rngIn: u32, w: BondWalkResult) {
  var rng = rngIn;
  // Surface growth / adsorption (bond-adsorption.wgsl): desorption safety valves run unconditionally,
  // every bond-dispatch cycle, for whichever kind this particle is -- BEFORE the candidate loop
  // below, and independent of whether any candidate was found nearby this cycle at all. See
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
    // Surface growth (bond-adsorption.wgsl): valence alone is not enough any more -- cc_bond
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
