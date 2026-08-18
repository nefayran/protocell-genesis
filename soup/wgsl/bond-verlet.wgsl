// perf2-report.md, candidate (c): the SAME Verlet list soup/wgsl/step.wgsl's soup_force_list_main
// reads (built by soup_build_verlet_list_main, a geometric superset of the raw interaction range
// including the requiresCatalyst check's own reach and every rule's wca_cut(pairB) reach -- both
// bounded by interactionRange, which the list's own VL.x=interactionRange+skin always covers) --
// so bond formation candidate collection can iterate the list instead of re-walking cells too, one
// fewer O(candidates) walk per step, not two. See bond-common.wgsl's header for this file's place
// in the bond.wgsl decomposition; bondFormDecide (bond-dispatch.wgsl) is shared unchanged.
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
