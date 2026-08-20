// The Verlet neighbour list: its build kernel, the drift-safety snapshot/guard kernels, and the
// list-reading force kernel. Split out of soup/wgsl/step.wgsl (2026-08-20, task 'electrostatics',
// CLAUDE.md's 400-600 line rule) -- that file stood at 596 lines against the hard 600 limit, and the
// rule is "split first, then add". Lines 466-596 of step.wgsl, moved verbatim -- same entry-point
// names, same @group/@binding indices, same bodies. WGSL does not care about source-file boundaries,
// only the token stream, so soup/src/soup-pipelines.ts concatenating this AFTER step.wgsl produces
// the same compiled module the single file did. The ONLY post-move edit is inside
// soup_force_list_main: two tokens (`let qi = chargeRO[i];` and the two extra arguments to
// nonbondedSoup) added for this task's screened-Coulomb term, exactly as in the grid-walk and
// brute-force paths -- and identically zero on any system without electrostatics.
//
// Why THIS is the seam. The list is a self-contained responsibility: five kernels
// (soup_build_verlet_list_main / soup_snapshot_positions_main / soup_reset_max_drift_main /
// soup_max_drift_main / soup_force_list_main), five bindings (14-19) nothing outside this file
// touches, and one owner on the TS side (soup/src/soup-grid-verlet.ts). It depends on step.wgsl
// (nonbondedSoup/bondedForce/mi3) and nothing depends on it, so it can only be the tail.

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
  let qi = chargeRO[i];
  let count = verletCount[i];
  for (var s = 0u; s < count; s = s + 1u) {
    let j = verletList[i * cap + s];
    f = f + nonbondedSoup(xi, pos2[j].xyz, ti, pos2[j].w, box, qi, chargeRO[j]);
  }
  outForce[i] = vec4<f32>(f, 0.0);
}
