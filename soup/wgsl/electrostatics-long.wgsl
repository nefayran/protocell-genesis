// Task 'long-range-electrostatics' (2026-08-20): the four KERNELS of the dedicated long-range
// electrostatic pass. A separate file from soup/wgsl/electrostatics.wgsl for one hard reason and one
// soft one. HARD: these kernels call mi3() and read pos2/outForce/GB, all of which are declared in
// step.wgsl / forces.wgsl, and WGSL has no forward declarations -- so this file must be concatenated
// AFTER step.wgsl, exactly as soup/wgsl/verlet.wgsl is, while electrostatics.wgsl must come BEFORE
// it (nonbondedSoup calls its esForceNear). SOFT: CLAUDE.md's 400-600 line rule -- electrostatics.wgsl
// carries the interaction, its truncation and the whole WHY; this file carries the plumbing that
// makes the range affordable.
//
// The rationale for every choice here -- why a real-space cutoff at a multiple of lambda_D rather
// than Ewald/PPPM or Wolf, why a head-only list rather than a longer shared one, why the compaction
// is serial, and what each of ES2's four components is -- is in electrostatics.wgsl's own ES2 header
// and in data/soup.json's electrostatics.basis item 12. Read those before changing anything here.

@compute @workgroup_size(1)
fn soup_build_head_index_main() {
  let n = arrayLength(&pos2);
  let kind = ES2.w;
  var c = 0u;
  let cap = arrayLength(&headIdx);
  for (var i = 0u; i < n; i = i + 1u) {
    if (pos2[i].w == kind) {
      if (c < cap) { headIdx[c] = i; }
      c = c + 1u;
    }
  }
  atomicStore(&esMeta[0], c);
  atomicStore(&esMeta[1], select(0u, 1u, c > cap));
}

@compute @workgroup_size(64)
fn soup_build_es_list_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  let nh = atomicLoad(&esMeta[0]);
  if (h >= nh) { return; }
  let box = GB.box.xyz;
  let i = headIdx[h];
  let xi = pos2[i].xyz;
  let listRange = ES2.y;
  let cap = u32(ES2.z);
  var count = 0u;
  for (var g = 0u; g < nh; g = g + 1u) {
    let j = headIdx[g];
    if (j == i) { continue; }
    let d = mi3(xi - pos2[j].xyz, box);
    if (length(d) < listRange) {
      if (count < cap) {
        esList[h * cap + count] = j;
        count = count + 1u;
      } else {
        atomicStore(&esMeta[1], 1u);
      }
    }
  }
  esCount[h] = count;
}

// ACCUMULATES into outForce -- it runs immediately after whichever force kernel WROTE outForce, in
// the same compute pass (WebGPU orders dispatches within a pass and inserts the memory barrier), so
// this is `+=` by construction. Dispatched from inside encodeSoupForce/Brute/List themselves
// (soup/src/soup-integrate.ts) rather than at their call sites, so no force path -- step, box scale,
// evaporation, cold-start relax, protonation recompute, readback -- can silently miss it.
@compute @workgroup_size(64)
fn soup_es_force_far_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x;
  let nh = atomicLoad(&esMeta[0]);
  if (h >= nh) { return; }
  let i = headIdx[h];
  let qi = chargeRO[i];
  if (qi == 0.0) { return; }
  let box = GB.box.xyz;
  let xi = pos2[i].xyz;
  let cap = u32(ES2.z);
  let count = esCount[h];
  var f = vec3<f32>(0.0);
  for (var s = 0u; s < count; s = s + 1u) {
    let j = esList[h * cap + s];
    let d = mi3(xi - pos2[j].xyz, box);
    f = f + esForceFar(d, length(d), qi, chargeRO[j]);
  }
  outForce[i] = outForce[i] + vec4<f32>(f, 0.0);
}

// The correctness reference for the pass above: same physics, no list, O(N^2) over every pair --
// mirroring soup_force_brute_main's relationship to soup_force_main. tests/soup-forces.test.ts
// compares the two, which is what proves the head-only list is COMPLETE over the new range.
@compute @workgroup_size(64)
fn soup_es_force_far_brute_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let n = arrayLength(&pos2);
  if (i >= n) { return; }
  let qi = chargeRO[i];
  if (qi == 0.0) { return; }
  let box = GB.box.xyz;
  let xi = pos2[i].xyz;
  var f = vec3<f32>(0.0);
  for (var j = 0u; j < n; j = j + 1u) {
    if (j == i) { continue; }
    let d = mi3(xi - pos2[j].xyz, box);
    f = f + esForceFar(d, length(d), qi, chargeRO[j]);
  }
  outForce[i] = outForce[i] + vec4<f32>(f, 0.0);
}
