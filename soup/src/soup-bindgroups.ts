// Bind-group construction, including the rebind-on-resize logic. Split out of soup/src/sim.ts's
// createSoup (file-size rule in CLAUDE.md) -- every bind group, and rebindGridDependent's own
// grouping rationale, moved verbatim: not one binding index or buffer reference changed.

import type { SoupPipelines } from './soup-pipelines'
import type { SoupBuffers } from './soup-buffers'

function bind(pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[], device: GPUDevice): GPUBindGroup {
  return device.createBindGroup({ label: `${pipeline.label}@${group}`, layout: pipeline.getBindGroupLayout(group), entries })
}
function buf(b: GPUBuffer): GPUBindingResource {
  return { buffer: b }
}

/** A per-particle vec4 buffer bound over only its FIRST `activeN` particles.
 *
 * Task 'evaporation' (2026-08-20): every soup kernel bounds itself with `arrayLength()` of one of
 * these four buffers (soup/wgsl/step.wgsl's posRW/velRW/posSortedRW/posAtRebuildRW and forces.wgsl's
 * pos2 -- grep `arrayLength` across soup/wgsl and engine/wgsl: there is no particle-count uniform at
 * all), and WGSL's arrayLength() of a runtime-sized array is the length of the BOUND RANGE, not of
 * the whole GPUBuffer. So binding these four with an explicit `size` is what lets the live particle
 * count fall (a dry phase, where solvent beads have left the system) and rise again (rehydration)
 * with NO buffer reallocated, NO shader change, and no index of any surviving particle moving -- the
 * removed beads are simply outside every kernel's own bound. The buffers stay allocated at the wet
 * N, which is also what makes rehydration cheap: the slots are still there to write into.
 *
 * `activeN * 16` is exactly the whole buffer for a system that never evaporates (storageBuffer sizes
 * posBuf/velBuf/posSortedBuf/posAtRebuildBuf at N*16 bytes), so every existing caller's arrayLength
 * is bit-identical to what it was before this parameter existed. */
function pbuf(b: GPUBuffer, activeN: number): GPUBindingResource {
  return { buffer: b, size: activeN * 16 }
}

export interface SoupBindGroups {
  // --- bind groups that reference the ncells-sized buffers (countsBuf/cellStartBuf/cursorBuf) ----
  // Grouped together here -- even though they belong to several different kernels (grid rebuild,
  // soup force, bond formation, the perf2 diagnostic, the Verlet-list build) -- because ALL of them
  // must be rebuilt together by rebindGridDependent() (below) whenever countsBuf/cellStartBuf/
  // cursorBuf are reallocated: a WebGPU bind group is a fixed reference to specific buffer OBJECTS,
  // so reassigning the buffer that holds a reference does nothing to a bind group already created
  // against the old one. Exactly the discipline engine/src/sim.ts's own rebindGridDependent()/
  // resizeGrid() established for the membrane's area move, generalised here to this soup's own
  // larger set of grid-dependent kernels (it has bond formation and a Verlet-list build the
  // membrane engine does not). soupForceBruteGroup1/soupForceListGroup1/bondFormListGroup1 (below,
  // NOT here) reference NEITHER buffer -- the brute-force kernel walks no grid at all, and the two
  // *List kernels walk the Verlet list (sized N*listCapacity, independent of ncells) instead of the
  // coarse grid -- so none of those three ever need rebinding on a resize.
  clearCountsBind: GPUBindGroup
  countBind: GPUBindGroup
  prefixBind: GPUBindGroup
  fillBind: GPUBindGroup
  soupForceGroup1: GPUBindGroup
  bondFormGroup1: GPUBindGroup
  forceStatsGroup1: GPUBindGroup
  buildVerletListBind: GPUBindGroup

  // --- fixed bind groups, never rebuilt on a resize ------------------------------------------
  gatherSortedBind: GPUBindGroup
  soupForceGroup0: GPUBindGroup
  soupForceBruteGroup0: GPUBindGroup
  soupForceBruteGroup1: GPUBindGroup
  kickDriftWrapGroup0: GPUBindGroup
  kickDriftWrapGroup1: GPUBindGroup
  kickThermostatGroup0: GPUBindGroup
  kickThermostatGroup1: GPUBindGroup
  bondFormGroup0: GPUBindGroup
  bondFormGroup2: GPUBindGroup
  bondBreakGroup1: GPUBindGroup
  bondBreakGroup2: GPUBindGroup
  forceStatsGroup0: GPUBindGroup
  forceStatsGroup3: GPUBindGroup
  snapshotPositionsBind: GPUBindGroup
  resetMaxDriftBind: GPUBindGroup
  maxDriftBind: GPUBindGroup
  soupForceListGroup0: GPUBindGroup
  soupForceListGroup1: GPUBindGroup
  bondFormListGroup0: GPUBindGroup
  bondFormListGroup1: GPUBindGroup
  bondFormListGroup2: GPUBindGroup
  /** Task 'loud-failure-and-liquid-water' (2026-08-20): none of these three reference an ncells-sized
   * buffer, so like snapshotPositions/resetMaxDrift/maxDrift they never need rebinding on a resize.
   * relaxStepBind DOES reference the grid UNIFORM (binding 3, for the box the wrap uses) -- that
   * buffer object is never reallocated by resizeSoupGrid (only rewritten), so it is not
   * grid-dependent in the bind-group sense either. */
  resetNonFiniteBind: GPUBindGroup
  scanNonFiniteBind: GPUBindGroup
  relaxStepBind: GPUBindGroup
}

/** Rebuilds every bind group that references countsBuf/cellStartBuf/cursorBuf -- called once at
 * creation time and again by soup/src/soup-buffers.ts's resizeSoupGrid (via the `onResized`
 * callback soup/src/sim.ts wires up) whenever a dry-wet box change moves the grid into a different
 * cell-count bracket. Mutates the 8 resizable fields on `bind` in place; the other bind groups
 * (built once by buildBindGroups below) are left untouched. */
export function rebindGridDependent(device: GPUDevice, pipe: SoupPipelines, sb: SoupBuffers, bind_: SoupBindGroups, activeN: number): void {
  // --- grid rebuild (engine/wgsl/neighbor.wgsl, unchanged) --------------------------------------
  bind_.clearCountsBind = bind(pipe.clearCounts, 0, [
    { binding: 0, resource: buf(sb.gridUniform) },
    { binding: 2, resource: buf(sb.countsBuf) },
  ], device)
  bind_.countBind = bind(pipe.count, 0, [
    { binding: 0, resource: buf(sb.gridUniform) },
    { binding: 1, resource: pbuf(sb.posBuf, activeN) },
    { binding: 2, resource: buf(sb.countsBuf) },
  ], device)
  bind_.prefixBind = bind(pipe.prefix, 0, [
    { binding: 0, resource: buf(sb.gridUniform) },
    { binding: 2, resource: buf(sb.countsBuf) },
    { binding: 4, resource: buf(sb.cellStartBuf) },
    { binding: 5, resource: buf(sb.cursorBuf) },
  ], device)
  bind_.fillBind = bind(pipe.fill, 0, [
    { binding: 0, resource: buf(sb.gridUniform) },
    { binding: 1, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.cellsBuf) },
    { binding: 5, resource: buf(sb.cursorBuf) },
  ], device)
  // --- soup force (forces.wgsl + step.wgsl), grid-walk variant ----------------------------------
  bind_.soupForceGroup1 = bind(pipe.soupForce, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 1, resource: buf(sb.forceBuf) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 4, resource: buf(sb.cellStartBuf) },
    { binding: 5, resource: buf(sb.cellsBuf) },
    { binding: 7, resource: buf(sb.bondSlotsBuf) },
    { binding: 8, resource: buf(sb.speciesUniform) },
    { binding: 13, resource: pbuf(sb.posSortedBuf, activeN) },
    // Surface growth / adsorption (adsorption-report.md): centerLink, read-only here
    // (soup/wgsl/step.wgsl's centerLinkRO) for the adsorption tether's own FENE contribution.
    { binding: 20, resource: buf(sb.centerLinkBuf) },
    // Task 'hydrophobic-asymmetry' (2026-08-19): soup/wgsl/step.wgsl's AttrScale table, read by
    // nonbondedSoup's attraction term (this pipeline's own soup_force_main/_unsorted body).
    { binding: 9, resource: buf(sb.attrScaleUniform) },
    // Task 'electrostatics' (2026-08-20): the per-particle charge (soup/wgsl/electrostatics.wgsl's
    // chargeRO) and the ES uniform. Required on exactly the three pipelines whose entry points reach
    // nonbondedSoup -- soupForce, soupForceBrute, soupForceList. `layout: auto` derives each layout
    // from what the entry point actually USES, so adding these to any other group would fail
    // bind-group validation outright (which is how a miswiring surfaces here, not silently).
    { binding: 24, resource: buf(sb.chargeBuf) },
    { binding: 25, resource: buf(sb.esUniformBuf) },
  ], device)
  // --- bond formation (forces.wgsl + bond.wgsl), grid-walk variant -----------------------------
  bind_.bondFormGroup1 = bind(pipe.bondForm, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 4, resource: buf(sb.cellStartBuf) },
    { binding: 5, resource: buf(sb.cellsBuf) },
    { binding: 6, resource: buf(sb.bondSlotsBuf) },
    { binding: 7, resource: buf(sb.speciesUniform) },
    { binding: 8, resource: buf(sb.eventsBuf) },
    { binding: 9, resource: buf(sb.bondRngBuf) },
    { binding: 13, resource: pbuf(sb.posSortedBuf, activeN) },
    { binding: 20, resource: buf(sb.centerLinkBuf) },
    { binding: 21, resource: buf(sb.centerHeldStepsBuf) },
    { binding: 22, resource: buf(sb.desorbEventsBuf) },
  ], device)
  // perf2-report.md, STEP 1 diagnosis: soup_force_stats_main's own group1 -- same grid buffers as
  // soup_force_main's group1 above.
  bind_.forceStatsGroup1 = bind(pipe.forceStats, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 4, resource: buf(sb.cellStartBuf) },
    { binding: 5, resource: buf(sb.cellsBuf) },
    { binding: 8, resource: buf(sb.speciesUniform) },
    // Task 'hydrophobic-asymmetry' (2026-08-19): this diagnostic's own withinAttr gate calls
    // shouldAttract(), which is now DERIVED from the AttrScale table (a zero cell is "does not
    // attract") rather than from the polar/solvent flags alone -- so this group needs binding 9
    // too. Before this task shouldAttract() read no uniform at all and this group deliberately
    // omitted it; leaving it out now fails bind-group validation outright (entry count 5 vs the
    // shader's 6), which is how it was caught -- tests/soup-forces.test.ts asserts the browser
    // logged no GPU warning, so a missing entry surfaces as a test failure, not as a silent zero.
    { binding: 9, resource: buf(sb.attrScaleUniform) },
  ], device)
  // perf2-report.md, candidate (c): builds the Verlet list FROM the coarse grid (cellStartBuf) --
  // must be rebuilt whenever that grid's own buffers are.
  bind_.buildVerletListBind = bind(pipe.buildVerletList, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 4, resource: buf(sb.cellStartBuf) },
    { binding: 5, resource: buf(sb.cellsBuf) },
    { binding: 14, resource: buf(sb.verletListBuf) },
    { binding: 15, resource: buf(sb.verletCountBuf) },
    { binding: 16, resource: buf(sb.verletOverflowBuf) },
    { binding: 19, resource: buf(sb.verletUniform) },
  ], device)
}

/** Builds every bind group createSoup needs -- the resizable set (via rebindGridDependent, above)
 * plus every fixed one that never changes for this system's lifetime. Called exactly once, right
 * after soup/src/soup-buffers.ts's allocateSoupBuffers. */
export function buildBindGroups(device: GPUDevice, pipe: SoupPipelines, sb: SoupBuffers, activeN: number): SoupBindGroups {
  const bind_ = {} as SoupBindGroups
  rebindGridDependent(device, pipe, sb, bind_, activeN)

  // perf2-report.md, candidate (b): gather bind group, only group 1 (no Params uniform needed for
  // a plain copy) -- pos2 (read) + cellIdx (read, the permutation) + posSortedRW (write target).
  // cellsBuf is sized N (the permutation, one slot per bead), never reallocated by resizeSoupGrid --
  // this bind group is unaffected by ncells and does not need rebinding.
  bind_.gatherSortedBind = bind(pipe.gatherSorted, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 5, resource: buf(sb.cellsBuf) },
    { binding: 13, resource: pbuf(sb.posSortedBuf, activeN) },
  ], device)

  // --- soup force + wrap (forces.wgsl + step.wgsl) -----------------------------------------------
  bind_.soupForceGroup0 = bind(pipe.soupForce, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  // perf2-report.md correctness gate: O(N^2) reference, no grid buffers needed at all. Its own
  // group0 -- NOT soupForceGroup0 -- because 'layout: auto' gives every pipeline a DISTINCT layout
  // object even when the referenced uniform is identical; reusing another pipeline's bind group
  // fails WebGPU validation (caught via a page-console listener, not silently -- see perf2-report.md).
  bind_.soupForceBruteGroup0 = bind(pipe.soupForceBrute, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  bind_.soupForceBruteGroup1 = bind(pipe.soupForceBrute, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 1, resource: buf(sb.forceBuf) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 7, resource: buf(sb.bondSlotsBuf) },
    { binding: 8, resource: buf(sb.speciesUniform) },
    { binding: 20, resource: buf(sb.centerLinkBuf) },
    // Water-calibration task (2026-08-19): see soupForceGroup1's own comment above.
    { binding: 9, resource: buf(sb.attrScaleUniform) },
    // Task 'electrostatics' (2026-08-20): the per-particle charge (soup/wgsl/electrostatics.wgsl's
    // chargeRO) and the ES uniform. Required on exactly the three pipelines whose entry points reach
    // nonbondedSoup -- soupForce, soupForceBrute, soupForceList. `layout: auto` derives each layout
    // from what the entry point actually USES, so adding these to any other group would fail
    // bind-group validation outright (which is how a miswiring surfaces here, not silently).
    { binding: 24, resource: buf(sb.chargeBuf) },
    { binding: 25, resource: buf(sb.esUniformBuf) },
  ], device)
  bind_.kickDriftWrapGroup0 = bind(pipe.kickDriftWrap, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  bind_.kickDriftWrapGroup1 = bind(pipe.kickDriftWrap, 1, [
    { binding: 6, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 10, resource: pbuf(sb.velBuf, activeN) },
    { binding: 11, resource: buf(sb.forceBuf) },
    // Task 'clay-surface' (2026-08-19): soup/wgsl/step.wgsl's frozenRO. Both integrator kernels read
    // it, so both of their group-1 layouts require the entry -- omitting it fails bind-group
    // validation outright (entry count vs the shader's), which is how it would be caught rather than
    // by the platelet silently drifting.
    { binding: 21, resource: buf(sb.frozenBuf) },
  ], device)
  bind_.kickThermostatGroup0 = bind(pipe.kickThermostat, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  bind_.kickThermostatGroup1 = bind(pipe.kickThermostat, 1, [
    { binding: 10, resource: pbuf(sb.velBuf, activeN) },
    { binding: 11, resource: buf(sb.forceBuf) },
    { binding: 12, resource: buf(sb.thermoRngBuf) },
    // Task 'clay-surface': see kickDriftWrapGroup1's own note above.
    { binding: 21, resource: buf(sb.frozenBuf) },
  ], device)

  // --- bond formation/breaking (forces.wgsl + bond.wgsl) -----------------------------------------
  bind_.bondFormGroup0 = bind(pipe.bondForm, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  bind_.bondFormGroup2 = bind(pipe.bondForm, 2, [{ binding: 0, resource: buf(sb.bondParamsUniform) }], device)
  bind_.bondBreakGroup1 = bind(pipe.bondBreak, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 6, resource: buf(sb.bondSlotsBuf) },
    { binding: 8, resource: buf(sb.eventsBuf) },
    { binding: 9, resource: buf(sb.bondRngBuf) },
  ], device)
  bind_.bondBreakGroup2 = bind(pipe.bondBreak, 2, [{ binding: 0, resource: buf(sb.bondParamsUniform) }], device)

  // perf2-report.md, STEP 1 diagnosis: bind groups for soup_force_stats_main -- reuses exactly the
  // same buffers soup_force_main's group0/1 do (it needs P.sigma/P.b_tt/P.wc via Params and the
  // same grid), plus its own group 3 output. forceStatsGroup1 itself lives in rebindGridDependent
  // above (it references cellStartBuf).
  bind_.forceStatsGroup0 = bind(pipe.forceStats, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  bind_.forceStatsGroup3 = bind(pipe.forceStats, 3, [{ binding: 0, resource: buf(sb.statsBuf) }], device)

  // perf2-report.md, candidate (c): Verlet list bind groups. buildVerletListBind itself lives in
  // rebindGridDependent above (it references cellStartBuf); snapshotPositions/resetMaxDrift/
  // maxDrift never reference P (no group 0 needed) -- only the geometry/positions and their own list
  // buffers, none of them ncells-sized, so none need rebinding on a resize.
  bind_.snapshotPositionsBind = bind(pipe.snapshotPositions, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 17, resource: pbuf(sb.posAtRebuildBuf, activeN) },
  ], device)
  bind_.resetMaxDriftBind = bind(pipe.resetMaxDrift, 1, [{ binding: 18, resource: buf(sb.maxDriftSqBuf) }], device)
  bind_.maxDriftBind = bind(pipe.maxDrift, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 17, resource: pbuf(sb.posAtRebuildBuf, activeN) },
    { binding: 18, resource: buf(sb.maxDriftSqBuf) },
  ], device)
  bind_.soupForceListGroup0 = bind(pipe.soupForceList, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  bind_.soupForceListGroup1 = bind(pipe.soupForceList, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 1, resource: buf(sb.forceBuf) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 7, resource: buf(sb.bondSlotsBuf) },
    { binding: 8, resource: buf(sb.speciesUniform) },
    { binding: 14, resource: buf(sb.verletListBuf) },
    { binding: 15, resource: buf(sb.verletCountBuf) },
    { binding: 19, resource: buf(sb.verletUniform) },
    { binding: 20, resource: buf(sb.centerLinkBuf) },
    // Water-calibration task (2026-08-19): see soupForceGroup1's own comment above.
    { binding: 9, resource: buf(sb.attrScaleUniform) },
    // Task 'electrostatics' (2026-08-20): the per-particle charge (soup/wgsl/electrostatics.wgsl's
    // chargeRO) and the ES uniform. Required on exactly the three pipelines whose entry points reach
    // nonbondedSoup -- soupForce, soupForceBrute, soupForceList. `layout: auto` derives each layout
    // from what the entry point actually USES, so adding these to any other group would fail
    // bind-group validation outright (which is how a miswiring surfaces here, not silently).
    { binding: 24, resource: buf(sb.chargeBuf) },
    { binding: 25, resource: buf(sb.esUniformBuf) },
  ], device)
  bind_.bondFormListGroup0 = bind(pipe.bondFormList, 0, [{ binding: 0, resource: buf(sb.paramsUniform) }], device)
  bind_.bondFormListGroup1 = bind(pipe.bondFormList, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 6, resource: buf(sb.bondSlotsBuf) },
    { binding: 7, resource: buf(sb.speciesUniform) },
    { binding: 8, resource: buf(sb.eventsBuf) },
    { binding: 9, resource: buf(sb.bondRngBuf) },
    { binding: 14, resource: buf(sb.verletListBuf) },
    { binding: 15, resource: buf(sb.verletCountBuf) },
    { binding: 19, resource: buf(sb.verletUniform) },
    { binding: 20, resource: buf(sb.centerLinkBuf) },
    { binding: 21, resource: buf(sb.centerHeldStepsBuf) },
    { binding: 22, resource: buf(sb.desorbEventsBuf) },
  ], device)
  bind_.bondFormListGroup2 = bind(pipe.bondFormList, 2, [{ binding: 0, resource: buf(sb.bondParamsUniform) }], device)

  // Task 'loud-failure-and-liquid-water' (2026-08-20). The scan reads BOTH state arrays the
  // integrator writes -- positions through forces.wgsl's own read-only view (binding 0) and
  // velocities through step.wgsl's velRW (binding 10) -- and writes only its own 2-slot counter
  // (binding 22). The minimiser writes positions (binding 6, step.wgsl's posRW), reads the force the
  // real force kernel just produced (binding 11) and the immobility flag (binding 21), and needs the
  // grid uniform (binding 3) for the box its periodic wrap uses plus its own displacement cap
  // (binding 23). Neither kernel references Params, so neither pipeline has a group 0 at all -- the
  // same shape maxDrift/snapshotPositions already have.
  bind_.resetNonFiniteBind = bind(pipe.resetNonFinite, 1, [{ binding: 22, resource: buf(sb.healthBuf) }], device)
  bind_.scanNonFiniteBind = bind(pipe.scanNonFinite, 1, [
    { binding: 0, resource: pbuf(sb.posBuf, activeN) },
    { binding: 10, resource: pbuf(sb.velBuf, activeN) },
    { binding: 22, resource: buf(sb.healthBuf) },
  ], device)
  bind_.relaxStepBind = bind(pipe.relaxStep, 1, [
    { binding: 6, resource: pbuf(sb.posBuf, activeN) },
    { binding: 3, resource: buf(sb.gridUniform) },
    { binding: 11, resource: buf(sb.forceBuf) },
    { binding: 21, resource: buf(sb.frozenBuf) },
    { binding: 23, resource: buf(sb.relaxUniform) },
  ], device)

  return bind_
}
