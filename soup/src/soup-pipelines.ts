// Compute pipeline creation, device-keyed cache. Split out of soup/src/sim.ts (file-size rule in
// CLAUDE.md) -- moved verbatim, including the module-level `cached` singleton and its own doc
// comment on why pipelines (unlike bind groups) are safe to share across systems on the same
// device.
//
// bond.wgsl itself was ALSO split by responsibility (soup/wgsl/bond-common.wgsl, bond-valence.wgsl,
// bond-adsorption.wgsl, bond-dispatch.wgsl, bond-verlet.wgsl) -- concatenated back into one shader
// module below in the exact same order the original single file had them in, so the compiled
// module is unchanged: WGSL does not care about source-file boundaries, only the token stream, and
// no entry-point name or binding index moved.

import forcesWgsl from '../../engine/wgsl/forces.wgsl?raw'
import neighborWgsl from '../../engine/wgsl/neighbor.wgsl?raw'
import stepWgsl from '../wgsl/step.wgsl?raw'
// Task 'electrostatics' (2026-08-20): TWO more files, and the ORDER below is load-bearing twice over.
// electrostatics.wgsl declares chargeRO/ES and esForce, which step.wgsl's nonbondedSoup CALLS, and
// WGSL has no forward declarations -- so it must come BEFORE step.wgsl. verlet.wgsl is the Verlet
// list responsibility split OUT of step.wgsl (which stood at 596 against CLAUDE.md's hard 600, and
// the rule is "split first, then add"); its soup_force_list_main calls step.wgsl's
// nonbondedSoup/bondedForce, so it must come AFTER. Neither changes the token stream the previously
// compiled entry points were built from, so no binding index and no kernel body moved.
import electrostaticsWgsl from '../wgsl/electrostatics.wgsl?raw'
import verletWgsl from '../wgsl/verlet.wgsl?raw'
// Task 'loud-failure-and-liquid-water' (2026-08-20): two NEW files, not additions to step.wgsl --
// that file stands at 596 lines against CLAUDE.md's hard 600 limit, and the rule is "split first,
// then add". Order matters twice over: health.wgsl's soupNonFinite() is called by relax.wgsl, and
// BOTH reuse pos2/velRW/posRW/forceRO/frozenRO/GB declared by forces.wgsl and step.wgsl above, so
// they must come last. WGSL does not care about source-file boundaries, only the token stream, so
// this changes nothing about the previously compiled entry points -- no binding index moved, no
// existing kernel body was touched.
import healthWgsl from '../wgsl/health.wgsl?raw'
import relaxWgsl from '../wgsl/relax.wgsl?raw'
import bondCommonWgsl from '../wgsl/bond-common.wgsl?raw'
import bondValenceWgsl from '../wgsl/bond-valence.wgsl?raw'
import bondAdsorptionWgsl from '../wgsl/bond-adsorption.wgsl?raw'
import bondDispatchWgsl from '../wgsl/bond-dispatch.wgsl?raw'
import bondVerletWgsl from '../wgsl/bond-verlet.wgsl?raw'

const bondWgsl = [bondCommonWgsl, bondValenceWgsl, bondAdsorptionWgsl, bondDispatchWgsl, bondVerletWgsl].join('\n')

export interface SoupPipelines {
  device: GPUDevice
  sortedGather: boolean
  soupForce: GPUComputePipeline
  /** perf2-report.md correctness gate: O(N^2) reference force, no grid -- see
   * soup/src/soup-readback.ts's forces()/forcesBruteForce(), mirroring engine/src/sim.ts's own pair. */
  soupForceBrute: GPUComputePipeline
  kickDriftWrap: GPUComputePipeline
  kickThermostat: GPUComputePipeline
  bondForm: GPUComputePipeline
  bondBreak: GPUComputePipeline
  clearCounts: GPUComputePipeline
  count: GPUComputePipeline
  prefix: GPUComputePipeline
  fill: GPUComputePipeline
  /** perf2-report.md, candidate (b): gathers positions into cell-sorted order (using the same
   * permutation fill_main already produces in `cellIdx`) so the O(candidates) neighbour walk in
   * soup_force_main/bond_form_main reads a contiguous array instead of scattering through `pos2`
   * at arbitrary original indices. */
  gatherSorted: GPUComputePipeline
  /** perf2-report.md, STEP 1 diagnosis: counts candidates examined and pairs within the actual
   * interaction range for one dispatch of the SAME cell walk soup_force_main runs -- never used in
   * the real step loop, only by soup/src/soup-readback.ts's forceCandidateStatsDEBUG. */
  forceStats: GPUComputePipeline
  /** perf2-report.md, candidate (c): Verlet list build/maintenance kernels -- only ever dispatched
   * when verletList.enabled is true. */
  buildVerletList: GPUComputePipeline
  snapshotPositions: GPUComputePipeline
  resetMaxDrift: GPUComputePipeline
  maxDrift: GPUComputePipeline
  soupForceList: GPUComputePipeline
  bondFormList: GPUComputePipeline
  /** Task 'loud-failure-and-liquid-water' (2026-08-20): the non-finite state guard (soup/wgsl/
   * health.wgsl) and the cold-start minimiser's one iteration (soup/wgsl/relax.wgsl). Neither is
   * dispatched inside encodeOneIntegrationStep: the scan runs once per step()-CHUNK at the sync point
   * assertVerletSafety already uses, and the minimiser runs only before step 1, from
   * soup/src/soup-relax.ts. */
  resetNonFinite: GPUComputePipeline
  scanNonFinite: GPUComputePipeline
  relaxStep: GPUComputePipeline
}

let cached: SoupPipelines | undefined

// perf2-report.md, candidate (b): which entry point to compile for the force/bond-form kernels --
// see NeighborGrid.sortedGather's doc comment (soup/src/rules.ts) for why this is a pipeline
// choice, not a runtime branch. Threaded into the (device-keyed) pipeline cache key too: this
// engine only ever runs with ONE data/soup.json per process (loaded once at module import), so in
// practice the cache is never asked for the other variant on the same device, but keying on it
// explicitly documents that dependency rather than leaving it implicit.
export function getSoupPipelines(device: GPUDevice, sortedGather: boolean): SoupPipelines {
  if (cached && cached.device === device && cached.sortedGather === sortedGather) return cached
  const forceModule = device.createShaderModule({
    code: `${forcesWgsl}\n${electrostaticsWgsl}\n${stepWgsl}\n${verletWgsl}\n${healthWgsl}\n${relaxWgsl}`,
  })
  const bondModule = device.createShaderModule({ code: `${forcesWgsl}\n${bondWgsl}` })
  const neighborModule = device.createShaderModule({ code: neighborWgsl })
  const cp = (module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } })
  cached = {
    device,
    sortedGather,
    soupForce: cp(forceModule, sortedGather ? 'soup_force_main' : 'soup_force_main_unsorted'),
    soupForceBrute: cp(forceModule, 'soup_force_brute_main'),
    kickDriftWrap: cp(forceModule, 'kick_drift_wrap_main'),
    kickThermostat: cp(forceModule, 'kick_thermostat_main'),
    bondForm: cp(bondModule, sortedGather ? 'bond_form_main' : 'bond_form_main_unsorted'),
    bondBreak: cp(bondModule, 'bond_break_main'),
    clearCounts: cp(neighborModule, 'clear_counts_main'),
    count: cp(neighborModule, 'count_main'),
    prefix: cp(neighborModule, 'prefix_main'),
    fill: cp(neighborModule, 'fill_main'),
    gatherSorted: cp(forceModule, 'soup_gather_sorted_main'),
    forceStats: cp(forceModule, 'soup_force_stats_main'),
    buildVerletList: cp(forceModule, 'soup_build_verlet_list_main'),
    snapshotPositions: cp(forceModule, 'soup_snapshot_positions_main'),
    resetMaxDrift: cp(forceModule, 'soup_reset_max_drift_main'),
    maxDrift: cp(forceModule, 'soup_max_drift_main'),
    soupForceList: cp(forceModule, 'soup_force_list_main'),
    bondFormList: cp(bondModule, 'bond_form_list_main'),
    resetNonFinite: cp(forceModule, 'soup_reset_nonfinite_main'),
    scanNonFinite: cp(forceModule, 'soup_scan_nonfinite_main'),
    relaxStep: cp(forceModule, 'soup_relax_step_main'),
  }
  return cached
}
