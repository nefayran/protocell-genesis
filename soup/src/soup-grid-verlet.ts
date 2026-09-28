// The neighbour grid and Verlet list, with their completeness guards. Split out of
// soup/src/sim.ts's createSoup (file-size rule in CLAUDE.md) -- the grid-rebuild/Verlet-rebuild
// dispatch sequences and the runtime drift-safety/overflow check, moved verbatim. Creation-time
// derivation of the SAME completeness guarantees (walk-radius coverage, list coverage, drift bound)
// lives in soup/src/soup-plan.ts's deriveGridGeometry instead -- that one throws before any GPU
// resource exists; this file's assertVerletSafety re-checks the analytical bound against a REAL
// measured trajectory, after every step()/box-scale call.

import { readBack } from '../../engine/src/gpu'
import type { SoupRuntime } from './soup-runtime'

// Task 'long-range-electrostatics' (2026-08-20): the dedicated long-range electrostatic list --
// compact the titratable beads' indices (deterministically, in index order: see
// soup/wgsl/electrostatics-long.wgsl for why not an atomic append), then build their own neighbour
// list at rc_es + skin. Rebuilt from INSIDE encodeGridRebuild, i.e. at exactly the same cadence and
// the same moments the main Verlet list is, so the same drift bound (skin/2, measured every step by
// soup_max_drift_main) covers it and no separate guard is needed. Skipped entirely without charge.
function encodeEsListRebuild(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  if (!rt.protonation?.es.enabled || rt.esHeads === 0) return
  pass.setPipeline(rt.pipe.buildHeadIndex)
  pass.setBindGroup(1, rt.bind.buildHeadIndexBind)
  pass.dispatchWorkgroups(1)
  pass.setPipeline(rt.pipe.buildEsList)
  pass.setBindGroup(1, rt.bind.buildEsListBind)
  pass.dispatchWorkgroups(rt.wgEsHeads)
}

export function encodeGridRebuild(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  const { pipe, bind } = rt
  pass.setPipeline(pipe.clearCounts)
  pass.setBindGroup(0, bind.clearCountsBind)
  pass.dispatchWorkgroups(rt.grid.wgCells)
  pass.setPipeline(pipe.count)
  pass.setBindGroup(0, bind.countBind)
  pass.dispatchWorkgroups(rt.wgN)
  pass.setPipeline(pipe.prefix)
  pass.setBindGroup(0, bind.prefixBind)
  pass.dispatchWorkgroups(1)
  pass.setPipeline(pipe.fill)
  pass.setBindGroup(0, bind.fillBind)
  pass.dispatchWorkgroups(rt.wgN)
  // perf2-report.md, candidate (b): gather positions into cell-sorted order right after `fill`
  // finalises `cellsBuf` (the permutation) for this step -- must run before soup_force_main/
  // bond_form_main (both read posSortedRW/posSortedRO this same step) and after fill (its own
  // permutation is this gather's input). Skipped entirely when sortedGather=false -- the
  // *_unsorted pipeline variants never read posSortedRW/posSortedRO, so this dispatch would be
  // pure waste (and its own cost must not be charged against the (a)-only measurement).
  if (rt.sortedGather) {
    pass.setPipeline(pipe.gatherSorted)
    pass.setBindGroup(1, bind.gatherSortedBind)
    pass.dispatchWorkgroups(rt.wgN)
  }
  encodeEsListRebuild(rt, pass)
}

// perf2-report.md, candidate (c): rebuild the coarse grid (unchanged, cheap -- cellDivisor is
// independent of this candidate) THEN the Verlet list from it, snapshot the positions this
// rebuild used (the drift-safety reference point), and zero the running max-drift counter --
// ORDER matters: snapshot/reset must follow the list build that just consumed the CURRENT
// positions, not precede it.
export function encodeVerletRebuild(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  encodeGridRebuild(rt, pass)
  const { pipe, bind } = rt
  pass.setPipeline(pipe.buildVerletList)
  pass.setBindGroup(1, bind.buildVerletListBind)
  pass.dispatchWorkgroups(rt.wgN)
  pass.setPipeline(pipe.snapshotPositions)
  pass.setBindGroup(1, bind.snapshotPositionsBind)
  pass.dispatchWorkgroups(rt.wgN)
  pass.setPipeline(pipe.resetMaxDrift)
  pass.setBindGroup(1, bind.resetMaxDriftBind)
  pass.dispatchWorkgroups(1)
}

// O(N), same order of cost as kick_drift_wrap_main -- see soup_max_drift_main's own header for
// why this per-step cost is acceptable (it is what lets the drift-safety guard be checked for
// real, not just trusted from the analytical bound in soup/src/soup-plan.ts's deriveGridGeometry).
export function encodeMaxDrift(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  pass.setPipeline(rt.pipe.maxDrift)
  pass.setBindGroup(1, rt.bind.maxDriftBind)
  pass.dispatchWorkgroups(rt.wgN)
}

// perf2-report.md, candidate (c): checked once per chunk (not once per step -- the whole point
// of STEP_CHUNK, soup/src/soup-integrate.ts, is to keep the GPU timeline free of per-step CPU
// round trips, and a live per-step drift-triggered rebuild would reintroduce exactly that), at the
// natural onSubmittedWorkDone() sync point step() already has. Throws immediately and does not
// continue if either the drift bound or the list capacity was ever violated -- this candidate's
// whole completeness guarantee rests on catching that for real, not trusting the analytical bound
// computed at creation time.
export async function assertVerletSafety(rt: SoupRuntime): Promise<void> {
  const { device, buf, verlet } = rt
  const rawOverflow = await readBack(device, buf.verletOverflowBuf, 4)
  const overflow = new Uint32Array(rawOverflow.buffer, rawOverflow.byteOffset, 1)[0]
  if (overflow !== 0) {
    throw new Error(
      `Verlet list: verletList.listCapacity=${verlet.listCapacity} was insufficient -- ` +
        `at least one particle found more candidates than the list holds (data may have been silently dropped)`,
    )
  }
  // Task 'long-range-electrostatics' (2026-08-20): the SAME guard for the dedicated long-range list.
  // esMeta[1] is set by either kernel -- by the compaction if there are more titratable beads than
  // headIdxBuf was sized for, or by the list build if one head found more neighbours within
  // rc_es+skin than longRangeListCapacity holds. Both are silent physics loss if not thrown on:
  // a truncated long-range list is exactly the truncated interaction this task exists to remove.
  if (rt.protonation?.es.enabled && rt.esHeads > 0) {
    const rawEs = await readBack(rt.device, rt.buf.esMetaBuf, 8)
    const meta = new Uint32Array(rawEs.buffer, rawEs.byteOffset, 2)
    if (meta[1] !== 0) {
      throw new Error(
        `long-range electrostatics list: longRangeListCapacity=${rt.protonation.es.listCapacity} ` +
          `or the headIdx size (${rt.esHeads}) was insufficient at rc_es=${rt.protonation.es.cutoff.toFixed(4)} ` +
          `(heads found=${meta[0]}) -- data may have been silently dropped`,
      )
    }
  }
  const rawDrift = await readBack(device, buf.maxDriftSqBuf, 4)
  const drift = Math.sqrt(Math.max(0, rawDrift[0]))
  const bound = verlet.skin / 2
  // Task 'loud-failure-and-liquid-water' (2026-08-20): the measured drift itself can be NaN, and
  // when it is, EVERY comparison below is false -- which is exactly how three separate divergences
  // (rho_tot 0.8444 / 0.75 / 0.80) passed this guard in silence while producing 68 049 / 60 495 /
  // 366 282 non-finite coordinates. soup/src/soup-health.ts's assertStateFinite now runs BEFORE this
  // function at every call site, so a NaN drift should be unreachable; this branch exists so that if
  // it ever IS reached the failure is loud instead of a false pass.
  if (!Number.isFinite(drift)) {
    throw new Error(
      `Verlet list: the measured drift is not a number (${drift}) -- the state is already non-finite, ` +
        `see soup/src/soup-health.ts's assertStateFinite`,
    )
  }
  if (drift > bound + 1e-6) {
    throw new Error(
      `Verlet list: the measured drift ${drift.toFixed(4)} exceeds skin/2=${bound.toFixed(4)} -- ` +
        `the analytic bound (verletList.basis) did not hold for the real trajectory, the rebuild was not frequent enough`,
    )
  }
}
