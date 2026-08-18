// The neighbour grid and Verlet list, with their completeness guards. Split out of
// soup/src/sim.ts's createSoup (file-size rule in CLAUDE.md) -- the grid-rebuild/Verlet-rebuild
// dispatch sequences and the runtime drift-safety/overflow check, moved verbatim. Creation-time
// derivation of the SAME completeness guarantees (walk-radius coverage, list coverage, drift bound)
// lives in soup/src/soup-plan.ts's deriveGridGeometry instead -- that one throws before any GPU
// resource exists; this file's assertVerletSafety re-checks the analytical bound against a REAL
// measured trajectory, after every step()/box-scale call.

import { readBack } from '../../engine/src/gpu'
import type { SoupRuntime } from './soup-runtime'

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
      `список Верле: verletList.listCapacity=${verlet.listCapacity} было недостаточно -- ` +
        `хотя бы одна частица нашла больше кандидатов, чем вмещает список (данные могли быть тихо отброшены)`,
    )
  }
  const rawDrift = await readBack(device, buf.maxDriftSqBuf, 4)
  const drift = Math.sqrt(Math.max(0, rawDrift[0]))
  const bound = verlet.skin / 2
  if (drift > bound + 1e-6) {
    throw new Error(
      `список Верле: измеренный дрейф ${drift.toFixed(4)} превышает skin/2=${bound.toFixed(4)} -- ` +
        `аналитическая граница (verletList.basis) не сработала для реальной траектории, перестройка была недостаточно частой`,
    )
  }
}
