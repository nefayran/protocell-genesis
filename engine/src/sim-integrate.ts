// The Langevin/velocity-Verlet integration step for engine/src/sim.ts — split out by
// responsibility (file-size rule, root CLAUDE.md). Moved verbatim from the original sim.ts: same
// kick/drift/wrap/grid-rebuild/kick/thermostat sequence, one command buffer per step(n) call (no
// per-step CPU<->GPU round trip).

import { encodeForceGrid } from './sim-forces'
import type { EngineRuntime } from './sim-runtime'

export function encodeGridRebuild(rt: EngineRuntime, pass: GPUComputePassEncoder): void {
  pass.setPipeline(rt.pipe.clearCounts)
  pass.setBindGroup(0, rt.clearCountsBind)
  pass.dispatchWorkgroups(rt.wgCells)
  pass.setPipeline(rt.pipe.count)
  pass.setBindGroup(0, rt.countBind)
  pass.dispatchWorkgroups(rt.wgN)
  pass.setPipeline(rt.pipe.prefix)
  pass.setBindGroup(0, rt.prefixBind)
  pass.dispatchWorkgroups(1)
  pass.setPipeline(rt.pipe.fill)
  pass.setBindGroup(0, rt.fillBind)
  pass.dispatchWorkgroups(rt.wgN)
}

export async function rebuildGridTimed(rt: EngineRuntime): Promise<void> {
  const t0 = performance.now()
  await rebuildGridUntimed(rt)
  rt.neighborBuildMs = performance.now() - t0
}

/** Same rebuild, without touching `neighborBuildMs`. Used everywhere the rebuild is incidental
 * (warm-up, the two energy evaluations inside every area-move trial) so the number Task 9 reports
 * as a neighbor-grid rebuild time stays the one measured by an explicit, deliberate rebuild
 * instead of whatever the last Monte Carlo trial happened to cost. */
export async function rebuildGridUntimed(rt: EngineRuntime): Promise<void> {
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeGridRebuild(rt, pass)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  await rt.device.queue.onSubmittedWorkDone()
}

export async function step(rt: EngineRuntime, n: number): Promise<void> {
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  for (let k = 0; k < n; k++) {
    pass.setPipeline(rt.pipe.kick)
    pass.setBindGroup(0, rt.kickBind)
    pass.dispatchWorkgroups(rt.wgN)

    pass.setPipeline(rt.pipe.drift)
    pass.setBindGroup(0, rt.driftBind)
    pass.dispatchWorkgroups(rt.wgN)

    pass.setPipeline(rt.pipe.wrap)
    pass.setBindGroup(0, rt.wrapBind)
    pass.dispatchWorkgroups(rt.wgN)

    encodeGridRebuild(rt, pass)
    encodeForceGrid(rt, pass)

    pass.setPipeline(rt.pipe.kick)
    pass.setBindGroup(0, rt.kickBind)
    pass.dispatchWorkgroups(rt.wgN)

    pass.setPipeline(rt.pipe.thermostat)
    pass.setBindGroup(0, rt.thermostatBind)
    pass.dispatchWorkgroups(rt.wgN)
  }
  pass.end()
  rt.device.queue.submit([enc.finish()])
  await rt.device.queue.onSubmittedWorkDone()
  rt.totalSteps += n
}
