// The LOUD non-finite-state guard: read soup/wgsl/health.wgsl's two counters and THROW if either is
// nonzero, naming the step and the count.
//
// This is the missing half of soup/src/soup-grid-verlet.ts's assertVerletSafety, and it is a
// deliberate EXTENSION of that machinery rather than a parallel one: same cadence (once per
// step()-chunk), same sync point (soup/src/soup-integrate.ts's
// `await device.queue.onSubmittedWorkDone()`), same "throw immediately and do not continue"
// discipline, same file-local error wording style. It is checked BEFORE assertVerletSafety, for a
// measured reason: once positions are NaN the measured drift is NaN too, `NaN > skin/2` is false, and
// assertVerletSafety reports a clean run forever (see soup/wgsl/health.wgsl's header for the three
// historical instances this cost). Checking finiteness first means the message names the real cause
// instead of an overflow flag that happens to trip later.
//
// WHY A SEPARATE SUBMIT INSTEAD OF FOLDING THE DISPATCH INTO encodeOneIntegrationStep. The scan needs
// to run ONCE per chunk, not once per step; encoding it inside the per-step sequence would multiply
// its cost by STEP_CHUNK for no gain. Its own tiny command buffer also means every existing call
// site of assertVerletSafety (soup/src/soup-integrate.ts and both of soup/src/soup-box-scale.ts's)
// can get the finiteness check by adding one line, with no change at all to any pass-encoding
// function -- so the integration step's own dispatch sequence is byte-identical to before this task.

import { readBack } from '../../engine/src/gpu'
import type { SoupRuntime } from './soup-runtime'

export interface NonFiniteCount {
  /** Non-finite COMPONENTS (not particles) among the x,y,z of every position -- the same unit the
   * three historical offline scans reported, so this number is directly comparable with them. */
  pos: number
  vel: number
}

/** Dispatches the O(N) exponent scan and reads back its two counters. Pure measurement, throws
 * nothing -- published on SoupSystem as `nonFiniteCount()` so a caller (or a throughput measurement)
 * can pay for the check explicitly and time it. */
export async function scanNonFinite(rt: SoupRuntime): Promise<NonFiniteCount> {
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(rt.pipe.resetNonFinite)
  pass.setBindGroup(1, rt.bind.resetNonFiniteBind)
  pass.dispatchWorkgroups(1)
  pass.setPipeline(rt.pipe.scanNonFinite)
  pass.setBindGroup(1, rt.bind.scanNonFiniteBind)
  pass.dispatchWorkgroups(rt.wgN)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  const raw = await readBack(rt.device, rt.buf.healthBuf, 8)
  const u = new Uint32Array(raw.buffer, raw.byteOffset, 2)
  return { pos: u[0], vel: u[1] }
}

/** THROWS if any position or velocity component of the live state is Inf or NaN. `sinceStep` (the
 * global step this chunk started at) is carried into the message purely so the reader can bracket
 * WHEN the divergence appeared to within one chunk -- the scan itself is a snapshot and cannot know
 * which step inside the chunk did it, and saying so in the message is cheaper than implying a
 * precision the check does not have. */
export async function assertStateFinite(rt: SoupRuntime, sinceStep?: number): Promise<void> {
  const c = await scanNonFinite(rt)
  if (c.pos === 0 && c.vel === 0) return
  const window = sinceStep !== undefined ? ` (appeared in the step interval ${sinceStep}..${rt.live.globalStep})` : ''
  throw new Error(
    `non-finite state at step=${rt.live.globalStep}: non-finite components of positions=${c.pos}, velocities=${c.vel} ` +
      `of ${rt.N * 3}${window} -- the computation diverged (Inf/NaN), any further numbers from this run are meaningless`,
  )
}
