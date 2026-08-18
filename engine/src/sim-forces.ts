// TS-side dispatch of the bonded (WCA-on-bonded-pairs included — a deliberate, hard-won decision,
// see forces.wgsl's own header) and non-bonded tail-attraction forces for engine/src/sim.ts —
// split out by responsibility (file-size rule, root CLAUDE.md). This file is intentionally thin:
// the physics itself lives entirely in engine/wgsl/forces.wgsl (force_main / force_brute_main);
// what is here is only which pipeline/bind-groups run and the two measurement entry points
// (forces()/forcesBruteForce()) System exposes. Moved verbatim from the original sim.ts.

import { readBack } from './gpu'
import { rebuildGridTimed } from './sim-integrate'
import type { EngineRuntime } from './sim-runtime'

export function encodeForceGrid(rt: EngineRuntime, pass: GPUComputePassEncoder): void {
  pass.setPipeline(rt.pipe.forceGrid)
  pass.setBindGroup(0, rt.forceGridGroup0)
  pass.setBindGroup(1, rt.forceGridGroup1)
  pass.dispatchWorkgroups(rt.wgN)
}

export function encodeForceBrute(rt: EngineRuntime, pass: GPUComputePassEncoder): void {
  pass.setPipeline(rt.pipe.forceBrute)
  pass.setBindGroup(0, rt.forceBruteGroup0)
  pass.setBindGroup(1, rt.forceBruteGroup1)
  pass.dispatchWorkgroups(rt.wgN)
}

export async function forces(rt: EngineRuntime): Promise<Float32Array> {
  await rebuildGridTimed(rt)
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeForceGrid(rt, pass)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  return readBack(rt.device, rt.forceBuf, rt.N * 16)
}

export async function forcesBruteForce(rt: EngineRuntime): Promise<Float32Array> {
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeForceBrute(rt, pass)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  return readBack(rt.device, rt.forceBuf, rt.N * 16)
}
