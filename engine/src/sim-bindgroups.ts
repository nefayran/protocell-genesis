// Bind-group construction for engine/src/sim.ts, including rebind-on-resize logic — split out by
// responsibility (file-size rule, root CLAUDE.md). Moved verbatim from the original sim.ts; same
// binding indices, same bind-group layout groups, same order.

import type { EngineRuntime } from './sim-runtime'

function bindGroup(device: GPUDevice, pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[]): GPUBindGroup {
  return device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries })
}

function buf(b: GPUBuffer) {
  return { buffer: b }
}

/** Bind groups that reference the ncells-sized buffers (countsBuf/cellStartBuf/cursorBuf) — must
 * be rebuilt every time those buffers are reallocated (sim-buffers.ts's resizeGrid), since a
 * WebGPU bind group is a fixed reference to specific buffer objects. Also called once at
 * creation time. */
export function rebindGridDependent(rt: EngineRuntime): void {
  rt.clearCountsBind = bindGroup(rt.device, rt.pipe.clearCounts, 0, [
    { binding: 0, resource: buf(rt.gridUniform) },
    { binding: 2, resource: buf(rt.countsBuf) },
  ])
  rt.countBind = bindGroup(rt.device, rt.pipe.count, 0, [
    { binding: 0, resource: buf(rt.gridUniform) },
    { binding: 1, resource: buf(rt.posBuf) },
    { binding: 2, resource: buf(rt.countsBuf) },
  ])
  rt.prefixBind = bindGroup(rt.device, rt.pipe.prefix, 0, [
    { binding: 0, resource: buf(rt.gridUniform) },
    { binding: 2, resource: buf(rt.countsBuf) },
    { binding: 4, resource: buf(rt.cellStartBuf) },
    { binding: 5, resource: buf(rt.cursorBuf) },
  ])
  rt.fillBind = bindGroup(rt.device, rt.pipe.fill, 0, [
    { binding: 0, resource: buf(rt.gridUniform) },
    { binding: 1, resource: buf(rt.posBuf) },
    { binding: 3, resource: buf(rt.cellsBuf) },
    { binding: 5, resource: buf(rt.cursorBuf) },
  ])
  rt.forceGridGroup1 = bindGroup(rt.device, rt.pipe.forceGrid, 1, [
    { binding: 0, resource: buf(rt.posBuf) },
    { binding: 1, resource: buf(rt.forceBuf) },
    { binding: 2, resource: buf(rt.potentialBuf) },
    { binding: 3, resource: buf(rt.gridUniform) },
    { binding: 4, resource: buf(rt.cellStartBuf) },
    { binding: 5, resource: buf(rt.cellsBuf) },
  ])
  rt.wgCells = Math.ceil(rt.ncells / 64)
}

/** Bind groups built once at creation time from buffers that never get reallocated — never
 * rebound afterward (unlike rebindGridDependent's group above). */
export function buildStaticBindGroups(rt: EngineRuntime): void {
  rt.forceGridGroup0 = bindGroup(rt.device, rt.pipe.forceGrid, 0, [{ binding: 0, resource: buf(rt.paramsUniform) }])
  rt.forceBruteGroup0 = bindGroup(rt.device, rt.pipe.forceBrute, 0, [{ binding: 0, resource: buf(rt.paramsUniform) }])
  rt.forceBruteGroup1 = bindGroup(rt.device, rt.pipe.forceBrute, 1, [
    { binding: 0, resource: buf(rt.posBuf) },
    { binding: 1, resource: buf(rt.forceBuf) },
    { binding: 2, resource: buf(rt.potentialBuf) },
    { binding: 3, resource: buf(rt.gridUniform) },
  ])

  rt.kickBind = bindGroup(rt.device, rt.pipe.kick, 0, [
    { binding: 0, resource: buf(rt.paramsUniform) },
    { binding: 2, resource: buf(rt.velBuf) },
    { binding: 3, resource: buf(rt.forceBuf) },
  ])
  rt.driftBind = bindGroup(rt.device, rt.pipe.drift, 0, [
    { binding: 0, resource: buf(rt.paramsUniform) },
    { binding: 1, resource: buf(rt.posBuf) },
    { binding: 2, resource: buf(rt.velBuf) },
  ])
  rt.wrapBind = bindGroup(rt.device, rt.pipe.wrap, 0, [
    { binding: 1, resource: buf(rt.posBuf) },
    { binding: 5, resource: buf(rt.boxUniform) },
  ])
  rt.thermostatBind = bindGroup(rt.device, rt.pipe.thermostat, 0, [
    { binding: 0, resource: buf(rt.paramsUniform) },
    { binding: 2, resource: buf(rt.velBuf) },
    { binding: 4, resource: buf(rt.rngBuf) },
  ])
}
