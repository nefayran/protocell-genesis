// GPU buffer allocation and the neighbor-grid resize path for engine/src/sim.ts — split out by
// responsibility (file-size rule, root CLAUDE.md). Moved verbatim from the original sim.ts.
//
// resizeGrid() takes an `onResized` callback rather than importing sim-bindgroups.ts directly —
// same reason soup/src/soup-buffers.ts's resizeSoupGrid does (see that split's own report): keeps
// "buffer lifetime" and "bind-group rebuilding" independently owned modules with no import cycle
// between them. sim.ts wires the callback to sim-bindgroups.ts's rebindGridDependent().

import { storageBuffer } from './gpu'
import { paramsToUniform, type Params } from './params'
import { computeDims } from './sim-grid-geometry'
import type { EngineRuntime } from './sim-runtime'

export interface EngineBuffers {
  posBuf: GPUBuffer
  velBuf: GPUBuffer
  forceBuf: GPUBuffer
  potentialBuf: GPUBuffer
  cellsBuf: GPUBuffer
  countsBuf: GPUBuffer
  cellStartBuf: GPUBuffer
  cursorBuf: GPUBuffer
  rngBuf: GPUBuffer
  paramsUniform: GPUBuffer
  gridUniform: GPUBuffer
  boxUniform: GPUBuffer
}

/** Allocates every buffer createSystem() needs before pipelines/bind groups can be built. Sized
 * exactly as the original inline allocation was: `cellsBuf` is sized N (one slot per bead, sorted
 * by cell), independent of the grid resize below, so it is never reallocated; `countsBuf`/
 * `cellStartBuf`/`cursorBuf` are sized `ncells` and ARE reallocated by resizeGrid(). */
export function allocateBuffers(
  device: GPUDevice,
  N: number,
  ncells: number,
  positions0: Float32Array,
  velocities0: Float32Array,
  rngState0: Uint32Array<ArrayBuffer>,
  p: Params,
): EngineBuffers {
  const posBuf = storageBuffer(device, positions0)
  const velBuf = storageBuffer(device, velocities0)
  const forceBuf = storageBuffer(device, new Float32Array(N * 4))
  const potentialBuf = storageBuffer(device, new Float32Array(N))
  const cellsBuf = storageBuffer(device, new Float32Array(N))
  const countsBuf = storageBuffer(device, new Float32Array(ncells))
  const cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
  const cursorBuf = storageBuffer(device, new Float32Array(ncells))
  const rngBuf = device.createBuffer({
    size: rngState0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(rngBuf, 0, rngState0)

  const paramsUniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(paramsUniform, 0, paramsToUniform(p))

  const gridUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const boxUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })

  return { posBuf, velBuf, forceBuf, potentialBuf, cellsBuf, countsBuf, cellStartBuf, cursorBuf, rngBuf, paramsUniform, gridUniform, boxUniform }
}

export function writeGridUniforms(
  device: GPUDevice,
  gridUniform: GPUBuffer,
  boxUniform: GPUBuffer,
  b: [number, number, number],
  d: [number, number, number],
): void {
  const bytes = new ArrayBuffer(32)
  new Uint32Array(bytes, 0, 4).set([d[0], d[1], d[2], 0])
  new Float32Array(bytes, 16, 4).set([b[0], b[1], b[2], 0])
  device.queue.writeBuffer(gridUniform, 0, bytes)
  device.queue.writeBuffer(boxUniform, 0, new Float32Array([b[0], b[1], b[2], 0]))
}

/** Recomputes dims for `newBox`; if the cell count changed, destroys and reallocates the
 * ncells-sized buffers and calls `onResized` (wired by the caller to rebindGridDependent()) so
 * everything that references them gets rebuilt. Called by areaMove() after every trial (accepted
 * or reverted) so the grid always matches the box actually in use — see sim-grid-geometry.ts's
 * computeDims doc comment for why a frozen grid silently breaks under compression. */
export async function resizeGrid(rt: EngineRuntime, newBox: [number, number, number], onResized: () => void): Promise<void> {
  const newDims = computeDims(newBox, rt.cellSize)
  if (newDims[0] !== rt.dims[0] || newDims[1] !== rt.dims[1] || newDims[2] !== rt.dims[2]) {
    rt.countsBuf.destroy()
    rt.cellStartBuf.destroy()
    rt.cursorBuf.destroy()
    rt.dims = newDims
    rt.ncells = rt.dims[0] * rt.dims[1] * rt.dims[2]
    rt.countsBuf = storageBuffer(rt.device, new Float32Array(rt.ncells))
    rt.cellStartBuf = storageBuffer(rt.device, new Float32Array(rt.ncells + 1))
    rt.cursorBuf = storageBuffer(rt.device, new Float32Array(rt.ncells))
    onResized()
  }
  writeGridUniforms(rt.device, rt.gridUniform, rt.boxUniform, newBox, rt.dims)
}
