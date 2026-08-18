// Shared mutable runtime context for engine/src/sim.ts's split modules — split out by
// responsibility (file-size rule, root CLAUDE.md). Type only, no logic: stands in for the
// original createSystem()'s closure over local `let`/`const` variables (same pattern
// soup/src/soup-runtime.ts already established for the sibling soup engine split). Every
// stateful module below (sim-buffers.ts, sim-bindgroups.ts, sim-integrate.ts, sim-forces.ts,
// sim-area-move.ts, sim-measure.ts) takes an `EngineRuntime` as an explicit parameter instead of
// closing over these values implicitly.

import type { Pipelines } from './sim-pipelines'
import type { CreateSystemOpts } from './sim-types'
import type { Params } from './params'

export interface EngineRuntime {
  device: GPUDevice
  pipe: Pipelines
  opts: CreateSystemOpts
  /** Immutable, exactly what the system was CREATED with (see setLiveParams's doc comment in
   * sim-types.ts for why this is kept distinct from `livep`). */
  p: Params
  /** Mutable copy of the params actually in force — retunable live via setLiveParams(). */
  livep: Params
  rng: () => number
  N: number
  /** Fixed for the system's lifetime: r_c + w_c (the largest interaction range), see
   * sim-grid-geometry.ts's computeDims doc comment. */
  cellSize: number
  /** The wc value the neighbor grid was actually sized for — the ceiling setLiveParams must
   * enforce. */
  builtForWc: number

  /** Live box, mutated in place by areaMove(). */
  liveBox: [number, number, number]
  dims: [number, number, number]
  ncells: number
  totalSteps: number
  neighborBuildMs: number

  // --- buffers, fixed size for the system's lifetime -------------------------------------------
  posBuf: GPUBuffer
  velBuf: GPUBuffer
  forceBuf: GPUBuffer
  potentialBuf: GPUBuffer
  cellsBuf: GPUBuffer
  rngBuf: GPUBuffer
  paramsUniform: GPUBuffer
  gridUniform: GPUBuffer
  boxUniform: GPUBuffer

  // --- buffers sized `ncells` — reallocated by resizeGrid() whenever dims changes --------------
  countsBuf: GPUBuffer
  cellStartBuf: GPUBuffer
  cursorBuf: GPUBuffer

  // --- bind groups built once, never rebound ---------------------------------------------------
  forceGridGroup0: GPUBindGroup
  forceBruteGroup0: GPUBindGroup
  forceBruteGroup1: GPUBindGroup
  kickBind: GPUBindGroup
  driftBind: GPUBindGroup
  wrapBind: GPUBindGroup
  thermostatBind: GPUBindGroup

  // --- bind groups that reference the ncells-sized buffers — rebuilt by rebindGridDependent()
  // every time those buffers are reallocated, since a WebGPU bind group is a fixed reference to
  // specific buffer objects.
  clearCountsBind: GPUBindGroup
  countBind: GPUBindGroup
  prefixBind: GPUBindGroup
  fillBind: GPUBindGroup
  forceGridGroup1: GPUBindGroup
  wgCells: number

  wgN: number
}
