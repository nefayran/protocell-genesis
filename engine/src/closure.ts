// Task 8: is the membrane CLOSED? Detects an enclosed cavity by flooding the box's grid from
// every OPEN boundary cell (see the note on the periodic-flood choice below) and measuring the
// volume of empty cells the flood never reaches. occupancy()/enclosedVolume() are the CPU
// reference -- the definition of truth this task's WGSL kernel (closure.wgsl) must agree with;
// enclosedVolumeGpuDetailed drives that kernel end to end.
//
// Periodic-flood choice: x and y are periodic in the PHYSICS (the neighbor grid and every force
// wrap them, see forces.wgsl's mi()), but the flood here answers a different question -- "is
// there a region this grid cannot reach from outside a localized closed object" -- and for that
// question all six faces of the box are treated as OPEN boundary here, with no wraparound between
// x=0 and x=nx-1 (same for y). Two reasons: (1) it is the simpler, more direct implementation of
// "flood from the boundary" on both CPU and GPU, with no modulo arithmetic in the flood step
// itself (occupancy() below still wraps x,y when marking cells near a bead, matching the physics
// of where a bead's influence actually reaches -- that part of the periodicity is real and kept);
// (2) it is what keeps the flat-sheet test honest without relying on the choice at all: z is
// never periodic in this engine (only x,y are wrapped anywhere else in the codebase), so a sheet
// spanning the full periodic x,y plane still touches the z=0 and z=nz-1 faces DIRECTLY on both
// sides of itself -- wrapping x,y would not change whether either side reaches a boundary, since
// the open z faces already guarantee it for any object that does not itself span all of z.
// Wrapping x,y in the flood would only matter for an object that exploits the periodic boundary
// itself (e.g. a tube closed around the periodic direction rather than by its own surface), which
// is not what this task's tests are about (a shell/vesicle centered well inside the box) -- so the
// simpler open-boundary choice is made deliberately, not by default, and does not need to be to
// pass the flat-sheet requirement.

import closureWgsl from '../wgsl/closure.wgsl?raw'
import { getGpu, storageBuffer } from './gpu'

export type Dims = [number, number, number]

/** Grid dims for a box tiled by cubes of side `cell`: ceil per axis so the grid always covers the
 * full box (a floor here could leave a sliver of the box outside the grid, which occupancy()
 * would then simply never see). */
export function dimsFor(box: [number, number, number], cell: number): Dims {
  return [
    Math.max(1, Math.ceil(box[0] / cell)),
    Math.max(1, Math.ceil(box[1] / cell)),
    Math.max(1, Math.ceil(box[2] / cell)),
  ]
}

/** Marks every grid cell (side `cell`, dims = dimsFor(box, cell)) whose CENTRE lies within
 * `radius` of any bead in `positions` (the engine's flat vec4-per-bead layout: x, y, z, type --
 * type is ignored, every bead counts). Walks only the cells in a bounding box around each bead
 * (reach = ceil(radius/cell) cells in every direction), never every cell against every bead, so
 * cost is O(beads * reach^3), not O(beads * ncells) -- the brief's "obход по ячейкам вокруг бида,
 * без полного перебора". x, y wrap periodically when measuring a cell's distance to a bead
 * (matching the box's own periodicity: a bead near x=0 also has physical reach into cells near
 * x=box.x); z does not (z is open throughout this engine, see sim.ts's wrapXY). */
export function occupancy(
  positions: Float32Array,
  box: [number, number, number],
  cell: number,
  radius: number,
): Uint8Array {
  const dims = dimsFor(box, cell)
  const [nx, ny, nz] = dims
  const occ = new Uint8Array(nx * ny * nz)
  const reach = Math.max(1, Math.ceil(radius / cell))
  const r2 = radius * radius
  const n = positions.length / 4

  for (let i = 0; i < n; i++) {
    const px = positions[i * 4]
    const py = positions[i * 4 + 1]
    const pz = positions[i * 4 + 2]
    const cx = Math.floor(px / cell)
    const cy = Math.floor(py / cell)
    const cz = Math.floor(pz / cell)
    for (let dz = -reach; dz <= reach; dz++) {
      const gz = cz + dz
      if (gz < 0 || gz >= nz) continue
      const ddz = (gz + 0.5) * cell - pz
      for (let dy = -reach; dy <= reach; dy++) {
        const gy = (((cy + dy) % ny) + ny) % ny
        let ddy = (gy + 0.5) * cell - py
        ddy -= Math.round(ddy / box[1]) * box[1]
        for (let dx = -reach; dx <= reach; dx++) {
          const gx = (((cx + dx) % nx) + nx) % nx
          let ddx = (gx + 0.5) * cell - px
          ddx -= Math.round(ddx / box[0]) * box[0]
          if (ddx * ddx + ddy * ddy + ddz * ddz <= r2) {
            occ[gx + nx * (gy + ny * gz)] = 1
          }
        }
      }
    }
  }
  return occ
}

/** CPU reference: breadth-first flood of every EMPTY cell reachable from any of the box's six
 * boundary faces (open boundary, no periodic wraparound -- see the module note above), then the
 * enclosed volume is the count of empty cells the flood never reached, times cell^3. This is the
 * definition of truth enclosedVolumeGpuDetailed's WGSL kernel is checked against. */
export function enclosedVolume(occ: Uint8Array, dims: [number, number, number], cell: number): number {
  const [nx, ny, nz] = dims
  if (occ.length !== nx * ny * nz) {
    throw new Error(`enclosedVolume: occ.length=${occ.length} не совпадает с dims=[${nx},${ny},${nz}]`)
  }
  const visited = new Uint8Array(occ.length)
  const idx = (x: number, y: number, z: number) => x + nx * (y + ny * z)
  const queue: number[] = []

  function seed(x: number, y: number, z: number) {
    if (x < 0 || x >= nx || y < 0 || y >= ny || z < 0 || z >= nz) return
    const i = idx(x, y, z)
    if (occ[i] === 0 && visited[i] === 0) {
      visited[i] = 1
      queue.push(i)
    }
  }

  for (let y = 0; y < ny; y++) for (let z = 0; z < nz; z++) { seed(0, y, z); seed(nx - 1, y, z) }
  for (let x = 0; x < nx; x++) for (let z = 0; z < nz; z++) { seed(x, 0, z); seed(x, ny - 1, z) }
  for (let x = 0; x < nx; x++) for (let y = 0; y < ny; y++) { seed(x, y, 0); seed(x, y, nz - 1) }

  let head = 0
  while (head < queue.length) {
    const i = queue[head++]
    const z = Math.floor(i / (nx * ny))
    const rem = i - z * nx * ny
    const y = Math.floor(rem / nx)
    const x = rem - y * nx
    seed(x + 1, y, z); seed(x - 1, y, z)
    seed(x, y + 1, z); seed(x, y - 1, z)
    seed(x, y, z + 1); seed(x, y, z - 1)
  }

  let unreached = 0
  for (let i = 0; i < occ.length; i++) if (occ[i] === 0 && visited[i] === 0) unreached++
  return unreached * cell * cell * cell
}

// --- GPU kernel driver -------------------------------------------------------------------------

interface ClosurePipelines {
  device: GPUDevice
  clear: GPUComputePipeline
  occupancy: GPUComputePipeline
  seed: GPUComputePipeline
  clearChanged: GPUComputePipeline
  propagate: GPUComputePipeline
}

let cachedPipelines: ClosurePipelines | undefined

function getClosurePipelines(device: GPUDevice): ClosurePipelines {
  if (cachedPipelines && cachedPipelines.device === device) return cachedPipelines
  const module = device.createShaderModule({ code: closureWgsl })
  const cp = (entryPoint: string) => device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })
  cachedPipelines = {
    device,
    clear: cp('clear_main'),
    occupancy: cp('occupancy_main'),
    seed: cp('seed_main'),
    clearChanged: cp('clear_changed_main'),
    propagate: cp('propagate_main'),
  }
  return cachedPipelines
}

async function readBackU32(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<Uint32Array> {
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(src, 0, dst, 0, bytes)
  device.queue.submit([enc.finish()])
  await dst.mapAsync(GPUMapMode.READ)
  const out = new Uint32Array(dst.getMappedRange().slice(0))
  dst.unmap()
  dst.destroy()
  return out
}

export interface EnclosedVolumeGpuResult {
  volume: number
  /** Number of propagate_main dispatches actually run before a pass changed nothing. */
  iterations: number
  /** Cap on `iterations` -- the grid's diagonal length in cells (see the doc comment below). */
  maxIterations: number
  /** Wall-clock, GPU-inclusive, for the whole computation (buffer setup through final readback). */
  ms: number
}

/** GPU counterpart of occupancy()+enclosedVolume(): occupancy in one atomicOr pass over beads
 * (closure.wgsl's occupancy_main), then iterative propagation of an "outside" label from every
 * open boundary cell (closure.wgsl's seed_main + propagate_main -- the SAME open-boundary choice
 * as the CPU reference above, so the two must agree) repeated with a change flag until a dispatch
 * flips nothing, capped at the grid's diagonal length in cells: ceil(sqrt(nx^2+ny^2+nz^2)) + 1. A
 * 6-connected flood from any boundary face reaches every cell within that many hops for the
 * star-convex exteriors this task's tests use (a hollow shell, a flat sheet) -- see
 * task-8-report.md for the iteration count actually measured against this cap. */
export async function enclosedVolumeGpuDetailed(
  positions: Float32Array,
  box: [number, number, number],
  opts: { cell: number; radius: number },
): Promise<EnclosedVolumeGpuResult> {
  const t0 = performance.now()
  const { device } = await getGpu()
  const pipe = getClosurePipelines(device)
  const dims = dimsFor(box, opts.cell)
  const [nx, ny, nz] = dims
  const ncells = nx * ny * nz
  const reach = Math.max(1, Math.ceil(opts.radius / opts.cell))
  const maxIterations = Math.ceil(Math.sqrt(nx * nx + ny * ny + nz * nz)) + 1

  const gridUniform = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const bytes = new ArrayBuffer(48)
  new Uint32Array(bytes, 0, 4).set([nx, ny, nz, reach])
  new Float32Array(bytes, 16, 4).set([box[0], box[1], box[2], opts.cell])
  new Float32Array(bytes, 32, 4).set([opts.radius, opts.radius * opts.radius, 0, 0])
  device.queue.writeBuffer(gridUniform, 0, bytes)

  const posBuf = storageBuffer(device, positions)
  const bufUsage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
  const occBuf = device.createBuffer({ size: ncells * 4, usage: bufUsage })
  const outsideBuf = device.createBuffer({ size: ncells * 4, usage: bufUsage })
  const changedBuf = device.createBuffer({ size: 4, usage: bufUsage })

  // WebGPU's layout:'auto' infers a DIFFERENT bind group layout per entry point -- only the
  // bindings that entry point's shader body statically references. clear_main touches G/occ/
  // outside but not pos/changed; occupancy_main touches G/pos/occ but not outside/changed; and so
  // on. One shared bind group built against any single pipeline's layout is therefore invalid for
  // the others (measured: WebGPU rejects it at createBindGroup with "binding index N not present
  // in the bind group layout", which fails validation silently from JS's point of view -- every
  // subsequent dispatch becomes a no-op against an invalid command buffer, occ/outside/changed
  // stay at their zero-initialized default, and the whole box reads back as one giant "cavity").
  // Each pipeline gets its own bind group with only the entries its own layout expects.
  const entriesFor = (want: readonly number[]): GPUBindGroupEntry[] =>
    [
      { binding: 0, resource: { buffer: gridUniform } },
      { binding: 1, resource: { buffer: posBuf } },
      { binding: 2, resource: { buffer: occBuf } },
      { binding: 3, resource: { buffer: outsideBuf } },
      { binding: 4, resource: { buffer: changedBuf } },
    ].filter((e) => want.includes(e.binding))

  const bindClear = device.createBindGroup({ layout: pipe.clear.getBindGroupLayout(0), entries: entriesFor([0, 2, 3]) })
  const bindOccupancy = device.createBindGroup({
    layout: pipe.occupancy.getBindGroupLayout(0),
    entries: entriesFor([0, 1, 2]),
  })
  const bindSeed = device.createBindGroup({ layout: pipe.seed.getBindGroupLayout(0), entries: entriesFor([0, 2, 3]) })
  const bindClearChanged = device.createBindGroup({
    layout: pipe.clearChanged.getBindGroupLayout(0),
    entries: entriesFor([4]),
  })
  const bindPropagate = device.createBindGroup({
    layout: pipe.propagate.getBindGroupLayout(0),
    entries: entriesFor([0, 2, 3, 4]),
  })

  const wgCells = Math.ceil(ncells / 64)
  const wgBeads = Math.ceil(positions.length / 4 / 64)

  function dispatch(pipeline: GPUComputePipeline, bind: GPUBindGroup, workgroups: number) {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    pass.setPipeline(pipeline)
    pass.setBindGroup(0, bind)
    pass.dispatchWorkgroups(workgroups)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  dispatch(pipe.clear, bindClear, wgCells)
  dispatch(pipe.occupancy, bindOccupancy, wgBeads)
  dispatch(pipe.seed, bindSeed, wgCells)

  let iterations = 0
  while (iterations < maxIterations) {
    dispatch(pipe.clearChanged, bindClearChanged, 1)
    dispatch(pipe.propagate, bindPropagate, wgCells)
    const changed = await readBackU32(device, changedBuf, 4)
    iterations++
    if (changed[0] === 0) break
  }

  const [outsideHost, occHost] = await Promise.all([
    readBackU32(device, outsideBuf, ncells * 4),
    readBackU32(device, occBuf, ncells * 4),
  ])

  posBuf.destroy()
  occBuf.destroy()
  outsideBuf.destroy()
  changedBuf.destroy()
  gridUniform.destroy()

  let unreached = 0
  for (let i = 0; i < ncells; i++) if (occHost[i] === 0 && outsideHost[i] === 0) unreached++
  const volume = unreached * opts.cell * opts.cell * opts.cell

  return { volume, iterations, maxIterations, ms: performance.now() - t0 }
}
