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
//
// Consequence of the open-boundary choice that DOES need active handling, not just a caveat: a
// perfectly closed object that merely sits near a box face has its own interior cells seeded as
// "outside" by seed_main/the CPU seed() walk, because seeding only looks at WHICH FACE a cell is
// on, never at what structure surrounds it. occupancy()/enclosedVolume() are deliberately dumb
// about this (they are the low-level, position-in-box-as-given primitives); the facades in
// index.ts (enclosedVolumeCpu/enclosedVolumeGpu) are what must not hand them a coincidentally
// off-centre object, since a SELF-ASSEMBLED vesicle (this project's headline object) forms
// wherever it forms in a periodic box, not necessarily mid-box. recenterOnLargestCluster() below
// is the fix: it relocates the dominant connected structure to the box centre (wrapping in x,y,
// plain-shifting in z) before occupancy() ever runs. This still does not, and cannot, help an
// object that requires the periodic boundary to close AT ALL -- e.g. a tube that only forms a
// closed loop by wrapping around x -- since recentring changes where the object sits, not whether
// its own surface is complete; that case is out of scope for this detector by design, the same way
// the open-boundary flood itself is.

import { largestClusterCenter } from './aggregate'
import closureWgsl from '../wgsl/closure.wgsl?raw'
import { getGpu, storageBuffer } from './gpu'
import { loadParams, wcaCutoff } from './params'

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

/** The flood itself, factored out of enclosedVolume() below so the cavity-labelling pass added
 * later in this file (cavities()) can reuse the EXACT SAME "outside" BFS rather than a second,
 * possibly-drifting copy of it -- this is a pure extraction, the traversal order, seeding rule and
 * termination condition are byte-for-byte what enclosedVolume() ran inline before. Breadth-first
 * flood of every EMPTY cell reachable from any of the box's six boundary faces (open boundary, no
 * periodic wraparound -- see the module note above). Returns the visited mask; callers decide what
 * to do with the unreached (occ=0, visited=0) cells. */
function floodOutside(occ: Uint8Array, dims: [number, number, number]): Uint8Array {
  const [nx, ny, nz] = dims
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

  return visited
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
  const visited = floodOutside(occ, dims)
  let unreached = 0
  for (let i = 0; i < occ.length; i++) if (occ[i] === 0 && visited[i] === 0) unreached++
  return unreached * cell * cell * cell
}

// --- cavity breakdown: WHICH unreached cells belong to WHICH disjoint pocket -----------------------
//
// enclosedVolume() above answers "how much empty space did the flood never reach, total" -- a
// single scalar that silently sums every disjoint unreached region together. That scalar is what
// the vesicle gate has always been checked against and stays checked against (data/soup.json /
// data/literature.json), so this section does not change it. But a single number cannot show a
// viewer WHERE a cavity is, whether it is one pocket or several unrelated ones, or how big any ONE
// of them actually is -- exactly the distinction this task exists to make honest (a run can report
// a total enclosedVolume that clears a threshold while consisting of several small, physically
// meaningless pockets, none of which is remotely vesicle-sized). cavities() below labels the SAME
// unreached cells floodOutside() already identified into connected components (6-connected, same
// open-boundary convention as the flood itself -- no periodic wrap, matching floodOutside()'s own
// choice) via a second, independent BFS/labelling pass. This is new analysis of the flood's output,
// not a change to the flood algorithm itself.

export interface Cavity {
  /** Number of grid cells in this cavity. */
  voxelCount: number
  /** voxelCount * cell^3, in the same reduced units as the box. */
  volume: number
  /** Flat cell indices (x + nx*(y+ny*z), same indexing as occ/visited) belonging to this cavity. */
  cells: Uint32Array
  /** Centre of mass of this cavity's cell CENTRES, in the same coordinate frame as the `occ` grid
   * passed in (i.e. NOT undone of any recentring the caller applied before building `occ`). */
  centre: [number, number, number]
}

/** (3*volume/(4*pi))^(1/3) -- the radius of a sphere with the same volume as `volume`. Pure
 * geometry, used to make a cavity's size intuitive (a viewer showing "voxelCount=2984" is not
 * intuitive; "equivalent radius 8.9 sigma" is). */
export function equivalentSphereRadius(volume: number): number {
  return Math.cbrt((3 * volume) / (4 * Math.PI))
}

/** Labels every EMPTY, flood-unreached cell of `occ` into connected components (6-connected, open
 * boundary -- see this section's header), sorted largest-voxelCount-first. `enclosedVolume(occ,
 * dims, cell)` above always equals the sum of every returned cavity's `volume` -- this function is
 * strictly a breakdown of the same unreached set, not a different measurement of it. */
export function cavities(occ: Uint8Array, dims: Dims, cell: number): Cavity[] {
  const [nx, ny, nz] = dims
  if (occ.length !== nx * ny * nz) {
    throw new Error(`cavities: occ.length=${occ.length} не совпадает с dims=[${nx},${ny},${nz}]`)
  }
  const visited = floodOutside(occ, dims)
  const labelled = new Uint8Array(occ.length) // 1 once assigned to some cavity, to avoid a second visit
  const cellOf = (i: number): [number, number, number] => {
    const z = Math.floor(i / (nx * ny))
    const rem = i - z * nx * ny
    const y = Math.floor(rem / nx)
    const x = rem - y * nx
    return [x, y, z]
  }
  const idx = (x: number, y: number, z: number) => x + nx * (y + ny * z)

  const result: Cavity[] = []
  const queue: number[] = []
  for (let start = 0; start < occ.length; start++) {
    if (occ[start] !== 0 || visited[start] !== 0 || labelled[start] !== 0) continue

    labelled[start] = 1
    queue.length = 0
    queue.push(start)
    const cellsArr: number[] = []
    let sx = 0
    let sy = 0
    let sz = 0
    let head = 0
    while (head < queue.length) {
      const i = queue[head++]
      cellsArr.push(i)
      const [x, y, z] = cellOf(i)
      sx += (x + 0.5) * cell
      sy += (y + 0.5) * cell
      sz += (z + 0.5) * cell
      const neighbors: [number, number, number][] = [
        [x + 1, y, z], [x - 1, y, z],
        [x, y + 1, z], [x, y - 1, z],
        [x, y, z + 1], [x, y, z - 1],
      ]
      for (const [nx2, ny2, nz2] of neighbors) {
        if (nx2 < 0 || nx2 >= nx || ny2 < 0 || ny2 >= ny || nz2 < 0 || nz2 >= nz) continue
        const ni = idx(nx2, ny2, nz2)
        if (occ[ni] === 0 && visited[ni] === 0 && labelled[ni] === 0) {
          labelled[ni] = 1
          queue.push(ni)
        }
      }
    }

    const n = cellsArr.length
    result.push({
      voxelCount: n,
      volume: n * cell * cell * cell,
      cells: Uint32Array.from(cellsArr),
      centre: [sx / n, sy / n, sz / n],
    })
  }

  result.sort((a, b) => b.voxelCount - a.voxelCount)
  return result
}

// --- recentring: the fix for the open-boundary flood's face-coincidence blind spot -------------

function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

/** Returns a COPY of `positions` translated so the periodic-aware (x,y) centre of mass of the
 * single largest connected cluster (largestClusterCenter() in aggregate.ts) sits at the box centre
 * in x,y, and that same cluster's plain z mean sits at the box's z midpoint. x,y are translated by
 * a wrapped shift (any translation of a periodic configuration is itself just another valid
 * configuration of it); z is a plain shift, never wrapped, since z has no periodic image anywhere
 * in this engine. The cutoff for "single largest connected cluster" is the same tail-tail
 * attraction range used everywhere else a cluster needs defining (largestClusterFractionOf in
 * index.ts, the self-assembly gate) -- not a new constant.
 *
 * See the module doc above for why this step exists: without it, a self-assembled object that
 * happens to sit near a box face reads as open even when it is genuinely closed, because
 * occupancy()/enclosedVolume()'s open-boundary flood seeds "outside" from every boundary face with
 * no idea what structure surrounds it. Recentring first removes the coincidence entirely, at the
 * cost of one extra union-find pass. The caller's own array is never mutated. */
/** The (shiftX, shiftY, shiftZ) recenterOnLargestCluster applies -- factored out so a caller that
 * needs to walk the SAME shift backwards (mapping something computed in the recentred frame, e.g.
 * a cavity's voxels, back into the original frame the caller's own positions are drawn in) can get
 * it without re-deriving it, and without recenterOnLargestCluster itself changing shape. */
function largestClusterShift(positions: Float32Array, box: [number, number, number]): [number, number, number] {
  const p = loadParams()
  const cutoff = wcaCutoff(p.beadSizes.tail_tail) + p.attraction.wc
  const [comX, comY, comZ] = largestClusterCenter(positions, box, cutoff)
  return [box[0] / 2 - comX, box[1] / 2 - comY, box[2] / 2 - comZ]
}

export function recenterOnLargestCluster(positions: Float32Array, box: [number, number, number]): Float32Array {
  const [shiftX, shiftY, shiftZ] = largestClusterShift(positions, box)
  const n = positions.length / 4
  const out = new Float32Array(positions.length)
  for (let i = 0; i < n; i++) {
    out[i * 4] = wrap1(positions[i * 4] + shiftX, box[0])
    out[i * 4 + 1] = wrap1(positions[i * 4 + 1] + shiftY, box[1])
    out[i * 4 + 2] = positions[i * 4 + 2] + shiftZ
    out[i * 4 + 3] = positions[i * 4 + 3]
  }
  return out
}

/** Recentres (see recenterOnLargestCluster above), then runs the CPU occupancy()/enclosedVolume()
 * pipeline in one call -- the pure-CPU convenience the closure.test.ts boundary-straddling test
 * uses directly (no GPU/browser needed), and what enclosedVolumeCpu (index.ts) reduces to once it
 * has snapshotted `sys`'s positions/box. */
export function enclosedVolumeFromPositions(
  positions: Float32Array,
  box: [number, number, number],
  opts: { cell: number; radius: number },
): number {
  const centered = recenterOnLargestCluster(positions, box)
  const occ = occupancy(centered, box, opts.cell, opts.radius)
  return enclosedVolume(occ, dimsFor(box, opts.cell), opts.cell)
}

/** A cavity from cavitiesFromPositions() below, with everything mapped OUT of the internal
 * recentred frame and back into the same coordinate frame `positions` was given in -- the frame a
 * caller's own particles are actually drawn in. `centre`/`voxelCentres` are therefore directly
 * plottable alongside the caller's own position data with no further transform. */
export interface CavityWorld {
  voxelCount: number
  volume: number
  /** Equivalent-sphere radius of `volume` -- see equivalentSphereRadius(). */
  radius: number
  centre: [number, number, number]
  /** Flat x,y,z triples, one per voxel (grid-cell centre) belonging to this cavity, length ===
   * voxelCount*3 -- what a viewer instances to draw the cavity. */
  voxelCentres: Float32Array
}

export interface CavitiesResult {
  /** Total number of disjoint cavities found (cavities.length, kept as its own field so a caller
   * that only wants the count -- and not to materialize every cavity's voxel list -- can read it
   * without counting the array itself). */
  cavityCount: number
  /** Every cavity found, largest voxelCount first. */
  cavities: CavityWorld[]
}

/** Same recentring + occupancy pipeline as enclosedVolumeFromPositions above, but returns the full
 * per-cavity breakdown (cavities()) instead of one summed scalar, with every cavity's centre and
 * voxel list mapped back out of the recentred frame into the frame `positions` was given in (see
 * recenterOnLargestCluster's own doc comment for why recentring exists at all: an open-boundary
 * flood run directly on `positions` would misread a self-assembled object sitting near a box face
 * as open, even when it is genuinely closed). `cavities[0].volume` summed with every other
 * `cavities[k].volume` equals what enclosedVolumeFromPositions(positions, box, opts) returns for
 * the SAME inputs -- this is a breakdown of that same measurement, not a second one. */
export function cavitiesFromPositions(
  positions: Float32Array,
  box: [number, number, number],
  opts: { cell: number; radius: number },
): CavitiesResult {
  const [shiftX, shiftY, shiftZ] = largestClusterShift(positions, box)
  const centered = recenterOnLargestCluster(positions, box)
  const occ = occupancy(centered, box, opts.cell, opts.radius)
  const dims = dimsFor(box, opts.cell)
  const [nx, ny] = dims
  const comps = cavities(occ, dims, opts.cell)

  const toWorld = (x: number, y: number, z: number): [number, number, number] => [
    wrap1(x - shiftX, box[0]),
    wrap1(y - shiftY, box[1]),
    z - shiftZ,
  ]

  const cavitiesOut: CavityWorld[] = comps.map((c) => {
    const voxelCentres = new Float32Array(c.cells.length * 3)
    for (let k = 0; k < c.cells.length; k++) {
      const i = c.cells[k]
      const z = Math.floor(i / (nx * ny))
      const rem = i - z * nx * ny
      const y = Math.floor(rem / nx)
      const x = rem - y * nx
      const [wx, wy, wz] = toWorld((x + 0.5) * opts.cell, (y + 0.5) * opts.cell, (z + 0.5) * opts.cell)
      voxelCentres[k * 3] = wx
      voxelCentres[k * 3 + 1] = wy
      voxelCentres[k * 3 + 2] = wz
    }
    const [cx, cy, cz] = toWorld(c.centre[0], c.centre[1], c.centre[2])
    return {
      voxelCount: c.voxelCount,
      volume: c.volume,
      radius: equivalentSphereRadius(c.volume),
      centre: [cx, cy, cz],
      voxelCentres,
    }
  })

  return { cavityCount: cavitiesOut.length, cavities: cavitiesOut }
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
  let converged = false
  while (iterations < maxIterations) {
    dispatch(pipe.clearChanged, bindClearChanged, 1)
    dispatch(pipe.propagate, bindPropagate, wgCells)
    const changed = await readBackU32(device, changedBuf, 4)
    iterations++
    if (changed[0] === 0) {
      converged = true
      break
    }
  }

  // A flood that hits maxIterations without a dispatch reporting "nothing changed" has not
  // actually finished propagating "outside" from the boundary -- some cells that are truly outside
  // are still marked as not-yet-reached. Reading `outside` off in that state and computing volume
  // from it anyway would silently count those still-unreached-but-actually-outside cells as
  // enclosed cavity, INFLATING the reported volume rather than failing loudly. maxIterations is
  // already a generous cap (the grid's own diagonal length in cells, see the doc comment above) --
  // hitting it without convergence means something is actually wrong (a bug, or a geometry this
  // detector's 6-connected boundary flood cannot reach in that many hops), not merely a slow case
  // that would have converged with more patience, so this is a real failure, not a warning.
  if (!converged) {
    posBuf.destroy()
    occBuf.destroy()
    outsideBuf.destroy()
    changedBuf.destroy()
    gridUniform.destroy()
    throw new Error(
      `enclosedVolumeGpuDetailed: outside-flood did not converge within maxIterations=${maxIterations} ` +
        `dispatches -- the reported volume would silently count still-unreached cells as enclosed ` +
        `cavity, inflating it, so this is a hard failure rather than a best-effort number`,
    )
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
