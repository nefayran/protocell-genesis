import forcesWgsl from '../wgsl/forces.wgsl?raw'
import neighborWgsl from '../wgsl/neighbor.wgsl?raw'
import integrateWgsl from '../wgsl/integrate.wgsl?raw'
import { getGpu, readBack, storageBuffer } from './gpu'
import { loadParams, paramsToUniform, wcaCutoff, type Params } from './params'

export type Layout = 'random' | 'bilayer' | 'vesicle'

export interface CreateSystemOpts {
  lipids: number
  box: [number, number, number]
  seed: number
  layout: Layout
  /** Overrides thermostat.gamma from params.json — needed to run the drift test at gamma=0. */
  gamma?: number
  /** Test-only escape hatch: when given, replaces the layout function's output with these exact
   * bead positions (4 floats per bead: x,y,z,type), skipping RNG entirely. `layout` is still
   * required by the type but is not consulted. Lets a test build a fixed, hand-picked
   * configuration (no RNG) to check against an independently-computed reference. */
  positions?: Float32Array
  /** Paired with `positions`: fixed initial velocities (4 floats per bead: vx,vy,vz,0). Defaults
   * to the same Gaussian(kT) initialization used otherwise if omitted. */
  velocities?: Float32Array
}

export interface System {
  /** Advances n Langevin/velocity-Verlet steps. Encoded as one command buffer: no per-step
   * CPU<->GPU round trip. */
  step(n: number): Promise<void>
  /** 4 floats per bead: x, y, z, type (0 = head, 1 = tail). */
  positions(): Promise<Float32Array>
  /** Mean of squared velocity components over all beads (mass = 1) — i.e. kinetic energy PER
   * DEGREE OF FREEDOM, which in these reduced units equals kT directly (not kT/2: equipartition
   * gives <0.5*m*v_x^2> = 0.5*kT per dof, so <v_x^2> = kT). Returning kT/2 here would fail the
   * equipartition test by exactly a factor of two, since it compares this value directly against
   * params.thermostat.kT. */
  kineticEnergyPerDof(): Promise<number>
  /** Kinetic + potential energy of the whole system. */
  totalEnergy(): Promise<number>
  /** Same physics as forces(), full O(N^2) pair loop instead of the neighbor grid — for
   * cross-checking the grid result. */
  forcesBruteForce(): Promise<Float32Array>
  forces(): Promise<Float32Array>
  /** Wall-clock time (ms, GPU-inclusive) of the most recent neighbor-grid rebuild. */
  neighborBuildMs: number
  /** Zero-tension Metropolis Monte Carlo move on the box's lateral area: proposes L_x, L_y (and
   * every bead's x, y) scaled by the same sqrt(s), s = exp(u), leaves L_z and z untouched, and
   * accepts with probability min(1, exp(-DeltaU/kT)) where DeltaU is the change in total
   * POTENTIAL energy only (kinetic energy is invariant under a coordinate rescale, and lateral
   * tension is zero so there is no gammaDeltaA term). Runs `trials` such moves and returns the
   * accepted fraction. Rebuilds the neighbor grid after every trial (accepted or not) since the
   * box the grid's cell/box uniform refers to may have changed. */
  areaMove(trials: number): Promise<number>
  /** Current box lengths — a live snapshot, since areaMove() mutates L_x, L_y in place. */
  readonly box: [number, number, number]
  /** Number of lipids the system was created with (fixed for its lifetime). */
  readonly lipids: number
  /** Cumulative count of integration steps taken via step(), across all calls so far. */
  readonly steps: number
}

// --- seeded RNG for reproducible layouts -------------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-7)
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/** Wraps x,y into [0, box) — z is left untouched (open boundary). Every bead position placed by
 * a layout function must go through this: the neighbor grid's "any pair within cutoff shares a
 * cell or a 3x3x3 neighbor" guarantee assumes coordinates already sit inside [0, box); a bead
 * placed just past the edge (e.g. a tail extending past x=box due to its lipid's orientation)
 * gets clamped into the wrong boundary cell instead of wrapping, and the grid then silently
 * misses that pair while the brute-force loop (which uses mi() directly, not cell membership)
 * still finds it — measured as a real ~0.2 force mismatch in the grid-vs-brute-force test, not a
 * float32 precision artifact. */
function wrapXY(pos: number[], box: [number, number, number]): number[] {
  const x = pos[0] - Math.floor(pos[0] / box[0]) * box[0]
  const y = pos[1] - Math.floor(pos[1] / box[1]) * box[1]
  return [x, y, pos[2]]
}

// --- layouts -------------------------------------------------------------------------------

/** Random lipid centers with random orientation; bond lengths taken from params so nothing here
 * is a bare model constant. Placement is rejection-sampled against a minimum bead separation
 * (the largest WCA bead radius in params): pure IID-uniform placement of ~1800 beads in a modest
 * box reliably lands some pair almost exactly on top of each other, and the WCA force there is
 * unbounded (~1/r^13) — the resulting ~1e9-magnitude force makes the grid-vs-brute-force
 * comparison fail on float32 summation-order noise alone, well before any real neighbor-list bug.
 * Distance checks are periodic-aware in x,y to match the physics (mi() in forces.wgsl). */
function layoutRandom(lipids: number, box: [number, number, number], p: Params, rng: () => number): Float32Array {
  const out = new Float32Array(lipids * 3 * 4)
  const placed: number[][] = []
  const minSep = Math.max(p.beadSizes.head_head, p.beadSizes.head_tail, p.beadSizes.tail_tail)
  const minSep2 = minSep * minSep
  const maxAttempts = 200

  function periodicDist2(a: number[], b: number[]): number {
    let dx = a[0] - b[0]
    let dy = a[1] - b[1]
    const dz = a[2] - b[2]
    dx -= Math.round(dx / box[0]) * box[0]
    dy -= Math.round(dy / box[1]) * box[1]
    return dx * dx + dy * dy + dz * dz
  }

  for (let k = 0; k < lipids; k++) {
    let head: number[] = [], tail1: number[] = [], tail2: number[] = []
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const cx = rng() * box[0]
      const cy = rng() * box[1]
      const cz = rng() * box[2]
      let ox = gaussian(rng)
      let oy = gaussian(rng)
      let oz = gaussian(rng)
      const len = Math.sqrt(ox * ox + oy * oy + oz * oz) || 1
      ox /= len
      oy /= len
      oz /= len
      head = wrapXY([cx, cy, cz], box)
      tail1 = wrapXY(
        [cx + ox * p.beadSizes.head_tail, cy + oy * p.beadSizes.head_tail, cz + oz * p.beadSizes.head_tail],
        box,
      )
      tail2 = wrapXY(
        [
          tail1[0] + ox * p.beadSizes.tail_tail,
          tail1[1] + oy * p.beadSizes.tail_tail,
          tail1[2] + oz * p.beadSizes.tail_tail,
        ],
        box,
      )
      const clash = placed.some(
        (b) => periodicDist2(head, b) < minSep2 || periodicDist2(tail1, b) < minSep2 || periodicDist2(tail2, b) < minSep2,
      )
      if (!clash || attempt === maxAttempts - 1) break
    }
    placed.push(head, tail1, tail2)
    const base = k * 12
    out.set([...head, 0], base)
    out.set([...tail1, 1], base + 4)
    out.set([...tail2, 1], base + 8)
  }
  return out
}

/** Two leaflets in the xy plane, heads pointing outward and tails toward the midplane. Grid
 * spacing and leaflet gap are derived from box size and bead sizes (never a bare literal) — good
 * enough for a stable starting configuration for the energy-drift test; hitting the accepted
 * area-per-lipid corridor is Task 5's job, not this one. */
function layoutBilayer(lipids: number, box: [number, number, number], p: Params, rng: () => number): Float32Array {
  const out = new Float32Array(lipids * 3 * 4)
  const perLeaflet = Math.ceil(lipids / 2)
  const nSide = Math.max(1, Math.ceil(Math.sqrt(perLeaflet)))
  const spacingX = box[0] / nSide
  const spacingY = box[1] / nSide
  const midZ = box[2] / 2
  const gap = p.beadSizes.tail_tail / 2
  const jitter = Math.min(spacingX, spacingY) * 0.1

  for (let k = 0; k < lipids; k++) {
    const leaflet = k < perLeaflet ? 1 : -1 // +1 top (heads at +z), -1 bottom (heads at -z)
    const idx = k < perLeaflet ? k : k - perLeaflet
    const a = idx % nSide
    const b = Math.floor(idx / nSide)
    const xRaw = (a + 0.5) * spacingX + (rng() - 0.5) * jitter
    const yRaw = (b + 0.5) * spacingY + (rng() - 0.5) * jitter
    const [x, y] = wrapXY([xRaw, yRaw, 0], box)

    const zTail2 = midZ + leaflet * gap
    const zTail1 = zTail2 + leaflet * p.beadSizes.tail_tail
    const zHead = zTail1 + leaflet * p.beadSizes.head_tail

    const base = k * 12
    out.set([x, y, zHead, 0], base)
    out.set([x, y, zTail1, 1], base + 4)
    out.set([x, y, zTail2, 1], base + 8)
  }
  return out
}

function initialVelocities(n: number, kT: number, rng: () => number): Float32Array {
  const out = new Float32Array(n * 4)
  const s = Math.sqrt(kT)
  for (let i = 0; i < n; i++) {
    out[i * 4 + 0] = gaussian(rng) * s
    out[i * 4 + 1] = gaussian(rng) * s
    out[i * 4 + 2] = gaussian(rng) * s
    out[i * 4 + 3] = 0
  }
  return out
}

// --- pipeline cache (one shared GPUDevice, compiled once) ---------------------------------------

interface Pipelines {
  device: GPUDevice
  forceGrid: GPUComputePipeline
  forceBrute: GPUComputePipeline
  clearCounts: GPUComputePipeline
  count: GPUComputePipeline
  prefix: GPUComputePipeline
  fill: GPUComputePipeline
  kick: GPUComputePipeline
  drift: GPUComputePipeline
  wrap: GPUComputePipeline
  thermostat: GPUComputePipeline
}

let cached: Pipelines | undefined

function getPipelines(device: GPUDevice): Pipelines {
  if (cached && cached.device === device) return cached
  const forceModule = device.createShaderModule({ code: forcesWgsl })
  const neighborModule = device.createShaderModule({ code: neighborWgsl })
  const integrateModule = device.createShaderModule({ code: integrateWgsl })
  const cp = (module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })
  cached = {
    device,
    forceGrid: cp(forceModule, 'force_main'),
    forceBrute: cp(forceModule, 'force_brute_main'),
    clearCounts: cp(neighborModule, 'clear_counts_main'),
    count: cp(neighborModule, 'count_main'),
    prefix: cp(neighborModule, 'prefix_main'),
    fill: cp(neighborModule, 'fill_main'),
    kick: cp(integrateModule, 'kick_main'),
    drift: cp(integrateModule, 'drift_main'),
    wrap: cp(integrateModule, 'wrap_main'),
    thermostat: cp(integrateModule, 'thermostat_main'),
  }
  return cached
}

// --- system --------------------------------------------------------------------------------

export async function createSystem(opts: CreateSystemOpts): Promise<System> {
  if (opts.layout === 'vesicle') {
    // Task 8. Deliberately not implemented here.
    throw new Error("layout 'vesicle' не реализован в Задаче 4 — относится к Задаче 8")
  }

  const base = loadParams()
  const p: Params = opts.gamma === undefined ? base : { ...base, thermostat: { ...base.thermostat, gamma: opts.gamma } }

  const N = opts.lipids * 3
  const box = opts.box
  // Live box, mutated in place by areaMove(); `box` above stays the immutable value opts was
  // called with (still needed below for the cell-size/dims computation, which is fixed for the
  // system's lifetime — see the areaMove doc comment near its definition for why that's safe).
  let liveBox: [number, number, number] = [box[0], box[1], box[2]]
  let totalSteps = 0
  const rng = mulberry32(opts.seed)

  const positions0 = opts.positions ?? (opts.layout === 'random' ? layoutRandom(opts.lipids, box, p, rng) : layoutBilayer(opts.lipids, box, p, rng))
  const velocities0 = opts.velocities ?? initialVelocities(N, p.thermostat.kT, rng)
  const rngState0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) rngState0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9

  // Cell size: at least r_c + w_c (the largest interaction range, the tail-tail attraction's
  // outer cutoff) so any pair within range of each other is guaranteed to fall in the same cell
  // or one of the 26 neighbors. A smaller cell silently drops forces. Fixed for the system's
  // lifetime (it depends only on params, not on the box), but `dims`/`ncells` derived from it are
  // NOT fixed: areaMove() below recomputes and reallocates them whenever the live box crosses a
  // cell-count boundary, which is exactly the "shrinking the box changes the grid" case the brief
  // warns about. Leaving dims frozen at the value computed here would let the box drift far
  // enough that box/dims < cellSize — at that point the ±1 neighbor-cell walk in force_main no
  // longer reaches every pair within the interaction range, WCA repulsion at close range goes
  // silently under-counted, and nothing then resists further compression: measured on this
  // engine, a run that kept dims frozen collapsed area/lipid from 1.352 to 0.697 over 200 area
  // moves instead of equilibrating in the literature corridor.
  const cellSize = wcaCutoff(p.beadSizes.tail_tail) + p.attraction.wc

  function computeDims(b: [number, number, number]): [number, number, number] {
    return [Math.max(1, Math.floor(b[0] / cellSize)), Math.max(1, Math.floor(b[1] / cellSize)), Math.max(1, Math.floor(b[2] / cellSize))]
  }

  // The ±1 neighbor-cell walk in force_main wraps periodically in x,y. With fewer than 3 cells on
  // a periodic axis, +1 and -1 land on the same wrapped cell (or, at 1 cell, all three land on
  // the cell itself), so that cell gets visited twice (2 cells) or three times (1 cell) per
  // particle — every non-bonded force and energy contribution from it is silently doubled or
  // tripled. Minimum-image convention (mi() in forces.wgsl) separately assumes each axis sees at
  // most one periodic image within range, i.e. box/2 must exceed every interaction's reach in
  // that axis — including the bend pair (head-tail2, reach ~r0) which mi() also wraps.
  function gridInvariantsHold(b: [number, number, number], d: [number, number, number]): boolean {
    return d[0] >= 3 && d[1] >= 3 && Math.min(b[0], b[1]) / 2 > p.bend.r0
  }

  let dims = computeDims(box)
  if (!gridInvariantsHold(box, dims)) {
    throw new Error(
      `сетка соседей: box=[${box[0]},${box[1]},${box[2]}] даёт cellSize=${cellSize.toFixed(4)}, dims=[${dims[0]},${dims[1]},${dims[2]}] ` +
        `и min(box.x,box.y)/2=${(Math.min(box[0], box[1]) / 2).toFixed(4)} — нужно dims>=3 на осях x,y и min(box.x,box.y)/2 > bend.r0=${p.bend.r0}`,
    )
  }
  let ncells = dims[0] * dims[1] * dims[2]

  const { device } = await getGpu()
  const pipe = getPipelines(device)

  const posBuf = storageBuffer(device, positions0)
  const velBuf = storageBuffer(device, velocities0)
  const forceBuf = storageBuffer(device, new Float32Array(N * 4))
  const potentialBuf = storageBuffer(device, new Float32Array(N))
  // Sized N (one slot per bead, sorted by cell), not ncells — independent of the grid resize
  // below, so it is never reallocated.
  const cellsBuf = storageBuffer(device, new Float32Array(N))
  // Sized ncells — reallocated by resizeGrid() whenever dims changes.
  let countsBuf = storageBuffer(device, new Float32Array(ncells))
  let cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
  let cursorBuf = storageBuffer(device, new Float32Array(ncells))
  const rngBuf = device.createBuffer({
    size: rngState0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(rngBuf, 0, rngState0)

  const paramsUniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(paramsUniform, 0, paramsToUniform(p))

  const gridUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const boxUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })

  function writeGridUniforms(b: [number, number, number], d: [number, number, number]) {
    const bytes = new ArrayBuffer(32)
    new Uint32Array(bytes, 0, 4).set([d[0], d[1], d[2], 0])
    new Float32Array(bytes, 16, 4).set([b[0], b[1], b[2], 0])
    device.queue.writeBuffer(gridUniform, 0, bytes)
    device.queue.writeBuffer(boxUniform, 0, new Float32Array([b[0], b[1], b[2], 0]))
  }

  const bind = (pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries })
  const buf = (b: GPUBuffer) => ({ buffer: b })

  // Bind groups that reference the ncells-sized buffers (countsBuf/cellStartBuf/cursorBuf) — must
  // be rebuilt by resizeGrid() every time those buffers are reallocated, since a WebGPU bind
  // group is a fixed reference to specific buffer objects.
  let clearCountsBind!: GPUBindGroup
  let countBind!: GPUBindGroup
  let prefixBind!: GPUBindGroup
  let fillBind!: GPUBindGroup
  let forceGridGroup1!: GPUBindGroup
  let wgCells = 0

  function rebindGridDependent() {
    clearCountsBind = bind(pipe.clearCounts, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 2, resource: buf(countsBuf) },
    ])
    countBind = bind(pipe.count, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 1, resource: buf(posBuf) },
      { binding: 2, resource: buf(countsBuf) },
    ])
    prefixBind = bind(pipe.prefix, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 2, resource: buf(countsBuf) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cursorBuf) },
    ])
    fillBind = bind(pipe.fill, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 1, resource: buf(posBuf) },
      { binding: 3, resource: buf(cellsBuf) },
      { binding: 5, resource: buf(cursorBuf) },
    ])
    forceGridGroup1 = bind(pipe.forceGrid, 1, [
      { binding: 0, resource: buf(posBuf) },
      { binding: 1, resource: buf(forceBuf) },
      { binding: 2, resource: buf(potentialBuf) },
      { binding: 3, resource: buf(gridUniform) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cellsBuf) },
    ])
    wgCells = Math.ceil(ncells / 64)
  }

  writeGridUniforms(box, dims)
  rebindGridDependent()

  // Recomputes dims for `newBox`; if the cell count changed, destroys and reallocates the
  // ncells-sized buffers and rebinds everything that references them. Called by areaMove() after
  // every trial (accepted or reverted) so the grid always matches the box actually in use — see
  // the comment on `cellSize` above for why a frozen grid silently breaks under compression.
  async function resizeGrid(newBox: [number, number, number]): Promise<void> {
    const newDims = computeDims(newBox)
    if (newDims[0] !== dims[0] || newDims[1] !== dims[1] || newDims[2] !== dims[2]) {
      countsBuf.destroy()
      cellStartBuf.destroy()
      cursorBuf.destroy()
      dims = newDims
      ncells = dims[0] * dims[1] * dims[2]
      countsBuf = storageBuffer(device, new Float32Array(ncells))
      cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
      cursorBuf = storageBuffer(device, new Float32Array(ncells))
      rebindGridDependent()
    }
    writeGridUniforms(newBox, dims)
  }

  const forceGridGroup0 = bind(pipe.forceGrid, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const forceBruteGroup0 = bind(pipe.forceBrute, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const forceBruteGroup1 = bind(pipe.forceBrute, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 1, resource: buf(forceBuf) },
    { binding: 2, resource: buf(potentialBuf) },
    { binding: 3, resource: buf(gridUniform) },
  ])

  const kickBind = bind(pipe.kick, 0, [
    { binding: 0, resource: buf(paramsUniform) },
    { binding: 2, resource: buf(velBuf) },
    { binding: 3, resource: buf(forceBuf) },
  ])
  const driftBind = bind(pipe.drift, 0, [
    { binding: 0, resource: buf(paramsUniform) },
    { binding: 1, resource: buf(posBuf) },
    { binding: 2, resource: buf(velBuf) },
  ])
  const wrapBind = bind(pipe.wrap, 0, [
    { binding: 1, resource: buf(posBuf) },
    { binding: 5, resource: buf(boxUniform) },
  ])
  const thermostatBind = bind(pipe.thermostat, 0, [
    { binding: 0, resource: buf(paramsUniform) },
    { binding: 2, resource: buf(velBuf) },
    { binding: 4, resource: buf(rngBuf) },
  ])

  const wgN = Math.ceil(N / 64)

  function encodeGridRebuild(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.clearCounts)
    pass.setBindGroup(0, clearCountsBind)
    pass.dispatchWorkgroups(wgCells)
    pass.setPipeline(pipe.count)
    pass.setBindGroup(0, countBind)
    pass.dispatchWorkgroups(wgN)
    pass.setPipeline(pipe.prefix)
    pass.setBindGroup(0, prefixBind)
    pass.dispatchWorkgroups(1)
    pass.setPipeline(pipe.fill)
    pass.setBindGroup(0, fillBind)
    pass.dispatchWorkgroups(wgN)
  }

  function encodeForceGrid(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.forceGrid)
    pass.setBindGroup(0, forceGridGroup0)
    pass.setBindGroup(1, forceGridGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  function encodeForceBrute(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.forceBrute)
    pass.setBindGroup(0, forceBruteGroup0)
    pass.setBindGroup(1, forceBruteGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  let neighborBuildMs = 0

  async function rebuildGridTimed(): Promise<void> {
    const t0 = performance.now()
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeGridRebuild(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    neighborBuildMs = performance.now() - t0
  }

  // Warm-up: pipelines are compiled lazily on first dispatch, not at createComputePipeline()
  // time, and that one-time compile latency would otherwise leak into the first measurement
  // (measured: 5.4ms for a 600-particle system built first on the page vs about a fifth of that
  // for a 3000-particle one built after pipelines were already warm). Run one untimed rebuild before
  // the timed one so `neighborBuildMs` reports the rebuild itself, not compilation.
  {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeGridRebuild(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }

  // Initial grid build + force evaluation, timed for Task 9's `neighborBuildMs`, and needed as
  // F(x0) for the first kick of step().
  await rebuildGridTimed()
  {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  async function step(n: number): Promise<void> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    for (let k = 0; k < n; k++) {
      pass.setPipeline(pipe.kick)
      pass.setBindGroup(0, kickBind)
      pass.dispatchWorkgroups(wgN)

      pass.setPipeline(pipe.drift)
      pass.setBindGroup(0, driftBind)
      pass.dispatchWorkgroups(wgN)

      pass.setPipeline(pipe.wrap)
      pass.setBindGroup(0, wrapBind)
      pass.dispatchWorkgroups(wgN)

      encodeGridRebuild(pass)
      encodeForceGrid(pass)

      pass.setPipeline(pipe.kick)
      pass.setBindGroup(0, kickBind)
      pass.dispatchWorkgroups(wgN)

      pass.setPipeline(pipe.thermostat)
      pass.setBindGroup(0, thermostatBind)
      pass.dispatchWorkgroups(wgN)
    }
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    totalSteps += n
  }

  async function positions(): Promise<Float32Array> {
    return readBack(device, posBuf, N * 16)
  }

  async function forces(): Promise<Float32Array> {
    await rebuildGridTimed()
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    return readBack(device, forceBuf, N * 16)
  }

  async function forcesBruteForce(): Promise<Float32Array> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceBrute(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    return readBack(device, forceBuf, N * 16)
  }

  async function kineticEnergyPerDof(): Promise<number> {
    const v = await readBack(device, velBuf, N * 16)
    let sumSq = 0
    for (let i = 0; i < N; i++) {
      const vx = v[i * 4], vy = v[i * 4 + 1], vz = v[i * 4 + 2]
      sumSq += vx * vx + vy * vy + vz * vz
    }
    return sumSq / (3 * N)
  }

  async function totalEnergy(): Promise<number> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeGridRebuild(pass)
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    const [v, u] = await Promise.all([readBack(device, velBuf, N * 16), readBack(device, potentialBuf, N * 4)])
    let kinetic = 0
    for (let i = 0; i < N; i++) {
      const vx = v[i * 4], vy = v[i * 4 + 1], vz = v[i * 4 + 2]
      kinetic += 0.5 * (vx * vx + vy * vy + vz * vz)
    }
    let potential = 0
    for (let i = 0; i < N; i++) potential += u[i]
    return kinetic + potential
  }

  // Proposal half-width for the area move's log-area step u ~ U(-AREA_MOVE_LOG_DELTA,
  // +AREA_MOVE_LOG_DELTA). This is a Monte Carlo move-size tuning knob, not a physical model
  // parameter (it does not appear in any Cooke & Deserno formula and has no effect on the
  // equilibrium distribution, only on how fast the chain explores it), so it lives here rather
  // than in data/params.json. Tuned (see task-5-report.md) so the accepted fraction lands in the
  // 0.2-0.6 range once the Jacobian-corrected, rigid-center-of-mass move below is in place —
  // retuned smaller again after WCA was restored on bonded pairs (that change stiffened the
  // potential enough that the previous value's acceptance dropped under 0.2).
  const AREA_MOVE_LOG_DELTA = 0.003

  async function totalPotentialGPU(): Promise<number> {
    await rebuildGridTimed()
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    const u = await readBack(device, potentialBuf, N * 4)
    let sum = 0
    for (let i = 0; i < N; i++) sum += u[i]
    return sum
  }

  // Minimum-image displacement of a scalar coordinate difference, matching mi() in forces.wgsl —
  // needed to find each lipid's true (unwrapped-relative-to-its-head) center of mass when the
  // head itself may sit anywhere in [0, box).
  function mi1(d: number, box: number): number {
    return d - Math.round(d / box) * box
  }

  async function areaMove(trials: number): Promise<number> {
    let accepted = 0
    for (let t = 0; t < trials; t++) {
      const oldBox: [number, number, number] = [liveBox[0], liveBox[1], liveBox[2]]

      const u = (rng() * 2 - 1) * AREA_MOVE_LOG_DELTA
      const sq = Math.sqrt(Math.exp(u))
      const proposedBox: [number, number, number] = [oldBox[0] * sq, oldBox[1] * sq, oldBox[2]]

      // Reject up front, with no GPU work at all, if the proposal would breach either grid
      // invariant createSystem() enforces at construction time (dims>=3 on the periodic axes,
      // min(box.x,box.y)/2 > bend.r0). AREA_MOVE_LOG_DELTA is tuned small enough that this should
      // essentially never fire once the chain is anywhere near the literature corridor — it is a
      // safety net, not the normal path.
      if (!gridInvariantsHold(proposedBox, computeDims(proposedBox))) continue

      const potentialBefore = await totalPotentialGPU()
      const before = await readBack(device, posBuf, N * 16)

      // Scale each LIPID's center of mass (x,y) by sqrt(s), then translate all three of its
      // beads by that same displacement — never rescale a bead's coordinate directly. This keeps
      // every intramolecular distance (FENE head-tail1, FENE tail1-tail2, bend head-tail2)
      // exactly unchanged by the move, so ΔU below reflects only intermolecular structure, which
      // is what the area coordinate is actually supposed to couple to. The center of mass itself
      // is computed via each lipid's minimum-image offset from its head (mi1), so a lipid whose
      // beads happen to straddle a periodic boundary still gets the correct displacement — the
      // displacement is then applied to the beads' real (possibly-wrapped) stored coordinates, so
      // intramolecular vectors stay bit-exact regardless of which periodic image the head sits in.
      const proposed = new Float32Array(before.length)
      for (let lip = 0; lip < opts.lipids; lip++) {
        const h = lip * 3, t1 = lip * 3 + 1, t2 = lip * 3 + 2
        const hx = before[h * 4], hy = before[h * 4 + 1]
        const t1x = hx + mi1(before[t1 * 4] - hx, oldBox[0])
        const t1y = hy + mi1(before[t1 * 4 + 1] - hy, oldBox[1])
        const t2x = hx + mi1(before[t2 * 4] - hx, oldBox[0])
        const t2y = hy + mi1(before[t2 * 4 + 1] - hy, oldBox[1])
        const comX = (hx + t1x + t2x) / 3
        const comY = (hy + t1y + t2y) / 3
        const dx = comX * (sq - 1)
        const dy = comY * (sq - 1)
        for (const b of [h, t1, t2]) {
          // Explicit re-wrap into [0, newBox) — mirrors wrap_main in integrate.wgsl — so a bead
          // translated just past the edge lands in the cell cell_of() actually expects (which
          // clamps rather than wraps) instead of waiting for the next step()'s wrap pass.
          let x = before[b * 4] + dx
          let y = before[b * 4 + 1] + dy
          x = x - Math.floor(x / proposedBox[0]) * proposedBox[0]
          y = y - Math.floor(y / proposedBox[1]) * proposedBox[1]
          proposed[b * 4] = x
          proposed[b * 4 + 1] = y
          proposed[b * 4 + 2] = before[b * 4 + 2]
          proposed[b * 4 + 3] = before[b * 4 + 3]
        }
      }
      device.queue.writeBuffer(posBuf, 0, proposed)
      liveBox = proposedBox
      await resizeGrid(proposedBox)

      const potentialAfter = await totalPotentialGPU()
      const dU = potentialAfter - potentialBefore
      // Zero-tension Metropolis with the configurational Jacobian from rescaling N=lipids
      // centers of mass by sqrt(A'/A): accept with min(1, exp(-(dU - lipids*kT*ln(A'/A))/kT)) =
      // min(1, exp(lipids*u - dU/kT)), since ln(A'/A) = ln(s) = u exactly. The entropic term
      // favors expansion (positive u) and is what balances the tail-tail attraction's pull
      // toward smaller area — see task-5-report.md for the measurement that showed this term is
      // required (its absence produces a monotonic collapse with a perfectly healthy aggregate
      // acceptance fraction, not a near-zero one).
      const acceptProb = Math.min(1, Math.exp(opts.lipids * u - dU / p.thermostat.kT))
      if (rng() < acceptProb) {
        accepted++
        // Proposed state kept; forceBuf/potentialBuf/grid already reflect it from the
        // totalPotentialGPU() call above — nothing left to resync.
      } else {
        device.queue.writeBuffer(posBuf, 0, before)
        liveBox = oldBox
        await resizeGrid(oldBox)
        // Resync forceBuf/potentialBuf/grid with the reverted positions/box so the next step()
        // kicks off of F(x) at the state actually being kept, not the discarded proposal.
        await totalPotentialGPU()
      }
    }
    return accepted / trials
  }

  return {
    step,
    positions,
    kineticEnergyPerDof,
    totalEnergy,
    areaMove,
    get box(): [number, number, number] {
      return [liveBox[0], liveBox[1], liveBox[2]]
    },
    lipids: opts.lipids,
    get steps() {
      return totalSteps
    },
    forces,
    forcesBruteForce,
    get neighborBuildMs() {
      return neighborBuildMs
    },
  }
}
