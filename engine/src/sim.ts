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
  const rng = mulberry32(opts.seed)

  const positions0 = opts.positions ?? (opts.layout === 'random' ? layoutRandom(opts.lipids, box, p, rng) : layoutBilayer(opts.lipids, box, p, rng))
  const velocities0 = opts.velocities ?? initialVelocities(N, p.thermostat.kT, rng)
  const rngState0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) rngState0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9

  // Cell size: at least r_c + w_c (the largest interaction range, the tail-tail attraction's
  // outer cutoff) so any pair within range of each other is guaranteed to fall in the same cell
  // or one of the 26 neighbors. A smaller cell silently drops forces.
  const cellSize = wcaCutoff(p.beadSizes.tail_tail) + p.attraction.wc
  const dims = [Math.max(1, Math.floor(box[0] / cellSize)), Math.max(1, Math.floor(box[1] / cellSize)), Math.max(1, Math.floor(box[2] / cellSize))]
  const ncells = dims[0] * dims[1] * dims[2]

  // The ±1 neighbor-cell walk in force_main wraps periodically in x,y. With fewer than 3 cells
  // on a periodic axis, +1 and -1 land on the same wrapped cell (or, at 1 cell, all three land
  // on the cell itself), so that cell gets visited twice (2 cells) or three times (1 cell) per
  // particle — every non-bonded force and energy contribution from it is silently doubled or
  // tripled. This is a geometric property of the algorithm, not something a smaller cellSize
  // could fix without breaking the "cell size >= interaction range" guarantee, so it must be
  // caught here rather than produce a quietly-wrong answer.
  if (dims[0] < 3 || dims[1] < 3) {
    throw new Error(
      `сетка соседей: box=[${box[0]},${box[1]},${box[2]}] даёт cellSize=${cellSize.toFixed(4)} и dims=[${dims[0]},${dims[1]},${dims[2]}] — ` +
        `периодическим осям x,y нужно dims>=3, иначе соседний обход по ±1 посещает одну и ту же обёрнутую ячейку дважды/трижды и удваивает/утраивает силы`,
    )
  }
  // Minimum-image convention (mi() in forces.wgsl) assumes each axis sees at most one periodic
  // image within range, i.e. box/2 must exceed every interaction's reach in that axis —
  // including the bend pair (head-tail2, reach ~r0) which mi() also wraps.
  if (Math.min(box[0], box[1]) / 2 <= p.bend.r0) {
    throw new Error(
      `сетка соседей: min(box.x,box.y)/2=${(Math.min(box[0], box[1]) / 2).toFixed(4)} должен быть больше bend.r0=${p.bend.r0} — ` +
        `иначе minimum-image для изгибной пары head-tail2 не однозначен`,
    )
  }

  const { device } = await getGpu()
  const pipe = getPipelines(device)

  const posBuf = storageBuffer(device, positions0)
  const velBuf = storageBuffer(device, velocities0)
  const forceBuf = storageBuffer(device, new Float32Array(N * 4))
  const potentialBuf = storageBuffer(device, new Float32Array(N))
  const countsBuf = storageBuffer(device, new Float32Array(ncells))
  const cellsBuf = storageBuffer(device, new Float32Array(N))
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
  {
    const buf = new ArrayBuffer(32)
    new Uint32Array(buf, 0, 4).set([dims[0], dims[1], dims[2], 0])
    new Float32Array(buf, 16, 4).set([box[0], box[1], box[2], 0])
    device.queue.writeBuffer(gridUniform, 0, buf)
  }

  const boxUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(boxUniform, 0, new Float32Array([box[0], box[1], box[2], 0]))

  const bind = (pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries })
  const buf = (b: GPUBuffer) => ({ buffer: b })

  const clearCountsBind = bind(pipe.clearCounts, 0, [
    { binding: 0, resource: buf(gridUniform) },
    { binding: 2, resource: buf(countsBuf) },
  ])
  const countBind = bind(pipe.count, 0, [
    { binding: 0, resource: buf(gridUniform) },
    { binding: 1, resource: buf(posBuf) },
    { binding: 2, resource: buf(countsBuf) },
  ])
  const prefixBind = bind(pipe.prefix, 0, [
    { binding: 0, resource: buf(gridUniform) },
    { binding: 2, resource: buf(countsBuf) },
    { binding: 4, resource: buf(cellStartBuf) },
    { binding: 5, resource: buf(cursorBuf) },
  ])
  const fillBind = bind(pipe.fill, 0, [
    { binding: 0, resource: buf(gridUniform) },
    { binding: 1, resource: buf(posBuf) },
    { binding: 3, resource: buf(cellsBuf) },
    { binding: 5, resource: buf(cursorBuf) },
  ])

  const forceGridGroup0 = bind(pipe.forceGrid, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const forceGridGroup1 = bind(pipe.forceGrid, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 1, resource: buf(forceBuf) },
    { binding: 2, resource: buf(potentialBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 4, resource: buf(cellStartBuf) },
    { binding: 5, resource: buf(cellsBuf) },
  ])
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
  const wgCells = Math.ceil(ncells / 64)

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

  return {
    step,
    positions,
    kineticEnergyPerDof,
    totalEnergy,
    forces,
    forcesBruteForce,
    get neighborBuildMs() {
      return neighborBuildMs
    },
  }
}
