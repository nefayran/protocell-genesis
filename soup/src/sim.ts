// Task 2: bond formation and breaking inside the dynamics.
//
// This module owns the soup's GPU state and orchestrates three shader modules per integration
// step:
//  - forces.wgsl (engine/wgsl, UNCHANGED) concatenated with soup/wgsl/step.wgsl -- the Langevin
//    force kernel (soup_force_main) plus the fused kick+drift+wrap and kick+thermostat kernels
//    (kick_drift_wrap_main, kick_thermostat_main). Reuses wca_dv/fene_dv/bend_dv/attr_dv/wca_cut
//    and the Params/GridDims uniforms verbatim; step.wgsl adds no potential of its own -- the
//    dynamic-topology bond walk, a full 3-axis mi3/wrap (a bulk soup has no vacuum-facing axis the
//    way a membrane does), and the same kick/drift/thermostat arithmetic
//    engine/wgsl/integrate.wgsl uses, fused into fewer dispatches (see step.wgsl's header: the
//    membrane engine's own per-step dispatch count was measurably NOT the bottleneck at its scale,
//    but interleaving many small dispatches dominated wall time here -- task-2-report.md).
//  - forces.wgsl concatenated with soup/wgsl/bond.wgsl -- bond_form_main/bond_break_main, the
//    Metropolis bond dynamics. See bond.wgsl's header for how detailed balance is carried end to
//    end from soup/src/rules.ts's own acceptanceProbability/attemptProbability, not re-derived.
//  - engine/wgsl/neighbor.wgsl (UNCHANGED) for the grid build -- direction-agnostic (no box/
//    periodicity reference), reused exactly as the membrane engine uses it.
//
// No numeric model constant is written in this file: every physical number comes from
// loadSoup()/loadParams() at runtime; params.test.ts's literal scanner enforces this over
// soup/src and soup/wgsl.

import forcesWgsl from '../../engine/wgsl/forces.wgsl?raw'
import neighborWgsl from '../../engine/wgsl/neighbor.wgsl?raw'
import stepWgsl from '../wgsl/step.wgsl?raw'
import bondWgsl from '../wgsl/bond.wgsl?raw'
import { getGpu, readBack, storageBuffer } from '../../engine/src/gpu'
import { loadParams, paramsToUniform, wcaCutoff, type Params } from '../../engine/src/params'
import { acceptanceProbability, assertRulesConsistent, attemptProbability, loadSoup, type Rule } from './rules'

export interface CreateSoupOpts {
  box: [number, number, number]
  seed: number
  kT: number
  /** Overrides data/soup.json's `start` counts by monomer id. Merged over the file's defaults, not
   * a full replacement -- an id not mentioned here keeps the file's count. */
  start?: Record<string, number>
  /** Convenience override for the catalyst monomer's count specifically (soup.json calls it `M`,
   * but this reads the schema's `kind: "catalyst"` monomer rather than assuming the id, so a
   * future rename of the id would not silently stop working). Takes precedence over `start` for
   * that one id. */
  catalystCount?: number
}

export interface SoupSystem {
  /** Advances n Langevin + bond-Monte-Carlo steps, encoded as one command buffer. */
  step(n: number): Promise<void>
  /** 4 floats per particle: x, y, z, kind index (position into data/soup.json's `monomers`). */
  particles(): Promise<Float32Array>
  /** Pairs of particle indices [i0, j0, i1, j1, ...], one entry per currently active bond. */
  bonds(): Promise<Uint32Array>
  /** Cumulative event counts since creation, keyed by data/soup.json rule id (e.g. "cc_bond"). */
  events(): Promise<Record<string, number>>
  /** Per-monomer-id particle counts, active bond count, and total charge (sum of each monomer's
   * `charge` field if data/soup.json ever defines one; the current schema does not, so this is 0
   * by construction -- not a hardcoded placeholder, a schema-driven sum that happens to be empty
   * today). Compared before/after step() by tests/soup-bonds.test.ts: particle counts and charge
   * must be EXACTLY unchanged (nothing here ever creates or destroys a particle), bonds must be
   * able to change (that is the whole point of this task). */
  invariants(): Promise<{ monomers: Record<string, number>; bonds: number; charge: number }>
}

const NONE_U32 = 0xffffffff

// --- seeded RNG for reproducible initial layouts -------------------------------------------------

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
  const u1 = Math.max(rng(), 1e-9)
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

// --- rule resolution: data/soup.json's symbolic (id, id) rules -> numeric kind-index uniforms ----

interface ResolvedRule {
  ruleIdx: number
  bond: Rule
  brk: Rule
  kindA: number
  kindB: number
  slotRoleA: number
  slotRoleB: number
}

// Slot role codes shared with soup/wgsl/bond.wgsl's roleOf(): 0 = chain pool (slots 0,1), 1 = head
// slot (slot 2 fixed), 2 = single slot (slot 0 fixed). Resolved from each side's monomer KIND
// (carbon/head), never from a hardcoded id, so a future rename of "C"/"O" in data/soup.json would
// not need a matching change here.
function slotRole(thisKind: string, otherKind: string): number {
  if (thisKind === 'carbon') return otherKind === 'carbon' ? 0 : 1
  if (thisKind === 'head') return 2
  throw new Error(`soup/src/sim.ts: правило связывает мономер вида "${thisKind}" — поддержаны только carbon и head`)
}

function resolveRules(soup: ReturnType<typeof loadSoup>): { rules: ResolvedRule[]; catalystKind: number } {
  const kindIndex = new Map(soup.monomers.map((m, idx) => [m.id, idx]))
  const catalystMonomer = soup.monomers.find((m) => m.kind === 'catalyst')
  if (!catalystMonomer) throw new Error('data/soup.json: не найден мономер вида catalyst')
  const catalystKind = kindIndex.get(catalystMonomer.id)!

  const bondRules = soup.rules.filter((r) => r.kind === 'bond')
  const rules: ResolvedRule[] = bondRules.map((bond, ruleIdx) => {
    const brk = soup.rules.find((r) => r.kind === 'break' && r.a === bond.a && r.b === bond.b)
    if (!brk) throw new Error(`data/soup.json: у правила ${bond.id} нет парного разрыва`)
    const ma = soup.monomers.find((m) => m.id === bond.a)
    const mb = soup.monomers.find((m) => m.id === bond.b)
    if (!ma || !mb) throw new Error(`data/soup.json: правило ${bond.id} ссылается на неописанный мономер`)
    return {
      ruleIdx,
      bond,
      brk,
      kindA: kindIndex.get(bond.a)!,
      kindB: kindIndex.get(bond.b)!,
      slotRoleA: slotRole(ma.kind, mb.kind),
      slotRoleB: slotRole(mb.kind, ma.kind),
    }
  })
  if (rules.length > 4) {
    throw new Error(`soup/src/sim.ts: ${rules.length} правил образования связи — BondParams вмещает не больше 4`)
  }
  return { rules, catalystKind }
}

function packVec4(values: number[]): number[] {
  const out = [0, 0, 0, 0]
  for (let i = 0; i < Math.min(4, values.length); i++) out[i] = values[i]
  return out
}

// --- system --------------------------------------------------------------------------------------

interface SoupPipelines {
  device: GPUDevice
  soupForce: GPUComputePipeline
  kickDriftWrap: GPUComputePipeline
  kickThermostat: GPUComputePipeline
  bondForm: GPUComputePipeline
  bondBreak: GPUComputePipeline
  clearCounts: GPUComputePipeline
  count: GPUComputePipeline
  prefix: GPUComputePipeline
  fill: GPUComputePipeline
}

let cached: SoupPipelines | undefined

function getSoupPipelines(device: GPUDevice): SoupPipelines {
  if (cached && cached.device === device) return cached
  const forceModule = device.createShaderModule({ code: `${forcesWgsl}\n${stepWgsl}` })
  const bondModule = device.createShaderModule({ code: `${forcesWgsl}\n${bondWgsl}` })
  const neighborModule = device.createShaderModule({ code: neighborWgsl })
  const cp = (module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })
  cached = {
    device,
    soupForce: cp(forceModule, 'soup_force_main'),
    kickDriftWrap: cp(forceModule, 'kick_drift_wrap_main'),
    kickThermostat: cp(forceModule, 'kick_thermostat_main'),
    bondForm: cp(bondModule, 'bond_form_main'),
    bondBreak: cp(bondModule, 'bond_break_main'),
    clearCounts: cp(neighborModule, 'clear_counts_main'),
    count: cp(neighborModule, 'count_main'),
    prefix: cp(neighborModule, 'prefix_main'),
    fill: cp(neighborModule, 'fill_main'),
  }
  return cached
}

export async function createSoup(opts: CreateSoupOpts): Promise<SoupSystem> {
  const soup = loadSoup()
  assertRulesConsistent(soup)
  if (soup.monomers.length > 4) {
    throw new Error(`data/soup.json: ${soup.monomers.length} видов мономеров — шейдер вмещает не больше 4`)
  }
  const baseParams = loadParams()
  const p: Params = { ...baseParams, thermostat: { ...baseParams.thermostat, kT: opts.kT } }

  const { rules, catalystKind } = resolveRules(soup)

  const startCounts: Record<string, number> = { ...soup.start, ...(opts.start ?? {}) }
  if (opts.catalystCount !== undefined) {
    const catalystMonomer = soup.monomers.find((m) => m.kind === 'catalyst')!
    startCounts[catalystMonomer.id] = opts.catalystCount
  }
  const countsByKind = soup.monomers.map((m) => startCounts[m.id] ?? 0)
  const N = countsByKind.reduce((a, b) => a + b, 0)
  if (N === 0) throw new Error('createSoup: стартовый состав пуст')

  const box = opts.box
  const rng = mulberry32(opts.seed)

  // Cell size: at least the largest interaction reach (WCA contact for the largest pairwise size,
  // plus the tail-tail attraction's outer cutoff w_c) so any pair within range of each other is
  // guaranteed to land in the same cell or one of its 26 neighbours -- same reasoning
  // engine/src/sim.ts's own `cellSize` comment gives, generalised from a fixed lipid pair to
  // whichever two of the soup's own species (by data/soup.json's radiusSigma) are largest.
  const maxRadiusSigma = Math.max(...soup.monomers.map((m) => m.radiusSigma))
  const maxB = p.sigma * maxRadiusSigma
  const cellSize = wcaCutoff(maxB) + p.attraction.wc

  function computeDims(b: [number, number, number]): [number, number, number] {
    return [Math.max(1, Math.floor(b[0] / cellSize)), Math.max(1, Math.floor(b[1] / cellSize)), Math.max(1, Math.floor(b[2] / cellSize))]
  }

  // Full 3-axis periodicity here (unlike engine/src/sim.ts's membrane, which leaves z open for a
  // bilayer in vacuum) -- a bulk soup has no preferred axis, so the ±1 neighbour-cell walk wraps
  // z too (soup/wgsl/step.wgsl, soup/wgsl/bond.wgsl), and needs dims>=3 on z as well as x,y for
  // the same reason engine/src/sim.ts's gridInvariantsHold needs it on x,y: with fewer than 3
  // cells on a periodic axis the ±1 wrap revisits a cell 2x or 3x, silently doubling or tripling
  // every force/bond-attempt contribution from it. min(box)/2 > bend.r0 guards the minimum-image
  // convention (mi3) the same way, generalised to all three axes.
  function gridInvariantsHold(b: [number, number, number], d: [number, number, number]): boolean {
    return (
      d[0] >= 3 &&
      d[1] >= 3 &&
      d[2] >= 3 &&
      Math.min(b[0], b[1], b[2]) / 2 > p.bend.r0
    )
  }

  const dims = computeDims(box)
  if (!gridInvariantsHold(box, dims)) {
    throw new Error(
      `сетка соседей: box=[${box[0]},${box[1]},${box[2]}] даёт cellSize=${cellSize.toFixed(4)}, dims=[${dims[0]},${dims[1]},${dims[2]}] ` +
        `и min(box)/2=${(Math.min(box[0], box[1], box[2]) / 2).toFixed(4)} — нужно dims>=3 на всех трёх осях и min(box)/2 > bend.r0=${p.bend.r0}`,
    )
  }
  const ncells = dims[0] * dims[1] * dims[2]

  const { device } = await getGpu()
  const pipe = getSoupPipelines(device)

  // --- initial state: a jittered lattice, not independent uniform placement -----------------------
  // Independent uniform placement has no minimum-separation guarantee: at data/soup.json's default
  // density (~13100 particles in a 30^3 box, ~1.27 sigma average spacing against WCA cores up to
  // ~1.35 sigma) it is only a matter of trials before two particles land near-coincident, and
  // wca_dv diverges as r -> 0 -- measured here as positions overflowing float32 range (+-2^23 and
  // beyond) after exactly ONE kick+drift, not a slow drift. A lattice with the SAME spacing has no
  // such tail risk (nearest-neighbour separation is bounded below by construction); a small jitter
  // keeps it from being a perfectly artificial starting configuration. Species are interleaved by
  // shuffling which lattice SITE each particle gets (a seeded Fisher-Yates), not by shuffling
  // positions within a kind's own block, so the soup starts well-mixed rather than segregated into
  // one spatial region per monomer kind.
  const positions0 = new Float32Array(N * 4)
  const velocities0 = new Float32Array(N * 4)
  {
    let nx = Math.max(1, Math.ceil(Math.cbrt(N)))
    while (nx * nx * nx < N) nx++
    const spacing: [number, number, number] = [box[0] / nx, box[1] / nx, box[2] / nx]
    const jitterFrac = 0.15
    const nSites = nx * nx * nx
    const siteOrder = Array.from({ length: nSites }, (_, i) => i)
    for (let i = siteOrder.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      const tmp = siteOrder[i]
      siteOrder[i] = siteOrder[j]
      siteOrder[j] = tmp
    }
    let idx = 0
    for (let kind = 0; kind < countsByKind.length; kind++) {
      for (let c = 0; c < countsByKind[kind]; c++) {
        const site = siteOrder[idx]
        const ix = site % nx
        const iy = Math.floor(site / nx) % nx
        const iz = Math.floor(site / (nx * nx))
        positions0[idx * 4 + 0] = (ix + 0.5) * spacing[0] + (rng() * 2 - 1) * jitterFrac * spacing[0]
        positions0[idx * 4 + 1] = (iy + 0.5) * spacing[1] + (rng() * 2 - 1) * jitterFrac * spacing[1]
        positions0[idx * 4 + 2] = (iz + 0.5) * spacing[2] + (rng() * 2 - 1) * jitterFrac * spacing[2]
        positions0[idx * 4 + 3] = kind
        const s = Math.sqrt(opts.kT)
        velocities0[idx * 4 + 0] = s * gaussian(rng)
        velocities0[idx * 4 + 1] = s * gaussian(rng)
        velocities0[idx * 4 + 2] = s * gaussian(rng)
        idx++
      }
    }
  }

  const bondSlots0 = new Uint32Array(N * 3).fill(NONE_U32)
  const bondRng0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) bondRng0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9
  const thermoRng0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) thermoRng0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 3266489917) ^ 0x85ebca6b

  const posBuf = storageBuffer(device, positions0)
  const velBuf = storageBuffer(device, velocities0)
  const forceBuf = storageBuffer(device, new Float32Array(N * 4))
  const cellsBuf = storageBuffer(device, new Float32Array(N))
  const countsBuf = storageBuffer(device, new Float32Array(ncells))
  const cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
  const cursorBuf = storageBuffer(device, new Float32Array(ncells))

  const bondSlotsBuf = device.createBuffer({
    size: bondSlots0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(bondSlotsBuf, 0, bondSlots0)

  const bondRngBuf = device.createBuffer({ size: bondRng0.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(bondRngBuf, 0, bondRng0)
  const thermoRngBuf = device.createBuffer({ size: thermoRng0.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(thermoRngBuf, 0, thermoRng0)

  const eventsInit = new Uint32Array(rules.length * 2)
  const eventsBuf = device.createBuffer({
    size: eventsInit.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(eventsBuf, 0, eventsInit)

  const paramsUniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(paramsUniform, 0, paramsToUniform(p))

  const gridUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const bytes = new ArrayBuffer(32)
    new Uint32Array(bytes, 0, 4).set([dims[0], dims[1], dims[2], 0])
    new Float32Array(bytes, 16, 4).set([box[0], box[1], box[2], 0])
    device.queue.writeBuffer(gridUniform, 0, bytes)
  }

  const speciesUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const radius = packVec4(soup.monomers.map((m) => m.radiusSigma))
    const polar = packVec4(soup.monomers.map((m) => (m.polar ? 1 : 0)))
    device.queue.writeBuffer(speciesUniform, 0, new Float32Array([...radius, ...polar]))
  }

  // BondParams: acceptProbForm/acceptProbBreak come DIRECTLY from soup/src/rules.ts's
  // acceptanceProbability(rule, kT) -- the same function tests/soup-rules.test.ts checks against
  // forwardBackwardRatio -- and attemptProbForm/attemptProbBreak from its attemptProbability(rule,
  // dt), so this is a straight upload of Task 1's own numbers, not a re-derivation.
  const dt = p.integrator.dt
  const eventRuleIds: [string, string][] = rules.map((r) => [r.bond.id, r.brk.id])
  const bondParamsUniform = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const kindA = packVec4(rules.map((r) => r.kindA))
    const kindB = packVec4(rules.map((r) => r.kindB))
    const attemptProbForm = packVec4(rules.map((r) => attemptProbability(r.bond, dt)))
    const attemptProbBreak = packVec4(rules.map((r) => attemptProbability(r.brk, dt)))
    const acceptProbForm = packVec4(rules.map((r) => acceptanceProbability(r.bond, opts.kT)))
    const acceptProbBreak = packVec4(rules.map((r) => acceptanceProbability(r.brk, opts.kT)))
    const requiresCatalyst = packVec4(rules.map((r) => (r.bond.requiresCatalyst ? 1 : 0)))
    const slotRoleA = packVec4(rules.map((r) => r.slotRoleA))
    const slotRoleB = packVec4(rules.map((r) => r.slotRoleB))
    device.queue.writeBuffer(
      bondParamsUniform,
      0,
      new Float32Array([
        ...kindA,
        ...kindB,
        ...attemptProbForm,
        ...attemptProbBreak,
        ...acceptProbForm,
        ...acceptProbBreak,
        ...requiresCatalyst,
        ...slotRoleA,
        ...slotRoleB,
        catalystKind,
        0,
        0,
        0,
      ]),
    )
  }

  const bind = (pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries })
  const buf = (b: GPUBuffer) => ({ buffer: b })

  // --- grid rebuild (engine/wgsl/neighbor.wgsl, unchanged) ---------------------------------------
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
  const wgCells = Math.ceil(ncells / 64)

  // --- soup force + wrap (forces.wgsl + step.wgsl) -----------------------------------------------
  const soupForceGroup0 = bind(pipe.soupForce, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const soupForceGroup1 = bind(pipe.soupForce, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 1, resource: buf(forceBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 4, resource: buf(cellStartBuf) },
    { binding: 5, resource: buf(cellsBuf) },
    { binding: 7, resource: buf(bondSlotsBuf) },
    { binding: 8, resource: buf(speciesUniform) },
  ])
  const kickDriftWrapGroup0 = bind(pipe.kickDriftWrap, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const kickDriftWrapGroup1 = bind(pipe.kickDriftWrap, 1, [
    { binding: 6, resource: buf(posBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 10, resource: buf(velBuf) },
    { binding: 11, resource: buf(forceBuf) },
  ])
  const kickThermostatGroup0 = bind(pipe.kickThermostat, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const kickThermostatGroup1 = bind(pipe.kickThermostat, 1, [
    { binding: 10, resource: buf(velBuf) },
    { binding: 11, resource: buf(forceBuf) },
    { binding: 12, resource: buf(thermoRngBuf) },
  ])

  // --- bond formation/breaking (forces.wgsl + bond.wgsl) -----------------------------------------
  const bondFormGroup0 = bind(pipe.bondForm, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const bondFormGroup1 = bind(pipe.bondForm, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 4, resource: buf(cellStartBuf) },
    { binding: 5, resource: buf(cellsBuf) },
    { binding: 6, resource: buf(bondSlotsBuf) },
    { binding: 7, resource: buf(speciesUniform) },
    { binding: 8, resource: buf(eventsBuf) },
    { binding: 9, resource: buf(bondRngBuf) },
  ])
  const bondFormGroup2 = bind(pipe.bondForm, 2, [{ binding: 0, resource: buf(bondParamsUniform) }])
  const bondBreakGroup1 = bind(pipe.bondBreak, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 6, resource: buf(bondSlotsBuf) },
    { binding: 8, resource: buf(eventsBuf) },
    { binding: 9, resource: buf(bondRngBuf) },
  ])
  const bondBreakGroup2 = bind(pipe.bondBreak, 2, [{ binding: 0, resource: buf(bondParamsUniform) }])

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

  function encodeSoupForce(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.soupForce)
    pass.setBindGroup(0, soupForceGroup0)
    pass.setBindGroup(1, soupForceGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  // Initial grid + force, needed as F(x0) for the first kick.
  {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeGridRebuild(pass)
    encodeSoupForce(pass)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  function encodeOneIntegrationStep(pass: GPUComputePassEncoder) {
    // First Verlet half-kick + drift + 3-axis periodic wrap, fused into one dispatch (see
    // soup/wgsl/step.wgsl's kick_drift_wrap_main header) -- exactly kick_main+drift_main+a
    // 3-axis wrap from engine/wgsl/integrate.wgsl's own formulas, not a new integrator.
    pass.setPipeline(pipe.kickDriftWrap)
    pass.setBindGroup(0, kickDriftWrapGroup0)
    pass.setBindGroup(1, kickDriftWrapGroup1)
    pass.dispatchWorkgroups(wgN)

    encodeGridRebuild(pass)
    encodeSoupForce(pass)

    // Bond Monte Carlo: formation then breaking, on the freshly rebuilt grid/positions. Two
    // separate, ordered dispatches within the same pass -- never concurrent with each other,
    // see bond.wgsl's header for why that ordering is what makes the i<j dedupe race-free.
    pass.setPipeline(pipe.bondForm)
    pass.setBindGroup(0, bondFormGroup0)
    pass.setBindGroup(1, bondFormGroup1)
    pass.setBindGroup(2, bondFormGroup2)
    pass.dispatchWorkgroups(wgN)

    pass.setPipeline(pipe.bondBreak)
    pass.setBindGroup(1, bondBreakGroup1)
    pass.setBindGroup(2, bondBreakGroup2)
    pass.dispatchWorkgroups(wgN)

    // Second Verlet half-kick + Langevin thermostat, fused into one dispatch.
    pass.setPipeline(pipe.kickThermostat)
    pass.setBindGroup(0, kickThermostatGroup0)
    pass.setBindGroup(1, kickThermostatGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  // Steps per submitted command buffer. A single createSoup+step(50_000) in one page works fine
  // (measured), but a SECOND system's step(50_000) in the SAME page hung forever at
  // onSubmittedWorkDone(), regardless of the second system's parameters (reproduced with
  // catalystCount 0 AND with identical params to the first system) -- narrowed by bisection to the
  // SIZE of the second submission specifically: the same second system's step(100) (a ~900-command
  // pass) completed instantly right after the first system's step(50_000), where step(50_000) (a
  // ~450,000-command pass -- 9 dispatchWorkgroups calls x 50,000 iterations) hung. So it is not
  // "two systems in one page" (engine/src/sim.ts's own gate6-bilayer.test.ts does that, successfully,
  // by calling step() many times with small n) and not catalystCount -- it is a single compute pass
  // grown too large, which some part of the browser's WebGPU stack fails to ever signal complete once
  // it is the SECOND such giant submission in one device session. Splitting step(n) into bounded
  // chunks, each its own submit()+onSubmittedWorkDone() round trip, keeps every single command buffer
  // at the same order of magnitude gate6-bilayer's proven-safe per-submission size (900 commands for
  // n=100) while still letting step(n) advance any n in one call -- the fix is entirely inside step()
  // and changes nothing about what gets computed.
  const STEP_CHUNK = 1000

  async function step(n: number): Promise<void> {
    let done = 0
    while (done < n) {
      const chunk = Math.min(STEP_CHUNK, n - done)
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      for (let k = 0; k < chunk; k++) encodeOneIntegrationStep(pass)
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      done += chunk
    }
  }

  async function stepPhasesDEBUG(n: number): Promise<Record<string, number>> {
    async function timePhase(label: string, fn: (pass: GPUComputePassEncoder) => void): Promise<[string, number]> {
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      for (let k = 0; k < n; k++) fn(pass)
      pass.end()
      const t0 = performance.now()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      return [label, (performance.now() - t0) / n]
    }
    const out: Record<string, number> = {}
    const phases: [string, (pass: GPUComputePassEncoder) => void][] = [
      [
        'kickDriftWrap',
        (pass) => {
          pass.setPipeline(pipe.kickDriftWrap)
          pass.setBindGroup(0, kickDriftWrapGroup0)
          pass.setBindGroup(1, kickDriftWrapGroup1)
          pass.dispatchWorkgroups(wgN)
        },
      ],
      ['gridRebuild', (pass) => encodeGridRebuild(pass)],
      ['soupForce', (pass) => encodeSoupForce(pass)],
      [
        'bondForm',
        (pass) => {
          pass.setPipeline(pipe.bondForm)
          pass.setBindGroup(0, bondFormGroup0)
          pass.setBindGroup(1, bondFormGroup1)
          pass.setBindGroup(2, bondFormGroup2)
          pass.dispatchWorkgroups(wgN)
        },
      ],
      [
        'bondBreak',
        (pass) => {
          pass.setPipeline(pipe.bondBreak)
          pass.setBindGroup(1, bondBreakGroup1)
          pass.setBindGroup(2, bondBreakGroup2)
          pass.dispatchWorkgroups(wgN)
        },
      ],
      [
        'kickThermostat',
        (pass) => {
          pass.setPipeline(pipe.kickThermostat)
          pass.setBindGroup(0, kickThermostatGroup0)
          pass.setBindGroup(1, kickThermostatGroup1)
          pass.dispatchWorkgroups(wgN)
        },
      ],
      [
        'fullStepEquivalent',
        (pass) => {
          pass.setPipeline(pipe.kickDriftWrap)
          pass.setBindGroup(0, kickDriftWrapGroup0)
          pass.setBindGroup(1, kickDriftWrapGroup1)
          pass.dispatchWorkgroups(wgN)
          encodeGridRebuild(pass)
          encodeSoupForce(pass)
          pass.setPipeline(pipe.bondForm)
          pass.setBindGroup(0, bondFormGroup0)
          pass.setBindGroup(1, bondFormGroup1)
          pass.setBindGroup(2, bondFormGroup2)
          pass.dispatchWorkgroups(wgN)
          pass.setPipeline(pipe.bondBreak)
          pass.setBindGroup(1, bondBreakGroup1)
          pass.setBindGroup(2, bondBreakGroup2)
          pass.dispatchWorkgroups(wgN)
          pass.setPipeline(pipe.kickThermostat)
          pass.setBindGroup(0, kickThermostatGroup0)
          pass.setBindGroup(1, kickThermostatGroup1)
          pass.dispatchWorkgroups(wgN)
        },
      ],
    ]
    for (const [label, fn] of phases) {
      const [l, ms] = await timePhase(label, fn)
      out[l] = ms
    }
    return out
  }

  async function particles(): Promise<Float32Array> {
    return readBack(device, posBuf, N * 16)
  }

  async function readBondSlots(): Promise<Uint32Array> {
    const raw = await readBack(device, bondSlotsBuf, N * 3 * 4)
    return new Uint32Array(raw.buffer, raw.byteOffset, N * 3)
  }

  async function bonds(): Promise<Uint32Array> {
    const slots = await readBondSlots()
    const out: number[] = []
    for (let i = 0; i < N; i++) {
      for (let s = 0; s < 3; s++) {
        const j = slots[i * 3 + s]
        if (j !== NONE_U32 && j > i) out.push(i, j)
      }
    }
    return new Uint32Array(out)
  }

  async function events(): Promise<Record<string, number>> {
    const raw = await readBack(device, eventsBuf, eventsInit.byteLength)
    const u32 = new Uint32Array(raw.buffer, raw.byteOffset, rules.length * 2)
    const out: Record<string, number> = {}
    for (let r = 0; r < rules.length; r++) {
      out[eventRuleIds[r][0]] = u32[r * 2 + 0]
      out[eventRuleIds[r][1]] = u32[r * 2 + 1]
    }
    return out
  }

  async function invariants(): Promise<{ monomers: Record<string, number>; bonds: number; charge: number }> {
    const pos = await particles()
    const monomers: Record<string, number> = {}
    for (const m of soup.monomers) monomers[m.id] = 0
    let charge = 0
    for (let i = 0; i < N; i++) {
      const kind = Math.round(pos[i * 4 + 3])
      const m = soup.monomers[kind]
      monomers[m.id] = (monomers[m.id] ?? 0) + 1
      charge += (m as unknown as { charge?: number }).charge ?? 0
    }
    const bondCount = (await bonds()).length / 2
    return { monomers, bonds: bondCount, charge }
  }

  return { step, particles, bonds, events, invariants, stepPhasesDEBUG: stepPhasesDEBUG as any }
}
