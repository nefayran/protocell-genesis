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
import { acceptanceProbability, assertRulesConsistent, attemptProbability, loadSoup, type Rule, type Soup } from './rules'
import { detectStage, type Stage, type StageEvidence } from './stages'

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
  /** Per-particle force from the grid path (rebuilds the grid for current positions first).
   * perf2-report.md correctness gate: compared against forcesBruteForce() to floating-point
   * tolerance, mirroring engine/src/sim.ts's own forces()/forcesBruteForce() pair. */
  forces(): Promise<Float32Array>
  /** Same physics as forces(), computed by an O(N^2) pair loop with no neighbour grid at all --
   * the reference implementation forces() is checked against. */
  forcesBruteForce(): Promise<Float32Array>
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
  /** The box this system was created with (`CreateSoupOpts.box`), fixed for the system's lifetime.
   * Task 4's reconciliation of the deviation Task 3 flagged: soup/src/stages.ts's detectStage
   * reads this instead of taking box as a second parameter, so `stageOf(sys)` (and this file's own
   * runUntil, below) never has to re-thread it. */
  box: [number, number, number]
  /** Steps in batches of `sampleEvery`, running detectStage after each batch and appending
   * `{steps, stage, evidence}` to a trace (also logged one line at a time, via console.log, so a
   * long run is diagnosable while it is still in progress). Stops early the first batch whose
   * stage equals the target `stage`; otherwise runs until `maxSteps` is exhausted. `steps` in the
   * trace and in the return value is the CUMULATIVE step count taken by this call (independent of
   * whatever `globalStep` step() itself has already advanced from earlier calls) -- it is what a
   * caller graphs a trajectory against, not a raw step()-internal counter. */
  runUntil(
    stage: Stage,
    opts: { maxSteps: number; sampleEvery: number },
  ): Promise<{ reached: boolean; steps: number; trace: { steps: number; stage: Stage; evidence: StageEvidence }[] }>
  /** Destroys every GPUBuffer this system owns. getGpu() memoizes ONE device for the whole page,
   * so a caller that creates a second SoupSystem in the same page (viewer/run.ts's "start a new
   * run" button, tests/run-ui.test.ts's second-run regression) leaves the FIRST system's buffers
   * (positions, velocities, bond slots, Verlet lists, every uniform -- ~20 GPUBuffers, see
   * createSoup's own allocations) alive on that shared device unless something explicitly destroys
   * them: unlike CPU memory, a GPUBuffer whose JS object becomes unreachable is not promptly freed
   * by garbage collection, so repeated runs without disposal accumulate GPU-side allocations
   * without bound. Pipelines (module-level `cached`, keyed by device) are NOT touched here -- they
   * hold no reference to any particular system's buffers (bind groups, which do, are always
   * recreated fresh per createSoup call) and are safe, and intended, to be reused by the next
   * system on the same device. Safe to call more than once (GPUBuffer.destroy() is a no-op on an
   * already-destroyed buffer per the WebGPU spec) and safe to call on a system whose step() is not
   * currently in flight; callers must not call step()/particles()/etc. afterward. */
  dispose(): void
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
  sortedGather: boolean
  soupForce: GPUComputePipeline
  /** perf2-report.md correctness gate: O(N^2) reference force, no grid -- see forces()/
   * forcesBruteForce() below, mirroring engine/src/sim.ts's own pair. */
  soupForceBrute: GPUComputePipeline
  kickDriftWrap: GPUComputePipeline
  kickThermostat: GPUComputePipeline
  bondForm: GPUComputePipeline
  bondBreak: GPUComputePipeline
  clearCounts: GPUComputePipeline
  count: GPUComputePipeline
  prefix: GPUComputePipeline
  fill: GPUComputePipeline
  /** perf2-report.md, candidate (b): gathers positions into cell-sorted order (using the same
   * permutation fill_main already produces in `cellIdx`) so the O(candidates) neighbour walk in
   * soup_force_main/bond_form_main reads a contiguous array instead of scattering through `pos2`
   * at arbitrary original indices. */
  gatherSorted: GPUComputePipeline
  /** perf2-report.md, STEP 1 diagnosis: counts candidates examined and pairs within the actual
   * interaction range for one dispatch of the SAME cell walk soup_force_main runs -- never used in
   * the real step loop, only by forceCandidateStatsDEBUG below. */
  forceStats: GPUComputePipeline
  /** perf2-report.md, candidate (c): Verlet list build/maintenance kernels -- only ever dispatched
   * when verletList.enabled is true. */
  buildVerletList: GPUComputePipeline
  snapshotPositions: GPUComputePipeline
  resetMaxDrift: GPUComputePipeline
  maxDrift: GPUComputePipeline
  soupForceList: GPUComputePipeline
  bondFormList: GPUComputePipeline
}

let cached: SoupPipelines | undefined

// perf2-report.md, candidate (b): which entry point to compile for the force/bond-form kernels --
// see NeighborGrid.sortedGather's doc comment (soup/src/rules.ts) for why this is a pipeline
// choice, not a runtime branch. Threaded into the (device-keyed) pipeline cache key too: this
// engine only ever runs with ONE data/soup.json per process (loaded once at module import), so in
// practice the cache is never asked for the other variant on the same device, but keying on it
// explicitly documents that dependency rather than leaving it implicit.
function getSoupPipelines(device: GPUDevice, sortedGather: boolean): SoupPipelines {
  if (cached && cached.device === device && cached.sortedGather === sortedGather) return cached
  const forceModule = device.createShaderModule({ code: `${forcesWgsl}\n${stepWgsl}` })
  const bondModule = device.createShaderModule({ code: `${forcesWgsl}\n${bondWgsl}` })
  const neighborModule = device.createShaderModule({ code: neighborWgsl })
  const cp = (module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipeline({ label: entryPoint, layout: 'auto', compute: { module, entryPoint } })
  cached = {
    device,
    sortedGather,
    soupForce: cp(forceModule, sortedGather ? 'soup_force_main' : 'soup_force_main_unsorted'),
    soupForceBrute: cp(forceModule, 'soup_force_brute_main'),
    kickDriftWrap: cp(forceModule, 'kick_drift_wrap_main'),
    kickThermostat: cp(forceModule, 'kick_thermostat_main'),
    bondForm: cp(bondModule, sortedGather ? 'bond_form_main' : 'bond_form_main_unsorted'),
    bondBreak: cp(bondModule, 'bond_break_main'),
    clearCounts: cp(neighborModule, 'clear_counts_main'),
    count: cp(neighborModule, 'count_main'),
    prefix: cp(neighborModule, 'prefix_main'),
    fill: cp(neighborModule, 'fill_main'),
    gatherSorted: cp(forceModule, 'soup_gather_sorted_main'),
    forceStats: cp(forceModule, 'soup_force_stats_main'),
    buildVerletList: cp(forceModule, 'soup_build_verlet_list_main'),
    snapshotPositions: cp(forceModule, 'soup_snapshot_positions_main'),
    resetMaxDrift: cp(forceModule, 'soup_reset_max_drift_main'),
    maxDrift: cp(forceModule, 'soup_max_drift_main'),
    soupForceList: cp(forceModule, 'soup_force_list_main'),
    bondFormList: cp(bondModule, 'bond_form_list_main'),
  }
  return cached
}

/** The neighbour-grid geometry a given `box` would get, and whether it is even valid -- the exact
 * derivation createSoup() itself needs before it may allocate a single GPU buffer (walkRadius from
 * the species' own interaction range, the grid dims that follow from box/cellSize, and the two
 * periodicity/minimum-image invariants below), pulled out into its own pure, GPU-free function so
 * a caller (a viewer letting the user pick their own box, item 3 of the 2026-08 UI-fixes task) can
 * preview the cost -- particle count, grid dims -- and catch an invalid box BEFORE calling the async,
 * GPU-allocating createSoup() at all, rather than only finding out from a caught exception after
 * paying for the attempt. createSoup() below calls this too, so there is exactly one definition of
 * "is this box valid" for the soup engine, not a second one drifting out of sync in a viewer. */
export interface SoupPlan {
  /** Total starting particle count for this composition (independent of box). */
  N: number
  box: [number, number, number]
  /** Neighbour-grid cell side, sigma. */
  cellSize: number
  /** How many cells the neighbour walk must cover on each side (Verlet-list build radius when the
   * list is enabled, this soup's own regular walk radius otherwise). */
  effectiveWalkRadius: number
  /** Grid cell counts, one per axis -- `floor(box[axis]/cellSize)`, at least 1. */
  dims: [number, number, number]
  ncells: number
  /** `2*effectiveWalkRadius+1` -- the minimum dims[axis] the periodic ±effectiveWalkRadius cell
   * walk needs on every axis: with fewer cells than this on a periodic axis, the wrap revisits a
   * cell more than once, silently multiplying every force/bond-attempt/list-build contribution
   * from it. The OTHER half of `valid` (not separately named) is the minimum-image guard,
   * min(box)/2 > bend.r0, unaffected by cell count -- it is about the bond's own reach. */
  minCells: number
  valid: boolean
  /** Populated (non-null) exactly when `!valid` -- the exact message createSoup() itself would
   * throw for this box, so a caller can show it without waiting for that throw. */
  reason: string | null
}

export function planSoupGrid(box: [number, number, number], startCounts?: Record<string, number>): SoupPlan {
  const soup = loadSoup()
  const p = loadParams()

  const counts: Record<string, number> = { ...soup.start, ...(startCounts ?? {}) }
  const N = soup.monomers.reduce((sum, m) => sum + (counts[m.id] ?? 0), 0)

  // Same derivation as createSoup's own interactionRange/cellSize/walkRadius/listBuildWalkRadius --
  // see that function's own comments for the reasoning; not repeated here since it does not depend
  // on `box` at all (only on data/soup.json's own species/neighborGrid/verletList settings), so a
  // caller previewing many candidate box sizes is not re-deriving anything box-independent.
  const maxRadiusSigma = Math.max(...soup.monomers.map((m) => m.radiusSigma))
  const maxB = p.sigma * maxRadiusSigma
  const interactionRange = wcaCutoff(maxB) + p.attraction.wc
  const cellSize = interactionRange / soup.neighborGrid.cellDivisor
  const walkRadius = Math.ceil(interactionRange / cellSize)
  const verlet = soup.verletList
  const listRange = interactionRange + verlet.skin
  const listBuildWalkRadius = Math.ceil(listRange / cellSize)
  const effectiveWalkRadius = verlet.enabled ? listBuildWalkRadius : walkRadius

  const dims: [number, number, number] = [
    Math.max(1, Math.floor(box[0] / cellSize)),
    Math.max(1, Math.floor(box[1] / cellSize)),
    Math.max(1, Math.floor(box[2] / cellSize)),
  ]
  const ncells = dims[0] * dims[1] * dims[2]
  const minCells = 2 * effectiveWalkRadius + 1
  const gridOk = dims[0] >= minCells && dims[1] >= minCells && dims[2] >= minCells
  const miOk = Math.min(box[0], box[1], box[2]) / 2 > p.bend.r0
  const valid = gridOk && miOk
  const reason = valid
    ? null
    : `сетка соседей: box=[${box[0]},${box[1]},${box[2]}] даёт cellSize=${cellSize.toFixed(4)}, dims=[${dims[0]},${dims[1]},${dims[2]}] ` +
      `и min(box)/2=${(Math.min(box[0], box[1], box[2]) / 2).toFixed(4)} — нужно dims>=${minCells} (2*effectiveWalkRadius+1) на всех трёх осях и min(box)/2 > bend.r0=${p.bend.r0}`

  return { N, box, cellSize, effectiveWalkRadius, dims, ncells, minCells, valid, reason }
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

  // Interaction range: at least the largest interaction reach (WCA contact for the largest
  // pairwise size, plus the tail-tail attraction's outer cutoff w_c) so any pair within range of
  // each other is guaranteed to land within `walkRadius` cells of each other -- same reasoning
  // engine/src/sim.ts's own `cellSize` comment gives, generalised from a fixed lipid pair to
  // whichever two of the soup's own species (by data/soup.json's radiusSigma) are largest.
  //
  // perf2-report.md, candidate (a): the ORIGINAL grid set cellSize = interactionRange and walked
  // 3x3x3 (walkRadius=1), which searches a cube of side 3*interactionRange for pairs that only
  // ever lie within a sphere of radius interactionRange -- diagnosed and measured (candidates
  // examined vs pairs within range, forceCandidateStatsDEBUG below) before this was touched.
  // `neighborGrid.cellDivisor` (data/soup.json, with a written basis) divides the cell side by
  // that many; the walk radius needed to keep the SAME completeness guarantee is derived, not
  // assumed, and ASSERTED right below rather than trusted to fall out of the arithmetic.
  // cellDivisor=1 reproduces the original cellSize/walkRadius exactly (bit-identical 3x3x3 walk),
  // the honest A/B control point for this change.
  const maxRadiusSigma = Math.max(...soup.monomers.map((m) => m.radiusSigma))
  const maxB = p.sigma * maxRadiusSigma
  const interactionRange = wcaCutoff(maxB) + p.attraction.wc
  const cellDivisor = soup.neighborGrid.cellDivisor
  const cellSize = interactionRange / cellDivisor
  const walkRadius = Math.ceil(interactionRange / cellSize)
  if (walkRadius * cellSize < interactionRange - 1e-6) {
    throw new Error(
      `сетка соседей: walkRadius=${walkRadius} * cellSize=${cellSize.toFixed(6)} = ${(walkRadius * cellSize).toFixed(6)} ` +
        `не покрывает interactionRange=${interactionRange.toFixed(6)} — гарантия полноты обхода нарушена`,
    )
  }

  // perf2-report.md, candidate (c): a Verlet list, built every `rebuildEvery` real steps on the
  // SAME (unshrunk) grid above, with a wider walk radius that covers interactionRange+skin instead
  // of just interactionRange -- the skin margin is what lets the list stay complete for
  // `rebuildEvery` steps without re-walking. Derived and asserted the same way walkRadius is above,
  // never assumed. When verlet.enabled is false, GB.dims.w (written below) carries the ORIGINAL
  // walkRadius instead, and the *_list_main kernels are never compiled/dispatched at all -- see
  // getSoupPipelines and encodeOneIntegrationStep.
  const verlet = soup.verletList
  const listRange = interactionRange + verlet.skin
  const listBuildWalkRadius = Math.ceil(listRange / cellSize)
  if (listBuildWalkRadius * cellSize < listRange - 1e-6) {
    throw new Error(
      `список Верле: listBuildWalkRadius=${listBuildWalkRadius} * cellSize=${cellSize.toFixed(6)} = ` +
        `${(listBuildWalkRadius * cellSize).toFixed(6)} не покрывает listRange=${listRange.toFixed(6)} (interactionRange+skin) — гарантия полноты обхода нарушена`,
    )
  }
  // Drift-safety condition (perf-report.md's own rejected-candidate-(a) analysis, generalised from
  // one step to `rebuildEvery` steps): a 2x-RMS-3D-speed worst-case outlier bound, evaluated for
  // THIS system's own kT (not a fixed assumed worst case) -- a particle drifting at that bound for
  // the WHOLE rebuild interval must still land within skin/2 of where the list last saw it, or the
  // list could be missing a real neighbour by the next rebuild. soup_max_drift_main backs this
  // analytical bound up with a REAL per-step measurement, checked by step() below.
  if (verlet.enabled) {
    const vBound = 2 * Math.sqrt(3 * opts.kT)
    const driftBound = 2 * verlet.rebuildEvery * p.integrator.dt * vBound
    if (driftBound > verlet.skin) {
      throw new Error(
        `список Верле: 2*rebuildEvery*dt*vBound=${driftBound.toFixed(4)} превышает skin=${verlet.skin} ` +
          `при kT=${opts.kT} — перестройка недостаточно частая (или skin недостаточен) для этой температуры`,
      )
    }
  }
  const effectiveWalkRadius = verlet.enabled ? listBuildWalkRadius : walkRadius

  // Full 3-axis periodicity here (unlike engine/src/sim.ts's membrane, which leaves z open for a
  // bilayer in vacuum) -- a bulk soup has no preferred axis, so the ±effectiveWalkRadius
  // neighbour-cell walk wraps z too (soup/wgsl/step.wgsl, soup/wgsl/bond.wgsl), and needs
  // dims>=2*effectiveWalkRadius+1 on every axis: with fewer than 2*effectiveWalkRadius+1 cells on a
  // periodic axis, the wrap revisits a cell more than once, silently multiplying every force/
  // bond-attempt/list-build contribution from it. min(box)/2 > bend.r0 guards the minimum-image
  // convention (mi3) the same way, generalised to all three axes and unaffected by walkRadius (it
  // is about the bond's own reach, not the neighbour grid). Both checks -- and computeDims below --
  // are planSoupGrid()'s own job now (see its doc comment): the SAME derivation a viewer previews
  // before calling this function at all, not a second copy that could drift from this one.
  const plan = planSoupGrid(box, startCounts)
  if (!plan.valid) throw new Error(plan.reason!)
  const dims = plan.dims
  const ncells = plan.ncells

  const { device } = await getGpu()
  const sortedGather = soup.neighborGrid.sortedGather
  const pipe = getSoupPipelines(device, sortedGather)

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
  // perf2-report.md, candidate (b): cell-sorted GATHER of positions, rebuilt every grid rebuild
  // (i.e. every step) from the SAME cellIdx permutation fill_main already produces -- see
  // soup_gather_sorted_main in soup/wgsl/step.wgsl. Positions/velocities/bondSlots themselves stay
  // in ORIGINAL index space (never physically reordered), so bond bookkeeping is untouched by
  // construction -- see this file's header note on why that choice was made over a full resort.
  const posSortedBuf = storageBuffer(device, new Float32Array(N * 4))
  // perf2-report.md, STEP 1 diagnosis: [candidatesExamined, pairsWithinRange], read back once per
  // forceCandidateStatsDEBUG call, never touched by the real step loop.
  const statsBuf = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })

  // perf2-report.md, candidate (c): Verlet list buffers. verletListBuf is a flat N*listCapacity
  // array (soup_build_verlet_list_main/soup_force_list_main/bond_form_list_main all index it as
  // i*listCapacity+slot); verletCountBuf is the per-particle count actually found this rebuild;
  // verletOverflowBuf is a single flag the build kernel sets (never clears) if any particle found
  // more than listCapacity candidates -- soup/src/sim.ts asserts it stays clear after every
  // rebuild rather than silently trusting listCapacity was big enough. posAtRebuildBuf/
  // maxDriftSqBuf implement the drift-safety guard: a snapshot taken at every rebuild and the
  // running max squared-drift since it, read back and checked after every step() call.
  const verletListBuf = device.createBuffer({
    size: N * verlet.listCapacity * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const verletCountBuf = device.createBuffer({ size: Math.max(4, N * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const verletOverflowBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(verletOverflowBuf, 0, new Uint32Array([0]))
  const posAtRebuildBuf = storageBuffer(device, new Float32Array(N * 4))
  const maxDriftSqBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(maxDriftSqBuf, 0, new Uint32Array([0]))
  // VL: x=listRange (interactionRange+skin), y=listCapacity (as f32, cast to u32 in WGSL) -- see
  // step.wgsl/bond.wgsl's own VL declarations.
  const verletUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(verletUniform, 0, new Float32Array([listRange, verlet.listCapacity, 0, 0]))

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
    // dims.w carries walkRadius (perf2-report.md, candidate (a)) -- unused by engine/wgsl/
    // neighbor.wgsl (its cell_of()/clear_counts_main only ever read dims.xyz), read by
    // soup/wgsl/step.wgsl's soup_force_main and soup/wgsl/bond.wgsl's bond_form_main to drive the
    // ±walkRadius cell walk instead of a hardcoded ±1.
    new Uint32Array(bytes, 0, 4).set([dims[0], dims[1], dims[2], effectiveWalkRadius])
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
  //
  // Perf fix (b), perf-report.md: bond_form_main/bond_break_main are only DISPATCHED every
  // bondAttemptInterval.steps real steps (see encodeOneIntegrationStep below), so the dt this
  // uniform is built from is multiplied by that interval -- attemptProbability's own formula
  // (attemptRate*dt) is linear in dt, so attemptRate*(dt*k) is exactly k independent per-step
  // attempts' worth of probability folded into one (Poisson thinning), keeping the average attempt
  // rate per REAL step unchanged. acceptanceProbability below is untouched -- it depends only on
  // energyKT/kT, never on dt, so this has no effect on the forward/backward ratio detailed balance
  // is carried by.
  const dt = p.integrator.dt
  const bondAttemptInterval = soup.bondAttemptInterval.steps
  const bondDt = dt * bondAttemptInterval
  const eventRuleIds: [string, string][] = rules.map((r) => [r.bond.id, r.brk.id])
  const bondParamsUniform = device.createBuffer({ size: 160, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const kindA = packVec4(rules.map((r) => r.kindA))
    const kindB = packVec4(rules.map((r) => r.kindB))
    const attemptProbForm = packVec4(rules.map((r) => attemptProbability(r.bond, bondDt)))
    const attemptProbBreak = packVec4(rules.map((r) => attemptProbability(r.brk, bondDt)))
    const acceptProbForm = packVec4(rules.map((r) => acceptanceProbability(r.bond, opts.kT)))
    const acceptProbBreak = packVec4(rules.map((r) => acceptanceProbability(r.brk, opts.kT)))
    const requiresCatalyst = packVec4(rules.map((r) => (r.bond.requiresCatalyst ? 1 : 0)))
    const slotRoleA = packVec4(rules.map((r) => r.slotRoleA))
    const slotRoleB = packVec4(rules.map((r) => r.slotRoleB))
    // data/soup.json's headPlacement.terminalOnly (rank D, basis in that file): whether
    // soup/wgsl/bond.wgsl's tryClaimSlot gates head/chain-slot claims on chain-end position, packed
    // as a 1/0 float the same way requiresCatalyst already is -- the flag is data, not a constant
    // written into soup/src or soup/wgsl.
    const headTerminalOnly = soup.headPlacement.terminalOnly ? 1 : 0
    // data/soup.json's headPlacement.chainCapacity (rank D, basis in that file): how many chain
    // slots a head's own claim (soup/wgsl/bond.wgsl's tryClaimSlot, role==2) may try -- uploaded as
    // a plain float the same way headTerminalOnly already is, replacing that struct's former
    // bpPad1 padding slot (see BondParams's own comment in bond.wgsl). assertRulesConsistent
    // (soup/src/rules.ts) has already checked this is an integer within the architectural 1..3
    // range by the time loadSoup() returns it here.
    const headChainCapacity = soup.headPlacement.chainCapacity
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
        headTerminalOnly,
        headChainCapacity,
        0,
      ]),
    )
  }

  const bind = (pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ label: `${pipeline.label}@${group}`, layout: pipeline.getBindGroupLayout(group), entries })
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

  // perf2-report.md, candidate (b): gather bind group, only group 1 (no Params uniform needed for
  // a plain copy) -- pos2 (read) + cellIdx (read, the permutation) + posSortedRW (write target).
  const gatherSortedBind = bind(pipe.gatherSorted, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 5, resource: buf(cellsBuf) },
    { binding: 13, resource: buf(posSortedBuf) },
  ])

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
    { binding: 13, resource: buf(posSortedBuf) },
  ])
  // perf2-report.md correctness gate: O(N^2) reference, no grid buffers needed at all. Its own
  // group0 -- NOT soupForceGroup0 -- because 'layout: auto' gives every pipeline a DISTINCT layout
  // object even when the referenced uniform is identical; reusing another pipeline's bind group
  // fails WebGPU validation (caught via a page-console listener, not silently -- see perf2-report.md).
  const soupForceBruteGroup0 = bind(pipe.soupForceBrute, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const soupForceBruteGroup1 = bind(pipe.soupForceBrute, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 1, resource: buf(forceBuf) },
    { binding: 3, resource: buf(gridUniform) },
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
    { binding: 13, resource: buf(posSortedBuf) },
  ])
  const bondFormGroup2 = bind(pipe.bondForm, 2, [{ binding: 0, resource: buf(bondParamsUniform) }])
  const bondBreakGroup1 = bind(pipe.bondBreak, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 6, resource: buf(bondSlotsBuf) },
    { binding: 8, resource: buf(eventsBuf) },
    { binding: 9, resource: buf(bondRngBuf) },
  ])
  const bondBreakGroup2 = bind(pipe.bondBreak, 2, [{ binding: 0, resource: buf(bondParamsUniform) }])

  // perf2-report.md, STEP 1 diagnosis: bind groups for soup_force_stats_main -- reuses exactly the
  // same buffers soup_force_main's group0/1 do (it needs P.sigma/P.b_tt/P.wc via Params and the
  // same grid), plus its own group 3 output.
  const forceStatsGroup0 = bind(pipe.forceStats, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const forceStatsGroup1 = bind(pipe.forceStats, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 4, resource: buf(cellStartBuf) },
    { binding: 5, resource: buf(cellsBuf) },
    { binding: 8, resource: buf(speciesUniform) },
  ])
  const forceStatsGroup3 = bind(pipe.forceStats, 3, [{ binding: 0, resource: buf(statsBuf) }])

  // perf2-report.md, candidate (c): Verlet list bind groups. buildVerletList/snapshotPositions/
  // resetMaxDrift/maxDrift never reference P (no group 0 needed) -- only the geometry/positions
  // and their own list buffers.
  const buildVerletListBind = bind(pipe.buildVerletList, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 4, resource: buf(cellStartBuf) },
    { binding: 5, resource: buf(cellsBuf) },
    { binding: 14, resource: buf(verletListBuf) },
    { binding: 15, resource: buf(verletCountBuf) },
    { binding: 16, resource: buf(verletOverflowBuf) },
    { binding: 19, resource: buf(verletUniform) },
  ])
  const snapshotPositionsBind = bind(pipe.snapshotPositions, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 17, resource: buf(posAtRebuildBuf) },
  ])
  const resetMaxDriftBind = bind(pipe.resetMaxDrift, 1, [{ binding: 18, resource: buf(maxDriftSqBuf) }])
  const maxDriftBind = bind(pipe.maxDrift, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 17, resource: buf(posAtRebuildBuf) },
    { binding: 18, resource: buf(maxDriftSqBuf) },
  ])
  const soupForceListGroup0 = bind(pipe.soupForceList, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const soupForceListGroup1 = bind(pipe.soupForceList, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 1, resource: buf(forceBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 7, resource: buf(bondSlotsBuf) },
    { binding: 8, resource: buf(speciesUniform) },
    { binding: 14, resource: buf(verletListBuf) },
    { binding: 15, resource: buf(verletCountBuf) },
    { binding: 19, resource: buf(verletUniform) },
  ])
  const bondFormListGroup0 = bind(pipe.bondFormList, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const bondFormListGroup1 = bind(pipe.bondFormList, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 3, resource: buf(gridUniform) },
    { binding: 6, resource: buf(bondSlotsBuf) },
    { binding: 7, resource: buf(speciesUniform) },
    { binding: 8, resource: buf(eventsBuf) },
    { binding: 9, resource: buf(bondRngBuf) },
    { binding: 14, resource: buf(verletListBuf) },
    { binding: 15, resource: buf(verletCountBuf) },
    { binding: 19, resource: buf(verletUniform) },
  ])
  const bondFormListGroup2 = bind(pipe.bondFormList, 2, [{ binding: 0, resource: buf(bondParamsUniform) }])

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
    // perf2-report.md, candidate (b): gather positions into cell-sorted order right after `fill`
    // finalises `cellsBuf` (the permutation) for this step -- must run before soup_force_main/
    // bond_form_main (both read posSortedRW/posSortedRO this same step) and after fill (its own
    // permutation is this gather's input). Skipped entirely when sortedGather=false -- the
    // *_unsorted pipeline variants never read posSortedRW/posSortedRO, so this dispatch would be
    // pure waste (and its own cost must not be charged against the (a)-only measurement).
    if (sortedGather) {
      pass.setPipeline(pipe.gatherSorted)
      pass.setBindGroup(1, gatherSortedBind)
      pass.dispatchWorkgroups(wgN)
    }
  }

  function encodeSoupForce(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.soupForce)
    pass.setBindGroup(0, soupForceGroup0)
    pass.setBindGroup(1, soupForceGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  function encodeSoupForceBrute(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.soupForceBrute)
    pass.setBindGroup(0, soupForceBruteGroup0)
    pass.setBindGroup(1, soupForceBruteGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  // perf2-report.md, candidate (c): rebuild the coarse grid (unchanged, cheap -- cellDivisor is
  // independent of this candidate) THEN the Verlet list from it, snapshot the positions this
  // rebuild used (the drift-safety reference point), and zero the running max-drift counter --
  // ORDER matters: snapshot/reset must follow the list build that just consumed the CURRENT
  // positions, not precede it.
  function encodeVerletRebuild(pass: GPUComputePassEncoder) {
    encodeGridRebuild(pass)
    pass.setPipeline(pipe.buildVerletList)
    pass.setBindGroup(1, buildVerletListBind)
    pass.dispatchWorkgroups(wgN)
    pass.setPipeline(pipe.snapshotPositions)
    pass.setBindGroup(1, snapshotPositionsBind)
    pass.dispatchWorkgroups(wgN)
    pass.setPipeline(pipe.resetMaxDrift)
    pass.setBindGroup(1, resetMaxDriftBind)
    pass.dispatchWorkgroups(1)
  }

  function encodeSoupForceList(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.soupForceList)
    pass.setBindGroup(0, soupForceListGroup0)
    pass.setBindGroup(1, soupForceListGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  // O(N), same order of cost as kick_drift_wrap_main -- see soup_max_drift_main's own header for
  // why this per-step cost is acceptable (it is what lets the drift-safety guard be checked for
  // real, not just trusted from the analytical bound above).
  function encodeMaxDrift(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.maxDrift)
    pass.setBindGroup(1, maxDriftBind)
    pass.dispatchWorkgroups(wgN)
  }

  function encodeBondFormList(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.bondFormList)
    pass.setBindGroup(0, bondFormListGroup0)
    pass.setBindGroup(1, bondFormListGroup1)
    pass.setBindGroup(2, bondFormListGroup2)
    pass.dispatchWorkgroups(wgN)
  }

  // Initial grid + force, needed as F(x0) for the first kick. perf2-report.md, candidate (c): when
  // verlet.enabled, this ALSO builds the first Verlet list and takes the first drift-safety
  // snapshot -- step()'s own per-step rebuild schedule (globalStep % rebuildEvery === 0) will
  // rebuild again at globalStep=0 using the post-first-kick positions, exactly mirroring how the
  // non-verlet path already always rebuilds fresh every step; this priming block's own job is only
  // to produce a valid F(x0) for that very first kick.
  {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    if (verlet.enabled) {
      encodeVerletRebuild(pass)
      encodeSoupForceList(pass)
    } else {
      encodeGridRebuild(pass)
      encodeSoupForce(pass)
    }
    pass.end()
    device.queue.submit([enc.finish()])
  }

  // doBonds: perf fix (b), perf-report.md. bond_form_main/bond_break_main measured ~42% of a full
  // step's cost (0.69 of 1.64ms at N=13100, 30^3 box) while the events they exist to catch are
  // rare (attemptProbability per real step is 0.0005 at this rule set's attemptRate/dt) -- most of
  // that cost is the 3x3x3 neighbour walk run for nothing, every step, whether or not this step is
  // one of the rare ones that actually attempts anything. Dispatching the pair only every
  // bondAttemptInterval.steps real steps (data/soup.json, with attemptProbForm/attemptProbBreak
  // already built from dt*bondAttemptInterval above) cuts that cost by ~bondAttemptInterval.steps
  // while leaving the average attempt rate per real step, and therefore the equilibrium bond count
  // detailed balance sets, unchanged (see bondAttemptInterval's basis and
  // tests/soup-bonds.test.ts's before/after equilibrium check).
  function encodeOneIntegrationStep(pass: GPUComputePassEncoder, doBonds: boolean, doListRebuild: boolean) {
    // First Verlet half-kick + drift + 3-axis periodic wrap, fused into one dispatch (see
    // soup/wgsl/step.wgsl's kick_drift_wrap_main header) -- exactly kick_main+drift_main+a
    // 3-axis wrap from engine/wgsl/integrate.wgsl's own formulas, not a new integrator.
    pass.setPipeline(pipe.kickDriftWrap)
    pass.setBindGroup(0, kickDriftWrapGroup0)
    pass.setBindGroup(1, kickDriftWrapGroup1)
    pass.dispatchWorkgroups(wgN)

    // perf2-report.md, candidate (c): when enabled, the cell walk that dominates both force and
    // bond-attempt cost runs only on scheduled rebuild steps (doListRebuild); every other step
    // reads the list built at the last rebuild instead of re-walking cells at all.
    if (verlet.enabled) {
      if (doListRebuild) encodeVerletRebuild(pass)
      encodeSoupForceList(pass)
      encodeMaxDrift(pass)
    } else {
      encodeGridRebuild(pass)
      encodeSoupForce(pass)
    }

    if (doBonds) {
      // Bond Monte Carlo: formation then breaking, on the freshly rebuilt grid/positions (or the
      // current Verlet list). Two separate, ordered dispatches within the same pass -- never
      // concurrent with each other, see bond.wgsl's header for why that ordering is what makes the
      // i<j dedupe race-free.
      if (verlet.enabled) {
        encodeBondFormList(pass)
      } else {
        pass.setPipeline(pipe.bondForm)
        pass.setBindGroup(0, bondFormGroup0)
        pass.setBindGroup(1, bondFormGroup1)
        pass.setBindGroup(2, bondFormGroup2)
        pass.dispatchWorkgroups(wgN)
      }

      pass.setPipeline(pipe.bondBreak)
      pass.setBindGroup(1, bondBreakGroup1)
      pass.setBindGroup(2, bondBreakGroup2)
      pass.dispatchWorkgroups(wgN)
    }

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

  // Runs continuously across separate step(n) calls (not reset per call) so the
  // bondAttemptInterval schedule stays regular regardless of how a caller chunks its own n's --
  // e.g. step(30) then step(70) attempts bonds on the same global step indices step(100) would.
  let globalStep = 0

  // perf2-report.md, candidate (c): checked once per chunk (not once per step -- the whole point
  // of STEP_CHUNK is to keep the GPU timeline free of per-step CPU round trips, and a live
  // per-step drift-triggered rebuild would reintroduce exactly that), at the natural
  // onSubmittedWorkDone() sync point step() already has. Throws immediately and does not continue
  // if either the drift bound or the list capacity was ever violated -- this candidate's whole
  // completeness guarantee rests on catching that for real, not trusting the analytical bound
  // computed at creation time.
  async function assertVerletSafety(): Promise<void> {
    const rawOverflow = await readBack(device, verletOverflowBuf, 4)
    const overflow = new Uint32Array(rawOverflow.buffer, rawOverflow.byteOffset, 1)[0]
    if (overflow !== 0) {
      throw new Error(
        `список Верле: verletList.listCapacity=${verlet.listCapacity} было недостаточно -- ` +
          `хотя бы одна частица нашла больше кандидатов, чем вмещает список (данные могли быть тихо отброшены)`,
      )
    }
    const rawDrift = await readBack(device, maxDriftSqBuf, 4)
    const drift = Math.sqrt(Math.max(0, rawDrift[0]))
    const bound = verlet.skin / 2
    if (drift > bound + 1e-6) {
      throw new Error(
        `список Верле: измеренный дрейф ${drift.toFixed(4)} превышает skin/2=${bound.toFixed(4)} -- ` +
          `аналитическая граница (verletList.basis) не сработала для реальной траектории, перестройка была недостаточно частой`,
      )
    }
  }

  async function step(n: number): Promise<void> {
    let done = 0
    while (done < n) {
      const chunk = Math.min(STEP_CHUNK, n - done)
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      for (let k = 0; k < chunk; k++) {
        encodeOneIntegrationStep(
          pass,
          globalStep % bondAttemptInterval === 0,
          verlet.enabled && globalStep % verlet.rebuildEvery === 0,
        )
        globalStep++
      }
      pass.end()
      device.queue.submit([enc.finish()])
      await device.queue.onSubmittedWorkDone()
      if (verlet.enabled) await assertVerletSafety()
      done += chunk
    }
  }

  // Profiling helper for the perf task (report:
  // .superpowers/sdd/2026-08-16-soup-to-vesicle/perf-report.md). No timestamp-query use even
  // though the adapter supports the feature: WebGPU only exposes timestampWrites at COMPUTE-PASS
  // granularity, and this engine deliberately fuses all 9 dispatches of one integration step into
  // ONE pass (see this file's header and step.wgsl's -- interleaving many small
  // dispatches/passes measurably dominated wall time before that fuse). Splitting per-stage
  // passes to get per-stage GPU timestamps would reintroduce exactly the overhead the fuse
  // removed, so it would not be measuring the thing production actually runs. Uses the
  // difference-of-variants method instead, exactly as the task brief's fallback describes.
  //
  // A first, buggy version of this ran EACH stage alone for n steps in sequence on one shared
  // system, in an order (kickDriftWrap first) that let one unsafe isolation corrupt every
  // measurement after it: kickDriftWrap alone applies +0.5*dt*F from a force that is NEVER
  // recomputed (soupForce is a separate phase), so velocity grows by a fixed increment every
  // single iteration with nothing to damp it -- after 3000 iterations that is a large,
  // accumulating, UNBOUNDED drift, and every phase measured afterward (including the final
  // "fullStepEquivalent") inherited that already-deranged state. Measured symptom: the isolated
  // stages summed to ~0.9ms/step but "fullStepEquivalent" (the very same 9 dispatches, just
  // bundled) came out at 21.4ms/step -- 23x its own parts, and 10x the documented ~2.1ms/step
  // (480 steps/s) baseline for this exact configuration. That gap is the corruption, not a real
  // cost; it never got used for anything.
  //
  // Fix: only ever isolate GROUPS that stay physically bounded no matter how large n is.
  //  - 'full': the real, unmodified per-step dispatch sequence, timed FIRST (before anything else
  //    touches the buffers) -- this is just step()'s own physics, proven stable over 50_000 steps,
  //    and is the number this whole profile is checked against.
  //  - 'gridBuild', 'force': read positions, WRITE only their own output buffer (cellStart/cells,
  //    forceBuf) -- positions never move during these, so running either alone for any n computes
  //    the identical answer every iteration. No corruption possible.
  //  - 'bondAttempts' (form+break together): mutates only bondSlots/events, never positions or
  //    velocities, and valence is capped at 3 -- self-limiting, not unbounded.
  //  - 'integration' (kickDriftWrap+kickThermostat together, exactly as they alternate in a real
  //    step): unlike kickDriftWrap ALONE, this pairs every kick with the thermostat's own
  //    Ornstein-Uhlenbeck damping (v -= gamma*dt*v, before adding noise) in the SAME iteration, so
  //    velocity relaxes to a stationary distribution instead of growing linearly -- bounded for
  //    any n, even though the force it integrates against is a stale snapshot (not recomputed
  //    inside this isolated group, same as every other bucket here).
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
    const encodeIntegration = (pass: GPUComputePassEncoder) => {
      pass.setPipeline(pipe.kickDriftWrap)
      pass.setBindGroup(0, kickDriftWrapGroup0)
      pass.setBindGroup(1, kickDriftWrapGroup1)
      pass.dispatchWorkgroups(wgN)
      pass.setPipeline(pipe.kickThermostat)
      pass.setBindGroup(0, kickThermostatGroup0)
      pass.setBindGroup(1, kickThermostatGroup1)
      pass.dispatchWorkgroups(wgN)
    }
    const encodeBondAttempts = (pass: GPUComputePassEncoder) => {
      pass.setPipeline(pipe.bondForm)
      pass.setBindGroup(0, bondFormGroup0)
      pass.setBindGroup(1, bondFormGroup1)
      pass.setBindGroup(2, bondFormGroup2)
      pass.dispatchWorkgroups(wgN)
      pass.setPipeline(pipe.bondBreak)
      pass.setBindGroup(1, bondBreakGroup1)
      pass.setBindGroup(2, bondBreakGroup2)
      pass.dispatchWorkgroups(wgN)
    }
    const phases: [string, (pass: GPUComputePassEncoder) => void][] = [
      // Worst case (every step does bonds too) -- the number to compare against gridBuild/
      // force/bondAttempts/integration's sum; the REAL amortized cost (bonds only every
      // bondAttemptInterval.steps steps) is what step()'s own timing reports.
      ['full', (pass) => encodeOneIntegrationStep(pass, true, true)],
      ['gridBuild', (pass) => encodeGridRebuild(pass)],
      ['force', (pass) => encodeSoupForce(pass)],
      ['bondAttempts', encodeBondAttempts],
      ['integration', encodeIntegration],
    ]
    for (const [label, fn] of phases) {
      const [l, ms] = await timePhase(label, fn)
      out[l] = ms
    }
    return out
  }

  // perf2-report.md, STEP 1 diagnosis (and re-checked after each grid change): rebuilds the grid
  // for the CURRENT position state (so counts describe the config this call sees, not a stale
  // one), zeroes the stats buffer, then runs soup_force_stats_main -- the exact same ±walkRadius
  // cell walk soup_force_main runs -- once, and reads its two counters back. Never part of the
  // real step loop; a diagnostic only, same role stepPhasesDEBUG plays for timing.
  async function forceCandidateStatsDEBUG(): Promise<{ candidatesExamined: number; pairsWithinRange: number; ratio: number }> {
    device.queue.writeBuffer(statsBuf, 0, new Uint32Array([0, 0]))
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeGridRebuild(pass)
    pass.setPipeline(pipe.forceStats)
    pass.setBindGroup(0, forceStatsGroup0)
    pass.setBindGroup(1, forceStatsGroup1)
    pass.setBindGroup(3, forceStatsGroup3)
    pass.dispatchWorkgroups(wgN)
    pass.end()
    device.queue.submit([enc.finish()])
    const raw = await readBack(device, statsBuf, 8)
    const u32 = new Uint32Array(raw.buffer, raw.byteOffset, 2)
    const candidatesExamined = u32[0]
    const pairsWithinRange = u32[1]
    return { candidatesExamined, pairsWithinRange, ratio: candidatesExamined / Math.max(1, pairsWithinRange) }
  }

  async function particles(): Promise<Float32Array> {
    return readBack(device, posBuf, N * 16)
  }

  // perf2-report.md correctness gate: mirrors engine/src/sim.ts's forces()/forcesBruteForce() pair
  // (checked by tests/sim.test.ts's "сетка соседей даёт те же силы, что и полный перебор") for the
  // soup's own dynamic-topology force kernel. forces() rebuilds fresh for the CURRENT positions
  // first -- when verlet.enabled, a FULL Verlet rebuild (never relying on a possibly-stale list
  // from whatever step count the caller happens to be at) then the list-based force kernel;
  // otherwise the same encodeGridRebuild/encodeSoupForce cell walk every real step already uses
  // (candidate (a)'s walk radius and candidate (b)'s gather both feed through it).
  // forcesBruteForce() needs neither -- it is the O(N^2) reference soup_force_brute_main runs
  // directly off pos2/bondSlots.
  async function forces(): Promise<Float32Array> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    if (verlet.enabled) {
      encodeVerletRebuild(pass)
      encodeSoupForceList(pass)
    } else {
      encodeGridRebuild(pass)
      encodeSoupForce(pass)
    }
    pass.end()
    device.queue.submit([enc.finish()])
    return readBack(device, forceBuf, N * 16)
  }

  async function forcesBruteForce(): Promise<Float32Array> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeSoupForceBrute(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    return readBack(device, forceBuf, N * 16)
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

  // Task 4: the continuous soup->vesicle run. Declared with `let sys!` and assigned AFTER the
  // object literal below so runUntil's own closure can call detectStage(sys) -- detectStage only
  // ever reads sys.particles()/sys.bonds()/sys.box (see stages.ts), never sys.runUntil itself, so
  // the circularity is only in the TYPE, not in anything actually read before `sys` is assigned.
  async function runUntil(
    stage: Stage,
    opts: { maxSteps: number; sampleEvery: number },
  ): Promise<{ reached: boolean; steps: number; trace: { steps: number; stage: Stage; evidence: StageEvidence }[] }> {
    const trace: { steps: number; stage: Stage; evidence: StageEvidence }[] = []
    let steps = 0
    let reached = false
    while (steps < opts.maxSteps) {
      const chunk = Math.min(opts.sampleEvery, opts.maxSteps - steps)
      await sys.step(chunk)
      steps += chunk
      const { stage: currentStage, evidence } = await detectStage(sys)
      trace.push({ steps, stage: currentStage, evidence })
      // Printed AS IT HAPPENS (not buffered to the end) -- the whole point per this task's brief:
      // a run long enough to matter (the pilot is minutes, the full-scale run tens of minutes) must
      // be diagnosable while it is still running, not only from the return value after the fact.
      console.log(
        `[runUntil] steps=${steps} stage=${currentStage} ` +
          `amphiphileFraction=${evidence.amphiphileFraction.toFixed(4)} ` +
          `largestAggregateFraction=${evidence.largestAggregateFraction.toFixed(4)} ` +
          `headPeaks=${evidence.headPeaks} enclosedVolume=${evidence.enclosedVolume.toFixed(4)}`,
      )
      if (currentStage === stage) {
        reached = true
        break
      }
    }
    return { reached, steps, trace }
  }

  // Every GPUBuffer createSoup allocates above, for dispose() to destroy -- listed exhaustively
  // rather than tracked via a running array at allocation time, so this list is a single place to
  // audit against createSoup's own allocations whenever a new buffer is added up there.
  function dispose(): void {
    posBuf.destroy()
    velBuf.destroy()
    forceBuf.destroy()
    cellsBuf.destroy()
    countsBuf.destroy()
    cellStartBuf.destroy()
    cursorBuf.destroy()
    posSortedBuf.destroy()
    statsBuf.destroy()
    verletListBuf.destroy()
    verletCountBuf.destroy()
    verletOverflowBuf.destroy()
    posAtRebuildBuf.destroy()
    maxDriftSqBuf.destroy()
    verletUniform.destroy()
    bondSlotsBuf.destroy()
    bondRngBuf.destroy()
    thermoRngBuf.destroy()
    eventsBuf.destroy()
    paramsUniform.destroy()
    gridUniform.destroy()
    speciesUniform.destroy()
    bondParamsUniform.destroy()
  }

  let sys!: SoupSystem
  sys = {
    step,
    particles,
    forces,
    forcesBruteForce,
    bonds,
    events,
    invariants,
    box,
    runUntil,
    dispose,
    stepPhasesDEBUG: stepPhasesDEBUG as any,
    forceCandidateStatsDEBUG: forceCandidateStatsDEBUG as any,
  }
  return sys
}
