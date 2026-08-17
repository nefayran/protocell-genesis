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
  /** Convenience override for data/soup.json's dryWetCycle.enabled, mirroring `catalystCount`'s own
   * pattern: `true` forces cycling on for THIS system regardless of the file's own default (the file
   * defaults to `false` precisely so no existing caller sees a behaviour change without asking for
   * it); `false`/`undefined` defers to the file. The schedule itself (cycles/periodSteps/dryFraction/
   * targetDryDensity/rampSteps/rampRelaxSteps) always comes from data/soup.json -- this override is a
   * plain boolean switch, never a number. */
  dryWetCycle?: boolean
}

export interface SoupSystem {
  /** Advances n Langevin + bond-Monte-Carlo steps, encoded as one command buffer. */
  step(n: number): Promise<void>
  /** Advances n real steps exactly like step(), EXCEPT that when this system was created with dry-
   * wet cycling enabled (CreateSoupOpts.dryWetCycle / data/soup.json's dryWetCycle.enabled), a wet<->
   * dry box transition landing inside this call is applied at the right global step (via
   * applyBoxScale, not step() itself -- step() is untouched by this task, so every existing caller's
   * behaviour is unchanged byte for byte) before continuing. When cycling is NOT enabled this is
   * `step(n)` exactly, nothing else -- soup/src/sim.ts's runUntil() below always calls this one
   * (never step() directly) so cycling "just works" through the same trace-producing loop every
   * other run already uses, without a second stepping path to keep in sync. Because a transition
   * ramp injects its own `rampRelaxSteps` between increments (data/soup.json's dryWetCycle.basis),
   * the number of real steps actually taken by one call CAN exceed `n` when a transition falls
   * inside it -- callers that need an exact step count should check `steps` before/after, not assume
   * n was taken verbatim. */
  stepCycled(n: number): Promise<void>
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
  /** Surface growth (surface-growth-report.md): the mutual catalyst<->tip association buffer,
   * `soup/wgsl/bond.wgsl`'s `centerLink` -- one entry per particle, `0xFFFFFFFF` (unassociated) or
   * the id of the particle it is currently linked to (a catalyst's currently-held chain-tip carbon,
   * or a carbon's currently-owning catalyst). Debug/verification readback only, mirroring
   * `bonds()`'s own role for `bondSlots` -- never read by the real step loop. */
  centerLinks(): Promise<Uint32Array>
  /** Surface growth / adsorption (adsorption-report.md): cumulative desorption event counts since
   * creation -- `stretch` is soup/wgsl/bond.wgsl's desorbStretch (the adsorption bond exceeded
   * FENE's own bonded range, P.r_inf), `timeout` is desorbTimeout (a centre held the same chain for
   * more than data/soup.json's adsorption.maxHoldSteps real steps without a propagation/termination
   * event). Neither overlaps `events()`'s own per-rule counts: a desorption is never a completed
   * amphiphile. */
  desorbEvents(): Promise<{ stretch: number; timeout: number }>
  /** Cumulative event counts since creation, keyed by data/soup.json rule id (e.g. "cc_bond"). */
  events(): Promise<Record<string, number>>
  /** Per-monomer-id particle counts, active bond count, and total charge (sum of each monomer's
   * `charge` field if data/soup.json ever defines one; the current schema does not, so this is 0
   * by construction -- not a hardcoded placeholder, a schema-driven sum that happens to be empty
   * today). Compared before/after step() by tests/soup-bonds.test.ts: particle counts and charge
   * must be EXACTLY unchanged (nothing here ever creates or destroys a particle), bonds must be
   * able to change (that is the whole point of this task). */
  invariants(): Promise<{ monomers: Record<string, number>; bonds: number; charge: number }>
  /** The system's CURRENT box -- a live snapshot, mirroring engine/src/sim.ts's own `System.box`
   * getter (its doc comment: "a live snapshot, since areaMove() mutates L_x, L_y in place"). Equal
   * to `CreateSoupOpts.box` for the system's whole lifetime UNLESS dry-wet cycling is enabled, in
   * which case applyBoxScale (driven by stepCycled()) mutates it between the system's own wet box
   * and its derived dry box (computeDryBox). Task 4's reconciliation of the deviation Task 3
   * flagged: soup/src/stages.ts's detectStage reads this instead of taking box as a second
   * parameter, so `stageOf(sys)` (and this file's own runUntil, below) never has to re-thread it --
   * and, since detectStage reads it live, a sample taken mid-cycle correctly measures against
   * whichever box that sample's snapshot actually sits in. */
  readonly box: [number, number, number]
  /** 'wet' | 'dry' | 'none' -- 'none' when dry-wet cycling is not enabled for this system (the
   * default), otherwise which segment of its current cycle the box is presently at (or was last set
   * to; a ramp's own intermediate increments do not change this label, only which BOUNDARY they are
   * walking toward does -- see data/soup.json's dryWetCycle.basis on why the transition itself is
   * spread over several small steps). Read live, like `box` above. */
  readonly cyclePhase: 'wet' | 'dry' | 'none'
  /** 1-based index of the cycle currently in progress, or 0 when cycling is not enabled OR every
   * configured cycle has already completed (settled back to wet -- see cyclePhaseAt's own doc
   * comment for why 0 specifically means "over", not "cycle zero"). */
  readonly cycleIndex: number
  /** Cumulative real integration steps taken so far via step()/stepCycled(), across all calls --
   * mirrors engine/src/sim.ts's own `System.steps`. Needed by a stepCycled() caller that wants to
   * know exactly how many real steps a call actually took, since a call spanning a wet<->dry
   * transition takes MORE than the `n` it was asked for (the ramp's own relax steps) -- see
   * stepCycled's own doc comment. */
  readonly steps: number
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
  ): Promise<{
    reached: boolean
    steps: number
    trace: { steps: number; stage: Stage; evidence: StageEvidence; cyclePhase: 'wet' | 'dry' | 'none'; cycleIndex: number }[]
  }>
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

// --- dry-wet cycling (task 'wet-dry-cycle'): pure, GPU-free coordinate map + schedule -------------
//
// Deamer's dry-wet cycling forces closure by changing the BOX, not the membrane's own physics --
// see data/soup.json's dryWetCycle.basis for the experimental motivation and this task's amplitude/
// period reasoning. The coordinate map below reuses the discipline engine/src/sim.ts's
// scaleLateralRigid established for the membrane's fixed 3-bead lipids (its own doc comment: "cost
// four review rounds"): scale each MOLECULE's center of mass, rebuild every bead around that scaled
// center from its UNCHANGED internal offset, never rescale a bead directly. Generalised here from a
// fixed head/tail1/tail2 triple to the soup's dynamic topology (bond graph connected components,
// including singleton unbonded monomers as size-1 "molecules") and from a lateral (x,y only) scale
// to a full 3-axis isotropic one -- a bulk soup has no vacuum-facing axis the way a membrane patch
// does (soup/src/sim.ts's own header, above), so "drying" contracts every periodic axis together.

// Minimum-image displacement of a scalar coordinate difference -- same role as engine/src/sim.ts's
// private mi1(), duplicated rather than imported (that one is a module-private helper, not exported,
// and this file's own step.wgsl/bond.wgsl mi3() is a GPU-side twin of the same formula).
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

// Wraps one coordinate into [0, box) -- the scalar form of engine/src/sim.ts's wrapXY(), generalised
// to all three axes by being called per-axis below (this soup has no open axis to skip).
function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

/** Pure per-molecule rigid coordinate map for a box change: groups particles into connected
 * components of the COVALENT bond graph (`bonds`, matching SoupSystem.bonds()'s own pair layout --
 * an unbonded monomer is its own size-1 component), scales each component's center of mass by
 * `newBox/oldBox` per axis, and rebuilds every member bead around that scaled center from its
 * UNCHANGED internal offset. No bead is ever rescaled directly, so every intramolecular (bonded)
 * distance survives the move exactly -- tests/soup-boxcycle.test.ts checks this property on
 * synthetic topologies with no GPU. The adsorption tether (soup/wgsl/bond.wgsl's centerLink) is
 * deliberately NOT one of these edges: it is a soft, desorbable association, not a covalent bond,
 * and its own length is ALLOWED to change by this move (data/soup.json's adsorption.basis already
 * gives it a desorption valve for exactly this kind of stretch) -- see soup/src/sim.ts's
 * applyBoxScaleOnce for how that is exercised for real.
 *
 * Offsets are accumulated by WALKING THE BOND GRAPH edge by edge (BFS from an arbitrary root in
 * each component), not by taking one minimum-image reading against a single fixed reference bead --
 * a real bug, caught by applyBoxScaleOnce's own runtime self-check on a live 250000-step run
 * (wet-dry-cycle-report.md): this project's cc_bond has no length cap, and at the elevated dry-phase
 * density this task's own cycle deliberately targets, a real carbon chain grew long/coiled enough
 * that ITS OWN two ends sat more than half the (already-shrunk) box apart -- exactly the case a
 * single-reference mi1() reading aliases (engine/src/aggregate.ts's unwrapAggregate has the SAME
 * known limitation, for the same reason, on a spatial-proximity cluster that has no explicit
 * topology to walk instead). Every individual COVALENT BOND, by construction, forms only at
 * reaction-contact distance (a couple sigma at most, soup/wgsl/bond.wgsl) -- far under half of ANY box this
 * system ever runs at -- so accumulating one mi1() step per EDGE, never against a bead that might be
 * many bonds and sigma away, cannot alias regardless of how long or coiled the chain is. */
export function scaleMoleculesRigid(
  positions: Float32Array,
  bonds: Uint32Array,
  oldBox: [number, number, number],
  newBox: [number, number, number],
): Float32Array {
  const n = positions.length / 4
  const sx = newBox[0] / oldBox[0]
  const sy = newBox[1] / oldBox[1]
  const sz = newBox[2] / oldBox[2]

  const adjacency: number[][] = Array.from({ length: n }, () => [])
  for (let k = 0; k < bonds.length; k += 2) {
    const i = bonds[k]
    const j = bonds[k + 1]
    adjacency[i].push(j)
    adjacency[j].push(i)
  }

  const visited = new Uint8Array(n)
  const ox = new Float64Array(n)
  const oy = new Float64Array(n)
  const oz = new Float64Array(n)
  const out = new Float32Array(positions.length)

  for (let root = 0; root < n; root++) {
    if (visited[root]) continue
    // BFS from `root`: ox/oy/oz[member] ends up as that member's offset relative to `root`'s own
    // (wrapped) position, accumulated one bond-length hop at a time -- see this function's own doc
    // comment for why that is the property a single-reference reading cannot guarantee.
    visited[root] = 1
    ox[root] = 0
    oy[root] = 0
    oz[root] = 0
    const queue = [root]
    const members = [root]
    let qi = 0
    while (qi < queue.length) {
      const p = queue[qi++]
      for (const c of adjacency[p]) {
        if (visited[c]) continue
        visited[c] = 1
        ox[c] = ox[p] + mi1(positions[c * 4] - positions[p * 4], oldBox[0])
        oy[c] = oy[p] + mi1(positions[c * 4 + 1] - positions[p * 4 + 1], oldBox[1])
        oz[c] = oz[p] + mi1(positions[c * 4 + 2] - positions[p * 4 + 2], oldBox[2])
        queue.push(c)
        members.push(c)
      }
    }

    let sumx = 0
    let sumy = 0
    let sumz = 0
    for (const b of members) {
      sumx += ox[b]
      sumy += oy[b]
      sumz += oz[b]
    }
    const cx = sumx / members.length
    const cy = sumy / members.length
    const cz = sumz / members.length
    const rx = positions[root * 4]
    const ry = positions[root * 4 + 1]
    const rz = positions[root * 4 + 2]
    const comX = wrap1(rx + cx, oldBox[0]) * sx
    const comY = wrap1(ry + cy, oldBox[1]) * sy
    const comZ = wrap1(rz + cz, oldBox[2]) * sz
    for (const b of members) {
      out[b * 4] = wrap1(comX + (ox[b] - cx), newBox[0])
      out[b * 4 + 1] = wrap1(comY + (oy[b] - cy), newBox[1])
      out[b * 4 + 2] = wrap1(comZ + (oz[b] - cz), newBox[2])
      out[b * 4 + 3] = positions[b * 4 + 3]
    }
  }
  return out
}

/** The dry box a wet `box`/`N` pair maps to under data/soup.json's dryWetCycle.targetDryDensity --
 * isotropic (volume scales as N/targetDryDensity, every axis scales by the same cube-root factor),
 * pulled out as its own pure function so a caller (createSoup below, tests/soup-boxcycle.test.ts) can
 * check the density it actually produces without a GPU. */
export function computeDryBox(box: [number, number, number], N: number, targetDryDensity: number): [number, number, number] {
  const wetVolume = box[0] * box[1] * box[2]
  const dryVolume = N / targetDryDensity
  const scale = Math.cbrt(dryVolume / wetVolume)
  return [box[0] * scale, box[1] * scale, box[2] * scale]
}

/** The wet/dry schedule a running system needs to know nothing about except "what step am I at" --
 * pure functions of a step count so tests/soup-boxcycle.test.ts can check the schedule with no GPU.
 * One cycle = a WET segment (this system's own creation box, `1-dryFraction` of `periodSteps`) THEN
 * a DRY segment (`dryFraction` of `periodSteps`) -- wet first because step 0 IS already the wet box
 * (CreateSoupOpts.box), so cycle 1 needs no box change at all until its own dry segment starts.
 * After `cycles` full cycles, the schedule settles at (and stays at) wet -- `cycleIndex` reports 0
 * once cycling is over, matching "did any cavity survive rehydration" needing a well-defined final
 * wet state to check, not an indefinitely repeating cycle. */
export interface CycleSchedule {
  periodSteps: number
  dryFraction: number
  cycles: number
}

function cycleTransitionSteps(cfg: CycleSchedule): number[] {
  const wetLen = cfg.periodSteps * (1 - cfg.dryFraction)
  const out: number[] = []
  for (let k = 0; k < cfg.cycles; k++) {
    out.push(k * cfg.periodSteps + wetLen) // wet -> dry, this cycle's own dry segment starts
    out.push((k + 1) * cfg.periodSteps) // dry -> wet (the LAST one is final rehydration, cycling ends)
  }
  return out
}

/** The phase ('wet'/'dry') and 1-based cycle number a given real-step count falls in -- `cycleIndex`
 * is 0 once `step` has passed every configured cycle (settled wet, cycling over). */
export function cyclePhaseAt(step: number, cfg: CycleSchedule): { phase: 'wet' | 'dry'; cycleIndex: number } {
  const total = cfg.periodSteps * cfg.cycles
  if (step >= total) return { phase: 'wet', cycleIndex: 0 }
  const within = step % cfg.periodSteps
  const wetLen = cfg.periodSteps * (1 - cfg.dryFraction)
  return { phase: within < wetLen ? 'wet' : 'dry', cycleIndex: Math.floor(step / cfg.periodSteps) + 1 }
}

/** The smallest transition step strictly AFTER `step` -- `Infinity` once cycling is over, the
 * sentinel soup/src/sim.ts's stepCycled() uses to fall through to a plain, unbounded step(). */
export function nextCycleTransition(step: number, cfg: CycleSchedule): number {
  let best = Infinity
  for (const t of cycleTransitionSteps(cfg)) {
    if (t > step && t < best) best = t
  }
  return best
}

// 3-axis minimum-image distance -- the same formula soup/wgsl/forces.wgsl's mi3() encodes on the
// GPU side, needed here only for applyBoxScaleOnce's own runtime self-check (below): a plain CPU
// re-check that every covalent bond's length really did survive a box change, independent of and in
// addition to tests/soup-boxcycle.test.ts's own pure-function check on synthetic topologies.
function mi3Distance(positions: Float32Array, box: [number, number, number], i: number, j: number): number {
  const dx = mi1(positions[i * 4] - positions[j * 4], box[0])
  const dy = mi1(positions[i * 4 + 1] - positions[j * 4 + 1], box[1])
  const dz = mi1(positions[i * 4 + 2] - positions[j * 4 + 2], box[2])
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
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
  // `dims`/`ncells` are NOT fixed for the system's lifetime: resizeSoupGrid() (defined once the GPU
  // buffers/bind groups it touches exist, below) recomputes and reallocates them whenever a dry-wet
  // box change (applyBoxScaleOnce) crosses a cell-count bracket -- mirrors engine/src/sim.ts's own
  // dims/ncells, which areaMove()'s resizeGrid recomputes the same way for the membrane's area move.
  // `cellSize`/`effectiveWalkRadius` above stay fixed for the system's lifetime (they depend only on
  // data/soup.json's species/neighborGrid/verletList settings, never on the box).
  let dims = plan.dims
  let ncells = plan.ncells

  // Dry-wet cycling (task 'wet-dry-cycle', data/soup.json's dryWetCycle.basis): resolved ONCE here,
  // at creation time, from the file's own schedule plus this call's box/N -- never touched again
  // unless applyBoxScale below changes liveBox. `cycleCfg` stays undefined (the guard every use
  // site below checks) when cycling is not requested, so this whole feature costs nothing when off.
  const cycleEnabled = opts.dryWetCycle ?? soup.dryWetCycle.enabled
  let cycleCfg: CycleSchedule | undefined
  let dryBox: [number, number, number] = box
  if (cycleEnabled) {
    const dwc = soup.dryWetCycle
    const wetVolume = box[0] * box[1] * box[2]
    const wetDensity = N / wetVolume
    if (dwc.targetDryDensity <= wetDensity) {
      throw new Error(
        `data/soup.json: dryWetCycle.targetDryDensity=${dwc.targetDryDensity} не превышает текущую ` +
          `плотность бульона ${wetDensity.toFixed(4)} (N=${N}, box=[${box}]) -- сухая фаза обязана концентрировать, не разбавлять`,
      )
    }
    dryBox = computeDryBox(box, N, dwc.targetDryDensity)
    // Guard: the neighbour grid must stay VALID (task requirement 2 -- "fewer than three cells on a
    // periodic axis, or a box too small for the minimum-image convention" -- exactly planSoupGrid's
    // own `valid`, generalised from a fixed "3" to this soup's own minCells=2*effectiveWalkRadius+1)
    // at the smallest box the cycle visits. A box change that merely CHANGES the cell count (wet
    // dims != dry dims) is deliberately NOT rejected here any more: resizeSoupGrid (below, used by
    // applyBoxScaleOnce) now destroys and reallocates the ncells-sized buffers and rebuilds every
    // bind group that references them whenever dims actually changes, mirroring
    // engine/src/sim.ts's own resizeGrid -- the SAME capability this task adds, generalised from the
    // membrane's area move to the soup's dry-wet cycle. planSoupGrid's `valid` is the one thing a
    // reallocation cannot fix (a genuinely invalid geometry, not a merely-different-but-legal cell
    // count), so it remains the sole guard here.
    const dryPlan = planSoupGrid(dryBox, startCounts)
    if (!dryPlan.valid) throw new Error(dryPlan.reason!)
    cycleCfg = { periodSteps: dwc.periodSteps, dryFraction: dwc.dryFraction, cycles: dwc.cycles }
  }

  // Live box, mutated only by applyBoxScale (dry-wet cycling) -- mirrors engine/src/sim.ts's own
  // liveBox/areaMove split exactly: `box` above stays the immutable value opts was created with (the
  // initial layout/grid-size derivation above all use it, once, correctly), `liveBox` is what the
  // `box`/`cyclePhase`/`cycleIndex` getters report and what every GPU-side box reference (gridUniform's
  // own float4 at byte offset 16) is kept in sync with after every applied box change.
  let liveBox: [number, number, number] = [box[0], box[1], box[2]]
  let cyclePhaseState: 'wet' | 'dry' | 'none' = cycleCfg ? cyclePhaseAt(0, cycleCfg).phase : 'none'
  let cycleIndexState = cycleCfg ? cyclePhaseAt(0, cycleCfg).cycleIndex : 0

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
  // Surface growth (surface-growth-report.md): one association slot per particle -- see
  // soup/wgsl/bond.wgsl's own header ("Surface growth") for what a catalyst vs a carbon stores in
  // it. All-unassociated at creation, exactly like bondSlots0.
  const centerLink0 = new Uint32Array(N).fill(NONE_U32)
  // Surface growth / adsorption (adsorption-report.md): per-particle hold-timeout clock
  // (soup/wgsl/bond.wgsl's centerHeldSteps) -- zero-filled, meaningful only once a centre claims a
  // chain (see that file's own comments on where it is reset/incremented/checked).
  const centerHeldSteps0 = new Uint32Array(N)
  // Surface growth / adsorption: [0]=stretch-triggered desorptions, [1]=timeout-triggered
  // desorptions -- see SoupSystem.desorbEvents()'s own doc comment.
  const desorbEventsInit = new Uint32Array(2)
  const bondRng0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) bondRng0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9
  const thermoRng0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) thermoRng0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 3266489917) ^ 0x85ebca6b

  const posBuf = storageBuffer(device, positions0)
  const velBuf = storageBuffer(device, velocities0)
  const forceBuf = storageBuffer(device, new Float32Array(N * 4))
  const cellsBuf = storageBuffer(device, new Float32Array(N))
  // Sized ncells (not N) -- reallocated by resizeSoupGrid() below whenever a dry-wet box change
  // moves dims to a different cell-count bracket, exactly like engine/src/sim.ts's own
  // countsBuf/cellStartBuf/cursorBuf under areaMove()'s resizeGrid.
  let countsBuf = storageBuffer(device, new Float32Array(ncells))
  let cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
  let cursorBuf = storageBuffer(device, new Float32Array(ncells))
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

  // Surface growth (surface-growth-report.md): centerLink, bound at group1 binding20
  // (soup/wgsl/bond.wgsl) into both bond-form bind groups below (cell-walk and Verlet-list
  // variants) -- bond_break_main never references it (co_break/cc_break's own reversal is
  // deliberately orthogonal to this bookkeeping, see that task's own report).
  const centerLinkBuf = device.createBuffer({
    size: centerLink0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(centerLinkBuf, 0, centerLink0)

  // Surface growth / adsorption (adsorption-report.md): centerHeldSteps/desorbEvents, bound at
  // group1 bindings 21/22 (soup/wgsl/bond.wgsl) into both bond-form bind groups below (cell-walk
  // and Verlet-list variants) -- neither is read by soup/wgsl/step.wgsl's force kernels (only
  // centerLink itself is, for the tether force) nor by bond_break_main (desorption is decided
  // entirely inside bond_form_main's own dispatch, see bondFormDecide).
  const centerHeldStepsBuf = device.createBuffer({
    size: centerHeldSteps0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(centerHeldStepsBuf, 0, centerHeldSteps0)
  const desorbEventsBuf = device.createBuffer({
    size: desorbEventsInit.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(desorbEventsBuf, 0, desorbEventsInit)

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
  // Surface growth / adsorption (adsorption-report.md): BondParams grew one more vec4
  // (adsorptionParams -- see bond.wgsl's own struct comment), so its uniform buffer grows from 160
  // to 176 bytes (11 vec4-aligned f32 groups instead of 10) -- the WRITE below is the single place
  // that size must stay in sync with bond.wgsl's own struct layout.
  const bondParamsUniform = device.createBuffer({ size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
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
    // Surface growth / adsorption (adsorption-report.md): data/soup.json's adsorption.maxHoldSteps
    // (real steps, x) and bondAttemptInterval.steps (real steps per bond-dispatch cycle, y -- what
    // soup/wgsl/bond.wgsl's desorbTimeout increments centerHeldSteps by each cycle, so x and the
    // running total it is compared against stay in the SAME real-step units). Both already exist as
    // plain numbers read from data/soup.json / derived above -- no new numeric literal here.
    const adsorptionMaxHoldSteps = soup.adsorption.maxHoldSteps
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
        adsorptionMaxHoldSteps,
        bondAttemptInterval,
        0,
        0,
      ]),
    )
  }

  const bind = (pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ label: `${pipeline.label}@${group}`, layout: pipeline.getBindGroupLayout(group), entries })
  const buf = (b: GPUBuffer) => ({ buffer: b })

  // --- bind groups that reference the ncells-sized buffers (countsBuf/cellStartBuf/cursorBuf) ----
  // Grouped together here -- even though they belong to several different kernels (grid rebuild,
  // soup force, bond formation, the perf2 diagnostic, the Verlet-list build) -- because ALL of them
  // must be rebuilt together by resizeSoupGrid() (below) whenever countsBuf/cellStartBuf/cursorBuf
  // are reallocated: a WebGPU bind group is a fixed reference to specific buffer OBJECTS, so
  // reassigning the `let` that holds a buffer does nothing to a bind group already created against
  // the old one. Exactly the discipline engine/src/sim.ts's own rebindGridDependent()/resizeGrid()
  // established for the membrane's area move, generalised here to this soup's own larger set of
  // grid-dependent kernels (it has bond formation and a Verlet-list build the membrane engine does
  // not). soupForceBruteGroup1/soupForceListGroup1/bondFormListGroup1 (below, NOT here) reference
  // NEITHER buffer -- the brute-force kernel walks no grid at all, and the two *List kernels walk
  // the Verlet list (sized N*listCapacity, independent of ncells) instead of the coarse grid -- so
  // none of those three ever need rebinding on a resize.
  let clearCountsBind!: GPUBindGroup
  let countBind!: GPUBindGroup
  let prefixBind!: GPUBindGroup
  let fillBind!: GPUBindGroup
  let soupForceGroup1!: GPUBindGroup
  let bondFormGroup1!: GPUBindGroup
  let forceStatsGroup1!: GPUBindGroup
  let buildVerletListBind!: GPUBindGroup
  let wgCells = 0

  function rebindGridDependent(): void {
    // --- grid rebuild (engine/wgsl/neighbor.wgsl, unchanged) --------------------------------------
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
    // --- soup force (forces.wgsl + step.wgsl), grid-walk variant ----------------------------------
    soupForceGroup1 = bind(pipe.soupForce, 1, [
      { binding: 0, resource: buf(posBuf) },
      { binding: 1, resource: buf(forceBuf) },
      { binding: 3, resource: buf(gridUniform) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cellsBuf) },
      { binding: 7, resource: buf(bondSlotsBuf) },
      { binding: 8, resource: buf(speciesUniform) },
      { binding: 13, resource: buf(posSortedBuf) },
      // Surface growth / adsorption (adsorption-report.md): centerLink, read-only here
      // (soup/wgsl/step.wgsl's centerLinkRO) for the adsorption tether's own FENE contribution.
      { binding: 20, resource: buf(centerLinkBuf) },
    ])
    // --- bond formation (forces.wgsl + bond.wgsl), grid-walk variant -----------------------------
    bondFormGroup1 = bind(pipe.bondForm, 1, [
      { binding: 0, resource: buf(posBuf) },
      { binding: 3, resource: buf(gridUniform) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cellsBuf) },
      { binding: 6, resource: buf(bondSlotsBuf) },
      { binding: 7, resource: buf(speciesUniform) },
      { binding: 8, resource: buf(eventsBuf) },
      { binding: 9, resource: buf(bondRngBuf) },
      { binding: 13, resource: buf(posSortedBuf) },
      { binding: 20, resource: buf(centerLinkBuf) },
      { binding: 21, resource: buf(centerHeldStepsBuf) },
      { binding: 22, resource: buf(desorbEventsBuf) },
    ])
    // perf2-report.md, STEP 1 diagnosis: soup_force_stats_main's own group1 -- same grid buffers as
    // soup_force_main's group1 above.
    forceStatsGroup1 = bind(pipe.forceStats, 1, [
      { binding: 0, resource: buf(posBuf) },
      { binding: 3, resource: buf(gridUniform) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cellsBuf) },
      { binding: 8, resource: buf(speciesUniform) },
    ])
    // perf2-report.md, candidate (c): builds the Verlet list FROM the coarse grid (cellStartBuf) --
    // must be rebuilt whenever that grid's own buffers are.
    buildVerletListBind = bind(pipe.buildVerletList, 1, [
      { binding: 0, resource: buf(posBuf) },
      { binding: 3, resource: buf(gridUniform) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cellsBuf) },
      { binding: 14, resource: buf(verletListBuf) },
      { binding: 15, resource: buf(verletCountBuf) },
      { binding: 16, resource: buf(verletOverflowBuf) },
      { binding: 19, resource: buf(verletUniform) },
    ])
    wgCells = Math.ceil(ncells / 64)
  }
  rebindGridDependent()

  /** Dry-wet cycling's own grid-rebuild capability (task 'grid-rebuild'), generalising
   * engine/src/sim.ts's own resizeGrid() (added there for the membrane's area move, reviewed) to the
   * soup engine: recomputes dims for `newBox` via planSoupGrid -- the single source of truth for
   * grid validity, the SAME derivation createSoup used at construction time -- and, if the cell
   * count actually changed, destroys and reallocates countsBuf/cellStartBuf/cursorBuf and rebuilds
   * every bind group referencing them (rebindGridDependent, above), then rewrites the grid uniform
   * (dims + box) unconditionally. Called by applyBoxScaleOnce after every dry-wet transition
   * increment -- this is what removes the blocker wet-dry-cycle-report.md measured: a box change
   * that moves dims into a different cell-count bracket (the FULL-scale case at box 46,
   * targetDryDensity=0.6: wet dims=[15,15,15] -> dry dims=[14,14,14]) no longer desyncs the
   * fixed-size grid buffers from the live box, it reallocates them to match. planSoupGrid's own
   * `valid` is the ONLY thing this cannot paper over -- a box with fewer than minCells cells on some
   * axis, or one that violates the minimum-image convention, is a genuinely invalid geometry, not a
   * cell-count change a reallocation can fix, so it still throws (task requirement 2: "let it fire
   * only on a genuinely invalid geometry ... not on a legal change of cell count"). The Verlet list
   * itself (verletListBuf, sized N*listCapacity -- independent of ncells) is never touched here; its
   * own rebuild (needed because neighbourhoods move under any box change) is the caller's job, done
   * immediately afterward by applyBoxScaleOnce's own encodeVerletRebuild call, whose
   * assertVerletSafety check afterward is what keeps task requirement 3 (the overflow guard "must
   * stay in force") true across a resize. */
  function resizeSoupGrid(newBox: [number, number, number]): void {
    const newPlan = planSoupGrid(newBox, startCounts)
    if (!newPlan.valid) throw new Error(newPlan.reason!)
    if (newPlan.dims[0] !== dims[0] || newPlan.dims[1] !== dims[1] || newPlan.dims[2] !== dims[2]) {
      countsBuf.destroy()
      cellStartBuf.destroy()
      cursorBuf.destroy()
      dims = newPlan.dims
      ncells = newPlan.ncells
      countsBuf = storageBuffer(device, new Float32Array(ncells))
      cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
      cursorBuf = storageBuffer(device, new Float32Array(ncells))
      rebindGridDependent()
    }
    const bytes = new ArrayBuffer(32)
    new Uint32Array(bytes, 0, 4).set([dims[0], dims[1], dims[2], effectiveWalkRadius])
    new Float32Array(bytes, 16, 4).set([newBox[0], newBox[1], newBox[2], 0])
    device.queue.writeBuffer(gridUniform, 0, bytes)
  }

  // perf2-report.md, candidate (b): gather bind group, only group 1 (no Params uniform needed for
  // a plain copy) -- pos2 (read) + cellIdx (read, the permutation) + posSortedRW (write target).
  // cellsBuf is sized N (the permutation, one slot per bead), never reallocated by resizeSoupGrid --
  // this bind group is unaffected by ncells and does not need rebinding.
  const gatherSortedBind = bind(pipe.gatherSorted, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 5, resource: buf(cellsBuf) },
    { binding: 13, resource: buf(posSortedBuf) },
  ])

  // --- soup force + wrap (forces.wgsl + step.wgsl) -----------------------------------------------
  const soupForceGroup0 = bind(pipe.soupForce, 0, [{ binding: 0, resource: buf(paramsUniform) }])
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
    { binding: 20, resource: buf(centerLinkBuf) },
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
  // same grid), plus its own group 3 output. forceStatsGroup1 itself lives in rebindGridDependent
  // above (it references cellStartBuf).
  const forceStatsGroup0 = bind(pipe.forceStats, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const forceStatsGroup3 = bind(pipe.forceStats, 3, [{ binding: 0, resource: buf(statsBuf) }])

  // perf2-report.md, candidate (c): Verlet list bind groups. buildVerletListBind itself lives in
  // rebindGridDependent above (it references cellStartBuf); snapshotPositions/resetMaxDrift/
  // maxDrift never reference P (no group 0 needed) -- only the geometry/positions and their own list
  // buffers, none of them ncells-sized, so none need rebinding on a resize.
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
    { binding: 20, resource: buf(centerLinkBuf) },
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
    { binding: 20, resource: buf(centerLinkBuf) },
    { binding: 21, resource: buf(centerHeldStepsBuf) },
    { binding: 22, resource: buf(desorbEventsBuf) },
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

  // Dry-wet cycling (task 'wet-dry-cycle'): applies ONE box change (liveBox -> targetBox), rebuilding
  // and re-scaling every molecule's center of mass -- see scaleMoleculesRigid's own doc comment for
  // the discipline this reuses from engine/src/sim.ts's areaMove. Never called directly by a public
  // API; only applyBoxScale (below, the ramped wrapper) and, through it, stepCycled call this.
  async function applyBoxScaleOnce(targetBox: [number, number, number]): Promise<void> {
    const before = await particles()
    const bondPairs = await bonds()

    // Runtime self-check (task requirement 2: "assert intramolecular distances are unchanged by a
    // box change") -- not only the pure-function unit test on synthetic topologies
    // (tests/soup-boxcycle.test.ts), but every real application, on the real bond graph this call
    // actually sees. Cheap: O(bonds), a tiny fraction of O(N).
    const distBefore = new Float64Array(bondPairs.length / 2)
    for (let k = 0; k < bondPairs.length; k += 2) {
      distBefore[k / 2] = mi3Distance(before, liveBox, bondPairs[k], bondPairs[k + 1])
    }
    const after = scaleMoleculesRigid(before, bondPairs, liveBox, targetBox)
    for (let k = 0; k < bondPairs.length; k += 2) {
      const d = mi3Distance(after, targetBox, bondPairs[k], bondPairs[k + 1])
      const beforeD = distBefore[k / 2]
      if (Math.abs(d - beforeD) > 1e-3) {
        throw new Error(
          `applyBoxScale: внутримолекулярное расстояние изменилось (${beforeD.toFixed(6)} -> ${d.toFixed(6)}) ` +
            `для связи ${bondPairs[k]}-${bondPairs[k + 1]} при box [${liveBox}] -> [${targetBox}] -- scaleMoleculesRigid нарушен`,
        )
      }
    }

    device.queue.writeBuffer(posBuf, 0, after)
    liveBox = targetBox
    // resizeSoupGrid (task 'grid-rebuild', generalising engine/src/sim.ts's own resizeGrid):
    // recomputes dims for targetBox and, if the cell count changed, destroys/reallocates
    // countsBuf/cellStartBuf/cursorBuf and rebuilds every bind group referencing them, THEN rewrites
    // gridUniform's dims+box -- replaces this call's former bare `writeBuffer(gridUniform, 16, ...)`
    // (box floats only), which is exactly what let a dims change silently desync the fixed-size grid
    // buffers from the live box (wet-dry-cycle-report.md's own measured blocker at box 46).
    resizeSoupGrid(targetBox)

    // Force an immediate rebuild (and, when the Verlet list is enabled, a fresh drift-safety
    // snapshot) so the NEXT real step's force/bond-attempt pass sees the density this call just
    // applied, and so the drift-safety guard measures real diffusive drift afterward, not the
    // artificial jump this rescale itself made -- exactly why engine/src/sim.ts's own areaMove
    // rebuilds the grid after every trial, accepted or not (its own comment: "the box the grid's
    // cell/box uniform refers to may have changed"). Also where the Verlet list overflow guard
    // (task requirement 3) gets its first chance to fire for this box, immediately rather than
    // waiting up to `verletList.rebuildEvery` real steps for the next scheduled rebuild -- now ALSO
    // exercised against a freshly reallocated grid, not merely a rewritten uniform, whenever this
    // box change actually crossed a cell-count bracket.
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    if (verlet.enabled) {
      encodeVerletRebuild(pass)
    } else {
      encodeGridRebuild(pass)
    }
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    if (verlet.enabled) await assertVerletSafety()
  }

  /** Spreads one wet<->dry box change over `cycleCfg.rampSteps` log-linear increments, each followed
   * by `cycleCfg.rampRelaxSteps` of ordinary dynamics -- data/soup.json's dryWetCycle.basis gives the
   * physical reasoning (the same "keep each proposed change small" discipline engine/src/sim.ts's
   * areaMove already applies to its own MC trials, generalised to a forced mechanical move that has
   * no Metropolis rejection to fall back on if one increment were too large). Only ever called from
   * stepCycled, which has already established `cycleCfg` is defined. */
  async function applyBoxScale(targetBox: [number, number, number], cfg: CycleSchedule & { rampSteps: number; rampRelaxSteps: number }): Promise<void> {
    const fromBox = liveBox
    const lnRatio: [number, number, number] = [
      Math.log(targetBox[0] / fromBox[0]),
      Math.log(targetBox[1] / fromBox[1]),
      Math.log(targetBox[2] / fromBox[2]),
    ]
    for (let s = 1; s <= cfg.rampSteps; s++) {
      // The FINAL increment lands on the exact target box (not a geometric-interpolation rounding of
      // it) -- liveBox after a full ramp must equal targetBox bit for bit, since the next ramp's own
      // `fromBox` (and the guard's own dims check at creation time) both reason about the EXACT wet/
      // dry box pair, not an accumulated floating-point drift of it.
      const interp: [number, number, number] =
        s === cfg.rampSteps
          ? targetBox
          : [
              fromBox[0] * Math.exp((lnRatio[0] * s) / cfg.rampSteps),
              fromBox[1] * Math.exp((lnRatio[1] * s) / cfg.rampSteps),
              fromBox[2] * Math.exp((lnRatio[2] * s) / cfg.rampSteps),
            ]
      await applyBoxScaleOnce(interp)
      if (s < cfg.rampSteps && cfg.rampRelaxSteps > 0) await step(cfg.rampRelaxSteps)
    }
  }

  async function stepCycled(n: number): Promise<void> {
    if (!cycleCfg) {
      await step(n)
      return
    }
    const cfg = { ...cycleCfg, rampSteps: soup.dryWetCycle.rampSteps, rampRelaxSteps: soup.dryWetCycle.rampRelaxSteps }
    let remaining = n
    while (remaining > 0) {
      const next = nextCycleTransition(globalStep, cfg)
      const advance = next === Infinity ? remaining : Math.min(remaining, next - globalStep)
      if (advance > 0) {
        await step(advance)
        remaining -= advance
      }
      if (next !== Infinity && globalStep === next) {
        const { phase, cycleIndex } = cyclePhaseAt(globalStep, cfg)
        await applyBoxScale(phase === 'dry' ? dryBox : box, cfg)
        cyclePhaseState = phase
        cycleIndexState = cycleIndex
      } else if (advance === 0 && remaining > 0) {
        // Defensive: advance===0 with steps still remaining and no transition to apply would spin
        // forever -- cannot happen given nextCycleTransition's own contract (it only ever returns a
        // step strictly greater than the current one, or Infinity), kept as a hard stop rather than
        // a silent infinite loop if that contract is ever violated by a future edit.
        throw new Error('stepCycled: расписание циклов зациклилось -- nextCycleTransition вернул текущий globalStep')
      }
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

  async function centerLinks(): Promise<Uint32Array> {
    const raw = await readBack(device, centerLinkBuf, N * 4)
    return new Uint32Array(raw.buffer, raw.byteOffset, N)
  }

  async function desorbEvents(): Promise<{ stretch: number; timeout: number }> {
    const raw = await readBack(device, desorbEventsBuf, desorbEventsInit.byteLength)
    const u32 = new Uint32Array(raw.buffer, raw.byteOffset, 2)
    return { stretch: u32[0], timeout: u32[1] }
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
  ): Promise<{
    reached: boolean
    steps: number
    trace: { steps: number; stage: Stage; evidence: StageEvidence; cyclePhase: 'wet' | 'dry' | 'none'; cycleIndex: number }[]
  }> {
    const trace: { steps: number; stage: Stage; evidence: StageEvidence; cyclePhase: 'wet' | 'dry' | 'none'; cycleIndex: number }[] = []
    let steps = 0
    let reached = false
    while (steps < opts.maxSteps) {
      const chunk = Math.min(opts.sampleEvery, opts.maxSteps - steps)
      // stepCycled, not step directly (task requirement 4: "record the cycle phase and count in the
      // trace"): identical to step() when this system was not created with dry-wet cycling, so every
      // caller without cycling sees no difference -- see stepCycled's own doc comment.
      await sys.stepCycled(chunk)
      steps += chunk
      const { stage: currentStage, evidence } = await detectStage(sys)
      trace.push({ steps, stage: currentStage, evidence, cyclePhase: sys.cyclePhase, cycleIndex: sys.cycleIndex })
      // Printed AS IT HAPPENS (not buffered to the end) -- the whole point per this task's brief:
      // a run long enough to matter (the pilot is minutes, the full-scale run tens of minutes) must
      // be diagnosable while it is still running, not only from the return value after the fact.
      console.log(
        `[runUntil] steps=${steps} stage=${currentStage} cyclePhase=${sys.cyclePhase} cycleIndex=${sys.cycleIndex} ` +
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
    centerLinkBuf.destroy()
    centerHeldStepsBuf.destroy()
    desorbEventsBuf.destroy()
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
    stepCycled,
    particles,
    forces,
    forcesBruteForce,
    bonds,
    centerLinks,
    desorbEvents,
    events,
    invariants,
    get box(): [number, number, number] {
      return [liveBox[0], liveBox[1], liveBox[2]]
    },
    get cyclePhase(): 'wet' | 'dry' | 'none' {
      return cyclePhaseState
    },
    get cycleIndex(): number {
      return cycleIndexState
    },
    get steps(): number {
      return globalStep
    },
    runUntil,
    dispose,
    stepPhasesDEBUG: stepPhasesDEBUG as any,
    forceCandidateStatsDEBUG: forceCandidateStatsDEBUG as any,
  }
  return sys
}
