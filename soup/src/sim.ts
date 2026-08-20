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
//  - forces.wgsl concatenated with soup/wgsl/bond-*.wgsl -- bond_form_main/bond_break_main, the
//    Metropolis bond dynamics. See soup/wgsl/bond-common.wgsl's header for how detailed balance is
//    carried end to end from soup/src/rules.ts's own acceptanceProbability/attemptProbability, not
//    re-derived.
//  - engine/wgsl/neighbor.wgsl (UNCHANGED) for the grid build -- direction-agnostic (no box/
//    periodicity reference), reused exactly as the membrane engine uses it.
//
// No numeric model constant is written in this file: every physical number comes from
// loadSoup()/loadParams() at runtime; params.test.ts's literal scanner enforces this over
// soup/src and soup/wgsl.
//
// Split (2026-08-18, file-size rule in CLAUDE.md) out of a former 2351-line monolith into modules
// by responsibility -- this file is now a thin composition point: it derives creation-time
// config, wires the modules below in the same order the original createSoup's body executed them
// in, and assembles the SoupSystem object. Every module keeps the code that used to live in this
// file's closure verbatim, parameterised over a shared `SoupRuntime` context (soup/src/
// soup-runtime.ts) instead of closing over local variables directly -- see each module's own
// header for what responsibility it owns:
//  - soup/src/soup-types.ts        -- CreateSoupOpts/SoupSystem/NONE_U32
//  - soup/src/soup-plan.ts         -- rule resolution, neighbour-grid/Verlet-list geometry+guards
//  - soup/src/soup-init-state.ts   -- initial lattice OR checkpoint-resume state
//  - soup/src/soup-pipelines.ts    -- compute pipeline creation (device-keyed cache)
//  - soup/src/soup-buffers.ts      -- GPU buffer allocation/ownership, the grid-resize path
//  - soup/src/soup-bindgroups.ts   -- bind-group construction, rebind-on-resize
//  - soup/src/soup-grid-verlet.ts  -- neighbour grid + Verlet list, completeness guards
//  - soup/src/soup-bond-chemistry.ts -- bond-kernel dispatch (the chemistry itself is in WGSL)
//  - soup/src/soup-integrate.ts    -- the integration step, chunked submission
//  - soup/src/soup-box-scale.ts    -- box-scaling / dry-wet cycling, rigid-COM scaling
//  - soup/src/soup-readback.ts     -- measurement/readback helpers (the checkpoint "capture" half)

import { getGpu } from '../../engine/src/gpu'
import { loadParams, type Params } from '../../engine/src/params'
import { assertRulesConsistent, loadSoup } from './rules'
import { detectStage, type Stage, type StageEvidence } from './stages'

import type { CreateSoupOpts, SoupSystem } from './soup-types'
import { resolveRules, planSoupGrid, deriveGridGeometry, type SoupPlan } from './soup-plan'
import { buildInitialState } from './soup-init-state'
import { getSoupPipelines } from './soup-pipelines'
import { allocateSoupBuffers, disposeSoupBuffers } from './soup-buffers'
import { buildBindGroups } from './soup-bindgroups'
import { encodeGridRebuild, encodeVerletRebuild } from './soup-grid-verlet'
import { encodeSoupForce, encodeSoupForceList, buildStepper } from './soup-integrate'
import {
  deriveCycleConfig,
  makeScaleBoxTo,
  makeGrowBoxTo,
  makeStepCycled,
  scaleMoleculesRigid,
  computeDryBox,
  cyclePhaseAt,
  nextCycleTransition,
  type CycleSchedule,
} from './soup-box-scale'
import { makeSoupAreaMove } from './soup-area-move'
// Task 'loud-failure-and-liquid-water' (2026-08-20): the cold-start minimiser (opt-in, before step 1
// only) and the non-finite state guard. The guard's own per-chunk hook lives in
// soup/src/soup-integrate.ts's stepper; what is wired HERE is only the two public methods.
import { makeRelaxColdStart } from './soup-relax'
// Task 'evaporation' (2026-08-20): real solvent removal/return. planEvaporation is pure (no GPU) and
// carries its own throwing preconditions; nothing here changes for a system that does not ask for it.
import { planEvaporation, evaporateSolventTo, rehydrateSolventTo } from './soup-evaporate'
import { scanNonFinite } from './soup-health'
import { clayEnabled, planClay, type ClayLayout } from './soup-clay'
import * as readback from './soup-readback'
import type { SoupRuntime } from './soup-runtime'

export type { CreateSoupOpts, SoupSystem } from './soup-types'
export { planSoupGrid, type SoupPlan } from './soup-plan'
export { scaleMoleculesRigid, computeDryBox, cyclePhaseAt, nextCycleTransition, type CycleSchedule } from './soup-box-scale'

export async function createSoup(opts: CreateSoupOpts): Promise<SoupSystem> {
  const soup = loadSoup()
  assertRulesConsistent(soup)
  // Task 'explicit-water' (2026-08-18): raised from 4 to 8 -- soup/wgsl/step.wgsl and
  // bond-common.wgsl's Species struct now packs radius/polar/solvent into array<vec4<f32>,2> each
  // (soup/src/soup-plan.ts's packSpeciesSlots matches it), not a single vec4, specifically to make
  // room for the 5th species (water) without yet another restructure the next time one more is
  // needed.
  if (soup.monomers.length > 8) {
    throw new Error(`data/soup.json: ${soup.monomers.length} видов мономеров — шейдер вмещает не больше 8`)
  }
  const baseParams = loadParams()
  const p: Params = { ...baseParams, thermostat: { ...baseParams.thermostat, kT: opts.kT } }

  const { rules, catalystKind } = resolveRules(soup)
  const eventRuleIds: [string, string][] = rules.map((r) => [r.bond.id, r.brk.id])

  const startCounts: Record<string, number> = { ...soup.start, ...(opts.start ?? {}) }
  if (opts.catalystCount !== undefined) {
    const catalystMonomer = soup.monomers.find((m) => m.kind === 'catalyst')!
    startCounts[catalystMonomer.id] = opts.catalystCount
  }
  const box = opts.box

  // Task 'clay-surface' (2026-08-19): the immobile mineral platelet. Its bead count is DERIVED from
  // the box and from the mineral bead's own WCA contact distance (soup/src/soup-clay.ts), never read
  // from `start` -- data/soup.json has no start entry for it at all, because a fixed count would tile
  // exactly one box and leave holes in every other. So it is injected into startCounts HERE, before N
  // is summed, and everything downstream (N, countsByKind, planSoupGrid, the invariants readback) sees
  // the platelet as an ordinary part of the composition.
  //
  // The catalytic surface sites are NOT added on top of the catalyst pool: `planClay` takes the
  // fraction data/soup.json's clay.siteCatalystFraction names out of THIS system's own catalyst count,
  // so a with-clay and a without-clay run carry identical catalyst totals and the growth comparison
  // cannot be explained by "one arm simply had more catalyst" (clay.basis §6).
  let clay: ClayLayout | null = null
  if (clayEnabled(soup, opts.clay)) {
    const catalystMonomer = soup.monomers.find((m) => m.kind === 'catalyst')!
    clay = planClay(soup, baseParams, box, startCounts[catalystMonomer.id] ?? 0, opts.claySiteCatalystFraction)
    startCounts[soup.clay!.mineralId] = clay.mineralCount
  } else if (soup.clay) {
    startCounts[soup.clay.mineralId] = 0
  }

  // Task 'evaporation' (2026-08-20): CAPACITY vs ACTIVE. `capacityN` is the creation (wet)
  // composition and is what every GPU buffer is sized for -- so the slots a dry phase empties are
  // still there for rehydration to write into. `N` is the LIVE count, equal to capacityN except when
  // resuming a checkpoint taken mid dry-phase, which carries its own (smaller) census. Both are the
  // same number for every caller that does not evaporate, so nothing below changes for them.
  const countsByKind = soup.monomers.map((m) => startCounts[m.id] ?? 0)
  const capacityN = countsByKind.reduce((a, b) => a + b, 0)
  const activeCounts: Record<string, number> = opts.resume?.activeCounts ? { ...opts.resume.activeCounts } : { ...startCounts }
  const N = soup.monomers.reduce((sum, m) => sum + (activeCounts[m.id] ?? 0), 0)
  if (capacityN === 0) throw new Error('createSoup: стартовый состав пуст')
  if (N > capacityN) {
    throw new Error(`createSoup: живой состав (${N}) больше стартового (${capacityN}) -- буферы выделяются под стартовый`)
  }

  // Neighbour-grid/Verlet-list geometry, WITH its three throwing completeness guards (walk-radius
  // coverage, list coverage, drift-safety bound) -- see soup/src/soup-plan.ts's deriveGridGeometry.
  const { effectiveWalkRadius, listRange } = deriveGridGeometry(soup, p, opts.kT)

  // Checkpoint/resume (task 'checkpoint-resume'): a resumed system's grid must be sized for
  // whatever box it actually LIVES in right now, not for the wet/creation box `box` -- a
  // checkpoint taken mid dry-phase has to reconstruct the dry grid directly, not the wet one
  // createSoup would otherwise default to. `initialStep`/`initialLiveBox` fall back to the
  // pre-existing behaviour (globalStep 0, the creation box) whenever opts.resume is absent, so
  // every existing caller sees no change at all.
  const initialStep = opts.resume?.globalStep ?? 0
  const initialLiveBox: [number, number, number] = opts.resume?.liveBox ?? box
  const plan: SoupPlan = planSoupGrid(initialLiveBox, startCounts)
  if (!plan.valid) throw new Error(plan.reason!)

  // Task 'evaporation' (2026-08-20): resolved BEFORE the cycle config, because the dry box a cycle
  // with solvent removal targets is sized for what is LEFT, not for the wet N.
  const evapRequested = (opts.dryWetCycle ?? soup.dryWetCycle.enabled) && (opts.evaporateSolvent ?? soup.dryWetCycle.evaporateSolvent ?? false)
  const evap = evapRequested ? planEvaporation(soup, p, box, startCounts) : undefined

  // Dry-wet cycling setup (task 'wet-dry-cycle'), with its own throwing validation -- see
  // soup/src/soup-box-scale.ts's deriveCycleConfig.
  const { cycleCfg, dryBox } = deriveCycleConfig(soup, opts, box, capacityN, startCounts, evap?.dryBox)

  const { device } = await getGpu()
  const sortedGather = soup.neighborGrid.sortedGather
  const pipe = getSoupPipelines(device, sortedGather)

  const initial = buildInitialState(soup, opts, capacityN, countsByKind, box, rules, eventRuleIds, clay, N)
  const { buf, grid } = allocateSoupBuffers({
    device,
    soup,
    p,
    N: capacityN,
    dims: plan.dims,
    ncells: plan.ncells,
    effectiveWalkRadius,
    initialLiveBox,
    verlet: soup.verletList,
    listRange,
    rules,
    catalystKind,
    bondAttemptInterval: soup.bondAttemptInterval.steps,
    kT: opts.kT,
    initial,
    solventAttractionScaleOverride: opts.solventAttractionScaleOverride,
    claySurfaceChemistry: opts.claySurfaceChemistry,
  })
  const bind = buildBindGroups(device, pipe, buf, N)

  const rt: SoupRuntime = {
    device,
    pipe,
    soup,
    p,
    rules,
    catalystKind,
    eventRuleIds,
    startCounts,
    N,
    wgN: Math.ceil(N / 64),
    verlet: soup.verletList,
    effectiveWalkRadius,
    bondAttemptInterval: soup.bondAttemptInterval.steps,
    sortedGather,
    // Task 'clay-surface': how many particles are immobile. Kept on the runtime (rather than
    // re-derived) so soup/src/soup-box-scale.ts's applyBoxScaleOnce can REFUSE a box change on a
    // system that carries a platelet -- an affine scale would stretch the sheet's own lattice and move
    // beads that by construction cannot move (clay.basis §7).
    frozenCount: initial.frozen0.reduce((a, b) => a + b, 0),
    clay,
    box,
    dryBox,
    cycleCfg,
    evap,
    buf,
    bind,
    grid,
    live: {
      liveBox: [initialLiveBox[0], initialLiveBox[1], initialLiveBox[2]],
      globalStep: initialStep,
      cyclePhase: cycleCfg ? cyclePhaseAt(initialStep, cycleCfg).phase : 'none',
      cycleIndex: cycleCfg ? cyclePhaseAt(initialStep, cycleCfg).cycleIndex : 0,
    },
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
    if (rt.verlet.enabled) {
      encodeVerletRebuild(rt, pass)
      encodeSoupForceList(rt, pass)
    } else {
      encodeGridRebuild(rt, pass)
      encodeSoupForce(rt, pass)
    }
    pass.end()
    device.queue.submit([enc.finish()])
  }

  const step = buildStepper(rt)
  const particles = () => readback.particles(rt)
  const bonds = () => readback.bonds(rt)
  const scaleBoxTo = makeScaleBoxTo(rt, particles, bonds, step)
  const growBoxTo = makeGrowBoxTo(rt, step)
  // Task 'hydrophobic-asymmetry' (2026-08-19), defect 2 -- the zero-tension MC area move, ported
  // from engine/src/sim-area-move.ts onto the soup's own scaleMoleculesRigid/applyBoxScaleOnce
  // machinery rather than duplicated (see soup/src/soup-area-move.ts's header for what is reused and
  // what genuinely could not be).
  const areaMove = makeSoupAreaMove({
    rt,
    particles,
    bonds,
    centerLinks: () => readback.centerLinks(rt),
    soup,
    seed: opts.seed,
    attractionOverride: opts.solventAttractionScaleOverride,
    claySurfaceChemistry: opts.claySurfaceChemistry,
  })
  const stepCycled = makeStepCycled(
    rt,
    particles,
    bonds,
    step,
    { rampSteps: soup.dryWetCycle.rampSteps, rampRelaxSteps: soup.dryWetCycle.rampRelaxSteps },
    () => readback.forces(rt),
    () => readback.centerLinks(rt),
    opts.seed,
  )

  // Task 4: the continuous soup->vesicle run. Declared with `let sys!` and assigned AFTER the
  // object literal below so runUntil's own closure can call detectStage(sys) -- detectStage only
  // ever reads sys.particles()/sys.bonds()/sys.box (see stages.ts), never sys.runUntil itself, so
  // the circularity is only in the TYPE, not in anything actually read before `sys` is assigned.
  async function runUntil(
    stage: Stage,
    runOpts: { maxSteps: number; sampleEvery: number },
  ): Promise<{
    reached: boolean
    steps: number
    trace: { steps: number; stage: Stage; evidence: StageEvidence; cyclePhase: 'wet' | 'dry' | 'none'; cycleIndex: number }[]
  }> {
    const trace: { steps: number; stage: Stage; evidence: StageEvidence; cyclePhase: 'wet' | 'dry' | 'none'; cycleIndex: number }[] = []
    let steps = 0
    let reached = false
    while (steps < runOpts.maxSteps) {
      const chunk = Math.min(runOpts.sampleEvery, runOpts.maxSteps - steps)
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

  function dispose(): void {
    disposeSoupBuffers(rt.buf)
  }

  let sys!: SoupSystem
  sys = {
    step,
    stepCycled,
    scaleBoxTo,
    growBoxTo,
    areaMove,
    particles,
    velocities: () => readback.velocities(rt),
    forces: () => readback.forces(rt),
    forcesBruteForce: () => readback.forcesBruteForce(rt),
    bonds,
    bondSlots: () => readback.readBondSlots(rt),
    centerLinks: () => readback.centerLinks(rt),
    centerHeldSteps: () => readback.centerHeldSteps(rt),
    frozen: () => readback.frozen(rt),
    clayPlanes: () => (clay ? [...clay.planeZ] : []),
    desorbEvents: () => readback.desorbEvents(rt),
    rngState: () => readback.rngState(rt),
    events: () => readback.events(rt),
    invariants: () => readback.invariants(rt, soup),
    nonFiniteCount: () => scanNonFinite(rt),
    relaxColdStart: makeRelaxColdStart(rt, () => readback.forces(rt)),
    evaporateDEBUG: async (targetSolvent: number) => {
      if (!evap) throw new Error('evaporateDEBUG: система создана без испарения растворителя (CreateSoupOpts.evaporateSolvent)')
      await evaporateSolventTo(rt, evap, targetSolvent, rt.live.liveBox)
    },
    rehydrateDEBUG: async (targetSolvent: number) => {
      if (!evap) throw new Error('rehydrateDEBUG: система создана без испарения растворителя (CreateSoupOpts.evaporateSolvent)')
      return rehydrateSolventTo(rt, evap, targetSolvent, rt.live.liveBox, particles, () => readback.forces(rt), opts.seed)
    },
    get box(): [number, number, number] {
      return [rt.live.liveBox[0], rt.live.liveBox[1], rt.live.liveBox[2]]
    },
    get cyclePhase(): 'wet' | 'dry' | 'none' {
      return rt.live.cyclePhase
    },
    get cycleIndex(): number {
      return rt.live.cycleIndex
    },
    get steps(): number {
      return rt.live.globalStep
    },
    runUntil,
    dispose,
    stepPhasesDEBUG: ((n: number) => readback.stepPhasesDEBUG(rt, n)) as any,
    forceCandidateStatsDEBUG: (() => readback.forceCandidateStatsDEBUG(rt)) as any,
  }
  return sys
}
