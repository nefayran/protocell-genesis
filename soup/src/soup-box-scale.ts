// The stateful GPU-side ramp for box-scaling and dry-wet cycling. Split out of soup/src/sim.ts
// (file-size rule in CLAUDE.md) -- applyBoxScaleOnce/applyBoxScale/scaleBoxTo/growBoxTo/stepCycled,
// moved verbatim. The pure, GPU-free coordinate map + schedule (scaleMoleculesRigid, computeDryBox,
// cyclePhaseAt, nextCycleTransition, deriveCycleConfig, etc.) lives in soup/src/
// soup-box-scale-math.ts instead -- this file re-exports the ones tests/soup-boxcycle.test.ts and
// soup/src/sim.ts still need under their original names, so nothing importing from `soup/src/sim`
// sees any difference.

import { resizeSoupGrid } from './soup-buffers'
import { encodeGridRebuild, encodeVerletRebuild, assertVerletSafety } from './soup-grid-verlet'
import { encodeSoupForce, encodeSoupForceList } from './soup-integrate'
import { assertStateFinite } from './soup-health'
import { rebindGridDependent } from './soup-bindgroups'
import type { SoupRuntime } from './soup-runtime'
import { scaleMoleculesRigid, proximityPairs, mi3Distance, cyclePhaseAt, nextCycleTransition, type CycleSchedule } from './soup-box-scale-math'
import { evaporationLadder, rehydrationLadder, evaporateSolventTo, rehydrateSolventTo, desorbOverstretchedTethers, type EvaporationPlan, type RehydrationReport } from './soup-evaporate'
import { relaxIterations } from './soup-relax'

export {
  scaleMoleculesRigid,
  proximityPairs,
  computeDryBox,
  cyclePhaseAt,
  nextCycleTransition,
  mi3Distance,
  deriveCycleConfig,
  type CycleSchedule,
} from './soup-box-scale-math'

/** Applies ONE box change (rt.live.liveBox -> targetBox), rebuilding and re-scaling every
 * molecule's center of mass -- see scaleMoleculesRigid's own doc comment (soup-box-scale-math.ts)
 * for the discipline this reuses from engine/src/sim.ts's areaMove. Never called directly by a
 * public API; only applyBoxScale (below, the ramped wrapper) and, through it, stepCycled call
 * this.
 *
 * `cohesion` (box-expansion task, 2026-08-18): OPTIONAL, undefined for every EXISTING caller
 * (dry-wet cycling never passes it, so its own behaviour is byte-for-byte unchanged). When present,
 * scaleMoleculesRigid's own rigid-unit grouping is computed over covalent bonds PLUS proximityPairs
 * (`before`-fresh -- see why that matters below) within `cohesion.cutoff` -- by default among ALL
 * particles (`cohesion.memberIndices` omitted), or restricted to a caller-named subset when given.
 * `cutoff` is expected to be the SAME wcaCutoff(...)+attraction.wc clustering cutoff
 * soup/src/aggregates.ts's own analyzeAggregates() uses for its own aggregate recognition. Measured
 * need for this and the path to what actually works (soup/cli/campaign.ts's own --expandTo, all
 * three checked in pure Node against the real checkpoint, no GPU, before touching the real system):
 *  1. NO cohesion at all: scaling this checkpoint's ~1282-amphiphile vesicle as 1282 independent
 *     covalent chains let two DIFFERENT, non-covalently-bonded chains that happen to sit on nearly
 *     opposite sides of a periodic wrap get pushed toward instead of away from each other by the
 *     per-molecule wrap-then-scale map (a real, reproducible geometric consequence of scaling small
 *     units independently inside a large, densely-packed, non-covalently-held structure): min
 *     pairwise distance among ALL particles dropped from 0.68 sigma (original) to 0.27 sigma after a
 *     single ~6.5% linear increment -- deep inside WCA's repulsive core, and exactly what produced
 *     the ~1.36e6 sigma assertVerletSafety drift the first real GPU run threw on, a few real steps
 *     into the very first increment's relaxation.
 *  2. Global cohesion cutoff, but the proximity graph computed ONCE (at the ORIGINAL box) and
 *     reused unchanged across every chained ramp increment: at this checkpoint's own box-average
 *     density (0.478/sigma^3), a 2.72 sigma cutoff sphere holds ~40 neighbours on average, enough to
 *     percolate ~5.5 MILLION proximity edges across nearly all 93200 particles into one giant rigid
 *     component -- which should trivially preserve every internal distance, but a STALE graph means
 *     the ~20 particles that fall OUTSIDE that giant component at the ORIGINAL box (or move across
 *     its boundary as the ramp proceeds) are scaled inconsistently with their real-time neighbours,
 *     degrading the worst pairwise distance to 0.20-0.33 sigma over a chain -- better than (1) but
 *     still a real, avoidable artefact.
 *  3. Restricting eligibility to just the recognised aggregate's own membership (a natural next
 *     idea: "only the densely-packed structure needs rigid treatment") made this WORSE, not better:
 *     the vesicle's ~11688 members do form one clean internal component under this cutoff, but
 *     leaving every DILUTE particle in physical contact with the vesicle's own surface OUTSIDE the
 *     cohesion graph opened a NEW boundary at that surface -- worst pairwise distance measured
 *     0.0048 sigma, an even harder overlap than (1).
 * What actually works, and is what this parameter ships as (memberIndices omitted, i.e. every
 * particle eligible): recompute proximityPairs FRESH from `before` on EVERY call, never reuse a
 * graph from an earlier box size. applyBoxScaleOnce already reads `before = particles()` fresh on
 * every invocation (one call per ramp increment, from applyBoxScale's own loop) -- so this is not
 * extra plumbing, only not accidentally caching the STALE graph attempt 2 tried. Verified in pure
 * Node against the real checkpoint, chaining the actual log-linear ramp step by step and recomputing
 * proximityPairs at the START of every step (exactly mirroring applyBoxScaleOnce's own call pattern):
 * 15-20 ramp increments hold the worst pairwise distance across the WHOLE 58->150 sigma chain at
 * 0.6825 sigma -- the SAME as the unperturbed baseline, i.e. no measurable degradation at all.
 * No potential, threshold, recogniser, or bond rule is touched by any of this -- only which
 * particles this ONE coordinate remap treats as sharing a rigid body, recomputed every increment. */
export async function applyBoxScaleOnce(
  rt: SoupRuntime,
  particles: () => Promise<Float32Array>,
  bonds: () => Promise<Uint32Array>,
  targetBox: [number, number, number],
  cohesion?: { cutoff: number; memberIndices?: Uint32Array },
): Promise<void> {
  // Task 'clay-surface' (2026-08-19): a box change is structurally incompatible with an immobile
  // phase, and this refuses rather than silently producing a wrong answer. An affine box scale moves
  // every particle's centre by the box ratio -- which for the mineral platelet would (a) move beads
  // that by construction cannot move (the integrator refuses to write their positions, so the CPU
  // and GPU pictures would immediately disagree) and (b) stretch the sheet's own lattice spacing away
  // from the mineral-mineral contact distance it is defined as, i.e. punch holes in a surface whose
  // whole purpose is to be impermeable. Every zero-tension area measurement in this project therefore
  // runs clay-free, which is what the bilayer gates need anyway -- see data/soup.json's clay.basis §7.
  if (rt.frozenCount > 0) {
    throw new Error(
      `изменение коробки невозможно на системе с минеральной пластиной: ${rt.frozenCount} неподвижных бидов ` +
        `(areaMove/scaleBoxTo/dryWetCycle требуют CreateSoupOpts.clay=false — см. data/soup.json clay.basis §7)`,
    )
  }
  const before = await particles()
  const bondPairs = await bonds()
  let rigidEdges = bondPairs
  if (cohesion !== undefined) {
    // eligible defaults to EVERY particle (memberIndices omitted) -- see proximityPairs' own doc
    // comment and this function's header (2nd/3rd attempts) for why a global cutoff, recomputed
    // FRESH from `before` on every call (this function always reads a fresh particles()/bonds()
    // snapshot, never a stale graph reused across ramp increments), is what is actually verified
    // safe -- restricting to a caller-named subset (e.g. just one recognised aggregate's members)
    // was tried and measured WORSE (new boundary artefacts against nearby non-member particles),
    // so `memberIndices` stays available for a future caller with a different need but is NOT what
    // this task's own campaign.ts passes.
    const eligible = new Uint8Array(rt.N)
    if (cohesion.memberIndices) {
      for (const idx of cohesion.memberIndices) eligible[idx] = 1
    } else {
      eligible.fill(1)
    }
    const prox = proximityPairs(before, rt.live.liveBox, cohesion.cutoff, eligible)
    rigidEdges = new Uint32Array(bondPairs.length + prox.length)
    rigidEdges.set(bondPairs, 0)
    rigidEdges.set(prox, bondPairs.length)
  }

  // Runtime self-check (task requirement 2: "assert intramolecular distances are unchanged by a
  // box change") -- not only the pure-function unit test on synthetic topologies
  // (tests/soup-boxcycle.test.ts), but every real application, on the real bond graph this call
  // actually sees. Cheap: O(bonds), a tiny fraction of O(N). Checked over `rigidEdges` (covalent
  // bonds, plus cohesion edges when `cohesion` is given) -- every edge that DEFINES a rigid
  // unit for this call, not just the covalent subset, since scaleMoleculesRigid preserves them all
  // identically by construction and a regression in any of them is equally real.
  const distBefore = new Float64Array(rigidEdges.length / 2)
  for (let k = 0; k < rigidEdges.length; k += 2) {
    distBefore[k / 2] = mi3Distance(before, rt.live.liveBox, rigidEdges[k], rigidEdges[k + 1])
  }
  const after = scaleMoleculesRigid(before, rigidEdges, rt.live.liveBox, targetBox)
  for (let k = 0; k < rigidEdges.length; k += 2) {
    const d = mi3Distance(after, targetBox, rigidEdges[k], rigidEdges[k + 1])
    const beforeD = distBefore[k / 2]
    if (Math.abs(d - beforeD) > 1e-3) {
      throw new Error(
        `applyBoxScale: внутримолекулярное расстояние изменилось (${beforeD.toFixed(6)} -> ${d.toFixed(6)}) ` +
          `для связи ${rigidEdges[k]}-${rigidEdges[k + 1]} при box [${rt.live.liveBox}] -> [${targetBox}] -- scaleMoleculesRigid нарушен`,
      )
    }
  }

  rt.device.queue.writeBuffer(rt.buf.posBuf, 0, after)
  rt.live.liveBox = targetBox
  // resizeSoupGrid (task 'grid-rebuild', generalising engine/src/sim.ts's own resizeGrid):
  // recomputes dims for targetBox and, if the cell count changed, destroys/reallocates
  // countsBuf/cellStartBuf/cursorBuf and rebuilds every bind group referencing them, THEN rewrites
  // gridUniform's dims+box -- replaces this call's former bare `writeBuffer(gridUniform, 16, ...)`
  // (box floats only), which is exactly what let a dims change silently desync the fixed-size grid
  // buffers from the live box (wet-dry-cycle-report.md's own measured blocker at box 46).
  resizeSoupGrid(rt.device, rt.buf, rt.grid, rt.startCounts, targetBox, rt.effectiveWalkRadius, () =>
    rebindGridDependent(rt.device, rt.pipe, rt.buf, rt.bind, rt.N),
  )

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
  // Task 'decisive-run' (2026-08-20) -- THE STALE F(x) DEFECT, NOW FIXED IN THIS SHARED PATH.
  // Until this task the block below rebuilt the grid and the Verlet list and STOPPED, leaving
  // forceBuf holding the force of the configuration that existed BEFORE the coordinate remap; but
  // soup/src/soup-integrate.ts's step() opens with kick_drift_wrap, which consumes forceBuf as
  // F(x_n), so the first half-kick after EVERY box change applied a force computed for geometry that
  // no longer exists. A velocity-Verlet integrator requires F(x_n) there -- exactly the staleness
  // soup/src/soup-relax.ts fixes for its own case in its closing block ("the positions moved, so
  // createSoup's own priming F(x0) is stale"), and the one soup/src/soup-evaporate.ts's
  // setActiveCount already fixes for its own case (its rebuild pass has always included the force).
  // The force dispatch is therefore encoded HERE, in the same pass, for every caller of a box
  // change: the dry-wet cycle's ramp, growBoxTo, scaleBoxTo and soup/src/soup-area-move.ts's
  // accepted-chain application.
  //
  // Cost: one extra force dispatch per box change (not per step) -- applyBoxScale's ramp is 6
  // increments per transition and areaMove applies at most one box change per call, so this is
  // arithmetically negligible against the rampRelaxSteps of real dynamics between them.
  //
  // What it does NOT fix, measured: tests/soup-grid-resize.test.ts's bracket-crossing test still
  // fails identically (see that test and this task's report) -- the box change it picks is simply
  // too large for its own ramp at that composition, so the staleness was a second, independent
  // defect rather than the cause of that failure. Fixing it anyway because it is wrong physics on
  // its own terms, and because the evaporating transition's own per-increment forces() call
  // (applyEvaporatingTransition, below) was until now the ONLY thing standing between this defect
  // and the cycling result -- which made the cycling result rest on a workaround instead of on a
  // correct integrator.
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  if (rt.verlet.enabled) {
    encodeVerletRebuild(rt, pass)
    encodeSoupForceList(rt, pass)
  } else {
    encodeGridRebuild(rt, pass)
    encodeSoupForce(rt, pass)
  }
  pass.end()
  rt.device.queue.submit([enc.finish()])
  await rt.device.queue.onSubmittedWorkDone()
  await assertStateFinite(rt)
  if (rt.verlet.enabled) await assertVerletSafety(rt)
}

/** Spreads one wet<->dry box change over `cfg.rampSteps` log-linear increments, each followed
 * by `cfg.rampRelaxSteps` of ordinary dynamics -- data/soup.json's dryWetCycle.basis gives the
 * physical reasoning (the same "keep each proposed change small" discipline engine/src/sim.ts's
 * areaMove already applies to its own MC trials, generalised to a forced mechanical move that has
 * no Metropolis rejection to fall back on if one increment were too large). Only ever called from
 * stepCycled, which has already established `rt.cycleCfg` is defined. */
export async function applyBoxScale(
  rt: SoupRuntime,
  particles: () => Promise<Float32Array>,
  bonds: () => Promise<Uint32Array>,
  step: (n: number) => Promise<void>,
  targetBox: [number, number, number],
  cfg: CycleSchedule & { rampSteps: number; rampRelaxSteps: number },
  cohesion?: { cutoff: number; memberIndices?: Uint32Array },
): Promise<void> {
  const fromBox = rt.live.liveBox
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
    await applyBoxScaleOnce(rt, particles, bonds, interp, cohesion)
    if (s < cfg.rampSteps && cfg.rampRelaxSteps > 0) await step(cfg.rampRelaxSteps)
  }
}

/** Public entry point for a ONE-OFF ramped box change to an ARBITRARY target box, driven by a
 * caller-chosen (rampSteps, rampRelaxSteps) rather than data/soup.json's dryWetCycle schedule --
 * dry-wet cycling (stepCycled/applyBoxScale above) always resolves its own targetBox from
 * cycleCfg's wet/dry pair, so it has no way to express "expand this live system to some other box
 * a caller names" (e.g. the box-expansion task's "grow the box well past the aggregate's own
 * diameter so a single unwrapped frame is unambiguous" -- .superpowers/sdd/
 * 2026-08-16-soup-to-vesicle/shell-pore-report.md's own closing recommendation). Reuses
 * applyBoxScale VERBATIM (same log-linear ramp, same scaleMoleculesRigid rigid-COM map, same
 * resizeSoupGrid reallocation, same per-increment bonded-distance self-check, same post-ramp
 * grid/Verlet rebuild) -- only the schedule numbers differ, never the mechanism; the dummy
 * CycleSchedule fields (periodSteps/dryFraction/cycles) are unused by applyBoxScale's own body
 * (it only ever reads cfg.rampSteps/cfg.rampRelaxSteps), so zeroing them here changes nothing it
 * touches. `rampSteps`/`rampRelaxSteps` are the CALLER's own experiment-design choice (not a model
 * parameter), so they are ordinary function arguments, not a data/soup.json field -- exactly like
 * measureBendingModulusDetailed's samples/stepsPerSample (engine/src/index.ts) already are. */
export function makeScaleBoxTo(
  rt: SoupRuntime,
  particles: () => Promise<Float32Array>,
  bonds: () => Promise<Uint32Array>,
  step: (n: number) => Promise<void>,
): (targetBox: [number, number, number], rampSteps: number, rampRelaxSteps: number, cohesion?: { cutoff: number; memberIndices?: Uint32Array }) => Promise<void> {
  return async function scaleBoxTo(targetBox, rampSteps, rampRelaxSteps, cohesion) {
    await applyBoxScale(rt, particles, bonds, step, targetBox, { periodSteps: 0, dryFraction: 0, cycles: 0, rampSteps, rampRelaxSteps }, cohesion)
  }
}

/** Box-expansion task, 2026-08-18: grows rt.live.liveBox to `targetBox` WITHOUT moving any particle
 * -- the safe alternative to scaleBoxTo's rigid-CoM rescaling for a checkpoint whose live
 * aggregate's own size is comparable to or exceeds the CURRENT box (this checkpoint's own
 * situation, per .superpowers/sdd/2026-08-16-soup-to-vesicle/shell-pore-report.md: mean head radius
 * ~29.5-29.8 sigma against box58's own half-width 29 sigma). scaleBoxTo's rigid-unit construction
 * (scaleMoleculesRigid's BFS-over-proximity-graph offset accumulation) assumes every rigid unit's
 * own diameter stays under half the box it is being scaled FROM -- true for one small covalent
 * molecule, demonstrably NOT true here for EITHER grouping actually tried against this checkpoint:
 *  1. cohesion restricted to just the ~1282-amphiphile vesicle's own ~11688 recognised members: its
 *     OWN diameter already exceeds box58/2 (the exact finding shell-pore-report.md made for
 *     unwrapAggregate()), so a BFS over its own proximity graph picks up a genuine topological
 *     wraparound -- measured directly against this checkpoint: applyBoxScaleOnce's own runtime
 *     self-check threw on the FIRST ramp increment, a pair whose distance changed from 0.957 to
 *     6.57 sigma (maxDelta over 3891 of 154776 rigid edges), purely from this effect.
 *  2. cohesion left unrestricted (every particle eligible): this checkpoint's own box-average
 *     density (0.478/sigma^3) percolates ~93178 of 93200 particles into ONE proximity-connected
 *     component at the aggregate-recognition cutoff -- which ALSO wraps box58 topologically (a
 *     system-spanning percolating cluster in a periodic box almost always does) -- measured: the
 *     SAME self-check threw on the first real GPU run, a covalent bond changing 0.957 -> 4.217
 *     sigma purely from being reached via a different (topologically inequivalent) BFS path than
 *     its own direct edge implies.
 * Neither failure is a numerical fluke or an insufficiently-fine ramp: a pure-Node, no-relax
 * chained-ramp sweep (15 to 1000 increments, checked independently before any GPU run) found the
 * SAME catastrophic worst-case pairwise overlap regardless of step count, confirming this is a
 * property of the FINAL box size a rigid-unit construction reaches, not of how many increments it
 * takes to get there.
 *
 * growBoxTo has none of these risks because it moves nothing: every particle keeps its EXACT
 * current (x,y,z) -- already inside [0, liveBox) under the CURRENT, smaller box, hence automatically
 * still inside the LARGER new box, no rewrap, no rescale, no BFS, and "did any distance change" is
 * true by construction for every pair, not just checked after the fact. Only liveBox and the
 * neighbour grid (resizeSoupGrid -- the SAME verified reallocate-and-rebind machinery scaleBoxTo/
 * dry-wet cycling already use) change. Physically this is "instantaneously remove the confining
 * walls": the densely-packed material stays exactly as it was, now sitting in one region of a much
 * larger periodic box, and ORDINARY Langevin dynamics (not a synthetic coordinate remap) is what
 * redistributes material into the newly available volume during the relaxation that follows -- a
 * more honest model of instantaneous dilution than an affine rescale for a system this far from
 * spatially homogeneous. Ramped over `rampSteps` box-size increments (each followed by
 * `rampRelaxSteps` of ordinary dynamics) purely for gradualness/observability, matching this task's
 * own "ramp it" instruction -- unlike scaleBoxTo's ramp, no increment size here carries any overlap
 * risk at all, since no position ever moves; a single jump would be equally safe. */
export function makeGrowBoxTo(rt: SoupRuntime, step: (n: number) => Promise<void>): (targetBox: [number, number, number], rampSteps: number, rampRelaxSteps: number) => Promise<void> {
  return async function growBoxTo(targetBox, rampSteps, rampRelaxSteps) {
    const fromBox = rt.live.liveBox
    const lnRatio: [number, number, number] = [
      Math.log(targetBox[0] / fromBox[0]),
      Math.log(targetBox[1] / fromBox[1]),
      Math.log(targetBox[2] / fromBox[2]),
    ]
    for (let s = 1; s <= rampSteps; s++) {
      const interp: [number, number, number] =
        s === rampSteps
          ? targetBox
          : [
              fromBox[0] * Math.exp((lnRatio[0] * s) / rampSteps),
              fromBox[1] * Math.exp((lnRatio[1] * s) / rampSteps),
              fromBox[2] * Math.exp((lnRatio[2] * s) / rampSteps),
            ]
      rt.live.liveBox = interp
      resizeSoupGrid(rt.device, rt.buf, rt.grid, rt.startCounts, interp, rt.effectiveWalkRadius, () =>
        rebindGridDependent(rt.device, rt.pipe, rt.buf, rt.bind, rt.N),
      )
      const enc = rt.device.createCommandEncoder()
      const pass = enc.beginComputePass()
      if (rt.verlet.enabled) {
        encodeVerletRebuild(rt, pass)
      } else {
        encodeGridRebuild(rt, pass)
      }
      pass.end()
      rt.device.queue.submit([enc.finish()])
      await rt.device.queue.onSubmittedWorkDone()
      await assertStateFinite(rt)
      if (rt.verlet.enabled) await assertVerletSafety(rt)
      if (s < rampSteps && rampRelaxSteps > 0) await step(rampRelaxSteps)
    }
  }
}

/** Task 'evaporation' (2026-08-20): one wet<->dry transition WITH solvent removal/return, walked as
 * a ladder of (active particle count, box) pairs -- soup/src/soup-evaporate.ts owns both ladders and
 * the arithmetic behind them; this function is only the driver that applies them to a live system,
 * reusing applyBoxScaleOnce VERBATIM for every box change (same rigid-CoM map, same per-increment
 * bonded-distance self-check, same resizeSoupGrid reallocation, same finiteness/Verlet guards).
 *
 * ORDER WITHIN AN INCREMENT, and why each way round:
 *  - drying: REMOVE first, then CONTRACT. Removal can never create an overlap (it only takes
 *    particles away), and doing it before the contraction means the organics are never compressed
 *    against solvent that is about to leave anyway. The ORGANIC density -- the one the chemistry
 *    responds to -- is therefore monotone non-decreasing across the whole transition: removing
 *    solvent at fixed box leaves it exactly unchanged, and every contraction raises it.
 *  - rehydrating: EXPAND first, then INSERT. Expansion is what creates the room; there is no room to
 *    insert into before it (dryWetCycle.basis item 5 gives the measured reason), and the whole pool
 *    returns on the final increment, at the wet box.
 * `rampRelaxSteps` of ordinary dynamics run between increments in BOTH directions -- an affine
 * contraction of a dense liquid genuinely does push contacting pairs into each other's cores, and an
 * affine expansion can too for two multi-bead molecules whose centres are closer than their own bead
 * offsets, so neither direction is safe to jump. */
export async function applyEvaporatingTransition(
  rt: SoupRuntime,
  plan: EvaporationPlan,
  phase: 'wet' | 'dry',
  particles: () => Promise<Float32Array>,
  bonds: () => Promise<Uint32Array>,
  forces: () => Promise<Float32Array>,
  centerLinks: () => Promise<Uint32Array>,
  step: (n: number) => Promise<void>,
  cfg: CycleSchedule & { rampSteps: number; rampRelaxSteps: number },
  targetDryDensity: number,
  seed: number,
): Promise<RehydrationReport[]> {
  const ladder = phase === 'dry' ? evaporationLadder(plan, targetDryDensity) : rehydrationLadder(plan)
  const reports: RehydrationReport[] = []
  const d0 = (rt.soup.coldStartRelax?.maxDisplacementSigma ?? 0.1) * rt.p.sigma
  for (let s = 0; s < ladder.length; s++) {
    const rung = ladder[s]
    if (phase === 'dry') {
      await evaporateSolventTo(rt, plan, rung.solvent, rt.live.liveBox)
      await applyBoxScaleOnce(rt, particles, bonds, rung.box)
    } else {
      await applyBoxScaleOnce(rt, particles, bonds, rung.box)
      if (rung.solvent > rt.N - plan.solventBlockStart) {
        reports.push(await rehydrateSolventTo(rt, plan, rung.solvent, rung.box, particles, forces, seed))
      }
    }
    // THE forces() CALL BELOW IS LOAD-BEARING, NOT A DIAGNOSTIC, and that was found the hard way.
    // applyBoxScaleOnce rebuilds the neighbour grid and the Verlet list after a box change but does
    // NOT recompute the force -- harmless at the pre-existing path's 0.53 % increments, and not
    // harmless at this path's 3.18 % ones. soup/src/soup-integrate.ts's step() opens with
    // kick_drift_wrap, which consumes forceBuf as F(x_n); after a box change that buffer still holds
    // the force of the PREVIOUS configuration, so the first half-kick applies a force from geometry
    // that no longer exists (dv = F*dt/2, and F can be ~1e4 right after a contraction, i.e. a shove
    // of order one sigma in a single step -- straight into a neighbour's core, whence the correct
    // force is astronomical). This is exactly the staleness soup/src/soup-relax.ts fixes for itself
    // ("the positions moved, so createSoup's own priming F(x0) is stale"). Two runs of this
    // transition WITHOUT this call diverged inside the ramp's own relaxation steps (assertVerletSafety
    // threw drifts of 2.98e7 and 2.13e14 sigma with the state still finite); runs with it have not.
    // readback.forces() rebuilds grid+list+force and reads the result back, so it both refreshes
    // F(x_n) and supplies the spike measurement below -- one round trip, both jobs.
    //
    // MEASURED, not assumed: an affine contraction of material that has already aggregated pushes
    // contacting pairs of DIFFERENT molecules into each other's cores -- the amplification is
    // |dCOM|/d for a pair whose centre separation is much larger than its bead-bead separation, so a
    // 3.2 % box step can be a ~10 % approach for two interpenetrating multi-bead molecules. The first
    // run of this transition at box 30 blew up on exactly that in the SECOND cycle (the first cycle
    // had nothing aggregated yet to squeeze): assertVerletSafety threw a drift of 2.98e7 sigma inside
    // the ramp's own relaxation steps. The cure is the same minimiser the cold start and the solvent
    // insertion use, run only when the increment actually left a force spike the integrator cannot
    // absorb, and only for as many iterations as the spike needs -- so an increment that was already
    // clean pays one force readback and nothing else.
    // The adsorption tether is the ONE distance a box change is allowed to alter, and FENE's force
    // changes SIGN past r_inf -- see desorbOverstretchedTethers' own doc comment for the two measured
    // divergences that led here. Threshold derived from the ramp's own per-increment factor.
    const lambda = Math.exp(Math.abs(Math.log(plan.wetBox[0] / plan.dryBox[0])) / ladder.length)
    const tether = await desorbOverstretchedTethers(rt, particles, centerLinks, rung.box, rt.p.fene.rInf / lambda)
    const spike = maxAbsForce(await forces(), rt.N)
    let relaxed = 0
    // The trigger is DERIVED, and the FIRST derivation of it was measurably too permissive. With
    // m = 1 in these units a force F displaces a bead by F*dt^2 in one step, and the first version
    // required only that this stay under one sigma (F < sigma/dt^2 = 1e4) -- which never fired, and
    // the ramp diverged anyway. The real condition for a Verlet integrator is that the force must not
    // change appreciably ACROSS that displacement, and for a WCA r^-12 core d(ln F)/d(ln r) = -13, so
    // F changes e-fold over dr = r/13. Requiring F*dt^2 <= sigma/13 gives F <= sigma/(13*dt^2) = 769
    // -- BELOW the 0.9e3-2.7e3 spikes actually measured, which is exactly why the run kept diverging.
    if (spike > rt.p.sigma / (13 * rt.p.integrator.dt * rt.p.integrator.dt)) {
      relaxed = plan.relaxIterations
      await relaxIterations(rt, relaxed, d0)
    }
    const after = relaxed > 0 ? maxAbsForce(await forces(), rt.N) : spike
    console.log(
      `[evaporation] ${phase === 'dry' ? 'испарение' : 'регидратация'} приращение=${s + 1}/${ladder.length} ` +
        `box=${rung.box[0].toFixed(4)} растворителя=${rt.N - plan.solventBlockStart} N=${rt.N} ` +
        `max|F|=${spike.toExponential(3)} итераций_минимизации=${relaxed} max|F|_после=${after.toExponential(3)} ` +
        `перетянутых_привязок=${tether.cleared} самая_длинная=${tether.longest.toFixed(4)} порог=${tether.maxLength.toFixed(4)}`,
    )
    if (s < ladder.length - 1 && cfg.rampRelaxSteps > 0) await step(cfg.rampRelaxSteps)
  }
  return reports
}

/** max over all particles and components of |F| -- a scalar computed from a plain force readback, the
 * same reduction soup/src/soup-relax.ts's own forceStats makes; never transfers the array anywhere. */
function maxAbsForce(f: Float32Array, n: number): number {
  let m = 0
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.abs(f[i * 4 + c])
      if (!Number.isFinite(v)) return Number.POSITIVE_INFINITY
      if (v > m) m = v
    }
  }
  return m
}

export function makeStepCycled(
  rt: SoupRuntime,
  particles: () => Promise<Float32Array>,
  bonds: () => Promise<Uint32Array>,
  step: (n: number) => Promise<void>,
  dryWetCycleSchedule: { rampSteps: number; rampRelaxSteps: number },
  forces: () => Promise<Float32Array>,
  centerLinks: () => Promise<Uint32Array>,
  seed: number,
): (n: number) => Promise<void> {
  return async function stepCycled(n: number): Promise<void> {
    if (!rt.cycleCfg) {
      await step(n)
      return
    }
    const cfg = { ...rt.cycleCfg, rampSteps: dryWetCycleSchedule.rampSteps, rampRelaxSteps: dryWetCycleSchedule.rampRelaxSteps }
    let remaining = n
    while (remaining > 0) {
      const next = nextCycleTransition(rt.live.globalStep, cfg)
      const advance = next === Infinity ? remaining : Math.min(remaining, next - rt.live.globalStep)
      if (advance > 0) {
        await step(advance)
        remaining -= advance
      }
      if (next !== Infinity && rt.live.globalStep === next) {
        const { phase, cycleIndex } = cyclePhaseAt(rt.live.globalStep, cfg)
        if (rt.evap) {
          // Task 'evaporation': the SAME transition point, but the solvent actually leaves/returns.
          // rt.evap is undefined for every system that did not ask for it, so the branch below is the
          // untouched pre-task path.
          const evapCfg = { ...cfg, rampSteps: rt.evap.rampSteps }
          const reports = await applyEvaporatingTransition(rt, rt.evap, phase, particles, bonds, forces, centerLinks, step, evapCfg, rt.soup.dryWetCycle.targetDryDensity, seed)
          for (const r of reports) {
            console.log(
              `[evaporation] регидратация: вставлено=${r.inserted} ниже_порога=${r.shortOfFloor} ` +
                `минимальное_расстояние=${r.minAchieved.toFixed(4)} итераций_минимизации=${r.relaxIterations} ` +
                `max|F| ${r.maxForceBefore.toExponential(4)} -> ${r.maxForceAfter.toExponential(4)} ` +
                `смещение_прежних rms=${r.preexistingRmsDisplacement.toFixed(4)} max=${r.preexistingMaxDisplacement.toFixed(4)} граница=${r.displacementBound.toFixed(4)}`,
            )
          }
        } else {
          await applyBoxScale(rt, particles, bonds, step, phase === 'dry' ? rt.dryBox : rt.box, cfg)
        }
        rt.live.cyclePhase = phase
        rt.live.cycleIndex = cycleIndex
      } else if (advance === 0 && remaining > 0) {
        // Defensive: advance===0 with steps still remaining and no transition to apply would spin
        // forever -- cannot happen given nextCycleTransition's own contract (it only ever returns a
        // step strictly greater than the current one, or Infinity), kept as a hard stop rather than
        // a silent infinite loop if that contract is ever violated by a future edit.
        throw new Error('stepCycled: расписание циклов зациклилось -- nextCycleTransition вернул текущий globalStep')
      }
    }
  }
}
