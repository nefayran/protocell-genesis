// Task 3: stage detection -- EVIDENCE, not narrative. detectStage never asserts a stage on its own
// say-so: it always hands back the measurements the plan names alongside the label, so a caller can
// look at the numbers and disagree with the label. The thresholds that turn those numbers into a
// label live in data/soup.json's `stageThresholds` (with their basis), not here -- this file reads
// them, it does not declare them, matching the "no numeric constants in soup/src" rule
// tests/params.test.ts's literal scanner enforces over this directory.
//
// Reuses rather than reimplements: findAmphiphiles (soup/src/amphiphile.ts), largestClusterFraction
// (engine/src/aggregate.ts, Task 6's cell-list + union-find, kept as a diagnostic -- see below),
// enclosedVolumeFromPositions (engine/src/closure.ts, Task 8's verified flood-fill closure detector),
// and computeHeadPeaks/memberIndicesOf/positionsFor (moved to soup/src/aggregates.ts, re-exported
// here -- see that file's own header for why the move was needed).
//
// Task 3 flagged a deviation here: SoupSystem (soup/src/sim.ts, Task 2) had no `box` field --
// unlike the membrane engine's System, which does -- so detectStage took box as an explicit second
// argument. The reconciliation: SoupSystem carries its own `box` field (set once in createSoup,
// never mutated), and detectStage reads `sys.box` instead of taking it as a parameter.
//
// Task 3b (task-3b-report.md) fixed two pilot-exposed defects in stageFromEvidence/computeHeadPeaks
// (the ladder-ordering fix -- a later stage structurally requires every earlier stage's own
// condition too -- and computeHeadPeaks' two Poisson-noise guards). Both are still in force below;
// computeHeadPeaks itself is UNCHANGED (moved, not rewritten -- soup/src/aggregates.ts's own header).
//
// task-3c/per-aggregate-report fixed two FURTHER defects a real dilute run (box 50sigma, 14200
// particles, 200000 steps -- ~15 visibly discrete, round aggregates on screen, textbook micelles)
// exposed in what the OLD ladder measured, not in the physics:
//  1. `largestAggregateFraction >= aggregationLow` gated `micelles` on ONE DOMINANT aggregate -- the
//     opposite of a micellar state, where matter is deliberately split among MANY small aggregates.
//     The dilute run above reported `amphiphiles` (largestAggregateFraction=0.1049, below 0.3)
//     despite amphiphileFraction=0.3377, the highest of any run so far.
//  2. headPeaks is a BOX-WIDE z-density profile, meaningful only for a planar bilayer spanning the
//     periodic box -- noise for discrete micelles/vesicles, which have no global "up"/"down" -- yet
//     it gated the bilayer stage regardless.
// Fixed by redefining the ladder on PER-AGGREGATE quantities (soup/src/aggregates.ts's
// analyzeAggregates(): cluster amphiphile members into every connected aggregate, not just the
// largest, then measure each of the largest few's own shape/head-layering/cavity). The box-wide
// largestAggregateFraction/headPeaks/enclosedVolume fields stay on StageEvidence as DIAGNOSTICS
// (still computed, still shown), but no longer gate micelles/bilayer -- only the new per-aggregate
// fields (qualifyingAggregateCount, amphiphileShareInQualifying, hasLamellarAggregate,
// hasVesicleAggregate) do, with vesicle still gated by the SAME existing enclosedVolume minimum
// (kept exactly as it was, per the task's own instruction -- loadStageThresholds()'s cross-check
// against data/literature.json's closure.target.min is unchanged).

import rawLiterature from '../../data/literature.json'
import rawSoup from '../../data/soup.json'
import {
  analyzeAggregates,
  computeHeadPeaks,
  derivedMinAmphiphilesPerAggregate,
  emptyAggregateAnalysis,
  memberIndicesOf,
  positionsFor,
  HEAD_PEAKS_UNAVAILABLE,
  type AggregateAnalysis,
  type AggregateThresholds,
} from './aggregates'
import { largestClusterFraction } from '../../engine/src/aggregate'
import { enclosedVolumeFromPositions } from '../../engine/src/closure'
import { loadParams, wcaCutoff } from '../../engine/src/params'
import { findAmphiphiles } from './amphiphile'
import { loadSoup } from './rules'
import type { SoupSystem } from './sim'

export type Stage = 'monomers' | 'amphiphiles' | 'micelles' | 'bilayer' | 'vesicle'

// Re-exported so every existing caller (tests/soup-amphiphile.test.ts, tests/run-ui.test.ts,
// viewer/run.ts, engine/src/index.ts) keeps importing these from soup/src/stages -- only their
// IMPLEMENTATION moved (verbatim) to soup/src/aggregates.ts, see that file's own header for why.
export { computeHeadPeaks, emptyAggregateAnalysis, memberIndicesOf, positionsFor, HEAD_PEAKS_UNAVAILABLE }
export type { AggregateAnalysis, AggregateShape } from './aggregates'

export interface StageEvidence {
  /** Fraction of carbon particles bound into a recognised amphiphile chain. */
  amphiphileFraction: number
  /** DIAGNOSTIC only (task-3c): fraction of amphiphile-member particles belonging to the single
   * largest connected aggregate among them. No longer gates any stage -- see this file's header for
   * why (a micellar state is defined by matter split among MANY aggregates, not concentrated in
   * one), kept only because it remains a useful number to look at. 0 when there are no amphiphiles. */
  largestAggregateFraction: number
  /** DIAGNOSTIC only (task-3c): the BOX-WIDE head-density-profile peak count (0/1/2/'unavailable',
   * see computeHeadPeaks' own doc comment in soup/src/aggregates.ts) -- meaningful only for a planar
   * bilayer spanning the periodic box, noise for discrete micelles/vesicles. No longer gates the
   * bilayer stage -- see aggregateAnalysis.hasLamellarAggregate for the per-aggregate replacement. */
  headPeaks: number | typeof HEAD_PEAKS_UNAVAILABLE
  /** DIAGNOSTIC only (task-3c): box-wide flood-fill cavity volume over every amphiphile-member
   * particle (recentred on the dominant cluster) -- superseded by aggregateAnalysis.aggregates[k]
   * .cavityVolume (flooded within each aggregate's OWN local bounding region) for stage-gating; kept
   * as a whole-box cross-check number. 0 when nothing is closed. */
  enclosedVolume: number
  /** The full per-aggregate breakdown (soup/src/aggregates.ts's analyzeAggregates()) this sample's
   * micelles/bilayer/vesicle verdict is actually computed from -- aggregate count, size histogram,
   * and full shape/head-shell/cavity detail for the largest few, so a caller (the viewer, a test)
   * can see WHY a stage did or did not fire, not just the label. */
  aggregateAnalysis: AggregateAnalysis
}

export interface StageThresholds extends AggregateThresholds {
  amphiphileFraction: number
  enclosedVolume: number
  closureCell: number
  closureRadius: number
  headDensityBins: number
  /** Minimum head particles per bin, ON AVERAGE, before the BOX-WIDE headPeaks diagnostic is
   * trusted at all -- defect 2 (task-3b), see HEAD_PEAKS_UNAVAILABLE's own doc comment. */
  minHeadsPerBin: number
  /** Plausible bilayer-thickness band (peak-to-peak separation, same reduced sigma units as the
   * box) a two-peak reading must fall inside to be trusted -- shared between the box-wide diagnostic
   * and the per-aggregate radial/transverse profiles (soup/src/aggregates.ts), since the underlying
   * physical band (a plausible wall thickness) is the same regardless of whether the wall is flat or
   * curved. */
  headPeakSeparationMin: number
  headPeakSeparationMax: number
  basis: string
}

/** The vesicle-closure gate's own physically-grounded minimum (data/literature.json, id "closure",
 * target.min) -- see that file's `conditions` field for the derivation: the smallest sphere whose
 * radius exceeds this project's own measured bilayer thickness. Read here (not re-derived) so
 * loadStageThresholds() below can assert the ladder's own vesicle threshold is the SAME number,
 * rather than trusting two independently hand-maintained copies to stay equal. */
function literatureClosureMin(): number {
  const gates = (rawLiterature as { gates: Array<{ id: string; target: { min?: number } }> }).gates
  const gate = gates.find((g) => g.id === 'closure')
  if (!gate || gate.target.min === undefined) {
    throw new Error('data/literature.json: отсутствует closure.target.min')
  }
  return gate.target.min
}

export function loadStageThresholds(): StageThresholds {
  const t = (rawSoup as unknown as { stageThresholds?: StageThresholds }).stageThresholds
  if (!t) throw new Error('data/soup.json: отсутствует поле stageThresholds')
  const literatureMin = literatureClosureMin()
  if (t.enclosedVolume !== literatureMin) {
    // The whole point of tying these two numbers together (so the ladder and the gate agree) is
    // that an editor changing ONE of them without the other must fail loudly, not quietly let a
    // run's stage label and its own literature-gate verdict disagree about what counts as a vesicle.
    throw new Error(
      `data/soup.json stageThresholds.enclosedVolume (${t.enclosedVolume}) не совпадает с ` +
        `data/literature.json closure.target.min (${literatureMin}) -- лестница стадий и ворота ` +
        `должны использовать один и тот же физически обоснованный минимум объёма полости`,
    )
  }
  const derivedMin = derivedMinAmphiphilesPerAggregate()
  if (t.minAmphiphilesPerAggregate !== derivedMin) {
    // Same reasoning, task-3c's own threshold: minAmphiphilesPerAggregate is a FORMULA (sphere of
    // one tail length's own radius, covered at the literature area-per-lipid midpoint -- see
    // derivedMinAmphiphilesPerAggregate's own doc comment), not a hand-picked number, so the value
    // written in data/soup.json must always equal what that formula gives -- an editor changing
    // data/params.json's sigma/carbon radiusSigma, or data/literature.json's area-per-lipid range,
    // without recomputing this threshold must fail loudly instead of silently drifting.
    throw new Error(
      `data/soup.json stageThresholds.minAmphiphilesPerAggregate (${t.minAmphiphilesPerAggregate}) не ` +
        `совпадает с производной формулой (${derivedMin}) -- см. derivedMinAmphiphilesPerAggregate() ` +
        `в soup/src/aggregates.ts`,
    )
  }
  return t
}

/** Defect 1's fix (task-3c redefinition): the stage ladder, now decided on PER-AGGREGATE quantities
 * (evidence.aggregateAnalysis) rather than one box-wide largestAggregateFraction/headPeaks pair --
 * see this file's header for why those two numbers structurally cannot tell a micelle from a
 * bilayer/vesicle. `amphiphileOk` (task-3b's own defect-1 fix) is still the ONE blanket prerequisite
 * every later stage requires -- no stage above `monomers` fires without real recognised amphiphiles.
 *
 * micelles/bilayer/vesicle themselves are DELIBERATELY NOT chained to each other beyond that shared
 * prerequisite, unlike task-3b's own ladder (where bilayer's threshold was simply a HIGHER bar on
 * the SAME scalar micelles used, so requiring micelles' condition too was free). Here they describe
 * mutually exclusive shapes of "the dominant structure": micelles means matter split among MANY
 * small aggregates (qualifyingAggregateCount>=2 by construction); bilayer means one FLAT aggregate
 * (isotropic-in-two-axes, thin in the third); vesicle means one CURVED, closed shell (isotropic in
 * all three axes, unlike a flat bilayer patch -- a real spherical shell's own gyration tensor has no
 * preferred flat direction). A system that coarsens from micelles into a single dominant aggregate
 * (the physically expected trajectory, engine/src/aggregate.ts's own header) genuinely LOSES the
 * multiplicity micelles requires at the very moment it gains the shape bilayer/vesicle require --
 * chaining bilayerOk/vesicleOk through micelleOk would make bilayer/vesicle structurally unreachable
 * for exactly the trajectory this project's own physics is meant to produce. Each rung's own
 * condition is checked in priority order (vesicle > bilayer > micelles > amphiphiles), so a snapshot
 * whose aggregate population happens to satisfy more than one rung's own shape condition at once
 * (a realistic transitional mix) reports the most advanced one, same as the original ladder did.
 * Pure function of the evidence + thresholds, no GPU, unit-testable on synthetic StageEvidence
 * objects. */
export function stageFromEvidence(evidence: StageEvidence, thresholds: StageThresholds): Stage {
  const analysis = evidence.aggregateAnalysis
  const amphiphileOk = evidence.amphiphileFraction >= thresholds.amphiphileFraction
  const micelleOk =
    amphiphileOk &&
    analysis.qualifyingAggregateCount >= thresholds.minMicelleAggregates &&
    analysis.amphiphileShareInQualifying >= thresholds.minAmphiphileShareInAggregates
  const bilayerOk = amphiphileOk && analysis.hasLamellarAggregate
  const vesicleOk = amphiphileOk && analysis.hasVesicleAggregate

  if (vesicleOk) return 'vesicle'
  if (bilayerOk) return 'bilayer'
  if (micelleOk) return 'micelles'
  if (amphiphileOk) return 'amphiphiles'
  return 'monomers'
}

/** Measures the stage-deciding numbers for one snapshot of `sys`, then maps them to a stage label
 * via data/soup.json's stageThresholds. See this file's header for how `box` is obtained (`sys.box`,
 * not a parameter). */
export async function detectStage(sys: SoupSystem): Promise<{ stage: Stage; evidence: StageEvidence }> {
  const soup = loadSoup()
  const thresholds = loadStageThresholds()
  const box = sys.box
  const particles = await sys.particles()
  const bonds = await sys.bonds()
  const n = particles.length / 4

  const amphiphiles = findAmphiphiles(particles, bonds, soup.monomers)
  const memberIdx = memberIndicesOf(amphiphiles)

  // --- amphiphileFraction: carbon particles bound into a recognised chain / carbon particles total
  const carbonKinds = new Set(soup.monomers.map((m, i) => (m.kind === 'carbon' ? i : -1)).filter((i) => i >= 0))
  let totalCarbon = 0
  for (let i = 0; i < n; i++) {
    if (carbonKinds.has(Math.round(particles[i * 4 + 3]))) totalCarbon++
  }
  const carbonInAmphiphiles = amphiphiles.reduce((sum, a) => sum + a.length, 0)
  const amphiphileFraction = totalCarbon > 0 ? carbonInAmphiphiles / totalCarbon : 0

  // Cutoff for "connected amphiphile aggregate", derived from this soup's own species sizes
  // (data/soup.json radiusSigma) -- the SAME cutoff largestAggregateFraction below and
  // analyzeAggregates' own clustering use, mirroring soup/src/sim.ts's own cellSize derivation, not
  // a new constant.
  const p = loadParams()
  const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc

  // --- largestAggregateFraction: DIAGNOSTIC only (task-3c) -- see StageEvidence's own doc comment.
  let largestAggregateFraction = 0
  if (memberIdx.size > 0) {
    const out = particles.slice()
    for (let i = 0; i < n; i++) out[i * 4 + 3] = memberIdx.has(i) ? 1 : 0
    // `true` = z periodic, the soup's own boundary condition -- see analyzeAggregates' call.
    largestAggregateFraction = largestClusterFraction(out, box, cutoff, true)
  }

  // --- headPeaks: DIAGNOSTIC only (task-3c) -- box-wide, see StageEvidence's own doc comment.
  const headPeaks = computeHeadPeaks(particles, box, soup.monomers, thresholds)

  // --- enclosedVolume: DIAGNOSTIC only (task-3c) -- box-wide flood over every amphiphile-member
  // particle, recentred on the dominant cluster internally (engine/src/closure.ts). Superseded by
  // aggregateAnalysis.aggregates[k].cavityVolume (per-aggregate local flood) for stage-gating.
  let enclosedVolume = 0
  if (memberIdx.size > 0) {
    const memberPositions = positionsFor(particles, memberIdx)
    enclosedVolume = enclosedVolumeFromPositions(memberPositions, box, {
      cell: thresholds.closureCell,
      radius: thresholds.closureRadius,
    })
  }

  // --- water particle indices (task 'explicit-water', 2026-08-18): every particle whose kind is
  // data/soup.json's solvent.waterId, if that species is present at all. A run with data/soup.json's
  // `start.W` at 0 (or that never mentions W) gives an EMPTY array here, and analyzeAggregates()
  // treats an empty array the same as `undefined` -- see its own doc comment -- so a water-free run
  // keeps the vacuum-cavity vesicle check exactly as it always was, no special-casing needed here.
  const waterKind = soup.monomers.findIndex((m) => m.id === soup.solvent.waterId)
  const waterIdx: number[] = []
  if (waterKind >= 0) {
    for (let i = 0; i < n; i++) {
      if (Math.round(particles[i * 4 + 3]) === waterKind) waterIdx.push(i)
    }
  }

  // --- the per-aggregate breakdown the ladder is actually decided from (task-3c).
  const aggregateAnalysis = analyzeAggregates(particles, box, soup.monomers, amphiphiles, memberIdx, cutoff, thresholds, waterIdx)

  const evidence: StageEvidence = {
    amphiphileFraction,
    largestAggregateFraction,
    headPeaks,
    enclosedVolume,
    aggregateAnalysis,
  }
  const stage = stageFromEvidence(evidence, thresholds)

  return { stage, evidence }
}
