// Task 3: stage detection -- EVIDENCE, not narrative. detectStage never asserts a stage on its own
// say-so: it always hands back the four measurements the plan names (amphiphileFraction,
// largestAggregateFraction, headPeaks, enclosedVolume) alongside the label, so a caller can look at
// the numbers and disagree with the label. The thresholds that turn those numbers into a label live
// in data/soup.json's `stageThresholds` (with their basis), not here -- this file reads them, it
// does not declare them, matching the "no numeric constants in soup/src" rule
// tests/params.test.ts's literal scanner enforces over this directory.
//
// Reuses rather than reimplements: findAmphiphiles (this task, soup/src/amphiphile.ts),
// largestClusterFraction (engine/src/aggregate.ts, Task 6's cell-list + union-find),
// densityProfileZ/bilayerThickness (engine/src/metrics.ts, Task 5's sub-bin-peak z-histogram), and
// enclosedVolumeFromPositions (engine/src/closure.ts, Task 8's verified flood-fill closure
// detector, which recentres on the dominant cluster internally). None of those four functions know
// about soup/src/rules.ts's monomer kinds -- they were built for the membrane engine's own
// head(w=0)/tail(w!=0) encoding -- so remapFlag() below builds a throwaway copy of `particles`
// with the 4th (kind-index) component replaced by whichever binary flag the function being called
// expects, particle by particle. This is the same "adapt the encoding at the call site" pattern
// engine/src/index.ts's own facades already use (e.g. largestClusterFractionOf's cutoff derived
// from the caller's own species sizes) -- it is not a new union-find or a new histogram.
//
// Task 3 flagged a deviation here: SoupSystem (soup/src/sim.ts, Task 2) had no `box` field --
// unlike the membrane engine's System, which does -- so detectStage took box as an explicit second
// argument. Task 4 (soup/src/sim.ts's runUntil) needs to call detectStage once per sample without
// the caller re-threading box through every call, and the plan's own Task 4 test calls
// `api.stageOf(sys)` with a single argument -- so the reconciliation chosen here is: SoupSystem now
// carries its own `box` field (set once in createSoup, never mutated), and detectStage reads
// `sys.box` instead of taking it as a parameter. Every measurement below (clustering, the
// z-density profile, the closure flood) still needs the box; it now gets it from `sys.box`.
//
// Task 3b (task-3b-report.md) fixed two pilot-exposed defects, both isolated into pure, GPU-free
// functions below so they are unit-testable on synthetic particle/evidence data without a browser:
//  - stageFromEvidence(): the pilot's 300k-step trace never showed the `amphiphiles` label even
//    once, because the OLD stage ladder tested each stage's OWN condition independently
//    (`largestAggregateFraction >= aggregationLow` alone decided `micelles`, with no requirement
//    that amphiphiles be present at all) -- so a handful of non-amphiphile particles clustering
//    together could satisfy `micelles`/`bilayer` on aggregation alone. Fixed into a genuine ladder:
//    each stage's condition now REQUIRES the previous stage's condition too (micelles = amphiphile
//    condition AND its own; bilayer = micelles' condition AND its own; vesicle = bilayer's condition
//    AND its own), so a later stage is structurally impossible without the earlier one.
//  - computeHeadPeaks(): `headPeaks` reported 2 at step 0 of the pilot -- zero bonds, zero
//    structure, 267 head particles spread over 60 z-bins (~4.45/bin) -- pure Poisson binning noise
//    misread as a bilayer signal. Guarded two ways: (a) the profile's own particle-per-bin average
//    must clear `minHeadsPerBin` before ANY peak count is trusted (returns HEAD_PEAKS_UNAVAILABLE
//    otherwise, not a number); (b) even with enough particles, two maxima only count as `2` if their
//    separation falls inside a plausible bilayer-thickness band (`headPeakSeparationMin/Max`) --
//    two local maxima that just happen to sit far apart are not a bilayer.

import rawLiterature from '../../data/literature.json'
import rawSoup from '../../data/soup.json'
import { largestClusterFraction } from '../../engine/src/aggregate'
import { enclosedVolumeFromPositions } from '../../engine/src/closure'
import { bilayerThickness, densityProfileZ, dropEscapedZ } from '../../engine/src/metrics'
import { loadParams, wcaCutoff } from '../../engine/src/params'
import { findAmphiphiles, type Amphiphile } from './amphiphile'
import { loadSoup, type Monomer } from './rules'
import type { SoupSystem } from './sim'

export type Stage = 'monomers' | 'amphiphiles' | 'micelles' | 'bilayer' | 'vesicle'

/** Sentinel returned by computeHeadPeaks() when the head-density profile does not carry enough
 * particles to trust ANY peak count read off it (defect 2, task-3b-report.md): 267 heads spread
 * over 60 z-bins (~4.45/bin) reported two peaks at step 0 of the pilot, before a single bond
 * existed -- pure Poisson binning noise, not structure. A caller that treats that "2" as a real
 * bilayer signal is trusting a coin flip; 'unavailable' says the instrument has nothing to say
 * here rather than making something up. */
export const HEAD_PEAKS_UNAVAILABLE = 'unavailable' as const

export interface StageEvidence {
  /** Fraction of carbon particles bound into a recognised amphiphile chain. */
  amphiphileFraction: number
  /** Fraction of amphiphile-member particles (heads + their chains) that belong to the single
   * largest connected aggregate among them -- 0 when there are no amphiphiles yet. */
  largestAggregateFraction: number
  /** 0 (no polar particles at all), 1 (one head-density peak, or two peaks whose separation is
   * not a plausible bilayer thickness -- not yet a bilayer, the normal case for most of a soup
   * trajectory), 2 (two peaks separated by a plausible bilayer thickness -- a trusted bilayer
   * candidate), or 'unavailable' (too few head particles per bin, on average, to trust ANY peak
   * count read off this profile -- see HEAD_PEAKS_UNAVAILABLE). */
  headPeaks: number | typeof HEAD_PEAKS_UNAVAILABLE
  /** Volume of any flood-unreachable cavity, in the same reduced units as the box. 0 when nothing
   * is closed. */
  enclosedVolume: number
}

export interface StageThresholds {
  amphiphileFraction: number
  aggregationLow: number
  aggregationHigh: number
  enclosedVolume: number
  closureCell: number
  closureRadius: number
  headDensityBins: number
  /** Minimum head particles per bin, ON AVERAGE (totalHeadCount / headDensityBins -- a property of
   * the run's own composition, not of any one snapshot), before headPeaks is trusted at all --
   * defect 2's guard, see HEAD_PEAKS_UNAVAILABLE. */
  minHeadsPerBin: number
  /** Plausible bilayer-thickness band (peak-to-peak separation, same reduced sigma units as the
   * box) a two-peak reading must fall inside to be trusted as headPeaks=2 rather than downgraded
   * to 1 -- defect 2's second guard: two local maxima that are merely far apart (e.g. one near each
   * box face on an otherwise unstructured profile) are not a bilayer just because bilayerPeaks()
   * found two maxima. */
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
    // The whole point of tying these two numbers together (task: "so the ladder and the gate
    // agree") is that an editor changing ONE of them without the other must fail loudly, not
    // quietly let a run's stage label and its own literature-gate verdict disagree about what
    // counts as a vesicle -- exactly the kind of gap that let enclosedVolume=1.1250 (9 flood-fill
    // cells) read as "vesicle" under the old min:1 threshold while the gate's own written
    // conditions never meant that as a physical criterion.
    throw new Error(
      `data/soup.json stageThresholds.enclosedVolume (${t.enclosedVolume}) не совпадает с ` +
        `data/literature.json closure.target.min (${literatureMin}) -- лестница стадий и ворота ` +
        `должны использовать один и тот же физически обоснованный минимум объёма полости`,
    )
  }
  return t
}

/** Every particle index that belongs to a recognised amphiphile (its head or any bead of its
 * chain) -- the vesicle wall's own material, as opposed to free monomers/donors/catalysts drifting
 * through the box. Exported so a caller that wants the SAME "wall" definition detectStage's own
 * closure/aggregation measurements use (e.g. a viewer highlighting the cavity those measurements
 * found) can rebuild it from `amphiphiles` it already has, instead of drifting from this
 * definition with a second, independently-written one. */
export function memberIndicesOf(amphiphiles: Amphiphile[]): Set<number> {
  const memberIdx = new Set<number>()
  for (const a of amphiphiles) {
    memberIdx.add(a.headIndex)
    for (const c of a.chain) memberIdx.add(c)
  }
  return memberIdx
}

/** `particles`, filtered down to just the indices in `idx` -- the flat vec4-per-particle layout
 * enclosedVolumeFromPositions/cavitiesFromPositions (engine/src/closure.ts) expect. Exported
 * alongside memberIndicesOf() for the same reason: one definition of "the wall's positions", used
 * both by detectStage's own enclosedVolume measurement below and by a viewer wanting the identical
 * cavity as voxels to draw. */
export function positionsFor(particles: Float32Array, idx: Set<number>): Float32Array {
  const idxArr = Array.from(idx)
  const out = new Float32Array(idxArr.length * 4)
  for (let k = 0; k < idxArr.length; k++) out.set(particles.subarray(idxArr[k] * 4, idxArr[k] * 4 + 4), k * 4)
  return out
}

/** Copy of `particles` with the 4th (kind-index) component replaced by `flagFor(particleIndex)` --
 * see this file's header for why: engine/src/aggregate.ts and engine/src/metrics.ts read that slot
 * as a fixed head(0)/tail(nonzero) code from the membrane engine, and this is how soup's own
 * kind-index encoding gets translated into that code at the call site instead of forking either
 * function for a second particle layout. */
function remapFlag(particles: Float32Array, flagFor: (i: number) => number): Float32Array {
  const out = particles.slice()
  const n = out.length / 4
  for (let i = 0; i < n; i++) out[i * 4 + 3] = flagFor(i)
  return out
}

/** Defect 2's fix: how many head-density peaks to trust, given a snapshot's raw particle array.
 * Pure CPU math (densityProfileZ/bilayerThickness are both pure, no GPU) -- unit-testable on
 * synthetic `particles` with no browser/WebGPU context, which is what
 * tests/soup-amphiphile.test.ts's new tests do.
 *
 * Three-way result, never a bare "2" on faith:
 *  - 0: no polar particles in this snapshot at all -- nothing to profile.
 *  - HEAD_PEAKS_UNAVAILABLE: there ARE polar particles, but too few of them per bin, on average
 *    (totalHeadCount / headDensityBins < minHeadsPerBin), for a peak count read off this histogram
 *    to mean anything -- see this file's header for the pilot's step-0 false positive this guards
 *    against. This is a property of the run's OWN composition (particle counts and bin count never
 *    change once a system is created), not of any one snapshot's structure -- so for a given
 *    system, headPeaks is either trustworthy for its whole trajectory or never trustworthy at all.
 *  - 1: enough particles per bin to trust the count, but either bilayerThickness() found no second
 *    peak (the normal case for most of a soup trajectory) or it found one whose separation from the
 *    first falls outside the plausible bilayer-thickness band -- two local maxima that are not, by
 *    this guard's judgement, two leaflets of the same structure.
 *  - 2: enough particles per bin, AND two peaks, AND their separation is inside the plausible band.
 *    The only value the bilayer/vesicle stages are allowed to treat as a real bilayer signal.
 *
 * Item 1a (2026-08 crash report): a live run died with `densityProfileZ: бусина с z=... вне [0,
 * ...)` while this function fed it a raw, unfiltered snapshot. densityProfileZ is RIGHT to throw
 * on out-of-range z (its own doc comment: clamping would manufacture a false peak at an edge bin)
 * -- the bug was here, the caller, handing it a particle it cannot profile. A soup particle really
 * can end up with z outside [0, box[2]) on a live snapshot (soup/wgsl/step.wgsl wraps every axis
 * every step, but that wrap is a float32 `x - floor(x/box)*box`, which can round to exactly `box`
 * -- or, on a diverging trajectory, drift far past it -- so "wrapped" is not a strict on-paper
 * guarantee at read-back time). Fixed the same way engine/src/index.ts's measureBilayerAveraged
 * already handles the identical situation for the membrane engine's own open z axis: drop the
 * escaped particles with dropEscapedZ (engine/src/metrics.ts) before profiling, never hand
 * densityProfileZ anything it would reject. If dropping empties out the head population below
 * the existing minHeadsPerBin guard, that guard now fires on the POST-drop count and this reports
 * HEAD_PEAKS_UNAVAILABLE -- an honest "can't say" rather than a crash. */
export function computeHeadPeaks(
  particles: Float32Array,
  box: [number, number, number],
  monomers: Monomer[],
  thresholds: Pick<StageThresholds, 'headDensityBins' | 'minHeadsPerBin' | 'headPeakSeparationMin' | 'headPeakSeparationMax'>,
): number | typeof HEAD_PEAKS_UNAVAILABLE {
  const n = particles.length / 4
  let totalHeads = 0
  for (let i = 0; i < n; i++) {
    if (monomers[Math.round(particles[i * 4 + 3])]?.polar) totalHeads++
  }
  if (totalHeads === 0) return 0

  const remapped = remapFlag(particles, (i) => (monomers[Math.round(particles[i * 4 + 3])]?.polar ? 0 : 1))
  // Drop anything densityProfileZ would throw on -- see this function's own doc comment above.
  const { positions: inBounds } = dropEscapedZ(remapped, box)
  let keptHeads = 0
  const m = inBounds.length / 4
  for (let i = 0; i < m; i++) if (inBounds[i * 4 + 3] === 0) keptHeads++
  if (keptHeads === 0) return HEAD_PEAKS_UNAVAILABLE // every head escaped -- nothing left to profile

  const avgHeadsPerBin = keptHeads / thresholds.headDensityBins
  if (avgHeadsPerBin < thresholds.minHeadsPerBin) return HEAD_PEAKS_UNAVAILABLE

  const profile = densityProfileZ(inBounds, box, thresholds.headDensityBins)
  try {
    const separation = bilayerThickness(profile) // throws => no second peak at all; caught below
    if (separation >= thresholds.headPeakSeparationMin && separation <= thresholds.headPeakSeparationMax) return 2
    return 1 // two maxima exist, but not a plausible bilayer separation -- not trusted as a bilayer
  } catch {
    return 1
  }
}

/** Defect 1's fix: the stage ladder. Each stage's condition REQUIRES every earlier stage's own
 * condition to hold too, not merely its own -- so `micelles` is structurally impossible without the
 * amphiphile condition, `bilayer` impossible without micelles' condition, `vesicle` impossible
 * without bilayer's. This is what makes the ladder a ladder: before this fix, `micelles`/`bilayer`
 * tested `largestAggregateFraction` alone, so a handful of NON-amphiphile particles clustering
 * together (large fraction, zero amphiphiles) could satisfy them -- see this file's header and
 * task-3b-report.md for the pilot trace this produced. Pure function of the evidence + thresholds,
 * no GPU, unit-testable on synthetic StageEvidence objects. */
export function stageFromEvidence(evidence: StageEvidence, thresholds: StageThresholds): Stage {
  const amphiphileOk = evidence.amphiphileFraction >= thresholds.amphiphileFraction
  const micelleOk = amphiphileOk && evidence.largestAggregateFraction >= thresholds.aggregationLow
  const bilayerOk = micelleOk && evidence.headPeaks === 2 && evidence.largestAggregateFraction >= thresholds.aggregationHigh
  const vesicleOk = bilayerOk && evidence.enclosedVolume > thresholds.enclosedVolume

  if (vesicleOk) return 'vesicle'
  if (bilayerOk) return 'bilayer'
  if (micelleOk) return 'micelles'
  if (amphiphileOk) return 'amphiphiles'
  return 'monomers'
}

/** Measures the four stage-deciding numbers for one snapshot of `sys`, then maps them to a stage
 * label via data/soup.json's stageThresholds. See this file's header for how `box` is obtained
 * (Task 4 reconciliation: `sys.box`, not a parameter). */
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

  // --- largestAggregateFraction: cluster only the particles that ARE amphiphile members (heads +
  // chains) -- free monomers, unreacted donors and catalysts have no business defining "the
  // aggregate", the same way the membrane engine's largestClusterFractionOf only ever clusters
  // tail beads. Cutoff derived from this soup's own species sizes (data/soup.json radiusSigma),
  // mirroring soup/src/sim.ts's own cellSize derivation -- not a new constant.
  let largestAggregateFraction = 0
  if (memberIdx.size > 0) {
    const p = loadParams()
    const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
    const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
    const remapped = remapFlag(particles, (i) => (memberIdx.has(i) ? 1 : 0))
    largestAggregateFraction = largestClusterFraction(remapped, box, cutoff)
  }

  // --- headPeaks: defect 2's fix, see computeHeadPeaks()'s own header for the two guards.
  const headPeaks = computeHeadPeaks(particles, box, soup.monomers, thresholds)

  // --- enclosedVolume: flood-fill closure over the amphiphile-member particles only (the vesicle
  // wall's own material) -- free monomers/donors/catalysts drifting through the box interior must
  // not be misread as part of the wall. enclosedVolumeFromPositions recentres on the dominant
  // cluster internally (engine/src/closure.ts), so no separate recentring step is needed here.
  let enclosedVolume = 0
  if (memberIdx.size > 0) {
    const memberPositions = positionsFor(particles, memberIdx)
    enclosedVolume = enclosedVolumeFromPositions(memberPositions, box, {
      cell: thresholds.closureCell,
      radius: thresholds.closureRadius,
    })
  }

  const evidence: StageEvidence = { amphiphileFraction, largestAggregateFraction, headPeaks, enclosedVolume }
  const stage = stageFromEvidence(evidence, thresholds)

  return { stage, evidence }
}
