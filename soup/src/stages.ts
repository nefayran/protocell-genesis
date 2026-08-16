// Task 3: stage detection -- EVIDENCE, not narrative. detectStage never asserts a stage on its own
// say-so: it always hands back the four measurements the plan names (amphiphileFraction,
// largestAggregateFraction, headPeaks, enclosedVolume) alongside the label, so a caller can look at
// the numbers and disagree with the label. The four thresholds that turn those numbers into a
// label live in data/soup.json's `stageThresholds` (with their basis), not here -- this file reads
// them, it does not declare them, matching the "no numeric constants in soup/src" rule
// tests/params.test.ts's literal scanner enforces over this directory.
//
// Reuses rather than reimplements: findAmphiphiles (this task, soup/src/amphiphile.ts),
// largestClusterFraction (engine/src/aggregate.ts, Task 6's cell-list + union-find),
// densityProfileZ/bilayerPeaks (engine/src/metrics.ts, Task 5's sub-bin-peak z-histogram), and
// enclosedVolumeFromPositions (engine/src/closure.ts, Task 8's verified flood-fill closure
// detector, which recentres on the dominant cluster internally). None of those four functions know
// about soup/src/rules.ts's monomer kinds -- they were built for the membrane engine's own
// head(w=0)/tail(w!=0) encoding -- so remapFlag() below builds a throwaway copy of `particles`
// with the 4th (kind-index) component replaced by whichever binary flag the function being called
// expects, particle by particle. This is the same "adapt the encoding at the call site" pattern
// engine/src/index.ts's own facades already use (e.g. largestClusterFractionOf's cutoff derived
// from the caller's own species sizes) -- it is not a new union-find or a new histogram.
//
// Deviation from the plan's literal `detectStage(sys: SoupSystem)` signature, documented in
// task-3-report.md: SoupSystem (soup/src/sim.ts, Task 2) has no `box` field -- unlike the membrane
// engine's System, which does -- and this task's Files list only permits modifying
// engine/src/index.ts, not soup/src/sim.ts. Every measurement here (clustering, the z-density
// profile, the closure flood) needs the box, so detectStage takes it as an explicit second
// argument rather than reading it off `sys`.

import rawSoup from '../../data/soup.json'
import { largestClusterFraction } from '../../engine/src/aggregate'
import { enclosedVolumeFromPositions } from '../../engine/src/closure'
import { bilayerPeaks, densityProfileZ } from '../../engine/src/metrics'
import { loadParams, wcaCutoff } from '../../engine/src/params'
import { findAmphiphiles } from './amphiphile'
import { loadSoup } from './rules'
import type { SoupSystem } from './sim'

export type Stage = 'monomers' | 'amphiphiles' | 'micelles' | 'bilayer' | 'vesicle'

export interface StageEvidence {
  /** Fraction of carbon particles bound into a recognised amphiphile chain. */
  amphiphileFraction: number
  /** Fraction of amphiphile-member particles (heads + their chains) that belong to the single
   * largest connected aggregate among them -- 0 when there are no amphiphiles yet. */
  largestAggregateFraction: number
  /** 0 (no polar particles at all), 1 (one head-density peak -- not yet a bilayer, the normal case
   * for most of a soup trajectory), or 2 (two peaks -- bilayerPeaks succeeded). */
  headPeaks: number
  /** Volume of any flood-unreachable cavity, in the same reduced units as the box. 0 when nothing
   * is closed. */
  enclosedVolume: number
}

interface StageThresholds {
  amphiphileFraction: number
  aggregationLow: number
  aggregationHigh: number
  enclosedVolume: number
  closureCell: number
  closureRadius: number
  headDensityBins: number
  basis: string
}

function loadStageThresholds(): StageThresholds {
  const t = (rawSoup as unknown as { stageThresholds?: StageThresholds }).stageThresholds
  if (!t) throw new Error('data/soup.json: отсутствует поле stageThresholds')
  return t
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

/** Measures the four stage-deciding numbers for one snapshot of `sys`, then maps them to a stage
 * label via data/soup.json's stageThresholds. See this file's header for the box-parameter
 * deviation from the plan's literal signature. */
export async function detectStage(
  sys: SoupSystem,
  box: [number, number, number],
): Promise<{ stage: Stage; evidence: StageEvidence }> {
  const soup = loadSoup()
  const thresholds = loadStageThresholds()
  const particles = await sys.particles()
  const bonds = await sys.bonds()
  const n = particles.length / 4

  const amphiphiles = findAmphiphiles(particles, bonds, soup.monomers)
  const memberIdx = new Set<number>()
  for (const a of amphiphiles) {
    memberIdx.add(a.headIndex)
    for (const c of a.chain) memberIdx.add(c)
  }

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

  // --- headPeaks: reuse densityProfileZ/bilayerPeaks by remapping polar particles into the "head"
  // (w=0) bucket those functions already read. bilayerPeaks THROWS when there is no second peak --
  // the normal case for most of a soup trajectory, per this task's brief -- so that throw is caught
  // here and downgraded to "one peak", never left to propagate as a test failure.
  let headPeaks = 0
  let hasPolarParticle = false
  for (let i = 0; i < n && !hasPolarParticle; i++) {
    const kind = Math.round(particles[i * 4 + 3])
    if (soup.monomers[kind]?.polar) hasPolarParticle = true
  }
  if (hasPolarParticle) {
    headPeaks = 1
    try {
      const remapped = remapFlag(particles, (i) => {
        const kind = Math.round(particles[i * 4 + 3])
        return soup.monomers[kind].polar ? 0 : 1
      })
      const profile = densityProfileZ(remapped, box, thresholds.headDensityBins)
      bilayerPeaks(profile) // throws => not a bilayer yet, caught below
      headPeaks = 2
    } catch {
      headPeaks = 1
    }
  }

  // --- enclosedVolume: flood-fill closure over the amphiphile-member particles only (the vesicle
  // wall's own material) -- free monomers/donors/catalysts drifting through the box interior must
  // not be misread as part of the wall. enclosedVolumeFromPositions recentres on the dominant
  // cluster internally (engine/src/closure.ts), so no separate recentring step is needed here.
  let enclosedVolume = 0
  if (memberIdx.size > 0) {
    const idxArr = Array.from(memberIdx)
    const memberPositions = new Float32Array(idxArr.length * 4)
    for (let k = 0; k < idxArr.length; k++) {
      memberPositions.set(particles.subarray(idxArr[k] * 4, idxArr[k] * 4 + 4), k * 4)
    }
    enclosedVolume = enclosedVolumeFromPositions(memberPositions, box, {
      cell: thresholds.closureCell,
      radius: thresholds.closureRadius,
    })
  }

  const evidence: StageEvidence = { amphiphileFraction, largestAggregateFraction, headPeaks, enclosedVolume }

  let stage: Stage
  if (enclosedVolume > thresholds.enclosedVolume) {
    stage = 'vesicle'
  } else if (headPeaks >= 2 && largestAggregateFraction >= thresholds.aggregationHigh) {
    stage = 'bilayer'
  } else if (largestAggregateFraction >= thresholds.aggregationLow) {
    stage = 'micelles'
  } else if (amphiphileFraction >= thresholds.amphiphileFraction) {
    stage = 'amphiphiles'
  } else {
    stage = 'monomers'
  }

  return { stage, evidence }
}
