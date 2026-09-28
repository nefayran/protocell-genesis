import { expect, test } from 'vitest'
import { findAmphiphiles } from '../soup/src/amphiphile'
import { analyzeAggregates, memberIndicesOf } from '../soup/src/aggregates'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { loadSoup } from '../soup/src/rules'
import { loadStageThresholds, stageFromEvidence, type StageEvidence } from '../soup/src/stages'

// task-3c/per-aggregate-report: synthetic, GPU-free configurations for analyzeAggregates()'s own
// shape/head-shell/cavity logic and the redefined micelles/bilayer/vesicle ladder -- these are the
// tests a real 200000-step dilute run cannot cheaply repeat, but a hand-built particle/bond array
// can pin exactly, in milliseconds to a few seconds (Scenario C's own closed-shell construction
// needs a genuinely dense wall to close, see its own header below).
//
// All four build amphiphiles the same way findAmphiphiles() already recognises: a head-carbon pair
// (one bond) is the shortest valid amphiphile (chain length 1 -- the SAME mode the real dilute run's
// own chain-length histogram peaks at, per data/soup.json's startBasis) -- reusing the real
// recogniser, not a stand-in, so these tests exercise the SAME "what counts as an amphiphile"
// definition detectStage's own live path does.

const soup = loadSoup()
const monomers = soup.monomers
const C = monomers.findIndex((m) => m.kind === 'carbon')
const O = monomers.findIndex((m) => m.kind === 'head')
const thresholds = loadStageThresholds()
const p = loadParams()
const memberRadii = monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
// The SAME cutoff detectStage's own largestAggregateFraction/analyzeAggregates calls use (soup/src/
// stages.ts) -- derived from this soup's own species sizes, not a new constant for these tests.
const CUTOFF = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc

function buildAmphiphile(headPos: [number, number, number], tailPos: [number, number, number], particles: number[], bonds: number[]): void {
  const headIdx = particles.length / 4
  particles.push(headPos[0], headPos[1], headPos[2], O)
  const tailIdx = particles.length / 4
  particles.push(tailPos[0], tailPos[1], tailPos[2], C)
  bonds.push(headIdx, tailIdx)
}

function randomDir(): [number, number, number] {
  const u = Math.random() * 2 - 1
  const phi = Math.random() * 2 * Math.PI
  const s = Math.sqrt(1 - u * u)
  return [s * Math.cos(phi), s * Math.sin(phi), u]
}

/** Evenly-spread points on a unit sphere (Fibonacci/golden-angle spiral) -- deterministic, unlike
 * randomDir(), so Scenario A's micelles are reproducible run to run. */
function fibonacciSphere(n: number): [number, number, number][] {
  const pts: [number, number, number][] = []
  const golden = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2
    const r = Math.sqrt(1 - y * y)
    const theta = golden * i
    pts.push([Math.cos(theta) * r, y, Math.sin(theta) * r])
  }
  return pts
}

/** Runs the real recogniser + real per-aggregate analysis + real ladder on a synthetic particle/
 * bond array -- amphiphileFraction is fixed at 1 (every particle here IS amphiphile material, by
 * construction, so the fraction is trivially 1 and these tests are entirely about what
 * analyzeAggregates()/stageFromEvidence() do with the STRUCTURE, not about the amphiphile-fraction
 * gate itself -- soup-amphiphile.test.ts already covers that gate). */
function stageFor(particlesArr: number[], bondsArr: number[], box: [number, number, number]) {
  const particles = new Float32Array(particlesArr)
  const bonds = new Uint32Array(bondsArr)
  const amphiphiles = findAmphiphiles(particles, bonds, monomers)
  const memberIdx = memberIndicesOf(amphiphiles)
  const analysis = analyzeAggregates(particles, box, monomers, amphiphiles, memberIdx, CUTOFF, thresholds)
  const evidence: StageEvidence = { amphiphileFraction: 1, largestAggregateFraction: 0, headPeaks: 0, enclosedVolume: 0, aggregateAnalysis: analysis }
  return { analysis, stage: stageFromEvidence(evidence, thresholds) }
}

// --- Scenario A: many small spherical micelles ----------------------------------------------------
// Catches: the exact defect this task fixes (largestAggregateFraction gating micelles on ONE
// dominant aggregate) -- three well-separated small spheres (heads on a shell, tails pointing
// inward, exactly the micelle shape the task's own evidence describes) must read `micelles`, not
// get stuck at `amphiphiles` for lacking a single big aggregate.
test('several small spherical micelles read as micelles', () => {
  const particles: number[] = []
  const bonds: number[] = []
  const centers: [number, number, number][] = [[20, 20, 20], [70, 20, 20], [20, 70, 20]]
  const R_HEAD = 1.2
  const R_TAIL = 0.5
  const PER_MICELLE = 20 // above minAmphiphilesPerAggregate (13) with margin
  for (const c of centers) {
    for (const d of fibonacciSphere(PER_MICELLE)) {
      buildAmphiphile(
        [c[0] + d[0] * R_HEAD, c[1] + d[1] * R_HEAD, c[2] + d[2] * R_HEAD],
        [c[0] + d[0] * R_TAIL, c[1] + d[1] * R_TAIL, c[2] + d[2] * R_TAIL],
        particles, bonds,
      )
    }
  }
  const box: [number, number, number] = [100, 100, 100]
  const { analysis, stage } = stageFor(particles, bonds, box)

  expect(analysis.aggregateCount).toBe(3)
  expect(analysis.sizeHistogram).toEqual([PER_MICELLE, PER_MICELLE, PER_MICELLE])
  expect(analysis.qualifyingAggregateCount).toBe(3)
  expect(analysis.amphiphileShareInQualifying).toBe(1)
  // A round, small aggregate is neither flat (bilayer) nor a two-shell object (vesicle) -- the
  // stage must land on `micelles` specifically, not accidentally skip past it.
  expect(analysis.hasLamellarAggregate).toBe(false)
  expect(analysis.hasVesicleAggregate).toBe(false)
  expect(stage).toBe('micelles')
})

// --- Scenario B: one flat slab (bilayer) ------------------------------------------------------------
// Catches: the second defect this task fixes (a box-wide z-density profile gating bilayer, meaningless
// for anything but a box-spanning sheet) AND exercises the TRANSVERSE (not radial) head profile --
// heads sit at the SAME unsigned radial distance from the patch's own centroid on both faces of a
// centred flat sheet (radialHeadShells folds them onto one peak), so only the signed, along-the-
// flat-axis transverse profile can see the two layers. A square patch (not a stretched rectangle) is
// used so BOTH shape gates (flatnessRatio and inPlaneSymmetry) pass on the same aggregate.
test('a single flat layer reads as bilayer (from the transverse head profile)', () => {
  const particles: number[] = []
  const bonds: number[] = []
  const NX = 13
  const NY = 13
  const SPACING = 1.4
  const Z_HEAD = 2.2
  const Z_TAIL = 1.0
  for (let ix = 0; ix < NX; ix++) {
    for (let iy = 0; iy < NY; iy++) {
      const x = 30 + ix * SPACING
      const y = 30 + iy * SPACING
      buildAmphiphile([x, y, 30 + Z_HEAD], [x, y, 30 + Z_TAIL], particles, bonds)
      buildAmphiphile([x, y, 30 - Z_HEAD], [x, y, 30 - Z_TAIL], particles, bonds)
    }
  }
  const box: [number, number, number] = [100, 100, 100]
  const { analysis, stage } = stageFor(particles, bonds, box)

  expect(analysis.aggregateCount).toBe(1)
  const shape = analysis.aggregates[0]
  expect(shape.flatnessRatio).toBeLessThanOrEqual(thresholds.lamellarFlatnessRatio)
  expect(shape.inPlaneSymmetry).toBeGreaterThanOrEqual(thresholds.lamellarInPlaneSymmetryMin)
  expect(shape.transverseHeadShells).toBe(2)
  expect(analysis.hasLamellarAggregate).toBe(true)
  // Not a closed object -- no cavity, so it must not (and structurally cannot) read as `vesicle`.
  expect(shape.cavityVolume).toBe(0)
  expect(analysis.hasVesicleAggregate).toBe(false)
  expect(stage).toBe('bilayer')
})

// --- Scenario C: a hollow shell with a large interior (vesicle) -------------------------------------
// Catches: the vesicle stage's own per-aggregate condition (two head SHELLS -- inner AND outer,
// which the RADIAL profile is the right axis for, unlike Scenario B's flat patch) AND a genuinely
// enclosed cavity clearing the EXISTING physically-derived minimum (kept exactly as it was). The
// wall's tails are placed at an INDEPENDENT random direction and a radius drawn uniformly across the
// WHOLE wall thickness [R_IN, R_OUT] -- a volumetric fill of the annulus. An earlier attempt that
// clustered every tail at one fixed mid-radius instead left a genuine, un-bridged radial gap on
// either side of that mid-shell (each half the wall's own thickness wide, far more than the closure
// detector's own bead reach), and the flood punched straight through it -- reproducing that failure
// here would be silent (a smaller-but-nonzero cavity, not an error), so this construction is
// deliberate, not incidental.
test('a hollow shell with a large inner cavity reads as vesicle', () => {
  const particles: number[] = []
  const bonds: number[] = []
  const R_IN = 8
  const R_OUT = 12.4 // separation 4.4sigma -- inside headPeakSeparationMin/Max's own band
  const THICKNESS = 1.0
  const N_OUTER = 8_000
  const N_INNER = 4_000
  const centre: [number, number, number] = [30, 30, 30]
  function volumetricTail(): [number, number, number] {
    const d = randomDir()
    const r = R_IN + Math.random() * (R_OUT - R_IN)
    return [centre[0] + d[0] * r, centre[1] + d[1] * r, centre[2] + d[2] * r]
  }
  for (let i = 0; i < N_OUTER; i++) {
    const d = randomDir()
    const rHead = R_OUT + (Math.random() - 0.5) * THICKNESS
    buildAmphiphile([centre[0] + d[0] * rHead, centre[1] + d[1] * rHead, centre[2] + d[2] * rHead], volumetricTail(), particles, bonds)
  }
  for (let i = 0; i < N_INNER; i++) {
    const d = randomDir()
    const rHead = R_IN + (Math.random() - 0.5) * THICKNESS
    buildAmphiphile([centre[0] + d[0] * rHead, centre[1] + d[1] * rHead, centre[2] + d[2] * rHead], volumetricTail(), particles, bonds)
  }
  const box: [number, number, number] = [60, 60, 60]
  const { analysis, stage } = stageFor(particles, bonds, box)

  expect(analysis.aggregateCount).toBe(1)
  const shape = analysis.aggregates[0]
  expect(shape.radialHeadShells).toBe(2)
  expect(shape.cavityVolume).toBeGreaterThan(thresholds.enclosedVolume)
  expect(analysis.hasVesicleAggregate).toBe(true)
  // A genuinely spherical shell has no preferred flat axis (unlike Scenario B's patch) -- must not
  // ALSO read as lamellar just because it happens to clear the vesicle gate.
  expect(analysis.hasLamellarAggregate).toBe(false)
  expect(stage).toBe('vesicle')
})

// --- Scenario D: a single percolating blob must NOT read micelles merely because it is big -----------
// Catches: the exact old defect in the opposite direction -- a single large aggregate is the
// OPPOSITE of a micellar state (which requires matter split among several aggregates), and the OLD
// `largestAggregateFraction >= aggregationLow` criterion would have called this micelles for being
// big alone. Deterministic dense cubic-lattice fill of a ball (guaranteed one connected component,
// no randomness needed) -- amphiphiles scattered with no shell/flat organisation at all, so it also
// must not accidentally read as bilayer/vesicle.
test('a single large percolating aggregate does not read as micelles merely because of its size', () => {
  const particles: number[] = []
  const bonds: number[] = []
  const centre: [number, number, number] = [30, 30, 30]
  const SPACING = 1.8
  const RADIUS = 9
  for (let ix = -6; ix <= 6; ix++) {
    for (let iy = -6; iy <= 6; iy++) {
      for (let iz = -6; iz <= 6; iz++) {
        const x = ix * SPACING
        const y = iy * SPACING
        const z = iz * SPACING
        if (Math.sqrt(x * x + y * y + z * z) > RADIUS) continue
        buildAmphiphile(
          [centre[0] + x, centre[1] + y, centre[2] + z],
          [centre[0] + x + 0.3, centre[1] + y, centre[2] + z],
          particles, bonds,
        )
      }
    }
  }
  const box: [number, number, number] = [60, 60, 60]
  const { analysis, stage } = stageFor(particles, bonds, box)

  expect(analysis.aggregateCount).toBe(1)
  expect(analysis.sizeHistogram[0]).toBeGreaterThan(thresholds.minAmphiphilesPerAggregate * 10) // genuinely big
  // The micelle gate requires MULTIPLE qualifying aggregates -- one big blob gives exactly one.
  expect(analysis.qualifyingAggregateCount).toBe(1)
  expect(stage).not.toBe('micelles')
  expect(stage).not.toBe('bilayer')
  expect(stage).not.toBe('vesicle')
})
