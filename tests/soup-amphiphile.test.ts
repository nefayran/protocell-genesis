import { expect, test } from 'vitest'
import { amphiphileHistogram, findAmphiphiles } from '../soup/src/amphiphile'
import { loadSoup } from '../soup/src/rules'
import { computeHeadPeaks, loadStageThresholds, stageFromEvidence } from '../soup/src/stages'

const P = (xs: number[][]) => new Float32Array(xs.flat())

test('цепь с одной полярной головой распознаётся как амфифил', () => {
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 2,3])
  const a = findAmphiphiles(parts, bonds, loadSoup().monomers)
  expect(a.length).toBe(1)
  expect(a[0].length).toBe(3)
})

test('цепь без головы и цепь с двумя головами амфифилами не считаются', () => {
  const m = loadSoup().monomers
  const noHead = findAmphiphiles(P([[0,0,0,0],[1,0,0,0],[2,0,0,0]]), new Uint32Array([0,1, 1,2]), m)
  expect(noHead.length).toBe(0)
  const twoHeads = findAmphiphiles(P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,1]]), new Uint32Array([0,1, 1,2, 2,3]), m)
  expect(twoHeads.length).toBe(0)
})

test('гистограмма длин считает цепи по числу углеродов', () => {
  const m = loadSoup().monomers
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0], [10,0,0,1],[11,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 3,4])
  expect(amphiphileHistogram(findAmphiphiles(parts, bonds, m))).toEqual({ 1: 1, 2: 1 })
})

// --- task-3b: two pilot-exposed stage-detection defects (task-3b-report.md) ------------------------
// Both fixes live in soup/src/stages.ts as pure, GPU-free functions (stageFromEvidence,
// computeHeadPeaks) precisely so they can be unit-tested here on synthetic data, with no browser/
// WebGPU context -- the pilot itself (30 samples, 300k steps) is not re-run.

// Defect 1: the OLD stage ladder tested each stage's OWN condition in isolation, so
// largestAggregateFraction alone (denominator = the small amphiphile-member pool only) could
// saturate past aggregationLow/High before amphiphileFraction ever crossed its own threshold --
// this pilot's own step 30,000 sample (amphiphileFraction=0.0855, largestAggregateFraction=0.6619)
// is exactly that: aggregation already past 0.3 while amphiphileFraction sat below the OLD 0.1
// threshold, so the trace jumped straight to `micelles`, skipping `amphiphiles` in all 30 samples.
test('амфифилы есть, агрегат ещё мал — стадия amphiphiles, не перепрыгнутая агрегацией (дефект 1)', () => {
  const thresholds = loadStageThresholds()
  const evidence = {
    amphiphileFraction: thresholds.amphiphileFraction + 0.01,
    largestAggregateFraction: thresholds.aggregationLow - 0.01,
    headPeaks: 0,
    enclosedVolume: 0,
  }
  expect(stageFromEvidence(evidence, thresholds)).toBe('amphiphiles')
})

// The direct flip side: a crisp `largestAggregateFraction`/`headPeaks` reading with NO amphiphiles
// recognised at all must not be read as `micelles` (or `bilayer`) just because the aggregate/peak
// numbers alone look like one -- the ladder makes both stages structurally require the amphiphile
// condition first.
test('крупный агрегат без единого распознанного амфифила НЕ считается micelles (дефект 1)', () => {
  const thresholds = loadStageThresholds()
  const evidence = { amphiphileFraction: 0, largestAggregateFraction: 0.95, headPeaks: 2, enclosedVolume: 0 }
  const stage = stageFromEvidence(evidence, thresholds)
  expect(stage).not.toBe('micelles')
  expect(stage).not.toBe('bilayer')
  expect(stage).toBe('monomers')
})

// Defect 2, first guard: the pilot reported headPeaks=2 at step 0 (zero bonds, zero structure) with
// 267 head particles spread across 60 z-bins (~4.45/bin, below minHeadsPerBin=10) -- reproduced here
// verbatim (same head count, same box scale) to confirm the guard now reports 'unavailable' instead
// of manufacturing a peak count out of Poisson noise. Second guard, same test: even well above the
// per-bin floor, two maxima sitting near opposite box faces (separation far outside the plausible
// headPeakSeparationMin/Max band) must not be trusted as a bilayer either.
test('шумовой профиль голов не даёт достоверных двух пиков (дефект 2)', () => {
  const thresholds = loadStageThresholds()
  const monomers = loadSoup().monomers
  const box: [number, number, number] = [20, 20, 20]

  // Too few heads per bin, on average -- the pilot's own numbers (task-4-pilot-report.md).
  const nSparse = 267
  const sparse = new Float32Array(nSparse * 4)
  for (let i = 0; i < nSparse; i++) {
    sparse[i * 4 + 2] = ((i + 0.5) * box[2]) / nSparse // spread thinly across the whole box
    sparse[i * 4 + 3] = 1 // 'O' -- polar head
  }
  expect(computeHeadPeaks(sparse, box, monomers, thresholds)).toBe('unavailable')

  // Plenty of heads per bin, but the two maxima sit near opposite box faces -- not a plausible
  // bilayer thickness, just two unrelated spikes.
  const nFaces = 700
  const faces = new Float32Array(nFaces * 4)
  for (let i = 0; i < nFaces; i++) {
    faces[i * 4 + 2] = i < nFaces / 2 ? 0.5 : box[2] - 0.5
    faces[i * 4 + 3] = 1
  }
  expect(computeHeadPeaks(faces, box, monomers, thresholds)).toBe(1)
})

// Defect 2's positive case: two well-separated, well-populated slabs (7 sigma apart, inside the
// plausible headPeakSeparationMin/Max band, well above minHeadsPerBin) must still be trusted as a
// real two-peak (bilayer-candidate) reading -- the guards must not make headPeaks=2 unreachable.
test('настоящий двухслойный профиль голов даёт достоверные два пика (дефект 2)', () => {
  const thresholds = loadStageThresholds()
  const monomers = loadSoup().monomers
  const box: [number, number, number] = [20, 20, 20]
  const n = 700
  const parts = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    parts[i * 4 + 2] = i < n / 2 ? 6 : 13 // two thin slabs, 7 sigma apart
    parts[i * 4 + 3] = 1
  }
  expect(computeHeadPeaks(parts, box, monomers, thresholds)).toBe(2)
})
