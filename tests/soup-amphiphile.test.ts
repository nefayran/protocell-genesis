import { expect, test } from 'vitest'
import { amphiphileHistogram, findAmphiphiles } from '../soup/src/amphiphile'
import { loadSoup } from '../soup/src/rules'
import { computeHeadPeaks, emptyAggregateAnalysis, loadStageThresholds, stageFromEvidence } from '../soup/src/stages'

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

// --- "two-tails" task: the recogniser must accept a head with TWO tails (data/soup.json's
// headPlacement.chainCapacity, now up to 2) as a Y whose branch point IS the head, while still
// rejecting a branch anywhere else (a genuinely branched tail) and a head buried mid-chain (which
// is a carbon carrying 2 chain bonds + 1 head bond, degree 3) -- these last two are exactly the
// configurations the recogniser's per-particle degree check exists to keep out.

test('голова с двумя хвостами (Y, точка ветвления — сама голова) распознаётся как один амфифил', () => {
  const m = loadSoup().monomers
  // O(0) bonded to C(1) and C(2); C(1) further bonded to C(3) -- a 2-carbon tail and a 1-carbon
  // tail hanging off the same head, the two-tailed amphiphile's own topology.
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,0]])
  const bonds = new Uint32Array([0,1, 0,2, 1,3])
  const a = findAmphiphiles(parts, bonds, m)
  expect(a.length).toBe(1)
  expect(a[0].headIndex).toBe(0)
  expect(a[0].length).toBe(3) // total non-polar count across BOTH tails
  expect([...a[0].tailLengths].sort((x, y) => x - y)).toEqual([1, 2]) // the two tails, reported separately
  expect(new Set(a[0].chain)).toEqual(new Set([1, 2, 3])) // both tails' particles, for aggregate membership
})

test('ветвящийся хвост (T-развилка НЕ на голове) амфифилом не считается', () => {
  const m = loadSoup().monomers
  // O(0)-C(1)-C(2), then C(2) branches into C(3) AND C(4) -- a genuine T-branch on a tail carbon,
  // nothing to do with the head, which still sits cleanly at one end (degree 1).
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,0],[4,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 2,3, 2,4])
  expect(findAmphiphiles(parts, bonds, m).length).toBe(0)
})

test('голова, зарытая в середине цепи (степень 3 у углерода), амфифилом не считается', () => {
  const m = loadSoup().monomers
  // C(0)-C(1)-C(2)-C(3), with the head O(4) attached to C(1) -- a carbon in the MIDDLE of the
  // chain, already holding 2 chain bonds, also carrying the head: degree 3, not a chain end.
  const parts = P([[0,0,0,0],[1,0,0,0],[2,0,0,0],[3,0,0,0],[4,0,0,1]])
  const bonds = new Uint32Array([0,1, 1,2, 2,3, 1,4])
  expect(findAmphiphiles(parts, bonds, m).length).toBe(0)
})

test('гистограмма длин считает цепи по числу углеродов', () => {
  const m = loadSoup().monomers
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0], [10,0,0,1],[11,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 3,4])
  expect(amphiphileHistogram(findAmphiphiles(parts, bonds, m))).toEqual({ 1: 1, 2: 1 })
})

// --- task-3b: pilot-exposed stage-detection defect (task-3b-report.md), ladder ordering -------------
// stageFromEvidence lives in soup/src/stages.ts as a pure, GPU-free function precisely so it can be
// unit-tested here on synthetic data, with no browser/WebGPU context -- the pilot itself (30
// samples, 300k steps) is not re-run. task-3c/per-aggregate-report redefined WHAT the micelles/
// bilayer/vesicle conditions consist of (per-aggregate quantities, soup/src/aggregates.ts's
// AggregateAnalysis, instead of largestAggregateFraction/headPeaks) but kept the LADDER PROPERTY
// task-3b's own fix established: a later stage is structurally impossible without every earlier
// stage's own condition too. These two tests, updated for the new evidence shape, still pin exactly
// that property.

// Defect 1 (task-3b): the OLD stage ladder tested each stage's OWN condition in isolation, so an
// aggregation-derived number could saturate before amphiphileFraction ever crossed its own
// threshold, skipping the `amphiphiles` label. Here: amphiphiles are present (fraction above
// threshold) but no aggregate has yet cleared the per-aggregate qualifying bar (an empty
// AggregateAnalysis, e.g. amphiphile-member particles have not yet found each other) -- the stage
// must read `amphiphiles`, not jump ahead on any other number.
test('амфифилы есть, агрегаты ещё не сложились — стадия amphiphiles (дефект 1, лестница)', () => {
  const thresholds = loadStageThresholds()
  const evidence = {
    amphiphileFraction: thresholds.amphiphileFraction + 0.01,
    largestAggregateFraction: 0,
    headPeaks: 0,
    enclosedVolume: 0,
    aggregateAnalysis: emptyAggregateAnalysis(),
  }
  expect(stageFromEvidence(evidence, thresholds)).toBe('amphiphiles')
})

// The direct flip side: a per-aggregate reading that would otherwise satisfy EVERY later stage's
// own condition (many qualifying aggregates, all the material in them, a lamellar shape, a shell +
// cavity) must still not be read as `micelles`/`bilayer`/`vesicle` when NO amphiphiles were
// recognised at all -- the ladder makes every later stage structurally require the amphiphile
// condition first, regardless of how saturated the aggregate analysis alone looks.
test('насыщенный анализ агрегатов без единого распознанного амфифила НЕ считается micelles (дефект 1, лестница)', () => {
  const thresholds = loadStageThresholds()
  const saturatedAnalysis = {
    aggregateCount: 3,
    sizeHistogram: [500, 400, 300],
    aggregates: [],
    qualifyingAggregateCount: 3,
    amphiphilesInQualifying: 1200,
    amphiphileShareInQualifying: 1,
    hasLamellarAggregate: true,
    hasVesicleAggregate: true,
  }
  const evidence = { amphiphileFraction: 0, largestAggregateFraction: 0.95, headPeaks: 2, enclosedVolume: 500, aggregateAnalysis: saturatedAnalysis }
  const stage = stageFromEvidence(evidence, thresholds)
  expect(stage).not.toBe('micelles')
  expect(stage).not.toBe('bilayer')
  expect(stage).not.toBe('vesicle')
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

// Item 1a (2026-08 crash report): a live run died mid-loop with `densityProfileZ: бусина с z=... вне
// [0, ...)` -- computeHeadPeaks was handing densityProfileZ a raw, unfiltered snapshot, and a soup
// particle can legitimately read back with z outside [0, box[2]) on a live snapshot (see
// computeHeadPeaks's own updated doc comment for why: soup/wgsl/step.wgsl's float32 wrap can round
// to exactly `box`, or worse on a diverging trajectory). densityProfileZ is right to throw on that
// (its own doc comment) -- the caller was wrong to feed it unfiltered data. Reproduced here with
// the exact two-slab profile from the previous test, but with one head bumped to z=box[2] exactly
// (the float32-rounding case) and a second bumped further out (z > box[2], the "genuinely escaped"
// case) -- computeHeadPeaks must not throw, and with only 2 of 700 heads dropped the remaining
// signal is still well above every guard, so the real two-peak reading must survive unharmed.
test('улетевшая за коробку бусина не рушит computeHeadPeaks (item 1a)', () => {
  const thresholds = loadStageThresholds()
  const monomers = loadSoup().monomers
  const box: [number, number, number] = [20, 20, 20]
  const n = 700
  const parts = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    parts[i * 4 + 2] = i < n / 2 ? 6 : 13 // two thin slabs, 7 sigma apart -- same as the passing case
    parts[i * 4 + 3] = 1
  }
  parts[0 * 4 + 2] = box[2] // exactly on the boundary -- z < box[2] fails, this is "escaped"
  parts[1 * 4 + 2] = box[2] + 4 // clearly past the box -- the open-axis-drift case
  expect(() => computeHeadPeaks(parts, box, monomers, thresholds)).not.toThrow()
  expect(computeHeadPeaks(parts, box, monomers, thresholds)).toBe(2)
})

// The all-escaped extreme: if every polar head has left the box, there is nothing left to profile
// at all -- must report 'unavailable' (an honest "can't say"), not throw and not silently invent a
// peak count from an empty profile.
test('все головы улетели за коробку — computeHeadPeaks сообщает н/д, не рушится (item 1a)', () => {
  const thresholds = loadStageThresholds()
  const monomers = loadSoup().monomers
  const box: [number, number, number] = [20, 20, 20]
  const n = 50
  const parts = new Float32Array(n * 4)
  for (let i = 0; i < n; i++) {
    parts[i * 4 + 2] = box[2] + 1 + i // every head past the box, spread out so it isn't one coincidence
    parts[i * 4 + 3] = 1
  }
  expect(() => computeHeadPeaks(parts, box, monomers, thresholds)).not.toThrow()
  expect(computeHeadPeaks(parts, box, monomers, thresholds)).toBe('unavailable')
})
