import { expect, test } from 'vitest'
import {
  asfPrediction,
  carbonChainLengths,
  ccBondFraction,
  floryPrediction,
  recoverAlphaFromChainLengths,
  vanHoffSlope,
} from '../soup/src/equilibrium'
import type { Monomer } from '../soup/src/rules'

// Pure math only -- no GPU, no browser -- mirroring tests/params.test.ts's own no-GPU discipline
// for anything that does not need the engine. The GPU-dependent checks (does the MEASURED
// distribution/bond-fraction sweep match these predictions) live in the calibration report, not a
// committed test, because they need multiple full soup runs (see this repo's own resource rules
// for this task) -- committing them here would make every `npm test` pay that cost.

const MONOMERS: Monomer[] = [
  { id: 'C', kind: 'carbon', radiusSigma: 1.0, polar: false },
  { id: 'O', kind: 'head', radiusSigma: 0.9, polar: true },
  { id: 'H', kind: 'donor', radiusSigma: 0.8, polar: false },
  { id: 'M', kind: 'catalyst', radiusSigma: 1.2, polar: false },
]

/** Builds the flat particle array this module's functions expect: one row per kind index in
 * `kinds` (positions are irrelevant to bond-graph-only functions, so all zero). */
function particlesOf(kinds: number[]): Float32Array {
  const out = new Float32Array(kinds.length * 4)
  for (let i = 0; i < kinds.length; i++) out[i * 4 + 3] = kinds[i]
  return out
}

test('floryPrediction нормирована и убывает геометрически', () => {
  const p = floryPrediction(2.0, 1.0, 0.05, 30)
  const vals = Object.values(p)
  expect(vals.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  for (let n = 3; n <= 10; n++) expect(p[n] / p[n - 1]).toBeCloseTo(p[3] / p[2], 6)
})

test('floryPrediction совпадает с формулой геометрического распределения при явном подсчёте', () => {
  const energyKT = 8.8
  const kT = 1.1
  const c = 0.0046466
  const K = Math.exp(energyKT / kT)
  const x = (K * c) / (1 + K * c)
  // maxLength must be wide enough that the truncated-and-renormalised window's own normalisation
  // factor is indistinguishable from 1 (x is close to 1 here, so the tail decays slowly) --
  // otherwise this is testing the renormalisation, not the raw geometric formula.
  const p = floryPrediction(energyKT, kT, c, 400)
  for (const n of [1, 2, 5, 10]) {
    expect(p[n]).toBeCloseTo((1 - x) * x ** (n - 1), 6)
  }
  // mean of the (untruncated) geometric distribution is 1/(1-x) -- sanity-check against the
  // truncated-and-renormalised version by summing n*p[n] over a window wide enough that the tail
  // mass is negligible.
  let mean = 0
  const wide = floryPrediction(energyKT, kT, c, 200)
  for (let n = 1; n <= 200; n++) mean += n * wide[n]
  expect(mean).toBeCloseTo(1 / (1 - x), 1)
})

test('vanHoffSlope восстанавливает заложенную энергию на синтетике', () => {
  const e = 8.8
  const c = 0.02
  const pts = [0.8, 1.0, 1.2, 1.5, 2.0].map((kT) => {
    const K = Math.exp(e / kT) * c
    return { kT, bondFraction: K / (1 + K) }
  })
  const f = vanHoffSlope(pts)
  expect(f.slope).toBeCloseTo(e, 3)
  expect(f.intercept).toBeCloseTo(Math.log(c), 3)
  expect(f.r2).toBeGreaterThan(0.999)
})

test('vanHoffSlope: точки не на прямой дают низкий r2 -- регрессия не притворяется идеальной', () => {
  const pts = [
    { kT: 0.8, bondFraction: 0.9 },
    { kT: 1.0, bondFraction: 0.1 },
    { kT: 1.5, bondFraction: 0.6 },
    { kT: 2.0, bondFraction: 0.3 },
  ]
  const f = vanHoffSlope(pts)
  expect(f.r2).toBeLessThan(0.9)
})

test('carbonChainLengths: цепь C-C-C с головой на конце -- одна тройка углеродов, голова не участвует', () => {
  // indices: 0=C,1=C,2=C,3=O(head), bonds: 0-1, 1-2 (chain), 2-3 (head cap)
  const particles = particlesOf([0, 0, 0, 1])
  const bonds = new Uint32Array([0, 1, 1, 2, 2, 3])
  const lengths = carbonChainLengths(particles, bonds, MONOMERS)
  expect(lengths).toEqual([3])
})

test('carbonChainLengths: голова НЕ сливает два разных углеродных фрагмента', () => {
  // Y-shape at a head with chainCapacity=2: head 4 bonded to carbon 0 and carbon 2, each carbon
  // extends into its own tail (0-1, 2-3). The C-C graph alone (ignoring the head) must see TWO
  // separate two-carbon chains, not one four-carbon chain through the head.
  const particles = particlesOf([0, 0, 0, 0, 1])
  const bonds = new Uint32Array([4, 0, 0, 1, 4, 2, 2, 3])
  const lengths = carbonChainLengths(particles, bonds, MONOMERS).sort()
  expect(lengths).toEqual([2, 2])
})

test('carbonChainLengths: непрореагировавший углерод -- собственная цепь длины 1', () => {
  const particles = particlesOf([0, 0])
  const bonds = new Uint32Array([])
  const lengths = carbonChainLengths(particles, bonds, MONOMERS).sort()
  expect(lengths).toEqual([1, 1])
})

test('ccBondFraction: половина углеродных слотов занята', () => {
  // 4 carbons, one C-C bond (2 slots occupied out of 8 total -- 2 carbons at degree 1, 2 at degree
  // 0) -- ccEdges=1, carbonCount=4, fraction = 1/4, matching ccEdges/carbonCount (NOT (2*edges)/
  // (2*carbonCount) written out, though they are the same number here by construction of this case
  // being checked against the direct definition below).
  const particles = particlesOf([0, 0, 0, 0])
  const bonds = new Uint32Array([0, 1])
  expect(ccBondFraction(particles, bonds, MONOMERS)).toBeCloseTo(1 / 4, 9)
})

test('ccBondFraction: полностью насыщенная цепь (каждый внутренний узел на 2 связях)', () => {
  // 3 carbons in a line: 0-1, 1-2 -- 2 edges, 3 carbons -> fraction = 2/3
  const particles = particlesOf([0, 0, 0])
  const bonds = new Uint32Array([0, 1, 1, 2])
  expect(ccBondFraction(particles, bonds, MONOMERS)).toBeCloseTo(2 / 3, 9)
})

test('ccBondFraction: связи C-O не считаются связями C-C', () => {
  const particles = particlesOf([0, 1]) // one carbon, one head, bonded
  const bonds = new Uint32Array([0, 1])
  expect(ccBondFraction(particles, bonds, MONOMERS)).toBe(0)
})

// --- ASF (kinetic-growth, 2026-08-17): same log-linear regression discipline as vanHoffSlope's own
// tests above, just against chain length instead of 1/kT -- see soup/src/equilibrium.ts's own doc
// comments for why alpha here is a KINETIC branching ratio, not floryPrediction's equilibrium x.

test('asfPrediction нормирована и убывает геометрически, отношение соседних членов = alpha', () => {
  const p = asfPrediction(14 / 15, 400)
  const vals = Object.values(p)
  expect(vals.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  for (let n = 3; n <= 20; n++) expect(p[n] / p[n - 1]).toBeCloseTo(p[3] / p[2], 6)
  expect(p[3] / p[2]).toBeCloseTo(14 / 15, 6)
})

test('asfPrediction: mean = 1/(1-alpha), проверено суммированием по широкому окну', () => {
  const alpha = 14 / 15
  const p = asfPrediction(alpha, 400)
  let mean = 0
  for (let n = 1; n <= 400; n++) mean += n * p[n]
  expect(mean).toBeCloseTo(1 / (1 - alpha), 1)
})

test('recoverAlphaFromChainLengths восстанавливает заложенное alpha на синтетическом ASF-ансамбле', () => {
  const alpha = 14 / 15
  const p = asfPrediction(alpha, 60)
  // Deterministic synthetic population (expected counts, not random draws) -- exactly the same
  // discipline soup-equilibrium's own vanHoffSlope test above uses for its synthetic points: this
  // checks the REGRESSION machinery recovers a KNOWN alpha exactly, independent of any GPU sampling
  // noise (that check lives in the report, not a committed test -- same reasoning as this file's own
  // header for why the GPU-dependent calibration checks are not committed here).
  const bigN = 1_000_000
  const lengths: number[] = []
  for (let n = 1; n <= 60; n++) {
    const count = Math.round(p[n] * bigN)
    for (let k = 0; k < count; k++) lengths.push(n)
  }
  const r = recoverAlphaFromChainLengths(lengths)
  expect(r.alpha).toBeCloseTo(alpha, 3)
  expect(r.r2).toBeGreaterThan(0.999)
})

test('recoverAlphaFromChainLengths: меньше двух различных длин -- бросает, а не тихо считает по одной точке', () => {
  expect(() => recoverAlphaFromChainLengths([5, 5, 5])).toThrow(/меньше двух/)
})
