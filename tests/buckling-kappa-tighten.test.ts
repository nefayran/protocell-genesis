import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'kappa-tightening' (2026-08-18), part 1: tighten kappa (bending modulus, buckling method)
// beyond kappa-lambda-report.md's own three-run range (43.9-101.7 eps, a 2.3x spread). Repeats the
// EXACT SAME protocol tests/buckling-kappa.test.ts already validated -- same lipid count, box, delta,
// sweep depth, equilibration/sampling budget, weighted-regression fit, onset exclusion, plateau
// windows -- at NEW seeds, to grow the sample from 6 (startK=2/3) values to enough for a real
// mean+SE, and adds one long-equilibration diagnostic run to check whether the original 20 000-step
// per-point equilibration budget was itself the source of the scatter (as opposed to the already-
// documented GPU floating-point non-reproducibility, task-5/6/7 reports: "not bit-reproducible run
// to run, even at a fixed seed" -- an atomic cell-fill slot-order effect this task is not authorised
// to touch, since fixing it would mean changing the WGSL kernels' own reduction order, i.e. changing
// physics). No physics/thresholds/potentials/bond-rules changed anywhere -- same formulas, same
// data/params.json constants (AREA_PER_LIPID matches the committed test's own header), only NEW
// seeds and (for the diagnostic run only) a NEW equilibration budget, both plain test parameters.

const AREA_PER_LIPID = 1.209
const LIPIDS = 1000
const LY = 26
const LZ = 40
const L0 = (LIPIDS / 2) * AREA_PER_LIPID / LY
const DELTA_L = 0.25
const STEPS_COMPRESSED = 10
const SAMPLE_COUNT = 20
const SAMPLE_SPACING = 250
const ONSET_EXCLUDE = 2

const OUT_DIR = 'verify/out'
const OUT_FILE = `${OUT_DIR}/buckling-kappa-tighten.json`

// Historical values from kappa-lambda-report.md SS3 ("A resource-budget honesty note" table),
// the three independent full runs already executed and reported for this same protocol -- carried
// forward here as prior data to COMBINE with, not re-measured (re-running them would not add new
// information; they are exactly reproducible in their OWN report, not in absolute value, since this
// engine is not bit-reproducible run to run, but the historical numbers themselves are fixed record).
const HISTORICAL_KAPPA_STARTK2_3: { run: number; startK2: number; startK3: number }[] = [
  { run: 1, startK2: 101.7, startK3: 88.07 },
  { run: 2, startK2: 55.05, startK3: 43.93 },
  { run: 3, startK2: 47.35, startK3: 45.67 },
]

function fitLineWeighted(pts: { Lx: number; mean: number; se: number }[]): { slope: number; slopeSe: number } {
  let sw = 0,
    swx = 0,
    swy = 0,
    swxx = 0,
    swxy = 0
  for (const p of pts) {
    const w = 1 / (p.se * p.se)
    sw += w
    swx += w * p.Lx
    swy += w * p.mean
    swxx += w * p.Lx * p.Lx
    swxy += w * p.Lx * p.mean
  }
  const denom = sw * swxx - swx * swx
  const slope = (sw * swxy - swx * swy) / denom
  const slopeSe = Math.sqrt(sw / denom)
  return { slope, slopeSe }
}

interface RunResult {
  seed: number
  equilSteps: number
  wallClockMs: number
  points: { k: number; Lx: number; mean: number; sd: number; se: number }[]
  windows: { startK: number; n: number; LxMid: number; kappaEst: number; kappaSe: number }[]
  startK2: number
  startK3: number
  spreadPct: number
}

async function runBucklingSweep(page: Awaited<ReturnType<typeof gpuPage>>, seed: number, equilSteps: number): Promise<RunResult> {
  const wallClockStart = Date.now()
  const points = await page.evaluate(
    async (
      lipids: number,
      l0: number,
      ly: number,
      lz: number,
      seedArg: number,
      deltaL: number,
      stepsCompressed: number,
      equilStepsArg: number,
      sampleCount: number,
      sampleSpacing: number,
    ) => {
      const api = (window as any).api

      async function samplePotentials(sys: any): Promise<number[]> {
        await sys.step(equilStepsArg)
        const N = sys.lipids * 3
        const out: number[] = []
        for (let i = 0; i < sampleCount; i++) {
          if (i > 0) await sys.step(sampleSpacing)
          const total = await sys.totalEnergy()
          const keDof = await sys.kineticEnergyPerDof()
          out.push(total - 1.5 * N * keDof)
        }
        return out
      }

      let box: [number, number, number] = [l0, ly, lz]
      let sys = await api.createSystem({ lipids, box, seed: seedArg, layout: 'bilayer' })
      const out: { k: number; Lx: number; potentials: number[] }[] = []
      for (let k = 0; k <= stepsCompressed; k++) {
        if (k > 0) {
          const newBox: [number, number, number] = [l0 - k * deltaL, ly, lz]
          const prevPositions = await sys.positions()
          const rescaled = api.scaleLateralRigid(prevPositions, box, newBox, lipids)
          sys = await api.createSystem({ lipids, box: newBox, seed: seedArg, layout: 'bilayer', positions: rescaled })
          box = newBox
        }
        const potentials = await samplePotentials(sys)
        out.push({ k, Lx: box[0], potentials })
      }
      return out
    },
    LIPIDS,
    L0,
    LY,
    LZ,
    seed,
    DELTA_L,
    STEPS_COMPRESSED,
    equilSteps,
    SAMPLE_COUNT,
    SAMPLE_SPACING,
  )
  const wallClockMs = Date.now() - wallClockStart

  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
  const sd = (xs: number[]) => {
    const m = mean(xs)
    return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
  }
  const stats = (points as { k: number; Lx: number; potentials: number[] }[]).map((pt) => {
    const m = mean(pt.potentials)
    const s = sd(pt.potentials)
    return { k: pt.k, Lx: pt.Lx, mean: m, sd: s, se: s / Math.sqrt(pt.potentials.length) }
  })

  const windows = [ONSET_EXCLUDE, ONSET_EXCLUDE + 1, ONSET_EXCLUDE + 2].map((startK) => {
    const pts = stats.filter((s) => s.k >= startK)
    const { slope, slopeSe } = fitLineWeighted(pts)
    const LxMid = mean(pts.map((p) => p.Lx))
    const tau = -slope / LY
    const tauSe = slopeSe / LY
    const kappaEst = (tau * LxMid * LxMid) / (4 * Math.PI * Math.PI)
    const kappaSe = (tauSe * LxMid * LxMid) / (4 * Math.PI * Math.PI)
    return { startK, n: pts.length, LxMid, kappaEst, kappaSe }
  })

  const startK2 = windows[0].kappaEst
  const startK3 = windows[1].kappaEst
  const spreadPct = (Math.abs(startK2 - startK3) / ((startK2 + startK3) / 2)) * 100

  console.log(
    `BUCKLING-TIGHTEN seed=${seed} equilSteps=${equilSteps} wallClockMs=${wallClockMs} ` +
      `startK2=${startK2.toFixed(2)} startK3=${startK3.toFixed(2)} spreadPct=${spreadPct.toFixed(1)}%`,
  )
  return { seed, equilSteps, wallClockMs, points: stats, windows, startK2, startK3, spreadPct }
}

function combinedStats(values: number[]): { mean: number; sd: number; se: number; n: number } {
  const n = values.length
  const mean = values.reduce((a, b) => a + b, 0) / n
  const sd = Math.sqrt(values.reduce((a, b) => a + (b - mean) ** 2, 0) / (n - 1))
  return { mean, sd, se: sd / Math.sqrt(n), n }
}

function appendAndWrite(newRuns: RunResult[]): void {
  mkdirSync(OUT_DIR, { recursive: true })
  let existing: { runs: RunResult[] } = { runs: [] }
  try {
    existing = JSON.parse(readFileSync(OUT_FILE, 'utf8'))
  } catch {
    // first write
  }
  const runs = [...existing.runs, ...newRuns]
  const allValues = [
    ...HISTORICAL_KAPPA_STARTK2_3.flatMap((h) => [h.startK2, h.startK3]),
    ...runs.flatMap((r) => [r.startK2, r.startK3]),
  ]
  const stats = combinedStats(allValues)
  writeFileSync(
    OUT_FILE,
    JSON.stringify(
      {
        gate: 'bending-modulus-kappa-tighten',
        method: 'same as tests/buckling-kappa.test.ts (Hu, Diggins & Deserno 2013), repeated at new seeds',
        historical: HISTORICAL_KAPPA_STARTK2_3,
        runs,
        combined: { ...stats, values: allValues },
      },
      null,
      2,
    ),
  )
  console.log(
    `BUCKLING-TIGHTEN combined n=${stats.n} kappaMean=${stats.mean.toFixed(3)} kappaSd=${stats.sd.toFixed(3)} kappaSe=${stats.se.toFixed(3)} ` +
      `range=[${Math.min(...allValues).toFixed(2)},${Math.max(...allValues).toFixed(2)}]`,
  )
}

test('buckling repeats A: seeds 101,102,103 (normal 20000-step equilibration, same protocol as buckling-kappa.test.ts)', async () => {
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))
  const runs: RunResult[] = []
  for (const seed of [101, 102, 103]) {
    runs.push(await runBucklingSweep(page, seed, 20_000))
  }
  appendAndWrite(runs)
  for (const r of runs) {
    expect(Number.isFinite(r.startK2)).toBe(true)
    expect(Number.isFinite(r.startK3)).toBe(true)
  }
}, 1_800_000)

test('buckling repeats B: seeds 104,105,106 (normal 20000-step equilibration, same protocol as buckling-kappa.test.ts)', async () => {
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))
  const runs: RunResult[] = []
  for (const seed of [104, 105, 106]) {
    runs.push(await runBucklingSweep(page, seed, 20_000))
  }
  appendAndWrite(runs)
  for (const r of runs) {
    expect(Number.isFinite(r.startK2)).toBe(true)
    expect(Number.isFinite(r.startK3)).toBe(true)
  }
}, 1_800_000)

test('equilibration diagnostic: seed 201 at DOUBLED (40000-step) per-point equilibration -- does the answer move outside the fixed-equilibration ensemble?', async () => {
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))
  const run = await runBucklingSweep(page, 201, 40_000)
  appendAndWrite([run])
  expect(Number.isFinite(run.startK2)).toBe(true)
  expect(Number.isFinite(run.startK3)).toBe(true)
}, 1_800_000)
