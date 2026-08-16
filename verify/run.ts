// Task 9's "full exam": boots Vite and drives a real Chrome tab (the same gpuPage() helper the
// vitest suite uses -- results are read via page.evaluate(), never --dump-dom, per the measured
// trap that --dump-dom serialises the page before its async GPU work has finished) through the
// measurement scenarios for gate 6's literature-checked numbers, evaluates the gates against
// data/literature.json (verify/gates.ts), and writes the two artifacts this task's brief asks for:
// verify/out/gates.json (machine-readable) and verify/out/report.html (the honesty surface a human
// reads). The bending-modulus measurement is NOT re-run here -- it is read from the artifact
// tests/gate6-kappa.test.ts already wrote (verify/out/kappa-measurement.json), per the brief: that
// fit is 200 samples x 100 steps (~3 minutes) and its outcome (UNPROVEN, by design -- see
// task-7-report.md) does not change by running it again.
import { execSync } from 'node:child_process'
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { gpuPage, shutdownGpu } from '../tests/helpers/gpu'
import { evaluateGates, type GateResult } from './gates'
import { renderReport } from './report'

const OUT_DIR = 'verify/out'
const KAPPA_FILE = `${OUT_DIR}/kappa-measurement.json`

function gitCommit(): string {
  try {
    return execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim()
  } catch {
    return 'unknown'
  }
}

interface BilayerScenarioResult {
  areaPerLipid: number
  thickness: number
  areaPerLipidMin: number
  areaPerLipidMax: number
  lnADriftPerMove: number
  lnADriftT: number
  acceptedFraction: number
  escapedMax: number
  samples: number
  steps: number
}

// Same scenario tests/gate6-bilayer.test.ts uses for the gate's headline number: a lattice-start
// bilayer at a fixed box, relaxed to its area-per-lipid plateau at zero lateral tension, then
// measured by an ENSEMBLE AVERAGE (measureBilayerAveraged) rather than one frame -- see that
// function's doc comment in engine/src/index.ts for why a single frame decides this gate by luck.
async function runBilayerScenario(): Promise<BilayerScenarioResult> {
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[bilayer] ${msg.text()}`))
  const m = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1000, box: [26, 26, 40], seed: 5, layout: 'bilayer' })
    await sys.step(3000)
    for (let i = 0; i < 400; i++) {
      await sys.step(100)
      await sys.areaMove(1)
    }
    return api.measureBilayerAveraged(sys, { samples: 400, stepsPerSample: 100 })
  })
  console.log(
    `BILAYER-SCENARIO area=${m.areaPerLipid.toFixed(4)} thickness=${m.thickness.toFixed(4)} ` +
      `min=${m.areaPerLipidMin.toFixed(4)} max=${m.areaPerLipidMax.toFixed(4)} accepted=${m.acceptedFraction.toFixed(3)} ` +
      `escapedMax=${m.escapedMax} steps=${m.steps}`,
  )
  return m
}

interface ThroughputResult {
  stepsPerSecond: number
  beads: number
  neighborBuildMs: number
  chainToBeadMapping: number
}

// Throughput + neighbor-grid rebuild time. `System.neighborBuildMs` is overwritten by every
// INCIDENTAL rebuild inside step()/areaMove() (sim.ts's own doc comment on the field), so this
// reads it right after a DELIBERATE, explicit rebuild (`sys.forces()`, which goes through
// rebuildGridTimed() on purpose) rather than trusting whatever the last Monte Carlo trial left
// behind.
async function runThroughputScenario(): Promise<ThroughputResult> {
  const STEPS = 20_000
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[throughput] ${msg.text()}`))
  const r = await page.evaluate(async (steps: number) => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1200, box: [28, 28, 28], seed: 1, layout: 'bilayer' })
    await sys.forces()
    const neighborBuildMs = sys.neighborBuildMs
    const beadsPerLipid = (await sys.positions()).length / 4 / sys.lipids
    const t0 = performance.now()
    await sys.step(steps)
    const wallMs = performance.now() - t0
    return { stepsPerSecond: steps / (wallMs / 1000), beads: sys.lipids * 3, neighborBuildMs, beadsPerLipid, wallMs }
  }, STEPS)
  console.log(
    `THROUGHPUT-SCENARIO stepsPerSecond=${r.stepsPerSecond.toFixed(1)} beads=${r.beads} ` +
      `neighborBuildMs=${r.neighborBuildMs.toFixed(3)} wallMs=${r.wallMs.toFixed(0)} beadsPerLipid=${r.beadsPerLipid}`,
  )
  return {
    stepsPerSecond: r.stepsPerSecond,
    beads: r.beads,
    neighborBuildMs: r.neighborBuildMs,
    chainToBeadMapping: r.beadsPerLipid,
  }
}

interface ClosureScenarioResult {
  volume: number
  label: string
}

// enclosedVolumeFromPositions (closure.ts) has no GPU dependency, but closure.ts's module also
// statically imports its WGSL kernel (`?raw`) for the GPU-side driver it shares the file with --
// resolvable by Vite, not by plain Node/tsx. Run this scenario through the same Vite-served
// window.api as the others rather than importing closure.ts directly from run.ts, so the ?raw
// import resolves the same way it does for every other test in this suite.
//
// The RNG (mulberry32 -- the same construction sim.ts uses for reproducible layouts) and the shell
// construction (a hollow sphere of beads with radial jitter, same shape as closure.test.ts's
// shell() helper) are seeded rather than Math.random()-based: an artifact this report re-publishes
// on every `npm run verify` should not change for no reason every time it is regenerated. This is
// a SYNTHETIC CHECK OBJECT for the flood-fill detector, not a membrane -- see the report's
// dedicated closure note (this task's brief, rule 3) for why this number must never be quoted as a
// vesicle's interior volume: that is a different, much more leak-prone number (the self-assembled/
// vesicle-layout lattice), discussed and rejected for this purpose in task-8-report.md.
//
// Passed to page.evaluate() as a plain STRING, not a function reference: tsx's esbuild transform
// (keepNames, on unconditionally) wraps EVERY named function/const-arrow binding in a call to a
// `__name` helper it defines once at module scope -- invisible to puppeteer, which captures only a
// function ARGUMENT's own source via Function.prototype.toString(), so that helper call throws
// "ReferenceError: __name is not defined" the instant it runs in the browser. Measured through two
// failed rewrites (a `function` declaration, then a `const` arrow) before finding that both get
// the same wrapper; a literal source string is never parsed by esbuild at all, so no wrapper is
// ever injected into it, and puppeteer's Runtime.evaluate awaits the returned promise exactly as it
// would for a function argument.
async function runClosureScenario(): Promise<ClosureScenarioResult> {
  const RADIUS = 8
  const THICKNESS = 1.5
  const COUNT = 40_000
  const SEED = 2026
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[closure] ${msg.text()}`))
  const source = `(async () => {
    const api = window.api
    let a = ${SEED} >>> 0
    function rng() {
      a = (a + 0x6d2b79f5) | 0
      let t = Math.imul(a ^ (a >>> 15), 1 | a)
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296
    }
    const centre = 20
    const out = []
    for (let i = 0; i < ${COUNT}; i++) {
      const u = rng() * 2 - 1
      const phi = rng() * 2 * Math.PI
      const r = ${RADIUS} + (rng() - 0.5) * ${THICKNESS}
      const s = Math.sqrt(1 - u * u)
      out.push(centre + r * s * Math.cos(phi), centre + r * s * Math.sin(phi), centre + r * u, 1)
    }
    const positions = new Float32Array(out)
    const box = [40, 40, 40]
    return api.enclosedVolumeFromPositions(positions, box, { cell: 0.5, radius: 0.6 })
  })()`
  const volume = (await page.evaluate(source)) as number
  console.log(
    `CLOSURE-SCENARIO synthetic shell radius=${RADIUS} thickness=${THICKNESS} count=${COUNT} volume=${volume.toFixed(3)}`,
  )
  return {
    volume,
    label:
      `объём полости синтетической проверочной оболочки детектора (радиус ${RADIUS}σ, толщина ${THICKNESS}σ, ` +
      `${COUNT} бидов, seed ${SEED}) = ${volume.toFixed(3)} σ³`,
  }
}

interface KappaArtifact {
  valid: boolean
  kappa: number | null
  kappaSd: number
  slope: number
  fitModes: number
  fitShells: number
  qMax: number
  ceilingQMax: number
  spectrumTable: Array<{
    q: number
    degeneracy: number
    mean: number
    degenerateSpreadSd: number
    degenerateSpreadRel: number
    inFitWindow: boolean
  }>
  wallClockMs: number
  samples: number
  stepsPerSample: number
  grid: number
  system: { lipids: number; box: [number, number, number]; seed: number; layout: string }
  heightFieldDefinition: string
}

function loadKappaArtifact(): KappaArtifact | undefined {
  if (!existsSync(KAPPA_FILE)) {
    console.warn(
      `KAPPA: ${KAPPA_FILE} не найден -- модуль изгиба не измерялся в этом чекауте. Task 9 сам не ` +
        `перезапускает это измерение (см. бриф); выполните один раз ` +
        `"npx vitest run tests/gate6-kappa.test.ts" (~3-4 минуты), затем повторите "npm run verify".`,
    )
    return undefined
  }
  return JSON.parse(readFileSync(KAPPA_FILE, 'utf8')) as KappaArtifact
}

async function main() {
  mkdirSync(OUT_DIR, { recursive: true })

  const kappaArtifact = loadKappaArtifact()

  const bilayer = await runBilayerScenario()
  const throughput = await runThroughputScenario()
  const closure = await runClosureScenario()

  const metrics: Record<string, number> = {
    areaPerLipid: bilayer.areaPerLipid,
    thickness: bilayer.thickness,
    enclosedVolume: closure.volume,
    chainToBeadMapping: throughput.chainToBeadMapping,
  }
  // bendingModulus is intentionally OMITTED rather than set to NaN/null when the artifact has no
  // valid fit: evaluateGates already treats an absent metric as unproven with value:null, which is
  // exactly the honest outcome here -- no measurement exists to report, so none is smuggled through
  // the metrics record as a sentinel.
  if (kappaArtifact?.valid && kappaArtifact.kappa !== null) {
    metrics.bendingModulus = kappaArtifact.kappa
  }

  const results: GateResult[] = evaluateGates(metrics)

  const commit = gitCommit()
  const generatedAt = new Date().toISOString()
  const performance = {
    stepsPerSecond: throughput.stepsPerSecond,
    beads: throughput.beads,
    neighborBuildMs: throughput.neighborBuildMs,
    scenario: 'lipids=1200 (3600 бидов), box=[28,28,28], layout=bilayer, 20000 шагов',
  }

  const gatesJson = {
    generatedAt,
    commit,
    gates: results,
    metrics,
    performance,
    bilayer: {
      areaPerLipidMin: bilayer.areaPerLipidMin,
      areaPerLipidMax: bilayer.areaPerLipidMax,
      lnADriftPerMove: bilayer.lnADriftPerMove,
      lnADriftT: bilayer.lnADriftT,
      acceptedFraction: bilayer.acceptedFraction,
      escapedMax: bilayer.escapedMax,
      samples: bilayer.samples,
      steps: bilayer.steps,
    },
    closure,
    kappa: kappaArtifact ?? null,
  }
  writeFileSync(`${OUT_DIR}/gates.json`, JSON.stringify(gatesJson, null, 2))
  console.log(`WROTE ${OUT_DIR}/gates.json`)

  const html = renderReport(results, {
    commit,
    generatedAt,
    performance,
    kappaDetail: kappaArtifact,
    closureDetail: closure,
  })
  writeFileSync(`${OUT_DIR}/report.html`, html)
  console.log(`WROTE ${OUT_DIR}/report.html`)

  for (const g of results) {
    console.log(`GATE ${g.id}: value=${g.value ?? 'null'} rank=${g.rank} verdict=${g.verdict}`)
  }
}

main()
  .catch((err) => {
    console.error(err)
    process.exitCode = 1
  })
  .finally(async () => {
    await shutdownGpu()
  })
