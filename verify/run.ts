// Task 9's "full exam": boots Vite and drives a real Chrome tab (the same gpuPage() helper the
// vitest suite uses -- results are read via page.evaluate(), never --dump-dom, per the measured
// trap that --dump-dom serialises the page before its async GPU work has finished) through the
// measurement scenarios for gate 6's literature-checked numbers, evaluates the gates against
// data/literature.json (verify/gates.ts), and writes the artifacts this task's brief asks for:
// verify/out/gates.json, verify/out/report.html and verify/out/kappa-measurement.json.
//
// The bending-modulus measurement IS run here, in this same process, deliberately -- an earlier
// version read tests/gate6-kappa.test.ts's own kappa-measurement.json off disk instead, which is a
// file ANOTHER process (vitest) can rewrite between this process rendering the report and writing
// that file, or between two `npm run verify` invocations racing a test run: a review caught exactly
// that (report showing a 58-75% spread, the same file on disk already at 2.8-64.3%; timestamps five
// seconds apart proved two different runs). Fix: obtain the measurement as ONE in-memory value in
// this process (runKappaScenario), and render the report AND write kappa-measurement.json from that
// SAME object, so the published pair is self-consistent by construction, not by luck. Every artifact
// this file writes is stamped with one runId/generatedAt pair generated once at the very top of
// main(), so any future mismatch would be visible to a reader instead of invisible.
// tests/gate6-kappa.test.ts still runs its own copy of this measurement (for its own strict pass/
// fail assertions) and writes it to a DIFFERENT path (verify/out/kappa-test-run.json) so the two
// writers can never collide on the same file again.
import { randomUUID } from 'node:crypto'
import { execSync } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { gpuPage, shutdownGpu } from '../tests/helpers/gpu'
import { collectCampaignGateInputs } from './campaign-gates'
import { evaluateGates, type GateResult } from './gates'
import { renderReport } from './report'

const OUT_DIR = 'verify/out'

interface CommitStamp {
  sha: string
  /** True iff `git status --porcelain` is non-empty at the moment this ran -- i.e. the working
   * tree that produced these artifacts differs from `sha`. This is normal, not a bug: this run is
   * meant to happen with the fix's code changes staged but not yet committed, and the final commit
   * packages the code together with the artifacts it produced -- a commit cannot contain its own
   * hash, so the honest stamp is "the last real commit, plus a flag that says more has changed
   * since". Silently reporting `sha` alone (as a previous version of this function did) reads as
   * "these numbers belong to this commit", which is false the moment the tree is dirty; the report
   * (verify/report.ts) surfaces this flag to the reader instead of hiding it. */
  dirty: boolean
}

function gitCommit(): CommitStamp {
  try {
    const sha = execSync('git rev-parse HEAD', { encoding: 'utf8' }).trim()
    const status = execSync('git status --porcelain', { encoding: 'utf8' }).trim()
    return { sha, dirty: status.length > 0 }
  } catch {
    return { sha: 'unknown', dirty: false }
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
      `cavity volume of the detector's synthetic test shell (radius ${RADIUS}σ, thickness ${THICKNESS}σ, ` +
      `${COUNT} beads, seed ${SEED}) = ${volume.toFixed(3)} σ³`,
  }
}

interface KappaArtifact {
  gate: string
  runId: string
  generatedAt: string
  valid: boolean
  kappa: number | null
  /** Omitted (not just left present-but-meaningless) when `kappa` is null: a standard deviation
   * describes the scatter of a number that exists. When the fit window is invalid there is no
   * kappa to have a spread around, so publishing kappaSd next to kappa:null would dress up a
   * non-measurement with a false air of quantified uncertainty. */
  kappaSd?: number
  slope: number
  fitModes: number
  fitShells: number
  qMax: number
  ceilingQMax: number
  literatureRange: [number, number]
  heightFieldDefinition: string
  grid: number
  system: { lipids: number; box: [number, number, number]; seed: number; layout: string }
  samples: number
  stepsPerSample: number
  wallClockMs: number
  noiseFloor: { predictedOldAllBeads32x32: unknown; predictedNewTailEndSameGrid: unknown }
  spectrumTable: Array<{
    q: number
    degeneracy: number
    mean: number
    degenerateSpreadSd: number
    degenerateSpreadRel: number
    inFitWindow: boolean
  }>
}

// Same measurement tests/gate6-kappa.test.ts runs (same system, same warm-up, same
// measureBendingModulusDetailed call, same per-shell degenerate-spread table construction) --
// duplicated rather than imported because vitest test files are not meant to be imported as
// modules, and run here (not read from that test's artifact) so the report renders from the exact
// value it also persists. Passed to page.evaluate() as a source STRING, not a function reference,
// for the same reason the closure scenario above is: tsx's esbuild `keepNames` transform wraps
// named function/const-arrow bindings (this scenario needs several, for the spectrum grouping) in
// calls to a `__name` helper invisible to a serialised function argument.
async function runKappaScenario(runId: string, generatedAt: string): Promise<KappaArtifact> {
  const LIPIDS = 10_000
  const BOX: [number, number, number] = [78, 78, 40]
  const SEED = 17
  const GRID = 16
  const MODES_CEILING = 8 // grid=16's own Nyquist index -- a generous search ceiling, not the window
  const SAMPLES = 200

  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[kappa] ${msg.text()}`))
  const source = `(async () => {
    const api = window.api
    const sys = await api.createSystem({ lipids: ${LIPIDS}, box: ${JSON.stringify(BOX)}, seed: ${SEED}, layout: 'bilayer' })
    for (let i = 0; i < 300; i++) {
      await sys.step(200)
      await sys.areaMove(1)
    }
    const raw = await sys.positions()
    const liveBox = sys.box
    const oldStats = api.columnNoiseStats(raw, liveBox, 32)
    const midBeads = api.tailEndBeads(raw)
    const newStats = api.columnNoiseStats(midBeads, liveBox, ${GRID})
    const detailed = await api.measureBendingModulusDetailed(sys, { grid: ${GRID}, modes: ${MODES_CEILING}, samples: ${SAMPLES} })

    const upToCeiling = detailed.spectrum.q
      .map((q, i) => ({ q, hq2: detailed.spectrum.hq2[i] }))
      .filter((e) => e.q <= detailed.ceilingQMax)
    const byQ = new Map()
    for (const e of upToCeiling) {
      const key = e.q.toFixed(6)
      if (!byQ.has(key)) byQ.set(key, [])
      byQ.get(key).push(e.hq2)
    }
    const spectrumTable = [...byQ.entries()]
      .map(([q, vals]) => {
        const mean = vals.reduce((a, b) => a + b, 0) / vals.length
        const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length)
        return {
          q: +q,
          degeneracy: vals.length,
          mean,
          degenerateSpreadSd: sd,
          degenerateSpreadRel: sd / mean,
          inFitWindow: +q <= detailed.qMax,
        }
      })
      .sort((a, b) => a.q - b.q)

    return {
      valid: detailed.valid,
      kappa: detailed.valid ? detailed.kappa : null,
      kappaSd: detailed.kappaSd,
      slope: detailed.slope,
      fitModes: detailed.fitModes,
      fitShells: detailed.fitShells,
      qMax: detailed.qMax,
      ceilingQMax: detailed.ceilingQMax,
      samples: detailed.samples,
      stepsPerSample: detailed.stepsPerSample,
      oldStats,
      newStats,
      spectrumTable,
    }
  })()`

  const wallClockStart = Date.now()
  const r = (await page.evaluate(source)) as {
    valid: boolean
    kappa: number | null
    kappaSd: number
    slope: number
    fitModes: number
    fitShells: number
    qMax: number
    ceilingQMax: number
    samples: number
    stepsPerSample: number
    oldStats: unknown
    newStats: unknown
    spectrumTable: KappaArtifact['spectrumTable']
  }
  const wallClockMs = Date.now() - wallClockStart

  console.log(
    `KAPPA-SCENARIO valid=${r.valid} kappa=${r.kappa ?? 'NaN'} slope=${r.slope.toFixed(4)} ` +
      `fitModes=${r.fitModes} fitShells=${r.fitShells} qMax=${r.qMax.toFixed(4)} wallClockMs=${wallClockMs}`,
  )

  return {
    gate: 'bending-modulus-kappa',
    runId,
    generatedAt,
    valid: r.valid,
    kappa: r.kappa,
    // Present only alongside an actual kappa -- see KappaArtifact.kappaSd's doc comment above.
    // `undefined` (not 0, not null) so JSON.stringify drops the key entirely rather than
    // publishing a false zero or a null that still visually implies "we tried to measure this".
    kappaSd: r.valid && r.kappa !== null ? r.kappaSd : undefined,
    slope: r.slope,
    fitModes: r.fitModes,
    fitShells: r.fitShells,
    qMax: r.qMax,
    ceilingQMax: r.ceilingQMax,
    literatureRange: [5, 50],
    heightFieldDefinition: 'tailEndBeads (tail2, the midplane-proximal bead of each lipid by construction)',
    grid: GRID,
    system: { lipids: LIPIDS, box: BOX, seed: SEED, layout: 'bilayer' },
    samples: r.samples,
    stepsPerSample: r.stepsPerSample,
    wallClockMs,
    noiseFloor: { predictedOldAllBeads32x32: r.oldStats, predictedNewTailEndSameGrid: r.newStats },
    spectrumTable: r.spectrumTable,
  }
}

async function main() {
  mkdirSync(OUT_DIR, { recursive: true })

  // Generated ONCE, before any scenario runs, and stamped onto every artifact this process writes
  // (gates.json, report.html, kappa-measurement.json) -- so if any future change reintroduces a
  // second writer for one of these paths, a mismatched runId/generatedAt makes that visible to a
  // reader instead of silently publishing two different runs as one.
  const runId = randomUUID()
  const generatedAt = new Date().toISOString()
  const commit = gitCommit()

  const bilayer = await runBilayerScenario()
  const throughput = await runThroughputScenario()
  const closure = await runClosureScenario()
  // Run LAST among the measurements (it is the slowest, ~3 minutes) but its result is used
  // identically by both writers below -- one in-memory object, no intervening read from disk.
  const kappaArtifact = await runKappaScenario(runId, generatedAt)

  // Task 'consolidation' (2026-08-20): the metrics this process measures ITSELF, in this run --
  // the solvent-free bilayer gates, the flood-fill detector's own synthetic check object, and the
  // chain-to-bead mapping. These four carry no `provenance` entry below precisely because their
  // provenance is "this run".
  const metrics: Record<string, number> = {
    areaPerLipid: bilayer.areaPerLipid,
    thickness: bilayer.thickness,
    enclosedVolume: closure.volume,
    chainToBeadMapping: throughput.chainToBeadMapping,
  }
  // ...and the metrics that cannot be measured inside one invocation of this pipeline (the
  // explicit-water patch, and everything that is a property of the box-54 campaign). Read from the
  // artifacts their own measuring code wrote, each stamped with that artifact's own mtime; a
  // MISSING artifact contributes no metric, so its gate comes out `unproven` with a reason instead
  // of carrying a stale number. See verify/campaign-gates.ts's header.
  const campaign = collectCampaignGateInputs()
  for (const [k, v] of Object.entries(campaign.metrics)) metrics[k] = v
  // bendingModulus is intentionally OMITTED rather than set to NaN/null when this run's own fit is
  // invalid: evaluateGates already treats an absent metric as unproven with value:null, which is
  // exactly the honest outcome here -- no valid measurement exists to report, so none is smuggled
  // through the metrics record as a sentinel.
  if (kappaArtifact.valid && kappaArtifact.kappa !== null) {
    metrics.bendingModulus = kappaArtifact.kappa
  }

  const results: GateResult[] = evaluateGates(metrics, { provenance: campaign.provenance, notes: campaign.notes })

  const performance = {
    stepsPerSecond: throughput.stepsPerSecond,
    beads: throughput.beads,
    neighborBuildMs: throughput.neighborBuildMs,
    scenario: 'lipids=1200 (3600 beads), box=[28,28,28], layout=bilayer, 20000 steps',
  }

  const gatesJson = {
    runId,
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
    kappa: kappaArtifact,
    // Everything needed to CHECK the campaign-sourced rows above rather than take them on trust:
    // the raw encapsulated-water count and the threshold it is a ratio of, both alpha estimates,
    // the percolation rows including their must-say-no controls, and which artifact each came from.
    campaign: { detail: campaign.detail, provenance: campaign.provenance, notes: campaign.notes },
  }

  // All three artifacts written from the SAME in-memory values (kappaArtifact, results, closure,
  // performance), one after another with nothing async in between -- no other process can rewrite
  // any of these three specific paths (tests/gate6-kappa.test.ts now writes elsewhere), so there is
  // no window left for the report and the on-disk kappa artifact to disagree.
  writeFileSync(`${OUT_DIR}/kappa-measurement.json`, JSON.stringify(kappaArtifact, null, 2))
  console.log(`WROTE ${OUT_DIR}/kappa-measurement.json`)

  writeFileSync(`${OUT_DIR}/gates.json`, JSON.stringify(gatesJson, null, 2))
  console.log(`WROTE ${OUT_DIR}/gates.json`)

  const html = renderReport(results, {
    runId,
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
