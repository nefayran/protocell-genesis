// Task 'big-box' (2026-08-20): WHAT THE NEIGHBOUR LIST ACTUALLY NEEDS.
//
// data/soup.json's verletList.listCapacity was 2500, set as a safe upper bound and never measured --
// and because the list is a flat `N * listCapacity * 4` byte buffer against this device's
// 4 294 967 292-byte storage-binding limit, that single unmeasured number WAS the project's particle
// ceiling (429 496) and therefore its box ceiling (81.27 sigma at liquid-water density), which is
// what put L* = 140.4-177.3 sigma (supply-window-report.md §3) out of reach.
//
// This CLI measures the real per-particle neighbour-count distribution across the configurations
// this project actually runs -- liquid water at 0.8 sigma^-3, the DRY phase of an evaporation cycle
// (the densest state the engine ever reaches, resumed from a real campaign checkpoint), a dense
// wet aggregate, the clay fixture (the one configuration already on record for overflowing 2500),
// and the long-range electrostatic head list -- plus a structure A/B (per-particle Verlet list vs
// cell-list traversal, which needs no per-particle array at all) and an N-scaling throughput probe.
//
// Run: nice -n 15 npx tsx soup/cli/measure-neighbours.ts
import { readFileSync } from 'node:fs'
import { gpuPage, shutdownGpu } from '../../tests/helpers/gpu'

const CK = 'data/checkpoints/swB54'

type Cfg = {
  label: string
  what: string
  checkpoint?: string
  box?: number
  start?: Record<string, number>
  clay?: boolean
  catalystCount?: number
  steps?: number
  electrostatics?: { enabled: boolean; pH: number; ionicStrengthMolar: number }
}

// The wet campaign composition of the last three campaigns (supply-window-report.md §5.1), scaled to
// each probe box at CONSTANT density, so a throughput number at a bigger box differs only in size.
function scaled(box: number): Record<string, number> {
  const f = (box / 54) ** 3
  return {
    C: Math.round(20084 * f),
    O: Math.round(6693 * f),
    H: Math.round(20084 * f),
    M: Math.round(521 * f),
    W: Math.round(125971 * f),
  }
}

const OCCUPANCY: Cfg[] = [
  { label: 'water-0.8', what: 'liquid water alone at rho_W=0.8, box 30', box: 30, start: { C: 0, O: 0, H: 0, M: 0, W: 21600 } },
  { label: 'coldstart-B54', what: 'the campaign composition at step 0, box 54 (rho_tot 1.10)', box: 54, start: scaled(54) },
  { label: 'dry-1.34', what: 'THE DRY PHASE of the evaporation cycle, resumed (rho_tot 1.34)', checkpoint: `${CK}/swB54-step24000.json` },
  { label: 'wet-aggregate', what: 'the settled percolating aggregate, resumed, charge on', checkpoint: `${CK}/swB54-step144400.json` },
  { label: 'clay-30', what: 'the clay fixture: box 30, platelet, catalystCount 200', box: 30, clay: true, catalystCount: 200 },
  // The DILUTE, WATER-FREE fixtures (tests/soup-checkpoint.test.ts): rho_box is only 0.135-0.174, so a
  // capacity derived from the box average is tiny -- but with no solvent the organics collapse into one
  // condensed droplet whose LOCAL density is set by the potential, not by the box. This is the
  // configuration that showed a density-derived capacity needs an absolute floor as well as a factor.
  { label: 'broth-18-dry', what: 'water-free broth, box 18, 6000 steps (condenses)', box: 18, start: { C: 250, O: 80, H: 250, M: 16, W: 0 }, steps: 6000 },
  { label: 'broth-20-dry', what: 'water-free broth, box 20, 6000 steps (condenses)', box: 20, start: { C: 450, O: 150, H: 450, M: 30, W: 0 }, steps: 6000 },
  { label: 'broth-30-dry', what: 'water-free broth at 4x the beads, box 30, 6000 steps', box: 30, start: { C: 1800, O: 600, H: 1800, M: 120, W: 0 }, steps: 6000 },
]

async function main(): Promise<void> {
  const page = await gpuPage()
  page.on('pageerror', (e) => console.error(`[measure] page error: ${e.message}`))
  try {
    const limits = await page.evaluate(async () => {
      const a = await (navigator as any).gpu.requestAdapter()
      return {
        maxStorageBufferBindingSize: a.limits.maxStorageBufferBindingSize,
        maxBufferSize: a.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: a.limits.maxStorageBuffersPerShaderStage,
        vendor: a.info?.vendor ?? '',
        architecture: a.info?.architecture ?? '',
      }
    })
    console.log(`LIMITS ${JSON.stringify(limits)}`)
    for (const cfg of OCCUPANCY) {
      const file = cfg.checkpoint ? JSON.parse(readFileSync(cfg.checkpoint, 'utf8')) : null
      const t0 = Date.now()
      const out = await page.evaluate(
        async (cfgJson: string, fileJson: string | null) => {
          const api = (window as any).api
          const c = JSON.parse(cfgJson)
          const f = fileJson ? JSON.parse(fileJson) : null
          const resume = f ? api.decodeCheckpointResume(f) : undefined
          const opts = f
            ? { box: f.config.box, seed: f.config.seed, kT: f.config.kT, start: f.config.start, catalystCount: f.config.catalystCount, dryWetCycle: f.config.dryWetCycle, dryWetCycles: f.config.dryWetCycles, evaporateSolvent: f.config.evaporateSolvent, electrostatics: f.config.electrostatics, resume, clay: false }
            : { box: [c.box, c.box, c.box], seed: 19, kT: 1.1, start: c.start, catalystCount: c.catalystCount, clay: c.clay ?? false, electrostatics: c.electrostatics }
          const sys = await api.createSoup(opts)
          try {
            if (c.steps) await sys.step(c.steps)
            return { occ: await sys.listOccupancyDEBUG(), vc: sys.verletConfig(), es: sys.electrostatics(), steps: sys.steps }
          } finally {
            sys.dispose()
          }
        },
        JSON.stringify(cfg),
        file ? JSON.stringify(file) : null,
      )
      const o = out.occ as any
      const m = o.main
      console.log(
        `OCC ${cfg.label} | ${cfg.what}\n` +
          `  N=${o.N} box=${o.box[0].toFixed(4)} rho_tot=${o.density.toFixed(5)} listRange=${o.listRange.toFixed(6)} ` +
          `uniformExpected=${o.uniformExpected.toFixed(1)} capacity(derived)=${out.vc.listCapacity} bytes=${out.vc.bytes} steps=${out.steps}\n` +
          (m
            ? `  MAIN max=${m.max} mean=${m.mean.toFixed(2)} sd=${m.sd.toFixed(2)} p50=${m.p50} p99=${m.p99} p999=${m.p999} ` +
              `atCapacity=${m.atCapacity} max/mean=${m.inhomogeneity.toFixed(4)} max/uniform=${(m.max / o.uniformExpected).toFixed(4)}\n` +
              `  MAIN-TAIL ${JSON.stringify(m.tail)}\n`
            : `  MAIN none (cell-list configuration)\n`) +
          (o.es
            ? `  ES heads=${o.es.n} listRange=${o.esListRange.toFixed(4)} capacity=${o.es.capacity} max=${o.es.max} mean=${o.es.mean.toFixed(2)} ` +
              `sd=${o.es.sd.toFixed(2)} p99=${o.es.p99} p999=${o.es.p999} atCapacity=${o.es.atCapacity} max/mean=${o.es.inhomogeneity.toFixed(4)}\n` +
              `  ES-TAIL ${JSON.stringify(o.es.tail)}\n`
            : `  ES none\n`) +
          `  measured in ${Date.now() - t0}ms`,
      )
    }

    // --- STRUCTURE A/B: per-particle Verlet list vs cell-list traversal, same state, same steps ----
    // Measured on the REAL settled wet checkpoint (12k+ bonds, charge on) -- a cold start has almost
    // no bonds and would understate both arms by the same factor the campaign's own trace shows
    // (15.8 ms/step at 1366 bonds against 25.8 at 12 670).
    const abFile = JSON.parse(readFileSync(`${CK}/swB54-step144400.json`, 'utf8'))
    for (const enabled of [true, false]) {
      const r = await page.evaluate(
        async (fileJson: string, listEnabled: boolean, n: number) => {
          const api = (window as any).api
          const f = JSON.parse(fileJson)
          const resume = api.decodeCheckpointResume(f)
          const sys = await api.createSoup({ box: f.config.box, seed: f.config.seed, kT: f.config.kT, start: f.config.start, catalystCount: f.config.catalystCount, dryWetCycle: f.config.dryWetCycle, dryWetCycles: f.config.dryWetCycles, evaporateSolvent: f.config.evaporateSolvent, electrostatics: f.config.electrostatics, resume, clay: false, verletOverride: { enabled: listEnabled } })
          try {
            await sys.step(50) // warm up: first submission carries pipeline/allocation cost
            const t0 = performance.now()
            await sys.step(n)
            const ms = (performance.now() - t0) / n
            const inv = await sys.invariants()
            return { ms, bonds: inv.bonds, vc: sys.verletConfig(), phases: await sys.stepPhasesDEBUG(20) }
          } finally {
            sys.dispose()
          }
        },
        JSON.stringify(abFile),
        enabled,
        400,
      )
      console.log(
        `AB structure=${enabled ? 'verlet-list' : 'cell-list'} msPerStep=${(r.ms as number).toFixed(4)} bonds=${r.bonds} ` +
          `listBytes=${(r.vc as any).bytes} phases=${JSON.stringify(r.phases)}`,
      )
    }

    // --- N-SCALING of throughput at CONSTANT density, cold start (no bonds yet) -------------------
    for (const box of [54, 68, 80, 85]) {
      const r = await page.evaluate(
        async (b: number, start: Record<string, number>, n: number) => {
          const api = (window as any).api
          // dryWetCycle/evaporateSolvent ON so the DERIVED capacity is the same number the campaign
          // itself runs with (densestDensityOf takes dryWetCycle.targetDryDensity into account) --
          // a throughput probe at a different capacity would measure a different memory stride.
          const sys = await api.createSoup({ box: [b, b, b], seed: 19, kT: 1.1, start, clay: false, dryWetCycle: true, dryWetCycles: 1, evaporateSolvent: true, electrostatics: { enabled: true, pH: 7.0, ionicStrengthMolar: 0.01 } })
          try {
            const relax = await sys.relaxColdStart()
            await sys.step(50)
            const t0 = performance.now()
            await sys.step(n)
            const ms = (performance.now() - t0) / n
            const inv = await sys.invariants()
            const occ = await sys.listOccupancyDEBUG()
            return { ms, bonds: inv.bonds, N: (occ as any).N, cap: (sys.verletConfig() as any).listCapacity, bytes: (sys.verletConfig() as any).bytes, maxF: relax.maxForceAfter, nonFinite: relax.nonFiniteAfter, occ }
          } finally {
            sys.dispose()
          }
        },
        box,
        scaled(box),
        200,
      )
      const m = (r.occ as any).main
      console.log(
        `SCALE box=${box} N=${r.N} msPerStep=${(r.ms as number).toFixed(4)} bonds=${r.bonds} cap=${r.cap} bytes=${r.bytes} ` +
          `maxFafterRelax=${(r.maxF as number).toExponential(4)} nonFinite=${r.nonFinite} ` +
          `occ(max=${m.max} mean=${m.mean.toFixed(2)} p999=${m.p999} atCapacity=${m.atCapacity})`,
      )
    }
  } finally {
    await shutdownGpu()
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
