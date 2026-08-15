import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Cosmetic label for the console line below (BENDING_BLOCKS lives in index.ts, not re-exported --
// the block count itself is read off kappaSd's own denominator; this asserts nothing).
const BENDING_BLOCKS_LOG = 10

// System size: 10 000 lipids on a ~78x78 box (area/lipid ~1.208, the measured plateau, so
// 10000/2*1.208 ~ 6040 sigma^2 -> L ~ 77.7 sigma). This replaces the original brief's 4000
// lipids/52 sigma box -- NOT because the physics needed changing, but because the fit window is
// now chosen BY THE DATA (selectFitWindow in spectrum.ts), and the 52 sigma box did not have
// enough independent |q| shells below the q^-4/q^-2 crossover for that window to ever satisfy its
// own minimum-shell requirement without reaching into protrusion-mode territory. A bigger box
// gives finer mode spacing (2*pi/L smaller) and hence more shells below any fixed physical q --
// the physically correct remedy for "not enough small-q modes", not a wider or more lenient fit
// window. `modes: 16` below is the grid's own Nyquist index for grid=32, i.e. a generous ceiling
// on how far the data-driven search may ever grow, not the window itself.
test('модуль изгиба бислоя попадает в измеренный диапазон 5–50 kT', async () => {
  const page = await gpuPage()
  // Forward the page's console to this process's stdout: measureBendingModulusDetailed's own
  // progress logging (BENDING-PROGRESS lines) and the pre-loop's equilibration progress below both
  // run inside the page, and without this listener they are invisible to whatever is tailing this
  // test's stdout for a long run's liveness/diagnosis.
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))

  const result = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 10000, box: [78, 78, 40], seed: 17, layout: 'bilayer' })
    const t0 = performance.now()
    for (let i = 0; i < 300; i++) {
      await sys.step(200)
      await sys.areaMove(1)
      if (i % 20 === 0 || i === 299) {
        console.log(
          `PRELOOP-PROGRESS round=${i + 1}/300 steps=${sys.steps} box=${sys.box[0].toFixed(3)} ` +
            `elapsedMs=${(performance.now() - t0).toFixed(0)}`,
        )
      }
    }
    // measureBendingModulusDetailed rather than the bare measureBendingModulus facade: same
    // computation (measureBendingModulus is a one-line wrapper around it, see index.ts), but also
    // returns the data-driven fit window, the block-averaged scatter, and the ensemble spectrum
    // itself -- all required by the brief's report contract and otherwise invisible from a bare
    // number. The gate's pass/fail still rests on exactly the number measureBendingModulus would
    // have returned (result.kappa).
    return api.measureBendingModulusDetailed(sys, { grid: 32, modes: 16, samples: 200 })
  })

  // NaN does not survive page.evaluate()'s JSON round-trip -- JSON.stringify(NaN) is "null", so
  // an invalid measurement's result.kappa arrives here as `null`, not `NaN`. Format defensively
  // rather than calling .toFixed on it directly (measured: that crashed this line's own logging
  // before it could report an invalid measurement -- exactly the run this line exists to describe).
  const fmt = (x: number | null) => (x === null || Number.isNaN(x) ? 'NaN' : x.toFixed(4))
  console.log(
    `GATE7 valid=${result.valid} kappa=${fmt(result.kappa)} +/- ${fmt(result.kappaSd)} ` +
      `(block sd, n=${BENDING_BLOCKS_LOG}) slope=${fmt(result.slope)} fitModes=${result.fitModes} ` +
      `fitShells=${result.fitShells} qMax=${fmt(result.qMax)} ceilingQMax=${fmt(result.ceilingQMax)} ` +
      `samples=${result.samples} stepsPerSample=${result.stepsPerSample}`,
  )
  // Full spectrum up to the ceiling (not just the fit window) so the q^-4 -> q^-2 crossover, if
  // any, is visible as data rather than asserted -- printed unconditionally (selectFitWindow never
  // throws, so this line always runs, valid measurement or not).
  const upToCeiling = result.spectrum.q
    .map((q: number, i: number) => ({ q, hq2: result.spectrum.hq2[i], inFit: q <= result.qMax }))
    .filter((e: { q: number }) => e.q <= result.ceilingQMax)
  console.log(`GATE7 spectrum (q<=ceilingQMax, ${upToCeiling.length} modes): ${JSON.stringify(upToCeiling)}`)

  // Two independent assertions: the data-driven window must have found a trustworthy fit at all
  // (valid), AND the resulting kappa must sit in the literature range. Splitting them means a
  // failure report distinguishes "the spectrum was never q^-4 in this window" from "it was, but
  // kappa landed outside 5-50 kT" -- two different findings, not one.
  expect(result.valid).toBe(true)
  expect(result.kappa).toBeGreaterThan(5)
  expect(result.kappa).toBeLessThan(50)
}, 1_800_000)
