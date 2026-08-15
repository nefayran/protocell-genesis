import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Cosmetic label for the console line below (BENDING_BLOCKS lives in index.ts, not re-exported --
// the block count itself is read off kappaSd's own denominator; this asserts nothing).
const BENDING_BLOCKS_LOG = 10

// System size: 10 000 lipids on a ~78x78 box (area/lipid ~1.208, the measured plateau, so
// 10000/2*1.208 ~ 6040 sigma^2 -> L ~ 77.7 sigma) -- kept from the coordinator's Ruling 2 (a bigger
// box was tried; it did not fix the invalid window on its own, but there was no reason to revert
// it once the real culprit -- intra-column bead-mixing noise -- was identified separately).
//
// grid=16 (was 32) and modes=8 (the new grid's own Nyquist index, was 16): a coarser grid trades
// away high-q modes this measurement never needed (only the lowest handful of shells are ever
// used) for 4x the beads per column, halving the intra-column noise's contribution -- see
// columnNoiseStats logged below, computed on the SAME snapshot the measurement itself uses so no
// separate diagnostic-only run is spent.
test('модуль изгиба бислоя попадает в измеренный диапазон 5–50 kT', async () => {
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))

  const result = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 10000, box: [78, 78, 40], seed: 17, layout: 'bilayer' })
    for (let i = 0; i < 300; i++) {
      await sys.step(200)
      await sys.areaMove(1)
    }

    // Quantify the white-noise-floor hypothesis on ONE equilibrated snapshot before the redefined
    // measurement runs: predicted floor = (bead-count-weighted mean intra-column z variance) /
    // (mean beads per column * grid^2), for BOTH the OLD construction (all beads, 32x32 -- the one
    // that produced the ~4.5-6e-5 plateau) and the NEW one (tail-end beads only, 16x16) actually
    // used below, so the predicted improvement is visible against the same starting point.
    const raw = await sys.positions()
    const box = sys.box
    const oldStats = api.columnNoiseStats(raw, box, 32)
    const midBeads = api.tailEndBeads(raw)
    const newStats = api.columnNoiseStats(midBeads, box, 16)
    console.log(`FLOOR-PREDICT old(all-beads,32x32): ${JSON.stringify(oldStats)}`)
    console.log(`FLOOR-PREDICT new(tailEndBeads,16x16): ${JSON.stringify(newStats)}`)

    return api.measureBendingModulusDetailed(sys, { grid: 16, modes: 8, samples: 200 })
  })

  const fmt = (x: number | null) => (x === null || Number.isNaN(x) ? 'NaN' : x.toFixed(4))
  console.log(
    `GATE7 valid=${result.valid} kappa=${fmt(result.kappa)} +/- ${fmt(result.kappaSd)} ` +
      `(block sd, n=${BENDING_BLOCKS_LOG}) slope=${fmt(result.slope)} fitModes=${result.fitModes} ` +
      `fitShells=${result.fitShells} qMax=${fmt(result.qMax)} ceilingQMax=${fmt(result.ceilingQMax)} ` +
      `samples=${result.samples} stepsPerSample=${result.stepsPerSample}`,
  )
  const upToCeiling = result.spectrum.q
    .map((q: number, i: number) => ({ q, hq2: result.spectrum.hq2[i], inFit: q <= result.qMax }))
    .filter((e: { q: number }) => e.q <= result.ceilingQMax)
  console.log(`GATE7 spectrum (q<=ceilingQMax, ${upToCeiling.length} modes): ${JSON.stringify(upToCeiling)}`)

  // Per-shell spread ACROSS degenerate modes (same |q|, different (mx,my)) next to the shell mean,
  // for the lowest handful of shells: lattice symmetry requires (n,0)/(0,n)/(-n,0)/(0,-n) etc to
  // have equal ENSEMBLE averages, so a large spread here (comparable to the mean) is a
  // self-contained proof that the ensemble has not converged at that wavelength -- no reference
  // value needed, unlike comparing kappa/slope against -4.
  const byQ = new Map<string, number[]>()
  for (let i = 0; i < upToCeiling.length; i++) {
    const key = upToCeiling[i].q.toFixed(6)
    if (!byQ.has(key)) byQ.set(key, [])
    byQ.get(key)!.push(upToCeiling[i].hq2)
  }
  const shells = [...byQ.entries()].map(([q, vals]) => {
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length
    const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / vals.length)
    return { q: +q, degeneracy: vals.length, mean, sd, relSpread: sd / mean, values: vals }
  })
  shells.sort((a, b) => a.q - b.q)
  console.log(`GATE7 degenerate-mode spread (lowest 6 shells): ${JSON.stringify(shells.slice(0, 6))}`)

  // Two independent assertions: the data-driven window must have found a trustworthy fit at all
  // (valid), AND the resulting kappa must sit in the literature range.
  expect(result.valid).toBe(true)
  expect(result.kappa).toBeGreaterThan(5)
  expect(result.kappa).toBeLessThan(50)
}, 1_800_000)
