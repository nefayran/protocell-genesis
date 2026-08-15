import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Gate 6: a bilayer held at zero lateral tension must sit at the published area per lipid and
// bilayer thickness. Both numbers are read from an EQUILIBRATED, ENSEMBLE-AVERAGED measurement, not
// from one frame at an arbitrary moment:
//  - the area coordinate is relaxed for 400 area moves before any sampling starts, and the drift of
//    ln A per move over the sampling window is measured (block-averaged standard error) so
//    "equilibrium" is demonstrated by a number instead of assumed;
//  - thickness comes from the head-density histogram accumulated over all 400 sampled
//    configurations, with each leaflet peak located by parabolic interpolation through its bin and
//    the two neighbours. One frame's estimate scatters by about +/-0.23 sigma (measured below and
//    reported by the same call), which is roughly the distance from the true value to the lower
//    literature bound — a single frame therefore decides this gate by luck, and used to: one run in
//    six landed exactly on the bound.
test('готовый бислой при нулевом натяжении держит площадь и толщину из литературы', async () => {
  const page = await gpuPage()
  const m = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1000, box: [26, 26, 40], seed: 5, layout: 'bilayer' })
    // Relax the lattice-start layout at a fixed box, then let the area coordinate relax. Measured
    // (task-5-report.md): from this start the area reaches its plateau in ~150 moves at the tuned
    // proposal width, so 400 is a generous margin before sampling begins.
    await sys.step(3000)
    for (let i = 0; i < 400; i++) {
      await sys.step(100)
      await sys.areaMove(1)
    }
    return api.measureBilayerAveraged(sys, { samples: 400, stepsPerSample: 100 })
  })

  console.log(
    `GATE6 area ${m.areaPerLipid.toFixed(4)} +/- ${m.areaPerLipidSd.toFixed(4)} (min ${m.areaPerLipidMin.toFixed(4)}, max ${m.areaPerLipidMax.toFixed(4)})  ` +
      `thickness ${m.thickness.toFixed(4)} (per-frame mean ${m.thicknessMean.toFixed(4)}, sd ${m.thicknessSd.toFixed(4)}, n=${m.samples})  ` +
      `lnA drift/move ${m.lnADriftPerMove.toExponential(3)} +/- ${m.lnADriftStdErr.toExponential(2)} (t=${m.lnADriftT.toFixed(2)})  ` +
      `accepted ${m.acceptedFraction.toFixed(3)}  escapedMax ${m.escapedMax}  box ${m.box[0].toFixed(3)}  steps ${m.steps}`,
  )

  // The literature bounds (Cooke & Deserno 2005): area per lipid 1.1-1.5 sigma^2, thickness 4-6 sigma.
  expect(m.areaPerLipid).toBeGreaterThan(1.1)
  expect(m.areaPerLipid).toBeLessThan(1.5)
  expect(m.thickness).toBeGreaterThan(4.0)
  expect(m.thickness).toBeLessThan(6.0)

  // Equilibrium, demonstrated rather than asserted: the area stays inside the corridor for EVERY
  // sample of the window, not just on average, and its drift is not a straight slide toward a bound.
  expect(m.areaPerLipidMin).toBeGreaterThan(1.1)
  expect(m.areaPerLipidMax).toBeLessThan(1.5)
  expect(Math.abs(m.lnADriftPerMove) * m.samples).toBeLessThan(0.05)

  // The profile must be built from essentially the whole membrane: evaporated lipids are dropped
  // from the histogram (they are no longer part of the bilayer), so bound how many may be dropped.
  expect(m.escapedMax).toBeLessThan(0.01 * 3 * m.lipids)
}, 600_000)

// The gate above starts inside the corridor, so on its own it cannot tell a system that CONVERGES
// to the published area from one that merely has not left yet. This test starts outside the
// corridor on both sides — one box too large, one too small — and requires the area to enter the
// corridor and stay there. Fewer lipids and a shorter window than the gate: its job is to show the
// direction of convergence, not to produce the gate's number.
test('площадь сходится в литературный коридор и из слишком большого, и из слишком малого бокса', async () => {
  const page = await gpuPage()
  const runs = await page.evaluate(async () => {
    const api = (window as any).api
    const lipids = 400
    const out: any[] = []
    // area per lipid = Lx*Ly/(lipids/2): 1.9 is well above the corridor, 0.9 below it. 0.9 rather
    // than something even smaller because the lattice layout, not the physics, sets the floor here:
    // at 0.8 the perfectly ordered start overlaps beads hard enough that the first few hundred
    // integration steps blow the configuration up (measured: total energy goes non-finite, beads
    // reach z ~ 1e14), so there is nothing left to converge. 0.9 survives the same warm-up and is
    // still clearly outside the corridor.
    for (const startArea of [1.9, 0.9]) {
      const L = Math.sqrt((startArea * lipids) / 2)
      const sys = await api.createSystem({ lipids, box: [L, L, 40], seed: 9, layout: 'bilayer' })
      await sys.step(3000)
      const series: number[] = []
      let accepted = 0
      // Three area trials per integration block instead of the gate's one: this run has to travel a
      // long way in ln A and only has to show where it ends up, not produce the gate's number.
      for (let i = 0; i < 700; i++) {
        await sys.step(100)
        accepted += (await sys.areaMove(3)) * 3
        series.push(api.areaPerLipid(sys.box, lipids))
      }
      const tail = series.slice(-200)
      out.push({
        startArea,
        start: series[0],
        series,
        tailMin: Math.min(...tail),
        tailMax: Math.max(...tail),
        tailMean: tail.reduce((a, b) => a + b, 0) / tail.length,
        acceptedFraction: accepted / (700 * 3),
      })
    }
    return out
  })

  for (const r of runs) {
    const trace = [0, 99, 199, 299, 399, 499, 599, 699].map((i) => `${i}:${r.series[i - 1 < 0 ? 0 : i - 1].toFixed(3)}`).join(' ')
    console.log(
      `CONVERGE start ${r.startArea} (first sample ${r.start.toFixed(3)})  tail mean ${r.tailMean.toFixed(4)} min ${r.tailMin.toFixed(4)} max ${r.tailMax.toFixed(4)}  accepted ${r.acceptedFraction.toFixed(3)}  trace ${trace}`,
    )
  }

  const big = runs[0]
  const small = runs[1]
  // Started outside the corridor on the correct side...
  expect(big.start).toBeGreaterThan(1.5)
  expect(small.start).toBeLessThan(1.1)
  // ...and the last 200 samples of each run sit inside it — entering AND staying, from both sides.
  expect(big.tailMin).toBeGreaterThan(1.1)
  expect(big.tailMax).toBeLessThan(1.5)
  expect(small.tailMin).toBeGreaterThan(1.1)
  expect(small.tailMax).toBeLessThan(1.5)
}, 600_000)
