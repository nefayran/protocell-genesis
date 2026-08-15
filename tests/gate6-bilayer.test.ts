import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('готовый бислой при нулевом натяжении держит площадь и толщину из литературы', async () => {
  const page = await gpuPage()
  const m = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1000, box: [26, 26, 40], seed: 5, layout: 'bilayer' })
    // Warm-up: relax the lattice-start layout at a fixed box before sampling area. Measured
    // (task-5-report.md, ruling-2 remeasurement) that structural quantities here converge by
    // step ~3000 and stay flat out to 15000 — this is not a slow-equilibration knob, it just
    // gets the perfectly-ordered starting rods off their initial artificial configuration before
    // the area coordinate starts moving.
    await sys.step(3000)
    for (let i = 0; i < 200; i++) {
      await sys.step(100)
      await sys.areaMove(1)
    }
    return api.measureBilayer(sys)
  })
  expect(m.areaPerLipid).toBeGreaterThan(1.1)
  expect(m.areaPerLipid).toBeLessThan(1.5)
  expect(m.thickness).toBeGreaterThan(4.0)
  expect(m.thickness).toBeLessThan(6.0)
}, 600_000)
