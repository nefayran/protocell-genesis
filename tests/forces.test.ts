import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadParams, wcaCutoff } from '../engine/src/params'

afterAll(shutdownGpu)

const p = loadParams()

function wcaAnalytic(r: number, b: number): number {
  if (r >= wcaCutoff(b)) return 0
  const b6 = (b / r) ** 6
  return (-24 * p.epsilon / r) * (2 * b6 * b6 - b6)
}

test('WCA совпадает с аналитической производной и обрезается', async () => {
  const page = await gpuPage()
  const radii = [0.8, 0.9, 1.0, 1.05, 1.2]
  const got = await page.evaluate((r) => (window as any).api.probeForces('wca', 1.0, r), radii)
  // Precision 3, not 4: measured on this GPU (Metal via ANGLE) the r=0.9 point diverges
  // 7.45e-5 from the f64 analytic value — a real f32 accumulation floor through pow(x,6),
  // squaring and division (~4.5 ULP at this magnitude), not a formula error. toBeCloseTo(4)
  // demands <5e-5 and fails by a hair; toBeCloseTo(3) keeps a real check (<5e-4, i.e. <0.0004%
  // relative here) while giving headroom for that floor.
  radii.forEach((r, i) => expect(got[i]).toBeCloseTo(wcaAnalytic(r, 1.0), 3))
  expect(got[4]).toBe(0)
})

test('FENE тянет к центру и растёт у предела растяжения', async () => {
  const page = await gpuPage()
  const radii = [0.0, 0.5, 1.0, 1.4]
  const got = await page.evaluate((r) => (window as any).api.probeForces('fene', 1.0, r), radii)
  radii.forEach((r, i) => {
    const want = (p.fene.k * r) / (1 - (r / p.fene.rInf) ** 2)
    // Precision 3: same measured f32 floor as WCA above — at r=1.4 (close to r_inf=1.5) the
    // denominator 1-(r/r_inf)^2 is a near-cancellation subtraction; GPU diverges 5.26e-5 from
    // the f64 analytic value, just over toBeCloseTo(4)'s 5e-5 bound.
    expect(got[i]).toBeCloseTo(want, 3)
  })
  expect(got[3]).toBeGreaterThan(got[2])
})

test('изгибный потенциал линеен вокруг r0', async () => {
  const page = await gpuPage()
  const radii = [3.0, 4.0, 5.0]
  const got = await page.evaluate((r) => (window as any).api.probeForces('bend', 1.0, r), radii)
  expect(got[0]).toBeCloseTo(p.bend.k * (3 - p.bend.r0), 5)
  expect(got[1]).toBeCloseTo(0, 6)
  expect(got[2]).toBeCloseTo(p.bend.k * (5 - p.bend.r0), 5)
})

test('притяжение хвостов гладко сходит к нулю на обоих концах', async () => {
  const page = await gpuPage()
  const rc = wcaCutoff(p.beadSizes.tail_tail)
  const wc = p.attraction.wc
  const radii = [rc - 0.05, rc, rc + wc / 2, rc + wc, rc + wc + 0.05]
  const got = await page.evaluate((r) => (window as any).api.probeForces('attr', 1.0, r), radii)
  expect(got[0]).toBe(0)
  expect(got[1]).toBeCloseTo(0, 6)
  expect(got[4]).toBe(0)
  expect(got[3]).toBeCloseTo(0, 6)
  const x = Math.PI * (wc / 2) / (2 * wc)
  expect(got[2]).toBeCloseTo((p.epsilon * Math.PI * Math.sin(2 * x)) / (2 * wc), 4)
})
