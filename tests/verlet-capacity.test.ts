import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadSoup } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { deriveListCapacity, densestDensityOf, verletListBytes } from '../soup/src/soup-plan'

afterAll(shutdownGpu)

// Task 'big-box' (2026-08-20). data/soup.json's verletList.listCapacity was 2500, set as a safe upper
// bound and never measured -- and since the list is a flat N*listCapacity*4 byte buffer against this
// device's 4 294 967 292-byte maxStorageBufferBindingSize, that one number WAS the project's particle
// ceiling (429 496) and its box ceiling (81.27 sigma at liquid-water density). The capacity is now
// DERIVED from the densest density a run reaches (soup/src/soup-plan.ts), and this file is the pin on
// the two things that must both hold for that to be safe:
//   (1) the derived number is big enough that nothing overflows in the configurations this project
//       runs -- measured, not assumed (soup/cli/measure-neighbours.ts is the full sweep);
//   (2) if it EVER is not big enough, the failure is a LOUD THROW and never a silent truncation.
// (2) is the one that needs a deliberately-overflowing run to prove, because this project has been
// bitten twice by a guard that could not fail: once by a real silent Verlet overflow, and once by
// assertVerletSafety passing forever after positions went NaN (every comparison against NaN is false).

const LIST_RANGE = (() => {
  const p = loadParams()
  const soup = loadSoup()
  return wcaCutoff(p.sigma * Math.max(...soup.monomers.map((m) => m.radiusSigma))) + p.attraction.wc + soup.verletList.skin
})()

test('ёмкость списка ВЫВОДИТСЯ из плотности, а не берётся из файла — и арифметика потолка сходится', () => {
  const soup = loadSoup()
  expect(LIST_RANGE).toBeCloseTo(4.446954, 6)

  // The campaign configuration: box 54, the arm-B composition at rho_tot 1.10091, dry-wet cycling on
  // (so the densest state is dryWetCycle.targetDryDensity = 1.34, which is what the capacity is sized
  // for, not the wet density it starts at).
  const N54 = 173353
  const box54: [number, number, number] = [54, 54, 54]
  const rhoDry = densestDensityOf(soup, N54, box54, N54, box54, true)
  expect(rhoDry).toBeCloseTo(1.34, 10)
  const cap = deriveListCapacity(soup, LIST_RANGE, N54, rhoDry)
  expect(cap.derived).toBe(true)
  expect(cap.uniform).toBeCloseTo(493.607, 3)
  expect(cap.capacity).toBe(1284)
  // Right-sized against the measured maxima (soup/cli/measure-neighbours.ts): 749 in the dry phase and
  // 866 in the settled percolating aggregate -- the aggregate, not the dry phase, is the binding
  // configuration, and 1284 covers it 1.48x over.
  expect(cap.capacity / 866).toBeGreaterThan(1.4)
  expect(cap.capacity / 749).toBeGreaterThan(1.7)

  // Without cycling the densest state is the creation composition itself.
  const rhoWet = densestDensityOf(soup, N54, box54, N54, box54, false)
  expect(rhoWet).toBeCloseTo(1.10091, 5)
  expect(deriveListCapacity(soup, LIST_RANGE, N54, rhoWet).capacity).toBe(1126) // the floor binds, 1055 derived

  // THE CEILING, both sides of the change. 2500 gave 429 496 particles; the derived capacity for a
  // pure-water run at rho_W = 0.8 is 767, which gives 1 399 924 -- 3.26x more particles, 1.49x more box.
  const BINDING_LIMIT = 4294967292
  expect(Math.floor(BINDING_LIMIT / (2500 * 4))).toBe(429496)
  // Pure water at rho_W = 0.8 derives 767 from its own density, but the CONDENSED-PHASE FLOOR (1126,
  // the largest neighbour count ever measured inside a condensed phase plus 30 %) is what a general
  // composition needs, so that is what a real system gets.
  expect(Math.ceil(2.6 * ((4 * Math.PI) / 3) * LIST_RANGE ** 3 * 0.8)).toBe(767)
  const capWater = deriveListCapacity(soup, LIST_RANGE, 21600, 0.8)
  expect(capWater.capacity).toBe(1126)
  const ceilingWater = Math.floor(BINDING_LIMIT / (capWater.capacity * 4))
  expect(ceilingWater).toBe(953589)
  expect((ceilingWater / 0.8) ** (1 / 3)).toBeCloseTo(106.029, 3)
  // The campaign's own configuration (rho_tot 1.10091, cycling to 1.34) gives capacity 1284 and
  // therefore 836 247 particles = box 91.24 sigma.
  expect(Math.floor(BINDING_LIMIT / (1284 * 4))).toBe(836247)
  expect((836247 / 1.10091) ** (1 / 3)).toBeCloseTo(91.242, 3)
  // ...and L* = 140.4-177.3 sigma (supply-window-report.md §3) is still OUTSIDE it. This is structural,
  // not a tuning shortfall: a flat per-particle list costs rho_wet*L^3 * f*(4/3)pi*R^3*rho_dry * 4
  // bytes, i.e. it grows as L^3 times the SQUARE of the density, so the capacity the physics demands
  // and the particle count the box demands rise together.
  expect((ceilingWater / 0.8) ** (1 / 3)).toBeLessThan(140.4)
  // Even at ZERO margin -- a capacity equal to the largest neighbour count ever measured, 866, which
  // would leave no headroom at all for a bigger sample -- the box tops out at 115.73 sigma of pure
  // water, still short of L*.
  expect((Math.floor(BINDING_LIMIT / (866 * 4)) / 0.8) ** (1 / 3)).toBeCloseTo(115.726, 3)
  expect(verletListBytes(483268, 1284)).toBe(2482064448)
})

test(
  'переполнение ёмкости — ГРОМКИЙ throw, а выведенная ёмкость того же бокса не переполняется',
  async () => {
    const page = await gpuPage()
    // Liquid water at rho_W = 0.8 in a box 30: measured occupancy max 304, mean 287.32 (see
    // soup/cli/measure-neighbours.ts). A capacity of 64 is therefore certain to overflow, and a
    // capacity of 767 (what the density derives) is certain not to.
    const start = { C: 0, O: 0, H: 0, M: 0, W: 21600 }
    const overflowed = await page.evaluate(async (st: Record<string, number>) => {
      const api = (window as any).api
      const sys = await api.createSoup({ box: [30, 30, 30], seed: 19, kT: 1.1, start: st, clay: false, verletOverride: { listCapacity: 64 } })
      try {
        await sys.step(1)
        return 'НЕ БРОСИЛ -- список молча усечён'
      } catch (e) {
        return (e as Error).message
      } finally {
        sys.dispose()
      }
    }, start)
    console.log(`VERLET-OVERFLOW ${overflowed}`)
    expect(overflowed).toMatch(/listCapacity=64/)
    expect(overflowed).toMatch(/недостаточно/)

    const ok = await page.evaluate(async (st: Record<string, number>) => {
      const api = (window as any).api
      const sys = await api.createSoup({ box: [30, 30, 30], seed: 19, kT: 1.1, start: st, clay: false })
      try {
        await sys.step(100)
        const occ = await sys.listOccupancyDEBUG()
        const vc = sys.verletConfig()
        return { cap: (vc as any).listCapacity, max: (occ as any).main.max, atCapacity: (occ as any).main.atCapacity, mean: (occ as any).main.mean }
      } finally {
        sys.dispose()
      }
    }, start)
    console.log(`VERLET-DERIVED-OK ${JSON.stringify(ok)}`)
    expect(ok.cap).toBe(1126)
    expect(ok.atCapacity).toBe(0)
    expect(ok.max).toBeLessThan(ok.cap)

    // THE CEILING ITSELF must also be loud. Measured before this guard existed: asking for a
    // 5 484 474 528-byte list at box 85 produced no JS error at all -- createBuffer's validation
    // failure is a console warning, every dispatch on that bind group silently became a no-op, and the
    // run returned max|F| = 0, 0 non-finite, 0 neighbours and 0.04 ms/step, i.e. it looked like a
    // spectacular success. Same silent-zeros class as engine/src/gpu.ts's readBack guard, one layer up.
    const refused = await page.evaluate(async (st: Record<string, number>) => {
      const api = (window as any).api
      try {
        const sys = await api.createSoup({ box: [30, 30, 30], seed: 19, kT: 1.1, start: st, clay: false, verletOverride: { listCapacity: 200000 } })
        sys.dispose()
        return 'НЕ БРОСИЛ -- устройство отказало бы молча'
      } catch (e) {
        return (e as Error).message
      }
    }, start)
    console.log(`VERLET-CEILING ${refused}`)
    expect(refused).toMatch(/17280000000 байт/)
    expect(refused).toMatch(/maxStorageBufferBindingSize=4294967292/)
  },
  180_000,
)
