import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// This test's two step(50_000) runs measured ~103-105s each (see task-2-debug-report.md), ~208s
// total -- comfortably over vitest's default 120_000ms testTimeout even when healthy, which is what
// made a genuine hang (see soup/src/sim.ts's STEP_CHUNK comment) indistinguishable from "just slow"
// until timed directly. 450_000ms gives >2x headroom over the measured total for a slower machine.
test(
  'связи образуются только на каталитическом центре там, где правило это требует',
  async () => {
    const page = await gpuPage()
    const r = await page.evaluate(async () => {
      const api = (window as any).api
      const withM = await api.createSoup({ box: [30, 30, 30], seed: 4, kT: 1.1, catalystCount: 200 })
      await withM.step(50000)
      const a = await withM.events()
      const noM = await api.createSoup({ box: [30, 30, 30], seed: 4, kT: 1.1, catalystCount: 0 })
      await noM.step(50000)
      const b = await noM.events()
      return { withCatalyst: a['cc_bond'] ?? 0, without: b['cc_bond'] ?? 0 }
    })
    expect(r.withCatalyst).toBeGreaterThan(100)
    expect(r.without).toBe(0)
  },
  450_000,
)

test('число мономеров каждого сорта и заряд сохраняются при работающих реакциях', async () => {
  const page = await gpuPage()
  const inv = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSoup({ box: [30, 30, 30], seed: 7, kT: 1.1 })
    const before = await sys.invariants()
    await sys.step(50000)
    const after = await sys.invariants()
    return { before, after }
  })
  expect(inv.after.monomers).toEqual(inv.before.monomers)
  expect(inv.after.charge).toBe(inv.before.charge)
  expect(inv.after.bonds).toBeGreaterThan(0)
})

test('при высокой температуре связей меньше, чем при низкой — равновесие определяется энергией', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const out: Record<string, number> = {}
    for (const kT of [0.9, 1.8]) {
      const sys = await api.createSoup({ box: [30, 30, 30], seed: 11, kT })
      await sys.step(80000)
      out[String(kT)] = (await sys.invariants()).bonds
    }
    return out
  })
  expect(r['1.8']).toBeLessThan(r['0.9'])
}, 1_800_000)
