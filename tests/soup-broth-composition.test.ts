import { afterAll, expect, test } from 'vitest'
import { loadSoup } from '../soup/src/rules'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'broth-composition' (2026-08-18). See
// .superpowers/sdd/2026-08-16-soup-to-vesicle/broth-composition-report.md for the full measurement
// trail this file's assertions are drawn from.

test('стартовый состав содержит явную воду и не содержит готового амфифила', () => {
  const s = loadSoup()
  expect(s.start.W).toBeGreaterThan(0) // the medium the user asked for -- "естественный бульон"
  // Every start id is a monomer, not a pre-assembled chain -- already covered by
  // tests/soup-rules.test.ts's own "только мономеры" test; this test adds the water-specific check
  // that test predates.
  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) expect(ids.has(k)).toBe(true)
})

test('соль/pH: файл честно объявляет их непредставимыми, а не имитирует', () => {
  const s = loadSoup()
  expect(s.saltPhLimitation).toBeDefined()
  expect(s.saltPhLimitation!.represented).toBe(false)
  expect(s.saltPhLimitation!.basis.length).toBeGreaterThan(10)
})

// The following is the MEASURED, negative finding this task's own report documents: with explicit
// water present, chain termination (co_bond, the reaction that caps a growing chain with a head)
// did not fire even once across every composition and step budget tested (up to 150000 steps) --
// diagnosed as water's own hydrophilic pull on the head group (a new interaction, absent from every
// prior rate calibration in this project's history, all of which predate explicit water)
// suppressing the head-to-growing-tip encounter rate far below what co_bond.attemptRate was tuned
// against. This test captures that CURRENT, reported behaviour at the shipped start composition, at
// a budget (20000 steps) cheap enough to run routinely -- not a design goal, a documented limitation
// that a future fix (recalibrating co_bond's rate for the water-present encounter geometry, out of
// this task's budget) should make FAIL, at which point this assertion must be revisited, not
// silently loosened.
test(
  'находка: с текущим составом+водой естественный рост не закрывает ни одной цепи головой за 20000 шагов',
  async () => {
    const page = await gpuPage()
    const r = await page.evaluate(async (kT: number, steps: number) => {
      const api = (window as any).api
      const soup = api.loadSoup()
      const sys = await api.createSoup({ box: [30, 30, 30], seed: 41, kT })
      await sys.step(steps)
      const particles: Float32Array = await sys.particles()
      const bonds: Uint32Array = await sys.bonds()
      const amph = api.findAmphiphiles(particles, bonds, soup.monomers)
      const events = await sys.events()
      return {
        amphiphileCount: amph.length,
        ccBondEvents: events['cc_bond'] ?? 0,
        coBondEvents: events['co_bond'] ?? 0,
      }
    }, 1.1, 20000)

    console.log('BROTH-GROWTH-FINDING', JSON.stringify(r))
    expect(r.ccBondEvents).toBeGreaterThan(0) // chains DO start growing
    expect(r.coBondEvents).toBe(0) // but never get capped -- the documented finding
    expect(r.amphiphileCount).toBe(0) // so no amphiphile is ever recognised via natural growth
  },
  120_000,
)
