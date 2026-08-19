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
// silently loosened. IT DID FAIL, on 2026-08-19, and NOT by a rate recalibration -- see the
// before/after measurement and the mechanism in the test body below.
test(
  'находка ОБНОВЛЕНА: после возврата дисперсии неполярное-неполярное рост ЗАКРЫВАЕТ цепи головой (co_bond > 0)',
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
    // Task 'hydrophobic-asymmetry' (2026-08-19). This test used to assert coBondEvents === 0 and
    // amphiphileCount === 0 -- broth-composition-report.md §5's honest finding that chains grew and
    // were NEVER capped by a head, so "mean tail length via natural catalysis" was not a number at
    // all. It was written as a live regression so that a real fix would break it. It broke.
    // MEASURED, same box/seed/kT/steps, one line changed in the physics (the restored apolar-apolar
    // dispersion attraction, data/soup.json's solvent.attractionScale.basis):
    //   before: ccBondEvents 263, coBondEvents 0,  amphiphileCount 0
    //   after:  ccBondEvents 263, coBondEvents 11, amphiphileCount 10
    // co_bond's own attemptRate was NOT re-tuned -- it is the same rank-D number as before, and
    // §9's open question 2 ("re-derive co_bond's encounter frequency with water present, by
    // measurement rather than by fitting the rate") is answered by measurement here rather than by a
    // fit. Physically: chain growth is unchanged (ccBondEvents identical, so this is not "more
    // chemistry happened"), but carbon chains and catalysts are non-polar and had lost EVERY
    // attractive term when 'explicit-water' removed the blanket non-polar attraction; with it back
    // they cohere again, and a polar head diffusing in the solvent meets a chain END at the surface
    // of such an aggregate instead of chasing a chain dispersed through the whole box. That is the
    // encounter-frequency collapse broth-composition-report.md §5 named as its best explanation,
    // now removed by its own stated cause rather than compensated by a rate.
    expect(r.ccBondEvents).toBeGreaterThan(0) // chains still start growing
    expect(r.coBondEvents).toBeGreaterThan(0) // ... and now DO get capped by a head
    expect(r.amphiphileCount).toBeGreaterThan(0) // so natural growth yields recognised amphiphiles
  },
  120_000,
)
