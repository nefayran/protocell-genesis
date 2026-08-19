import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { findBracketCrossingBox } from './helpers/gridSearch'

afterAll(shutdownGpu)

// Task 'grid-rebuild': the real thing, on the GPU, not just the pure coordinate map
// (tests/soup-boxcycle.test.ts's own bracket-crossing test covers scaleMoleculesRigid itself).
// findBracketCrossingBox picks the smallest wet/dry box pair that moves the neighbour grid's cell
// count into a different bracket -- exactly the defect class wet-dry-cycle-report.md measured
// failing outright at full scale (box 46, targetDryDensity=0.6: wet dims=[15,15,15] -> dry
// dims=[14,14,14]). This test creates a soup at that box with dry-wet cycling enabled, steps it
// PAST the real first wet->dry transition (so resizeSoupGrid's reallocation branch runs for real,
// through applyBoxScaleOnce -> stepCycled, not via any internal-only call), and checks the two
// things a silent buffer/bind-group desync would break: no particle gained or lost, and the
// grid-walk force kernel still agrees with the O(N^2) brute-force reference to floating-point
// tolerance -- the SAME correctness gate tests/soup-forces.test.ts applies to the unresized grid,
// now applied to a grid that has actually been reallocated and rebound mid-run.
test('box, реально пересекающий границу числа ячеек, реаллоцирует сетку: частицы и силы (грид против brute-force) сохраняются', async () => {
  const { box, start, dryBox, dryDims, wetDims } = findBracketCrossingBox()
  expect(dryDims).not.toEqual(wetDims) // sanity: this run really exercises the reallocation branch

  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async (box: [number, number, number], start: Record<string, number>) => {
    const api = (window as any).api
    // clay: false (task 'clay-surface', 2026-08-19) -- the clay-free control: a box change refuses to run
    // on a system with an immobile mineral phase, so the dry-wet grid-resize path is measured without one.
    const sys = await api.createSoup({ box, seed: 13, kT: 1.1, start, dryWetCycle: true, clay: false })
    const before = await sys.invariants()

    // data/soup.json's own schedule: the first wet->dry transition falls at
    // periodSteps*dryFraction real steps. stepCycled applies the ramp (and therefore
    // resizeSoupGrid, once per increment) automatically once the schedule reaches it -- this call
    // steps a little PAST that point so the transition has definitely landed before this test
    // reads anything back.
    const dwc = api.loadSoup().dryWetCycle
    const transitionAt = dwc.periodSteps * dwc.dryFraction
    await sys.stepCycled(transitionAt + 1)

    const boxAfter = sys.box
    const phaseAfter = sys.cyclePhase
    const after = await sys.invariants()
    const f = await sys.forces()
    const b = await sys.forcesBruteForce()
    let maxDiff = 0
    let sumAbsRef = 0
    for (let i = 0; i < f.length; i++) {
      maxDiff = Math.max(maxDiff, Math.abs(f[i] - b[i]))
      sumAbsRef += Math.abs(b[i])
    }
    return { before, after, boxAfter, phaseAfter, maxDiff, meanAbsRef: sumAbsRef / f.length, n: f.length, steps: sys.steps }
  }, box, start)

  expect(consoleWarnings, `браузер сообщил об ошибке/предупреждении GPU во время теста:\n${consoleWarnings.join('\n')}`).toEqual([])
  expect(r.phaseAfter).toBe('dry') // the transition this test exists to exercise really landed
  for (let i = 0; i < 3; i++) expect(r.boxAfter[i]).toBeCloseTo(dryBox[i], 6)
  // No particle created or destroyed by the box change/grid reallocation.
  expect(r.after.monomers).toEqual(r.before.monomers)
  expect(r.after.charge).toBe(r.before.charge)
  // The resized grid's own force kernel still agrees with the O(N^2) brute-force reference -- same
  // absolute tolerance tests/soup-forces.test.ts uses for the unresized grid.
  expect(r.maxDiff).toBeLessThan(1e-2)
})
