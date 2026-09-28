import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 4's full-scale test, taken verbatim from the plan (docs/superpowers/plans/2026-08-16-soup-to-vesicle.md).
// This is the ~90-minute continuous GPU run the task brief explicitly forbids starting from this
// task: "run the PILOT, not the full run... Do NOT start the full-scale run in this task." So this
// test is committed (runUntil's own "Test:" deliverable) but intentionally NOT executed here --
// see .superpowers/sdd/2026-08-16-soup-to-vesicle/task-4-pilot-report.md for the pilot that WAS run
// in its place, and for why this test is left un-run rather than weakened to pass cheaply.
// The project's answer to this test's question is no (docs/verdict.md): no campaign closed a vesicle,
// so the test is expected to fail, after a run of about 90 minutes. It is kept as the statement of the
// goal and runs only when asked for with RUN_GOAL_TEST=1.
test.skipIf(!process.env.RUN_GOAL_TEST)('a closed vesicle arises from a broth without ready-made amphiphiles', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    // clay: false (task 'clay-surface', 2026-08-19): this test was NOT run by that task (its budget is
    // millions of steps), so it is pinned clay-free rather than silently changed. A clay-assisted
    // vesicle run is its own measurement, not a side effect of adding the platelet to the file.
    const sys = await api.createSoup({ box: [40, 40, 40], seed: 19, kT: 1.1, catalystCount: 400, clay: false })
    const startAmph = (await api.stageOf(sys)).evidence.amphiphileFraction
    const run = await sys.runUntil('vesicle', { maxSteps: 4_000_000, sampleEvery: 20_000 })
    return { startAmph, ...run }
  })
  expect(r.startAmph).toBe(0)
  expect(r.trace.map((t: { stage: string }) => t.stage)).toContain('amphiphiles')
  expect(r.trace.map((t: { stage: string }) => t.stage)).toContain('bilayer')
  expect(r.reached).toBe(true)
  expect(r.trace[r.trace.length - 1].evidence.enclosedVolume).toBeGreaterThan(1)
}, 3_600_000)
