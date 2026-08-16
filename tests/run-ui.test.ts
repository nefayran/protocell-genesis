import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Run-control page (viewer/run.html + viewer/run.ts): the user starts/pauses/stops the run
// themselves instead of a script launching a long one for them. Kept tiny on purpose -- the
// "tiny" size preset (~470 particles, box 16^3) is this page's own DEFAULT selection, and the step
// cap set below is small, so this test costs seconds, not minutes.
test('управление прогоном: старт держит счёт, пауза останавливает его, стоп останавливает навсегда', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  // Controls exist.
  for (const id of ['size-select', 'stage-select', 'step-cap', 'start-btn', 'pause-btn', 'stop-btn']) {
    const handle = await page.$(`#${id}`)
    expect(handle, `#${id} должен существовать`).not.toBeNull()
  }

  // window.runUI exposes the required shape before anything has started.
  const initialShape = await page.evaluate(() => {
    const ui = (window as any).runUI
    return {
      hasState: 'state' in ui,
      hasSteps: 'steps' in ui,
      hasStepsPerSecond: 'stepsPerSecond' in ui,
      hasStage: 'stage' in ui,
      hasEvidence: 'evidence' in ui,
      hasTrace: 'trace' in ui,
      state: ui.state,
    }
  })
  expect(initialShape.hasState).toBe(true)
  expect(initialShape.hasSteps).toBe(true)
  expect(initialShape.hasStepsPerSecond).toBe(true)
  expect(initialShape.hasStage).toBe(true)
  expect(initialShape.hasEvidence).toBe(true)
  expect(initialShape.hasTrace).toBe(true)
  expect(initialShape.state).toBe('idle')

  // Keep the run tiny: a small step cap on top of the page's own tiny/16^3 default preset.
  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '6000'
  })

  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')

  const runningState = await page.evaluate(() => ({ state: (window as any).runUI.state, steps: (window as any).runUI.steps }))
  expect(runningState.state).toBe('running')
  expect(runningState.steps).toBeGreaterThan(0)

  // --- PAUSE: steps must stop advancing ---------------------------------------------------------
  await page.click('#pause-btn')
  await page.waitForFunction('window.runUI.state === "paused"')
  // Let any in-flight step() batch (already submitted before the click) finish settling.
  await new Promise((r) => setTimeout(r, 600))
  const pausedSteps1 = await page.evaluate(() => (window as any).runUI.steps)
  await new Promise((r) => setTimeout(r, 600))
  const pausedSteps2 = await page.evaluate(() => (window as any).runUI.steps)
  const pausedState = await page.evaluate(() => (window as any).runUI.state)
  expect(pausedState).toBe('paused')
  expect(pausedSteps2).toBe(pausedSteps1)

  // --- STOP: state becomes 'stopped' and stepping has ceased for good --------------------------
  await page.click('#stop-btn')
  await page.waitForFunction('window.runUI.state === "stopped"')
  await new Promise((r) => setTimeout(r, 600))
  const stoppedSteps1 = await page.evaluate(() => (window as any).runUI.steps)
  await new Promise((r) => setTimeout(r, 600))
  const stoppedSteps2 = await page.evaluate(() => (window as any).runUI.steps)
  const stoppedState = await page.evaluate(() => (window as any).runUI.state)
  expect(stoppedState).toBe('stopped')
  expect(stoppedSteps2).toBe(stoppedSteps1)

  // Progress numbers reported at least once, honestly (not fabricated).
  const finalUI = await page.evaluate(() => {
    const ui = (window as any).runUI
    return { stepsPerSecond: ui.stepsPerSecond, traceLength: ui.trace.length, stage: ui.stage }
  })
  expect(finalUI.stepsPerSecond).toBeGreaterThan(0)
  expect(finalUI.traceLength).toBeGreaterThan(0)
  expect(typeof finalUI.stage).toBe('string')
})
