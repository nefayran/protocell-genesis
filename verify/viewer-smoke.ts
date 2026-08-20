// Closeout task (2026-08-21): the ONE headless start that proves the page a user actually opens
// still runs and still draws. Deliberately not a test -- tests/run-ui.test.ts owns the assertions;
// this prints the two numbers a reader of the closeout report needs (steps advanced, and how much
// of the canvas is not background) so "the viewer works" is a measurement and not a claim.
import { gpuPage, shutdownGpu } from '../tests/helpers/gpu'

const page = await gpuPage()
await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })
await page.$eval('#step-cap', (el) => { (el as HTMLInputElement).value = '8000' })
await page.click('#start-btn')
await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')
for (let i = 0; i < 20; i++) {
  const s = await page.evaluate(() => ({ st: (window as any).runUI.state, n: (window as any).runUI.steps, sps: (window as any).runUI.stepsPerSecond, stage: (window as any).runUI.stage }))
  console.log(`POLL ${i} state=${s.st} steps=${s.n} steps/s=${typeof s.sps === 'number' ? s.sps.toFixed(0) : s.sps} stage=${s.stage}`)
  if (s.st !== 'running') break
  await new Promise((r) => setTimeout(r, 500))
}
// "Draws" measured the way tests/run-ui.test.ts's ocean-look test measures it -- a WebGPU canvas
// read back through a 2d drawImage() comes back blank on a page that is demonstrably rendering
// (measured here first: corner [0,0,0], 0 of 30 000 pixels differing), so the instrument is
// sceneDebug's own counters plus a real page screenshot, not a pixel guess through a 2d context.
const drawn = await page.evaluate(() => ({
  renderedFrameCount: (window as any).sceneDebug.renderedFrameCount,
  instances: (window as any).sceneDebug.instanceCounts(),
}))
const shot = await page.screenshot({ encoding: 'binary' })
;(drawn as any).screenshotBytes = shot.length
console.log('DRAWN ' + JSON.stringify(drawn))
const final = await page.evaluate(() => ({ st: (window as any).runUI.state, n: (window as any).runUI.steps, stage: (window as any).runUI.stage, trace: ((window as any).runUI.trace ?? []).length }))
console.log('FINAL ' + JSON.stringify(final))
await shutdownGpu()
