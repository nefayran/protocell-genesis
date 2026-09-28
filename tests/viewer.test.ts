import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('the viewer draws frames and shows the time-scale badge', async () => {
  const page = await gpuPage()
  // Corrected from the brief: `new URL('viewer/index.html', page.url())` resolves relative to
  // tests/runner.html (the page gpuPage() just navigated to) and would request
  // /tests/viewer/index.html -- a 404. The leading slash resolves from the origin instead.
  await page.goto(new URL('/viewer/index.html', page.url()).href, { waitUntil: 'load' })
  await page.waitForFunction('window.viewer && window.viewer.frames > 5')
  const state = await page.evaluate(() => ({
    frames: (window as any).viewer.frames,
    badge: (window as any).viewer.timeScaleBadge,
    kT: (window as any).viewer.kT,
  }))
  expect(state.frames).toBeGreaterThan(5)
  expect(state.badge).toContain('τ')
  expect(state.kT).toBe(1.1)

  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)
})
