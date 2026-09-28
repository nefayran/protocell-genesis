import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('headless Chrome provides a WebGPU adapter and runs a compute pass', async () => {
  const page = await gpuPage()
  const info = await page.evaluate(() => (window as any).api.gpuSmoke())
  // NOT a hardcoded 'apple': this test's only job is proving a real WebGPU adapter answered and a
  // compute pass actually ran -- pinning the vendor string to this machine's own GPU would fail the
  // whole suite (this test runs first, afterAll(shutdownGpu) or not) on any other hardware, which
  // blocks anyone else from reproducing this exam on their own machine. Assert the vendor is a real,
  // non-empty string and log what it actually was, instead of asserting a value that is true here by
  // accident of which laptop ran it.
  console.log(`GPU-SMOKE vendor=${info.vendor}`)
  expect(typeof info.vendor).toBe('string')
  expect(info.vendor.length).toBeGreaterThan(0)
  expect(info.doubled).toEqual([2, 4, 6, 8])
})
