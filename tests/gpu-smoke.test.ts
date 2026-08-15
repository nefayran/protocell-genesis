import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('headless Chrome отдаёт адаптер WebGPU и исполняет compute-проход', async () => {
  const page = await gpuPage()
  const info = await page.evaluate(() => (window as any).api.gpuSmoke())
  expect(info.vendor).toBe('apple')
  expect(info.doubled).toEqual([2, 4, 6, 8])
})
