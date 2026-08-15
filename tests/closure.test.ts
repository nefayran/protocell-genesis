import { afterAll, expect, test } from 'vitest'
import { enclosedVolume, occupancy } from '../engine/src/closure'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

function shell(radius: number, thickness: number, count: number): Float32Array {
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    const u = Math.random() * 2 - 1
    const phi = Math.random() * 2 * Math.PI
    const r = radius + (Math.random() - 0.5) * thickness
    const s = Math.sqrt(1 - u * u)
    out.push(20 + r * s * Math.cos(phi), 20 + r * s * Math.sin(phi), 20 + r * u, 1)
  }
  return new Float32Array(out)
}

test('замкнутая оболочка даёт полость близкую к объёму шара', () => {
  const box: [number, number, number] = [40, 40, 40]
  const occ = occupancy(shell(8, 1.5, 40_000), box, 0.5, 0.6)
  const v = enclosedVolume(occ, [80, 80, 80], 0.5)
  const ideal = (4 / 3) * Math.PI * 7.5 ** 3
  expect(v).toBeGreaterThan(ideal * 0.7)
  expect(v).toBeLessThan(ideal * 1.3)
})

test('плоский лист полости не даёт', () => {
  const box: [number, number, number] = [40, 40, 40]
  const pos: number[] = []
  for (let i = 0; i < 20_000; i++) pos.push(Math.random() * 40, Math.random() * 40, 20 + (Math.random() - 0.5), 1)
  const occ = occupancy(new Float32Array(pos), box, 0.5, 0.6)
  expect(enclosedVolume(occ, [80, 80, 80], 0.5)).toBe(0)
})

test('версия на GPU совпадает с эталоном на TypeScript', async () => {
  const page = await gpuPage()
  const rel = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 2000, box: [40, 40, 40], seed: 23, layout: 'vesicle' })
    const cpu = await api.enclosedVolumeCpu(sys, { cell: 0.5, radius: 0.6 })
    const gpu = await api.enclosedVolumeGpu(sys, { cell: 0.5, radius: 0.6 })
    return Math.abs(gpu - cpu) / Math.max(cpu, 1)
  })
  expect(rel).toBeLessThan(0.01)
})
