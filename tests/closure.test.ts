import { afterAll, expect, test } from 'vitest'
import { enclosedVolume, enclosedVolumeFromPositions, occupancy } from '../engine/src/closure'
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

/** Translates every bead's x,y by (dx,dy), wrapping into [0, box) -- used to move the synthetic
 * shell's centre away from the box midpoint and toward/across a periodic face, without touching z
 * (z is never periodic in this engine, so an unbounded shift there would push beads outside the
 * domain instead of testing anything about the periodic-flood fix). */
function shiftXY(positions: Float32Array, box: [number, number, number], dx: number, dy: number): Float32Array {
  const out = positions.slice()
  for (let i = 0; i < out.length; i += 4) {
    out[i] = ((out[i] + dx) % box[0] + box[0]) % box[0]
    out[i + 1] = ((out[i + 1] + dy) % box[1] + box[1]) % box[1]
  }
  return out
}

test('замкнутая оболочка даёт полость близкую к объёму шара', () => {
  const box: [number, number, number] = [40, 40, 40]
  const occ = occupancy(shell(8, 1.5, 40_000), box, 0.5, 0.6)
  const v = enclosedVolume(occ, [80, 80, 80], 0.5)
  const ideal = (4 / 3) * Math.PI * 7.5 ** 3
  expect(v).toBeGreaterThan(ideal * 0.7)
  expect(v).toBeLessThan(ideal * 1.3)
})

test('оболочка, пересекающая границу по x, даёт ту же полость после рецентровки', () => {
  const box: [number, number, number] = [40, 40, 40]
  const centered = shell(8, 1.5, 40_000)
  const straddling = shiftXY(centered, box, -20, 0) // centre moves from x=20 to x=0: half the shell wraps to x~38-40
  const vCentered = enclosedVolumeFromPositions(centered, box, { cell: 0.5, radius: 0.6 })
  const vStraddling = enclosedVolumeFromPositions(straddling, box, { cell: 0.5, radius: 0.6 })
  // Without recentring, the open-boundary flood would seed x=0 and x=nx-1 as outside, splitting
  // this shell's two halves apart at the seam and leaking almost the whole cavity out through it
  // (this is the exact bug: v_straddling would collapse toward 0 instead of matching v_centered).
  expect(vStraddling).toBeGreaterThan(vCentered * 0.7)
  expect(vStraddling).toBeLessThan(vCentered * 1.3)
})

test('оболочка в углу коробки (x и y сразу) даёт ту же полость после рецентровки', () => {
  const box: [number, number, number] = [40, 40, 40]
  const centered = shell(8, 1.5, 40_000)
  const corner = shiftXY(centered, box, -20, -20) // centre moves from (20,20) to (0,0): both periodic faces at once
  const vCentered = enclosedVolumeFromPositions(centered, box, { cell: 0.5, radius: 0.6 })
  const vCorner = enclosedVolumeFromPositions(corner, box, { cell: 0.5, radius: 0.6 })
  expect(vCorner).toBeGreaterThan(vCentered * 0.7)
  expect(vCorner).toBeLessThan(vCentered * 1.3)
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
