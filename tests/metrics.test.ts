import { expect, test } from 'vitest'
import {
  areaPerLipid,
  bilayerThickness,
  centerMembraneZ,
  densityProfileZ,
  dropEscapedZ,
  sumProfiles,
} from '../engine/src/metrics'

test('площадь на липид считается по половине числа липидов на слой', () => {
  expect(areaPerLipid([20, 20, 30], 1000)).toBeCloseTo(0.8, 12)
})

test('толщина берётся как расстояние между пиками плотности голов', () => {
  const box: [number, number, number] = [10, 10, 20]
  const pos: number[] = []
  for (let i = 0; i < 200; i++) {
    const z = i < 100 ? 7.5 : 12.5
    pos.push(Math.random() * 10, Math.random() * 10, z, 0)
  }
  const profile = densityProfileZ(new Float32Array(pos), box, 200)
  expect(bilayerThickness(profile)).toBeCloseTo(5, 1)
})

// --- honest-estimator requirements (review of Task 5) -----------------------------------------

/** Deterministic Gaussian-shaped head profile: for every bin, put round(amp*exp(-d^2/2s^2)) head
 * beads exactly at that bin's centre. No RNG, so the test measures the estimator, not luck. */
function gaussianHeads(centers: number[], box: [number, number, number], bins: number, sd: number, amp: number): Float32Array {
  const dz = box[2] / bins
  const pos: number[] = []
  for (let b = 0; b < bins; b++) {
    const zc = (b + 0.5) * dz
    let count = 0
    for (const c of centers) count += Math.round(amp * Math.exp(-((zc - c) ** 2) / (2 * sd * sd)))
    for (let k = 0; k < count; k++) pos.push(0, 0, zc, 0)
  }
  return new Float32Array(pos)
}

test('толщина находится точнее ширины бина: пики уточняются параболой по трём точкам', () => {
  const box: [number, number, number] = [26, 26, 40]
  const bins = 200 // dz = 0.2, i.e. bin centres land on ...,17.9,18.1,... — never on 17.93/21.07
  const truth = 21.07 - 17.93 // 3.14: deliberately not a multiple of dz, and not near a bin centre
  const profile = densityProfileZ(gaussianHeads([17.93, 21.07], box, bins, 0.35, 500), box, bins)
  const got = bilayerThickness(profile)
  // A bin-centre-only estimator can only answer 3.0 or 3.2 here; sub-bin interpolation has to land
  // much closer than half a bin to the true separation.
  expect(Math.abs(got - truth)).toBeLessThan(0.05)
  expect(Math.abs(got / (box[2] / bins) - Math.round(got / (box[2] / bins)))).toBeGreaterThan(1e-6)
})

test('вылетевшая по z бусина — ошибка, а не молчаливый краевой бин', () => {
  const box: [number, number, number] = [10, 10, 20]
  const pos = new Float32Array([5, 5, 21, 0])
  expect(() => densityProfileZ(pos, box, 200)).toThrow(/вне \[0, 20\)/)
})

test('один пик вместо двух — ошибка, а не z[0] в качестве второго пика', () => {
  const box: [number, number, number] = [10, 10, 20]
  const pos: number[] = []
  for (let i = 0; i < 100; i++) pos.push(0, 0, 10, 0)
  const profile = densityProfileZ(new Float32Array(pos), box, 200)
  expect(() => bilayerThickness(profile)).toThrow(/второй пик не найден/)
})

test('профили складываются по бинам и не смешивают разные сетки', () => {
  const box: [number, number, number] = [10, 10, 20]
  const a = densityProfileZ(new Float32Array([0, 0, 5, 0, 0, 0, 15, 1]), box, 20)
  const b = densityProfileZ(new Float32Array([0, 0, 5, 0]), box, 20)
  const s = sumProfiles([a, b])
  expect(s.head[5]).toBe(2)
  expect(s.tail[15]).toBe(1)
  expect(s.z).toEqual(a.z)
  const coarse = densityProfileZ(new Float32Array([0, 0, 5, 0]), box, 10)
  expect(() => sumProfiles([a, coarse])).toThrow(/разных сетках/)
})

test('улетевшие по z бусины отбрасываются и пересчитываются, а не портят профиль', () => {
  const box: [number, number, number] = [10, 10, 20]
  const pos = new Float32Array([0, 0, 5, 0, 0, 0, 25, 1, 0, 0, -1, 1, 0, 0, 15, 0])
  const { positions, escaped } = dropEscapedZ(pos, box)
  expect(escaped).toBe(2)
  expect(positions.length / 4).toBe(2)
  expect(Array.from(positions.slice(0, 4))).toEqual([0, 0, 5, 0])
  // and what survives is a legal input for the profile (which would otherwise throw)
  expect(() => densityProfileZ(positions, box, 20)).not.toThrow()
})

test('центрирование по z сдвигает центр масс мембраны в середину бокса', () => {
  const box: [number, number, number] = [10, 10, 20]
  const pos = new Float32Array([0, 0, 5, 0, 0, 0, 7, 1, 0, 0, 9, 1])
  const centered = centerMembraneZ(pos, box)
  const meanZ = (centered[2] + centered[6] + centered[10]) / 3
  expect(meanZ).toBeCloseTo(10, 12)
  expect(centered[6] - centered[2]).toBeCloseTo(2, 12) // relative geometry untouched
  expect(pos[2]).toBe(5) // input not mutated
})
