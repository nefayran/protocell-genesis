import { expect, test } from 'vitest'
import { areaPerLipid, bilayerThickness, densityProfileZ } from '../engine/src/metrics'

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
