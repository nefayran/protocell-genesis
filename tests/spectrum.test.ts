import { expect, test } from 'vitest'
import { fitBendingModulus, spectrum, synthesizeHeightField } from '../engine/src/spectrum'

test('оценка κ восстанавливает величину, заложенную в синтетическое поле', () => {
  const n = 64
  const box: [number, number, number] = [40, 40, 40]
  const kT = 1.1
  for (const kappaTrue of [8, 20, 40]) {
    const h = synthesizeHeightField(n, box, kappaTrue, kT, 42)
    const s = spectrum(h, n, box)
    const kappa = fitBendingModulus(s, kT, box[0] * box[1], (2 * Math.PI * 8) / box[0])
    expect(kappa).toBeGreaterThan(kappaTrue * 0.85)
    expect(kappa).toBeLessThan(kappaTrue * 1.15)
  }
})
