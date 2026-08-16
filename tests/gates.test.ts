import { expect, test } from 'vitest'
import { evaluateGates } from '../verify/gates'
import { renderReport } from '../verify/report'

test('метрика в коридоре проходит, вне коридора падает', () => {
  const ok = evaluateGates({ areaPerLipid: 1.2, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(ok.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('passed')
  const bad = evaluateGates({ areaPerLipid: 0.7, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(bad.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('failed')
})

test('отсутствующая метрика даёт «недоказано», а не «пройдено»', () => {
  const r = evaluateGates({ areaPerLipid: 1.2 })
  expect(r.find((g) => g.id === 'bending-modulus')!.verdict).toBe('unproven')
  expect(r.find((g) => g.id === 'bending-modulus')!.value).toBeNull()
})

test('ворота ранга D никогда не выходят пройденными', () => {
  const r = evaluateGates({ chainToBeadMapping: 1 })
  expect(r.find((g) => g.id === 'chain-to-bead-mapping')!.verdict).toBe('unproven')
})

test('отчёт содержит вердикт, число, коридор и ссылку', () => {
  const html = renderReport(evaluateGates({ areaPerLipid: 1.2 }), { commit: 'abc123' })
  expect(html).toContain('1.1')
  expect(html).toContain('cond-mat/0509218')
  expect(html).toContain('недоказано')
})
