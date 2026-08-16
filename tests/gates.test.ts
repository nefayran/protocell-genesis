import { expect, test } from 'vitest'
import literature from '../data/literature.json'
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
  // closure carries rank D for a different reason (see verify/gates.ts's rule-1 comment): its
  // corridor is real, but no self-assembled vesicle has ever produced enclosedVolume, only a
  // synthetic test shell. A value that clears the corridor by three orders of magnitude must still
  // read as unproven, not as a pass on the strength of a fixture.
  const closure = evaluateGates({ enclosedVolume: 1284.875 }).find((g) => g.id === 'closure')!
  expect(closure.verdict).toBe('unproven')
  expect(closure.rank).toBe('D')
})

// Pins data/literature.json's corridors to the spec's published numbers (Cooke & Deserno 2005) so
// a future edit to the JSON cannot silently widen a corridor without a test failing here first --
// this is the guard rule 5 of the final review asked for: "bounds are never widened" needs
// something that actually breaks if they are.
test('literature.json держит ровно спецификационные коридоры', () => {
  const gates = (literature as { gates: Array<{ id: string; target: { min?: number; max?: number } }> }).gates
  const byId = (id: string) => gates.find((g) => g.id === id)!.target
  expect(byId('area-per-lipid')).toEqual({ min: 1.1, max: 1.5 })
  expect(byId('bilayer-thickness')).toEqual({ min: 4.0, max: 6.0 })
  expect(byId('bending-modulus')).toEqual({ min: 5, max: 50 })
})

test('отчёт содержит вердикт, число, коридор и ссылку', () => {
  const html = renderReport(evaluateGates({ areaPerLipid: 1.2 }), { commit: { sha: 'abc123', dirty: false } })
  // NOT plain '1.1': that also matches the area gate's own condition text ("kT/eps = 1.1"), so it
  // would pass even if the corridor cell were empty or wrong. The corridor's own rendered text
  // (corridorText in verify/report.ts) is "1.1 – 1.5 sigma^2" -- assert that exact, specific string.
  expect(html).toContain('1.1 – 1.5 sigma^2')
  expect(html).toContain('cond-mat/0509218')
  expect(html).toContain('недоказано')
})
