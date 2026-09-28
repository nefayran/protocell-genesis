import { expect, test } from 'vitest'
import { assertRulesConsistent, forwardBackwardRatio, loadSoup } from '../soup/src/rules'

test('every formation rule has a paired break and detailed balance', () => {
  const s = loadSoup()
  expect(() => assertRulesConsistent(s)).not.toThrow()
  for (const r of s.rules.filter((x) => x.kind === 'bond')) {
    const back = s.rules.find((x) => x.kind === 'break' && x.a === r.a && x.b === r.b)
    expect(back).toBeDefined()
    expect(forwardBackwardRatio(r)).toBeCloseTo(Math.exp(-r.energyKT), 10)
  }
})

test('all synthesis rates are declared with rank D and a justification', () => {
  for (const r of loadSoup().rules) {
    expect(r.rank).toBe('D')
    expect(r.basis.length).toBeGreaterThan(10)
  }
})

test('the starting composition contains only monomers and no ready-made amphiphiles', () => {
  const s = loadSoup()
  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) expect(ids.has(k)).toBe(true)
  expect(Object.keys(s.start).length).toBeGreaterThan(2)
})

test('an inconsistent rule set is detected', () => {
  const s = loadSoup()
  const broken = { ...s, rules: s.rules.filter((r) => r.kind !== 'break') }
  expect(() => assertRulesConsistent(broken)).toThrow(/paired break/)
})
