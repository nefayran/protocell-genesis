import { expect, test } from 'vitest'
import { assertRulesConsistent, forwardBackwardRatio, loadSoup } from '../soup/src/rules'

test('каждое правило образования имеет парный разрыв и детальный баланс', () => {
  const s = loadSoup()
  expect(() => assertRulesConsistent(s)).not.toThrow()
  for (const r of s.rules.filter((x) => x.kind === 'bond')) {
    const back = s.rules.find((x) => x.kind === 'break' && x.a === r.a && x.b === r.b)
    expect(back).toBeDefined()
    expect(forwardBackwardRatio(r)).toBeCloseTo(Math.exp(-r.energyKT), 10)
  }
})

test('все скорости синтеза объявлены рангом D с обоснованием', () => {
  for (const r of loadSoup().rules) {
    expect(r.rank).toBe('D')
    expect(r.basis.length).toBeGreaterThan(10)
  }
})

test('стартовый состав содержит только мономеры и не содержит готовых амфифилов', () => {
  const s = loadSoup()
  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) expect(ids.has(k)).toBe(true)
  expect(Object.keys(s.start).length).toBeGreaterThan(2)
})

test('несогласованный набор правил выявляется', () => {
  const s = loadSoup()
  const broken = { ...s, rules: s.rules.filter((r) => r.kind !== 'break') }
  expect(() => assertRulesConsistent(broken)).toThrow(/разрыв/)
})
