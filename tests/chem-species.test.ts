import { expect, test } from 'vitest'
import { buildAlkanoicAcid, loadSpecies, molarMass, parseFormula } from '../chem/src/species'

const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

test('формула разбирается и даёт молярную массу', () => {
  expect(parseFormula('H2O')).toEqual({ H: 2, O: 1 })
  expect(parseFormula('C10H20O2')).toEqual({ C: 10, H: 20, O: 2 })
  expect(molarMass('H2O')).toBeCloseTo(18.015, 2)
  expect(molarMass('CO')).toBeCloseTo(28.010, 2)
})

test('геометрия воды совпадает со справочными значениями', () => {
  const w = loadSpecies()['H2O']
  const o = w.atoms.find((a) => a.element === 'O')!.position
  const hs = w.atoms.filter((a) => a.element === 'H').map((a) => a.position)
  expect(dist(o, hs[0])).toBeCloseTo(0.09572, 5)
  expect(dist(o, hs[1])).toBeCloseTo(0.09572, 5)
  const cos = ((hs[0][0] - o[0]) * (hs[1][0] - o[0]) + (hs[0][1] - o[1]) * (hs[1][1] - o[1]) + (hs[0][2] - o[2]) * (hs[1][2] - o[2])) / (0.09572 * 0.09572)
  expect((Math.acos(cos) * 180) / Math.PI).toBeCloseTo(104.52, 1)
})

test('CO и H2 двухатомны с правильной длиной связи', () => {
  const s = loadSpecies()
  expect(dist(s['CO'].atoms[0].position, s['CO'].atoms[1].position)).toBeCloseTo(0.1128, 5)
  expect(dist(s['H2'].atoms[0].position, s['H2'].atoms[1].position)).toBeCloseTo(0.0741, 5)
})

test('цепь кислоты строится зигзагом с правильными связями и составом', () => {
  const c10 = buildAlkanoicAcid(10)
  expect(parseFormula(c10.formula)).toEqual({ C: 10, H: 20, O: 2 })
  const cs = c10.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  for (let i = 1; i < cs.length; i++) expect(dist(cs[i - 1], cs[i])).toBeCloseTo(0.154, 4)
  const span = dist(cs[0], cs[cs.length - 1])
  expect(span).toBeGreaterThan(0.154 * (cs.length - 1) * 0.7)
  expect(span).toBeLessThan(0.154 * (cs.length - 1))
})

test('заряды видов взяты из данных, а не выдуманы', () => {
  const s = loadSpecies()
  expect(s['HCOO-'].charge).toBe(-1)
  expect(s['H+'].charge).toBe(1)
  expect(s['CO'].charge).toBe(0)
})
