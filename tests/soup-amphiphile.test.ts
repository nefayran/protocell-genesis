import { expect, test } from 'vitest'
import { amphiphileHistogram, findAmphiphiles } from '../soup/src/amphiphile'
import { loadSoup } from '../soup/src/rules'

const P = (xs: number[][]) => new Float32Array(xs.flat())

test('цепь с одной полярной головой распознаётся как амфифил', () => {
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 2,3])
  const a = findAmphiphiles(parts, bonds, loadSoup().monomers)
  expect(a.length).toBe(1)
  expect(a[0].length).toBe(3)
})

test('цепь без головы и цепь с двумя головами амфифилами не считаются', () => {
  const m = loadSoup().monomers
  const noHead = findAmphiphiles(P([[0,0,0,0],[1,0,0,0],[2,0,0,0]]), new Uint32Array([0,1, 1,2]), m)
  expect(noHead.length).toBe(0)
  const twoHeads = findAmphiphiles(P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,1]]), new Uint32Array([0,1, 1,2, 2,3]), m)
  expect(twoHeads.length).toBe(0)
})

test('гистограмма длин считает цепи по числу углеродов', () => {
  const m = loadSoup().monomers
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0], [10,0,0,1],[11,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 3,4])
  expect(amphiphileHistogram(findAmphiphiles(parts, bonds, m))).toEqual({ 1: 1, 2: 1 })
})
