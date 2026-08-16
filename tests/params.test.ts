import { expect, test } from 'vitest'
import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { loadParams, wcaCutoff } from '../engine/src/params'

test('параметры совпадают с опубликованными значениями Cooke & Deserno 2005', () => {
  const p = loadParams()
  expect(p.beadSizes.head_head).toBe(0.95)
  expect(p.beadSizes.head_tail).toBe(0.95)
  expect(p.beadSizes.tail_tail).toBe(1.0)
  expect(p.fene.k).toBe(30)
  expect(p.fene.rInf).toBe(1.5)
  expect(p.bend.k).toBe(10)
  expect(p.bend.r0).toBe(4)
  expect(p.attraction.wc).toBe(1.6)
  expect(p.thermostat.gamma).toBe(1)
  expect(p.thermostat.kT).toBe(1.1)
  expect(p.integrator.dt).toBe(0.01)
})

test('обрезка WCA равна 2^(1/6)·b', () => {
  expect(wcaCutoff(1)).toBeCloseTo(1.1224620483, 9)
  expect(wcaCutoff(0.95)).toBeCloseTo(0.95 * 2 ** (1 / 6), 9)
})

test('в движке нет вписанных констант модели', () => {
  // Forbidden literals with word-boundary regexes to avoid false positives
  // E.g., 31.5 contains 1.5 as substring, but regex requires no adjacent digit/dot
  const forbidden = [
    /(?<![\d.])0\.95(?![\d.])/,
    /(?<![\d.])1\.6(?![\d.])/,
    /(?<![\d.])30\.0(?![\d.])/,
    /(?<![\d.])1\.5(?![\d.])/,
    /(?<![\d.])4\.0(?![\d.])/,
    /(?<![\d.])1\.1(?![\d.])/,
    /(?<![\d.])0\.01(?![\d.])/,
  ]
  const forbiddenLiterals = ['0.95', '1.6', '30.0', '1.5', '4.0', '1.1', '0.01']
  const dirs = ['engine/src', 'engine/wgsl', 'chem/src', 'soup/src', 'soup/wgsl']
  const offenders: string[] = []
  for (const dir of dirs) {
    if (!existsSync(dir)) continue // future task adds soup/wgsl; nothing to scan until it exists
    for (const name of readdirSync(dir)) {
      if (name === 'params.ts') continue
      const text = readFileSync(join(dir, name), 'utf8')
      for (let i = 0; i < forbidden.length; i++) {
        if (forbidden[i].test(text)) offenders.push(`${dir}/${name}: ${forbiddenLiterals[i]}`)
      }
    }
  }
  expect(offenders).toEqual([])
})
