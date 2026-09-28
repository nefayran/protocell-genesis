import { afterAll, expect, test } from 'vitest'
import { backmapLipid } from '../chem/src/backmap'
import { gpuPage, shutdownGpu } from './helpers/gpu'

const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

test('an unfolded molecule keeps its composition and bond lengths', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  expect(cs.length).toBe(12)
  expect(m.atoms.filter((a) => a.element === 'O').length).toBe(2)
  for (let i = 1; i < cs.length; i++) expect(d(cs[i - 1], cs[i])).toBeCloseTo(0.154, 4)
  expect(m.bonds.length).toBeGreaterThan(cs.length)
})

test('the molecule axis coincides with the head-tail direction', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  const axis = [cs[cs.length - 1][0] - cs[0][0], cs[cs.length - 1][1] - cs[0][1], cs[cs.length - 1][2] - cs[0][2]]
  const len = Math.hypot(...axis)
  expect(Math.abs(axis[2] / len)).toBeGreaterThan(0.9)
})

test('the carboxyl group sits at the head end, not at the tail end', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const os = m.atoms.filter((a) => a.element === 'O').map((a) => a.position)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  for (const o of os) expect(d(o, cs[0])).toBeLessThan(d(o, cs[cs.length - 1]))
})

test('the bead-to-nanometre scale is set explicitly and changes the size of the molecule', () => {
  const a = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const b = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 16, 0.8)
  const span = (m: typeof a) => {
    const cs = m.atoms.filter((x) => x.element === 'C').map((x) => x.position)
    return d(cs[0], cs[cs.length - 1])
  }
  expect(span(b)).toBeGreaterThan(span(a))
})

afterAll(shutdownGpu)

test('the scene draws molecules atom by atom and says so honestly', () => {
  return (async () => {
    const page = await gpuPage()
    await page.goto(new URL('/viewer/molecular.html', page.url()).href, { waitUntil: 'load' })
    await page.waitForFunction('window.molecular && window.molecular.frames > 5', { timeout: 60_000 })
    const state = await page.evaluate(() => ({
      frames: (window as any).molecular.frames,
      molecules: (window as any).molecular.molecules,
      atoms: (window as any).molecular.atoms,
      bonds: (window as any).molecular.bonds,
      badge: (window as any).molecular.reconstructionBadge as string,
    }))
    expect(state.frames).toBeGreaterThan(5)
    expect(state.atoms).toBeGreaterThan(state.molecules * 10)
    expect(state.bonds).toBeGreaterThan(state.atoms)
    expect(state.badge.toLowerCase()).toContain('reconstructed')

    const shot = await page.screenshot({ encoding: 'binary' })
    expect(shot.length).toBeGreaterThan(5000)
  })()
})

test('a one-carbon acid and a head coinciding with the tail do not crash the unfolding', () => {
  // A soup run really produces chains of length 1, and their chain axis is degenerate:
  // the last carbon IS the first. This used to throw inside vNormalize and silently killed
  // the whole run loop, leaving the UI showing "running" forever.
  const one = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 1, 0.8)
  expect(one.atoms.filter((a) => a.element === 'C').length).toBe(1)
  expect(one.atoms.every((a) => a.position.every((c) => Number.isFinite(c)))).toBe(true)

  const coincident = backmapLipid([1, 1, 1], [1, 1, 1], [1, 1, 1], 8, 0.8)
  expect(coincident.atoms.every((a) => a.position.every((c) => Number.isFinite(c)))).toBe(true)
})
