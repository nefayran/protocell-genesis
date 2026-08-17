import { expect, test } from 'vitest'
import {
  computeDryBox,
  cyclePhaseAt,
  nextCycleTransition,
  scaleMoleculesRigid,
  type CycleSchedule,
} from '../soup/src/sim'

// Task 'wet-dry-cycle': pure, GPU-free checks that a box change (the dry-wet cycle's own mechanism)
// preserves every intramolecular distance exactly -- the discipline this project's engine/src/sim.ts
// already established for the membrane's fixed 3-bead lipids (tests/sim.test.ts's own
// "карта area move сохраняет все внутримолекулярные расстояния точно"), reused here for the soup's
// dynamic bond-graph topology (variable-size connected components, including unbonded singletons).

function mi(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

function periodicDistance(pos: Float32Array, box: [number, number, number], i: number, j: number): number {
  const dx = mi(pos[i * 4] - pos[j * 4], box[0])
  const dy = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1])
  const dz = mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

// A synthetic snapshot covering every topology findAmphiphiles()/the bond graph can produce:
//  - particles 0,1,2: a 2-tailed amphiphile (head=1, tails 0 and 2) -- a branch point.
//  - particles 3,4,5,6: a plain 4-bead chain (no head at all -- a bare carbon chain).
//  - particle 7: an unbonded singleton monomer (a free catalyst/monomer never claimed by anything).
//  - particles 8,9: a 2-bead pair straddling a periodic boundary (bead 9's raw x is near box[0],
//    bead 8's near 0 -- the mi1()-relative-offset construction is exactly what this exercises; a
//    naive "rescale the raw coordinate" map would stretch this bond by nearly a full box length).
function buildFixture(): { positions: Float32Array; bonds: Uint32Array; box: [number, number, number] } {
  const box: [number, number, number] = [20, 20, 20]
  const n = 10
  const positions = new Float32Array(n * 4)
  const set = (i: number, x: number, y: number, z: number) => {
    positions[i * 4] = x
    positions[i * 4 + 1] = y
    positions[i * 4 + 2] = z
    positions[i * 4 + 3] = 0
  }
  set(0, 5.0, 5.0, 5.0) // tail 1
  set(1, 6.0, 5.0, 5.0) // head (branch point)
  set(2, 6.0, 6.0, 5.0) // tail 2
  set(3, 10.0, 10.0, 10.0)
  set(4, 11.0, 10.0, 10.0)
  set(5, 11.0, 11.0, 10.0)
  set(6, 11.0, 11.0, 11.0)
  set(7, 2.0, 17.0, 3.0) // unbonded singleton
  set(8, 0.4, 8.0, 8.0) // straddles x=0
  set(9, 19.6, 8.0, 8.0) // straddles x=box[0] -- mi(8-9) is short (0.8), the raw difference is not
  const bonds = new Uint32Array([0, 1, 1, 2, 3, 4, 4, 5, 5, 6, 8, 9])
  return { positions, bonds, box }
}

function allBondDistances(positions: Float32Array, box: [number, number, number], bonds: Uint32Array): number[] {
  const out: number[] = []
  for (let k = 0; k < bonds.length; k += 2) out.push(periodicDistance(positions, box, bonds[k], bonds[k + 1]))
  return out
}

test('scaleMoleculesRigid сохраняет все внутримолекулярные (по ковалентным связям) расстояния точно, включая ветвление, свободные мономеры и связь через границу', () => {
  const { positions, bonds, box } = buildFixture()
  const before = allBondDistances(positions, box, bonds)

  // An isotropic contraction on all three axes -- exactly what one dry-wet transition applies
  // (data/soup.json's dryWetCycle.basis: isotropic, since a bulk soup has no preferred axis).
  const newBox: [number, number, number] = [box[0] * 0.929, box[1] * 0.929, box[2] * 0.929]
  const after = scaleMoleculesRigid(positions, bonds, box, newBox)
  const afterDist = allBondDistances(after, newBox, bonds)

  expect(before.length).toBe(bonds.length / 2)
  for (let k = 0; k < before.length; k++) {
    expect(afterDist[k]).toBeCloseTo(before[k], 5)
  }
})

test('scaleMoleculesRigid — инволюция: сжатие затем растяжение назад восстанавливает исходные позиции', () => {
  const { positions, bonds, box } = buildFixture()
  const newBox: [number, number, number] = [box[0] * 0.929, box[1] * 0.929, box[2] * 0.929]
  const mid = scaleMoleculesRigid(positions, bonds, box, newBox)
  const back = scaleMoleculesRigid(mid, bonds, newBox, box)

  for (let i = 0; i < positions.length; i++) {
    if ((i + 1) % 4 === 0) continue // the kind/type slot -- unrelated to the coordinate map
    expect(back[i]).toBeCloseTo(positions[i], 5)
  }
})

test('scaleMoleculesRigid реально меняет ИНТЕРмолекулярные (не связанные) расстояния — иначе это не "плотность выросла", а бесполезный no-op', () => {
  const { positions, bonds, box } = buildFixture()
  const newBox: [number, number, number] = [box[0] * 0.929, box[1] * 0.929, box[2] * 0.929]
  const after = scaleMoleculesRigid(positions, bonds, box, newBox)
  // Particle 7 (singleton) vs particle 3 (a different molecule) -- unrelated components, so their
  // separation SHOULD shrink by the box's own scale factor, unlike any bonded pair above.
  const before = periodicDistance(positions, box, 7, 3)
  const scaled = periodicDistance(after, newBox, 7, 3)
  expect(scaled).toBeLessThan(before)
})

function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

// Regression for the real bug this task's own live GPU run caught (wet-dry-cycle-report.md): a
// single-reference "offset relative to the component's first member" reading aliases once a
// component's own span exceeds half the box -- exactly what an UNCAPPED carbon chain (cc_bond has no
// length limit) can reach at an elevated dry-phase density. A 24-bead chain, spaced 1 sigma apart,
// wrapped several times around a box far smaller than its own ~23 sigma end-to-end extent, is the
// smallest fixture that reproduces it: any single fixed reference more than box/2 away from a distant
// member computes a wrong offset for that member, while a hop-by-hop bond-graph walk (this function's
// actual implementation) cannot, since every individual bond is ~1 sigma -- always far under box/2.
test('scaleMoleculesRigid: длинная цепь, обёрнутая через границу несколько раз (span > box/2), не искажает связи', () => {
  const box: [number, number, number] = [10, 10, 10]
  const beadCount = 24
  const positions = new Float32Array(beadCount * 4)
  for (let i = 0; i < beadCount; i++) {
    // Straight line, 1 sigma spacing, unwrapped span ~23 sigma against box=10 (span >> box/2=5) --
    // wrapped into [0, box) per-bead, so the raw stored coordinates jump discontinuously across the
    // periodic boundary multiple times along the chain, exactly like a real long chain drifting
    // through a periodic box over many steps.
    positions[i * 4] = wrap1(i * 1.0, box[0])
    positions[i * 4 + 1] = 5.0
    positions[i * 4 + 2] = 5.0
    positions[i * 4 + 3] = 0
  }
  const bonds = new Uint32Array(2 * (beadCount - 1))
  for (let i = 0; i < beadCount - 1; i++) {
    bonds[i * 2] = i
    bonds[i * 2 + 1] = i + 1
  }
  const before = allBondDistances(positions, box, bonds)
  // Sanity: this fixture really does exercise wrapping -- consecutive beads' RAW coordinates are not
  // monotonic (the chain crosses x=0/x=box[0] more than once), the case a naive unwrap gets wrong.
  const rawXs = Array.from({ length: beadCount }, (_, i) => positions[i * 4])
  const wrapsAcrossBoundary = rawXs.some((x, i) => i > 0 && Math.abs(x - rawXs[i - 1]) > box[0] / 2)
  expect(wrapsAcrossBoundary).toBe(true)

  const newBox: [number, number, number] = [box[0] * 0.929, box[1] * 0.929, box[2] * 0.929]
  const after = scaleMoleculesRigid(positions, bonds, box, newBox)
  const afterDist = allBondDistances(after, newBox, bonds)
  for (let k = 0; k < before.length; k++) {
    expect(afterDist[k]).toBeCloseTo(before[k], 5)
  }
})

test('computeDryBox: плотность после сжатия равна targetDryDensity, изотропно по всем трём осям', () => {
  const box: [number, number, number] = [20, 20, 20]
  const N = 3839
  const dryBox = computeDryBox(box, N, 0.6)
  const dryVolume = dryBox[0] * dryBox[1] * dryBox[2]
  expect(N / dryVolume).toBeCloseTo(0.6, 6)
  // isotropic -- every axis scaled by the identical factor.
  expect(dryBox[0]).toBeCloseTo(dryBox[1], 10)
  expect(dryBox[1]).toBeCloseTo(dryBox[2], 10)
  expect(dryBox[0]).toBeLessThan(box[0]) // a HIGHER target density must CONTRACT the box
})

test('cyclePhaseAt/nextCycleTransition: расписание wet-first-then-dry по циклам, settle wet после последнего', () => {
  const cfg: CycleSchedule = { periodSteps: 50000, dryFraction: 0.5, cycles: 5 }
  // step 0: already wet (the system's own creation box), cycle 1.
  expect(cyclePhaseAt(0, cfg)).toEqual({ phase: 'wet', cycleIndex: 1 })
  // just before the first wet->dry transition: still wet, cycle 1.
  expect(cyclePhaseAt(24999, cfg)).toEqual({ phase: 'wet', cycleIndex: 1 })
  // AT the transition step: dry, cycle 1.
  expect(cyclePhaseAt(25000, cfg)).toEqual({ phase: 'dry', cycleIndex: 1 })
  // AT the next cycle's own start: back to wet, cycle 2.
  expect(cyclePhaseAt(50000, cfg)).toEqual({ phase: 'wet', cycleIndex: 2 })
  // AT and past the final rehydration (cycles*periodSteps): settled wet, cycling over (cycleIndex 0).
  expect(cyclePhaseAt(250000, cfg)).toEqual({ phase: 'wet', cycleIndex: 0 })
  expect(cyclePhaseAt(999999, cfg)).toEqual({ phase: 'wet', cycleIndex: 0 })

  expect(nextCycleTransition(0, cfg)).toBe(25000)
  expect(nextCycleTransition(25000, cfg)).toBe(50000)
  expect(nextCycleTransition(24999, cfg)).toBe(25000)
  expect(nextCycleTransition(249999, cfg)).toBe(250000)
  expect(nextCycleTransition(250000, cfg)).toBe(Infinity) // cycling is over -- no further transitions
})
