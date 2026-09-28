import { expect, test } from 'vitest'
import literature from '../data/literature.json'
import { evaluateGates } from '../verify/gates'
import { renderReport } from '../verify/report'

test('a metric inside the corridor passes, outside the corridor fails', () => {
  const ok = evaluateGates({ areaPerLipid: 1.2, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(ok.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('passed')
  const bad = evaluateGates({ areaPerLipid: 0.7, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(bad.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('failed')
})

test('a missing metric gives "unproven", not "passed"', () => {
  const r = evaluateGates({ areaPerLipid: 1.2 })
  expect(r.find((g) => g.id === 'bending-modulus')!.verdict).toBe('unproven')
  expect(r.find((g) => g.id === 'bending-modulus')!.value).toBeNull()
})

test('a rank-D gate never comes out passed', () => {
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
test('literature.json holds exactly the specification corridors', () => {
  const gates = (literature as { gates: Array<{ id: string; target: { min?: number; max?: number } }> }).gates
  const byId = (id: string) => gates.find((g) => g.id === id)!.target
  expect(byId('area-per-lipid')).toEqual({ min: 1.1, max: 1.5 })
  expect(byId('bilayer-thickness')).toEqual({ min: 4.0, max: 6.0 })
  expect(byId('bending-modulus')).toEqual({ min: 5, max: 50 })
})

// Task 'consolidation' (2026-08-20): the corridors this task ADDED get the same guard as the three
// above -- the project's rule is that a gate is published passed/failed/unproven and its bound is
// never widened, so every bound needs something that breaks if it moves. The two explicit-water
// gates deliberately pin to the SAME numbers as their solvent-free namesakes: they are the same
// literature corridor measured in a different medium, and letting them drift apart would turn "the
// bilayer holds in water" into "the bilayer holds in water, by a corridor of its own".
test('literature.json: the corridors added by the consolidation do not drift apart', () => {
  const gates = (literature as { gates: Array<{ id: string; target: { min?: number; max?: number }; rank: string }> }).gates
  const byId = (id: string) => gates.find((g) => g.id === id)!
  expect(byId('area-per-lipid-water').target).toEqual(byId('area-per-lipid').target)
  expect(byId('bilayer-thickness-water').target).toEqual(byId('bilayer-thickness').target)
  // A closed vesicle must hold AT LEAST the threshold's worth of solvent -- the ratio's floor is 1
  // by construction, not by choice, and the raw threshold is recomputed per snapshot.
  expect(byId('vesicle-closure-water').target).toEqual({ min: 1.0 })
  // A finite object wraps ZERO axes. Not a tolerance: the definition.
  expect(byId('aggregate-percolation').target).toEqual({ max: 0 })
  expect(byId('vesicle-verdict').target).toEqual({ min: 1.0 })
  // The chain-to-bead mapping is ours (spec §4), so the window it implies stays rank D and the gate
  // stays unproven whatever it measures -- pinned here so nobody promotes it by editing one field.
  expect(byId('mean-tail-length').target).toEqual({ min: 2.0, max: 3.0 })
  expect(byId('mean-tail-length').rank).toBe('D')
  expect(byId('chain-length-asf').rank).toBe('D')
})

// The three verdicts this project's own final numbers must produce. Written with the measured values
// inline rather than read from verify/out/gates.json on purpose: this is a test of the RULE, and it
// must fail if a future edit makes "no vesicle" or "the object percolates" read as anything but a
// failure.
test('the final campaign gates come out failed on the measured numbers, not unproven', () => {
  const r = evaluateGates({
    encapsulatedWaterOverThreshold: 0 / 320.8907983202498,
    wrappingAxes: 3,
    vesicleAggregates: 0,
  })
  const g = (id: string) => r.find((x) => x.id === id)!
  expect(g('vesicle-closure-water').verdict).toBe('failed')
  expect(g('vesicle-closure-water').rank).toBe('B')
  expect(g('aggregate-percolation').verdict).toBe('failed')
  expect(g('vesicle-verdict').verdict).toBe('failed')
  // ...and a hypothetical finite object with a full lumen would pass the same three, so the gates
  // are not simply wired to fail.
  const ok = evaluateGates({ encapsulatedWaterOverThreshold: 1.4, wrappingAxes: 0, vesicleAggregates: 1 })
  for (const id of ['vesicle-closure-water', 'aggregate-percolation', 'vesicle-verdict']) {
    expect(ok.find((x) => x.id === id)!.verdict).toBe('passed')
  }
})

test('a missing input artifact gives unproven gates with a reason, not a silent old number', () => {
  const r = evaluateGates(
    {},
    { notes: { 'area-per-lipid-water': 'artifact missing, the number is not substituted' } },
  )
  const g = r.find((x) => x.id === 'area-per-lipid-water')!
  expect(g.verdict).toBe('unproven')
  expect(g.value).toBeNull()
  expect(g.note).toContain('is not substituted')
  // provenance is attached only where the pipeline knows one, and never invented
  expect(g.provenance).toBeUndefined()
  const withProv = evaluateGates({ wrappingAxes: 3 }, { provenance: { 'aggregate-percolation': 'artifact X' } })
  expect(withProv.find((x) => x.id === 'aggregate-percolation')!.provenance).toBe('artifact X')
})

test('the report contains the verdict, the number, the corridor and the reference', () => {
  const html = renderReport(evaluateGates({ areaPerLipid: 1.2 }), { commit: { sha: 'abc123', dirty: false } })
  // NOT plain '1.1': that also matches the area gate's own condition text ("kT/eps = 1.1"), so it
  // would pass even if the corridor cell were empty or wrong. The corridor's own rendered text
  // (corridorText in verify/report.ts) is "1.1 – 1.5 sigma^2" -- assert that exact, specific string.
  expect(html).toContain('1.1 – 1.5 sigma^2')
  expect(html).toContain('cond-mat/0509218')
  // The verdict cell, not the bare word: in English 'unproven' is also the row's CSS class and appears
  // in gate condition texts, so a bare substring would pass even with the verdict label missing.
  expect(html).toContain('<td class="verdict">unproven</td>')
})
