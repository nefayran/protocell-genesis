// PINS the mechanism task 'coalescence' (2026-08-20) established --
// .superpowers/sdd/2026-08-16-soup-to-vesicle/coalescence-report.md. A real gate, not a harness: it
// needs no GPU, no env variable and no checkpoint, because every number it asserts is either in a
// committed data file, in a committed measurement artifact, or synthetic geometry.
//
// Three claims, one test each:
//  1. FREE-HEAD POISONING IS STRUCTURALLY IMPOSSIBLE IN THIS MODEL, and the measurement agrees.
//     data/soup.json's own pair-depth table gives a free polar head ZERO attraction to any bead of
//     an aggregate (polarPolar = polarApolar = 0.0), so the only free-head/aggregate interaction is
//     the WCA core -- exclusion. The measured surface enrichment of free heads relative to water is
//     therefore below 1 at every snapshot, which is what the artifact must keep saying.
//  2. THE AGGREGATES DO NOT EXCHANGE MATERIAL. Over the decisive lineage's 126 000 settled wet steps
//     the trace carries at most one merge per interval and zero fissions -- the frozen-distribution
//     claim, pinned against the artifact so a future change that unfreezes it is visible.
//  3. THE CLUSTERING RULE'S DEFAULT LEAVES z OPEN -- correct for the membrane engine (a bilayer patch
//     in vacuum has no image across z) and WRONG for the soup, which wraps all three axes, where it
//     split every z-straddling aggregate and counted it twice. Task 'final-campaign' (2026-08-20)
//     fixed that by giving buildClusterUnionFind a `periodicZ` flag every soup call site passes.
//     Both branches are pinned on synthetic geometry so neither can move silently.
//  4. AND THE SAME CLASS OF DEFECT ON x AND y, which nobody has measured because x and y were
//     periodic from the start: a blob straddling the x face and one straddling the y face must each
//     read as ONE aggregate under both branches. This is the test that would have caught the z bug
//     if it had been written for z, and it is now written for all three axes.
import { expect, test } from 'vitest'
import { existsSync, readFileSync } from 'node:fs'
import { clusterComponents } from '../engine/src/aggregate'
import { clusterComponents3D } from './helpers/coalescence-geometry'
import { loadSoup } from '../soup/src/rules'

const ARTIFACT = 'verify/out/coalescence-dec54-trace.json'

test('free-head poisoning: no attractive term exists, and the measured surface enrichment is below 1', () => {
  const soup = loadSoup() as unknown as {
    solvent: { attractionScale: { pairEpsilon: { levels: Record<string, { epsilonKJ: number }> } } }
  }
  const levels = soup.solvent.attractionScale.pairEpsilon.levels
  // A free head is polar; an aggregate is polar heads + apolar tails. Both those pair classes are 0.
  expect(levels.polarPolar.epsilonKJ, 'head-head depth').toBe(0)
  expect(levels.polarApolar.epsilonKJ, 'head-tail depth').toBe(0)
  // ... while water attracts a head at 1.14x the tail-tail reference, so water OUTCOMPETES the
  // surface for a free head. That ordering is the mechanism, and it is what the file must keep.
  expect(levels.solventPolar.epsilonKJ / levels.apolarApolar.epsilonKJ).toBeGreaterThan(1)

  if (!existsSync(ARTIFACT)) {
    console.log(`COAL-PIN skipped enrichment half: ${ARTIFACT} absent`)
    return
  }
  const trace = JSON.parse(readFileSync(ARTIFACT, 'utf8')) as {
    step: number
    coverage: { enrichmentVsWater: number; gRatioByBin: number[] }
  }[]
  expect(trace.length, 'trace must carry snapshots').toBeGreaterThan(0)
  for (const r of trace) {
    expect(r.coverage.enrichmentVsWater, `step ${r.step}: free heads must be DEPLETED at the surface`).toBeLessThan(1)
    // And depleted hardest in the first shell, which is what "excluded, not adsorbed" looks like.
    const firstShell = r.coverage.gRatioByBin.filter((v) => v >= 0).slice(0, 3)
    for (const v of firstShell) expect(v, `step ${r.step}: g_O/g_W in the first shell`).toBeLessThan(1)
  }
  console.log(
    `COAL-PIN enrichmentVsWater over ${trace.length} snapshots: ` +
      `[${Math.min(...trace.map((r) => r.coverage.enrichmentVsWater)).toFixed(4)}, ${Math.max(...trace.map((r) => r.coverage.enrichmentVsWater)).toFixed(4)}]`,
  )
})

test('the aggregates do not exchange material: at most one merge per interval, zero fissions', () => {
  if (!existsSync(ARTIFACT)) {
    console.log(`COAL-PIN skipped: ${ARTIFACT} absent`)
    return
  }
  const trace = JSON.parse(readFileSync(ARTIFACT, 'utf8')) as {
    step: number
    transition: null | { merges: number; fissions: number; encounters: { merged: number; stillInContact: number; driftedApart: number } }
  }[]
  let merges = 0
  let fissions = 0
  let encMerged = 0
  let encTotal = 0
  for (const r of trace) {
    if (!r.transition) continue
    expect(r.transition.fissions, `step ${r.step}: a fission would mean the distribution is NOT frozen`).toBe(0)
    expect(r.transition.merges, `step ${r.step}: merges per 10 500 steps`).toBeLessThanOrEqual(1)
    merges += r.transition.merges
    fissions += r.transition.fissions
    encMerged += r.transition.encounters.merged
    encTotal += r.transition.encounters.merged + r.transition.encounters.stillInContact + r.transition.encounters.driftedApart
  }
  console.log(`COAL-PIN merges=${merges} fissions=${fissions} encounters=${encTotal} ofWhichMerged=${encMerged}`)
  expect(encTotal, 'the encounter statistic must be non-trivial').toBeGreaterThan(20)
  expect(encMerged / encTotal, 'merge probability per encounter-interval').toBeLessThan(0.1)
})

test('the DEFAULT clustering rule leaves z open, so an aggregate across the z face is counted twice', () => {
  const box: [number, number, number] = [20, 20, 20]
  const cutoff = 2
  // One compact blob straddling the z = 0 face: half at z = 19.5, half at z = 0.5, i.e. 1.0 apart
  // across the periodic image and well inside the cutoff.
  const pts: number[] = []
  const push = (x: number, y: number, z: number) => pts.push(x, y, z, 1)
  for (let k = 0; k < 4; k++) push(10 + k * 0.5, 10, 19.5)
  for (let k = 0; k < 4; k++) push(10 + k * 0.5, 10, 0.5)
  const pos = new Float32Array(pts)
  const zOpen = clusterComponents(pos, box, cutoff)
  const full = clusterComponents3D(pos, box, cutoff)
  console.log(`COAL-PIN clustering: zOpen=${zOpen.map((c) => c.length).join('+')} periodic3d=${full.map((c) => c.length).join('+')}`)
  expect(zOpen.length, 'engine/src/aggregate.ts splits a z-straddling aggregate -- see its own comment').toBe(2)
  expect(full.length, 'with the z image restored it is ONE aggregate').toBe(1)
  expect(full[0].length).toBe(8)
})

test('periodicZ = true joins the z-straddling blob, and agrees with the independent 3D implementation', () => {
  const box: [number, number, number] = [20, 20, 20]
  const cutoff = 2
  const pts: number[] = []
  const push = (x: number, y: number, z: number) => pts.push(x, y, z, 1)
  for (let k = 0; k < 4; k++) push(10 + k * 0.5, 10, 19.5)
  for (let k = 0; k < 4; k++) push(10 + k * 0.5, 10, 0.5)
  const pos = new Float32Array(pts)
  // THE FIX: the soup's own argument. One aggregate, and byte-for-byte the same partition the
  // independent clusterComponents3D helper (a separate implementation, written as the control that
  // sized the defect) produces -- two implementations agreeing is what makes this a fix and not a
  // second convention.
  const fixed = clusterComponents(pos, box, cutoff, true)
  const control = clusterComponents3D(pos, box, cutoff)
  expect(fixed.length, 'z periodic: ONE aggregate').toBe(1)
  expect(fixed[0].length).toBe(8)
  expect(fixed.map((c) => [...c].sort((a, b) => a - b))).toEqual(control.map((c) => [...c].sort((a, b) => a - b)))
  // And the flag must not join what is genuinely apart: the same blob moved to mid-box, split in
  // two by a gap of 4 sigma (> cutoff), stays two aggregates under BOTH branches.
  const apart: number[] = []
  for (let k = 0; k < 4; k++) apart.push(10 + k * 0.5, 10, 6, 1)
  for (let k = 0; k < 4; k++) apart.push(10 + k * 0.5, 10, 14, 1)
  const apartPos = new Float32Array(apart)
  expect(clusterComponents(apartPos, box, cutoff, true).length, 'a real 8 sigma gap is still a gap').toBe(2)
  expect(clusterComponents(apartPos, box, cutoff, false).length).toBe(2)
  console.log(`COAL-PIN clustering z-periodic: joined=${fixed.map((c) => c.length).join('+')} genuinelyApart=2`)
})

test('the same defect class on x and y: a blob across the x or y face is ONE aggregate, both branches', () => {
  const box: [number, number, number] = [20, 20, 20]
  const cutoff = 2
  const straddle = (axis: 0 | 1): Float32Array => {
    const pts: number[] = []
    for (const near of [19.5, 0.5]) {
      for (let k = 0; k < 4; k++) {
        const c = [10, 10, 10]
        c[axis] = near
        c[axis === 0 ? 1 : 0] = 10 + k * 0.5
        pts.push(c[0], c[1], c[2], 1)
      }
    }
    return new Float32Array(pts)
  }
  for (const axis of [0, 1] as const) {
    const name = axis === 0 ? 'x' : 'y'
    for (const periodicZ of [false, true]) {
      const comps = clusterComponents(straddle(axis), box, cutoff, periodicZ)
      expect(comps.length, `${name} face, periodicZ=${periodicZ}: x and y have ALWAYS been periodic`).toBe(1)
      expect(comps[0].length).toBe(8)
    }
  }
  console.log('COAL-PIN clustering x/y faces: 1 aggregate of 8 on both axes, both periodicZ branches')
})
