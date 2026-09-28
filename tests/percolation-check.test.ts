// Does the largest aggregate PERCOLATE -- i.e. is "one aggregate holding ~100 % of the supply" a
// compact object or a box-spanning network? Task 'final-campaign' (2026-08-20),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/final-campaign-report.md.
//
// WHY THIS FILE EXISTS. coalescence-report.md §8.1 raised the question and answered it with a
// PROXY: the radius of gyration of the single aggregate (13.15-15.31 sigma) against the r_g of a
// uniformly FILLED box (15.000 sigma at L = 30). That proxy is weak in two ways -- r_g is computed
// on unwrapAggregate()'s output, whose minimum-image-from-one-reference frame is exactly what stops
// being trustworthy once an aggregate's extent approaches the box (periodic-measurement-report.md's
// own finding), and "r_g close to the filled-box value" is consistent with a compact object of the
// same span as well as with a network. This file replaces the proxy with the DIRECT test.
//
// THE TEST (standard wrapping-cluster criterion, no new constant). Take the largest aggregate's own
// member particles. Replicate them once along one axis (+L) with NO periodicity on that axis, join
// pairs within the SAME cutoff the project already uses for "connected aggregate", and ask whether
// any particle ends up in the same connected component as its OWN +L image. If it does, the
// aggregate connects to itself across the boundary on that axis: it wraps, and "one aggregate" is a
// percolating network rather than an object. If it does not, the aggregate is genuinely finite and
// its span is a real span. Run per axis, so a network that spans one axis only is reported as such.
//
// Everything reused: findAmphiphiles/memberIndicesOf/positionsFor (soup), clusterComponents with
// periodicZ = true (engine, this task's own fix), the cutoff re-derived from data/params.json +
// data/soup.json radii exactly as soup/src/stages.ts derives it. No threshold is introduced: the
// output of this file is a boolean per axis plus the aggregate's own occupied-cell span, and the
// only comparison made is against the box itself.
import { existsSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { dirname } from 'node:path'
import { expect, test } from 'vitest'
import { clusterComponents } from '../engine/src/aggregate'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { findAmphiphiles } from '../soup/src/amphiphile'
import { memberIndicesOf, positionsFor } from '../soup/src/aggregates'
import { loadSoup } from '../soup/src/rules'
import { wrapsOnAxis } from './helpers/periodic-geometry'

const FILES = (process.env.PERC_CHECKPOINTS ?? '').split(/\s+/).filter(Boolean)
const ARTIFACT = process.env.PERC_ARTIFACT ?? 'verify/out/percolation-check.json'
/** Task 'electrostatics' (2026-08-20): which of these files are THE campaign under test and which are
 * the must-say-no controls. verify/campaign-gates.ts used to decide that by looking for the substring
 * 'zfB54' -- one campaign's LABEL hardcoded into the gate pipeline, which silently turned the
 * percolation gate to `unproven` the first time a campaign with a different label produced the
 * artifact. Each row now carries its own `role`, so the label lives with the run that made it; the
 * gate keeps the old substring rule as a fallback for artifacts written before this field existed. */
const CAMPAIGN_LABEL = process.env.PERC_CAMPAIGN_LABEL ?? 'zfB54'

// wrapsOnAxis moved to tests/helpers/periodic-geometry.ts (task 'confined-parcel', 2026-08-21) --
// a pure move, verbatim, because tests/soup-confine.test.ts's setup check needs the SAME criterion
// and a second copy of this project's discriminating measurement is how the two drift apart.

test('does the largest aggregate wrap the box on any axis?', () => {
  if (FILES.length === 0) {
    console.log('PERC skipped: set PERC_CHECKPOINTS')
    return
  }
  const soup = loadSoup()
  const p = loadParams()
  const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
  const out: unknown[] = []
  for (const f of FILES) {
    if (!existsSync(f)) throw new Error(`PERC: no file ${f}`)
    const r = decodeCheckpointResume(JSON.parse(readFileSync(f, 'utf8')))
    const pos = r.positions
    const box = r.liveBox as [number, number, number]
    // Bond pairs from the checkpoint's own raw slot rows, exactly as tests/continuous-run-audit.ts
    // builds them -- the checkpoint's authoritative representation, not a derived list.
    const N = pos.length / 4
    const pairs: number[] = []
    for (let i = 0; i < N; i++) {
      for (let sl = 0; sl < 3; sl++) {
        const j = r.bondSlots[i * 3 + sl]
        if (j === 0xffffffff || i >= j) continue
        pairs.push(i, j)
      }
    }
    const amph = findAmphiphiles(pos, new Uint32Array(pairs), soup.monomers)
    const memberIdx = memberIndicesOf(amph)
    const memberPositions = positionsFor(pos, memberIdx)
    const idxArr = Array.from(memberIdx)
    const comps = clusterComponents(memberPositions, box, cutoff, true)
    const largest = comps[0]
    const member = new Float32Array(largest.length * 4)
    largest.forEach((k, j) => member.set(memberPositions.subarray(k * 4, k * 4 + 4), j * 4))
    const origSet = new Set(largest.map((k) => idxArr[k]))
    const amphInLargest = amph.filter((a) => origSet.has(a.headIndex)).length
    // Occupied-cell span per axis on a coarse `cutoff`-wide grid: how much of the box the object's
    // own members touch at all. A percolating network touches every slab; a compact object does not.
    const nc = Math.max(1, Math.floor(box[0] / cutoff))
    const occupied: Set<number>[] = [new Set(), new Set(), new Set()]
    for (let i = 0; i < member.length / 4; i++) {
      for (let a = 0; a < 3; a++) {
        const v = ((member[i * 4 + a] % box[a]) + box[a]) % box[a]
        occupied[a].add(Math.min(nc - 1, Math.floor((v / box[a]) * nc)))
      }
    }
    const wraps = [0, 1, 2].map((a) => wrapsOnAxis(member, box, cutoff, a as 0 | 1 | 2))
    const rec = {
      file: f,
      step: r.globalStep,
      box: box[0],
      aggregates: comps.length,
      amphiphilesInLargest: amphInLargest,
      particlesInLargest: largest.length,
      wrapsX: wraps[0],
      wrapsY: wraps[1],
      wrapsZ: wraps[2],
      wrappingAxes: wraps.filter(Boolean).length,
      slabsTouchedOfTotal: occupied.map((s) => s.size).concat([nc]),
      role: f.includes(CAMPAIGN_LABEL) ? 'campaign' : 'control',
    }
    out.push(rec)
    console.log(`PERC ${JSON.stringify(rec)}`)
    // The one thing this file ASSERTS rather than reports: the wrapping test must be able to say NO.
    // A test that answered "wraps" for every input would be measuring its own construction, so the
    // aggregate's own members are additionally checked against a control -- the SAME test run on a
    // deliberately compact subset (the members inside one cutoff-wide slab), which must NOT wrap.
    const slabMembers: number[] = []
    for (let i = 0; i < member.length / 4; i++) {
      const v = ((member[i * 4 + 2] % box[2]) + box[2]) % box[2]
      if (v < cutoff) slabMembers.push(i)
    }
    if (slabMembers.length > 4) {
      const slab = new Float32Array(slabMembers.length * 4)
      slabMembers.forEach((k, j) => slab.set(member.subarray(k * 4, k * 4 + 4), j * 4))
      expect(wrapsOnAxis(slab, box, cutoff, 2), 'control: a one-slab subset must NOT wrap in z').toBe(false)
    }
  }
  mkdirSync(dirname(ARTIFACT), { recursive: true })
  writeFileSync(ARTIFACT, JSON.stringify(out, null, 2))
  expect(out.length).toBe(FILES.length)
})
