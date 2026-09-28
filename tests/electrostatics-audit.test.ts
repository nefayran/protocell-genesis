// Off-GPU audit of a CHARGED soup lineage -- task 'electrostatics' (2026-08-20),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/electrostatics-report.md.
//
// A measurement harness, not a gate: with ES_AUDIT_CHECKPOINTS unset it SKIPS (the same discipline
// tests/coalescence-mechanisms.test.ts and tests/tail-length-oc-sweep.test.ts already use), so a
// fresh clone spends nothing on it. A vitest file rather than an `npx tsx` script for the same
// reason those two are: soup/src/aggregates.ts reaches engine/src/closure.ts, which imports a .wgsl
// module with vite's `?raw`, which node alone cannot load.
//
// Everything is measured with the project's OWN functions -- findAmphiphiles, buildAggregates with
// the same wcaCutoff+wc cutoff, mergeStats, pairingStats -- so no number here can disagree with what
// the live run reported for the same checkpoint. What is NEW is four things the neutral lineage had
// no way to ask:
//
//  1. THE PROTONATION STATE, split by environment. alpha over ALL heads, over heads INSIDE the
//     largest aggregate, and over FREE heads (degree 0), each converted to an apparent pKa by
//     Henderson-Hasselbalch read backwards. The DIFFERENCE between the aggregate's alpha and the free
//     heads' alpha is the interfacial pKa SHIFT -- the quantity the literature reports (~0.7 between
//     10 and 100 mM NaCl) and the quantity this model must PRODUCE rather than be given, since its
//     pKa_intrinsic is the monomer acid's 4.9.
//
//  2. ACID-SOAP PAIRING, as a correlation and not as a species. No hydrogen bond was added (see
//     soup/src/electrostatics.ts's header item 4), so the only pairing that can exist here is charge
//     ALTERNATION: the fraction of head-head contacts that are UNLIKE, against the fraction a random
//     assignment at the same alpha would give (2*alpha*(1-alpha)). Plus the LIFETIME of a pair, from
//     the identity of each head's nearest unlike partner across consecutive checkpoints.
//
//  3. WHY the network is (or is not) breakable: the largest COVALENT component. The aggregate that
//     `final-campaign-report.md` measured wrapping 3 of 3 axes is a CONTACT cluster; if its members
//     are also one connected component of the BOND graph, then the connectivity is covalent and head
//     repulsion -- which acts between beads, not on bonds -- structurally cannot cut it. That is a
//     prediction with an obvious test, and it is run here on the same checkpoints.
//
//  4. Monomer exchange and fissions with charge on, against the neutral lineage's measured 0 and 0.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { loadSoup } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { findAmphiphiles } from '../soup/src/amphiphile'
import { loadStageThresholds } from '../soup/src/stages'
import { memberIndicesOf } from '../soup/src/aggregates'
import { buildAggregates, mergeStats, type Agg } from './helpers/coalescence-geometry'
// Task 'acid-soap-pairing' (2026-08-23): nearestUnlike MOVED to soup/src/acid-soap.ts (pure move,
// same body) so this audit and the pair-strength sweep cannot disagree about what an acid-soap pair is
// -- the same reason this file's header gives for sharing pairingStats/mergeStats.
import { nearestUnlike } from '../soup/src/acid-soap'
import { apparentPKa, makeEsBasis, pairingStats } from '../soup/src/electrostatics'

const EMPTY = 0xffffffff

function stepOf(path: string): number {
  const m = /step(\d+)/.exec(path)
  return m ? Number(m[1]) : 0
}

/** Connected components of the BOND graph restricted to `idx`, largest first size. Charge acts
 * between beads; a covalent bond is not something a pair potential can break, so this is the
 * measurement that decides whether head repulsion COULD break the object at all. */
function covalentComponents(bondSlots: Uint32Array, idx: readonly number[]): number[] {
  const inSet = new Set(idx)
  const parent = new Map<number, number>()
  const find = (x: number): number => {
    let r = x
    while ((parent.get(r) ?? r) !== r) r = parent.get(r)!
    while ((parent.get(x) ?? x) !== x) {
      const nx = parent.get(x)!
      parent.set(x, r)
      x = nx
    }
    return r
  }
  const union = (a: number, b: number): void => {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent.set(ra, rb)
  }
  for (const i of idx) parent.set(i, i)
  for (const i of idx) {
    for (let s = 0; s < 3; s++) {
      const j = bondSlots[i * 3 + s]
      if (j === EMPTY || !inSet.has(j)) continue
      union(i, j)
    }
  }
  const sizes = new Map<number, number>()
  for (const i of idx) {
    const r = find(i)
    sizes.set(r, (sizes.get(r) ?? 0) + 1)
  }
  return Array.from(sizes.values()).sort((a, b) => b - a)
}


test('electrostatics: protonation, the acid-soap pair, covalent connectivity, exchange and divisions', () => {
  const raw = process.env.ES_AUDIT_CHECKPOINTS
  if (!raw || raw.trim().length === 0) {
    console.log('ES-AUDIT skipped: ES_AUDIT_CHECKPOINTS unset (measurement harness, not a gate)')
    return
  }
  const files = raw.trim().split(/[\s,]+/).filter((f) => existsSync(f))
  const artifactPath = process.env.ES_AUDIT_ARTIFACT ?? 'verify/out/electrostatics-audit.json'
  const soup = loadSoup()
  const p = loadParams()
  const t = loadStageThresholds()
  const headKind = soup.monomers.findIndex((m) => m.kind === 'head')
  const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
  // "Two heads are touching" = the head-head WCA contact separation data/params.json itself declares
  // (beadSizes.head_head), never a round number.
  const contact = wcaCutoff(p.sigma * p.beadSizes.head_head)
  const artifact: unknown[] = []
  let prev: { step: number; aggs: Agg[]; pairs: Map<number, number>; charges: Float32Array } | null = null

  for (const path of files.slice().sort((a, b) => stepOf(a) - stepOf(b))) {
    const file = JSON.parse(readFileSync(path, 'utf8'))
    const r = decodeCheckpointResume(file)
    const pos = r.positions
    const N = pos.length / 4
    const box = r.liveBox as [number, number, number]
    const es = makeEsBasis(soup, p, file.config?.electrostatics ?? { enabled: false })
    // A neutral lineage's checkpoint has no charges at all -- read as all-zero, which is exactly what
    // it physically was, so the control runs through this same code path.
    const charges = r.charges ?? new Float32Array(N)

    const deg = new Int32Array(N)
    const pairsList: number[] = []
    for (let i = 0; i < N; i++) {
      for (let s = 0; s < 3; s++) {
        const j = r.bondSlots[i * 3 + s]
        if (j === EMPTY) continue
        deg[i]++
        if (i < j) pairsList.push(i, j)
      }
    }
    const bonds = new Uint32Array(pairsList)
    const amph = findAmphiphiles(pos, bonds, soup.monomers)
    const amphHeads = new Set(amph.map((a) => a.headIndex))
    const memberIdx = memberIndicesOf(amph)
    const aggs = buildAggregates(pos, box, amphHeads, memberIdx, cutoff, t.minAmphiphilesPerAggregate, headKind, 'periodic3d')
    const largest = aggs.length > 0 ? aggs.reduce((a, b) => (b.amphiphileCount > a.amphiphileCount ? b : a)) : null

    // --- 1. protonation, split by environment -------------------------------------------------
    const allHeads: number[] = []
    const freeHeads: number[] = []
    for (let i = 0; i < N; i++) {
      if (Math.round(pos[i * 4 + 3]) !== headKind) continue
      allHeads.push(i)
      if (deg[i] === 0) freeHeads.push(i)
    }
    const aggHeads = largest ? largest.heads : []
    const alphaOf = (idx: readonly number[]): number => {
      if (idx.length === 0) return NaN
      let on = 0
      for (const i of idx) if (charges[i] !== 0) on++
      return on / idx.length
    }
    const alphaAll = alphaOf(allHeads)
    const alphaAgg = alphaOf(aggHeads)
    const alphaFree = alphaOf(freeHeads)

    // --- 2. acid-soap pairing, as a correlation ------------------------------------------------
    const pr = pairingStats(pos, charges, box, es, contact)
    const pairs = nearestUnlike(pos, charges, allHeads, box, contact)
    let survived = 0
    let survivedEither = 0
    if (prev) {
      for (const [i, j] of prev.pairs) {
        if (pairs.get(i) === j) survived++
        if (pairs.has(i)) survivedEither++
      }
    }

    // --- 3. covalent connectivity of the largest aggregate -------------------------------------
    const covSizes = largest ? covalentComponents(r.bondSlots, largest.idx) : []
    const covLargest = covSizes.length > 0 ? covSizes[0] : 0
    const covCount = covSizes.length

    // --- 4. exchange / fissions ----------------------------------------------------------------
    const ms = prev ? mergeStats(prev.aggs, aggs, 0.5) : null

    const rec = {
      file: path,
      step: r.globalStep,
      box: Number(box[0].toFixed(4)),
      pH: es.enabled ? es.pH : null,
      ionicStrengthMolar: es.enabled ? es.ionicStrengthMolar : null,
      chargeEnabled: es.enabled,
      amphiphiles: amph.length,
      aggregates: aggs.length,
      largest: largest?.amphiphileCount ?? 0,
      sizeDistribution: aggs.map((a) => a.amphiphileCount).sort((a, b) => b - a),
      freeAmphiphileFraction: Number((1 - (largest ? aggs.reduce((s, a) => s + a.amphiphileCount, 0) : 0) / Math.max(1, amph.length)).toFixed(5)),
      heads: allHeads.length,
      freeHeads: freeHeads.length,
      alphaAll: Number(alphaAll.toFixed(5)),
      alphaInLargestAggregate: Number(alphaAgg.toFixed(5)),
      alphaFreeHeads: Number(alphaFree.toFixed(5)),
      pKaAppAll: Number(apparentPKa(alphaAll, es.pH).toFixed(4)),
      pKaAppAggregate: Number(apparentPKa(alphaAgg, es.pH).toFixed(4)),
      pKaAppFreeHeads: Number(apparentPKa(alphaFree, es.pH).toFixed(4)),
      interfacialShift: Number((apparentPKa(alphaAgg, es.pH) - apparentPKa(alphaFree, es.pH)).toFixed(4)),
      shiftVsIntrinsic: Number((apparentPKa(alphaAgg, es.pH) - es.pKaIntrinsic).toFixed(4)),
      pairedFraction: Number(pr.pairedFraction.toFixed(5)),
      unlikeContactFraction: Number(pr.unlikeFraction.toFixed(5)),
      unlikeContactFractionRandom: Number(pr.unlikeFractionRandom.toFixed(5)),
      unlikeExcessRatio: pr.unlikeFractionRandom > 0 ? Number((pr.unlikeFraction / pr.unlikeFractionRandom).toFixed(4)) : null,
      pairsNow: pairs.size,
      pairsSurvivedFromPrev: prev ? survived : null,
      pairsWhosePartnerChanged: prev ? survivedEither - survived : null,
      pairSurvivalFraction: prev && prev.pairs.size > 0 ? Number((survived / prev.pairs.size).toFixed(5)) : null,
      stepsSincePrev: prev ? r.globalStep - prev.step : null,
      covalentComponentsInLargest: covCount,
      largestCovalentComponent: covLargest,
      covalentSpanFraction: largest && largest.idx.length > 0 ? Number((covLargest / largest.idx.length).toFixed(5)) : null,
      merges: ms?.merges ?? null,
      fissions: ms?.fissions ?? null,
      exchangedFraction: ms ? Number(ms.exchangedFraction.toFixed(6)) : null,
      matchedHeads: ms?.matchedHeads ?? null,
      flatness: largest ? Number(largest.flatnessRatio.toFixed(4)) : null,
      inPlaneSymmetry: largest ? Number(largest.inPlaneSymmetry.toFixed(4)) : null,
      radiusOfGyration: largest ? Number(largest.radiusOfGyration.toFixed(4)) : null,
    }
    artifact.push(rec)
    console.log(`ES-AUDIT ${JSON.stringify(rec)}`)
    prev = { step: r.globalStep, aggs, pairs, charges }
  }

  mkdirSync(dirname(artifactPath), { recursive: true })
  writeFileSync(artifactPath, JSON.stringify(artifact, null, 2))
  console.log(`ES-AUDIT wrote ${artifact.length} records to ${artifactPath}`)
  expect(artifact.length).toBeGreaterThan(0)
})
