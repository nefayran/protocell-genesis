// Off-GPU discrimination between the three candidate coalescence mechanisms -- task 'coalescence'
// (2026-08-20), .superpowers/sdd/2026-08-16-soup-to-vesicle/coalescence-report.md.
//
// A measurement harness, not a gate: with COAL_CHECKPOINTS unset it SKIPS, exactly as
// tests/tail-length-oc-sweep.test.ts does, so a fresh clone spends nothing on it. It is a vitest
// file rather than an `npx tsx` script for the same reason tests/continuous-run-audit.test.ts is:
// soup/src/aggregates.ts reaches engine/src/closure.ts, which imports a .wgsl module with vite's
// `?raw`, which node cannot load.
//
// Everything is measured with the project's OWN functions -- findAmphiphiles, clusterComponents with
// the same wcaCutoff+wc cutoff, shapeOf -- so no number here can disagree with what the live run
// reported for the same checkpoint. What is new is only WHICH objects are measured (every qualifying
// aggregate, not the largest five) and WHAT is measured about their surroundings (see
// tests/helpers/coalescence-geometry.ts's header for the g_O/g_W normalisation argument).
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { loadSoup } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { findAmphiphiles } from '../soup/src/amphiphile'
import { loadStageThresholds } from '../soup/src/stages'
import { memberIndicesOf } from '../soup/src/aggregates'
import { buildAggregates, mergeStats, pairGaps, solventProfile, type Agg } from './helpers/coalescence-geometry'

const EMPTY = 0xffffffff

test('coalescence: free-head coverage, aggregate contacts/merges and shape over every aggregate', () => {
  const raw = process.env.COAL_CHECKPOINTS
  if (!raw || raw.trim().length === 0) {
    console.log('COAL skipped: COAL_CHECKPOINTS unset (measurement harness, not a gate)')
    return
  }
  const files = raw.trim().split(/[\s,]+/).filter((f) => existsSync(f))
  const artifactPath = process.env.COAL_ARTIFACT ?? 'verify/out/coalescence-trace.json'
  const soup = loadSoup()
  const p = loadParams()
  const t = loadStageThresholds()
  const headKind = soup.monomers.findIndex((m) => m.kind === 'head')
  const waterKind = soup.monomers.findIndex((m) => m.id === soup.solvent.waterId)
  const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
  // "Contact" for a solvent bead against a member bead: the WCA contact separation of the head-tail
  // bead pair data/params.json itself declares (beadSizes.head_tail = 0.95), not a round number.
  const contact = wcaCutoff(p.sigma * p.beadSizes.head_tail)
  const artifact: unknown[] = []
  let prev: { step: number; aggs: Agg[]; gaps: { a: number; b: number; gap: number }[] } | null = null

  for (const path of files.slice().sort((a, b) => stepOf(a) - stepOf(b))) {
    const r = decodeCheckpointResume(JSON.parse(readFileSync(path, 'utf8')))
    const pos = r.positions
    const N = pos.length / 4
    const box = r.liveBox as [number, number, number]

    const deg = new Int32Array(N)
    const pairs: number[] = []
    for (let i = 0; i < N; i++) {
      for (let s = 0; s < 3; s++) {
        const j = r.bondSlots[i * 3 + s]
        if (j === EMPTY) continue
        deg[i]++
        if (i < j) pairs.push(i, j)
      }
    }
    const bonds = new Uint32Array(pairs)
    const amph = findAmphiphiles(pos, bonds, soup.monomers)
    const heads = new Set(amph.map((a) => a.headIndex))
    const memberIdx = memberIndicesOf(amph)
    const freeHeadIdx: number[] = []
    const waterIdx: number[] = []
    for (let i = 0; i < N; i++) {
      const k = Math.round(pos[i * 4 + 3])
      if (k === headKind && deg[i] === 0) freeHeadIdx.push(i)
      else if (k === waterKind) waterIdx.push(i)
    }
    let tailSum = 0
    let tailN = 0
    let twoTailed = 0
    for (const a of amph) {
      if (a.tailLengths.length >= 2) twoTailed++
      for (const x of a.tailLengths) {
        tailSum += x
        tailN++
      }
    }

    const aggs = buildAggregates(pos, box, heads, memberIdx, cutoff, t.minAmphiphilesPerAggregate, headKind)
    // THE CONTROL on the project's own z-open clustering rule (see clusterComponents3D's header):
    // the identical connectivity with the z periodic image restored. Reported beside, never instead.
    const aggs3d = buildAggregates(pos, box, heads, memberIdx, cutoff, t.minAmphiphilesPerAggregate, headKind, 'periodic3d')
    const prof = solventProfile(pos, box, aggs3d, freeHeadIdx, waterIdx, { binWidth: 0.25, bins: 24, contact, gapCut: 2.5 })
    const gaps = pairGaps(pos, box, aggs3d, 8)
    const bulkFH = freeHeadIdx.length / (box[0] * box[1] * box[2])
    const bulkW = waterIdx.length / (box[0] * box[1] * box[2])
    // g_O(r)/g_W(r): the two histograms share the shell volume exactly, so the ratio needs only the
    // bulk-count ratio to become a real enrichment ratio.
    const gRatio: number[] = []
    for (let b = 0; b < prof.bins; b++) {
      const o = prof.freeHead[b]
      const w = prof.water[b]
      gRatio.push(w > 0 ? Number(((o / w) * (bulkW / bulkFH)).toFixed(4)) : -1)
    }
    const totalHeadSites = aggs3d.reduce((s, a) => s + a.heads.length, 0)
    const bridging = Array.from(prof.bridgingWater.entries()).sort((x, y) => y[1] - x[1])
    const ms = prev ? mergeStats(prev.aggs, aggs3d, 0.5) : null
    // "Does a near contact proceed to a merge?" -- the direct P2 statistic, from the run's own
    // trajectory: every pair whose surfaces were within `encounterGap` at the PREVIOUS sample is
    // followed to this one and classified. `successor` is the dominant-destination map mergeStats
    // already builds, so this asks about the SAME objects the merge count is about.
    const encounterGap = 4
    let encMerged = 0
    let encStill = 0
    let encApart = 0
    if (prev && ms) {
      const nowGap = new Map<string, number>()
      for (const g of gaps) nowGap.set(`${g.a}-${g.b}`, g.gap)
      for (const g of prev.gaps) {
        if (g.gap > encounterGap) continue
        const sa = ms.successor.get(g.a) ?? -1
        const sb = ms.successor.get(g.b) ?? -1
        if (sa >= 0 && sa === sb) encMerged++
        else {
          const k = sa < sb ? `${sa}-${sb}` : `${sb}-${sa}`
          const gg = nowGap.get(k)
          if (gg !== undefined && gg <= encounterGap) encStill++
          else encApart++
        }
      }
    }

    const record = {
      step: r.globalStep,
      file: path,
      box,
      N,
      amphiphileCount: amph.length,
      meanPerTail: tailN ? Number((tailSum / tailN).toFixed(3)) : 0,
      twoTailedHeads: twoTailed,
      freeHeads: freeHeadIdx.length,
      water: waterIdx.length,
      aggregateCount: aggs.length,
      // --- MECHANISM 1: free-head surface coverage
      coverage: {
        contactDistance: Number(contact.toFixed(4)),
        freeHeadsInContact: prof.freeHeadInContact,
        waterInContact: prof.waterInContact,
        headSites: totalHeadSites,
        freeHeadsPerHeadSite: totalHeadSites ? Number((prof.freeHeadInContact / totalHeadSites).toFixed(4)) : 0,
        waterPerHeadSite: totalHeadSites ? Number((prof.waterInContact / totalHeadSites).toFixed(4)) : 0,
        /** Enrichment over bulk, from the counts alone: (in-contact fraction of the species) /
         * (its bulk fraction of the box). Volume-free because both species are divided by the
         * SAME contact shell. */
        freeHeadContactFractionOfPool: Number((prof.freeHeadInContact / Math.max(1, freeHeadIdx.length)).toExponential(4)),
        waterContactFractionOfPool: Number((prof.waterInContact / Math.max(1, waterIdx.length)).toExponential(4)),
        enrichmentVsWater: Number(
          ((prof.freeHeadInContact / Math.max(1, freeHeadIdx.length)) / (prof.waterInContact / Math.max(1, waterIdx.length))).toFixed(4),
        ),
        gRatioByBin: gRatio,
      },
      // --- MECHANISM 2: contacts, bridging water, merges/fissions
      contacts: {
        clusteringForContacts: 'periodic3d',
        pairsUnder8: gaps.length,
        pairsUnder4: gaps.filter((g) => g.gap <= 4).length,
        pairsUnderCutoff: gaps.filter((g) => g.gap <= cutoff).length,
        closestGaps: gaps.slice(0, 8).map((g) => ({ a: g.a, b: g.b, gap: Number(g.gap.toFixed(3)) })),
        bridgingWaterTopPairs: bridging.slice(0, 8).map(([k, v]) => ({ pair: k, water: v })),
        bridgingWaterTotal: bridging.reduce((s, [, v]) => s + v, 0),
      },
      transition: ms
        ? {
            fromStep: prev!.step,
            merges: ms.merges,
            fissions: ms.fissions,
            persisted: ms.persisted,
            exchangedFraction: Number(ms.exchangedFraction.toFixed(4)),
            matchedHeads: ms.matchedHeads,
            detail: ms.detail.slice(0, 12),
            encounters: { gap: encounterGap, merged: encMerged, stillInContact: encStill, driftedApart: encApart },
          }
        : null,
      // --- MECHANISM 3: shape of EVERY aggregate, not just the largest
      shapes: aggs.map((a) => ({
        n: a.amphiphileCount,
        rg: Number(a.radiusOfGyration.toFixed(2)),
        flat: Number(a.flatnessRatio.toFixed(4)),
        inPl: Number(a.inPlaneSymmetry.toFixed(4)),
      })),
      shapeSummary: summarise(aggs),
      periodic3d: { aggregateCount: aggs3d.length, shapeSummary: summarise(aggs3d), shapes: aggs3d.map((a) => ({ n: a.amphiphileCount, rg: Number(a.radiusOfGyration.toFixed(2)), flat: Number(a.flatnessRatio.toFixed(4)), inPl: Number(a.inPlaneSymmetry.toFixed(4)) })) },
      lamellarCount: aggs.filter((a) => a.flatnessRatio <= t.lamellarFlatnessRatio && a.inPlaneSymmetry >= t.lamellarInPlaneSymmetryMin).length,
      lamellarCount3d: aggs3d.filter((a) => a.flatnessRatio <= t.lamellarFlatnessRatio && a.inPlaneSymmetry >= t.lamellarInPlaneSymmetryMin).length,
    }
    artifact.push(record)
    console.log('COAL ' + JSON.stringify(record))
    expect(aggs.length, `${path}: no qualifying aggregate`).toBeGreaterThan(0)
    prev = { step: r.globalStep, aggs: aggs3d, gaps }
  }
  mkdirSync('verify/out', { recursive: true })
  writeFileSync(artifactPath, JSON.stringify(artifact, null, 1))
  console.log(`COAL artifact written: ${artifactPath} (${artifact.length} snapshots)`)
}, 3_600_000)

const stepOf = (p: string) => Number(/step(\d+)\.json$/.exec(p)?.[1] ?? 0)

function summarise(aggs: Agg[]) {
  if (aggs.length === 0) return null
  const n = aggs.map((a) => a.amphiphileCount)
  const inPl = aggs.map((a) => a.inPlaneSymmetry)
  const flat = aggs.map((a) => a.flatnessRatio)
  const mean = (x: number[]) => x.reduce((s, v) => s + v, 0) / x.length
  const sd = (x: number[]) => {
    const m = mean(x)
    return Math.sqrt(x.reduce((s, v) => s + (v - m) * (v - m), 0) / x.length)
  }
  return {
    sizes: n,
    sizeMean: Number(mean(n).toFixed(2)),
    sizeSd: Number(sd(n).toFixed(2)),
    sizeMax: Math.max(...n),
    inPlaneMean: Number(mean(inPl).toFixed(4)),
    inPlaneSd: Number(sd(inPl).toFixed(4)),
    inPlaneMax: Number(Math.max(...inPl).toFixed(4)),
    aboveLamellar: inPl.filter((v) => v >= 0.5).length,
    flatMean: Number(mean(flat).toFixed(4)),
  }
}
