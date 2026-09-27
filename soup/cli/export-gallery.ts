// Snapshot export for the gallery page (viewer/gallery.html).
//
// The gallery shows real campaign checkpoints, not a re-run: this CLI decodes a fixed list of them,
// classifies every organic bead the same way soup/src/stages.ts's detectStage() does (amphiphiles
// from the bond graph, aggregates by the same tail-contact cutoff), and writes one compact binary per
// snapshot plus an index with the measured numbers the page prints next to it. Wrapping axes and
// encapsulated water are not recomputed here: they are joined from the traces in verify/out/, which
// are what the published gates were decided from.
//
// Checkpoints are gigabytes and stay out of the repository, so the exported files are committed and
// this CLI is only needed to regenerate them from a local data/checkpoints/.
//
// Binary layout per snapshot (little-endian): Uint16 x,y,z per bead, quantised over the live box,
// followed by one Uint8 class per bead (see CLASS below). Bead count and box are in index.json.
//
// Run: npx vite-node soup/cli/export-gallery.ts
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { decodeCheckpointResume, type CheckpointFile } from '../src/checkpoint'
import { loadSoup } from '../src/rules'
import { findAmphiphiles } from '../src/amphiphile'
import { memberIndicesOf, positionsFor } from '../src/aggregates'
import { NONE_U32 } from '../src/soup-types'
import { clusterComponents } from '../../engine/src/aggregate'
import { loadParams, wcaCutoff } from '../../engine/src/params'

const OUT = 'viewer/gallery-data'

const CLASS = {
  headLargest: 0,
  tailLargest: 1,
  headOther: 2,
  tailOther: 3,
  freeOrganic: 4,
  water: 5,
} as const

// One water bead in WATER_STRIDE is kept: enough to show where the liquid is without hiding the
// organics behind 351 181 points.
const WATER_STRIDE = 16

interface Spec {
  campaign: 'periodic' | 'confined'
  file: string
  caption: string
}

const SNAPSHOTS: Spec[] = [
  { campaign: 'periodic', file: 'data/checkpoints/bbB76/bbB76-step3000.json', caption: 'Early soup: the first amphiphiles, in hundreds of small clusters' },
  { campaign: 'periodic', file: 'data/checkpoints/bbB76/bbB76-step11000.json', caption: 'Chains keep growing; clusters are still small' },
  { campaign: 'periodic', file: 'data/checkpoints/bbB76/bbB76-step23400.json', caption: 'Dry phase: water evaporated, the box shrank, everything touches' },
  { campaign: 'periodic', file: 'data/checkpoints/bbB76/bbB76-step41100.json', caption: 'Rewetted: one aggregate that wraps through all three axes' },
  { campaign: 'periodic', file: 'data/checkpoints/bbB76/bbB76-step148200.json', caption: 'Final snapshot: still one sheet through the box, no water inside' },
  { campaign: 'confined', file: 'data/checkpoints/cpR36/cpR36-step2000.json', caption: 'A finite parcel of water inside a neutral wall' },
  { campaign: 'confined', file: 'data/checkpoints/cpR36/cpR36-step31800.json', caption: 'Dry phase: the parcel shrinks with the box' },
  { campaign: 'confined', file: 'data/checkpoints/cpR36/cpR36-step105200.json', caption: 'Rewetted: one aggregate that no longer wraps' },
  { campaign: 'confined', file: 'data/checkpoints/cpR36/cpR36-step225200.json', caption: 'Final snapshot: finite, edged, and still no water inside' },
]

interface PercolationRow { file: string; wrappingAxes: number }
interface TraceRow {
  step: number
  stage: string
  largest?: { encapsulatedWater?: { encapsulatedCount: number; encapsulationThresholdCount: number } | string | null }[]
}

const percolation: PercolationRow[] = [
  ...JSON.parse(readFileSync('verify/out/gates-percolation.json', 'utf8')),
  ...JSON.parse(readFileSync('verify/out/gates-percolation-confined.json', 'utf8')),
]
const traces: Record<Spec['campaign'], TraceRow[]> = {
  periodic: JSON.parse(readFileSync('verify/out/closeout-campaign-B76-trace.json', 'utf8')),
  confined: JSON.parse(readFileSync('verify/out/confined-campaign-R36-trace.json', 'utf8')),
}

const soup = loadSoup()
const p = loadParams()
// The same aggregate cutoff detectStage() derives from the species sizes.
const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
const waterKind = soup.monomers.findIndex((m) => m.id === soup.solvent.waterId)

mkdirSync(OUT, { recursive: true })
const index: Record<string, unknown>[] = []

for (const spec of SNAPSHOTS) {
  const file = JSON.parse(readFileSync(spec.file, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  const N = file.N
  const box = file.liveBox
  const pos = resume.positions

  const edges: number[] = []
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < 3; s++) {
      const q = resume.bondSlots[i * 3 + s]
      if (q !== NONE_U32 && q > i) edges.push(i, q)
    }
  }
  const amphiphiles = findAmphiphiles(pos, new Uint32Array(edges), soup.monomers)
  const memberIdx = memberIndicesOf(amphiphiles)
  const idxArr = Array.from(memberIdx)
  // `true`: the soup wraps all three axes, as in analyzeAggregates().
  const components = clusterComponents(positionsFor(pos, memberIdx), box, cutoff, true)
  const inLargest = new Set((components[0] ?? []).map((k) => idxArr[k]))
  const heads = new Set(amphiphiles.map((a) => a.headIndex))
  const amphiphilesInLargest = amphiphiles.filter((a) => inLargest.has(a.headIndex)).length

  const kept: number[] = []
  const cls: number[] = []
  let waterSeen = 0
  for (let i = 0; i < N; i++) {
    const kind = Math.round(pos[i * 4 + 3])
    let c: number
    if (memberIdx.has(i)) {
      const head = heads.has(i)
      c = inLargest.has(i) ? (head ? CLASS.headLargest : CLASS.tailLargest) : head ? CLASS.headOther : CLASS.tailOther
    } else if (kind === waterKind) {
      if (waterSeen++ % WATER_STRIDE !== 0) continue
      c = CLASS.water
    } else {
      c = CLASS.freeOrganic
    }
    kept.push(i)
    cls.push(c)
  }

  const n = kept.length
  const xyz = new Uint16Array(n * 3)
  for (let k = 0; k < n; k++) {
    const i = kept[k]
    for (let a = 0; a < 3; a++) {
      const u = pos[i * 4 + a] / box[a]
      const wrapped = u - Math.floor(u)
      xyz[k * 3 + a] = Math.min(65535, Math.round(wrapped * 65535))
    }
  }
  const id = `${spec.campaign}-${file.globalStep}`
  const bin = Buffer.concat([Buffer.from(xyz.buffer), Buffer.from(Uint8Array.from(cls).buffer)])
  writeFileSync(`${OUT}/${id}.bin`, bin)

  const trace = traces[spec.campaign].find((t) => t.step === file.globalStep)
  const water = trace?.largest?.[0]?.encapsulatedWater
  const perc = percolation.find((r) => r.file === spec.file)
  const confine = file.config.confine
  const counts: Record<string, number> = {}
  for (const c of cls) counts[c] = (counts[c] ?? 0) + 1

  index.push({
    id,
    campaign: spec.campaign,
    caption: spec.caption,
    step: file.globalStep,
    phase: file.cyclePhase,
    box,
    particles: N,
    beads: n,
    classCounts: counts,
    waterStride: WATER_STRIDE,
    amphiphiles: amphiphiles.length,
    aggregates: components.length,
    amphiphilesInLargest,
    stage: trace?.stage ?? null,
    wrappingAxes: perc ? perc.wrappingAxes : null,
    encapsulatedWater: water && typeof water === 'object' ? water.encapsulatedCount : null,
    closureThreshold: water && typeof water === 'object' ? Math.round(water.encapsulationThresholdCount) : null,
    parcelRadius: confine ? confine.radiusSigma * (box[0] / file.config.box[0]) : null,
  })
  console.log(`${id}: ${n} beads, ${amphiphiles.length} amphiphiles, ${components.length} aggregates, ${bin.length} bytes`)
}

writeFileSync(`${OUT}/index.json`, JSON.stringify({ classes: CLASS, snapshots: index }, null, 2) + '\n')
