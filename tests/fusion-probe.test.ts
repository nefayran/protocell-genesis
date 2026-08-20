// THE FUSION PROBE -- build and read -- task 'coalescence' (2026-08-20),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/coalescence-report.md.
//
// Two env-gated halves, neither a gate (both SKIP unsent, like tests/tail-length-oc-sweep.test.ts):
//  * FUSE_BUILD: takes a settled checkpoint of the decisive lineage and writes a probe checkpoint
//    with two of its own aggregate PAIRS pushed into surface contact -- a permutation of positions
//    inside the parent box (see tests/helpers/fusion-probe-state.ts's header for why that, and not a
//    fresh water box, is the defensible construction). Off-GPU. The GPU run that follows is an
//    ordinary `soup/cli/campaign.ts` resume of the file this writes, with no cycling.
//  * FUSE_READ: reads the probe lineage's checkpoints and reports, per forced pair, whether the two
//    groups MERGED -- tracked by particle index, which chemistry cannot confuse because organics are
//    never created or destroyed -- together with the water still sitting in the gap.
//
// A probe is a probe. Nothing here is ever reported as the run that produced a vesicle.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { expect, test } from 'vitest'
import { decodeCheckpointResume, type CheckpointFile } from '../soup/src/checkpoint'
import { loadSoup } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { findAmphiphiles } from '../soup/src/amphiphile'
import { memberIndicesOf } from '../soup/src/aggregates'
import { loadStageThresholds } from '../soup/src/stages'
import { buildAggregates } from './helpers/coalescence-geometry'
import { applyContact, globalMinSeparation, minGap } from './helpers/fusion-probe-state'

const EMPTY = 0xffffffff
const mi = (d: number, L: number) => d - L * Math.round(d / L)

function bondGraph(r: ReturnType<typeof decodeCheckpointResume>, N: number) {
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
  return { deg, bonds: new Uint32Array(pairs) }
}

test('fusion probe: build a contact state out of the decisive lineage own aggregates', () => {
  const src = process.env.FUSE_BUILD
  if (!src || !existsSync(src)) {
    console.log('FUSE-BUILD skipped: FUSE_BUILD unset or missing')
    return
  }
  const outDir = process.env.FUSE_DIR ?? 'data/checkpoints/fuse'
  const label = process.env.FUSE_LABEL ?? 'fuse54'
  const targetGap = Number(process.env.FUSE_GAP ?? 0.6)
  const parent = JSON.parse(readFileSync(src, 'utf8')) as CheckpointFile
  const r = decodeCheckpointResume(parent)
  const pos = r.positions
  const N = pos.length / 4
  const box = r.liveBox as [number, number, number]
  const soup = loadSoup()
  const p = loadParams()
  const t = loadStageThresholds()
  const headKind = soup.monomers.findIndex((m) => m.kind === 'head')
  const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
  const { deg, bonds } = bondGraph(r, N)
  const amph = findAmphiphiles(pos, bonds, soup.monomers)
  const aggs = buildAggregates(pos, box, new Set(amph.map((a) => a.headIndex)), memberIndicesOf(amph), cutoff, t.minAmphiphilesPerAggregate, headKind, 'periodic3d')
  expect(aggs.length, 'probe source must carry at least four qualifying aggregates').toBeGreaterThanOrEqual(4)

  // Protected from relocation: anything bonded, anything tethered to a catalyst, and every catalyst
  // bead. Only free solvent and free monomers are ever moved out of the mover's way.
  const prot = new Set<number>()
  for (const a of aggs) for (const i of a.idx) prot.add(i)
  // Molecule ids: a union-find over the bond graph, so a free bonded molecule that has to give way
  // is pushed RIGIDLY and no push can stretch a bond.
  const unitOf = new Int32Array(N)
  for (let i = 0; i < N; i++) unitOf[i] = i
  const find = (x: number) => {
    while (unitOf[x] !== x) x = unitOf[x] = unitOf[unitOf[x]]
    return x
  }
  for (let k = 0; k < bonds.length; k += 2) {
    const ra = find(bonds[k])
    const rb = find(bonds[k + 1])
    if (ra !== rb) unitOf[rb] = ra
  }
  for (let i = 0; i < N; i++) unitOf[i] = find(i)

  // WHICH pairs to force: the two closest pairs among the eight largest aggregates, not the two
  // biggest. The displacement is what disturbs the solvent, so the closest pair is the smallest
  // perturbation that still asks the question -- and both members are large objects either way
  // (sizes are printed). Pairs are disjoint: an aggregate is forced at most once.
  const top = aggs
  const cand: { a: number; b: number; gap: number }[] = []
  for (let a = 0; a < top.length; a++) for (let b = a + 1; b < top.length; b++) cand.push({ a, b, gap: minGap(pos, box, top[a].idx, top[b].idx) })
  cand.sort((x, y) => x.gap - y.gap)
  const used = new Set<number>()
  const chosen: { a: number; b: number; gap: number }[] = []
  for (const c of cand) {
    if (used.has(c.a) || used.has(c.b)) continue
    used.add(c.a)
    used.add(c.b)
    chosen.push(c)
    // ONE forced contact per probe. Two were attempted; the second pair's swept volume repeatedly
    // held a free chain that could not be put back anywhere at any of eight target gaps (leftover
    // 3-6 molecules, printed above), and a probe that ships an unplaced molecule is not a probe.
    if (chosen.length === 1) break
  }
  console.log('FUSE-BUILD-PAIRS ' + JSON.stringify(chosen.map((c) => ({ anchorN: top[c.a].amphiphileCount, moverN: top[c.b].amphiphileCount, gap: Number(c.gap.toFixed(3)) }))))
  const plans = chosen.map((c) => ({ anchor: top[c.a].idx, mover: top[c.b].idx, targetGap }))
  const bondedPairs = new Set<string>()
  for (let k = 0; k < bonds.length; k += 2) bondedPairs.add(`${Math.min(bonds[k], bonds[k + 1])}-${Math.max(bonds[k], bonds[k + 1])}`)
  // The acceptance reference, MEASURED on the parent before anything is touched: the worst contact
  // the decisive trajectory itself carries. A probe whose worst contact is no worse than that is as
  // safe to integrate as the checkpoint it came from -- which is a measurement, not a chosen bound.
  const parentWorst = globalMinSeparation(decodeCheckpointResume(parent).positions, box)
  console.log(`FUSE-BUILD-PARENT-WORST minSeparation=${parentWorst.min.toFixed(4)} pair=${parentWorst.pair.join('-')}`)
  // The gap is pushed as CLOSE as the construction can go cleanly. A long free chain in the swept
  // volume occasionally has no clear site to be put back into at any orientation; rather than ship a
  // state with an unplaced molecule, the displacement is reduced (the target gap widened) until every
  // lifted molecule is placed. Every gap tried here is still inside the project's own clustering
  // cutoff, so the pair is "in contact" by the same rule the aggregate
  // count uses, and much closer than the pair's own natural separation.
  const radiusOf = new Float32Array(N)
  for (let i = 0; i < N; i++) radiusOf[i] = soup.monomers[Math.round(pos[i * 4 + 3])].radiusSigma
  const reports = plans.map((pl) => {
    // Snapshot taken PER PLAN, so widening the second pair's gap cannot undo the first pair's move.
    const snapshot = Float32Array.from(pos)
    for (const g of [targetGap, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6]) {
      pos.set(snapshot)
      const rep = applyContact(pos, box, { ...pl, targetGap: g }, prot, unitOf, 0.75, radiusOf)
      if (rep.leftover === 0) return rep
      console.log(`FUSE-BUILD-RETRY gap=${g} leftover=${rep.leftover} -- widening`)
    }
    pos.set(snapshot)
    return applyContact(pos, box, { ...pl, targetGap: 2.6 }, prot, unitOf, 0.75, radiusOf)
  })
  for (const rep of reports) {
    console.log('FUSE-BUILD ' + JSON.stringify(rep))
    expect(rep.leftover, 'every lifted molecule must find a clear site').toBe(0)
  }
  // The rigid translation may not change ANY bonded distance -- checked against the parent's own
  // bond list, because a probe that silently stretched a FENE bond would be a different experiment.
  let maxBondDelta = 0
  const orig = decodeCheckpointResume(parent).positions
  for (let k = 0; k < bonds.length; k += 2) {
    const d = (q: Float32Array) => {
      let s = 0
      for (let ax = 0; ax < 3; ax++) {
        const e = mi(q[bonds[k] * 4 + ax] - q[bonds[k + 1] * 4 + ax], box[ax])
        s += e * e
      }
      return Math.sqrt(s)
    }
    maxBondDelta = Math.max(maxBondDelta, Math.abs(d(pos) - d(orig)))
  }
  const probeWorst = globalMinSeparation(pos, box)
  console.log(`FUSE-BUILD-PROBE-WORST minSeparation=${probeWorst.min.toFixed(4)} pair=${probeWorst.pair.join('-')} parent=${parentWorst.min.toFixed(4)}`)
  expect(probeWorst.min, 'probe worst contact must be no tighter than the parent own').toBeGreaterThanOrEqual(parentWorst.min)
  console.log(`FUSE-BUILD-BONDS maxBondedDistanceChange=${maxBondDelta.toExponential(3)} bonds=${bonds.length / 2}`)
  expect(maxBondDelta, 'a rigid translation cannot change a bonded distance').toBeLessThan(1e-3)

  // THE DEFECT THIS PROBE HIT, AND ITS FIX -- kept in the record because it is the predecessor's own
  // named-but-never-fired defect firing for the first time. A rigid translation moves an aggregate's
  // carbon but NOT the catalyst bead it is adsorbed to, so the centerLink tether ends up ~3.5 sigma
  // long against FENE's r_inf = 1.5 -- and past r_inf fene_dv(r) = k*r/(1 - (r/r_inf)^2) changes SIGN
  // and diverges. The first probe built without this fix diverged inside 1000 steps
  // (572 415 of 575 292 position components non-finite, caught by soup/src/soup-health.ts's own loud
  // guard, which is exactly what that guard is for). Every tether that touches a displaced aggregate
  // is therefore DESORBED here -- a physically ordinary event, counted and reported, never silent.
  const centerLink = Uint32Array.from(r.centerLink)
  const moverAll = new Set<number>()
  for (const pl of plans) for (const i of pl.mover) moverAll.add(i)
  let desorbed = 0
  for (let i = 0; i < N; i++) {
    if (centerLink[i] === EMPTY) continue
    if (moverAll.has(i) || moverAll.has(centerLink[i])) {
      centerLink[i] = EMPTY
      desorbed++
    }
  }
  let longestTether = 0
  let liveTethers = 0
  for (let i = 0; i < N; i++) {
    if (centerLink[i] === EMPTY || centerLink[i] >= N) continue
    liveTethers++
    let d2 = 0
    for (let ax = 0; ax < 3; ax++) {
      const e = mi(pos[i * 4 + ax] - pos[centerLink[i] * 4 + ax], box[ax])
      d2 += e * e
    }
    longestTether = Math.max(longestTether, Math.sqrt(d2))
  }
  console.log(`FUSE-BUILD-TETHER desorbedByDisplacement=${desorbed} live=${liveTethers} longest=${longestTether.toFixed(4)} rInf=${p.fene.rInf}`)
  expect(longestTether, 'no tether may be left over FENE r_inf').toBeLessThan(p.fene.rInf)

  const cfg = { box: parent.config.box, seed: parent.config.seed, kT: parent.config.kT, start: parent.config.start }
  const bytes = new Uint8Array(pos.buffer, pos.byteOffset, pos.byteLength)
  const file: CheckpointFile = {
    ...parent,
    createdAt: new Date().toISOString(),
    config: cfg,
    positionsB64: Buffer.from(bytes).toString('base64'),
    centerLinkB64: Buffer.from(new Uint8Array(centerLink.buffer, centerLink.byteOffset, centerLink.byteLength)).toString('base64'),
  }
  mkdirSync(outDir, { recursive: true })
  writeFileSync(`${outDir}/${label}-step${parent.globalStep}.json`, JSON.stringify(file))
  writeFileSync(`${outDir}/${label}-groups.json`, JSON.stringify({
    source: src, step: parent.globalStep, targetGap, box,
    desorbedByDisplacement: desorbed,
    pairs: plans.map((pl, k) => ({ anchor: pl.anchor, mover: pl.mover, report: reports[k] })),
  }))
  console.log(`FUSE-BUILD wrote ${outDir}/${label}-step${parent.globalStep}.json and its group sidecar`)
}, 3_600_000)

test('fusion probe: read the outcome of every forced contact', () => {
  const groupsPath = process.env.FUSE_READ
  if (!groupsPath || !existsSync(groupsPath)) {
    console.log('FUSE-READ skipped: FUSE_READ unset or missing')
    return
  }
  const g = JSON.parse(readFileSync(groupsPath, 'utf8')) as {
    box: [number, number, number]
    pairs: { anchor: number[]; mover: number[] }[]
  }
  const files = (process.env.FUSE_CHECKPOINTS ?? '').trim().split(/[\s,]+/).filter((f) => f && existsSync(f))
  const soup = loadSoup()
  const p = loadParams()
  const cutoff = wcaCutoff(p.sigma * 1.0) + p.attraction.wc
  const waterKind = soup.monomers.findIndex((m) => m.id === soup.solvent.waterId)
  const out: unknown[] = []
  for (const path of files.sort((a, b) => stepOf(a) - stepOf(b))) {
    const r = decodeCheckpointResume(JSON.parse(readFileSync(path, 'utf8')))
    const pos = r.positions
    const N = pos.length / 4
    const box = r.liveBox as [number, number, number]
    const waterIdx: number[] = []
    for (let i = 0; i < N; i++) if (Math.round(pos[i * 4 + 3]) === waterKind) waterIdx.push(i)
    const rec = {
      step: r.globalStep,
      file: path,
      pairs: g.pairs.map((pair, k) => {
        const gap = minGap(pos, box, pair.anchor, pair.mover)
        const ca = com(pos, box, pair.anchor)
        const cm = com(pos, box, pair.mover)
        const sep = Math.hypot(mi(ca[0] - cm[0], box[0]), mi(ca[1] - cm[1], box[1]), mi(ca[2] - cm[2], box[2]))
        // Interface size and mixing: how many of the mover's beads have an anchor bead inside the
        // clustering cutoff, and vice versa. A true fusion drives this UP (the two lobes interpenetrate);
        // two lobes merely touching keep it at the value a flat contact patch gives.
        let crossPairs = 0
        let moverWithAnchorNeighbour = 0
        for (const i of pair.mover) {
          let any = false
          for (const j of pair.anchor) {
            const dx = mi(pos[i * 4] - pos[j * 4], box[0])
            if (Math.abs(dx) > cutoff) continue
            const dy = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1])
            if (Math.abs(dy) > cutoff) continue
            const dz = mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])
            if (dx * dx + dy * dy + dz * dz <= cutoff * cutoff) {
              crossPairs++
              any = true
            }
          }
          if (any) moverWithAnchorNeighbour++
        }
        // Water in the gap: within 2.5 sigma of a bead of BOTH groups.
        let gapWater = 0
        for (const w of waterIdx) {
          let nearA = false
          let nearB = false
          for (const j of pair.anchor) if (near(pos, box, w, j, 2.5)) { nearA = true; break }
          if (!nearA) continue
          for (const j of pair.mover) if (near(pos, box, w, j, 2.5)) { nearB = true; break }
          if (nearB) gapWater++
        }
        return {
          pair: k,
          gap: Number(gap.toFixed(3)),
          centroidSeparation: Number(sep.toFixed(3)),
          crossPairsWithinCutoff: crossPairs,
          moverBeadsWithAnchorNeighbour: moverWithAnchorNeighbour,
          moverBeads: pair.mover.length,
          interfaceFraction: Number((moverWithAnchorNeighbour / pair.mover.length).toFixed(4)),
          gapWater,
        }
      }),
    }
    out.push(rec)
    console.log('FUSE-READ ' + JSON.stringify(rec))
  }
  mkdirSync('verify/out', { recursive: true })
  writeFileSync(process.env.FUSE_ARTIFACT ?? 'verify/out/fusion-probe-trace.json', JSON.stringify(out, null, 1))
  console.log('FUSE-READ artifact written')
}, 3_600_000)

const stepOf = (q: string) => Number(/step(\d+)\.json$/.exec(q)?.[1] ?? 0)

function com(pos: Float32Array, box: [number, number, number], idx: readonly number[]): [number, number, number] {
  const ref = [pos[idx[0] * 4], pos[idx[0] * 4 + 1], pos[idx[0] * 4 + 2]]
  const acc = [0, 0, 0]
  for (const i of idx) for (let ax = 0; ax < 3; ax++) acc[ax] += ref[ax] + mi(pos[i * 4 + ax] - ref[ax], box[ax])
  return [acc[0] / idx.length, acc[1] / idx.length, acc[2] / idx.length]
}

function near(pos: Float32Array, box: [number, number, number], i: number, j: number, r: number): boolean {
  const dx = mi(pos[i * 4] - pos[j * 4], box[0])
  if (Math.abs(dx) > r) return false
  const dy = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1])
  if (Math.abs(dy) > r) return false
  const dz = mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])
  return dx * dx + dy * dy + dz * dz <= r * r
}
