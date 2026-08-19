// Why a released catalytic centre never re-nucleates -- pure CPU, read straight off a checkpoint's
// own centerLink/bondSlots/positions payload. Task 'catalyst-turnover-and-window' (2026-08-20),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/catalyst-turnover-and-window-report.md sec 1.
//
// verify/center-occupancy.ts (the predecessor task's diagnostic) established WHAT is wrong: at step
// 300 000, 0 of 1021 catalysts hold a chain tip while 5125 bare carbons and 58 387 free heads are
// still in the box, after 711 desorption timeouts. It could not say WHY, because occupancy alone
// cannot distinguish "the link was left occupied" (it was not -- 0 linked means every link is FREE)
// from "the geometric precondition for nucleation stopped occurring".
//
// soup/wgsl/bond-adsorption.wgsl's propagateOnCenter nucleation branch needs THREE things in mutual
// contact at the same instant: a bare carbon i (zero chain bonds), a second bare carbon j within
// wca_cut(bPairB(C,C)) of i, AND a free catalyst within wca_cut(bPairB(C,M)) of BOTH (of i via
// bondFormWalk's own catalystNear test, of j via the dNuc check). This script counts each of those
// populations separately at every checkpoint, so a three-body coincidence going to zero is
// distinguishable from a two-body one.
//
// Usage: npx tsx verify/center-nucleation.ts <checkpoint.json> [...]
import { readFileSync } from 'node:fs'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { loadSoup } from '../soup/src/rules'

const EMPTY = 0xffffffff
const soup = loadSoup()
const TWO_ONE_SIXTH = Math.pow(2, 1 / 6)
const radius = (kind: string) => soup.monomers.find(m => m.kind === kind)!.radiusSigma
// bPairB/wca_cut, engine/wgsl/forces.wgsl:17 and soup/wgsl/bond-common.wgsl:123, sigma=1.
const wcaCut = (ka: string, kb: string) => TWO_ONE_SIXTH * (radius(ka) + radius(kb)) * 0.5
const R_CC = wcaCut('carbon', 'carbon')   // cc_bond candidate contact distance
const R_CM = wcaCut('carbon', 'catalyst') // catalystNear / dNuc contact distance

type Frame = { step: number; catFree: Uint8Array; catIdx: Int32Array }
const frames: Frame[] = []

for (const path of process.argv.slice(2)) {
  const file = JSON.parse(readFileSync(path, 'utf8'))
  const r = decodeCheckpointResume(file)
  const box = r.liveBox ?? file.config.box
  const pos = r.positions
  const N = pos.length / 4
  const kindOf = (i: number) => soup.monomers[Math.round(pos[i * 4 + 3])].kind

  const deg = new Int32Array(N)
  for (let i = 0; i < N; i++) for (let s = 0; s < 3; s++) if (r.bondSlots[i * 3 + s] !== EMPTY) deg[i]++

  const bareC: number[] = []
  const cats: number[] = []
  const catFreeArr: number[] = []
  let owned = 0
  for (let i = 0; i < N; i++) {
    const k = kindOf(i)
    if (k === 'carbon' && deg[i] === 0) { bareC.push(i); if (r.centerLink[i] !== EMPTY) owned++ }
    if (k === 'catalyst') { cats.push(i); catFreeArr.push(r.centerLink[i] === EMPTY ? 1 : 0) }
  }
  frames.push({ step: r.globalStep, catFree: Uint8Array.from(catFreeArr), catIdx: Int32Array.from(cats) })

  // uniform grid over the free catalysts, cell = R_CM so one shell of 27 cells covers the reach
  const cell = Math.max(R_CC, R_CM)
  const nx = Math.max(1, Math.floor(box[0] / cell)), ny = Math.max(1, Math.floor(box[1] / cell)), nz = Math.max(1, Math.floor(box[2] / cell))
  const hx = box[0] / nx, hy = box[1] / ny, hz = box[2] / nz
  const wrap = (v: number, n: number) => ((v % n) + n) % n
  const mi = (d: number, L: number) => d - L * Math.round(d / L)
  const build = (ids: number[]) => {
    const head = new Int32Array(nx * ny * nz).fill(-1)
    const next = new Int32Array(ids.length).fill(-1)
    for (let a = 0; a < ids.length; a++) {
      const i = ids[a]
      const cx = wrap(Math.floor(pos[i * 4] / hx), nx), cy = wrap(Math.floor(pos[i * 4 + 1] / hy), ny), cz = wrap(Math.floor(pos[i * 4 + 2] / hz), nz)
      const c = cx + nx * (cy + ny * cz)
      next[a] = head[c]; head[c] = a
    }
    return { head, next }
  }
  const neigh = (g: { head: Int32Array; next: Int32Array }, ids: number[], i: number, cut: number, fn: (j: number) => void) => {
    const cx = wrap(Math.floor(pos[i * 4] / hx), nx), cy = wrap(Math.floor(pos[i * 4 + 1] / hy), ny), cz = wrap(Math.floor(pos[i * 4 + 2] / hz), nz)
    for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const c = wrap(cx + dx, nx) + nx * (wrap(cy + dy, ny) + ny * wrap(cz + dz, nz))
      for (let a = g.head[c]; a !== -1; a = g.next[a]) {
        const j = ids[a]
        if (j === i) continue
        const ddx = mi(pos[i * 4] - pos[j * 4], box[0]), ddy = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1]), ddz = mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])
        if (ddx * ddx + ddy * ddy + ddz * ddz < cut * cut) fn(j)
      }
    }
  }
  const freeCats = cats.filter((_, a) => catFreeArr[a] === 1)
  const gCat = build(freeCats)
  const gBare = build(bareC)

  // two-body: a bare carbon within reaction contact of a FREE catalyst
  let twoBody = 0
  const catNear = new Map<number, number>()
  for (const i of bareC) {
    let found = -1
    neigh(gCat, freeCats, i, R_CM, j => { if (found < 0) found = j })
    if (found >= 0) { twoBody++; catNear.set(i, found) }
  }
  // bare-bare pairs in cc_bond contact
  let barePairs = 0
  for (const i of bareC) { let n = 0; neigh(gBare, bareC, i, R_CC, () => n++); barePairs += n }
  barePairs /= 2
  // three-body: the full nucleation precondition
  let triples = 0
  for (const [i, cat] of catNear) {
    let ok = false
    neigh(gBare, bareC, i, R_CC, j => {
      if (ok) return
      const ddx = mi(pos[j * 4] - pos[cat * 4], box[0]), ddy = mi(pos[j * 4 + 1] - pos[cat * 4 + 1], box[1]), ddz = mi(pos[j * 4 + 2] - pos[cat * 4 + 2], box[2])
      if (ddx * ddx + ddy * ddy + ddz * ddz < R_CM * R_CM) ok = true
    })
    if (ok) triples++
  }
  // RELAXED precondition: the same nucleation, but with the TIP being the carbon that is actually
  // in contact with the catalyst (i, the invoking particle, whose catalyst proximity bondFormWalk
  // has ALREADY established) instead of j, the arbitrary other one. Counted exactly as the kernel
  // would evaluate it: the candidate pair is collected only by the SMALLER-indexed side (j > i in
  // bondFormWalk), and catalystNear/catalystId is resolved for that same smaller-indexed particle.
  let relaxedAsCoded = 0, relaxedEitherSide = 0
  for (const i of bareC) {
    let hasSmallerSidePartner = false, hasAnyPartner = false
    neigh(gBare, bareC, i, R_CC, j => { hasAnyPartner = true; if (j > i) hasSmallerSidePartner = true })
    const nearFree = catNear.has(i)
    if (nearFree && hasSmallerSidePartner) relaxedAsCoded++
    if (nearFree && hasAnyPartner) relaxedEitherSide++
  }
  // the re-adsorption (reclaim) population: an owner-less chain end (exactly one C-C bond, no head
  // cap) sitting within reaction contact of a free catalyst
  let looseEnds = 0, looseEndsNearFreeCat = 0
  for (let i = 0; i < N; i++) {
    if (kindOf(i) !== 'carbon' || deg[i] !== 1 || r.centerLink[i] !== EMPTY) continue
    let cc = 0
    for (let s = 0; s < 3; s++) { const pj = r.bondSlots[i * 3 + s]; if (pj !== EMPTY && kindOf(pj) === 'carbon') cc++ }
    if (cc !== 1) continue
    looseEnds++
    let found = false
    neigh(gCat, freeCats, i, R_CM, () => { found = true })
    if (found) looseEndsNearFreeCat++
  }
  // nearest-catalyst distance distribution for bare carbons (brute force over 1021 catalysts)
  const dists: number[] = []
  for (const i of bareC) {
    let best = Infinity
    for (const c of cats) {
      const ddx = mi(pos[i * 4] - pos[c * 4], box[0]), ddy = mi(pos[i * 4 + 1] - pos[c * 4 + 1], box[1]), ddz = mi(pos[i * 4 + 2] - pos[c * 4 + 2], box[2])
      const d2 = ddx * ddx + ddy * ddy + ddz * ddz
      if (d2 < best) best = d2
    }
    dists.push(Math.sqrt(best))
  }
  dists.sort((a, b) => a - b)
  const q = (f: number) => dists.length ? dists[Math.min(dists.length - 1, Math.floor(f * dists.length))].toFixed(3) : 'n/a'
  console.log(`NUC ${path} step=${r.globalStep} bareC=${bareC.length} bareCowned=${owned} freeCats=${freeCats.length}/${cats.length} `
    + `barePairsInContact=${barePairs} bareC_nearFreeCat=${twoBody} NUCLEATION_TRIPLES=${triples} `
    + `RELAXED_ASCODED=${relaxedAsCoded} RELAXED_EITHERSIDE=${relaxedEitherSide} `
    + `looseChainEnds=${looseEnds} looseEndsNearFreeCat=${looseEndsNearFreeCat} `
    + `dNearestCat[min,p10,med,p90]=[${q(0)},${q(0.1)},${q(0.5)},${q(0.9)}] R_CC=${R_CC.toFixed(5)} R_CM=${R_CM.toFixed(5)}`)
}

// Per-centre fate across the frames given, in the order given: once a centre is released, does it
// ever hold a tip again?
if (frames.length > 1) {
  const n = frames[0].catIdx.length
  let everRelinked = 0, releasedOnce = 0, neverLinked = 0, linkedAtEnd = 0
  for (let a = 0; a < n; a++) {
    let seenFree = false, seenLinkedAfterFree = false, seenLinked = false
    for (const f of frames) {
      const linked = f.catFree[a] === 0
      if (linked) { seenLinked = true; if (seenFree) seenLinkedAfterFree = true }
      else seenFree = true
    }
    if (!seenLinked) neverLinked++
    if (seenLinked && seenFree) releasedOnce++
    if (seenLinkedAfterFree) everRelinked++
    if (frames[frames.length - 1].catFree[a] === 0) linkedAtEnd++
  }
  console.log(`FATE frames=${frames.map(f => f.step).join(',')} catalysts=${n} neverLinkedAtAnySample=${neverLinked} `
    + `linkedThenFreeAtSomeSample=${releasedOnce} RE-LINKED_AFTER_BEING_FREE=${everRelinked} linkedAtLastFrame=${linkedAtEnd}`)
}
