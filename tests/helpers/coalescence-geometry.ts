// Pure-CPU geometry for the coalescence question -- task 'coalescence' (2026-08-20),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/coalescence-report.md.
//
// Three mechanisms had to be told apart (free-head poisoning / a hydration-fusion barrier /
// spontaneous curvature), and each one's signature is a geometric property of a snapshot or of a
// pair of consecutive snapshots. Everything here reads a decoded checkpoint's raw positions and the
// project's OWN clustering rule (engine/src/aggregate.ts's clusterComponents with the same
// wcaCutoff+wc cutoff soup/src/aggregates.ts already uses) -- there is no second definition of what
// an aggregate is anywhere in this file.
//
// The one non-obvious choice, and why it is the honest one: the free-head enrichment is reported as
// the RATIO of the free-head radial profile to the WATER radial profile, g_O(r)/g_W(r). Both species
// probe the identical aggregate geometry from the identical bulk, so the shell volume -- which is the
// hard thing to define for a 34-object box of rods straddling periodic faces -- cancels EXACTLY and
// no normalisation is assumed. Water is also the physically right reference: it is the competing
// adsorbate, and "is a free head more likely than a water bead to be found touching this surface" is
// precisely the poisoning question.

import { clusterComponents, shapeOf, unwrapAggregate } from '../../engine/src/aggregate'

export interface Agg {
  id: number
  /** Amphiphile-member particle indices (carbon+head) of this aggregate. */
  idx: number[]
  /** Head-bead particle indices -- the surface sites the coverage number is per. */
  heads: number[]
  amphiphileCount: number
  particleCount: number
  centre: [number, number, number]
  radiusOfGyration: number
  principalMoments: [number, number, number]
  flatnessRatio: number
  inPlaneSymmetry: number
  /** Bounding-sphere radius about `centre` in unwrapped coordinates -- used only to prune pair work. */
  maxRadius: number
}

const mi = (d: number, L: number) => d - L * Math.round(d / L)

export function subPositions(pos: Float32Array, idx: readonly number[]): Float32Array {
  const out = new Float32Array(idx.length * 4)
  for (let k = 0; k < idx.length; k++) {
    const i = idx[k]
    out[k * 4] = pos[i * 4]
    out[k * 4 + 1] = pos[i * 4 + 1]
    out[k * 4 + 2] = pos[i * 4 + 2]
    out[k * 4 + 3] = pos[i * 4 + 3]
  }
  return out
}

/** Every aggregate (not only the largest few, which is what soup/src/aggregates.ts's own
 * detailAggregateCount caps at) with full shape, above `minAmph` amphiphiles. */
export function buildAggregates(
  pos: Float32Array,
  box: [number, number, number],
  amphHeadIndex: ReadonlySet<number>,
  memberIdx: ReadonlySet<number>,
  cutoff: number,
  minAmph: number,
  headKind: number,
  /** 'project' = engine/src/aggregate.ts's own z-open rule, the one every published number in this
   * project was measured with; 'periodic3d' = the same rule with the z image restored (see
   * clusterComponents3D below). Both are reported side by side; neither replaces the other. */
  clustering: 'project' | 'periodic3d' = 'project',
): Agg[] {
  const idxArr = Array.from(memberIdx)
  const memberPositions = subPositions(pos, idxArr)
  const comps = clustering === 'periodic3d' ? clusterComponents3D(memberPositions, box, cutoff) : clusterComponents(memberPositions, box, cutoff)
  const out: Agg[] = []
  let id = 0
  for (const local of comps) {
    const idx = local.map((k) => idxArr[k])
    const heads: number[] = []
    let amphiphileCount = 0
    for (const i of idx) {
      if (Math.round(pos[i * 4 + 3]) === headKind) heads.push(i)
      if (amphHeadIndex.has(i)) amphiphileCount++
    }
    if (amphiphileCount < minAmph) continue
    const sub = subPositions(pos, idx)
    const un = unwrapAggregate(sub, box, [true, true, true])
    const s = shapeOf(un)
    let maxR = 0
    for (let k = 0; k < idx.length; k++) {
      const dx = un[k * 4] - s.centre[0]
      const dy = un[k * 4 + 1] - s.centre[1]
      const dz = un[k * 4 + 2] - s.centre[2]
      const r = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (r > maxR) maxR = r
    }
    const pm = s.principalMoments
    out.push({
      id: id++,
      idx,
      heads,
      amphiphileCount,
      particleCount: idx.length,
      centre: [
        ((s.centre[0] % box[0]) + box[0]) % box[0],
        ((s.centre[1] % box[1]) + box[1]) % box[1],
        ((s.centre[2] % box[2]) + box[2]) % box[2],
      ],
      radiusOfGyration: s.radiusOfGyration,
      principalMoments: [pm[0], pm[1], pm[2]],
      flatnessRatio: pm[2] > 0 ? pm[0] / pm[2] : 0,
      inPlaneSymmetry: pm[2] > 0 ? pm[1] / pm[2] : 0,
      maxRadius: maxR,
    })
  }
  out.sort((a, b) => b.amphiphileCount - a.amphiphileCount)
  out.forEach((a, k) => (a.id = k))
  return out
}


// --- fully periodic clustering, as a CONTROL on the project's own z-open rule -------------------
//
// engine/src/aggregate.ts's buildClusterUnionFind was written for the membrane engine's own boundary
// convention -- "z is unbounded (open boundary) so it gets a plain integer cell index with no
// wrapping", its own comment -- because a bilayer patch in vacuum genuinely has no image across z.
// The SOUP is a bulk system that wraps all three axes every step (soup/src/sim.ts's own header: "a
// bulk soup has no preferred axis"), so an aggregate straddling the z face is split in two by that
// rule and counted twice. This function is the same connectivity rule with the z image restored, so
// the size of that effect can be MEASURED rather than argued about. Nothing else differs: same
// cutoff, same union-find semantics, same component ordering.
export function clusterComponents3D(positions: Float32Array, box: [number, number, number], cutoff: number): number[][] {
  const n = positions.length / 4
  if (n === 0) return []
  const parent = new Int32Array(n)
  for (let i = 0; i < n; i++) parent[i] = i
  const find = (x: number): number => {
    let r = x
    while (parent[r] !== r) r = parent[r]
    while (parent[x] !== r) {
      const nx = parent[x]
      parent[x] = r
      x = nx
    }
    return r
  }
  const union = (a: number, b: number) => {
    const ra = find(a)
    const rb = find(b)
    if (ra !== rb) parent[rb] = ra
  }
  const nc: [number, number, number] = [
    Math.max(1, Math.floor(box[0] / cutoff)),
    Math.max(1, Math.floor(box[1] / cutoff)),
    Math.max(1, Math.floor(box[2] / cutoff)),
  ]
  const w: [number, number, number] = [box[0] / nc[0], box[1] / nc[1], box[2] / nc[2]]
  const buckets = new Map<number, number[]>()
  const cells = new Int32Array(n * 3)
  for (let i = 0; i < n; i++) {
    const c: number[] = []
    for (let ax = 0; ax < 3; ax++) {
      const v = ((positions[i * 4 + ax] % box[ax]) + box[ax]) % box[ax]
      c.push(Math.min(nc[ax] - 1, Math.floor(v / w[ax])))
      cells[i * 3 + ax] = c[ax]
    }
    const key = (c[2] * nc[1] + c[1]) * nc[0] + c[0]
    const arr = buckets.get(key)
    if (arr) arr.push(i)
    else buckets.set(key, [i])
  }
  const c2 = cutoff * cutoff
  for (let i = 0; i < n; i++) {
    for (let dz = -1; dz <= 1; dz++) {
      const kz = (((cells[i * 3 + 2] + dz) % nc[2]) + nc[2]) % nc[2]
      for (let dy = -1; dy <= 1; dy++) {
        const ky = (((cells[i * 3 + 1] + dy) % nc[1]) + nc[1]) % nc[1]
        for (let dx = -1; dx <= 1; dx++) {
          const kx = (((cells[i * 3] + dx) % nc[0]) + nc[0]) % nc[0]
          const arr = buckets.get((kz * nc[1] + ky) * nc[0] + kx)
          if (!arr) continue
          for (const j of arr) {
            if (j <= i) continue
            let r2 = 0
            for (let ax = 0; ax < 3; ax++) {
              const d = mi(positions[i * 4 + ax] - positions[j * 4 + ax], box[ax])
              r2 += d * d
            }
            if (r2 <= c2) union(i, j)
          }
        }
      }
    }
  }
  const byRoot = new Map<number, number[]>()
  for (let i = 0; i < n; i++) {
    const r = find(i)
    const a = byRoot.get(r)
    if (a) a.push(i)
    else byRoot.set(r, [i])
  }
  return Array.from(byRoot.values()).sort((a, b) => b.length - a.length)
}

// --- a uniform cell list over the aggregates' member beads ---------------------------------------

interface MemberGrid {
  cell: number
  n: [number, number, number]
  start: Int32Array
  items: Int32Array
  px: Float32Array
  py: Float32Array
  pz: Float32Array
  agg: Int32Array
  box: [number, number, number]
}

function buildMemberGrid(pos: Float32Array, box: [number, number, number], aggs: Agg[], rcut: number): MemberGrid {
  let m = 0
  for (const a of aggs) m += a.idx.length
  const px = new Float32Array(m)
  const py = new Float32Array(m)
  const pz = new Float32Array(m)
  const agg = new Int32Array(m)
  let k = 0
  for (const a of aggs) {
    for (const i of a.idx) {
      px[k] = pos[i * 4]
      py[k] = pos[i * 4 + 1]
      pz[k] = pos[i * 4 + 2]
      agg[k] = a.id
      k++
    }
  }
  const n: [number, number, number] = [
    Math.max(1, Math.floor(box[0] / rcut)),
    Math.max(1, Math.floor(box[1] / rcut)),
    Math.max(1, Math.floor(box[2] / rcut)),
  ]
  const nCells = n[0] * n[1] * n[2]
  const counts = new Int32Array(nCells + 1)
  const cellOf = new Int32Array(m)
  const wrapCell = (v: number, L: number, nn: number) => {
    let c = Math.floor((((v % L) + L) % L) / (L / nn))
    if (c >= nn) c = nn - 1
    return c
  }
  for (let i = 0; i < m; i++) {
    const cx = wrapCell(px[i], box[0], n[0])
    const cy = wrapCell(py[i], box[1], n[1])
    const cz = wrapCell(pz[i], box[2], n[2])
    const c = (cz * n[1] + cy) * n[0] + cx
    cellOf[i] = c
    counts[c + 1]++
  }
  for (let c = 0; c < nCells; c++) counts[c + 1] += counts[c]
  const start = counts
  const cursor = Int32Array.from(start.subarray(0, nCells))
  const items = new Int32Array(m)
  for (let i = 0; i < m; i++) items[cursor[cellOf[i]]++] = i
  return { cell: rcut, n, start, items, px, py, pz, agg, box }
}

/** Nearest two DISTINCT aggregates to a point, within rcut. -1 when none. */
function nearestTwo(g: MemberGrid, x: number, y: number, z: number, rcut: number, skipMember: number): [number, number, number, number] {
  const { box, n } = g
  const hx = box[0] / n[0]
  const hy = box[1] / n[1]
  const hz = box[2] / n[2]
  const cx = Math.min(n[0] - 1, Math.floor((((x % box[0]) + box[0]) % box[0]) / hx))
  const cy = Math.min(n[1] - 1, Math.floor((((y % box[1]) + box[1]) % box[1]) / hy))
  const cz = Math.min(n[2] - 1, Math.floor((((z % box[2]) + box[2]) % box[2]) / hz))
  let d1 = Infinity
  let a1 = -1
  let d2 = Infinity
  let a2 = -1
  const r2cut = rcut * rcut
  for (let oz = -1; oz <= 1; oz++) {
    const kz = (((cz + oz) % n[2]) + n[2]) % n[2]
    for (let oy = -1; oy <= 1; oy++) {
      const ky = (((cy + oy) % n[1]) + n[1]) % n[1]
      for (let ox = -1; ox <= 1; ox++) {
        const kx = (((cx + ox) % n[0]) + n[0]) % n[0]
        const c = (kz * n[1] + ky) * n[0] + kx
        for (let s = g.start[c]; s < g.start[c + 1]; s++) {
          const j = g.items[s]
          if (j === skipMember) continue
          const dx = mi(x - g.px[j], box[0])
          const dy = mi(y - g.py[j], box[1])
          const dz = mi(z - g.pz[j], box[2])
          const r2 = dx * dx + dy * dy + dz * dz
          if (r2 >= r2cut) continue
          const a = g.agg[j]
          const r = Math.sqrt(r2)
          if (a === a1) {
            if (r < d1) d1 = r
          } else if (a === a2) {
            if (r < d2) d2 = r
            if (d2 < d1) {
              const td = d1
              const ta = a1
              d1 = d2
              a1 = a2
              d2 = td
              a2 = ta
            }
          } else if (r < d1) {
            d2 = d1
            a2 = a1
            d1 = r
            a1 = a
          } else if (r < d2) {
            d2 = r
            a2 = a
          }
        }
      }
    }
  }
  return [d1, a1, d2, a2]
}

export interface SolventProfile {
  /** Bin width, and the histogram of nearest-aggregate distance for each species. */
  binWidth: number
  bins: number
  freeHead: Int32Array
  water: Int32Array
  /** Free heads / water beads whose nearest member bead is within `contact`. */
  freeHeadInContact: number
  waterInContact: number
  /** Per aggregate id: free heads and water beads in contact with it. */
  freeHeadPerAgg: Int32Array
  waterPerAgg: Int32Array
  /** Water beads simultaneously within `gapCut` of two DIFFERENT aggregates, keyed "a-b". */
  bridgingWater: Map<string, number>
  totalFreeHeads: number
  totalWater: number
}

export function solventProfile(
  pos: Float32Array,
  box: [number, number, number],
  aggs: Agg[],
  freeHeadIdx: readonly number[],
  waterIdx: readonly number[],
  opts: { binWidth: number; bins: number; contact: number; gapCut: number },
): SolventProfile {
  const rcut = opts.binWidth * opts.bins
  const g = buildMemberGrid(pos, box, aggs, rcut)
  const freeHead = new Int32Array(opts.bins)
  const water = new Int32Array(opts.bins)
  const freeHeadPerAgg = new Int32Array(aggs.length)
  const waterPerAgg = new Int32Array(aggs.length)
  const bridgingWater = new Map<string, number>()
  let fhContact = 0
  let wContact = 0
  const scan = (idx: readonly number[], hist: Int32Array, perAgg: Int32Array, bridging: boolean) => {
    let contact = 0
    for (const i of idx) {
      const [d1, a1, d2, a2] = nearestTwo(g, pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2], rcut, -1)
      if (a1 < 0) continue
      const b = Math.floor(d1 / opts.binWidth)
      if (b < opts.bins) hist[b]++
      if (d1 <= opts.contact) {
        contact++
        perAgg[a1]++
      }
      if (bridging && a2 >= 0 && d1 <= opts.gapCut && d2 <= opts.gapCut) {
        const key = a1 < a2 ? `${a1}-${a2}` : `${a2}-${a1}`
        bridgingWater.set(key, (bridgingWater.get(key) ?? 0) + 1)
      }
    }
    return contact
  }
  fhContact = scan(freeHeadIdx, freeHead, freeHeadPerAgg, false)
  wContact = scan(waterIdx, water, waterPerAgg, true)
  return {
    binWidth: opts.binWidth,
    bins: opts.bins,
    freeHead,
    water,
    freeHeadInContact: fhContact,
    waterInContact: wContact,
    freeHeadPerAgg,
    waterPerAgg,
    bridgingWater,
    totalFreeHeads: freeHeadIdx.length,
    totalWater: waterIdx.length,
  }
}

export interface PairGap {
  a: number
  b: number
  /** Minimum member-bead-to-member-bead distance between the two aggregates, minimum-image. */
  gap: number
}

/** Every aggregate pair whose closest approach is under `rcut`, by brute force over member beads
 * with a bounding-sphere prefilter (the prefilter only skips pairs that provably cannot be closer
 * than rcut, so the result is exact). */
export function pairGaps(pos: Float32Array, box: [number, number, number], aggs: Agg[], rcut: number): PairGap[] {
  const out: PairGap[] = []
  for (let a = 0; a < aggs.length; a++) {
    for (let b = a + 1; b < aggs.length; b++) {
      const A = aggs[a]
      const B = aggs[b]
      const dx = mi(A.centre[0] - B.centre[0], box[0])
      const dy = mi(A.centre[1] - B.centre[1], box[1])
      const dz = mi(A.centre[2] - B.centre[2], box[2])
      const dc = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (dc - A.maxRadius - B.maxRadius > rcut) continue
      let best = Infinity
      for (const i of A.idx) {
        const xi = pos[i * 4]
        const yi = pos[i * 4 + 1]
        const zi = pos[i * 4 + 2]
        for (const j of B.idx) {
          const ex = mi(xi - pos[j * 4], box[0])
          const ey = mi(yi - pos[j * 4 + 1], box[1])
          const ez = mi(zi - pos[j * 4 + 2], box[2])
          const r2 = ex * ex + ey * ey + ez * ez
          if (r2 < best) best = r2
        }
      }
      best = Math.sqrt(best)
      if (best <= rcut) out.push({ a, b, gap: best })
    }
  }
  out.sort((p, q) => p.gap - q.gap)
  return out
}

export interface MergeStats {
  /** Aggregates at t whose amphiphile heads ended up mostly inside ONE aggregate at t+1 together
   * with another aggregate's heads -- a merge. */
  merges: number
  /** Aggregates at t whose heads split across two or more aggregates at t+1, each taking >= share. */
  fissions: number
  /** Aggregates that neither merged nor split (their heads stayed in one aggregate that received no
   * other aggregate's heads). */
  persisted: number
  /** Amphiphiles that changed aggregate identity between the two snapshots, over the amphiphiles
   * present in both -- the material-exchange rate. */
  exchangedFraction: number
  matchedHeads: number
  detail: string[]
  /** prev aggregate id -> the next aggregate id that took most of its heads (-1 when none did) --
   * the successor map the encounter bookkeeping needs to ask "did THIS pair become one object". */
  successor: Map<number, number>
}

/** Merge/fission bookkeeping between two snapshots, by HEAD-BEAD identity (a particle index is
 * stable for the whole run -- organics are never created or destroyed), restricted to heads that
 * belong to a qualifying aggregate in BOTH snapshots so that chemistry (a head losing its tail)
 * cannot masquerade as a fission. */
export function mergeStats(prev: Agg[], next: Agg[], shareMin: number): MergeStats {
  const ownerNext = new Map<number, number>()
  for (const a of next) for (const h of a.heads) ownerNext.set(h, a.id)
  const ownerPrev = new Map<number, number>()
  for (const a of prev) for (const h of a.heads) ownerPrev.set(h, a.id)
  let matched = 0
  let moved = 0
  // prev id -> map of next id -> count
  const fwd = new Map<number, Map<number, number>>()
  for (const a of prev) {
    const m = new Map<number, number>()
    for (const h of a.heads) {
      const nid = ownerNext.get(h)
      if (nid === undefined) continue
      matched++
      m.set(nid, (m.get(nid) ?? 0) + 1)
    }
    fwd.set(a.id, m)
  }
  // next id -> how many DIFFERENT prev aggregates contributed >= shareMin of their own heads
  const contributors = new Map<number, number[]>()
  const detail: string[] = []
  let fissions = 0
  for (const a of prev) {
    const m = fwd.get(a.id)!
    const tot = Array.from(m.values()).reduce((s, v) => s + v, 0)
    if (tot === 0) continue
    let big = 0
    for (const [nid, c] of m) {
      if (c / tot >= shareMin) {
        big++
        const arr = contributors.get(nid) ?? []
        arr.push(a.id)
        contributors.set(nid, arr)
      }
    }
    if (big >= 2) fissions++
    const dominant = Array.from(m.entries()).sort((p, q) => q[1] - p[1])[0]
    moved += tot - dominant[1]
  }
  let merges = 0
  for (const [nid, arr] of contributors) if (arr.length >= 2) { merges++; detail.push(`merge->${nid}<-${arr.join('+')}`) }
  const persisted = prev.length - merges - fissions
  const successor = new Map<number, number>()
  for (const a of prev) {
    const m = fwd.get(a.id)!
    const top = Array.from(m.entries()).sort((p, q) => q[1] - p[1])[0]
    successor.set(a.id, top ? top[0] : -1)
  }
  return { merges, fissions, persisted, exchangedFraction: matched > 0 ? moved / matched : 0, matchedHeads: matched, detail, successor }
}
