// Energy-formula and pairwise-bookkeeping machinery for tests/rim-lambda-insitu.test.ts — moved
// verbatim out of that file by responsibility (file-size rule, root CLAUDE.md). Every energy formula
// below is a literal transcription of the antiderivatives already written in
// engine/wgsl/forces.wgsl (wca_v/fene_v/bend_v/attr_v) and the pairing/gating logic already written
// in soup/wgsl/step.wgsl (nonbondedSoup/bondedForce) — read here, not altered. No new physics.

import { loadParams } from '../../engine/src/params'
import { loadSoup } from '../../soup/src/rules'
import type { Amphiphile } from '../../soup/src/amphiphile'

// --- energy formulas, transcribed verbatim from engine/wgsl/forces.wgsl (wca_v/fene_v/bend_v/
// attr_v) -- see that file's own comments for the derivation of each; nothing here is a new formula.
function wcaCut(b: number): number {
  return Math.pow(2, 1 / 6) * b
}
export function wcaV(r: number, b: number, epsilon: number): number {
  if (r >= wcaCut(b)) return 0
  const s6 = Math.pow(b / r, 6)
  const q = 2 * s6 - 1
  return epsilon * q * q
}
export function feneV(r: number, k: number, rInf: number): number {
  const x = r / rInf
  return -0.5 * k * rInf * rInf * Math.log(1 - x * x)
}
export function bendV(r: number, k: number, r0: number): number {
  const d = r - r0
  return 0.5 * k * d * d
}
export function attrV(r: number, rc: number, wc: number, epsilon: number): number {
  if (r < rc) return -epsilon
  if (r > rc + wc) return 0
  const x = (Math.PI * (r - rc)) / (2 * wc)
  const c = Math.cos(x)
  return -epsilon * c * c
}

// Minimum-image displacement/distance, 3-axis periodic -- matches soup/wgsl/step.wgsl's mi3(), the
// soup's own full-3D wrap (unlike the membrane engine's x,y-only mi()), since this checkpoint is a
// soup snapshot, not a membrane patch.
export function mi3(d: number, L: number): number {
  return d - Math.round(d / L) * L
}
export function dist3(px: Float32Array, i: number, j: number, box: [number, number, number]): number {
  const dx = mi3(px[i * 4] - px[j * 4], box[0])
  const dy = mi3(px[i * 4 + 1] - px[j * 4 + 1], box[1])
  const dz = mi3(px[i * 4 + 2] - px[j * 4 + 2], box[2])
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

// Cell-list pair enumerator over a GIVEN index list (into the full `positions` array), 3-axis
// periodic on the FULL box -- soup/wgsl/step.wgsl's soupForceWalk walks a 3x3x3 (or wider) cell
// neighbourhood; this is the same idea at O(n) instead of O(n^2), restricted to a caller-chosen
// particle subset (the aggregate's own members) rather than the whole 93200-particle system, since
// this file is only ever asked for aggregate-internal (lipid-only, no-solvent) energy -- the same
// convention tests/line-tension.test.ts's own prepared patch/reference pair already used (those
// systems contained ONLY lipids, zero solvent, by construction).
export function forEachPairWithinCutoff(
  idx: number[],
  positions: Float32Array,
  box: [number, number, number],
  cutoff: number,
  fn: (i: number, j: number, r: number) => void,
): void {
  const n = idx.length
  const nx = Math.max(1, Math.floor(box[0] / cutoff))
  const ny = Math.max(1, Math.floor(box[1] / cutoff))
  const nz = Math.max(1, Math.floor(box[2] / cutoff))
  const wx = box[0] / nx
  const wy = box[1] / ny
  const wz = box[2] / nz
  const cellOf = (k: number): [number, number, number] => {
    const p = idx[k]
    const x = ((positions[p * 4] % box[0]) + box[0]) % box[0]
    const y = ((positions[p * 4 + 1] % box[1]) + box[1]) % box[1]
    const z = ((positions[p * 4 + 2] % box[2]) + box[2]) % box[2]
    return [Math.min(nx - 1, Math.floor(x / wx)), Math.min(ny - 1, Math.floor(y / wy)), Math.min(nz - 1, Math.floor(z / wz))]
  }
  const buckets = new Map<string, number[]>()
  const cells: [number, number, number][] = []
  for (let k = 0; k < n; k++) {
    const c = cellOf(k)
    cells.push(c)
    const key = `${c[0]},${c[1]},${c[2]}`
    const b = buckets.get(key)
    if (b) b.push(k)
    else buckets.set(key, [k])
  }
  const cutoff2 = cutoff * cutoff
  for (let k = 0; k < n; k++) {
    const [cx, cy, cz] = cells[k]
    const i = idx[k]
    for (let dx = -1; dx <= 1; dx++) {
      const ncx = ((cx + dx) % nx + nx) % nx
      for (let dy = -1; dy <= 1; dy++) {
        const ncy = ((cy + dy) % ny + ny) % ny
        for (let dz = -1; dz <= 1; dz++) {
          const ncz = ((cz + dz) % nz + nz) % nz
          const bucket = buckets.get(`${ncx},${ncy},${ncz}`)
          if (!bucket) continue
          for (const kk of bucket) {
            if (kk <= k) continue
            const j = idx[kk]
            const r = dist3(positions, i, j, box)
            if (r * r <= cutoff2) fn(i, j, r)
          }
        }
      }
    }
  }
}

export interface EnergyBasis {
  sigma: number
  epsilon: number
  feneK: number
  feneRInf: number
  bendK: number
  bendR0: number
  wc: number
  rc: number // wcaCutoff(sigma * tail_tail) -- the FIXED attraction-well onset, forces.wgsl's own
  // b_tt convention (attr_v/attr_dv use P.b_tt, not the per-pair WCA radius) -- see forces.wgsl.
  radiusOf: (kindIndex: number) => number
  polarOf: (kindIndex: number) => boolean
}

export function makeBasis(): EnergyBasis {
  const p = loadParams()
  const soup = loadSoup()
  const radiusByKind = soup.monomers.map((m) => m.radiusSigma)
  const polarByKind = soup.monomers.map((m) => m.polar)
  return {
    sigma: p.sigma,
    epsilon: p.epsilon,
    feneK: p.fene.k,
    feneRInf: p.fene.rInf,
    bendK: p.bend.k,
    bendR0: p.bend.r0,
    wc: p.attraction.wc,
    rc: wcaCut(p.sigma * p.beadSizes.tail_tail),
    radiusOf: (k) => radiusByKind[k],
    polarOf: (k) => polarByKind[k],
  }
}

/** Bonded (FENE + bend) + nonbonded (WCA + attraction) potential energy among exactly the particles
 * in `workingSet` -- both endpoints of every term must be in the set, matching the prepared patch's
 * own "lipid-only, no solvent" convention (line-tension.test.ts's patch/reference systems contained
 * ONLY lipids). `adjacency` is the FULL covalent bond graph (soup/wgsl/step.wgsl's bondedForce: FENE
 * for every occupied slot, bend for every pair of a shared bonded middle's own partners -- credited
 * once per triple here, by its unique middle particle, mirroring that file's own per-particle loop
 * collapsed to unique triples). */
export function pairwiseEnergy(
  workingSet: Set<number>,
  adjacency: number[][],
  positions: Float32Array,
  box: [number, number, number],
  cutoff: number,
  basis: EnergyBasis,
): { fene: number; bend: number; nonbonded: number; total: number } {
  let fene = 0
  for (const i of workingSet) {
    for (const j of adjacency[i]) {
      if (j <= i || !workingSet.has(j)) continue
      const r = dist3(positions, i, j, box)
      fene += feneV(r, basis.feneK, basis.feneRInf)
    }
  }
  let bend = 0
  for (const m of workingSet) {
    const partners = adjacency[m].filter((p) => workingSet.has(p))
    for (let a = 0; a < partners.length; a++) {
      for (let b = a + 1; b < partners.length; b++) {
        const r = dist3(positions, partners[a], partners[b], box)
        bend += bendV(r, basis.bendK, basis.bendR0)
      }
    }
  }
  let nonbonded = 0
  const idx = Array.from(workingSet)
  forEachPairWithinCutoff(idx, positions, box, cutoff, (i, j, r) => {
    const ki = Math.round(positions[i * 4 + 3])
    const kj = Math.round(positions[j * 4 + 3])
    const b = basis.sigma * (basis.radiusOf(ki) + basis.radiusOf(kj)) * 0.5
    nonbonded += wcaV(r, b, basis.epsilon)
    if (!basis.polarOf(ki) && !basis.polarOf(kj)) nonbonded += attrV(r, basis.rc, basis.wc, basis.epsilon)
  })
  return { fene, bend, nonbonded, total: fene + bend + nonbonded }
}

/** Local coordination number of every particle in `idx`: how many OTHER members of `idx` sit within
 * `cutoff` -- the packing-density proxy this file uses to tell a rim (edge-exposed, structurally
 * missing neighbours on the open side) amphiphile from an interior one, the in-situ analogue of "has
 * a free edge" for an object with no periodic twin to subtract against. */
export function coordinationNumbers(idx: number[], positions: Float32Array, box: [number, number, number], cutoff: number): Map<number, number> {
  const deg = new Map<number, number>()
  for (const i of idx) deg.set(i, 0)
  forEachPairWithinCutoff(idx, positions, box, cutoff, (i, j) => {
    deg.set(i, (deg.get(i) ?? 0) + 1)
    deg.set(j, (deg.get(j) ?? 0) + 1)
  })
  return deg
}

/** Greedy nearest-neighbour chain length through `pts` (visits every point once, always hopping to
 * the closest unvisited one) -- a real, measured contour length through the ACTUAL rim-classified
 * head positions, not an assumed formula. Returns both the open-path length and the length with the
 * final point re-joined to the first (closed-loop reading), since a rim is topologically a loop. */
export function greedyChainLength(pts: [number, number, number][], box: [number, number, number]): { open: number; closed: number } {
  const n = pts.length
  if (n < 2) return { open: 0, closed: 0 }
  const visited = new Array(n).fill(false)
  let cur = 0
  visited[0] = true
  let open = 0
  const order = [0]
  for (let step = 1; step < n; step++) {
    let best = -1
    let bestD = Infinity
    for (let k = 0; k < n; k++) {
      if (visited[k]) continue
      const dx = mi3(pts[cur][0] - pts[k][0], box[0])
      const dy = mi3(pts[cur][1] - pts[k][1], box[1])
      const dz = mi3(pts[cur][2] - pts[k][2], box[2])
      const d = dx * dx + dy * dy + dz * dz
      if (d < bestD) {
        bestD = d
        best = k
      }
    }
    open += Math.sqrt(bestD)
    visited[best] = true
    order.push(best)
    cur = best
  }
  const first = pts[order[0]]
  const last = pts[order[order.length - 1]]
  const dx = mi3(last[0] - first[0], box[0])
  const dy = mi3(last[1] - first[1], box[1])
  const dz = mi3(last[2] - first[2], box[2])
  const closingEdge = Math.sqrt(dx * dx + dy * dy + dz * dz)
  return { open, closed: open + closingEdge }
}

/** Everything evaluateRimSplit() needs from the calling test that it does not receive as its own
 * rimAmph/interiorAmph parameters -- moved out of tests/rim-lambda-insitu.test.ts's first test body
 * (file-size rule, root CLAUDE.md) as an explicit context object instead of a closure capture. */
export interface RimEvalContext {
  aggOriginalIdx: number[]
  aggSet: Set<number>
  adjacency: number[][]
  positions: Float32Array
  box: [number, number, number]
  cutoff: number
  basis: EnergyBasis
  radiusByOriginal: Map<number, number>
  unwrappedDirOf: (originalIdx: number) => [number, number, number]
}

/** Splits the aggregate's amphiphiles into a caller-chosen rim/interior partition and reports the
 * nonbonded energy excess of the rim relative to the interior baseline, plus the rim's own measured
 * contour length by two independent methods (median-nearest-neighbour-spacing x count, and a greedy
 * nearest-neighbour chain) -- see tests/rim-lambda-insitu.test.ts's own header for why the PRIMARY
 * lambda isolates the nonbonded (WCA+attraction) term rather than the full fene+bend+nonbonded
 * excess. Moved verbatim out of that file's first test (it was a nested function closing over the
 * same values `ctx` now carries explicitly). */
export function evaluateRimSplit(rimAmph: Amphiphile[], interiorAmph: Amphiphile[], ctx: RimEvalContext) {
  const { aggOriginalIdx, aggSet, adjacency, positions, box, cutoff, basis, radiusByOriginal, unwrappedDirOf } = ctx
  const interiorSet = new Set<number>()
  for (const a of interiorAmph) {
    interiorSet.add(a.headIndex)
    for (const c of a.chain) interiorSet.add(c)
  }
  const eTotal = pairwiseEnergy(aggSet, adjacency, positions, box, cutoff, basis)
  const eInterior = pairwiseEnergy(interiorSet, adjacency, positions, box, cutoff, basis)
  // PER-PARTICLE baseline, not per-amphiphile: amphiphile chain length varies (this soup's own
  // emergent Flory-like distribution, not a fixed 3-bead lipid), and FENE/bend/nonbonded energy
  // scales with chain length almost independently of rim-vs-interior packing -- normalising by
  // amphiphile COUNT would confound "this amphiphile is longer" with "this amphiphile is at the
  // edge". Normalising by member PARTICLE count instead treats every bead's own local energy
  // density as the unit being compared, which is what the rim/bulk packing difference actually
  // acts on.
  const rimParticleCount = aggOriginalIdx.length - interiorSet.size
  const perParticleInterior = eInterior.total / interiorSet.size
  const eExcessFull = eTotal.total - aggOriginalIdx.length * perParticleInterior

  // Primary in-situ lambda: the NONBONDED (WCA+attraction) excess only, not fene+bend. This
  // matches what the prepared-patch measurement (line-tension.test.ts) actually isolated: patch
  // and reference there shared the EXACT SAME lipid topology (same fixed 3-bead FENE/bend terms
  // on both sides), so their subtraction cancelled fene+bend exactly and left only the nonbonded
  // edge cost -- the classical membrane-physics definition of line tension (lost lateral cohesive
  // contacts at a free edge), not intramolecular chain strain. Here there is no periodic twin to
  // subtract against, and this snapshot's rim-classified amphiphiles have a materially SHORTER
  // mean chain length than the interior ones (see meanChainLengthRim/Interior below) -- since this
  // soup's chain length varies per amphiphile (unlike the fixed-lipid patch), a fene+bend excess
  // mixes "this amphiphile is at the edge" with "this amphiphile happens to be short", which the
  // prepared patch could never produce. Isolating nonbonded removes that confound at the source.
  const perParticleInteriorNb = eInterior.nonbonded / interiorSet.size
  const eExcessNb = eTotal.nonbonded - aggOriginalIdx.length * perParticleInteriorNb

  const rimHeadPts = rimAmph.map((a) => {
    const i = a.headIndex
    return [positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2]] as [number, number, number]
  })
  // Method A: count x measured median nearest-rim-head spacing.
  const nn: number[] = rimHeadPts.map((p, i) => {
    let best = Infinity
    for (let j = 0; j < rimHeadPts.length; j++) {
      if (j === i) continue
      const q = rimHeadPts[j]
      const dx = mi3(p[0] - q[0], box[0])
      const dy = mi3(p[1] - q[1], box[1])
      const dz = mi3(p[2] - q[2], box[2])
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (d < best) best = d
    }
    return best
  })
  nn.sort((x, y) => x - y)
  const medianNN = nn[Math.floor(nn.length / 2)]
  const rimLengthA = rimAmph.length * medianNN

  // Method B: greedy nearest-neighbour chain through the rim heads themselves.
  const chain = greedyChainLength(rimHeadPts, box)

  // Full (fene+bend+nonbonded) excess, reported alongside for transparency -- see the note above
  // on why it is confounded by chain length and NOT the headline number.
  const lambdaFull = eExcessFull / rimLengthA
  // Primary (nonbonded-only) lambda, all three rim-length conventions.
  const lambdaA = eExcessNb / rimLengthA
  const lambdaBOpen = eExcessNb / chain.open
  const lambdaBClosed = eExcessNb / chain.closed

  const meanChainRim = rimAmph.reduce((s, a) => s + a.length, 0) / rimAmph.length
  const meanChainInterior = interiorAmph.reduce((s, a) => s + a.length, 0) / interiorAmph.length
  const meanRadiusRim = rimAmph.reduce((s, a) => s + (radiusByOriginal.get(a.headIndex) ?? 0), 0) / rimAmph.length
  const meanRadiusInterior = interiorAmph.reduce((s, a) => s + (radiusByOriginal.get(a.headIndex) ?? 0), 0) / interiorAmph.length

  // Angular clustering check: do the rim-classified heads sit in ONE localised patch on the
  // shell (consistent with bordering a single real hole/rim) or scattered all over (consistent
  // with generic packing noise, not a rim)? Mean pairwise angle (from the shape's own centroid)
  // among rim heads, vs. the ~90 degree value uniformly-random points on a sphere would give.
  const rimDirs = rimAmph.map((a) => unwrappedDirOf(a.headIndex))
  let angleSum = 0
  let anglePairs = 0
  for (let i = 0; i < rimDirs.length; i++) {
    for (let j = i + 1; j < rimDirs.length; j++) {
      const dot = Math.max(-1, Math.min(1, rimDirs[i][0] * rimDirs[j][0] + rimDirs[i][1] * rimDirs[j][1] + rimDirs[i][2] * rimDirs[j][2]))
      angleSum += (Math.acos(dot) * 180) / Math.PI
      anglePairs++
    }
  }
  const meanPairwiseAngleDeg = angleSum / anglePairs

  return {
    numRimAmphiphiles: rimAmph.length,
    numInteriorAmphiphiles: interiorAmph.length,
    meanChainLengthRim: meanChainRim,
    meanChainLengthInterior: meanChainInterior,
    meanRadiusRim,
    meanRadiusInterior,
    meanPairwiseAngleDeg,
    rimParticleCount,
    interiorParticleCount: interiorSet.size,
    eTotal: eTotal.total,
    eTotalBreakdown: eTotal,
    eInteriorOnly: eInterior.total,
    eInteriorBreakdown: eInterior,
    perParticleInterior,
    perParticleInteriorNb,
    eExcessFull,
    eExcessNb,
    lambdaFull,
    eExcess: eExcessNb,
    medianNNSpacing: medianNN,
    rimLengthCountSpacing: rimLengthA,
    rimLengthChainOpen: chain.open,
    rimLengthChainClosed: chain.closed,
    lambdaCountSpacing: lambdaA,
    lambdaChainOpen: lambdaBOpen,
    lambdaChainClosed: lambdaBClosed,
  }
}
