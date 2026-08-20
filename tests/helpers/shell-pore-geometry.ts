// Shell/pore geometry measurement machinery for tests/shell-pore-check.test.ts — moved verbatim out
// of that file by responsibility (file-size rule, root CLAUDE.md). Reuses this project's own
// already-tested functions throughout (findAmphiphiles, clusterComponents, unwrapAggregate, shapeOf,
// densityProfileZ, bilayerPeaks, occupancy/enclosedVolume/dimsFor, analyzeAggregates) and adds no
// new physics, threshold, potential, recogniser or bond rule.

import { loadParams, wcaCutoff } from '../../engine/src/params'
import type { Monomer } from '../../soup/src/rules'
import { findAmphiphiles, type Amphiphile } from '../../soup/src/amphiphile'
import { memberIndicesOf, positionsFor, analyzeAggregates } from '../../soup/src/aggregates'
import { loadStageThresholds } from '../../soup/src/stages'
import { clusterComponents, shapeOf, unwrapAggregate } from '../../engine/src/aggregate'
import { densityProfileZ, bilayerPeaks } from '../../engine/src/metrics'
import { occupancy, enclosedVolume, dimsFor } from '../../engine/src/closure'
import { fibonacciSphere, angleDeg, mi1 } from './geometry-primitives'

export function norm(a: [number, number, number]): [number, number, number] {
  const m = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) || 1
  return [a[0] / m, a[1] / m, a[2] / m]
}
export function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

// --- largest aggregate, same recognition pipeline as tests/rim-lambda-insitu.test.ts -------------
export interface Largest {
  aggOriginalIdx: number[]
  aggAmphiphiles: Amphiphile[]
  allAmphiphiles: Amphiphile[]
  memberIdx: Set<number>
  cutoff: number
  aggregateCount: number
  edges: [number, number][] // covalent bond pairs (ORIGINAL indices) with BOTH ends in this aggregate
}
export function findLargest(positions: Float32Array, bonds: Uint32Array, box: [number, number, number], monomers: Monomer[]): Largest {
  const params = loadParams()
  const amphiphiles = findAmphiphiles(positions, bonds, monomers)
  const memberIdx = memberIndicesOf(amphiphiles)
  const memberPositions = positionsFor(positions, memberIdx)
  const idxArr = Array.from(memberIdx)
  const memberRadii = monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(params.sigma * Math.max(...memberRadii)) + params.attraction.wc
  const components = clusterComponents(memberPositions, box, cutoff, true) // z periodic: bulk soup
  const largest = components[0]
  const aggOriginalIdx = largest.map((k) => idxArr[k])
  const aggSet = new Set(aggOriginalIdx)
  const aggAmphiphiles = amphiphiles.filter((a) => aggSet.has(a.headIndex))
  const edges: [number, number][] = []
  for (let k = 0; k < bonds.length; k += 2) {
    const i = bonds[k], j = bonds[k + 1]
    if (aggSet.has(i) && aggSet.has(j)) edges.push([i, j])
  }
  return { aggOriginalIdx, aggAmphiphiles, allAmphiphiles: amphiphiles, memberIdx, cutoff, aggregateCount: components.length, edges }
}

// --- a CORRECT unwrap for an aggregate whose own radius approaches box/2 --------------------------
// unwrapAggregate() (engine/src/aggregate.ts) places every member via minimum-image relative to a
// SINGLE reference particle -- exact whenever the aggregate's own diameter stays under half the box
// length, but not guaranteed once it does not (see checkUnwrapSanity's own doc comment below, and the
// measured bond-length blowup this file's own report documents: ~3% of bonded pairs land tens of
// sigma apart in that frame, for EVERY one of the ten checkpoints, at this box=58/meanHeadRadius~29
// scale). properUnwrap() below fixes this the standard way for a polymer/aggregate whose extent can
// exceed box/2: walk the SAME proximity graph that defined the aggregate in the first place
// (clusterComponents' own connectivity rule -- particles within `cutoff` of each other, the identical
// cutoff soup/src/stages.ts's own detectStage uses) via BFS, placing each newly-visited particle via
// minimum-image relative to the NEIGHBOUR that discovered it, never relative to one fixed, possibly-
// distant reference. Since every graph edge is by construction a REAL pair within `cutoff` (2.7225
// sigma here, far under box/2=29), a single minimum-image step relative to that edge's own two
// endpoints is always correct, and chaining such steps along a connected path reconstructs a globally
// coherent frame regardless of the aggregate's own overall diameter. This does not touch
// engine/src/aggregate.ts -- it is new analysis code, kept in this file only, and does not change
// what unwrapAggregate() itself does or how any other caller of it behaves.
export function properUnwrap(aggOriginalIdx: number[], positions: Float32Array, box: [number, number, number], cutoff: number, rootLocal = 0): { unwrapped: Float32Array; unreached: number } {
  const n = aggOriginalIdx.length
  const cellSize = cutoff
  const nx = Math.max(1, Math.floor(box[0] / cellSize))
  const ny = Math.max(1, Math.floor(box[1] / cellSize))
  const nz = Math.max(1, Math.floor(box[2] / cellSize))
  const wx = box[0] / nx, wy = box[1] / ny, wz = box[2] / nz
  const cellOf = (k: number): [number, number, number] => {
    const o = aggOriginalIdx[k]
    const x = ((positions[o * 4] % box[0]) + box[0]) % box[0]
    const y = ((positions[o * 4 + 1] % box[1]) + box[1]) % box[1]
    const z = ((positions[o * 4 + 2] % box[2]) + box[2]) % box[2]
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
  const adjacency: number[][] = Array.from({ length: n }, () => [])
  for (let k = 0; k < n; k++) {
    const [cx, cy, cz] = cells[k]
    const o = aggOriginalIdx[k]
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
            const o2 = aggOriginalIdx[kk]
            const ddx = mi1(positions[o * 4] - positions[o2 * 4], box[0])
            const ddy = mi1(positions[o * 4 + 1] - positions[o2 * 4 + 1], box[1])
            const ddz = mi1(positions[o * 4 + 2] - positions[o2 * 4 + 2], box[2])
            if (ddx * ddx + ddy * ddy + ddz * ddz <= cutoff2) {
              adjacency[k].push(kk)
              adjacency[kk].push(k)
            }
          }
        }
      }
    }
  }
  const out = new Float32Array(n * 4)
  const visited = new Uint8Array(n)
  out.set(positions.subarray(aggOriginalIdx[rootLocal] * 4, aggOriginalIdx[rootLocal] * 4 + 4), rootLocal * 4)
  visited[rootLocal] = 1
  const queue = [rootLocal]
  let head = 0
  while (head < queue.length) {
    const k = queue[head++]
    const ok = aggOriginalIdx[k]
    for (const kk of adjacency[k]) {
      if (visited[kk]) continue
      const okk = aggOriginalIdx[kk]
      const ddx = mi1(positions[okk * 4] - positions[ok * 4], box[0])
      const ddy = mi1(positions[okk * 4 + 1] - positions[ok * 4 + 1], box[1])
      const ddz = mi1(positions[okk * 4 + 2] - positions[ok * 4 + 2], box[2])
      out[kk * 4] = out[k * 4] + ddx
      out[kk * 4 + 1] = out[k * 4 + 1] + ddy
      out[kk * 4 + 2] = out[k * 4 + 2] + ddz
      out[kk * 4 + 3] = positions[okk * 4 + 3]
      visited[kk] = 1
      queue.push(kk)
    }
  }
  let unreached = 0
  for (let k = 0; k < n; k++) if (!visited[k]) unreached++
  return { unwrapped: out, unreached }
}

// --- sanity check on unwrapAggregate() itself: a genuine covalent (FENE) bond is ALWAYS ~1 sigma in
// real, physical space, by construction of data/params.json's fene.rInf=1.5 cap -- so in ANY
// correctly unwrapped frame, every bonded pair's distance must stay small and tightly clustered. This
// project's own unwrapAggregate() (engine/src/aggregate.ts) unwraps every member relative to a SINGLE
// reference particle via per-axis minimum-image -- correct whenever the aggregate's own diameter stays
// under half the box length, but not guaranteed once an aggregate's own radius approaches box/2 (a
// particle on the far side of a large, curved object can sit closer to the box's OTHER periodic image
// of the reference than to the reference's own true image, aliasing it to the wrong side). This
// checkpoint's own box is 58 sigma/side (box/2=29 sigma) and the aggregate's own measured mean head
// radius is ~29 sigma -- i.e. the shell's own radius is comparable to box/2, exactly the regime where
// this failure mode is live, not a remote edge case. A bonded-pair distance blowing up to tens of
// sigma in the "unwrapped" frame is the direct, unambiguous signature of it happening.
export function checkUnwrapSanity(agg: Largest, unwrapped: Float32Array): { maxBondLength: number; meanBondLength: number; brokenCount: number; brokenThreshold: number } {
  const posOf = new Map<number, number>()
  agg.aggOriginalIdx.forEach((orig, k) => posOf.set(orig, k))
  const brokenThreshold = 5 // sigma -- FENE rInf=1.5 caps any real bond well under this; anything past it is not a stretched bond, it is a wrap artefact
  let maxBondLength = 0
  let sum = 0
  let brokenCount = 0
  for (const [i, j] of agg.edges) {
    const ki = posOf.get(i)!
    const kj = posOf.get(j)!
    const dx = unwrapped[ki * 4] - unwrapped[kj * 4]
    const dy = unwrapped[ki * 4 + 1] - unwrapped[kj * 4 + 1]
    const dz = unwrapped[ki * 4 + 2] - unwrapped[kj * 4 + 2]
    const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
    sum += d
    if (d > maxBondLength) maxBondLength = d
    if (d > brokenThreshold) brokenCount++
  }
  return { maxBondLength, meanBondLength: sum / agg.edges.length, brokenCount, brokenThreshold }
}

// --- unwrap + shape + hole search, one aggregate --------------------------------------------------
export interface ShellGeometry {
  centre: [number, number, number]
  radiusOfGyration: number
  flatnessRatio: number
  holeCentre: [number, number, number]
  holeRadiusDeg: number
  solidAngleFraction: number
  unwrapped: Float32Array // aligned with aggOriginalIdx order -- properUnwrap()'s output, not unwrapAggregate()'s
  unreached: number
  meanHeadRadius: number
}
export function shellGeometry(agg: Largest, positions: Float32Array, box: [number, number, number]): ShellGeometry {
  const { aggOriginalIdx } = agg
  const { unwrapped, unreached } = properUnwrap(aggOriginalIdx, positions, box, agg.cutoff)
  const shape = shapeOf(unwrapped)
  const [l0, , l2] = shape.principalMoments
  const flatnessRatio = l2 > 1e-12 ? l0 / l2 : 1

  const headDirs: { dir: [number, number, number]; radius: number }[] = []
  for (const a of agg.aggAmphiphiles) {
    const k = aggOriginalIdx.indexOf(a.headIndex)
    const dx = unwrapped[k * 4] - shape.centre[0]
    const dy = unwrapped[k * 4 + 1] - shape.centre[1]
    const dz = unwrapped[k * 4 + 2] - shape.centre[2]
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1
    headDirs.push({ dir: [dx / r, dy / r, dz / r], radius: r })
  }
  const meanHeadRadius = headDirs.reduce((s, h) => s + h.radius, 0) / headDirs.length

  const candidates = fibonacciSphere(800)
  let holeCentre: [number, number, number] = candidates[0]
  let holeRadiusDeg = -1
  for (const c of candidates) {
    let minAngle = Infinity
    for (const h of headDirs) {
      const ang = angleDeg(c, h.dir)
      if (ang < minAngle) minAngle = ang
    }
    if (minAngle > holeRadiusDeg) {
      holeRadiusDeg = minAngle
      holeCentre = c
    }
  }
  const solidAngleFraction = (1 - Math.cos((holeRadiusDeg * Math.PI) / 180)) / 2

  return { centre: shape.centre, radiusOfGyration: shape.radiusOfGyration, flatnessRatio, holeCentre, holeRadiusDeg, solidAngleFraction, unwrapped, unreached, meanHeadRadius }
}

// --- side-by-side comparison: unwrapAggregate() (broken at this scale) vs properUnwrap() (fixed) --
export function compareUnwrapMethods(agg: Largest, positions: Float32Array, box: [number, number, number]) {
  const raw = new Float32Array(agg.aggOriginalIdx.length * 4)
  for (let k = 0; k < agg.aggOriginalIdx.length; k++) raw.set(positions.subarray(agg.aggOriginalIdx[k] * 4, agg.aggOriginalIdx[k] * 4 + 4), k * 4)
  const broken = unwrapAggregate(raw, box, [true, true, true])
  const brokenSanity = checkUnwrapSanity(agg, broken)
  const brokenShape = shapeOf(broken)
  return { brokenSanity, brokenShape }
}

// --- Task 1: radial head/tail density profile of the largest aggregate, using this project's own
// densityProfileZ/bilayerPeaks (engine/src/metrics.ts), packed the same way soup/src/aggregates.ts's
// own (unexported) profileAlong() already does for its own radial head-shell diagnostic -- the
// coordinate goes in the position's z-slot, the box is a throwaway [1,1,span]. --------------------
export function radialProfile(agg: Largest, geom: ShellGeometry, positions: Float32Array, monomers: Monomer[], bins: number) {
  const { aggOriginalIdx } = agg
  const n = aggOriginalIdx.length
  const synthetic = new Float32Array(n * 4)
  let maxR = 0
  for (let k = 0; k < n; k++) {
    const dx = geom.unwrapped[k * 4] - geom.centre[0]
    const dy = geom.unwrapped[k * 4 + 1] - geom.centre[1]
    const dz = geom.unwrapped[k * 4 + 2] - geom.centre[2]
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (r > maxR) maxR = r
    const kindIdx = Math.round(positions[aggOriginalIdx[k] * 4 + 3])
    const isTail = monomers[kindIdx]?.polar ? 0 : 1 // 0=head, nonzero=tail -- densityProfileZ's own convention
    synthetic[k * 4 + 2] = r
    synthetic[k * 4 + 3] = isTail
  }
  const span = Math.max(1e-6, maxR * 1.0001)
  const box: [number, number, number] = [1, 1, span]
  const profile = densityProfileZ(synthetic, box, bins)
  let peaks: { lower: number; upper: number } | null = null
  try {
    peaks = bilayerPeaks(profile)
  } catch {
    peaks = null
  }
  return { profile, peaks, maxR }
}

// --- Task 3: honest measured cavity volume via this project's own analyzeAggregates() pipeline ---
export function honestCavityVolume(positions: Float32Array, box: [number, number, number], monomers: Monomer[], agg: Largest): number {
  const thresholds = loadStageThresholds()
  const analysis = analyzeAggregates(positions, box, monomers, agg.allAmphiphiles, agg.memberIdx, agg.cutoff, thresholds)
  // aggregates[] is sorted largest-particle-count-first (clusterComponents' own order) -- the same
  // order agg.aggOriginalIdx was drawn from, so aggregates[0] IS the same aggregate this file's own
  // geometry/hole-search analysed.
  return analysis.aggregates.length > 0 ? analysis.aggregates[0].cavityVolume : 0
}

// --- Task 3: "plugged" counterfactual, two independent ways --------------------------------------
// (a) analytic: volume of a sphere at the INNER head-peak radius from the radial profile -- "if this
//     shell's own inner leaflet were a perfect closed sphere of the radius it already has".
// (b) literal flood-with-plug: seal the hole with a synthetic cap of beads at the mean head radius,
//     then run the SAME local-bounding-box flood soup/src/aggregates.ts's (unexported)
//     localCavityVolume() uses, built here from the REAL exported occupancy()/enclosedVolume()/
//     dimsFor() primitives (engine/src/closure.ts) -- only the padding/shift wrapper is duplicated
//     (it is not exported), the flood algorithm itself is not reimplemented.
export function localCavityVolumeWrapper(positions: Float32Array, cell: number, radius: number): number {
  const n = positions.length / 4
  if (n === 0) return 0
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity
  let minZ = Infinity, maxZ = -Infinity
  for (let i = 0; i < n; i++) {
    const x = positions[i * 4], y = positions[i * 4 + 1], z = positions[i * 4 + 2]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
    if (z < minZ) minZ = z
    if (z > maxZ) maxZ = z
  }
  const pad = radius * 4
  const shiftX = pad - minX
  const shiftY = pad - minY
  const shiftZ = pad - minZ
  const localBox: [number, number, number] = [maxX - minX + 2 * pad, maxY - minY + 2 * pad, maxZ - minZ + 2 * pad]
  const shifted = new Float32Array(positions.length)
  for (let i = 0; i < n; i++) {
    shifted[i * 4] = positions[i * 4] + shiftX
    shifted[i * 4 + 1] = positions[i * 4 + 1] + shiftY
    shifted[i * 4 + 2] = positions[i * 4 + 2] + shiftZ
    shifted[i * 4 + 3] = positions[i * 4 + 3]
  }
  const occ = occupancy(shifted, localBox, cell, radius)
  return enclosedVolume(occ, dimsFor(localBox, cell), cell)
}

export function pluggedFlood(geom: ShellGeometry, cell: number, radius: number): number {
  const base = geom.unwrapped

  // Orthonormal basis (u,v) perpendicular to holeCentre for a polar-cap parametrisation.
  const holeC = geom.holeCentre
  let ref: [number, number, number] = [1, 0, 0]
  if (Math.abs(holeC[0]) > 0.9) ref = [0, 1, 0]
  const u = norm(cross(holeC, ref))
  const v = norm(cross(holeC, u))

  // Seal the ENTIRE measured hole (margin +2deg over the measured angular radius, so the plug fully
  // covers what the search found, not just up to its own boundary) with a fine grid of synthetic
  // beads at the shell's own mean head radius -- ring spacing ~= 2*radius (bead diameter equivalent
  // in this closure detector's own convention) so occupancy() seals it the same way a real, tightly
  // packed head layer would.
  const capRad = ((geom.holeRadiusDeg + 2) * Math.PI) / 180
  const ringStep = (0.5 * radius) / geom.meanHeadRadius // angular step giving ~0.5*radius arc spacing --
  // deliberately much tighter than the real membrane's own bead spacing so a sparse synthetic plug
  // cannot leave sub-cell gaps the flood leaks through (occupancy()'s own cell=0.5/radius=0.6 grid is
  // coarse enough that a plug spaced at bead-diameter distance measurably under-seals -- see the
  // report's own note on this).
  const plug: number[] = []
  for (let theta = 0; theta <= capRad; theta += ringStep) {
    const ringRadius = Math.sin(theta)
    const circumference = 2 * Math.PI * ringRadius
    const nPhi = Math.max(1, Math.round(circumference / ringStep))
    for (let j = 0; j < nPhi; j++) {
      const phi = (2 * Math.PI * j) / nPhi
      const dir: [number, number, number] = [
        Math.cos(theta) * holeC[0] + Math.sin(theta) * (Math.cos(phi) * u[0] + Math.sin(phi) * v[0]),
        Math.cos(theta) * holeC[1] + Math.sin(theta) * (Math.cos(phi) * u[1] + Math.sin(phi) * v[1]),
        Math.cos(theta) * holeC[2] + Math.sin(theta) * (Math.cos(phi) * u[2] + Math.sin(phi) * v[2]),
      ]
      plug.push(geom.centre[0] + dir[0] * geom.meanHeadRadius, geom.centre[1] + dir[1] * geom.meanHeadRadius, geom.centre[2] + dir[2] * geom.meanHeadRadius, 0)
    }
  }

  console.log(`SHELL-PORE-PLUG plugBeadCount=${plug.length / 4} capRadDeg=${((capRad * 180) / Math.PI).toFixed(2)} ringStepDeg=${((ringStep * 180) / Math.PI).toFixed(3)}`)
  const combined = new Float32Array(base.length + plug.length)
  combined.set(base, 0)
  combined.set(plug, base.length)
  return localCavityVolumeWrapper(combined, cell, radius)
}

// --- Task 4: local head:tail ratio and local area-per-lipid, rim vs an equal-solid-angle antipodal
// patch, plus rim vs "everything else". Local area-per-lipid generalises engine/src/metrics.ts's own
// areaPerLipid(box,lipids) = boxArea/(lipids/2) to a spherical-cap-annulus area over its own local
// head count -- same formula, a local patch instead of the whole periodic box. --------------------
export interface Bin { heads: number; tails: number; radiusSum: number; radiusCount: number }
export function localRimVsBulk(agg: Largest, geom: ShellGeometry, positions: Float32Array, monomers: Monomer[], marginDeg: number) {
  const { aggOriginalIdx } = agg
  const antipode: [number, number, number] = [-geom.holeCentre[0], -geom.holeCentre[1], -geom.holeCentre[2]]
  const lo = geom.holeRadiusDeg
  const hi = geom.holeRadiusDeg + marginDeg

  const rim: Bin = { heads: 0, tails: 0, radiusSum: 0, radiusCount: 0 }
  const antipodal: Bin = { heads: 0, tails: 0, radiusSum: 0, radiusCount: 0 }
  const rest: Bin = { heads: 0, tails: 0, radiusSum: 0, radiusCount: 0 }

  for (let k = 0; k < aggOriginalIdx.length; k++) {
    const dx = geom.unwrapped[k * 4] - geom.centre[0]
    const dy = geom.unwrapped[k * 4 + 1] - geom.centre[1]
    const dz = geom.unwrapped[k * 4 + 2] - geom.centre[2]
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1
    const dir: [number, number, number] = [dx / r, dy / r, dz / r]
    const kindIdx = Math.round(positions[aggOriginalIdx[k] * 4 + 3])
    const isHead = !!monomers[kindIdx]?.polar
    const angFromHole = angleDeg(dir, geom.holeCentre)
    const angFromAnti = angleDeg(dir, antipode)
    const inRim = angFromHole >= lo && angFromHole <= hi
    const inAntipodal = angFromAnti >= lo && angFromAnti <= hi
    const target = inRim ? rim : rest
    if (isHead) target.heads++
    else target.tails++
    target.radiusSum += r
    target.radiusCount++
    if (inAntipodal) {
      if (isHead) antipodal.heads++
      else antipodal.tails++
      antipodal.radiusSum += r
      antipodal.radiusCount++
    }
  }

  const solidAngleAnnulus = 2 * Math.PI * (Math.cos((lo * Math.PI) / 180) - Math.cos((hi * Math.PI) / 180))
  const areaPerLipidOf = (bin: Bin) => {
    if (bin.heads === 0) return Infinity
    const meanR = bin.radiusSum / bin.radiusCount
    const area = solidAngleAnnulus * meanR * meanR
    return area / (bin.heads / 2)
  }
  const headTailRatioOf = (bin: Bin) => (bin.tails > 0 ? bin.heads / bin.tails : Infinity)
  const pack = (bin: Bin) => ({ ...bin, headTailRatio: headTailRatioOf(bin), areaPerLipid: areaPerLipidOf(bin), meanRadius: bin.radiusSum / bin.radiusCount })

  return { marginDeg, lo, hi, rim: pack(rim), antipodal: pack(antipodal), rest: pack(rest) }
}
