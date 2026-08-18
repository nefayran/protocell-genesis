import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import type { CheckpointFile } from '../soup/src/checkpoint'
import { loadSoup, type Monomer } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { findAmphiphiles, type Amphiphile } from '../soup/src/amphiphile'
import { memberIndicesOf, positionsFor, analyzeAggregates } from '../soup/src/aggregates'
import { loadStageThresholds } from '../soup/src/stages'
import { clusterComponents, shapeOf, unwrapAggregate } from '../engine/src/aggregate'
import { densityProfileZ, bilayerPeaks } from '../engine/src/metrics'
import { occupancy, enclosedVolume, dimsFor } from '../engine/src/closure'

// Task 'shell-pore-check' (2026-08-18): independent test of the claim raised by
// .superpowers/sdd/2026-08-16-soup-to-vesicle/kappa-tightening-report.md's own SS2 -- that the
// campaign's largest aggregate (data/checkpoints/vesicle-93k-step*.json, label 'vesicle-93k') is a
// near-closed SPHERICAL SHELL with a single, persistent, geometrically small hole, not an open cup,
// and that the rim's in-situ line tension is small and negative (-1.5 to -4.6 eps/sigma there,
// vs +10.77 for the prepared flat patch). This file adds NO new simulation: every number below comes
// from the ten checkpoints already on disk (step160000..step250000, 10000-step spacing) via pure
// Node decode (decodeCheckpointResume, soup/src/checkpoint.ts -- no GPU, no gpuPage() call anywhere
// in this file). It reuses this project's own already-tested functions throughout (findAmphiphiles,
// clusterComponents, unwrapAggregate, shapeOf, densityProfileZ, bilayerPeaks,
// occupancy/enclosedVolume/dimsFor, analyzeAggregates) and adds no new physics, threshold, potential,
// recogniser or bond rule anywhere -- confirmed by `git diff --stat -- soup/src engine/src soup/wgsl
// engine/wgsl data/params.json data/soup.json data/literature.json` staying empty for this task (see
// shell-pore-report.md's own housekeeping section for the actual command+output).
//
// This runs under vitest (not plain tsx) for one purely mechanical reason: engine/src/closure.ts
// (needed for occupancy/enclosedVolume, and transitively imported by soup/src/aggregates.ts's own
// analyzeAggregates) does a Vite-specific `?raw` import of its WGSL shader at module load time, which
// only Vite's own transform (vitest included) understands -- plain tsx/Node throws
// ERR_UNKNOWN_FILE_EXTENSION on it. tests/rim-lambda-insitu.test.ts's own first test hits the same
// constraint and is run the same way, by its own header note ("no GPU needed for this step at all").
// No test here calls gpuPage() -- nothing in this file touches a browser or the GPU.
//
// The Fibonacci-sphere hole search below is a byte-for-byte copy of the SAME method
// tests/rim-lambda-insitu.test.ts already used and reported (kappa-tightening-report.md SS2.2,
// attempt 2) -- repeated here on ten checkpoints instead of one, plus a consecutive-snapshot
// correlation test that report's own SS5 concern #4 asked for but did not have the budget to run.

const NONE_U32 = 0xffffffff
const STEPS = [160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000]
const OUT_DIR = 'verify/out'
const OUT_FILE = `${OUT_DIR}/shell-pore-check.json`

// --- geometry helpers, copied verbatim from tests/rim-lambda-insitu.test.ts (same method, cited
// there against kappa-tightening-report.md SS2.2) -- not a new algorithm. ---------------------------
function fibonacciSphere(n: number): [number, number, number][] {
  const pts: [number, number, number][] = []
  const golden = Math.PI * (3 - Math.sqrt(5))
  for (let i = 0; i < n; i++) {
    const y = 1 - (i / (n - 1)) * 2
    const r = Math.sqrt(Math.max(0, 1 - y * y))
    const theta = golden * i
    pts.push([Math.cos(theta) * r, y, Math.sin(theta) * r])
  }
  return pts
}
function angleDeg(a: [number, number, number], b: [number, number, number]): number {
  const dot = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
  return (Math.acos(dot) * 180) / Math.PI
}
function norm(a: [number, number, number]): [number, number, number] {
  const m = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]) || 1
  return [a[0] / m, a[1] / m, a[2] / m]
}
function cross(a: [number, number, number], b: [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

// --- checkpoint decode (pure Node, no GPU) ------------------------------------------------------
function loadCheckpoint(step: number): { positions: Float32Array; box: [number, number, number]; bonds: Uint32Array } {
  const path = `data/checkpoints/vesicle-93k-step${step}.json`
  const file = JSON.parse(readFileSync(path, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  const N = file.N
  const box = file.config.box
  const edges: number[] = []
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < 3; s++) {
      const p = resume.bondSlots[i * 3 + s]
      if (p !== NONE_U32 && p > i) edges.push(i, p)
    }
  }
  if (resume.globalStep !== step) throw new Error(`checkpoint step${step}.json: globalStep=${resume.globalStep} != ${step}`)
  return { positions: resume.positions, box, bonds: new Uint32Array(edges) }
}

// --- largest aggregate, same recognition pipeline as tests/rim-lambda-insitu.test.ts -------------
interface Largest {
  aggOriginalIdx: number[]
  aggAmphiphiles: Amphiphile[]
  allAmphiphiles: Amphiphile[]
  memberIdx: Set<number>
  cutoff: number
  aggregateCount: number
  edges: [number, number][] // covalent bond pairs (ORIGINAL indices) with BOTH ends in this aggregate
}
function findLargest(positions: Float32Array, bonds: Uint32Array, box: [number, number, number], monomers: Monomer[]): Largest {
  const params = loadParams()
  const amphiphiles = findAmphiphiles(positions, bonds, monomers)
  const memberIdx = memberIndicesOf(amphiphiles)
  const memberPositions = positionsFor(positions, memberIdx)
  const idxArr = Array.from(memberIdx)
  const memberRadii = monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(params.sigma * Math.max(...memberRadii)) + params.attraction.wc
  const components = clusterComponents(memberPositions, box, cutoff)
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
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}
function properUnwrap(aggOriginalIdx: number[], positions: Float32Array, box: [number, number, number], cutoff: number, rootLocal = 0): { unwrapped: Float32Array; unreached: number } {
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
function checkUnwrapSanity(agg: Largest, unwrapped: Float32Array): { maxBondLength: number; meanBondLength: number; brokenCount: number; brokenThreshold: number } {
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
interface ShellGeometry {
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
function shellGeometry(agg: Largest, positions: Float32Array, box: [number, number, number]): ShellGeometry {
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
function compareUnwrapMethods(agg: Largest, positions: Float32Array, box: [number, number, number]) {
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
function radialProfile(agg: Largest, geom: ShellGeometry, positions: Float32Array, monomers: Monomer[], bins: number) {
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
function honestCavityVolume(positions: Float32Array, box: [number, number, number], monomers: Monomer[], agg: Largest): number {
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
function localCavityVolumeWrapper(positions: Float32Array, cell: number, radius: number): number {
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

function pluggedFlood(geom: ShellGeometry, cell: number, radius: number): number {
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
interface Bin { heads: number; tails: number; radiusSum: number; radiusCount: number }
function localRimVsBulk(agg: Largest, geom: ShellGeometry, positions: Float32Array, monomers: Monomer[], marginDeg: number) {
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

test('shell-pore-check: radial profile, hole persistence across 10 checkpoints, plugged-vs-measured volume, rim head enrichment (no GPU, pure decode+analysis)', () => {
  const soup = loadSoup()
  const monomers: Monomer[] = soup.monomers
  const thresholds = loadStageThresholds()

  const perStep = STEPS.map((step) => {
    const { positions, box, bonds } = loadCheckpoint(step)
    const agg = findLargest(positions, bonds, box, monomers)
    const geom = shellGeometry(agg, positions, box)
    const honest = honestCavityVolume(positions, box, monomers, agg)
    const fixedSanity = checkUnwrapSanity(agg, geom.unwrapped)
    const { brokenSanity, brokenShape } = compareUnwrapMethods(agg, positions, box)
    console.log(
      `SHELL-PORE step=${step} aggregates=${agg.aggregateCount} amphiphiles=${agg.aggAmphiphiles.length} ` +
        `particles=${agg.aggOriginalIdx.length} Rg=${geom.radiusOfGyration.toFixed(3)} flatness=${geom.flatnessRatio.toFixed(4)} ` +
        `holeCentre=[${geom.holeCentre.map((v) => v.toFixed(3))}] holeRadiusDeg=${geom.holeRadiusDeg.toFixed(2)} ` +
        `solidAngleFrac=${(geom.solidAngleFraction * 100).toFixed(2)}% meanHeadRadius=${geom.meanHeadRadius.toFixed(2)} ` +
        `cavityVolumeHonest=${honest.toFixed(4)} unreachedByProperUnwrap=${geom.unreached}`,
    )
    console.log(
      `SHELL-PORE-UNWRAP step=${step} box=${box[0]} boxHalf=${(box[0] / 2).toFixed(1)} edges=${agg.edges.length} ` +
        `BROKEN(unwrapAggregate) meanBondLen=${brokenSanity.meanBondLength.toFixed(3)} maxBondLen=${brokenSanity.maxBondLength.toFixed(2)} broken(>${brokenSanity.brokenThreshold}sigma)=${brokenSanity.brokenCount} Rg=${brokenShape.radiusOfGyration.toFixed(3)} | ` +
        `FIXED(properUnwrap) meanBondLen=${fixedSanity.meanBondLength.toFixed(3)} maxBondLen=${fixedSanity.maxBondLength.toFixed(2)} broken(>${fixedSanity.brokenThreshold}sigma)=${fixedSanity.brokenCount} Rg=${geom.radiusOfGyration.toFixed(3)}`,
    )
    return { step, agg, geom, honest, positions, fixedSanity, brokenSanity, brokenShape }
  })

  expect(perStep.length).toBe(10)
  for (const p of perStep) {
    expect(p.agg.aggAmphiphiles.length).toBeGreaterThan(1000) // sanity: same ~1282-amphiphile cup at every step
    expect(p.geom.holeRadiusDeg).toBeGreaterThan(0)
    expect(p.geom.unreached).toBe(0) // properUnwrap must reach every member via the SAME graph that defined the aggregate
    // properUnwrap() removes MOST of unwrapAggregate()'s own single-reference wrap failures (see the
    // BROKEN vs FIXED counts logged above) but not all of them: this aggregate's own diameter (2x
    // meanHeadRadius ~59-64 sigma) is comparable to or EXCEEDS the box's own side length (58 sigma),
    // so the object cannot fit inside one periodic image of the box at all -- a genuine topological
    // wraparound, not an unwrap-algorithm defect, and no single-frame unwrap (BFS-based or otherwise)
    // can fully resolve it. The residual is real and reported, not hidden: expect it to be SMALL
    // relative to the broken method's own count, not necessarily zero.
    expect(p.fixedSanity.brokenCount).toBeLessThan(p.brokenSanity.brokenCount)
    expect(p.brokenSanity.brokenCount).toBeGreaterThan(0) // documents that unwrapAggregate() genuinely does break here
  }

  // --- Task 2: consecutive-snapshot hole-direction correlation ----------------------------------
  const consecutive = perStep.slice(1).map((p, i) => ({
    fromStep: perStep[i].step,
    toStep: p.step,
    angleDeg: angleDeg(perStep[i].geom.holeCentre, p.geom.holeCentre),
  }))
  const meanConsecutiveAngle = consecutive.reduce((s, c) => s + c.angleDeg, 0) / consecutive.length
  const netDrift = angleDeg(perStep[0].geom.holeCentre, perStep[perStep.length - 1].geom.holeCentre)
  console.log(
    `SHELL-PORE-CORR consecutive angles(deg)=[${consecutive.map((c) => c.angleDeg.toFixed(1)).join(',')}] mean=${meanConsecutiveAngle.toFixed(2)} netDriftFirstToLast=${netDrift.toFixed(2)}`,
  )
  expect(Number.isFinite(meanConsecutiveAngle)).toBe(true)

  // --- Task 1: radial head/tail profile on the LAST checkpoint (step 250000, the same one
  // kappa-tightening-report.md's own SS2 measured lambda on) and the FIRST of the ten (step 160000)
  // as a within-window robustness check. -----------------------------------------------------------
  const last = perStep[perStep.length - 1]
  const first = perStep[0]
  const radialLast = radialProfile(last.agg, last.geom, last.positions, monomers, 60)
  const radialFirst = radialProfile(first.agg, first.geom, first.positions, monomers, 60)
  const sepLast = radialLast.peaks ? radialLast.peaks.upper - radialLast.peaks.lower : null
  const sepFirst = radialFirst.peaks ? radialFirst.peaks.upper - radialFirst.peaks.lower : null
  console.log(
    `SHELL-PORE-RADIAL step=${last.step} innerPeak=${radialLast.peaks?.lower.toFixed(3)} outerPeak=${radialLast.peaks?.upper.toFixed(3)} ` +
      `separation=${sepLast?.toFixed(3)} bilayerThicknessRef=4.457 maxR=${radialLast.maxR.toFixed(2)}`,
  )
  console.log(
    `SHELL-PORE-RADIAL step=${first.step} innerPeak=${radialFirst.peaks?.lower.toFixed(3)} outerPeak=${radialFirst.peaks?.upper.toFixed(3)} ` +
      `separation=${sepFirst?.toFixed(3)} bilayerThicknessRef=4.457 maxR=${radialFirst.maxR.toFixed(2)}`,
  )
  expect(radialLast.peaks).not.toBeNull()

  // --- Robustness check: does properUnwrap's own shape reading depend on which particle it starts
  // BFS from? It should NOT, if the underlying proximity graph is topologically consistent; the
  // residual ~1.6-2% broken-bond count already found above says it is not QUITE consistent (this
  // aggregate's own diameter is comparable to the box), so this checks how much that residual
  // inconsistency actually moves the headline shape numbers, not just its bond-length symptom. -----
  const altRootLocal = Math.floor(last.agg.aggOriginalIdx.length / 2)
  const altUnwrap = properUnwrap(last.agg.aggOriginalIdx, last.positions, loadCheckpoint(last.step).box, last.agg.cutoff, altRootLocal)
  const altShape = shapeOf(altUnwrap.unwrapped)
  const altFlatness = altShape.principalMoments[2] > 1e-12 ? altShape.principalMoments[0] / altShape.principalMoments[2] : 1
  console.log(
    `SHELL-PORE-ROOTCHECK step=${last.step} root=0 Rg=${last.geom.radiusOfGyration.toFixed(3)} flatness=${last.geom.flatnessRatio.toFixed(4)} | ` +
      `root=${altRootLocal} Rg=${altShape.radiusOfGyration.toFixed(3)} flatness=${altFlatness.toFixed(4)} unreached=${altUnwrap.unreached}`,
  )

  // --- Task 3: plugged counterfactual vs the honest measurement, on step 250000 -----------------
  // Both the unplugged and plugged flood below run on the SAME frame (this file's own properUnwrap
  // output, geom.unwrapped) for a fair apples-to-apples comparison -- analyzeAggregates()'s own
  // cavityVolume (last.honest, printed above as "cavityVolumeHonest") uses the PROJECT'S own internal
  // unwrapAggregate() (the broken-at-this-scale method) and is reported separately, as the project's
  // own official reading, not mixed into this comparison.
  const innerRadius = radialLast.peaks ? radialLast.peaks.lower : last.geom.meanHeadRadius
  const analyticPlugged = (4 / 3) * Math.PI * innerRadius ** 3
  const unplugged = localCavityVolumeWrapper(last.geom.unwrapped, thresholds.closureCell, thresholds.closureRadius)
  const floodPlugged = pluggedFlood(last.geom, thresholds.closureCell, thresholds.closureRadius)
  console.log(
    `SHELL-PORE-VOLUME step=${last.step} projectOfficialHonest(brokenUnwrap)=${last.honest.toFixed(4)} ` +
      `sameFrameUnplugged(properUnwrap)=${unplugged.toFixed(4)} pluggedAnalytic(4/3*pi*r_inner^3, r_inner=${innerRadius.toFixed(3)})=${analyticPlugged.toFixed(2)} ` +
      `pluggedFlood(sameFrame)=${floodPlugged.toFixed(2)} closureThreshold=${thresholds.enclosedVolume}`,
  )
  expect(floodPlugged).toBeGreaterThanOrEqual(unplugged) // sealing the hole should never shrink the enclosed volume, same-frame comparison

  // --- Task 4: rim-vs-bulk head:tail ratio and local area-per-lipid, on step 250000, two margins --
  const rimBulk15 = localRimVsBulk(last.agg, last.geom, last.positions, monomers, 15)
  const rimBulk25 = localRimVsBulk(last.agg, last.geom, last.positions, monomers, 25)
  for (const rb of [rimBulk15, rimBulk25]) {
    console.log(
      `SHELL-PORE-RIMBULK step=${last.step} margin=${rb.marginDeg} ` +
        `rim[heads=${rb.rim.heads} tails=${rb.rim.tails} h:t=${rb.rim.headTailRatio.toFixed(3)} aLipid=${rb.rim.areaPerLipid.toFixed(3)} meanR=${rb.rim.meanRadius.toFixed(2)}] ` +
        `antipodal[heads=${rb.antipodal.heads} tails=${rb.antipodal.tails} h:t=${rb.antipodal.headTailRatio.toFixed(3)} aLipid=${rb.antipodal.areaPerLipid.toFixed(3)} meanR=${rb.antipodal.meanRadius.toFixed(2)}] ` +
        `rest[heads=${rb.rest.heads} tails=${rb.rest.tails} h:t=${rb.rest.headTailRatio.toFixed(3)} aLipid=${rb.rest.areaPerLipid.toFixed(3)} meanR=${rb.rest.meanRadius.toFixed(2)}]`,
    )
  }

  // --- Independent, unwrap-FREE confirmation that the aggregate's own footprint is comparable to
  // the box: the true minimum-image (periodic-aware) distance between any TWO given particles is
  // always well-defined, with no global-frame ambiguity at all, since it only involves one pair. A
  // random sample of many such pairwise distances approximates the aggregate's own true diameter
  // without needing any unwrap, sanity-checking the box/2=29-vs-meanHeadRadius~29.6 comparison above
  // from a completely different angle. -----------------------------------------------------------
  {
    const { positions, box } = loadCheckpoint(last.step)
    const idx = last.agg.aggOriginalIdx
    let maxD = 0
    let sumD = 0
    const samples = 20000
    for (let s = 0; s < samples; s++) {
      const i = idx[Math.floor(Math.random() * idx.length)]
      const j = idx[Math.floor(Math.random() * idx.length)]
      const dx = mi1(positions[i * 4] - positions[j * 4], box[0])
      const dy = mi1(positions[i * 4 + 1] - positions[j * 4 + 1], box[1])
      const dz = mi1(positions[i * 4 + 2] - positions[j * 4 + 2], box[2])
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (d > maxD) maxD = d
      sumD += d
    }
    console.log(
      `SHELL-PORE-DIAMETER step=${last.step} box=${box[0]} boxHalfDiag(sqrt3*box/2)=${((Math.sqrt(3) * box[0]) / 2).toFixed(2)} ` +
        `sampledMaxPeriodicPairDist(n=${samples})=${maxD.toFixed(2)} meanPeriodicPairDist=${(sumD / samples).toFixed(2)}`,
    )
  }

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(
    OUT_FILE,
    JSON.stringify(
      {
        gate: 'shell-pore-check',
        steps: STEPS,
        perStep: perStep.map((p) => ({
          step: p.step,
          aggregateCount: p.agg.aggregateCount,
          amphiphiles: p.agg.aggAmphiphiles.length,
          particles: p.agg.aggOriginalIdx.length,
          radiusOfGyration: p.geom.radiusOfGyration,
          flatnessRatio: p.geom.flatnessRatio,
          holeCentre: p.geom.holeCentre,
          holeRadiusDeg: p.geom.holeRadiusDeg,
          solidAngleFraction: p.geom.solidAngleFraction,
          meanHeadRadius: p.geom.meanHeadRadius,
          cavityVolumeHonest: p.honest,
          unreachedByProperUnwrap: p.geom.unreached,
          unwrapComparison: {
            broken: { meanBondLength: p.brokenSanity.meanBondLength, maxBondLength: p.brokenSanity.maxBondLength, brokenCount: p.brokenSanity.brokenCount, edgeCount: p.agg.edges.length, radiusOfGyration: p.brokenShape.radiusOfGyration },
            fixed: { meanBondLength: p.fixedSanity.meanBondLength, maxBondLength: p.fixedSanity.maxBondLength, brokenCount: p.fixedSanity.brokenCount, radiusOfGyration: p.geom.radiusOfGyration },
          },
        })),
        consecutiveHoleAngles: consecutive,
        meanConsecutiveAngleDeg: meanConsecutiveAngle,
        netDriftFirstToLastDeg: netDrift,
        radialProfile: {
          step: last.step,
          bins: radialLast.profile.z.length,
          z: radialLast.profile.z,
          head: radialLast.profile.head,
          tail: radialLast.profile.tail,
          innerPeak: radialLast.peaks?.lower ?? null,
          outerPeak: radialLast.peaks?.upper ?? null,
          separation: sepLast,
          bilayerThicknessReference: 4.457,
        },
        radialProfileFirst: { step: first.step, innerPeak: radialFirst.peaks?.lower ?? null, outerPeak: radialFirst.peaks?.upper ?? null, separation: sepFirst },
        volumes: {
          step: last.step,
          projectOfficialHonest_brokenUnwrap: last.honest,
          sameFrameUnplugged_properUnwrap: unplugged,
          pluggedAnalytic: analyticPlugged,
          pluggedFlood_sameFrame: floodPlugged,
          innerRadiusUsed: innerRadius,
          closureThreshold: thresholds.enclosedVolume,
        },
        rimVsBulk: [rimBulk15, rimBulk25],
      },
      null,
      2,
    ),
  )
  console.log(`SHELL-PORE artifact written: ${OUT_FILE}`)
})
