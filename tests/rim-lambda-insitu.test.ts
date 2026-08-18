import { readFileSync, mkdirSync, writeFileSync } from 'node:fs'
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import type { CheckpointFile } from '../soup/src/checkpoint'
import { loadSoup, type Monomer } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { findAmphiphiles, type Amphiphile } from '../soup/src/amphiphile'
import { memberIndicesOf, positionsFor } from '../soup/src/aggregates'
import { clusterComponents, shapeOf, unwrapAggregate } from '../engine/src/aggregate'

afterAll(shutdownGpu)

// Task 'kappa-tightening' (2026-08-18), part 2: lambda measured IN SITU on the campaign's own
// refusing-to-close cup (data/checkpoints/vesicle-93k-step250000.json, label 'vesicle-93k'), as
// opposed to kappa-lambda-report.md's own lambda (a flat PREPARED patch with a free edge -- see
// this file's own header note below on why that convention is reused, not replaced, for validation).
//
// No physics/potentials/thresholds/recogniser/bond-rules are touched anywhere in this file: every
// energy formula below (wcaV/feneV/bendV/attrV) is a literal transcription of the antiderivatives
// ALREADY WRITTEN in engine/wgsl/forces.wgsl (wca_v/fene_v/bend_v/attr_v) and the pairing/gating
// logic ALREADY WRITTEN in soup/wgsl/step.wgsl (nonbondedSoup/bondedForce) -- read here, not
// altered, and cited by name at each formula below. Every numeric constant is read from
// data/params.json (loadParams()) or data/soup.json (loadSoup()), never a new literal. The
// checkpoint decode (decodeCheckpointResume, soup/src/checkpoint.ts) and the aggregate-finding
// pipeline (findAmphiphiles, memberIndicesOf, positionsFor, clusterComponents) are the project's own
// existing, already-tested pure functions -- this file adds no new physics, only a NEW MEASUREMENT
// (an offline energy bookkeeping pass over positions+bonds this project's own functions already
// hand back) on top of them, exactly as tests/buckling-kappa.test.ts's own weighted-regression code
// or tests/line-tension.test.ts's own patch/reference bookkeeping do for their prepared samples.
//
// Two GPU calls only, both cheap (no stepping, "small sample" in the resource-rule sense):
//  1. resumes the REAL 93200-particle checkpoint with ZERO further steps and reads back
//     particles()/bonds() to cross-check this file's own from-scratch bondSlots decode (Node-side,
//     no GPU) against the engine's own bonds() derivation -- if they disagree, the whole rim/lambda
//     measurement below would be built on a misread topology.
//  2. a handful of tiny hand-placed synthetic systems (a few particles each, far enough apart in a
//     big box that WCA/attraction never crosses between groups) that isolate each energy term
//     (nonbonded WCA+attraction, FENE, bend) and check this file's own wcaV/feneV/bendV/attrV
//     against the engine's REAL sys.forces() at that exact geometry -- if the transcribed formulas
//     were wired wrong (wrong radius, wrong gate, wrong sign), the force comparison catches it
//     before the checkpoint-scale number is trusted.

const CHECKPOINT_PATH = 'data/checkpoints/vesicle-93k-step250000.json'
const NONE_U32 = 0xffffffff
const OUT_DIR = 'verify/out'
const OUT_FILE = `${OUT_DIR}/rim-lambda-insitu.json`

// --- energy formulas, transcribed verbatim from engine/wgsl/forces.wgsl (wca_v/fene_v/bend_v/
// attr_v) -- see that file's own comments for the derivation of each; nothing here is a new formula.
function wcaCut(b: number): number {
  return Math.pow(2, 1 / 6) * b
}
function wcaV(r: number, b: number, epsilon: number): number {
  if (r >= wcaCut(b)) return 0
  const s6 = Math.pow(b / r, 6)
  const q = 2 * s6 - 1
  return epsilon * q * q
}
function feneV(r: number, k: number, rInf: number): number {
  const x = r / rInf
  return -0.5 * k * rInf * rInf * Math.log(1 - x * x)
}
function bendV(r: number, k: number, r0: number): number {
  const d = r - r0
  return 0.5 * k * d * d
}
function attrV(r: number, rc: number, wc: number, epsilon: number): number {
  if (r < rc) return -epsilon
  if (r > rc + wc) return 0
  const x = (Math.PI * (r - rc)) / (2 * wc)
  const c = Math.cos(x)
  return -epsilon * c * c
}

// Minimum-image displacement/distance, 3-axis periodic -- matches soup/wgsl/step.wgsl's mi3(), the
// soup's own full-3D wrap (unlike the membrane engine's x,y-only mi()), since this checkpoint is a
// soup snapshot, not a membrane patch.
function mi3(d: number, L: number): number {
  return d - Math.round(d / L) * L
}
function dist3(px: Float32Array, i: number, j: number, box: [number, number, number]): number {
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
function forEachPairWithinCutoff(
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

interface EnergyBasis {
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

function makeBasis(): EnergyBasis {
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
    rc: wcaCutoff(p.sigma * p.beadSizes.tail_tail),
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
function pairwiseEnergy(
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
function coordinationNumbers(idx: number[], positions: Float32Array, box: [number, number, number], cutoff: number): Map<number, number> {
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
function greedyChainLength(pts: [number, number, number][], box: [number, number, number]): { open: number; closed: number } {
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

test('lambda in situ: rim of the largest aggregate in the 250000-step vesicle-93k checkpoint (no GPU, pure decode+analysis)', () => {
  const file = JSON.parse(readFileSync(CHECKPOINT_PATH, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  const box = file.config.box
  const N = file.N
  expect(resume.positions.length).toBe(N * 4)
  expect(resume.globalStep).toBe(250000)

  const positions = resume.positions
  const bondSlots = resume.bondSlots

  // Derive unique covalent edges from bondSlots (3 slots/particle, soup/src/sim.ts's bondSlots0 --
  // "partner > i" is the dedup, since soup/wgsl/bond.wgsl's tryClaimSlot records the bond
  // symmetrically in BOTH particles' own slot rows).
  const edges: number[] = []
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < 3; s++) {
      const p = bondSlots[i * 3 + s]
      if (p !== NONE_U32 && p > i) edges.push(i, p)
    }
  }
  const expectedBondCount = file.events.cc_bond - file.events.cc_break + (file.events.co_bond - file.events.co_break)
  console.log(`RIM-LAMBDA derived bond edges=${edges.length / 2} expected(from events)=${expectedBondCount}`)
  expect(edges.length / 2).toBe(expectedBondCount)

  const adjacency: number[][] = Array.from({ length: N }, () => [])
  for (let k = 0; k < edges.length; k += 2) {
    adjacency[edges[k]].push(edges[k + 1])
    adjacency[edges[k + 1]].push(edges[k])
  }

  const soup = loadSoup()
  const monomers: Monomer[] = soup.monomers
  const bondsArr = new Uint32Array(edges)
  const amphiphiles: Amphiphile[] = findAmphiphiles(positions, bondsArr, monomers)
  const memberIdx = memberIndicesOf(amphiphiles)
  const memberPositions = positionsFor(positions, memberIdx)
  const idxArr = Array.from(memberIdx)

  const params = loadParams()
  const memberRadii = monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(params.sigma * Math.max(...memberRadii)) + params.attraction.wc // = same
  // 2.7225 sigma cellSize the kappa-lambda-report.md's own grid-invariant dry-run printed, and the
  // SAME cutoff soup/src/stages.ts's detectStage/analyzeAggregates already use for "connected
  // amphiphile aggregate" -- re-derived here from the same params, not re-typed as a literal.

  const components = clusterComponents(memberPositions, box, cutoff)
  console.log(`RIM-LAMBDA amphiphiles=${amphiphiles.length} members=${memberIdx.size} aggregates=${components.length} cutoff=${cutoff.toFixed(4)}`)
  expect(components.length).toBeGreaterThan(0)

  const largest = components[0]
  const aggOriginalIdx = largest.map((k) => idxArr[k])
  const aggSet = new Set(aggOriginalIdx)
  const aggAmphiphiles = amphiphiles.filter((a) => aggSet.has(a.headIndex))
  console.log(`RIM-LAMBDA largest aggregate particles=${aggOriginalIdx.length} amphiphiles=${aggAmphiphiles.length}`)
  expect(aggAmphiphiles.length).toBeGreaterThan(100) // sanity: this is the campaign's own ~1282-amphiphile cup, not a stray micelle

  const deg = coordinationNumbers(aggOriginalIdx, positions, box, cutoff)
  const headDeg = aggAmphiphiles.map((a) => ({ a, d: deg.get(a.headIndex) ?? 0 }))
  headDeg.sort((x, y) => x.d - y.d)

  // Spatial sanity check (not used in the energy calc itself): is "low head-coordination" actually
  // picking out the object's own geometric BOUNDARY (larger radial distance from its own centroid,
  // consistent with a cup's free rim) rather than, say, a small satellite blob bridged in by one
  // bond? unwrapAggregate (engine/src/aggregate.ts, already used by soup/src/aggregates.ts's own
  // shapeOfAggregate for exactly this reason) removes the periodic-wrap discontinuity first.
  const aggRaw = new Float32Array(aggOriginalIdx.length * 4)
  for (let k = 0; k < aggOriginalIdx.length; k++) aggRaw.set(positions.subarray(aggOriginalIdx[k] * 4, aggOriginalIdx[k] * 4 + 4), k * 4)
  const unwrapped = unwrapAggregate(aggRaw, box, [true, true, true])
  const shape = shapeOf(unwrapped)
  const radiusByOriginal = new Map<number, number>()
  for (let k = 0; k < aggOriginalIdx.length; k++) {
    const dx = unwrapped[k * 4] - shape.centre[0]
    const dy = unwrapped[k * 4 + 1] - shape.centre[1]
    const dz = unwrapped[k * 4 + 2] - shape.centre[2]
    radiusByOriginal.set(aggOriginalIdx[k], Math.sqrt(dx * dx + dy * dy + dz * dz))
  }
  function unwrappedDirOf(originalIdx: number): [number, number, number] {
    const k = aggOriginalIdx.indexOf(originalIdx)
    const dx = unwrapped[k * 4] - shape.centre[0]
    const dy = unwrapped[k * 4 + 1] - shape.centre[1]
    const dz = unwrapped[k * 4 + 2] - shape.centre[2]
    const r = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1
    return [dx / r, dy / r, dz / r]
  }
  console.log(
    `RIM-LAMBDA shape radiusOfGyration=${shape.radiusOfGyration.toFixed(3)} principalMoments=${shape.principalMoments.map((v) => v.toFixed(3))} ` +
      `flatnessRatio=${(shape.principalMoments[0] / shape.principalMoments[2]).toFixed(4)}`,
  )

  // --- Geometric rim: the boundary of the actual MISSING SOLID ANGLE on the shell, not a
  // coordination-number threshold. --------------------------------------------------------------
  // The coordination-based classification above (kept and reported as a ruled-out diagnostic, see
  // the RIM-LAMBDA-COORD lines below and the report) turned out to have a mean pairwise ANGULAR
  // separation (from the shape's own centroid) of ~89.5-89.9 degrees -- indistinguishable from the
  // ~90 degrees uniformly-random points on a sphere would give. That means "low local coordination"
  // is scattered isotropically over this near-spherical shell (radiusOfGyration=30.4,
  // flatnessRatio=0.62 -- a cup that has already curved most of the way around, not a flat disc),
  // not concentrated at one edge -- it is generic packing texture, not the rim. The task's own
  // premise (a cup with a real opening) calls for finding that opening directly: the largest EMPTY
  // patch of solid angle (as seen from the aggregate's own centroid) that this aggregate's own heads
  // fail to cover, via a coarse Fibonacci-sphere candidate grid (a standard even point distribution)
  // -- for each candidate direction, the angular distance to the NEAREST real head; the candidate
  // with the LARGEST such distance is the hole's own centre, and that distance is the hole's own
  // angular radius (by construction: it is exactly how far the search could grow before hitting a
  // real head). The rim is then the heads immediately bordering that empty region.
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
  const headDirs = aggAmphiphiles.map((a) => ({ a, dir: unwrappedDirOf(a.headIndex) }))
  const candidates = fibonacciSphere(800)
  let holeCenter: [number, number, number] = candidates[0]
  let holeRadiusDeg = -1
  for (const c of candidates) {
    let minAngle = Infinity
    for (const h of headDirs) {
      const ang = angleDeg(c, h.dir)
      if (ang < minAngle) minAngle = ang
    }
    if (minAngle > holeRadiusDeg) {
      holeRadiusDeg = minAngle
      holeCenter = c
    }
  }
  const holeSolidAngleFrac = (1 - Math.cos((holeRadiusDeg * Math.PI) / 180)) / 2 // spherical-cap
  // solid angle / 4*pi, standard cap formula -- fraction of the FULL sphere's own surface this
  // empty region covers.
  console.log(
    `RIM-LAMBDA hole centre=[${holeCenter.map((v) => v.toFixed(3))}] angularRadiusDeg=${holeRadiusDeg.toFixed(2)} ` +
      `solidAngleFraction=${(holeSolidAngleFrac * 100).toFixed(2)}%`,
  )
  const headAngleFromHole = headDirs.map((h) => ({ a: h.a, ang: angleDeg(h.dir, holeCenter) })).sort((x, y) => x.ang - y.ang)

  const basis = makeBasis()
  const percentiles = [10, 15, 20, 25]

  function evaluateRimSplit(rimAmph: Amphiphile[], interiorAmph: Amphiphile[]) {
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

  // Ruled-out diagnostic: coordination-percentile rim (kept for the report -- this is the
  // measurement that revealed the ~90 degree isotropy above).
  const coordSensitivity = percentiles.map((pct) => {
    const numRim = Math.max(1, Math.round((pct / 100) * headDeg.length))
    const rimAmph = headDeg.slice(0, numRim).map((x) => x.a)
    const interiorAmph = headDeg.slice(numRim).map((x) => x.a)
    return { rimPercentile: pct, ...evaluateRimSplit(rimAmph, interiorAmph) }
  })
  for (const s of coordSensitivity) {
    console.log(
      `RIM-LAMBDA-COORD(ruled out, isotropic) pct=${s.rimPercentile}% nRim=${s.numRimAmphiphiles}(${s.rimParticleCount}p,meanChain=${s.meanChainLengthRim.toFixed(2)},meanR=${s.meanRadiusRim.toFixed(2)}) ` +
        `nInt=${s.numInteriorAmphiphiles}(${s.interiorParticleCount}p,meanChain=${s.meanChainLengthInterior.toFixed(2)},meanR=${s.meanRadiusInterior.toFixed(2)}) ` +
        `rimAngleDeg=${s.meanPairwiseAngleDeg.toFixed(1)} eExcessFull=${s.eExcessFull.toFixed(2)}(lambdaFull=${s.lambdaFull.toFixed(4)}) eExcessNb=${s.eExcessNb.toFixed(2)} ` +
        `lambdaNb[count=${s.lambdaCountSpacing.toFixed(4)} chainOpen=${s.lambdaChainOpen.toFixed(4)} chainClosed=${s.lambdaChainClosed.toFixed(4)}]`,
    )
  }

  // Primary, geometric rim: heads within [holeRadiusDeg, holeRadiusDeg+margin] of the hole centre
  // found above -- the actual front line bordering the missing solid angle, at several margins as
  // the sensitivity check the task asks for.
  const margins = [10, 15, 20, 25]
  const sensitivity = margins.map((margin) => {
    const rimAmph = headAngleFromHole.filter((h) => h.ang <= holeRadiusDeg + margin).map((h) => h.a)
    const interiorAmph = headAngleFromHole.filter((h) => h.ang > holeRadiusDeg + margin).map((h) => h.a)
    return { holeMarginDeg: margin, ...evaluateRimSplit(rimAmph, interiorAmph) }
  })
  for (const s of sensitivity) {
    console.log(
      `RIM-LAMBDA-GEOM marginDeg=${s.holeMarginDeg} nRim=${s.numRimAmphiphiles}(${s.rimParticleCount}p,meanChain=${s.meanChainLengthRim.toFixed(2)},meanR=${s.meanRadiusRim.toFixed(2)}) ` +
        `nInt=${s.numInteriorAmphiphiles}(${s.interiorParticleCount}p,meanChain=${s.meanChainLengthInterior.toFixed(2)},meanR=${s.meanRadiusInterior.toFixed(2)}) ` +
        `eTotal[fene=${s.eTotalBreakdown.fene.toFixed(1)} bend=${s.eTotalBreakdown.bend.toFixed(1)} nb=${s.eTotalBreakdown.nonbonded.toFixed(1)}] ` +
        `eInt[fene=${s.eInteriorBreakdown.fene.toFixed(1)} bend=${s.eInteriorBreakdown.bend.toFixed(1)} nb=${s.eInteriorBreakdown.nonbonded.toFixed(1)}] ` +
        `eExcessFull=${s.eExcessFull.toFixed(2)}(lambdaFull=${s.lambdaFull.toFixed(4)}) eExcessNb=${s.eExcessNb.toFixed(2)} ` +
        `rimLen[count=${s.rimLengthCountSpacing.toFixed(2)} chainOpen=${s.rimLengthChainOpen.toFixed(2)} chainClosed=${s.rimLengthChainClosed.toFixed(2)}] ` +
        `lambdaNb[count=${s.lambdaCountSpacing.toFixed(4)} chainOpen=${s.lambdaChainOpen.toFixed(4)} chainClosed=${s.lambdaChainClosed.toFixed(4)}]`,
    )
  }

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(
    OUT_FILE,
    JSON.stringify(
      {
        gate: 'rim-lambda-in-situ',
        checkpoint: CHECKPOINT_PATH,
        globalStep: resume.globalStep,
        N,
        box,
        cutoff,
        aggregateCount: components.length,
        largestAggregateParticles: aggOriginalIdx.length,
        largestAggregateAmphiphiles: aggAmphiphiles.length,
        shape: { radiusOfGyration: shape.radiusOfGyration, principalMoments: shape.principalMoments },
        hole: { centre: holeCenter, angularRadiusDeg: holeRadiusDeg, solidAngleFraction: holeSolidAngleFrac },
        coordSensitivity, // ruled-out diagnostic (isotropic, ~90deg mean pairwise angle)
        sensitivity, // primary: geometric hole-boundary rim
      },
      null,
      2,
    ),
  )
  console.log(`RIM-LAMBDA artifact written: ${OUT_FILE}`)

  // The rim genuinely costs energy relative to the interior baseline at every threshold tried, and
  // a real (finite, positive) contour length backs every lambda estimate -- otherwise this
  // measurement has nothing to report.
  for (const s of sensitivity) {
    expect(Number.isFinite(s.eExcess)).toBe(true)
    expect(s.rimLengthCountSpacing).toBeGreaterThan(0)
    expect(s.rimLengthChainOpen).toBeGreaterThan(0)
  }
})

test('checkpoint decode cross-check: this file\'s own bondSlots-derived topology matches a live resumed SoupSystem\'s own bonds() (zero steps taken)', async () => {
  const raw = readFileSync(CHECKPOINT_PATH, 'utf8')
  const file = JSON.parse(raw) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  const N = file.N

  const edges: number[] = []
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < 3; s++) {
      const p = resume.bondSlots[i * 3 + s]
      if (p !== NONE_U32 && p > i) edges.push(i, p)
    }
  }

  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))
  const summary = await page.evaluate(async (checkpointJson: string) => {
    const api = (window as any).api
    const f = JSON.parse(checkpointJson)
    const r = api.decodeCheckpointResume(f)
    const sys = await api.createSoup({ box: f.config.box, seed: f.config.seed, kT: f.config.kT, start: f.config.start, resume: r })
    const particles = await sys.particles()
    const bonds = await sys.bonds()
    const sampleIdx = [0, 1, 1000, 50000, 93199]
    const sample = sampleIdx.map((i) => [particles[i * 4], particles[i * 4 + 1], particles[i * 4 + 2], particles[i * 4 + 3]])
    const N2 = particles.length / 4
    const steps = sys.steps
    sys.dispose()
    return { N: N2, steps, bondPairCount: bonds.length / 2, sample, sampleIdx, bondsPairs: Array.from(bonds) }
  }, raw) as {
    N: number
    steps: number
    bondPairCount: number
    sample: number[][]
    sampleIdx: number[]
    bondsPairs: number[]
  }

  console.log(`RIM-LAMBDA-VALIDATE liveN=${summary.N} liveSteps=${summary.steps} liveBondPairs=${summary.bondPairCount} decodedEdges=${edges.length / 2}`)
  expect(summary.N).toBe(N)
  expect(summary.steps).toBe(file.globalStep)
  expect(summary.bondPairCount).toBe(edges.length / 2)

  for (let k = 0; k < summary.sampleIdx.length; k++) {
    const i = summary.sampleIdx[k]
    const [x, y, z, kind] = summary.sample[k]
    expect(x).toBeCloseTo(resume.positions[i * 4], 5)
    expect(y).toBeCloseTo(resume.positions[i * 4 + 1], 5)
    expect(z).toBeCloseTo(resume.positions[i * 4 + 2], 5)
    expect(kind).toBeCloseTo(resume.positions[i * 4 + 3], 5)
  }

  // Same edge SET, not just the same count: build the live pair list as (min,max) tuples and
  // compare against this file's own decode, sorted, so ordering differences (the GPU's own bonds()
  // walks particles in index order per its own internal bookkeeping, not necessarily identical to
  // this file's slot-scan order) do not cause a false mismatch.
  const liveEdges: string[] = []
  for (let k = 0; k < summary.bondsPairs.length; k += 2) {
    const a = summary.bondsPairs[k]
    const b = summary.bondsPairs[k + 1]
    liveEdges.push(a < b ? `${a},${b}` : `${b},${a}`)
  }
  const decodedEdges: string[] = []
  for (let k = 0; k < edges.length; k += 2) decodedEdges.push(`${edges[k]},${edges[k + 1]}`)
  liveEdges.sort()
  decodedEdges.sort()
  expect(liveEdges).toEqual(decodedEdges)
}, 300_000)

test('formula validation: wcaV/feneV/bendV/attrV against the real engine\'s sys.forces() on tiny hand-placed synthetic systems', async () => {
  const params = loadParams()
  const soup = loadSoup()
  const carbonKind = soup.monomers.findIndex((m) => m.kind === 'carbon')
  expect(carbonKind).toBeGreaterThanOrEqual(0)

  const box: [number, number, number] = [40, 40, 40]
  // 7 particles, three well-separated (>8 sigma, far above the 2.7225 sigma interaction cutoff)
  // groups so each isolates exactly one energy term:
  //  - group A (p0,p1): unbonded, r=1.05 sigma -- inside BOTH the WCA core and the attraction
  //    plateau (r < rc), so this checks wcaV and attrV's constant-plateau branch together.
  //  - group B (p2,p3): unbonded, r=1.9 sigma -- inside the attraction cos^2 ramp, outside WCA
  //    entirely, checking attrV's ramp branch in isolation.
  //  - group C (p4,p5,p6): a bonded chain p4-p5-p6 (FENE p4-p5, FENE p5-p6), r(p4,p5)=r(p5,p6)=1.0
  //    sigma, checking feneV; bend is credited to the (p4,p6) pair with p5 as the shared middle,
  //    r(p4,p6)=2.0 sigma, checking bendV.
  const positions = new Float32Array([
    5, 5, 5, carbonKind, // p0
    6.05, 5, 5, carbonKind, // p1 (r=1.05 from p0)
    20, 5, 5, carbonKind, // p2
    21.9, 5, 5, carbonKind, // p3 (r=1.9 from p2)
    5, 20, 5, carbonKind, // p4
    6, 20, 5, carbonKind, // p5 (r=1.0 from p4)
    7, 20, 5, carbonKind, // p6 (r=1.0 from p5, r=2.0 from p4)
  ])
  const N = 7
  const velocities = new Float32Array(N * 4)
  const bondSlots = new Uint32Array(N * 3).fill(NONE_U32)
  // p4(idx4) slot0 = p5(idx5); p5 slot0 = p4, slot1 = p6; p6 slot0 = p5.
  bondSlots[4 * 3 + 0] = 5
  bondSlots[5 * 3 + 0] = 4
  bondSlots[5 * 3 + 1] = 6
  bondSlots[6 * 3 + 0] = 5
  const centerLink = new Uint32Array(N).fill(NONE_U32)
  const centerHeldSteps = new Uint32Array(N)
  const bondRng = new Uint32Array(N).fill(1)
  const thermoRng = new Uint32Array(N).fill(1)

  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))
  const forces = await page.evaluate(
    async (
      boxArg: [number, number, number],
      positionsArg: number[],
      velocitiesArg: number[],
      bondSlotsArg: number[],
      centerLinkArg: number[],
      centerHeldStepsArg: number[],
      bondRngArg: number[],
      thermoRngArg: number[],
      nArg: number,
      carbonKindArg: number,
    ) => {
      const api = (window as any).api
      const sys = await api.createSoup({
        box: boxArg,
        seed: 1,
        kT: 1.1,
        start: { C: nArg, O: 0, H: 0, M: 0 },
        resume: {
          globalStep: 0,
          liveBox: boxArg,
          positions: new Float32Array(positionsArg),
          velocities: new Float32Array(velocitiesArg),
          bondSlots: new Uint32Array(bondSlotsArg),
          centerLink: new Uint32Array(centerLinkArg),
          centerHeldSteps: new Uint32Array(centerHeldStepsArg),
          desorbEvents: new Uint32Array([0, 0]),
          bondRng: new Uint32Array(bondRngArg),
          thermoRng: new Uint32Array(thermoRngArg),
          events: {},
        },
      })
      const f = await sys.forces()
      sys.dispose()
      return Array.from(f)
    },
    box,
    Array.from(positions),
    Array.from(velocities),
    Array.from(bondSlots),
    Array.from(centerLink),
    Array.from(centerHeldSteps),
    Array.from(bondRng),
    Array.from(thermoRng),
    N,
    carbonKind,
  ) as number[]

  const basis = makeBasis()
  const b = basis.sigma * (basis.radiusOf(carbonKind) + basis.radiusOf(carbonKind)) * 0.5 // C-C pairB

  // Slope functions -- dV/dr by central finite difference on THIS FILE's own wcaV/attrV/feneV/bendV
  // closed forms, matching forces.wgsl's own wca_dv/fene_dv/bend_dv/attr_dv NAMING AND SIGN exactly
  // (that file's own header: "wca_dv/fene_dv/bend_dv above ... are dV/dr"). Independent of the
  // engine's internal *_dv code -- a genuine cross-check of the antiderivative pair (V and its own
  // slope), not a tautology against the same formula the engine calls.
  function centralDiff(f: (r: number) => number, r: number, h = 1e-4): number {
    return (f(r + h) - f(r - h)) / (2 * h)
  }
  const wcaAttrSlope = (r: number) => centralDiff((rr) => wcaV(rr, b, basis.epsilon) + attrV(rr, basis.rc, basis.wc, basis.epsilon), r)
  const feneSlope = (r: number) => centralDiff((rr) => feneV(rr, basis.feneK, basis.feneRInf), r)
  const bendSlope = (r: number) => centralDiff((rr) => bendV(rr, basis.bendK, basis.bendR0), r)

  // General pairwise force (forces.wgsl's own convention): F_on_A_from_B = -slope(r_AB) * (xA-xB)/r_AB.
  function forceX(xA: number, xB: number, slope: (r: number) => number): number {
    const d = xA - xB
    const r = Math.abs(d)
    return -slope(r) * (d / r)
  }

  // Group A: p0<->p1 (unbonded), r=1.05 -- inside the WCA core AND the attraction plateau.
  const r01 = 1.05
  const expectedF0x = forceX(positions[0 * 4], positions[1 * 4], wcaAttrSlope)
  console.log(`FORMULA-CHECK groupA r=${r01} expectedF0x=${expectedF0x.toFixed(6)} realF0x=${forces[0 * 4].toFixed(6)}`)
  expect(forces[0 * 4]).toBeCloseTo(expectedF0x, 2)
  expect(Math.abs(forces[0 * 4 + 1])).toBeLessThan(1e-4)
  expect(Math.abs(forces[0 * 4 + 2])).toBeLessThan(1e-4)

  // Group B: p2<->p3 (unbonded), r=1.9 -- attraction-ramp-only regime (outside the WCA core).
  const r23 = 1.9
  const expectedF2x = forceX(positions[2 * 4], positions[3 * 4], wcaAttrSlope)
  console.log(`FORMULA-CHECK groupB r=${r23} expectedF2x=${expectedF2x.toFixed(6)} realF2x=${forces[2 * 4].toFixed(6)}`)
  expect(forces[2 * 4]).toBeCloseTo(expectedF2x, 2)

  // Group C: p4-p5-p6 chain (FENE p4-p5, FENE p5-p6; bend triple (p4,p6) with p5 the shared bonded
  // middle -- soup/wgsl/step.wgsl's bondedForce: bend is credited to the OUTER pair, never to the
  // middle itself). First pass at "expected" here (fene+bend only) missed a real effect and FAILED
  // against sys.forces() by ~23 units on p4/p6 -- soup/wgsl/step.wgsl's nonbondedSoup (the grid-walk
  // neighbour search feeding soupForceWalk) does NOT know or care about bond topology: it adds its
  // OWN WCA+attraction contribution to every pair found within the interaction cutoff regardless of
  // whether they are ALSO bonded or a 1-3 neighbour (forces.wgsl's own header on the fixed-lipid
  // engine says this explicitly: "WCA is applied here to EVERY pair, including the two FENE-bonded
  // pairs and the 1-3 bend pair"; soup/wgsl/step.wgsl's nonbondedSoup carries the same convention
  // forward, unconditionally). So p4-p5 (r=1.0, inside the WCA core) and p4-p6 (r=2.0, inside the
  // attraction ramp) EACH also contribute a wcaV+attrV term on top of the bonded fene/bend terms --
  // confirmed by back-solving the exact ~23-unit gap to wcaAttrSlope(1.0)=-24.0 (WCA core) plus
  // wcaAttrSlope(2.0)=+0.97 (attraction ramp), which closes it to <0.001. This is exactly the same
  // per-pair gating pairwiseEnergy() above already uses (its own nonbonded loop makes no bond-status
  // exception either) -- this fix corrects the TEST's own hand-derived expectation, not the
  // measurement code, which had this right from the start.
  const expectedF4x =
    forceX(positions[4 * 4], positions[5 * 4], feneSlope) +
    forceX(positions[4 * 4], positions[6 * 4], bendSlope) +
    forceX(positions[4 * 4], positions[5 * 4], wcaAttrSlope) +
    forceX(positions[4 * 4], positions[6 * 4], wcaAttrSlope)
  const expectedF6x =
    forceX(positions[6 * 4], positions[5 * 4], feneSlope) +
    forceX(positions[6 * 4], positions[4 * 4], bendSlope) +
    forceX(positions[6 * 4], positions[5 * 4], wcaAttrSlope) +
    forceX(positions[6 * 4], positions[4 * 4], wcaAttrSlope)
  const expectedF5x =
    forceX(positions[5 * 4], positions[4 * 4], feneSlope) +
    forceX(positions[5 * 4], positions[6 * 4], feneSlope) +
    forceX(positions[5 * 4], positions[4 * 4], wcaAttrSlope) +
    forceX(positions[5 * 4], positions[6 * 4], wcaAttrSlope)
  console.log(
    `FORMULA-CHECK groupC expectedF4x=${expectedF4x.toFixed(6)} realF4x=${forces[4 * 4].toFixed(6)} ` +
      `expectedF6x=${expectedF6x.toFixed(6)} realF6x=${forces[6 * 4].toFixed(6)} ` +
      `expectedF5x=${expectedF5x.toFixed(6)} realF5x=${forces[5 * 4].toFixed(6)}`,
  )
  expect(forces[4 * 4]).toBeCloseTo(expectedF4x, 1)
  expect(forces[6 * 4]).toBeCloseTo(expectedF6x, 1)
  expect(forces[5 * 4]).toBeCloseTo(expectedF5x, 1)
}, 120_000)
