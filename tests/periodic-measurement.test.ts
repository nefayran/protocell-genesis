import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import type { CheckpointFile } from '../soup/src/checkpoint'
import { loadSoup, type Monomer } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { findAmphiphiles } from '../soup/src/amphiphile'
import { clusterComponents } from '../engine/src/aggregate'
import { loadStageThresholds } from '../soup/src/stages'
import periodicParams from '../data/periodic-measurement.json'

// Task 'periodic-measurement' (2026-08-18). Branch stage-a-atoms, no worktree, no new simulation:
// every real-data number below comes from the ten checkpoints already on disk
// (data/checkpoints/vesicle-93k-step160000.json .. step250000.json), decoded purely in Node
// (decodeCheckpointResume) -- no GPU, no gpuPage(), no page.evaluate() anywhere in this file.
//
// WHY THIS FILE EXISTS: .superpowers/sdd/2026-08-16-soup-to-vesicle/shell-pore-report.md and
// expanded-box-report.md both establish that the dense campaign's largest aggregate (~1282
// amphiphiles) spans a box of only 58 sigma/side -- its own mean head radius (~29.5-29.8 sigma) is
// comparable to or exceeds box/2=29 sigma, and its own covalent bonds pass through the periodic
// boundary too tightly to survive any box expansion (FENE divergence on wraparound-dependent bonds,
// expanded-box-report.md SS2 run 5). Consequently EVERY geometric reading obtained via a single
// unwrapped frame -- this project's own unwrapAggregate(), or shell-pore-report.md's own corrected
// properUnwrap() -- is unreliable at this scale (shell-pore-report.md SS0): a particle on the far
// side of an object whose own diameter approaches the box can alias to the wrong periodic image no
// matter which reference particle or BFS walk order the unwrap starts from.
//
// This file's whole point is to measure the SAME object WITHOUT ever building a global unwrapped
// frame. Minimum-image distances/directions between any two points are always well-defined
// regardless of the object's own size (they involve exactly one pair, no global frame at all) --
// what is ill-defined is a single COORDINATE-VALUED centre of mass, which this file replaces with a
// periodic (circular-mean) centre, defined per axis and refused when the axis is not concentrated
// enough to support one (data/periodic-measurement.json's own written basis). Every downstream
// measurement (radial profile, cavity flood, hole search) is built ONLY from minimum-image
// displacements relative to that periodic centre -- no unwrap() call, no BFS-over-covalent-bonds
// walk, anywhere in this file.
//
// Runs under vitest, not plain tsx, for the same mechanical reason shell-pore-check.test.ts already
// documents: soup/src/stages.ts (loadStageThresholds, imported below for closureCell/closureRadius/
// headDensityBins/enclosedVolume -- reused, not reinvented) transitively imports engine/src/
// closure.ts, which does a Vite-specific `?raw` import of its WGSL shader at module load time; only
// Vite's own transform (vitest included) resolves that. No test in this file calls gpuPage() --
// nothing here touches a browser or the GPU.
//
// No numeric model constant is added to soup/src or engine/src (neither is touched at all -- every
// function below is new analysis code, kept in this file only, exactly the precedent
// shell-pore-check.test.ts's own properUnwrap() already set: "this does not touch
// engine/src/aggregate.ts -- it is new analysis code, kept in this file only"). The one genuinely
// new tunable parameter this task introduces (the circular-concentration refusal threshold) lives in
// data/periodic-measurement.json with its own written basis, per this task's own constraint; the
// Fibonacci-sphere hole-search resolution is reused verbatim from shell-pore-check.test.ts's own
// choice (same file, same basis) for direct comparability, also recorded there. Every other
// threshold this file reads (closureCell, closureRadius, headDensityBins, enclosedVolume) is the
// project's own existing, already-justified data/soup.json value (loadStageThresholds()), not a new
// number.

const NONE_U32 = 0xffffffff
const STEPS = [160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000]
const OUT_DIR = 'verify/out'
const OUT_FILE = `${OUT_DIR}/periodic-measurement.json`
const CONCENTRATION_ALPHA = periodicParams.circularConcentration.alpha
const FIB_SAMPLES = periodicParams.fibonacciSphereSamples.count

/** Rayleigh's test for circular uniformity (data/periodic-measurement.json's own written basis):
 * the minimum mean-resultant-length R that n points must show, at significance level alpha, before
 * their circular mean is distinguishable from a genuinely uniform (no-preferred-direction) axis.
 * Falls with n (more points make a smaller true concentration detectable) -- unlike a fixed R bar,
 * which this task's own first attempt found refuses even a large, genuinely non-uniform, real
 * membrane object (see the data file's basis for the measured numbers that forced this revision). */
function rayleighRMin(n: number, alpha: number): number {
  return Math.sqrt(-Math.log(alpha) / n)
}

// --- generic periodic geometry primitives, no unwrap anywhere -------------------------------------

/** Minimum-image displacement of one coordinate on a periodic axis of length `box` -- the same
 * convention as engine/src/aggregate.ts's own mi1/miAxis and sim.ts's mi(), duplicated here (not
 * imported) because this file adds no dependency on those modules' internals, only on their already-
 * exported, unwrap-based functions it deliberately does NOT call. */
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

/** Wraps one coordinate into [0, box). */
function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

export interface AxisCircularStat {
  /** Mean resultant length of the unit vectors formed from this axis' own coordinates mapped to
   * angles -- 1 for a perfectly localized coordinate, 0 for one uniformly spread across the whole
   * periodic box (Mardia & Jupp 1999; see data/periodic-measurement.json for the full derivation). */
  R: number
  /** The circular mean angle itself (radians, atan2 range), regardless of whether R clears rMin --
   * reported so a caller can see WHAT number was rejected, not just that one was. */
  angle: number
  /** The angle converted back to a box coordinate, or null when R fails the Rayleigh significance
   * test below -- refused, per this task's own brief ("refuse to report a centre when [the
   * concentration] is too low"). */
  coordinate: number | null
  trusted: boolean
  /** The Rayleigh-test threshold R had to clear for this many points -- reported alongside R itself
   * so a reader can see the actual bar, not just the pass/fail bit. */
  rMin: number
}

/** Circular mean of one axis' own coordinates (Mardia & Jupp 1999, ch.2): map each coordinate to an
 * angle on the box's own period, average the unit vectors, convert back. This is the textbook fix
 * for averaging a periodic quantity -- a plain arithmetic mean of, say, [1, box-1] would report
 * box/2 (the wrong side of the box entirely) for two points that are actually right next to each
 * other across the seam; the circular mean instead reports ~0 (or ~box), the physically correct
 * answer, because it averages ANGLES, which correctly wrap. Trust is decided by the Rayleigh test
 * (rayleighRMin, data/periodic-measurement.json's own written basis), not a fixed R bar. */
function circularAxisStat(coordsRaw: readonly number[], boxLen: number, alpha: number): AxisCircularStat {
  const n = coordsRaw.length
  if (n === 0) throw new Error('circularAxisStat: пустой список координат')
  let sc = 0
  let ss = 0
  for (const x of coordsRaw) {
    const theta = (2 * Math.PI * wrap1(x, boxLen)) / boxLen
    sc += Math.cos(theta)
    ss += Math.sin(theta)
  }
  sc /= n
  ss /= n
  const R = Math.sqrt(sc * sc + ss * ss)
  const angle = Math.atan2(ss, sc)
  const rMin = rayleighRMin(n, alpha)
  const trusted = R >= rMin
  const coordinate = trusted ? wrap1((angle / (2 * Math.PI)) * boxLen, boxLen) : null
  return { R, angle, coordinate, trusted, rMin }
}

export interface PeriodicCentre {
  axes: [AxisCircularStat, AxisCircularStat, AxisCircularStat]
  /** null unless all three axes are individually trusted -- a centre is only as good as its worst
   * axis; reporting two trustworthy coordinates and one meaningless one as a "position" would be
   * exactly the failure this task's brief warns against, just hidden inside a 3-vector instead of a
   * scalar. */
  centre: [number, number, number] | null
}

/** The periodic centre of `idx` (indices into the flat vec4-per-particle `positions` array), one
 * circular mean per axis -- see circularAxisStat's own doc comment. Never unwraps anything; every
 * axis is handled independently and directly off the raw (periodically wrapped, or not -- wrap1
 * handles either) coordinate. */
function periodicCentre(positions: Float32Array, idx: readonly number[], box: [number, number, number], alpha: number): PeriodicCentre {
  const axes = [0, 1, 2].map((axis) => {
    const coords = idx.map((i) => positions[i * 4 + axis])
    return circularAxisStat(coords, box[axis], alpha)
  }) as [AxisCircularStat, AxisCircularStat, AxisCircularStat]
  const centre = axes.every((a) => a.trusted)
    ? ([axes[0].coordinate as number, axes[1].coordinate as number, axes[2].coordinate as number] as [number, number, number])
    : null
  return { axes, centre }
}

/** Minimum-image displacement vector from `centre` to `pos`, one mi1() per axis -- exact and unique
 * regardless of the source object's own size, unlike a coordinate-valued unwrap. */
function miVectorFromCentre(pos: readonly [number, number, number], centre: readonly [number, number, number], box: readonly [number, number, number]): [number, number, number] {
  return [mi1(pos[0] - centre[0], box[0]), mi1(pos[1] - centre[1], box[1]), mi1(pos[2] - centre[2], box[2])]
}

// --- Task 2: periodic radial head/tail profile -----------------------------------------------------

export interface RadialBin {
  rLo: number
  rHi: number
  headCount: number
  tailCount: number
  /** count / shell volume (4/3*pi*(rHi^3-rLo^3)) -- the volume-normalised reading
   * shell-pore-report.md SS1 found more honest than a raw count histogram (a raw count profile rises
   * with r purely from the growing shell volume, even for beads spread with constant density). */
  headDensity: number
  tailDensity: number
}

export interface RadialProfileResult {
  bins: RadialBin[]
  /** min(box)/2 -- the absolute ceiling past which a minimum-image distance from any fixed point can
   * alias to the WRONG periodic image (see this file's header). No bin extends past this. */
  trustLimit: number
  headTotal: number
  tailTotal: number
  /** Heads/tails whose periodic distance from the centre already reached or exceeded trustLimit --
   * excluded from every bin, counted here so the profile's own coverage is reported honestly rather
   * than silently dropping members. */
  headBeyondTrust: number
  tailBeyondTrust: number
}

/** Radial density profile of head/tail beads about a PERIODIC centre, using minimum-image distances
 * only -- no unwrap, no coordinate-valued frame. Bins run from 0 to box/2 (min over axes) because a
 * minimum-image distance is only unambiguous out to there (this file's header); anything at or past
 * that limit is excluded and counted, not silently folded into the last bin. `isHead` classifies a
 * member by whatever convention the caller's own `positions` encoding uses (the project's own
 * monomers[].polar for real checkpoints, or a synthetic w-field convention for the validation tests
 * below -- kept as a caller-supplied predicate specifically so the SAME function is exercised by
 * both, rather than a synthetic-only reimplementation standing in for it). */
function periodicRadialProfile(
  idx: readonly number[],
  positions: Float32Array,
  isHead: (i: number) => boolean,
  centre: readonly [number, number, number],
  box: readonly [number, number, number],
  bins: number,
): RadialProfileResult {
  const trustLimit = Math.min(box[0], box[1], box[2]) / 2
  const dr = trustLimit / bins
  const headCounts = new Array(bins).fill(0)
  const tailCounts = new Array(bins).fill(0)
  let headTotal = 0
  let tailTotal = 0
  let headBeyondTrust = 0
  let tailBeyondTrust = 0
  for (const i of idx) {
    const head = isHead(i)
    if (head) headTotal++
    else tailTotal++
    const v = miVectorFromCentre([positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2]], centre, box)
    const r = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if (r >= trustLimit) {
      if (head) headBeyondTrust++
      else tailBeyondTrust++
      continue
    }
    const b = Math.min(bins - 1, Math.floor(r / dr))
    if (head) headCounts[b]++
    else tailCounts[b]++
  }
  const out: RadialBin[] = []
  for (let b = 0; b < bins; b++) {
    const rLo = b * dr
    const rHi = (b + 1) * dr
    const shellVolume = (4 / 3) * Math.PI * (rHi ** 3 - rLo ** 3)
    out.push({
      rLo,
      rHi,
      headCount: headCounts[b],
      tailCount: tailCounts[b],
      headDensity: headCounts[b] / shellVolume,
      tailDensity: tailCounts[b] / shellVolume,
    })
  }
  return { bins: out, trustLimit, headTotal, tailTotal, headBeyondTrust, tailBeyondTrust }
}

export interface TwoPeaks {
  primary: number
  secondary: number | null
  separation: number | null
}

/** The two most prominent maxima of `counts` (bin values), at least `minSeparation` apart on
 * `centers` -- the SAME masked-argmax method engine/src/metrics.ts's bilayerPeaks() uses (global
 * max, mask a window around it, argmax again), reimplemented here directly on a raw bin array
 * instead of a ZProfile/densityProfileZ object, since the periodic radial profile above is not a
 * z-profile and does not have a monotonic-decreasing-background assumption built in anywhere --
 * unlike bilayerPeaks(), this makes NO claim that a two-maxima reading means "two shells": callers
 * must look at the actual profile (this function's caller logs and stores it in full) to say whether
 * that reading sits on a genuine bimodal profile or on an arbitrary bump in an otherwise monotonic
 * background, per this task's own instruction not to force a two-shell reading. */
function twoPeaks(counts: readonly number[], centers: readonly number[], minSeparation: number): TwoPeaks {
  let a = 0
  for (let i = 1; i < counts.length; i++) if (counts[i] > counts[a]) a = i
  if (counts[a] === 0) return { primary: NaN, secondary: null, separation: null }
  const masked = counts.map((v, i) => (Math.abs(centers[i] - centers[a]) <= minSeparation ? 0 : v))
  let b = 0
  for (let i = 1; i < masked.length; i++) if (masked[i] > masked[b]) b = i
  if (masked[b] === 0) return { primary: centers[a], secondary: null, separation: null }
  const lower = Math.min(centers[a], centers[b])
  const upper = Math.max(centers[a], centers[b])
  return { primary: centers[a], secondary: centers[b], separation: upper - lower }
}

// --- Task 3: periodic flood-fill for the cavity ----------------------------------------------------
//
// engine/src/closure.ts's occupancy()/floodOutside()/enclosedVolume() treat all six box faces as
// OPEN boundary, by deliberate design documented at that module's own header -- correct for a
// membrane-engine object that sits well inside the box, wrong for an object whose own extent is
// comparable to the box (exactly this task's subject): an aggregate that genuinely touches or spans
// a face would have that face's own cells seeded as "outside" regardless of what real structure
// surrounds them, which is precisely the kind of spurious small pocket shell-pore-report.md SS3
// already caught the OLD (non-periodic, unwrapped-frame) detector manufacturing. The functions below
// are a PERIODIC variant, alongside (not instead of) the existing one: occupancy wraps all three
// axes (matching the soup engine's own physics -- "a bulk soup has no preferred axis", soup/src/
// sim.ts's own header), and the flood seeds from cells that are DEMONSTRABLY outside the aggregate --
// the single emptiest cell by minimum-image distance from the periodic centre, i.e. the periodic
// box's own farthest point from the object's own location, which for a compact object is nowhere
// near its own material regardless of the object's own radius (no outer-radius estimate is needed to
// pick it: for a periodic cube, the point diagonally opposite the centre -- distance sqrt(3)*box/2 --
// is always at least as far from the centre as any face-midpoint direction, so it stays clear of an
// object whose own radius is comparable to box/2 even though such an object's own material would
// already touch every face). engine/src/closure.ts itself is not imported or modified anywhere in
// this file -- the existing non-periodic detector stays exactly as it is for the tests that already
// rely on it (tests/closure.test.ts, tests/soup-aggregates.test.ts and everything that reaches
// analyzeAggregates()); this is new, additional code.

interface PeriodicOccupancy {
  occ: Uint8Array
  dims: [number, number, number]
}

/** Same rule as engine/src/closure.ts's occupancy() (mark every grid cell whose CENTRE lies within
 * `radius` of some bead in `idx`), generalised to wrap all THREE axes -- that file wraps only x,y (a
 * deliberate choice correct for its own membrane-engine z-open convention, see its own header); the
 * soup engine this task's data comes from wraps x, y AND z, so a periodic flood over it must too. */
function occupancyPeriodic(positions: Float32Array, idx: readonly number[], box: readonly [number, number, number], cell: number, radius: number): PeriodicOccupancy {
  const dims: [number, number, number] = [Math.max(1, Math.ceil(box[0] / cell)), Math.max(1, Math.ceil(box[1] / cell)), Math.max(1, Math.ceil(box[2] / cell))]
  const [nx, ny, nz] = dims
  const occ = new Uint8Array(nx * ny * nz)
  const reach = Math.max(1, Math.ceil(radius / cell))
  const r2 = radius * radius
  for (const i of idx) {
    const px = wrap1(positions[i * 4], box[0])
    const py = wrap1(positions[i * 4 + 1], box[1])
    const pz = wrap1(positions[i * 4 + 2], box[2])
    const cx = Math.floor(px / cell)
    const cy = Math.floor(py / cell)
    const cz = Math.floor(pz / cell)
    for (let dz = -reach; dz <= reach; dz++) {
      const gz = (((cz + dz) % nz) + nz) % nz
      let ddz = (gz + 0.5) * cell - pz
      ddz -= Math.round(ddz / box[2]) * box[2]
      for (let dy = -reach; dy <= reach; dy++) {
        const gy = (((cy + dy) % ny) + ny) % ny
        let ddy = (gy + 0.5) * cell - py
        ddy -= Math.round(ddy / box[1]) * box[1]
        for (let dx = -reach; dx <= reach; dx++) {
          const gx = (((cx + dx) % nx) + nx) % nx
          let ddx = (gx + 0.5) * cell - px
          ddx -= Math.round(ddx / box[0]) * box[0]
          if (ddx * ddx + ddy * ddy + ddz * ddz <= r2) occ[gx + nx * (gy + ny * gz)] = 1
        }
      }
    }
  }
  return { occ, dims }
}

/** Breadth-first flood of every EMPTY cell reachable from `seeds`, wrapping all three axes (unlike
 * engine/src/closure.ts's floodOutside(), which is deliberately open-boundary -- see this section's
 * header). Seeds are supplied by the caller, not derived from "which face a cell is on" (there are no
 * faces once every axis wraps): see periodicCavityVolume() below for how they are chosen. */
function floodOutsidePeriodic(occ: Uint8Array, dims: readonly [number, number, number], seeds: readonly number[]): Uint8Array {
  const [nx, ny, nz] = dims
  const visited = new Uint8Array(occ.length)
  const idx = (x: number, y: number, z: number) => x + nx * (y + ny * z)
  const queue: number[] = []
  for (const s of seeds) {
    if (occ[s] === 0 && visited[s] === 0) {
      visited[s] = 1
      queue.push(s)
    }
  }
  let head = 0
  while (head < queue.length) {
    const i = queue[head++]
    const z = Math.floor(i / (nx * ny))
    const rem = i - z * nx * ny
    const y = Math.floor(rem / nx)
    const x = rem - y * nx
    const neighbours: [number, number, number][] = [
      [((x + 1) % nx + nx) % nx, y, z],
      [((x - 1) % nx + nx) % nx, y, z],
      [x, ((y + 1) % ny + ny) % ny, z],
      [x, ((y - 1) % ny + ny) % ny, z],
      [x, y, ((z + 1) % nz + nz) % nz],
      [x, y, ((z - 1) % nz + nz) % nz],
    ]
    for (const [nx2, ny2, nz2] of neighbours) {
      const ni = idx(nx2, ny2, nz2)
      if (occ[ni] === 0 && visited[ni] === 0) {
        visited[ni] = 1
        queue.push(ni)
      }
    }
  }
  return visited
}

/** The single empty grid cell with the greatest minimum-image distance from `centre` -- "the
 * emptiest region far from the periodic centre" the task's brief asks for, made concrete: no
 * outer-radius estimate is needed (see this section's header for why the periodic box's own farthest
 * point from any fixed centre is always at least as extreme as any face-based seed a non-periodic
 * flood could pick). A single seed is sufficient for a standard flood-fill as long as it belongs to
 * the true connected exterior; periodicCavityVolume() below reports enough of this cell's own
 * geometry (its distance from the centre, and the total reached/unreached cell counts) for a reader
 * to judge whether that held, rather than trusting it silently. Throws if the whole grid is occupied
 * (nothing to seed from) -- a real failure, not a number to fabricate. */
function farthestEmptyCellSeed(occ: Uint8Array, dims: readonly [number, number, number], cell: number, box: readonly [number, number, number], centre: readonly [number, number, number]): number {
  const [nx, ny, nz] = dims
  let best = -1
  let bestD2 = -1
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const i = x + nx * (y + ny * z)
        if (occ[i] !== 0) continue
        const cx = (x + 0.5) * cell
        const cy = (y + 0.5) * cell
        const cz = (z + 0.5) * cell
        const dx = mi1(cx - centre[0], box[0])
        const dy = mi1(cy - centre[1], box[1])
        const dz = mi1(cz - centre[2], box[2])
        const d2 = dx * dx + dy * dy + dz * dz
        if (d2 > bestD2) {
          bestD2 = d2
          best = i
        }
      }
    }
  }
  if (best < 0) {
    throw new Error('farthestEmptyCellSeed: сетка полностью занята -- нет ни одной пустой клетки для затравки периодической заливки')
  }
  return best
}

export interface PeriodicCavityResult {
  volume: number
  unreached: number
  totalEmpty: number
  seedCell: number
  seedDistFromCentre: number
  /** sqrt(3)*box/2 (min over axes if the box is not cubic) -- the periodic box's own absolute
   * ceiling on any point's distance from a fixed centre, reported alongside seedDistFromCentre so a
   * reader can judge how close the chosen seed sat to that ceiling (close to it: a normal, safe
   * seed; far below it: a warning sign the object may leave no clean "outside" region at all, see
   * this section's header). */
  theoreticalMaxDist: number
}

/** Periodic counterpart of engine/src/closure.ts's enclosedVolumeFromPositions(): builds the
 * occupancy grid over `idx`'s own members wrapping all three axes, seeds the flood from the single
 * farthest empty cell from `centre` (no outer-radius estimate needed, see above), floods with
 * periodic wraparound, and reports the unreached volume -- the SAME "unreached empty cells * cell^3"
 * definition the existing detector uses, just computed on a periodic grid with a periodic seed
 * instead of an open-boundary one. */
function periodicCavityVolume(positions: Float32Array, idx: readonly number[], box: readonly [number, number, number], centre: readonly [number, number, number], cell: number, radius: number): PeriodicCavityResult {
  const { occ, dims } = occupancyPeriodic(positions, idx, box, cell, radius)
  const seed = farthestEmptyCellSeed(occ, dims, cell, box, centre)
  const visited = floodOutsidePeriodic(occ, dims, [seed])
  let unreached = 0
  let totalEmpty = 0
  for (let i = 0; i < occ.length; i++) {
    if (occ[i] === 0) {
      totalEmpty++
      if (visited[i] === 0) unreached++
    }
  }
  const [nx, ny] = dims
  const z = Math.floor(seed / (nx * ny))
  const rem = seed - z * nx * ny
  const y = Math.floor(rem / nx)
  const x = rem - y * nx
  const cellSide = box[0] / dims[0] // dims/cell relationship is uniform per axis by construction (dimsFor-style ceil)
  const scx = (x + 0.5) * cellSide
  const scy = (y + 0.5) * cellSide
  const scz = (z + 0.5) * (box[2] / dims[2])
  const ddx = mi1(scx - centre[0], box[0])
  const ddy = mi1(scy - centre[1], box[1])
  const ddz = mi1(scz - centre[2], box[2])
  const seedDistFromCentre = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
  const theoreticalMaxDist = (Math.sqrt(3) * Math.min(box[0], box[1], box[2])) / 2
  return { volume: unreached * cell * cell * cell, unreached, totalEmpty, seedCell: seed, seedDistFromCentre, theoreticalMaxDist }
}

// --- Task 4 helper: hole detection via angular coverage deficit, minimum-image directions only ----
//
// Fibonacci-sphere sampling + max-min-angle search, copied verbatim (not re-derived) from the SAME
// method tests/shell-pore-check.test.ts already used and cited there against kappa-tightening-
// report.md SS2.2's own attempt-2 hole search -- the only change here is what feeds it: minimum-
// image DIRECTIONS from a periodic centre (this file), never directions read off an unwrapped frame.

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

function angleDeg(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  const dot = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
  return (Math.acos(dot) * 180) / Math.PI
}

export interface HoleSearchResult {
  holeCentre: [number, number, number]
  holeRadiusDeg: number
  solidAngleFraction: number
}

function holeSearch(headDirs: readonly [number, number, number][], samples: number): HoleSearchResult {
  const candidates = fibonacciSphere(samples)
  let holeCentre = candidates[0]
  let holeRadiusDeg = -1
  for (const c of candidates) {
    let minAngle = Infinity
    for (const h of headDirs) {
      const ang = angleDeg(c, h)
      if (ang < minAngle) minAngle = ang
    }
    if (minAngle > holeRadiusDeg) {
      holeRadiusDeg = minAngle
      holeCentre = c
    }
  }
  const solidAngleFraction = (1 - Math.cos((holeRadiusDeg * Math.PI) / 180)) / 2
  return { holeCentre, holeRadiusDeg, solidAngleFraction }
}

/** Minimum-image unit directions from `centre` to every head member of `idx` whose periodic distance
 * from it stays under trustLimit=min(box)/2 -- see this file's header for why a direction computed
 * past that limit can alias to the wrong periodic image and must not enter a hole search at all,
 * rather than being included with a caveat. Reports how many heads were excluded, since a real
 * aggregate whose own radius approaches box/2 (this task's whole subject) can lose a large share of
 * its own material to this cutoff -- a fact this task reports honestly rather than hides by silently
 * dropping members. */
function headDirectionsWithinTrust(
  idx: readonly number[],
  positions: Float32Array,
  isHead: (i: number) => boolean,
  centre: readonly [number, number, number],
  box: readonly [number, number, number],
): { dirs: [number, number, number][]; excluded: number; total: number } {
  const trustLimit = Math.min(box[0], box[1], box[2]) / 2
  const dirs: [number, number, number][] = []
  let total = 0
  let excluded = 0
  for (const i of idx) {
    if (!isHead(i)) continue
    total++
    const v = miVectorFromCentre([positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2]], centre, box)
    const r = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if (r >= trustLimit || r < 1e-9) {
      excluded++
      continue
    }
    dirs.push([v[0] / r, v[1] / r, v[2] / r])
  }
  return { dirs, excluded, total }
}

// --- synthetic-data plumbing (validation only) -----------------------------------------------------

/** Seeded PRNG (mulberry32) -- same construction already used inline in verify/run.ts's own
 * synthetic-shell scenario, reused here so the validation fixtures below are reproducible rather
 * than Math.random()-based. */
function makeRng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Target area (sigma^2) a random head bead must cover, on its own leaflet's sphere, before that
 * leaflet is trusted to seal at the production occupancy radius (closureRadius=0.6): a randomly
 * (Poisson-)placed disk-covering process leaves a point uncovered with probability
 * exp(-N*pi*radius^2/area) (same vacancy-probability reasoning as buildSolidBlob's own doc comment,
 * here per unit AREA on a 2D sphere instead of per unit VOLUME in a 3D ball). At
 * TARGET_AREA_PER_HEAD_BEAD=0.1, the per-cell vacancy probability is
 * exp(-pi*0.6^2/0.1)=exp(-11.31)~=1.2e-5 -- negligible even summed over the few thousand grid cells
 * one leaflet's own surface covers. A first attempt at this fixture split a single, much smaller,
 * FIXED head count 50/50 between the two leaflets (3000 each) -- far below what either leaflet's own
 * (very different) area needs, and measured a spuriously small cavity (100.75 instead of the
 * expected ~5000+ sigma^3) from exactly the gaps this constant is chosen to rule out; kept as the
 * reasoning trail for why leaflet head counts are DERIVED from each leaflet's own area below, not a
 * single shared constant split evenly. */
const TARGET_AREA_PER_HEAD_BEAD = 0.1

/** A hollow bilayer-shaped shell (inner+outer head shell at rIn/rOut, tail beads filling the
 * annulus between them, w=0 for head / w=1 for tail -- the same convention densityProfileZ/
 * findAmphiphiles use elsewhere, applied directly here since these fixtures have no monomers table)
 * centred at `centre`, optionally with a full-thickness angular hole (both leaflets removed within
 * `holeHalfAngleDeg` of `holeDir`) punched through it -- a literal channel connecting inside to
 * outside, not just a thin gap in one leaflet, so a flood that reaches the hole necessarily leaks.
 * Each leaflet's own head count is derived from its own area (TARGET_AREA_PER_HEAD_BEAD's own doc
 * comment) rather than taken as a single shared parameter, since the inner and outer leaflets of a
 * bilayer this thick relative to its own radius have meaningfully different areas (4*pi*r^2). */
function buildShell(
  rng: () => number,
  centre: readonly [number, number, number],
  rIn: number,
  rOut: number,
  tailCount: number,
  hole?: { dir: [number, number, number]; halfAngleDeg: number },
): Float32Array {
  const randomDir = (): [number, number, number] => {
    const u = rng() * 2 - 1
    const phi = rng() * 2 * Math.PI
    const s = Math.sqrt(Math.max(0, 1 - u * u))
    return [s * Math.cos(phi), s * Math.sin(phi), u]
  }
  const keep = (dir: readonly [number, number, number]): boolean => {
    if (!hole) return true
    return angleDeg(dir, hole.dir) > hole.halfAngleDeg
  }
  const drawDir = (): [number, number, number] => {
    let dir = randomDir()
    while (!keep(dir)) dir = randomDir()
    return dir
  }
  const headCountFor = (r: number) => Math.ceil((4 * Math.PI * r * r) / TARGET_AREA_PER_HEAD_BEAD)
  const out: number[] = []
  for (const r of [rIn, rOut]) {
    const n = headCountFor(r)
    for (let i = 0; i < n; i++) {
      const dir = drawDir()
      out.push(centre[0] + dir[0] * r, centre[1] + dir[1] * r, centre[2] + dir[2] * r, 0)
    }
  }
  for (let i = 0; i < tailCount; i++) {
    const dir = drawDir()
    const r = rIn + rng() * (rOut - rIn)
    out.push(centre[0] + dir[0] * r, centre[1] + dir[1] * r, centre[2] + dir[2] * r, 1)
  }
  return new Float32Array(out)
}

/** A solid, uniformly-filled ball (all beads w=1/tail) -- the "no cavity at all" synthetic case.
 * Density must be high enough that a random (Poisson-like) gap in the interior essentially never
 * occurs at the production cell/radius (closureCell=0.5, closureRadius=0.6): the expected bead count
 * within one occupancy-sealing radius of a random point is lambda = density*(4/3)*pi*radius^3, and
 * P(zero beads within reach) = exp(-lambda) -- at the count/radius picked by this test's own caller
 * (150_000 beads in a radius-15 ball, density~=(3*150000)/(4*pi*15^3)~=10.61/sigma^3),
 * lambda~=10.61*0.9048~=9.6, P(gap)~=exp(-9.6)~=6.8e-5, times ~33500 interior grid cells at cell=0.5
 * gives an expected false-cavity count under 3 cells (~0.4 sigma^3) -- negligible next to the
 * project's own closure threshold (370.8656 sigma^3) this test checks against. An earlier attempt at
 * 40_000 beads (density~=2.83/sigma^3, lambda~=2.56, P(gap)~=7.7%) measured a spurious 911.125
 * sigma^3 "cavity" from exactly this effect, not a bug in the flood itself -- kept as the reasoning
 * trail for why the count below is what it is, not a rounder or smaller number. */
function buildSolidBlob(rng: () => number, centre: readonly [number, number, number], radius: number, count: number): Float32Array {
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    const u = rng() * 2 - 1
    const phi = rng() * 2 * Math.PI
    const s = Math.sqrt(Math.max(0, 1 - u * u))
    const r = radius * Math.cbrt(rng()) // cube-root radius so the FILL is uniform by volume, not biased toward the centre
    out.push(centre[0] + r * s * Math.cos(phi), centre[1] + r * s * Math.sin(phi), centre[2] + r * u, 1)
  }
  return new Float32Array(out)
}

function isHeadW(positions: Float32Array, i: number): boolean {
  return positions[i * 4 + 3] === 0
}

function allIdx(positions: Float32Array): number[] {
  return Array.from({ length: positions.length / 4 }, (_, i) => i)
}

// ==================================================================================================
// SYNTHETIC VALIDATION -- run before anything real is trusted, per this task's own brief.
// ==================================================================================================

const SYN_BOX: [number, number, number] = [50, 50, 50]
const SYN_CENTRE: [number, number, number] = [25, 25, 25]
const BILAYER_THICKNESS = 4.457 // this project's own measured value (data/literature.json's closure gate, verify/out/gates.json) -- reused, not a new number
const SYN_R_IN = 15 - BILAYER_THICKNESS / 2
const SYN_R_OUT = 15 + BILAYER_THICKNESS / 2
// Tail count: generous, not fine-tuned like the head counts above -- the annulus fill is only
// needed for the head:tail radial-profile check, sealing against the flood is provided entirely by
// the two head leaflets (TARGET_AREA_PER_HEAD_BEAD's own doc comment). Scaled off the SAME area-based
// reasoning (total leaflet area / TARGET_AREA_PER_HEAD_BEAD, times 2 for headroom) purely so this
// number tracks SYN_R_IN/SYN_R_OUT if either ever changes, rather than being independently hand-picked.
const SYN_TAIL_COUNT = Math.ceil((2 * (4 * Math.PI * SYN_R_IN ** 2 + 4 * Math.PI * SYN_R_OUT ** 2)) / TARGET_AREA_PER_HEAD_BEAD)
const CLOSURE = loadStageThresholds() // closureCell/closureRadius/enclosedVolume/headDensityBins -- existing thresholds, reused verbatim

test('synthetic: circular centre refuses an axis that is uniformly spread across the box, keeps the other two', () => {
  const rng = makeRng(1)
  const n = 4000
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    // x,y: tight Gaussian-like jitter around (10,40) -- a real, localized object.
    const x = 10 + (rng() - 0.5) * 2
    const y = 40 + (rng() - 0.5) * 2
    // z: uniform across the WHOLE box -- no preferred position at all on this axis.
    const z = rng() * SYN_BOX[2]
    out.push(x, y, z, 1)
  }
  const positions = new Float32Array(out)
  const idx = allIdx(positions)
  const pc = periodicCentre(positions, idx, SYN_BOX, CONCENTRATION_ALPHA)

  console.log(
    `SYN-CONCENTRATION Rx=${pc.axes[0].R.toFixed(4)} Ry=${pc.axes[1].R.toFixed(4)} Rz=${pc.axes[2].R.toFixed(4)} ` +
      `trusted=[${pc.axes.map((a) => a.trusted)}] coords=[${pc.axes.map((a) => a.coordinate?.toFixed(2) ?? 'null')}] alpha=${CONCENTRATION_ALPHA}`,
  )

  expect(pc.axes[0].trusted).toBe(true)
  expect(pc.axes[1].trusted).toBe(true)
  expect(pc.axes[2].trusted).toBe(false) // the uniformly-spread axis must be refused
  expect(pc.axes[2].coordinate).toBeNull()
  expect(pc.axes[0].coordinate).not.toBeNull()
  expect(pc.axes[0].coordinate).toBeCloseTo(10, 0)
  expect(pc.axes[1].coordinate).toBeCloseTo(40, 0)
  expect(pc.centre).toBeNull() // overall centre refused: one bad axis is enough
})

test('synthetic: periodic tools give the SAME reading for a centred shell and the same shell straddling all three periodic faces', () => {
  const rngShared = makeRng(2)
  const centred = buildShell(rngShared, SYN_CENTRE, SYN_R_IN, SYN_R_OUT, SYN_TAIL_COUNT)
  // Wrap every coordinate through a translation that moves the centre from (25,25,25) to (0,0,0) --
  // every axis straddles its own periodic face simultaneously (the hardest case, matching
  // tests/closure.test.ts's own "corner" test convention).
  const straddling = centred.slice()
  for (let i = 0; i < straddling.length; i += 4) {
    straddling[i] = wrap1(straddling[i] - SYN_CENTRE[0], SYN_BOX[0])
    straddling[i + 1] = wrap1(straddling[i + 1] - SYN_CENTRE[1], SYN_BOX[1])
    straddling[i + 2] = wrap1(straddling[i + 2] - SYN_CENTRE[2], SYN_BOX[2])
  }
  const idxC = allIdx(centred)
  const idxS = allIdx(straddling)

  const pcC = periodicCentre(centred, idxC, SYN_BOX, CONCENTRATION_ALPHA)
  const pcS = periodicCentre(straddling, idxS, SYN_BOX, CONCENTRATION_ALPHA)
  expect(pcC.centre).not.toBeNull()
  expect(pcS.centre).not.toBeNull()
  for (let a = 0; a < 3; a++) expect(pcC.axes[a].R).toBeCloseTo(pcS.axes[a].R, 6)

  const profC = periodicRadialProfile(idxC, centred, (i) => isHeadW(centred, i), pcC.centre as [number, number, number], SYN_BOX, CLOSURE.headDensityBins)
  const profS = periodicRadialProfile(idxS, straddling, (i) => isHeadW(straddling, i), pcS.centre as [number, number, number], SYN_BOX, CLOSURE.headDensityBins)
  const sigma = loadParams().sigma
  const peaksC = twoPeaks(
    profC.bins.map((b) => b.headCount),
    profC.bins.map((b) => (b.rLo + b.rHi) / 2),
    sigma,
  )
  const peaksS = twoPeaks(
    profS.bins.map((b) => b.headCount),
    profS.bins.map((b) => (b.rLo + b.rHi) / 2),
    sigma,
  )

  const cavC = periodicCavityVolume(centred, idxC, SYN_BOX, pcC.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)
  const cavS = periodicCavityVolume(straddling, idxS, SYN_BOX, pcS.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)

  console.log(
    `SYN-EQUALITY centred: peaks=[${peaksC.primary.toFixed(3)},${peaksC.secondary?.toFixed(3)}] sep=${peaksC.separation?.toFixed(3)} cavity=${cavC.volume.toFixed(3)} | ` +
      `straddling: peaks=[${peaksS.primary.toFixed(3)},${peaksS.secondary?.toFixed(3)}] sep=${peaksS.separation?.toFixed(3)} cavity=${cavS.volume.toFixed(3)}`,
  )

  // The whole point: a periodic method must not care where in the box the object happens to sit.
  expect(cavS.volume).toBe(cavC.volume)
  expect(peaksS.primary).toBeCloseTo(peaksC.primary, 3)
  expect(peaksS.secondary as number).toBeCloseTo(peaksC.secondary as number, 3)
  expect(peaksS.separation as number).toBeCloseTo(peaksC.separation as number, 3)
  // And it must actually find a real cavity, not report zero for a genuinely closed shell.
  const idealInner = (4 / 3) * Math.PI * SYN_R_IN ** 3
  expect(cavC.volume).toBeGreaterThan(idealInner * 0.6)
  expect(cavC.volume).toBeLessThan(idealInner * 1.3)
  expect(cavC.volume).toBeGreaterThan(CLOSURE.enclosedVolume) // clears the project's own vesicle-closure minimum
})

test('synthetic: a known, full-thickness hole is detected via angular deficit with no unwrap, and the cavity leaks', () => {
  const rng = makeRng(3)
  const holeDir = (() => {
    const v: [number, number, number] = [1, 1, 1]
    const m = Math.sqrt(3)
    return [v[0] / m, v[1] / m, v[2] / m] as [number, number, number]
  })()
  const HOLE_HALF_ANGLE = 30
  const closed = buildShell(rng, SYN_CENTRE, SYN_R_IN, SYN_R_OUT, SYN_TAIL_COUNT)
  const holed = buildShell(makeRng(3), SYN_CENTRE, SYN_R_IN, SYN_R_OUT, SYN_TAIL_COUNT, { dir: holeDir, halfAngleDeg: HOLE_HALF_ANGLE })

  const idxClosed = allIdx(closed)
  const idxHoled = allIdx(holed)
  const pcClosed = periodicCentre(closed, idxClosed, SYN_BOX, CONCENTRATION_ALPHA)
  const pcHoled = periodicCentre(holed, idxHoled, SYN_BOX, CONCENTRATION_ALPHA)
  expect(pcClosed.centre).not.toBeNull()
  expect(pcHoled.centre).not.toBeNull()

  const { dirs: dirsClosed } = headDirectionsWithinTrust(idxClosed, closed, (i) => isHeadW(closed, i), pcClosed.centre as [number, number, number], SYN_BOX)
  const { dirs: dirsHoled, excluded, total } = headDirectionsWithinTrust(idxHoled, holed, (i) => isHeadW(holed, i), pcHoled.centre as [number, number, number], SYN_BOX)
  const holeOnClosed = holeSearch(dirsClosed, FIB_SAMPLES)
  const holeOnHoled = holeSearch(dirsHoled, FIB_SAMPLES)

  const cavClosed = periodicCavityVolume(closed, idxClosed, SYN_BOX, pcClosed.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)
  const cavHoled = periodicCavityVolume(holed, idxHoled, SYN_BOX, pcHoled.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)

  console.log(
    `SYN-HOLE closedShell: holeRadiusDeg=${holeOnClosed.holeRadiusDeg.toFixed(2)} cavity=${cavClosed.volume.toFixed(3)} | ` +
      `holedShell: knownHalfAngle=${HOLE_HALF_ANGLE} foundHoleRadiusDeg=${holeOnHoled.holeRadiusDeg.toFixed(2)} ` +
      `foundHoleCentre=[${holeOnHoled.holeCentre.map((v) => v.toFixed(3))}] angleToKnownDir=${angleDeg(holeOnHoled.holeCentre, holeDir).toFixed(2)}deg ` +
      `cavity=${cavHoled.volume.toFixed(3)} headsExcludedBeyondTrust=${excluded}/${total}`,
  )

  // A truly closed shell should show only the ordinary Poisson-noise-sized gaps a finite random
  // sample leaves, well under a real punched hole.
  expect(holeOnClosed.holeRadiusDeg).toBeLessThan(HOLE_HALF_ANGLE * 0.6)
  // The punched hole must be found close to where it actually is (Fibonacci-sphere resolution at
  // 800 samples plus the finite bead sample means "close", not "exact").
  expect(angleDeg(holeOnHoled.holeCentre, holeDir)).toBeLessThan(15)
  expect(holeOnHoled.holeRadiusDeg).toBeGreaterThan(HOLE_HALF_ANGLE * 0.6)
  expect(holeOnHoled.holeRadiusDeg).toBeLessThan(HOLE_HALF_ANGLE * 1.4)
  // A full-thickness channel this wide must let the flood leak substantially relative to the closed case.
  expect(cavHoled.volume).toBeLessThan(cavClosed.volume * 0.5)
})

test('synthetic: a solid blob with no interior gives no cavity', () => {
  const rng = makeRng(4)
  const blob = buildSolidBlob(rng, SYN_CENTRE, 15, 150_000)
  const idx = allIdx(blob)
  const pc = periodicCentre(blob, idx, SYN_BOX, CONCENTRATION_ALPHA)
  expect(pc.centre).not.toBeNull()
  const cavity = periodicCavityVolume(blob, idx, SYN_BOX, pc.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)
  console.log(`SYN-BLOB cavity=${cavity.volume.toFixed(4)} totalEmpty=${cavity.totalEmpty} unreached=${cavity.unreached} closureThreshold=${CLOSURE.enclosedVolume}`)
  expect(cavity.volume).toBeLessThan(CLOSURE.enclosedVolume / 100) // same "must not be mistaken for a real cavity" bound tests/closure.test.ts's own precedent uses
})

// ==================================================================================================
// REAL DATA -- the last ten dense-campaign checkpoints, same STEPS as shell-pore-report.md for direct
// comparability. No unwrap anywhere below this line either.
// ==================================================================================================

function loadCheckpoint(step: number): { positions: Float32Array; box: [number, number, number] } {
  const path = `data/checkpoints/vesicle-93k-step${step}.json`
  const file = JSON.parse(readFileSync(path, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  if (resume.globalStep !== step) throw new Error(`checkpoint step${step}.json: globalStep=${resume.globalStep} != ${step}`)
  return { positions: resume.positions, box: file.config.box }
}

/** Original indices of the largest aggregate's own members (heads+chains of every recognised
 * amphiphile whose head sits in the largest connected component) -- same recognition pipeline
 * shell-pore-check.test.ts's own findLargest() used (findAmphiphiles + clusterComponents, same
 * cutoff derivation), simplified here since this file never needs the covalent-bond edge list (no
 * unwrap sanity check is performed anywhere in this file -- there is no unwrap to sanity-check). */
function findLargestMembers(positions: Float32Array, bonds: Uint32Array, box: [number, number, number], monomers: Monomer[]): { idx: number[]; amphiphileCount: number; aggregateCount: number } {
  const params = loadParams()
  const amphiphiles = findAmphiphiles(positions, bonds, monomers)
  const memberIdx = new Set<number>()
  for (const a of amphiphiles) {
    memberIdx.add(a.headIndex)
    for (const c of a.chain) memberIdx.add(c)
  }
  const idxArr = Array.from(memberIdx)
  const memberPositions = new Float32Array(idxArr.length * 4)
  for (let k = 0; k < idxArr.length; k++) memberPositions.set(positions.subarray(idxArr[k] * 4, idxArr[k] * 4 + 4), k * 4)
  const memberRadii = monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(params.sigma * Math.max(...memberRadii)) + params.attraction.wc
  const components = clusterComponents(memberPositions, box, cutoff)
  const largestLocal = components[0]
  const aggOriginalIdx = largestLocal.map((k) => idxArr[k])
  const aggSet = new Set(aggOriginalIdx)
  const amphiphileCount = amphiphiles.filter((a) => aggSet.has(a.headIndex)).length
  return { idx: aggOriginalIdx, amphiphileCount, aggregateCount: components.length }
}

function loadBonds(step: number): Uint32Array {
  const path = `data/checkpoints/vesicle-93k-step${step}.json`
  const file = JSON.parse(readFileSync(path, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  const N = file.N
  const edges: number[] = []
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < 3; s++) {
      const p = resume.bondSlots[i * 3 + s]
      if (p !== NONE_U32 && p > i) edges.push(i, p)
    }
  }
  return new Uint32Array(edges)
}

test('periodic re-analysis of the ten dense-campaign checkpoints: circular concentration, head/tail profiles, periodic cavity, hole without unwrapping', () => {
  const soup = loadSoup()
  const monomers: Monomer[] = soup.monomers
  const thresholds = CLOSURE
  const isHeadReal = (positions: Float32Array) => (i: number) => !!monomers[Math.round(positions[i * 4 + 3])]?.polar

  const perStep = STEPS.map((step) => {
    const { positions, box } = loadCheckpoint(step)
    const bonds = loadBonds(step)
    const agg = findLargestMembers(positions, bonds, box, monomers)
    const pc = periodicCentre(positions, agg.idx, box, CONCENTRATION_ALPHA)

    console.log(
      `PERIODIC-CENTRE step=${step} amphiphiles=${agg.amphiphileCount} aggregates=${agg.aggregateCount} particles=${agg.idx.length} ` +
        `Rx=${pc.axes[0].R.toFixed(4)} Ry=${pc.axes[1].R.toFixed(4)} Rz=${pc.axes[2].R.toFixed(4)} ` +
        `trusted=[${pc.axes.map((a) => a.trusted)}] centre=${pc.centre ? `[${pc.centre.map((v) => v.toFixed(3))}]` : 'REFUSED'} alpha=${CONCENTRATION_ALPHA}`,
    )

    if (!pc.centre) {
      // Refused per this file's own rule -- report it and skip the downstream measurements that
      // would need a well-defined centre, rather than silently substituting something else.
      return { step, agg, pc, radial: null, cavity: null, hole: null }
    }
    const centre = pc.centre

    const radial = periodicRadialProfile(agg.idx, positions, isHeadReal(positions), centre, box, thresholds.headDensityBins)
    const sigma = loadParams().sigma
    const peaks = twoPeaks(
      radial.bins.map((b) => b.headCount),
      radial.bins.map((b) => (b.rLo + b.rHi) / 2),
      sigma,
    )
    console.log(
      `PERIODIC-RADIAL step=${step} trustLimit=${radial.trustLimit.toFixed(3)} headTotal=${radial.headTotal} headBeyondTrust=${radial.headBeyondTrust} ` +
        `(${((100 * radial.headBeyondTrust) / radial.headTotal).toFixed(1)}%) tailTotal=${radial.tailTotal} tailBeyondTrust=${radial.tailBeyondTrust} ` +
        `(${((100 * radial.tailBeyondTrust) / radial.tailTotal).toFixed(1)}%) headPeakPrimary=${peaks.primary.toFixed(3)} ` +
        `headPeakSecondary=${peaks.secondary?.toFixed(3) ?? 'none'} separation=${peaks.separation?.toFixed(3) ?? 'none'} bilayerThicknessRef=${BILAYER_THICKNESS}`,
    )
    console.log(
      `PERIODIC-RADIAL-BINS step=${step} headCounts=[${radial.bins.map((b) => b.headCount).join(',')}] tailCounts=[${radial.bins.map((b) => b.tailCount).join(',')}]`,
    )

    const cavity = periodicCavityVolume(positions, agg.idx, box, centre, thresholds.closureCell, thresholds.closureRadius)
    console.log(
      `PERIODIC-CAVITY step=${step} volume=${cavity.volume.toFixed(4)} unreached=${cavity.unreached} totalEmpty=${cavity.totalEmpty} ` +
        `seedDistFromCentre=${cavity.seedDistFromCentre.toFixed(3)} theoreticalMaxDist=${cavity.theoreticalMaxDist.toFixed(3)} closureThreshold=${thresholds.enclosedVolume}`,
    )

    const { dirs, excluded, total } = headDirectionsWithinTrust(agg.idx, positions, isHeadReal(positions), centre, box)
    const hole = holeSearch(dirs, FIB_SAMPLES)
    console.log(
      `PERIODIC-HOLE step=${step} headsUsed=${dirs.length} headsExcludedBeyondTrust=${excluded}/${total} ` +
        `(${((100 * excluded) / total).toFixed(1)}%) holeRadiusDeg=${hole.holeRadiusDeg.toFixed(2)} solidAngleFrac=${(hole.solidAngleFraction * 100).toFixed(2)}% ` +
        `holeCentre=[${hole.holeCentre.map((v) => v.toFixed(3))}]`,
    )

    return { step, agg, pc, radial, peaks, cavity, hole, headsExcluded: excluded, headsTotal: total }
  })

  expect(perStep.length).toBe(10)
  for (const p of perStep) {
    expect(p.agg.amphiphileCount).toBeGreaterThan(1000) // sanity: same ~1282-amphiphile object at every step
    // The periodic centre must be reportable at every one of these checkpoints for the rest of this
    // test to say anything about them -- if a future checkpoint ever refuses, that itself is the
    // finding and this assertion is what would surface it instead of a silent null downstream.
    expect(p.pc.centre).not.toBeNull()
  }

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(
    OUT_FILE,
    JSON.stringify(
      {
        gate: 'periodic-measurement',
        alpha: CONCENTRATION_ALPHA,
        fibonacciSphereSamples: FIB_SAMPLES,
        steps: STEPS,
        perStep: perStep.map((p) => ({
          step: p.step,
          amphiphileCount: p.agg.amphiphileCount,
          aggregateCount: p.agg.aggregateCount,
          particleCount: p.agg.idx.length,
          circularConcentration: { Rx: p.pc.axes[0].R, Ry: p.pc.axes[1].R, Rz: p.pc.axes[2].R, trusted: p.pc.axes.map((a) => a.trusted) },
          centre: p.pc.centre,
          radial: p.radial
            ? {
                trustLimit: p.radial.trustLimit,
                headTotal: p.radial.headTotal,
                headBeyondTrust: p.radial.headBeyondTrust,
                tailTotal: p.radial.tailTotal,
                tailBeyondTrust: p.radial.tailBeyondTrust,
                headPeakPrimary: p.peaks?.primary ?? null,
                headPeakSecondary: p.peaks?.secondary ?? null,
                separation: p.peaks?.separation ?? null,
                bilayerThicknessReference: BILAYER_THICKNESS,
                bins: p.radial.bins,
              }
            : null,
          cavity: p.cavity,
          hole: p.hole ? { ...p.hole, headsUsed: p.headsExcluded !== undefined ? p.headsTotal! - p.headsExcluded! : null, headsExcludedBeyondTrust: p.headsExcluded, headsTotal: p.headsTotal } : null,
        })),
      },
      null,
      2,
    ),
  )
  console.log(`PERIODIC-MEASUREMENT artifact written: ${OUT_FILE}`)
})
