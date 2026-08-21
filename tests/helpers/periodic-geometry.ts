// Periodic-aware shell/cavity measurement machinery for tests/periodic-measurement.test.ts — moved
// verbatim out of that file by responsibility (file-size rule, root CLAUDE.md); no formula changed.
// See that test file's own header for the full rationale (why every measurement here works purely
// from minimum-image displacements, never a global unwrapped frame).

import { fibonacciSphere, angleDeg, mi1 } from './geometry-primitives'
import { clusterComponents } from '../../engine/src/aggregate'

/** Rayleigh's test for circular uniformity (data/periodic-measurement.json's own written basis):
 * the minimum mean-resultant-length R that n points must show, at significance level alpha, before
 * their circular mean is distinguishable from a genuinely uniform (no-preferred-direction) axis.
 * Falls with n (more points make a smaller true concentration detectable) -- unlike a fixed R bar,
 * which this task's own first attempt found refuses even a large, genuinely non-uniform, real
 * membrane object (see the data file's basis for the measured numbers that forced this revision). */
export function rayleighRMin(n: number, alpha: number): number {
  return Math.sqrt(-Math.log(alpha) / n)
}

// --- generic periodic geometry primitives, no unwrap anywhere -------------------------------------

/** Wraps one coordinate into [0, box). */
export function wrap1(v: number, box: number): number {
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
export function circularAxisStat(coordsRaw: readonly number[], boxLen: number, alpha: number): AxisCircularStat {
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
export function periodicCentre(positions: Float32Array, idx: readonly number[], box: [number, number, number], alpha: number): PeriodicCentre {
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
export function miVectorFromCentre(pos: readonly [number, number, number], centre: readonly [number, number, number], box: readonly [number, number, number]): [number, number, number] {
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
export function periodicRadialProfile(
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
export function twoPeaks(counts: readonly number[], centers: readonly number[], minSeparation: number): TwoPeaks {
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

export interface PeriodicOccupancy {
  occ: Uint8Array
  dims: [number, number, number]
}

/** Same rule as engine/src/closure.ts's occupancy() (mark every grid cell whose CENTRE lies within
 * `radius` of some bead in `idx`), generalised to wrap all THREE axes -- that file wraps only x,y (a
 * deliberate choice correct for its own membrane-engine z-open convention, see its own header); the
 * soup engine this task's data comes from wraps x, y AND z, so a periodic flood over it must too. */
export function occupancyPeriodic(positions: Float32Array, idx: readonly number[], box: readonly [number, number, number], cell: number, radius: number): PeriodicOccupancy {
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
export function floodOutsidePeriodic(occ: Uint8Array, dims: readonly [number, number, number], seeds: readonly number[]): Uint8Array {
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
export function periodicCavityVolume(positions: Float32Array, idx: readonly number[], box: readonly [number, number, number], centre: readonly [number, number, number], cell: number, radius: number): PeriodicCavityResult {
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

export interface HoleSearchResult {
  holeCentre: [number, number, number]
  holeRadiusDeg: number
  solidAngleFraction: number
}

export function holeSearch(headDirs: readonly [number, number, number][], samples: number): HoleSearchResult {
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
export function headDirectionsWithinTrust(
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

// --- the wrapping-cluster criterion --------------------------------------------------------------
//
// Moved here VERBATIM from tests/percolation-check.test.ts (task 'confined-parcel', 2026-08-21) --
// not one line of the construction, the axis handling or the return value changed -- because a
// SECOND caller now needs it: tests/soup-confine.test.ts's setup check, which must show that a
// confined run reads 0 of 3 wrapped axes BY CONSTRUCTION. A duplicated percolation criterion is
// exactly how two of them drift apart, and this one is the project's discriminating measurement.

/** True iff the given member positions connect to their own +L image on `axis`: replicate along
 * that axis with the axis made OPEN (so the only way particle i can meet image i is a genuine chain
 * of contacts running the whole length of the box) and test component identity. The two axes that
 * are NOT being tested stay periodic, which is what makes this a test of THIS axis alone. */
export function wrapsOnAxis(member: Float32Array, box: [number, number, number], cutoff: number, axis: 0 | 1 | 2): boolean {
  const n = member.length / 4
  const doubled = new Float32Array(n * 2 * 4)
  doubled.set(member, 0)
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 4; a++) doubled[(n + i) * 4 + a] = member[i * 4 + a]
    doubled[(n + i) * 4 + axis] = member[i * 4 + axis] + box[axis]
  }
  // The replicated axis is given twice the length and NO periodic image of its own, so nothing can
  // join through the far face of the doubled cell; the other two axes keep the box's periodicity.
  const openBox: [number, number, number] = [box[0], box[1], box[2]]
  openBox[axis] = box[axis] * 4 // >= 2x the doubled extent: no minimum-image shortcut on this axis
  const label = new Int32Array(n * 2).fill(-1)
  const comps = clusterComponents(doubled, openBox, cutoff, true)
  comps.forEach((c, k) => c.forEach((i) => (label[i] = k)))
  for (let i = 0; i < n; i++) if (label[i] === label[n + i]) return true
  return false
}
