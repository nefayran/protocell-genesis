// Task 'explicit-water' (2026-08-18): the closure observable for a vesicle whose interior can now
// genuinely contain solvent instead of vacuum. Counting EMPTY grid cells (engine/src/closure.ts's
// enclosedVolume(), or soup/src/aggregates.ts's localCavityVolume() built on it) is the wrong
// measure once water exists: occupancy() only marks cells within `radius` of an AMPHIPHILE-MEMBER
// particle (soup/src/aggregates.ts's memberIdx -- heads and chain carbons, never water, since water
// never bonds), so a cell full of water sitting inside a closed shell reads as "empty" to that
// detector exactly as a true vacuum would. This file adds the observable the project's own brief
// asks for instead: count water PARTICLES that cannot reach the bulk, via the SAME periodic
// (never-unwrap) flood-fill methodology .superpowers/sdd/2026-08-16-soup-to-vesicle/
// periodic-measurement-report.md just validated (occupancyPeriodic/floodOutsidePeriodic/
// farthestEmptyCellSeed/periodicCavityVolume, tests/periodic-measurement.test.ts) -- ported here as
// real source (not test-only) because a stage-gating observable belongs in soup/src, not in a test
// file. The periodic centre (circular mean per axis, Rayleigh-refused when an axis carries no
// positional information) is reused unchanged from that same report's own design.
//
// engine/src/closure.ts's vacuum-cavity detector is left completely untouched -- every existing
// caller (soup/src/aggregates.ts's localCavityVolume, tests/closure.test.ts,
// tests/soup-aggregates.test.ts) keeps seeing byte-identical behaviour. This module is new,
// additive analysis, exactly the precedent tests/periodic-measurement.test.ts's own header set for
// its non-unwrapping tools ("alongside, not instead of").
//
// No numeric model constant lives in this file (the project-wide rule, tests/params.test.ts's
// literal scanner covers soup/src): `cell`/`radius` are the SAME data/soup.json
// stageThresholds.closureCell/closureRadius every other cavity measurement in this project already
// uses (soup/src/aggregates.ts's localCavityVolume, soup/src/stages.ts), passed in by the caller,
// never hardcoded here; the one new "threshold" this module produces (encapsulationThresholdCount)
// is DERIVED at call time from the existing enclosedVolume minimum (data/soup.json /
// data/literature.json, already rank D and already validated) times a bulk water number density
// MEASURED from the very same flood on the very same snapshot -- never a stored/invented number.

/** Minimum-image displacement of one coordinate on a periodic axis of length `box`. Same convention
 * as engine/src/aggregate.ts's mi1/miAxis and tests/periodic-measurement.test.ts's own mi1,
 * duplicated (not imported) for the same reason that file gives: no dependency on any
 * unwrap-based helper, only on the minimum-image primitive itself. */
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

/** Wraps one coordinate into [0, box). */
function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

// --- periodic (circular-mean) centre, ported from tests/periodic-measurement.test.ts -------------
//
// Reused verbatim (same Rayleigh-test significance level, same formula) rather than re-derived --
// see data/periodic-measurement.json's own written basis for why a fixed R bar was tried first and
// replaced: a fixed bar conflates "this axis carries no positional information" with "this object
// is large relative to the box", and refused even a well-defined real membrane object measured in
// that task. `alpha` is threaded in by the caller (data/periodic-measurement.json's own
// circularConcentration.alpha, the one place that tunable lives) rather than re-declared here.

function rayleighRMin(n: number, alpha: number): number {
  return Math.sqrt(-Math.log(alpha) / n)
}

export interface AxisCircularStat {
  R: number
  coordinate: number | null
  trusted: boolean
}

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
  const trusted = R >= rayleighRMin(n, alpha)
  const coordinate = trusted ? wrap1((angle / (2 * Math.PI)) * boxLen, boxLen) : null
  return { R, coordinate, trusted }
}

/** Periodic centre of `idx` (indices into the flat vec4-per-particle `positions`), one circular
 * mean per axis -- never unwraps anything. `null` when any axis is refused (see
 * circularAxisStat). Callers needing a centre to seed a flood from should treat `null` as "cannot
 * measure this snapshot", not silently fall back to an arithmetic mean (exactly what a coordinate-
 * valued unwrap would be aliasing risk for at this project's own box-spanning scale --
 * periodic-measurement-report.md's whole point). */
export function periodicCentreOf(
  positions: Float32Array,
  idx: readonly number[],
  box: readonly [number, number, number],
  alpha: number,
): [number, number, number] | null {
  const axes = [0, 1, 2].map((axis) => circularAxisStat(idx.map((i) => positions[i * 4 + axis]), box[axis], alpha))
  if (!axes.every((a) => a.trusted)) return null
  return [axes[0].coordinate as number, axes[1].coordinate as number, axes[2].coordinate as number]
}

// --- periodic occupancy + flood, ported from tests/periodic-measurement.test.ts -------------------

interface PeriodicOccupancy {
  occ: Uint8Array
  dims: [number, number, number]
}

/** Marks every grid cell whose centre lies within `radius` of some particle in `idx`, wrapping all
 * three axes (the soup engine has no preferred axis, unlike the membrane engine's own z-open
 * occupancy() in engine/src/closure.ts, which this function deliberately does not call). */
function occupancyPeriodic(
  positions: Float32Array,
  idx: readonly number[],
  box: readonly [number, number, number],
  cell: number,
  radius: number,
): PeriodicOccupancy {
  const dims: [number, number, number] = [
    Math.max(1, Math.ceil(box[0] / cell)),
    Math.max(1, Math.ceil(box[1] / cell)),
    Math.max(1, Math.ceil(box[2] / cell)),
  ]
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
 * engine/src/closure.ts's floodOutside(), deliberately open-boundary there). */
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

/** The single empty cell farthest (by minimum-image distance) from `centre` -- see
 * tests/periodic-measurement.test.ts's own header for why no outer-radius estimate is needed (the
 * periodic box's own farthest point from any fixed centre is always at least as extreme as any
 * face-based seed a non-periodic flood could pick). Throws if the whole grid is occupied. */
function farthestEmptyCellSeed(
  occ: Uint8Array,
  dims: readonly [number, number, number],
  cell: number,
  box: readonly [number, number, number],
  centre: readonly [number, number, number],
): number {
  const [nx, ny, nz] = dims
  let best = -1
  let bestD2 = -1
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        const i = x + nx * (y + ny * z)
        if (occ[i] !== 0) continue
        const dx = mi1((x + 0.5) * cell - centre[0], box[0])
        const dy = mi1((y + 0.5) * cell - centre[1], box[1])
        const dz = mi1((z + 0.5) * cell - centre[2], box[2])
        const d2 = dx * dx + dy * dy + dz * dz
        if (d2 > bestD2) {
          bestD2 = d2
          best = i
        }
      }
    }
  }
  if (best < 0) {
    throw new Error('farthestEmptyCellSeed: сетка полностью занята -- нет пустой клетки для затравки периодической заливки')
  }
  return best
}

// --- the new observable: encapsulated water --------------------------------------------------------

export interface EncapsulatedWaterResult {
  /** Water particles whose own cell is empty (not within `radius` of a membrane-member particle)
   * AND unreached by the periodic flood from the bulk -- water that cannot get to the bulk. This is
   * the observable the vesicle gate should read once water is present (see this module's header). */
  encapsulatedCount: number
  totalWater: number
  /** Bulk (reached, empty) water number density, MEASURED on this same snapshot -- reachedEmptyWater
   * / (reachedEmptyCells * cell^3). Used to derive encapsulationThresholdCount below; also reported
   * on its own so a caller can sanity-check it against the run's own box-average water density. */
  bulkWaterDensity: number
  /** enclosedVolumeMin (data/soup.json stageThresholds.enclosedVolume / data/literature.json's
   * closure.target.min, unchanged, rank D, already validated) times `bulkWaterDensity` above -- the
   * water-bead-COUNT floor equivalent to that volume floor, at THIS snapshot's own measured bulk
   * water density. Derived at call time, never a stored constant (per this task's own instruction:
   * "if you do change a threshold for the water observable, derive it from a measured quantity").
   * NaN when bulkWaterDensity is 0 (no reached empty water at all to measure a density from -- see
   * the field's own caveat below). */
  encapsulationThresholdCount: number
  /** True iff encapsulatedCount clears encapsulationThresholdCount -- the vesicle-closure verdict
   * this observable exists to produce. Always false when encapsulationThresholdCount is NaN (no
   * bulk water to compare against, so no threshold can be derived -- reported as "not closed", not
   * as a crash, since the honest reading of "we cannot measure a bulk density" is "we cannot
   * confirm closure", not "closure confirmed by omission"). */
  closed: boolean
  seedDistFromCentre: number
  theoreticalMaxDist: number
  totalEmptyCells: number
  unreachedCells: number
}

/** Counts water particles trapped inside a membrane structure -- never unwrapping any coordinate,
 * built on the SAME periodic occupancy/flood/seed machinery periodic-measurement-report.md
 * validated for the vacuum-cavity case, applied here to a DIFFERENT question: occupancy is built
 * from `memberIdx` (the membrane material -- amphiphile heads+chain carbons, exactly what
 * soup/src/aggregates.ts's localCavityVolume also floods around), the flood identifies which EMPTY
 * cells the bulk exterior can reach, and every water particle (`waterIdx`) whose own cell is empty
 * and unreached is counted as encapsulated.
 *
 * `centre` must come from periodicCentreOf() above (or an equivalent periodic centre) -- passed in
 * rather than computed here so a caller that already has one (e.g. from a broader per-aggregate
 * analysis) does not pay for it twice, mirroring tests/periodic-measurement.test.ts's own
 * periodicCavityVolume() signature. */
export function encapsulatedWaterVolume(
  positions: Float32Array,
  memberIdx: readonly number[],
  waterIdx: readonly number[],
  box: readonly [number, number, number],
  centre: readonly [number, number, number],
  cell: number,
  radius: number,
  enclosedVolumeMin: number,
): EncapsulatedWaterResult {
  const { occ, dims } = occupancyPeriodic(positions, memberIdx, box, cell, radius)
  const seed = farthestEmptyCellSeed(occ, dims, cell, box, centre)
  const visited = floodOutsidePeriodic(occ, dims, [seed])

  const [nx, ny, nz] = dims
  let totalEmptyCells = 0
  let unreachedCells = 0
  for (let i = 0; i < occ.length; i++) {
    if (occ[i] === 0) {
      totalEmptyCells++
      if (visited[i] === 0) unreachedCells++
    }
  }

  const cellOfPoint = (px: number, py: number, pz: number): number => {
    const wx = wrap1(px, box[0])
    const wy = wrap1(py, box[1])
    const wz = wrap1(pz, box[2])
    const gx = Math.min(nx - 1, Math.floor(wx / cell))
    const gy = Math.min(ny - 1, Math.floor(wy / cell))
    const gz = Math.min(nz - 1, Math.floor(wz / cell))
    return gx + nx * (gy + ny * gz)
  }

  let encapsulatedCount = 0
  let reachedEmptyWater = 0
  for (const i of waterIdx) {
    const c = cellOfPoint(positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2])
    if (occ[c] !== 0) continue // sits inside the membrane's own excluded-volume shell, not solvent space
    if (visited[c] === 0) encapsulatedCount++
    else reachedEmptyWater++
  }

  let reachedEmptyCells = 0
  for (let i = 0; i < occ.length; i++) if (occ[i] === 0 && visited[i] !== 0) reachedEmptyCells++
  const bulkWaterDensity = reachedEmptyCells > 0 ? reachedEmptyWater / (reachedEmptyCells * cell * cell * cell) : 0
  const encapsulationThresholdCount = bulkWaterDensity > 0 ? enclosedVolumeMin * bulkWaterDensity : NaN

  const z = Math.floor(seed / (nx * ny))
  const rem = seed - z * nx * ny
  const y = Math.floor(rem / nx)
  const x = rem - y * nx
  const ddx = mi1((x + 0.5) * cell - centre[0], box[0])
  const ddy = mi1((y + 0.5) * cell - centre[1], box[1])
  const ddz = mi1((z + 0.5) * cell - centre[2], box[2])
  const seedDistFromCentre = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
  const theoreticalMaxDist = (Math.sqrt(3) * Math.min(box[0], box[1], box[2])) / 2

  return {
    encapsulatedCount,
    totalWater: waterIdx.length,
    bulkWaterDensity,
    encapsulationThresholdCount,
    closed: Number.isFinite(encapsulationThresholdCount) && encapsulatedCount > encapsulationThresholdCount,
    seedDistFromCentre,
    theoreticalMaxDist,
    totalEmptyCells,
    unreachedCells,
  }
}
