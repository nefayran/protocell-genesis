import { expect, test } from 'vitest'
import periodicParams from '../data/periodic-measurement.json'
import { loadStageThresholds } from '../soup/src/stages'
import { encapsulatedWaterVolume, periodicCentreOf } from '../soup/src/water-closure'

// Task 'explicit-water' (2026-08-18): synthetic validation of the new encapsulated-water observable
// (soup/src/water-closure.ts), run BEFORE anything real is trusted -- same discipline
// tests/periodic-measurement.test.ts's own header set for its periodic flood tools, applied here to
// a different question ("how many WATER PARTICLES are trapped", not "how much empty volume is
// trapped"). Pure CPU/Node, no GPU, no gpuPage() -- these are synthetic beads with a known answer,
// built directly, exactly like periodic-measurement.test.ts's own SYN-* cases.
//
// The membrane shell geometry (box 50, rIn=12.7715/rOut=17.2285, thickness 4.457sigma) is the SAME
// one periodic-measurement.test.ts already validated its cavity-volume tools on -- reused, not
// reinvented, for direct comparability and because it is itself derived from this project's own
// measured bilayer thickness (data/literature.json's closure gate), not an arbitrary shape.

const SYN_BOX: [number, number, number] = [50, 50, 50]
const R_IN = 12.7715
const R_OUT = 17.2285
const ALPHA = periodicParams.circularConcentration.alpha
const CLOSURE = loadStageThresholds() // closureCell/closureRadius/enclosedVolume -- existing, reused verbatim

function makeRng(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

function randomDir(rng: () => number): [number, number, number] {
  const u = rng() * 2 - 1
  const phi = rng() * 2 * Math.PI
  const s = Math.sqrt(Math.max(0, 1 - u * u))
  return [s * Math.cos(phi), s * Math.sin(phi), u]
}

function angleDeg(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  const dot = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
  return (Math.acos(dot) * 180) / Math.PI
}

/** A hollow bilayer-shaped shell (member beads only, kind is irrelevant to this observable) at
 * `centre`, optionally with a full-thickness angular hole -- same fixture shape
 * tests/periodic-measurement.test.ts's own buildShell produces, simplified (no head/tail split,
 * this observable does not care) and rebuilt here rather than imported (test files do not import
 * each other in this project). */
// Volume-uniform radial fill of the [R_IN,R_OUT] annulus: r = cbrt(R_IN^3 + u*(R_OUT^3-R_IN^3)),
// the same cube-root technique fillBall/buildSolidBlob use for a solid ball, generalised to an
// annulus's own lower bound. A first version of this fixture sampled r UNIFORMLY IN r (not in r^3)
// -- that under-fills the OUTER surface relative to the inner one (a shell's volume element grows
// as r^2 dr, so equal-width r-bins near R_OUT cover more real volume per bead than near R_IN),
// measured to leave a leak: with MEMBER_COUNT below, the naive-r fixture traps ZERO water at either
// radius (the flood escapes through the sparse outer surface) where the volume-uniform fill below
// correctly seals it -- kept as the reasoning trail, not silently fixed with no record.
function buildMembers(
  rng: () => number,
  centre: readonly [number, number, number],
  count: number,
  hole?: { dir: [number, number, number]; halfAngleDeg: number },
): number[][] {
  const keep = (dir: readonly [number, number, number]): boolean => !hole || angleDeg(dir, hole.dir) > hole.halfAngleDeg
  const out: number[][] = []
  const rIn3 = R_IN ** 3
  const rOut3 = R_OUT ** 3
  for (let i = 0; i < count; i++) {
    let dir = randomDir(rng)
    while (!keep(dir)) dir = randomDir(rng)
    const r = Math.cbrt(rIn3 + rng() * (rOut3 - rIn3))
    out.push([centre[0] + dir[0] * r, centre[1] + dir[1] * r, centre[2] + dir[2] * r])
  }
  return out
}

/** Member count for buildMembers() above, dense enough that a Poisson gap at closureRadius=0.6
 * (data/soup.json's own already-validated closure cell/radius) is negligible -- SAME volume-density
 * target periodic-measurement.test.ts's own buildSolidBlob uses (150_000 beads in a radius-15 ball,
 * density ~=150000/((4/3)*pi*15^3)~=10.61/sigma^3), applied here to the ANNULUS volume instead of a
 * full ball, since this fixture's own sealing surface is the annulus's own inner+outer boundary, not
 * a filled interior. */
const BLOB_DENSITY = 150_000 / ((4 / 3) * Math.PI * 15 ** 3)
const MEMBER_COUNT = Math.ceil(BLOB_DENSITY * ((4 / 3) * Math.PI * (R_OUT ** 3 - R_IN ** 3)))

/** Water uniformly filling the ball of radius `radius` around `centre` (cube-root radius sampling
 * for a volume-uniform fill, same technique as periodic-measurement.test.ts's own buildSolidBlob). */
function fillBall(rng: () => number, centre: readonly [number, number, number], radius: number, count: number): number[][] {
  const out: number[][] = []
  for (let i = 0; i < count; i++) {
    const dir = randomDir(rng)
    const r = radius * Math.cbrt(rng())
    out.push([centre[0] + dir[0] * r, centre[1] + dir[1] * r, centre[2] + dir[2] * r])
  }
  return out
}

/** Water uniformly filling the box, rejecting any draw within `excludeRadius` (periodic minimum-
 * image distance) of `centre` -- the "definitely outside the shell" bulk fill. */
function fillBoxExcluding(
  rng: () => number,
  box: readonly [number, number, number],
  centre: readonly [number, number, number],
  excludeRadius: number,
  count: number,
): number[][] {
  const out: number[][] = []
  let guard = 0
  while (out.length < count) {
    if (++guard > count * 1000) throw new Error('fillBoxExcluding: too many rejections -- is excludeRadius comparable to box?')
    const p: [number, number, number] = [rng() * box[0], rng() * box[1], rng() * box[2]]
    const dx = mi1(p[0] - centre[0], box[0])
    const dy = mi1(p[1] - centre[1], box[1])
    const dz = mi1(p[2] - centre[2], box[2])
    if (dx * dx + dy * dy + dz * dz >= excludeRadius * excludeRadius) out.push(p)
  }
  return out
}

/** Packs member coords then water coords into one flat vec4-per-particle array (w is unused by
 * water-closure.ts, left at 0), returning the array plus the index ranges for each group. */
function assemble(members: number[][], water: number[][]): { positions: Float32Array; memberIdx: number[]; waterIdx: number[] } {
  const n = members.length + water.length
  const positions = new Float32Array(n * 4)
  const memberIdx: number[] = []
  const waterIdx: number[] = []
  let k = 0
  for (const [x, y, z] of members) {
    positions.set([x, y, z, 0], k * 4)
    memberIdx.push(k)
    k++
  }
  for (const [x, y, z] of water) {
    positions.set([x, y, z, 1], k * 4)
    waterIdx.push(k)
    k++
  }
  return { positions, memberIdx, waterIdx }
}

/** Rigidly translates every coordinate in `positions` by `shift` (per axis) and wraps into
 * [0,box) -- used to build the boundary-straddling twin of a centred fixture, mirroring
 * tests/periodic-measurement.test.ts's own "corner" convention for the SAME equality check. */
function shiftAndWrap(positions: Float32Array, box: readonly [number, number, number], shift: readonly [number, number, number]): Float32Array {
  const n = positions.length / 4
  const out = new Float32Array(positions.length)
  for (let i = 0; i < n; i++) {
    out[i * 4] = wrap1(positions[i * 4] + shift[0], box[0])
    out[i * 4 + 1] = wrap1(positions[i * 4 + 1] + shift[1], box[1])
    out[i * 4 + 2] = wrap1(positions[i * 4 + 2] + shift[2], box[2])
    out[i * 4 + 3] = positions[i * 4 + 3]
  }
  return out
}

test('SYN-EQUALITY: centred vs. boundary-straddling shell give the SAME encapsulated water count', () => {
  const rng = makeRng(2001)
  const centre: [number, number, number] = [25, 25, 25]
  const members = buildMembers(rng, centre, MEMBER_COUNT)
  const innerWater = fillBall(rng, centre, R_IN - 1, 900)
  const outerWater = fillBoxExcluding(rng, SYN_BOX, centre, R_OUT + 1, 6000)
  const { positions: centred, memberIdx, waterIdx } = assemble(members, [...innerWater, ...outerWater])

  const straddling = shiftAndWrap(centred, SYN_BOX, [25, 25, 25]) // moves centre 25->0/box, all three axes at once

  const pcC = periodicCentreOf(centred, memberIdx, SYN_BOX, ALPHA)
  const pcS = periodicCentreOf(straddling, memberIdx, SYN_BOX, ALPHA)
  expect(pcC).not.toBeNull()
  expect(pcS).not.toBeNull()

  const resC = encapsulatedWaterVolume(centred, memberIdx, waterIdx, SYN_BOX, pcC as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius, CLOSURE.enclosedVolume)
  const resS = encapsulatedWaterVolume(straddling, memberIdx, waterIdx, SYN_BOX, pcS as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius, CLOSURE.enclosedVolume)

  console.log(
    `SYN-EQUALITY centred: encapsulated=${resC.encapsulatedCount}/${resC.totalWater} bulkDensity=${resC.bulkWaterDensity.toFixed(5)} threshold=${resC.encapsulationThresholdCount.toFixed(2)} closed=${resC.closed} | ` +
      `straddling: encapsulated=${resS.encapsulatedCount}/${resS.totalWater} bulkDensity=${resS.bulkWaterDensity.toFixed(5)} threshold=${resS.encapsulationThresholdCount.toFixed(2)} closed=${resS.closed}`,
  )

  // The equality standard this task's own brief sets: a periodic method must not care where in the
  // box the object happens to sit.
  expect(resS.encapsulatedCount).toBe(resC.encapsulatedCount)
  expect(resS.totalEmptyCells).toBe(resC.totalEmptyCells)
  expect(resS.unreachedCells).toBe(resC.unreachedCells)
  expect(resS.bulkWaterDensity).toBeCloseTo(resC.bulkWaterDensity, 9)

  // Sanity: most inner water is genuinely trapped (a closed shell), most outer water is genuinely
  // bulk (reachable) -- the equality above would be a vacuous pass if BOTH numbers were 0 or both
  // equalled totalWater.
  expect(resC.encapsulatedCount).toBeGreaterThan(innerWater.length * 0.9)
  expect(resC.encapsulatedCount).toBeLessThan(innerWater.length * 1.1)
  expect(resC.closed).toBe(true)
})

test('SYN-BLOB: a solid membrane blob with no cavity traps zero water', () => {
  const rng = makeRng(2002)
  const centre: [number, number, number] = [25, 25, 25]
  // Solid ball (no hollow interior at all) at the same density scale periodic-measurement.test.ts's
  // own buildSolidBlob uses (dense enough that a Poisson gap at closureRadius=0.6 is negligible --
  // see that file's own comment for the vacancy-probability arithmetic; reused here, not re-derived,
  // since it is a property of the occupancy grid/radius, not of this fixture's shape).
  const memberCoords = fillBall(rng, centre, R_OUT, 150_000)
  const water = fillBoxExcluding(rng, SYN_BOX, centre, R_OUT + 1, 6000)
  const { positions, memberIdx, waterIdx } = assemble(memberCoords, water)
  const pc = periodicCentreOf(positions, memberIdx, SYN_BOX, ALPHA)
  expect(pc).not.toBeNull()
  const res = encapsulatedWaterVolume(positions, memberIdx, waterIdx, SYN_BOX, pc as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius, CLOSURE.enclosedVolume)
  console.log(`SYN-BLOB encapsulated=${res.encapsulatedCount}/${res.totalWater} totalEmpty=${res.totalEmptyCells} unreached=${res.unreachedCells}`)
  expect(res.encapsulatedCount).toBe(0)
  expect(res.closed).toBe(false)
})

test('SYN-HOLE: a full-thickness hole lets encapsulated water escape (zero or near it)', () => {
  const rngClosed = makeRng(2003)
  const centre: [number, number, number] = [25, 25, 25]
  const closedMembers = buildMembers(rngClosed, centre, MEMBER_COUNT)
  const innerWater = fillBall(rngClosed, centre, R_IN - 1, 900)
  const outerWater = fillBoxExcluding(rngClosed, SYN_BOX, centre, R_OUT + 1, 6000)
  const closedAssembled = assemble(closedMembers, [...innerWater, ...outerWater])
  const pcClosed = periodicCentreOf(closedAssembled.positions, closedAssembled.memberIdx, SYN_BOX, ALPHA)
  expect(pcClosed).not.toBeNull()
  const resClosed = encapsulatedWaterVolume(
    closedAssembled.positions,
    closedAssembled.memberIdx,
    closedAssembled.waterIdx,
    SYN_BOX,
    pcClosed as [number, number, number],
    CLOSURE.closureCell,
    CLOSURE.closureRadius,
    CLOSURE.enclosedVolume,
  )

  const rngHoled = makeRng(2003) // SAME seed: the holed case differs only by the hole itself
  const holeDir: [number, number, number] = [1, 1, 1].map((v) => v / Math.sqrt(3)) as [number, number, number]
  const holedMembers = buildMembers(rngHoled, centre, MEMBER_COUNT, { dir: holeDir, halfAngleDeg: 30 })
  const innerWater2 = fillBall(rngHoled, centre, R_IN - 1, 900)
  const outerWater2 = fillBoxExcluding(rngHoled, SYN_BOX, centre, R_OUT + 1, 6000)
  const holedAssembled = assemble(holedMembers, [...innerWater2, ...outerWater2])
  const pcHoled = periodicCentreOf(holedAssembled.positions, holedAssembled.memberIdx, SYN_BOX, ALPHA)
  expect(pcHoled).not.toBeNull()
  const resHoled = encapsulatedWaterVolume(
    holedAssembled.positions,
    holedAssembled.memberIdx,
    holedAssembled.waterIdx,
    SYN_BOX,
    pcHoled as [number, number, number],
    CLOSURE.closureCell,
    CLOSURE.closureRadius,
    CLOSURE.enclosedVolume,
  )

  console.log(
    `SYN-HOLE closedShell: encapsulated=${resClosed.encapsulatedCount}/${resClosed.totalWater} closed=${resClosed.closed} | ` +
      `holedShell: encapsulated=${resHoled.encapsulatedCount}/${resHoled.totalWater} closed=${resHoled.closed}`,
  )

  // The closed twin (same seed, same water placement, no hole) must show genuine encapsulation --
  // otherwise "the hole let water out" would be indistinguishable from "nothing was ever trapped".
  expect(resClosed.encapsulatedCount).toBeGreaterThan(innerWater.length * 0.9)
  // The holed shell leaks: "zero or near it", per this task's own brief -- bounded well below the
  // closed shell's own count, not required to hit exactly 0 (a few water beads can sit in cells the
  // flood's own finite cell size does not connect through a thin/grazing hole).
  expect(resHoled.encapsulatedCount).toBeLessThan(resClosed.encapsulatedCount * 0.05)
  expect(resHoled.closed).toBe(false)
})
