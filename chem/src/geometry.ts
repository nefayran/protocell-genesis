/**
 * Pure vector-geometry helpers for building molecular coordinates from bond
 * lengths and angles (nanometres, degrees). No model constants live here —
 * every length or angle a caller uses comes from data/molecules.json.
 */

export type Vec3 = [number, number, number]

export function vAdd(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

export function vSub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

export function vScale(a: Vec3, s: number): Vec3 {
  return [a[0] * s, a[1] * s, a[2] * s]
}

export function vDot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

export function vCross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

export function vLength(a: Vec3): number {
  return Math.sqrt(vDot(a, a))
}

export function vNormalize(a: Vec3): Vec3 {
  const len = vLength(a)
  if (len === 0) throw new Error('geometry: normalising a zero vector')
  return vScale(a, 1 / len)
}

export function degToRad(deg: number): number {
  return (deg * Math.PI) / 180
}

/** A point at `length` along the direction `angleDeg` from the +x axis, in the xy plane. */
export function atAngleXY(length: number, angleDeg: number): Vec3 {
  const r = degToRad(angleDeg)
  return [length * Math.cos(r), length * Math.sin(r), 0]
}

/** Rotate `v` around unit-length-normalized `axis` by `angleDeg` (Rodrigues' rotation formula). */
export function rotateAroundAxis(v: Vec3, axis: Vec3, angleDeg: number): Vec3 {
  const k = vNormalize(axis)
  const r = degToRad(angleDeg)
  const cos = Math.cos(r)
  const sin = Math.sin(r)
  const term1 = vScale(v, cos)
  const term2 = vScale(vCross(k, v), sin)
  const term3 = vScale(k, vDot(k, v) * (1 - cos))
  return vAdd(vAdd(term1, term2), term3)
}

/** Any unit vector perpendicular to `v` (`v` need not be unit length, must be non-zero). */
export function perpendicular(v: Vec3): Vec3 {
  const u = vNormalize(v)
  const ref: Vec3 = Math.abs(u[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]
  return vNormalize(vCross(u, ref))
}

/**
 * Planar zig-zag backbone in the xy plane, all-anti conformation: `n` vertices
 * joined by `n - 1` bonds of `bondLength`, with every interior vertex angle
 * equal to `bondAngleDeg` (e.g. the tetrahedral angle for a saturated chain).
 * Consecutive bond vectors alternate reflection across the chain (x) axis —
 * the standard construction for an all-anti backbone. This is what makes the
 * chain's end-to-end span shorter than the sum of its bond lengths.
 */
export function zigzagChain(n: number, bondLength: number, bondAngleDeg: number): Vec3[] {
  if (n <= 0) return []
  // The chain axis bisects the reflex angle at each vertex, so each bond makes
  // angle (90 - bondAngleDeg / 2) with that axis.
  const half = 90 - bondAngleDeg / 2
  const steps: [Vec3, Vec3] = [atAngleXY(bondLength, half), atAngleXY(bondLength, -half)]
  const positions: Vec3[] = [[0, 0, 0]]
  for (let i = 1; i < n; i++) {
    positions.push(vAdd(positions[i - 1], steps[(i - 1) % 2]))
  }
  return positions
}

/**
 * Complete a regular tetrahedron: given two existing unit bond directions
 * `d1`, `d2` already at `angleDeg` to each other, return the other two unit
 * directions so all six pairwise angles equal `angleDeg` (a CH2-type centre
 * with two known ring/chain neighbours and two unknown substituents).
 */
export function tetrahedralPair(d1: Vec3, d2: Vec3, angleDeg: number): [Vec3, Vec3] {
  const theta = degToRad(angleDeg)
  const bIn = vNormalize(vAdd(d1, d2))
  const n = vNormalize(vCross(d1, d2))
  const a = Math.cos(theta) / Math.cos(theta / 2)
  const b = Math.sqrt(Math.max(0, 1 - a * a))
  return [vAdd(vScale(bIn, a), vScale(n, b)), vAdd(vScale(bIn, a), vScale(n, -b))]
}

/**
 * Complete a regular tetrahedron: given one existing unit bond direction
 * `d1`, return the other three unit directions, 120° apart around `d1`, each
 * at `angleDeg` to `d1` and to each other (a CH3-type centre).
 */
export function tetrahedralTriple(d1: Vec3, angleDeg: number): [Vec3, Vec3, Vec3] {
  const theta = degToRad(angleDeg)
  const u = perpendicular(d1)
  const v = vCross(vNormalize(d1), u)
  const out: Vec3[] = []
  for (let k = 0; k < 3; k++) {
    const phi = (2 * Math.PI * k) / 3
    const radial = vAdd(vScale(u, Math.cos(phi)), vScale(v, Math.sin(phi)))
    out.push(vAdd(vScale(d1, Math.cos(theta)), vScale(radial, Math.sin(theta))))
  }
  return out as [Vec3, Vec3, Vec3]
}

/**
 * Trigonal-planar completion: given one existing unit bond direction `d1` and
 * the plane's `normal`, return the other two unit directions, each rotated
 * `angleDeg` from `d1` around `normal` (a sp2 centre with one known
 * substituent and two unknown ones, all three mutually `angleDeg` apart when
 * `angleDeg` is 120°).
 */
export function trigonalPair(d1: Vec3, normal: Vec3, angleDeg: number): [Vec3, Vec3] {
  return [rotateAroundAxis(d1, normal, angleDeg), rotateAroundAxis(d1, normal, -angleDeg)]
}
