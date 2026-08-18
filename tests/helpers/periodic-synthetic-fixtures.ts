// Synthetic validation fixtures for tests/periodic-measurement.test.ts — moved verbatim out of
// that file by responsibility (file-size rule, root CLAUDE.md); validation-only, no new physics.

import { angleDeg } from './geometry-primitives'

/** Seeded PRNG (mulberry32) -- same construction already used inline in verify/run.ts's own
 * synthetic-shell scenario, reused here so the validation fixtures below are reproducible rather
 * than Math.random()-based. */
export function makeRng(seed: number): () => number {
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
export const TARGET_AREA_PER_HEAD_BEAD = 0.1

/** A hollow bilayer-shaped shell (inner+outer head shell at rIn/rOut, tail beads filling the
 * annulus between them, w=0 for head / w=1 for tail -- the same convention densityProfileZ/
 * findAmphiphiles use elsewhere, applied directly here since these fixtures have no monomers table)
 * centred at `centre`, optionally with a full-thickness angular hole (both leaflets removed within
 * `holeHalfAngleDeg` of `holeDir`) punched through it -- a literal channel connecting inside to
 * outside, not just a thin gap in one leaflet, so a flood that reaches the hole necessarily leaks.
 * Each leaflet's own head count is derived from its own area (TARGET_AREA_PER_HEAD_BEAD's own doc
 * comment) rather than taken as a single shared parameter, since the inner and outer leaflets of a
 * bilayer this thick relative to its own radius have meaningfully different areas (4*pi*r^2). */
export function buildShell(
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
export function buildSolidBlob(rng: () => number, centre: readonly [number, number, number], radius: number, count: number): Float32Array {
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

export function isHeadW(positions: Float32Array, i: number): boolean {
  return positions[i * 4 + 3] === 0
}

export function allIdx(positions: Float32Array): number[] {
  return Array.from({ length: positions.length / 4 }, (_, i) => i)
}
