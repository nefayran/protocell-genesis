// Shared 3D geometry primitives — verified byte-for-byte identical (same name, same body) across
// tests/periodic-measurement.test.ts, tests/rim-lambda-insitu.test.ts (previously inline inside its
// first test) and tests/shell-pore-check.test.ts before this split; unified here per the file-size
// task's own instruction to move genuinely-duplicate in-page helpers to tests/helpers/. Nothing here
// is new: every function is a verbatim move.
//
// mi1's identical formula also appeared as `mi3` in tests/rim-lambda-insitu.test.ts, but that name
// carries a different provenance in that file's own comments (matches soup/wgsl/step.wgsl's mi3(),
// the soup engine's 3-axis convention, as opposed to the membrane engine's mi()/mi1()) — kept
// separate there rather than renamed into this shared import, since unifying by NAME across two
// files that intentionally chose different names would be more than a pure move.

/** Golden-angle (Fibonacci) spiral sampling of the unit sphere — deterministic, near-uniform
 * coverage, no RNG needed. */
export function fibonacciSphere(n: number): [number, number, number][] {
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

/** Angle in degrees between two unit vectors. */
export function angleDeg(a: readonly [number, number, number], b: readonly [number, number, number]): number {
  const dot = Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2]))
  return (Math.acos(dot) * 180) / Math.PI
}

/** Minimum-image displacement of a scalar coordinate difference on a periodic axis of length
 * `box` — matches engine/src/sim.ts's own mi1()/mi() in forces.wgsl. */
export function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}
