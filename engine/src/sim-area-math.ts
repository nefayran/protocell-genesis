// Pure, GPU-free coordinate map for the area move (Task 6: factored out so it is unit-testable
// without a GPU — see tests/sim.test.ts). Split out of sim.ts by responsibility (file-size rule,
// root CLAUDE.md); moved verbatim, exported name/signature unchanged (tests/sim.test.ts,
// tests/buckling-kappa*.test.ts import scaleLateralRigid directly from engine/src/sim, which
// re-exports it from here).

// Minimum-image displacement of a scalar coordinate difference, matching mi() in forces.wgsl —
// needed to find each lipid's true (unwrapped-relative-to-its-head) offsets when the head itself
// may sit anywhere in [0, box).
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

/** Wraps one coordinate into [0, box) — the scalar form of wrapXY() in sim-layouts.ts. */
function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

/** Pure per-lipid rigid coordinate map for the area move: scales each lipid's lateral (x,y) center
 * of mass by the box's lateral scale factors (newBox/oldBox per axis) and rebuilds its three beads
 * around that new center from their UNCHANGED internal offsets — recovered via minimum-image
 * relative to the head, so a bead that starts wrapped around a periodic boundary from its own head
 * is still measured correctly. Bead coordinates are never individually rescaled, only rebuilt from
 * fixed offsets, so every intramolecular distance (FENE head-tail1, FENE tail1-tail2, bend
 * head-tail2) survives the move exactly. z passes through unchanged (the area move never touches
 * L_z or any bead's z). `positions` is the flat vec4-per-bead layout (x, y, z, type).
 *
 * This is the exact map areaMove() applies to every trial's proposal below — extracted here (not
 * duplicated) so a plain CPU unit test can check its two required properties with no GPU: (1)
 * intramolecular-distance invariance, and (2) that it is an involution under oldBox -> newBox ->
 * oldBox (mapping to newBox and back recovers the exact stored positions, to floating-point noise,
 * as long as no offset exceeds half of either box — true for any physical bond length against a
 * membrane-scale box; see the derivation on AREA_MOVE_LOG_DELTA and the acceptance formula below
 * for why the wrapped-center construction is what makes this hold). */
export function scaleLateralRigid(
  positions: Float32Array,
  oldBox: [number, number, number],
  newBox: [number, number, number],
  lipids: number,
): Float32Array<ArrayBuffer> {
  const sx = newBox[0] / oldBox[0]
  const sy = newBox[1] / oldBox[1]
  const out = new Float32Array(positions.length)
  for (let lip = 0; lip < lipids; lip++) {
    const h = lip * 3, t1 = lip * 3 + 1, t2 = lip * 3 + 2
    const hx = positions[h * 4], hy = positions[h * 4 + 1]
    const ox = [0, mi1(positions[t1 * 4] - hx, oldBox[0]), mi1(positions[t2 * 4] - hx, oldBox[0])]
    const oy = [0, mi1(positions[t1 * 4 + 1] - hy, oldBox[1]), mi1(positions[t2 * 4 + 1] - hy, oldBox[1])]
    const cx = (ox[0] + ox[1] + ox[2]) / 3
    const cy = (oy[0] + oy[1] + oy[2]) / 3
    const comX = wrap1(hx + cx, oldBox[0]) * sx
    const comY = wrap1(hy + cy, oldBox[1]) * sy
    const beads = [h, t1, t2]
    for (let k = 0; k < 3; k++) {
      const b = beads[k]
      out[b * 4] = wrap1(comX + (ox[k] - cx), newBox[0])
      out[b * 4 + 1] = wrap1(comY + (oy[k] - cy), newBox[1])
      out[b * 4 + 2] = positions[b * 4 + 2]
      out[b * 4 + 3] = positions[b * 4 + 3]
    }
  }
  return out
}
