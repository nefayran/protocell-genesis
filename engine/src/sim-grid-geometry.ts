// Pure, GPU-free neighbor-grid geometry for engine/src/sim.ts — split out by responsibility
// (file-size rule, root CLAUDE.md). Moved verbatim from the original sim.ts's createSystem body;
// the only mechanical change is that the two functions below no longer close over `cellSize`/
// `p.bend.r0` from createSystem's own scope (impossible once they live in a separate module) —
// every call site now passes those same values in explicitly instead. No formula changed.

/** Cell size: at least r_c + w_c (the largest interaction range, the tail-tail attraction's
 * outer cutoff) so any pair within range of each other is guaranteed to fall in the same cell
 * or one of the 26 neighbors. A smaller cell silently drops forces. */
export function computeDims(b: [number, number, number], cellSize: number): [number, number, number] {
  return [Math.max(1, Math.floor(b[0] / cellSize)), Math.max(1, Math.floor(b[1] / cellSize)), Math.max(1, Math.floor(b[2] / cellSize))]
}

// The ±1 neighbor-cell walk in force_main wraps periodically in x,y. With fewer than 3 cells on
// a periodic axis, +1 and -1 land on the same wrapped cell (or, at 1 cell, all three land on
// the cell itself), so that cell gets visited twice (2 cells) or three times (1 cell) per
// particle — every non-bonded force and energy contribution from it is silently doubled or
// tripled. Minimum-image convention (mi() in forces.wgsl) separately assumes each axis sees at
// most one periodic image within range, i.e. box/2 must exceed every interaction's reach in
// that axis — including the bend pair (head-tail2, reach ~r0) which mi() also wraps.
export function gridInvariantsHold(b: [number, number, number], d: [number, number, number], bendR0: number): boolean {
  return d[0] >= 3 && d[1] >= 3 && Math.min(b[0], b[1]) / 2 > bendR0
}
