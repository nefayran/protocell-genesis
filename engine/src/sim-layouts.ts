// Initial-configuration layouts for engine/src/sim.ts's createSystem — split out by responsibility
// (file-size rule, root CLAUDE.md). Seeded RNG + the three layout functions (random/bilayer/vesicle)
// + initial-velocity sampling. Moved verbatim from the original sim.ts; no numeric constant changed.

import type { Params } from './params'

// --- seeded RNG for reproducible layouts -------------------------------------------------------

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-7)
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

/** Wraps x,y into [0, box) — z is left untouched (open boundary). Every bead position placed by
 * a layout function must go through this: the neighbor grid's "any pair within cutoff shares a
 * cell or a 3x3x3 neighbor" guarantee assumes coordinates already sit inside [0, box); a bead
 * placed just past the edge (e.g. a tail extending past x=box due to its lipid's orientation)
 * gets clamped into the wrong boundary cell instead of wrapping, and the grid then silently
 * misses that pair while the brute-force loop (which uses mi() directly, not cell membership)
 * still finds it — measured as a real ~0.2 force mismatch in the grid-vs-brute-force test, not a
 * float32 precision artifact. */
export function wrapXY(pos: number[], box: [number, number, number]): number[] {
  const x = pos[0] - Math.floor(pos[0] / box[0]) * box[0]
  const y = pos[1] - Math.floor(pos[1] / box[1]) * box[1]
  return [x, y, pos[2]]
}

// --- layouts -------------------------------------------------------------------------------

/** Random lipid centers with random orientation; bond lengths taken from params so nothing here
 * is a bare model constant. Placement is rejection-sampled against a minimum bead separation
 * (the largest WCA bead radius in params): pure IID-uniform placement of ~1800 beads in a modest
 * box reliably lands some pair almost exactly on top of each other, and the WCA force there is
 * unbounded (~1/r^13) — the resulting ~1e9-magnitude force makes the grid-vs-brute-force
 * comparison fail on float32 summation-order noise alone, well before any real neighbor-list bug.
 * Distance checks are periodic-aware in x,y to match the physics (mi() in forces.wgsl). */
export function layoutRandom(lipids: number, box: [number, number, number], p: Params, rng: () => number): Float32Array {
  const out = new Float32Array(lipids * 3 * 4)
  const placed: number[][] = []
  const minSep = Math.max(p.beadSizes.head_head, p.beadSizes.head_tail, p.beadSizes.tail_tail)
  const minSep2 = minSep * minSep
  const maxAttempts = 200

  function periodicDist2(a: number[], b: number[]): number {
    let dx = a[0] - b[0]
    let dy = a[1] - b[1]
    const dz = a[2] - b[2]
    dx -= Math.round(dx / box[0]) * box[0]
    dy -= Math.round(dy / box[1]) * box[1]
    return dx * dx + dy * dy + dz * dz
  }

  for (let k = 0; k < lipids; k++) {
    let head: number[] = [], tail1: number[] = [], tail2: number[] = []
    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      const cx = rng() * box[0]
      const cy = rng() * box[1]
      const cz = rng() * box[2]
      let ox = gaussian(rng)
      let oy = gaussian(rng)
      let oz = gaussian(rng)
      const len = Math.sqrt(ox * ox + oy * oy + oz * oz) || 1
      ox /= len
      oy /= len
      oz /= len
      head = wrapXY([cx, cy, cz], box)
      tail1 = wrapXY(
        [cx + ox * p.beadSizes.head_tail, cy + oy * p.beadSizes.head_tail, cz + oz * p.beadSizes.head_tail],
        box,
      )
      tail2 = wrapXY(
        [
          tail1[0] + ox * p.beadSizes.tail_tail,
          tail1[1] + oy * p.beadSizes.tail_tail,
          tail1[2] + oz * p.beadSizes.tail_tail,
        ],
        box,
      )
      const clash = placed.some(
        (b) => periodicDist2(head, b) < minSep2 || periodicDist2(tail1, b) < minSep2 || periodicDist2(tail2, b) < minSep2,
      )
      if (!clash || attempt === maxAttempts - 1) break
    }
    placed.push(head, tail1, tail2)
    const base = k * 12
    out.set([...head, 0], base)
    out.set([...tail1, 1], base + 4)
    out.set([...tail2, 1], base + 8)
  }
  return out
}

/** Two leaflets in the xy plane, heads pointing outward and tails toward the midplane. Grid
 * spacing and leaflet gap are derived from box size and bead sizes (never a bare literal) — good
 * enough for a stable starting configuration for the energy-drift test; hitting the accepted
 * area-per-lipid corridor is Task 5's job, not this one. */
export function layoutBilayer(lipids: number, box: [number, number, number], p: Params, rng: () => number): Float32Array {
  const out = new Float32Array(lipids * 3 * 4)
  const perLeaflet = Math.ceil(lipids / 2)
  const nSide = Math.max(1, Math.ceil(Math.sqrt(perLeaflet)))
  const spacingX = box[0] / nSide
  const spacingY = box[1] / nSide
  const midZ = box[2] / 2
  const gap = p.beadSizes.tail_tail / 2
  const jitter = Math.min(spacingX, spacingY) * 0.1

  for (let k = 0; k < lipids; k++) {
    const leaflet = k < perLeaflet ? 1 : -1 // +1 top (heads at +z), -1 bottom (heads at -z)
    const idx = k < perLeaflet ? k : k - perLeaflet
    const a = idx % nSide
    const b = Math.floor(idx / nSide)
    const xRaw = (a + 0.5) * spacingX + (rng() - 0.5) * jitter
    const yRaw = (b + 0.5) * spacingY + (rng() - 0.5) * jitter
    const [x, y] = wrapXY([xRaw, yRaw, 0], box)

    const zTail2 = midZ + leaflet * gap
    const zTail1 = zTail2 + leaflet * p.beadSizes.tail_tail
    const zHead = zTail1 + leaflet * p.beadSizes.head_tail

    const base = k * 12
    out.set([x, y, zHead, 0], base)
    out.set([x, y, zTail1, 1], base + 4)
    out.set([x, y, zTail2, 1], base + 8)
  }
  return out
}

// --- Task 8: vesicle layout ------------------------------------------------------------------

// Target area per lipid for the OUTER leaflet's head shell (see layoutVesicle below) -- the value
// this engine's own self-assembly settled on for a flat bilayer (task-6-report.md: largest-
// cluster fraction 0.9975 at 1e6 steps, area/lipid ~1.208 sigma^2), not a Cooke & Deserno model
// constant, so it lives here next to AREA_MOVE_LOG_DELTA rather than in data/params.json -- it
// tunes a STARTING configuration for Task 8's closure detector only; the story's own vesicle
// comes from self-assembly (Task 6), not from this layout.
const VESICLE_AREA_PER_LIPID = 1.208

/** Two concentric spherical shells: heads outward on the outer leaflet, heads inward on the
 * inner leaflet, tails meeting near the midpoint between the shells -- the spherical analogue of
 * layoutBilayer's planar construction below (same radial offsets, moving away from a mid-radius
 * instead of a mid-plane: halfGap, then tail_tail, then head_tail).
 *
 * Radius: let L = halfGap + tail_tail + head_tail (mid-to-head radial offset) and
 * K = lipids * VESICLE_AREA_PER_LIPID / (4*pi). The two head-shell radii are R = midR + L (outer)
 * and midR - L (inner); requiring the OUTER shell's area per lipid to equal
 * VESICLE_AREA_PER_LIPID and splitting the lipid count between leaflets in proportion to their
 * head-shell areas (R_out^2 : R_in^2, per the brief -- not equally) together reduce to
 * R_out^2 + R_in^2 = K, i.e. a quadratic in R_out whose positive root is
 * R_out = L + sqrt(K/2 - L^2) (task-8-report.md carries the full derivation). One consequence of
 * splitting by shell area: the INNER leaflet lands on the same area per lipid automatically
 * (N_in/R_in^2 = N_out/R_out^2 = 4*pi/VESICLE_AREA_PER_LIPID by construction).
 *
 * Points on each shell come from a golden-angle (Fibonacci) spiral: deterministic, no rejection
 * sampling, and no pole clustering -- reproducible by construction, independent of `rng` (kept in
 * the signature only so this layout matches layoutRandom/layoutBilayer's call shape). */
export function layoutVesicle(lipids: number, box: [number, number, number], p: Params): Float32Array {
  const halfGap = p.beadSizes.tail_tail / 2
  const L = halfGap + p.beadSizes.tail_tail + p.beadSizes.head_tail
  const K = (lipids * VESICLE_AREA_PER_LIPID) / (4 * Math.PI)
  const discriminant = K / 2 - L * L
  if (discriminant <= 0) {
    throw new Error(
      `layoutVesicle: lipids=${lipids} слишком мало для двухслойной сферы ` +
        `(K/2-L^2=${discriminant.toFixed(4)} <= 0) — нужно больше липидов`,
    )
  }
  const rHeadOut = L + Math.sqrt(discriminant)
  const midR = rHeadOut - L
  const rHeadIn = midR - L
  if (rHeadIn <= 0) {
    throw new Error(`layoutVesicle: внутренний радиус головного слоя ${rHeadIn.toFixed(4)} <= 0 — нужно больше липидов`)
  }
  const maxReach = Math.min(box[0], box[1], box[2]) / 2
  if (rHeadOut >= maxReach) {
    throw new Error(
      `layoutVesicle: внешний радиус ${rHeadOut.toFixed(4)} не помещается в box=[${box[0]},${box[1]},${box[2]}] ` +
        `с центром в середине (нужно min(box)/2 > R)`,
    )
  }
  const center: [number, number, number] = [box[0] / 2, box[1] / 2, box[2] / 2]

  const nOut = Math.round((lipids * rHeadOut * rHeadOut) / (rHeadOut * rHeadOut + rHeadIn * rHeadIn))
  const nIn = lipids - nOut

  const out = new Float32Array(lipids * 3 * 4)
  const goldenAngle = Math.PI * (3 - Math.sqrt(5))

  function place(count: number, sign: 1 | -1, offset: number) {
    const rTail2 = midR + sign * halfGap
    const rTail1 = rTail2 + sign * p.beadSizes.tail_tail
    const rHead = rTail1 + sign * p.beadSizes.head_tail
    for (let i = 0; i < count; i++) {
      // Golden-angle spiral on the unit sphere (Marsaglia/Fibonacci construction): near-uniform
      // coverage, deterministic, no RNG needed.
      const y = count > 1 ? 1 - (2 * i) / (count - 1) : 0
      const radial = Math.sqrt(Math.max(0, 1 - y * y))
      const theta = goldenAngle * i
      const dir: [number, number, number] = [Math.cos(theta) * radial, y, Math.sin(theta) * radial]

      const head = wrapXY([center[0] + dir[0] * rHead, center[1] + dir[1] * rHead, center[2] + dir[2] * rHead], box)
      const tail1 = wrapXY(
        [center[0] + dir[0] * rTail1, center[1] + dir[1] * rTail1, center[2] + dir[2] * rTail1],
        box,
      )
      const tail2 = wrapXY(
        [center[0] + dir[0] * rTail2, center[1] + dir[1] * rTail2, center[2] + dir[2] * rTail2],
        box,
      )
      const base = (offset + i) * 12
      out.set([...head, 0], base)
      out.set([...tail1, 1], base + 4)
      out.set([...tail2, 1], base + 8)
    }
  }

  place(nOut, 1, 0)
  place(nIn, -1, nOut)
  return out
}

export function initialVelocities(n: number, kT: number, rng: () => number): Float32Array {
  const out = new Float32Array(n * 4)
  const s = Math.sqrt(kT)
  for (let i = 0; i < n; i++) {
    out[i * 4 + 0] = gaussian(rng) * s
    out[i * 4 + 1] = gaussian(rng) * s
    out[i * 4 + 2] = gaussian(rng) * s
    out[i * 4 + 3] = 0
  }
  return out
}
