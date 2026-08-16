import forcesWgsl from '../wgsl/forces.wgsl?raw'
import neighborWgsl from '../wgsl/neighbor.wgsl?raw'
import integrateWgsl from '../wgsl/integrate.wgsl?raw'
import { getGpu, readBack, storageBuffer } from './gpu'
import { loadParams, paramsToUniform, wcaCutoff, type Params } from './params'

export type Layout = 'random' | 'bilayer' | 'vesicle'

export interface CreateSystemOpts {
  lipids: number
  box: [number, number, number]
  seed: number
  layout: Layout
  /** Overrides thermostat.gamma from params.json — needed to run the drift test at gamma=0. */
  gamma?: number
  /** Widens the neighbor-grid cell size as if attraction.wc were at least this value, WITHOUT
   * changing the actual physics wc used at creation (that still comes from params.json/gamma
   * override above). Needed by viewer/, whose w_c slider calls System.setLiveParams() after
   * creation to retune a RUNNING system — the grid's cell size is fixed at creation time (see the
   * `cellSize` comment below for why a frozen-too-small grid silently drops forces), so a caller
   * that plans to raise wc above its initial value after the fact must say so up front here. */
  maxWc?: number
  /** Test-only escape hatch: when given, replaces the layout function's output with these exact
   * bead positions (4 floats per bead: x,y,z,type), skipping RNG entirely. `layout` is still
   * required by the type but is not consulted. Lets a test build a fixed, hand-picked
   * configuration (no RNG) to check against an independently-computed reference. */
  positions?: Float32Array
  /** Paired with `positions`: fixed initial velocities (4 floats per bead: vx,vy,vz,0). Defaults
   * to the same Gaussian(kT) initialization used otherwise if omitted. */
  velocities?: Float32Array
}

export interface System {
  /** Advances n Langevin/velocity-Verlet steps. Encoded as one command buffer: no per-step
   * CPU<->GPU round trip. */
  step(n: number): Promise<void>
  /** 4 floats per bead: x, y, z, type (0 = head, 1 = tail). */
  positions(): Promise<Float32Array>
  /** Mean of squared velocity components over all beads (mass = 1) — i.e. kinetic energy PER
   * DEGREE OF FREEDOM, which in these reduced units equals kT directly (not kT/2: equipartition
   * gives <0.5*m*v_x^2> = 0.5*kT per dof, so <v_x^2> = kT). Returning kT/2 here would fail the
   * equipartition test by exactly a factor of two, since it compares this value directly against
   * params.thermostat.kT. */
  kineticEnergyPerDof(): Promise<number>
  /** Kinetic + potential energy of the whole system. */
  totalEnergy(): Promise<number>
  /** Same physics as forces(), full O(N^2) pair loop instead of the neighbor grid — for
   * cross-checking the grid result. */
  forcesBruteForce(): Promise<Float32Array>
  forces(): Promise<Float32Array>
  /** Wall-clock time (ms, GPU-inclusive) of the most recent neighbor-grid rebuild. */
  neighborBuildMs: number
  /** Zero-tension Metropolis Monte Carlo move on the box's lateral area. Proposes s = exp(u),
   * u ~ U(-delta, +delta), multiplies L_x and L_y by sqrt(s) (L_z and every bead's z untouched),
   * and displaces each LIPID rigidly: its center of mass in x,y is scaled by sqrt(s) and its three
   * beads are rebuilt around that scaled center from their unchanged internal offsets, so every
   * intramolecular distance survives the move exactly and DeltaU is purely intermolecular. Accepts
   * with
   * min(1, exp(-(DeltaU - N*kT*ln(A'/A))/kT)) = min(1, exp(N*u - DeltaU/kT)) where N is the number
   * of LIPIDS (the objects whose coordinates are being scaled, hence the count that enters the
   * configurational Jacobian) and DeltaU is the change in total POTENTIAL energy only (kinetic
   * energy is untouched by a positional move, and lateral tension is zero so there is no
   * gamma*DeltaA term). Runs `trials` such moves and returns the accepted fraction. Rebuilds the
   * neighbor grid after every trial (accepted or not) since the box the grid's cell/box uniform
   * refers to may have changed. */
  areaMove(trials: number): Promise<number>
  /** Rewrites the params uniform buffer in place (kT and/or wc), WITHOUT recreating the system or
   * touching any other engine state — the running simulation keeps its positions, velocities,
   * step count and neighbor grid exactly as they were. Only subsequent kick/force/thermostat
   * passes see the new value. Added for viewer/'s two sliders (kT/epsilon and w_c), which must
   * retune a live simulation rather than restart it.
   *
   * Validates and THROWS rather than clamping: wc above the value the neighbor grid was built for
   * (params.attraction.wc, or createSystem's `maxWc` override if given) would silently drop
   * forces beyond cell range -- the exact failure mode `cellSize`'s doc comment above describes,
   * just reached live instead of at construction; kT outside data/params.json's `kTRange` is
   * rejected the same way. A clamp would hide the caller's mistake; the throw names the requested
   * value, the value the grid was actually built for (or the configured range), and the resulting
   * cellSize, so misuse fails at the call site instead of showing up later as wrong physics. */
  setLiveParams(overrides: { kT?: number; wc?: number }): void
  /** Current box lengths — a live snapshot, since areaMove() mutates L_x, L_y in place. */
  readonly box: [number, number, number]
  /** Number of lipids the system was created with (fixed for its lifetime). */
  readonly lipids: number
  /** Cumulative count of integration steps taken via step(), across all calls so far. */
  readonly steps: number
}

// --- seeded RNG for reproducible layouts -------------------------------------------------------

function mulberry32(seed: number): () => number {
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
function wrapXY(pos: number[], box: [number, number, number]): number[] {
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
function layoutRandom(lipids: number, box: [number, number, number], p: Params, rng: () => number): Float32Array {
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
function layoutBilayer(lipids: number, box: [number, number, number], p: Params, rng: () => number): Float32Array {
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
function layoutVesicle(lipids: number, box: [number, number, number], p: Params): Float32Array {
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

// --- pure coordinate map for the area move (Task 6: factored out so it is unit-testable without a
// GPU — see tests/sim.test.ts) --------------------------------------------------------------------

// Minimum-image displacement of a scalar coordinate difference, matching mi() in forces.wgsl —
// needed to find each lipid's true (unwrapped-relative-to-its-head) offsets when the head itself
// may sit anywhere in [0, box).
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

/** Wraps one coordinate into [0, box) — the scalar form of wrapXY() above. */
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
): Float32Array {
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

function initialVelocities(n: number, kT: number, rng: () => number): Float32Array {
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

// --- pipeline cache (one shared GPUDevice, compiled once) ---------------------------------------

interface Pipelines {
  device: GPUDevice
  forceGrid: GPUComputePipeline
  forceBrute: GPUComputePipeline
  clearCounts: GPUComputePipeline
  count: GPUComputePipeline
  prefix: GPUComputePipeline
  fill: GPUComputePipeline
  kick: GPUComputePipeline
  drift: GPUComputePipeline
  wrap: GPUComputePipeline
  thermostat: GPUComputePipeline
}

let cached: Pipelines | undefined

function getPipelines(device: GPUDevice): Pipelines {
  if (cached && cached.device === device) return cached
  const forceModule = device.createShaderModule({ code: forcesWgsl })
  const neighborModule = device.createShaderModule({ code: neighborWgsl })
  const integrateModule = device.createShaderModule({ code: integrateWgsl })
  const cp = (module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })
  cached = {
    device,
    forceGrid: cp(forceModule, 'force_main'),
    forceBrute: cp(forceModule, 'force_brute_main'),
    clearCounts: cp(neighborModule, 'clear_counts_main'),
    count: cp(neighborModule, 'count_main'),
    prefix: cp(neighborModule, 'prefix_main'),
    fill: cp(neighborModule, 'fill_main'),
    kick: cp(integrateModule, 'kick_main'),
    drift: cp(integrateModule, 'drift_main'),
    wrap: cp(integrateModule, 'wrap_main'),
    thermostat: cp(integrateModule, 'thermostat_main'),
  }
  return cached
}

// --- system --------------------------------------------------------------------------------

export async function createSystem(opts: CreateSystemOpts): Promise<System> {
  const base = loadParams()
  const p: Params = opts.gamma === undefined ? base : { ...base, thermostat: { ...base.thermostat, gamma: opts.gamma } }

  const N = opts.lipids * 3
  const box = opts.box
  // Live box, mutated in place by areaMove(); `box` above stays the immutable value opts was
  // called with, used below for the initial cell-size/dims computation and the layout. `cellSize`
  // really is fixed for the system's lifetime (it depends only on params); `dims`/`ncells` are NOT
  // — resizeGrid() recomputes them from the live box after every area-move trial.
  let liveBox: [number, number, number] = [box[0], box[1], box[2]]
  let totalSteps = 0
  const rng = mulberry32(opts.seed)

  const positions0 =
    opts.positions ??
    (opts.layout === 'random'
      ? layoutRandom(opts.lipids, box, p, rng)
      : opts.layout === 'vesicle'
        ? layoutVesicle(opts.lipids, box, p)
        : layoutBilayer(opts.lipids, box, p, rng))
  const velocities0 = opts.velocities ?? initialVelocities(N, p.thermostat.kT, rng)
  const rngState0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) rngState0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9

  // Cell size: at least r_c + w_c (the largest interaction range, the tail-tail attraction's
  // outer cutoff) so any pair within range of each other is guaranteed to fall in the same cell
  // or one of the 26 neighbors. A smaller cell silently drops forces. Fixed for the system's
  // lifetime (it depends only on params, not on the box), but `dims`/`ncells` derived from it are
  // NOT fixed: areaMove() below recomputes and reallocates them whenever the live box crosses a
  // cell-count boundary, which is exactly the "shrinking the box changes the grid" case the brief
  // warns about. Leaving dims frozen at the value computed here would let the box drift far
  // enough that box/dims < cellSize — at that point the ±1 neighbor-cell walk in force_main no
  // longer reaches every pair within the interaction range, WCA repulsion at close range goes
  // silently under-counted, and nothing then resists further compression: measured on this
  // engine, a run that kept dims frozen collapsed area/lipid from 1.352 to 0.697 over 200 area
  // moves instead of equilibrating in the literature corridor.
  // The wc value the neighbor grid was actually sized for — the ceiling setLiveParams below must
  // enforce, since raising wc past this after creation is exactly the silently-dropped-forces
  // failure mode the comment above describes, just triggered live instead of at construction.
  const builtForWc = Math.max(p.attraction.wc, opts.maxWc ?? p.attraction.wc)
  const cellSize = wcaCutoff(p.beadSizes.tail_tail) + builtForWc

  function computeDims(b: [number, number, number]): [number, number, number] {
    return [Math.max(1, Math.floor(b[0] / cellSize)), Math.max(1, Math.floor(b[1] / cellSize)), Math.max(1, Math.floor(b[2] / cellSize))]
  }

  // The ±1 neighbor-cell walk in force_main wraps periodically in x,y. With fewer than 3 cells on
  // a periodic axis, +1 and -1 land on the same wrapped cell (or, at 1 cell, all three land on
  // the cell itself), so that cell gets visited twice (2 cells) or three times (1 cell) per
  // particle — every non-bonded force and energy contribution from it is silently doubled or
  // tripled. Minimum-image convention (mi() in forces.wgsl) separately assumes each axis sees at
  // most one periodic image within range, i.e. box/2 must exceed every interaction's reach in
  // that axis — including the bend pair (head-tail2, reach ~r0) which mi() also wraps.
  function gridInvariantsHold(b: [number, number, number], d: [number, number, number]): boolean {
    return d[0] >= 3 && d[1] >= 3 && Math.min(b[0], b[1]) / 2 > p.bend.r0
  }

  let dims = computeDims(box)
  if (!gridInvariantsHold(box, dims)) {
    throw new Error(
      `сетка соседей: box=[${box[0]},${box[1]},${box[2]}] даёт cellSize=${cellSize.toFixed(4)}, dims=[${dims[0]},${dims[1]},${dims[2]}] ` +
        `и min(box.x,box.y)/2=${(Math.min(box[0], box[1]) / 2).toFixed(4)} — нужно dims>=3 на осях x,y и min(box.x,box.y)/2 > bend.r0=${p.bend.r0}`,
    )
  }
  let ncells = dims[0] * dims[1] * dims[2]

  const { device } = await getGpu()
  const pipe = getPipelines(device)

  const posBuf = storageBuffer(device, positions0)
  const velBuf = storageBuffer(device, velocities0)
  const forceBuf = storageBuffer(device, new Float32Array(N * 4))
  const potentialBuf = storageBuffer(device, new Float32Array(N))
  // Sized N (one slot per bead, sorted by cell), not ncells — independent of the grid resize
  // below, so it is never reallocated.
  const cellsBuf = storageBuffer(device, new Float32Array(N))
  // Sized ncells — reallocated by resizeGrid() whenever dims changes.
  let countsBuf = storageBuffer(device, new Float32Array(ncells))
  let cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
  let cursorBuf = storageBuffer(device, new Float32Array(ncells))
  const rngBuf = device.createBuffer({
    size: rngState0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(rngBuf, 0, rngState0)

  const paramsUniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(paramsUniform, 0, paramsToUniform(p))

  const gridUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  const boxUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })

  function writeGridUniforms(b: [number, number, number], d: [number, number, number]) {
    const bytes = new ArrayBuffer(32)
    new Uint32Array(bytes, 0, 4).set([d[0], d[1], d[2], 0])
    new Float32Array(bytes, 16, 4).set([b[0], b[1], b[2], 0])
    device.queue.writeBuffer(gridUniform, 0, bytes)
    device.queue.writeBuffer(boxUniform, 0, new Float32Array([b[0], b[1], b[2], 0]))
  }

  const bind = (pipeline: GPUComputePipeline, group: number, entries: GPUBindGroupEntry[]) =>
    device.createBindGroup({ layout: pipeline.getBindGroupLayout(group), entries })
  const buf = (b: GPUBuffer) => ({ buffer: b })

  // Bind groups that reference the ncells-sized buffers (countsBuf/cellStartBuf/cursorBuf) — must
  // be rebuilt by resizeGrid() every time those buffers are reallocated, since a WebGPU bind
  // group is a fixed reference to specific buffer objects.
  let clearCountsBind!: GPUBindGroup
  let countBind!: GPUBindGroup
  let prefixBind!: GPUBindGroup
  let fillBind!: GPUBindGroup
  let forceGridGroup1!: GPUBindGroup
  let wgCells = 0

  function rebindGridDependent() {
    clearCountsBind = bind(pipe.clearCounts, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 2, resource: buf(countsBuf) },
    ])
    countBind = bind(pipe.count, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 1, resource: buf(posBuf) },
      { binding: 2, resource: buf(countsBuf) },
    ])
    prefixBind = bind(pipe.prefix, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 2, resource: buf(countsBuf) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cursorBuf) },
    ])
    fillBind = bind(pipe.fill, 0, [
      { binding: 0, resource: buf(gridUniform) },
      { binding: 1, resource: buf(posBuf) },
      { binding: 3, resource: buf(cellsBuf) },
      { binding: 5, resource: buf(cursorBuf) },
    ])
    forceGridGroup1 = bind(pipe.forceGrid, 1, [
      { binding: 0, resource: buf(posBuf) },
      { binding: 1, resource: buf(forceBuf) },
      { binding: 2, resource: buf(potentialBuf) },
      { binding: 3, resource: buf(gridUniform) },
      { binding: 4, resource: buf(cellStartBuf) },
      { binding: 5, resource: buf(cellsBuf) },
    ])
    wgCells = Math.ceil(ncells / 64)
  }

  writeGridUniforms(box, dims)
  rebindGridDependent()

  // Recomputes dims for `newBox`; if the cell count changed, destroys and reallocates the
  // ncells-sized buffers and rebinds everything that references them. Called by areaMove() after
  // every trial (accepted or reverted) so the grid always matches the box actually in use — see
  // the comment on `cellSize` above for why a frozen grid silently breaks under compression.
  async function resizeGrid(newBox: [number, number, number]): Promise<void> {
    const newDims = computeDims(newBox)
    if (newDims[0] !== dims[0] || newDims[1] !== dims[1] || newDims[2] !== dims[2]) {
      countsBuf.destroy()
      cellStartBuf.destroy()
      cursorBuf.destroy()
      dims = newDims
      ncells = dims[0] * dims[1] * dims[2]
      countsBuf = storageBuffer(device, new Float32Array(ncells))
      cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
      cursorBuf = storageBuffer(device, new Float32Array(ncells))
      rebindGridDependent()
    }
    writeGridUniforms(newBox, dims)
  }

  const forceGridGroup0 = bind(pipe.forceGrid, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const forceBruteGroup0 = bind(pipe.forceBrute, 0, [{ binding: 0, resource: buf(paramsUniform) }])
  const forceBruteGroup1 = bind(pipe.forceBrute, 1, [
    { binding: 0, resource: buf(posBuf) },
    { binding: 1, resource: buf(forceBuf) },
    { binding: 2, resource: buf(potentialBuf) },
    { binding: 3, resource: buf(gridUniform) },
  ])

  const kickBind = bind(pipe.kick, 0, [
    { binding: 0, resource: buf(paramsUniform) },
    { binding: 2, resource: buf(velBuf) },
    { binding: 3, resource: buf(forceBuf) },
  ])
  const driftBind = bind(pipe.drift, 0, [
    { binding: 0, resource: buf(paramsUniform) },
    { binding: 1, resource: buf(posBuf) },
    { binding: 2, resource: buf(velBuf) },
  ])
  const wrapBind = bind(pipe.wrap, 0, [
    { binding: 1, resource: buf(posBuf) },
    { binding: 5, resource: buf(boxUniform) },
  ])
  const thermostatBind = bind(pipe.thermostat, 0, [
    { binding: 0, resource: buf(paramsUniform) },
    { binding: 2, resource: buf(velBuf) },
    { binding: 4, resource: buf(rngBuf) },
  ])

  const wgN = Math.ceil(N / 64)

  function encodeGridRebuild(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.clearCounts)
    pass.setBindGroup(0, clearCountsBind)
    pass.dispatchWorkgroups(wgCells)
    pass.setPipeline(pipe.count)
    pass.setBindGroup(0, countBind)
    pass.dispatchWorkgroups(wgN)
    pass.setPipeline(pipe.prefix)
    pass.setBindGroup(0, prefixBind)
    pass.dispatchWorkgroups(1)
    pass.setPipeline(pipe.fill)
    pass.setBindGroup(0, fillBind)
    pass.dispatchWorkgroups(wgN)
  }

  function encodeForceGrid(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.forceGrid)
    pass.setBindGroup(0, forceGridGroup0)
    pass.setBindGroup(1, forceGridGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  function encodeForceBrute(pass: GPUComputePassEncoder) {
    pass.setPipeline(pipe.forceBrute)
    pass.setBindGroup(0, forceBruteGroup0)
    pass.setBindGroup(1, forceBruteGroup1)
    pass.dispatchWorkgroups(wgN)
  }

  let neighborBuildMs = 0

  async function rebuildGridTimed(): Promise<void> {
    const t0 = performance.now()
    await rebuildGridUntimed()
    neighborBuildMs = performance.now() - t0
  }

  /** Same rebuild, without touching `neighborBuildMs`. Used everywhere the rebuild is incidental
   * (warm-up, the two energy evaluations inside every area-move trial) so the number Task 9 reports
   * as a neighbor-grid rebuild time stays the one measured by an explicit, deliberate rebuild
   * instead of whatever the last Monte Carlo trial happened to cost. */
  async function rebuildGridUntimed(): Promise<void> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeGridRebuild(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
  }

  // Warm-up: pipelines are compiled lazily on first dispatch, not at createComputePipeline()
  // time, and that one-time compile latency would otherwise leak into the first measurement
  // (measured: 5.4ms for a 600-particle system built first on the page vs about a fifth of that
  // for a 3000-particle one built after pipelines were already warm). Run one untimed rebuild before
  // the timed one so `neighborBuildMs` reports the rebuild itself, not compilation.
  await rebuildGridUntimed()

  // Initial grid build + force evaluation, timed for Task 9's `neighborBuildMs`, and needed as
  // F(x0) for the first kick of step().
  await rebuildGridTimed()
  {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
  }

  async function step(n: number): Promise<void> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    for (let k = 0; k < n; k++) {
      pass.setPipeline(pipe.kick)
      pass.setBindGroup(0, kickBind)
      pass.dispatchWorkgroups(wgN)

      pass.setPipeline(pipe.drift)
      pass.setBindGroup(0, driftBind)
      pass.dispatchWorkgroups(wgN)

      pass.setPipeline(pipe.wrap)
      pass.setBindGroup(0, wrapBind)
      pass.dispatchWorkgroups(wgN)

      encodeGridRebuild(pass)
      encodeForceGrid(pass)

      pass.setPipeline(pipe.kick)
      pass.setBindGroup(0, kickBind)
      pass.dispatchWorkgroups(wgN)

      pass.setPipeline(pipe.thermostat)
      pass.setBindGroup(0, thermostatBind)
      pass.dispatchWorkgroups(wgN)
    }
    pass.end()
    device.queue.submit([enc.finish()])
    await device.queue.onSubmittedWorkDone()
    totalSteps += n
  }

  async function positions(): Promise<Float32Array> {
    return readBack(device, posBuf, N * 16)
  }

  // Mutable copy of the params actually in force, distinct from the immutable `p` above (which
  // still reflects exactly what the system was CREATED with) — see setLiveParams's doc comment on
  // the System interface for why this exists.
  let livep: Params = p

  function setLiveParams(overrides: { kT?: number; wc?: number }): void {
    // Validate loudly, before touching anything, rather than clamping: a silent clamp hides a
    // caller's mistake exactly the way a too-small grid hides one -- the failure this method
    // exists to prevent must fail AT THE CALL, not show up later as wrong physics. Same style as
    // createSystem's own gridInvariantsHold/discriminant guards.
    if (overrides.wc !== undefined && overrides.wc > builtForWc) {
      throw new Error(
        `setLiveParams: wc=${overrides.wc} превышает значение, под которое построена сетка соседей ` +
          `(builtForWc=${builtForWc}, cellSize=${cellSize.toFixed(4)}) — пересоздайте систему с ` +
          `опцией maxWc>=${overrides.wc}, иначе силы будут молча теряться`,
      )
    }
    if (overrides.kT !== undefined && (overrides.kT < p.kTRange[0] || overrides.kT > p.kTRange[1])) {
      throw new Error(
        `setLiveParams: kT=${overrides.kT} вне data/params.json kTRange=[${p.kTRange[0]},${p.kTRange[1]}]`,
      )
    }
    livep = {
      ...livep,
      thermostat: overrides.kT !== undefined ? { ...livep.thermostat, kT: overrides.kT } : livep.thermostat,
      attraction: overrides.wc !== undefined ? { ...livep.attraction, wc: overrides.wc } : livep.attraction,
    }
    device.queue.writeBuffer(paramsUniform, 0, paramsToUniform(livep))
  }

  async function forces(): Promise<Float32Array> {
    await rebuildGridTimed()
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    return readBack(device, forceBuf, N * 16)
  }

  async function forcesBruteForce(): Promise<Float32Array> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceBrute(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    return readBack(device, forceBuf, N * 16)
  }

  async function kineticEnergyPerDof(): Promise<number> {
    const v = await readBack(device, velBuf, N * 16)
    let sumSq = 0
    for (let i = 0; i < N; i++) {
      const vx = v[i * 4], vy = v[i * 4 + 1], vz = v[i * 4 + 2]
      sumSq += vx * vx + vy * vy + vz * vz
    }
    return sumSq / (3 * N)
  }

  async function totalEnergy(): Promise<number> {
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeGridRebuild(pass)
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    const [v, u] = await Promise.all([readBack(device, velBuf, N * 16), readBack(device, potentialBuf, N * 4)])
    let kinetic = 0
    for (let i = 0; i < N; i++) {
      const vx = v[i * 4], vy = v[i * 4 + 1], vz = v[i * 4 + 2]
      kinetic += 0.5 * (vx * vx + vy * vy + vz * vz)
    }
    let potential = 0
    for (let i = 0; i < N; i++) potential += u[i]
    return kinetic + potential
  }

  // Proposal half-width for the area move's log-area step u ~ U(-AREA_MOVE_LOG_DELTA,
  // +AREA_MOVE_LOG_DELTA). This is a Monte Carlo move-size tuning knob, not a physical model
  // parameter (it does not appear in any Cooke & Deserno formula and has no effect on the
  // equilibrium distribution, only on how fast the chain explores it), so it lives here rather
  // than in data/params.json. Tuned (see task-5-report.md) so the accepted fraction lands in the
  // 0.2-0.6 range: measured 0.54 / 0.36 / 0.23 at half-widths 0.006 / 0.010 / 0.016 once the
  // rigid-lipid displacement below became exact. Earlier, much smaller values were forced by the
  // periodic-image defect in that displacement, which put a spurious quadratic cost on every
  // proposal and pushed acceptance under 0.2 for anything but a tiny step; with the defect fixed a
  // step three to four times larger is accepted at the same rate, and the chain reaches its area
  // plateau in ~150 moves (before the fix it never reached one — it slid out of the corridor).
  const AREA_MOVE_LOG_DELTA = 0.012

  async function totalPotentialGPU(): Promise<number> {
    await rebuildGridUntimed()
    const enc = device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceGrid(pass)
    pass.end()
    device.queue.submit([enc.finish()])
    const u = await readBack(device, potentialBuf, N * 4)
    let sum = 0
    for (let i = 0; i < N; i++) sum += u[i]
    return sum
  }

  async function areaMove(trials: number): Promise<number> {
    let accepted = 0
    for (let t = 0; t < trials; t++) {
      const oldBox: [number, number, number] = [liveBox[0], liveBox[1], liveBox[2]]

      const u = (rng() * 2 - 1) * AREA_MOVE_LOG_DELTA
      const sq = Math.sqrt(Math.exp(u))
      const proposedBox: [number, number, number] = [oldBox[0] * sq, oldBox[1] * sq, oldBox[2]]

      // Reject up front, with no GPU work at all, if the proposal would breach either grid
      // invariant createSystem() enforces at construction time (dims>=3 on the periodic axes,
      // min(box.x,box.y)/2 > bend.r0). AREA_MOVE_LOG_DELTA is tuned small enough that this should
      // essentially never fire once the chain is anywhere near the literature corridor — it is a
      // safety net, not the normal path.
      if (!gridInvariantsHold(proposedBox, computeDims(proposedBox))) continue

      const potentialBefore = await totalPotentialGPU()
      const before = await readBack(device, posBuf, N * 16)

      // Rigid lipid displacement: scale each LIPID's center of mass (x,y) by sqrt(s) and REBUILD its
      // three beads around the scaled center from their unchanged internal offsets. Bead coordinates
      // are never rescaled themselves, so every intramolecular distance (FENE head-tail1, FENE
      // tail1-tail2, bend head-tail2) survives the move exactly and ΔU below reflects only
      // intermolecular structure — which is what the area coordinate is supposed to couple to.
      //
      // Both the center of mass and the offsets must be handled in the periodic sense, and BOTH
      // matter (each of the two was measured to break the move on its own):
      //  - the center is built from the head coordinate plus minimum-image offsets (mi1) and then
      //    WRAPPED into [0, oldBox). Only a wrapped center makes this move the identity in scaled
      //    coordinates s = R/L (s stays in [0,1), so s' = sqrt(s)R/(sqrt(s)L) = s), which is what
      //    makes the configurational Jacobian exactly (A'/A)^N_lipids and the move an involution
      //    under u -> -u. An unwrapped center displaces its lipid by up to L*(sqrt(s)-1) away from
      //    the homogeneous value every other lipid gets.
      //  - each bead is then placed at comScaled + offset rather than at storedCoordinate +
      //    comX*(sqrt(s)-1). Those two differ by exactly L*(sqrt(s)-1) ~ 0.04 sigma for any bead
      //    whose stored coordinate sits on the far side of a periodic boundary from its own center
      //    (about a tenth of the lipids at these box sizes): translating the stored coordinate and
      //    re-wrapping lands such a bead one box length off its lipid, i.e. STRETCHES or COMPRESSES
      //    that bond by 0.04 sigma. The measured damage was severe and not obvious: the bond
      //    perturbation is quadratic in u and one-signed, which showed up as a quenched curvature of
      //    d2U/dlnA2 of order a million (so a u = 0.003 proposal cost ~13 kT whichever direction it
      //    went), a per-configuration scatter of +/-1500 in dU/dlnA, and a Metropolis chain that
      //    walked the area monotonically out of the literature corridor while every aggregate
      //    diagnostic (acceptance fraction, energy conservation) looked healthy.
      // Explicit re-wrap into [0, newBox) inside scaleLateralRigid() mirrors wrap_main in
      // integrate.wgsl — so a bead landing just past the edge sits in the cell cell_coord()
      // actually expects (it clamps rather than wraps) instead of waiting for the next step()'s
      // wrap pass.
      const proposed = scaleLateralRigid(before, oldBox, proposedBox, opts.lipids)
      device.queue.writeBuffer(posBuf, 0, proposed)
      liveBox = proposedBox
      await resizeGrid(proposedBox)

      const potentialAfter = await totalPotentialGPU()
      const dU = potentialAfter - potentialBefore
      // Zero-tension Metropolis with the configurational Jacobian from rescaling N=lipids
      // centers of mass by sqrt(A'/A): accept with min(1, exp(-(dU - lipids*kT*ln(A'/A))/kT)) =
      // min(1, exp(lipids*u - dU/kT)), since ln(A'/A) = ln(s) = u exactly. The entropic term
      // favors expansion (positive u) and is what balances the tail-tail attraction's pull
      // toward smaller area — see task-5-report.md for the measurement that showed this term is
      // required (its absence produces a monotonic collapse with a perfectly healthy aggregate
      // acceptance fraction, not a near-zero one).
      const acceptProb = Math.min(1, Math.exp(opts.lipids * u - dU / p.thermostat.kT))
      if (rng() < acceptProb) {
        accepted++
        // Proposed state kept; forceBuf/potentialBuf/grid already reflect it from the
        // totalPotentialGPU() call above — nothing left to resync.
      } else {
        device.queue.writeBuffer(posBuf, 0, before)
        liveBox = oldBox
        await resizeGrid(oldBox)
        // Resync forceBuf/potentialBuf/grid with the reverted positions/box so the next step()
        // kicks off of F(x) at the state actually being kept, not the discarded proposal.
        await totalPotentialGPU()
      }
    }
    return accepted / trials
  }

  return {
    step,
    positions,
    kineticEnergyPerDof,
    totalEnergy,
    areaMove,
    setLiveParams,
    get box(): [number, number, number] {
      return [liveBox[0], liveBox[1], liveBox[2]]
    },
    lipids: opts.lipids,
    get steps() {
      return totalSteps
    },
    forces,
    forcesBruteForce,
    get neighborBuildMs() {
      return neighborBuildMs
    },
  }
}
