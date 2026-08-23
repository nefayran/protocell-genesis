// Task 'acid-soap-pairing' (2026-08-23): the acid-soap PAIR'S IDENTITY, and the survival statistic a
// pair LIFETIME is read off. Its own module rather than an addition to soup/src/electrostatics.ts
// (581 lines against CLAUDE.md's hard 600 -- the rule is "split first, then add", and nothing here
// needs splitting out of that file to make room), and rather than a copy inside a test: the rule
// "one protonated head with one deprotonated head, nearest, inside the contact radius" now exists in
// exactly three places that must agree -- soup/wgsl/pair.wgsl's acidSoapScale() XOR (the force),
// soup/src/electrostatics.ts's pairingStats' `paired` count (the population), and this file (the
// identity). Moved here verbatim from tests/electrostatics-audit.test.ts, which now imports it, so
// the audit's published pairSurvivalFraction and this task's lifetime are the same measurement.
//
// Pure, GPU-free and array-in/array-out, so `window.api` can call it INSIDE the page and reduce the
// result to scalars before anything crosses the CDP boundary -- the same reason soup/src/
// soup-confine.ts's wallStats is shaped that way.
//
// No model constant is written here: `contact` is the head-head WCA contact distance the caller
// derives from data/params.json's rank-A beadSizes.

/** Each head's nearest UNLIKE-charge head neighbour inside `contact`, or absent. The pair's identity,
 * so a lifetime can be measured by asking how many pairs survive to the next checkpoint. */
export function nearestUnlike(
  pos: Float32Array,
  charges: Float32Array,
  heads: readonly number[],
  box: [number, number, number],
  contact: number,
): Map<number, number> {
  const mi = (d: number, L: number): number => d - L * Math.round(d / L)
  const out = new Map<number, number>()
  // Cell list over heads at side `contact`.
  const nx = Math.max(1, Math.floor(box[0] / contact))
  const ny = Math.max(1, Math.floor(box[1] / contact))
  const nz = Math.max(1, Math.floor(box[2] / contact))
  const wx = box[0] / nx, wy = box[1] / ny, wz = box[2] / nz
  const wrap = (v: number, L: number): number => ((v % L) + L) % L
  const key = (i: number): number =>
    Math.min(nx - 1, Math.floor(wrap(pos[i * 4], box[0]) / wx)) +
    nx * (Math.min(ny - 1, Math.floor(wrap(pos[i * 4 + 1], box[1]) / wy)) +
      ny * Math.min(nz - 1, Math.floor(wrap(pos[i * 4 + 2], box[2]) / wz)))
  const buckets = new Map<number, number[]>()
  for (const i of heads) {
    const k = key(i)
    const b = buckets.get(k)
    if (b) b.push(i)
    else buckets.set(k, [i])
  }
  for (const i of heads) {
    const cx = Math.min(nx - 1, Math.floor(wrap(pos[i * 4], box[0]) / wx))
    const cy = Math.min(ny - 1, Math.floor(wrap(pos[i * 4 + 1], box[1]) / wy))
    const cz = Math.min(nz - 1, Math.floor(wrap(pos[i * 4 + 2], box[2]) / wz))
    let best = Infinity
    let bestJ = -1
    for (let dz = -1; dz <= 1; dz++) {
      const az = ((cz + dz) % nz + nz) % nz
      for (let dy = -1; dy <= 1; dy++) {
        const ay = ((cy + dy) % ny + ny) % ny
        for (let dx = -1; dx <= 1; dx++) {
          const ax = ((cx + dx) % nx + nx) % nx
          const b = buckets.get(ax + nx * (ay + ny * az))
          if (!b) continue
          for (const j of b) {
            if (j === i) continue
            if ((charges[i] === 0) === (charges[j] === 0)) continue
            const ddx = mi(pos[i * 4] - pos[j * 4], box[0])
            const ddy = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1])
            const ddz = mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])
            const r = Math.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            if (r <= contact && r < best) {
              best = r
              bestJ = j
            }
          }
        }
      }
    }
    if (bestJ >= 0) out.set(i, bestJ)
  }
  return out
}

export interface PairSurvival {
  /** Pairs present at the EARLIER checkpoint. */
  before: number
  /** Pairs present now. */
  now: number
  /** Of `before`, how many are still the SAME unordered pair now. */
  survived: number
  /** Of `before`, how many still have SOME unlike partner but a DIFFERENT one -- an exchange, not a
   * break. Reported separately because the two have different physical meanings and a single
   * "survival" number hides the difference. */
  repartnered: number
  survivalFraction: number
  /** Steps between the two checkpoints -- the interval the fraction is a survival over. */
  stepsElapsed: number
  /** First-order lifetime in STEPS implied by that fraction: -dt / ln(f). Infinite when nothing broke
   * (f = 1) and 0 when everything did, both reported as-is rather than clamped -- a lifetime longer
   * than the interval it was measured over is a lower bound, and saying so is the honest reading. */
  lifetimeSteps: number
}

/** How many of `before`'s pairs are still the same pair in `now`. Unordered: a pair is counted as
 * surviving if either direction still maps to the same partner, because nearestUnlikeHead is a
 * nearest-neighbour map and is therefore not automatically symmetric (i's nearest unlike head can be
 * j while j's is k). */
export function pairSurvival(
  before: Map<number, number>,
  now: Map<number, number>,
  stepsElapsed: number,
): PairSurvival {
  const seen = new Set<string>()
  let total = 0
  let survived = 0
  let repartnered = 0
  for (const [i, j] of before) {
    const k = i < j ? `${i}:${j}` : `${j}:${i}`
    if (seen.has(k)) continue
    seen.add(k)
    total++
    if (now.get(i) === j || now.get(j) === i) survived++
    else if (now.has(i) || now.has(j)) repartnered++
  }
  const nowSeen = new Set<string>()
  for (const [i, j] of now) nowSeen.add(i < j ? `${i}:${j}` : `${j}:${i}`)
  const f = total > 0 ? survived / total : 0
  return {
    before: total,
    now: nowSeen.size,
    survived,
    repartnered,
    survivalFraction: f,
    stepsElapsed,
    lifetimeSteps: f >= 1 ? Infinity : f <= 0 ? 0 : -stepsElapsed / Math.log(f),
  }
}

/** The acid-soap well, in the SAME shape and depth units the force uses: attr_v's own cos^2 ramp
 * (engine/wgsl/forces.wgsl, transcribed once in soup/src/soup-potential.ts and once here), times the
 * normalised depth. `scale` is soup/src/soup-attraction.ts's acidSoapScaleOf(); `rcAttr`/`wc`/
 * `epsilon` are the rank-A shape constants from data/params.json. Returns a NEGATIVE number (an
 * attraction) or 0 beyond the well. */
export interface AcidSoapWell {
  scale: number
  rcAttr: number
  wc: number
  epsilon: number
}

export function acidSoapWell(r: number, w: AcidSoapWell): number {
  if (w.scale <= 0) return 0
  if (r < w.rcAttr) return -w.scale * w.epsilon
  if (r > w.rcAttr + w.wc) return 0
  const c = Math.cos((Math.PI * (r - w.rcAttr)) / (2 * w.wc))
  return -w.scale * w.epsilon * c * c
}

/** THE WORK OF HAVING THE CHARGE ON HEAD `i`, from the acid-soap pair term alone -- i.e.
 * U_pair(i deprotonated) - U_pair(i protonated) at fixed environment, which is exactly the shape the
 * constant-pH Monte Carlo's electrostatic term already has (soup/src/electrostatics.ts's
 * protonationSweep computes `uEs` the same way and this is added to it).
 *
 * WHY THIS IS NOT OPTIONAL. Flipping a head's protonation state changes which of its neighbours it
 * pairs with, so it changes the system's potential energy through the pair term as well as through
 * the screened Coulomb. A sampler that omits it would sample a distribution inconsistent with the
 * potential the dynamics integrates -- the same class of silent inconsistency as a stale F(x). And it
 * is the whole MECHANISM of the predicted pH window: a head sitting among CHARGED neighbours gains
 * pair energy by becoming neutral and vice versa, so the pair term is a restoring force on alpha
 * toward 1/2, which is where half the heads are deprotonated and every acid has a soap to pair with.
 *
 * Deprotonated i pairs with the NEUTRAL neighbours; protonated i pairs with the CHARGED ones. Hence
 * the difference is sum(neutral j) - sum(charged j), both over the well. */
export function acidSoapSiteWork(
  positions: Float32Array,
  charges: Float32Array,
  i: number,
  neighbours: Int32Array,
  count: number,
  box: [number, number, number],
  w: AcidSoapWell,
): number {
  if (w.scale <= 0) return 0
  const reach = w.rcAttr + w.wc
  const reach2 = reach * reach
  let u = 0
  for (let s = 0; s < count; s++) {
    const j = neighbours[s]
    if (j === i) continue
    let dx = positions[i * 4] - positions[j * 4]
    let dy = positions[i * 4 + 1] - positions[j * 4 + 1]
    let dz = positions[i * 4 + 2] - positions[j * 4 + 2]
    dx -= Math.round(dx / box[0]) * box[0]
    dy -= Math.round(dy / box[1]) * box[1]
    dz -= Math.round(dz / box[2]) * box[2]
    const r2 = dx * dx + dy * dy + dz * dz
    if (r2 >= reach2 || r2 <= 0) continue
    const v = acidSoapWell(Math.sqrt(r2), w)
    u += charges[j] === 0 ? v : -v
  }
  return u
}
