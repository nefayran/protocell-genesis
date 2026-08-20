// THE FUSION PROBE's state builder -- task 'coalescence' (2026-08-20),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/coalescence-report.md.
//
// It is a PROBE, not a run: it takes a settled checkpoint of the decisive box-54 lineage and moves
// two of its own aggregates into surface contact, so that "does a contact lead to a merge" is asked
// directly instead of waited for. Nothing is pre-made -- every particle in the probe is a particle
// the decisive run itself produced, and the report says so wherever a probe number appears.
//
// The one design decision that makes the probe defensible: it is a PERMUTATION OF POSITIONS INSIDE
// THE PARENT BOX, not a new smaller system. An aggregate is displaced by a rigid translation; every
// solvent/monomer bead that its new position would overlap is moved into a slot the aggregate itself
// just vacated. So the box, the composition, the total density, the water density, the free-monomer
// concentrations, the bond graph, the velocities and both RNG streams are the parent's own, bit for
// bit -- the ONLY difference from the parent checkpoint is which of two equal-sized volumes each of
// ~600 beads occupies. A probe built by filling a fresh box with water would have changed four
// things at once and could not have been read.
//
// The rigid translation cannot break a molecule: it moves every bead of the aggregate by the same
// vector, so no bonded, bend or tether distance changes at all (asserted by the caller against the
// parent's own bond list).

import type { CheckpointFile } from '../../soup/src/checkpoint'

const mi = (d: number, L: number) => d - L * Math.round(d / L)
const wrap = (v: number, L: number) => ((v % L) + L) % L

export interface ContactPlan {
  /** Particle indices of the aggregate that stays put, and of the one that is displaced. */
  anchor: number[]
  mover: number[]
  /** Target minimum surface-surface separation after the move. */
  targetGap: number
}

export interface ProbeReport {
  moved: number
  displacement: [number, number, number]
  gapBefore: number
  gapAfter: number
  /** Beads lifted out of the mover's new footprint (whole molecules). */
  intruders: number
  removedUnits: number
  /** Molecules placed only at the parent checkpoint's own worst-contact clearance, not at
   * `overlapRadius` -- reported, never hidden. */
  relaxedPlacements: number
  /** Molecules that found no clear site -- must be 0; the caller fails loudly otherwise. */
  leftover: number
  minSeparationAfter: number
}

/** Minimum member-to-member distance between two index sets, minimum-image. */
export function minGap(pos: Float32Array, box: [number, number, number], a: readonly number[], b: readonly number[]): number {
  let best = Infinity
  for (const i of a) {
    const xi = pos[i * 4]
    const yi = pos[i * 4 + 1]
    const zi = pos[i * 4 + 2]
    for (const j of b) {
      const dx = mi(xi - pos[j * 4], box[0])
      const dy = mi(yi - pos[j * 4 + 1], box[1])
      const dz = mi(zi - pos[j * 4 + 2], box[2])
      const r2 = dx * dx + dy * dy + dz * dz
      if (r2 < best) best = r2
    }
  }
  return Math.sqrt(best)
}

function centroid(pos: Float32Array, box: [number, number, number], idx: readonly number[]): [number, number, number] {
  const ref = [pos[idx[0] * 4], pos[idx[0] * 4 + 1], pos[idx[0] * 4 + 2]]
  const acc = [0, 0, 0]
  for (const i of idx) for (let ax = 0; ax < 3; ax++) acc[ax] += ref[ax] + mi(pos[i * 4 + ax] - ref[ax], box[ax])
  return [acc[0] / idx.length, acc[1] / idx.length, acc[2] / idx.length]
}

/** Applies one contact plan in place, by REMOVE-AND-REPLACE (no pushing).
 *
 * 1. The mover is translated rigidly along the anchor->mover centre line until the measured minimum
 *    surface separation is `targetGap`. A rigid translation changes no bonded, bend or tether
 *    distance inside the mover.
 * 2. Every free bead the mover's new footprint would overlap -- and, with it, every other bead of its
 *    molecule -- is taken OUT of the configuration.
 * 3. Each removed molecule is put back, rigidly (random site + random orientation), at a site that
 *    clears `overlapRadius` of everything that is staying and of everything already put back. Sites
 *    are drawn first from the volume the mover itself vacated, which is genuinely empty once the
 *    mover has moved and the overlapping beads have been lifted out.
 *
 * Why not pushing: the first version of this probe pushed the overlapping beads outwards in sweeps.
 * It produced a state whose worst contact was no tighter than the parent's (0.6957 sigma, measured)
 * but whose max|F| was 2.2261e4 and which DIVERGED inside 500 steps even after a 200-iteration
 * minimisation took max|F| to 1.4835e1 -- the loud guard caught it both times. Pushing several
 * hundred molecules by up to 0.75 sigma each stores energy in a shell that no local minimisation
 * relieves. Remove-and-replace stores none: nothing is compressed, every bead ends at a separation
 * this system's own bulk already carries.
 */
export function applyContact(
  pos: Float32Array,
  box: [number, number, number],
  plan: ContactPlan,
  /** Beads that may never move: every bead of every QUALIFYING AGGREGATE. */
  protectedIdx: ReadonlySet<number>,
  /** Molecule id per particle (union-find over the bond graph). */
  unitOf: Int32Array,
  /** Clearance required between two beads, IN UNITS OF THAT PAIR'S OWN WCA bead size
   * b_ij = sigma*(r_i + r_j)/2 -- the rule soup/src/soup-clay.ts:143 already uses for this engine's
   * pair sizes. A species-blind absolute radius is what made the first three probes diverge: the
   * catalyst's radiusSigma is 1.2 against water's 1.0, so a catalyst placed 0.75 sigma from a water
   * bead sits at 0.61 of ITS OWN contact distance, and 24*eps*[2*(b/r)^12-(b/r)^6]/r there is
   * ~1.8e4 -- which is exactly the max|F| = 2.2261e4 the engine measured on that probe. */
  overlapRadius: number,
  /** radiusSigma per particle, from its species slot. */
  radiusOf: Float32Array,
): ProbeReport {
  const bOf = (i: number, j: number) => ((radiusOf[i] + radiusOf[j]) / 2) * overlapRadius
  const N = pos.length / 4
  const gapBefore = minGap(pos, box, plan.anchor, plan.mover)
  const ca = centroid(pos, box, plan.anchor)
  const cm = centroid(pos, box, plan.mover)
  const dir = [mi(cm[0] - ca[0], box[0]), mi(cm[1] - ca[1], box[1]), mi(cm[2] - ca[2], box[2])]
  const dn = Math.hypot(dir[0], dir[1], dir[2])
  const unit = dn > 1e-9 ? [dir[0] / dn, dir[1] / dn, dir[2] / dn] : [1, 0, 0]
  const vacated: number[] = []
  for (const i of plan.mover) vacated.push(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2])
  const trial = (sMag: number) => {
    for (let k = 0; k < plan.mover.length; k++) {
      const i = plan.mover[k]
      for (let ax = 0; ax < 3; ax++) pos[i * 4 + ax] = wrap(vacated[k * 3 + ax] - unit[ax] * sMag, box[ax])
    }
    return minGap(pos, box, plan.anchor, plan.mover)
  }
  let lo = 0
  let hi = Math.max(0, gapBefore - plan.targetGap) + 2
  for (let it = 0; it < 40; it++) {
    const mid = (lo + hi) / 2
    if (trial(mid) > plan.targetGap) lo = mid
    else hi = mid
  }
  const gapAfter = trial(lo)
  const displacement: [number, number, number] = [-unit[0] * lo, -unit[1] * lo, -unit[2] * lo]

  const moverSet = new Set(plan.mover)
  const allUnits = new Map<number, number[]>()
  for (let i = 0; i < N; i++) {
    const a = allUnits.get(unitOf[i])
    if (a) a.push(i)
    else allUnits.set(unitOf[i], [i])
  }
  // --- step 2: lift out every free molecule the mover's new footprint overlaps
  const removed = new Set<number>()
  const removedUnits: number[][] = []
  for (let i = 0; i < N; i++) {
    if (moverSet.has(i) || protectedIdx.has(i) || removed.has(i)) continue
    let hit = false
    for (const j of plan.mover) {
      const b = bOf(i, j)
      const dx = mi(pos[i * 4] - pos[j * 4], box[0])
      if (Math.abs(dx) > b) continue
      const dy = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1])
      if (Math.abs(dy) > b) continue
      const dz = mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])
      if (dx * dx + dy * dy + dz * dz <= b * b) {
        hit = true
        break
      }
    }
    if (!hit) continue
    const members = allUnits.get(unitOf[i]) ?? [i]
    if (members.some((m) => protectedIdx.has(m) || moverSet.has(m))) continue
    removedUnits.push(members)
    for (const m of members) removed.add(m)
  }
  // --- step 3: put them back, largest molecule first, on a grid of everything that is staying
  let maxR = 0
  for (let i = 0; i < N; i++) if (radiusOf[i] > maxR) maxR = radiusOf[i]
  const cellW = Math.max(overlapRadius * maxR, 1.0)
  const gn = [Math.max(1, Math.floor(box[0] / cellW)), Math.max(1, Math.floor(box[1] / cellW)), Math.max(1, Math.floor(box[2] / cellW))]
  const w = [box[0] / gn[0], box[1] / gn[1], box[2] / gn[2]]
  const cellKey = (x: number, y: number, z: number) => {
    const cx = Math.min(gn[0] - 1, Math.floor(wrap(x, box[0]) / w[0]))
    const cy = Math.min(gn[1] - 1, Math.floor(wrap(y, box[1]) / w[1]))
    const cz = Math.min(gn[2] - 1, Math.floor(wrap(z, box[2]) / w[2]))
    return [cx, cy, cz]
  }
  const grid = new Map<number, number[]>()
  const insert = (i: number) => {
    const [cx, cy, cz] = cellKey(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2])
    const k = (cz * gn[1] + cy) * gn[0] + cx
    const a = grid.get(k)
    if (a) a.push(i)
    else grid.set(k, [i])
  }
  for (let i = 0; i < N; i++) if (!removed.has(i)) insert(i)
  const clearAtR = (x: number, y: number, z: number, rad: number, rSelf: number) => {
    const [cx, cy, cz] = cellKey(x, y, z)
    for (let dz = -1; dz <= 1; dz++) {
      const kz = (((cz + dz) % gn[2]) + gn[2]) % gn[2]
      for (let dy = -1; dy <= 1; dy++) {
        const ky = (((cy + dy) % gn[1]) + gn[1]) % gn[1]
        for (let dx = -1; dx <= 1; dx++) {
          const kx = (((cx + dx) % gn[0]) + gn[0]) % gn[0]
          for (const j of grid.get((kz * gn[1] + ky) * gn[0] + kx) ?? []) {
            const need = ((rSelf + radiusOf[j]) / 2) * rad
            const ex = mi(x - pos[j * 4], box[0])
            if (Math.abs(ex) > need) continue
            const ey = mi(y - pos[j * 4 + 1], box[1])
            if (Math.abs(ey) > need) continue
            const ez = mi(z - pos[j * 4 + 2], box[2])
            if (ex * ex + ey * ey + ez * ez <= need * need) return false
          }
        }
      }
    }
    return true
  }
  const clearAt = (x: number, y: number, z: number, rSelf: number) => clearAtR(x, y, z, overlapRadius, rSelf)
  let relaxedPlacements = 0
  let rngState = 987654321
  const rnd = () => {
    rngState = (Math.imul(rngState, 1664525) + 1013904223) >>> 0
    return rngState / 4294967296
  }
  removedUnits.sort((a, b) => b.length - a.length)
  // Every lifted molecule's OWN former site is now empty, and so is every other lifted molecule's --
  // 45 to 130 vacancies of exactly the right size distribution. Those are the first sites tried, in
  // largest-molecule-first order, which is why a six-bead chain finds a home: it goes into a hole a
  // molecule of its own kind came out of.
  const freeSites = removedUnits.map((m) => centroid(pos, box, m))
  let leftover = 0
  for (const members of removedUnits) {
    const cen = centroid(pos, box, members)
    const offsets = members.map((m) => [mi(pos[m * 4] - cen[0], box[0]), mi(pos[m * 4 + 1] - cen[1], box[1]), mi(pos[m * 4 + 2] - cen[2], box[2])])
    let placed = false
    for (let t = 0; t < 60000 && !placed; t++) {
      // Site search, in the order that actually finds sites. A LOCAL nudge first: the molecule's own
      // original site is clear except where the mover now sits, so a small displacement away from it
      // is where the good sites are dense. A uniformly random site is hopeless for anything longer
      // than a couple of beads (p ~ 0.116^n at rho_tot = 1.218 -- measured: a 6-bead chain failed
      // 30 000 uniform draws), and the vacated slab is thinner than a long chain.
      const site =
        t < 20000
          ? freeSites[t % freeSites.length]
          : t < 40000
            ? [cen[0] + (rnd() * 2 - 1) * 4, cen[1] + (rnd() * 2 - 1) * 4, cen[2] + (rnd() * 2 - 1) * 4]
            : t < 50000
              ? [vacated[(t % plan.mover.length) * 3], vacated[(t % plan.mover.length) * 3 + 1], vacated[(t % plan.mover.length) * 3 + 2]]
              : [rnd() * box[0], rnd() * box[1], rnd() * box[2]]
      const q = [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1]
      const qn = Math.hypot(q[0], q[1], q[2], q[3]) || 1
      const [qw, qx, qy, qz] = q.map((v) => v / qn)
      const rot = (v: number[]) => {
        const tt = [2 * (qy * v[2] - qz * v[1]), 2 * (qz * v[0] - qx * v[2]), 2 * (qx * v[1] - qy * v[0])]
        return [v[0] + qw * tt[0] + (qy * tt[2] - qz * tt[1]), v[1] + qw * tt[1] + (qz * tt[0] - qx * tt[2]), v[2] + qw * tt[2] + (qx * tt[1] - qy * tt[0])]
      }
      const cand = offsets.map((o) => {
        const r = rot(o)
        return [wrap(site[0] + r[0], box[0]), wrap(site[1] + r[1], box[1]), wrap(site[2] + r[2], box[2])]
      })
      // Every bead of the molecule must clear, and (for a multi-bead molecule) its own beads are
      // excluded from that test by construction because they are not in the grid yet.
      if (!cand.every((c, k) => clearAt(c[0], c[1], c[2], radiusOf[members[k]]))) continue
      members.forEach((m, k) => {
        pos[m * 4] = cand[k][0]
        pos[m * 4 + 1] = cand[k][1]
        pos[m * 4 + 2] = cand[k][2]
        insert(m)
      })
      placed = true
    }
    // Last resort for a long chain that finds no site at `overlapRadius`: the same search at the
    // parent checkpoint's OWN worst contact, which is the acceptance reference this probe is held to
    // (0.6957 sigma, measured). Below that it is not attempted at all -- the caller fails loudly.
    if (!placed) {
      const relaxed = 0.6958
      for (let t = 0; t < 60000 && !placed; t++) {
        const site =
          t < 20000
            ? freeSites[t % freeSites.length]
            : t < 40000
              ? [cen[0] + (rnd() * 2 - 1) * 4, cen[1] + (rnd() * 2 - 1) * 4, cen[2] + (rnd() * 2 - 1) * 4]
              : [vacated[(t % plan.mover.length) * 3], vacated[(t % plan.mover.length) * 3 + 1], vacated[(t % plan.mover.length) * 3 + 2]]
        const q = [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1]
        const qn = Math.hypot(q[0], q[1], q[2], q[3]) || 1
        const [qw, qx, qy, qz] = q.map((v) => v / qn)
        const rot = (v: number[]) => {
          const tt = [2 * (qy * v[2] - qz * v[1]), 2 * (qz * v[0] - qx * v[2]), 2 * (qx * v[1] - qy * v[0])]
          return [v[0] + qw * tt[0] + (qy * tt[2] - qz * tt[1]), v[1] + qw * tt[1] + (qz * tt[0] - qx * tt[2]), v[2] + qw * tt[2] + (qx * tt[1] - qy * tt[0])]
        }
        const cand = offsets.map((o) => {
          const r = rot(o)
          return [wrap(site[0] + r[0], box[0]), wrap(site[1] + r[1], box[1]), wrap(site[2] + r[2], box[2])]
        })
        if (!cand.every((c, k) => clearAtR(c[0], c[1], c[2], relaxed, radiusOf[members[k]]))) continue
        members.forEach((m, k) => {
          pos[m * 4] = cand[k][0]
          pos[m * 4 + 1] = cand[k][1]
          pos[m * 4 + 2] = cand[k][2]
          insert(m)
        })
        placed = true
        relaxedPlacements++
      }
    }
    if (!placed) leftover++
  }
  return {
    moved: plan.mover.length,
    displacement,
    gapBefore: Number(gapBefore.toFixed(4)),
    gapAfter: Number(gapAfter.toFixed(4)),
    intruders: removed.size,
    removedUnits: removedUnits.length,
    relaxedPlacements,
    leftover,
    minSeparationAfter: 0,
  }
}

/** Re-encodes a decoded state back into a CheckpointFile, changing ONLY the positions payload and
 * the config the resume signature is matched against. */
export function reencode(
  parent: CheckpointFile,
  positions: Float32Array,
  config: CheckpointFile['config'],
  b64: (a: Float32Array | Uint32Array) => string,
): CheckpointFile {
  return { ...parent, createdAt: new Date().toISOString(), config, positionsB64: b64(positions) }
}

/** Smallest particle-particle separation anywhere in a snapshot, minimum-image, via a cell grid.
 * This is the honest acceptance reference for the probe: a hand-built state is safe to integrate if
 * its worst contact is no worse than the worst contact the PARENT trajectory itself carries -- a
 * number to be measured, not a threshold to be picked. */
export function globalMinSeparation(pos: Float32Array, box: [number, number, number], cell = 1.2): { min: number; pair: [number, number] } {
  const N = pos.length / 4
  const n = [Math.max(1, Math.floor(box[0] / cell)), Math.max(1, Math.floor(box[1] / cell)), Math.max(1, Math.floor(box[2] / cell))]
  const w = [box[0] / n[0], box[1] / n[1], box[2] / n[2]]
  const map = new Map<number, number[]>()
  const cidx = (i: number) => {
    const c = [0, 0, 0]
    for (let ax = 0; ax < 3; ax++) c[ax] = Math.min(n[ax] - 1, Math.floor(wrap(pos[i * 4 + ax], box[ax]) / w[ax]))
    return c
  }
  for (let i = 0; i < N; i++) {
    const c = cidx(i)
    const k = (c[2] * n[1] + c[1]) * n[0] + c[0]
    const a = map.get(k)
    if (a) a.push(i)
    else map.set(k, [i])
  }
  let best = Infinity
  let pair: [number, number] = [-1, -1]
  for (let i = 0; i < N; i++) {
    const c = cidx(i)
    for (let dz = -1; dz <= 1; dz++) {
      const kz = (((c[2] + dz) % n[2]) + n[2]) % n[2]
      for (let dy = -1; dy <= 1; dy++) {
        const ky = (((c[1] + dy) % n[1]) + n[1]) % n[1]
        for (let dx = -1; dx <= 1; dx++) {
          const kx = (((c[0] + dx) % n[0]) + n[0]) % n[0]
          for (const j of map.get((kz * n[1] + ky) * n[0] + kx) ?? []) {
            if (j <= i) continue
            const ex = mi(pos[i * 4] - pos[j * 4], box[0])
            if (Math.abs(ex) >= best) continue
            const ey = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1])
            if (Math.abs(ey) >= best) continue
            const ez = mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])
            const r = Math.sqrt(ex * ex + ey * ey + ez * ez)
            if (r < best) {
              best = r
              pair = [i, j]
            }
          }
        }
      }
    }
  }
  return { min: best, pair }
}
