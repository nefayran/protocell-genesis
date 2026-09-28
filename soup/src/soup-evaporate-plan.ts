// The PURE, GPU-free half of solvent evaporation: the plan (what the dry phase targets and how many
// minimiser iterations an inserted bead needs), the two ramps that walk between the wet and dry
// states, and the insertion sampler. Split out of soup/src/soup-evaporate.ts by responsibility
// (CLAUDE.md's 400-600 line rule) when task 'confined-parcel' (2026-08-21) pushed that file to 615
// lines: the rule is "split first, then add", and the split line is exactly the one the file's own
// section comment already drew -- everything here is arithmetic over counts, volumes and positions
// with no GPUDevice, no dispatch and no readback anywhere, while everything left behind mutates a
// LIVE system. A pure move: not one formula, one clamp, one message or one default changed, and
// soup/src/soup-evaporate.ts re-exports every name below so no importer sees any difference.
//
// See soup/src/soup-evaporate.ts's own header for the mechanism as a whole and for the measured
// failures (the cold-start-style overlap an inserted bead lands in, the FENE sign change past r_inf)
// that shaped these numbers.

import { wcaCutoff } from '../../engine/src/params'
import type { Params } from '../../engine/src/params'
import type { Soup } from './rules'
import { computeDryBox } from './soup-box-scale-math'


/** How many candidate positions a returning solvent bead is offered before the best one is taken.
 * Purely an algorithmic knob on the rejection sampler (like soup/src/soup-relax.ts's own
 * RELAX_SYNC_EVERY): it cannot change what is modelled, only how hard the sampler tries before
 * falling back to "the roomiest of the candidates seen". */
const INSERTION_CANDIDATES = 64

export interface EvaporationPlan {
  /** kind index (position in data/soup.json's `monomers`) of the solvent species. */
  solventKind: number
  /** First particle index of the solvent block. Every index BELOW this is organic and never moves. */
  solventBlockStart: number
  wetSolventCount: number
  drySolventCount: number
  /** N minus the wet solvent pool -- the particles that never leave. */
  organicCount: number
  wetBox: [number, number, number]
  dryBox: [number, number, number]
  rampSteps: number
  minSeparationSigma: number
  /** Derived, NOT a config field: see data/soup.json's dryWetCycle.basis item 7. */
  relaxIterations: number
  /** V_wet / V_dry -- the factor by which the surviving organics actually concentrate. */
  concentrationFactor: number
}

/** Resolves everything the evaporating cycle needs from data/soup.json plus this system's own box/N,
 * WITH its throwing preconditions. Pure: no GPU, so tests can check the arithmetic and the refusals
 * without a device. */
export function planEvaporation(
  soup: Soup,
  p: Params,
  box: [number, number, number],
  startCounts: Record<string, number>,
  /** Task 'confined-parcel' (2026-08-21): the volume the material occupies at a given box when the run
   * confines it to a parcel -- threaded into computeDryBox and the mean-spacing term, both densities.
   * Omitted (every unconfined run) is the box volume. */
  occupiedVolumeOf?: (b: [number, number, number]) => number,
): EvaporationPlan {
  const dwc = soup.dryWetCycle
  if (dwc.residualSolventFraction === undefined || dwc.evaporationRampSteps === undefined || dwc.insertionMinSeparationSigma === undefined) {
    throw new Error('data/soup.json: solvent evaporation requested, but dryWetCycle does not contain its fields (residualSolventFraction / evaporationRampSteps / insertionMinSeparationSigma)')
  }
  const solventKind = soup.monomers.findIndex((m) => m.id === soup.solvent.waterId)
  if (solventKind < 0) throw new Error(`data/soup.json: solvent.waterId=${soup.solvent.waterId} not found among monomers`)
  const counts = soup.monomers.map((m) => startCounts[m.id] ?? 0)
  // Precondition (see this file's header): the solvent must be the LAST non-empty block, or
  // truncating it would renumber non-solvent particles and silently corrupt the bond graph.
  for (let k = solventKind + 1; k < counts.length; k++) {
    if (counts[k] > 0) {
      throw new Error(
        `solvent evaporation: the solvent block (${soup.solvent.waterId}, kind ${solventKind}) must be the last non-empty block of the composition, ` +
          `but kind ${soup.monomers[k].id} has ${counts[k]} particles -- truncation would shift the indices of non-water particles and break the bond graph`,
      )
    }
  }
  const wetSolventCount = counts[solventKind]
  if (wetSolventCount <= 0) throw new Error('solvent evaporation: the composition contains no solvent')
  const solventBlockStart = counts.slice(0, solventKind).reduce((a, b) => a + b, 0)
  const organicCount = solventBlockStart
  const drySolventCount = Math.round(wetSolventCount * dwc.residualSolventFraction)
  if (drySolventCount >= wetSolventCount) {
    throw new Error(
      `solvent evaporation: residualSolventFraction=${dwc.residualSolventFraction} removes not a single bead ` +
        `(wet ${wetSolventCount}, dry ${drySolventCount}) -- the dry phase must remove the solvent`,
    )
  }
  const dryBox = computeDryBox(box, organicCount + drySolventCount, dwc.targetDryDensity, occupiedVolumeOf)
  // The one number that is derived rather than configured: the displacement budget an inserted bead
  // needs, in units of the minimiser's own first-iteration cap. Two terms, both geometric:
  //  - lift out of the core it was placed in: wcaCutoff(stiffest solvent pair) - insertionMinSeparation;
  //  - migrate as far as the mean inter-particle spacing at the wet density, rho^(-1/3), because a
  //    bead placed in a locally jammed neighbourhood has to REACH free volume, not merely back off
  //    one neighbour -- the first version of this derivation had only the first term and measured
  //    max|F| still at 1.2e4 after the minimisation, i.e. visibly not converged.
  // The normalised descent's total displacement bound is d0*(it+1)/2 (soup/src/soup-relax.ts), so
  // it = ceil(2*need/d0) is exactly "enough budget to get there", not a tuned number.
  const maxRadius = Math.max(...soup.monomers.map((m) => m.radiusSigma))
  const solventRadius = soup.monomers[solventKind].radiusSigma
  const stiffestCutoff = wcaCutoff(p.sigma * (solventRadius + maxRadius) * 0.5)
  const d0 = (soup.coldStartRelax?.maxDisplacementSigma ?? 0.1) * p.sigma
  const wetVolume = occupiedVolumeOf ? occupiedVolumeOf(box) : box[0] * box[1] * box[2]
  const meanSpacing = Math.cbrt(wetVolume / (organicCount + wetSolventCount))
  const need = Math.max(0, stiffestCutoff - dwc.insertionMinSeparationSigma * p.sigma) + meanSpacing
  return {
    solventKind,
    solventBlockStart,
    wetSolventCount,
    drySolventCount,
    organicCount,
    wetBox: box,
    dryBox,
    rampSteps: dwc.evaporationRampSteps,
    minSeparationSigma: dwc.insertionMinSeparationSigma,
    relaxIterations: Math.max(1, Math.ceil((2 * need) / d0)),
    concentrationFactor: wetVolume / (occupiedVolumeOf ? occupiedVolumeOf(dryBox) : dryBox[0] * dryBox[1] * dryBox[2]),
  }
}

/** The wet->dry ladder: `rampSteps` entries, each a (box, solvent count) pair, the last being exactly
 * the dry box and the residual solvent count.
 *
 * The BOX is what is stepped uniformly in ln L (so every increment is the same linear contraction --
 * the 3.18 % this project has measured safe, dryWetCycle.basis item 8) and the solvent count FOLLOWS
 * from a log-linear ramp of the total density between the wet density and targetDryDensity. Doing it
 * the other way round (uniform steps in solvent count, box derived) makes the last increments 4.9 %
 * linear, since the count falls linearly while the volume falls as its cube root. */
export function evaporationLadder(
  plan: EvaporationPlan,
  targetDryDensity: number,
  /** Task 'confined-parcel' (2026-08-21): THE FOURTH place that assumed the box IS the volume, and
   * the only one this task found by a DIVERGENCE rather than by reading. Both densities below are
   * densities: the wet one the ramp starts from, and the per-rung one it interpolates to the dry
   * target. Taken over the box in a confined run, rho_wet reads 0.0525 instead of 1.1009, so
   * ln(1.34/rho_wet) is 3.24 instead of 0.197 and EVERY rung's solvent target lands far above the
   * pool -- the clamp then holds the solvent at its full wet count for all 17 intermediate rungs
   * while the box (and with it the parcel) contracts by the whole wet/dry ratio. Measured: the run
   * reached 2.32 sigma^-3 by rung 9 and 614 769 of 645 453 position components went non-finite
   * between steps 12 800 and 13 000. The finiteness guard caught it LOUDLY, which is the only reason
   * this is a paragraph and not a wrong number in a table. Omitted (every unconfined run) is the box
   * volume, bit-identical to before. */
  occupiedVolumeOf?: (b: [number, number, number]) => number,
): { box: [number, number, number]; solvent: number }[] {
  const { wetBox, dryBox, rampSteps, organicCount, wetSolventCount, drySolventCount } = plan
  const vol = (b: [number, number, number]) => (occupiedVolumeOf ? occupiedVolumeOf(b) : b[0] * b[1] * b[2])
  const nWet = organicCount + wetSolventCount
  const rhoWet = nWet / vol(wetBox)
  const lnBox = Math.log(dryBox[0] / wetBox[0])
  const lnRho = Math.log(targetDryDensity / rhoWet)
  const out: { box: [number, number, number]; solvent: number }[] = []
  for (let s = 1; s <= rampSteps; s++) {
    if (s === rampSteps) {
      out.push({ box: dryBox, solvent: drySolventCount })
      continue
    }
    const f = s / rampSteps
    const scale = Math.exp(lnBox * f)
    const box: [number, number, number] = [wetBox[0] * scale, wetBox[1] * scale, wetBox[2] * scale]
    const rho = rhoWet * Math.exp(lnRho * f)
    const solvent = Math.round(rho * vol(box)) - organicCount
    // Clamped to stay monotone and inside the pool: the ramp is a schedule, not a constraint the
    // arithmetic may violate at a rounding boundary.
    const prev = out.length > 0 ? out[out.length - 1].solvent : wetSolventCount
    out.push({ box, solvent: Math.min(prev, Math.max(drySolventCount, solvent)) })
  }
  return out
}

/** The dry->wet ladder: the SAME box ladder walked backwards, with the solvent held at the residual
 * count the whole way and the whole pool returned on the FINAL increment, at the wet box.
 *
 * Why the solvent cannot come back gradually, measured rather than assumed: an affine expansion
 * widens every gap by the same 3.18 % per increment, so it never opens a cavity a whole bead wide,
 * and at total densities of 1.2-1.34 a WCA liquid has no sigma-sized cavities to find. The only place
 * the pool fits is the fully-expanded box with the organics already dilute (rho_org 0.418). The cost
 * of that -- (rampSteps-1)*rampRelaxSteps real steps per rehydration spent solvent-free -- is stated
 * in dryWetCycle.basis item 5 and reported as an artefact, not hidden. */
export function rehydrationLadder(plan: EvaporationPlan): { box: [number, number, number]; solvent: number }[] {
  const { wetBox, dryBox, rampSteps, wetSolventCount, drySolventCount } = plan
  const lnBox = Math.log(wetBox[0] / dryBox[0])
  const out: { box: [number, number, number]; solvent: number }[] = []
  for (let s = 1; s <= rampSteps; s++) {
    if (s === rampSteps) {
      out.push({ box: wetBox, solvent: wetSolventCount })
      continue
    }
    const scale = Math.exp((lnBox * s) / rampSteps)
    out.push({ box: [dryBox[0] * scale, dryBox[1] * scale, dryBox[2] * scale], solvent: drySolventCount })
  }
  return out
}

/** Rejection-sampled positions for `count` returning solvent beads: uniform in `box`, at least
 * `minSep` from every particle already present AND from every bead already placed by this call, via
 * a cell list of side `minSep`. Best-of-INSERTION_CANDIDATES when the floor cannot be met, so the
 * sampler degrades into "the roomiest candidate seen" rather than failing -- and it REPORTS how often
 * that happened (`shortOfFloor`) plus the worst separation it actually achieved, so the fallback
 * cannot hide.
 *
 * Uniformly at random in the volume, deliberately: the box is fully periodic, so there is no
 * air-water interface for a surface-preferential placement to be defined against (dryWetCycle.basis
 * item 4). */
export function sampleInsertionPositions(
  active: Float32Array,
  activeN: number,
  box: [number, number, number],
  count: number,
  minSep: number,
  rng: () => number,
  /** Task 'confined-parcel' (2026-08-21): where a candidate may be drawn from; omitted is the whole
   * box, as before. A confined run MUST pass the parcel draw (soup/src/soup-confine.ts's
   * sampleInParcel): the accept rule below is "farthest from anything already there", and the vacuum
   * outside the parcel is farther from everything than any point inside it, so an unrestricted draw
   * would put every rehydrated bead in the vacuum on its first candidate, every time. */
  drawCandidate?: () => [number, number, number],
): { positions: Float32Array; shortOfFloor: number; minAchieved: number } {
  const nx = Math.max(1, Math.floor(box[0] / minSep))
  const ny = Math.max(1, Math.floor(box[1] / minSep))
  const nz = Math.max(1, Math.floor(box[2] / minSep))
  const wx = box[0] / nx
  const wy = box[1] / ny
  const wz = box[2] / nz
  const cells = new Map<number, number[]>()
  const xs: number[] = []
  const ys: number[] = []
  const zs: number[] = []
  const cellOf = (x: number, y: number, z: number): number => {
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = Math.min(nz - 1, Math.floor(z / wz))
    return cx + nx * (cy + ny * cz)
  }
  const push = (x: number, y: number, z: number): void => {
    const idx = xs.length
    xs.push(x)
    ys.push(y)
    zs.push(z)
    const key = cellOf(x, y, z)
    const b = cells.get(key)
    if (b) b.push(idx)
    else cells.set(key, [idx])
  }
  for (let i = 0; i < activeN; i++) {
    push(active[i * 4], active[i * 4 + 1], active[i * 4 + 2])
  }
  const nearest = (x: number, y: number, z: number): number => {
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = Math.min(nz - 1, Math.floor(z / wz))
    let best = Infinity
    for (let dx = -1; dx <= 1; dx++) {
      const ax = ((cx + dx) % nx + nx) % nx
      for (let dy = -1; dy <= 1; dy++) {
        const ay = ((cy + dy) % ny + ny) % ny
        for (let dz = -1; dz <= 1; dz++) {
          const az = ((cz + dz) % nz + nz) % nz
          const bucket = cells.get(ax + nx * (ay + ny * az))
          if (!bucket) continue
          for (const j of bucket) {
            let d2 = 0
            for (const [a, bb, L] of [
              [x, xs[j], box[0]],
              [y, ys[j], box[1]],
              [z, zs[j], box[2]],
            ] as [number, number, number][]) {
              const d = a - bb
              const m = d - Math.round(d / L) * L
              d2 += m * m
            }
            if (d2 < best) best = d2
          }
        }
      }
    }
    return Math.sqrt(best)
  }
  const positions = new Float32Array(count * 4)
  const min2 = minSep
  let shortOfFloor = 0
  let minAchieved = Infinity
  for (let k = 0; k < count; k++) {
    let bx = 0
    let by = 0
    let bz = 0
    let bd = -1
    for (let t = 0; t < INSERTION_CANDIDATES; t++) {
      const [x, y, z] = drawCandidate ? drawCandidate() : [rng() * box[0], rng() * box[1], rng() * box[2]]
      const d = nearest(x, y, z)
      if (d > bd) {
        bd = d
        bx = x
        by = y
        bz = z
      }
      if (d >= min2) break
    }
    if (bd < min2) shortOfFloor++
    if (bd < minAchieved) minAchieved = bd
    positions[k * 4] = bx
    positions[k * 4 + 1] = by
    positions[k * 4 + 2] = bz
    push(bx, by, bz)
  }
  return { positions, shortOfFloor, minAchieved: Number.isFinite(minAchieved) ? minAchieved : 0 }
}
