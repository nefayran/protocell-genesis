// Structural metrics for the bilayer gate (Task 5): area per lipid and bilayer thickness from the
// z-density profile of head beads. Pure CPU math — no GPU calls — so these are unit-testable
// without a browser/WebGPU context. The GPU-touching half (the zero-tension Monte Carlo area move)
// lives on System in sim.ts, since it needs the live position/box buffers.

import { loadParams } from './params'

export function areaPerLipid(box: [number, number, number], lipids: number): number {
  return (box[0] * box[1]) / (lipids / 2)
}

export interface ZProfile {
  z: number[]
  head: number[]
  tail: number[]
}

/** Histograms bead z-coordinates into `bins` bins spanning [0, box[2]), split by bead type (w
 * component: 0 = head, nonzero = tail). `positions` is the flat vec4-per-bead layout used
 * throughout the engine (x, y, z, type). */
export function densityProfileZ(positions: Float32Array, box: [number, number, number], bins: number): ZProfile {
  const dz = box[2] / bins
  const head = new Array(bins).fill(0)
  const tail = new Array(bins).fill(0)
  for (let i = 0; i < positions.length; i += 4) {
    const z = positions[i + 2]
    const b = Math.min(bins - 1, Math.max(0, Math.floor(z / dz)))
    if (positions[i + 3] === 0) head[b] += 1
    else tail[b] += 1
  }
  const z = Array.from({ length: bins }, (_, i) => (i + 0.5) * dz)
  return { z, head, tail }
}

/** The two most prominent head-density maxima, at least one sigma apart. Splitting the profile at
 * the box's geometric midpoint and taking an argmax on each side (the original approach) silently
 * breaks if the membrane's center of mass has drifted: both true peaks can land in the same half
 * (so one side's argmax is a shoulder, not a peak), and nothing in this engine keeps a bilayer
 * centered in z — only x,y are periodic/wrapped; z is open. This finder instead takes the global
 * maximum, zeroes a +/-1 sigma window around it (so it cannot re-select a shoulder of the same
 * peak), and takes the next maximum in what remains — robust to wherever the membrane actually
 * sits. Measured on a real run (task-5-report.md) that the two approaches agree when the profile
 * is genuinely bimodal with well-separated peaks; this finder is what stays correct when it isn't. */
export function bilayerPeaks(profile: ZProfile): { lower: number; upper: number } {
  const { head, z } = profile
  const minSeparation = loadParams().sigma // 1 in these reduced units; the two peaks must be at
  // least this far apart to count as two leaflets rather than one peak and its own shoulder.
  let a = 0
  for (let i = 1; i < head.length; i++) if (head[i] > head[a]) a = i
  const masked = head.map((v, i) => (Math.abs(z[i] - z[a]) <= minSeparation ? 0 : v))
  let b = 0
  for (let i = 1; i < masked.length; i++) if (masked[i] > masked[b]) b = i
  return z[a] <= z[b] ? { lower: z[a], upper: z[b] } : { lower: z[b], upper: z[a] }
}

/** Distance between the two head-density peaks (the two leaflets) — see bilayerPeaks() above for
 * how they are found. */
export function bilayerThickness(profile: ZProfile): number {
  const { lower, upper } = bilayerPeaks(profile)
  return upper - lower
}
