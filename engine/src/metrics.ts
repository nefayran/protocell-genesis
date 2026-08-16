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
    // z is NOT periodic in this engine (only x,y are wrapped), and nothing keeps a membrane's
    // center of mass pinned to the box midplane, so a bead really can walk out of [0, box_z).
    // Clamping it into the first or last bin — the previous behaviour — is silent and actively
    // dangerous: enough clamped beads manufacture a spike at an edge bin, and bilayerPeaks() below
    // will happily report that spike as a leaflet. Fail loudly instead; the caller either recentres
    // or has a real problem to report.
    if (!(z >= 0 && z < box[2])) {
      throw new Error(
        `densityProfileZ: бусина с z=${z} вне [0, ${box[2]}) — профиль по z не определён для ` +
          `вылетевших бусин (z не периодичен), клампить их в краевой бин нельзя: это создаёт ложный пик`,
      )
    }
    const b = Math.min(bins - 1, Math.floor(z / dz))
    if (positions[i + 3] === 0) head[b] += 1
    else tail[b] += 1
  }
  const z = Array.from({ length: bins }, (_, i) => (i + 0.5) * dz)
  return { z, head, tail }
}

/** Bin-by-bin sum of several profiles sharing one z grid — the accumulated histogram a thickness
 * estimator needs. A single snapshot puts ~500 head beads per leaflet into 0.2-sigma bins, which
 * (measured, task-5-report.md) scatters the peak-to-peak distance by roughly +/-0.3 sigma frame to
 * frame; summing many independent configurations first shrinks that scatter as 1/sqrt(samples)
 * instead of asking one frame to carry the whole answer. */
export function sumProfiles(profiles: ZProfile[]): ZProfile {
  if (profiles.length === 0) throw new Error('sumProfiles: пустой список профилей')
  const bins = profiles[0].z.length
  const head = new Array(bins).fill(0)
  const tail = new Array(bins).fill(0)
  for (const pr of profiles) {
    if (pr.z.length !== bins || pr.z[0] !== profiles[0].z[0]) {
      throw new Error('sumProfiles: профили построены на разных сетках по z — суммировать нельзя')
    }
    for (let i = 0; i < bins; i++) {
      head[i] += pr.head[i]
      tail[i] += pr.tail[i]
    }
  }
  return { z: profiles[0].z.slice(), head, tail }
}

/** Copy of `positions` with every z shifted so the mean bead z sits at the box midplane. The
 * membrane's center of mass drifts freely along the open z axis (measured about half a sigma over
 * one gate run), so histograms accumulated over many configurations would otherwise be smeared by
 * that drift rather than by the physics. Shifting is not wrapping: z has no periodic image, and a
 * membrane far enough off-centre that recentring pushes beads out of the box is a real problem for
 * densityProfileZ() to report. */
export function centerMembraneZ(positions: Float32Array, box: [number, number, number]): Float32Array {
  const n = positions.length / 4
  let sum = 0
  for (let i = 0; i < n; i++) sum += positions[i * 4 + 2]
  const shift = box[2] / 2 - sum / n
  const out = positions.slice()
  for (let i = 0; i < n; i++) out[i * 4 + 2] += shift
  return out
}

/** Splits `positions` into the beads that lie inside [0, box_z) and a count of those that do not.
 * Lipids do occasionally evaporate off a solvent-free membrane into the vacuum above or below it,
 * and since z has no periodic image they then walk away for good (measured: one lipid, i.e. 3 beads
 * of 3000, left the box during a 2000-move gate run). Such beads are no longer part of the bilayer,
 * so they must not enter its density profile — but dropping them silently is how a profile starts
 * lying, hence the count comes back with the data for the caller to bound. */
export function dropEscapedZ(
  positions: Float32Array,
  box: [number, number, number],
): { positions: Float32Array; escaped: number } {
  const n = positions.length / 4
  const kept: number[] = []
  let escaped = 0
  for (let i = 0; i < n; i++) {
    const z = positions[i * 4 + 2]
    if (z >= 0 && z < box[2]) {
      kept.push(positions[i * 4], positions[i * 4 + 1], z, positions[i * 4 + 3])
    } else {
      escaped++
    }
  }
  return { positions: new Float32Array(kept), escaped }
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
  if (head[a] === 0) throw new Error('bilayerPeaks: профиль голов пуст — пиков нет')
  const masked = head.map((v, i) => (Math.abs(z[i] - z[a]) <= minSeparation ? 0 : v))
  let b = 0
  for (let i = 1; i < masked.length; i++) if (masked[i] > masked[b]) b = i
  // An all-zero masked profile means every head bead sits within one sigma of the single maximum:
  // there is no second leaflet to measure. Returning z[0] (what an unguarded argmax does) would
  // hand the caller the box's bottom edge dressed up as a peak position — a silently wrong
  // thickness instead of a failure.
  if (masked[b] === 0) {
    throw new Error(
      `bilayerPeaks: второй пик не найден — все головы лежат в пределах ${minSeparation} от ` +
        `максимума при z=${z[a]}; это не бислой`,
    )
  }
  const za = subBinPeak(head, z, a)
  const zb = subBinPeak(head, z, b)
  return za <= zb ? { lower: za, upper: zb } : { lower: zb, upper: za }
}

/** Peak position refined below bin width by fitting a parabola through the peak bin and its two
 * neighbours — the standard three-point interpolation. Without it the estimator can only ever
 * return a bin centre, so with 0.2-sigma bins the thickness is quantised in 0.2-sigma steps and a
 * true value near a bound reads as exactly on it. Falls back to the bin centre when the three
 * points do not form a downward parabola (flat top, or the peak sitting on the profile's edge). */
function subBinPeak(values: number[], z: number[], i: number): number {
  if (i === 0 || i === values.length - 1 || z.length < 2) return z[i]
  const dz = z[1] - z[0]
  const denom = values[i - 1] - 2 * values[i] + values[i + 1]
  if (denom >= 0) return z[i]
  const offset = (0.5 * (values[i - 1] - values[i + 1])) / denom
  return z[i] + Math.max(-0.5, Math.min(0.5, offset)) * dz
}

/** Distance between the two head-density peaks (the two leaflets) — see bilayerPeaks() above for
 * how they are found. */
export function bilayerThickness(profile: ZProfile): number {
  const { lower, upper } = bilayerPeaks(profile)
  return upper - lower
}
