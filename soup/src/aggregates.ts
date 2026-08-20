// soup-to-vesicle, per-aggregate analysis (task-3c/per-aggregate-report): corrects two defects in
// how soup/src/stages.ts used to decide the micelles/bilayer/vesicle stages --
//  1. `largestAggregateFraction >= aggregationLow` gated `micelles` on ONE dominant aggregate, the
//     OPPOSITE of a micellar state (matter deliberately split among MANY small aggregates). A real
//     dilute run (box 50sigma, 14200 particles, 200000 steps) showed ~15 visibly discrete, round
//     aggregates on screen -- textbook micelles -- and was reported as `amphiphiles` because that
//     one number never exceeded 0.3.
//  2. headPeaks (soup/src/stages.ts's computeHeadPeaks, moved into this file verbatim -- see below)
//     is a BOX-WIDE z-density profile: meaningful only for a planar bilayer spanning the periodic
//     box. For discrete micelles/vesicles there is no global "up" and "down", so it is noise, and it
//     used to gate the bilayer stage anyway.
//
// The fix computed here is PER AGGREGATE, not box-wide: cluster the amphiphile-member particles
// (clusterComponents(), engine/src/aggregate.ts -- the SAME tail-contact cutoff/connectivity rule
// detectStage's own largestAggregateFraction already used, just broken into every component instead
// of only the largest), then for the largest few measure genuine shape (gyration tensor + principal
// moments, engine/src/aggregate.ts's shapeOf()) and head-vs-tail layering read off that AGGREGATE's
// OWN centre (radially, for a curved micelle/vesicle) or own flat axis (transversely, for a lamellar
// patch -- unsigned radial distance folds a centred flat sheet's two faces onto the SAME peak, so it
// cannot see two layers there; see profileAlong()'s own header), plus whether it encloses a cavity by
// flooding only that aggregate's own padded local bounding region (not the whole simulation box).
//
// No numeric model constant is written here (the project-wide rule, tests/params.test.ts's literal
// scanner) -- every threshold this file reads comes from data/soup.json's stageThresholds via the
// AggregateThresholds fields soup/src/stages.ts's loadStageThresholds() already validates.
//
// computeHeadPeaks()/HEAD_PEAKS_UNAVAILABLE/memberIndicesOf()/positionsFor() were moved here VERBATIM
// from soup/src/stages.ts (task-3b's own defect-2 fix): this module needs computeHeadPeaks for its
// own per-aggregate radial/transverse profiles (see profileAlong()), and stages.ts needs it for its
// existing box-wide headPeaks diagnostic -- putting it here and having stages.ts import+re-export it
// is the only way to share one implementation without a stages.ts<->aggregates.ts import cycle
// (stages.ts already needs to import analyzeAggregates() from this file). Behaviour is unchanged --
// tests/soup-amphiphile.test.ts's defect-2 tests import computeHeadPeaks from stages.ts (unchanged
// import path, via stages.ts's re-export) and pass exactly as before.

import { clusterComponents, shapeOf, unwrapAggregate } from '../../engine/src/aggregate'
import { dimsFor, enclosedVolume, occupancy } from '../../engine/src/closure'
import { bilayerThickness, densityProfileZ, dropEscapedZ } from '../../engine/src/metrics'
import { loadParams, wcaCutoff } from '../../engine/src/params'
import rawLiterature from '../../data/literature.json'
import periodicMeasurementParams from '../../data/periodic-measurement.json'
import type { Amphiphile } from './amphiphile'
import { loadSoup, type Monomer } from './rules'
import { encapsulatedWaterVolume, periodicCentreOf, type EncapsulatedWaterResult } from './water-closure'

/** Sentinel returned by computeHeadPeaks() (box-wide) and by this file's own per-aggregate
 * radial/transverse peak counts when the underlying profile does not carry enough particles to
 * trust ANY peak count read off it -- see computeHeadPeaks()'s own doc comment below. */
export const HEAD_PEAKS_UNAVAILABLE = 'unavailable' as const

/** The four box-wide-profile fields computeHeadPeaks() needs -- a `Pick` so both the box-wide caller
 * (soup/src/stages.ts, headDensityBins/minHeadsPerBin) and this file's own per-aggregate callers
 * (aggregateHeadDensityBins/aggregateMinHeadsPerBin, renamed into this shape at the call site) can
 * share one function without either side owning the other's field names. */
export interface HeadPeakThresholds {
  headDensityBins: number
  minHeadsPerBin: number
  headPeakSeparationMin: number
  headPeakSeparationMax: number
}

/** How many head-density peaks to trust, given a snapshot's raw particle array and a z-like axis
 * (box-wide z for soup/src/stages.ts's own diagnostic; a synthetic per-aggregate radial/transverse
 * coordinate, packed into the same "z" slot by profileAlong() below, for this file's own per-
 * aggregate shells). Pure CPU math (densityProfileZ/bilayerThickness are both pure, no GPU) --
 * unit-testable on synthetic `particles` with no browser/WebGPU context.
 *
 * Three-way result, never a bare "2" on faith:
 *  - 0: no polar particles in this snapshot/profile at all -- nothing to profile.
 *  - HEAD_PEAKS_UNAVAILABLE: there ARE polar particles, but too few of them per bin, on average, for
 *    a peak count read off this histogram to mean anything (Poisson binning noise misread as
 *    structure -- see task-3b-report.md for the box-wide pilot case this guard was built for: 267
 *    heads / 60 bins ~= 4.45/bin reported two peaks at step 0, zero bonds, zero structure).
 *  - 1: enough particles per bin to trust the count, but either bilayerThickness() found no second
 *    peak, or it found one whose separation falls outside the plausible band.
 *  - 2: enough particles per bin, AND two peaks, AND their separation is inside the plausible band --
 *    the only value a caller may treat as a real two-layer signal.
 *
 * Drops any particle densityProfileZ would throw on (dropEscapedZ) before profiling -- a live
 * snapshot's z (or, per-aggregate, an unwrapped-but-still-synthetic coordinate) can legitimately walk
 * outside [0, box[2]), and densityProfileZ is right to refuse rather than manufacture a false edge
 * peak by clamping (see that function's own doc comment, engine/src/metrics.ts). */
export function computeHeadPeaks(
  particles: Float32Array,
  box: [number, number, number],
  monomers: Monomer[],
  thresholds: HeadPeakThresholds,
): number | typeof HEAD_PEAKS_UNAVAILABLE {
  const n = particles.length / 4
  let totalHeads = 0
  for (let i = 0; i < n; i++) {
    if (monomers[Math.round(particles[i * 4 + 3])]?.polar) totalHeads++
  }
  if (totalHeads === 0) return 0

  const remapped = particles.slice()
  for (let i = 0; i < n; i++) {
    remapped[i * 4 + 3] = monomers[Math.round(particles[i * 4 + 3])]?.polar ? 0 : 1
  }
  const { positions: inBounds } = dropEscapedZ(remapped, box)
  let keptHeads = 0
  const m = inBounds.length / 4
  for (let i = 0; i < m; i++) if (inBounds[i * 4 + 3] === 0) keptHeads++
  if (keptHeads === 0) return HEAD_PEAKS_UNAVAILABLE // every head escaped -- nothing left to profile

  const avgHeadsPerBin = keptHeads / thresholds.headDensityBins
  if (avgHeadsPerBin < thresholds.minHeadsPerBin) return HEAD_PEAKS_UNAVAILABLE

  const profile = densityProfileZ(inBounds, box, thresholds.headDensityBins)
  try {
    const separation = bilayerThickness(profile) // throws => no second peak at all; caught below
    if (separation >= thresholds.headPeakSeparationMin && separation <= thresholds.headPeakSeparationMax) return 2
    return 1 // two maxima exist, but not a plausible bilayer separation -- not trusted as a bilayer
  } catch {
    return 1
  }
}

/** Every particle index that belongs to a recognised amphiphile (its head or any bead of its
 * chain) -- the vesicle wall's own material, as opposed to free monomers/donors/catalysts drifting
 * through the box. */
export function memberIndicesOf(amphiphiles: Amphiphile[]): Set<number> {
  const memberIdx = new Set<number>()
  for (const a of amphiphiles) {
    memberIdx.add(a.headIndex)
    for (const c of a.chain) memberIdx.add(c)
  }
  return memberIdx
}

/** `particles`, filtered down to just the indices in `idx` -- the flat vec4-per-particle layout this
 * file's own clustering/shape functions expect. */
export function positionsFor(particles: Float32Array, idx: Set<number>): Float32Array {
  const idxArr = Array.from(idx)
  const out = new Float32Array(idxArr.length * 4)
  for (let k = 0; k < idxArr.length; k++) out.set(particles.subarray(idxArr[k] * 4, idxArr[k] * 4 + 4), k * 4)
  return out
}

// --- per-aggregate thresholds ------------------------------------------------------------------

/** The per-aggregate fields of data/soup.json's stageThresholds -- see that file's `basis` for each
 * one's derivation. A `Pick`-friendly interface (soup/src/stages.ts's StageThresholds satisfies it
 * structurally) so this module never needs to import StageThresholds itself (which would reach back
 * into stages.ts and create the very import cycle this file's header explains avoiding). */
export interface AggregateThresholds {
  minAmphiphilesPerAggregate: number
  minMicelleAggregates: number
  minAmphiphileShareInAggregates: number
  lamellarFlatnessRatio: number
  lamellarInPlaneSymmetryMin: number
  aggregateHeadDensityBins: number
  aggregateMinHeadsPerBin: number
  headPeakSeparationMin: number
  headPeakSeparationMax: number
  /** The vesicle-closure gate's own existing minimum (data/soup.json / data/literature.json,
   * literally the same field/value stages.ts's loadStageThresholds() already cross-checks) --
   * reused, per the task's own instruction, not re-derived. */
  enclosedVolume: number
  closureCell: number
  closureRadius: number
  detailAggregateCount: number
}

/** Re-derives minAmphiphilesPerAggregate from first principles, so loadStageThresholds()
 * (soup/src/stages.ts) can assert data/soup.json's own copy has not silently drifted from this
 * formula -- the same pattern that file already uses for enclosedVolume vs data/literature.json's
 * closure.target.min. Derivation (data/soup.json's own stageThresholds.basis carries the same text):
 * the number of amphiphiles needed to fully cover a SPHERE of radius "one tail length" at the
 * literature-calibrated area-per-lipid gate's own midpoint.
 *  - "one tail length": this soup's own carbon bead's WCA contact radius (wcaCutoff(sigma*
 *    carbon.radiusSigma)) -- the physical reach of a single tail SEGMENT, chosen over the emergent
 *    chain-length distribution itself (mode 1 carbon per chain, per data/soup.json's own startBasis)
 *    precisely because that distribution is a RESULT of the run, not a constant a threshold should
 *    be built from.
 *  - area per lipid: the midpoint of data/literature.json's area-per-lipid gate's own [min,max]
 *    target range (Cooke & Deserno 2005, rank A) -- an already-literature-anchored number, not a
 *    new assumption.
 *  - N_min = ceil(4*pi*r^2 / a_mid). */
export function derivedMinAmphiphilesPerAggregate(): number {
  const p = loadParams()
  const soup = loadSoup()
  const carbon = soup.monomers.find((m) => m.kind === 'carbon')
  if (!carbon) throw new Error('data/soup.json: не найден мономер вида carbon')
  const tailLength = wcaCutoff(p.sigma * carbon.radiusSigma)
  const gates = (rawLiterature as { gates: Array<{ id: string; target: { min?: number; max?: number } }> }).gates
  const gate = gates.find((g) => g.id === 'area-per-lipid')
  if (!gate || gate.target.min === undefined || gate.target.max === undefined) {
    throw new Error('data/literature.json: отсутствует area-per-lipid.target.min/max')
  }
  const areaPerLipidMid = (gate.target.min + gate.target.max) / 2
  const sphereArea = 4 * Math.PI * tailLength * tailLength
  return Math.ceil(sphereArea / areaPerLipidMid)
}

// --- per-aggregate report ------------------------------------------------------------------------

export interface AggregateShape {
  particleCount: number
  amphiphileCount: number
  centre: [number, number, number]
  radiusOfGyration: number
  /** Ascending gyration-tensor eigenvalues -- see engine/src/aggregate.ts's shapeOf() for the shape
   * these ratios read off (sphere ~= all equal; oblate/lamellar has [0] << [1]~=[2]; prolate/rod has
   * [0]~=[1] << [2]). */
  principalMoments: [number, number, number]
  /** principalMoments[0]/principalMoments[2] -- near 0 for a flat sheet/shell wall, ~1 for a sphere. */
  flatnessRatio: number
  /** principalMoments[1]/principalMoments[2] -- near 1 for a flat sheet extended roughly equally in
   * its own plane (or a sphere); well below 1 for a rod/ribbon (also has a small [0], but with [1]
   * and [2] unequal too -- this ratio is what tells the two apart). */
  inPlaneSymmetry: number
  /** Head-vs-tail layering off UNSIGNED radial distance from this aggregate's own centre -- right
   * for a curved object (micelle: one head shell outside a tail core; vesicle: an inner AND outer
   * head shell reads as two peaks at two DIFFERENT radii). */
  radialHeadShells: number | typeof HEAD_PEAKS_UNAVAILABLE
  /** Head-vs-tail layering off the SIGNED coordinate along this aggregate's own smallest-principal-
   * moment axis -- right for a centred flat/lamellar patch, where the two faces sit at the SAME
   * unsigned radial distance from the centroid (radialHeadShells folds them onto one peak there) but
   * at opposite signed positions along the flat axis. Only computed (and only meaningful) when
   * flatnessRatio/inPlaneSymmetry already read as oblate -- 0 otherwise, since there is no flat axis
   * to project onto for a shape that is not flat. */
  transverseHeadShells: number | typeof HEAD_PEAKS_UNAVAILABLE
  /** Enclosed cavity volume from flooding ONLY this aggregate's own padded local bounding region
   * (localCavityVolume() below) -- 0 when nothing is closed, the normal case for a micelle or a
   * lamellar patch. This is the VACUUM-cavity detector (counts empty grid cells) -- kept exactly as
   * it was; see `encapsulatedWater` below for what replaces it once water particles exist (task
   * 'explicit-water', 2026-08-18). */
  cavityVolume: number
  /** Task 'explicit-water' (2026-08-18): water particles trapped inside this aggregate (cannot
   * reach the bulk), measured by soup/src/water-closure.ts's own periodic (never-unwrap) flood --
   * see that module's header for why counting empty CELLS (cavityVolume above) is the wrong measure
   * once water exists (occupancy() never marks a cell as occupied just because water sits in it, so
   * a water-filled interior reads as "empty" to that detector exactly as true vacuum would).
   * `undefined` when the caller did not supply any water-particle indices to analyzeAggregates()
   * (e.g. every existing call site that predates this task, or a real run with data/soup.json's `W`
   * count at 0) -- isVesicleShape() below falls back to the vacuum-cavity check in that case, so
   * every existing caller/test keeps seeing byte-identical behaviour. `null` when water indices
   * WERE supplied but this aggregate's own periodic centre could not be trusted (see
   * soup/src/water-closure.ts's periodicCentreOf -- an axis with too little positional
   * concentration to support a centre at all), which is a real "cannot measure this", not a zero. */
  encapsulatedWater?: EncapsulatedWaterResult | null
}

export interface AggregateAnalysis {
  aggregateCount: number
  /** Amphiphile count per aggregate, largest-particle-count-first (clusterComponents()'s own
   * ordering) -- the size histogram the UI shows. */
  sizeHistogram: number[]
  /** Full shape/head-shell/cavity detail for the `detailAggregateCount` largest aggregates -- see
   * clusterComponents()'s own doc comment (engine/src/aggregate.ts) for why the stage-deciding
   * lamellar/vesicle candidate, holding most of the system's own amphiphile material, cannot be
   * missed by this cutoff. */
  aggregates: AggregateShape[]
  /** Aggregates whose own amphiphileCount clears minAmphiphilesPerAggregate -- the micelle-stage
   * "real object, not a transient pair" gate. */
  qualifyingAggregateCount: number
  amphiphilesInQualifying: number
  /** amphiphilesInQualifying / total recognised amphiphiles (0 when there are none at all). */
  amphiphileShareInQualifying: number
  /** True iff some detailed aggregate's shape reads as flat/lamellar AND its own radial-or-
   * transverse head profile shows two layers -- the bilayer stage's own per-aggregate condition. */
  hasLamellarAggregate: boolean
  /** True iff some detailed aggregate's own radial head profile shows two shells AND it encloses a
   * cavity clearing the existing physically-derived minimum -- the vesicle stage's own per-aggregate
   * condition. */
  hasVesicleAggregate: boolean
}

/** Bilayer's own shape gate: flat (principalMoments[0] well below the other two) AND extended
 * roughly equally in its own plane (principalMoments[1]~=[2]), as opposed to a rod/ribbon (also has
 * a small principalMoments[0], but principalMoments[1] would be well below [2] too). */
export function isLamellarShape(
  shape: Pick<AggregateShape, 'flatnessRatio' | 'inPlaneSymmetry'>,
  t: Pick<AggregateThresholds, 'lamellarFlatnessRatio' | 'lamellarInPlaneSymmetryMin'>,
): boolean {
  return shape.flatnessRatio <= t.lamellarFlatnessRatio && shape.inPlaneSymmetry >= t.lamellarInPlaneSymmetryMin
}

/** Bilayer stage condition on one already-analysed aggregate: lamellar shape AND a trusted two-layer
 * head reading off EITHER axis (radial or transverse -- the task's own wording; see
 * transverseHeadShells' own doc comment for why a centred flat sheet needs the transverse axis while
 * a curved object needs the radial one, and either can be the one that actually shows it). */
export function isBilayerShape(
  shape: Pick<AggregateShape, 'flatnessRatio' | 'inPlaneSymmetry' | 'radialHeadShells' | 'transverseHeadShells'>,
  t: Pick<AggregateThresholds, 'lamellarFlatnessRatio' | 'lamellarInPlaneSymmetryMin'>,
): boolean {
  return isLamellarShape(shape, t) && (shape.radialHeadShells === 2 || shape.transverseHeadShells === 2)
}

/** Vesicle stage condition on one already-analysed aggregate: two head SHELLS (inner + outer, read
 * off the radial profile -- a vesicle has no meaningful flat axis to project onto, unlike a bilayer
 * patch) AND an enclosed interior.
 *
 * WHICH CLOSURE CHECK, task 'explicit-water' (2026-08-18): when the caller supplied water-particle
 * indices to analyzeAggregates() (`shape.encapsulatedWater` is present), closure is decided by
 * ENCAPSULATED WATER (soup/src/water-closure.ts) -- the physically correct question once a solvent
 * exists (an enclosed interior full of water at the same chemical potential as the outside costs
 * nothing, so counting EMPTY grid cells, as `cavityVolume` does, would call a water-filled vesicle
 * unclosed). `shape.encapsulatedWater` being `null` (present but this snapshot's own periodic centre
 * could not be trusted) is treated as NOT closed, not as "fall back to the old check" -- a
 * measurement that could not be taken is not evidence of closure. When no water indices were
 * supplied at all (`encapsulatedWater` is `undefined` -- every pre-existing caller, or a real run
 * with zero water particles), this is EXACTLY the old vacuum-cavity check
 * (`cavityVolume > enclosedVolume`), unchanged -- see AggregateThresholds.enclosedVolume's own doc
 * comment for that minimum's own derivation, kept per this task's instruction not to touch it. */
export function isVesicleShape(
  shape: Pick<AggregateShape, 'radialHeadShells' | 'cavityVolume'> & Partial<Pick<AggregateShape, 'encapsulatedWater'>>,
  t: Pick<AggregateThresholds, 'enclosedVolume'>,
): boolean {
  if (shape.radialHeadShells !== 2) return false
  if (shape.encapsulatedWater !== undefined) return shape.encapsulatedWater !== null && shape.encapsulatedWater.closed
  return shape.cavityVolume > t.enclosedVolume
}

/** Packs a scalar-per-member coordinate (`coordFor`) and each member's own original kind index
 * (`kindOf`) into the same flat vec4-per-particle layout computeHeadPeaks() expects, with the
 * coordinate written into the "z" slot and a throwaway [1,1,span] "box" -- the same "adapt the
 * encoding at the call site, don't fork the function" pattern this project already leans on (e.g.
 * soup/src/stages.ts's own detectStage() remaps particles' kind index into a binary head/tail flag
 * inline before calling largestClusterFraction). `span` is shifted/padded so every coordinate lands
 * strictly inside
 * [0, box[2]) (densityProfileZ throws otherwise, by design -- see that function's own doc comment). */
function profileAlong(coordFor: (i: number) => number, kindOf: (i: number) => number, count: number): { particles: Float32Array; box: [number, number, number] } {
  const particles = new Float32Array(count * 4)
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < count; i++) {
    const c = coordFor(i)
    if (c < lo) lo = c
    if (c > hi) hi = c
  }
  const span = Math.max(1e-6, (hi - lo) * 1.0001)
  for (let i = 0; i < count; i++) {
    particles[i * 4 + 2] = coordFor(i) - lo
    particles[i * 4 + 3] = kindOf(i)
  }
  return { particles, box: [1, 1, span] }
}

/** Floods ONLY this aggregate's own padded local bounding region -- not the whole simulation box --
 * for an enclosed cavity, reusing occupancy()/enclosedVolume()/dimsFor() (engine/src/closure.ts)
 * exactly as they already are (open-boundary flood, no periodicity assumed within this local grid).
 * `positions` must already be one coherent, unwrapped local frame (unwrapAggregate()'s own output).
 * `pad` (in the same reduced sigma units as `radius`) is chosen generously above `radius` so
 * occupancy()'s own x,y modulo wrap (built for a genuinely periodic simulation box, irrelevant here)
 * never wraps around near a real bead -- every bead sits at least `pad` cells from this LOCAL grid's
 * own edge -- and so the flood always has real open exterior space to mark before it could ever
 * reach that edge. */
function localCavityVolume(positions: Float32Array, cell: number, radius: number): number {
  const n = positions.length / 4
  if (n === 0) return 0
  let minX = Infinity, maxX = -Infinity
  let minY = Infinity, maxY = -Infinity
  let minZ = Infinity, maxZ = -Infinity
  for (let i = 0; i < n; i++) {
    const x = positions[i * 4], y = positions[i * 4 + 1], z = positions[i * 4 + 2]
    if (x < minX) minX = x
    if (x > maxX) maxX = x
    if (y < minY) minY = y
    if (y > maxY) maxY = y
    if (z < minZ) minZ = z
    if (z > maxZ) maxZ = z
  }
  const pad = radius * 4
  const shiftX = pad - minX
  const shiftY = pad - minY
  const shiftZ = pad - minZ
  const localBox: [number, number, number] = [maxX - minX + 2 * pad, maxY - minY + 2 * pad, maxZ - minZ + 2 * pad]
  const shifted = new Float32Array(positions.length)
  for (let i = 0; i < n; i++) {
    shifted[i * 4] = positions[i * 4] + shiftX
    shifted[i * 4 + 1] = positions[i * 4 + 1] + shiftY
    shifted[i * 4 + 2] = positions[i * 4 + 2] + shiftZ
    shifted[i * 4 + 3] = positions[i * 4 + 3]
  }
  const occ = occupancy(shifted, localBox, cell, radius)
  return enclosedVolume(occ, dimsFor(localBox, cell), cell)
}

/** Full shape/head-shell/cavity analysis of ONE aggregate, given as the ORIGINAL particle indices of
 * its members (in any order). `particles`/`box` are the live snapshot's own raw (periodically
 * wrapped) arrays -- this function unwraps just this aggregate's own members before measuring
 * anything, so a member straddling a periodic face never inflates the gyration tensor or the
 * radial/transverse profiles (see engine/src/aggregate.ts's unwrapAggregate() header for why the
 * fully-periodic soup engine needs this, unlike the membrane engine's open-z convention). */
function shapeOfAggregate(
  originalIdx: number[],
  amphiphileCount: number,
  particles: Float32Array,
  box: [number, number, number],
  monomers: Monomer[],
  t: AggregateThresholds,
  waterIdx?: readonly number[],
): AggregateShape {
  const raw = new Float32Array(originalIdx.length * 4)
  for (let k = 0; k < originalIdx.length; k++) {
    raw.set(particles.subarray(originalIdx[k] * 4, originalIdx[k] * 4 + 4), k * 4)
  }
  const unwrapped = unwrapAggregate(raw, box, [true, true, true])
  const shape = shapeOf(unwrapped)
  const [l0, l1, l2] = shape.principalMoments
  const flatnessRatio = l2 > 1e-12 ? l0 / l2 : 1
  const inPlaneSymmetry = l2 > 1e-12 ? l1 / l2 : 1

  const n = unwrapped.length / 4
  const kindOf = (i: number) => unwrapped[i * 4 + 3]
  const headThresholds: HeadPeakThresholds = {
    headDensityBins: t.aggregateHeadDensityBins,
    minHeadsPerBin: t.aggregateMinHeadsPerBin,
    headPeakSeparationMin: t.headPeakSeparationMin,
    headPeakSeparationMax: t.headPeakSeparationMax,
  }

  const radialCoord = (i: number): number => {
    const dx = unwrapped[i * 4] - shape.centre[0]
    const dy = unwrapped[i * 4 + 1] - shape.centre[1]
    const dz = unwrapped[i * 4 + 2] - shape.centre[2]
    return Math.sqrt(dx * dx + dy * dy + dz * dz)
  }
  const radial = profileAlong(radialCoord, kindOf, n)
  const radialHeadShells = computeHeadPeaks(radial.particles, radial.box, monomers, headThresholds)

  // Transverse (signed, along the flat axis) profile: only worth computing once the shape already
  // reads as oblate -- for a sphere/rod there is no flat axis, and shapeOf()'s own smallestAxis is
  // numerically arbitrary in that case (see its doc comment).
  let transverseHeadShells: number | typeof HEAD_PEAKS_UNAVAILABLE = 0
  if (isLamellarShape({ flatnessRatio, inPlaneSymmetry }, t)) {
    const axis = shape.smallestAxis
    const transverseCoord = (i: number): number => {
      const dx = unwrapped[i * 4] - shape.centre[0]
      const dy = unwrapped[i * 4 + 1] - shape.centre[1]
      const dz = unwrapped[i * 4 + 2] - shape.centre[2]
      return dx * axis[0] + dy * axis[1] + dz * axis[2]
    }
    const transverse = profileAlong(transverseCoord, kindOf, n)
    transverseHeadShells = computeHeadPeaks(transverse.particles, transverse.box, monomers, headThresholds)
  }

  // Task 'explicit-water' (2026-08-18): computed on the ORIGINAL, still-periodically-wrapped
  // `particles`/`box` and this aggregate's own `originalIdx` -- deliberately NOT on `unwrapped`
  // above, since the whole point of the periodic (never-unwrap) method is to stay correct even when
  // this aggregate's own extent is comparable to the box (periodic-measurement-report.md's own
  // finding for this project's largest campaign aggregate), a regime where unwrapAggregate()'s own
  // single-reference-particle walk is exactly what can alias. `waterIdx` absent/empty -> undefined,
  // so isVesicleShape() falls back to the vacuum-cavity check untouched (see that function's own
  // doc comment).
  let encapsulatedWater: AggregateShape['encapsulatedWater']
  if (waterIdx && waterIdx.length > 0) {
    const alpha = periodicMeasurementParams.circularConcentration.alpha
    const centre = periodicCentreOf(particles, originalIdx, box, alpha)
    encapsulatedWater = centre
      ? encapsulatedWaterVolume(particles, originalIdx, waterIdx, box, centre, t.closureCell, t.closureRadius, t.enclosedVolume)
      : null
  }

  return {
    particleCount: n,
    amphiphileCount,
    centre: shape.centre,
    radiusOfGyration: shape.radiusOfGyration,
    principalMoments: shape.principalMoments,
    flatnessRatio,
    inPlaneSymmetry,
    radialHeadShells,
    transverseHeadShells,
    cavityVolume: localCavityVolume(unwrapped, t.closureCell, t.closureRadius),
    encapsulatedWater,
  }
}

/** The all-zero/empty AggregateAnalysis -- what analyzeAggregates() itself returns when there are no
 * amphiphiles or no connected aggregates among them, and what a caller resetting a run's UI state
 * (viewer/run.ts) uses so there is exactly one definition of "nothing analysed yet" rather than two
 * hand-written copies that could drift apart. */
export function emptyAggregateAnalysis(): AggregateAnalysis {
  return {
    aggregateCount: 0,
    sizeHistogram: [],
    aggregates: [],
    qualifyingAggregateCount: 0,
    amphiphilesInQualifying: 0,
    amphiphileShareInQualifying: 0,
    hasLamellarAggregate: false,
    hasVesicleAggregate: false,
  }
}

/** The full per-aggregate breakdown for one snapshot: clusters the amphiphile-member particles
 * (`memberIdx`/`particles` restricted to it, via clusterComponents() -- the SAME connectivity rule
 * `cutoff` already defines for detectStage's own largestAggregateFraction, just broken into every
 * component instead of collapsed to the largest), counts amphiphiles per aggregate (an amphiphile's
 * whole bonded chain sits within `cutoff` of itself by construction -- data/params.json's own FENE
 * rInf (loaded, never hardcoded here) is well under this soup's tail-tail attraction cutoff -- so
 * checking headIndex membership alone correctly attributes the WHOLE amphiphile to one aggregate),
 * then measures full shape/head-
 * shell/cavity detail for the `detailAggregateCount` largest. */
export function analyzeAggregates(
  particles: Float32Array,
  box: [number, number, number],
  monomers: Monomer[],
  amphiphiles: Amphiphile[],
  memberIdx: Set<number>,
  cutoff: number,
  thresholds: AggregateThresholds,
  /** Task 'explicit-water' (2026-08-18): indices of every water-species particle in `particles`,
   * or omitted entirely -- every pre-existing call site omits this and sees byte-identical
   * behaviour (see AggregateShape.encapsulatedWater's own doc comment). soup/src/stages.ts's
   * detectStage() is the one caller that supplies it, from data/soup.json's own solvent.waterId. */
  waterIdx?: readonly number[],
): AggregateAnalysis {
  if (memberIdx.size === 0) return emptyAggregateAnalysis()

  const idxArr = Array.from(memberIdx)
  const memberPositions = positionsFor(particles, memberIdx)
  // `true` = z is periodic: the soup wraps all three axes every step (soup/src/sim.ts), so leaving
  // z open here SPLIT every z-straddling aggregate and counted it twice -- coalescence-report.md §6
  // measured that at +11.9 % on the largest aggregate of the decisive box-54 lineage.
  const components = clusterComponents(memberPositions, box, cutoff, true)
  if (components.length === 0) return emptyAggregateAnalysis()

  const prepared = components.map((local) => {
    const originalIdx = local.map((k) => idxArr[k])
    const origSet = new Set(originalIdx)
    let amphiphileCount = 0
    for (const a of amphiphiles) if (origSet.has(a.headIndex)) amphiphileCount++
    return { originalIdx, amphiphileCount }
  })

  const sizeHistogram = prepared.map((p) => p.amphiphileCount)
  const totalAmphiphiles = amphiphiles.length
  const qualifying = prepared.filter((p) => p.amphiphileCount >= thresholds.minAmphiphilesPerAggregate)
  const amphiphilesInQualifying = qualifying.reduce((s, p) => s + p.amphiphileCount, 0)
  const amphiphileShareInQualifying = totalAmphiphiles > 0 ? amphiphilesInQualifying / totalAmphiphiles : 0

  const detailCount = Math.min(thresholds.detailAggregateCount, prepared.length)
  const aggregates: AggregateShape[] = []
  for (let k = 0; k < detailCount; k++) {
    aggregates.push(shapeOfAggregate(prepared[k].originalIdx, prepared[k].amphiphileCount, particles, box, monomers, thresholds, waterIdx))
  }

  return {
    aggregateCount: components.length,
    sizeHistogram,
    aggregates,
    qualifyingAggregateCount: qualifying.length,
    amphiphilesInQualifying,
    amphiphileShareInQualifying,
    hasLamellarAggregate: aggregates.some((s) => isBilayerShape(s, thresholds)),
    hasVesicleAggregate: aggregates.some((s) => isVesicleShape(s, thresholds)),
  }
}
