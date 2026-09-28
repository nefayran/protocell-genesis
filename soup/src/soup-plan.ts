// Pre-GPU planning: everything createSoup needs to derive and validate BEFORE it may allocate a
// single GPU resource -- rule resolution (data/soup.json's symbolic (id, id) rules -> numeric
// kind-index uniforms), neighbour-grid geometry, and the Verlet-list/drift-safety completeness
// derivation. Split out of soup/src/sim.ts (file-size rule in CLAUDE.md) as one "pre-GPU planning"
// module: every function here is pure/GPU-free, moved verbatim, not rewritten.

import { loadParams, wcaCutoff, type Params } from '../../engine/src/params'
import { loadSoup, type Rule, type Soup } from './rules'

// --- rule resolution: data/soup.json's symbolic (id, id) rules -> numeric kind-index uniforms ----

export interface ResolvedRule {
  ruleIdx: number
  bond: Rule
  brk: Rule
  kindA: number
  kindB: number
  slotRoleA: number
  slotRoleB: number
}

// Slot role codes shared with soup/wgsl/bond-common.wgsl's roleOf(): 0 = chain pool (slots 0,1), 1 =
// head slot (slot 2 fixed), 2 = single slot (slot 0 fixed). Resolved from each side's monomer KIND
// (carbon/head), never from a hardcoded id, so a future rename of "C"/"O" in data/soup.json would
// not need a matching change here.
export function slotRole(thisKind: string, otherKind: string): number {
  if (thisKind === 'carbon') return otherKind === 'carbon' ? 0 : 1
  if (thisKind === 'head') return 2
  throw new Error(`soup/src/soup-plan.ts: a rule bonds a monomer of kind "${thisKind}", but only carbon and head are supported`)
}

export function resolveRules(soup: ReturnType<typeof loadSoup>): { rules: ResolvedRule[]; catalystKind: number } {
  const kindIndex = new Map(soup.monomers.map((m, idx) => [m.id, idx]))
  const catalystMonomer = soup.monomers.find((m) => m.kind === 'catalyst')
  if (!catalystMonomer) throw new Error('data/soup.json: no monomer of kind catalyst found')
  const catalystKind = kindIndex.get(catalystMonomer.id)!

  const bondRules = soup.rules.filter((r) => r.kind === 'bond')
  const rules: ResolvedRule[] = bondRules.map((bond, ruleIdx) => {
    const brk = soup.rules.find((r) => r.kind === 'break' && r.a === bond.a && r.b === bond.b)
    if (!brk) throw new Error(`data/soup.json: rule ${bond.id} has no paired break`)
    const ma = soup.monomers.find((m) => m.id === bond.a)
    const mb = soup.monomers.find((m) => m.id === bond.b)
    if (!ma || !mb) throw new Error(`data/soup.json: rule ${bond.id} refers to an undescribed monomer`)
    return {
      ruleIdx,
      bond,
      brk,
      kindA: kindIndex.get(bond.a)!,
      kindB: kindIndex.get(bond.b)!,
      slotRoleA: slotRole(ma.kind, mb.kind),
      slotRoleB: slotRole(mb.kind, ma.kind),
    }
  })
  if (rules.length > 4) {
    throw new Error(`soup/src/soup-plan.ts: ${rules.length} bond formation rules, but BondParams holds at most 4`)
  }
  return { rules, catalystKind }
}

export function packVec4(values: number[]): number[] {
  const out = [0, 0, 0, 0]
  for (let i = 0; i < Math.min(4, values.length); i++) out[i] = values[i]
  return out
}

/** Task 'explicit-water' (2026-08-18): the same zero-padded packing packVec4 does, widened from one
 * vec4 (4 slots) to two (8 slots) -- matches soup/wgsl/step.wgsl/bond-common.wgsl's own Species
 * struct (`array<vec4<f32>, 2>` per field), sized for up to 8 monomer kinds so the 5th species
 * (water) fits without restructuring the uniform again. */
export function packSpeciesSlots(values: number[]): number[] {
  const out = new Array(8).fill(0)
  for (let i = 0; i < Math.min(8, values.length); i++) out[i] = values[i]
  return out
}

// --- neighbour-grid geometry ------------------------------------------------------------------

/** The neighbour-grid geometry a given `box` would get, and whether it is even valid -- the exact
 * derivation createSoup() itself needs before it may allocate a single GPU buffer (walkRadius from
 * the species' own interaction range, the grid dims that follow from box/cellSize, and the two
 * periodicity/minimum-image invariants below), pulled out into its own pure, GPU-free function so
 * a caller (a viewer letting the user pick their own box, item 3 of the 2026-08 UI-fixes task) can
 * preview the cost -- particle count, grid dims -- and catch an invalid box BEFORE calling the async,
 * GPU-allocating createSoup() at all, rather than only finding out from a caught exception after
 * paying for the attempt. createSoup() calls this too, so there is exactly one definition of
 * "is this box valid" for the soup engine, not a second one drifting out of sync in a viewer. */
export interface SoupPlan {
  /** Total starting particle count for this composition (independent of box). */
  N: number
  box: [number, number, number]
  /** Neighbour-grid cell side, sigma. */
  cellSize: number
  /** How many cells the neighbour walk must cover on each side (Verlet-list build radius when the
   * list is enabled, this soup's own regular walk radius otherwise). */
  effectiveWalkRadius: number
  /** Grid cell counts, one per axis -- `floor(box[axis]/cellSize)`, at least 1. */
  dims: [number, number, number]
  ncells: number
  /** `2*effectiveWalkRadius+1` -- the minimum dims[axis] the periodic ±effectiveWalkRadius cell
   * walk needs on every axis: with fewer cells than this on a periodic axis, the wrap revisits a
   * cell more than once, silently multiplying every force/bond-attempt/list-build contribution
   * from it. The OTHER half of `valid` (not separately named) is the minimum-image guard,
   * min(box)/2 > bend.r0, unaffected by cell count -- it is about the bond's own reach. */
  minCells: number
  valid: boolean
  /** Populated (non-null) exactly when `!valid` -- the exact message createSoup() itself would
   * throw for this box, so a caller can show it without waiting for that throw. */
  reason: string | null
}

export function planSoupGrid(box: [number, number, number], startCounts?: Record<string, number>): SoupPlan {
  const soup = loadSoup()
  const p = loadParams()

  const counts: Record<string, number> = { ...soup.start, ...(startCounts ?? {}) }
  const N = soup.monomers.reduce((sum, m) => sum + (counts[m.id] ?? 0), 0)

  // Same derivation as createSoup's own interactionRange/cellSize/walkRadius/listBuildWalkRadius
  // (deriveGridGeometry below) -- not repeated here since it does not depend on `box` at all (only
  // on data/soup.json's own species/neighborGrid/verletList settings), so a caller previewing many
  // candidate box sizes is not re-deriving anything box-independent.
  const maxRadiusSigma = Math.max(...soup.monomers.map((m) => m.radiusSigma))
  const maxB = p.sigma * maxRadiusSigma
  const interactionRange = wcaCutoff(maxB) + p.attraction.wc
  const cellSize = interactionRange / soup.neighborGrid.cellDivisor
  const walkRadius = Math.ceil(interactionRange / cellSize)
  const verlet = soup.verletList
  const listRange = interactionRange + verlet.skin
  const listBuildWalkRadius = Math.ceil(listRange / cellSize)
  const effectiveWalkRadius = verlet.enabled ? listBuildWalkRadius : walkRadius

  const dims: [number, number, number] = [
    Math.max(1, Math.floor(box[0] / cellSize)),
    Math.max(1, Math.floor(box[1] / cellSize)),
    Math.max(1, Math.floor(box[2] / cellSize)),
  ]
  const ncells = dims[0] * dims[1] * dims[2]
  const minCells = 2 * effectiveWalkRadius + 1
  const gridOk = dims[0] >= minCells && dims[1] >= minCells && dims[2] >= minCells
  const miOk = Math.min(box[0], box[1], box[2]) / 2 > p.bend.r0
  const valid = gridOk && miOk
  const reason = valid
    ? null
    : `neighbour grid: box=[${box[0]},${box[1]},${box[2]}] gives cellSize=${cellSize.toFixed(4)}, dims=[${dims[0]},${dims[1]},${dims[2]}] ` +
      `and min(box)/2=${(Math.min(box[0], box[1], box[2]) / 2).toFixed(4)}; it needs dims>=${minCells} (2*effectiveWalkRadius+1) on all three axes and min(box)/2 > bend.r0=${p.bend.r0}`

  return { N, box, cellSize, effectiveWalkRadius, dims, ncells, minCells, valid, reason }
}

// --- creation-time grid/Verlet-list geometry, WITH the throwing completeness guards --------------

export interface GridGeometry {
  interactionRange: number
  cellSize: number
  walkRadius: number
  listRange: number
  listBuildWalkRadius: number
  effectiveWalkRadius: number
}

/** createSoup's own interactionRange/cellSize/walkRadius/Verlet-list geometry derivation, WITH the
 * three throwing completeness guards (walk-radius coverage, Verlet-list coverage, drift-safety
 * bound) -- moved verbatim out of createSoup's body (file-size rule in CLAUDE.md). Kept separate
 * from planSoupGrid's OWN (box-dependent) derivation above even though the two overlap heavily: this
 * one throws with createSoup's own error wording and additionally checks the drift-safety bound
 * (needs `opts.kT`, which planSoupGrid never receives), so it is not a drop-in call to planSoupGrid
 * -- exactly the duplication the original sim.ts already had between the two, preserved rather than
 * "cleaned up" (a behaviour change this refactor must not make). */
export function deriveGridGeometry(soup: Soup, p: Params, kT: number): GridGeometry {
  // Interaction range: at least the largest interaction reach (WCA contact for the largest
  // pairwise size, plus the tail-tail attraction's outer cutoff w_c) so any pair within range of
  // each other is guaranteed to land within `walkRadius` cells of each other -- same reasoning
  // engine/src/sim.ts's own `cellSize` comment gives, generalised from a fixed lipid pair to
  // whichever two of the soup's own species (by data/soup.json's radiusSigma) are largest.
  //
  // perf2-report.md, candidate (a): the ORIGINAL grid set cellSize = interactionRange and walked
  // 3x3x3 (walkRadius=1), which searches a cube of side 3*interactionRange for pairs that only
  // ever lie within a sphere of radius interactionRange -- diagnosed and measured (candidates
  // examined vs pairs within range, forceCandidateStatsDEBUG in soup/src/soup-readback.ts) before
  // this was touched. `neighborGrid.cellDivisor` (data/soup.json, with a written basis) divides the
  // cell side by that many; the walk radius needed to keep the SAME completeness guarantee is
  // derived, not assumed, and ASSERTED right below rather than trusted to fall out of the
  // arithmetic. cellDivisor=1 reproduces the original cellSize/walkRadius exactly (bit-identical
  // 3x3x3 walk), the honest A/B control point for this change.
  const maxRadiusSigma = Math.max(...soup.monomers.map((m) => m.radiusSigma))
  const maxB = p.sigma * maxRadiusSigma
  const interactionRange = wcaCutoff(maxB) + p.attraction.wc
  const cellDivisor = soup.neighborGrid.cellDivisor
  const cellSize = interactionRange / cellDivisor
  const walkRadius = Math.ceil(interactionRange / cellSize)
  if (walkRadius * cellSize < interactionRange - 1e-6) {
    throw new Error(
      `neighbour grid: walkRadius=${walkRadius} * cellSize=${cellSize.toFixed(6)} = ${(walkRadius * cellSize).toFixed(6)} ` +
        `does not cover interactionRange=${interactionRange.toFixed(6)}, so the traversal completeness guarantee is broken`,
    )
  }

  // perf2-report.md, candidate (c): a Verlet list, built every `rebuildEvery` real steps on the
  // SAME (unshrunk) grid above, with a wider walk radius that covers interactionRange+skin instead
  // of just interactionRange -- the skin margin is what lets the list stay complete for
  // `rebuildEvery` steps without re-walking. Derived and asserted the same way walkRadius is above,
  // never assumed. When verlet.enabled is false, GB.dims.w (soup/src/soup-buffers.ts's gridUniform
  // write) carries the ORIGINAL walkRadius instead, and the *_list_main kernels are never compiled/
  // dispatched at all -- see soup/src/soup-pipelines.ts and soup/src/soup-integrate.ts.
  const verlet = soup.verletList
  const listRange = interactionRange + verlet.skin
  const listBuildWalkRadius = Math.ceil(listRange / cellSize)
  if (listBuildWalkRadius * cellSize < listRange - 1e-6) {
    throw new Error(
      `Verlet list: listBuildWalkRadius=${listBuildWalkRadius} * cellSize=${cellSize.toFixed(6)} = ` +
        `${(listBuildWalkRadius * cellSize).toFixed(6)} does not cover listRange=${listRange.toFixed(6)} (interactionRange+skin), so the traversal completeness guarantee is broken`,
    )
  }
  // Drift-safety condition (perf-report.md's own rejected-candidate-(a) analysis, generalised from
  // one step to `rebuildEvery` steps): a 2x-RMS-3D-speed worst-case outlier bound, evaluated for
  // THIS system's own kT (not a fixed assumed worst case) -- a particle drifting at that bound for
  // the WHOLE rebuild interval must still land within skin/2 of where the list last saw it, or the
  // list could be missing a real neighbour by the next rebuild. soup_max_drift_main backs this
  // analytical bound up with a REAL per-step measurement, checked by soup-grid-verlet.ts's
  // assertVerletSafety.
  if (verlet.enabled) {
    const vBound = 2 * Math.sqrt(3 * kT)
    const driftBound = 2 * verlet.rebuildEvery * p.integrator.dt * vBound
    if (driftBound > verlet.skin) {
      throw new Error(
        `Verlet list: 2*rebuildEvery*dt*vBound=${driftBound.toFixed(4)} exceeds skin=${verlet.skin} ` +
          `at kT=${kT}: the rebuild is not frequent enough (or skin is too small) for this temperature`,
      )
    }
  }
  const effectiveWalkRadius = verlet.enabled ? listBuildWalkRadius : walkRadius

  return { interactionRange, cellSize, walkRadius, listRange, listBuildWalkRadius, effectiveWalkRadius }
}

// --- the Verlet list's per-particle capacity, DERIVED (task 'big-box', 2026-08-20) ---------------

/** WHY THIS EXISTS. `data/soup.json`'s `verletList.listCapacity` was 2500, set as a safe upper bound
 * and never measured -- and because the list is a flat `N * listCapacity * 4` byte buffer against
 * this device's 4 294 967 292-byte storage-binding limit, that one unmeasured number WAS the
 * project's particle ceiling (429 496) and therefore its box ceiling (81.27 sigma at liquid-water
 * density). Deriving it from the density instead is the same treatment the long-range
 * electrostatic list already gets (soup/src/electrostatics.ts's makeEsBasis, whose own comment
 * spells out the identical reasoning), and for the identical reason: an overflow is not a
 * performance problem, it is a silently dropped neighbour.
 *
 * The estimate is the UNIFORM expectation (4/3)*pi*listRange^3 * rho at the densest box this run
 * will ever visit -- rho from the CAPACITY census (the wet composition every buffer is sized for)
 * over min(box)^3, which under-counts a non-cubic box's volume and therefore over-counts the
 * density, the safe direction -- times `capacitySafetyFactor`, which is the measured inhomogeneity
 * headroom (see data/soup.json's verletList.basis for the four configurations it was measured on).
 * Clamped below by 64 (a tiny system must still hold its handful of neighbours) and above by
 * capacityN (no particle can have more neighbours than there are other particles).
 *
 * It is NOT capped by the file's own listCapacity: that field is the fallback for a caller with no
 * census/box to derive from, exactly as `electrostatics.longRangeListCapacity` is for its own list. */
export function deriveListCapacity(
  soup: Soup,
  listRange: number,
  capacityN: number,
  densestDensity: number,
): { capacity: number; uniform: number; safetyFactor: number; derived: boolean } {
  const vl = soup.verletList
  const f = vl.capacitySafetyFactor
  if (!(capacityN > 0) || !(densestDensity > 0) || !(f > 0)) {
    return { capacity: vl.listCapacity, uniform: 0, safetyFactor: f, derived: false }
  }
  const uniform = ((4 * Math.PI) / 3) * listRange ** 3 * densestDensity
  // The FLOOR matters as much as the factor, and it is the part the first version got wrong. In a
  // DILUTE system the box-average density says almost nothing about the local one: the water-free
  // broth fixtures sit at rho_box 0.135-0.174, which derives a capacity of 168, and then the organics
  // collapse into one condensed droplet whose local density is set by the POTENTIAL, not by the box --
  // and the capacity guard threw (loudly, correctly) on two of them. `capacityFloor` is the largest
  // neighbour count ever measured inside a condensed phase plus a margin, so the derived capacity can
  // never fall below what a condensed droplet needs however dilute the box is. Still clamped by
  // capacityN, which makes a small system overflow-proof by construction.
  const capacity = Math.max(64, Math.min(capacityN, Math.max(soup.verletList.capacityFloor, Math.ceil(f * uniform))))
  return { capacity, uniform, safetyFactor: f, derived: true }
}

/** The DENSEST total number density this run will ever reach -- what the capacity above is derived
 * from. Three candidates, all real states of a real run and all known at creation: the creation
 * (wet) composition in the creation box; the live composition in the live box (a checkpoint resumed
 * mid dry-phase carries FEWER beads in a SMALLER box, and which of the two wins is not obvious); and,
 * when dry-wet cycling is on, `dryWetCycle.targetDryDensity` itself -- which is exactly the density
 * the dry phase is CONSTRUCTED to reach (soup/src/soup-box-scale-math.ts's computeDryBox derives the
 * dry box from it), so it needs no estimate at all.
 *
 * Getting this wrong in the safe direction costs memory; getting it wrong in the unsafe direction
 * costs a thrown run (never a silent truncation -- soup/src/soup-grid-verlet.ts's assertVerletSafety).
 * The first version of this function used capacityN/min(box)^3, which mixed the WET census with the
 * DRY box and over-estimated the density by 3.65x on a resumed evaporating checkpoint -- enough to
 * ask for a 6.25 GB list against a 4.29 GB device limit. */
export function densestDensityOf(
  soup: Soup,
  capacityN: number,
  creationBox: [number, number, number],
  liveN: number,
  liveBox: [number, number, number],
  cycling: boolean,
  /** Task 'confined-parcel' (2026-08-21): when this run confines its material to a parcel smaller
   * than the box, the density that matters is over the PARCEL. Given as a function of a box (rather
   * than as one number) because both the creation and the live box are asked about below and the
   * parcel scales with each. Omitted (every unconfined run) leaves the box volume exactly as it was.
   *
   * This is not a refinement: without it, the confined geometry's box-average density is 5.8x below
   * its real one, deriveListCapacity would size the Verlet list for that, and a list overflow is a
   * silently dropped neighbour. `capacityFloor` would have caught part of it and no more. */
  parcelVolumeOf?: (box: [number, number, number]) => number,
): number {
  const vol = (b: [number, number, number]) => (parcelVolumeOf ? parcelVolumeOf(b) : b[0] * b[1] * b[2])
  return Math.max(
    capacityN / vol(creationBox),
    liveN / vol(liveBox),
    cycling ? soup.dryWetCycle.targetDryDensity : 0,
  )
}

/** The memory arithmetic that made the ceiling, as a function a report and a test can both call
 * instead of re-deriving it. `bindingLimit` is the device's own maxStorageBufferBindingSize. */
export function verletListBytes(capacityN: number, listCapacity: number): number {
  return capacityN * listCapacity * 4
}
