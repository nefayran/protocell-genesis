// The immobile mineral PLATELET: where its beads sit, which of them are catalytic surface sites,
// and which particles of the whole system are frozen. Task 'clay-surface' (2026-08-19) --
// data/soup.json's `clay.basis` carries the literature, the ranks and the explicit list of reported
// clay effects this cannot represent (surface charge first among them). This module is pure
// geometry + bookkeeping: no GPU, no numeric model constant (params.test.ts's literal scanner
// covers soup/src), every number derived from the live box, data/params.json's rank-A sigma and the
// monomer's own radiusSigma.
//
// WHY THE COUNT IS DERIVED AND NOT A COMPOSITION NUMBER. A sheet is only a sheet if its beads are
// close enough to be impermeable, so its bead count is fixed by the box and by the mineral bead's
// own contact distance -- a `start` count would tile exactly one box and silently make holes in
// every other. data/soup.json's `start` therefore has no mineral entry at all; soup/src/sim.ts asks
// this module for the count before it sums N.
//
// WHY IMMOBILITY IS PER PARTICLE. The catalytic sites on the platelet are beads of the SAME
// catalyst species the broth already has (`M`), so that soup/wgsl/bond-adsorption.wgsl's whole
// centre machinery -- the requiresCatalyst gate, the centerLink FENE tether, occupancy,
// maxHoldSteps, desorbStretch/desorbTimeout -- applies to a surface site with no change at all.
// Part of that species' pool is frozen in the sheet and the rest diffuses, which a per-species
// switch could not express.

import type { Soup } from './rules'
import type { Params } from '../../engine/src/params'
import { wcaCutoff } from '../../engine/src/params'

/** Everything soup/src/soup-init-state.ts needs to lay the platelet out, and everything a
 * measurement needs to know where it is -- computed once, from the creation box. */
export interface ClayLayout {
  /** Monomer index (the value a particle carries in position.w) of the mineral species. */
  mineralKind: number
  /** Monomer index of the catalyst species whose beads become surface sites. */
  catalystKind: number
  /** Lattice sites per axis of ONE sheet, and the ACTUAL spacing used (box/n, so the sheet is
   * commensurate with the periodic box rather than merely close to the target spacing). */
  nx: number
  ny: number
  spacingX: number
  spacingY: number
  /** z of each sheet plane. `sheets` entries, centred on the box. */
  planeZ: number[]
  /** Total beads in the platelet = nx*ny*sheets, of which `siteCount` are catalyst beads and the
   * rest are mineral beads. */
  latticeCount: number
  siteCount: number
  mineralCount: number
  /** Contact distance below which a NON-mineral bead would sit inside a sheet bead's WCA core --
   * the half-thickness of the slab soup-init-state.ts keeps empty when it places everything else.
   * Derived as the largest wca_cut(pairB(mineral, other)) over the species actually present. */
  exclusionHalfWidth: number
}

/** True when this system carries a platelet: the file says so (or a caller overrode it) AND the
 * file actually declares a mineral monomer. */
export function clayEnabled(soup: Soup, override?: boolean): boolean {
  const c = soup.clay
  if (!c) return false
  return override ?? c.enabled
}

/** The sheet's lattice spacing: the WCA contact distance of the mineral-mineral pair, i.e. exactly
 * where the repulsion between two neighbouring sheet beads vanishes. pairB is the same
 * Lorentz-Berthelot arithmetic mean soup/wgsl/step.wgsl's pairB() uses, so the spacing this returns
 * is the same distance the force kernel itself calls "touching". */
function mineralSpacingTarget(p: Params, radiusSigma: number): number {
  return wcaCutoff(p.sigma * radiusSigma)
}

/**
 * Derives the whole platelet geometry for `box`. Throws (rather than silently making a one-bead
 * sheet) if the box is too small to hold a commensurate sheet at contact spacing.
 *
 * `catalystTotal` is the catalyst count this system was actually created with (CreateSoupOpts.start
 * / catalystCount already applied), because the site count is a FRACTION OF THAT POOL, moved onto
 * the surface rather than added to it -- so a with-clay and a without-clay run have identical
 * catalyst totals and the growth comparison is not confounded by "more catalyst present".
 */
export function planClay(
  soup: Soup,
  p: Params,
  box: [number, number, number],
  catalystTotal: number,
  siteFractionOverride?: number,
): ClayLayout {
  const c = soup.clay
  if (!c) throw new Error('planClay: data/soup.json не содержит секции clay')
  const mineralKind = soup.monomers.findIndex((m) => m.id === c.mineralId)
  if (mineralKind < 0) throw new Error(`planClay: clay.mineralId="${c.mineralId}" не найден среди monomers`)
  const catalystKind = soup.monomers.findIndex((m) => m.kind === 'catalyst')
  if (catalystKind < 0) throw new Error('planClay: в monomers нет частицы с kind="catalyst" — некому быть центром на поверхности')

  const mineralRadius = soup.monomers[mineralKind].radiusSigma
  const target = mineralSpacingTarget(p, mineralRadius)
  const nx = Math.round(box[0] / target)
  const ny = Math.round(box[1] / target)
  if (nx < 2 || ny < 2) {
    throw new Error(
      `planClay: коробка ${box[0]}x${box[1]} вмещает решётку ${nx}x${ny} при шаге ${target.toFixed(4)}σ — ` +
        `пластина из одного ряда бидов не является поверхностью`,
    )
  }
  const spacingX = box[0] / nx
  const spacingY = box[1] / ny

  // Sheet planes, centred on the box. For sheets=1 (what data/soup.json ships) this is exactly the
  // box midplane. For a stack, planes are spaced by the same contact distance the in-plane lattice
  // uses -- NOT by a separate interlayer number, because this model does not represent the
  // interlayer at all (clay.basis §2); a stack here is only a thicker slab.
  const planeZ: number[] = []
  const stackSpan = (c.sheets - 1) * target
  for (let s = 0; s < c.sheets; s++) planeZ.push(box[2] / 2 - stackSpan / 2 + s * target)

  const latticeCount = nx * ny * c.sheets
  // Task 'clay-surface-chemistry' (2026-08-19): the fraction is SWEEPABLE per system
  // (CreateSoupOpts.claySiteCatalystFraction) because the predecessor's headline growth result is a
  // function of it and it was never swept -- see that report's concern §3. The file's own value stays
  // the default; an override is validated here exactly as the file's value is validated in
  // soup/src/rules-validate.ts, so a sweep cannot ask for a fraction the schema would refuse.
  const siteFraction = siteFractionOverride ?? c.siteCatalystFraction
  if (!(siteFraction >= 0) || !(siteFraction <= 1)) {
    throw new Error(`planClay: доля центров на пластине ${siteFraction} должна лежать в [0,1]`)
  }
  const siteCount = Math.min(Math.round(siteFraction * catalystTotal), latticeCount)
  const mineralCount = latticeCount - siteCount

  // The slab kept clear of everything else: the largest distance at which ANY other species would
  // still be inside a sheet bead's WCA core. Taken over every declared species (not just the ones
  // with a nonzero count) so the value does not silently depend on this run's composition.
  //
  // Task 'clay-surface-chemistry' (2026-08-19), MEASURED DEFECT of the predecessor's version: the
  // maximum ran over the MINERAL bead's radius only, but a catalytic SITE is a bead of the catalyst
  // species and that species is the LARGEST in the file (radiusSigma 1.2 against the mineral's 1.0).
  // So a mobile catalyst could be placed at 1.2347 sigma from the plane -- inside the 1.3470 sigma WCA
  // core of a site bead it happened to sit above -- i.e. a cold-start overlap against a bead that by
  // construction cannot move away, which is exactly what data/soup.json's startBasis §2 warns about.
  // Measured overlap energy: 0.47 eps (small, which is why it never showed at the shipped 25 % site
  // fraction), but the fix costs nothing: the maximum runs over the radii of every species ACTUALLY IN
  // the sheet, which is the mineral bead plus -- when there are sites -- the catalyst bead.
  const sheetRadii = [mineralRadius]
  if (siteCount > 0) sheetRadii.push(soup.monomers[catalystKind].radiusSigma)
  let exclusionHalfWidth = 0
  for (const rSheet of sheetRadii) {
    for (const m of soup.monomers) {
      const b = p.sigma * (rSheet + m.radiusSigma) * 0.5
      exclusionHalfWidth = Math.max(exclusionHalfWidth, wcaCutoff(b))
    }
  }

  return { mineralKind, catalystKind, nx, ny, spacingX, spacingY, planeZ, latticeCount, siteCount, mineralCount, exclusionHalfWidth }
}

/** The (x,y,z) of lattice position `k` of the platelet, in the sheet-major order planClay's counts
 * assume: k = ((sheet*ny) + iy)*nx + ix. Beads sit at cell CENTRES so the sheet is symmetric under
 * the periodic wrap on both axes. */
export function clayLatticePosition(L: ClayLayout, k: number): [number, number, number] {
  const perSheet = L.nx * L.ny
  const sheet = Math.floor(k / perSheet)
  const within = k % perSheet
  const iy = Math.floor(within / L.nx)
  const ix = within % L.nx
  return [(ix + 0.5) * L.spacingX, (iy + 0.5) * L.spacingY, L.planeZ[sheet]]
}

/**
 * Which lattice positions carry a catalyst site rather than a mineral bead. Spread as evenly as the
 * lattice allows (a fixed stride over the sheet-major index), NOT randomly: a random draw would make
 * the site pattern depend on the seed, and then a with/without-clay comparison at "the same seed"
 * would still differ in where the sites are. An even stride also keeps sites isolated from each
 * other, which is what a 3-4% surface coverage is supposed to mean.
 */
export function claySiteIndices(L: ClayLayout): Set<number> {
  const out = new Set<number>()
  if (L.siteCount <= 0) return out
  const stride = L.latticeCount / L.siteCount
  for (let s = 0; s < L.siteCount; s++) out.add(Math.min(L.latticeCount - 1, Math.round(s * stride)))
  // Rounding can collide on a very dense site fraction; fill any shortfall with the first free
  // positions so the returned count always equals siteCount (asserted by the caller).
  let probe = 0
  while (out.size < L.siteCount && probe < L.latticeCount) {
    if (!out.has(probe)) out.add(probe)
    probe++
  }
  return out
}

/** Signed distance from `z` to the nearest sheet plane, under the periodic wrap in z -- the
 * coordinate every profile measurement in tests/soup-clay.test.ts is binned against. Returns the
 * ABSOLUTE distance: the two faces of a sheet are physically equivalent here, so folding them
 * together doubles the statistics per bin instead of splitting them. */
export function distanceToPlatelet(z: number, planeZ: number[], boxZ: number): number {
  let best = Infinity
  for (const p0 of planeZ) {
    let d = z - p0
    d -= Math.round(d / boxZ) * boxZ
    best = Math.min(best, Math.abs(d))
  }
  return best
}
