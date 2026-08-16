/**
 * Backmapping (Task 8): reconstructs full atomistic geometry for the coarse-grained lipids the
 * validated Cooke & Deserno engine (engine/src/sim.ts) actually moves. The three coarse beads
 * (head, tail1, tail2) fix the molecule's POSITION and ORIENTATION -- they come straight out of
 * the checked simulation -- but the atom-by-atom skeleton inside each lipid is never simulated:
 * it is built once from literature bond lengths/angles (chem/src/species.ts's
 * `buildAlkanoicAcid`) and then rigidly rotated and translated onto the bead frame. No model
 * constant lives in this file -- chain length and the bead-to-nanometre scale are call
 * parameters, and every bond length/angle traces back to data/molecules.json via species.ts.
 */
import { buildAlkanoicAcid } from './species'
import { vAdd, vCross, vDot, vLength, vNormalize, vScale, vSub, perpendicular, rotateAroundAxis, type Vec3 } from './geometry'

/** Below this span (nm) an axis carries no orientation: far under any real bond length, so it can
 *  only mean "the two reference points coincide", never a short-but-meaningful separation. */
const DEGENERATE_AXIS_NM = 1e-9

export interface BackmapAtom {
  element: string
  position: Vec3
}

export interface BackmapResult {
  atoms: BackmapAtom[]
  bonds: [number, number][]
}

/**
 * Rotation (as an axis + angle) that carries unit vector `from` onto unit vector `to`.
 * Degenerate cases (parallel / antiparallel) are handled explicitly since cross(from, to) has
 * zero length there and cannot supply a rotation axis on its own.
 */
function rotationAligning(from: Vec3, to: Vec3): { axis: Vec3; angleDeg: number } {
  const cross = vCross(from, to)
  const crossLen = vLength(cross)
  const dot = Math.max(-1, Math.min(1, vDot(from, to)))
  if (crossLen < 1e-9) {
    // Parallel (dot > 0, no rotation needed) or antiparallel (dot < 0, a 180 degree flip around
    // any perpendicular axis carries `from` onto `to`).
    return dot > 0 ? { axis: [0, 0, 1], angleDeg: 0 } : { axis: perpendicular(from), angleDeg: 180 }
  }
  return { axis: vScale(cross, 1 / crossLen), angleDeg: (Math.acos(dot) * 180) / Math.PI }
}

/**
 * Reconstructs one n-carbon alkanoic-acid lipid's full atomistic geometry from its three
 * coarse-grained bead positions (head, tail1, tail2 -- engine/src/sim.ts's per-lipid bead order,
 * in reduced sigma units) and a chain length. The reconstructed backbone's own bond lengths and
 * angles come from `buildAlkanoicAcid` (literature geometry, in nanometres) untouched; only its
 * rigid placement -- the rotation that lines its chain axis up with the bead-derived
 * head-to-tail direction, and the translation that puts its carboxyl carbon at the head bead's
 * position -- is derived from the coarse-grained frame. `sigmaNm` is the explicit nanometres-per-
 * sigma scale used to convert that bead-space frame into the nanometre space the reconstructed
 * atoms already live in; it never touches the atoms' own bond lengths.
 */
export function backmapLipid(
  head: Vec3,
  tail1: Vec3,
  tail2: Vec3,
  carbons: number,
  sigmaNm: number,
): BackmapResult {
  const species = buildAlkanoicAcid(carbons)
  const carbonPositions = species.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  const chainOrigin = carbonPositions[0]

  const headNm = vScale(head, sigmaNm)
  const tailNm = vScale(tail2, sigmaNm)

  // Both axes can legitimately be degenerate, and neither is an error:
  // a single-carbon acid has no chain axis at all (last carbon IS the first), and a soup run
  // really does produce chains of length one; and head and last tail can coincide numerically.
  // A degenerate axis carries no orientation information, so the molecule is placed unrotated
  // rather than throwing — orientation is undefined here, not wrong.
  const localSpan = vSub(carbonPositions[carbonPositions.length - 1], chainOrigin)
  const targetSpan = vSub(tailNm, headNm)
  const degenerate = vLength(localSpan) < DEGENERATE_AXIS_NM || vLength(targetSpan) < DEGENERATE_AXIS_NM

  const { axis, angleDeg } = degenerate
    ? { axis: [0, 0, 1] as Vec3, angleDeg: 0 }
    : rotationAligning(vNormalize(localSpan), vNormalize(targetSpan))

  const atoms: BackmapAtom[] = species.atoms.map((a) => {
    const relative = vSub(a.position, chainOrigin)
    const rotated = rotateAroundAxis(relative, axis, angleDeg)
    return { element: a.element, position: vAdd(rotated, headNm) }
  })

  // tail1 is not used to fix orientation (two points, head and tail2, already fully determine
  // the chain axis) -- it exists in the signature because it is part of the engine's per-lipid
  // bead triple and callers naturally have it on hand, but only head and tail2 are read here.
  void tail1

  return { atoms, bonds: species.bonds.map(([a, b]) => [a, b]) }
}

/**
 * Same reconstruction as `backmapLipid`, applied to every lipid encoded in a coarse-grained
 * System.positions() buffer (4 floats per bead -- x, y, z, type -- three consecutive beads per
 * lipid: head, tail1, tail2, per engine/src/sim.ts). Bond indices from each lipid are offset so
 * every atom index in the returned `bonds` array refers into the single merged `atoms` array.
 */
export function backmapSystem(
  positions: Float32Array,
  opts: { carbons: number; sigmaNm: number },
): BackmapResult {
  const floatsPerBead = 4
  const beadsPerLipid = 3
  const floatsPerLipid = floatsPerBead * beadsPerLipid
  const lipids = Math.floor(positions.length / floatsPerLipid)

  const atoms: BackmapAtom[] = []
  const bonds: [number, number][] = []

  for (let lip = 0; lip < lipids; lip++) {
    const base = lip * floatsPerLipid
    const head: Vec3 = [positions[base], positions[base + 1], positions[base + 2]]
    const tail1: Vec3 = [positions[base + 4], positions[base + 5], positions[base + 6]]
    const tail2: Vec3 = [positions[base + 8], positions[base + 9], positions[base + 10]]

    const molecule = backmapLipid(head, tail1, tail2, opts.carbons, opts.sigmaNm)
    const offset = atoms.length
    for (const atom of molecule.atoms) atoms.push(atom)
    for (const [a, b] of molecule.bonds) bonds.push([a + offset, b + offset])
  }

  return { atoms, bonds }
}
