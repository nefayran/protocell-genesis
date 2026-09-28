// The zero-tension Metropolis Monte Carlo area move for the SOUP path -- task
// 'hydrophobic-asymmetry' (2026-08-19), defect 2: with a fixed box, area-per-lipid was an ASSUMED
// construction constant (the literature corridor's midpoint) and every bilayer-in-water measurement
// inherited that assumption. With this move the lateral box relaxes and area-per-lipid becomes a
// MEASUREMENT.
//
// PORTED, NOT COPIED. The mechanism comes from engine/src/sim-area-move.ts (whose own doc comments
// carry the history: without the configurational Jacobian the box collapsed 1.352 -> 0.53 sigma^2,
// and the fix was verified by the identity <dU/dlnA>_bead - <dU/dlnA>_COM = 2*N*kT). Its module
// cannot be called directly here -- it is typed on EngineRuntime and reaches into the membrane
// engine's own grid/bind-group/potential-buffer functions, and its coordinate map scaleLateralRigid
// assumes a fixed 3-bead lipid at a known index stride, which the soup (dynamic bond topology,
// plus 4500 unbonded solvent beads) does not have. What IS reused, rather than re-implemented:
//  - the coordinate map: soup/src/soup-box-scale-math.ts's scaleMoleculesRigid, which is already
//    the generalisation of scaleLateralRigid to the soup's bond-graph molecules (a free water bead
//    is a size-1 molecule and therefore scales as its own centre of mass, which is exactly right);
//  - the GPU-side application: soup/src/soup-box-scale.ts's applyBoxScaleOnce, including its
//    grid/Verlet reallocation, its rebuild, and its runtime self-check that no bonded distance moved.
// New here: the Metropolis criterion itself, the proposal's box geometry, and the energy the
// criterion needs (soup/src/soup-potential.ts -- the soup force path has no potential kernel).
//
// TWO THINGS THIS PORT HAD TO GET RIGHT, both stated in data/soup.json's areaMove.basis and both
// measured in tests/soup-area-move.test.ts rather than argued:
//  1. The SOLVENT scales with the lipids. scaleMoleculesRigid moves every connected component,
//     water included, so water is part of the system whose box changes -- and water also counts in
//     the Jacobian's N (see moleculeCount below). The identity test below fails loudly, by a large
//     margin, if the solvent is left behind.
//  2. The acceptance keeps the entropic term with the correct N: Nmol*kT*ln(V'/V), the general form
//     of engine/src/sim-area-move.ts's Nlipids*kT*ln(A'/A). For mode 'lateral-fixed-z' the two are
//     literally the same expression (V'/V = A'/A). For mode 'lateral-fixed-volume' -- the one the
//     bilayer-in-water gate requires, because with explicit solvent a fixed-Lz area move equilibrates
//     against the SOLVENT's absolute pressure instead of against membrane tension -- ln(V'/V) is
//     identically zero, so the term evaluates to zero. That is the same formula at a different
//     point, not a dropped term: an isotropic (ideal-gas-like) pressure does no work on a
//     volume-preserving deformation.
//
// No numeric model constant is written here: logDelta and mode come from data/soup.json's areaMove.

import { loadParams, type Params } from '../../engine/src/params'
import type { AreaMove, Soup } from './rules'
import { planSoupGrid } from './soup-plan'
import { scaleMoleculesRigid } from './soup-box-scale-math'
import { applyBoxScaleOnce } from './soup-box-scale'
import { makePotentialBasis, soupPotential, type PotentialBasis } from './soup-potential'
import type { EsBasis } from './electrostatics'
import type { SoupRuntime } from './soup-runtime'

export type AreaMoveMode = AreaMove['mode']

export type Box = [number, number, number]

/** The proposed box for a log-area step `u`. Lateral axes always scale by sqrt(exp(u)) (so the
 * projected area scales by exp(u) exactly, making ln(A'/A) = u with no rounding); the difference
 * between the modes is entirely what happens to L_z:
 *  - 'lateral-fixed-z': L_z untouched -- engine/src/sim-area-move.ts's own move, correct for a
 *    solvent-free patch with an open z axis, where zero lateral pressure IS zero tension.
 *  - 'lateral-fixed-volume': L_z divided by exp(u), so the volume is invariant -- required once the
 *    box is filled with explicit solvent (see this file's header and data/soup.json's
 *    areaMove.basis). */
export function proposeBox(box: Box, u: number, mode: AreaMoveMode): Box {
  const s = Math.exp(u)
  const q = Math.sqrt(s)
  return mode === 'lateral-fixed-volume' ? [box[0] * q, box[1] * q, box[2] / s] : [box[0] * q, box[1] * q, box[2]]
}

/** Straight affine rescale of every BEAD coordinate (no molecule grouping at all), used only by the
 * verification identity in tests/soup-area-move.test.ts: the identity compares the free-energy
 * derivative measured through this map against the one measured through the rigid-molecule map, and
 * their difference is a pure count times kT. Never used by the move itself -- rescaling beads
 * individually would stretch every covalent bond, which is precisely what scaleMoleculesRigid
 * exists to avoid. */
export function scaleBeadsAffine(positions: Float32Array, oldBox: Box, newBox: Box): Float32Array {
  const out = new Float32Array(positions.length)
  const n = positions.length / 4
  const s: Box = [newBox[0] / oldBox[0], newBox[1] / oldBox[1], newBox[2] / oldBox[2]]
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) {
      const v = positions[i * 4 + a] * s[a]
      out[i * 4 + a] = v - Math.floor(v / newBox[a]) * newBox[a]
    }
    out[i * 4 + 3] = positions[i * 4 + 3]
  }
  return out
}

/** Number of INDEPENDENT molecules: connected components of the covalent bond graph, counting every
 * unbonded particle as its own component. This is the N of the configurational Jacobian
 * (V'/V)^N, because scaleMoleculesRigid displaces exactly one degree of freedom per component (its
 * centre of mass) and leaves internal offsets alone. Water beads are components of size 1 and DO
 * count -- the solvent is part of the system whose volume changes. */
export function moleculeCount(bonds: Uint32Array, n: number): number {
  const parent = new Int32Array(n)
  for (let i = 0; i < n; i++) parent[i] = i
  const find = (x: number): number => {
    let r = x
    while (parent[r] !== r) r = parent[r]
    while (parent[x] !== r) {
      const next = parent[x]
      parent[x] = r
      x = next
    }
    return r
  }
  let components = n
  for (let k = 0; k < bonds.length; k += 2) {
    const a = find(bonds[k])
    const b = find(bonds[k + 1])
    if (a !== b) {
      parent[a] = b
      components--
    }
  }
  return components
}

export interface AreaMoveResult {
  trials: number
  accepted: number
  acceptedFraction: number
  /** The box after the chain -- also written to rt.live.liveBox and to the GPU. */
  box: Box
  /** One entry per trial (the CURRENT box after that trial, accepted or not), so a caller can show
   * that the box really moved and then settled instead of asserting it did. */
  lateralTrajectory: number[]
  /** Potential energy at the start and at the end of the chain, for the record. */
  energyStart: number
  energyEnd: number
}

export interface SoupAreaMoveDeps {
  rt: SoupRuntime
  particles: () => Promise<Float32Array>
  bonds: () => Promise<Uint32Array>
  centerLinks: () => Promise<Uint32Array>
  soup: Soup
  seed: number
  attractionOverride?: number
  /** Task 'clay-surface-chemistry': the CPU Metropolis energy must use the SAME mineral depth row the
   * GPU force kernel got, so the selected surface chemistry travels with the override. */
  claySurfaceChemistry?: string
  /** Task 'acid-soap-pairing' (2026-08-23): the charge-assisted head-head depth this system's GPU
   * uniform was written with -- same reason as claySurfaceChemistry above. */
  acidSoapScaleOverride?: number
  /** Task 'acid-soap-pairing' (2026-08-23): the RESOLVED electrostatic basis of this system (sim.ts's
   * own `es`), so the Metropolis energy screens charge exactly as the force kernel does. Absent keeps
   * the pre-task behaviour, in which makePotentialBasis derived the basis from the FILE (shipped
   * `enabled: false`) and the electrostatic energy was therefore identically zero. */
  es?: EsBasis
  /** Task 'acid-soap-pairing' (2026-08-23): the live per-particle charges, i.e. the protonation state
   * the constant-pH Monte Carlo last wrote. Read once per areaMove() call, at the same point the
   * positions/bonds/links are read -- the chain proposes only BOX changes, so no charge can change
   * inside it. Absent (or an unenabled basis) leaves both charge-reading terms identically zero. */
  charges?: () => Promise<Float32Array>
}

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function makeSoupAreaMove(deps: SoupAreaMoveDeps): (trials: number, opts?: { mode?: AreaMoveMode }) => Promise<AreaMoveResult> {
  const { rt, particles, bonds, centerLinks, soup, seed } = deps
  const p: Params = loadParams()
  const rng = mulberry32(seed ^ 0x5eed_a1ea)
  let basis: PotentialBasis | null = null

  return async function areaMove(trials: number, opts?: { mode?: AreaMoveMode }): Promise<AreaMoveResult> {
    // Task 'confined-parcel' (2026-08-21): REFUSED under confinement, rather than returning a number
    // that would not mean what its name says. This move proposes a BOX change at fixed volume and
    // reads a lateral tension off the acceptance statistics; in a confined run the box is not what
    // sets the system's volume (the wall is), the wall does work on the system as the box moves, and
    // soup/src/soup-potential.ts -- the energy this move's Metropolis criterion is built on -- carries
    // no wall term at all, so the proposed and current energies would differ by an unaccounted
    // one-body contribution. There is no zero-tension area to quote inside a parcel.
    if (rt.confine) {
      throw new Error(
        'areaMove: an MC area move makes no sense in a confined parcel -- the volume is set by the wall, not the box, ' +
          'the wall does work on the system when the box changes, and soupPotential does not include its contribution. ' +
          'Zero tension is measured in a periodic box (CreateSoupOpts.confine absent)',
      )
    }
    const am = soup.areaMove
    if (am === undefined) {
      throw new Error("data/soup.json: no areaMove section, so the MC area move is not configured (see soup/src/soup-area-move.ts)")
    }
    const mode: AreaMoveMode = opts?.mode ?? am.mode
    if (basis === null) {
      basis = makePotentialBasis(
        soup,
        p,
        deps.attractionOverride,
        deps.claySurfaceChemistry,
        undefined,
        deps.acidSoapScaleOverride,
      )
      // Task 'acid-soap-pairing' (2026-08-23): the system's OWN resolved electrostatic basis rather
      // than one re-derived from the file, so the screened-Coulomb cutoff, coefficient and shift in
      // the Metropolis criterion are bit-identical to the ones the force kernel runs. On an
      // uncharged system this assigns a disabled basis over a disabled basis.
      if (deps.es !== undefined) basis.es = deps.es
    }

    const startBox: Box = [rt.live.liveBox[0], rt.live.liveBox[1], rt.live.liveBox[2]]
    const bondPairs = await bonds()
    const links = await centerLinks()
    let curPos = await particles()
    // Task 'acid-soap-pairing' (2026-08-23): the protonation state, read ONCE for the whole chain --
    // the chain proposes box changes only, and no charge can change inside it (the constant-pH sweep
    // runs from the stepper, not from here). `undefined` on a system with no charge readback at all,
    // which makes both charge-reading terms identically zero exactly as before this task.
    const chargesNow = basis.es.enabled && deps.charges !== undefined ? await deps.charges() : undefined
    let curBox: Box = startBox
    // kT from rt.p, which soup/src/sim.ts already built as loadParams() with THIS system's own
    // opts.kT substituted -- the same kT the thermostat uniform was written from, so the Metropolis
    // criterion and the dynamics cannot judge the configuration at different temperatures.
    const kT = rt.p.thermostat.kT
    const nMol = moleculeCount(bondPairs, rt.N)
    let energy = soupPotential(curPos, bondPairs, curBox, basis, links, chargesNow).total
    const energyStart = energy
    const lateralTrajectory: number[] = []
    let accepted = 0

    for (let t = 0; t < trials; t++) {
      const u = (rng() * 2 - 1) * am.logDelta
      const proposedBox = proposeBox(curBox, u, mode)
      // Reject with no work at all if the proposal would breach the neighbour-grid/minimum-image
      // invariants createSoup enforces at construction (planSoupGrid's own `valid`), mirroring
      // engine/src/sim-area-move.ts's gridInvariantsHold pre-check: a safety net, not the normal
      // path, since logDelta is small.
      if (!planSoupGrid(proposedBox, rt.startCounts).valid) {
        lateralTrajectory.push(curBox[0])
        continue
      }
      const proposedPos = scaleMoleculesRigid(curPos, bondPairs, curBox, proposedBox)
      const proposedEnergy = soupPotential(proposedPos, bondPairs, proposedBox, basis, links, chargesNow).total
      const dU = proposedEnergy - energy
      const volRatio =
        (proposedBox[0] * proposedBox[1] * proposedBox[2]) / (curBox[0] * curBox[1] * curBox[2])
      // min(1, exp(-(dU - Nmol*kT*ln(V'/V))/kT)) -- see this file's header for why ln(V'/V) is the
      // general form of engine/src/sim-area-move.ts's ln(A'/A), and why it is identically 0 (not
      // dropped) in the fixed-volume mode.
      const acceptProb = Math.min(1, Math.exp(nMol * Math.log(volRatio) - dU / kT))
      if (rng() < acceptProb) {
        accepted++
        curPos = proposedPos
        curBox = proposedBox
        energy = proposedEnergy
      }
      lateralTrajectory.push(curBox[0])
    }

    // ONE GPU application for the whole chain, at the box the chain ended on. This is exact, not an
    // approximation of the intermediate steps: scaleMoleculesRigid wraps each molecule's centre into
    // [0, L) and then multiplies by the per-axis ratio, so a wrapped centre stays wrapped and the
    // composition of two scalings equals the single scaling by the product of their ratios, with
    // internal offsets untouched in both. tests/soup-area-move.test.ts checks that composition
    // property directly (no GPU). The upshot is that a REJECTED trial costs no GPU work at all --
    // unlike the membrane engine's version, which pays two potential passes and a grid resize per
    // trial because its energy lives on the GPU.
    if (curBox[0] !== startBox[0] || curBox[1] !== startBox[1] || curBox[2] !== startBox[2]) {
      await applyBoxScaleOnce(rt, particles, bonds, curBox)
    }

    return {
      trials,
      accepted,
      acceptedFraction: trials > 0 ? accepted / trials : 0,
      box: curBox,
      lateralTrajectory,
      energyStart,
      energyEnd: energy,
    }
  }
}
