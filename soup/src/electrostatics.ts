// Task 'electrostatics' (2026-08-20): the screened-Coulomb interaction and the PROTONATION
// EQUILIBRIUM, on the CPU -- the twin of soup/wgsl/electrostatics.wgsl (which carries the force only)
// and the single place both the GPU uniform and the Monte Carlo that moves protonation states are
// derived from, so the two cannot drift.
//
// WHY, in this project's own measured terms. `final-campaign-report.md` §8 named its binding
// constraint "(3) CONNECTIVITY -- the supply PERCOLATES": at the two-tailed-rich composition the
// organic phase wrapped 3 of 3 axes at every wet checkpoint, i.e. the object had no inside to
// enclose anything in. Neutral heads predict exactly that -- nothing sets an effective head area, so
// nothing stops aggregates cross-linking. Real fatty acids are carboxylic ACIDS: the head is a
// protonation equilibrium (COOH <-> COO- + H+), the deprotonated head carries -e, like heads repel,
// and that repulsion IS what sets a0 in Israelachvili's p = v/(a0*l_c). The fatty-acid vesicle window
// (pH ~7-9; apparent pKa falling ~0.7 as NaCl rises 10 -> 100 mM -- design.md's "Gate 5", PMC7575682)
// exists BECAUSE of that equilibrium: above it micelles, below it oil/crystal.
//
// THE SCHEME, and what each choice biases.
//
// 1. INTERACTION -- Debye-Huckel screened Coulomb, U(r) = kT*(l_B/sigma)*q_i*q_j*exp(-r/lambda_D)/r,
//    shifted-force truncated at the SAME nonbonded cutoff wca_cut(b_tt)+wc the attraction already
//    ends at. Standard CG treatment of an implicit electrolyte (Debye & Huckel 1923; Israelachvili
//    3rd ed. ch. 14; MARTINI's own screened Coulomb cut at 1.2 nm, Marrink et al. JPCB 111 (2007)
//    7812 -- and note this model's cutoff, 2.7225 sigma, is 1.8-2.7 nm over the sigma->nm range
//    below, i.e. LONGER-ranged than MARTINI's own electrostatics cutoff, not shorter). Bias: a
//    mean-field screening length is not explicit counterions, so it cannot produce ion condensation,
//    charge inversion, or any correlation between the screening cloud and the interface's shape.
//
// 2. THE SIGMA -> NM MAPPING IS A RANGE, AND THE RANGE IS CARRIED. This project has no length
//    calibration (see data/soup.json's electrostatics.basis): sigma is a Cooke & Deserno bead.
//    Bracketing it by the measured area per lipid (1.1777 sigma^2 against ~0.6-0.7 nm^2 for a
//    two-tailed lipid -> sigma ~ 0.71-0.77 nm) and by the measured bilayer thickness (4.799 sigma
//    against 4-5 nm -> sigma ~ 0.83-1.04 nm) gives sigma in [0.65, 1.00] nm, and the shipped value is
//    0.80 nm. That range is exactly the uncertainty in the interaction strength, because
//    A = kT*l_B/sigma and kappa = sigma/lambda_D both scale with it: A in [0.781, 1.202] eps*sigma and
//    lambda_D in [0.961, 1.479] sigma at 100 mM, which moves the charged-head contact repulsion
//    (U_sf at the head-head bead separation, i.e. beadSizes.head_head) over [0.2245, 0.4234] kT --
//    a factor 1.886 between the ends. No calibration is
//    invented to narrow it; every result below is reported with that factor attached.
//
// 3. PROTONATION IS A DYNAMICAL VARIABLE, NOT A LABEL. Discrete constant-pH Monte Carlo: pick a head
//    at random, propose flipping its state, accept by Metropolis on
//      dG(protonated -> deprotonated) = kT*ln(10)*(pKa_intrinsic - pH) + dU_es
//    where dU_es is the screened-Coulomb work of putting the charge into THIS bead's own current
//    environment. This is the standard constant-pH MC / semi-grand form (Baptista, Teixeira &
//    Soares, J. Chem. Phys. 117 (2002) 4184; Mongan & Case, Curr. Opin. Struct. Biol. 15 (2005) 157;
//    Donnini et al., JCTC 7 (2011) 1962), and it is what makes the pKa SHIFT emergent rather than
//    inserted: pKa_intrinsic is the MONOMER acid's pKa (rank B literature), and any elevation of the
//    apparent pKa at a crowded charged interface has to be produced by dU_es. Detailed balance:
//    single-site flips with a randomly chosen site, forward and reverse dG differing only in sign at
//    fixed environment, so P(f)/P(r) = exp(-dG/kT) exactly -- pinned numerically in
//    tests/soup-electrostatics.test.ts against the exact Boltzmann weights of a two-head system.
//    Bias, stated: (a) the sweep runs every `sweepEverySteps` integration steps, so protonation is
//    quasi-static with respect to the dynamics rather than continuously coupled -- if the real
//    relaxation of the H+ equilibrium were slower than that, this over-equilibrates it; (b) there is
//    no explicit H+ and no titration of the medium, so pH is an external constant and a crowded
//    interface cannot locally deplete protons; (c) charge state does not enter the BOND Metropolis
//    (which uses only its rule's energyKT, as it already did for WCA and attraction).
//
// 4. NO ACID-SOAP DIMER IS INSERTED. The hydrogen-bonded COOH...COO- pair that behaves as one
//    double-tailed amphiphile is NOT added as a species and NOT given an attraction term of its own:
//    data/soup.json's polarPolar depth is 0 and this task does not touch it. So the only thing that
//    can produce pairing here is the interaction itself -- a charged head's like neighbours are
//    penalised, so its nearest head neighbour is preferentially NEUTRAL, which is the acid-soap pair's
//    structural signature (charge alternation) without its hydrogen bond. `pairingStats` below measures
//    exactly that and nothing more, and the report says which of the two it is.
//
// Every number comes from data/soup.json's `electrostatics` section (ranks and bases there) or from
// data/params.json's rank-A kT/sigma/wc; tests/params.test.ts's literal scanner enforces that over
// all of soup/src.

import { wcaCutoff, type Params } from '../../engine/src/params'
import type { Soup } from './rules'
// Task 'acid-soap-pairing': pair well + per-site work in their OWN module, soup/src/acid-soap.ts.
import { acidSoapSiteWork, type AcidSoapWell } from './acid-soap'
import { acidSoapScaleOf } from './soup-attraction'

/** Everything the force term, the potential term and the protonation MC need, derived once. */
export interface EsBasis {
  enabled: boolean
  /** kind index (position in data/soup.json's `monomers`) of the titratable species. */
  chargedKind: number
  /** Charge of the DEPROTONATED state, in units of e (negative: a carboxylate). */
  chargeDeprotonated: number
  /** A = kT * l_B/sigma, in eps*sigma (kT itself is in eps). */
  coeffA: number
  /** kappa = sigma/lambda_D, in sigma^-1. */
  kappa: number
  /** Shifted-force cutoff, in sigma. Task 'long-range-electrostatics' (2026-08-20): this is the
   * DEDICATED electrostatic cutoff, longRangeDebyeLengths * lambda_D capped by the minimum-image
   * ceiling -- NOT the Lennard-Jones one any more. */
  cutoff: number
  /** The engine's existing nonbonded cutoff wca_cut(b_tt)+wc = 2.7224620 sigma. Kept only for
   * reporting: it is what `cutoff` USED to be, and the number every published truncation figure of
   * the predecessor task was measured at. */
  nbCutoff: number
  /** Radius at which the force term is SPLIT between nonbondedSoup (which keeps everything below
   * it, where the existing neighbour walk already guarantees completeness) and the dedicated
   * long-range pass (which adds exactly the remainder up to `cutoff`). Equal to the grid's own
   * interactionRange, so no existing completeness guarantee is relied on beyond its own reach. */
  splitRadius: number
  /** cutoff + verletList.skin -- the radius the dedicated head-only list is BUILT to, so it stays
   * complete for verletList.rebuildEvery steps under the same drift bound the main list uses. */
  listRange: number
  /** Per-head capacity of that list (data/soup.json's longRangeListCapacity). */
  listCapacity: number
  /** cutoff / lambda_D -- how many Debye lengths the cutoff spans. The number that says whether the
   * range is right, and the one the report states per ionic strength. */
  debyeLengthsSpanned: number
  /** exp(-x)*(1+x) at x = debyeLengthsSpanned: the fraction of the INTEGRATED interaction
   * (int u(r) 4 pi r^2 dr) still discarded by the truncation. Exact for a Yukawa. */
  discardedIntegratedFraction: number
  /** exp(-x): the fraction of the unscreened CONTACT value still standing at the cutoff -- the
   * quantity the predecessor task reported as "removes 49 %". */
  discardedContactFraction: number
  /** longRangeDebyeLengths * lambda_D before the minimum-image cap, and the cap itself, so a report
   * can say WHICH of the two bound the cutoff. */
  targetCutoff: number
  imageCap: number
  /** u1(rc) and f1(rc), the shift constants, for UNIT charge product. */
  shiftU: number
  shiftF: number
  pKaIntrinsic: number
  pH: number
  ionicStrengthMolar: number
  sigmaNm: number
  /** lambda_D in sigma, kept for reporting. */
  debyeSigma: number
  sweepEverySteps: number
  kT: number
  /** Task 'acid-soap-pairing': the pair well; the constant-pH criterion needs it. 0 = pre-task. */
  acidSoap: AcidSoapWell
}

export interface EsOverrides {
  enabled?: boolean
  /** Task 'acid-soap-pairing': normalised pair depth for this system (absent = the file's own). */
  acidSoapScale?: number
  pH?: number
  ionicStrengthMolar?: number
  /** Task 'long-range-electrostatics' (2026-08-20): the SMALLEST box side this run will ever visit
   * (createSoup passes min(liveBox, dryBox)), which is what the minimum-image ceiling
   * longRangeMaxBoxFraction * this is computed from. Omitted -> no ceiling, which is only correct
   * for a caller that has already bounded the cutoff itself. */
  minBoxSigma?: number
  /** Task 'long-range-electrostatics' (2026-08-20): how many titratable beads the system holds. Used
   * ONLY to derive the long-range list's per-head capacity from the tightest box's own head density
   * (see EsBasis.listCapacity). Omitted -> the file's fallback longRangeListCapacity. */
  maxHeads?: number
  /** An EXPLICIT cutoff in sigma, bypassing both the lambda_D multiple and the ceiling. Exists for
   * the range-CONVERGENCE study (tests/es-range-calibration.test.ts), which has to sweep the cutoff
   * on a frozen configuration; never used by a real run. */
  cutoffSigma?: number
  /** Task 'confined-parcel' (2026-08-21): the volume the titratable beads are actually CONFINED to at
   * the tightest box this run visits -- the parcel's volume, not min(box)^3. Omitted (every
   * unconfined run) keeps the pre-task min(box)^3 denominator exactly. It matters because the
   * confined geometry puts the whole system inside a sphere occupying ~1/5.8 of the box, so
   * maxHeads/min(box)^3 under-states the real head density by that factor and would derive a
   * long-range list capacity ~20x too small -- a silently dropped interaction, which is the exact
   * defect the long-range task removed. */
  densityVolumeSigma3?: number
}

/** Derives the basis from data/soup.json + data/params.json, with per-run overrides for the two
 * EXPERIMENT-DESIGN numbers (pH, ionic strength). A soup file with no `electrostatics` section, or one
 * with `enabled: false` and no override, yields `coeffA = 0` -- which makes both the GPU term and this
 * module's own energies identically zero, so every pre-task run is bit-identical. */
export function makeEsBasis(soup: Soup, p: Params, over?: EsOverrides): EsBasis {
  const es = soup.electrostatics
  const kT = p.thermostat.kT
  const cutoff = wcaCutoff(p.sigma * p.beadSizes.tail_tail) + p.attraction.wc
  const interactionRange = wcaCutoff(p.sigma * Math.max(...soup.monomers.map((m) => m.radiusSigma))) + p.attraction.wc
  if (!es) {
    return {
      enabled: false, chargedKind: -1, chargeDeprotonated: 0, coeffA: 0, kappa: 0, cutoff,
      nbCutoff: cutoff, splitRadius: interactionRange, listRange: cutoff, listCapacity: 0,
      debyeLengthsSpanned: 0, discardedIntegratedFraction: 0, discardedContactFraction: 0,
      targetCutoff: 0, imageCap: 0,
      shiftU: 0, shiftF: 0, pKaIntrinsic: 0, pH: 0, ionicStrengthMolar: 0, sigmaNm: 0,
      debyeSigma: 0, sweepEverySteps: 0, kT,
      acidSoap: { scale: 0, rcAttr: 0, wc: 0, epsilon: 0 },
    }
  }
  const enabled = over?.enabled ?? es.enabled
  const pH = over?.pH ?? es.pH
  const ionicStrengthMolar = over?.ionicStrengthMolar ?? es.ionicStrengthMolar
  const chargedKind = soup.monomers.findIndex((m) => m.id === es.chargedKind)
  if (chargedKind < 0) throw new Error(`electrostatics.chargedKind='${es.chargedKind}' нет среди monomers`)
  if (!(ionicStrengthMolar > 0)) throw new Error(`electrostatics: ionicStrengthMolar должен быть > 0, дано ${ionicStrengthMolar}`)
  if (!(es.sigmaToNm > 0)) throw new Error(`electrostatics: sigmaToNm должен быть > 0, дано ${es.sigmaToNm}`)
  const sigmaNm = es.sigmaToNm
  // lambda_D = (lambda_D at 1 M) / sqrt(I[M]), then expressed in sigma.
  const debyeSigma = es.debyeLengthNmAtUnitMolar / Math.sqrt(ionicStrengthMolar) / sigmaNm
  const coeffA = enabled ? kT * (es.bjerrumLengthNm / sigmaNm) : 0
  const kappa = 1 / debyeSigma
  // Task 'long-range-electrostatics' (2026-08-20). THE CUTOFF IS NO LONGER THE LENNARD-JONES ONE.
  // rc_es = longRangeDebyeLengths * lambda_D, i.e. a multiple of the interaction's OWN decay length,
  // capped by the minimum-image ceiling longRangeMaxBoxFraction * min(box) and floored at the old
  // nonbonded cutoff (so it can only ever be longer, never shorter, than what the predecessor ran).
  // Expressing it as a multiple of lambda_D rather than as a length is the whole point: the
  // discarded fraction exp(-x)*(1+x) is then IDENTICAL in every ionic strength, so a salt comparison
  // is no longer biased by truncation in the low-salt arm alone (which is exactly what made the
  // predecessor's measured pKa shift a lower bound: 84 % discarded at 10 mM against 34 % at 100 mM).
  const targetCutoff = es.longRangeDebyeLengths * debyeSigma
  const imageCap = over?.minBoxSigma !== undefined ? es.longRangeMaxBoxFraction * over.minBoxSigma : Number.POSITIVE_INFINITY
  const esCutoff = over?.cutoffSigma ?? Math.max(cutoff, Math.min(targetCutoff, imageCap))
  if (over?.minBoxSigma !== undefined && esCutoff > over.minBoxSigma / 2) {
    throw new Error(
      `electrostatics: обрезка rc_es=${esCutoff.toFixed(6)} нарушает соглашение минимального образа ` +
        `при min(box)=${over.minBoxSigma.toFixed(4)} (нужно rc_es <= ${(over.minBoxSigma / 2).toFixed(4)}) -- ` +
        `бокс слишком мал для ${es.longRangeDebyeLengths} дебаевских длин при I=${ionicStrengthMolar} М`,
    )
  }
  // The long-range list's per-head capacity, DERIVED from the tightest box this run visits, not typed.
  // Getting this wrong is not a performance question: an overflow is a silently dropped interaction,
  // which is the exact defect this task removes. It is computed at the UNIFORM head density of the
  // SMALLEST box (min(box)^3 under-counts the volume of a non-cubic box, i.e. over-counts the density,
  // which is the safe direction), times longRangeListSafetyFactor for the fact that heads sit on an
  // aggregate's surface. Measured on the predecessor's own box-54 wet checkpoint: mean 1154.2, max
  // 1332 neighbours at listRange 16.7, i.e. an inhomogeneity factor of 1.15 -- and the DRY box of the
  // same run, at 3.2x the head density, needs ~3690, which is what overflowed a capacity of 2500.
  const listRange = esCutoff + soup.verletList.skin
  let listCapacity = es.longRangeListCapacity
  if (over?.maxHeads !== undefined && over.maxHeads > 0 && over?.minBoxSigma !== undefined) {
    const denom = over.densityVolumeSigma3 !== undefined && over.densityVolumeSigma3 > 0 ? over.densityVolumeSigma3 : over.minBoxSigma ** 3
    const uniform = ((4 * Math.PI) / 3) * listRange ** 3 * (over.maxHeads / denom)
    listCapacity = Math.max(64, Math.min(over.maxHeads, Math.ceil(es.longRangeListSafetyFactor * uniform)))
  }
  const shiftU = coeffA * Math.exp(-kappa * esCutoff) / esCutoff
  const shiftF = shiftU * (1 / esCutoff + kappa)
  const x = esCutoff / debyeSigma
  return {
    enabled, chargedKind, chargeDeprotonated: es.chargeDeprotonated, coeffA, kappa, cutoff: esCutoff,
    nbCutoff: cutoff, splitRadius: interactionRange, listRange, listCapacity,
    debyeLengthsSpanned: x,
    discardedIntegratedFraction: Math.exp(-x) * (1 + x),
    discardedContactFraction: Math.exp(-x),
    targetCutoff, imageCap,
    shiftU, shiftF, pKaIntrinsic: es.pKaIntrinsic, pH, ionicStrengthMolar, sigmaNm, debyeSigma,
    sweepEverySteps: es.sweepEverySteps, kT,
    // Task 'acid-soap-pairing': the pair well -- same rank-A shape constants, same normalised depth.
    acidSoap: {
      scale: acidSoapScaleOf(soup, over?.acidSoapScale),
      rcAttr: wcaCutoff(p.sigma * p.beadSizes.tail_tail),
      wc: p.attraction.wc,
      epsilon: p.epsilon,
    },
  }
}

/** The second vec4 soup/wgsl/electrostatics.wgsl's ES2 uniform expects: [splitRadius, listRange,
 * listCapacity, 0]. The force term is split BY RADIUS between nonbondedSoup (r < splitRadius) and
 * the dedicated long-range pass (splitRadius <= r < cutoff), with the same shift constants, so no
 * force path can lose electrostatics wholesale and the two halves sum to the untruncated term. */
export function esUniform2(b: EsBasis): Float32Array {
  return new Float32Array([b.splitRadius, b.listRange, b.listCapacity, b.chargedKind])
}

/** One line with every number that decides whether the RANGE is right: the cutoff, what bound it,
 * how many Debye lengths it spans and what fraction of the interaction it still discards. Printed by
 * the tests and by the campaign so no report has to re-derive it. */
export function esRangeSummary(b: EsBasis): string {
  const bound = b.cutoff <= b.nbCutoff + 1e-9 ? 'nbCutoff' : b.cutoff < b.targetCutoff - 1e-9 ? 'minImage' : 'debyeMultiple'
  return (
    `ES-RANGE I=${b.ionicStrengthMolar} lambdaD=${b.debyeSigma.toFixed(4)}sig rc_es=${b.cutoff.toFixed(4)}sig ` +
    `(=${b.debyeLengthsSpanned.toFixed(3)} lambdaD, bound=${bound}, target=${b.targetCutoff.toFixed(4)}, ` +
    `imageCap=${Number.isFinite(b.imageCap) ? b.imageCap.toFixed(4) : 'none'}) ` +
    `discarded_integrated=${b.discardedIntegratedFraction.toFixed(4)} discarded_contact=${b.discardedContactFraction.toFixed(4)} ` +
    `nbCutoff_was=${b.nbCutoff.toFixed(7)} listRange=${b.listRange.toFixed(4)} cap=${b.listCapacity}`
  )
}

/** The vec4 soup/wgsl/electrostatics.wgsl's ES uniform expects: [A, kappa, rc, F(rc)]. */
export function esUniform(b: EsBasis): Float32Array {
  return new Float32Array([b.coeffA, b.kappa, b.cutoff, b.shiftF])
}

/** Shifted-force screened-Coulomb PAIR ENERGY for charge product qq. Zero at and beyond the cutoff. */
export function esPairEnergy(r: number, qq: number, b: EsBasis): number {
  if (b.coeffA === 0 || qq === 0 || r >= b.cutoff || r <= 0) return 0
  const u1 = (b.coeffA * Math.exp(-b.kappa * r)) / r
  return qq * (u1 - b.shiftU + b.shiftF * (r - b.cutoff))
}

/** Shifted-force screened-Coulomb PAIR FORCE MAGNITUDE (positive = repulsive), = -d/dr of
 * esPairEnergy. The WGSL twin is esForceMag in soup/wgsl/electrostatics.wgsl. */
export function esPairForceMag(r: number, qq: number, b: EsBasis): number {
  if (b.coeffA === 0 || qq === 0 || r >= b.cutoff || r <= 0) return 0
  const u1 = (b.coeffA * Math.exp(-b.kappa * r)) / r
  return qq * (u1 * (1 / r + b.kappa) - b.shiftF)
}

/** Henderson-Hasselbalch deprotonated fraction of an ISOLATED acid -- the value the MC below must
 * reproduce when dU_es is zero, which is how its sampling is checked rather than asserted. */
export function hendersonAlpha(pH: number, pKa: number): number {
  return 1 / (1 + Math.pow(10, pKa - pH))
}

/** Apparent pKa implied by a MEASURED deprotonated fraction at a known pH -- Henderson-Hasselbalch
 * read backwards. This is how the interfacial pKa SHIFT is measured (the shift is
 * pKa_apparent - pKa_intrinsic), never assumed. */
export function apparentPKa(alpha: number, pH: number): number {
  if (alpha <= 0 || alpha >= 1) return NaN
  return pH - Math.log10(alpha / (1 - alpha))
}

/** PCG32-based uniform stream. Same generator family soup/wgsl/step.wgsl's pcgSoup uses, on the CPU,
 * with an EXPLICIT state so the protonation chain can be checkpointed and resumed exactly. */
export interface PcgState { state: number }
export function pcgNext(s: PcgState): number {
  // 32-bit PCG-XSH-RR-style mix, all arithmetic forced through >>> 0 so it stays exact in doubles.
  let v = (Math.imul(s.state, 747796405) + 2891336453) >>> 0
  s.state = v
  v = (Math.imul((v >>> ((v >>> 28) + 4)) ^ v, 277803737) >>> 0)
  v = ((v >>> 22) ^ v) >>> 0
  return v / 4294967296
}

export interface SweepResult {
  attempts: number
  accepted: number
  heads: number
  deprotonated: number
  alpha: number
  /** Mean electrostatic work over ACCEPTED deprotonation moves, in kT -- the quantity the apparent
   * pKa shift is made of. */
  dEsMeanKT: number
}

/** Cell list over the CHARGEABLE beads only, for the neighbour sums below. Heads are a few percent of
 * N (9296 of 191778 in the decisive composition), so this is cheap even at full scale. */
function headCells(positions: Float32Array, heads: Int32Array, box: [number, number, number], cutoff: number) {
  const nx = Math.max(1, Math.floor(box[0] / cutoff))
  const ny = Math.max(1, Math.floor(box[1] / cutoff))
  const nz = Math.max(1, Math.floor(box[2] / cutoff))
  const wx = box[0] / nx, wy = box[1] / ny, wz = box[2] / nz
  const buckets = new Map<number, number[]>()
  const key = (x: number, y: number, z: number): number => {
    const cx = Math.min(nx - 1, Math.floor(((x % box[0]) + box[0]) % box[0] / wx))
    const cy = Math.min(ny - 1, Math.floor(((y % box[1]) + box[1]) % box[1] / wy))
    const cz = Math.min(nz - 1, Math.floor(((z % box[2]) + box[2]) % box[2] / wz))
    return cx + nx * (cy + ny * cz)
  }
  for (let h = 0; h < heads.length; h++) {
    const i = heads[h]
    const k = key(positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2])
    const b = buckets.get(k)
    if (b) b.push(i)
    else buckets.set(k, [i])
  }
  // Task 'long-range-electrostatics' (2026-08-20): fills a CALLER-OWNED buffer and returns the count,
  // instead of allocating a fresh array per call. With the cutoff now up to 15.2 sigma the cell grid
  // is only 2-3 cells per axis at these boxes, so the walk legitimately returns most of the heads --
  // 9296 of them, once per attempt, 9296 attempts per sweep. Allocating and growing an array 86
  // million times per sweep was measured to dominate the sweep's whole cost; the arithmetic itself
  // does not. Same candidates, same order, no allocation.
  const neighboursInto = (i: number, out: Int32Array): number => {
    const x = ((positions[i * 4] % box[0]) + box[0]) % box[0]
    const y = ((positions[i * 4 + 1] % box[1]) + box[1]) % box[1]
    const z = ((positions[i * 4 + 2] % box[2]) + box[2]) % box[2]
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = Math.min(nz - 1, Math.floor(z / wz))
    let m = 0
    const seen = new Set<number>()
    for (let dz = -1; dz <= 1; dz++) {
      const az = ((cz + dz) % nz + nz) % nz
      for (let dy = -1; dy <= 1; dy++) {
        const ay = ((cy + dy) % ny + ny) % ny
        for (let dx = -1; dx <= 1; dx++) {
          const ax = ((cx + dx) % nx + nx) % nx
          const k = ax + nx * (ay + ny * az)
          if (seen.has(k)) continue
          seen.add(k)
          const bb = buckets.get(k)
          if (bb) for (let q = 0; q < bb.length; q++) out[m++] = bb[q]
        }
      }
    }
    return m
  }
  const neighboursOf = (i: number): number[] => {
    const x = ((positions[i * 4] % box[0]) + box[0]) % box[0]
    const y = ((positions[i * 4 + 1] % box[1]) + box[1]) % box[1]
    const z = ((positions[i * 4 + 2] % box[2]) + box[2]) % box[2]
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = Math.min(nz - 1, Math.floor(z / wz))
    const out: number[] = []
    const seen = new Set<number>()
    for (let dz = -1; dz <= 1; dz++) {
      const az = ((cz + dz) % nz + nz) % nz
      for (let dy = -1; dy <= 1; dy++) {
        const ay = ((cy + dy) % ny + ny) % ny
        for (let dx = -1; dx <= 1; dx++) {
          const ax = ((cx + dx) % nx + nx) % nx
          const k = ax + nx * (ay + ny * az)
          if (seen.has(k)) continue
          seen.add(k)
          const b = buckets.get(k)
          if (b) for (const j of b) out.push(j)
        }
      }
    }
    return out
  }
  return { neighboursOf, neighboursInto }
}

function mi(d: number, L: number): number {
  return d - Math.round(d / L) * L
}
function dist2(positions: Float32Array, i: number, j: number, box: [number, number, number]): number {
  const dx = mi(positions[i * 4] - positions[j * 4], box[0])
  const dy = mi(positions[i * 4 + 1] - positions[j * 4 + 1], box[1])
  const dz = mi(positions[i * 4 + 2] - positions[j * 4 + 2], box[2])
  return dx * dx + dy * dy + dz * dz
}
function dist(positions: Float32Array, i: number, j: number, box: [number, number, number]): number {
  const dx = mi(positions[i * 4] - positions[j * 4], box[0])
  const dy = mi(positions[i * 4 + 1] - positions[j * 4 + 1], box[1])
  const dz = mi(positions[i * 4 + 2] - positions[j * 4 + 2], box[2])
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

/** Indices of every bead of the titratable species, in index order. */
export function headIndices(positions: Float32Array, chargedKind: number): Int32Array {
  const n = positions.length / 4
  const out: number[] = []
  for (let i = 0; i < n; i++) if ((positions[i * 4 + 3] | 0) === chargedKind) out.push(i)
  return Int32Array.from(out)
}

/** ONE constant-pH Monte Carlo sweep: `heads.length` single-site flip attempts at randomly chosen
 * heads. MUTATES `charges` in place and returns what it did. Pure (no GPU, no fs) so it can be
 * exercised in Node, which is what makes the detailed-balance pin cheap. */
export function protonationSweep(
  positions: Float32Array,
  charges: Float32Array,
  box: [number, number, number],
  b: EsBasis,
  rng: PcgState,
): SweepResult {
  const heads = headIndices(positions, b.chargedKind)
  const cells = headCells(positions, heads, box, b.cutoff)
  // dG_intrinsic for protonated -> deprotonated, in ENERGY units (eps): kT*ln(10)*(pKa - pH).
  const dGintr = b.kT * Math.LN10 * (b.pKaIntrinsic - b.pH)
  const q = b.chargeDeprotonated
  let attempts = 0
  let accepted = 0
  let dEsSum = 0
  let dEsCount = 0
  const nbuf = new Int32Array(heads.length)
  const cut2 = b.cutoff * b.cutoff
  for (let a = 0; a < heads.length; a++) {
    const i = heads[Math.min(heads.length - 1, Math.floor(pcgNext(rng) * heads.length))]
    attempts++
    // Electrostatic work of HAVING the charge on i, given every other charge as it stands now. The
    // reject test is on the SQUARED distance (task 'long-range-electrostatics', 2026-08-20): at the
    // long cutoff most candidates the cell walk returns are out of range, and a sqrt per rejection is
    // the single most expensive thing in the sweep.
    let uEs = 0
    const m = cells.neighboursInto(i, nbuf)
    for (let s = 0; s < m; s++) {
      const j = nbuf[s]
      if (j === i || charges[j] === 0) continue
      const r2 = dist2(positions, i, j, box)
      if (r2 >= cut2) continue
      uEs += esPairEnergy(Math.sqrt(r2), q * charges[j], b)
    }
    // Task 'acid-soap-pairing' (2026-08-23): the SAME "work of having the charge on i" for the pair
    // term, over this walk's own buffer (the well's reach is far shorter than b.cutoff, so the
    // head-only list covers it); 0 at depth 0. Not optional: see acidSoapSiteWork's own header.
    uEs += acidSoapSiteWork(positions, charges, i, nbuf, m, box, b.acidSoap)
    const isCharged = charges[i] !== 0
    // Forward (protonated -> deprotonated) costs +dGintr +uEs; the reverse is the same number negated
    // at fixed environment, which is exactly what makes P(f)/P(r) = exp(-dG/kT).
    const dG = isCharged ? -(dGintr + uEs) : dGintr + uEs
    if (dG <= 0 || pcgNext(rng) < Math.exp(-dG / b.kT)) {
      charges[i] = isCharged ? 0 : q
      accepted++
      if (!isCharged) {
        dEsSum += uEs / b.kT
        dEsCount++
      }
    }
  }
  let deprotonated = 0
  for (let h = 0; h < heads.length; h++) if (charges[heads[h]] !== 0) deprotonated++
  return {
    attempts,
    accepted,
    heads: heads.length,
    deprotonated,
    alpha: heads.length > 0 ? deprotonated / heads.length : 0,
    dEsMeanKT: dEsCount > 0 ? dEsSum / dEsCount : 0,
  }
}

export interface PairingStats {
  heads: number
  /** Heads with at least one head neighbour inside `contactRadius`. */
  withNeighbour: number
  /** Of those contacts, how many are UNLIKE (one charged, one neutral) -- the acid-soap pair's own
   * structural signature. */
  unlikeContacts: number
  likeContacts: number
  /** Fraction of contacts that are unlike, and the fraction a RANDOM assignment of the same alpha
   * would give (2*alpha*(1-alpha)) -- the null this must beat to be a correlation at all. */
  unlikeFraction: number
  unlikeFractionRandom: number
  /** Heads counted as PAIRED: nearest head neighbour inside contactRadius and of unlike charge. */
  paired: number
  pairedFraction: number
  contactRadius: number
  alpha: number
}

/** Acid-soap pairing, measured and not assumed. `contactRadius` is the head-head WCA contact
 * distance wca_cut(sigma*head_head) -- the separation at which two heads are touching, derived from
 * data/params.json's rank-A beadSizes, not chosen. */
export function pairingStats(
  positions: Float32Array,
  charges: Float32Array,
  box: [number, number, number],
  b: EsBasis,
  contactRadius: number,
): PairingStats {
  const heads = headIndices(positions, b.chargedKind)
  const cells = headCells(positions, heads, box, Math.max(contactRadius, b.cutoff))
  let withNeighbour = 0, unlike = 0, like = 0, paired = 0, charged = 0
  for (let h = 0; h < heads.length; h++) {
    const i = heads[h]
    if (charges[i] !== 0) charged++
    let best = Infinity
    let bestJ = -1
    let any = false
    for (const j of cells.neighboursOf(i)) {
      if (j === i) continue
      const r = dist(positions, i, j, box)
      if (r > contactRadius) continue
      any = true
      if ((charges[i] === 0) !== (charges[j] === 0)) unlike++
      else like++
      if (r < best) {
        best = r
        bestJ = j
      }
    }
    if (any) withNeighbour++
    if (bestJ >= 0 && (charges[i] === 0) !== (charges[bestJ] === 0)) paired++
  }
  const alpha = heads.length > 0 ? charged / heads.length : 0
  const contacts = unlike + like
  return {
    heads: heads.length,
    withNeighbour,
    unlikeContacts: unlike,
    likeContacts: like,
    unlikeFraction: contacts > 0 ? unlike / contacts : 0,
    unlikeFractionRandom: 2 * alpha * (1 - alpha),
    paired,
    pairedFraction: heads.length > 0 ? paired / heads.length : 0,
    contactRadius,
    alpha,
  }
}

/** TOTAL screened-Coulomb energy of a configuration -- every unordered pair of CHARGED beads within
 * the (now dedicated, long) cutoff, summed with the same shifted-force antiderivative the GPU force
 * differentiates. Task 'long-range-electrostatics' (2026-08-20): this replaces the term
 * soup/src/soup-potential.ts used to add inside its own pair loop, because that loop's cell list is
 * built at the LENNARD-JONES cutoff (2.7224620 sigma) and can no longer see the whole electrostatic
 * range. Its own cell list is over the titratable beads only -- a few percent of N -- so the longer
 * range costs almost nothing here, exactly as it does on the GPU (soup/wgsl/electrostatics.wgsl).
 * Identically 0 when the basis is disabled, which keeps every pre-task potential bit-identical. */
export function esTotalEnergy(
  positions: Float32Array,
  charges: Float32Array | undefined,
  box: [number, number, number],
  b: EsBasis,
): number {
  if (!b.enabled || b.coeffA === 0 || charges === undefined) return 0
  const heads = headIndices(positions, b.chargedKind)
  const cells = headCells(positions, heads, box, b.cutoff)
  let u = 0
  const nbuf = new Int32Array(heads.length)
  const cut2 = b.cutoff * b.cutoff
  for (let h = 0; h < heads.length; h++) {
    const i = heads[h]
    if (charges[i] === 0) continue
    const m = cells.neighboursInto(i, nbuf)
    for (let s = 0; s < m; s++) {
      const j = nbuf[s]
      if (j <= i || charges[j] === 0) continue
      const r2 = dist2(positions, i, j, box)
      if (r2 >= cut2) continue
      u += esPairEnergy(Math.sqrt(r2), charges[i] * charges[j], b)
    }
  }
  return u
}
