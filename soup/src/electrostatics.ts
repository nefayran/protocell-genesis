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
  /** Shifted-force cutoff, in sigma. */
  cutoff: number
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
}

export interface EsOverrides {
  enabled?: boolean
  pH?: number
  ionicStrengthMolar?: number
}

/** Derives the basis from data/soup.json + data/params.json, with per-run overrides for the two
 * EXPERIMENT-DESIGN numbers (pH, ionic strength). A soup file with no `electrostatics` section, or one
 * with `enabled: false` and no override, yields `coeffA = 0` -- which makes both the GPU term and this
 * module's own energies identically zero, so every pre-task run is bit-identical. */
export function makeEsBasis(soup: Soup, p: Params, over?: EsOverrides): EsBasis {
  const es = soup.electrostatics
  const kT = p.thermostat.kT
  const cutoff = wcaCutoff(p.sigma * p.beadSizes.tail_tail) + p.attraction.wc
  if (!es) {
    return {
      enabled: false, chargedKind: -1, chargeDeprotonated: 0, coeffA: 0, kappa: 0, cutoff,
      shiftU: 0, shiftF: 0, pKaIntrinsic: 0, pH: 0, ionicStrengthMolar: 0, sigmaNm: 0,
      debyeSigma: 0, sweepEverySteps: 0, kT,
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
  const shiftU = coeffA * Math.exp(-kappa * cutoff) / cutoff
  const shiftF = shiftU * (1 / cutoff + kappa)
  return {
    enabled, chargedKind, chargeDeprotonated: es.chargeDeprotonated, coeffA, kappa, cutoff,
    shiftU, shiftF, pKaIntrinsic: es.pKaIntrinsic, pH, ionicStrengthMolar, sigmaNm, debyeSigma,
    sweepEverySteps: es.sweepEverySteps, kT,
  }
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
  return { neighboursOf }
}

function mi(d: number, L: number): number {
  return d - Math.round(d / L) * L
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
  for (let a = 0; a < heads.length; a++) {
    const i = heads[Math.min(heads.length - 1, Math.floor(pcgNext(rng) * heads.length))]
    attempts++
    // Electrostatic work of HAVING the charge on i, given every other charge as it stands now.
    let uEs = 0
    for (const j of cells.neighboursOf(i)) {
      if (j === i || charges[j] === 0) continue
      const r = dist(positions, i, j, box)
      if (r >= b.cutoff) continue
      uEs += esPairEnergy(r, q * charges[j], b)
    }
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
