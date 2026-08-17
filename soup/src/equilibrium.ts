// Energy calibration (2026-08-17): rate-free rigor checks for the chain-growth bond energy,
// independent of any rank-D attemptRate -- the Flory equilibrium-length distribution and the
// van't Hoff recovery of the set energy from the bond fraction's temperature dependence. Pure CPU
// math (no GPU calls), reading the same flat particle/bond layout soup/src/amphiphile.ts's
// findAmphiphiles already takes -- this file does NOT modify that recogniser; it walks the
// CARBON-ONLY (C-C) bond subgraph directly, because the Flory relation this project's own
// diagnosis invokes ("mean chain length = 1/(1-p)") describes the free carbon
// homopolymerisation itself, not the head-capped amphiphile TAIL
// (`soup/src/amphiphile.ts`'s `Amphiphile.tailLengths`): `headPlacement.terminalOnly` freezes one
// end of a tail the moment a head lands on it, a separate structural constraint layered on top of
// the underlying C-C equilibrium, and the prior head-count sweep
// (.superpowers/sdd/2026-08-16-soup-to-vesicle/tail-length-report.md) already showed head
// availability is a secondary, weaker lever than the chain-bond energy itself -- so the raw C-C
// population, not the amphiphile-tail population, is the right one to check the Flory relation
// against.
//
// No numeric model constant lives here -- every energy, kT and concentration below is a
// caller-supplied argument (data/soup.json / data/params.json / a real measurement own those
// numbers); tests/params.test.ts's literal scanner covers this directory the same way it covers
// every other file in soup/src.

import type { Monomer } from './rules'

/** Every maximal run of C-C bonded carbon particles in a snapshot's bond graph, IGNORING any bond
 * to a non-carbon particle (a head, donor or catalyst) -- the free homopolymer population the
 * Flory relation models. A carbon with zero C-C bonds is its own length-1 chain (an unreacted
 * monomer, not excluded -- p_1 = (1-x) is the single largest term of the Flory distribution
 * itself, not a degenerate case to special-case away). Returns one entry per chain (its carbon
 * count), in no particular order. Pure function of `particles`/`bonds` (the same flat
 * vec4-per-particle / index-pair layout `soup/src/amphiphile.ts`'s `findAmphiphiles` takes) plus
 * `monomers` (for `kind === 'carbon'`) -- no GPU. */
export function carbonChainLengths(particles: Float32Array, bonds: Uint32Array, monomers: Monomer[]): number[] {
  const n = particles.length / 4
  const isCarbon = (i: number): boolean => monomers[Math.round(particles[i * 4 + 3])]?.kind === 'carbon'

  const parent = new Int32Array(n)
  for (let i = 0; i < n; i++) parent[i] = i
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]
      x = parent[x]
    }
    return x
  }
  for (let k = 0; k < bonds.length; k += 2) {
    const i = bonds[k]
    const j = bonds[k + 1]
    if (!isCarbon(i) || !isCarbon(j)) continue // only C-C edges join this graph -- a C-O edge never merges two carbon components
    const ra = find(i)
    const rb = find(j)
    if (ra !== rb) parent[ra] = rb
  }

  const sizeByRoot = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    if (!isCarbon(i)) continue
    const root = find(i)
    sizeByRoot.set(root, (sizeByRoot.get(root) ?? 0) + 1)
  }
  return Array.from(sizeByRoot.values())
}

/** Fraction of a carbon's own 2 chain-bond slots (`soup/wgsl/bond.wgsl`'s 2-slot carbon chain
 * pool -- architectural, unrelated to `headPlacement.chainCapacity`, which is the HEAD's own slot
 * count) that is actually occupied by a C-C bond: `(2 * ccEdges) / (2 * carbonCount)` = `ccEdges /
 * carbonCount`. This is the `p` the diagnosis's own `mean = 1/(1-p)` and `p = Kc/(1+Kc)` refer
 * to -- the probability that a given growth site continues the chain, i.e. the fraction of
 * maximally-possible C-C bonds (at this valence) actually realised -- not a bond count over total
 * particle count, and not a count over bonds of every kind (a C-O bond never occupies a C-C
 * slot). */
export function ccBondFraction(particles: Float32Array, bonds: Uint32Array, monomers: Monomer[]): number {
  const n = particles.length / 4
  const isCarbon = (i: number): boolean => monomers[Math.round(particles[i * 4 + 3])]?.kind === 'carbon'
  let carbonCount = 0
  for (let i = 0; i < n; i++) if (isCarbon(i)) carbonCount++
  if (carbonCount === 0) return 0
  let ccEdges = 0
  for (let k = 0; k < bonds.length; k += 2) {
    if (isCarbon(bonds[k]) && isCarbon(bonds[k + 1])) ccEdges++
  }
  return ccEdges / carbonCount
}

/** The Flory most-probable-distribution prediction for a reversible linear homopolymerisation at
 * detailed balance: given the bond energy actually set in `data/soup.json` (`energyKT`, in the
 * same units `soup/src/rules.ts`'s `acceptanceProbability` divides by `kT`), the system's actual
 * thermostat `kT`, and `carbonConcentration` -- see this module's own report for how that number
 * is obtained; it is NOT the raw carbon number density, because bond FORMATION here requires a
 * real spatial encounter (this is a genuinely spatial Monte Carlo, not a well-mixed mass-action
 * system) while only bond BREAKING carries the Metropolis `exp(-energyKT/kT)` factor, so the
 * product `K*carbonConcentration` folds together the pure Boltzmann energy dependence AND an
 * encounter-probability factor this module does not attempt to derive analytically. That encounter
 * factor is a property of the box/composition/dynamics alone (unaffected by `energyKT`), so it is
 * legitimate to calibrate it ONCE against an independently measured baseline (a DIFFERENT energy,
 * already-collected data) and hold it fixed while predicting a NEW energy's distribution -- exactly
 * the "no fitting to the target" falsifiability check this calibration requires. Returns the
 * normalised probability that a chain drawn at random has exactly `n` carbons, for `n` =
 * `1..maxLength`.
 *
 * Derivation: for a chain end with equilibrium constant `K = exp(energyKT / kT)` for adding one
 * more carbon unit at concentration `c`, the continuation probability is `x = Kc / (1 + Kc)` (the
 * calibration's own "p = Kc/(1+Kc)"), and the resulting length distribution is geometric, `p_n =
 * (1 - x) * x^(n-1)` -- the classic Flory most-probable distribution, whose mean is exactly
 * `1/(1-x)`. Renormalised over the returned `[1, maxLength]` window so truncation past
 * `maxLength` does not silently leave the reported distribution summing to less than 1. */
export function floryPrediction(
  energyKT: number,
  kT: number,
  carbonConcentration: number,
  maxLength: number,
): Record<number, number> {
  const K = Math.exp(energyKT / kT)
  const Kc = K * carbonConcentration
  const x = Kc / (1 + Kc)
  const raw: number[] = []
  for (let n = 1; n <= maxLength; n++) raw.push((1 - x) * x ** (n - 1))
  const total = raw.reduce((a, b) => a + b, 0)
  const out: Record<number, number> = {}
  for (let n = 1; n <= maxLength; n++) out[n] = raw[n - 1] / total
  return out
}

/** Ordinary-least-squares van't Hoff recovery: regresses `ln[p/(1-p)]` (`points[i].bondFraction`
 * is `p`, e.g. `ccBondFraction` above) against `1/kT`. Since `p/(1-p) = Kc = c *
 * exp(energyKT/kT)`, `ln[p/(1-p)] = ln(c) + energyKT*(1/kT)` -- a straight line in `1/kT` whose
 * SLOPE is the bond energy actually governing the measured bond fraction, independent of the
 * (unknown, and irrelevant to this recovery) concentration term folded into the intercept.
 * `slope` is that recovered energy; `intercept` is `ln(c)`; `r2` is the coefficient of
 * determination, reported so a caller can see whether the points actually lie on a line before
 * trusting the slope at all. */
export function vanHoffSlope(
  points: { kT: number; bondFraction: number }[],
): { slope: number; intercept: number; r2: number } {
  const xs = points.map((p) => 1 / p.kT)
  const ys = points.map((p) => Math.log(p.bondFraction / (1 - p.bondFraction)))
  const n = xs.length
  const xm = xs.reduce((a, b) => a + b, 0) / n
  const ym = ys.reduce((a, b) => a + b, 0) / n
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - xm) * (ys[i] - ym)
    sxx += (xs[i] - xm) ** 2
  }
  const slope = sxy / sxx
  const intercept = ym - slope * xm
  let ssRes = 0
  let ssTot = 0
  for (let i = 0; i < n; i++) {
    const pred = intercept + slope * xs[i]
    ssRes += (ys[i] - pred) ** 2
    ssTot += (ys[i] - ym) ** 2
  }
  const r2 = ssTot > 0 ? 1 - ssRes / ssTot : 1
  return { slope, intercept, r2 }
}
