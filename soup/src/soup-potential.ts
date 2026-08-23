// Total potential energy of a soup configuration, on the CPU.
//
// Task 'hydrophobic-asymmetry' (2026-08-19), defect 2: the zero-tension Monte Carlo area move
// (soup/src/soup-area-move.ts) needs U(configuration, box) for its Metropolis criterion, and the
// soup path has no potential kernel at all -- soup/src/sim.ts's own header states it outright
// ("step.wgsl adds no potential of its own"), unlike the membrane engine, whose force kernel writes
// per-particle potential into potentialBuf for exactly this purpose. Rather than duplicate the
// GPU cell walk in WGSL a second time (soup/wgsl/step.wgsl is already near the file-size limit and
// the walk would have to be maintained twice), the energy is evaluated here and PROVEN consistent
// with the GPU force field by measurement: tests/soup-area-move.test.ts differentiates this
// function numerically along one particle's coordinate and compares against SoupSystem.forces()
// (F = -grad U), which is a stronger check than reading both listings side by side.
//
// Every formula below is the antiderivative already written in engine/wgsl/forces.wgsl
// (wca_v/fene_v/bend_v/attr_v) -- transcribed, not invented -- and the term structure mirrors
// soup/wgsl/step.wgsl's own soupForceWalk/bondedForce exactly:
//  - WCA and the attraction act on EVERY pair inside the interaction range, bonded pairs included
//    (soupForceWalk's cell walk skips only j == i; unlike the membrane engine, the soup has no
//    intramolecular exclusion, and this file must match the soup, not the engine),
//  - FENE once per covalent bond,
//  - bend once per unordered pair of a shared bonded middle's partners (bondedForce credits the
//    bend to the two OUTER particles of the triple, never to the middle),
//  - FENE once per adsorption tether pair (bond-adsorption.wgsl's centerLink, mutual, so one energy
//    term per linked pair).
// No numeric model constant is written here: every number comes from loadParams()/loadSoup().

import type { Params } from '../../engine/src/params'
import type { Soup } from './rules'
import { CLASS_POLAR, acidSoapScaleOf, attractionScaleTable, speciesClasses } from './soup-attraction'
import { esTotalEnergy, makeEsBasis, type EsBasis, type EsOverrides } from './electrostatics'
import { NONE_U32 } from './soup-types'
import type { Soup as SoupType } from './rules'

function wcaCut(b: number): number {
  return Math.pow(2, 1 / 6) * b
}
function wcaV(r: number, b: number, epsilon: number): number {
  if (r >= wcaCut(b)) return 0
  const s6 = Math.pow(b / r, 6)
  const q = 2 * s6 - 1
  return epsilon * q * q
}
function feneV(r: number, k: number, rInf: number): number {
  const x = r / rInf
  return -0.5 * k * rInf * rInf * Math.log(1 - x * x)
}
function bendV(r: number, k: number, r0: number): number {
  const d = r - r0
  return 0.5 * k * d * d
}
/** attr_v's SHAPE (onset rc, width wc, depth epsilon) is rank A and identical for every pair; the
 * per-class table only multiplies its magnitude -- see data/soup.json's solvent.attractionRule. */
function attrV(r: number, rc: number, wc: number, epsilon: number): number {
  if (r < rc) return -epsilon
  if (r > rc + wc) return 0
  const c = Math.cos((Math.PI * (r - rc)) / (2 * wc))
  return -epsilon * c * c
}

export interface PotentialBasis {
  epsilon: number
  feneK: number
  feneRInf: number
  bendK: number
  bendR0: number
  wc: number
  /** Fixed attraction-well onset wca_cut(P.b_tt) -- forces.wgsl's own convention (attr_v uses
   * b_tt, NOT the per-pair WCA radius), so the well's position is the same for every pair. */
  rcAttr: number
  /** Per-kind-index WCA radius (P.sigma * radiusSigma), arithmetic-mean mixed per pair exactly as
   * step.wgsl's pairB() does. */
  bRadius: Float64Array
  /** Per-kind-index species class (soup/src/soup-attraction.ts). */
  classes: Uint32Array
  /** Per-class-pair attraction depth multiplier. */
  attr: number[][]
  /** Task 'acid-soap-pairing' (2026-08-23): the charge-assisted head-head depth multiplier, ADDED to
   * the polar-polar cell for exactly those head pairs where one bead is protonated and the other is
   * not -- the CPU twin of soup/wgsl/pair.wgsl's acidSoapScale(). 0 without the section (and on any
   * caller that passes no charges at all), which keeps every pre-task energy bit-identical. */
  acidSoap: number
  /** Largest distance at which any pair can contribute -- the CPU cell list's own cutoff. */
  cutoff: number
  /** Task 'electrostatics' (2026-08-20): the screened-Coulomb basis, whose cutoff is BY CONSTRUCTION
   * the same rcAttr+wc this cell list already covers (soup/src/electrostatics.ts), so adding the term
   * changes no cell size and no walk. Zero coefficient on a system without electrostatics, which makes
   * the term identically zero and this file's output bit-identical to before the task. */
  es: EsBasis
}

export function makePotentialBasis(
  soup: Soup,
  p: Params,
  attractionOverride?: number,
  claySurfaceChemistry?: string,
  esOverrides?: EsOverrides,
  acidSoapOverride?: number,
): PotentialBasis {
  const bRadius = new Float64Array(soup.monomers.map((m) => p.sigma * m.radiusSigma))
  const rcAttr = wcaCut(p.sigma * p.beadSizes.tail_tail)
  let maxWca = 0
  for (let a = 0; a < bRadius.length; a++) {
    for (let b = 0; b < bRadius.length; b++) maxWca = Math.max(maxWca, wcaCut((bRadius[a] + bRadius[b]) / 2))
  }
  return {
    epsilon: p.epsilon,
    feneK: p.fene.k,
    feneRInf: p.fene.rInf,
    bendK: p.bend.k,
    bendR0: p.bend.r0,
    wc: p.attraction.wc,
    rcAttr,
    bRadius,
    classes: speciesClasses(soup),
    attr: attractionScaleTable(soup, attractionOverride, claySurfaceChemistry),
    acidSoap: acidSoapScaleOf(soup, acidSoapOverride, attractionOverride),
    cutoff: Math.max(maxWca, rcAttr + p.attraction.wc),
    // Task 'acid-soap-pairing' (2026-08-23): the pair depth travels into the electrostatic basis as
    // well, because the constant-pH sampler that basis also serves needs it (soup/src/acid-soap.ts).
    es: makeEsBasis(soup as unknown as SoupType, p, { ...esOverrides, acidSoapScale: acidSoapOverride }),
  }
}

export interface PotentialTerms {
  nonbonded: number
  fene: number
  bend: number
  tether: number
  total: number
}

/** Adjacency built from SoupSystem.bonds()'s flat (i,j) pair list. */
export function bondAdjacency(bonds: Uint32Array, n: number): number[][] {
  const adj: number[][] = Array.from({ length: n }, () => [])
  for (let k = 0; k < bonds.length; k += 2) {
    adj[bonds[k]].push(bonds[k + 1])
    adj[bonds[k + 1]].push(bonds[k])
  }
  return adj
}

/** Total potential energy of `positions` (flat x,y,z,kind per particle) in `box`, 3-axis periodic --
 * the CPU twin of the sum of soup/wgsl/step.wgsl's own force terms. `centerLink`, when given, adds
 * the adsorption tether's FENE energy (one term per mutually linked pair). */
export function soupPotential(
  positions: Float32Array,
  bonds: Uint32Array,
  box: [number, number, number],
  basis: PotentialBasis,
  centerLink?: Uint32Array,
  charges?: Float32Array,
): PotentialTerms {
  const n = positions.length / 4
  const cutoff = basis.cutoff
  const nx = Math.max(1, Math.floor(box[0] / cutoff))
  const ny = Math.max(1, Math.floor(box[1] / cutoff))
  const nz = Math.max(1, Math.floor(box[2] / cutoff))
  const wx = box[0] / nx
  const wy = box[1] / ny
  const wz = box[2] / nz
  const ncell = nx * ny * nz

  // Counting sort into cells (same shape as the GPU grid build, at CPU scale).
  const counts = new Int32Array(ncell + 1)
  const cellOf = new Int32Array(n)
  for (let i = 0; i < n; i++) {
    const x = positions[i * 4] - Math.floor(positions[i * 4] / box[0]) * box[0]
    const y = positions[i * 4 + 1] - Math.floor(positions[i * 4 + 1] / box[1]) * box[1]
    const z = positions[i * 4 + 2] - Math.floor(positions[i * 4 + 2] / box[2]) * box[2]
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = Math.min(nz - 1, Math.floor(z / wz))
    const c = cx + nx * (cy + ny * cz)
    cellOf[i] = c
    counts[c + 1]++
  }
  for (let c = 0; c < ncell; c++) counts[c + 1] += counts[c]
  const cursor = counts.slice(0, ncell)
  const order = new Int32Array(n)
  for (let i = 0; i < n; i++) order[cursor[cellOf[i]]++] = i

  // Unique neighbour cells per cell. Deduplicated on purpose: when an axis holds fewer than 3
  // cells (a box barely wider than the interaction range, which the area move can approach as it
  // shrinks Lx/Ly), the wrapped -1/0/+1 offsets alias onto the SAME cell and a naive walk would
  // count those pairs two or three times. Each unordered pair is then taken exactly once by the
  // j > i test below, which is only valid because this neighbourhood relation is symmetric.
  const neighbours: Int32Array[] = []
  for (let cz = 0; cz < nz; cz++) {
    for (let cy = 0; cy < ny; cy++) {
      for (let cx = 0; cx < nx; cx++) {
        const seen = new Set<number>()
        for (let dz = -1; dz <= 1; dz++) {
          const ncz = (((cz + dz) % nz) + nz) % nz
          for (let dy = -1; dy <= 1; dy++) {
            const ncy = (((cy + dy) % ny) + ny) % ny
            for (let dx = -1; dx <= 1; dx++) {
              const ncx = (((cx + dx) % nx) + nx) % nx
              seen.add(ncx + nx * (ncy + ny * ncz))
            }
          }
        }
        neighbours.push(Int32Array.from(seen))
      }
    }
  }

  const cutoff2 = cutoff * cutoff
  let nonbonded = 0
  for (let cSelf = 0; cSelf < ncell; cSelf++) {
    const nb = neighbours[cSelf]
    for (let a = counts[cSelf]; a < counts[cSelf + 1]; a++) {
      const i = order[a]
      const xi = positions[i * 4], yi = positions[i * 4 + 1], zi = positions[i * 4 + 2]
      const ki = positions[i * 4 + 3] | 0
      const bi = basis.bRadius[ki]
      const attrRow = basis.attr[basis.classes[ki]]
      for (let q = 0; q < nb.length; q++) {
        const c = nb[q]
        for (let bq = counts[c]; bq < counts[c + 1]; bq++) {
          const j = order[bq]
          if (j <= i) continue
          let ddx = xi - positions[j * 4]
          let ddy = yi - positions[j * 4 + 1]
          let ddz = zi - positions[j * 4 + 2]
          ddx -= Math.round(ddx / box[0]) * box[0]
          ddy -= Math.round(ddy / box[1]) * box[1]
          ddz -= Math.round(ddz / box[2]) * box[2]
          const r2 = ddx * ddx + ddy * ddy + ddz * ddz
          if (r2 > cutoff2 || r2 <= 0) continue
          const r = Math.sqrt(r2)
          const kj = positions[j * 4 + 3] | 0
          nonbonded += wcaV(r, (bi + basis.bRadius[kj]) / 2, basis.epsilon)
          let sc = attrRow[basis.classes[kj]]
          // Task 'acid-soap-pairing' (2026-08-23): the charge-assisted head-head depth, on the SAME
          // attr_dv/attrV ramp, for exactly one protonated head with one deprotonated head -- the
          // transcription of soup/wgsl/pair.wgsl's acidSoapScale(), including its XOR. Without
          // charges (every pre-task caller) or without the section this branch is never taken.
          if (
            basis.acidSoap > 0 &&
            charges !== undefined &&
            basis.classes[ki] === CLASS_POLAR &&
            basis.classes[kj] === CLASS_POLAR &&
            (charges[i] === 0) !== (charges[j] === 0)
          ) {
            sc += basis.acidSoap
          }
          if (sc > 0) nonbonded += sc * attrV(r, basis.rcAttr, basis.wc, basis.epsilon)
          // Task 'long-range-electrostatics' (2026-08-20): the screened-Coulomb term is NO LONGER
          // summed here. This loop's cell list is built at the Lennard-Jones cutoff
          // (basis.cutoff = 2.7224620 sigma) and the electrostatic cutoff is now a multiple of the
          // Debye length, i.e. up to 15.2 sigma -- so this walk cannot see the whole range and
          // adding it here would silently truncate it back to the old, wrong one. It is summed
          // instead by esTotalEnergy below, over its OWN head-only cell list at its own cutoff,
          // exactly mirroring the GPU's dedicated long-range pass.
        }
      }
    }
  }

  const dist = (i: number, j: number): number => {
    let dx = positions[i * 4] - positions[j * 4]
    let dy = positions[i * 4 + 1] - positions[j * 4 + 1]
    let dz = positions[i * 4 + 2] - positions[j * 4 + 2]
    dx -= Math.round(dx / box[0]) * box[0]
    dy -= Math.round(dy / box[1]) * box[1]
    dz -= Math.round(dz / box[2]) * box[2]
    return Math.sqrt(dx * dx + dy * dy + dz * dz)
  }

  let fene = 0
  for (let k = 0; k < bonds.length; k += 2) {
    fene += feneV(dist(bonds[k], bonds[k + 1]), basis.feneK, basis.feneRInf)
  }

  const adj = bondAdjacency(bonds, n)
  let bend = 0
  for (let m = 0; m < n; m++) {
    const partners = adj[m]
    for (let a = 0; a < partners.length; a++) {
      for (let b = a + 1; b < partners.length; b++) {
        bend += bendV(dist(partners[a], partners[b]), basis.bendK, basis.bendR0)
      }
    }
  }

  let tether = 0
  if (centerLink) {
    for (let i = 0; i < n; i++) {
      const owner = centerLink[i]
      if (owner === NONE_U32 || owner <= i) continue
      tether += feneV(dist(i, owner), basis.feneK, basis.feneRInf)
    }
  }

  // Task 'long-range-electrostatics' (2026-08-20): the whole screened-Coulomb energy, over its own
  // (long) cutoff and its own head-only cell list -- see esTotalEnergy's header. Identically 0
  // without charge, so every pre-task potential is bit-identical.
  nonbonded += esTotalEnergy(positions, charges, box, basis.es)

  return { nonbonded, fene, bend, tether, total: nonbonded + fene + bend + tether }
}
