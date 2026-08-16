import atomsRaw from '../../data/atoms.json'
import moleculesRaw from '../../data/molecules.json'
import {
  atAngleXY,
  rotateAroundAxis,
  tetrahedralPair,
  tetrahedralTriple,
  trigonalPair,
  vAdd,
  vNormalize,
  vScale,
  vSub,
  zigzagChain,
  type Vec3,
} from './geometry'

export interface AtomRef {
  element: string
  position: Vec3
}

export interface Species {
  id: string
  formula: string
  charge: number
  atoms: AtomRef[]
  bonds: [number, number][]
  source: string
}

interface ElementInfo {
  mass: number
  vdw: number
  covalent: number
  color: string
}

interface AtomsData {
  source: string
  rank: string
  elements: Record<string, ElementInfo>
}

interface SpeciesEntry {
  id: string
  formula: string
  charge: number
  kind: 'single' | 'diatomic' | 'bent' | 'linear3' | 'formate'
  bond?: string
  angle?: string
}

interface MoleculesData {
  rank: string
  bondLengths: Record<string, number>
  angles: Record<string, number>
  sources: Record<string, string>
  species: SpeciesEntry[]
}

function loadAtoms(): AtomsData {
  return atomsRaw as unknown as AtomsData
}

function loadMolecules(): MoleculesData {
  return moleculesRaw as unknown as MoleculesData
}

function requireBond(mol: MoleculesData, key: string): number {
  const v = mol.bondLengths[key]
  if (v === undefined) throw new Error(`data/molecules.json: нет длины связи "${key}"`)
  return v
}

function requireAngle(mol: MoleculesData, key: string): number {
  const v = mol.angles[key]
  if (v === undefined) throw new Error(`data/molecules.json: нет угла "${key}"`)
  return v
}

function sourcesOf(mol: MoleculesData, keys: string[]): string {
  const parts = keys.map((k) => mol.sources[k]).filter((s): s is string => Boolean(s))
  return Array.from(new Set(parts)).join('; ')
}

/** Parse a chemical formula ("C10H20O2") into element -> count. */
export function parseFormula(formula: string): Record<string, number> {
  const counts: Record<string, number> = {}
  const re = /([A-Z][a-z]?)(\d*)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(formula))) {
    const [, element, numStr] = m
    if (!element) continue
    counts[element] = (counts[element] ?? 0) + (numStr ? parseInt(numStr, 10) : 1)
  }
  return counts
}

/** Expand a formula into a flat, left-to-right ordered list of element symbols. */
function flattenFormula(formula: string): string[] {
  const out: string[] = []
  const re = /([A-Z][a-z]?)(\d*)/g
  let m: RegExpExecArray | null
  while ((m = re.exec(formula))) {
    const n = m[2] ? parseInt(m[2], 10) : 1
    for (let i = 0; i < n; i++) out.push(m[1])
  }
  return out
}

/** Split an AB2 formula (e.g. H2O, CO2) into its central atom and its two identical terminal atoms. */
function splitAB2(formula: string): { central: string; terminal: string } {
  const counts = parseFormula(formula)
  const entries = Object.entries(counts)
  const central = entries.find(([, c]) => c === 1)?.[0]
  const terminal = entries.find(([, c]) => c === 2)?.[0]
  if (!central || !terminal || entries.length !== 2) {
    throw new Error(`species: формула "${formula}" не подходит под AB2-геометрию`)
  }
  return { central, terminal }
}

/** Assemble the canonical (Hill-order) formula string from an actual atom count. */
function hillFormula(counts: Record<string, number>): string {
  const keys = Object.keys(counts).sort((a, b) => {
    if (a === 'C') return -1
    if (b === 'C') return 1
    if (a === 'H') return -1
    if (b === 'H') return 1
    return a.localeCompare(b)
  })
  return keys.map((el) => `${el}${counts[el] > 1 ? counts[el] : ''}`).join('')
}

/** Molar mass (g/mol) of a formula, from data/atoms.json element masses. */
export function molarMass(formula: string): number {
  const atoms = loadAtoms()
  const counts = parseFormula(formula)
  let mass = 0
  for (const [element, n] of Object.entries(counts)) {
    const info = atoms.elements[element]
    if (!info) throw new Error(`data/atoms.json: неизвестный элемент "${element}"`)
    mass += info.mass * n
  }
  return mass
}

function buildSpeciesEntry(entry: SpeciesEntry, mol: MoleculesData): Species {
  let atoms: AtomRef[] = []
  let bonds: [number, number][] = []
  let source = ''

  switch (entry.kind) {
    case 'single': {
      const els = flattenFormula(entry.formula)
      atoms = [{ element: els[0], position: [0, 0, 0] }]
      source = 'одноатомный ион — геометрия не применима'
      break
    }
    case 'diatomic': {
      const bondLen = requireBond(mol, entry.bond!)
      const els = flattenFormula(entry.formula)
      if (els.length !== 2) throw new Error(`species: диатомная формула должна давать 2 атома: "${entry.formula}"`)
      atoms = [
        { element: els[0], position: [0, 0, 0] },
        { element: els[1], position: [bondLen, 0, 0] },
      ]
      bonds = [[0, 1]]
      source = sourcesOf(mol, [entry.bond!])
      break
    }
    case 'bent': {
      const { central, terminal } = splitAB2(entry.formula)
      const bondLen = requireBond(mol, entry.bond!)
      const angleDeg = requireAngle(mol, entry.angle!)
      atoms = [
        { element: central, position: [0, 0, 0] },
        { element: terminal, position: atAngleXY(bondLen, angleDeg / 2) },
        { element: terminal, position: atAngleXY(bondLen, -angleDeg / 2) },
      ]
      bonds = [[0, 1], [0, 2]]
      source = sourcesOf(mol, [entry.bond!, entry.angle!])
      break
    }
    case 'linear3': {
      const { central, terminal } = splitAB2(entry.formula)
      const bondLen = requireBond(mol, entry.bond!)
      atoms = [
        { element: central, position: [0, 0, 0] },
        { element: terminal, position: [bondLen, 0, 0] },
        { element: terminal, position: [-bondLen, 0, 0] },
      ]
      bonds = [[0, 1], [0, 2]]
      source = sourcesOf(mol, [entry.bond!])
      break
    }
    case 'formate': {
      const cOBond = requireBond(mol, 'C_O_formate')
      const chBond = requireBond(mol, 'CH')
      const angleDeg = requireAngle(mol, 'formate_OCO')
      const normal: Vec3 = [0, 0, 1]
      const dO1: Vec3 = [1, 0, 0]
      const dOa = rotateAroundAxis(dO1, normal, angleDeg / 2)
      const dOb = rotateAroundAxis(dO1, normal, -angleDeg / 2)
      // The C-H bond points along the reversed bisector of the two C-O bonds
      // (planar sp2 carbon; the brief fixes only its length, not a separate angle).
      const dH: Vec3 = [-1, 0, 0]
      atoms = [
        { element: 'C', position: [0, 0, 0] },
        { element: 'O', position: vScale(dOa, cOBond) },
        { element: 'O', position: vScale(dOb, cOBond) },
        { element: 'H', position: vScale(dH, chBond) },
      ]
      bonds = [[0, 1], [0, 2], [0, 3]]
      source = sourcesOf(mol, ['C_O_formate', 'formate_OCO', 'CH'])
      break
    }
    default:
      throw new Error(`species: неизвестный вид геометрии "${entry.kind}"`)
  }

  return { id: entry.id, formula: entry.formula, charge: entry.charge, atoms, bonds, source }
}

/** All fixed-composition species declared in data/molecules.json, built into 3D geometry. */
export function loadSpecies(): Record<string, Species> {
  const mol = loadMolecules()
  const out: Record<string, Species> = {}
  for (const entry of mol.species) out[entry.id] = buildSpeciesEntry(entry, mol)
  return out
}

/**
 * Build an n-carbon saturated alkanoic (fatty) acid: an all-anti zig-zag
 * carbon backbone with a carboxyl group on carbon 0 and enough hydrogens on
 * every other carbon to fill its four bonds. The formula is composed from the
 * atoms actually placed, so it cannot silently disagree with the geometry.
 */
export function buildAlkanoicAcid(n: number): Species {
  if (n < 1) throw new Error('buildAlkanoicAcid: n должно быть >= 1')
  const mol = loadMolecules()

  const ccBond = requireBond(mol, 'CC_single')
  const chBond = requireBond(mol, 'CH')
  const tetra = requireAngle(mol, 'tetrahedral')
  const trigonal = requireAngle(mol, 'trigonal')
  const carbonylBond = requireBond(mol, 'C_O_carboxyl_double')
  const hydroxylCOBond = requireBond(mol, 'C_O_carboxyl_single')
  const ohBond = requireBond(mol, 'OH_carboxyl')

  const chain = zigzagChain(n, ccBond, tetra)
  const atoms: AtomRef[] = chain.map((p) => ({ element: 'C', position: p }))
  const bonds: [number, number][] = []
  for (let i = 1; i < n; i++) bonds.push([i - 1, i])

  const addAtom = (element: string, position: Vec3, bondFrom: number) => {
    atoms.push({ element, position })
    bonds.push([bondFrom, atoms.length - 1])
  }

  const normal: Vec3 = [0, 0, 1] // the zig-zag backbone's plane normal

  // Carboxyl group on carbon 0: sp2, three substituents at `trigonal` apart.
  // With a chain neighbour (n > 1) that neighbour is the third substituent and
  // fixes the other two directions; with no chain (n === 1, formic acid) there
  // is no anchor, so an arbitrary in-plane axis stands in for it.
  const anchor: Vec3 = n > 1 ? vNormalize(vSub(chain[1], chain[0])) : [1, 0, 0]
  const [carbonylDir, hydroxylDir] = trigonalPair(anchor, normal, trigonal)
  addAtom('O', vAdd(chain[0], vScale(carbonylDir, carbonylBond)), 0)
  const hydroxylOIndex = atoms.length
  addAtom('O', vAdd(chain[0], vScale(hydroxylDir, hydroxylCOBond)), 0)
  addAtom('H', vAdd(atoms[hydroxylOIndex].position, vScale(hydroxylDir, ohBond)), hydroxylOIndex)
  if (n === 1) {
    // Formic acid: with no chain neighbour, `anchor` played the role a chain
    // bond would have (fixing the other two directions via trigonalPair
    // above); its own direction is then free for the carbon's third
    // substituent, a hydrogen.
    addAtom('H', vScale(anchor, chBond), 0)
  }

  // Remaining carbons: sp3, fill to four bonds with hydrogens.
  for (let i = 1; i < n; i++) {
    const dPrev = vNormalize(vSub(chain[i - 1], chain[i]))
    if (i < n - 1) {
      const dNext = vNormalize(vSub(chain[i + 1], chain[i]))
      const [h1, h2] = tetrahedralPair(dPrev, dNext, tetra)
      addAtom('H', vAdd(chain[i], vScale(h1, chBond)), i)
      addAtom('H', vAdd(chain[i], vScale(h2, chBond)), i)
    } else {
      const [h1, h2, h3] = tetrahedralTriple(dPrev, tetra)
      addAtom('H', vAdd(chain[i], vScale(h1, chBond)), i)
      addAtom('H', vAdd(chain[i], vScale(h2, chBond)), i)
      addAtom('H', vAdd(chain[i], vScale(h3, chBond)), i)
    }
  }

  const counts: Record<string, number> = {}
  for (const a of atoms) counts[a.element] = (counts[a.element] ?? 0) + 1
  const formula = hillFormula(counts)

  const source = sourcesOf(mol, [
    'CC_single', 'CH', 'tetrahedral', 'trigonal',
    'C_O_carboxyl_double', 'C_O_carboxyl_single', 'OH_carboxyl',
  ])

  return { id: `C${n}_acid`, formula, charge: 0, atoms, bonds, source }
}
