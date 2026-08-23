// The nonbonded-attraction DEPTH table, in ONE place, read by both sides that need it:
// soup/src/soup-buffers.ts uploads it as the AttrScale uniform soup/wgsl/step.wgsl's
// pairAttrScale() reads, and soup/src/soup-potential.ts evaluates the same numbers on the CPU for
// the area move's Metropolis energy. Two copies of a pairwise table drifting apart is exactly what
// data/soup.json's own solvent.attractionRule warns about, so neither side owns the numbers --
// data/soup.json does, and this module is the only thing that turns them into a per-class array.
//
// Task 'hydrophobic-asymmetry' (2026-08-19): before this task the attraction was a BOOLEAN
// (soup/wgsl/step.wgsl's shouldAttract) times one global scale, with the apolar-apolar (tail-tail)
// pair structurally excluded. It is now a per-CLASS depth multiplier, and apolar-apolar is back --
// see data/soup.json's solvent.attractionScale.basis for the reversal, the MARTINI ratios and the
// ranks. No numeric model constant is written here: every number comes from loadSoup().
//
// Classes are DERIVED from the per-species flags data/soup.json already carries, not declared per
// monomer id: mineral=true -> MINERAL, else solvent=true -> SOLVENT, else polar=true -> POLAR, else
// APOLAR. That is the same derivation soup/wgsl/step.wgsl's speciesClass() does on the GPU side,
// kept textually parallel.

import type { ClaySurfaceChemistry, Monomer, Soup } from './rules'

export const CLASS_APOLAR = 0
export const CLASS_POLAR = 1
export const CLASS_SOLVENT = 2
// Task 'clay-surface' (2026-08-19): the mineral phase gets its OWN class rather than borrowing the
// head's `polar` or the water's `solvent` row -- a clay basal surface is not a carboxyl head and not
// a water bead, and the whole point of the platelet is that its depths against water/head/tail
// differ from both. Appended as the LAST class so every pre-existing class index (and therefore
// every pre-existing pairEpsilon key) is unchanged.
export const CLASS_MINERAL = 3
export const CLASS_COUNT = 4

const CLASS_NAMES = ['apolar', 'polar', 'solvent', 'mineral'] as const

/** APOLAR/POLAR/SOLVENT/MINERAL for one monomer -- the CPU twin of step.wgsl's speciesClass().
 * `mineral` is tested FIRST: it is the most specific flag, and data/soup.json's own validation
 * (soup/src/rules-validate.ts) refuses a monomer that claims mineral together with polar/solvent, so
 * the order can never silently reclassify an existing species. */
export function speciesClassOf(m: Monomer): number {
  if (m.mineral) return CLASS_MINERAL
  if (m.solvent) return CLASS_SOLVENT
  if (m.polar) return CLASS_POLAR
  return CLASS_APOLAR
}

/** Per-kind-index class lookup, in data/soup.json's own monomer order (the same index particle
 * positions carry in their .w component). */
export function speciesClasses(soup: Soup): Uint32Array {
  return new Uint32Array(soup.monomers.map(speciesClassOf))
}

/** data/soup.json key for the unordered class pair: higher class first, lower one capitalised
 * ("solventApolar", "polarApolar", "apolarApolar", ...) -- the naming data/soup.json's
 * solvent.attractionScale.pairEpsilon.levels uses. */
function pairKey(ci: number, cj: number): string {
  const hi = CLASS_NAMES[Math.max(ci, cj)]
  const lo = CLASS_NAMES[Math.min(ci, cj)]
  return hi + lo[0].toUpperCase() + lo.slice(1)
}

/** The SURFACE CHEMISTRY in force for this system: data/soup.json's clay.surfaceChemistry, or the
 * caller's override (CreateSoupOpts.claySurfaceChemistry). Task 'clay-surface-chemistry' (2026-08-19):
 * the platelet's hydrophilicity is a MAPPING CHOICE, not a measured property of this tree, and the
 * published CG clay mapping disagrees with the hydrophilic one -- so the mapping is selectable and
 * both limits stay measurable. Throws by name rather than falling back to a default, because a
 * silently-wrong surface chemistry is exactly the kind of thing a measurement would then attribute to
 * physics. */
export function claySurfaceChemistryOf(soup: Soup, override?: string): ClaySurfaceChemistry & { name: string } {
  const c = soup.clay
  if (!c) throw new Error('claySurfaceChemistryOf: data/soup.json не содержит секции clay')
  const name = override ?? c.surfaceChemistry
  const chem = c.surfaceChemistries?.[name]
  if (chem === undefined) {
    throw new Error(
      `data/soup.json: химия поверхности глины "${name}" не объявлена в clay.surfaceChemistries ` +
        `(есть: ${Object.keys(c.surfaceChemistries ?? {}).join(', ') || 'ни одной'})`,
    )
  }
  return { name, ...chem }
}

/** Resolves one class-pair key through the selected surface chemistry: a MINERAL pair is looked up by
 * the level name that chemistry aliases it to, every other pair by its own name. This is the only
 * place the indirection exists, so the CPU potential and the GPU uniform cannot disagree about which
 * surface the run has. */
function levelKeyFor(soup: Soup, ci: number, cj: number, chemistry?: string): string {
  const key = pairKey(ci, cj)
  if (ci !== CLASS_MINERAL && cj !== CLASS_MINERAL) return key
  if (!soup.clay) return key
  const alias = claySurfaceChemistryOf(soup, chemistry).pairs[key]
  if (alias === undefined) {
    throw new Error(
      `data/soup.json: clay.surfaceChemistries["${chemistry ?? soup.clay.surfaceChemistry}"].pairs не задаёт пару "${key}"`,
    )
  }
  return alias
}

/** The symmetric CLASS_COUNT x CLASS_COUNT depth-multiplier table on attr_dv's magnitude:
 * levels[pair].epsilonKJ / levels[reference].epsilonKJ, times the global epsilonScale. The
 * reference pair therefore comes out at exactly `epsilonScale` (1.0 in the file), which is what
 * ties the ABSOLUTE scale to data/params.json's rank-A Cooke & Deserno epsilon instead of importing
 * MARTINI's kJ/mol into this model's energy unit -- only the RATIOS are borrowed. `override`
 * replaces the file's epsilonScale for one system (CreateSoupOpts.solventAttractionScaleOverride).
 *
 * A file with no attractionScale at all (an older fixture) reads as the pre-task behaviour it
 * described: water-water/water-head/water-tail at depth 1, everything else 0 -- including every
 * MINERAL cell, which is correct for such a fixture: a file too old to carry a pair table is also
 * too old to declare a mineral monomer, so those cells are never read. */
export function attractionScaleTable(soup: Soup, override?: number, chemistry?: string): number[][] {
  const sc = soup.solvent.attractionScale
  const table: number[][] = []
  for (let ci = 0; ci < CLASS_COUNT; ci++) table.push(new Array(CLASS_COUNT).fill(0))
  const global = override ?? sc?.epsilonScale ?? 1
  if (!Number.isFinite(global) || global <= 0) {
    throw new Error(`soup attraction: epsilonScale=${global} должен быть конечным положительным числом`)
  }
  if (sc?.pairEpsilon === undefined) {
    // Pre-'hydrophobic-asymmetry' shape: the boolean rule (solvent with solvent, or solvent with
    // polar) at one uniform depth. Kept so a fixture written against the old schema still loads.
    for (let ci = 0; ci < CLASS_COUNT; ci++) {
      for (let cj = 0; cj < CLASS_COUNT; cj++) {
        const anySolvent = ci === CLASS_SOLVENT || cj === CLASS_SOLVENT
        const bothSolventOrPolar = ci !== CLASS_APOLAR && cj !== CLASS_APOLAR
        table[ci][cj] = anySolvent && bothSolventOrPolar ? global : 0
      }
    }
    return table
  }
  const pe = sc.pairEpsilon
  const ref = pe.levels[pe.reference]
  if (ref === undefined || !(Number.isFinite(ref.epsilonKJ) && ref.epsilonKJ > 0)) {
    throw new Error(
      `data/soup.json: solvent.attractionScale.pairEpsilon.reference="${pe.reference}" не указывает на уровень с положительным epsilonKJ`,
    )
  }
  for (let ci = 0; ci < CLASS_COUNT; ci++) {
    for (let cj = 0; cj < CLASS_COUNT; cj++) {
      const key = levelKeyFor(soup, ci, cj, chemistry)
      const lvl = pe.levels[key]
      if (lvl === undefined) {
        throw new Error(`data/soup.json: solvent.attractionScale.pairEpsilon.levels не содержит пары "${key}"`)
      }
      table[ci][cj] = (global * lvl.epsilonKJ) / ref.epsilonKJ
    }
  }
  return table
}

/** Task 'acid-soap-pairing' (2026-08-23): the CHARGE-ASSISTED head-head depth multiplier, in the
 * SAME units every cell of the class table is in -- levels.acidSoapPair.epsilonKJ divided by the
 * reference level, times the global epsilonScale. Zero (and therefore structurally absent) when
 * data/soup.json declares no acidSoapPair section or declares it at 0, which is what keeps every
 * pre-task run bit-identical.
 *
 * WHY IT IS NOT A CELL OF THE CLASS TABLE. Every cell of `pairEpsilon` is a function of the two
 * SPECIES alone; this one is a function of the two beads' PROTONATION STATES, which are dynamical
 * variables the constant-pH Monte Carlo owns (soup/src/electrostatics.ts). A polarPolar cell would
 * attract acid-acid and soap-soap as well, and the atomic measurement this number comes from says
 * those do NOT associate in water (the neutral acid dimer measures UNFAVOURABLE once 32 waters
 * compete for the hydrogen bonds). So the rule has to read charge, and the depth travels as its own
 * scalar -- one extra vec4 on the uniform, not a fifth row. */
export function acidSoapScaleOf(soup: Soup, override?: number, epsilonScaleOverride?: number): number {
  const sc = soup.solvent.attractionScale
  const pe = sc?.pairEpsilon
  if (override !== undefined) {
    if (!Number.isFinite(override) || override < 0) {
      throw new Error(`soup attraction: acidSoapScaleOverride=${override} должен быть конечным неотрицательным числом`)
    }
    return override
  }
  const as = sc?.acidSoapPair
  if (as === undefined || pe === undefined) return 0
  if (!Number.isFinite(as.epsilonKJ) || as.epsilonKJ < 0) {
    throw new Error(`data/soup.json: solvent.attractionScale.acidSoapPair.epsilonKJ=${as.epsilonKJ} должен быть конечным неотрицательным`)
  }
  const ref = pe.levels[pe.reference]
  if (ref === undefined || !(ref.epsilonKJ > 0)) {
    throw new Error(`data/soup.json: acidSoapPair нельзя нормировать -- reference="${pe.reference}" не даёт положительного epsilonKJ`)
  }
  const global = epsilonScaleOverride ?? sc?.epsilonScale ?? 1
  return (global * as.epsilonKJ) / ref.epsilonKJ
}

/** The same table flattened for the GPU uniform soup/wgsl/pair.wgsl declares as
 * `rows: array<vec4<f32>, CLASS_COUNT>` (16-byte row stride, one row per class) followed by ONE
 * more vec4 `pair`, whose .x is the charge-assisted head-head depth (task 'acid-soap-pairing',
 * 2026-08-23) and whose other three components are spare. With four classes the ROWS are exactly
 * full -- there is no spare component left in them, so a FIFTH class would need the uniform's own
 * shape changed on both sides, not just one more entry here. */
export function attractionScaleUniform(
  soup: Soup,
  override?: number,
  chemistry?: string,
  acidSoapOverride?: number,
): Float32Array<ArrayBuffer> {
  const table = attractionScaleTable(soup, override, chemistry)
  const out = new Float32Array((CLASS_COUNT + 1) * 4)
  for (let ci = 0; ci < CLASS_COUNT; ci++) {
    for (let cj = 0; cj < CLASS_COUNT; cj++) out[ci * 4 + cj] = table[ci][cj]
  }
  out[CLASS_COUNT * 4] = acidSoapScaleOf(soup, acidSoapOverride, override)
  return out
}

export const ATTR_SCALE_UNIFORM_BYTES = (CLASS_COUNT + 1) * 4 * 4
