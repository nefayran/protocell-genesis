// Schema VALIDATION for data/soup.json -- every "this file is internally consistent" invariant, in
// one place. Split out of soup/src/rules.ts (file-size rule in CLAUDE.md: rules.ts stood at 598
// lines, one line under the hard 600 limit, and task 'clay-surface' needed to add a section to it)
// as a straight, verbatim move of assertRulesConsistent: not one message, threshold or check
// changed by the split itself. The split is BY RESPONSIBILITY, not by line count -- rules.ts now
// owns the schema's TYPES plus loading and the detailed-balance arithmetic (what the rest of the
// engine reads), while this module owns the throwing checks nothing else calls into.
//
// `import type` only, in the one direction that matters: this module imports the types from
// rules.ts and rules.ts re-exports this module's function, so there is no runtime import cycle
// (the type import is erased entirely at compile time).

import type { Soup, Monomer } from './rules'

const ALLOWED_MONOMER_KINDS = new Set<Monomer['kind']>(['carbon', 'head', 'donor', 'catalyst', 'water', 'clay'])

/**
 * Checks three invariants of the rule set and one invariant of the starting composition.
 * Throws an Error whose message names the offender, at the first violation:
 *  - a rule has a rank other than 'D' (this model has no measured rate constants, so
 *    claiming a higher rank would be dishonest);
 *  - a rule has an empty or trivial basis (the rate justification must be substantive);
 *  - a formation rule ('bond') has no paired break ('break') with the same a/b,
 *    or the pair's energies differ; without that, detailed balance is undefined;
 *  - a monomer's kind is not one of the four elementary ones ('carbon'|'head'|'donor'|'catalyst'),
 *    or start refers to an id not declared in monomers; this is the input-side check for
 *    "monomers only, no ready-made amphiphile".
 */
export function assertRulesConsistent(s: Soup): void {
  for (const r of s.rules) {
    if (r.rank !== 'D') {
      throw new Error(
        `data/soup.json: rule ${r.id} has rank ${r.rank}, but synthesis rates without measured constants allow only rank D`,
      )
    }
    if (!r.basis || r.basis.trim().length <= 10) {
      throw new Error(`data/soup.json: rule ${r.id} has no substantive justification (basis)`)
    }
  }

  for (const r of s.rules.filter((x) => x.kind === 'bond')) {
    const back = s.rules.find((x) => x.kind === 'break' && x.a === r.a && x.b === r.b)
    if (!back) {
      throw new Error(
        `data/soup.json: formation rule ${r.id} (${r.a}-${r.b}) has no paired break rule, so detailed balance is undefined`,
      )
    }
    if (back.energyKT !== r.energyKT) {
      throw new Error(
        `data/soup.json: rules ${r.id} and ${back.id} disagree on energy (${r.energyKT} vs ${back.energyKT}): the paired break must have the same energy, otherwise detailed balance is violated`,
      )
    }
  }

  for (const m of s.monomers) {
    if (!ALLOWED_MONOMER_KINDS.has(m.kind)) {
      throw new Error(
        `data/soup.json: monomer ${m.id} has kind "${m.kind}", which is not in the set of elementary building blocks`,
      )
    }
  }

  const sv = s.solvent
  const waterMonomer = s.monomers.find((m) => m.id === sv.waterId)
  if (!waterMonomer) {
    throw new Error(`data/soup.json: solvent.waterId="${sv.waterId}" not found among monomers`)
  }
  if (waterMonomer.kind !== 'water' || !waterMonomer.solvent) {
    throw new Error(
      `data/soup.json: monomer "${sv.waterId}" is named solvent.waterId but does not have kind="water" and solvent=true`,
    )
  }
  if (!sv.attractionRule || sv.attractionRule.trim().length <= 10) {
    throw new Error('data/soup.json: solvent.attractionRule has no substantive description')
  }
  if (!sv.basis || sv.basis.trim().length <= 10) {
    throw new Error('data/soup.json: solvent has no substantive justification (basis)')
  }
  if (sv.attractionScale !== undefined) {
    const sc = sv.attractionScale
    if (!(Number.isFinite(sc.epsilonScale) && sc.epsilonScale > 0)) {
      throw new Error(`data/soup.json: solvent.attractionScale.epsilonScale=${sc.epsilonScale} must be a finite positive number`)
    }
    if (!sc.basis || sc.basis.trim().length <= 10) {
      throw new Error('data/soup.json: solvent.attractionScale has no substantive justification (basis)')
    }
    if (sc.pairEpsilon !== undefined) {
      const pe = sc.pairEpsilon
      const ref = pe.levels?.[pe.reference]
      if (!ref || !(Number.isFinite(ref.epsilonKJ) && ref.epsilonKJ > 0)) {
        throw new Error(
          `data/soup.json: solvent.attractionScale.pairEpsilon.reference="${pe.reference}" does not point to a level with positive epsilonKJ`,
        )
      }
      for (const [key, lvl] of Object.entries(pe.levels)) {
        if (!Number.isFinite(lvl.epsilonKJ) || lvl.epsilonKJ < 0) {
          throw new Error(`data/soup.json: solvent.attractionScale.pairEpsilon.levels.${key}.epsilonKJ=${lvl.epsilonKJ} must be a finite non-negative number`)
        }
      }
    }
  }

  if (s.areaMove !== undefined) {
    const am = s.areaMove
    if (!(Number.isFinite(am.logDelta) && am.logDelta > 0)) {
      throw new Error(`data/soup.json: areaMove.logDelta=${am.logDelta} must be a finite positive number`)
    }
    if (am.mode !== 'lateral-fixed-volume' && am.mode !== 'lateral-fixed-z') {
      throw new Error(`data/soup.json: areaMove.mode="${am.mode}" is not in the set {lateral-fixed-volume, lateral-fixed-z}`)
    }
    if (!am.basis || am.basis.trim().length <= 10) {
      throw new Error('data/soup.json: areaMove has no substantive justification (basis)')
    }
  }

  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) {
    if (!ids.has(k)) {
      throw new Error(
        `data/soup.json: the starting composition refers to "${k}", which is not declared as a monomer; it may be a ready-made amphiphile rather than a building block`,
      )
    }
  }

  const bai = s.bondAttemptInterval
  if (!Number.isInteger(bai.steps) || bai.steps < 1) {
    throw new Error(`data/soup.json: bondAttemptInterval.steps=${bai.steps} must be an integer >= 1`)
  }
  if (!bai.basis || bai.basis.trim().length <= 10) {
    throw new Error('data/soup.json: bondAttemptInterval has no substantive justification (basis)')
  }

  const ng = s.neighborGrid
  if (!Number.isInteger(ng.cellDivisor) || ng.cellDivisor < 1) {
    throw new Error(`data/soup.json: neighborGrid.cellDivisor=${ng.cellDivisor} must be an integer >= 1`)
  }
  if (typeof ng.sortedGather !== 'boolean') {
    throw new Error('data/soup.json: neighborGrid.sortedGather must be a boolean')
  }
  if (!ng.basis || ng.basis.trim().length <= 10) {
    throw new Error('data/soup.json: neighborGrid has no substantive justification (basis)')
  }

  const vl = s.verletList
  if (typeof vl.enabled !== 'boolean') {
    throw new Error('data/soup.json: verletList.enabled must be a boolean')
  }
  if (!(vl.skin > 0)) {
    throw new Error(`data/soup.json: verletList.skin=${vl.skin} must be a positive number`)
  }
  if (!Number.isInteger(vl.rebuildEvery) || vl.rebuildEvery < 1) {
    throw new Error(`data/soup.json: verletList.rebuildEvery=${vl.rebuildEvery} must be an integer >= 1`)
  }
  if (!Number.isInteger(vl.listCapacity) || vl.listCapacity < 1) {
    throw new Error(`data/soup.json: verletList.listCapacity=${vl.listCapacity} must be an integer >= 1`)
  }
  if (!(vl.capacitySafetyFactor >= 1)) {
    throw new Error(
      `data/soup.json: verletList.capacitySafetyFactor=${vl.capacitySafetyFactor} must be a number >= 1 ` +
        `(a multiplier on the uniform estimate of the neighbour count, see soup/src/soup-plan.ts's deriveListCapacity)`,
    )
  }
  if (!Number.isInteger(vl.capacityFloor) || vl.capacityFloor < 1) {
    throw new Error(`data/soup.json: verletList.capacityFloor=${vl.capacityFloor} must be an integer >= 1`)
  }
  if (!vl.basis || vl.basis.trim().length <= 10) {
    throw new Error('data/soup.json: verletList has no substantive justification (basis)')
  }

  const hp = s.headPlacement
  if (typeof hp.terminalOnly !== 'boolean') {
    throw new Error('data/soup.json: headPlacement.terminalOnly must be a boolean')
  }
  // Architectural ceiling, not a physics one: every particle (any kind) owns exactly 3 bondSlots
  // rows (soup/src/sim.ts's bondSlots0 -- N*3, uniform across kinds), so a head literally cannot
  // claim a 4th chain slot regardless of what data/soup.json asks for.
  const MAX_ARCHITECTURAL_SLOTS = 3
  if (!Number.isInteger(hp.chainCapacity) || hp.chainCapacity < 1 || hp.chainCapacity > MAX_ARCHITECTURAL_SLOTS) {
    throw new Error(
      `data/soup.json: headPlacement.chainCapacity=${hp.chainCapacity} must be an integer from 1 to ${MAX_ARCHITECTURAL_SLOTS} (soup/src/sim.ts's per-particle bondSlots row)`,
    )
  }
  if (!hp.basis || hp.basis.trim().length <= 10) {
    throw new Error('data/soup.json: headPlacement has no substantive justification (basis)')
  }

  const ad = s.adsorption
  if (!Number.isInteger(ad.occupancy) || ad.occupancy < 1) {
    throw new Error(`data/soup.json: adsorption.occupancy=${ad.occupancy} must be an integer >= 1`)
  }
  // Architectural ceiling, not a physics one (mirrors headPlacement.chainCapacity's own check
  // above): soup/wgsl/bond.wgsl's centerLink is a single u32 slot per particle, not an array --
  // raising occupancy past 1 needs that buffer restructured first (data/soup.json's own basis
  // explains why this was considered and deliberately deferred, not overlooked).
  if (ad.occupancy !== 1) {
    throw new Error(
      `data/soup.json: adsorption.occupancy=${ad.occupancy} -- soup/wgsl/bond.wgsl's centerLink -- ` +
        `one u32 slot per particle -- supports only 1; raising this number requires a separate buffer restructuring`,
    )
  }
  if (!Number.isInteger(ad.maxHoldSteps) || ad.maxHoldSteps < 1) {
    throw new Error(`data/soup.json: adsorption.maxHoldSteps=${ad.maxHoldSteps} must be an integer >= 1`)
  }
  if (!ad.basis || ad.basis.trim().length <= 10) {
    throw new Error('data/soup.json: adsorption has no substantive justification (basis)')
  }

  const dwc = s.dryWetCycle
  if (typeof dwc.enabled !== 'boolean') {
    throw new Error('data/soup.json: dryWetCycle.enabled must be a boolean')
  }
  if (!Number.isInteger(dwc.cycles) || dwc.cycles < 1) {
    throw new Error(`data/soup.json: dryWetCycle.cycles=${dwc.cycles} must be an integer >= 1`)
  }
  if (!Number.isInteger(dwc.periodSteps) || dwc.periodSteps < 1) {
    throw new Error(`data/soup.json: dryWetCycle.periodSteps=${dwc.periodSteps} must be an integer >= 1`)
  }
  if (!(dwc.dryFraction > 0) || !(dwc.dryFraction < 1)) {
    throw new Error(`data/soup.json: dryWetCycle.dryFraction=${dwc.dryFraction} must lie strictly between 0 and 1`)
  }
  if (!(dwc.targetDryDensity > 0)) {
    throw new Error(`data/soup.json: dryWetCycle.targetDryDensity=${dwc.targetDryDensity} must be a positive number`)
  }
  if (!Number.isInteger(dwc.rampSteps) || dwc.rampSteps < 1) {
    throw new Error(`data/soup.json: dryWetCycle.rampSteps=${dwc.rampSteps} must be an integer >= 1`)
  }
  if (!Number.isInteger(dwc.rampRelaxSteps) || dwc.rampRelaxSteps < 0) {
    throw new Error(`data/soup.json: dryWetCycle.rampRelaxSteps=${dwc.rampRelaxSteps} must be an integer >= 0`)
  }
  if (!dwc.basis || dwc.basis.trim().length <= 10) {
    throw new Error('data/soup.json: dryWetCycle has no substantive justification (basis)')
  }
  // Task 'evaporation' (2026-08-20): the four solvent-removal fields. Optional (like saltPhLimitation/
  // clay/coldStartRelax) so every pre-task Soup literal in tests/fixtures stays valid, and validated
  // the same way as everything else whenever present.
  if (dwc.evaporateSolvent !== undefined && typeof dwc.evaporateSolvent !== 'boolean') {
    throw new Error('data/soup.json: dryWetCycle.evaporateSolvent must be a boolean')
  }
  if (dwc.residualSolventFraction !== undefined && !(dwc.residualSolventFraction >= 0 && dwc.residualSolventFraction < 1)) {
    throw new Error(
      `data/soup.json: dryWetCycle.residualSolventFraction=${dwc.residualSolventFraction} must lie in [0,1): ` +
        `the dry phase must remove the solvent, not keep all of it`,
    )
  }
  if (dwc.evaporationRampSteps !== undefined && (!Number.isInteger(dwc.evaporationRampSteps) || dwc.evaporationRampSteps < 1)) {
    throw new Error(`data/soup.json: dryWetCycle.evaporationRampSteps=${dwc.evaporationRampSteps} must be an integer >= 1`)
  }
  if (dwc.insertionMinSeparationSigma !== undefined && !(dwc.insertionMinSeparationSigma > 0)) {
    throw new Error(
      `data/soup.json: dryWetCycle.insertionMinSeparationSigma=${dwc.insertionMinSeparationSigma} must be positive`,
    )
  }
  if (dwc.evaporateSolvent) {
    for (const k of ['residualSolventFraction', 'evaporationRampSteps', 'insertionMinSeparationSigma'] as const) {
      if (dwc[k] === undefined) {
        throw new Error(`data/soup.json: dryWetCycle.evaporateSolvent=true, but field ${k} is not set`)
      }
    }
  }

  // Salt/pH limitation (task 'broth-composition'): optional (mirrors CheckpointDefaults' own
  // pattern) so every pre-existing Soup literal in tests/fixtures stays valid, but validated the
  // same way as every other section whenever it IS present.
  if (s.saltPhLimitation) {
    const sp = s.saltPhLimitation
    if (typeof sp.represented !== 'boolean') {
      throw new Error('data/soup.json: saltPhLimitation.represented must be a boolean')
    }
    if (!sp.basis || sp.basis.trim().length <= 10) {
      throw new Error('data/soup.json: saltPhLimitation has no substantive justification (basis)')
    }
  }

  // Clay platelet (task 'clay-surface'): optional, same pattern as saltPhLimitation above, and
  // validated section by section whenever present. NOTE what is deliberately NOT checked here
  // because it does not exist as a field: a bead count or a lattice spacing -- both are derived
  // geometry (soup/src/soup-clay.ts), so there is no number here that could disagree with the box.
  if (s.clay) {
    const c = s.clay
    if (typeof c.enabled !== 'boolean') {
      throw new Error('data/soup.json: clay.enabled must be a boolean')
    }
    const mineral = s.monomers.find((m) => m.id === c.mineralId)
    if (!mineral) {
      throw new Error(`data/soup.json: clay.mineralId="${c.mineralId}" not found among monomers`)
    }
    if (mineral.kind !== 'clay' || !mineral.mineral) {
      throw new Error(
        `data/soup.json: monomer "${c.mineralId}" is named clay.mineralId but does not have kind="clay" and mineral=true`,
      )
    }
    if (mineral.polar || mineral.solvent) {
      throw new Error(
        `data/soup.json: mineral monomer "${c.mineralId}" cannot also be polar/solvent: ` +
          `the MINERAL class is derived from mineral=true and has its own row in pairEpsilon`,
      )
    }
    if (!Number.isInteger(c.sheets) || c.sheets < 1) {
      throw new Error(`data/soup.json: clay.sheets=${c.sheets} must be an integer >= 1`)
    }
    if (!(c.siteCatalystFraction >= 0) || !(c.siteCatalystFraction <= 1)) {
      throw new Error(
        `data/soup.json: clay.siteCatalystFraction=${c.siteCatalystFraction} must lie in [0,1]`,
      )
    }
    if (c.rank !== 'D') {
      throw new Error(
        `data/soup.json: clay is declared with rank ${c.rank}; the sheet geometry and the fraction of sites on it are ` +
          `choices of this model, not measured quantities, so only rank D is allowed`,
      )
    }
    if (!c.basis || c.basis.trim().length <= 10) {
      throw new Error('data/soup.json: clay has no substantive justification (basis)')
    }
    // Task 'clay-surface-chemistry' (2026-08-19): the SURFACE CHEMISTRY is selectable, and both limits
    // of the hydrophilicity bracket must stay declared -- the real mineral is charged and neither limit
    // is it, so a single "the" clay surface would be a claim this engine cannot make. Checked here (a
    // schema failure) rather than at system creation (a GPU-path failure).
    const chems = c.surfaceChemistries
    if (!chems || typeof chems !== 'object' || Object.keys(chems).length < 2) {
      throw new Error(
        'data/soup.json: clay.surfaceChemistries must declare both limits of the hydrophilicity bracket ' +
          '(the hydrophilic limit P5 and the apolar limit SC6/C1); a single "the" surface would be a claim ' +
          'this engine cannot make: the real mineral is charged, and there is no charge here',
      )
    }
    if (typeof c.surfaceChemistry !== 'string' || chems[c.surfaceChemistry] === undefined) {
      throw new Error(
        `data/soup.json: clay.surfaceChemistry="${c.surfaceChemistry}" is not declared in clay.surfaceChemistries ` +
          `(available: ${Object.keys(chems).join(', ')})`,
      )
    }
    if (!c.surfaceChemistryBasis || c.surfaceChemistryBasis.trim().length <= 10) {
      throw new Error('data/soup.json: clay.surfaceChemistryBasis does not explain why there are two limits')
    }
    const levels = s.solvent.attractionScale?.pairEpsilon?.levels
    for (const [name, chem] of Object.entries(chems)) {
      if (chem.rank !== 'A' && chem.rank !== 'B' && chem.rank !== 'C' && chem.rank !== 'D') {
        throw new Error(`data/soup.json: clay.surfaceChemistries["${name}"].rank=${chem.rank}: only A..D are allowed`)
      }
      if (!chem.basis || chem.basis.trim().length <= 10) {
        throw new Error(`data/soup.json: clay.surfaceChemistries["${name}"] has no substantive justification (basis)`)
      }
      for (const key of ['mineralApolar', 'mineralPolar', 'mineralSolvent', 'mineralMineral']) {
        const alias = chem.pairs?.[key]
        if (typeof alias !== 'string') {
          throw new Error(`data/soup.json: clay.surfaceChemistries["${name}"].pairs does not set the pair "${key}"`)
        }
        // A chemistry is an ALIAS onto a level that already exists with its own martini source line --
        // never a number of its own. This check is what makes that structural rather than a convention.
        if (levels && levels[alias] === undefined) {
          throw new Error(
            `data/soup.json: clay.surfaceChemistries["${name}"].pairs.${key}="${alias}" does not point to a level ` +
              `in solvent.attractionScale.pairEpsilon.levels: a surface chemistry is an alias of a level, not a depth of its own`,
          )
        }
      }
    }
    // A clay-class column in the depth table is not optional once a mineral monomer exists: without
    // it soup/src/soup-attraction.ts would throw at system-creation time instead of here, i.e. the
    // schema error would surface as a GPU-path failure rather than as a schema failure.
    const pe = s.solvent.attractionScale?.pairEpsilon
    if (pe) {
      for (const key of ['mineralApolar', 'mineralPolar', 'mineralSolvent', 'mineralMineral']) {
        if (pe.levels[key] === undefined) {
          throw new Error(
            `data/soup.json: mineral monomer "${c.mineralId}" is declared, but solvent.attractionScale.pairEpsilon.levels has no pair "${key}"`,
          )
        }
      }
    }
  }

  // Checkpoint/resume: optional (see CheckpointDefaults' own doc comment for why), but validated
  // the same way as every other section here whenever it IS present.
  // Task 'loud-failure-and-liquid-water' (2026-08-20): the cold-start minimisation protocol. Same
  // shape as every other optional section -- a present section must be internally sensible and must
  // carry a substantive basis, an absent one is simply "this file predates the stage".
  if (s.coldStartRelax) {
    const cr = s.coldStartRelax
    if (!Number.isInteger(cr.iterations) || cr.iterations < 1) {
      throw new Error(`data/soup.json: coldStartRelax.iterations=${cr.iterations} must be an integer >= 1`)
    }
    if (!Number.isFinite(cr.maxDisplacementSigma) || cr.maxDisplacementSigma <= 0) {
      throw new Error(
        `data/soup.json: coldStartRelax.maxDisplacementSigma=${cr.maxDisplacementSigma} must be a finite positive number`,
      )
    }
    if (!cr.basis || cr.basis.trim().length <= 10) {
      throw new Error('data/soup.json: coldStartRelax has no substantive justification (basis)')
    }
  }

  if (s.checkpoint) {
    const cp = s.checkpoint
    if (!Number.isInteger(cp.everySteps) || cp.everySteps < 1) {
      throw new Error(`data/soup.json: checkpoint.everySteps=${cp.everySteps} must be an integer >= 1`)
    }
    if (!cp.dir || cp.dir.trim().length === 0) {
      throw new Error('data/soup.json: checkpoint.dir must not be empty')
    }
    if (!cp.basis || cp.basis.trim().length <= 10) {
      throw new Error('data/soup.json: checkpoint has no substantive justification (basis)')
    }
  }
}
