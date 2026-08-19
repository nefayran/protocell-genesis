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
 * Проверяет три инварианта набора правил и один инвариант стартового состава.
 * Бросает Error с сообщением, называющим нарушителя, при первом нарушении:
 *  - у правила rank не 'D' (для этой модели измеренных констант скорости нет —
 *    заявлять более высокий ранг было бы нечестно);
 *  - у правила пустое или тривиальное basis (обоснование скорости обязано быть содержательным);
 *  - у правила образования ('bond') нет парного разрыва ('break') с теми же a/b,
 *    либо энергии пары не совпадают — без этого детальный баланс не определён;
 *  - у мономера вид не из четырёх элементарных ('carbon'|'head'|'donor'|'catalyst'),
 *    либо start ссылается на id, не объявленный в monomers — это и есть проверка
 *    «только мономеры, без готового амфифила» на входе.
 */
export function assertRulesConsistent(s: Soup): void {
  for (const r of s.rules) {
    if (r.rank !== 'D') {
      throw new Error(
        `data/soup.json: правило ${r.id} имеет ранг ${r.rank}, а для скоростей синтеза без измеренных констант допускается только ранг D`,
      )
    }
    if (!r.basis || r.basis.trim().length <= 10) {
      throw new Error(`data/soup.json: правило ${r.id} не имеет содержательного обоснования (basis)`)
    }
  }

  for (const r of s.rules.filter((x) => x.kind === 'bond')) {
    const back = s.rules.find((x) => x.kind === 'break' && x.a === r.a && x.b === r.b)
    if (!back) {
      throw new Error(
        `data/soup.json: у правила образования ${r.id} (${r.a}-${r.b}) нет парного правила разрыва — детальный баланс не определён`,
      )
    }
    if (back.energyKT !== r.energyKT) {
      throw new Error(
        `data/soup.json: правила ${r.id} и ${back.id} расходятся по энергии (${r.energyKT} против ${back.energyKT}) — парный разрыв обязан иметь ту же энергию, иначе детальный баланс нарушен`,
      )
    }
  }

  for (const m of s.monomers) {
    if (!ALLOWED_MONOMER_KINDS.has(m.kind)) {
      throw new Error(
        `data/soup.json: мономер ${m.id} имеет вид "${m.kind}", не входящий в набор элементарных строительных блоков`,
      )
    }
  }

  const sv = s.solvent
  const waterMonomer = s.monomers.find((m) => m.id === sv.waterId)
  if (!waterMonomer) {
    throw new Error(`data/soup.json: solvent.waterId="${sv.waterId}" не найден среди monomers`)
  }
  if (waterMonomer.kind !== 'water' || !waterMonomer.solvent) {
    throw new Error(
      `data/soup.json: мономер "${sv.waterId}" назван solvent.waterId, но не имеет kind="water" и solvent=true`,
    )
  }
  if (!sv.attractionRule || sv.attractionRule.trim().length <= 10) {
    throw new Error('data/soup.json: solvent.attractionRule не имеет содержательного описания')
  }
  if (!sv.basis || sv.basis.trim().length <= 10) {
    throw new Error('data/soup.json: solvent не имеет содержательного обоснования (basis)')
  }
  if (sv.attractionScale !== undefined) {
    const sc = sv.attractionScale
    if (!(Number.isFinite(sc.epsilonScale) && sc.epsilonScale > 0)) {
      throw new Error(`data/soup.json: solvent.attractionScale.epsilonScale=${sc.epsilonScale} должен быть конечным положительным числом`)
    }
    if (!sc.basis || sc.basis.trim().length <= 10) {
      throw new Error('data/soup.json: solvent.attractionScale не имеет содержательного обоснования (basis)')
    }
    if (sc.pairEpsilon !== undefined) {
      const pe = sc.pairEpsilon
      const ref = pe.levels?.[pe.reference]
      if (!ref || !(Number.isFinite(ref.epsilonKJ) && ref.epsilonKJ > 0)) {
        throw new Error(
          `data/soup.json: solvent.attractionScale.pairEpsilon.reference="${pe.reference}" не указывает на уровень с положительным epsilonKJ`,
        )
      }
      for (const [key, lvl] of Object.entries(pe.levels)) {
        if (!Number.isFinite(lvl.epsilonKJ) || lvl.epsilonKJ < 0) {
          throw new Error(`data/soup.json: solvent.attractionScale.pairEpsilon.levels.${key}.epsilonKJ=${lvl.epsilonKJ} должен быть конечным неотрицательным числом`)
        }
      }
    }
  }

  if (s.areaMove !== undefined) {
    const am = s.areaMove
    if (!(Number.isFinite(am.logDelta) && am.logDelta > 0)) {
      throw new Error(`data/soup.json: areaMove.logDelta=${am.logDelta} должен быть конечным положительным числом`)
    }
    if (am.mode !== 'lateral-fixed-volume' && am.mode !== 'lateral-fixed-z') {
      throw new Error(`data/soup.json: areaMove.mode="${am.mode}" не входит в набор {lateral-fixed-volume, lateral-fixed-z}`)
    }
    if (!am.basis || am.basis.trim().length <= 10) {
      throw new Error('data/soup.json: areaMove не имеет содержательного обоснования (basis)')
    }
  }

  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) {
    if (!ids.has(k)) {
      throw new Error(
        `data/soup.json: стартовый состав ссылается на "${k}", который не объявлен как мономер — это может быть готовый амфифил, а не строительный блок`,
      )
    }
  }

  const bai = s.bondAttemptInterval
  if (!Number.isInteger(bai.steps) || bai.steps < 1) {
    throw new Error(`data/soup.json: bondAttemptInterval.steps=${bai.steps} должен быть целым числом >= 1`)
  }
  if (!bai.basis || bai.basis.trim().length <= 10) {
    throw new Error('data/soup.json: bondAttemptInterval не имеет содержательного обоснования (basis)')
  }

  const ng = s.neighborGrid
  if (!Number.isInteger(ng.cellDivisor) || ng.cellDivisor < 1) {
    throw new Error(`data/soup.json: neighborGrid.cellDivisor=${ng.cellDivisor} должен быть целым числом >= 1`)
  }
  if (typeof ng.sortedGather !== 'boolean') {
    throw new Error('data/soup.json: neighborGrid.sortedGather должен быть булевым значением')
  }
  if (!ng.basis || ng.basis.trim().length <= 10) {
    throw new Error('data/soup.json: neighborGrid не имеет содержательного обоснования (basis)')
  }

  const vl = s.verletList
  if (typeof vl.enabled !== 'boolean') {
    throw new Error('data/soup.json: verletList.enabled должен быть булевым значением')
  }
  if (!(vl.skin > 0)) {
    throw new Error(`data/soup.json: verletList.skin=${vl.skin} должен быть положительным числом`)
  }
  if (!Number.isInteger(vl.rebuildEvery) || vl.rebuildEvery < 1) {
    throw new Error(`data/soup.json: verletList.rebuildEvery=${vl.rebuildEvery} должен быть целым числом >= 1`)
  }
  if (!Number.isInteger(vl.listCapacity) || vl.listCapacity < 1) {
    throw new Error(`data/soup.json: verletList.listCapacity=${vl.listCapacity} должен быть целым числом >= 1`)
  }
  if (!vl.basis || vl.basis.trim().length <= 10) {
    throw new Error('data/soup.json: verletList не имеет содержательного обоснования (basis)')
  }

  const hp = s.headPlacement
  if (typeof hp.terminalOnly !== 'boolean') {
    throw new Error('data/soup.json: headPlacement.terminalOnly должен быть булевым значением')
  }
  // Architectural ceiling, not a physics one: every particle (any kind) owns exactly 3 bondSlots
  // rows (soup/src/sim.ts's bondSlots0 -- N*3, uniform across kinds), so a head literally cannot
  // claim a 4th chain slot regardless of what data/soup.json asks for.
  const MAX_ARCHITECTURAL_SLOTS = 3
  if (!Number.isInteger(hp.chainCapacity) || hp.chainCapacity < 1 || hp.chainCapacity > MAX_ARCHITECTURAL_SLOTS) {
    throw new Error(
      `data/soup.json: headPlacement.chainCapacity=${hp.chainCapacity} должен быть целым числом от 1 до ${MAX_ARCHITECTURAL_SLOTS} (soup/src/sim.ts's per-particle bondSlots row)`,
    )
  }
  if (!hp.basis || hp.basis.trim().length <= 10) {
    throw new Error('data/soup.json: headPlacement не имеет содержательного обоснования (basis)')
  }

  const ad = s.adsorption
  if (!Number.isInteger(ad.occupancy) || ad.occupancy < 1) {
    throw new Error(`data/soup.json: adsorption.occupancy=${ad.occupancy} должен быть целым числом >= 1`)
  }
  // Architectural ceiling, not a physics one (mirrors headPlacement.chainCapacity's own check
  // above): soup/wgsl/bond.wgsl's centerLink is a single u32 slot per particle, not an array --
  // raising occupancy past 1 needs that buffer restructured first (data/soup.json's own basis
  // explains why this was considered and deliberately deferred, not overlooked).
  if (ad.occupancy !== 1) {
    throw new Error(
      `data/soup.json: adsorption.occupancy=${ad.occupancy} -- soup/wgsl/bond.wgsl's centerLink -- ` +
        `один u32-слот на частицу -- поддерживает только 1; поднять это число требует отдельной перестройки буфера`,
    )
  }
  if (!Number.isInteger(ad.maxHoldSteps) || ad.maxHoldSteps < 1) {
    throw new Error(`data/soup.json: adsorption.maxHoldSteps=${ad.maxHoldSteps} должен быть целым числом >= 1`)
  }
  if (!ad.basis || ad.basis.trim().length <= 10) {
    throw new Error('data/soup.json: adsorption не имеет содержательного обоснования (basis)')
  }

  const dwc = s.dryWetCycle
  if (typeof dwc.enabled !== 'boolean') {
    throw new Error('data/soup.json: dryWetCycle.enabled должен быть булевым значением')
  }
  if (!Number.isInteger(dwc.cycles) || dwc.cycles < 1) {
    throw new Error(`data/soup.json: dryWetCycle.cycles=${dwc.cycles} должен быть целым числом >= 1`)
  }
  if (!Number.isInteger(dwc.periodSteps) || dwc.periodSteps < 1) {
    throw new Error(`data/soup.json: dryWetCycle.periodSteps=${dwc.periodSteps} должен быть целым числом >= 1`)
  }
  if (!(dwc.dryFraction > 0) || !(dwc.dryFraction < 1)) {
    throw new Error(`data/soup.json: dryWetCycle.dryFraction=${dwc.dryFraction} должен лежать строго между 0 и 1`)
  }
  if (!(dwc.targetDryDensity > 0)) {
    throw new Error(`data/soup.json: dryWetCycle.targetDryDensity=${dwc.targetDryDensity} должен быть положительным числом`)
  }
  if (!Number.isInteger(dwc.rampSteps) || dwc.rampSteps < 1) {
    throw new Error(`data/soup.json: dryWetCycle.rampSteps=${dwc.rampSteps} должен быть целым числом >= 1`)
  }
  if (!Number.isInteger(dwc.rampRelaxSteps) || dwc.rampRelaxSteps < 0) {
    throw new Error(`data/soup.json: dryWetCycle.rampRelaxSteps=${dwc.rampRelaxSteps} должен быть целым числом >= 0`)
  }
  if (!dwc.basis || dwc.basis.trim().length <= 10) {
    throw new Error('data/soup.json: dryWetCycle не имеет содержательного обоснования (basis)')
  }

  // Salt/pH limitation (task 'broth-composition'): optional (mirrors CheckpointDefaults' own
  // pattern) so every pre-existing Soup literal in tests/fixtures stays valid, but validated the
  // same way as every other section whenever it IS present.
  if (s.saltPhLimitation) {
    const sp = s.saltPhLimitation
    if (typeof sp.represented !== 'boolean') {
      throw new Error('data/soup.json: saltPhLimitation.represented должен быть булевым значением')
    }
    if (!sp.basis || sp.basis.trim().length <= 10) {
      throw new Error('data/soup.json: saltPhLimitation не имеет содержательного обоснования (basis)')
    }
  }

  // Clay platelet (task 'clay-surface'): optional, same pattern as saltPhLimitation above, and
  // validated section by section whenever present. NOTE what is deliberately NOT checked here
  // because it does not exist as a field: a bead count or a lattice spacing -- both are derived
  // geometry (soup/src/soup-clay.ts), so there is no number here that could disagree with the box.
  if (s.clay) {
    const c = s.clay
    if (typeof c.enabled !== 'boolean') {
      throw new Error('data/soup.json: clay.enabled должен быть булевым значением')
    }
    const mineral = s.monomers.find((m) => m.id === c.mineralId)
    if (!mineral) {
      throw new Error(`data/soup.json: clay.mineralId="${c.mineralId}" не найден среди monomers`)
    }
    if (mineral.kind !== 'clay' || !mineral.mineral) {
      throw new Error(
        `data/soup.json: мономер "${c.mineralId}" назван clay.mineralId, но не имеет kind="clay" и mineral=true`,
      )
    }
    if (mineral.polar || mineral.solvent) {
      throw new Error(
        `data/soup.json: минеральный мономер "${c.mineralId}" не может быть одновременно polar/solvent — ` +
          `класс MINERAL выводится из mineral=true и имеет собственную строку в pairEpsilon`,
      )
    }
    if (!Number.isInteger(c.sheets) || c.sheets < 1) {
      throw new Error(`data/soup.json: clay.sheets=${c.sheets} должен быть целым числом >= 1`)
    }
    if (!(c.siteCatalystFraction >= 0) || !(c.siteCatalystFraction <= 1)) {
      throw new Error(
        `data/soup.json: clay.siteCatalystFraction=${c.siteCatalystFraction} должен лежать в [0,1]`,
      )
    }
    if (c.rank !== 'D') {
      throw new Error(
        `data/soup.json: clay заявлен рангом ${c.rank}; геометрия пластины и доля центров на ней — ` +
          `выбор этой модели, а не измеренные величины, поэтому допускается только ранг D`,
      )
    }
    if (!c.basis || c.basis.trim().length <= 10) {
      throw new Error('data/soup.json: clay не имеет содержательного обоснования (basis)')
    }
    // Task 'clay-surface-chemistry' (2026-08-19): the SURFACE CHEMISTRY is selectable, and both limits
    // of the hydrophilicity bracket must stay declared -- the real mineral is charged and neither limit
    // is it, so a single "the" clay surface would be a claim this engine cannot make. Checked here (a
    // schema failure) rather than at system creation (a GPU-path failure).
    const chems = c.surfaceChemistries
    if (!chems || typeof chems !== 'object' || Object.keys(chems).length < 2) {
      throw new Error(
        'data/soup.json: clay.surfaceChemistries должен объявлять ОБЕ границы вилки гидрофильности ' +
          '(гидрофильный предел P5 и аполярный предел SC6/C1) — одна «та самая» поверхность была бы утверждением, ' +
          'которого этот движок сделать не может: настоящий минерал заряжен, а заряда здесь нет',
      )
    }
    if (typeof c.surfaceChemistry !== 'string' || chems[c.surfaceChemistry] === undefined) {
      throw new Error(
        `data/soup.json: clay.surfaceChemistry="${c.surfaceChemistry}" не объявлен в clay.surfaceChemistries ` +
          `(есть: ${Object.keys(chems).join(', ')})`,
      )
    }
    if (!c.surfaceChemistryBasis || c.surfaceChemistryBasis.trim().length <= 10) {
      throw new Error('data/soup.json: clay.surfaceChemistryBasis не объясняет, почему пределов два')
    }
    const levels = s.solvent.attractionScale?.pairEpsilon?.levels
    for (const [name, chem] of Object.entries(chems)) {
      if (chem.rank !== 'A' && chem.rank !== 'B' && chem.rank !== 'C' && chem.rank !== 'D') {
        throw new Error(`data/soup.json: clay.surfaceChemistries["${name}"].rank=${chem.rank} — допускаются только A..D`)
      }
      if (!chem.basis || chem.basis.trim().length <= 10) {
        throw new Error(`data/soup.json: clay.surfaceChemistries["${name}"] не имеет содержательного обоснования (basis)`)
      }
      for (const key of ['mineralApolar', 'mineralPolar', 'mineralSolvent', 'mineralMineral']) {
        const alias = chem.pairs?.[key]
        if (typeof alias !== 'string') {
          throw new Error(`data/soup.json: clay.surfaceChemistries["${name}"].pairs не задаёт пару "${key}"`)
        }
        // A chemistry is an ALIAS onto a level that already exists with its own martini source line --
        // never a number of its own. This check is what makes that structural rather than a convention.
        if (levels && levels[alias] === undefined) {
          throw new Error(
            `data/soup.json: clay.surfaceChemistries["${name}"].pairs.${key}="${alias}" не указывает на уровень ` +
              `в solvent.attractionScale.pairEpsilon.levels — химия поверхности это ПСЕВДОНИМ уровня, а не своя глубина`,
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
            `data/soup.json: объявлен минеральный мономер "${c.mineralId}", но solvent.attractionScale.pairEpsilon.levels не содержит пары "${key}"`,
          )
        }
      }
    }
  }

  // Checkpoint/resume: optional (see CheckpointDefaults' own doc comment for why), but validated
  // the same way as every other section here whenever it IS present.
  if (s.checkpoint) {
    const cp = s.checkpoint
    if (!Number.isInteger(cp.everySteps) || cp.everySteps < 1) {
      throw new Error(`data/soup.json: checkpoint.everySteps=${cp.everySteps} должен быть целым числом >= 1`)
    }
    if (!cp.dir || cp.dir.trim().length === 0) {
      throw new Error('data/soup.json: checkpoint.dir не должен быть пустым')
    }
    if (!cp.basis || cp.basis.trim().length <= 10) {
      throw new Error('data/soup.json: checkpoint не имеет содержательного обоснования (basis)')
    }
  }
}
