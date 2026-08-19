import raw from '../../data/soup.json'

/** Один из элементарных строительных блоков — не готовый амфифил. `water` (task 'explicit-water',
 * 2026-08-18) is the solvent bead: `solvent=true` marks it for the attraction rule below, `polar`
 * stays false for it (polar denotes the carboxyl HEAD group specifically, not "hydrophilic" in
 * general -- see soup/wgsl/step.wgsl's shouldAttract for how the two flags combine). */
export interface Monomer {
  id: string
  kind: 'carbon' | 'head' | 'donor' | 'catalyst' | 'water'
  radiusSigma: number
  polar: boolean
  /** True only for the solvent species (data/soup.json's `W`). Drives soup/wgsl/step.wgsl's
   * nonbondedSoup attraction gate together with `polar`: water-water and water-head attract,
   * water-tail and head-head do not -- see data/soup.json's `solvent.basis` for the full argument,
   * including why the old blanket "both nonpolar" tail-tail attraction was removed rather than
   * kept alongside this (double-counting the same hydrophobic-exclusion physics). Optional so every
   * existing Monomer literal in this codebase (tests, older fixtures) that predates this field
   * stays valid -- undefined reads as false, exactly like a monomer that never mentions `polar`
   * would read as non-polar under the same `?? false` convention used elsewhere in this file. */
  solvent?: boolean
}

/**
 * Правило образования или разрыва связи между двумя видами мономеров.
 * У каждого 'bond' обязана быть парная 'break' с ТОЙ ЖЕ энергией — это и есть
 * детальный баланс: абсолютная скорость (rank D, оценка) не откалибрована,
 * но отношение вперёд/назад калибровано энергией и поэтому честное.
 */
export interface Rule {
  id: string
  kind: 'bond' | 'break'
  a: string
  b: string
  requiresCatalyst: boolean
  energyKT: number
  attemptRate: number
  rank: 'A' | 'B' | 'C' | 'D'
  basis: string
}

export interface Sweep {
  kT: number[]
  carbonDensity: number[]
  catalystCount: number[]
}

/**
 * Performance-only batching of the bond Monte Carlo (perf-report.md, candidate fix (b)):
 * bond_form_main/bond_break_main run every `steps` integration steps instead of every step, with
 * attemptProbability's dt multiplied by `steps` to keep the average attempt rate per real step
 * unchanged (Poisson thinning) -- acceptanceProbability (the one number carrying detailed balance)
 * is untouched by this. Not a physical rate, so no `rank`/energy field the way Rule has.
 */
export interface BondAttemptInterval {
  steps: number
  basis: string
}

/**
 * Performance-only sizing of the neighbour grid (perf2-report.md, candidate (a)): the cell side is
 * the interaction range divided by `cellDivisor`, and the walk radius (in cells) is derived from
 * it and asserted, never assumed -- see soup/src/sim.ts's own assertion right after this value is
 * read. `cellDivisor=1` reproduces the original 3x3x3 walk exactly (same cell side, walk radius 1)
 * and is the honest A/B control point for this change, the same role
 * `bondAttemptInterval.steps=1` plays for perf-report.md's fix (b).
 */
export interface NeighborGrid {
  cellDivisor: number
  /** Selects which shader entry point soup/src/sim.ts compiles for the force/bond-form kernels --
   * `soup_force_main`/`bond_form_main` (read the cell-sorted gather) when true,
   * `soup_force_main_unsorted`/`bond_form_main_unsorted` (read positions at the original,
   * scattered index directly) when false. Both are one WGSL implementation parametrised by a
   * `useSorted` argument (soup/wgsl/step.wgsl's soupForceWalk, soup/wgsl/bond.wgsl's
   * bondFormWalk) picked at pipeline-creation time, not a runtime branch in the hot loop -- see
   * perf2-report.md, candidate (b), for why this flag exists (an honest, code-identical A/B).
   */
  sortedGather: boolean
  basis: string
}

/**
 * Performance-only Verlet neighbour list (perf2-report.md, candidate (c)): built every
 * `rebuildEvery` real steps instead of every step, with a `skin` margin so it stays complete
 * between rebuilds. `enabled=false` reproduces the pre-(c) behaviour (cell walk every step,
 * unmodified) exactly, the honest A/B control point for this candidate -- the same role
 * `bondAttemptInterval.steps=1` and `neighborGrid.cellDivisor=1` play for their own candidates.
 * See soup/src/sim.ts for the drift-safety assertion this pair (skin, rebuildEvery) must satisfy
 * at system-creation time, and soup/wgsl/step.wgsl's soup_max_drift_main for the per-step empirical
 * check backing that analytical bound up.
 */
export interface VerletList {
  enabled: boolean
  skin: number
  rebuildEvery: number
  listCapacity: number
  basis: string
}

/**
 * Modelling constraint (rank D, like every other rule in this file): a carboxyl head sits on a
 * TERMINAL carbon in real amphiphile chemistry, never buried mid-chain. `terminalOnly=true` makes
 * that a condition on the bond-forming attempt itself (soup/wgsl/bond.wgsl's tryClaimSlot):
 * (1) a head may claim its slot on a carbon only while that carbon has at most one existing C-C
 * bond, and (2) a carbon that already carries a head may not accept a further C-C bond. Both are
 * expressed through the SAME atomic claim-then-rollback idiom the valence cap already uses, not a
 * separate lock -- see the WGSL file for the reversibility argument (breaking a C-O bond off a
 * terminal-head carbon returns to a state where forming it is again allowed; a C-C bond this
 * rule blocks can never have formed, so its own reverse never needs to fire). `terminalOnly=false`
 * reproduces pre-change behaviour exactly (the same honest A/B control point every other switch in
 * this file provides) and is the value to flip back to if this constraint is ever suspected of
 * corrupting detailed balance.
 */
export interface HeadPlacement {
  terminalOnly: boolean
  /** How many chain (C-O) bonds a head may hold at once -- soup/wgsl/bond.wgsl's tryClaimSlot
   * (role==2) claims slots 0..chainCapacity-1 on the head's own bondSlots row, the same
   * claim-then-rollback discipline the 2-slot carbon chain pool already uses, just generalised to
   * a configurable count instead of two hardcoded branches. Bounded by the architectural per-
   * particle slot count (3, soup/src/sim.ts's bondSlots0) regardless of kind -- see
   * assertRulesConsistent below. Raised from 1 to 2 (data/soup.json's own basis, "two-tails" task)
   * so a head can carry two tails, the packing-parameter geometry real membrane lipids use.
   */
  chainCapacity: number
  basis: string
}

/**
 * Surface growth / adsorption (task 'adsorption', 2026-08-17, adsorption-report.md): the
 * catalytic-centre occupancy and desorption-safety-valve parameters the FENE-tethered
 * catalyst<->tip mechanism needs (soup/wgsl/bond.wgsl's centerLink/centerHeldSteps/desorbStretch/
 * desorbTimeout). Rank D like every other model choice in this file -- see data/soup.json's own
 * `basis` for the physical reasoning behind each field.
 */
export interface Adsorption {
  /** How many chains a single catalytic centre may hold adsorbed at once. The current
   * implementation (soup/wgsl/bond.wgsl's centerLink) is a single scalar slot per particle, so
   * only 1 is supported today -- kept as an explicit, validated field (not a silent assumption) so
   * a future multi-site model has one place to raise it, with its own basis, rather than changing
   * behaviour underneath an unvalidated number. */
  occupancy: number
  /** Real integration steps a centre may hold the SAME chain tip without a single successful
   * propagation or termination event on it (both reset the running count) before the timeout
   * desorption safety valve releases it regardless of distance -- the deadlock-impossible-by-
   * construction guarantee, on top of (not instead of) the FENE tether itself. */
  maxHoldSteps: number
  basis: string
}

/**
 * Dry-wet cycling (task 'wet-dry-cycle', 2026-08-17,
 * .superpowers/sdd/2026-08-16-soup-to-vesicle/wet-dry-cycle-report.md): the experimentally
 * demonstrated route to closed vesicles (Deamer and co-workers, dry-wet cycling of fatty-acid
 * amphiphiles) that every prior run in this project did NOT have access to -- every earlier run held
 * volume/temperature/concentration fixed for its whole duration. Rank D like every other model choice
 * in this file: the SCHEDULE (period, amplitude, ramp) is an engineering choice this task made from
 * the density regime this project itself already measured (0.45 sigma^-3 single-cup, 0.72
 * runaway-polymerisation), not a literature rate constant. `enabled=false` is the file's own default
 * -- soup/src/sim.ts's createSoup() only cycles the box when this is true OR its own
 * CreateSoupOpts.dryWetCycle override says so, so every existing run/test/viewer session that never
 * asks for cycling sees IDENTICAL behaviour to before this task, byte for byte.
 */
export interface DryWetCycle {
  enabled: boolean
  /** How many full wet-then-dry cycles to run before settling back to (and staying at) the wet box. */
  cycles: number
  /** Real integration steps per full cycle (wet segment + dry segment together). */
  periodSteps: number
  /** Fraction of `periodSteps` spent DRY (contracted) -- the remaining `1-dryFraction` is spent WET
   * (at the box this system was created with). */
  dryFraction: number
  /** The box-average particle density (sigma^-3) the dry segment's box is sized to reach -- the
   * cycle's "amplitude". soup/src/sim.ts's createSoup() derives the dry box from THIS system's own
   * N and starting box (dryVolume = N/targetDryDensity), not from a literal box size, so the same
   * config value reproduces the same physical density regardless of which box/composition a caller
   * requests. */
  targetDryDensity: number
  /** The wet<->dry transition is NOT one instantaneous jump: it is spread over this many discrete
   * log-linear box increments (see soup/src/sim.ts's applyBoxScale), each one immediately followed by
   * `rampRelaxSteps` of ordinary dynamics before the next increment -- the same discipline
   * engine/src/sim.ts's own area move keeps each proposed step small (AREA_MOVE_LOG_DELTA) rather
   * than jumping straight to a target area, generalised from a stochastic MC step to a forced
   * mechanical one. */
  rampSteps: number
  /** Real dynamics steps run between two consecutive ramp increments, letting WCA overlaps introduced
   * by that increment's compression relax before the next one lands -- see rampSteps' own comment. */
  rampRelaxSteps: number
  basis: string
}

/**
 * Checkpoint/resume (task 'checkpoint-resume'): the DEFAULT interval/location a long soup run
 * saves state at, engineering-only (no rank/energy the way Rule has, same as BondAttemptInterval) --
 * see data/soup.json's own basis for why the number is what it is. Optional (not one of Soup's
 * REQUIRED keys, unlike every physics-bearing section above) so this file stays loadable by any
 * existing caller/test that predates this field, without a matching change on their end.
 * soup/cli/campaign.ts's own --every/--dir flags override this file's numbers when given, exactly
 * how CreateSoupOpts.catalystCount already overrides data/soup.json's own `start.M`.
 */
export interface CheckpointDefaults {
  everySteps: number
  dir: string
  basis: string
}

/**
 * Task 'explicit-water' (2026-08-18): the nonbonded-attraction scheme's own documentation, not a
 * second implementation of it -- soup/wgsl/step.wgsl's shouldAttract() computes the rule directly
 * from each species' own `polar`/`solvent` flags (already uploaded per monomer), so there is
 * nothing numeric here for params.test.ts's literal-scanner rule to catch and nothing that could
 * drift from the WGSL side by having two copies of a pairwise table. `attractionRule` is a
 * human-readable restatement of that same formula, for a reader who has not opened the WGSL.
 */
export interface Solvent {
  /** data/soup.json monomer id of the solvent species -- read by callers that need to find water
   * particles (e.g. soup/src/water-closure.ts) without hardcoding the id "W". */
  waterId: string
  attractionRule: string
  basis: string
}

/**
 * Task 'broth-composition' (2026-08-18): this model carries no electrostatics -- no partial
 * charges, no Debye screening, no explicit Na+/Cl-, no Henderson-Hasselbalch protonation -- so
 * ionic strength and pH cannot be represented, not even approximately by something that would look
 * like chemistry without being it (data/soup.json's own `basis` spells out why a pseudo-pH knob was
 * deliberately not added). `represented` is always `false` today; kept as an explicit field (not a
 * comment) so a future task that DID add electrostatics has one place to flip it, with its own
 * basis, rather than this limitation silently going stale.
 */
export interface SaltPhLimitation {
  represented: boolean
  basis: string
}

export interface Soup {
  /** Единственная явная калибровка временнóй шкалы модели (kappa_t на экране в отчётах). */
  kappaT: number
  monomers: Monomer[]
  rules: Rule[]
  start: Record<string, number>
  sweep: Sweep
  neighborGrid: NeighborGrid
  verletList: VerletList
  bondAttemptInterval: BondAttemptInterval
  headPlacement: HeadPlacement
  adsorption: Adsorption
  dryWetCycle: DryWetCycle
  solvent: Solvent
  saltPhLimitation?: SaltPhLimitation
  checkpoint?: CheckpointDefaults
}

const ALLOWED_MONOMER_KINDS = new Set<Monomer['kind']>(['carbon', 'head', 'donor', 'catalyst', 'water'])

const REQUIRED = [
  'kappaT', 'monomers', 'rules', 'start', 'sweep', 'neighborGrid', 'verletList', 'bondAttemptInterval',
  'headPlacement', 'adsorption', 'dryWetCycle', 'solvent',
] as const

export function loadSoup(): Soup {
  const s = raw as unknown as Soup
  for (const key of REQUIRED) {
    if ((s as unknown as Record<string, unknown>)[key] === undefined) {
      throw new Error(`data/soup.json: отсутствует поле ${key}`)
    }
  }
  return s
}

/**
 * Отношение обратной скорости к прямой при детальном балансе для правила
 * образования связи: exp(-энергия связи в единицах kT). При acceptanceProbability
 * с kT=1 (базовая калибровка kappaT) это ровно то отношение, которое получает
 * система из связки acceptanceProbability(bond)/acceptanceProbability(break).
 */
export function forwardBackwardRatio(r: Rule): number {
  return Math.exp(-r.energyKT)
}

/** Вероятность ПОПЫТКИ реакции за шаг интегратора; dt приходит извне (engine/src/params), не хардкодится здесь. */
export function attemptProbability(r: Rule, dt: number): number {
  return r.attemptRate * dt
}

/**
 * Вероятность ПРИНЯТИЯ попытки по Метрополису. Образование связи снижает
 * энергию на energyKT (ΔE = -energyKT) и поэтому принимается всегда;
 * разрыв — та же связь в обратную сторону (ΔE = +energyKT) и принимается
 * с вероятностью exp(-energyKT/kT). Оба случая — одна формула, поэтому
 * прямое и обратное направления автоматически удовлетворяют детальному
 * балансу и не могут разъехаться по отдельным константам.
 */
export function acceptanceProbability(r: Rule, kT: number): number {
  const deltaE = r.kind === 'bond' ? -r.energyKT : r.energyKT
  return Math.min(1, Math.exp(-deltaE / kT))
}

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
