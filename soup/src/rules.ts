import raw from '../../data/soup.json'

/** One of the elementary building blocks, not a ready-made amphiphile. `water` (task 'explicit-water',
 * 2026-08-18) is the solvent bead: `solvent=true` marks it for the attraction rule below, `polar`
 * stays false for it (polar denotes the carboxyl HEAD group specifically, not "hydrophilic" in
 * general -- see soup/wgsl/step.wgsl's shouldAttract for how the two flags combine). */
export interface Monomer {
  id: string
  kind: 'carbon' | 'head' | 'donor' | 'catalyst' | 'water' | 'clay'
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
  /** True only for the MINERAL phase (data/soup.json's clay bead `K`, task 'clay-surface',
   * 2026-08-19). Two things follow from it, and they are deliberately the SAME flag rather than
   * two, because in this model there is nothing mobile that is mineral and nothing immobile that is
   * not:
   *  - interaction CLASS: soup/src/soup-attraction.ts's speciesClassOf checks `mineral` BEFORE
   *    `solvent`/`polar`, so a mineral bead gets its own row/column in
   *    solvent.attractionScale.pairEpsilon (mineralSolvent / mineralPolar / mineralApolar /
   *    mineralMineral) instead of borrowing the head's or the water's;
   *  - it makes a particle ELIGIBLE to be frozen. Immobility itself is per-PARTICLE, not per
   *    species (soup/src/soup-clay.ts's `frozen` array, soup/wgsl/step.wgsl's frozenRO): the same
   *    catalyst species `M` has both free beads diffusing in the broth and beads immobilised as
   *    surface sites on the platelet, so a species-wide switch could not express it. */
  mineral?: boolean
}

/**
 * A rule for forming or breaking a bond between two monomer kinds.
 * Every 'bond' must have a paired 'break' with the same energy; that is what
 * detailed balance means here: the absolute rate (rank D, an estimate) is not calibrated,
 * but the forward/backward ratio is calibrated by the energy and is therefore honest.
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
  /** FALLBACK ONLY (task 'big-box', 2026-08-20). The per-particle capacity a real system runs with is
   * DERIVED from its own densest box's density (soup/src/soup-plan.ts's deriveListCapacity); this
   * value is what a caller with no census/box to derive from gets, the same role
   * `electrostatics.longRangeListCapacity` plays for the long-range list. */
  listCapacity: number
  /** Multiplier on the uniform-density expectation that becomes the derived per-particle capacity --
   * the measured inhomogeneity headroom, not a guess. See this section's `basis` in data/soup.json. */
  capacitySafetyFactor: number
  /** Absolute lower bound on the derived per-particle capacity -- the largest neighbour count ever
   * measured inside a CONDENSED phase, plus a margin. A dilute box derives a tiny capacity from its
   * own average density and then condenses into a droplet whose local density has nothing to do with
   * it; this floor is what covers that. */
  capacityFloor: number
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
  /** Task 'evaporation' (2026-08-20): whether the dry phase REMOVES SOLVENT BEADS FROM THE SYSTEM
   * (and rehydration puts them back), instead of merely compressing the box at fixed composition.
   * Optional and default-off (see the basis, item 1): a system that does not ask for it takes the
   * pre-existing box-scaling-only path byte for byte, which is what tests/soup-drywet-cycling.test.ts
   * pins. Overridable per system by CreateSoupOpts.evaporateSolvent, the same boolean-only pattern
   * `enabled` itself uses. */
  evaporateSolvent?: boolean
  /** Fraction of the wet solvent pool that REMAINS in the dry phase -- 1/1400, straight from
   * Ross & Deamer 2016's own measured ~1400-fold volume reduction (basis item 2). Not a free
   * parameter: the volume of an aqueous solution IS essentially the volume of its water, so a
   * 1400-fold volume reduction at fixed solute content maps one-to-one onto removing 1 - 1/1400 of
   * the solvent. */
  residualSolventFraction?: number
  /** How many log-linear increments the EVAPORATING wet<->dry transition is spread over -- separate
   * from `rampSteps` because removing the solvent makes the box change an order of magnitude larger
   * (|ln(L_wet/L_dry)| 0.388 against 0.032), so reusing `rampSteps` would make each increment 6.3 %
   * linear instead of the 3.2 % this project has actually measured safe (basis item 8). */
  evaporationRampSteps?: number
  /** Minimum separation (in sigma) a returning solvent bead is placed at from every particle already
   * present, by rejection sampling against a cell list -- derived from the random-sequential-addition
   * saturation fraction, basis item 6. Deliberately INSIDE the WCA core, which is why the insertion
   * is followed by a minimisation that moves ONLY the inserted beads (basis item 7). */
  insertionMinSeparationSigma?: number
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
 * Task 'explicit-water' (2026-08-18), amended by task 'hydrophobic-asymmetry' (2026-08-19): the
 * nonbonded-attraction scheme's own documentation, not a second implementation of it --
 * soup/wgsl/step.wgsl's speciesClass()/pairAttrScale() derive the rule from each species' own
 * `polar`/`solvent` flags (already uploaded per monomer) plus the class ratio table
 * soup/src/soup-attraction.ts builds from `attractionScale.pairEpsilon`, so there is exactly ONE
 * place the depths live (data/soup.json) and one place they are turned into a table.
 * `attractionRule` is a human-readable restatement of that same formula, for a reader who has not
 * opened the WGSL.
 */
/**
 * Water-calibration task (2026-08-19) introduced `epsilonScale`, a dimensionless multiplier on
 * attr_dv's own magnitude (the SAME Cooke & Deserno formula/range, data/params.json's epsilon/wc,
 * both rank A and untouched). Task 'hydrophobic-asymmetry' (2026-08-19) kept it as the GLOBAL
 * multiplier of the whole per-class table `pairEpsilon` below (it is unchanged at 1) and moved the
 * question of WHICH pairs attract, and how deeply relative to one another, into that table --
 * including the restored apolar-apolar (tail-tail) term the earlier task had removed. This is still
 * NOT a second physical constant alongside epsilon: epsilon sets the absolute well depth (energy
 * units) and the reference pair's own multiplier is exactly 1, so tail-tail keeps the rank-A depth;
 * the table only answers the question Cooke & Deserno's solvent-free calibration never had an
 * opinion on ("how deep are the water pairs, and the water-tail pair, RELATIVE to tail-tail").
 * Optional on Solvent so a pre-existing fixture/test that never mentions it keeps working: absent
 * reads as the pre-task water-only behaviour (soup/src/soup-attraction.ts's fallback branch).
 */
export interface SolventAttractionScale {
  epsilonScale: number
  /** Task 'hydrophobic-asymmetry' (2026-08-19): per-CLASS attraction DEPTHS, in MARTINI's own
   * kJ/mol, used ONLY as ratios to `reference` (see data/soup.json's own basis for the file the
   * numbers were read out of, the class->MARTINI-type mapping, and why the absolute scale stays
   * tied to data/params.json's rank-A epsilon instead of importing kJ/mol). Optional: a fixture
   * written before this task keeps the old boolean water-only behaviour -- soup/src/
   * soup-attraction.ts's own fallback branch. */
  pairEpsilon?: PairEpsilon
  /** Task 'acid-soap-pairing' (2026-08-23): the CHARGE-ASSISTED head-head pair depth -- the only
   * attraction here that is not a function of the two species alone: it fires only between a
   * PROTONATED and a DEPROTONATED head, never between two of the same state, so it cannot be a cell
   * of `pairEpsilon` (a polarPolar cell would attract acid-acid and soap-soap too, which the atomic
   * measurement says does NOT happen in water). Same units and `reference` normalisation as the
   * levels; absent/0 keeps every pre-task run bit-identical. Full basis: data/soup.json's own. */
  acidSoapPair?: AcidSoapPair
  basis: string
}

/** Task 'acid-soap-pairing' (2026-08-23): the measured acid-carboxylate association, mapped onto this
 * model's attraction depth. `epsilonKJ` is what the engine USES; the two `measured*` fields record the
 * atomic number, which is an ENERGY and not a free energy (entropy of pairing UNCOMPUTED) and is
 * therefore an UPPER BOUND -- data/soup.json's own basis has the sweep and the ranks. */
export interface AcidSoapPair {
  epsilonKJ: number
  measuredUpperBoundKJ: number
  measuredKcalPerMol: number
  rank: 'A' | 'B' | 'C' | 'D'
  basis: string
}

/** One cell of solvent.attractionScale.pairEpsilon.levels: the literature depth plus the exact
 * parameter-file line it was read from (`martini` is null for a pair deliberately kept at zero by
 * the rank-A Cooke & Deserno structure, not taken from MARTINI at all). */
export interface PairEpsilonLevel {
  epsilonKJ: number
  martini: string | null
}

export interface PairEpsilon {
  /** Key into `levels` whose depth every other depth is divided by -- its own multiplier therefore
   * comes out at exactly epsilonScale. */
  reference: string
  units: string
  levels: Record<string, PairEpsilonLevel>
}

/**
 * Task 'hydrophobic-asymmetry' (2026-08-19), defect 2: the zero-tension Monte Carlo area move for
 * the soup path (soup/src/soup-area-move.ts), so area-per-lipid is MEASURED rather than assumed
 * from a fixed box. `logDelta` is the proposal half-width in ln(area) -- a Markov-chain step-size
 * knob with no effect on the equilibrium distribution, living in data/soup.json only because
 * tests/params.test.ts forbids numeric constants inside soup/src. `mode` selects which box
 * deformation the move proposes, and with explicit solvent that choice is forced rather than
 * cosmetic: see data/soup.json's areaMove.basis for the measurement-level reason
 * ('lateral-fixed-volume' realises gamma=0, 'lateral-fixed-z' would instead realise "solvent
 * absolute lateral pressure = 0").
 */
export interface AreaMove {
  logDelta: number
  mode: 'lateral-fixed-volume' | 'lateral-fixed-z'
  basis: string
}

export interface Solvent {
  /** data/soup.json monomer id of the solvent species -- read by callers that need to find water
   * particles (e.g. soup/src/water-closure.ts) without hardcoding the id "W". */
  waterId: string
  attractionRule: string
  basis: string
  attractionScale?: SolventAttractionScale
}

/**
 * Task 'electrostatics' (2026-08-20): SCREENED ELECTROSTATICS plus a PROTONATION EQUILIBRIUM on the
 * titratable head. This is the section `saltPhLimitation` was deliberately left as an explicit field
 * for -- see data/soup.json's own `electrostatics.basis` for every literature citation, every rank,
 * and the list of what is still NOT represented, and soup/src/electrostatics.ts's header for the
 * scheme and what each choice biases. Optional so every pre-task fixture/test still loads: absent, or
 * `enabled: false`, makes the interaction coefficient exactly 0, which makes both the GPU term
 * (soup/wgsl/electrostatics.wgsl) and the CPU twin identically zero -- so a run without it is
 * bit-identical to every predecessor's.
 */
export interface Electrostatics {
  /** The file's own default (false). Overridden per system by CreateSoupOpts.electrostatics -- pH,
   * ionic strength and this switch are EXPERIMENT-design parameters (like --steps), not model
   * constants, which is why all three are overridable and none of them is baked into a run. */
  enabled: boolean
  /** Monomer id of the titratable species -- read rather than hardcoding "O", mirroring
   * solvent.waterId / clay.mineralId. EVERY bead of it titrates, bonded or free (basis item 10). */
  chargedKind: string
  /** Charge of the DEPROTONATED state in units of e. -1: a carboxylate. */
  chargeDeprotonated: number
  /** How many nanometres one sigma stands for. Rank C and a RANGE, not a calibration this project
   * has: `sigmaToNmRange` is carried through every derived quantity and every reported result,
   * because both the coupling A = kT*l_B/sigma and the screening kappa = sigma/lambda_D scale with
   * it (basis item 2). */
  sigmaToNm: number
  sigmaToNmRange: [number, number]
  sigmaToNmRank: 'A' | 'B' | 'C' | 'D'
  /** Bjerrum length of water at 298 K, nm (rank B literature). */
  bjerrumLengthNm: number
  bjerrumLengthRank: 'A' | 'B' | 'C' | 'D'
  /** Debye length of a 1:1 electrolyte at 1 M and 298 K, nm -- lambda_D = this / sqrt(I[M]) (rank B). */
  debyeLengthNmAtUnitMolar: number
  debyeLengthRank: 'A' | 'B' | 'C' | 'D'
  /** Ionic strength in mol/l. THE salt parameter: it is what makes 10 mM and 100 mM two different
   * interactions rather than two labels. Overridable per run. */
  ionicStrengthMolar: number
  ionicStrengthRank: 'A' | 'B' | 'C' | 'D'
  /** pKa of the MONOMER acid (rank B literature), deliberately NOT the interfacial apparent pKa: the
   * shift at a crowded charged interface must EMERGE from the electrostatic work, not be inserted. */
  pKaIntrinsic: number
  pKaIntrinsicRank: 'A' | 'B' | 'C' | 'D'
  /** The run's pH. Overridable per run; this is the swept variable. */
  pH: number
  /** Integration steps between constant-pH Monte Carlo sweeps (rank D, engineering -- it is the
   * sync point step() already takes, so a sweep costs no extra pipeline stall). */
  sweepEverySteps: number
  sweepEveryStepsRank: 'A' | 'B' | 'C' | 'D'
  /** Task 'long-range-electrostatics' (2026-08-20): how many DEBYE LENGTHS the screened-Coulomb
   * cutoff spans, i.e. rc_es = this * lambda_D. A Coulomb term cannot share a Lennard-Jones cutoff
   * (the first version of this section did, at 2.7224620 sigma, and at 10 mM that discarded 84 % of
   * the integrated interaction and produced a salt shift of the apparent pKa of -0.124 against the
   * literature's ~0.7). Expressed as a MULTIPLE of the screening length rather than as a length so
   * the truncated fraction -- exp(-x)(1+x) with x = rc/lambda_D -- is IDENTICAL in every ionic
   * strength, which is what makes a salt comparison unbiased by truncation. See the file's own
   * basis, item 12, for why 4 and why not Ewald/PPPM or Wolf. */
  longRangeDebyeLengths: number
  longRangeDebyeLengthsRank: 'A' | 'B' | 'C' | 'D'
  /** Minimum-image ceiling on rc_es as a fraction of the SMALLEST box side the run will visit (rank
   * D, engineering: below 0.5 with a margin). When 4*lambda_D exceeds it, the cutoff is truncated by
   * it and the report must state how many Debye lengths it then spans. */
  longRangeMaxBoxFraction: number
  longRangeMaxBoxFractionRank: 'A' | 'B' | 'C' | 'D'
  /** Per-head capacity of the DEDICATED long-range neighbour list (rank D -- the same number
   * verletList.listCapacity uses). Overflow is a loud throw, never a silent truncation. */
  longRangeListCapacity: number
  longRangeListCapacityRank: 'A' | 'B' | 'C' | 'D'
  /** Safety factor on the DERIVED per-head capacity (rank D). The capacity is not a typed number: it
   * is longRangeListSafetyFactor * (4pi/3)*listRange^3 * (heads / min(box)^3), bounded above by the
   * head count, i.e. the uniform-density occupancy of the TIGHTEST box the run visits times a margin
   * for the fact that heads sit on an aggregate's surface rather than uniformly. `longRangeListCapacity`
   * above is only the fallback for a caller that does not know the head count. This exists because
   * taking 2500 (verletList's own capacity) and checking it against the WET box overflowed on the
   * first dry step of the campaign -- see the file's basis, item 12. */
  longRangeListSafetyFactor: number
  longRangeListSafetyFactorRank: 'A' | 'B' | 'C' | 'D'
  rank: 'A' | 'B' | 'C' | 'D'
  basis: string
}

/**
 * Task 'broth-composition' (2026-08-18) declared ionic strength and pH UNREPRESENTABLE and kept
 * `represented` as an explicit field (not a comment) precisely so a future task that DID add
 * electrostatics would have one place to flip it with its own basis. Task 'electrostatics'
 * (2026-08-20) is that task and flipped it to `true` -- for the part that is genuinely represented
 * (charge on the head, Debye screening set by a real ionic strength, a protonation equilibrium whose
 * state changes during the run) and no more: data/soup.json's own `basis` now restates, item by item,
 * what stays OUT (no explicit H+, no titration of the medium, no explicit Na+/Cl-, no acid-soap
 * hydrogen bond, no electrostatics in the bond Metropolis, and an uncalibrated sigma->nm mapping
 * carried as a range). Read that field, not this comment, for the current boundary.
 */
export interface SaltPhLimitation {
  represented: boolean
  basis: string
}

/**
 * Task 'clay-surface' (2026-08-19): the immobile mineral PLATELET -- a rigid sheet of frozen beads
 * standing for a montmorillonite-like clay basal surface. See data/soup.json's `clay.basis` for the
 * literature, the ranks, and the explicit list of reported clay effects this cannot represent
 * (surface charge first among them -- `saltPhLimitation` already declares there is no electrostatics
 * in this engine at all).
 *
 * There is deliberately NO bead count and NO lattice spacing here: both are DERIVED (soup/src/
 * soup-clay.ts) from the live box and from the mineral bead's own WCA contact distance, i.e. from
 * data/params.json's rank-A sigma and the monomer's own radiusSigma, so the sheet is a dense,
 * impermeable, box-spanning plane at whatever box a caller asks for rather than a count that only
 * happens to tile one particular box.
 */
/**
 * ONE named mapping of the mineral class onto the depth table, i.e. one SURFACE CHEMISTRY the
 * platelet can be given. Task 'clay-surface-chemistry' (2026-08-19): the predecessor shipped exactly
 * one (the P5 hydrophilic mapping) and its own report's concern §1 was that the published CG clay
 * mapping DISAGREES with it -- Sposito et al. PNAS 1999 96:3358 has the uncharged siloxane basal
 * surface hydrophobic, and Khan & Goel JPCB 2019 123:9011 puts the basal beads at an apolar subtype
 * with the wetting carried by CHARGED beads this engine cannot represent. Neither mapping is the real
 * charged mineral, so both are kept SELECTABLE and the honest statement is the bracket between them.
 *
 * `pairs` maps each mineral pair key soup/src/soup-attraction.ts asks for (mineralSolvent /
 * mineralPolar / mineralApolar / mineralMineral) onto the name of a level in
 * solvent.attractionScale.pairEpsilon.levels. It is an ALIAS, never a number: a chemistry can only
 * point at a depth that already exists in the table with its own martini_v2.1.itp source line, so
 * adding a chemistry cannot introduce an uncited energy.
 */
export interface ClaySurfaceChemistry {
  pairs: Record<string, string>
  rank: 'A' | 'B' | 'C' | 'D'
  basis: string
}

export interface Clay {
  /** Whether the shipped composition carries the platelet. A caller overrides it per system with
   * CreateSoupOpts.clay (the same "convenience boolean, file's default when absent" pattern
   * dryWetCycle already uses) -- which is how every with/without-clay measurement is paired. */
  enabled: boolean
  /** data/soup.json monomer id of the mineral bead, read instead of hardcoding "K" (mirrors
   * solvent.waterId's own reasoning). Must be a monomer with kind="clay" and mineral=true. */
  mineralId: string
  /** How many parallel bead layers the platelet is. 1 = a single tetrahedral-octahedral-tetrahedral
   * (TOT) layer, which is what this task ships; see the basis for why the INTERLAYER (the quasi-2D
   * reaction environment between two stacked layers) is therefore NOT represented. */
  sheets: number
  /** Fraction of the broth's OWN catalyst pool that sits immobilised on the platelet as a surface
   * site, instead of diffusing freely. Not an addition: the sites are TAKEN from the same
   * `start` count, so a with-clay and a without-clay run have identical catalyst totals and the
   * comparison is not confounded by "more catalyst". Rank D -- a choice, see the basis. */
  siteCatalystFraction: number
  /** Which entry of `surfaceChemistries` the shipped platelet uses. Overridden per system with
   * CreateSoupOpts.claySurfaceChemistry, which is how the two limits are measured side by side. */
  surfaceChemistry: string
  /** Why there are two limits rather than one choice -- literature, ranks and the bracket. */
  surfaceChemistryBasis: string
  /** Every selectable surface chemistry, by name. Both bounds of the hydrophilicity bracket stay
   * here permanently: the mineral this stands for is CHARGED, and neither limit is it. */
  surfaceChemistries: Record<string, ClaySurfaceChemistry>
  rank: 'A' | 'B' | 'C' | 'D'
  basis: string
}

/** Task 'loud-failure-and-liquid-water' (2026-08-20): the cold-start energy-minimisation protocol.
 * NOT physics -- see this section's own `basis` in data/soup.json, and soup/src/soup-relax.ts's
 * header, for why it cannot appear inside any trajectory statistic. Optional so every pre-task
 * fixture still loads; absent means `relaxColdStart()` must be given both numbers explicitly or it
 * throws (it never falls back to a literal). */
export interface ColdStartRelax {
  /** How many displacement-capped steepest-descent iterations to run. The cap decays linearly to
   * zero across them, so this also fixes the total displacement bound. */
  iterations: number
  /** The FIRST iteration's displacement cap, in units of data/params.json's rank-A sigma. */
  maxDisplacementSigma: number
  basis: string
}

export interface Soup {
  /** The single explicit calibration of the model's time scale (kappa_t on screen in the reports). */
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
  /** Optional so a pre-'hydrophobic-asymmetry' fixture still loads; soup/src/soup-area-move.ts
   * throws a named error if a caller asks for an area move on a file that has no such section. */
  areaMove?: AreaMove
  saltPhLimitation?: SaltPhLimitation
  /** Optional so every pre-'electrostatics' fixture still loads; absent reads as "no electrostatics
   * at all", which is bit-identical to `enabled: false`. */
  electrostatics?: Electrostatics
  /** Optional so every pre-'clay-surface' fixture still loads; absent reads as "no mineral phase"
   * exactly like `clay.enabled: false`. */
  clay?: Clay
  checkpoint?: CheckpointDefaults
  coldStartRelax?: ColdStartRelax
}

const REQUIRED = [
  'kappaT', 'monomers', 'rules', 'start', 'sweep', 'neighborGrid', 'verletList', 'bondAttemptInterval',
  'headPlacement', 'adsorption', 'dryWetCycle', 'solvent',
] as const

export function loadSoup(): Soup {
  const s = raw as unknown as Soup
  for (const key of REQUIRED) {
    if ((s as unknown as Record<string, unknown>)[key] === undefined) {
      throw new Error(`data/soup.json: missing field ${key}`)
    }
  }
  return s
}

/**
 * The ratio of the backward rate to the forward rate under detailed balance for a bond
 * formation rule: exp(-bond energy in units of kT). With acceptanceProbability
 * at kT=1 (the base kappaT calibration) this is exactly the ratio that the
 * system gets from the pair acceptanceProbability(bond)/acceptanceProbability(break).
 */
export function forwardBackwardRatio(r: Rule): number {
  return Math.exp(-r.energyKT)
}

/** Probability of a reaction attempt per integrator step; dt comes from outside (engine/src/params) and is not hardcoded here. */
export function attemptProbability(r: Rule, dt: number): number {
  return r.attemptRate * dt
}

/**
 * Metropolis probability of accepting an attempt. Bond formation lowers the
 * energy by energyKT (ΔE = -energyKT) and is therefore always accepted;
 * a break is the same bond in the opposite direction (ΔE = +energyKT) and is accepted
 * with probability exp(-energyKT/kT). Both cases are one formula, so the
 * forward and backward directions automatically satisfy detailed balance
 * and cannot drift apart through separate constants.
 */
export function acceptanceProbability(r: Rule, kT: number): number {
  const deltaE = r.kind === 'bond' ? -r.energyKT : r.energyKT
  return Math.min(1, Math.exp(-deltaE / kT))
}

// --- schema validation -------------------------------------------------------------------------
// assertRulesConsistent lives in soup/src/rules-validate.ts since task 'clay-surface' (2026-08-19):
// this file had reached 598 lines against CLAUDE.md's hard 600 limit, and the rule is "split first,
// then add". Re-exported HERE, under its original name, so every existing importer
// (soup/src/sim.ts, tests/soup-rules.test.ts, soup/cli/campaign.ts, ...) is untouched by the move.
export { assertRulesConsistent } from './rules-validate'
