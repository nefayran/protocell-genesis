// Public API types for the soup engine (soup/src/sim.ts's createSoup/SoupSystem), split out of
// sim.ts itself (file-size rule in CLAUDE.md: sim.ts had grown to 2351 lines, four times the
// 400-600 target) so sim.ts can stay a thin composition point. Pure types plus the one shared
// sentinel constant every bond-slot/centerLink buffer uses -- no behaviour lives here, this is a
// straight move of sim.ts's own CreateSoupOpts/SoupSystem/NONE_U32, unchanged.

import type { Stage, StageEvidence } from './stages'
import type { RelaxColdStartResult } from './soup-relax'
import type { NonFiniteCount } from './soup-health'
import type { RehydrationReport } from './soup-evaporate'

export const NONE_U32 = 0xffffffff

export interface CreateSoupOpts {
  box: [number, number, number]
  seed: number
  kT: number
  /** Overrides data/soup.json's `start` counts by monomer id. Merged over the file's defaults, not
   * a full replacement -- an id not mentioned here keeps the file's count. */
  start?: Record<string, number>
  /** Convenience override for the catalyst monomer's count specifically (soup.json calls it `M`,
   * but this reads the schema's `kind: "catalyst"` monomer rather than assuming the id, so a
   * future rename of the id would not silently stop working). Takes precedence over `start` for
   * that one id. */
  catalystCount?: number
  /** Convenience override for data/soup.json's dryWetCycle.enabled, mirroring `catalystCount`'s own
   * pattern: `true` forces cycling on for THIS system regardless of the file's own default (the file
   * defaults to `false` precisely so no existing caller sees a behaviour change without asking for
   * it); `false`/`undefined` defers to the file. The schedule itself (cycles/periodSteps/dryFraction/
   * targetDryDensity/rampSteps/rampRelaxSteps) always comes from data/soup.json -- this override is a
   * plain boolean switch, never a number. */
  dryWetCycle?: boolean
  /** Task 'decisive-run' (2026-08-20): overrides data/soup.json's dryWetCycle.cycles for THIS system
   * only -- the schedule's cycle COUNT, nothing else (period, dry fraction, ramp and target density
   * all still come from the file). Exists because the evaporation task measured that the gain does
   * NOT compound: one drying event delivered 13x in aggregate size while cycles 2-6 delivered nothing
   * measurable and the yield declined monotonically, so the schedule that trend justifies is ONE
   * cycle -- and expressing that must not mean editing a data file per run. An experiment-design
   * number in exactly the sense --steps and --every already are; undefined keeps the file's own value
   * byte for byte. */
  dryWetCycles?: number
  /** Task 'evaporation' (2026-08-20): overrides data/soup.json's dryWetCycle.evaporateSolvent for
   * THIS system -- the same boolean-only convenience pattern `dryWetCycle` itself uses. `true` makes
   * the dry phase REMOVE solvent beads from the system (and rehydration put them back) instead of
   * merely compressing the box at fixed composition; `false`/`undefined` defers to the file, which
   * defaults to off so no existing caller changes behaviour. Requires `dryWetCycle` as well (solvent
   * removal happens at a wet<->dry transition; there is no other trigger for it). */
  evaporateSolvent?: boolean
  /** Water-calibration task (2026-08-19): overrides data/soup.json's `solvent.attractionScale.
   * epsilonScale` for THIS system only -- the same "convenience override, file's own default when
   * absent" pattern `catalystCount` already uses. Exists so the calibration sweep
   * (tests/tmp-water-calib-probe.test.ts, and any future re-sweep) does not need to rewrite
   * data/soup.json between candidate values; the file's own field is what a real run (and the
   * committed gates) actually uses. */
  solventAttractionScaleOverride?: number
  /** Task 'acid-soap-pairing' (2026-08-23): overrides data/soup.json's
   * solvent.attractionScale.acidSoapPair depth for THIS system, ALREADY NORMALISED (i.e. in the same
   * units every cell of the class table is in: 1.0 = the apolar-apolar reference depth). Exists
   * because the strength has to be SWEPT -- the atomic measurement it comes from is an upper bound,
   * not a calibration -- and a sweep must not mean rewriting a data file between arms. The file's own
   * value is what every real run uses. 0 disables the term outright, which is the arm that proves
   * every other number in this engine is unchanged by the task. */
  acidSoapScaleOverride?: number
  /** Task 'clay-surface' (2026-08-19): overrides data/soup.json's `clay.enabled` for THIS system --
   * the same convenience-boolean pattern `dryWetCycle` uses. `false` is how every clay-free control
   * arm is created, and every measurement of the platelet is a with/without pair at otherwise
   * identical composition and seed. The platelet's geometry (bead count, lattice spacing, plane
   * position, site count) is never a caller's choice: it is derived from the box, from
   * data/params.json's rank-A sigma and from data/soup.json's clay section. */
  clay?: boolean
  /** Task 'clay-surface-chemistry' (2026-08-19): which of data/soup.json's clay.surfaceChemistries
   * this system's platelet uses, overriding the file's clay.surfaceChemistry. The two limits of the
   * hydrophilicity bracket ("hydrophilic" = the P5 mapping, "apolar" = the published uncharged-
   * siloxane mapping) are measured by running otherwise identical arms that differ only in this
   * string. An unknown name THROWS (soup/src/soup-attraction.ts) rather than falling back. */
  claySurfaceChemistry?: string
  /** Overrides data/soup.json's clay.siteCatalystFraction for THIS system -- the knob the predecessor
   * report's concern §3 named as unswept, and the one the growth result is a function of. 0 means a
   * platelet with NO catalytic sites at all (the whole catalyst pool stays free in the broth), which
   * is the arm that separates "the surface did it" from "a quarter of the catalysts were immobilised". */
  claySiteCatalystFraction?: number
  /** THE CONTROL THE PREDECESSOR DID NOT RUN (its concern §3): immobilise this many catalyst beads at
   * their ordinary BULK positions, with NO platelet in the box. Every other draw -- positions,
   * velocities, RNG streams -- is untouched, so such a system is bit-identical to the clay-free arm
   * except in WHICH beads the integrator refuses to move. Ignored (throws) together with `clay`, since
   * the point is a surface-free control. */
  frozenBulkCatalysts?: number
  /** Task 'electrostatics' (2026-08-20): per-run overrides of data/soup.json's `electrostatics`
   * section. Only three things are overridable and all three are EXPERIMENT-design parameters, not
   * model constants: whether charge is on at all, the pH (the swept variable), and the ionic strength
   * (which sets the Debye length). Everything else -- the Bjerrum length, the intrinsic pKa, the
   * sigma->nm mapping and its range, the sweep cadence -- always comes from the file with its own
   * rank and basis. Absent defers to the file, which ships `enabled: false`, so every pre-task caller
   * is unchanged. */
  electrostatics?: {
    enabled?: boolean
    pH?: number
    ionicStrengthMolar?: number
    /** Task 'acid-soap-pairing' (2026-08-23): the volume the TITRATABLE beads actually occupy at the
     * tightest box, for sizing the long-range head list -- passed straight through to
     * EsOverrides.densityVolumeSigma3 (task 'confined-parcel' added the field for the parcel case and
     * derives it automatically there; a confined run therefore ignores this). Needed because a
     * PREBUILT BILAYER PATCH concentrates every head into a slab a few sigma thick inside a box tens
     * of sigma deep, so maxHeads/min(box)^3 under-states the head density several-fold and the list
     * comes out too small -- which this engine catches LOUDLY (assertVerletSafety throws by name)
     * rather than silently dropping interaction, and this is the field that answers it honestly
     * instead of inflating a safety factor. Absent = min(box)^3, i.e. every pre-task caller. */
    densityVolumeSigma3?: number
  }
  /** Task 'big-box' (2026-08-20): per-run override of data/soup.json's `verletList` -- ONLY the two
   * fields that are experiment design rather than model constants. `enabled: false` selects
   * cell-list traversal (the grid walk every step, no per-particle neighbour array at all, O(N)
   * memory) instead of the flat N*listCapacity list; `listCapacity` pins the per-particle capacity
   * instead of letting soup/src/soup-plan.ts's deriveListCapacity derive it from this system's own
   * densest box. Both exist so the structure/capacity A/B can be measured without rewriting a data
   * file between arms -- the file's own values are what every real run uses. Absent = the file. */
  verletOverride?: { enabled?: boolean; listCapacity?: number }
  /** Task 'confined-parcel' (2026-08-21): confine the system to a finite sphere of water with a soft,
   * neutral, repulsive wall instead of running it in a fully periodic box -- see
   * soup/src/soup-confine.ts for the whole design, including how periodicity is removed (by making
   * every mi3/wrap provably the identity, not by branching eleven force paths) and what the wall's
   * neutrality does and does not bias. Absent = the fully periodic box every published gate and every
   * previous campaign was measured in, with ZERO extra dispatches, so those numbers cannot move. */
  confine?: { radiusSigma: number; stiffness: number }
  /** Checkpoint/resume (task 'checkpoint-resume'): when present, createSoup skips the jittered-
   * lattice initial layout and every zero-filled buffer below, loading this system's ENTIRE mutable
   * state from a prior checkpoint instead -- everything step()/stepCycled() can change: positions,
   * velocities, the bond-slot graph, the catalyst<->chain adsorption links and their hold counters,
   * cumulative desorption/rule-event counts, the two per-particle GPU RNG streams (bond
   * Monte Carlo, Langevin thermostat noise), the real step counter, and the live box (which can
   * differ from `box` above when a dry-wet cycle was mid-transition at checkpoint time -- `box`
   * stays this call's WET/creation box, exactly as it always has, so cycleCfg/dryBox re-derive
   * identically to the original run; only the mutable liveBox/cyclePhase/cycleIndex the getters
   * report are seeded from the checkpoint instead of from step 0).
   *
   * What is deliberately NOT here, and why it does not need to be: the coarse neighbour grid
   * (countsBuf/cellStartBuf/cursorBuf), the Verlet list itself (verletListBuf/verletCountBuf), the
   * drift-safety snapshot/counter (posAtRebuildBuf/maxDriftSqBuf) and the overflow flag
   * (verletOverflowBuf) are never part of a checkpoint -- every one of them is a pure function of
   * the CURRENT positions (soup/src/checkpoint.ts's own header spells this out with the exact
   * mechanism: they are rebuilt from `positions` below by this same function's own priming compute
   * pass, the identical rebuild applyBoxScaleOnce already forces after every dry-wet box change for
   * the same reason). Restoring them from a stale reading would be no more correct than rebuilding
   * fresh, and rebuilding fresh needs no new state at all. */
  resume?: {
    globalStep: number
    liveBox: [number, number, number]
    /** Task 'evaporation' (2026-08-20): the LIVE per-monomer census at checkpoint time, which can
     * differ from `start` above because a checkpoint taken mid dry-phase carries FEWER solvent beads
     * than the composition the run was created with. When present it is what N is taken from (so the
     * arrays below match), while `start` stays the creation composition -- the wet/dry box pair and
     * the whole cycle schedule are derived from THAT, so a resumed evaporating run reproduces the
     * same ladder the original did. Absent (every pre-task checkpoint) reads as "the census equals
     * `start`", i.e. exactly the previous behaviour. */
    activeCounts?: Record<string, number>
    positions: Float32Array
    velocities: Float32Array
    bondSlots: Uint32Array
    centerLink: Uint32Array
    centerHeldSteps: Uint32Array
    desorbEvents: Uint32Array
    bondRng: Uint32Array
    thermoRng: Uint32Array
    /** Keyed by data/soup.json rule id, matching SoupSystem.events()'s own return shape -- resolved
     * to the numeric per-rule uniform layout by rule id lookup (not by array position), so this
     * still lines up correctly even if data/soup.json's rule ORDER ever changes between the
     * checkpointed run and the resuming one. */
    events: Record<string, number>
    /** Task 'electrostatics' (2026-08-20): the per-particle charge array, i.e. the PROTONATION STATE.
     * As much a part of the mutable state as the bond graph is -- a resume that dropped it would
     * silently re-draw every head's protonation from the Henderson-Hasselbalch prior and throw away
     * whatever the interface had equilibrated to. Optional so every pre-task checkpoint still loads
     * (absent = draw fresh, which is what those runs did anyway since they had no charge at all). */
    charges?: Float32Array
    /** The constant-pH Monte Carlo's own RNG state, so a resumed run continues the same chain rather
     * than restarting it at the seed. */
    protonationRng?: number
  }
}

export interface SoupSystem {
  /** Advances n Langevin + bond-Monte-Carlo steps, encoded as one command buffer. */
  step(n: number): Promise<void>
  /** Advances n real steps exactly like step(), EXCEPT that when this system was created with dry-
   * wet cycling enabled (CreateSoupOpts.dryWetCycle / data/soup.json's dryWetCycle.enabled), a wet<->
   * dry box transition landing inside this call is applied at the right global step (via
   * applyBoxScale, not step() itself -- step() is untouched by this task, so every existing caller's
   * behaviour is unchanged byte for byte) before continuing. When cycling is NOT enabled this is
   * `step(n)` exactly, nothing else -- soup/src/sim.ts's runUntil() below always calls this one
   * (never step() directly) so cycling "just works" through the same trace-producing loop every
   * other run already uses, without a second stepping path to keep in sync. Because a transition
   * ramp injects its own `rampRelaxSteps` between increments (data/soup.json's dryWetCycle.basis),
   * the number of real steps actually taken by one call CAN exceed `n` when a transition falls
   * inside it -- callers that need an exact step count should check `steps` before/after, not assume
   * n was taken verbatim. */
  stepCycled(n: number): Promise<void>
  /** One-off ramped box change to an arbitrary target box, for a caller outside the dry-wet cycle
   * schedule (e.g. expanding a resumed checkpoint's box so its aggregate is small compared with the
   * box before measuring shape -- see soup/src/sim.ts's own doc comment on this function for why it
   * is a thin wrapper around the SAME applyBoxScale ramp/rigid-COM-scale/grid-rebuild mechanism
   * stepCycled already uses, not a new code path). Throws if any rigid-unit-defining distance
   * changes during any increment (applyBoxScaleOnce's own runtime self-check, covalent bonds plus
   * cohesion's own proximity edges when given) -- a caller does not need to re-verify that
   * separately, only confirm the call did not throw. `cohesion`, when given, makes every proximity-
   * connected pair within `cohesion.cutoff` of each other (by default among ALL particles;
   * `cohesion.memberIndices` restricts eligibility to a caller-named subset) scale as ONE rigid body
   * instead of as N independently-scaled covalent molecules, RECOMPUTED FRESH on every ramp
   * increment (never a graph reused from an earlier box size) -- see applyBoxScaleOnce's own doc
   * comment (soup/src/soup-box-scale.ts) for the three configurations actually measured against this
   * checkpoint (no cohesion; a STALE global graph; a member-restricted graph) and why only the
   * fresh-every-increment, unrestricted default measured as truly safe (no degradation at all from
   * the unperturbed baseline, chained across the full 58->150 sigma ramp in pure Node). */
  scaleBoxTo(
    targetBox: [number, number, number],
    rampSteps: number,
    rampRelaxSteps: number,
    cohesion?: { cutoff: number; memberIndices?: Uint32Array },
  ): Promise<void>
  /** Box-expansion task: grows liveBox to `targetBox` WITHOUT moving any particle (only the box and
   * the neighbour grid change) -- the safe alternative to scaleBoxTo when the live aggregate's own
   * size is comparable to or exceeds the current box, so scaleBoxTo's rigid-unit BFS construction
   * would pick up a genuine topological wraparound regardless of cohesion grouping -- see
   * soup/src/soup-box-scale.ts's own doc comment on this function for the two measured failures
   * (unrestricted and member-restricted cohesion, both against this exact scenario) that motivated
   * it. */
  growBoxTo(targetBox: [number, number, number], rampSteps: number, rampRelaxSteps: number): Promise<void>
  /** 4 floats per particle: x, y, z, kind index (position into data/soup.json's `monomers`). */
  particles(): Promise<Float32Array>
  /** Checkpoint/resume (task 'checkpoint-resume'): 4 floats per particle, x/y/z/w velocity --
   * mirrors particles()'s own layout and role, the other half of the Langevin state a checkpoint
   * needs to continue the SAME trajectory rather than one that restarts every particle at rest. */
  velocities(): Promise<Float32Array>
  /** Checkpoint/resume (task 'checkpoint-resume'): the raw per-particle bond-slot rows this system's
   * bondSlotsBuf holds -- 3 u32 slots per particle (NONE_U32 where unused), the exact bookkeeping
   * bonds() derives its (i,j) pair list FROM. bonds()'s own pairs are not enough to resume from:
   * which of a particle's 3 rows a given partner sits in is what soup/wgsl/bond-valence.wgsl's
   * roleOf() checks against slot-role/valence limits on every future bond attempt, and re-deriving
   * an arbitrary (if physically equivalent) slot assignment from an unordered pair list is a
   * needless risk when the exact row is already sitting in a buffer one readback away. */
  bondSlots(): Promise<Uint32Array>
  /** Checkpoint/resume (task 'checkpoint-resume'): the raw per-particle hold-timeout clock
   * (soup/wgsl/bond-adsorption.wgsl's centerHeldSteps) -- the adsorption.basis timeout desorption
   * valve counts against this, so a resumed run that zeroed it would give every held centre a free
   * extra maxHoldSteps of grace it never had. */
  centerHeldSteps(): Promise<Uint32Array>
  /** Task 'clay-surface' (2026-08-19): the per-particle immobility flag as the GPU holds it -- 1 for
   * a bead of the rigid mineral platelet, 0 otherwise. All zeros when this system has no platelet. */
  frozen(): Promise<Uint32Array>
  /** Task 'big-box' (2026-08-20): the measured per-particle neighbour-count distribution of the main
   * Verlet list and of the long-range electrostatic head list, plus the uniform-density expectation
   * both derived capacities are a multiple of. `main` is null on a cell-list system (there is no
   * per-particle list to measure); `es` is null without charge. Reduced to scalars inside the page. */
  listOccupancyDEBUG(): Promise<Record<string, unknown>>
  /** The Verlet configuration this system actually runs with, including the DERIVED listCapacity and
   * the byte size of the list it implies -- synchronous, reads no GPU buffer. */
  verletConfig(): Record<string, unknown>
  /** Task 'electrostatics' (2026-08-20): per-particle charge in units of e -- 0 for every bead except
   * a deprotonated head. Part of the checkpoint; read by every off-GPU measurement of the
   * deprotonated fraction, the apparent pKa and acid-soap pairing. */
  charges(): Promise<Float32Array>
  /** The resolved electrostatics basis plus how many constant-pH sweeps have run and what the last
   * one did. Synchronous: it reads no GPU buffer. */
  electrostatics(): Record<string, unknown>
  /** Task 'confined-parcel' (2026-08-21): the resolved confinement plus everything derived from it --
   * the live radius, the parcel volume, the clearances the no-wrap argument rests on. `null` on a
   * fully periodic system, which is what every measurement path branches on. */
  confinement(): Record<string, unknown> | null
  /** Forces ONE constant-pH sweep now, regardless of the schedule -- for tests and for the sweep
   * probe. Returns null on a system without electrostatics. */
  protonationSweepDEBUG(): Promise<unknown>
  protonationRngState(): number
  /** z of each of the platelet's sheet planes, empty when this system has no platelet -- what a
   * distance-to-surface profile is binned against. */
  clayPlanes(): number[]
  /** Per-particle force from the grid path (rebuilds the grid for current positions first).
   * perf2-report.md correctness gate: compared against forcesBruteForce() to floating-point
   * tolerance, mirroring engine/src/sim.ts's own forces()/forcesBruteForce() pair. */
  forces(): Promise<Float32Array>
  /** Same physics as forces(), computed by an O(N^2) pair loop with no neighbour grid at all --
   * the reference implementation forces() is checked against. */
  forcesBruteForce(): Promise<Float32Array>
  /** Task 'hydrophobic-asymmetry' (2026-08-19), defect 2: runs `trials` zero-tension Metropolis
   * area moves, so area-per-lipid becomes a MEASUREMENT instead of the fixed box's assumption. The
   * move's own mechanism, the two modes and why the explicit-solvent gate needs the volume-preserving
   * one are documented in soup/src/soup-area-move.ts and data/soup.json's areaMove.basis. The
   * returned `lateralTrajectory` is one L_x reading per trial, so a caller can SHOW that the box
   * moved and settled. `opts.mode` overrides the file's mode for one call (the identity-verification
   * test needs the fixed-L_z branch, where the entropic term is non-zero and therefore testable). */
  areaMove(trials: number, opts?: { mode?: 'lateral-fixed-volume' | 'lateral-fixed-z' }): Promise<{
    trials: number
    accepted: number
    acceptedFraction: number
    box: [number, number, number]
    lateralTrajectory: number[]
    energyStart: number
    energyEnd: number
  }>
  /** Pairs of particle indices [i0, j0, i1, j1, ...], one entry per currently active bond. */
  bonds(): Promise<Uint32Array>
  /** Surface growth (surface-growth-report.md): the mutual catalyst<->tip association buffer,
   * `soup/wgsl/bond-adsorption.wgsl`'s `centerLink` -- one entry per particle, `0xFFFFFFFF`
   * (unassociated) or the id of the particle it is currently linked to (a catalyst's currently-held
   * chain-tip carbon, or a carbon's currently-owning catalyst). Debug/verification readback only,
   * mirroring `bonds()`'s own role for `bondSlots` -- never read by the real step loop. */
  centerLinks(): Promise<Uint32Array>
  /** Surface growth / adsorption (adsorption-report.md): cumulative desorption event counts since
   * creation -- `stretch` is soup/wgsl/bond-adsorption.wgsl's desorbStretch (the adsorption bond
   * exceeded FENE's own bonded range, P.r_inf), `timeout` is desorbTimeout (a centre held the same
   * chain for more than data/soup.json's adsorption.maxHoldSteps real steps without a propagation/
   * termination event). Neither overlaps `events()`'s own per-rule counts: a desorption is never a
   * completed amphiphile. */
  desorbEvents(): Promise<{ stretch: number; timeout: number }>
  /** Checkpoint/resume (task 'checkpoint-resume'): the two per-particle GPU RNG streams -- bond
   * Monte Carlo (soup/wgsl/bond-common.wgsl's bondRng) and Langevin thermostat noise
   * (soup/wgsl/step.wgsl's thermoRng) -- read back EXACTLY, one u32 state word per particle, the
   * same buffer bond_form_main/bond_break_main/kick_thermostat_main themselves read and rewrite
   * every dispatch. Restoring these is what lets a resumed run's random draws continue the SAME
   * stream a crash cut off, rather than starting a fresh stream from the creation seed -- the one
   * piece of state this engine's own RNG story does NOT have to report as "differs on resume" (see
   * soup/src/checkpoint.ts's header for what, if anything, still does). */
  rngState(): Promise<{ bond: Uint32Array; thermo: Uint32Array }>
  /** Cumulative event counts since creation, keyed by data/soup.json rule id (e.g. "cc_bond"). */
  events(): Promise<Record<string, number>>
  /** Per-monomer-id particle counts, active bond count, and total charge (sum of each monomer's
   * `charge` field if data/soup.json ever defines one; the current schema does not, so this is 0
   * by construction -- not a hardcoded placeholder, a schema-driven sum that happens to be empty
   * today). Compared before/after step() by tests/soup-bonds.test.ts: particle counts and charge
   * must be EXACTLY unchanged (nothing here ever creates or destroys a particle), bonds must be
   * able to change (that is the whole point of this task). */
  invariants(): Promise<{ monomers: Record<string, number>; bonds: number; charge: number }>
  /** Task 'loud-failure-and-liquid-water' (2026-08-20): how many position/velocity COMPONENTS of the
   * live state are Inf or NaN, measured by the same O(N) GPU scan step() now runs once per
   * 1000-step chunk (soup/wgsl/health.wgsl). Published so a caller can pay for the check explicitly
   * -- e.g. to time it, or to read the count without provoking the throw that step() would. A
   * healthy system returns {pos: 0, vel: 0}. */
  nonFiniteCount(): Promise<NonFiniteCount>
  /** Task 'loud-failure-and-liquid-water' (2026-08-20): displacement-capped steepest-descent energy
   * minimisation of the INITIAL coordinates -- the standard MD cure for a cold start whose lattice
   * puts unlike-radius pairs inside each other's repulsive cores, which is what has blown up every
   * attempt at liquid-density water in this project.
   *
   * OPT-IN BY DESIGN, and that is the measurement-neutrality argument: nothing in createSoup calls
   * this, so every existing caller, test, gate and checkpoint lineage is bit-identical to before the
   * stage existed. THROWS if the system has already taken a step, so it cannot be inside a
   * trajectory. Reads its two numbers from data/soup.json's `coldStartRelax` unless the caller
   * overrides them explicitly (a sweep does; a real run does not). See soup/src/soup-relax.ts and
   * soup/wgsl/relax.wgsl for what it does and does not touch. */
  relaxColdStart(opts?: { iterations?: number; maxDisplacementSigma?: number }): Promise<RelaxColdStartResult>
  /** Task 'evaporation' (2026-08-20): removes/returns solvent beads AT THE CURRENT BOX, with no ramp
   * and no dynamics. Exists so the two operations can be proved in ISOLATION -- inside a real
   * wet<->dry transition they are inseparable from the box change and the ramp's relaxation steps, so
   * "removal did not touch the bond graph" and "rehydration did not move a single pre-existing
   * particle" would not be checkable claims. Never called by stepCycled, the CLI, the viewer or any
   * measurement path: only by tests/soup-evaporation.test.ts. Throws unless this system was created
   * with solvent evaporation. */
  evaporateDEBUG(targetSolvent: number): Promise<void>
  rehydrateDEBUG(targetSolvent: number): Promise<RehydrationReport>
  /** Task 'decisive-run' (2026-08-20): the force buffer AS IT STANDS -- no grid/list/force rebuild
   * first -- i.e. the very F(x_n) the next kick will consume. forces() recomputes before reading
   * back, which makes a stale forceBuf unobservable through it; this is what
   * tests/soup-stale-force.test.ts uses to pin applyBoxScaleOnce's force refresh. Encodes no
   * dispatch, changes no state. */
  forcesNoRebuildDEBUG(): Promise<Float32Array>
  /** Profiling hooks from soup/src/soup-readback.ts: milliseconds per step for each group of step
   * phases, each group run `n` times, and the force kernel's two candidate-pair counters.
   * Diagnostics only, never part of the step loop. */
  stepPhasesDEBUG(n: number): Promise<Record<string, number>>
  forceCandidateStatsDEBUG(): Promise<{ candidatesExamined: number; pairsWithinRange: number; ratio: number }>
  /** Task 'decisive-run' (2026-08-20): the SAME minimiser the cold start, the solvent insertion and
   * the evaporating ramp's guard use (soup/src/soup-relax.ts's relaxIterations), applied MID-RUN at
   * the current box with no box change and no solvent movement. Exists for ONE measurement: the
   * minimisation-only control arm -- a run that receives exactly the minimisations the cycled arm
   * received, at the same global steps, with no evaporation cycle, so the share of the cycling
   * benefit attributable to minimisation alone can be measured instead of argued about. Deliberately
   * does NOT carry relaxColdStart's "globalStep must be 0" refusal (being inside the trajectory is
   * the whole point here), which is exactly why it is a separately-named DEBUG hook and not that
   * entry point. Reachable only from soup/cli/campaign.ts's --minimiseAt and from
   * tests/soup-stale-force.test.ts. */
  minimiseNowDEBUG(iterations: number): Promise<{
    iterations: number
    globalStep: number
    maxForceBefore: number
    maxForceAfter: number
    displacementBound: number
  }>
  /** The system's CURRENT box -- a live snapshot, mirroring engine/src/sim.ts's own `System.box`
   * getter (its doc comment: "a live snapshot, since areaMove() mutates L_x, L_y in place"). Equal
   * to `CreateSoupOpts.box` for the system's whole lifetime UNLESS dry-wet cycling is enabled, in
   * which case applyBoxScale (driven by stepCycled()) mutates it between the system's own wet box
   * and its derived dry box (computeDryBox). Task 4's reconciliation of the deviation Task 3
   * flagged: soup/src/stages.ts's detectStage reads this instead of taking box as a second
   * parameter, so `stageOf(sys)` (and this file's own runUntil, below) never has to re-thread it --
   * and, since detectStage reads it live, a sample taken mid-cycle correctly measures against
   * whichever box that sample's snapshot actually sits in. */
  readonly box: [number, number, number]
  /** 'wet' | 'dry' | 'none' -- 'none' when dry-wet cycling is not enabled for this system (the
   * default), otherwise which segment of its current cycle the box is presently at (or was last set
   * to; a ramp's own intermediate increments do not change this label, only which BOUNDARY they are
   * walking toward does -- see data/soup.json's dryWetCycle.basis on why the transition itself is
   * spread over several small steps). Read live, like `box` above. */
  readonly cyclePhase: 'wet' | 'dry' | 'none'
  /** 1-based index of the cycle currently in progress, or 0 when cycling is not enabled OR every
   * configured cycle has already completed (settled back to wet -- see cyclePhaseAt's own doc
   * comment for why 0 specifically means "over", not "cycle zero"). */
  readonly cycleIndex: number
  /** Cumulative real integration steps taken so far via step()/stepCycled(), across all calls --
   * mirrors engine/src/sim.ts's own `System.steps`. Needed by a stepCycled() caller that wants to
   * know exactly how many real steps a call actually took, since a call spanning a wet<->dry
   * transition takes MORE than the `n` it was asked for (the ramp's own relax steps) -- see
   * stepCycled's own doc comment. */
  readonly steps: number
  /** Steps in batches of `sampleEvery`, running detectStage after each batch and appending
   * `{steps, stage, evidence}` to a trace (also logged one line at a time, via console.log, so a
   * long run is diagnosable while it is still in progress). Stops early the first batch whose
   * stage equals the target `stage`; otherwise runs until `maxSteps` is exhausted. `steps` in the
   * trace and in the return value is the CUMULATIVE step count taken by this call (independent of
   * whatever `globalStep` step() itself has already advanced from earlier calls) -- it is what a
   * caller graphs a trajectory against, not a raw step()-internal counter. */
  runUntil(
    stage: Stage,
    opts: { maxSteps: number; sampleEvery: number },
  ): Promise<{
    reached: boolean
    steps: number
    trace: { steps: number; stage: Stage; evidence: StageEvidence; cyclePhase: 'wet' | 'dry' | 'none'; cycleIndex: number }[]
  }>
  /** Destroys every GPUBuffer this system owns. getGpu() memoizes ONE device for the whole page,
   * so a caller that creates a second SoupSystem in the same page (viewer/run.ts's "start a new
   * run" button, tests/run-ui.test.ts's second-run regression) leaves the FIRST system's buffers
   * (positions, velocities, bond slots, Verlet lists, every uniform -- ~20 GPUBuffers, see
   * soup/src/soup-buffers.ts's own allocations) alive on that shared device unless something
   * explicitly destroys them: unlike CPU memory, a GPUBuffer whose JS object becomes unreachable is
   * not promptly freed by garbage collection, so repeated runs without disposal accumulate GPU-side
   * allocations without bound. Pipelines (soup/src/soup-pipelines.ts's module-level `cached`, keyed
   * by device) are NOT touched here -- they hold no reference to any particular system's buffers
   * (bind groups, which do, are always recreated fresh per createSoup call) and are safe, and
   * intended, to be reused by the next system on the same device. Safe to call more than once
   * (GPUBuffer.destroy() is a no-op on an already-destroyed buffer per the WebGPU spec) and safe to
   * call on a system whose step() is not currently in flight; callers must not call
   * step()/particles()/etc. afterward. */
  dispose(): void
}
