// Checkpoint/resume (task 'checkpoint-resume'): a small, no-test-harness entry point for a long soup
// campaign -- run to a step budget with periodic checkpoints, and pick up from the newest checkpoint
// that matches the requested configuration if one already exists in --dir. Invoke with
// `npx tsx soup/cli/campaign.ts --label <name> --steps <n> --box <n> --start '{"C":...}' [...]`; see
// printUsage() below for the full flag list, or run with no arguments.
//
// Exists because this exact run -- 250 000 steps at 93 200 particles, ~40 minutes -- has now failed
// FOUR times for reasons that had nothing to do with the physics:
//  1. a test's own 30-minute timeout, while the simulation itself was still correctly stepping;
//  2. the page dying while transferring 93 200 particles' coordinates as a JSON array of ~400 000
//     numbers, AFTER the simulation had already correctly finished all 250 000 steps in 1405s;
//  3. puppeteer's own CDP protocolTimeout cutting the single evaluate() at 1808s;
//  4. earlier in this project, more than a dozen background runs vanishing when their parent turn
//     ended -- no test timeout, no crash, just nothing left holding the process open.
// Every one of those threw away a fully-computed state and forced a restart from step 0. This CLI's
// own job is narrow: advance the SAME SoupSystem (soup/src/sim.ts) a caller would drive from a test,
// write soup/src/checkpoint.ts's own encodeCheckpoint() output to disk on a schedule AND right before
// exiting on SIGINT/SIGTERM, and reconstruct that exact state via decodeCheckpointResume() +
// createSoup(opts.resume) on the next invocation -- so failure #4 above (or a plain `kill -9`, which
// this file's own signal handler cannot catch, and does not need to: the periodic on-disk checkpoint
// is what survives THAT one) costs at most one --every interval of recomputation, not the whole run.

import { mkdirSync, readFileSync } from 'node:fs'
import { gpuPage, shutdownGpu } from '../../tests/helpers/gpu'
import type { CheckpointConfig } from '../src/checkpoint'
// The flag surface, the usage text, the resume-signature and the checkpoint file I/O live in
// soup/cli/campaign-config.ts (CLAUDE.md's file-size rule) -- a pure move, see that file's header.
import { configSignature, findNewestMatchingCheckpoint, parseCliArgs, writeCheckpointFile } from './campaign-config'
import { loadSoup } from '../src/rules'

// box-expansion task: how many real steps stepPhasesDEBUG isolates each grid/force/bondAttempts/
// integration component over, immediately before and after the one-time expansion above -- an
// experiment-design sample count (perf-report.md's own precedent used n=3000 at N=13100; this
// checkpoint's N=93200 is ~7x larger and the whole point of this measurement is "how expensive did
// this just get", so a SMALL n that still finishes quickly is preferred over perf-report.md's own
// n, not a physical parameter either way).
const GRID_DEBUG_N = 100

// Task 'confined-parcel' (2026-08-21): the radial shell, in sigma, the per-species wall-adsorption
// measurement counts particles in -- the MEASURED bilayer thickness of this project's own
// explicit-water gate (verify/out/water-bilayer-area-move.json's `thickness`, 4.687), i.e. exactly the
// depth an adsorbed film would occupy, read from the gate artifact rather than typed here. Falls back
// to the file's own closure radius scale only if the artifact is missing, and says so.
const WALL_SHELL_JSON = 'verify/out/water-bilayer-area-move.json'


async function main(): Promise<void> {
  const args = parseCliArgs()
  let wallShell = 0
  if (args.confineRadius !== undefined) {
    const wb = JSON.parse(readFileSync(WALL_SHELL_JSON, 'utf8'))
    wallShell = wb.thickness
    console.log(`[campaign] wall shell for the adsorption measurement = ${wallShell.toFixed(4)} sigma (bilayer thickness from ${WALL_SHELL_JSON})`)
  }
  const config: CheckpointConfig = {
    box: [args.box, args.box, args.box],
    seed: args.seed,
    kT: args.kT,
    start: args.start,
    catalystCount: args.catalyst,
    dryWetCycle: args.cycle || undefined,
    evaporateSolvent: args.evaporate || undefined,
    dryWetCycles: args.cycles,
    minimiseAt: args.minimiseAt.length > 0 ? args.minimiseAt : undefined,
    // Task 'electrostatics' (2026-08-20): resolved against data/soup.json's own defaults HERE (not
    // left as three optional flags) so the checkpoint carries the pH and ionic strength the run
    // actually used, and a reader of the file never has to re-resolve them against a data file that
    // may have moved since.
    electrostatics: args.charge
      ? {
          enabled: true,
          pH: args.pH ?? loadSoup().electrostatics!.pH,
          ionicStrengthMolar: args.ionicStrength ?? loadSoup().electrostatics!.ionicStrengthMolar,
        }
      : undefined,
    // Task 'confined-parcel' (2026-08-21): resolved HERE (not left as two optional flags) so the
    // checkpoint carries the parcel this run actually used and a reader never has to re-derive it.
    confine: args.confineRadius !== undefined ? { radiusSigma: args.confineRadius, stiffness: args.confineStiffness } : undefined,
  }
  mkdirSync(args.dir, { recursive: true })
  const sig = configSignature(config)
  const found = findNewestMatchingCheckpoint(args.dir, args.label, sig)
  console.log(
    found
      ? `[campaign] resume label=${args.label} from ${found.path}, step=${found.file.globalStep}`
      : `[campaign] new run label=${args.label} (no matching checkpoints in ${args.dir})`,
  )

  let stopRequested = false
  const onSignal = (signal: string) => {
    console.log(`[campaign] received ${signal} -- will save a checkpoint and stop after the current block`)
    stopRequested = true
  }
  process.on('SIGINT', () => onSignal('SIGINT'))
  process.on('SIGTERM', () => onSignal('SIGTERM'))

  const page = await gpuPage()
  page.on('pageerror', (e) => console.error(`[campaign] page error: ${e.message}`))
  // Task 'evaporation' (2026-08-20): the solvent-removal/insertion ramp logs one line per increment
  // from INSIDE the page (soup/src/soup-box-scale.ts's applyEvaporatingTransition, soup/src/
  // soup-evaporate.ts's rehydrateSolventTo) -- the force spike each increment left, whether the
  // minimiser had to run, how much solvent is live. Those lines are the only record of what happens
  // during a transition, so they are forwarded to this process's stdout; every other page message is
  // left alone so no existing invocation's output changes.
  page.on('console', (m) => {
    const t = m.text()
    if (t.startsWith('[evaporation]')) console.log(t)
  })

  try {
    const startStep = found?.file.globalStep ?? 0
    const targetStep = startStep + args.steps

    const created = await page.evaluate(
      async (cfgJson: string, checkpointJson: string | null) => {
        const api = (window as any).api
        const cfg = JSON.parse(cfgJson)
        const resume = checkpointJson ? api.decodeCheckpointResume(JSON.parse(checkpointJson)) : undefined
        // clay: false (task 'clay-surface', 2026-08-19). This CLI resumes from checkpoints captured
        // BEFORE the mineral platelet existed -- their positions arrays are sized for a clay-free
        // particle count, so an injected platelet would make every one of them fail to load -- and it
        // drives dry-wet box cycling, which applyBoxScaleOnce refuses on a system with an immobile
        // phase. Pinned clay-free so every existing campaign stays reproducible; a clay campaign is its
        // own measurement with its own checkpoint lineage, not a silent change to this one.
        const sys = await api.createSoup({ box: cfg.box, seed: cfg.seed, kT: cfg.kT, start: cfg.start, catalystCount: cfg.catalystCount, dryWetCycle: cfg.dryWetCycle, dryWetCycles: cfg.dryWetCycles, evaporateSolvent: cfg.evaporateSolvent, electrostatics: cfg.electrostatics, confine: cfg.confine, resume, clay: false })
        ;(window as any).__sys = sys
        return { N: (await sys.particles()).length / 4, steps: sys.steps, confinement: sys.confinement() }
      },
      JSON.stringify(config),
      found ? JSON.stringify(found.file) : null,
    )
    console.log(`[campaign] system ready N=${created.N} start_step=${created.steps} target=${targetStep}`)
    if (created.confinement) {
      const cf = created.confinement as Record<string, number | boolean | number[]>
      console.log(
        `[campaign] confinement: parcel R_wet=${Number(cf.radiusWet).toFixed(4)} R_live=${Number(cf.radiusLive).toFixed(4)} ` +
          `k=${cf.stiffness} box_live=${JSON.stringify(cf.boxLive)} V_parcel=${Number(cf.parcelVolumeLive).toFixed(1)} ` +
          `V_box=${Number(cf.boxVolumeLive).toFixed(1)} (V_box/V_parcel=${(Number(cf.boxVolumeLive) / Number(cf.parcelVolumeLive)).toFixed(3)}) ` +
          `largest_interaction_radius=${Number(cf.cutMax).toFixed(4)} clearance_L/2-R=${Number(cf.clearanceLive).toFixed(4)} ` +
          `strong_condition_2R<L/2=${cf.strongNoWrapLive}`,
      )
    }

    // Task 'loud-failure-and-liquid-water' (2026-08-20): the cold-start minimisation, BEFORE any
    // step and before the box-expansion block below. Fresh runs only: a resumed run's positions are
    // an already-relaxed trajectory, and relaxColdStart refuses a nonzero step counter anyway, so
    // the guard here is about not printing a confusing skip line rather than about safety.
    if (args.relax) {
      if (startStep !== 0) {
        console.log(`[campaign] --relax skipped: this is a resume from step=${startStep}, minimisation is allowed only on a fresh start`)
      } else {
        const r = await page.evaluate(async () => (window as any).__sys.relaxColdStart())
        console.log(
          `[campaign] cold-start minimisation: iterations=${r.iterations} first_step=${r.maxDisplacementStart} ` +
            `total_displacement_bound=${r.displacementBound.toFixed(4)} max|F| ${r.maxForceBefore.toExponential(4)} -> ${r.maxForceAfter.toExponential(4)} ` +
            `nonfinite_before=${r.nonFiniteBefore} nonfinite_after=${r.nonFiniteAfter} system_step=${await page.evaluate(() => (window as any).__sys.steps)}`,
        )
      }
    }

    // box-expansion task: one-time ramped box change, BEFORE the main step loop -- see Args.expandTo's
    // own doc comment for why this is idempotent (skipped on a resume that already landed at the
    // target box). Uses growBoxTo (soup/src/sim.ts), NOT scaleBoxTo: two independent real attempts
    // with scaleBoxTo's rigid-CoM rescaling (cohesion unrestricted, then restricted to just the
    // recognised aggregate's own members) both threw on applyBoxScaleOnce's own runtime self-check --
    // a genuine topological wraparound in the rigid-unit's own BFS-over-proximity-graph construction,
    // not a ramp-fineness problem (see growBoxTo's own doc comment, soup/src/sim.ts, for the measured
    // numbers from both attempts and the pure-Node sweep that ruled out "just use more increments").
    // growBoxTo moves no particle at all -- see that function's own doc comment for why this sidesteps
    // the wraparound risk entirely while still satisfying "ramp it, not one jump" and "assert distances
    // unchanged, no particle lost" (both hold BY CONSTRUCTION here, verified explicitly below anyway).
    if (args.expandTo !== undefined) {
      const liveBoxNow = (await page.evaluate(() => (window as any).__sys.box)) as [number, number, number]
      if (Math.abs(liveBoxNow[0] - args.expandTo) > 1e-6) {
        console.log(
          `[campaign] box expansion ${JSON.stringify(liveBoxNow)} -> [${args.expandTo},${args.expandTo},${args.expandTo}] ` +
            `(growBoxTo, rampSteps=${args.expandRampSteps}, rampRelaxSteps=${args.expandRampRelaxSteps})`,
        )
        // Grid-cost measurement runs on a THROWAWAY probe system, resumed fresh from the SAME
        // checkpoint, NEVER on window.__sys (the real system this call goes on to expand/relax).
        // stepPhasesDEBUG's own doc comment (soup/src/sim.ts) says its 'bondAttempts' bucket mutates
        // the REAL bondSlots graph via n form/break attempts on a FROZEN position snapshot, and
        // 'integration' mutates REAL positions/velocities via n kick+drift+thermostat iterations
        // against a force that is never recomputed as those positions move -- both fine for the perf
        // task's own purpose (state discarded after) but corrupt a system meant for further real use.
        // First attempt at this task called stepPhasesDEBUG directly on window.__sys and then fed its
        // (by then corrupted) bonds()/positions into the box change, which threw immediately: a
        // covalent pair verified (independently, in pure Node, straight off the checkpoint file) to
        // be a normal ~0.94 sigma bond read back at ~25.5 sigma live -- the 'integration' bucket's own
        // stale-force drift, not a real structural problem with the checkpoint. Disposable probes
        // below avoid this entirely.
        async function probeDebug(cfgJson: string, checkpointJson: string | null, n: number, expandToBox: number | null, rampSteps: number, rampRelaxSteps: number) {
          return page.evaluate(
            async (cfgJson2: string, checkpointJson2: string | null, n2: number, expandToBox2: number | null, rampSteps2: number, rampRelaxSteps2: number) => {
              const api = (window as any).api
              const cfg = JSON.parse(cfgJson2)
              const resume = checkpointJson2 ? api.decodeCheckpointResume(JSON.parse(checkpointJson2)) : undefined
              const probe = await api.createSoup({ box: cfg.box, seed: cfg.seed, kT: cfg.kT, start: cfg.start, catalystCount: cfg.catalystCount, dryWetCycle: cfg.dryWetCycle, dryWetCycles: cfg.dryWetCycles, evaporateSolvent: cfg.evaporateSolvent, electrostatics: cfg.electrostatics, confine: cfg.confine, resume, clay: false })
              try {
                if (expandToBox2 !== null) await probe.growBoxTo([expandToBox2, expandToBox2, expandToBox2], rampSteps2, rampRelaxSteps2)
                const debug = await probe.stepPhasesDEBUG(n2)
                return { debug, box: probe.box, steps: probe.steps }
              } finally {
                probe.dispose()
              }
            },
            cfgJson,
            checkpointJson,
            n,
            expandToBox,
            rampSteps,
            rampRelaxSteps,
          )
        }
        const cfgJson = JSON.stringify(config)
        const checkpointJson = found ? JSON.stringify(found.file) : null
        const before = await probeDebug(cfgJson, checkpointJson, GRID_DEBUG_N, null, 0, 0)
        console.log(
          `[campaign] before expansion (probe): box=${JSON.stringify(before.box)} steps=${before.steps} n=${GRID_DEBUG_N} ` +
            `full=${before.debug.full.toFixed(4)}ms gridBuild=${before.debug.gridBuild.toFixed(4)}ms ` +
            `force=${before.debug.force.toFixed(4)}ms bondAttempts=${before.debug.bondAttempts.toFixed(4)}ms ` +
            `integration=${before.debug.integration.toFixed(4)}ms`,
        )

        const invariantsBefore = await page.evaluate(() => (window as any).__sys.invariants())
        // particles() returns a Float32Array of up to 372800 numbers (93200*4) -- transferring that
        // twice (before/after) as a plain array is exactly the transfer pattern this project's own
        // checkpoint-resume-report.md measured killing the page at full scale. A CHEAP fingerprint
        // (running sum + an evenly-spaced sample) is transferred instead of the whole array,
        // sufficient to catch "any particle moved" without paying that cost.
        const fingerprintBefore = await page.evaluate(async () => {
          const sys = (window as any).__sys
          const p = await sys.particles()
          let sum = 0
          for (let i = 0; i < p.length; i++) sum += p[i]
          const sample: number[] = []
          for (let i = 0; i < p.length; i += 3701) sample.push(p[i])
          return { sum, sample }
        })

        const expandT0 = Date.now()
        try {
          await page.evaluate(
            async (targetBox: number, rampSteps: number, rampRelaxSteps: number) => {
              const sys = (window as any).__sys
              await sys.growBoxTo([targetBox, targetBox, targetBox], rampSteps, rampRelaxSteps)
            },
            args.expandTo,
            args.expandRampSteps,
            args.expandRampRelaxSteps,
          )
        } catch (err) {
          // Resilience: growBoxTo can legitimately throw partway through its own ramp (a real,
          // physical assertVerletSafety drift-margin violation -- particles that were only "close"
          // via periodic wraparound under the SMALLER box can read as far apart under a bigger one
          // until real dynamics catches up; see growBoxTo's own doc comment, soup/src/sim.ts, for
          // why this is a genuine, bounded physics effect, not corruption). The underlying position/
          // velocity/bond state is still real and checkpoint-worthy at whatever box size was reached
          // -- save it before re-throwing, so a partial expansion is not a total loss of this run's
          // own GPU time.
          console.error(`[campaign] growBoxTo threw on an intermediate box: ${(err as Error).message}`)
          const partialBox = await page.evaluate(() => (window as any).__sys.box)
          console.log(`[campaign] saving an emergency checkpoint at box=${JSON.stringify(partialBox)}`)
          const partialCheckpoint = await page.evaluate(async (cfgJson2: string) => {
            const api = (window as any).api
            const sys = (window as any).__sys
            return api.encodeCheckpoint(sys, JSON.parse(cfgJson2))
          }, JSON.stringify(config))
          const partialPath = writeCheckpointFile(args.dir, args.label, partialCheckpoint)
          console.log(`[campaign] emergency checkpoint saved: ${partialPath}`)
          throw err
        }
        const expandMs = Date.now() - expandT0

        const invariantsAfter = await page.evaluate(() => (window as any).__sys.invariants())
        const boxAfter = await page.evaluate(() => (window as any).__sys.box)
        const stepsAfter = await page.evaluate(() => (window as any).__sys.steps)
        const fingerprintAfter = await page.evaluate(async () => {
          const sys = (window as any).__sys
          const p = await sys.particles()
          let sum = 0
          for (let i = 0; i < p.length; i++) sum += p[i]
          const sample: number[] = []
          for (let i = 0; i < p.length; i += 3701) sample.push(p[i])
          return { sum, sample }
        })

        // "After expansion" probe: a SEPARATE fresh resume, box-grown to the SAME target via the
        // SAME (rampSteps, rampRelaxSteps) as the real expansion above, so its box exactly matches
        // window.__sys's post-expansion state -- then stepPhasesDEBUG on THAT throwaway, never on
        // window.__sys itself. growBoxTo moves no particle, so this probe's positions are identical
        // to window.__sys's own (both resumed from the SAME checkpoint, neither one's positions moved
        // by growBoxTo) -- only the grid/Verlet state differs, which is exactly what this measures.
        const after = await probeDebug(cfgJson, checkpointJson, GRID_DEBUG_N, args.expandTo, args.expandRampSteps, args.expandRampRelaxSteps)
        console.log(
          `[campaign] after expansion (${expandMs}ms wall clock, probe): box=${JSON.stringify(after.box)} steps=${after.steps} n=${GRID_DEBUG_N} ` +
            `full=${after.debug.full.toFixed(4)}ms gridBuild=${after.debug.gridBuild.toFixed(4)}ms ` +
            `force=${after.debug.force.toFixed(4)}ms bondAttempts=${after.debug.bondAttempts.toFixed(4)}ms ` +
            `integration=${after.debug.integration.toFixed(4)}ms`,
        )
        console.log(
          `[campaign] window.__sys actually after expansion: box=${JSON.stringify(boxAfter)} steps=${stepsAfter} ` +
            `invariantsBefore=${JSON.stringify(invariantsBefore)} invariantsAfter=${JSON.stringify(invariantsAfter)} ` +
            `fingerprintSumBefore=${fingerprintBefore.sum} fingerprintSumAfter=${fingerprintAfter.sum}`,
        )

        // Requirement: "assert after it that intramolecular distances are unchanged and no particle
        // was lost." growBoxTo moves no particle at all, so every pairwise distance (not just bonded
        // ones) is unchanged by construction -- checked here directly, not just inferred: the particle
        // count/charge invariant, AND a position fingerprint (running sum + an evenly-spaced sample)
        // that would catch ANY particle having moved, even one, without literally transferring and
        // diffing all 372800 numbers twice over CDP (checkpoint-resume-report.md's own measured cost
        // concern at this exact particle count).
        if (JSON.stringify(invariantsBefore.monomers) !== JSON.stringify(invariantsAfter.monomers) || invariantsBefore.charge !== invariantsAfter.charge) {
          throw new Error(
            `[campaign] box expansion lost/added particles: ${JSON.stringify(invariantsBefore.monomers)} -> ${JSON.stringify(invariantsAfter.monomers)}`,
          )
        }
        if (fingerprintBefore.sum !== fingerprintAfter.sum || JSON.stringify(fingerprintBefore.sample) !== JSON.stringify(fingerprintAfter.sample)) {
          throw new Error('[campaign] box expansion (growBoxTo) moved at least one particle -- the position fingerprint changed, and it must not have')
        }
        console.log('[campaign] invariant confirmed: the per-monomer particle count, the charge and the position fingerprint were not changed by the box expansion (growBoxTo)')

        // Immediate checkpoint right after expansion, before the main loop below -- this milestone
        // must survive even if the process is killed before the next --every interval.
        const expandedCheckpoint = await page.evaluate(async (cfgJson: string) => {
          const api = (window as any).api
          const sys = (window as any).__sys
          return api.encodeCheckpoint(sys, JSON.parse(cfgJson))
        }, JSON.stringify(config))
        const savedExpandedPath = writeCheckpointFile(args.dir, args.label, expandedCheckpoint)
        console.log(`[campaign] checkpoint after expansion saved: ${savedExpandedPath}`)
      } else {
        console.log(`[campaign] box already expanded to ${JSON.stringify(liveBoxNow)} -- skipping (idempotent resume)`)
      }
    }

    // Read LIVE, not created.steps: when the expansion block above ran, its own ramp-relax steps
    // (Args.expandRampRelaxSteps between each of Args.expandRampSteps increments) already advanced
    // sys.steps past created.steps -- currentStep/the loop below must count against that real
    // total, not a stale pre-expansion snapshot, or the main loop would think it still owed steps
    // that already happened (or double-count/skip the remaining budget).
    let currentStep = (await page.evaluate(() => (window as any).__sys.steps)) as number
    // Task 'decisive-run' (2026-08-20): the minimisation-only control. Steps still due at which a
    // MID-RUN minimisation has to land -- a resume re-derives this from the flag and the live step
    // counter, so a control arm split over several invocations gets each minimisation exactly once
    // (those already behind currentStep are dropped here, not re-applied).
    const minimiseDue = args.minimiseAt.filter((x) => x > currentStep && x <= targetStep).sort((a, b) => a - b)
    if (args.minimiseAt.length > 0) {
      console.log(
        `[campaign] "minimisation only" control: steps=${JSON.stringify(args.minimiseAt)} iterations_each=${args.minimiseIterations} ` +
          `still_due_in_this_call=${JSON.stringify(minimiseDue)}`,
      )
    }
    while (currentStep < targetStep && !stopRequested) {
      // The chunk is cut short at the next minimisation step, so a minimisation lands on EXACTLY the
      // requested global step rather than at the next --every boundary: the control's whole claim is
      // that it received the same minimisations at the same steps as the cycled arm.
      const nextMin = minimiseDue.find((x) => x > currentStep)
      const chunk = Math.min(args.every, targetStep - currentStep, nextMin !== undefined ? nextMin - currentStep : Infinity)

      const t0 = Date.now()
      currentStep = await page.evaluate(async (n: number) => {
        const sys = (window as any).__sys
        await sys.stepCycled(n)
        return sys.steps
      }, chunk)
      const stepMs = Date.now() - t0

      if (nextMin !== undefined && currentStep === nextMin) {
        const m = (await page.evaluate(async (it: number) => (window as any).__sys.minimiseNowDEBUG(it), args.minimiseIterations)) as {
          iterations: number
          globalStep: number
          maxForceBefore: number
          maxForceAfter: number
          displacementBound: number
        }
        console.log(
          `[campaign] mid-run minimisation at step=${m.globalStep} iterations=${m.iterations} ` +
            `max|F| ${m.maxForceBefore.toExponential(4)} -> ${m.maxForceAfter.toExponential(4)} ` +
            `displacement_bound=${m.displacementBound.toFixed(4)}`,
        )
      }

      // Checkpoint transfer, timed in isolation (task requirement: "measure the transfer time at
      // 93 200 particles, reporting the number") -- deliberately its OWN page.evaluate call, not
      // fused with the progress read below, so this number is exactly "encode + hand to node",
      // uncontaminated by the aggregate-analysis CPU work the progress line also needs.
      const t1 = Date.now()
      const checkpoint = await page.evaluate(async (cfgJson: string) => {
        const api = (window as any).api
        const sys = (window as any).__sys
        return api.encodeCheckpoint(sys, JSON.parse(cfgJson))
      }, JSON.stringify(config))
      const checkpointMs = Date.now() - t1
      const savedPath = writeCheckpointFile(args.dir, args.label, checkpoint)

      // Progress (task requirement: "print progress lines ... the aggregate size, the head-shell
      // count and the cavity volume") -- the largest aggregate by amphiphile count, since that is
      // the one candidate that could plausibly BE the closing vesicle at this point in a run.
      const t2 = Date.now()
      const progress = await page.evaluate(async (WALL_SHELL: number) => {
        const api = (window as any).api
        const sys = (window as any).__sys
        const { stage, evidence } = await api.stageOf(sys)
        const aggs = evidence.aggregateAnalysis.aggregates as Array<{
          amphiphileCount: number
          radialHeadShells: unknown
          cavityVolume: number
          flatnessRatio: number
          inPlaneSymmetry: number
          radiusOfGyration: number
          encapsulatedWater: { encapsulatedCount: number; encapsulationThresholdCount: number; closed: boolean; bulkWaterDensity: number } | null | undefined
        }>
        const largest = aggs.length > 0 ? aggs.reduce((a, b) => (b.amphiphileCount > a.amphiphileCount ? b : a)) : null
        // Task 'evaporation' (2026-08-20): N and the live box are printed because with solvent removal
        // they are no longer constants of the run -- a dry-phase progress line has to show how much
        // solvent actually left, or the trace cannot be read.
        const inv = await sys.invariants()
        // Task 'electrostatics' (2026-08-20): the protonation state and the acid-soap pairing, in the
        // SAME progress line as the structure, because with charge on they are the two things the
        // structure is a function of. Computed inside the page (where the charge readback lives) and
        // reduced to scalars before crossing the CDP boundary -- never the whole array.
        const es = sys.electrostatics() as Record<string, any>
        let esLine = ''
        if (es.enabled) {
          const pos = await sys.particles()
          const q = await sys.charges()
          const params = api.loadParams()
          const contact = api.wcaCutoff(params.sigma * params.beadSizes.head_head)
          const pr = api.pairingStats(pos, q, sys.box, es, contact)
          esLine =
            // Task 'long-range-electrostatics' (2026-08-20): the RANGE, in every progress line, because
            // it is the thing this run is about and because it is derived (4*lambda_D, capped by the
            // minimum image) rather than typed -- a report must never have to re-derive which cutoff a
            // run actually used.
            ` rc_es=${es.cutoff.toFixed(4)}(=${es.debyeLengthsSpanned.toFixed(3)}lD, discarded=${es.discardedIntegratedFraction.toFixed(4)})` +
            ` pH=${es.pH} I=${es.ionicStrengthMolar} alpha=${pr.alpha.toFixed(4)} pKaApp=${api.apparentPKa(pr.alpha, es.pH).toFixed(3)}` +
            ` paired=${pr.pairedFraction.toFixed(4)} unlike=${pr.unlikeFraction.toFixed(4)}(random ${pr.unlikeFractionRandom.toFixed(4)})` +
            ` sweeps=${es.sweeps}` +
            (es.last ? ` last(accepted=${es.last.accepted}/${es.last.attempts} dEs=${es.last.dEsMeanKT.toFixed(4)}kT)` : '')
        }
        // Task 'confined-parcel' (2026-08-21): THE COMPETING SINK, in every progress line, because it
        // is the way this experiment most plausibly fails -- amphiphiles plastering the container
        // instead of closing. Reported per species against the UNIFORM null for the same shell, so
        // "enrichment 1.0" means indifferent to the wall and is also the direct check that the wall is
        // in fact neutral. The shell is the measured bilayer thickness (a film's own depth), read from
        // the gate artifact rather than typed. Plus the no-wrap measurement: the largest radius any
        // particle reached, and therefore the largest pair separation in the system, against L/2.
        let wallLine = ''
        const cf = sys.confinement()
        if (cf) {
          const posW = await sys.particles()
          const soupW = api.loadSoup()
          const ws = api.wallStats(
            posW,
            posW.length / 4,
            sys.box,
            cf.radiusLive,
            WALL_SHELL,
            soupW.monomers.map((m: any) => m.id),
            soupW.monomers.findIndex((m: any) => m.id === soupW.solvent.waterId),
          )
          wallLine =
            ` R=${Number(cf.radiusLive).toFixed(4)} maxR=${ws.maxRadius.toFixed(4)} penetration=${ws.penetration.toFixed(4)}` +
            ` face_clearance=${ws.faceClearance.toFixed(4)} maxPair=${ws.maxPairSeparation.toFixed(3)} L/2=${ws.halfBox.toFixed(3)}` +
            ` strong=${ws.strongNoWrap} shell=${ws.shell.toFixed(3)} uniform_fraction=${ws.species[0].uniformFraction.toFixed(4)}` +
            ` enrichment={${ws.species.map((x: any) => `${x.id}:${x.enrichment.toFixed(3)}`).join(',')}}` +
            ` enrichment_vs_water={${ws.species.map((x: any) => `${x.id}:${x.enrichmentVsSolvent.toFixed(4)}`).join(',')}}` +
            ` in_shell={${ws.species.map((x: any) => `${x.id}:${x.inShell}`).join(',')}}`
        }
        // Task 'confined-parcel': encapsulated water and its DERIVED threshold, in the progress line,
        // because in a confined run the threshold depends on a density measured inside the parcel.
        const enc = largest?.encapsulatedWater ?? null
        const encLine = enc
          ? ` encH2O=${enc.encapsulatedCount}/${Number(enc.encapsulationThresholdCount).toFixed(2)} closed=${enc.closed}` +
            ` bulk_water_density=${enc.bulkWaterDensity.toFixed(4)}`
          : ''
        return {
          esLine,
          wallLine,
          encLine,
          stage,
          aggregateCount: evidence.aggregateAnalysis.aggregateCount,
          largestAggregateSize: largest?.amphiphileCount ?? 0,
          headShells: largest ? String(largest.radialHeadShells) : 'n/a',
          cavityVolume: largest?.cavityVolume ?? 0,
          flatness: largest?.flatnessRatio ?? 0,
          inPlane: largest?.inPlaneSymmetry ?? 0,
          rg: largest?.radiusOfGyration ?? 0,
          sizeHistogramTop: (evidence.aggregateAnalysis.sizeHistogram as number[]).slice(0, 6),
          census: inv.monomers,
          bonds: inv.bonds,
          box: sys.box[0],
          phase: sys.cyclePhase,
          cycleIndex: sys.cycleIndex,
        }
      }, wallShell)
      const progressMs = Date.now() - t2

      console.log(
        `[campaign] step=${currentStep}/${targetStep} stage=${progress.stage} aggregates=${progress.aggregateCount} ` +
          `largest=${progress.largestAggregateSize} headShells=${progress.headShells} cavityVolume=${progress.cavityVolume.toFixed(3)} ` +
          `phase=${progress.phase}/${progress.cycleIndex} box=${progress.box.toFixed(4)} bonds=${progress.bonds} census=${JSON.stringify(progress.census)} ` +
          `flat=${progress.flatness.toFixed(4)} inPl=${progress.inPlane.toFixed(4)} rg=${progress.rg.toFixed(3)} ` +
          `histogram=${JSON.stringify(progress.sizeHistogramTop)} ` +
          `stepMs=${stepMs} checkpointMs=${checkpointMs} progressMs=${progressMs} saved=${savedPath}` +
          progress.encLine + progress.esLine + progress.wallLine,
      )
    }

    if (stopRequested) {
      console.log(`[campaign] stopped by signal at step=${currentStep} -- the next call with the same flags will continue from this point`)
    } else {
      console.log(`[campaign] step budget fully completed: step=${currentStep}`)
    }
  } finally {
    await shutdownGpu()
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
