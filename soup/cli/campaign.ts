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

import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import { gpuPage, shutdownGpu } from '../../tests/helpers/gpu'
import { loadParams } from '../../engine/src/params'
import { loadSoup } from '../src/rules'
import type { CheckpointConfig, CheckpointFile } from '../src/checkpoint'

interface Args {
  label: string
  steps: number
  every: number
  dir: string
  box: number
  seed: number
  kT: number
  start: Record<string, number>
  catalyst?: number
  cycle: boolean
  /** Task 'loud-failure-and-liquid-water' (2026-08-20): run the cold-start energy minimisation
   * (SoupSystem.relaxColdStart, soup/src/soup-relax.ts) once, on a FRESH run only, before the first
   * step. Skipped on every resume by construction -- relaxColdStart itself throws at globalStep != 0,
   * and the resumed state is a already-relaxed trajectory, not a cold lattice. Deliberately NOT part
   * of the checkpoint config signature: it is a property of how step 0 was reached, not of the
   * composition/box/seed a resume has to match, and adding it would orphan every existing checkpoint
   * lineage in data/checkpoints. */
  relax: boolean
  /** box-expansion task (2026-08-18, .superpowers/sdd/2026-08-16-soup-to-vesicle/expanded-box-
   * report.md): a ONE-TIME ramped box change applied right after this call's system is created
   * (resumed or fresh), before the main step loop, via SoupSystem.growBoxTo (soup/src/sim.ts) --
   * grows the box and rebuilds the neighbour grid (the SAME resizeSoupGrid mechanism dry-wet
   * cycling already uses) WITHOUT moving any particle. An earlier version of this flag drove
   * scaleBoxTo's rigid-CoM rescaling instead (dry-wet cycling's own mechanism, aimed at an
   * arbitrary target box); that measured a genuine topological wraparound in this specific
   * checkpoint's own rigid-unit BFS construction (both with an unrestricted and a member-restricted
   * cohesion group -- see growBoxTo's own doc comment, soup/src/sim.ts, for the two measured
   * failures), so this flag now drives growBoxTo instead -- see that function's own doc comment for
   * why moving nothing sidesteps the wraparound risk entirely. Idempotent across repeated resumes of
   * the SAME checkpoint lineage: skipped whenever the resumed system's live box already equals this
   * target (see main() below), so re-running this campaign with the same flags after expansion has
   * already happened does not re-expand a second time. Undefined (the default) never touches the
   * box at all -- every existing invocation of this CLI is unaffected. */
  expandTo?: number
  /** How many box-size increments growBoxTo splits the expansion into (each followed by
   * expandRampRelaxSteps of ordinary dynamics) -- an experiment-design choice for THIS one-off
   * task, not a physical model parameter (same status as --steps/--every themselves), so it is an
   * ordinary CLI flag with a written default here, not a data/soup.json field. growBoxTo moves no
   * particle at all, so no increment size here carries any overlap/wraparound risk the way
   * scaleBoxTo's own ramp would -- a single jump would be equally safe geometrically. Ramped anyway,
   * per this task's own "ramp it, not one jump" instruction, and because gradual box growth gives
   * ordinary dynamics repeated, evenly-spaced opportunities to start redistributing material into
   * the newly available volume DURING the expansion, not only after it. Default 15, matching the
   * scale of dry-wet cycling's own ramp (data/soup.json dryWetCycle.rampSteps=6) rather than a
   * value tuned for any overlap-avoidance reason (none applies here). */
  expandRampSteps: number
  /** Ordinary dynamics steps between box-growth increments. Default 60: enough real steps between
   * increments for the aggregate/free-monomer boundary to start responding to the newly available
   * volume gradually rather than all at once at the very end of the ramp; not tuned against any
   * overlap risk (growBoxTo has none), only against giving the "brief post-expansion relaxation"
   * this task's own brief calls for a head start that is spread across the ramp instead of only
   * following it. */
  expandRampRelaxSteps: number
}

// box-expansion task: how many real steps stepPhasesDEBUG isolates each grid/force/bondAttempts/
// integration component over, immediately before and after the one-time expansion above -- an
// experiment-design sample count (perf-report.md's own precedent used n=3000 at N=13100; this
// checkpoint's N=93200 is ~7x larger and the whole point of this measurement is "how expensive did
// this just get", so a SMALL n that still finishes quickly is preferred over perf-report.md's own
// n, not a physical parameter either way).
const GRID_DEBUG_N = 100

function printUsage(): void {
  console.log(
    [
      'Использование:',
      '  npx tsx soup/cli/campaign.ts --label <имя> --steps <n> --box <n> --start \'{"C":300,"O":100,"H":300,"M":20}\' [флаги]',
      '',
      'Обязательные:',
      '  --label <имя>     различает файлы контрольных точек одного каталога (несколько кампаний могут делить --dir)',
      '  --steps <n>       сколько ЕЩЁ шагов сделать в ЭТОМ вызове -- при резюме отсчитывается от найденного шага, не от нуля',
      '  --box <n>         кубический бокс (одно число, все три оси)',
      '  --start <json>    стартовый состав по id мономера, например \'{"C":300,"O":100,"H":300,"M":20}\'',
      '',
      'Опциональные:',
      `  --every <n>       шагов между контрольными точками (по умолчанию из data/soup.json's checkpoint.everySteps)`,
      `  --dir <path>      каталог контрольных точек (по умолчанию из data/soup.json's checkpoint.dir)`,
      '  --seed <n>        (по умолчанию 1)',
      '  --kT <n>          (по умолчанию data/params.json thermostat.kT)',
      '  --catalyst <n>    переопределяет число катализатора отдельно от --start (как CreateSoupOpts.catalystCount)',
      '  --cycle           включает сухо-влажное циклирование (data/soup.json dryWetCycle) для ЭТОЙ системы',
      '  --relax           минимизация энергии холодного старта (data/soup.json coldStartRelax) ДО первого шага;',
      '                    только для СВЕЖЕГО прогона, при резюме молча пропускается',
      '  --expandTo <n>            одноразовое рамп-расширение живого бокса до [n,n,n] сразу после создания/резюме,',
      '                            ДО основного цикла шагов (idempotent: пропускается, если бокс уже расширен)',
      '  --expandRampSteps <n>     шагов рампы для --expandTo (по умолчанию 15)',
      '  --expandRampRelaxSteps <n>  шагов обычной динамики между приращениями рампы (по умолчанию 60)',
    ].join('\n'),
  )
}

function parseCliArgs(): Args {
  const soup = loadSoup()
  const params = loadParams()
  if (!soup.checkpoint) {
    throw new Error("data/soup.json: отсутствует секция 'checkpoint' -- campaign.ts не может выбрать интервал/каталог по умолчанию без неё")
  }
  const { values } = parseArgs({
    options: {
      label: { type: 'string' },
      steps: { type: 'string' },
      every: { type: 'string' },
      dir: { type: 'string' },
      box: { type: 'string' },
      seed: { type: 'string' },
      kT: { type: 'string' },
      start: { type: 'string' },
      catalyst: { type: 'string' },
      cycle: { type: 'boolean', default: false },
      relax: { type: 'boolean', default: false },
      expandTo: { type: 'string' },
      expandRampSteps: { type: 'string' },
      expandRampRelaxSteps: { type: 'string' },
      help: { type: 'boolean', default: false },
    },
    allowPositionals: false,
  })
  if (values.help) {
    printUsage()
    process.exit(0)
  }
  const missing = (['label', 'steps', 'box', 'start'] as const).filter((k) => values[k] === undefined)
  if (missing.length > 0) {
    printUsage()
    throw new Error(`campaign.ts: обязательные флаги отсутствуют: ${missing.map((m) => `--${m}`).join(', ')}`)
  }
  return {
    label: String(values.label),
    steps: Number(values.steps),
    every: values.every !== undefined ? Number(values.every) : soup.checkpoint.everySteps,
    dir: values.dir !== undefined ? String(values.dir) : soup.checkpoint.dir,
    box: Number(values.box),
    seed: values.seed !== undefined ? Number(values.seed) : 1,
    kT: values.kT !== undefined ? Number(values.kT) : params.thermostat.kT,
    start: JSON.parse(String(values.start)),
    catalyst: values.catalyst !== undefined ? Number(values.catalyst) : undefined,
    cycle: Boolean(values.cycle),
    relax: Boolean(values.relax),
    expandTo: values.expandTo !== undefined ? Number(values.expandTo) : undefined,
    expandRampSteps: values.expandRampSteps !== undefined ? Number(values.expandRampSteps) : 15,
    expandRampRelaxSteps: values.expandRampRelaxSteps !== undefined ? Number(values.expandRampRelaxSteps) : 60,
  }
}

/** A deterministic string identifying "this exact run configuration" -- two checkpoint files with
 * the same signature are two snapshots of what would be the SAME run, and only among those is
 * "newest" (highest globalStep) a meaningful thing to pick; a file with any other signature belongs
 * to a different composition/box/seed and picking it would silently resume the wrong experiment. */
function configSignature(c: CheckpointConfig): string {
  const startSorted = Object.fromEntries(Object.entries(c.start ?? {}).sort(([a], [b]) => a.localeCompare(b)))
  return JSON.stringify({
    box: c.box,
    seed: c.seed,
    kT: c.kT,
    start: startSorted,
    catalystCount: c.catalystCount ?? null,
    dryWetCycle: c.dryWetCycle ?? null,
  })
}

function findNewestMatchingCheckpoint(dir: string, label: string, sig: string): { path: string; file: CheckpointFile } | null {
  if (!existsSync(dir)) return null
  const prefix = `${label}-step`
  let best: { path: string; file: CheckpointFile } | null = null
  for (const name of readdirSync(dir)) {
    if (!name.startsWith(prefix) || !name.endsWith('.json')) continue
    const path = join(dir, name)
    let file: CheckpointFile
    try {
      file = JSON.parse(readFileSync(path, 'utf8'))
    } catch {
      // A checkpoint killed mid-write (soup/src/checkpoint.ts's own header: base64, not a number
      // array, is what makes the WRITE itself fast too, but does not make it atomic on its own --
      // see writeCheckpointFile below for the tmp-then-rename discipline that actually guarantees
      // this branch should never fire for a checkpoint THIS file wrote) must not abort the whole
      // scan; skip it and keep looking.
      continue
    }
    if (configSignature(file.config) !== sig) continue
    if (!best || file.globalStep > best.file.globalStep) best = { path, file }
  }
  return best
}

/** Write-then-rename, not a direct writeFileSync to the final name: a checkpoint killed mid-write
 * must never be mistaken for a valid one on the next invocation's scan (findNewestMatchingCheckpoint
 * above). rename() on the same filesystem is atomic (POSIX), so the final path either has the
 * COMPLETE previous write or the complete new one, never a half-written mix -- the same discipline
 * every crash-safe append-only log uses, applied here because "resumable after a crash" is this
 * whole task's own requirement, not just for the physics state but for the file that carries it. */
function writeCheckpointFile(dir: string, label: string, file: CheckpointFile): string {
  const finalPath = join(dir, `${label}-step${file.globalStep}.json`)
  const tmpPath = `${finalPath}.tmp`
  writeFileSync(tmpPath, JSON.stringify(file))
  renameSync(tmpPath, finalPath)
  return finalPath
}

async function main(): Promise<void> {
  const args = parseCliArgs()
  const config: CheckpointConfig = {
    box: [args.box, args.box, args.box],
    seed: args.seed,
    kT: args.kT,
    start: args.start,
    catalystCount: args.catalyst,
    dryWetCycle: args.cycle || undefined,
  }
  mkdirSync(args.dir, { recursive: true })
  const sig = configSignature(config)
  const found = findNewestMatchingCheckpoint(args.dir, args.label, sig)
  console.log(
    found
      ? `[campaign] резюме label=${args.label} из ${found.path}, шаг=${found.file.globalStep}`
      : `[campaign] новый запуск label=${args.label} (совпадающих контрольных точек в ${args.dir} нет)`,
  )

  let stopRequested = false
  const onSignal = (signal: string) => {
    console.log(`[campaign] получен ${signal} -- сохраню контрольную точку и остановлюсь после текущего блока`)
    stopRequested = true
  }
  process.on('SIGINT', () => onSignal('SIGINT'))
  process.on('SIGTERM', () => onSignal('SIGTERM'))

  const page = await gpuPage()
  page.on('pageerror', (e) => console.error(`[campaign] page error: ${e.message}`))

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
        const sys = await api.createSoup({ box: cfg.box, seed: cfg.seed, kT: cfg.kT, start: cfg.start, catalystCount: cfg.catalystCount, dryWetCycle: cfg.dryWetCycle, resume, clay: false })
        ;(window as any).__sys = sys
        return { N: (await sys.particles()).length / 4, steps: sys.steps }
      },
      JSON.stringify(config),
      found ? JSON.stringify(found.file) : null,
    )
    console.log(`[campaign] система готова N=${created.N} стартовый_шаг=${created.steps} цель=${targetStep}`)

    // Task 'loud-failure-and-liquid-water' (2026-08-20): the cold-start minimisation, BEFORE any
    // step and before the box-expansion block below. Fresh runs only: a resumed run's positions are
    // an already-relaxed trajectory, and relaxColdStart refuses a nonzero step counter anyway, so
    // the guard here is about not printing a confusing skip line rather than about safety.
    if (args.relax) {
      if (startStep !== 0) {
        console.log(`[campaign] --relax пропущен: это резюме с шага=${startStep}, минимизация допустима только на свежем старте`)
      } else {
        const r = await page.evaluate(async () => (window as any).__sys.relaxColdStart())
        console.log(
          `[campaign] минимизация холодного старта: итераций=${r.iterations} шаг_первой=${r.maxDisplacementStart} ` +
            `граница_суммарного_смещения=${r.displacementBound.toFixed(4)} max|F| ${r.maxForceBefore.toExponential(4)} -> ${r.maxForceAfter.toExponential(4)} ` +
            `нефинитных_до=${r.nonFiniteBefore} нефинитных_после=${r.nonFiniteAfter} шаг_системы=${await page.evaluate(() => (window as any).__sys.steps)}`,
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
          `[campaign] расширение бокса ${JSON.stringify(liveBoxNow)} -> [${args.expandTo},${args.expandTo},${args.expandTo}] ` +
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
              const probe = await api.createSoup({ box: cfg.box, seed: cfg.seed, kT: cfg.kT, start: cfg.start, catalystCount: cfg.catalystCount, dryWetCycle: cfg.dryWetCycle, resume, clay: false })
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
          `[campaign] ДО расширения (probe): box=${JSON.stringify(before.box)} steps=${before.steps} n=${GRID_DEBUG_N} ` +
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
          console.error(`[campaign] growBoxTo бросил на промежуточном боксе: ${(err as Error).message}`)
          const partialBox = await page.evaluate(() => (window as any).__sys.box)
          console.log(`[campaign] сохраняю аварийную контрольную точку на боксе=${JSON.stringify(partialBox)}`)
          const partialCheckpoint = await page.evaluate(async (cfgJson2: string) => {
            const api = (window as any).api
            const sys = (window as any).__sys
            return api.encodeCheckpoint(sys, JSON.parse(cfgJson2))
          }, JSON.stringify(config))
          const partialPath = writeCheckpointFile(args.dir, args.label, partialCheckpoint)
          console.log(`[campaign] аварийная контрольная точка сохранена: ${partialPath}`)
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

        // "После расширения" probe: a SEPARATE fresh resume, box-grown to the SAME target via the
        // SAME (rampSteps, rampRelaxSteps) as the real expansion above, so its box exactly matches
        // window.__sys's post-expansion state -- then stepPhasesDEBUG on THAT throwaway, never on
        // window.__sys itself. growBoxTo moves no particle, so this probe's positions are identical
        // to window.__sys's own (both resumed from the SAME checkpoint, neither one's positions moved
        // by growBoxTo) -- only the grid/Verlet state differs, which is exactly what this measures.
        const after = await probeDebug(cfgJson, checkpointJson, GRID_DEBUG_N, args.expandTo, args.expandRampSteps, args.expandRampRelaxSteps)
        console.log(
          `[campaign] ПОСЛЕ расширения (${expandMs}ms стенных часов, probe): box=${JSON.stringify(after.box)} steps=${after.steps} n=${GRID_DEBUG_N} ` +
            `full=${after.debug.full.toFixed(4)}ms gridBuild=${after.debug.gridBuild.toFixed(4)}ms ` +
            `force=${after.debug.force.toFixed(4)}ms bondAttempts=${after.debug.bondAttempts.toFixed(4)}ms ` +
            `integration=${after.debug.integration.toFixed(4)}ms`,
        )
        console.log(
          `[campaign] window.__sys реально после расширения: box=${JSON.stringify(boxAfter)} steps=${stepsAfter} ` +
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
            `[campaign] расширение бокса потеряло/добавило частицы: ${JSON.stringify(invariantsBefore.monomers)} -> ${JSON.stringify(invariantsAfter.monomers)}`,
          )
        }
        if (fingerprintBefore.sum !== fingerprintAfter.sum || JSON.stringify(fingerprintBefore.sample) !== JSON.stringify(fingerprintAfter.sample)) {
          throw new Error('[campaign] расширение бокса (growBoxTo) сдвинуло хотя бы одну частицу -- отпечаток позиций изменился, а не должен был')
        }
        console.log('[campaign] инвариант подтверждён: число частиц по мономерам, заряд и отпечаток позиций не изменились расширением бокса (growBoxTo)')

        // Immediate checkpoint right after expansion, before the main loop below -- this milestone
        // must survive even if the process is killed before the next --every interval.
        const expandedCheckpoint = await page.evaluate(async (cfgJson: string) => {
          const api = (window as any).api
          const sys = (window as any).__sys
          return api.encodeCheckpoint(sys, JSON.parse(cfgJson))
        }, JSON.stringify(config))
        const savedExpandedPath = writeCheckpointFile(args.dir, args.label, expandedCheckpoint)
        console.log(`[campaign] контрольная точка после расширения сохранена: ${savedExpandedPath}`)
      } else {
        console.log(`[campaign] бокс уже расширен до ${JSON.stringify(liveBoxNow)} -- пропускаю (idempotent-резюме)`)
      }
    }

    // Read LIVE, not created.steps: when the expansion block above ran, its own ramp-relax steps
    // (Args.expandRampRelaxSteps between each of Args.expandRampSteps increments) already advanced
    // sys.steps past created.steps -- currentStep/the loop below must count against that real
    // total, not a stale pre-expansion snapshot, or the main loop would think it still owed steps
    // that already happened (or double-count/skip the remaining budget).
    let currentStep = (await page.evaluate(() => (window as any).__sys.steps)) as number
    while (currentStep < targetStep && !stopRequested) {
      const chunk = Math.min(args.every, targetStep - currentStep)

      const t0 = Date.now()
      currentStep = await page.evaluate(async (n: number) => {
        const sys = (window as any).__sys
        await sys.stepCycled(n)
        return sys.steps
      }, chunk)
      const stepMs = Date.now() - t0

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
      const progress = await page.evaluate(async () => {
        const api = (window as any).api
        const sys = (window as any).__sys
        const { stage, evidence } = await api.stageOf(sys)
        const aggs = evidence.aggregateAnalysis.aggregates as Array<{ amphiphileCount: number; radialHeadShells: unknown; cavityVolume: number }>
        const largest = aggs.length > 0 ? aggs.reduce((a, b) => (b.amphiphileCount > a.amphiphileCount ? b : a)) : null
        return {
          stage,
          aggregateCount: evidence.aggregateAnalysis.aggregateCount,
          largestAggregateSize: largest?.amphiphileCount ?? 0,
          headShells: largest ? String(largest.radialHeadShells) : 'n/a',
          cavityVolume: largest?.cavityVolume ?? 0,
        }
      })
      const progressMs = Date.now() - t2

      console.log(
        `[campaign] шаг=${currentStep}/${targetStep} stage=${progress.stage} агрегатов=${progress.aggregateCount} ` +
          `крупнейший=${progress.largestAggregateSize} headShells=${progress.headShells} cavityVolume=${progress.cavityVolume.toFixed(3)} ` +
          `stepMs=${stepMs} checkpointMs=${checkpointMs} progressMs=${progressMs} сохранено=${savedPath}`,
      )
    }

    if (stopRequested) {
      console.log(`[campaign] остановлен по сигналу на шаге=${currentStep} -- следующий вызов с теми же флагами продолжит с этой точки`)
    } else {
      console.log(`[campaign] бюджет шагов выполнен полностью: шаг=${currentStep}`)
    }
  } finally {
    await shutdownGpu()
  }
}

main().catch((err) => {
  console.error(err)
  process.exitCode = 1
})
