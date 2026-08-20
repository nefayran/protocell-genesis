// Configuration side of the soup campaign CLI (soup/cli/campaign.ts): the flag surface, its usage
// text, the run-configuration signature that decides which checkpoint a resume is allowed to pick
// up, and the crash-safe checkpoint file I/O. Split out of campaign.ts by responsibility
// (CLAUDE.md's 400-600 line rule) when task 'decisive-run' (2026-08-20) added three more flags to
// a file already at 579 lines -- a pure move: not one default, one usage line, one signature field
// or one rename/write discipline changed in the process. campaign.ts keeps the RUN itself (create
// or resume, the optional minimisation/expansion prologue, the step/checkpoint/progress loop).

import { existsSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { parseArgs } from 'node:util'
import { loadParams } from '../../engine/src/params'
import { loadSoup } from '../src/rules'
import type { CheckpointConfig, CheckpointFile } from '../src/checkpoint'

export interface Args {
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
  /** Task 'evaporation' (2026-08-20): make the dry phase REMOVE solvent beads from the system (and
   * rehydration put them back) instead of only compressing the box -- data/soup.json's
   * dryWetCycle.evaporateSolvent for THIS run. Part of the checkpoint config signature, since a run
   * with it and one without are different experiments, not two snapshots of one. Requires --cycle. */
  evaporate: boolean
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
  /** Task 'decisive-run' (2026-08-20): overrides data/soup.json's dryWetCycle.cycles for THIS run.
   * The evaporation task measured that the gain does not compound (largest aggregate 189 after cycle
   * 1, then 169/162/168/166/160, with yield declining monotonically as co_break accumulated), so the
   * schedule that trend justifies is ONE drying event -- and that has to be expressible as a flag,
   * not as an edit to a data file that every other lineage also reads. Part of the resume signature
   * (see configSignature), because a one-cycle and a six-cycle run are two experiments. */
  cycles?: number
  /** Task 'decisive-run' (2026-08-20): global steps at which to apply a MID-RUN energy minimisation
   * (SoupSystem.minimiseNowDEBUG). THE MINIMISATION-ONLY CONTROL: the evaporation report's own
   * concern 1 named the missing arm -- "a run that receives exactly those 9 + 6 minimisations at the
   * same global steps with NO cycling" -- because a minimisation removes overlaps and could
   * plausibly deliver part of the cycled arm's benefit by itself. Empty (the default) touches
   * nothing. Part of the resume signature. */
  minimiseAt: number[]
  /** Iterations per --minimiseAt minimisation. Must equal the cycled arm's own count for the control
   * to be a control -- that count is EvaporationPlan.relaxIterations, printed by the cycled run's
   * own [evaporation] lines, and it is derived from the composition rather than chosen (29 at box 30
   * -- see planEvaporation). Defaults to data/soup.json's coldStartRelax.iterations only so the flag
   * has a defined meaning on its own; a real control passes the cycled arm's number explicitly. */
  minimiseIterations: number
}

export function printUsage(): void {
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
      '  --evaporate       сухая фаза УБИРАЕТ биды растворителя из системы, регидратация возвращает их (требует --cycle)',
      '  --relax           минимизация энергии холодного старта (data/soup.json coldStartRelax) ДО первого шага;',
      '                    только для СВЕЖЕГО прогона, при резюме молча пропускается',
      '  --expandTo <n>            одноразовое рамп-расширение живого бокса до [n,n,n] сразу после создания/резюме,',
      '                            ДО основного цикла шагов (idempotent: пропускается, если бокс уже расширен)',
      '  --expandRampSteps <n>     шагов рампы для --expandTo (по умолчанию 15)',
      '  --expandRampRelaxSteps <n>  шагов обычной динамики между приращениями рампы (по умолчанию 60)',
      '  --cycles <n>      число циклов сухо-влажного цикла для ЭТОГО прогона (переопределяет data/soup.json',
      '                    dryWetCycle.cycles; входит в подпись резюме -- один цикл и шесть суть разные опыты)',
      '  --minimiseAt <s1,s2,...>  глобальные шаги, на которых применить минимизацию энергии ПОСРЕДИ прогона',
      '                    (контроль «только минимизации», без циклирования; входит в подпись резюме)',
      '  --minimiseIterations <n>  итераций на каждую такую минимизацию (по умолчанию coldStartRelax.iterations;',
      '                    для контроля берётся число из [evaporation]-строк циклированного плеча)',
    ].join('\n'),
  )
}

export function parseCliArgs(): Args {
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
      evaporate: { type: 'boolean', default: false },
      relax: { type: 'boolean', default: false },
      expandTo: { type: 'string' },
      expandRampSteps: { type: 'string' },
      expandRampRelaxSteps: { type: 'string' },
      cycles: { type: 'string' },
      minimiseAt: { type: 'string' },
      minimiseIterations: { type: 'string' },
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
    evaporate: Boolean(values.evaporate),
    relax: Boolean(values.relax),
    expandTo: values.expandTo !== undefined ? Number(values.expandTo) : undefined,
    expandRampSteps: values.expandRampSteps !== undefined ? Number(values.expandRampSteps) : 15,
    expandRampRelaxSteps: values.expandRampRelaxSteps !== undefined ? Number(values.expandRampRelaxSteps) : 60,
    cycles: values.cycles !== undefined ? Number(values.cycles) : undefined,
    minimiseAt:
      values.minimiseAt !== undefined
        ? String(values.minimiseAt)
            .split(/[\s,]+/)
            .filter((x) => x.length > 0)
            .map((x) => Number(x))
            .sort((a, b) => a - b)
        : [],
    minimiseIterations: values.minimiseIterations !== undefined ? Number(values.minimiseIterations) : soup.coldStartRelax?.iterations ?? 200,
  }
}

/** A deterministic string identifying "this exact run configuration" -- two checkpoint files with
 * the same signature are two snapshots of what would be the SAME run, and only among those is
 * "newest" (highest globalStep) a meaningful thing to pick; a file with any other signature belongs
 * to a different composition/box/seed and picking it would silently resume the wrong experiment. */
export function configSignature(c: CheckpointConfig): string {
  const startSorted = Object.fromEntries(Object.entries(c.start ?? {}).sort(([a], [b]) => a.localeCompare(b)))
  return JSON.stringify({
    box: c.box,
    seed: c.seed,
    kT: c.kT,
    start: startSorted,
    catalystCount: c.catalystCount ?? null,
    dryWetCycle: c.dryWetCycle ?? null,
    evaporateSolvent: c.evaporateSolvent ?? null,
    // Task 'decisive-run' (2026-08-20): SPREAD, not a plain `?? null` field. A plain field would add
    // a key to every signature this function has ever produced and orphan every checkpoint lineage
    // on disk; a conditional spread leaves the JSON byte-identical for a run that passes neither
    // flag, while still making a one-cycle run and a minimised run refuse to resume from anything
    // else.
    ...(c.dryWetCycles !== undefined ? { dryWetCycles: c.dryWetCycles } : {}),
    ...(c.minimiseAt !== undefined && c.minimiseAt.length > 0 ? { minimiseAt: c.minimiseAt } : {}),
  })
}

export function findNewestMatchingCheckpoint(dir: string, label: string, sig: string): { path: string; file: CheckpointFile } | null {
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
export function writeCheckpointFile(dir: string, label: string, file: CheckpointFile): string {
  const finalPath = join(dir, `${label}-step${file.globalStep}.json`)
  const tmpPath = `${finalPath}.tmp`
  writeFileSync(tmpPath, JSON.stringify(file))
  renameSync(tmpPath, finalPath)
  return finalPath
}
