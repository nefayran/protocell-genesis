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
}

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
        const sys = await api.createSoup({ box: cfg.box, seed: cfg.seed, kT: cfg.kT, start: cfg.start, catalystCount: cfg.catalystCount, dryWetCycle: cfg.dryWetCycle, resume })
        ;(window as any).__sys = sys
        return { N: (await sys.particles()).length / 4, steps: sys.steps }
      },
      JSON.stringify(config),
      found ? JSON.stringify(found.file) : null,
    )
    console.log(`[campaign] система готова N=${created.N} стартовый_шаг=${created.steps} цель=${targetStep}`)

    let currentStep = created.steps
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
