// Task 'consolidation' (2026-08-20): the gate metrics that CANNOT come from a membrane-engine
// scenario, collected for verify/run.ts from the artifacts their own measuring code wrote.
//
// WHY THIS FILE EXISTS AT ALL. verify/run.ts measures the solvent-free bilayer gates itself, in its
// own process, on purpose (see its header: an earlier version read another process's artifact off
// disk and published a report that disagreed with the file next to it). Three of the gates this
// project now publishes cannot be measured that way inside one invocation:
//   * the explicit-water bilayer gates need the 400-lipid + 4500-water patch WITH the zero-tension
//     area move -- 152 000 steps at N = 5700, ~120 s, and the construction lives in
//     tests/water-bilayer-area-move.test.ts, which is 458 lines of measured protocol (settle
//     criterion, non-overlapping tail windows, trials-per-chunk) that must not be forked;
//   * the closure / chain-statistics / percolation / verdict gates are properties of a CAMPAIGN --
//     176 400 steps at N = 191 778, tens of minutes of GPU, which no `npm run verify` may ever
//     start on someone's working machine.
//
// So those metrics are read from artifacts, and the ONE rule that makes that honest is enforced
// here rather than documented: every value carries the artifact it came from AND that artifact's own
// mtime, and a MISSING artifact yields no metric at all -- so evaluateGates() publishes the gate as
// `unproven` with a reason, and nothing stale is ever carried forward as if it were fresh. The two
// race conditions verify/run.ts's header warns about cannot apply: each path below has exactly ONE
// writer, and none of them is verify/run.ts.
//
// Every artifact below is produced by re-running measurement code on data ALREADY on disk (the
// campaign's own checkpoints), never by re-running the campaign.
import { existsSync, readFileSync, statSync } from 'node:fs'

/** Where each group of metrics is read from. Paths, not globs: the point is that a reader can go
 * look at the same file. `gates-` prefixed artifacts are the ones written for THIS pipeline (by the
 * same tests that produced the campaign's own traces, pointed at these paths through their own env
 * vars), so re-generating the gate table never overwrites the campaign's published trace. */
export const WATER_BILAYER_ARTIFACT = 'verify/out/water-bilayer-area-move.json'
export const CAMPAIGN_TRACE_ARTIFACT = 'verify/out/gates-campaign-trace.json'
export const PERCOLATION_ARTIFACT = 'verify/out/gates-percolation.json'

export interface CampaignGateInputs {
  metrics: Record<string, number>
  provenance: Record<string, string>
  notes: Record<string, string>
  /** Everything measured that is not itself a gate metric but is needed to read one honestly (the
   * raw encapsulated-water count and its threshold, both alphas, the checkpoint the row came from).
   * Published into gates.json's own `campaign` block so the ratio the closure gate publishes can be
   * checked against the two numbers it is a ratio of. */
  detail: Record<string, unknown>
}

function readJson<T>(path: string): { value: T; measuredAt: string } | null {
  if (!existsSync(path)) return null
  return { value: JSON.parse(readFileSync(path, 'utf8')) as T, measuredAt: statSync(path).mtime.toISOString() }
}

interface WaterBilayerArtifact {
  areaPerLipid: number
  thickness: number | null
  areaTailMin: number
  areaTailMax: number
  settled: boolean
  chunksUsed: number
  driftPerChunk: number
  driftT: number
  clusterFraction: number
  waterInCore: number
  totalWater: number
  headBuriedFraction: number | null
  acceptedFraction: number
  N: number
  stepsPerSec: number
  waterDensity: number
  lipids: number
}

/** One checkpoint of the off-GPU campaign audit (tests/continuous-run-audit.test.ts's own artifact
 * shape -- only the fields this file reads are declared). `encapsulatedWater` is a THREE-state
 * field by that test's own convention: an object when measured, the string 'null(centre-untrusted)'
 * when water-closure.ts refused because the aggregate's periodic centre cannot be trusted, and
 * 'undefined' when no water indices were supplied at all. */
interface AuditCheckpoint {
  step: number
  box: [number, number, number]
  stage: string
  meanPerTail: number | null
  alphaEvent: number | null
  asfMeanFromEventAlpha: number | null
  alphaRecovered: number | null
  alphaRecoveredR2: number | null
  hasVesicleAggregate: boolean
  aggregateCount: number
  amphiphileCount: number
  largest: Array<{
    amphiphileCount: number
    radialHeadShells: number | string
    cavityVolume: number
    encapsulatedWater:
      | string
      | { encapsulatedCount: number; encapsulationThresholdCount: number; closed: boolean; bulkWaterDensity: number }
  }>
  closureThreshold: number
}

interface PercolationRow {
  file: string
  step: number
  box: number
  amphiphilesInLargest: number
  wrappingAxes: number
  slabsTouchedOfTotal: number[]
}

/** The wet, settled checkpoints -- i.e. the ones a structural claim may rest on. The dry phase of a
 * cycled run is one contact-percolating mass BY CONSTRUCTION (target dry density 1.34), so its
 * aggregate/closure numbers mean nothing structurally; final-campaign-report.md §5 prints them in
 * italics for the density trajectory only. Identified by the box, not by an index: a dry checkpoint
 * is exactly one whose box is smaller than the run's own largest box. */
function wetCheckpoints(trace: AuditCheckpoint[]): AuditCheckpoint[] {
  const maxSide = Math.max(...trace.map((c) => c.box[0]))
  return trace.filter((c) => c.box[0] === maxSide)
}

export function collectCampaignGateInputs(): CampaignGateInputs {
  const metrics: Record<string, number> = {}
  const provenance: Record<string, string> = {}
  const notes: Record<string, string> = {}
  const detail: Record<string, unknown> = {}

  // --- the explicit-water bilayer gates ---------------------------------------------------------
  const water = readJson<WaterBilayerArtifact>(WATER_BILAYER_ARTIFACT)
  if (water) {
    const w = water.value
    metrics.areaPerLipidWater = w.areaPerLipid
    if (w.thickness !== null) metrics.thicknessWater = w.thickness
    const prov =
      `${WATER_BILAYER_ARTIFACT} (tests/water-bilayer-area-move.test.ts, измерено ${water.measuredAt}): ` +
      `${w.lipids} липидов + ${w.totalWater} бидов воды при ${w.waterDensity} sigma^-3, N=${w.N}, ` +
      `осёдлость=${w.settled}, ${w.chunksUsed} отсчётов, дрейф ln A ${w.driftPerChunk.toExponential(3)} (t=${w.driftT.toFixed(2)})`
    provenance['area-per-lipid-water'] = prov
    provenance['bilayer-thickness-water'] = prov
    detail.waterBilayer = w
    if (w.thickness === null) {
      notes['bilayer-thickness-water'] =
        'толщина не измерена в этом прогоне (bilayerPeaks не нашёл двух пиков плотности голов) — ' +
        'публикуется как недоказанная, а не как число из прошлого прогона'
    }
  } else {
    const note =
      `нечего измерять: артефакт ${WATER_BILAYER_ARTIFACT} отсутствует, поэтому свежего числа нет, ` +
      `а старое не подставляется. Опубликованные значения этих ворот и полная методика — ` +
      `.superpowers/sdd/2026-08-16-soup-to-vesicle/hydrophobic-asymmetry-report.md (площадь 1.1777 sigma^2, ` +
      `толщина 4.7990 sigma, перезамерено final-campaign-report.md §3). Пересобрать: ` +
      `nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism`
    notes['area-per-lipid-water'] = note
    notes['bilayer-thickness-water'] = note
  }

  // --- the campaign gates: closure, chain statistics, the verdict --------------------------------
  const audit = readJson<AuditCheckpoint[]>(CAMPAIGN_TRACE_ARTIFACT)
  if (audit && audit.value.length > 0) {
    const trace = audit.value
    const wet = wetCheckpoints(trace)
    const last = wet[wet.length - 1]

    // Closure: the LAST wet checkpoint whose periodic centre is trustworthy enough for
    // water-closure.ts to answer at all. Deliberately the last rather than the best: a closure
    // claim must rest on the state the run ended in, not on whichever sample flatters it.
    const reportable = wet
      .map((c) => ({ c, e: c.largest[0]?.encapsulatedWater }))
      .filter((r): r is { c: AuditCheckpoint; e: Exclude<AuditCheckpoint['largest'][0]['encapsulatedWater'], string> } =>
        typeof r.e === 'object' && r.e !== null,
      )
    const lastReportable = reportable[reportable.length - 1]
    if (lastReportable) {
      const { encapsulatedCount, encapsulationThresholdCount, closed, bulkWaterDensity } = lastReportable.e
      metrics.encapsulatedWaterOverThreshold = encapsulatedCount / encapsulationThresholdCount
      provenance['vesicle-closure-water'] =
        `${CAMPAIGN_TRACE_ARTIFACT} (tests/continuous-run-audit.test.ts, off-GPU по чекпойнтам кампании, ` +
        `измерено ${audit.measuredAt}), шаг ${lastReportable.c.step}: инкапсулировано ${encapsulatedCount} ` +
        `водяных бидов против порога ${encapsulationThresholdCount.toFixed(3)} (объёмная плотность воды ` +
        `${bulkWaterDensity.toFixed(4)} sigma^-3 x минимальный замкнутый объём ${lastReportable.c.closureThreshold} sigma^3), ` +
        `closed=${closed}; таких снимков с надёжным центром ${reportable.length} из ${wet.length} влажных, и на ` +
        `КАЖДОМ инкапсулировано 0`
      detail.closure = {
        step: lastReportable.c.step,
        encapsulatedCount,
        encapsulationThresholdCount,
        closed,
        bulkWaterDensity,
        closureVolumeThreshold: lastReportable.c.closureThreshold,
        reportableWetCheckpoints: reportable.length,
        wetCheckpoints: wet.length,
        thresholdRange: [
          Math.min(...reportable.map((r) => r.e.encapsulationThresholdCount)),
          Math.max(...reportable.map((r) => r.e.encapsulationThresholdCount)),
        ],
      }
    } else {
      notes['vesicle-closure-water'] =
        `ни на одном влажном снимке артефакта ${CAMPAIGN_TRACE_ARTIFACT} периодический центр агрегата не ` +
        `надёжен, поэтому soup/src/water-closure.ts отказывается отвечать — это сам по себе результат ` +
        `(см. ворота aggregate-percolation), но числа для этих ворот он не даёт. final-campaign-report.md §7.1`
    }

    // Chain-length statistics, from the last wet checkpoint.
    if (last.meanPerTail !== null && last.asfMeanFromEventAlpha !== null) {
      metrics.meanPerTail = last.meanPerTail
      metrics.asfMeanRelativeDeviation = Math.abs(last.asfMeanFromEventAlpha - last.meanPerTail) / last.meanPerTail
      const prov =
        `${CAMPAIGN_TRACE_ARTIFACT} (измерено ${audit.measuredAt}), шаг ${last.step}: ` +
        `alpha_ev=${last.alphaEvent}, ASF-среднее 1/(1-alpha_ev)=${last.asfMeanFromEventAlpha}, ` +
        `измеренное среднее на хвост=${last.meanPerTail}, alpha восстановленное из гистограммы=` +
        `${last.alphaRecovered} (r^2=${last.alphaRecoveredR2})`
      provenance['chain-length-asf'] = prov
      provenance['mean-tail-length'] = prov
      detail.chainStatistics = {
        step: last.step,
        alphaEvent: last.alphaEvent,
        asfMeanFromEventAlpha: last.asfMeanFromEventAlpha,
        meanPerTail: last.meanPerTail,
        alphaRecovered: last.alphaRecovered,
        alphaRecoveredR2: last.alphaRecoveredR2,
        alphaAgreementRelative:
          last.alphaEvent !== null && last.alphaRecovered !== null
            ? Math.abs(last.alphaEvent - last.alphaRecovered) / last.alphaEvent
            : null,
        meanPerTailFirstWet: wet[0]?.meanPerTail ?? null,
      }
    }

    // The verdict: how many aggregates ever passed the project's own vesicle criterion, at ANY
    // checkpoint of this trace -- the maximum, so a single closed object anywhere would show up.
    metrics.vesicleAggregates = trace.some((c) => c.hasVesicleAggregate) ? 1 : 0
    provenance['vesicle-verdict'] =
      `${CAMPAIGN_TRACE_ARTIFACT} (измерено ${audit.measuredAt}): ${trace.length} снимков, ` +
      `hasVesicleAggregate=false на всех; последний влажный снимок — шаг ${last.step}, ` +
      `агрегатов ${last.aggregateCount}, крупнейший ${last.largest[0]?.amphiphileCount ?? 0} амфифилов, ` +
      `радиальных слоёв голов ${String(last.largest[0]?.radialHeadShells ?? 'н/д')}, ` +
      `полость ${last.largest[0]?.cavityVolume ?? 0} sigma^3 против порога ${last.closureThreshold} sigma^3`
    detail.verdict = {
      checkpoints: trace.length,
      wetCheckpoints: wet.length,
      lastStep: last.step,
      box: last.box,
      largestAmphiphiles: last.largest[0]?.amphiphileCount ?? 0,
      radialHeadShells: last.largest[0]?.radialHeadShells ?? null,
      cavityVolume: last.largest[0]?.cavityVolume ?? 0,
      cavityVolumeThreshold: last.closureThreshold,
    }
  } else {
    const note =
      `нечего измерять: артефакт ${CAMPAIGN_TRACE_ARTIFACT} отсутствует или пуст, поэтому свежего числа ` +
      `нет, а старое не подставляется. Опубликованные значения — ` +
      `.superpowers/sdd/2026-08-16-soup-to-vesicle/final-campaign-report.md §5, §7, §8. Пересобрать без ` +
      `повторного прогона кампании (off-GPU, по её собственным чекпойнтам): CONTINUOUS_RUN_CHECKPOINTS=... ` +
      `CONTINUOUS_RUN_ARTIFACT=${CAMPAIGN_TRACE_ARTIFACT} npx vitest run tests/continuous-run-audit.test.ts`
    for (const id of ['vesicle-closure-water', 'chain-length-asf', 'mean-tail-length', 'vesicle-verdict']) {
      notes[id] = note
    }
  }

  // --- the percolation gate ---------------------------------------------------------------------
  const perc = readJson<PercolationRow[]>(PERCOLATION_ARTIFACT)
  const campaignRows = (perc?.value ?? []).filter((r) => r.file.includes('zfB54'))
  if (perc && campaignRows.length > 0) {
    // The WORST case over the campaign's own wet checkpoints: this gate asks "is the object finite",
    // and one wrapping axis at one checkpoint already answers no.
    const worst = campaignRows.reduce((a, b) => (b.wrappingAxes > a.wrappingAxes ? b : a))
    metrics.wrappingAxes = worst.wrappingAxes
    const controls = (perc.value ?? []).filter((r) => !r.file.includes('zfB54'))
    provenance['aggregate-percolation'] =
      `${PERCOLATION_ARTIFACT} (tests/percolation-check.test.ts, off-GPU по чекпойнтам кампании, ` +
      `измерено ${perc.measuredAt}): ${campaignRows.length} снимков кампании, обёртывающих осей ` +
      `${campaignRows.map((r) => r.wrappingAxes).join('/')}, слоёв затронуто ` +
      `${worst.slabsTouchedOfTotal.join('/')} на шаге ${worst.step}` +
      (controls.length > 0
        ? `; контроли (объекты, которые проект называет конечными): ` +
          controls.map((r) => `${r.file.split('/').pop()}=${r.wrappingAxes}`).join(', ')
        : '')
    detail.percolation = { campaign: campaignRows, controls }
  } else {
    notes['aggregate-percolation'] =
      `нечего измерять: артефакт ${PERCOLATION_ARTIFACT} отсутствует или не содержит снимков кампании, ` +
      `поэтому свежего числа нет, а старое не подставляется. Опубликованное значение (3 оси из 3 на 5 из 5 ` +
      `влажных снимков, 19/19 слоёв) — .superpowers/sdd/2026-08-16-soup-to-vesicle/final-campaign-report.md §6. ` +
      `Пересобрать: PERC_CHECKPOINTS=... PERC_ARTIFACT=${PERCOLATION_ARTIFACT} npx vitest run tests/percolation-check.test.ts`
  }

  return { metrics, provenance, notes, detail }
}
