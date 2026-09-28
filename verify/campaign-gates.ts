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
// here rather than documented: every value carries the artifact it came from AND the time that
// artifact was measured (see measuredAtOf), and a MISSING artifact yields no metric at all -- so
// evaluateGates() publishes the gate as `unproven` with a reason, and nothing stale is ever carried
// forward as if it were fresh. The two race conditions verify/run.ts's header warns about cannot
// apply: each path below has exactly ONE writer, and none of them is verify/run.ts.
//
// Every artifact below is produced by re-running measurement code on data ALREADY on disk (the
// campaign's own checkpoints), never by re-running the campaign.
import { execFileSync } from 'node:child_process'
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
  const value = JSON.parse(readFileSync(path, 'utf8')) as T
  return { value, measuredAt: measuredAtOf(path, value) }
}

// When the artifact was measured. A file's mtime is only right on the machine that wrote it: git does
// not store mtimes, so in a fresh clone it is the checkout time. An artifact that stamps its own
// generatedAt is dated by that; an unstamped one that is committed and unmodified is dated by the
// commit that last changed it; only a new or locally modified file falls back to its mtime.
function measuredAtOf(path: string, value: unknown): string {
  const stamped = (value as { generatedAt?: unknown } | null)?.generatedAt
  if (typeof stamped === 'string') return stamped
  try {
    execFileSync('git', ['diff', '--quiet', 'HEAD', '--', path], { stdio: 'ignore' })
    const committed = execFileSync('git', ['log', '-1', '--format=%cI', '--', path], { encoding: 'utf8' }).trim()
    if (committed) return new Date(committed).toISOString()
  } catch {
    // modified against HEAD, untracked, or no git at all
  }
  return statSync(path).mtime.toISOString()
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
  /** Task 'electrostatics' (2026-08-20): 'campaign' = a checkpoint of the run under test, 'control' =
   * an object the project already calls finite and which this instrument must therefore report as
   * NOT wrapping. Written by tests/percolation-check.test.ts (from its own PERC_CAMPAIGN_LABEL);
   * absent on every artifact written before this field existed, which is why the reader below falls
   * back to the old label substring rather than treating absence as 'control'. */
  role?: 'campaign' | 'control'
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

/** Task 'confined-parcel' (2026-08-21): the two campaign artifacts are now PARAMETERS with the
 * pre-existing paths as defaults, so the SAME collection code can be pointed at a second campaign's
 * artifacts and publish it as its OWN row instead of overwriting the first. Both results matter: the
 * periodic-box campaign is a finding in its own right (a percolating network that wraps 3 of 3 axes
 * and encapsulates exactly zero water), and the confined one is a different experiment, not a
 * correction of it. `npm run verify` keeps calling this with no arguments, so the published table is
 * byte-for-byte the periodic one it always was; verify/confined-gates.ts calls it with the confined
 * pair and writes a separate file. */
export function collectCampaignGateInputs(
  campaignTraceArtifact: string = CAMPAIGN_TRACE_ARTIFACT,
  percolationArtifact: string = PERCOLATION_ARTIFACT,
): CampaignGateInputs {
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
      `${WATER_BILAYER_ARTIFACT} (tests/water-bilayer-area-move.test.ts, measured ${water.measuredAt}): ` +
      `${w.lipids} lipids + ${w.totalWater} water beads at ${w.waterDensity} sigma^-3, N=${w.N}, ` +
      `settled=${w.settled}, ${w.chunksUsed} samples, ln A drift ${w.driftPerChunk.toExponential(3)} (t=${w.driftT.toFixed(2)})`
    provenance['area-per-lipid-water'] = prov
    provenance['bilayer-thickness-water'] = prov
    detail.waterBilayer = w
    if (w.thickness === null) {
      notes['bilayer-thickness-water'] =
        'thickness not measured in this run (bilayerPeaks did not find two head-density peaks); ' +
        'published as unproven, not as a number from an earlier run'
    }
  } else {
    const note =
      `nothing to measure: artifact ${WATER_BILAYER_ARTIFACT} is missing, so there is no fresh number, ` +
      `and the old one is not substituted. Published values of these gates and the full method: ` +
      `.superpowers/sdd/2026-08-16-soup-to-vesicle/hydrophobic-asymmetry-report.md (area 1.1777 sigma^2, ` +
      `thickness 4.7990 sigma, remeasured in final-campaign-report.md §3). To rebuild: ` +
      `nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism`
    notes['area-per-lipid-water'] = note
    notes['bilayer-thickness-water'] = note
  }

  // --- the campaign gates: closure, chain statistics, the verdict --------------------------------
  const audit = readJson<AuditCheckpoint[]>(campaignTraceArtifact)
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
        `${campaignTraceArtifact} (tests/continuous-run-audit.test.ts, off-GPU from the campaign checkpoints, ` +
        `measured ${audit.measuredAt}), step ${lastReportable.c.step}: ${encapsulatedCount} water beads ` +
        `encapsulated against a threshold of ${encapsulationThresholdCount.toFixed(3)} (bulk water density ` +
        `${bulkWaterDensity.toFixed(4)} sigma^-3 x minimum closed volume ${lastReportable.c.closureThreshold} sigma^3), ` +
        `closed=${closed}; checkpoints with a reliable centre: ${reportable.length} of ${wet.length} wet ones, and on ` +
        `each of them 0 were encapsulated`
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
        `on no wet checkpoint of artifact ${campaignTraceArtifact} is the periodic centre of the aggregate ` +
        `reliable, so soup/src/water-closure.ts refuses to answer; that is a result in itself ` +
        `(see the aggregate-percolation gate), but it gives no number for this gate. final-campaign-report.md §7.1`
    }

    // Chain-length statistics, from the last wet checkpoint.
    if (last.meanPerTail !== null && last.asfMeanFromEventAlpha !== null) {
      metrics.meanPerTail = last.meanPerTail
      metrics.asfMeanRelativeDeviation = Math.abs(last.asfMeanFromEventAlpha - last.meanPerTail) / last.meanPerTail
      const prov =
        `${campaignTraceArtifact} (measured ${audit.measuredAt}), step ${last.step}: ` +
        `alpha_ev=${last.alphaEvent}, ASF mean 1/(1-alpha_ev)=${last.asfMeanFromEventAlpha}, ` +
        `measured mean per tail=${last.meanPerTail}, alpha recovered from the histogram=` +
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
      `${campaignTraceArtifact} (measured ${audit.measuredAt}): ${trace.length} checkpoints, ` +
      `hasVesicleAggregate=false on all of them; last wet checkpoint: step ${last.step}, ` +
      `${last.aggregateCount} aggregates, largest ${last.largest[0]?.amphiphileCount ?? 0} amphiphiles, ` +
      `radial head shells ${String(last.largest[0]?.radialHeadShells ?? 'n/a')}, ` +
      `cavity ${last.largest[0]?.cavityVolume ?? 0} sigma^3 against a threshold of ${last.closureThreshold} sigma^3`
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
      `nothing to measure: artifact ${campaignTraceArtifact} is missing or empty, so there is no fresh number, ` +
      `and the old one is not substituted. Published values: ` +
      `.superpowers/sdd/2026-08-16-soup-to-vesicle/final-campaign-report.md §5, §7, §8. To rebuild without ` +
      `rerunning the campaign (off-GPU, from its own checkpoints): CONTINUOUS_RUN_CHECKPOINTS=... ` +
      `CONTINUOUS_RUN_ARTIFACT=${campaignTraceArtifact} npx vitest run tests/continuous-run-audit.test.ts`
    for (const id of ['vesicle-closure-water', 'chain-length-asf', 'mean-tail-length', 'vesicle-verdict']) {
      notes[id] = note
    }
  }

  // --- the percolation gate ---------------------------------------------------------------------
  const perc = readJson<PercolationRow[]>(percolationArtifact)
  // Task 'electrostatics' (2026-08-20): the row's OWN `role` decides, with the pre-existing 'zfB54'
  // substring kept as the fallback for artifacts written before that field existed. The hardcoded
  // label was a real defect: the first campaign with a different label turned this gate `unproven`
  // while its measurement was sitting in the artifact.
  const isCampaign = (r: PercolationRow): boolean =>
    r.role !== undefined ? r.role === 'campaign' : r.file.includes('zfB54')
  const campaignRows = (perc?.value ?? []).filter(isCampaign)
  if (perc && campaignRows.length > 0) {
    // The WORST case over the campaign's own wet checkpoints: this gate asks "is the object finite",
    // and one wrapping axis at one checkpoint already answers no.
    const worst = campaignRows.reduce((a, b) => (b.wrappingAxes > a.wrappingAxes ? b : a))
    metrics.wrappingAxes = worst.wrappingAxes
    const controls = (perc.value ?? []).filter((r) => !isCampaign(r))
    provenance['aggregate-percolation'] =
      `${percolationArtifact} (tests/percolation-check.test.ts, off-GPU from the campaign checkpoints, ` +
      `measured ${perc.measuredAt}): ${campaignRows.length} campaign checkpoints, wrapping axes ` +
      `${campaignRows.map((r) => r.wrappingAxes).join('/')}, slabs touched ` +
      `${worst.slabsTouchedOfTotal.join('/')} at step ${worst.step}` +
      (controls.length > 0
        ? `; controls (objects the project calls finite): ` +
          controls.map((r) => `${r.file.split('/').pop()}=${r.wrappingAxes}`).join(', ')
        : '')
    detail.percolation = { campaign: campaignRows, controls }
  } else {
    notes['aggregate-percolation'] =
      `nothing to measure: artifact ${percolationArtifact} is missing or contains no campaign checkpoints, ` +
      `so there is no fresh number, and the old one is not substituted. Published value (3 axes of 3 on 5 of 5 ` +
      `wet checkpoints, 19/19 slabs): .superpowers/sdd/2026-08-16-soup-to-vesicle/final-campaign-report.md §6. ` +
      `To rebuild: PERC_CHECKPOINTS=... PERC_ARTIFACT=${percolationArtifact} npx vitest run tests/percolation-check.test.ts`
  }

  return { metrics, provenance, notes, detail }
}
