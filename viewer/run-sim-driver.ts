// The run-control panel's state machine, sim-stepping half: the async loop that advances the sim in
// bounded batches, its watchdog, the atomistic-slice reconstruction it feeds into each snapshot, and
// the two ways a run ends (finishRun/failRun). Split out of viewer/run.ts (2026-08-18); the DOM
// wiring (buttons, size/stage/step-cap inputs, setState's own text/enable-state painting) stays in
// run-control-panel.ts, which calls into this module's createRunLifecycle().
import { backmapLipid, type BackmapAtom } from '../chem/src/backmap'
import { findAmphiphiles, type Amphiphile } from '../soup/src/amphiphile'
import { detectStage, HEAD_PEAKS_UNAVAILABLE, type StageThresholds } from '../soup/src/stages'
import type { Soup } from '../soup/src/rules'
import type { SoupSystem } from '../soup/src/sim'
import { computeCavitySample } from './run-cavity'
import type { ProgressReadout } from './run-readout'
import { currentElapsedMs, type RunRuntime } from './run-runtime'
import { MAX_TRACE_LINES, SAMPLE_INTERVAL_MS, SIGMA_NM, SLICE_HALF_WIDTH_FRAC, STAGES, STAGE_LABEL, STEP_BATCH, STEP_WATCHDOG_MS, wrap, type RunState, type Snapshot, type TraceEntry } from './run-types'

/** Builds this snapshot's atomistic slice (backmapLipid per amphiphile whose head falls inside
 * the fixed spatial band -- exactly viewer/molecular.ts's own convention). Takes an already-
 * recognised `amphiphiles` list (simDriver's own single findAmphiphiles() call per tick) rather
 * than recomputing it here a second time -- the same list feeds the cavity-drawing member
 * positions below, and both uses must agree on exactly which particles count as "amphiphile
 * members" for this tick. */
function buildAtomisticSlice(
  amphiphiles: Amphiphile[],
  particles: Float32Array,
  boxNow: [number, number, number],
): { atomistic: Snapshot['atomistic']; atomisticSet: Set<number> } {
  const halfWidth = boxNow[0] * SLICE_HALF_WIDTH_FRAC
  const cx = boxNow[0] / 2
  const inSlice = (a: Amphiphile): boolean => {
    const x = particles[a.headIndex * 4]
    return x >= cx - halfWidth && x <= cx + halfWidth
  }
  const atomistic: Snapshot['atomistic'] = []
  const atomisticSet = new Set<number>()
  for (const a of amphiphiles) {
    if (!inSlice(a)) continue
    const headPos: [number, number, number] = [
      particles[a.headIndex * 4],
      particles[a.headIndex * 4 + 1],
      particles[a.headIndex * 4 + 2],
    ]
    const farIdx = a.chain[a.chain.length - 1]
    const farRaw: [number, number, number] = [particles[farIdx * 4], particles[farIdx * 4 + 1], particles[farIdx * 4 + 2]]
    // Minimum-image the far end relative to the head before handing it to backmapLipid -- a
    // chain that happens to straddle a periodic face must still read as one short chain, not a
    // molecule stretched across the whole box (same reasoning as run-types.ts's own wrap()).
    const tail2: [number, number, number] = [
      headPos[0] + wrap(farRaw[0] - headPos[0], boxNow[0]),
      headPos[1] + wrap(farRaw[1] - headPos[1], boxNow[1]),
      headPos[2] + wrap(farRaw[2] - headPos[2], boxNow[2]),
    ]
    const molecule = backmapLipid(headPos, headPos, tail2, a.length, SIGMA_NM)
    atomistic.push(molecule)
    atomisticSet.add(a.headIndex)
    for (const c of a.chain) atomisticSet.add(c)
  }
  return { atomistic, atomisticSet }
}

/** Races mySys.step(batch) against a fixed deadline instead of awaiting it unconditionally --
 * see run-types.ts's own STEP_WATCHDOG_MS doc comment for why an unconditional await is exactly the
 * shape of bug this project has already hit once (a second big submission's onSubmittedWorkDone()
 * that never settled). Returns 'ok' if step() won the race; 'timeout' if the deadline did. If
 * step() itself REJECTS (throws), that rejection propagates out of this function's own `await` as a
 * normal exception -- Promise.race does not swallow it, so simDriver's try/catch below still sees
 * it. A step() that wins the race late (after a timeout was already declared) is left to settle on
 * its own; nothing reads its result at that point, and Promise.race attaching its own reaction to
 * it is enough that a late rejection is not reported as an unhandled one either. */
async function stepWithWatchdog(mySys: SoupSystem, batch: number): Promise<'ok' | 'timeout'> {
  let timer: ReturnType<typeof setTimeout> | undefined
  const timeout = new Promise<'timeout'>((resolve) => {
    timer = setTimeout(() => resolve('timeout'), STEP_WATCHDOG_MS)
  })
  const result = await Promise.race([mySys.step(batch).then(() => 'ok' as const), timeout])
  clearTimeout(timer)
  return result
}

export interface RunLifecycle {
  /** Async loop: advances the sim in bounded STEP_BATCH chunks, sampling evidence/trace at most
   * every SAMPLE_INTERVAL_MS. Every iteration checks `myGeneration` against the live
   * `rt.runGeneration` and `rt.runUI.state` -- a STOP (which bumps runGeneration) or a PAUSE makes
   * the very next check exit/idle instead of this loop ever touching sys.step() again. That is the
   * whole mechanism by which STOP "releases the GPU": WebGPU has no persistent kernel between
   * command-buffer submissions, so once this loop stops calling step(), no further work is ever
   * submitted to the device and GPU load returns to whatever the browser/OS shows at idle.
   *
   * The whole body runs under one try/catch: previously an exception ANYWHERE in this loop (a
   * genuinely wedged step() surfaced via stepWithWatchdog's 'timeout', or any other unexpected
   * throw) left `runUI.state` stuck at 'running' forever with nothing on screen to say so -- the
   * run LOOKED alive because nothing ever told it otherwise. failRun() below is what makes a dead
   * run look dead. */
  simDriver(mySys: SoupSystem, myGeneration: number): Promise<void>
  finishRun(reason: string): void
  /** `kind` (item 1b) is which of the two distinct failure modes this call site is reporting -- see
   * RunUI.errorKind's own doc comment (run-types.ts) and run-control-panel.ts's errorLabel() for
   * why the status line needs this rather than a single hardcoded message. Set on `rt.runUI`
   * BEFORE setState('error') so errorLabel() reads the right value the first time it runs. */
  failRun(reason: string, kind: 'timeout' | 'exception'): void
}

export function createRunLifecycle(
  rt: RunRuntime,
  readout: ProgressReadout,
  setState: (s: RunState) => void,
  soup: Soup,
  thresholds: StageThresholds,
): RunLifecycle {
  function paint(): void {
    readout.paintProgress(rt.runUI, rt.stepCap, rt.bondCount, thresholds, currentElapsedMs(rt))
  }

  function finishRun(reason: string): void {
    rt.runGeneration++ // cancels any in-flight simDriver iteration for good
    rt.activeElapsedMs = currentElapsedMs(rt) // freeze the clock BEFORE the state flip changes what it reads
    setState('stopped')
    paint()
    readout.appendTraceLine(`[stopped] ${reason}`)
    // Release this run's GPUBuffers now, not just before the NEXT run starts -- a run that is
    // stopped and never restarted should not hold ~20 buffers on the shared device indefinitely.
    rt.activeSys?.dispose()
    rt.activeSys = null
  }

  function failRun(reason: string, kind: 'timeout' | 'exception'): void {
    rt.runGeneration++
    rt.activeElapsedMs = currentElapsedMs(rt)
    const message = `${reason} (steps done: ${rt.runUI.steps})`
    rt.runUI.error = message
    rt.runUI.errorKind = kind
    setState('error')
    paint()
    readout.appendTraceLine(`[error] ${message}`)
    console.error('[viewer/run] simDriver failed:', message)
    rt.activeSys?.dispose()
    rt.activeSys = null
  }

  async function simDriver(mySys: SoupSystem, myGeneration: number): Promise<void> {
    try {
      while (true) {
        if (rt.runGeneration !== myGeneration) return // superseded by a STOP (and possibly a new run)
        if (rt.runUI.state !== 'running') {
          // Paused (manually or by visibility): yield without stepping, poll cheaply.
          await new Promise((r) => setTimeout(r, 50))
          continue
        }
        const batch = Math.min(STEP_BATCH, rt.stepCap - rt.runUI.steps)
        if (batch <= 0) {
          finishRun('step cap reached')
          return
        }
        const stepResult = await stepWithWatchdog(mySys, batch)
        if (rt.runGeneration !== myGeneration) return
        if (stepResult === 'timeout') {
          failRun(`the GPU is not responding: step() did not return within ${(STEP_WATCHDOG_MS / 1000).toFixed(0)} s`, 'timeout')
          return
        }
        rt.runUI.steps += batch

        const pos = await mySys.particles()
        const bnds = await mySys.bonds()
        if (rt.runGeneration !== myGeneration) return
        rt.bondCount = bnds.length / 2

        const now = performance.now()
        const amphiphiles = findAmphiphiles(pos, bnds, soup.monomers)
        const { atomistic, atomisticSet } = buildAtomisticSlice(amphiphiles, pos, rt.box)
        rt.latestSnapshot = {
          particles: pos,
          bonds: bnds,
          box: rt.box,
          atomistic,
          atomisticSet,
          amphiphileCount: amphiphiles.length,
          largestCavity: rt.lastCavity,
          cavityCount: rt.lastCavityCount,
        }
        readout.setAtomBadgeFull(
          `The skeleton of the reconstructed atoms comes from the VERIFIED coarse-grained dynamics of the ` +
            `soup; the atomic geometry uses literature bond lengths and angles (chem/src/backmap.ts), not an ` +
            `independent atomistic simulation. Shown atom by atom: ${atomistic.length} of ${amphiphiles.length} ` +
            `amphiphiles found (the rest are the same coarse-grained monomers).`,
        )

        if (rt.sampleDue || now - rt.lastSampleAt >= SAMPLE_INTERVAL_MS) {
          rt.sampleDue = false
          const dtSec = (now - rt.lastSampleAt) / 1000
          rt.runUI.stepsPerSecond = dtSec > 0 ? (rt.runUI.steps - rt.lastSampleSteps) / dtSec : rt.runUI.stepsPerSecond
          rt.lastSampleAt = now
          rt.lastSampleSteps = rt.runUI.steps

          const { stage, evidence } = await detectStage(mySys)
          if (rt.runGeneration !== myGeneration) return
          rt.runUI.stage = stage
          rt.runUI.evidence = evidence
          const entry: TraceEntry = { steps: rt.runUI.steps, stage, evidence }
          rt.runUI.trace.push(entry)
          if (rt.runUI.trace.length > MAX_TRACE_LINES) rt.runUI.trace.shift()

          // --- cavity breakdown: the SAME amphiphile-member positions detectStage's own
          // enclosedVolume just measured above, run through run-cavity.ts's computeCavitySample for
          // the full per-cavity list instead of one summed scalar.
          const { cavity, cavityCount } = computeCavitySample(pos, amphiphiles, rt.box, thresholds)
          rt.lastCavity = cavity
          rt.lastCavityCount = cavityCount
          rt.runUI.cavityCount = cavityCount
          rt.runUI.largestCavity = cavity
            ? { voxelCount: cavity.voxelCount, volume: cavity.volume, radius: cavity.radius, centre: cavity.centre }
            : null
          // The snapshot built just above this block still carries the PREVIOUS tick's cavity --
          // overwrite it now that this tick's own cavity is known, so draw() never lags a whole
          // sample interval behind the readout it is drawn to match.
          rt.latestSnapshot.largestCavity = cavity
          rt.latestSnapshot.cavityCount = cavityCount

          readout.appendTraceLine(
            `step=${entry.steps} stage=${stage} amph=${evidence.amphiphileFraction.toFixed(4)} ` +
              `agg=${evidence.largestAggregateFraction.toFixed(4)} peaks=${
                evidence.headPeaks === HEAD_PEAKS_UNAVAILABLE ? 'n/a' : evidence.headPeaks
              } volume=${evidence.enclosedVolume.toFixed(4)} cavities=${cavityCount} ` +
              `largest_cavity_voxels=${cavity?.voxelCount ?? 0}`,
          )

          paint()

          // Compare positions on the ladder, not identity: a sample can jump two stages at once
          // (aggregation is fast once amphiphiles exist), and an identity check would miss the stop.
          if (STAGES.indexOf(stage) >= STAGES.indexOf(rt.targetStage)) {
            finishRun(
              stage === rt.targetStage
                ? `target stage "${STAGE_LABEL[rt.targetStage]}" reached`
                : `stage "${STAGE_LABEL[stage]}" reached, which is not below the target "${STAGE_LABEL[rt.targetStage]}"`,
            )
            return
          }
        }
      }
    } catch (err) {
      if (rt.runGeneration !== myGeneration) return // a superseded generation's own error, not this run's
      failRun(`error in the run loop: ${(err as Error).message}`, 'exception')
    }
  }

  return { simDriver, finishRun, failRun }
}
