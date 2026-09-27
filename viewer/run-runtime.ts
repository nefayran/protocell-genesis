// The shared mutable context every stateful run-*.ts module below takes as a parameter, standing
// in for viewer/run.ts's original main()'s closure over local `let`/`const` variables -- same
// pattern soup/src/soup-runtime.ts already established for the sim.ts split. Type + one factory
// function only, no DOM/three.js/GPU logic.
import type { SoupSystem } from '../soup/src/sim'
import type { CavityWorld } from '../engine/src/closure'
import { emptyAggregateAnalysis, type Stage } from '../soup/src/stages'
import { DEFAULT_STEP_CAP, type RunUI, type SceneMeshes, type Snapshot } from './run-types'

export interface RunRuntime {
  /** The object published as `window.runUI` -- created once, mutated in place for the whole page
   * lifetime (never reassigned), so every module holding a reference to `runtime.runUI` always
   * sees the latest state. */
  runUI: RunUI

  // --- current run's own config, set fresh by startRun() on every START ------------------------
  box: [number, number, number]
  targetStage: Stage
  stepCap: number
  bondCount: number

  // --- three.js instanced meshes, rebuilt per run from that run's own N -------------------------
  meshes: SceneMeshes | null

  // --- the currently live SoupSystem, if any -- tracked here (not just simDriver's own local
  // parameter) purely so startRun() can dispose() the PREVIOUS run's system before handing the
  // device to a new one. See SoupSystem.dispose()'s own doc comment (soup/src/sim.ts) for why this
  // matters: getGpu() memoizes one GPUDevice for the whole page, so without this, every "start a
  // new run" click would leak the just-finished run's ~20 GPUBuffers forever.
  activeSys: SoupSystem | null

  // Runs monotonically upward across the page's whole lifetime; simDriver checks it each iteration
  // so STOP can cancel an in-flight tick's *next* iteration immediately -- see run-sim-driver.ts's
  // use of this for why a stale simDriver from a previous run can never write into a new run's
  // state.
  runGeneration: number

  latestSnapshot: Snapshot | null

  // Cavity info persists ACROSS ticks between samples (unlike evidence/stage, which only ever exist
  // as of the last sample too, but are read straight off runUI) -- draw() reads
  // Snapshot.largestCavity every rendered frame, including the many ticks between two samples, so
  // without this the voxel cloud would flicker to "nothing" every tick that is not itself a sample.
  lastCavity: CavityWorld | null
  lastCavityCount: number

  // Wall-clock bookkeeping: elapsed only accumulates while ACTUALLY running (not paused, not
  // hidden-auto-paused) -- pausing must freeze the clock, not just stop the step counter, or "elapsed"
  // would silently include idle time and make "steps/s" look wrong.
  activeElapsedMs: number
  lastResumeAt: number
  lastSampleAt: number
  lastSampleSteps: number
  // Forces the very NEXT sim tick to sample/paint immediately regardless of SAMPLE_INTERVAL_MS --
  // set on every start/resume so progress is never blank waiting for the first 250ms to elapse
  // (the throttle is a ceiling on repaint RATE, not a floor before the first paint appears).
  sampleDue: boolean

  // Auto-pause on tab hidden vs a user's own manual pause are different things: only an
  // auto-pause resumes itself when the tab comes back; a manual pause must stay paused until the
  // user clicks resume themselves.
  autoPausedByVisibility: boolean
}

export function createRunRuntime(initialBox: [number, number, number]): RunRuntime {
  return {
    runUI: {
      state: 'idle',
      steps: 0,
      stepsPerSecond: 0,
      stage: 'monomers',
      evidence: { amphiphileFraction: 0, largestAggregateFraction: 0, headPeaks: 0, enclosedVolume: 0, aggregateAnalysis: emptyAggregateAnalysis() },
      trace: [],
      error: null,
      errorKind: null,
      cavityCount: 0,
      largestCavity: null,
    },
    box: initialBox,
    targetStage: 'vesicle',
    stepCap: DEFAULT_STEP_CAP,
    bondCount: 0,
    meshes: null,
    activeSys: null,
    runGeneration: 0,
    latestSnapshot: null,
    lastCavity: null,
    lastCavityCount: 0,
    activeElapsedMs: 0,
    lastResumeAt: 0,
    lastSampleAt: 0,
    lastSampleSteps: 0,
    sampleDue: true,
    autoPausedByVisibility: false,
  }
}

/** Wall-clock elapsed while ACTUALLY running, live: accumulated time up to the last
 * pause/resume/stop transition (`rt.activeElapsedMs`) plus, only while currently running, the time
 * since the last resume. Computed fresh on every read rather than mutated in the hot loop, so
 * pause/resume/stop can each freeze or resume the clock by touching `activeElapsedMs` and
 * `lastResumeAt` exactly once, with no double-counting between a pause and a later stop. */
export function currentElapsedMs(rt: RunRuntime): number {
  return rt.activeElapsedMs + (rt.runUI.state === 'running' ? performance.now() - rt.lastResumeAt : 0)
}
