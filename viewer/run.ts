// Run-control page: the user starts a soup run, watches it, pauses it and stops it -- nobody
// starts a long GPU run FOR them from a script. Sibling of viewer/main.ts and viewer/molecular.ts
// (same three.js/orbit-controls structure), but the thing under everyone's control here is the
// *lifecycle* of the run, not just a couple of live sliders.
//
// Split by responsibility (2026-08-18, same task that split soup/src/sim.ts) into:
//  - run-types.ts     -- shared types, page-level constants, tiny pure helpers (wrap/formatElapsed/
//                         elementColor/elementRadius). No DOM/three.js/GPU side effects.
//  - run-runtime.ts   -- the shared mutable RunRuntime context every stateful module below takes as
//                         a parameter, standing in for this file's ORIGINAL closure over local
//                         `let`/`const` variables (box, activeSys, runGeneration, latestSnapshot,
//                         lastCavity, elapsed bookkeeping, ...).
//  - run-scene.ts     -- the three.js scene and its ocean look: renderer/scene/camera/controls, the
//                         depth-graded background + drifting caustic bands, key/fill lighting, fog,
//                         box-diagonal camera framing (`frustumCulled = false` plus the box-diagonal
//                         near/far planes both live here -- without them the scene vanishes at some
//                         zoom/angle, see that file's own header), and per-run InstancedMesh capacity
//                         construction.
//  - run-render.ts    -- the per-frame render loop and the instanced-particle buffer updates: draw()
//                         from a Snapshot, plus the requestAnimationFrame loop throttled independent
//                         of simDriver's own rate, plus the sceneDebug test hooks.
//  - run-cavity.ts    -- the cavity/vesicle voxel highlight: mesh construction/update, the flood-fill
//                         computation, the #cavity DOM panel and its honesty note.
//  - run-readout.ts   -- the live readout of measurements (steps/s, elapsed, ETA, stage ladder,
//                         evidence, per-aggregate panel, trace log) and the collapsible honesty-note/
//                         atom-badge notes.
//  - run-sim-driver.ts -- the run-control panel's sim-stepping half: simDriver's bounded-batch async
//                         loop, its watchdog (must not attribute every failure to "GPU не отвечает" --
//                         see RunUI.errorKind's own doc comment), the atomistic-slice reconstruction,
//                         and how a run ends (finishRun/failRun), including the restart path that lets
//                         a new run start after one finished (SoupSystem.dispose()).
//  - run-control-panel.ts -- the run-control panel's DOM half: start/stop/pause, the parameter inputs
//                         (grid size and the rest), their live validation (including the guard that no
//                         monomer-only state counts as a run target), and the state-machine text.
//
// This file is now a thin composition point: load data, build the runtime + every module above in
// the same order the original single main() did, wire the two closures that would otherwise need a
// construction-order cycle (see createControlPanel's own doc comment), and expose window.runUI/
// window.sceneDebug with exactly the shapes they had before the split.
import { loadParams } from '../engine/src/params'
import { loadSoup } from '../soup/src/rules'
import { loadStageThresholds } from '../soup/src/stages'
import { createCavityPanel } from './run-cavity'
import { createControlPanel } from './run-control-panel'
import { createProgressReadout } from './run-readout'
import { createRunRenderer } from './run-render'
import { createRunRuntime } from './run-runtime'
import { createRunScene } from './run-scene'
import { createRunLifecycle, type RunLifecycle } from './run-sim-driver'
import { BOND_RADIUS, DEFAULT_SIZE_KEY, SIZE_PRESETS, type RunUI, type SceneDebugHooks } from './run-types'

function main(): void {
  const params = loadParams()
  const soup = loadSoup()
  const thresholds = loadStageThresholds()

  const runtime = createRunRuntime(SIZE_PRESETS[DEFAULT_SIZE_KEY].box)
  ;(window as unknown as { runUI: RunUI }).runUI = runtime.runUI

  const canvas = document.getElementById('scene') as HTMLCanvasElement
  const runScene = createRunScene(canvas)

  const cavityPanel = createCavityPanel(runScene.camera, runScene.controls, runtime.runUI, thresholds)
  const readout = createProgressReadout(() => cavityPanel.paint(), soup)

  const runRenderer = createRunRenderer(runScene, runtime, soup.monomers, BOND_RADIUS, thresholds.closureCell)
  ;(window as unknown as { sceneDebug: SceneDebugHooks }).sceneDebug = runRenderer.sceneDebug

  // run-sim-driver.ts's simDriver calls into run-control-panel.ts's setState (to transition to
  // 'stopped'/'error' from deep inside its own loop), and run-control-panel.ts's startRun() calls
  // into run-sim-driver.ts's simDriver -- neither module can be constructed strictly before the
  // other. Broken here: the control panel is built first with a lazy accessor for the lifecycle
  // (`() => lifecycle`), then the lifecycle is built with the panel's own `setState`; by the time a
  // user can click a button, both closures have settled -- see createControlPanel's own doc comment.
  let lifecycle!: RunLifecycle
  const controlPanel = createControlPanel(runtime, runScene, readout, () => lifecycle, soup, params, thresholds)
  lifecycle = createRunLifecycle(runtime, readout, controlPanel.setState, soup, thresholds)

  // --- render loop: independent of simDriver, capped at RENDER_INTERVAL_MS (run-render.ts) --------
  runRenderer.startRenderLoop()
}

main()
