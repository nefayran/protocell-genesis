// The run-control panel and its state machine: start/stop/pause, the parameter inputs (size preset,
// box side, particle scale, head count, target stage, step cap) and their live validation preview,
// and the failure/watchdog status text. Split out of viewer/run.ts (2026-08-18). Owns #controls and
// #state-value/#visibility-note; delegates the actual sim-stepping loop to run-sim-driver.ts's
// RunLifecycle and the measurement readout to run-readout.ts's ProgressReadout.
import { equivalentSphereRadius } from '../engine/src/closure'
import type { Params } from '../engine/src/params'
import type { Soup } from '../soup/src/rules'
import { planSoupGrid, createSoup, type SoupSystem } from '../soup/src/sim'
import { emptyAggregateAnalysis, type Stage, type StageThresholds } from '../soup/src/stages'
import type { ProgressReadout } from './run-readout'
import type { RunLifecycle } from './run-sim-driver'
import { currentElapsedMs, type RunRuntime } from './run-runtime'
import type { RunScene } from './run-scene'
import { DEFAULT_SIZE_KEY, DEFAULT_STEP_CAP, RUN_SEED, SIZE_PRESETS, SOLVENT_VISIBLE_BY_DEFAULT, STAGES, STAGE_LABEL, type RunState } from './run-types'

export interface ControlPanel {
  setState(s: RunState): void
}

/** `getLifecycle` is a lazy accessor rather than a plain `RunLifecycle` parameter: run-sim-driver.ts's
 * createRunLifecycle() itself needs THIS module's own `setState` (to transition to 'stopped'/'error'
 * from deep inside simDriver's loop), so the two can't be constructed in either order without one of
 * them existing before the other. run.ts breaks the cycle by constructing this panel first (with a
 * closure that reads a `let lifecycle` assigned right after), then constructing the lifecycle with
 * this panel's own `setState` -- by the time a user can click a button, both closures have settled. */
export function createControlPanel(
  rt: RunRuntime,
  runScene: RunScene,
  readout: ProgressReadout,
  getLifecycle: () => RunLifecycle,
  soup: Soup,
  params: Params,
  thresholds: StageThresholds,
): ControlPanel {
  // --- DOM handles ---------------------------------------------------------------------------
  const sizeSelect = document.getElementById('size-select') as HTMLSelectElement
  const boxSideInput = document.getElementById('box-side-input') as HTMLInputElement
  const particleScaleInput = document.getElementById('particle-scale-input') as HTMLInputElement
  const headCountInput = document.getElementById('head-count-input') as HTMLInputElement
  const sizePreviewEl = document.getElementById('size-preview') as HTMLElement
  const sizeErrorEl = document.getElementById('size-error') as HTMLElement
  const stageSelect = document.getElementById('stage-select') as HTMLSelectElement
  const stepCapInput = document.getElementById('step-cap') as HTMLInputElement
  const startBtn = document.getElementById('start-btn') as HTMLButtonElement
  const pauseBtn = document.getElementById('pause-btn') as HTMLButtonElement
  const stopBtn = document.getElementById('stop-btn') as HTMLButtonElement
  const stateValue = document.getElementById('state-value') as HTMLElement
  const visibilityNoteEl = document.getElementById('visibility-note') as HTMLElement

  for (const [key, preset] of Object.entries(SIZE_PRESETS)) {
    const opt = document.createElement('option')
    opt.value = key
    opt.textContent = preset.label
    sizeSelect.appendChild(opt)
  }
  sizeSelect.value = DEFAULT_SIZE_KEY

  // --- item 3 (2026-08 UI-fixes task): the user picks box side + particle scale directly instead
  // of choosing between two fixed guesses. The presets above are kept only as a convenience "fill
  // these two fields for me" shortcut -- picking one sets `baseStart` (this composition, scaled by
  // 1x) and the box-side field, but both fields stay freely editable afterward, and typing into
  // either no longer requires the preset dropdown to agree.
  let baseStart: Record<string, number> = SIZE_PRESETS[DEFAULT_SIZE_KEY].start ?? soup.start
  boxSideInput.value = String(SIZE_PRESETS[DEFAULT_SIZE_KEY].box[0])
  particleScaleInput.value = '1'
  headCountInput.value = String(baseStart.O)

  /** `baseStart` scaled by `scale` and rounded to whole particles per species -- the "particle
   * scale" control's own effect. Never negative (a scale below what rounds to 0 for every species
   * is caught by planSoupGrid's own N===0 downstream check in createSoup, not specially guarded
   * here). Its own `O` entry is only the FALLBACK default for the head-count field below -- the
   * actual composition control overrides it, see currentSizeSelection(). */
  function scaledStart(scale: number): Record<string, number> {
    return Object.fromEntries(Object.entries(baseStart).map(([id, n]) => [id, Math.max(0, Math.round(n * scale))]))
  }

  /** Composition control: the head count (data/soup.json's monomer `O`) is exposed DIRECTLY as its
   * own field, independent of `particleScaleInput`, rather than as a carbon-to-head RATIO. Chosen
   * over a ratio because it reads unambiguously as a count of molecules -- the same unit the
   * particle-count preview already shows -- with no division the reader has to do in their head to
   * find out what a change means, and because the measurement this control exists to support (see
   * this task's own brief) is itself phrased directly in heads ("a third and a quarter of the
   * heads"), which maps onto this field with zero arithmetic. `particleScaleInput` still scales
   * C/H/M together (unchanged); this field scales O on its own, so the carbon:head RATIO (what the
   * task also offered as an alternative control) is exactly `startCounts.C / startCounts.O`, freely
   * choosable by moving this field independently of the other two -- the ratio is a derived
   * reading of these two direct counts, not a second control a user would have to reconcile with
   * this one. */
  function currentHeadCount(scaledDefault: number): number {
    const raw = Number(headCountInput.value)
    return Number.isFinite(raw) ? Math.round(raw) : scaledDefault
  }

  /** The box/startCounts the CURRENT form fields describe, regardless of whether a run has ever
   * started -- read fresh from the inputs every time (never rt.box/rt.stepCap, which only update on
   * START) so the preview always reflects what the user is looking at right now. Falls back to the
   * tiny preset's own numbers for anything that fails to parse (empty field, non-numeric typing
   * mid-edit) rather than propagating NaN into the preview/guard. `startCounts.O` may come back
   * NEGATIVE here on purpose: currentHeadCount() only falls back to the scaled default for input
   * Number() cannot parse into a finite value at all (e.g. mid-typing a bare "-" or letters) -- an
   * empty field or a literal "0" is a genuine, valid head count of zero (not a fallback trigger),
   * and a genuinely-typed negative number is passed through unclamped. validateSizeSelection() is
   * what turns a negative count into a refusal, not a silent clamp, so a deliberately invalid
   * composition is rejected with a message rather than quietly reinterpreted as something the user
   * didn't type. */
  function currentSizeSelection(): { box: [number, number, number]; startCounts: Record<string, number> } {
    const side = Number(boxSideInput.value)
    const scale = Number(particleScaleInput.value)
    const boxSide = Number.isFinite(side) && side > 0 ? side : SIZE_PRESETS[DEFAULT_SIZE_KEY].box[0]
    const particleScale = Number.isFinite(scale) && scale > 0 ? scale : 1
    const startCounts = scaledStart(particleScale)
    startCounts.O = currentHeadCount(startCounts.O)
    return { box: [boxSide, boxSide, boxSide], startCounts }
  }

  /** Item 3's other guard, alongside planSoupGrid's neighbour-grid check: the vesicle-closure
   * gate's own minimum cavity volume (thresholds.enclosedVolume, tied 1:1 to data/literature.json's
   * measured-bilayer-thickness-derived closure.target.min -- see loadStageThresholds's own doc
   * comment) needs room to physically exist inside the box at all. A cavity of that minimum volume
   * needs a sphere of its own equivalent radius PLUS a wall at least one plausible bilayer
   * thickness (headPeakSeparationMin -- the SAME physically-grounded band the stage ladder itself
   * trusts, not a new number) thick around it on every side, so the box's smallest side must clear
   * twice (radius + wall). A box below this can never show a qualifying cavity no matter how long
   * it runs -- worth refusing up front rather than after a wasted run. */
  function closureMinBoxSide(): number {
    return 2 * (equivalentSphereRadius(thresholds.enclosedVolume) + thresholds.headPeakSeparationMin)
  }

  /** Both of item 3's guards against the CURRENT form selection -- null when the selection is fine
   * to start, otherwise the human-readable reason (possibly both reasons, one per line) the start
   * button must refuse. Shared by refreshSizePreview() (so the refusal is visible before the user
   * even clicks START) and startRun() (so START itself never proceeds on a selection this already
   * flagged, no matter how fast the user clicks past the live preview). */
  function validateSizeSelection(boxNow: [number, number, number], startCounts: Record<string, number>): string | null {
    const reasons: string[] = []
    // Composition guards, ahead of the box/grid guards below (cheap, no dependence on `box` at
    // all, and the more directly "the user just typed something nonsensical" of the two failure
    // classes this function reports). Both are refused rather than silently clamped/ignored --
    // createSoup() itself would otherwise either throw a much less specific error (`startCounts.O`
    // negative propagating into a GPU buffer size) or, for N===0, its own already-existing
    // 'стартовый состав пуст' throw -- refusing here, before createSoup is ever called, gives the
    // SAME guarantee this task's other two guards already have: a clear message, never a crash.
    if (startCounts.O < 0) {
      reasons.push(
        `число голов (O)=${startCounts.O} — состав не может быть отрицательным; введите 0 или больше`,
      )
    }
    const totalN = Object.values(startCounts).reduce((a, b) => a + b, 0)
    if (totalN <= 0) {
      reasons.push(`стартовый состав пуст: суммарное число частиц по всем видам равно ${totalN}`)
    }
    const plan = planSoupGrid(boxNow, startCounts)
    if (!plan.valid) reasons.push(plan.reason!)
    const minSide = closureMinBoxSide()
    if (Math.min(...boxNow) < minSide) {
      reasons.push(
        `бокс слишком мал для полости-кандидата в везикулу: наименьшая сторона (${Math.min(...boxNow).toFixed(2)}σ) ` +
          `должна быть не меньше ${minSide.toFixed(2)}σ (2×(экв. радиус минимальной полости ` +
          `${equivalentSphereRadius(thresholds.enclosedVolume).toFixed(2)}σ + минимальная толщина стенки ` +
          `${thresholds.headPeakSeparationMin}σ)) — иначе замкнутый объём порога закрытия физически не поместится, ` +
          `независимо от того, сколько шагов прогон сделает`,
      )
    }
    return reasons.length > 0 ? reasons.join('\n') : null
  }

  /** Repaints the particle-count/grid-dims preview and the refusal banner from the CURRENT form
   * fields -- called on load and on every edit to size-select/box-side-input/particle-scale-input,
   * so the cost (and any guard violation) is visible before START is ever clicked, per item 3's own
   * "so the cost is visible in advance" requirement. */
  function refreshSizePreview(): void {
    const { box: boxNow, startCounts } = currentSizeSelection()
    const plan = planSoupGrid(boxNow, startCounts)
    // Per-species breakdown alongside the existing total/grid preview -- so the composition
    // control's own effect (and the cost it implies) is visible BEFORE start, same requirement the
    // total/grid numbers already satisfy. Order follows data/soup.json's own monomer list, not this
    // object's insertion order, so it reads the same regardless of which field the user touched last.
    const perSpecies = soup.monomers.map((m) => `${m.id}:${startCounts[m.id] ?? 0}`).join(' ')
    sizePreviewEl.textContent =
      `частиц: ${plan.N} (${perSpecies}) · сетка соседей: ${plan.dims[0]}×${plan.dims[1]}×${plan.dims[2]} = ${plan.ncells} ячеек`
    const reason = validateSizeSelection(boxNow, startCounts)
    sizeErrorEl.hidden = reason === null
    sizeErrorEl.textContent = reason ?? ''
  }

  // Which species are the medium comes from data/soup.json's own per-species `solvent` flag, never
  // from a hardcoded "W" here -- adding a second solvent species to that file would need no edit.
  const solventIds = soup.monomers.filter((m) => m.solvent ?? false).map((m) => m.id)
  const waterVisibleInput = document.getElementById('water-visible-input') as HTMLInputElement
  waterVisibleInput.checked = SOLVENT_VISIBLE_BY_DEFAULT

  function applySolventVisibility(): void {
    if (!rt.meshes) return
    for (const id of solventIds) {
      const mesh = rt.meshes.monomerMesh[id]
      if (mesh) mesh.visible = waterVisibleInput.checked
    }
  }
  waterVisibleInput.addEventListener('change', applySolventVisibility)

  sizeSelect.addEventListener('change', () => {
    const preset = SIZE_PRESETS[sizeSelect.value]
    baseStart = preset.start ?? soup.start
    boxSideInput.value = String(preset.box[0])
    particleScaleInput.value = '1'
    // Presets stay the default composition too, per this task's own requirement: picking a preset
    // resets the head-count field back to that preset's own O count, exactly like it already resets
    // box side and particle scale -- the user can move it independently again afterward.
    headCountInput.value = String(baseStart.O)
    refreshSizePreview()
  })
  boxSideInput.addEventListener('input', refreshSizePreview)
  particleScaleInput.addEventListener('input', refreshSizePreview)
  headCountInput.addEventListener('input', refreshSizePreview)
  refreshSizePreview()

  // `monomers` is the state every run STARTS in, so offering it as a target makes the run
  // declare success on its first sample. Only stages that require the physics to do something
  // are selectable; the ladder itself (run-readout.ts) still shows all five.
  for (const stage of STAGES.filter((s) => s !== 'monomers')) {
    const opt = document.createElement('option')
    opt.value = stage
    opt.textContent = STAGE_LABEL[stage]
    stageSelect.appendChild(opt)
  }
  stageSelect.value = 'vesicle'

  stepCapInput.value = String(DEFAULT_STEP_CAP)

  // Item 1b (2026-08 crash report): the status line must not name a cause it does not know. Before
  // this fix every 'error' state rendered the SAME hardcoded "GPU не отвечает", true only for a
  // genuine watchdog timeout -- an exception thrown inside the sample loop (e.g. a metric function
  // rejecting bad data) got the identical label, misattributing the failure to a GPU that had, in
  // fact, answered every command. rt.runUI.errorKind (set by failRun, one call site per cause) is
  // what this now reads instead of guessing.
  function errorLabel(): string {
    if (rt.runUI.errorKind === 'timeout') return 'ОШИБКА — GPU не отвечает'
    return `ОШИБКА — сбой в коде: ${rt.runUI.error ?? 'неизвестная ошибка'}`
  }

  function setState(s: RunState): void {
    rt.runUI.state = s
    stateValue.textContent =
      s === 'idle'
        ? 'простой'
        : s === 'running'
          ? 'идёт'
          : s === 'paused'
            ? 'пауза'
            : s === 'error'
              ? errorLabel()
              : 'остановлен'
    startBtn.disabled = s === 'running' || s === 'paused'
    pauseBtn.disabled = s !== 'running' && s !== 'paused'
    pauseBtn.textContent = s === 'paused' ? 'ПРОДОЛЖИТЬ' : 'ПАУЗА'
    stopBtn.disabled = s !== 'running' && s !== 'paused'
    const controlsLocked = s === 'running' || s === 'paused'
    sizeSelect.disabled = controlsLocked
    boxSideInput.disabled = controlsLocked
    particleScaleInput.disabled = controlsLocked
    headCountInput.disabled = controlsLocked
    stageSelect.disabled = controlsLocked
    stepCapInput.disabled = controlsLocked
  }

  function paint(): void {
    readout.paintProgress(rt.runUI, rt.stepCap, rt.bondCount, thresholds, currentElapsedMs(rt))
  }

  async function startRun(): Promise<void> {
    const { box: boxNow, startCounts } = currentSizeSelection()
    const refusal = validateSizeSelection(boxNow, startCounts)
    if (refusal) {
      // Item 3 (2026-08 UI-fixes task): refuse cleanly instead of letting createSoup's own
      // planSoupGrid check throw (or worse, an invalid neighbour grid misbehave on the GPU) -- see
      // validateSizeSelection's own doc comment for what each possible reason means. Nothing about
      // run state changes here: this is refused before anything starts, not a failed run, so the
      // trace log / evidence / step counter from any PREVIOUS run are deliberately left alone.
      sizeErrorEl.hidden = false
      sizeErrorEl.textContent = refusal
      return
    }

    rt.targetStage = stageSelect.value as Stage
    rt.stepCap = Math.max(1, Math.floor(Number(stepCapInput.value) || DEFAULT_STEP_CAP))
    rt.box = boxNow

    rt.runUI.steps = 0
    rt.runUI.stepsPerSecond = 0
    rt.runUI.stage = 'monomers'
    rt.runUI.evidence = { amphiphileFraction: 0, largestAggregateFraction: 0, headPeaks: 0, enclosedVolume: 0, aggregateAnalysis: emptyAggregateAnalysis() }
    rt.runUI.trace = []
    rt.runUI.error = null
    rt.runUI.errorKind = null
    rt.runUI.cavityCount = 0
    rt.runUI.largestCavity = null
    rt.bondCount = 0
    rt.lastCavity = null
    rt.lastCavityCount = 0
    rt.activeElapsedMs = 0
    rt.lastResumeAt = performance.now()
    rt.lastSampleAt = performance.now()
    rt.lastSampleSteps = 0
    rt.sampleDue = true
    rt.autoPausedByVisibility = false

    // The DOM equivalent of the trace/error resets above -- this run's log starts empty, matching
    // rt.runUI.trace being reset just above.
    readout.resetTrace()

    // Defensive: finishRun()/failRun() already dispose() the previous run's system on every path
    // that ends a run (stop, step cap, target stage reached, watchdog, exception). This is a
    // second, belt-and-suspenders release right before handing the shared device to a brand new
    // system, in case some future code path ever calls startRun() again without having gone
    // through one of those -- see SoupSystem.dispose()'s own doc comment for why an undisposed
    // system's GPUBuffers do not otherwise get freed on this shared, memoized device.
    rt.activeSys?.dispose()
    rt.activeSys = null

    setState('running')
    readout.appendTraceLine(`[старт] бокс=${rt.box[0]}×${rt.box[1]}×${rt.box[2]} цель=${rt.targetStage} предел_шагов=${rt.stepCap}`)

    const sys = await createSoup({ box: rt.box, seed: RUN_SEED, kT: params.thermostat.kT, start: startCounts })
    rt.activeSys = sys
    const n = Object.values(startCounts).reduce((a, b) => a + b, 0)
    rt.meshes = runScene.buildMeshes(rt.meshes, soup.monomers, n, rt.box, thresholds.closureCell)
    applySolventVisibility() // fresh meshes start at the default; re-assert whatever the user chose

    rt.runGeneration++
    void getLifecycle().simDriver(sys, rt.runGeneration)
  }

  function togglePause(): void {
    if (rt.runUI.state === 'running') {
      rt.activeElapsedMs = currentElapsedMs(rt) // freeze BEFORE flipping state, see currentElapsedMs's doc
      setState('paused')
      paint()
    } else if (rt.runUI.state === 'paused') {
      rt.autoPausedByVisibility = false
      rt.lastResumeAt = performance.now()
      rt.lastSampleAt = performance.now() // next sample window starts fresh from resume, not spanning the paused gap
      rt.sampleDue = true
      setState('running')
    }
  }

  function stopRun(): void {
    getLifecycle().finishRun('остановлено пользователем')
  }

  startBtn.addEventListener('click', () => {
    startRun().catch((err) => {
      console.error(err)
      readout.appendTraceLine(`[ошибка] ${(err as Error).message}`)
      setState('stopped')
    })
  })
  pauseBtn.addEventListener('click', togglePause)
  stopBtn.addEventListener('click', stopRun)

  // Machine-friendliness requirement: pause automatically when the tab is hidden, resume only when
  // it becomes visible again -- and ONLY if this pause was the automatic kind, never overriding a
  // user's own manual pause.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (rt.runUI.state === 'running') {
        rt.autoPausedByVisibility = true
        rt.activeElapsedMs = currentElapsedMs(rt) // freeze BEFORE flipping state, see currentElapsedMs's doc
        setState('paused')
        paint()
      }
      visibilityNoteEl.classList.add('active')
      visibilityNoteEl.textContent = 'Вкладка скрыта — прогон на автопаузе, GPU не грузится в фоне.'
    } else {
      visibilityNoteEl.classList.remove('active')
      visibilityNoteEl.textContent =
        'Пока эта заметка синяя — вкладка видима, счёт идёт. Как только вкладка уходит из видимости ' +
        '(переключение окна/таба), прогон автоматически ставится на паузу и не грузит GPU в фоне; при ' +
        'возврате на вкладку он сам продолжается — но только если это была автопауза, а не ручная.'
      if (rt.autoPausedByVisibility && rt.runUI.state === 'paused') {
        rt.autoPausedByVisibility = false
        rt.lastResumeAt = performance.now()
        rt.lastSampleAt = performance.now()
        rt.sampleDue = true
        setState('running')
      }
    }
  })

  setState('idle')
  paint()

  return { setState }
}
