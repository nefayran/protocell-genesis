// Run-control page: the user starts a soup run, watches it, pauses it and stops it -- nobody
// starts a long GPU run FOR them from a script. Sibling of viewer/main.ts and viewer/molecular.ts
// (same three.js/orbit-controls structure), but the thing under everyone's control here is the
// *lifecycle* of the run, not just a couple of live sliders.
//
// Two independent loops, on purpose (the brief's own machine-friendliness requirement):
//  - simDriver(): an async loop that calls soup/src/sim.ts's step() in BOUNDED batches (STEP_BATCH),
//    awaited every time -- the await is what lets the event loop (clicks, rAF, visibilitychange)
//    run between batches, so a long run never blocks the UI thread the way one giant step() would.
//  - renderLoop(): a requestAnimationFrame loop that redraws the LATEST snapshot simDriver produced,
//    throttled to RENDER_INTERVAL_MS regardless of how fast simDriver is producing new snapshots --
//    this is the independent render-rate cap the brief asks for. It keeps running even while
//    paused/stopped so OrbitControls stays interactive on the last frame.
// Neither loop ever calls the other; they only share `snapshot` (latest positions/bonds/evidence)
// and `runUI` (the published state object).
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import atomsRaw from '../data/atoms.json'
import { backmapLipid, type BackmapAtom } from '../chem/src/backmap'
import { cavitiesFromPositions, equivalentSphereRadius, type CavityWorld } from '../engine/src/closure'
import { loadParams } from '../engine/src/params'
import { findAmphiphiles, type Amphiphile } from '../soup/src/amphiphile'
import { loadSoup } from '../soup/src/rules'
import { createSoup, planSoupGrid, type SoupSystem } from '../soup/src/sim'
import {
  detectStage,
  HEAD_PEAKS_UNAVAILABLE,
  loadStageThresholds,
  memberIndicesOf,
  positionsFor,
  type Stage,
  type StageEvidence,
} from '../soup/src/stages'

interface ElementInfo {
  vdw: number
  color: string
}
interface AtomsData {
  elements: Record<string, ElementInfo>
}
const atomsData = atomsRaw as unknown as AtomsData

// --- page-level presentation constants (soup/src stays free of these; this is display-only). ------

// Steps advanced per simDriver tick. Bounded on purpose -- see this file's header. Small enough
// that even the "default" ~13100-particle preset finishes a tick in well under a second at this
// engine's measured 2086-2302 steps/s, so the UI (clicks, pause, the render loop) is never starved
// for more than that.
const STEP_BATCH = 250

// Progress numbers (steps/s, elapsed, ETA, evidence, trace) repaint at most this often -- "a few
// times a second, no faster" from the brief. 250ms = 4/s.
const SAMPLE_INTERVAL_MS = 250

// 3D view redraws at most this often, independent of how fast simDriver is producing snapshots --
// "at most a few dozen frames a second" from the brief. 1000/30 caps it at 30fps.
const RENDER_INTERVAL_MS = 1000 / 30

// Scrolling trace is bounded so a long run's log can't grow the DOM without limit.
const MAX_TRACE_LINES = 300

// Watchdog on each simDriver tick's mySys.step(batch) call. getGpu() memoizes ONE GPUDevice for
// the whole page (engine/src/gpu.ts); soup/src/sim.ts's own step() chunks at STEP_CHUNK=1000 so a
// single onSubmittedWorkDone() round trip is already known-safe at that size (see that file's own
// header on the second-large-submission browser bug this project hit once already) -- but nothing
// in this page ever destroyed a finished run's GPUBuffers before starting the next one, so a
// second SoupSystem's own step() could, in principle, hit the SAME "await never settles" failure
// mode this project has already seen once, or any OTHER unexpected exception, and previously that
// left the page silently stuck at "идёт" forever with no visible sign anything was wrong. This
// bounds that: if step() has not returned by this deadline, the run is declared wedged instead of
// waiting forever. 15s is generous margin -- STEP_BATCH=250 measured at ~500-2300 steps/s for
// every preset this page offers (well under 1s/batch) -- while still surfacing a real hang to the
// person watching within a reasonable, interactive time.
const STEP_WATCHDOG_MS = 15_000

// Nanometres per reduced sigma -- same viewer-only convention as viewer/molecular.ts (sigma~1nm is
// the standard literature mapping for this bead model, not a measured quantity; chosen here purely
// so the atomistic reconstruction's coordinates read directly in the same numbers as the coarse
// sim's own box/particle coordinates).
const SIGMA_NM = 1.0

// Half-width (fraction of box x-size) of the fixed spatial band drawn atom-by-atom; amphiphiles
// outside it are drawn as the same coarse spheres every other monomer gets. Fixed (not
// camera-tracking) so which amphiphiles are atomistic is stable frame to frame, exactly the
// reasoning viewer/molecular.ts's own SLICE_HALF_WIDTH uses.
const SLICE_HALF_WIDTH_FRAC = 0.15

// Ball-and-stick sizing, identical convention to viewer/molecular.ts: atoms shrunk below their true
// van-der-Waals radius so bonds stay visible as sticks. Reused for the COARSE monomer spheres too
// (radius/colour both sourced from data/atoms.json), so the atomistic slice and the coarse rest of
// the soup share one visual vocabulary instead of two unrelated size scales.
const ATOM_RADIUS_SCALE = 0.34
const BOND_RADIUS = atomsData.elements.C.vdw * 0.16

// data/soup.json's monomer ids happen to spell out real element symbols for three of the four kinds
// (C=carbon, O=head, H=donor) -- not a coincidence in this model's naming, and reused here so the
// coarse spheres and the atomistic reconstruction draw from the exact same palette. The catalyst
// ("M") has no atomic identity in this coarse-grained model at all; "Na" is an arbitrary, honestly
// documented stand-in chosen only to keep it visually distinct, not a claim about its chemistry.
const MONOMER_ELEMENT: Record<string, string> = { C: 'C', O: 'O', H: 'H', M: 'Na' }

// Cavity voxel cloud: display-only styling for the closure detector's own unreached grid cells (see
// buildCavityDrawing() below). Colour/opacity are chosen only to read as visually distinct from
// every monomer sphere (data/atoms.json's element palette) and every bond cylinder (plain white) --
// a translucent teal that no monomer or bond uses. CAVITY_CUBE_SCALE shrinks each cube slightly
// below the detector's own grid cell (thresholds.closureCell) so adjacent voxels of the same cavity
// still read as a granular cloud instead of one solid slab -- purely cosmetic, never fed back into
// any measurement.
const CAVITY_COLOR = 0x2de6c0
const CAVITY_OPACITY = 0.35
const CAVITY_CUBE_SCALE = 0.85

// Safe upper bounds for InstancedMesh capacities, derived rather than guessed: soup/src/stages.ts's
// own header notes valence is capped at 3, so bonds <= N*3/2 and this file's two-half-cylinder
// convention needs N*3 segments at most. ATOMS_PER_CARBON_ESTIMATE covers the reconstructed
// ball-and-stick backbone (backbone C + its H's, plus the fixed few-atom carboxyl group at one
// end) generously above the ~3 atoms/carbon a plain alkanoic-acid chain works out to.
const MAX_BOND_SEGMENTS_PER_PARTICLE = 3 * 2
const ATOMS_PER_CARBON_ESTIMATE = 5

// --- run-size presets (page-level; the physics/thresholds these feed are all read from
// data/soup.json by createSoup itself -- these are just which knobs of CreateSoupOpts the user is
// offered). "tiny" is deliberately the DEFAULT selection: this page's whole point is to put the
// user in charge of how much machine a run costs, so the safe/cheap choice should not require the
// user to already know to pick it. -----------------------------------------------------------------
interface SizePreset {
  label: string
  box: [number, number, number]
  start?: Record<string, number>
}
const SIZE_PRESETS: Record<string, SizePreset> = {
  tiny: { label: 'малый (проверочный, ~470 частиц)', box: [16, 16, 16], start: { C: 200, O: 50, H: 200, M: 20 } },
  default: { label: 'стандартный бульон (data/soup.json, ~13100 частиц)', box: [30, 30, 30] },
}
const DEFAULT_SIZE_KEY = 'tiny'
const DEFAULT_STEP_CAP = 20_000
const RUN_SEED = 42

const STAGES: Stage[] = ['monomers', 'amphiphiles', 'micelles', 'bilayer', 'vesicle']
const STAGE_LABEL: Record<Stage, string> = {
  monomers: 'monomers',
  amphiphiles: 'amphiphiles',
  micelles: 'micelles',
  bilayer: 'bilayer',
  vesicle: 'vesicle',
}

// 'error': the watchdog (or an unexpected exception in simDriver) declared the run wedged/dead.
// Deliberately its own state rather than reusing 'stopped' with a side flag: a wedged run must be
// visibly DIFFERENT from a normal stop, not just carry an extra field a casual look would miss.
type RunState = 'idle' | 'running' | 'paused' | 'stopped' | 'error'

interface TraceEntry {
  steps: number
  stage: Stage
  evidence: StageEvidence
}

interface RunUI {
  state: RunState
  steps: number
  stepsPerSecond: number
  stage: Stage
  evidence: StageEvidence
  trace: TraceEntry[]
  /** Non-null exactly when state === 'error' -- what the watchdog/catch reported, with the last
   * completed step count folded into the message so the on-screen state itself carries that
   * number (also still readable from `steps`, which the failure path freezes rather than zeroes). */
  error: string | null
  /** Non-null exactly when state === 'error' -- WHICH of the two distinct failure modes this was,
   * so the status line can name what actually happened instead of a fixed guess (item 1b, 2026-08
   * crash report: the watchdog and an exception used to share one hardcoded "GPU не отвечает"
   * label, which was a lie whenever the real cause was an exception in metric code -- see
   * setState()'s own use of this field below). 'timeout': stepWithWatchdog's deadline fired, the
   * GPU-bound step() call itself never returned -- "GPU не отвечает" is an honest description ONLY
   * of this case. 'exception': something in the sample/step loop THREW (its own message is what
   * `error` carries) -- the GPU answered fine; the failure is in this page's own code. */
  errorKind: 'timeout' | 'exception' | null
  /** How many disjoint cavities the closure detector found on the latest sample -- 0 whenever
   * nothing is closed at all (the normal state for most of a run). */
  cavityCount: number
  /** The largest of those cavities' own summary numbers (voxelCount/volume/radius/centre), without
   * its full voxel list -- that list lives on Snapshot.largestCavity for drawing only, not on this
   * UI-readout object. null exactly when cavityCount === 0. */
  largestCavity: { voxelCount: number; volume: number; radius: number; centre: [number, number, number] } | null
}

/** Item 4 (2026-08 UI-fixes task): three "the page must never imply more than it delivers"
 * disclaimers -- honesty-note (step units are reduced τ, not seconds; sps/elapsed are measured, ETA
 * is a forecast), cavity-honesty (a flood-fill pocket is not a vesicle interior), atom-badge (beads
 * are not atoms; the shown atomic detail is reconstructed, not independently simulated) -- used to
 * sit on screen as full paragraphs, always fully expanded. Correct, and required (this project's
 * own honesty rule), but loud: a long-running page kept three multi-sentence blocks permanently
 * open. Collapses each into one short summary line plus a small toggle that reveals the EXACT SAME
 * full text on demand -- nothing here is deleted, and nothing moves into a tooltip (which reliably
 * never gets read, the reason the task rejects that option outright). Remembers the open/closed
 * choice for the rest of this browser SESSION via sessionStorage (not localStorage: a fresh tab
 * should default back to collapsed, not silently inherit a choice made days ago in a different
 * session). Returns a setter for the full text, since one of the three (atom-badge) recomputes its
 * full text every sampled tick -- the summary/toggle DOM nodes stay put; only the hidden full-text
 * node's content changes, so the toggle's own open/closed state is never disturbed by a repaint. */
function initCollapsibleNote(container: HTMLElement, storageKey: string, summary: string): (full: string) => void {
  container.innerHTML =
    `<div class="note-summary"><span>${summary}</span>` +
    `<button class="note-toggle" type="button" aria-expanded="false">ⓘ подробнее</button></div>` +
    `<div class="note-full" hidden></div>`
  const toggleBtn = container.querySelector('.note-toggle') as HTMLButtonElement
  const fullEl = container.querySelector('.note-full') as HTMLElement
  const storeKey = `run-note-expanded:${storageKey}`
  const setExpanded = (expanded: boolean): void => {
    fullEl.hidden = !expanded
    toggleBtn.setAttribute('aria-expanded', String(expanded))
    toggleBtn.textContent = expanded ? 'ⓘ свернуть' : 'ⓘ подробнее'
    sessionStorage.setItem(storeKey, expanded ? '1' : '0')
  }
  toggleBtn.addEventListener('click', () => setExpanded(fullEl.hidden))
  setExpanded(sessionStorage.getItem(storeKey) === '1')
  return (full: string) => {
    fullEl.textContent = full
  }
}

function main(): void {
  const params = loadParams()
  const soup = loadSoup()

  // --- DOM handles ---------------------------------------------------------------------------
  const sizeSelect = document.getElementById('size-select') as HTMLSelectElement
  const boxSideInput = document.getElementById('box-side-input') as HTMLInputElement
  const particleScaleInput = document.getElementById('particle-scale-input') as HTMLInputElement
  const sizePreviewEl = document.getElementById('size-preview') as HTMLElement
  const sizeErrorEl = document.getElementById('size-error') as HTMLElement
  const stageSelect = document.getElementById('stage-select') as HTMLSelectElement
  const stepCapInput = document.getElementById('step-cap') as HTMLInputElement
  const startBtn = document.getElementById('start-btn') as HTMLButtonElement
  const pauseBtn = document.getElementById('pause-btn') as HTMLButtonElement
  const stopBtn = document.getElementById('stop-btn') as HTMLButtonElement

  const stateValue = document.getElementById('state-value') as HTMLElement
  const stepsValue = document.getElementById('steps-value') as HTMLElement
  const spsValue = document.getElementById('sps-value') as HTMLElement
  const elapsedValue = document.getElementById('elapsed-value') as HTMLElement
  const etaValue = document.getElementById('eta-value') as HTMLElement
  const stageValue = document.getElementById('stage-value') as HTMLElement
  const ladderEl = document.getElementById('ladder') as HTMLElement
  const evidenceEl = document.getElementById('evidence') as HTMLElement
  const traceLogEl = document.getElementById('trace-log') as HTMLElement
  const visibilityNoteEl = document.getElementById('visibility-note') as HTMLElement
  const honestyNoteEl = document.getElementById('honesty-note') as HTMLElement
  const atomBadgeEl = document.getElementById('atom-badge') as HTMLElement

  const cavityCountEl = document.getElementById('cavity-count') as HTMLElement
  const cavityVoxelsEl = document.getElementById('cavity-voxels') as HTMLElement
  const cavityVolumeEl = document.getElementById('cavity-volume') as HTMLElement
  const cavityRadiusEl = document.getElementById('cavity-radius') as HTMLElement
  const cavityCentreEl = document.getElementById('cavity-centre') as HTMLElement
  const cavityAimBtn = document.getElementById('cavity-aim-btn') as HTMLButtonElement
  const cavityHonestyEl = document.getElementById('cavity-honesty') as HTMLElement

  const thresholds = loadStageThresholds()
  initCollapsibleNote(
    cavityHonestyEl,
    'cavity-honesty',
    'ЧЕСТНО: полость из клеток сетки — карман между бидами, не внутренность везикулы.',
  )(
    `ЧЕСТНО: полость из нескольких клеток сетки — это карман между бидами, а не внутренность ` +
      `везикулы. Порог считается физически: минимальный объём — это шар радиусом, ПРЕВЫШАЮЩИМ ` +
      `измеренную толщину бислоя (data/literature.json's closure gate, ≈${thresholds.enclosedVolume.toFixed(2)} σ³ ` +
      `≈ ${equivalentSphereRadius(thresholds.enclosedVolume).toFixed(2)}σ экв. радиуса) — меньшая полость ` +
      `физически не может быть внутренностью мембраны, которая должна её огибать.`,
  )

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

  /** `baseStart` scaled by `scale` and rounded to whole particles per species -- the "particle
   * scale" control's own effect. Never negative (a scale below what rounds to 0 for every species
   * is caught by planSoupGrid's own N===0 downstream check in createSoup, not specially guarded
   * here). */
  function scaledStart(scale: number): Record<string, number> {
    return Object.fromEntries(Object.entries(baseStart).map(([id, n]) => [id, Math.max(0, Math.round(n * scale))]))
  }

  /** The box/startCounts the CURRENT form fields describe, regardless of whether a run has ever
   * started -- read fresh from the inputs every time (never the run's own `box`/`stepCap` state
   * below, which only updates on START) so the preview always reflects what the user is looking
   * at right now. Falls back to the tiny preset's own numbers for anything that fails to parse
   * (empty field, non-numeric typing mid-edit) rather than propagating NaN into the preview/guard. */
  function currentSizeSelection(): { box: [number, number, number]; startCounts: Record<string, number> } {
    const side = Number(boxSideInput.value)
    const scale = Number(particleScaleInput.value)
    const boxSide = Number.isFinite(side) && side > 0 ? side : SIZE_PRESETS[DEFAULT_SIZE_KEY].box[0]
    const particleScale = Number.isFinite(scale) && scale > 0 ? scale : 1
    return { box: [boxSide, boxSide, boxSide], startCounts: scaledStart(particleScale) }
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
    sizePreviewEl.textContent =
      `частиц: ${plan.N} · сетка соседей: ${plan.dims[0]}×${plan.dims[1]}×${plan.dims[2]} = ${plan.ncells} ячеек`
    const reason = validateSizeSelection(boxNow, startCounts)
    sizeErrorEl.hidden = reason === null
    sizeErrorEl.textContent = reason ?? ''
  }

  sizeSelect.addEventListener('change', () => {
    const preset = SIZE_PRESETS[sizeSelect.value]
    baseStart = preset.start ?? soup.start
    boxSideInput.value = String(preset.box[0])
    particleScaleInput.value = '1'
    refreshSizePreview()
  })
  boxSideInput.addEventListener('input', refreshSizePreview)
  particleScaleInput.addEventListener('input', refreshSizePreview)
  refreshSizePreview()

  // `monomers` is the state every run STARTS in, so offering it as a target makes the run
  // declare success on its first sample. Only stages that require the physics to do something
  // are selectable; the ladder below still shows all five.
  for (const stage of STAGES.filter((s) => s !== 'monomers')) {
    const opt = document.createElement('option')
    opt.value = stage
    opt.textContent = STAGE_LABEL[stage]
    stageSelect.appendChild(opt)
  }
  stageSelect.value = 'vesicle'

  stepCapInput.value = String(DEFAULT_STEP_CAP)

  ladderEl.innerHTML = STAGES.map((s) => `<div class="stage" data-stage="${s}">${STAGE_LABEL[s]}</div>`).join('')

  initCollapsibleNote(
    honestyNoteEl,
    'honesty',
    'ОЦЕНКА vs ИЗМЕРЕНО: шаги — приведённые единицы (τ), не секунды; ETA — прогноз, не гарантия.',
  )(
    `ОЦЕНКА vs ИЗМЕРЕНО: «шагов/с» и «прошло (реал.)» измерены по системным часам браузера; ` +
      `«осталось (оцен.)» — прогноз из текущей измеренной скорости, не гарантия. Единицы шагов — ` +
      `ПРИВЕДЁННЫЕ (τ, σ=1), это не секунды реального мира: κ_t = ${soup.kappaT} — единственная явная ` +
      `калибровка временной шкалы модели (data/soup.json), настоящей секундной привязки для бульона нет.`,
  )

  // atom-badge's full text is recomputed every sampled tick (it names how many amphiphiles got the
  // atomistic treatment THIS tick) -- see simDriver's own use of this setter below. The summary/
  // toggle themselves are built once, here, so the toggle's open/closed choice survives every repaint.
  const setAtomBadgeFull = initCollapsibleNote(
    atomBadgeEl,
    'atom-badge',
    'Бусины — не атомы; показанная атомная детализация реконструирована, не симулирована отдельно.',
  )

  // --- three.js scene --------------------------------------------------------------------------
  const canvas = document.getElementById('scene') as HTMLCanvasElement
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setSize(window.innerWidth, window.innerHeight)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b0d10)

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true

  scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const sun = new THREE.DirectionalLight(0xffffff, 0.9)
  sun.position.set(1, 1.4, 1)
  scene.add(sun)

  /** Item 2 (2026-08 UI-fixes task): the near/far planes used to be fixed (0.1/1000) -- sized for
   * the two old hardcoded presets (box 16/30), so they never had to move. Item 3 lets the user pick
   * an arbitrary box side now, and a fixed far=1000 clips a large box's own far corner the moment
   * the box's own diagonal (times frameCamera's own ~2x placement below) approaches it, exactly
   * "nothing disappears by clipping" in reverse -- while a fixed near=0.1 on a TINY box wastes most
   * of the depth buffer's precision on distances the camera is never placed within. Both planes are
   * now derived from the box's own diagonal every time this is called (on every run start, since
   * box can change run to run), not tied to either preset. */
  function frameCamera(box: [number, number, number]): void {
    const center = new THREE.Vector3(box[0] / 2, box[1] / 2, box[2] / 2)
    const diag = Math.sqrt(box[0] ** 2 + box[1] ** 2 + box[2] ** 2)
    camera.near = Math.max(diag / 1000, 1e-3)
    camera.far = diag * 20
    camera.updateProjectionMatrix()
    camera.position.set(box[0] * 1.6, box[1] * 1.2, box[2] * 1.4)
    camera.lookAt(center)
    controls.target.copy(center)
    controls.update()
  }

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  })

  function elementColor(element: string): THREE.Color {
    const info = atomsData.elements[element]
    if (!info) throw new Error(`viewer/run: неизвестный элемент "${element}" в data/atoms.json`)
    return new THREE.Color(info.color)
  }
  function elementRadius(element: string): number {
    const info = atomsData.elements[element]
    if (!info) throw new Error(`viewer/run: неизвестный элемент "${element}" в data/atoms.json`)
    return info.vdw * ATOM_RADIUS_SCALE
  }

  // Instanced meshes are (re)built per run, sized from THAT run's own N -- a fresh run can pick a
  // different size preset, so capacities must not be fixed at module load.
  interface SceneMeshes {
    monomerMesh: Record<string, THREE.InstancedMesh>
    bondMesh: THREE.InstancedMesh
    atomMesh: THREE.InstancedMesh
    atomBondMesh: THREE.InstancedMesh
    cavityMesh: THREE.InstancedMesh
  }
  let meshes: SceneMeshes | null = null

  // Item 2 (2026-08 UI-fixes task): every InstancedMesh below gets frustumCulled = false. three.js
  // culls a mesh against its GEOMETRY's own bounding sphere transformed by the MESH's own
  // matrixWorld -- for an InstancedMesh that mesh-level transform is the identity at the scene
  // origin (per-instance transforms live in instanceMatrix, which the built-in frustum check never
  // looks at), so the sphere it actually tests is a unit-ish sphere sitting at (0,0,0), nowhere near
  // where the thousands of actual instances are scattered across the box. Depending on camera
  // distance/angle that coincidence can go either way -- the whole mesh vanishes while its instances
  // are plainly on screen, or (rarer) it stays "visible" by luck -- which is exactly the "scene
  // vanishes at some zoom levels and angles" symptom. Maintaining a correct per-frame bounding
  // sphere covering the whole box would work too, but these five meshes are the entire draw call
  // count of this page; disabling culling on them costs nothing measurable and removes the bug
  // class outright rather than chasing whichever camera pose exposes it next.
  function buildMeshes(n: number, box: [number, number, number]): SceneMeshes {
    if (meshes) {
      for (const m of Object.values(meshes.monomerMesh)) scene.remove(m)
      scene.remove(meshes.bondMesh, meshes.atomMesh, meshes.atomBondMesh, meshes.cavityMesh)
    }
    const monomerMesh: Record<string, THREE.InstancedMesh> = {}
    for (const m of soup.monomers) {
      const element = MONOMER_ELEMENT[m.id]
      const geo = new THREE.SphereGeometry(elementRadius(element), 10, 8)
      const mat = new THREE.MeshStandardMaterial({ color: elementColor(element) })
      const mesh = new THREE.InstancedMesh(geo, mat, n)
      mesh.count = 0
      mesh.frustumCulled = false
      monomerMesh[m.id] = mesh
      scene.add(mesh)
    }
    const bondGeo = new THREE.CylinderGeometry(1, 1, 1, 6)
    const bondMat = new THREE.MeshStandardMaterial({ color: 0xffffff })
    const bondMesh = new THREE.InstancedMesh(bondGeo, bondMat, Math.max(1, n * MAX_BOND_SEGMENTS_PER_PARTICLE))
    bondMesh.count = 0
    bondMesh.frustumCulled = false
    scene.add(bondMesh)

    const atomCap = Math.max(1, n * ATOMS_PER_CARBON_ESTIMATE)
    const atomGeo = new THREE.SphereGeometry(1, 10, 8)
    const atomMat = new THREE.MeshStandardMaterial({ color: 0xffffff })
    const atomMesh = new THREE.InstancedMesh(atomGeo, atomMat, atomCap)
    atomMesh.count = 0
    atomMesh.frustumCulled = false
    scene.add(atomMesh)

    const atomBondGeo = new THREE.CylinderGeometry(1, 1, 1, 6)
    const atomBondMat = new THREE.MeshStandardMaterial({ color: 0xffffff })
    const atomBondMesh = new THREE.InstancedMesh(atomBondGeo, atomBondMat, atomCap * 2)
    atomBondMesh.count = 0
    atomBondMesh.frustumCulled = false
    scene.add(atomBondMesh)

    // Cavity voxel cloud (see CAVITY_COLOR's own doc comment): capacity is the box's own total
    // grid-cell count at this run's closure resolution (thresholds.closureCell) -- a cavity can
    // never have more voxels than the grid itself, so this is a structural bound, not a guess (same
    // reasoning as MAX_BOND_SEGMENTS_PER_PARTICLE/ATOMS_PER_CARBON_ESTIMATE above).
    const cellsPerAxis = box.map((side) => Math.max(1, Math.ceil(side / thresholds.closureCell)))
    const cavityCap = Math.max(1, cellsPerAxis[0] * cellsPerAxis[1] * cellsPerAxis[2])
    const cavityGeo = new THREE.BoxGeometry(1, 1, 1)
    const cavityMat = new THREE.MeshBasicMaterial({
      color: CAVITY_COLOR,
      transparent: true,
      opacity: CAVITY_OPACITY,
      depthWrite: false,
    })
    const cavityMesh = new THREE.InstancedMesh(cavityGeo, cavityMat, cavityCap)
    cavityMesh.count = 0
    cavityMesh.frustumCulled = false
    scene.add(cavityMesh)

    frameCamera(box)
    return { monomerMesh, bondMesh, atomMesh, atomBondMesh, cavityMesh }
  }

  const dummy = new THREE.Object3D()
  const up = new THREE.Vector3(0, 1, 0)

  /** Shortest periodic image of `d` along one axis of length `L` (minimum-image convention -- the
   * same rule the physics itself uses, engine/src/sim.ts's own mi3 comment). Positions read back
   * from the GPU are wrapped into [0,box); drawing a straight cylinder between two RAW wrapped
   * positions would show a bond stretching across the whole box whenever it happens to straddle a
   * periodic face. This is a display-only correction -- it never touches soup/src's own state. */
  function wrap(d: number, L: number): number {
    if (d > L / 2) return d - L
    if (d < -L / 2) return d + L
    return d
  }

  function drawHalfCylinders(
    mesh: THREE.InstancedMesh,
    segments: { from: THREE.Vector3; to: THREE.Vector3; color: THREE.Color }[],
  ): void {
    let i = 0
    for (const seg of segments) {
      const dir = seg.to.clone().sub(seg.from)
      const len = dir.length()
      if (len < 1e-9) continue
      dummy.position.copy(seg.from).add(dir.clone().multiplyScalar(0.5))
      dummy.quaternion.setFromUnitVectors(up, dir.clone().normalize())
      dummy.scale.set(BOND_RADIUS, len, BOND_RADIUS)
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
      mesh.setColorAt(i, seg.color)
      i++
      if (i >= mesh.instanceMatrix.count) break // capacity safety, see buildMeshes
    }
    mesh.count = Math.min(i, mesh.instanceMatrix.count)
    mesh.instanceMatrix.needsUpdate = true
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
  }

  /** One redraw of the whole scene from a snapshot -- positions/bonds/box for the coarse view, plus
   * which amphiphiles (if any) fall inside the atomistic slice this snapshot. Pure DOM/three.js
   * side effects; never touches the GPU sim itself (that is simDriver's job only). */
  function draw(snap: Snapshot): void {
    if (!meshes) return
    const { particles, bonds, box, atomistic, atomisticSet } = snap
    const n = particles.length / 4

    // --- coarse monomer spheres, one InstancedMesh per monomer id, skipping anything drawn
    // atom-by-atom instead ------------------------------------------------------------------
    const counters: Record<string, number> = {}
    for (const m of soup.monomers) counters[m.id] = 0
    for (let i = 0; i < n; i++) {
      if (atomisticSet.has(i)) continue
      const kind = Math.round(particles[i * 4 + 3])
      const m = soup.monomers[kind]
      const mesh = meshes.monomerMesh[m.id]
      dummy.position.set(particles[i * 4], particles[i * 4 + 1], particles[i * 4 + 2])
      dummy.scale.setScalar(1)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      const idx = counters[m.id]++
      if (idx < mesh.instanceMatrix.count) mesh.setMatrixAt(idx, dummy.matrix)
    }
    for (const m of soup.monomers) {
      const mesh = meshes.monomerMesh[m.id]
      mesh.count = Math.min(counters[m.id], mesh.instanceMatrix.count)
      mesh.instanceMatrix.needsUpdate = true
    }

    // --- coarse bonds: both ends outside the atomistic slice only (the atomistic reconstruction
    // draws its own sticks for the rest) -----------------------------------------------------
    const coarseSegments: { from: THREE.Vector3; to: THREE.Vector3; color: THREE.Color }[] = []
    for (let k = 0; k < bonds.length; k += 2) {
      const ia = bonds[k]
      const ib = bonds[k + 1]
      if (atomisticSet.has(ia) || atomisticSet.has(ib)) continue
      const pA = new THREE.Vector3(particles[ia * 4], particles[ia * 4 + 1], particles[ia * 4 + 2])
      const rawDelta = new THREE.Vector3(
        particles[ib * 4] - pA.x,
        particles[ib * 4 + 1] - pA.y,
        particles[ib * 4 + 2] - pA.z,
      )
      const delta = new THREE.Vector3(wrap(rawDelta.x, box[0]), wrap(rawDelta.y, box[1]), wrap(rawDelta.z, box[2]))
      const pB = pA.clone().add(delta)
      const mid = pA.clone().add(pB).multiplyScalar(0.5)
      const colorA = elementColor(MONOMER_ELEMENT[soup.monomers[Math.round(particles[ia * 4 + 3])].id])
      const colorB = elementColor(MONOMER_ELEMENT[soup.monomers[Math.round(particles[ib * 4 + 3])].id])
      coarseSegments.push({ from: pA, to: mid, color: colorA }, { from: mid, to: pB, color: colorB })
    }
    drawHalfCylinders(meshes.bondMesh, coarseSegments)

    // --- atomistic ball-and-stick for the slice's amphiphiles ---------------------------------
    let ai = 0
    const atomSegments: { from: THREE.Vector3; to: THREE.Vector3; color: THREE.Color }[] = []
    for (const { atoms, bonds: aBonds } of atomistic) {
      for (const a of atoms) {
        dummy.position.set(a.position[0], a.position[1], a.position[2])
        dummy.scale.setScalar(elementRadius(a.element))
        dummy.rotation.set(0, 0, 0)
        dummy.updateMatrix()
        if (ai < meshes.atomMesh.instanceMatrix.count) {
          meshes.atomMesh.setMatrixAt(ai, dummy.matrix)
          meshes.atomMesh.setColorAt(ai, elementColor(a.element))
        }
        ai++
      }
      for (const [ia, ib] of aBonds) {
        const atomA = atoms[ia]
        const atomB = atoms[ib]
        const pA = new THREE.Vector3(...atomA.position)
        const pB = new THREE.Vector3(...atomB.position)
        const mid = pA.clone().add(pB).multiplyScalar(0.5)
        atomSegments.push(
          { from: pA, to: mid, color: elementColor(atomA.element) },
          { from: mid, to: pB, color: elementColor(atomB.element) },
        )
      }
    }
    meshes.atomMesh.count = Math.min(ai, meshes.atomMesh.instanceMatrix.count)
    meshes.atomMesh.instanceMatrix.needsUpdate = true
    if (meshes.atomMesh.instanceColor) meshes.atomMesh.instanceColor.needsUpdate = true
    drawHalfCylinders(meshes.atomBondMesh, atomSegments)

    // --- largest cavity's voxel cloud (see CAVITY_COLOR's own doc comment) -------------------
    const cavityMesh = meshes.cavityMesh
    const cubeSide = thresholds.closureCell * CAVITY_CUBE_SCALE
    const voxels = snap.largestCavity?.voxelCentres ?? null
    const voxelCount = voxels ? voxels.length / 3 : 0
    const cavityInstances = Math.min(voxelCount, cavityMesh.instanceMatrix.count)
    for (let k = 0; k < cavityInstances; k++) {
      dummy.position.set(voxels![k * 3], voxels![k * 3 + 1], voxels![k * 3 + 2])
      dummy.scale.setScalar(cubeSide)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      cavityMesh.setMatrixAt(k, dummy.matrix)
    }
    cavityMesh.count = cavityInstances
    cavityMesh.instanceMatrix.needsUpdate = true

    controls.update()
    renderer.render(scene, camera)
  }

  // --- run lifecycle -----------------------------------------------------------------------------
  interface Snapshot {
    particles: Float32Array
    bonds: Uint32Array
    box: [number, number, number]
    atomistic: { atoms: BackmapAtom[]; bonds: [number, number][] }[]
    atomisticSet: Set<number>
    amphiphileCount: number
    /** The largest disjoint cavity the closure detector found on this snapshot's amphiphile-member
     * positions (the exact same "wall" material detectStage's own enclosedVolume measures), already
     * mapped back into this snapshot's own (un-recentred) coordinate frame -- see
     * cavitiesFromPositions' own doc comment. null when no cavity was found at all. Recomputed only
     * on sampled ticks (SAMPLE_INTERVAL_MS), same throttle as the stage/evidence readout -- the grid
     * flood is cheap, but there is no reason to redo it faster than the numbers it feeds ever
     * repaint. */
    largestCavity: CavityWorld | null
    cavityCount: number
  }

  const runUI: RunUI = {
    state: 'idle',
    steps: 0,
    stepsPerSecond: 0,
    stage: 'monomers',
    evidence: { amphiphileFraction: 0, largestAggregateFraction: 0, headPeaks: 0, enclosedVolume: 0 },
    trace: [],
    error: null,
    errorKind: null,
    cavityCount: 0,
    largestCavity: null,
  }
  ;(window as unknown as { runUI: RunUI }).runUI = runUI

  let targetStage: Stage = 'vesicle'
  let stepCap = DEFAULT_STEP_CAP
  let box: [number, number, number] = SIZE_PRESETS[DEFAULT_SIZE_KEY].box
  let bondCount = 0

  // --- test-support hook (item 2, 2026-08 UI-fixes task) ------------------------------------------
  // tests/run-ui.test.ts's camera sweep needs to move the camera through several distances/angles
  // around the CURRENT box and check that something non-background actually rendered at each one --
  // there is no DOM state that proves "the instanced meshes are visible", the only ground truth is
  // the rendered pixels, read back here via WebGL readPixels (never --dump-dom, per this task's own
  // constraint) so the test stays inside page.evaluate the same way every other assertion here does.
  // Exposed the same way `runUI` already is: a small, explicitly-named object on `window`, not a
  // production feature.
  interface SceneDebugHooks {
    /** Places the camera at `distanceScale` * the CURRENT box's own diagonal from its centre, at
     * spherical angles (thetaDeg around the vertical axis, phiDeg from it), looking at the centre,
     * then renders one frame immediately so nonBackgroundPixelFraction() below has something fresh
     * to read. */
    setCameraOrbit(distanceScale: number, thetaDeg: number, phiDeg: number): void
    /** Fraction of the canvas's own pixels that differ from the scene's background colour
     * (0x0b0d10) by more than a small tolerance (float/antialiasing noise) -- 0 means "the canvas
     * shows nothing but empty background", which is exactly the item-2 symptom this guards. */
    nonBackgroundPixelFraction(): number
  }
  const sceneDebug: SceneDebugHooks = {
    setCameraOrbit(distanceScale, thetaDeg, phiDeg) {
      const center = new THREE.Vector3(box[0] / 2, box[1] / 2, box[2] / 2)
      const diag = Math.sqrt(box[0] ** 2 + box[1] ** 2 + box[2] ** 2)
      const r = diag * distanceScale
      const theta = (thetaDeg * Math.PI) / 180
      const phi = (phiDeg * Math.PI) / 180
      camera.position.set(
        center.x + r * Math.sin(phi) * Math.cos(theta),
        center.y + r * Math.cos(phi),
        center.z + r * Math.sin(phi) * Math.sin(theta),
      )
      camera.lookAt(center)
      controls.target.copy(center)
      controls.update()
      if (latestSnapshot) draw(latestSnapshot)
      else renderer.render(scene, camera)
    },
    nonBackgroundPixelFraction() {
      const gl = renderer.getContext()
      const w = canvas.width
      const h = canvas.height
      const buf = new Uint8Array(w * h * 4)
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, buf)
      const bg = [0x0b, 0x0d, 0x10]
      let diff = 0
      for (let i = 0; i < buf.length; i += 4) {
        if (Math.abs(buf[i] - bg[0]) > 6 || Math.abs(buf[i + 1] - bg[1]) > 6 || Math.abs(buf[i + 2] - bg[2]) > 6) diff++
      }
      return diff / (w * h)
    },
  }
  ;(window as unknown as { sceneDebug: SceneDebugHooks }).sceneDebug = sceneDebug
  // Cavity info persists ACROSS ticks between samples (unlike evidence/stage, which only ever
  // exist as of the last sample too, but are read straight off runUI) -- draw() reads
  // Snapshot.largestCavity every rendered frame, including the many ticks between two samples, so
  // without this the voxel cloud would flicker to "nothing" every tick that is not itself a sample.
  let lastCavity: CavityWorld | null = null
  let lastCavityCount = 0

  // Wall-clock bookkeeping: elapsed only accumulates while ACTUALLY running (not paused, not
  // hidden-auto-paused) -- pausing must freeze the clock, not just stop the step counter, or "прошло"
  // would silently include idle time and make "шагов/с" look wrong.
  let activeElapsedMs = 0
  let lastResumeAt = 0
  let lastSampleAt = 0
  let lastSampleSteps = 0
  // Forces the very NEXT sim tick to sample/paint immediately regardless of SAMPLE_INTERVAL_MS --
  // set on every start/resume so progress is never blank waiting for the first 250ms to elapse
  // (the throttle is a ceiling on repaint RATE, not a floor before the first paint appears).
  let sampleDue = true

  // Auto-pause on tab hidden vs a user's own manual pause are different things: only an
  // auto-pause resumes itself when the tab comes back; a manual pause must stay paused until the
  // user clicks resume themselves.
  let autoPausedByVisibility = false

  // Runs monotonically upward across a run's whole lifetime; simDriver checks it each iteration so
  // STOP can cancel an in-flight tick's *next* iteration immediately -- see runGeneration's use
  // below for why a stale simDriver from a previous run can never write into a new run's state.
  let runGeneration = 0

  let latestSnapshot: Snapshot | null = null

  // The currently live SoupSystem, if any -- tracked at this scope (not just simDriver's own local
  // `mySys` parameter) purely so startRun() can dispose() the PREVIOUS run's system before handing
  // the device to a new one. See SoupSystem.dispose()'s own doc comment (soup/src/sim.ts) for why
  // this matters: getGpu() memoizes one GPUDevice for the whole page, so without this, every
  // "start a new run" click would leak the just-finished run's ~20 GPUBuffers forever.
  let activeSys: SoupSystem | null = null

  // Item 1b (2026-08 crash report): the status line must not name a cause it does not know. Before
  // this fix every 'error' state rendered the SAME hardcoded "GPU не отвечает", true only for a
  // genuine watchdog timeout -- an exception thrown inside the sample loop (e.g. a metric function
  // rejecting bad data) got the identical label, misattributing the failure to a GPU that had, in
  // fact, answered every command. runUI.errorKind (set by failRun, one call site per cause) is what
  // this now reads instead of guessing.
  function errorLabel(): string {
    if (runUI.errorKind === 'timeout') return 'ОШИБКА — GPU не отвечает'
    return `ОШИБКА — сбой в коде: ${runUI.error ?? 'неизвестная ошибка'}`
  }

  function setState(s: RunState): void {
    runUI.state = s
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
    stageSelect.disabled = controlsLocked
    stepCapInput.disabled = controlsLocked
  }

  /** Wall-clock elapsed while ACTUALLY running, live: accumulated time up to the last
   * pause/resume/stop transition (`activeElapsedMs`) plus, only while currently running, the time
   * since the last resume. Computed fresh on every read rather than mutated in the hot loop, so
   * pause/resume/stop can each freeze or resume the clock by touching `activeElapsedMs` and
   * `lastResumeAt` exactly once, with no double-counting between a pause and a later stop. */
  function currentElapsedMs(): number {
    return activeElapsedMs + (runUI.state === 'running' ? performance.now() - lastResumeAt : 0)
  }

  function formatElapsed(ms: number): string {
    const totalSec = Math.floor(ms / 1000)
    const m = Math.floor(totalSec / 60)
    const s = totalSec % 60
    return `${m}:${String(s).padStart(2, '0')}`
  }

  function updateLadder(reachedIdx: number): void {
    for (const el of Array.from(ladderEl.children) as HTMLElement[]) {
      const stage = el.dataset.stage as Stage
      const idx = STAGES.indexOf(stage)
      el.classList.toggle('reached', idx <= reachedIdx)
      el.classList.toggle('current', idx === reachedIdx)
    }
  }

  function appendTraceLine(text: string): void {
    traceLogEl.textContent += (traceLogEl.textContent ? '\n' : '') + text
    const lines = traceLogEl.textContent.split('\n')
    if (lines.length > MAX_TRACE_LINES) traceLogEl.textContent = lines.slice(lines.length - MAX_TRACE_LINES).join('\n')
    traceLogEl.scrollTop = traceLogEl.scrollHeight
  }

  function paintProgress(): void {
    stepsValue.textContent = String(runUI.steps)
    spsValue.textContent = runUI.stepsPerSecond > 0 ? runUI.stepsPerSecond.toFixed(0) : '–'
    elapsedValue.textContent = formatElapsed(currentElapsedMs())
    const remaining = stepCap - runUI.steps
    etaValue.textContent =
      runUI.stepsPerSecond > 0 && remaining > 0 ? `~${formatElapsed((remaining / runUI.stepsPerSecond) * 1000)}` : '–'
    stageValue.textContent = STAGE_LABEL[runUI.stage]
    updateLadder(STAGES.indexOf(runUI.stage))
    const ev = runUI.evidence
    evidenceEl.innerHTML =
      `<div class="row"><span>доля амфифилов</span><span>${ev.amphiphileFraction.toFixed(4)}</span></div>` +
      `<div class="row"><span>доля крупн. агрегата</span><span>${ev.largestAggregateFraction.toFixed(4)}</span></div>` +
      `<div class="row"><span>пики голов</span><span>${ev.headPeaks === HEAD_PEAKS_UNAVAILABLE ? 'н/д' : ev.headPeaks}</span></div>` +
      `<div class="row"><span>замкн. объём</span><span>${ev.enclosedVolume.toFixed(4)}</span></div>` +
      `<div class="row"><span>связей</span><span>${bondCount}</span></div>`

    cavityCountEl.textContent = String(runUI.cavityCount)
    const largest = runUI.largestCavity
    cavityVoxelsEl.textContent = largest ? String(largest.voxelCount) : '–'
    cavityVolumeEl.textContent = largest ? largest.volume.toFixed(4) : '–'
    cavityRadiusEl.textContent = largest ? largest.radius.toFixed(3) : '–'
    cavityCentreEl.textContent = largest
      ? `${largest.centre[0].toFixed(1)}, ${largest.centre[1].toFixed(1)}, ${largest.centre[2].toFixed(1)}`
      : '–'
    cavityAimBtn.disabled = !largest
  }

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
      // molecule stretched across the whole box (same reasoning as this file's own wrap() above).
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
   * see STEP_WATCHDOG_MS's own doc comment for why an unconditional await is exactly the shape of
   * bug this project has already hit once (a second big submission's onSubmittedWorkDone() that
   * never settled). Returns 'ok' if step() won the race; 'timeout' if the deadline did. If step()
   * itself REJECTS (throws), that rejection propagates out of this function's own `await` as a
   * normal exception -- Promise.race does not swallow it, so simDriver's try/catch below still
   * sees it. A step() that wins the race late (after a timeout was already declared) is left to
   * settle on its own; nothing reads its result at that point, and Promise.race attaching its own
   * reaction to it is enough that a late rejection is not reported as an unhandled one either. */
  async function stepWithWatchdog(mySys: SoupSystem, batch: number): Promise<'ok' | 'timeout'> {
    let timer: ReturnType<typeof setTimeout> | undefined
    const timeout = new Promise<'timeout'>((resolve) => {
      timer = setTimeout(() => resolve('timeout'), STEP_WATCHDOG_MS)
    })
    const result = await Promise.race([mySys.step(batch).then(() => 'ok' as const), timeout])
    clearTimeout(timer)
    return result
  }

  /** Async loop: advances the sim in bounded STEP_BATCH chunks, sampling evidence/trace at most
   * every SAMPLE_INTERVAL_MS. Every iteration checks `myGeneration` against the live
   * `runGeneration` and `runUI.state` -- a STOP (which bumps runGeneration) or a PAUSE makes the
   * very next check exit/idle instead of this loop ever touching sys.step() again. That is the
   * whole mechanism by which STOP "releases the GPU": WebGPU has no persistent kernel between
   * command-buffer submissions, so once this loop stops calling step(), no further work is ever
   * submitted to the device and GPU load returns to whatever the browser/OS shows at idle.
   *
   * The whole body runs under one try/catch: previously an exception ANYWHERE in this loop (a
   * genuinely wedged step() surfaced via stepWithWatchdog's 'timeout', or any other unexpected
   * throw) left `runUI.state` stuck at 'running' forever with nothing on screen to say so -- the
   * run LOOKED alive because nothing ever told it otherwise. failRun() below is what makes a dead
   * run look dead. */
  async function simDriver(mySys: SoupSystem, myGeneration: number): Promise<void> {
    try {
      while (true) {
        if (runGeneration !== myGeneration) return // superseded by a STOP (and possibly a new run)
        if (runUI.state !== 'running') {
          // Paused (manually or by visibility): yield without stepping, poll cheaply.
          await new Promise((r) => setTimeout(r, 50))
          continue
        }
        const batch = Math.min(STEP_BATCH, stepCap - runUI.steps)
        if (batch <= 0) {
          finishRun('предел шагов достигнут')
          return
        }
        const stepResult = await stepWithWatchdog(mySys, batch)
        if (runGeneration !== myGeneration) return
        if (stepResult === 'timeout') {
          failRun(`GPU не отвечает: step() не вернулся за ${(STEP_WATCHDOG_MS / 1000).toFixed(0)}с`, 'timeout')
          return
        }
        runUI.steps += batch

        const pos = await mySys.particles()
        const bnds = await mySys.bonds()
        if (runGeneration !== myGeneration) return
        bondCount = bnds.length / 2

        const now = performance.now()
        const amphiphiles = findAmphiphiles(pos, bnds, soup.monomers)
        const { atomistic, atomisticSet } = buildAtomisticSlice(amphiphiles, pos, box)
        latestSnapshot = {
          particles: pos,
          bonds: bnds,
          box,
          atomistic,
          atomisticSet,
          amphiphileCount: amphiphiles.length,
          largestCavity: lastCavity,
          cavityCount: lastCavityCount,
        }
        setAtomBadgeFull(
          `Скелет реконструированных атомов взят из ПРОВЕРЕННОЙ огрублённой динамики бульона; атомная ` +
            `геометрия — литературные длины связей/углы (chem/src/backmap.ts), не независимая ` +
            `атомистическая симуляция. Атом за атомом показано ${atomistic.length} из ${amphiphiles.length} ` +
            `найденных амфифилов (остальные — те же коарс-грейн мономеры).`,
        )

        if (sampleDue || now - lastSampleAt >= SAMPLE_INTERVAL_MS) {
          sampleDue = false
          const dtSec = (now - lastSampleAt) / 1000
          runUI.stepsPerSecond = dtSec > 0 ? (runUI.steps - lastSampleSteps) / dtSec : runUI.stepsPerSecond
          lastSampleAt = now
          lastSampleSteps = runUI.steps

          const { stage, evidence } = await detectStage(mySys)
          if (runGeneration !== myGeneration) return
          runUI.stage = stage
          runUI.evidence = evidence
          const entry: TraceEntry = { steps: runUI.steps, stage, evidence }
          runUI.trace.push(entry)
          if (runUI.trace.length > MAX_TRACE_LINES) runUI.trace.shift()

          // --- cavity breakdown: the SAME amphiphile-member positions detectStage's own
          // enclosedVolume just measured above, run through cavitiesFromPositions for the full
          // per-cavity list instead of one summed scalar -- see this file's Snapshot.largestCavity
          // doc comment for why this must reuse memberIndicesOf/positionsFor (soup/src/stages.ts)
          // rather than a second, independently-written "what counts as the wall" definition.
          const memberIdx = memberIndicesOf(amphiphiles)
          const cavityResult =
            memberIdx.size > 0
              ? cavitiesFromPositions(positionsFor(pos, memberIdx), box, {
                  cell: thresholds.closureCell,
                  radius: thresholds.closureRadius,
                })
              : { cavityCount: 0, cavities: [] }
          lastCavity = cavityResult.cavities[0] ?? null
          lastCavityCount = cavityResult.cavityCount
          runUI.cavityCount = lastCavityCount
          runUI.largestCavity = lastCavity
            ? { voxelCount: lastCavity.voxelCount, volume: lastCavity.volume, radius: lastCavity.radius, centre: lastCavity.centre }
            : null
          // The snapshot built just above this block still carries the PREVIOUS tick's cavity --
          // overwrite it now that this tick's own cavity is known, so draw() never lags a whole
          // sample interval behind the readout it is drawn to match.
          latestSnapshot.largestCavity = lastCavity
          latestSnapshot.cavityCount = lastCavityCount

          appendTraceLine(
            `шаг=${entry.steps} стадия=${stage} амф=${evidence.amphiphileFraction.toFixed(4)} ` +
              `агр=${evidence.largestAggregateFraction.toFixed(4)} пики=${
                evidence.headPeaks === HEAD_PEAKS_UNAVAILABLE ? 'н/д' : evidence.headPeaks
              } объём=${evidence.enclosedVolume.toFixed(4)} полостей=${lastCavityCount} ` +
              `крупн.полость.вокс=${lastCavity?.voxelCount ?? 0}`,
          )

          paintProgress()

          // Compare positions on the ladder, not identity: a sample can jump two stages at once
          // (aggregation is fast once amphiphiles exist), and an identity check would miss the stop.
          if (STAGES.indexOf(stage) >= STAGES.indexOf(targetStage)) {
            finishRun(
              stage === targetStage
                ? `целевая стадия «${STAGE_LABEL[targetStage]}» достигнута`
                : `стадия «${STAGE_LABEL[stage]}» достигнута, это не ниже цели «${STAGE_LABEL[targetStage]}»`,
            )
            return
          }
        }
      }
    } catch (err) {
      if (runGeneration !== myGeneration) return // a superseded generation's own error, not this run's
      failRun(`ошибка в цикле прогона: ${(err as Error).message}`, 'exception')
    }
  }

  function finishRun(reason: string): void {
    runGeneration++ // cancels any in-flight simDriver iteration for good
    activeElapsedMs = currentElapsedMs() // freeze the clock BEFORE the state flip changes what it reads
    setState('stopped')
    paintProgress()
    appendTraceLine(`[остановлено] ${reason}`)
    // Release this run's GPUBuffers now, not just before the NEXT run starts -- a run that is
    // stopped and never restarted should not hold ~20 buffers on the shared device indefinitely.
    activeSys?.dispose()
    activeSys = null
  }

  /** A run declared dead by the watchdog or by an uncaught exception (see simDriver's own
   * try/catch) -- distinct from finishRun(): sets state to 'error' (not 'stopped') and records
   * `runUI.error` with the last completed step count folded in, so the page LOOKS wedged instead
   * of looking like a normal, intentional stop. Still tears down the GPU side exactly like
   * finishRun() does -- a dead run must not keep holding its buffers either.
   *
   * `kind` (item 1b) is which of the two distinct failure modes this call site is reporting -- see
   * runUI.errorKind's own doc comment and errorLabel() above for why the status line needs this
   * rather than a single hardcoded message. Set on `runUI` BEFORE setState('error') so errorLabel()
   * reads the right value the first time it runs. */
  function failRun(reason: string, kind: 'timeout' | 'exception'): void {
    runGeneration++
    activeElapsedMs = currentElapsedMs()
    const message = `${reason} (пройдено шагов: ${runUI.steps})`
    runUI.error = message
    runUI.errorKind = kind
    setState('error')
    paintProgress()
    appendTraceLine(`[ошибка] ${message}`)
    console.error('[viewer/run] simDriver failed:', message)
    activeSys?.dispose()
    activeSys = null
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

    targetStage = stageSelect.value as Stage
    stepCap = Math.max(1, Math.floor(Number(stepCapInput.value) || DEFAULT_STEP_CAP))
    box = boxNow

    runUI.steps = 0
    runUI.stepsPerSecond = 0
    runUI.stage = 'monomers'
    runUI.evidence = { amphiphileFraction: 0, largestAggregateFraction: 0, headPeaks: 0, enclosedVolume: 0 }
    runUI.trace = []
    runUI.error = null
    runUI.errorKind = null
    runUI.cavityCount = 0
    runUI.largestCavity = null
    traceLogEl.textContent = ''
    bondCount = 0
    lastCavity = null
    lastCavityCount = 0
    activeElapsedMs = 0
    lastResumeAt = performance.now()
    lastSampleAt = performance.now()
    lastSampleSteps = 0
    sampleDue = true
    autoPausedByVisibility = false

    // Defensive: finishRun()/failRun() already dispose() the previous run's system on every path
    // that ends a run (stop, step cap, target stage reached, watchdog, exception). This is a
    // second, belt-and-suspenders release right before handing the shared device to a brand new
    // system, in case some future code path ever calls startRun() again without having gone
    // through one of those -- see SoupSystem.dispose()'s own doc comment for why an undisposed
    // system's GPUBuffers do not otherwise get freed on this shared, memoized device.
    activeSys?.dispose()
    activeSys = null

    setState('running')
    appendTraceLine(`[старт] бокс=${box[0]}×${box[1]}×${box[2]} цель=${targetStage} предел_шагов=${stepCap}`)

    const sys = await createSoup({ box, seed: RUN_SEED, kT: params.thermostat.kT, start: startCounts })
    activeSys = sys
    const n = Object.values(startCounts).reduce((a, b) => a + b, 0)
    meshes = buildMeshes(n, box)

    runGeneration++
    void simDriver(sys, runGeneration)
  }

  function togglePause(): void {
    if (runUI.state === 'running') {
      activeElapsedMs = currentElapsedMs() // freeze BEFORE flipping state, see currentElapsedMs's doc
      setState('paused')
      paintProgress()
    } else if (runUI.state === 'paused') {
      autoPausedByVisibility = false
      lastResumeAt = performance.now()
      lastSampleAt = performance.now() // next sample window starts fresh from resume, not spanning the paused gap
      sampleDue = true
      setState('running')
    }
  }

  function stopRun(): void {
    finishRun('остановлено пользователем')
  }

  startBtn.addEventListener('click', () => {
    startRun().catch((err) => {
      console.error(err)
      appendTraceLine(`[ошибка] ${(err as Error).message}`)
      setState('stopped')
    })
  })
  pauseBtn.addEventListener('click', togglePause)
  stopBtn.addEventListener('click', stopRun)

  /** Points the camera at the largest cavity's own centre, preserving the CURRENT camera-to-target
   * offset (distance and angle) rather than jumping to some fixed distance -- so a user who has
   * already zoomed/rotated to a view they like keeps that same framing, just re-aimed. No-op (the
   * button is disabled, see paintProgress()) when no cavity has been found yet. */
  function aimAtCavity(): void {
    const largest = runUI.largestCavity
    if (!largest) return
    const target = new THREE.Vector3(...largest.centre)
    const offset = camera.position.clone().sub(controls.target)
    controls.target.copy(target)
    camera.position.copy(target).add(offset)
    controls.update()
  }
  cavityAimBtn.addEventListener('click', aimAtCavity)

  // Machine-friendliness requirement: pause automatically when the tab is hidden, resume only when
  // it becomes visible again -- and ONLY if this pause was the automatic kind, never overriding a
  // user's own manual pause.
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      if (runUI.state === 'running') {
        autoPausedByVisibility = true
        activeElapsedMs = currentElapsedMs() // freeze BEFORE flipping state, see currentElapsedMs's doc
        setState('paused')
        paintProgress()
      }
      visibilityNoteEl.classList.add('active')
      visibilityNoteEl.textContent = 'Вкладка скрыта — прогон на автопаузе, GPU не грузится в фоне.'
    } else {
      visibilityNoteEl.classList.remove('active')
      visibilityNoteEl.textContent =
        'Пока эта заметка синяя — вкладка видима, счёт идёт. Как только вкладка уходит из видимости ' +
        '(переключение окна/таба), прогон автоматически ставится на паузу и не грузит GPU в фоне; при ' +
        'возврате на вкладку он сам продолжается — но только если это была автопауза, а не ручная.'
      if (autoPausedByVisibility && runUI.state === 'paused') {
        autoPausedByVisibility = false
        lastResumeAt = performance.now()
        lastSampleAt = performance.now()
        sampleDue = true
        setState('running')
      }
    }
  })

  setState('idle')
  paintProgress()

  // --- render loop: independent of simDriver, capped at RENDER_INTERVAL_MS -----------------------
  let lastRenderAt = 0
  function renderLoop(now: number): void {
    requestAnimationFrame(renderLoop)
    if (now - lastRenderAt < RENDER_INTERVAL_MS) return
    lastRenderAt = now
    if (latestSnapshot) draw(latestSnapshot)
    else {
      controls.update()
      renderer.render(scene, camera)
    }
  }
  frameCamera(box)
  requestAnimationFrame(renderLoop)
}

main()
