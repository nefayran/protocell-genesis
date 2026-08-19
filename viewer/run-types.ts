// Shared types, page-level presentation constants and tiny pure helpers for the run-control page
// (viewer/run.ts and its sibling run-*.ts modules). Split out of viewer/run.ts (2026-08-18, same
// task that split soup/src/sim.ts) purely so every other module has one place to import these from
// without re-declaring them -- this file itself has no DOM/three.js/GPU side effects.
import * as THREE from 'three'
import atomsRaw from '../data/atoms.json'
import type { BackmapAtom } from '../chem/src/backmap'
import type { CavityWorld } from '../engine/src/closure'
import type { Stage, StageEvidence } from '../soup/src/stages'

export interface ElementInfo {
  vdw: number
  color: string
}
export interface AtomsData {
  elements: Record<string, ElementInfo>
}
export const atomsData = atomsRaw as unknown as AtomsData

// --- page-level presentation constants (soup/src stays free of these; this is display-only). ------

// Steps advanced per simDriver tick. Bounded on purpose -- see run-sim-driver.ts's own header. Small
// enough that even the "default" ~13100-particle preset finishes a tick in well under a second at
// this engine's measured 2086-2302 steps/s, so the UI (clicks, pause, the render loop) is never
// starved for more than that.
export const STEP_BATCH = 250

// Progress numbers (steps/s, elapsed, ETA, evidence, trace) repaint at most this often -- "a few
// times a second, no faster" from the brief. 250ms = 4/s.
export const SAMPLE_INTERVAL_MS = 250

// 3D view redraws at most this often, independent of how fast simDriver is producing snapshots --
// "at most a few dozen frames a second" from the brief. 1000/30 caps it at 30fps.
export const RENDER_INTERVAL_MS = 1000 / 30

// Scrolling trace is bounded so a long run's log can't grow the DOM without limit.
export const MAX_TRACE_LINES = 300

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
export const STEP_WATCHDOG_MS = 15_000

// Nanometres per reduced sigma -- same viewer-only convention as viewer/molecular.ts (sigma~1nm is
// the standard literature mapping for this bead model, not a measured quantity; chosen here purely
// so the atomistic reconstruction's coordinates read directly in the same numbers as the coarse
// sim's own box/particle coordinates).
export const SIGMA_NM = 1.0

// Half-width (fraction of box x-size) of the fixed spatial band drawn atom-by-atom; amphiphiles
// outside it are drawn as the same coarse spheres every other monomer gets. Fixed (not
// camera-tracking) so which amphiphiles are atomistic is stable frame to frame, exactly the
// reasoning viewer/molecular.ts's own SLICE_HALF_WIDTH uses.
export const SLICE_HALF_WIDTH_FRAC = 0.15

// Ball-and-stick sizing, identical convention to viewer/molecular.ts: atoms shrunk below their true
// van-der-Waals radius so bonds stay visible as sticks. Reused for the COARSE monomer spheres too
// (radius/colour both sourced from data/atoms.json), so the atomistic slice and the coarse rest of
// the soup share one visual vocabulary instead of two unrelated size scales.
export const ATOM_RADIUS_SCALE = 0.34
export const BOND_RADIUS = atomsData.elements.C.vdw * 0.16

// data/soup.json's monomer ids happen to spell out real element symbols for three of the four kinds
// (C=carbon, O=head, H=donor) -- not a coincidence in this model's naming, and reused here so the
// coarse spheres and the atomistic reconstruction draw from the exact same palette. The catalyst
// ("M") has no atomic identity in this coarse-grained model at all; "Na" is an arbitrary, honestly
// documented stand-in chosen only to keep it visually distinct, not a claim about its chemistry.
// The solvent bead ("W", data/soup.json's `solvent.waterId`) maps to oxygen: a coarse water bead
// stands for H2O, whose only heavy atom IS oxygen, so this reuses a real palette entry rather than
// inventing an element -- data/atoms.json keeps holding measured atomic data only. It shares the
// head group's colour by construction; SOLVENT_OPACITY below is what keeps the two distinguishable
// on screen (the medium reads as a haze, the heads as solid spheres).
export const MONOMER_ELEMENT: Record<string, string> = { C: 'C', O: 'O', H: 'H', M: 'Na', W: 'O' }

// Task 'clay-surface' (2026-08-19): the mineral platelet gets its own visual identity, NOT another
// entry in MONOMER_ELEMENT. Two reasons, in order: (a) a clay bead has no single element to look up
// (an aluminosilicate layer is Si/Al/O/OH plus interlayer cations, and data/atoms.json holds measured
// per-ELEMENT data only, so borrowing "Si" would put a colour in the palette that claims a chemistry
// the model does not carry); (b) drawing it as one more sphere is exactly what the brief rules out --
// on screen the platelet must read as a mineral slab. So mineral beads are drawn as thin, flat,
// axis-aligned PLATES (a BoxGeometry per bead, sized to the sheet's own lattice spacing, which is
// derived in run-scene.ts the same way soup/src/soup-clay.ts derives it) in a matte grey-olive with
// flat shading -- a slate look, dull where the water beads are glossy. Look choice, documented here
// rather than buried: no model constant, nothing measured, and nothing about it feeds the physics.
export const MINERAL_COLOR = 0x8f9179
export const MINERAL_ROUGHNESS = 0.92
export const MINERAL_METALNESS = 0.02

// The solvent is the MEDIUM, not an object in it: at any broth-like composition it outnumbers every
// other species several times over, so drawing it as solid spheres both hides the chemistry behind
// a wall of water and multiplies transparent overdraw on a machine this page is meant to stay cheap
// on. Hence a much lower opacity than the 0.94 of the coarse monomer spheres, and hidden by default
// (viewer/run.html's #water-visible-input turns it on) -- a look/cost choice, not a model constant.
export const SOLVENT_OPACITY = 0.1
export const SOLVENT_VISIBLE_BY_DEFAULT = false

// Safe upper bounds for InstancedMesh capacities, derived rather than guessed: soup/src/stages.ts's
// own header notes valence is capped at 3, so bonds <= N*3/2 and this file's two-half-cylinder
// convention needs N*3 segments at most. ATOMS_PER_CARBON_ESTIMATE covers the reconstructed
// ball-and-stick backbone (backbone C + its H's, plus the fixed few-atom carboxyl group at one
// end) generously above the ~3 atoms/carbon a plain alkanoic-acid chain works out to.
export const MAX_BOND_SEGMENTS_PER_PARTICLE = 3 * 2
export const ATOMS_PER_CARBON_ESTIMATE = 5

// --- run-size presets (page-level; the physics/thresholds these feed are all read from
// data/soup.json by createSoup itself -- these are just which knobs of CreateSoupOpts the user is
// offered). "tiny" is deliberately the DEFAULT selection: this page's whole point is to put the
// user in charge of how much machine a run costs, so the safe/cheap choice should not require the
// user to already know to pick it. -----------------------------------------------------------------
export interface SizePreset {
  label: string
  box: [number, number, number]
  start?: Record<string, number>
}
// Task 'broth-composition' (2026-08-18): data/soup.json's own `start` now includes explicit water
// (W). CreateSoupOpts.start MERGES over the file's defaults (an id this preset does not mention
// keeps the file's count) -- so leaving W unmentioned here would silently pull the file's full
// water count (10700, sized for the box-30 "default" preset's own density) into this box-16 preset
// too, multiplying its particle count roughly 4.5x and defeating the whole point of "tiny" staying
// cheap. `W: 100` is a deliberately TOKEN amount (density ~0.024 sigma^-3, nowhere near the
// measured 0.8 sigma^-3 liquid density -- see data/soup.json's solvent.basis) -- enough to exercise
// the water code path (the solvent-attraction rule, the water-aware closure detector) at trivial
// cost, not a scientifically meaningful liquid. A real liquid-density preview belongs in "default"
// or a size a user picks knowing the cost, not in the one preset whose whole job is staying cheap.
// Task 'clay-surface' (2026-08-19): the particle counts in these labels now INCLUDE the mineral
// platelet, whose bead count is derived from the box (soup/src/soup-clay.ts), not from `start` -- at
// box 16 that is a 14x14 sheet (~196 beads), at box 30 a 27x27 one (~729). The platelet is part of
// the shipped composition (data/soup.json's clay.enabled), so it is part of what these presets run;
// `clay: false` is what a clay-free control arm passes, and no preset here does.
export const SIZE_PRESETS: Record<string, SizePreset> = {
  tiny: { label: 'малый (проверочный, ~760 частиц с пластиной глины)', box: [16, 16, 16], start: { C: 200, O: 50, H: 200, M: 20, W: 100 } },
  default: { label: 'стандартный бульон + глина (data/soup.json, ~15000 частиц)', box: [30, 30, 30] },
}
export const DEFAULT_SIZE_KEY = 'tiny'
export const DEFAULT_STEP_CAP = 20_000
export const RUN_SEED = 42

export const STAGES: Stage[] = ['monomers', 'amphiphiles', 'micelles', 'bilayer', 'vesicle']
export const STAGE_LABEL: Record<Stage, string> = {
  monomers: 'monomers',
  amphiphiles: 'amphiphiles',
  micelles: 'micelles',
  bilayer: 'bilayer',
  vesicle: 'vesicle',
}

// 'error': the watchdog (or an unexpected exception in simDriver) declared the run wedged/dead.
// Deliberately its own state rather than reusing 'stopped' with a side flag: a wedged run must be
// visibly DIFFERENT from a normal stop, not just carry an extra field a casual look would miss.
export type RunState = 'idle' | 'running' | 'paused' | 'stopped' | 'error'

export interface TraceEntry {
  steps: number
  stage: Stage
  evidence: StageEvidence
}

export interface RunUI {
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
   * run-control-panel.ts's setState()'s own use of this field). 'timeout': stepWithWatchdog's
   * deadline fired, the GPU-bound step() call itself never returned -- "GPU не отвечает" is an
   * honest description ONLY of this case. 'exception': something in the sample/step loop THREW
   * (its own message is what `error` carries) -- the GPU answered fine; the failure is in this
   * page's own code. */
  errorKind: 'timeout' | 'exception' | null
  /** How many disjoint cavities the closure detector found on the latest sample -- 0 whenever
   * nothing is closed at all (the normal state for most of a run). */
  cavityCount: number
  /** The largest of those cavities' own summary numbers (voxelCount/volume/radius/centre), without
   * its full voxel list -- that list lives on Snapshot.largestCavity for drawing only, not on this
   * UI-readout object. null exactly when cavityCount === 0. */
  largestCavity: { voxelCount: number; volume: number; radius: number; centre: [number, number, number] } | null
}

// --- run lifecycle -----------------------------------------------------------------------------
export interface Snapshot {
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

// Instanced meshes are (re)built per run, sized from THAT run's own N -- a fresh run can pick a
// different size preset, so capacities must not be fixed at module load.
export interface SceneMeshes {
  monomerMesh: Record<string, THREE.InstancedMesh>
  bondMesh: THREE.InstancedMesh
  atomMesh: THREE.InstancedMesh
  atomBondMesh: THREE.InstancedMesh
  cavityMesh: THREE.InstancedMesh
}

// Item 2 (2026-08 UI-fixes task): test-support hook. tests/run-ui.test.ts's camera sweep needs to
// move the camera through several distances/angles around the CURRENT box and check that something
// non-background actually rendered at each one -- there is no DOM state that proves "the instanced
// meshes are visible", the only ground truth is the rendered pixels, read back via WebGL readPixels
// (never --dump-dom, per this task's own constraint) so the test stays inside page.evaluate the same
// way every other assertion here does. Exposed the same way `runUI` already is: a small, explicitly-
// named object on `window`, not a production feature.
export interface SceneDebugHooks {
  /** Places the camera at `distanceScale` * the CURRENT box's own diagonal from its centre, at
   * spherical angles (thetaDeg around the vertical axis, phiDeg from it), looking at the centre,
   * then renders one frame immediately so nonBackgroundPixelFraction() below has something fresh
   * to read. */
  setCameraOrbit(distanceScale: number, thetaDeg: number, phiDeg: number): void
  /** Fraction of the canvas's own pixels that differ from what the ocean backdrop ALONE would
   * render at this exact camera pose/instant -- 0 means "the canvas shows nothing but the
   * backdrop", which is exactly the item-2 symptom this guards. Rewritten for the ocean-look task
   * (2026-08): the backdrop used to be a flat colour (0x0b0d10) so comparing against that constant
   * was enough, but it is now an animated gradient+caustic CanvasTexture (see run-scene.ts's own
   * paintOceanBackdrop), so a fixed reference colour would call almost every pixel "non-background"
   * regardless of whether any instance actually drew anything -- silently defeating this exact
   * regression guard. Instead this renders the SAME camera pose twice, once as normal and once with
   * every InstancedMesh hidden, and diffs the two readbacks; whatever the backdrop happens to look
   * like at this instant cancels out identically in both, so only real content shows up as a
   * difference. Visibility is restored and a normal frame re-rendered before returning. */
  nonBackgroundPixelFraction(): number
  /** Count of frames actually rendered (i.e. that passed renderLoop's RENDER_INTERVAL_MS gate),
   * incremented once per such frame -- a plain counter, not a computed rate, so a caller can
   * sample it twice around a real wall-clock delay and divide, the same honest measured-not-
   * guessed convention runUI.stepsPerSecond already uses for the sim side. Added for the
   * ocean-look task's own frame-rate-before/after requirement; harmless to leave in place. */
  renderedFrameCount: number
}

/** Shortest periodic image of `d` along one axis of length `L` (minimum-image convention -- the
 * same rule the physics itself uses, engine/src/sim.ts's own mi3 comment). Positions read back
 * from the GPU are wrapped into [0,box); drawing a straight cylinder between two RAW wrapped
 * positions would show a bond stretching across the whole box whenever it happens to straddle a
 * periodic face. This is a display-only correction -- it never touches soup/src's own state. */
export function wrap(d: number, L: number): number {
  if (d > L / 2) return d - L
  if (d < -L / 2) return d + L
  return d
}

export function formatElapsed(ms: number): string {
  const totalSec = Math.floor(ms / 1000)
  const m = Math.floor(totalSec / 60)
  const s = totalSec % 60
  return `${m}:${String(s).padStart(2, '0')}`
}

export function elementColor(element: string): THREE.Color {
  const info = atomsData.elements[element]
  if (!info) throw new Error(`viewer/run: неизвестный элемент "${element}" в data/atoms.json`)
  return new THREE.Color(info.color)
}
export function elementRadius(element: string): number {
  const info = atomsData.elements[element]
  if (!info) throw new Error(`viewer/run: неизвестный элемент "${element}" в data/atoms.json`)
  return info.vdw * ATOM_RADIUS_SCALE
}
