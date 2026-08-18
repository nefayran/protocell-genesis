// Cavity/vesicle voxel highlight -- split out of viewer/run.ts (2026-08-18). Owns the whole #cavity
// DOM subtree (readout numbers, the "aim camera" button, the cavity-honesty collapsible note) plus
// the cavity InstancedMesh's construction/per-frame update and the flood-fill computation that feeds
// both. Never touches the coarse/atomistic monomer or bond meshes -- those are run-scene.ts's/
// run-render.ts's own job.
import * as THREE from 'three'
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import {
  cavitiesFromPositions,
  equivalentSphereRadius,
  type CavityWorld,
} from '../engine/src/closure'
import { memberIndicesOf, positionsFor, type StageThresholds } from '../soup/src/stages'
import type { Amphiphile } from '../soup/src/amphiphile'
import { initCollapsibleNote } from './run-readout'
import type { RunUI } from './run-types'

// Cavity voxel cloud: display-only styling for the closure detector's own unreached grid cells --
// colour/opacity are chosen only to read as visually distinct from every monomer sphere (data/
// atoms.json's element palette) and every bond cylinder (plain white) -- a translucent teal that no
// monomer or bond uses. CAVITY_CUBE_SCALE shrinks each cube slightly below the detector's own grid
// cell (thresholds.closureCell) so adjacent voxels of the same cavity still read as a granular cloud
// instead of one solid slab -- purely cosmetic, never fed back into any measurement.
const CAVITY_COLOR = 0x2de6c0
const CAVITY_OPACITY = 0.35
const CAVITY_CUBE_SCALE = 0.85

/** Cavity voxel cloud's own InstancedMesh: capacity is the box's own total grid-cell count at this
 * run's closure resolution (thresholds.closureCell) -- a cavity can never have more voxels than the
 * grid itself, so this is a structural bound, not a guess (same reasoning as run-scene.ts's own
 * MAX_BOND_SEGMENTS_PER_PARTICLE/ATOMS_PER_CARBON_ESTIMATE). Called once per buildMeshes() (i.e. per
 * run start), never per frame. */
export function buildCavityMesh(box: [number, number, number], closureCell: number): THREE.InstancedMesh {
  const cellsPerAxis = box.map((side) => Math.max(1, Math.ceil(side / closureCell)))
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
  // Item 2 (2026-08 UI-fixes task): see run-scene.ts's own buildMeshes() doc comment for why every
  // InstancedMesh on this page gets frustumCulled = false.
  cavityMesh.frustumCulled = false
  return cavityMesh
}

const dummy = new THREE.Object3D()

/** Per-frame instance-buffer update for the cavity voxel cloud -- the largest cavity's voxel cloud,
 * moved verbatim out of viewer/run.ts's original draw(). Called from run-render.ts's draw() at the
 * exact same point in the per-frame sequence. */
export function updateCavityMesh(
  cavityMesh: THREE.InstancedMesh,
  closureCell: number,
  largestCavity: CavityWorld | null,
): void {
  const cubeSide = closureCell * CAVITY_CUBE_SCALE
  const voxels = largestCavity?.voxelCentres ?? null
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
}

/** The cavity breakdown computed once per SAMPLED tick (not every simDriver tick): the SAME
 * amphiphile-member positions detectStage's own enclosedVolume measures, run through
 * cavitiesFromPositions for the full per-cavity list instead of one summed scalar -- see
 * Snapshot.largestCavity's own doc comment (run-types.ts) for why this must reuse
 * memberIndicesOf/positionsFor (soup/src/stages.ts) rather than a second, independently-written
 * "what counts as the wall" definition. Moved verbatim out of viewer/run.ts's original simDriver. */
export function computeCavitySample(
  pos: Float32Array,
  amphiphiles: Amphiphile[],
  box: [number, number, number],
  thresholds: StageThresholds,
): { cavity: CavityWorld | null; cavityCount: number } {
  const memberIdx = memberIndicesOf(amphiphiles)
  const cavityResult =
    memberIdx.size > 0
      ? cavitiesFromPositions(positionsFor(pos, memberIdx), box, {
          cell: thresholds.closureCell,
          radius: thresholds.closureRadius,
        })
      : { cavityCount: 0, cavities: [] }
  return { cavity: cavityResult.cavities[0] ?? null, cavityCount: cavityResult.cavityCount }
}

export interface CavityPanel {
  /** Repaints #cavity's own readout numbers (found/voxels/volume/radius/centre) and the aim
   * button's disabled state from the CURRENT `runUI` (the same object reference passed in at
   * construction, mutated in place elsewhere -- see run-runtime.ts's own doc comment). Called from
   * run-readout.ts's paintProgress() at the exact point the original single paintProgress() painted
   * this same panel. */
  paint(): void
}

/** Builds the whole #cavity panel: readout numbers, the "aim camera at cavity" button, and the
 * cavity-honesty collapsible note. `runUI` is the ONE long-lived RunUI object (see run-runtime.ts) --
 * paint() and the aim button's click handler both read its CURRENT fields at call time, exactly like
 * the original single main()'s closures did. */
export function createCavityPanel(
  camera: THREE.PerspectiveCamera,
  controls: OrbitControls,
  runUI: RunUI,
  thresholds: StageThresholds,
): CavityPanel {
  const cavityCountEl = document.getElementById('cavity-count') as HTMLElement
  const cavityVoxelsEl = document.getElementById('cavity-voxels') as HTMLElement
  const cavityVolumeEl = document.getElementById('cavity-volume') as HTMLElement
  const cavityRadiusEl = document.getElementById('cavity-radius') as HTMLElement
  const cavityCentreEl = document.getElementById('cavity-centre') as HTMLElement
  const cavityAimBtn = document.getElementById('cavity-aim-btn') as HTMLButtonElement
  const cavityHonestyEl = document.getElementById('cavity-honesty') as HTMLElement

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

  function paint(): void {
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

  /** Points the camera at the largest cavity's own centre, preserving the CURRENT camera-to-target
   * offset (distance and angle) rather than jumping to some fixed distance -- so a user who has
   * already zoomed/rotated to a view they like keeps that same framing, just re-aimed. No-op (the
   * button is disabled, see paint() above) when no cavity has been found yet. */
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

  return { paint }
}
