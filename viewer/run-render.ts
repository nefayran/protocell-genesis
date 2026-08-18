// The per-frame render loop and the instanced-particle buffer updates: one redraw of the whole
// scene from a Snapshot (coarse monomer spheres, coarse bonds, the atomistic ball-and-stick slice,
// the cavity voxel cloud) plus the requestAnimationFrame loop that throttles how often that redraw
// happens, independent of how fast simDriver is producing new snapshots. Split out of
// viewer/run.ts (2026-08-18).
import * as THREE from 'three'
import { updateCavityMesh } from './run-cavity'
import type { RunRuntime } from './run-runtime'
import type { RunScene } from './run-scene'
import { elementColor, elementRadius, MONOMER_ELEMENT, RENDER_INTERVAL_MS, wrap, type SceneDebugHooks, type Snapshot } from './run-types'

const dummy = new THREE.Object3D()
const up = new THREE.Vector3(0, 1, 0)

function drawHalfCylinders(
  mesh: THREE.InstancedMesh,
  bondRadius: number,
  segments: { from: THREE.Vector3; to: THREE.Vector3; color: THREE.Color }[],
): void {
  let i = 0
  for (const seg of segments) {
    const dir = seg.to.clone().sub(seg.from)
    const len = dir.length()
    if (len < 1e-9) continue
    dummy.position.copy(seg.from).add(dir.clone().multiplyScalar(0.5))
    dummy.quaternion.setFromUnitVectors(up, dir.clone().normalize())
    dummy.scale.set(bondRadius, len, bondRadius)
    dummy.updateMatrix()
    mesh.setMatrixAt(i, dummy.matrix)
    mesh.setColorAt(i, seg.color)
    i++
    if (i >= mesh.instanceMatrix.count) break // capacity safety, see run-scene.ts's own buildMeshes
  }
  mesh.count = Math.min(i, mesh.instanceMatrix.count)
  mesh.instanceMatrix.needsUpdate = true
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true
}

export interface RunRenderer {
  draw(snap: Snapshot): void
  sceneDebug: SceneDebugHooks
  /** Calls frameCamera(runtime.box) once, then starts the requestAnimationFrame loop -- same order
   * as the original main()'s own final two statements. */
  startRenderLoop(): void
}

export function createRunRenderer(
  runScene: RunScene,
  runtime: RunRuntime,
  monomers: { id: string }[],
  bondRadius: number,
  closureCell: number,
): RunRenderer {
  const { scene, camera, renderer, controls, canvas, paintOceanBackdrop, frameCamera } = runScene

  /** One redraw of the whole scene from a snapshot -- positions/bonds/box for the coarse view, plus
   * which amphiphiles (if any) fall inside the atomistic slice this snapshot. Pure DOM/three.js
   * side effects; never touches the GPU sim itself (that is run-sim-driver.ts's job only). */
  function draw(snap: Snapshot): void {
    const meshes = runtime.meshes
    if (!meshes) return
    const { particles, bonds, box, atomistic, atomisticSet } = snap
    const n = particles.length / 4

    // --- coarse monomer spheres, one InstancedMesh per monomer id, skipping anything drawn
    // atom-by-atom instead ------------------------------------------------------------------
    const counters: Record<string, number> = {}
    for (const m of monomers) counters[m.id] = 0
    for (let i = 0; i < n; i++) {
      if (atomisticSet.has(i)) continue
      const kind = Math.round(particles[i * 4 + 3])
      const m = monomers[kind]
      const mesh = meshes.monomerMesh[m.id]
      dummy.position.set(particles[i * 4], particles[i * 4 + 1], particles[i * 4 + 2])
      dummy.scale.setScalar(1)
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      const idx = counters[m.id]++
      if (idx < mesh.instanceMatrix.count) mesh.setMatrixAt(idx, dummy.matrix)
    }
    for (const m of monomers) {
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
      const colorA = elementColor(MONOMER_ELEMENT[monomers[Math.round(particles[ia * 4 + 3])].id])
      const colorB = elementColor(MONOMER_ELEMENT[monomers[Math.round(particles[ib * 4 + 3])].id])
      coarseSegments.push({ from: pA, to: mid, color: colorA }, { from: mid, to: pB, color: colorB })
    }
    drawHalfCylinders(meshes.bondMesh, bondRadius, coarseSegments)

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
    drawHalfCylinders(meshes.atomBondMesh, bondRadius, atomSegments)

    // --- largest cavity's voxel cloud (see run-cavity.ts's own doc comments) -----------------
    updateCavityMesh(meshes.cavityMesh, closureCell, snap.largestCavity)

    controls.update()
    renderer.render(scene, camera)
  }

  // --- test-support hook (item 2, 2026-08 UI-fixes task) ------------------------------------------
  // tests/run-ui.test.ts's camera sweep needs to move the camera through several distances/angles
  // around the CURRENT box and check that something non-background actually rendered at each one --
  // there is no DOM state that proves "the instanced meshes are visible", the only ground truth is
  // the rendered pixels, read back here via WebGL readPixels (never --dump-dom, per this task's own
  // constraint) so the test stays inside page.evaluate the same way every other assertion here does.
  // Exposed the same way `runUI` already is: a small, explicitly-named object on `window`, not a
  // production feature.
  const sceneDebug: SceneDebugHooks = {
    renderedFrameCount: 0,
    setCameraOrbit(distanceScale, thetaDeg, phiDeg) {
      const box = runtime.box
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
      if (runtime.latestSnapshot) draw(runtime.latestSnapshot)
      else renderer.render(scene, camera)
    },
    nonBackgroundPixelFraction() {
      const gl = renderer.getContext()
      const w = canvas.width
      const h = canvas.height
      const withContent = new Uint8Array(w * h * 4)
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, withContent)

      const meshes = runtime.meshes
      const toggled: THREE.InstancedMesh[] = meshes
        ? [meshes.bondMesh, meshes.atomMesh, meshes.atomBondMesh, meshes.cavityMesh, ...Object.values(meshes.monomerMesh)]
        : []
      const prevVisible = toggled.map((m) => m.visible)
      for (const m of toggled) m.visible = false
      renderer.render(scene, camera)
      const backdropOnly = new Uint8Array(w * h * 4)
      gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, backdropOnly)
      toggled.forEach((m, i) => (m.visible = prevVisible[i]))
      renderer.render(scene, camera) // restore the real frame

      let diff = 0
      for (let i = 0; i < withContent.length; i += 4) {
        if (
          Math.abs(withContent[i] - backdropOnly[i]) > 6 ||
          Math.abs(withContent[i + 1] - backdropOnly[i + 1]) > 6 ||
          Math.abs(withContent[i + 2] - backdropOnly[i + 2]) > 6
        )
          diff++
      }
      return diff / (w * h)
    },
  }

  // --- render loop: independent of simDriver, capped at RENDER_INTERVAL_MS -----------------------
  let lastRenderAt = 0
  function renderLoop(now: number): void {
    requestAnimationFrame(renderLoop)
    if (now - lastRenderAt < RENDER_INTERVAL_MS) return
    lastRenderAt = now
    paintOceanBackdrop(now) // background-only redraw; see run-scene.ts's own doc comment -- never touches particles
    sceneDebug.renderedFrameCount++
    if (runtime.latestSnapshot) draw(runtime.latestSnapshot)
    else {
      controls.update()
      renderer.render(scene, camera)
    }
  }
  function startRenderLoop(): void {
    frameCamera(runtime.box)
    requestAnimationFrame(renderLoop)
  }

  return { draw, sceneDebug, startRenderLoop }
}
