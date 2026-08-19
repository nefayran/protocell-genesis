// The three.js scene and its ocean look: renderer/scene/camera/controls, the depth-graded
// background + drifting caustic bands, key/fill lighting, fog, box-diagonal camera framing, and the
// per-run InstancedMesh construction (capacities only -- per-frame instance updates are
// run-render.ts's job). Split out of viewer/run.ts (2026-08-18).
//
// Item 2 (2026-08 UI-fixes task): the near/far planes used to be fixed (0.1/1000) -- sized for the
// two old hardcoded presets (box 16/30), so they never had to move. Letting the user pick an
// arbitrary box side means a fixed far=1000 clips a large box's own far corner the moment the box's
// own diagonal (times frameCamera's own ~2x placement below) approaches it, exactly "nothing
// disappears by clipping" in reverse -- while a fixed near=0.1 on a TINY box wastes most of the
// depth buffer's precision on distances the camera is never placed within. Both planes are derived
// from the box's own diagonal every time frameCamera() runs (on every run start, since box can
// change run to run), not tied to either preset.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { buildCavityMesh } from './run-cavity'
import {
  ATOMS_PER_CARBON_ESTIMATE,
  elementColor,
  elementRadius,
  MAX_BOND_SEGMENTS_PER_PARTICLE,
  MINERAL_COLOR,
  MINERAL_METALNESS,
  MINERAL_ROUGHNESS,
  MONOMER_ELEMENT,
  SOLVENT_OPACITY,
  SOLVENT_VISIBLE_BY_DEFAULT,
  type SceneMeshes,
} from './run-types'
import { loadParams, wcaCutoff } from '../engine/src/params'

export interface RunScene {
  scene: THREE.Scene
  camera: THREE.PerspectiveCamera
  renderer: THREE.WebGLRenderer
  controls: OrbitControls
  canvas: HTMLCanvasElement
  /** Background-only redraw -- never touches `scene`'s real content, `particles`, or any
   * InstancedMesh. See this module's own construction of `bgTexture` below for what it paints. */
  paintOceanBackdrop(tMs: number): void
  frameCamera(box: [number, number, number]): void
  /** (Re)builds every InstancedMesh for a fresh run of `n` particles inside `box`, disposing the
   * previous run's meshes (if any) first, then calls frameCamera(box). `closureCell` sizes the
   * cavity mesh's own capacity (see run-cavity.ts's buildCavityMesh). */
  buildMeshes(
    prevMeshes: SceneMeshes | null,
    monomers: { id: string; solvent?: boolean; mineral?: boolean; radiusSigma?: number }[],
    n: number,
    box: [number, number, number],
    closureCell: number,
  ): SceneMeshes
}

export function createRunScene(canvas: HTMLCanvasElement): RunScene {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setSize(window.innerWidth, window.innerHeight)

  const scene = new THREE.Scene()

  // --- ocean backdrop: depth-graded gradient + drifting caustic bands, BACKGROUND ONLY -----------
  // scene.background is set to a small CanvasTexture (32x256 -- three.js always stretches a plain
  // Texture background to cover the whole canvas, so the SOURCE image can be tiny and cheap to
  // redraw) instead of a flat colour. paintOceanBackdrop() below is the only thing that ever writes
  // into this canvas/texture; it never reaches into `scene`'s real content, `particles`, or any
  // InstancedMesh -- the vertical gradient (lighter near the top of the texture = "toward the
  // surface", darker toward the bottom = "deeper") and the faint sine-driven bands standing in for
  // caustic ripples/light shafts are the WHOLE effect, and they live only on this texture's pixels.
  // Redrawn once per rendered frame (same RENDER_INTERVAL_MS cadence as everything else -- see
  // run-render.ts's render loop), which is where the "gentle ambient motion" comes from; the canvas
  // itself is far too small for that redraw to register against the actual per-particle cost of a
  // frame.
  const bgCanvas = document.createElement('canvas')
  bgCanvas.width = 32
  bgCanvas.height = 256
  const bgCtx = bgCanvas.getContext('2d') as CanvasRenderingContext2D
  const bgTexture = new THREE.CanvasTexture(bgCanvas)
  scene.background = bgTexture

  const OCEAN_SURFACE_RGB = [26, 66, 74] as const
  const OCEAN_DEEP_RGB = [4, 9, 18] as const
  function paintOceanBackdrop(tMs: number): void {
    const w = bgCanvas.width
    const h = bgCanvas.height
    const grad = bgCtx.createLinearGradient(0, 0, 0, h)
    grad.addColorStop(0, `rgb(${OCEAN_SURFACE_RGB.join(',')})`)
    grad.addColorStop(1, `rgb(${OCEAN_DEEP_RGB.join(',')})`)
    bgCtx.fillStyle = grad
    bgCtx.fillRect(0, 0, w, h)

    // Faint drifting bands standing in for caustic ripples/light shafts -- decoration on this
    // texture alone, never mistakable for a monomer/bond/atom (those are drawn as actual 3D
    // geometry elsewhere; this is flat, blurred, and always sits behind everything else).
    const t = tMs / 1000
    bgCtx.globalCompositeOperation = 'lighter'
    for (let i = 0; i < 3; i++) {
      const y = ((Math.sin(t * 0.12 + i * 2.1) * 0.5 + 0.5) * 0.75 + 0.05) * h
      const alpha = 0.05 + 0.03 * Math.sin(t * 0.35 + i * 1.3)
      bgCtx.fillStyle = `rgba(150, 220, 210, ${Math.max(0, alpha)})`
      bgCtx.fillRect(0, y, w, Math.max(2, h * 0.03))
    }
    bgCtx.globalCompositeOperation = 'source-over'
    bgTexture.needsUpdate = true
  }
  paintOceanBackdrop(0)

  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000)
  const controls = new OrbitControls(camera, renderer.domElement)
  controls.enableDamping = true

  // Light from above (soft key, slightly warm-white like sunlight filtered through water) plus a
  // cooler, dimmer fill from below (as if bounced back up through the water column) -- replaces the
  // old single flat ambient+sun pair. Three light objects total, still a fixed, particle-count-
  // independent shading cost. scene.fog (set in frameCamera below, since it depends on the box's own
  // size) gives distant particles the same water-haze falloff on top of this.
  scene.add(new THREE.AmbientLight(0xbfe4ff, 0.35))
  const keyLight = new THREE.DirectionalLight(0xeaf7ff, 0.85)
  keyLight.position.set(0.6, 1.6, 0.5)
  scene.add(keyLight)
  const fillLight = new THREE.DirectionalLight(0x2f7a92, 0.3)
  fillLight.position.set(-0.4, -1.3, -0.3)
  scene.add(fillLight)

  function frameCamera(box: [number, number, number]): void {
    const center = new THREE.Vector3(box[0] / 2, box[1] / 2, box[2] / 2)
    const diag = Math.sqrt(box[0] ** 2 + box[1] ** 2 + box[2] ** 2)
    camera.near = Math.max(diag / 1000, 1e-3)
    camera.far = diag * 20
    camera.updateProjectionMatrix()
    // Water-haze distance fade on the particles themselves: standard THREE.Fog blends each
    // fragment toward this colour by distance from the camera -- it is shading, computed at draw
    // time from the camera's own depth, and never touches a single position/instanceMatrix/
    // instanceColor value. Sized off the box's own diagonal (same basis as near/far above) so it
    // scales with whichever preset/box side the user picked, same as everywhere else in this file.
    scene.fog = new THREE.Fog(0x061018, diag * 0.6, diag * 2.4)
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
  function buildMeshes(
    prevMeshes: SceneMeshes | null,
    monomers: { id: string; solvent?: boolean; mineral?: boolean; radiusSigma?: number }[],
    n: number,
    box: [number, number, number],
    closureCell: number,
  ): SceneMeshes {
    if (prevMeshes) {
      for (const m of Object.values(prevMeshes.monomerMesh)) scene.remove(m)
      scene.remove(prevMeshes.bondMesh, prevMeshes.atomMesh, prevMeshes.atomBondMesh, prevMeshes.cavityMesh)
    }
    const monomerMesh: Record<string, THREE.InstancedMesh> = {}
    for (const m of monomers) {
      // Task 'clay-surface' (2026-08-19): the mineral platelet is drawn as a slab of thin flat plates,
      // one per bead, not as spheres -- see MINERAL_COLOR's own note in run-types.ts for why it does
      // not go through MONOMER_ELEMENT/data/atoms.json at all. The plate's footprint is the sheet's
      // OWN lattice spacing, derived here exactly as soup/src/soup-clay.ts derives it (round the box
      // side to a whole number of mineral-mineral contact distances, then divide the box by that), so
      // neighbouring plates meet edge to edge at every box size instead of leaving gaps at some. Its
      // thickness is one bead: pairB(mineral,mineral) = sigma*radiusSigma, the same contact-distance
      // parameter the force kernel uses -- a single bead layer looks like a single mineral layer.
      if (m.mineral) {
        const pp = loadParams()
        const radiusSigma = m.radiusSigma ?? 1
        const contact = wcaCutoff(pp.sigma * radiusSigma)
        const plateX = box[0] / Math.max(1, Math.round(box[0] / contact))
        const plateY = box[1] / Math.max(1, Math.round(box[1] / contact))
        const geoPlate = new THREE.BoxGeometry(plateX, plateY, pp.sigma * radiusSigma)
        const matPlate = new THREE.MeshStandardMaterial({
          color: new THREE.Color(MINERAL_COLOR),
          roughness: MINERAL_ROUGHNESS,
          metalness: MINERAL_METALNESS,
          flatShading: true,
        })
        const meshPlate = new THREE.InstancedMesh(geoPlate, matPlate, n)
        meshPlate.count = 0
        meshPlate.frustumCulled = false
        monomerMesh[m.id] = meshPlate
        scene.add(meshPlate)
        continue
      }
      const element = MONOMER_ELEMENT[m.id]
      const geo = new THREE.SphereGeometry(elementRadius(element), 10, 8)
      // Water-like response, applied uniformly to the WHOLE material (never per-instance): lower
      // roughness gives the soft specular highlight a wet sphere shows under the key light above;
      // opacity just under 1 is the "slight translucency" -- the base colour is still exactly
      // elementColor(element) from data/atoms.json, untouched.
      const solvent = m.solvent ?? false
      const mat = new THREE.MeshStandardMaterial({
        color: elementColor(element),
        roughness: 0.35,
        metalness: 0.05,
        transparent: true,
        opacity: solvent ? SOLVENT_OPACITY : 0.94,
        // Medium, not object: without this the near faces of the water haze hide everything behind
        // them in depth order, which is the opposite of what a solvent should do to the view.
        depthWrite: !solvent,
      })
      const mesh = new THREE.InstancedMesh(geo, mat, n)
      mesh.count = 0
      mesh.frustumCulled = false
      // Hidden by default for the reason SOLVENT_VISIBLE_BY_DEFAULT documents; the checkbox in
      // run-control-panel.ts flips this, and run-render.ts skips the per-instance work while off.
      if (solvent) mesh.visible = SOLVENT_VISIBLE_BY_DEFAULT
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
    // Same wet-look tuning as the coarse monomer spheres above; per-instance colour is still set
    // by setColorAt(ai, elementColor(a.element)) in run-render.ts's draw(), untouched by this.
    const atomMat = new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.35, metalness: 0.05, transparent: true, opacity: 0.94 })
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

    const cavityMesh = buildCavityMesh(box, closureCell)
    scene.add(cavityMesh)

    frameCamera(box)
    return { monomerMesh, bondMesh, atomMesh, atomBondMesh, cavityMesh }
  }

  return { scene, camera, renderer, controls, canvas, paintOceanBackdrop, frameCamera, buildMeshes }
}
