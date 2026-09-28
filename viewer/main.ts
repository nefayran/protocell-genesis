// Interactive viewer (Task 10): a live view of the running GPU simulation, a two-slider parameter
// panel that retunes it without restarting it, and an always-visible honesty badge (spec
// requirement: a viewer must never let someone mistake a coarse-grained bead for an atom).
//
// three.js draws from the SAME positions() the engine already exposes -- no separate renderer-side
// physics, no WebGPURenderer (plain WebGLRenderer reading back CPU-side Float32Arrays is enough for
// a few hundred spheres and keeps this file independent of whether the engine's own WebGPU pipeline
// changes shape).
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { loadParams } from '../engine/src/params'
import { createSystem } from '../engine/src/sim'
import { createPanel } from './panel'

// Simulation steps advanced per rendered frame. Chosen for a responsive view: at this engine's
// measured throughput (~3600 steps/s at 3600 beads, task-9-report.md) the ~900-bead system this
// viewer builds steps well inside a 16ms frame budget even at 20 steps/frame, while 20 steps still
// visibly advances the dynamics between renders instead of redrawing a near-frozen frame 1 step at
// a time.
const STEPS_PER_FRAME = 20

// w_c slider bounds -- a UI choice (the brief's range), not a physics constant; the physics wc
// itself always comes from data/params.json (initial value) or setLiveParams (live value). Lives
// here rather than in engine/src/params.ts because viewer/ is not scanned by the "no bare model
// constants" guard test and this number has no Cooke & Deserno meaning outside this slider.
const WC_SLIDER_RANGE: [number, number] = [1.0, 2.0]

// Demo system size/box: small enough to step and read back every frame at interactive rates, large
// enough to show a recognisable bilayer patch. box.x,y must be large enough that the neighbor grid
// (sized to cover w_c up to WC_SLIDER_RANGE[1], via maxWc below) still keeps dims>=3 on the
// periodic axes -- see sim.ts's gridInvariantsHold.
const LIPIDS = 400
const BOX: [number, number, number] = [18, 18, 26]

async function main(): Promise<void> {
  const params = loadParams()

  const canvas = document.getElementById('scene') as HTMLCanvasElement
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setSize(window.innerWidth, window.innerHeight)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b0d10)

  const center = new THREE.Vector3(BOX[0] / 2, BOX[1] / 2, BOX[2] / 2)
  const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 500)
  camera.position.set(BOX[0] * 1.8, BOX[1] * 1.3, BOX[2] * 1.1)
  camera.lookAt(center)

  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.copy(center)
  controls.enableDamping = true

  scene.add(new THREE.AmbientLight(0xffffff, 0.7))
  const sun = new THREE.DirectionalLight(0xffffff, 0.9)
  sun.position.set(1, 1.4, 1)
  scene.add(sun)

  // maxWc: the grid must stay valid for the FULL slider range from the start, since w_c can be
  // raised live via setLiveParams without ever recreating the system (a grid sized only for the
  // initial params.json wc would silently drop forces once the slider goes above it -- see
  // CreateSystemOpts.maxWc's doc comment in sim.ts).
  const sys = await createSystem({
    lipids: LIPIDS,
    box: BOX,
    seed: 1,
    layout: 'bilayer',
    maxWc: WC_SLIDER_RANGE[1],
  })

  // Beads are heads (type 0, one per lipid) or tails (type 1, two per lipid) -- visually distinct
  // InstancedMeshes, sized from the bead-size params rather than a bare literal.
  const headRadius = params.beadSizes.head_head / 2
  const tailRadius = params.beadSizes.tail_tail / 2
  const headGeo = new THREE.SphereGeometry(headRadius, 12, 8)
  const tailGeo = new THREE.SphereGeometry(tailRadius, 10, 8)
  const headMat = new THREE.MeshStandardMaterial({ color: 0x4fc3f7 }) // heads: cool blue
  const tailMat = new THREE.MeshStandardMaterial({ color: 0xffa552 }) // tails: warm orange

  const headMesh = new THREE.InstancedMesh(headGeo, headMat, sys.lipids)
  const tailMesh = new THREE.InstancedMesh(tailGeo, tailMat, sys.lipids * 2)
  scene.add(headMesh, tailMesh)

  const dummy = new THREE.Object3D()

  function updateInstances(pos: Float32Array): void {
    let hi = 0
    let ti = 0
    const beadCount = pos.length / 4
    for (let i = 0; i < beadCount; i++) {
      dummy.position.set(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2])
      dummy.updateMatrix()
      if (pos[i * 4 + 3] === 0) headMesh.setMatrixAt(hi++, dummy.matrix)
      else tailMesh.setMatrixAt(ti++, dummy.matrix)
    }
    headMesh.instanceMatrix.needsUpdate = true
    tailMesh.instanceMatrix.needsUpdate = true
  }

  // --- honesty badge ---------------------------------------------------------------------------
  // THE BADGE IS A HONESTY REQUIREMENT, not decoration: it states the integration step size in
  // reduced units, how much reduced (simulated) time tau has elapsed, and says in plain words that
  // these are coarse-grained beads standing in for whole lipid segments, not atoms. Rendered into a
  // permanently visible on-screen element (#badge in index.html) -- never a tooltip.
  function renderBadge(): string {
    const tau = sys.steps * params.integrator.dt
    return (
      `integration step δt = ${params.integrator.dt}τ · shown ${tau.toFixed(1)}τ of reduced ` +
      `time · these are coarse-grained beads (3 per lipid: head + 2 tails), not atoms; the scale is the model's, ` +
      `not physical`
    )
  }

  const badgeEl = document.getElementById('badge') as HTMLElement
  const framesEl = document.getElementById('frames') as HTMLElement

  function applyKT(v: number): void {
    sys.setLiveParams({ kT: v })
    viewer.kT = v
    panel.setKT(v)
  }
  function applyWc(v: number): void {
    sys.setLiveParams({ wc: v })
    viewer.wc = v
    panel.setWc(v)
  }

  // Published up front (before the first frame) and MUTATED in place every frame rather than
  // replaced with a fresh object -- a test or external script reading window.viewer mid-flight
  // must see live fields, not a stale snapshot from an object that got swapped out from under it.
  const viewer = {
    frames: 0,
    kT: params.thermostat.kT,
    wc: params.attraction.wc,
    timeScaleBadge: renderBadge(),
    setKT: applyKT,
    setWc: applyWc,
  }
  ;(window as unknown as { viewer: typeof viewer }).viewer = viewer
  badgeEl.textContent = viewer.timeScaleBadge

  const panel = createPanel({
    kTRange: params.kTRange,
    wcRange: WC_SLIDER_RANGE,
    initialKT: params.thermostat.kT,
    initialWc: params.attraction.wc,
    onKT: applyKT,
    onWc: applyWc,
  })

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  })

  async function frame(): Promise<void> {
    await sys.step(STEPS_PER_FRAME)
    updateInstances(await sys.positions())
    controls.update()
    renderer.render(scene, camera)
    viewer.frames++
    viewer.timeScaleBadge = renderBadge()
    badgeEl.textContent = viewer.timeScaleBadge
    framesEl.textContent = `frame ${viewer.frames}`
    requestAnimationFrame(() => {
      frame()
    })
  }
  requestAnimationFrame(() => {
    frame()
  })
}

main().catch((err) => {
  console.error(err)
  document.body.innerHTML = `<pre style="color:#f66;padding:20px;white-space:pre-wrap;">${String(
    (err as Error)?.stack ?? err,
  )}</pre>`
})
