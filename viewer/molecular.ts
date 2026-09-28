// Task 8: the coarse-grained membrane redrawn as real molecules, atom by atom -- a sibling of
// viewer/main.ts (same three.js/orbit-controls/badge structure), not a rewrite of it.
//
// THE HONESTY REQUIREMENT this page exists to satisfy: every atom's position is RECONSTRUCTED
// from literature bond geometry (chem/src/species.ts + chem/src/backmap.ts), not independently
// simulated -- only the molecule's overall position and orientation come from the validated
// coarse-grained run (engine/src/sim.ts), the same run whose area-per-lipid and bilayer thickness
// were checked against Cooke & Deserno 2005. The on-screen badge says exactly that, permanently,
// with no interaction required to see it.
//
// Atom counts explode fast (12 carbons/lipid ~ 38 atoms/lipid), so only a spatial SLICE of the
// membrane is drawn atom-by-atom; the rest is drawn as the same coarse beads main.ts uses. The
// split is reported on screen as a plain count, not hidden in a corner.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import atomsRaw from '../data/atoms.json'
import { loadParams } from '../engine/src/params'
import { createSystem } from '../engine/src/sim'
import { backmapLipid, backmapSystem, type BackmapAtom } from '../chem/src/backmap'

interface ElementInfo {
  vdw: number
  color: string
}
interface AtomsData {
  elements: Record<string, ElementInfo>
}
const atomsData = atomsRaw as unknown as AtomsData

// --- exposed parameters (brief: chain length must be a call parameter the scene can swap out for
// whatever the chemistry stage eventually produces -- never buried as a literal deep in the
// reconstruction logic; both backmapLipid/backmapSystem already take it as an argument, this is
// just where the viewer's own choice of value lives). ---------------------------------------------
const CARBONS_PER_LIPID = 12

// Nanometres per reduced-unit sigma. Not measured by this engine (Cooke & Deserno's model is
// dimensionless by construction) -- this is the standard literature mapping for this specific
// bead model: sigma ~ 1 nm, chosen so the engine's own validated bilayer-thickness gate corridor
// (4.0-6.0 sigma, data/literature.json) reads directly as 4-6 nm, the real range for a lipid
// bilayer. A viewer-only assumption, not a simulated quantity -- flagged here rather than in
// chem/src, which must stay free of model constants.
const SIGMA_NM = 1.0

// Simulation steps advanced per rendered frame -- same reasoning as main.ts's STEPS_PER_FRAME.
const STEPS_PER_FRAME = 10

// Demo membrane size and box. x,y match main.ts's proven values (dims>=3 on the neighbor grid,
// area/lipid near the Cooke & Deserno corridor for layoutBilayer's grid spacing); z is shrunk from
// main.ts's 26 to 10 so the bilayer fills more of the frame instead of floating in mostly empty
// box -- z has no periodic-grid constraint (engine/src/sim.ts's gridInvariantsHold only checks
// x,y), so this is free to choose independently.
const LIPIDS = 400
const BOX: [number, number, number] = [18, 18, 10]

// Half-width (sigma) of the spatial band, centered on the box, drawn atom-by-atom; every lipid
// outside it is drawn as the same coarse beads main.ts uses. A fixed slice (rather than tracking
// the camera) keeps which lipids are atomistic stable frame to frame instead of flickering as the
// camera moves.
const SLICE_HALF_WIDTH = BOX[0] * 0.1

// Ball-and-stick sizing: atoms shrunk well below their true van-der-Waals radius (data/atoms.json)
// so bonds stay visible between them (a literal VdW-radius draw would show touching/overlapping
// spheres with no visible stick, per the brief). Bond radius is derived from carbon's own VdW
// radius rather than an invented number, so it still traces back to data/atoms.json.
const ATOM_RADIUS_SCALE = 0.34
const BOND_RADIUS = atomsData.elements.C.vdw * 0.16

async function main(): Promise<void> {
  const params = loadParams()

  const canvas = document.getElementById('scene') as HTMLCanvasElement
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
  renderer.setPixelRatio(window.devicePixelRatio)
  renderer.setSize(window.innerWidth, window.innerHeight)

  const scene = new THREE.Scene()
  scene.background = new THREE.Color(0x0b0d10)

  const center = new THREE.Vector3(BOX[0] / 2, BOX[1] / 2, BOX[2] / 2)
  // Edge-on framing (brief: "a slab seen edge-on reads better than a wall of spheres"): the
  // camera looks along the membrane's own plane (roughly the -y axis) so the bilayer's thickness
  // (a handful of sigma, along z) reads as a visible band across the frame's full width (x),
  // instead of looking down the z axis (the membrane's normal) onto a packed sheet of head
  // groups where no bilayer structure is visible at all.
  const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 500)
  // layoutBilayer's membrane normal is the z axis (heads outward at +/-z, midplane at box.z/2) --
  // three.js's camera.up defaults to the y axis, which would roll the whole scene sideways in an
  // edge-on shot (the box's real "up" is z here, not y). Without this the slab renders as a
  // diagonal wall instead of the intended horizontal band.
  camera.up.set(0, 0, 1)
  camera.position.set(center.x + BOX[0] * 0.15, center.y - BOX[1] * 1.4, center.z + BOX[2] * 0.35)
  camera.lookAt(center)

  const controls = new OrbitControls(camera, renderer.domElement)
  controls.target.copy(center)
  controls.enableDamping = true

  scene.add(new THREE.AmbientLight(0xffffff, 0.75))
  const sun = new THREE.DirectionalLight(0xffffff, 0.9)
  sun.position.set(1, -1, 1.4)
  scene.add(sun)

  const sys = await createSystem({
    lipids: LIPIDS,
    box: BOX,
    seed: 1,
    layout: 'bilayer',
  })

  // --- coarse beads for the lipids NOT drawn atom-by-atom (same visual language as main.ts) ------
  const headRadius = params.beadSizes.head_head / 2
  const tailRadius = params.beadSizes.tail_tail / 2
  const headGeo = new THREE.SphereGeometry(headRadius, 12, 8)
  const tailGeo = new THREE.SphereGeometry(tailRadius, 10, 8)
  const headMat = new THREE.MeshStandardMaterial({ color: 0x4fc3f7 })
  const tailMat = new THREE.MeshStandardMaterial({ color: 0xffa552 })
  const headMesh = new THREE.InstancedMesh(headGeo, headMat, sys.lipids)
  const tailMesh = new THREE.InstancedMesh(tailGeo, tailMat, sys.lipids * 2)
  scene.add(headMesh, tailMesh)

  // --- ball-and-stick instanced meshes for the atomistic slice --------------------------------
  // Capacities are a safe upper bound (as if every lipid in the system were ever in the slice at
  // once), sized from one sample reconstruction rather than a guessed literal -- they only bound
  // buffer allocation, not what actually gets drawn each frame (mesh.count below tracks that).
  const sample = backmapLipid([0, 0, 2], [0, 0, 1], [0, 0, 0], CARBONS_PER_LIPID, SIGMA_NM)
  const atomCapacity = sys.lipids * sample.atoms.length
  const bondSegmentCapacity = sys.lipids * sample.bonds.length * 2 // two half-cylinders per bond

  const atomGeo = new THREE.SphereGeometry(1, 12, 8)
  const atomMat = new THREE.MeshStandardMaterial({ color: 0xffffff })
  const atomMesh = new THREE.InstancedMesh(atomGeo, atomMat, atomCapacity)
  scene.add(atomMesh)

  // Unit cylinder along the local +Y axis, radius 1 -- per-instance transform below scales it to
  // the actual bond radius/length and rotates it onto the real bond direction.
  const bondGeo = new THREE.CylinderGeometry(1, 1, 1, 6)
  const bondMat = new THREE.MeshStandardMaterial({ color: 0xffffff })
  const bondMesh = new THREE.InstancedMesh(bondGeo, bondMat, bondSegmentCapacity)
  scene.add(bondMesh)

  const dummy = new THREE.Object3D()
  const color = new THREE.Color()
  const up = new THREE.Vector3(0, 1, 0)
  const pA = new THREE.Vector3()
  const pB = new THREE.Vector3()
  const mid = new THREE.Vector3()
  const dir = new THREE.Vector3()

  function elementColor(element: string): THREE.Color {
    const info = atomsData.elements[element]
    if (!info) throw new Error(`viewer/molecular: unknown element "${element}" in data/atoms.json`)
    return color.set(info.color)
  }

  function elementRadius(element: string): number {
    const info = atomsData.elements[element]
    if (!info) throw new Error(`viewer/molecular: unknown element "${element}" in data/atoms.json`)
    return info.vdw * ATOM_RADIUS_SCALE
  }

  function drawAtomsAndBonds(atoms: BackmapAtom[], bonds: [number, number][]): number {
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i]
      dummy.position.set(a.position[0], a.position[1], a.position[2])
      dummy.scale.setScalar(elementRadius(a.element))
      dummy.rotation.set(0, 0, 0)
      dummy.updateMatrix()
      atomMesh.setMatrixAt(i, dummy.matrix)
      atomMesh.setColorAt(i, elementColor(a.element))
    }
    atomMesh.count = atoms.length
    atomMesh.instanceMatrix.needsUpdate = true
    if (atomMesh.instanceColor) atomMesh.instanceColor.needsUpdate = true

    // Each bond is drawn as two half-cylinders, one from the bond's midpoint to each end atom,
    // coloured by that end atom's own element -- the standard two-tone CPK ball-and-stick
    // convention, and honestly reported: window.molecular.bonds below counts exactly these
    // rendered segments, not the underlying graph edges.
    let seg = 0
    for (const [ia, ib] of bonds) {
      const atomA = atoms[ia]
      const atomB = atoms[ib]
      pA.set(atomA.position[0], atomA.position[1], atomA.position[2])
      pB.set(atomB.position[0], atomB.position[1], atomB.position[2])
      mid.copy(pA).add(pB).multiplyScalar(0.5)
      for (const [from, to, element] of [
        [pA, mid, atomA.element],
        [mid, pB, atomB.element],
      ] as const) {
        dir.copy(to).sub(from)
        const len = dir.length()
        dummy.position.copy(from).add(dir.clone().multiplyScalar(0.5))
        dummy.quaternion.setFromUnitVectors(up, dir.clone().normalize())
        dummy.scale.set(BOND_RADIUS, len, BOND_RADIUS)
        dummy.updateMatrix()
        bondMesh.setMatrixAt(seg, dummy.matrix)
        bondMesh.setColorAt(seg, elementColor(element))
        seg++
      }
    }
    bondMesh.count = seg
    bondMesh.instanceMatrix.needsUpdate = true
    if (bondMesh.instanceColor) bondMesh.instanceColor.needsUpdate = true
    return seg
  }

  /** Lipid indices whose head bead falls inside the fixed atomistic slab. */
  function sliceIndices(pos: Float32Array): number[] {
    const out: number[] = []
    const lo = center.x - SLICE_HALF_WIDTH
    const hi = center.x + SLICE_HALF_WIDTH
    for (let lip = 0; lip < sys.lipids; lip++) {
      const headX = pos[lip * 12]
      if (headX >= lo && headX <= hi) out.push(lip)
    }
    return out
  }

  function sliceArray(pos: Float32Array, indices: number[]): Float32Array {
    const out = new Float32Array(indices.length * 12)
    for (let k = 0; k < indices.length; k++) {
      out.set(pos.subarray(indices[k] * 12, indices[k] * 12 + 12), k * 12)
    }
    return out
  }

  function drawBeads(pos: Float32Array, atomisticSet: Set<number>): void {
    let hi = 0
    let ti = 0
    for (let lip = 0; lip < sys.lipids; lip++) {
      if (atomisticSet.has(lip)) continue // this lipid is drawn atom-by-atom instead
      const base = lip * 12
      for (let bead = 0; bead < 3; bead++) {
        const o = base + bead * 4
        dummy.position.set(pos[o], pos[o + 1], pos[o + 2])
        dummy.scale.setScalar(1)
        dummy.rotation.set(0, 0, 0)
        dummy.updateMatrix()
        if (pos[o + 3] === 0) headMesh.setMatrixAt(hi++, dummy.matrix)
        else tailMesh.setMatrixAt(ti++, dummy.matrix)
      }
    }
    headMesh.count = hi
    tailMesh.count = ti
    headMesh.instanceMatrix.needsUpdate = true
    tailMesh.instanceMatrix.needsUpdate = true
  }

  // --- honesty badge -----------------------------------------------------------------------------
  function renderBadge(shown: number, total: number): string {
    return (
      `The heavy-skeleton positions come from validated coarse-grained dynamics: ` +
      `the area per lipid and the thickness of this same membrane are checked against Cooke & Deserno 2005. The atomic ` +
      `detail is reconstructed from reference geometry (literature bond lengths and angles), not ` +
      `computed by an independent atomistic simulation: this is a reconstruction, not a simulation. ` +
      `${shown} of ${total} molecules are shown atom by atom; the rest are the same coarse-grained beads.`
    )
  }

  const badgeEl = document.getElementById('badge') as HTMLElement
  const countsEl = document.getElementById('counts') as HTMLElement

  const molecular = {
    frames: 0,
    molecules: 0,
    atoms: 0,
    bonds: 0,
    reconstructionBadge: renderBadge(0, sys.lipids),
  }
  ;(window as unknown as { molecular: typeof molecular }).molecular = molecular
  badgeEl.textContent = molecular.reconstructionBadge

  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight
    camera.updateProjectionMatrix()
    renderer.setSize(window.innerWidth, window.innerHeight)
  })

  async function frame(): Promise<void> {
    await sys.step(STEPS_PER_FRAME)
    const pos = await sys.positions()

    const indices = sliceIndices(pos)
    drawBeads(pos, new Set(indices))

    const { atoms, bonds } = backmapSystem(sliceArray(pos, indices), {
      carbons: CARBONS_PER_LIPID,
      sigmaNm: SIGMA_NM,
    })
    const drawnSegments = drawAtomsAndBonds(atoms, bonds)

    controls.update()
    renderer.render(scene, camera)

    molecular.frames++
    molecular.molecules = indices.length
    molecular.atoms = atoms.length
    molecular.bonds = drawnSegments
    molecular.reconstructionBadge = renderBadge(indices.length, sys.lipids)
    badgeEl.textContent = molecular.reconstructionBadge
    countsEl.textContent =
      `frame ${molecular.frames} · atomistic ${molecular.molecules}/${sys.lipids} molecules · ` +
      `${molecular.atoms} atoms · ${molecular.bonds} bonds (segments)`

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
