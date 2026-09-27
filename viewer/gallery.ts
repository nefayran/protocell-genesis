/// <reference types="vite/client" />
// Snapshot gallery: renders the checkpoints exported by soup/cli/export-gallery.ts with WebGL, so it
// runs in any current browser (the live simulation in run.html needs WebGPU). Beads are drawn as
// shaded sphere impostors: one point per bead, with the sphere's normal and depth reconstructed in
// the fragment shader, which keeps 150 000 beads interactive on a laptop GPU.
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import galleryIndex from './gallery-data/index.json'

interface Snapshot {
  id: string
  campaign: 'periodic' | 'confined'
  caption: string
  step: number
  phase: 'wet' | 'dry' | 'none'
  box: [number, number, number]
  particles: number
  beads: number
  amphiphiles: number
  aggregates: number
  amphiphilesInLargest: number
  wrappingAxes: number | null
  encapsulatedWater: number | null
  closureThreshold: number | null
  parcelRadius: number | null
}

// index.json is written by the exporter; its inferred type widens the literal unions, hence the cast.
const snapshots = galleryIndex.snapshots as unknown as Snapshot[]
const binUrls = import.meta.glob('./gallery-data/*.bin', { query: '?url', import: 'default', eager: true }) as Record<string, string>

// Class ids written by the exporter, grouped into the three layers the page can toggle.
const LAYERS = {
  amphiphiles: { classes: [0, 1, 2, 3], opaque: true },
  free: { classes: [4], opaque: false },
  water: { classes: [5], opaque: false },
} as const
type LayerName = keyof typeof LAYERS

// Radius in sigma and sRGB colour per class. Heads are drawn slightly larger than tails so the
// surface of an aggregate reads as a surface.
const STYLE: Record<number, { radius: number; color: string; alpha: number }> = {
  0: { radius: 0.56, color: '#f2dfb8', alpha: 1 },
  1: { radius: 0.5, color: '#2f7f86', alpha: 1 },
  2: { radius: 0.56, color: '#f2dfb8', alpha: 1 },
  3: { radius: 0.5, color: '#2f7f86', alpha: 1 },
  4: { radius: 0.3, color: '#8795a3', alpha: 0.32 },
  5: { radius: 0.22, color: '#3d7fd6', alpha: 0.28 },
}
const BACKGROUND = new THREE.Color('#05080d')

const params = new URLSearchParams(location.search)
const shotMode = params.get('shot') === '1'
if (shotMode) document.body.classList.add('shot')
if (shotMode && params.get('label') === '1') document.body.classList.add('labelled')

const canvas = document.getElementById('scene') as HTMLCanvasElement
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: shotMode })
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
renderer.setClearColor(BACKGROUND, 1)

const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera(32, 1, 0.5, 2000)
const controls = new OrbitControls(camera, canvas)
controls.enableDamping = true
controls.autoRotateSpeed = 0.55

const vertexShader = /* glsl */ `
  attribute vec3 aColor;
  attribute float aRadius;
  attribute float aAlpha;
  uniform float uPixelsPerUnit;
  varying vec3 vColor;
  varying float vAlpha;
  varying vec3 vViewPos;
  varying float vRadius;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = max(1.5, 2.0 * aRadius * uPixelsPerUnit / -mv.z);
    vColor = aColor;
    vAlpha = aAlpha;
    vViewPos = mv.xyz;
    vRadius = aRadius;
  }
`

const fragmentShader = /* glsl */ `
  uniform mat4 projectionMatrix;
  uniform vec3 uBackground;
  uniform float uFogNear;
  uniform float uFogFar;
  uniform bool uOpaque;
  varying vec3 vColor;
  varying float vAlpha;
  varying vec3 vViewPos;
  varying float vRadius;
  void main() {
    vec2 p = gl_PointCoord * 2.0 - 1.0;
    p.y = -p.y;
    float d2 = dot(p, p);
    if (d2 > 1.0) discard;
    vec3 n = vec3(p, sqrt(1.0 - d2));
    vec3 key = normalize(vec3(-0.45, 0.6, 0.66));
    float diffuse = max(dot(n, key), 0.0);
    float spec = pow(max(dot(reflect(-key, n), vec3(0.0, 0.0, 1.0)), 0.0), 28.0);
    float rim = pow(1.0 - n.z, 2.0);
    vec3 color = vColor * (0.28 + 0.72 * diffuse) + vec3(0.22) * spec + vColor * rim * 0.18;
    vec3 fragView = vViewPos + vec3(p * vRadius, n.z * vRadius);
    float fog = smoothstep(uFogNear, uFogFar, -fragView.z);
    color = mix(color, uBackground, fog * 0.82);
    if (uOpaque) {
      vec4 clip = projectionMatrix * vec4(fragView, 1.0);
      gl_FragDepth = 0.5 * (clip.z / clip.w) + 0.5;
      gl_FragColor = vec4(color, 1.0);
    } else {
      gl_FragDepth = gl_FragCoord.z;
      gl_FragColor = vec4(color, vAlpha * (1.0 - fog * 0.7));
    }
  }
`

const uniforms = {
  uPixelsPerUnit: { value: 1 },
  uBackground: { value: new THREE.Vector3(BACKGROUND.r, BACKGROUND.g, BACKGROUND.b) },
  uFogNear: { value: 0 },
  uFogFar: { value: 1 },
}

function material(opaque: boolean): THREE.ShaderMaterial {
  return new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: { ...uniforms, uOpaque: { value: opaque } },
    transparent: !opaque,
    depthWrite: opaque,
  })
}

const layerObjects: Record<LayerName, THREE.Points> = {
  amphiphiles: new THREE.Points(new THREE.BufferGeometry(), material(true)),
  free: new THREE.Points(new THREE.BufferGeometry(), material(false)),
  water: new THREE.Points(new THREE.BufferGeometry(), material(false)),
}
layerObjects.free.renderOrder = 1
layerObjects.water.renderOrder = 2
for (const obj of Object.values(layerObjects)) scene.add(obj)

const boxLines = new THREE.LineSegments(
  new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
  new THREE.LineBasicMaterial({ color: '#2c3d50' }),
)
scene.add(boxLines)

// The parcel's wall, drawn as a sparse globe so it reads as a boundary without hiding anything.
function globe(): THREE.LineSegments {
  const pts: number[] = []
  const seg = 96
  for (let lat = -60; lat <= 60; lat += 30) {
    const phi = (lat * Math.PI) / 180
    for (let i = 0; i < seg; i++) {
      const a = (i / seg) * Math.PI * 2
      const b = ((i + 1) / seg) * Math.PI * 2
      pts.push(Math.cos(phi) * Math.cos(a), Math.sin(phi), Math.cos(phi) * Math.sin(a))
      pts.push(Math.cos(phi) * Math.cos(b), Math.sin(phi), Math.cos(phi) * Math.sin(b))
    }
  }
  for (let lon = 0; lon < 180; lon += 30) {
    const t = (lon * Math.PI) / 180
    for (let i = 0; i < seg; i++) {
      const a = (i / seg) * Math.PI * 2
      const b = ((i + 1) / seg) * Math.PI * 2
      pts.push(Math.cos(a) * Math.cos(t), Math.sin(a), Math.cos(a) * Math.sin(t))
      pts.push(Math.cos(b) * Math.cos(t), Math.sin(b), Math.cos(b) * Math.sin(t))
    }
  }
  const g = new THREE.BufferGeometry()
  g.setAttribute('position', new THREE.Float32BufferAttribute(pts, 3))
  return new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: '#2a5d86', transparent: true, opacity: 0.45 }))
}
const parcel = globe()
scene.add(parcel)

const cache = new Map<string, Promise<ArrayBuffer>>()
function load(id: string): Promise<ArrayBuffer> {
  let p = cache.get(id)
  if (!p) {
    const url = binUrls[`./gallery-data/${id}.bin`]
    if (!url) throw new Error(`no exported file for snapshot ${id}`)
    p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`)
      return r.arrayBuffer()
    })
    cache.set(id, p)
  }
  return p
}

function fillLayers(snap: Snapshot, buffer: ArrayBuffer): void {
  const n = snap.beads
  if (buffer.byteLength !== n * 7) throw new Error(`${snap.id}: expected ${n * 7} bytes, got ${buffer.byteLength}`)
  const xyz = new Uint16Array(buffer, 0, n * 3)
  const cls = new Uint8Array(buffer, n * 6, n)
  const [bx, by, bz] = snap.box
  for (const [name, layer] of Object.entries(LAYERS) as [LayerName, (typeof LAYERS)[LayerName]][]) {
    const wanted = new Set<number>(layer.classes)
    let count = 0
    for (let i = 0; i < n; i++) if (wanted.has(cls[i])) count++
    const pos = new Float32Array(count * 3)
    const col = new Float32Array(count * 3)
    const rad = new Float32Array(count)
    const alp = new Float32Array(count)
    const tmp = new THREE.Color()
    let k = 0
    for (let i = 0; i < n; i++) {
      const c = cls[i]
      if (!wanted.has(c)) continue
      pos[k * 3] = (xyz[i * 3] / 65535 - 0.5) * bx
      pos[k * 3 + 1] = (xyz[i * 3 + 2] / 65535 - 0.5) * bz
      pos[k * 3 + 2] = (xyz[i * 3 + 1] / 65535 - 0.5) * by
      const s = STYLE[c]
      tmp.set(s.color)
      col[k * 3] = tmp.r
      col[k * 3 + 1] = tmp.g
      col[k * 3 + 2] = tmp.b
      rad[k] = s.radius
      alp[k] = s.alpha
      k++
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    g.setAttribute('aColor', new THREE.BufferAttribute(col, 3))
    g.setAttribute('aRadius', new THREE.BufferAttribute(rad, 1))
    g.setAttribute('aAlpha', new THREE.BufferAttribute(alp, 1))
    layerObjects[name].geometry.dispose()
    layerObjects[name].geometry = g
  }
  boxLines.scale.set(bx, bz, by)
  boxLines.visible = snap.campaign === 'periodic'
  parcel.visible = snap.parcelRadius !== null
  if (snap.parcelRadius !== null) parcel.scale.setScalar(snap.parcelRadius)
}

// Framing is fixed per campaign, from its wet extent, so the dry phase visibly shrinks.
function extentOf(campaign: Snapshot['campaign']): number {
  const wet = snapshots.find((s) => s.campaign === campaign && s.phase === 'wet')!
  return campaign === 'periodic' ? wet.box[0] * 1.2 : wet.parcelRadius! * 2
}

let current: Snapshot | null = null
let framedCampaign: Snapshot['campaign'] | null = null

function frame(campaign: Snapshot['campaign']): void {
  const extent = extentOf(campaign)
  const az = (Number(params.get('az') ?? 38) * Math.PI) / 180
  const el = (Number(params.get('el') ?? 22) * Math.PI) / 180
  const zoom = Number(params.get('zoom') ?? 1)
  // The field of view is vertical; on a portrait screen the narrower horizontal one decides the fit.
  const fitVertical = extent / (2 * Math.tan((camera.fov * Math.PI) / 360))
  const dist = (Math.max(fitVertical, fitVertical / camera.aspect) * 1.5) / zoom
  camera.position.set(dist * Math.cos(el) * Math.sin(az), dist * Math.sin(el), dist * Math.cos(el) * Math.cos(az))
  controls.target.set(0, 0, 0)
  controls.minDistance = extent * 0.25
  controls.maxDistance = extent * 6
  controls.update()
  framedCampaign = campaign
}

const fmt = (x: number) => x.toLocaleString('en-US')
const el = (id: string) => document.getElementById(id)!

function showNumbers(s: Snapshot): void {
  el('n-step').textContent = fmt(s.step)
  el('n-phase').textContent = s.phase === 'dry' ? 'dry (water evaporated)' : 'wet'
  el('n-box').textContent = `${s.box[0].toFixed(1)} σ`
  el('n-particles').textContent = fmt(s.particles)
  el('n-amph').textContent = fmt(s.amphiphiles)
  el('n-aggs').textContent = fmt(s.aggregates)
  el('n-share').textContent = s.amphiphiles > 0 ? `${((100 * s.amphiphilesInLargest) / s.amphiphiles).toFixed(2)} %` : '–'
  el('n-wrap').textContent = s.wrappingAxes === null ? 'not measured' : `${s.wrappingAxes} of 3`
  const wet = s.phase !== 'dry' && s.encapsulatedWater !== null
  el('n-water').textContent = s.phase === 'dry' ? 'n/a, dry phase' : wet ? `${fmt(s.encapsulatedWater!)} beads` : 'not measured'
  el('n-need').textContent = wet && s.closureThreshold !== null ? `≥ ${fmt(s.closureThreshold)} beads` : '–'
  el('n-verdict').textContent =
    s.campaign === 'periodic'
      ? 'After the merge the aggregate wrapped all three axes in 23 of 23 wet snapshots and trapped no water in any of them.'
      : 'Without a periodic box the aggregate wrapped no axis in any snapshot, and the water trapped inside stayed at 0.'
}

function renderStepButtons(campaign: Snapshot['campaign']): void {
  const box = el('steps')
  box.textContent = ''
  for (const s of snapshots.filter((x) => x.campaign === campaign)) {
    const b = document.createElement('button')
    b.textContent = `step ${fmt(s.step)}${s.phase === 'dry' ? ' · dry' : ''}`
    b.setAttribute('aria-pressed', String(current?.id === s.id))
    b.addEventListener('click', () => void select(s.id))
    box.appendChild(b)
  }
  el('tab-periodic').setAttribute('aria-pressed', String(campaign === 'periodic'))
  el('tab-confined').setAttribute('aria-pressed', String(campaign === 'confined'))
}

let requestSeq = 0
async function select(id: string): Promise<void> {
  const snap = snapshots.find((s) => s.id === id)
  if (!snap) throw new Error(`unknown snapshot ${id}`)
  const seq = ++requestSeq
  el('loading').style.display = 'grid'
  const buffer = await load(id)
  if (seq !== requestSeq) return
  fillLayers(snap, buffer)
  current = snap
  if (framedCampaign !== snap.campaign) frame(snap.campaign)
  renderStepButtons(snap.campaign)
  el('caption').textContent = snap.caption
  el('shot-label').innerHTML =
    `step ${fmt(snap.step)}${snap.phase === 'dry' ? ' · dry phase' : ''}<br><span>${fmt(snap.amphiphiles)} amphiphiles in ${fmt(snap.aggregates)} ` +
    `${snap.aggregates === 1 ? 'aggregate' : 'aggregates'}</span>`
  showNumbers(snap)
  el('loading').style.display = 'none'
  const url = new URL(location.href)
  if (!shotMode) {
    url.searchParams.set('snap', id)
    history.replaceState(null, '', url)
  }
}

function applyToggles(): void {
  layerObjects.free.visible = (el('show-free') as HTMLInputElement).checked
  layerObjects.water.visible = (el('show-water') as HTMLInputElement).checked
  controls.autoRotate = (el('rotate') as HTMLInputElement).checked
}

for (const id of ['show-free', 'show-water', 'rotate']) el(id).addEventListener('change', applyToggles)
el('tab-periodic').addEventListener('click', () => void select(snapshots.filter((s) => s.campaign === 'periodic').at(-1)!.id))
el('tab-confined').addEventListener('click', () => void select(snapshots.filter((s) => s.campaign === 'confined').at(-1)!.id))

if (params.has('free')) (el('show-free') as HTMLInputElement).checked = params.get('free') === '1'
if (params.has('water')) (el('show-water') as HTMLInputElement).checked = params.get('water') === '1'
if (params.has('rotate')) (el('rotate') as HTMLInputElement).checked = params.get('rotate') === '1'
applyToggles()

function resize(): void {
  const w = window.innerWidth
  const h = window.innerHeight
  renderer.setSize(w, h, false)
  camera.aspect = w / h
  // On a phone the panel sits at the bottom, so the picture is shifted up to stay clear of it.
  if (w < 760 && !shotMode) camera.setViewOffset(w, h, 0, h * 0.2, w, h)
  else camera.clearViewOffset()
  camera.updateProjectionMatrix()
  uniforms.uPixelsPerUnit.value = renderer.getDrawingBufferSize(new THREE.Vector2()).y / (2 * Math.tan((camera.fov * Math.PI) / 360))
}
window.addEventListener('resize', resize)
resize()

let framesSinceLoad = 0
function tick(): void {
  controls.update()
  const d = camera.position.distanceTo(controls.target)
  const r = current ? extentOf(current.campaign) * 0.75 : 1
  uniforms.uFogNear.value = Math.max(0, d - r * 0.6)
  uniforms.uFogFar.value = d + r * 1.4
  renderer.render(scene, camera)
  if (current && ++framesSinceLoad === 3) (window as unknown as { __galleryReady: boolean }).__galleryReady = true
  requestAnimationFrame(tick)
}

const initial = params.get('snap') ?? snapshots.filter((s) => s.campaign === 'periodic').at(-1)!.id
select(initial)
  .then(() => {
    framesSinceLoad = 0
    requestAnimationFrame(tick)
  })
  .catch((err: Error) => {
    el('loading').textContent = `Could not load the snapshot: ${err.message}`
    throw err
  })
