# L0-L2: membrane engine, metrics, and automated exam - implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring stage C of the simulation to a working state: self-assembly of a lipid bilayer and a closed vesicle in WebGPU, together with metrics that are checked against published values, and a report that renders a verdict for gate 6.

**Architecture:** The physics lives in WGSL and runs on the same code in the interactive viewer and in verification. The model's numeric constants are not hardcoded; they are read from `data/params.json`. Verification runs through the system Chrome in headless mode under puppeteer-core control; metrics are read by calling a function on the page, not from a DOM dump. Pure math (undulation spectrum, cavity flood fill, gate verdicts) is implemented in TypeScript and tested without a GPU, and its GPU versions are checked against this reference implementation.

**Tech Stack:** TypeScript, Vite, Vitest, WGSL (WebGPU), three.js, puppeteer-core with the system Chrome. Node 22 (already on the system). No UI framework.

**Spec:** `docs/superpowers/specs/2026-08-15-protocell-genesis-design.md`

## Global Constraints

- `engine/` must not contain a single hardcoded numeric model constant: everything is read from `data/params.json`. Task 2 adds a guard test that checks this by scanning files.
- The Cooke & Deserno 2005 model parameters are used exactly at their published values: `b_hh = b_ht = 0.95sigma`, `b_tt = sigma`, WCA with cutoff `r_c = 2^(1/6)*b`, FENE `k_bond = 30epsilon/sigma^2` and `r_inf = 1.5sigma`, bending `k_bend = 10epsilon/sigma^2` around `4sigma`, tail attraction `-epsilon cos^2[pi(r-r_c)/(2w_c)]`, Langevin thermostat `Gamma = 1/tau`, step `dt = 0.01tau`, operating point `kT/epsilon = 1.1` and `w_c = 1.6sigma`.
- Reduced units: `sigma = 1`, `epsilon = 1`, `m = 1`, `tau = sigma*sqrt(m/epsilon) = 1`.
- Gate 6 literature targets: area per lipid **1.1-1.5 sigma^2**, bilayer thickness **~5sigma**, bending modulus **5-50 kT**.
- Verification must read the result via an in-page call (CDP `page.evaluate`). Using `--dump-dom` is forbidden: it has been measured to return content before the async GPU work finishes, and a live GPU looks absent.
- Chrome launches with the `--enable-unsafe-webgpu` flag; the path to the system Chrome is `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`. Measured: adapter `apple/metal-3`, the compute pass executes.
- Every claim about behavior is backed by a measurement that lands in `verify/out/gates.json`. Phrasing like "runs fast" or "looks correct" does not belong in the report.
- GPU reductions are order-dependent, so all GPU-vs-CPU comparisons use a tolerance, and group sizes are fixed.

---

## File Structure

Created in this plan:

- `package.json`, `tsconfig.json`, `vite.config.ts` - build and tests.
- `data/params.json` - model parameters, the single source of constants.
- `data/literature.json` - gate targets with references and conditions.
- `engine/src/gpu.ts` - WebGPU device acquisition, shader loading, helper buffers.
- `engine/src/params.ts` - reading and validating `data/params.json`, derived quantities.
- `engine/src/forces.ts` - force-kernel pipeline assembly, entry point for probe tests.
- `engine/src/sim.ts` - system state, integration step, MC box-area move.
- `engine/src/metrics.ts` - area per lipid, density profile, thickness.
- `engine/src/spectrum.ts` - height field, discrete transform, kappa estimation.
- `engine/src/closure.ts` - occupancy grid, outside flood fill, cavity volume (TypeScript reference).
- `engine/src/index.ts` - public facade: everything called from tests and from the viewer.
- `engine/wgsl/forces.wgsl` - potentials and their derivatives.
- `engine/wgsl/neighbor.wgsl` - neighbor grid.
- `engine/wgsl/integrate.wgsl` - Langevin step.
- `engine/wgsl/closure.wgsl` - GPU flood fill.
- `verify/gates.ts` - computing gate verdicts from metrics and `literature.json`.
- `verify/report.ts` - generating `verify/out/report.html`.
- `verify/run.ts` - running the full exam with a single command.
- `viewer/index.html`, `viewer/main.ts` - three.js viewer, panel, time-scale badge.
- `tests/helpers/gpu.ts` - launching Vite and Chrome, the page runner.
- `tests/runner.html` - page that publishes the engine facade to `window.api`.
- `tests/*.test.ts` - per-task tests.

---

### Task 1: Project scaffolding and the GPU bridge

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`
- Create: `engine/src/gpu.ts`, `engine/src/index.ts`
- Create: `tests/runner.html`, `tests/helpers/gpu.ts`
- Test: `tests/gpu-smoke.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `gpuSmoke(): Promise<{vendor: string, architecture: string, doubled: number[]}>` from `engine/src/index.ts`; `gpuPage(): Promise<Page>` and `shutdownGpu(): Promise<void>` from `tests/helpers/gpu.ts`.

- [ ] **Step 1: Set up dependencies and configuration**

`package.json`:

```json
{
  "name": "protocell-genesis",
  "private": true,
  "type": "module",
  "scripts": {
    "dev": "vite",
    "test": "vitest run",
    "verify": "vite build && tsx verify/run.ts"
  },
  "devDependencies": {
    "@types/three": "^0.169.0",
    "@webgpu/types": "^0.1.44",
    "puppeteer-core": "^23.0.0",
    "tsx": "^4.19.0",
    "typescript": "^5.6.0",
    "vite": "^5.4.0",
    "vitest": "^2.1.0"
  },
  "dependencies": {
    "three": "^0.169.0"
  }
}
```

`tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "ESNext",
    "moduleResolution": "bundler",
    "strict": true,
    "resolveJsonModule": true,
    "types": ["@webgpu/types", "vitest/globals"],
    "noEmit": true
  },
  "include": ["engine", "viewer", "verify", "tests"]
}
```

`vite.config.ts`:

```ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  root: '.',
  server: { port: 0 },
  test: { environment: 'node', testTimeout: 120_000, hookTimeout: 60_000 },
})
```

Import `defineConfig` specifically from `vitest/config`, not from `vite`: the `vite` version has no `test` field, and test settings, including timeouts, will be silently ignored.

Run: `npm install`

- [ ] **Step 2: Write a failing test for the GPU bridge**

`tests/gpu-smoke.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('headless Chrome returns a WebGPU adapter and executes a compute pass', async () => {
  const page = await gpuPage()
  const info = await page.evaluate(() => (window as any).api.gpuSmoke())
  expect(info.vendor).toBe('apple')
  expect(info.doubled).toEqual([2, 4, 6, 8])
})
```

- [ ] **Step 3: Run the test and confirm it fails**

Run: `npx vitest run tests/gpu-smoke.test.ts`
Expected: FAIL - the `./helpers/gpu` module is not found.

- [ ] **Step 4: Implement the page runner and the launch helper**

`tests/runner.html`:

```html
<!doctype html>
<meta charset="utf-8" />
<title>engine runner</title>
<script type="module">
  import * as api from '/engine/src/index.ts'
  window.api = api
  window.__ready = true
</script>
```

`tests/helpers/gpu.ts`:

```ts
import { createServer, type ViteDevServer } from 'vite'
import puppeteer, { type Browser, type Page } from 'puppeteer-core'

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

let server: ViteDevServer | null = null
let browser: Browser | null = null

export async function gpuPage(): Promise<Page> {
  if (!server) {
    server = await createServer({ server: { port: 0 } })
    await server.listen()
  }
  if (!browser) {
    browser = await puppeteer.launch({
      executablePath: CHROME,
      headless: true,
      args: ['--enable-unsafe-webgpu', '--no-sandbox'],
    })
  }
  const base = server.resolvedUrls!.local[0]
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  await page.goto(new URL('tests/runner.html', base).href, { waitUntil: 'load' })
  await page.waitForFunction('window.__ready === true')
  return page
}

export async function shutdownGpu(): Promise<void> {
  await browser?.close()
  await server?.close()
  browser = null
  server = null
}
```

- [ ] **Step 5: Implement `gpuSmoke`**

`engine/src/gpu.ts`:

```ts
export interface Gpu {
  device: GPUDevice
  adapterInfo: { vendor: string; architecture: string }
}

export async function getGpu(): Promise<Gpu> {
  if (!navigator.gpu) throw new Error('navigator.gpu is missing')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('WebGPU adapter was not granted')
  const device = await adapter.requestDevice()
  const info = adapter.info ?? ({} as GPUAdapterInfo)
  return {
    device,
    adapterInfo: { vendor: info.vendor ?? '', architecture: info.architecture ?? '' },
  }
}

export async function readBack(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<Float32Array> {
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(src, 0, dst, 0, bytes)
  device.queue.submit([enc.finish()])
  await dst.mapAsync(GPUMapMode.READ)
  const out = new Float32Array(dst.getMappedRange().slice(0))
  dst.unmap()
  dst.destroy()
  return out
}

export function storageBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buf, 0, data)
  return buf
}
```

`engine/src/index.ts`:

```ts
import { getGpu, readBack, storageBuffer } from './gpu'

export async function gpuSmoke() {
  const { device, adapterInfo } = await getGpu()
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
  b[i.x] = b[i.x] * 2.0;
}`,
  })
  const buf = storageBuffer(device, new Float32Array([1, 2, 3, 4]))
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } })
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: buf } }],
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bind)
  pass.dispatchWorkgroups(1)
  pass.end()
  device.queue.submit([enc.finish()])
  const doubled = Array.from(await readBack(device, buf, 16))
  return { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture, doubled }
}
```

- [ ] **Step 6: Run the test and confirm it passes**

Run: `npx vitest run tests/gpu-smoke.test.ts`
Expected: PASS. If it fails with "adapter was not granted", check that the `--enable-unsafe-webgpu` flag reached Chrome, and do not try to read the result via `--dump-dom`.

- [ ] **Step 7: Commit**

```bash
git add package.json tsconfig.json vite.config.ts engine/src/gpu.ts engine/src/index.ts tests/runner.html tests/helpers/gpu.ts tests/gpu-smoke.test.ts package-lock.json
git commit -m "feat: WebGPU bridge and headless test harness"
```

---

### Task 2: Model parameters as the single source of constants

**Files:**
- Create: `data/params.json`, `engine/src/params.ts`
- Test: `tests/params.test.ts`

**Interfaces:**
- Consumes: nothing.
- Produces: `loadParams(): Params`, the `Params` type with fields `sigma, epsilon, beadSizes.{head_head,head_tail,tail_tail}, fene.{k,rInf}, bend.{k,r0}, attraction.wc, thermostat.{gamma,kT}, integrator.dt`, and `wcaCutoff(b: number): number`.

- [ ] **Step 1: Write failing tests**

`tests/params.test.ts`:

```ts
import { expect, test } from 'vitest'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { loadParams, wcaCutoff } from '../engine/src/params'

test('parameters match the published Cooke & Deserno 2005 values', () => {
  const p = loadParams()
  expect(p.beadSizes.head_head).toBe(0.95)
  expect(p.beadSizes.head_tail).toBe(0.95)
  expect(p.beadSizes.tail_tail).toBe(1.0)
  expect(p.fene.k).toBe(30)
  expect(p.fene.rInf).toBe(1.5)
  expect(p.bend.k).toBe(10)
  expect(p.bend.r0).toBe(4)
  expect(p.attraction.wc).toBe(1.6)
  expect(p.thermostat.gamma).toBe(1)
  expect(p.thermostat.kT).toBe(1.1)
  expect(p.integrator.dt).toBe(0.01)
})

test('the WCA cutoff equals 2^(1/6)*b', () => {
  expect(wcaCutoff(1)).toBeCloseTo(1.1224620483, 9)
  expect(wcaCutoff(0.95)).toBeCloseTo(0.95 * 2 ** (1 / 6), 9)
})

test('the engine has no hardcoded model constants', () => {
  const forbidden = ['0.95', '1.6', '30.0', '1.5', '4.0', '1.1', '0.01']
  const dirs = ['engine/src', 'engine/wgsl']
  const offenders: string[] = []
  for (const dir of dirs) {
    for (const name of readdirSync(dir)) {
      if (name === 'params.ts') continue
      const text = readFileSync(join(dir, name), 'utf8')
      for (const lit of forbidden) if (text.includes(lit)) offenders.push(`${dir}/${name}: ${lit}`)
    }
  }
  expect(offenders).toEqual([])
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/params.test.ts`
Expected: FAIL - `engine/src/params` not found.

- [ ] **Step 3: Implement the data and the loader**

`data/params.json`:

```json
{
  "source": "Cooke & Deserno 2005, arXiv:cond-mat/0509218",
  "rank": "A",
  "units": "sigma=1, epsilon=1, m=1, tau=1",
  "sigma": 1.0,
  "epsilon": 1.0,
  "beadSizes": { "head_head": 0.95, "head_tail": 0.95, "tail_tail": 1.0 },
  "fene": { "k": 30.0, "rInf": 1.5 },
  "bend": { "k": 10.0, "r0": 4.0 },
  "attraction": { "wc": 1.6 },
  "thermostat": { "gamma": 1.0, "kT": 1.1 },
  "integrator": { "dt": 0.01 },
  "kTRange": [0.6, 1.1]
}
```

`engine/src/params.ts`:

```ts
import raw from '../../data/params.json'

export interface Params {
  source: string
  rank: string
  sigma: number
  epsilon: number
  beadSizes: { head_head: number; head_tail: number; tail_tail: number }
  fene: { k: number; rInf: number }
  bend: { k: number; r0: number }
  attraction: { wc: number }
  thermostat: { gamma: number; kT: number }
  integrator: { dt: number }
  kTRange: [number, number]
}

const REQUIRED = [
  'sigma', 'epsilon', 'beadSizes', 'fene', 'bend', 'attraction', 'thermostat', 'integrator',
] as const

export function loadParams(): Params {
  const p = raw as unknown as Params
  for (const key of REQUIRED) {
    if ((p as Record<string, unknown>)[key] === undefined) {
      throw new Error(`data/params.json: missing field ${key}`)
    }
  }
  return p
}

export function wcaCutoff(b: number): number {
  return Math.pow(2, 1 / 6) * b
}

/** Flat array for the WGSL uniform buffer. Order must match struct Params in forces.wgsl. */
export function paramsToUniform(p: Params): Float32Array {
  return new Float32Array([
    p.sigma, p.epsilon,
    p.beadSizes.head_head, p.beadSizes.head_tail, p.beadSizes.tail_tail,
    p.fene.k, p.fene.rInf,
    p.bend.k, p.bend.r0,
    p.attraction.wc,
    p.thermostat.kT, p.thermostat.gamma,
    p.integrator.dt,
    0, 0, 0,
  ])
}
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/params.test.ts`
Expected: PASS all three.

- [ ] **Step 5: Commit**

```bash
git add data/params.json engine/src/params.ts tests/params.test.ts
git commit -m "feat: model parameters as the single source of constants"
```

---

### Task 3: Force kernels and analytic verification

**Files:**
- Create: `engine/wgsl/forces.wgsl`, `engine/src/forces.ts`
- Modify: `engine/src/index.ts` (export `probeForces`)
- Test: `tests/forces.test.ts`

**Interfaces:**
- Consumes: `loadParams`, `paramsToUniform`, `getGpu`, `storageBuffer`, `readBack`.
- Produces: `probeForces(kind: 'wca' | 'fene' | 'bend' | 'attr', b: number, radii: number[]): Promise<number[]>` - returns `dV/dr` at the given points. Sign convention: a positive value is repulsion.

- [ ] **Step 1: Write failing tests**

`tests/forces.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadParams, wcaCutoff } from '../engine/src/params'

afterAll(shutdownGpu)

const p = loadParams()

function wcaAnalytic(r: number, b: number): number {
  if (r >= wcaCutoff(b)) return 0
  const b6 = (b / r) ** 6
  return (-24 * p.epsilon / r) * (2 * b6 * b6 - b6)
}

test('WCA matches the analytic derivative and cuts off correctly', async () => {
  const page = await gpuPage()
  const radii = [0.8, 0.9, 1.0, 1.05, 1.2]
  const got = await page.evaluate((r) => (window as any).api.probeForces('wca', 1.0, r), radii)
  radii.forEach((r, i) => expect(got[i]).toBeCloseTo(wcaAnalytic(r, 1.0), 4))
  expect(got[4]).toBe(0)
})

test('FENE pulls toward the center and grows near the extension limit', async () => {
  const page = await gpuPage()
  const radii = [0.0, 0.5, 1.0, 1.4]
  const got = await page.evaluate((r) => (window as any).api.probeForces('fene', 1.0, r), radii)
  radii.forEach((r, i) => {
    const want = (p.fene.k * r) / (1 - (r / p.fene.rInf) ** 2)
    expect(got[i]).toBeCloseTo(want, 4)
  })
  expect(got[3]).toBeGreaterThan(got[2])
})

test('the bending potential is linear around r0', async () => {
  const page = await gpuPage()
  const radii = [3.0, 4.0, 5.0]
  const got = await page.evaluate((r) => (window as any).api.probeForces('bend', 1.0, r), radii)
  expect(got[0]).toBeCloseTo(p.bend.k * (3 - p.bend.r0), 5)
  expect(got[1]).toBeCloseTo(0, 6)
  expect(got[2]).toBeCloseTo(p.bend.k * (5 - p.bend.r0), 5)
})

test('tail attraction goes smoothly to zero at both ends', async () => {
  const page = await gpuPage()
  const rc = wcaCutoff(p.beadSizes.tail_tail)
  const wc = p.attraction.wc
  const radii = [rc - 0.05, rc, rc + wc / 2, rc + wc, rc + wc + 0.05]
  const got = await page.evaluate((r) => (window as any).api.probeForces('attr', 1.0, r), radii)
  expect(got[0]).toBe(0)
  expect(got[1]).toBeCloseTo(0, 6)
  expect(got[4]).toBe(0)
  expect(got[3]).toBeCloseTo(0, 6)
  const x = Math.PI * (wc / 2) / (2 * wc)
  expect(got[2]).toBeCloseTo((p.epsilon * Math.PI * Math.sin(2 * x)) / (2 * wc), 4)
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/forces.test.ts`
Expected: FAIL - `api.probeForces` is not a function.

- [ ] **Step 3: Implement the kernels**

`engine/wgsl/forces.wgsl`:

```wgsl
struct Params {
  sigma: f32, epsilon: f32,
  b_hh: f32, b_ht: f32, b_tt: f32,
  k_fene: f32, r_inf: f32,
  k_bend: f32, r_bend: f32,
  wc: f32,
  kT: f32, gamma: f32,
  dt: f32,
  pad0: f32, pad1: f32, pad2: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> radii: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> probe: vec4<f32>; // x = kind, y = b

fn wca_cut(b: f32) -> f32 { return pow(2.0, 0.16666667) * b; }

fn wca_dv(r: f32, b: f32) -> f32 {
  if (r >= wca_cut(b) || r <= 0.0) { return 0.0; }
  let s6 = pow(b / r, 6.0);
  return -24.0 * P.epsilon / r * (2.0 * s6 * s6 - s6);
}

fn fene_dv(r: f32) -> f32 {
  let x = r / P.r_inf;
  return P.k_fene * r / (1.0 - x * x);
}

fn bend_dv(r: f32) -> f32 { return P.k_bend * (r - P.r_bend); }

fn attr_dv(r: f32) -> f32 {
  let rc = wca_cut(P.b_tt);
  if (r < rc || r > rc + P.wc) { return 0.0; }
  let x = 3.14159265 * (r - rc) / (2.0 * P.wc);
  return P.epsilon * 3.14159265 * sin(2.0 * x) / (2.0 * P.wc);
}

@compute @workgroup_size(64)
fn probe_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&radii)) { return; }
  let r = radii[i];
  let kind = u32(probe.x);
  var v = 0.0;
  if (kind == 0u) { v = wca_dv(r, probe.y); }
  else if (kind == 1u) { v = fene_dv(r); }
  else if (kind == 2u) { v = bend_dv(r); }
  else { v = attr_dv(r); }
  out[i] = v;
}
```

`engine/src/forces.ts`:

```ts
import forcesWgsl from '../wgsl/forces.wgsl?raw'
import { getGpu, readBack, storageBuffer } from './gpu'
import { loadParams, paramsToUniform } from './params'

const KINDS = { wca: 0, fene: 1, bend: 2, attr: 3 } as const

export async function probeForces(kind: keyof typeof KINDS, b: number, radii: number[]): Promise<number[]> {
  const { device } = await getGpu()
  const module = device.createShaderModule({ code: forcesWgsl })
  const uni = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(uni, 0, paramsToUniform(loadParams()))
  const probe = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(probe, 0, new Float32Array([KINDS[kind], b, 0, 0]))
  const rin = storageBuffer(device, new Float32Array(radii))
  const rout = storageBuffer(device, new Float32Array(radii.length))
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'probe_main' } })
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: rin } },
      { binding: 2, resource: { buffer: rout } },
      { binding: 3, resource: { buffer: probe } },
    ],
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bind)
  pass.dispatchWorkgroups(Math.ceil(radii.length / 64))
  pass.end()
  device.queue.submit([enc.finish()])
  return Array.from(await readBack(device, rout, radii.length * 4))
}
```

Add to `engine/src/index.ts`: `export { probeForces } from './forces'`.

The `../wgsl/forces.wgsl?raw` import works in Vite with no plugins and no config changes needed. Do not add `assetsInclude: ['**/*.wgsl']`: that turns the shader into an asset, and `?raw` starts returning a path instead of the text.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/forces.test.ts`
Expected: PASS all four. If WCA diverges at the fifth digit, that is expected for f32, and the `toBeCloseTo(..., 4)` tolerance accounts for it; a divergence at the first digit means an error in the formula, not a precision issue.

- [ ] **Step 5: Commit**

```bash
git add engine/wgsl/forces.wgsl engine/src/forces.ts engine/src/index.ts vite.config.ts tests/forces.test.ts
git commit -m "feat: force kernels verified against analytic derivatives"
```

---

### Task 4: Langevin integrator, neighbor grid, equipartition

**Files:**
- Create: `engine/wgsl/integrate.wgsl`, `engine/wgsl/neighbor.wgsl`, `engine/src/sim.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/sim.test.ts`

**Interfaces:**
- Consumes: `probeForces` is not needed; `getGpu`, `paramsToUniform`, `loadParams` are used.
- Produces: from `engine/src/sim.ts`:
  - `createSystem(opts: {lipids: number, box: [number, number, number], seed: number, layout: 'random' | 'bilayer' | 'vesicle', gamma?: number}): Promise<System>` - `gamma` overrides the friction from `params.json` (needed for verifying energy conservation with `gamma: 0`), `layout: 'vesicle'` is described in Task 8
  - `System.step(n: number): Promise<void>`
  - `System.positions(): Promise<Float32Array>` - 4 floats per bead (x, y, z, type; type 0 = head, 1 = tail)
  - `System.kineticEnergyPerDof(): Promise<number>`
  - `System.totalEnergy(): Promise<number>`
  - `System.forcesBruteForce(): Promise<Float32Array>` - the same forces without the neighbor grid, for cross-checking
  - `System.forces(): Promise<Float32Array>`

- [ ] **Step 1: Write failing tests**

`tests/sim.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadParams } from '../engine/src/params'

afterAll(shutdownGpu)
const p = loadParams()

test('the neighbor grid gives the same forces as brute force', async () => {
  const page = await gpuPage()
  const diff = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 600, box: [20, 20, 20], seed: 7, layout: 'random' })
    const a = await sys.forces()
    const b = await sys.forcesBruteForce()
    let max = 0
    for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]))
    return max
  })
  expect(diff).toBeLessThan(1e-4)
})

test('the thermostat brings the system to the target temperature', async () => {
  const page = await gpuPage()
  const kT = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1000, box: [24, 24, 24], seed: 11, layout: 'random' })
    await sys.step(20_000)
    return sys.kineticEnergyPerDof()
  })
  expect(kT).toBeGreaterThan(p.thermostat.kT * 0.95)
  expect(kT).toBeLessThan(p.thermostat.kT * 1.05)
})

test('with no friction, total energy drifts only slightly', async () => {
  const page = await gpuPage()
  const drift = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 200, box: [16, 16, 16], seed: 3, layout: 'bilayer', gamma: 0 })
    const e0 = await sys.totalEnergy()
    await sys.step(2000)
    const e1 = await sys.totalEnergy()
    return Math.abs((e1 - e0) / e0)
  })
  expect(drift).toBeLessThan(0.02)
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/sim.test.ts`
Expected: FAIL - `api.createSystem` is not a function.

- [ ] **Step 3: Implement the neighbor grid**

`engine/wgsl/neighbor.wgsl`:

```wgsl
struct Grid { dims: vec4<u32>, box: vec4<f32> };
@group(0) @binding(0) var<uniform> G: Grid;
@group(0) @binding(1) var<storage, read> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cells: array<u32>;

fn cell_of(p: vec3<f32>) -> u32 {
  let n = vec3<f32>(f32(G.dims.x), f32(G.dims.y), f32(G.dims.z));
  let f = floor((p / G.box.xyz) * n);
  let c = clamp(vec3<u32>(max(f, vec3<f32>(0.0))), vec3<u32>(0u), G.dims.xyz - vec3<u32>(1u));
  return c.x + G.dims.x * (c.y + G.dims.y * c.z);
}

@compute @workgroup_size(64)
fn count_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  atomicAdd(&counts[cell_of(pos[i].xyz)], 1u);
}

@compute @workgroup_size(64)
fn fill_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  let c = cell_of(pos[i].xyz);
  let slot = atomicAdd(&counts[c], 1u);
  cells[slot] = i;
}
```

Workflow: `count_main` counts occupancy, the CPU (or a separate kernel) takes the prefix sums into a cell-start buffer, `counts` is zeroed, and `fill_main` scatters the indices. Compute the prefix sum on the CPU from the counted buffer on this first pass; that is acceptable because the grid size is small (thousands of cells), and it is replaced with a GPU version only if a measurement shows it is a bottleneck. Record the measurement in `verify/out/gates.json` under the field `neighborBuildMs`.

- [ ] **Step 4: Implement the Langevin step**

`engine/wgsl/integrate.wgsl`:

```wgsl
struct Params {
  sigma: f32, epsilon: f32,
  b_hh: f32, b_ht: f32, b_tt: f32,
  k_fene: f32, r_inf: f32,
  k_bend: f32, r_bend: f32,
  wc: f32,
  kT: f32, gamma: f32,
  dt: f32,
  pad0: f32, pad1: f32, pad2: f32,
};
struct Step { seed: u32, index: u32, pad0: u32, pad1: u32 };

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> pos: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> vel: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> force: array<vec4<f32>>;
@group(0) @binding(4) var<uniform> S: Step;

fn pcg(v: u32) -> u32 {
  var state = v * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn unit_normal(seed: u32) -> f32 {
  let u1 = max(f32(pcg(seed)) / 4294967296.0, 1e-7);
  let u2 = f32(pcg(seed + 1u)) / 4294967296.0;
  return sqrt(-2.0 * log(u1)) * cos(6.28318531 * u2);
}

@compute @workgroup_size(64)
fn step_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= arrayLength(&pos)) { return; }
  let base = pcg(i * 2654435761u ^ (S.index * 40503u) ^ S.seed);
  let noise = vec3<f32>(unit_normal(base), unit_normal(base + 7u), unit_normal(base + 17u));
  let sigma_v = sqrt(2.0 * P.gamma * P.kT * P.dt);
  var v = vel[i].xyz + P.dt * force[i].xyz - P.gamma * P.dt * vel[i].xyz + sigma_v * noise;
  var x = pos[i].xyz + P.dt * v;
  vel[i] = vec4<f32>(v, 0.0);
  pos[i] = vec4<f32>(x, pos[i].w);
}
```

Periodic boundaries are applied in a separate short kernel after the step, so that force and coordinates do not diverge within the same pass: `x = x - floor(x / box) * box` componentwise for x and y; no walls are applied along z, and the box is taken large enough there.

`engine/src/sim.ts` assembles the three pipelines (grid, forces, integrator), holds the position, velocity, and force buffers, and implements the methods listed in Interfaces. `layout: 'bilayer'` arranges lipids into two layers on a grid in the xy plane with heads facing out; `layout: 'random'` scatters lipids randomly with a random orientation direction; both use `seed` for reproducibility.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `npx vitest run tests/sim.test.ts`
Expected: PASS all three. Diagnostics: if the temperature comes out above the target, check that the noise is scaled as `sqrt(2*Gamma*kT*dt)`, not `sqrt(2*Gamma*kT/dt)`; if the forces do not match brute force, check that the grid cell radius is not smaller than the maximum interaction radius `r_c + w_c`.

- [ ] **Step 6: Commit**

```bash
git add engine/wgsl/neighbor.wgsl engine/wgsl/integrate.wgsl engine/src/sim.ts engine/src/index.ts tests/sim.test.ts
git commit -m "feat: Langevin integrator and neighbor grid with equipartition test"
```

---

### Task 5: Zero tension, area per lipid, bilayer thickness

**Files:**
- Create: `engine/src/metrics.ts`
- Modify: `engine/src/sim.ts` (add the MC area move), `engine/src/index.ts`
- Test: `tests/metrics.test.ts`

**Interfaces:**
- Consumes: `System` from Task 4.
- Produces:
  - `System.areaMove(trials: number): Promise<number>` - a Monte Carlo box-area move: `L_x` and `L_y` are multiplied by `sqrt(s)`, `L_z` is unchanged; returns the fraction of accepted moves.
  - `areaPerLipid(box: [number, number, number], lipids: number): number`
  - `densityProfileZ(positions: Float32Array, box: [number, number, number], bins: number): {z: number[], head: number[], tail: number[]}`
  - `bilayerThickness(profile: ReturnType<typeof densityProfileZ>): number` - the distance between the head-density maxima.
  - `measureBilayer(sys: System): Promise<{areaPerLipid: number, thickness: number, box: [number, number, number], lipids: number, steps: number}>` - a facade in `engine/src/index.ts`.

- [ ] **Step 1: Write failing tests for the pure math**

`tests/metrics.test.ts`:

```ts
import { expect, test } from 'vitest'
import { areaPerLipid, bilayerThickness, densityProfileZ } from '../engine/src/metrics'

test('area per lipid is computed from half the number of lipids per layer', () => {
  expect(areaPerLipid([20, 20, 30], 1000)).toBeCloseTo(0.8, 12)
})

test('thickness is taken as the distance between the head-density peaks', () => {
  const box: [number, number, number] = [10, 10, 20]
  const pos: number[] = []
  for (let i = 0; i < 200; i++) {
    const z = i < 100 ? 7.5 : 12.5
    pos.push(Math.random() * 10, Math.random() * 10, z, 0)
  }
  const profile = densityProfileZ(new Float32Array(pos), box, 200)
  expect(bilayerThickness(profile)).toBeCloseTo(5, 1)
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/metrics.test.ts`
Expected: FAIL - `engine/src/metrics` not found.

- [ ] **Step 3: Implement the metrics**

`engine/src/metrics.ts`:

```ts
export function areaPerLipid(box: [number, number, number], lipids: number): number {
  return (box[0] * box[1]) / (lipids / 2)
}

export interface ZProfile { z: number[]; head: number[]; tail: number[] }

export function densityProfileZ(
  positions: Float32Array,
  box: [number, number, number],
  bins: number,
): ZProfile {
  const dz = box[2] / bins
  const head = new Array(bins).fill(0)
  const tail = new Array(bins).fill(0)
  for (let i = 0; i < positions.length; i += 4) {
    const z = positions[i + 2]
    const b = Math.min(bins - 1, Math.max(0, Math.floor(z / dz)))
    if (positions[i + 3] === 0) head[b] += 1
    else tail[b] += 1
  }
  const z = Array.from({ length: bins }, (_, i) => (i + 0.5) * dz)
  return { z, head, tail }
}

export function bilayerThickness(profile: ZProfile): number {
  const mid = Math.floor(profile.head.length / 2)
  const argmax = (from: number, to: number) => {
    let best = from
    for (let i = from; i < to; i++) if (profile.head[i] > profile.head[best]) best = i
    return best
  }
  const lower = argmax(0, mid)
  const upper = argmax(mid, profile.head.length)
  return profile.z[upper] - profile.z[lower]
}
```

MC area move in `sim.ts`: propose a multiplier `s = exp(u)`, `u ~ U(-0.005, 0.005)`, and scale `L_x, L_y` by `s` and `1/s`? No: at zero tension it is the area itself that changes, so scale both `L_x` and `L_y` by `sqrt(s)`, and scale the particles' x and y coordinates by `sqrt(s)` as well; leave `L_z` unchanged. Accept with probability `min(1, exp(-DeltaU/kT))`, where `DeltaU` is the difference in total potential energy before and after (surface tension gamma = 0, so the `gamma*DeltaA` term is absent). Call the move once every 100 integration steps.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/metrics.test.ts`
Expected: PASS both.

- [ ] **Step 5: Write a failing test for the bilayer gate 6**

`tests/gate6-bilayer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('a formed bilayer at zero tension holds the area and thickness from the literature', async () => {
  const page = await gpuPage()
  const m = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1000, box: [26, 26, 40], seed: 5, layout: 'bilayer' })
    for (let i = 0; i < 200; i++) { await sys.step(100); await sys.areaMove(1) }
    return api.measureBilayer(sys)
  })
  expect(m.areaPerLipid).toBeGreaterThan(1.1)
  expect(m.areaPerLipid).toBeLessThan(1.5)
  expect(m.thickness).toBeGreaterThan(4.0)
  expect(m.thickness).toBeLessThan(6.0)
}, 600_000)
```

- [ ] **Step 6: Implement `measureBilayer` and make it pass**

`measureBilayer(sys)` in `engine/src/index.ts`: read the positions, compute `areaPerLipid` from the current box, build a profile over 200 bins, return `{areaPerLipid, thickness, box, lipids, steps}`.

Run: `npx vitest run tests/gate6-bilayer.test.ts`
Expected: PASS. Diagnostics on failure: an area noticeably above the target range means the MC move is not being accepted (check the acceptance fraction; it should be 0.2-0.6; if it is 0, there is a sign error in `DeltaU`); an area below the target range at `kT/epsilon = 1.1` means a gel phase - check that `w_c = 1.6sigma` actually reached the shader.

- [ ] **Step 7: Commit**

```bash
git add engine/src/metrics.ts engine/src/sim.ts engine/src/index.ts tests/metrics.test.ts tests/gate6-bilayer.test.ts
git commit -m "feat: zero-tension area move, area per lipid and thickness metrics"
```

---

### Task 6: Self-assembly from a random solution

**Files:**
- Create: `engine/src/aggregate.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/self-assembly.test.ts`

**Interfaces:**
- Consumes: `System`, `densityProfileZ`.
- Produces: `clusters(positions: Float32Array, box: [number, number, number], cutoff: number): number[]` - cluster sizes by tail connectivity; `largestClusterFraction(positions, box, cutoff): number`; a facade `largestClusterFractionOf(sys: System): Promise<number>` in `engine/src/index.ts` that reads positions and calls `largestClusterFraction` with `cutoff = r_c + w_c`.

- [ ] **Step 1: Write a failing clustering test on synthetic data**

`tests/self-assembly.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { clusters, largestClusterFraction } from '../engine/src/aggregate'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('clustering distinguishes two distant groups', () => {
  const pos: number[] = []
  for (let i = 0; i < 10; i++) pos.push(1 + i * 0.2, 1, 1, 1)
  for (let i = 0; i < 6; i++) pos.push(20 + i * 0.2, 20, 20, 1)
  const sizes = clusters(new Float32Array(pos), [40, 40, 40], 0.5).sort((a, b) => b - a)
  expect(sizes).toEqual([10, 6])
})

test('a single large aggregate grows from a random solution', async () => {
  const page = await gpuPage()
  const frac = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1200, box: [28, 28, 28], seed: 13, layout: 'random' })
    await sys.step(400_000)
    return api.largestClusterFractionOf(sys)
  })
  expect(frac).toBeGreaterThan(0.8)
}, 900_000)
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/self-assembly.test.ts`
Expected: FAIL - `engine/src/aggregate` not found.

- [ ] **Step 3: Implement clustering**

`engine/src/aggregate.ts`: merge by tail-bead distance accounting for periodicity along x and y, using union-find with a cell grid of size `cutoff`, returning component sizes; `largestClusterFraction` divides the largest component by the number of tail beads.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/self-assembly.test.ts`
Expected: PASS both. If the largest-cluster fraction gets stuck around 0.2, that is a micelle stage; increase the number of steps and record the measured time to convergence in `gates.json` under the field `assemblySteps`, rather than "tuning up" the attraction.

- [ ] **Step 5: Commit**

```bash
git add engine/src/aggregate.ts engine/src/index.ts tests/self-assembly.test.ts
git commit -m "feat: cluster analysis and self-assembly from random solution"
```

---

### Task 7: Undulation spectrum and bending modulus

**Files:**
- Create: `engine/src/spectrum.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/spectrum.test.ts`

**Interfaces:**
- Consumes: positions from `System`.
- Produces:
  - `heightField(positions: Float32Array, box: [number, number, number], n: number): Float32Array` - the midsurface on an `n x n` grid.
  - `spectrum(h: Float32Array, n: number, box: [number, number, number]): {q: number[], hq2: number[]}`
  - `fitBendingModulus(s: {q: number[], hq2: number[]}, kT: number, area: number, qMax: number): number`
  - `synthesizeHeightField(n: number, box: [number, number, number], kappa: number, kT: number, seed: number): Float32Array` - a generator for a field with known kappa, for a round-trip check.
  - a facade `measureBendingModulus(sys: System, opts: {grid: number, modes: number, samples: number}): Promise<number>` in `engine/src/index.ts`.

- [ ] **Step 1: Write a failing round-trip test**

`tests/spectrum.test.ts`:

```ts
import { expect, test } from 'vitest'
import { fitBendingModulus, spectrum, synthesizeHeightField } from '../engine/src/spectrum'

test('the kappa estimate recovers the value built into the synthetic field', () => {
  const n = 64
  const box: [number, number, number] = [40, 40, 40]
  const kT = 1.1
  for (const kappaTrue of [8, 20, 40]) {
    const h = synthesizeHeightField(n, box, kappaTrue, kT, 42)
    const s = spectrum(h, n, box)
    const kappa = fitBendingModulus(s, kT, box[0] * box[1], (2 * Math.PI * 8) / box[0])
    expect(kappa).toBeGreaterThan(kappaTrue * 0.85)
    expect(kappa).toBeLessThan(kappaTrue * 1.15)
  }
})
```

- [ ] **Step 2: Run and confirm the test fails**

Run: `npx vitest run tests/spectrum.test.ts`
Expected: FAIL - `engine/src/spectrum` not found.

- [ ] **Step 3: Implement the spectrum**

The normalization convention is fixed once and used both in the generator and in the estimator: `h_q = (1/N)*Sum_r h(r)*exp(-i q.r)`, where `N = n^2`, and `<|h_q|^2> = kT/(A*kappa*q^4)`. The `synthesizeHeightField` generator fills the amplitudes `|h_q| = sqrt(kT/(A*kappa*q^4))` with random phases and inverse-transforms; `fitBendingModulus` takes a linear regression of `log<|h_q|^2>` against `log q` over the region `q <= qMax`, checks that the slope is close to -4 (otherwise throws an error with the measured slope), and extracts kappa from the intercept.

The discrete transform is done with a direct double loop over modes with `q <= qMax`: a 64x64 grid and a dozen modes take a fraction of a second, so a fast transform is not needed.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `npx vitest run tests/spectrum.test.ts`
Expected: PASS for all three kappa values. If the recovered kappa is systematically off by a factor of two to four, that is a normalization error, not a physics error: check the `1/N` factor on both sides.

- [ ] **Step 5: Write the gate-6 bending-modulus test**

`tests/gate6-kappa.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('the bilayer bending modulus falls within the measured 5-50 kT range', async () => {
  const page = await gpuPage()
  const kappa = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 4000, box: [52, 52, 40], seed: 17, layout: 'bilayer' })
    for (let i = 0; i < 300; i++) { await sys.step(200); await sys.areaMove(1) }
    return api.measureBendingModulus(sys, { grid: 32, modes: 8, samples: 200 })
  })
  expect(kappa).toBeGreaterThan(5)
  expect(kappa).toBeLessThan(50)
}, 1_800_000)
```

- [ ] **Step 6: Implement `measureBendingModulus` and make it pass**

The function averages `|h_q|^2` over `samples` configurations, sampled at even intervals of steps, then calls `fitBendingModulus`.

Run: `npx vitest run tests/gate6-kappa.test.ts`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add engine/src/spectrum.ts engine/src/index.ts tests/spectrum.test.ts tests/gate6-kappa.test.ts
git commit -m "feat: undulation spectrum and bending modulus measurement"
```

---

### Task 8: Closed-vesicle detection by flood fill

**Files:**
- Create: `engine/src/closure.ts`, `engine/wgsl/closure.wgsl`
- Modify: `engine/src/index.ts`
- Test: `tests/closure.test.ts`

**Interfaces:**
- Consumes: positions from `System`.
- Produces:
  - `occupancy(positions: Float32Array, box: [number, number, number], cell: number, radius: number): Uint8Array`
  - `enclosedVolume(occ: Uint8Array, dims: [number, number, number], cell: number): number` - a TypeScript reference: flood fill from the boundary, the volume of un-flooded empty cells.
  - facades in `engine/src/index.ts`: `enclosedVolumeCpu(sys: System, opts: {cell: number, radius: number}): Promise<number>` - reads positions and computes with the reference implementation; `enclosedVolumeGpu(sys: System, opts: {cell: number, radius: number}): Promise<number>` - computes with the `closure.wgsl` kernel.

- [ ] **Step 1: Write failing tests on synthetic data**

`tests/closure.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { enclosedVolume, occupancy } from '../engine/src/closure'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

function shell(radius: number, thickness: number, count: number): Float32Array {
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    const u = Math.random() * 2 - 1
    const phi = Math.random() * 2 * Math.PI
    const r = radius + (Math.random() - 0.5) * thickness
    const s = Math.sqrt(1 - u * u)
    out.push(20 + r * s * Math.cos(phi), 20 + r * s * Math.sin(phi), 20 + r * u, 1)
  }
  return new Float32Array(out)
}

test('a closed shell yields a cavity close to the volume of a sphere', () => {
  const box: [number, number, number] = [40, 40, 40]
  const occ = occupancy(shell(8, 1.5, 40_000), box, 0.5, 0.6)
  const v = enclosedVolume(occ, [80, 80, 80], 0.5)
  const ideal = (4 / 3) * Math.PI * 7.5 ** 3
  expect(v).toBeGreaterThan(ideal * 0.7)
  expect(v).toBeLessThan(ideal * 1.3)
})

test('a flat sheet yields no cavity', () => {
  const box: [number, number, number] = [40, 40, 40]
  const pos: number[] = []
  for (let i = 0; i < 20_000; i++) pos.push(Math.random() * 40, Math.random() * 40, 20 + (Math.random() - 0.5), 1)
  const occ = occupancy(new Float32Array(pos), box, 0.5, 0.6)
  expect(enclosedVolume(occ, [80, 80, 80], 0.5)).toBe(0)
})

test('the GPU version matches the TypeScript reference', async () => {
  const page = await gpuPage()
  const rel = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 2000, box: [40, 40, 40], seed: 23, layout: 'vesicle' })
    const cpu = await api.enclosedVolumeCpu(sys, { cell: 0.5, radius: 0.6 })
    const gpu = await api.enclosedVolumeGpu(sys, { cell: 0.5, radius: 0.6 })
    return Math.abs(gpu - cpu) / Math.max(cpu, 1)
  })
  expect(rel).toBeLessThan(0.01)
})
```

Task 4 gets an addition: `layout` accepts a third value, `'vesicle'` - lipids are arranged in two concentric layers on a sphere of a radius chosen to fit the number of lipids at an area of 1.2 sigma^2 per lipid. This is only a starting configuration for the detector tests; the actual vesicle arises from self-assembly in Task 6.

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/closure.test.ts`
Expected: FAIL - `engine/src/closure` not found.

- [ ] **Step 3: Implement the reference and the GPU version**

`closure.ts`: the occupancy grid marks cells whose centers are within `radius` of any bead (traversing the cells around each bead, not a full brute-force scan); flood fill is a breadth-first search starting from all empty boundary cells; the volume is computed from the unvisited empty cells.

`closure.wgsl`: occupancy in a single pass over the beads using `atomicOr`; flood fill is iterative propagation of the "outside" label to neighbors with a change flag in a buffer, repeating until there is no change (cap the number of iterations by the grid's diagonal length and record the measurement in `gates.json`).

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/closure.test.ts`
Expected: PASS all three.

- [ ] **Step 5: Commit**

```bash
git add engine/src/closure.ts engine/wgsl/closure.wgsl engine/src/sim.ts engine/src/index.ts tests/closure.test.ts
git commit -m "feat: closed-vesicle detection by outside flood fill"
```

---

### Task 9: Literature targets, gate verdicts, report

**Files:**
- Create: `data/literature.json`, `verify/gates.ts`, `verify/report.ts`, `verify/run.ts`
- Test: `tests/gates.test.ts`

**Interfaces:**
- Consumes: metrics from Tasks 5-8.
- Produces:
  - `evaluateGates(metrics: Record<string, number>): GateResult[]`, where `GateResult = {id: string, title: string, value: number | null, target: {min?: number, max?: number}, unit: string, rank: 'A'|'B'|'C'|'D', verdict: 'passed'|'failed'|'unproven', source: string}`
  - `renderReport(results: GateResult[], meta: Record<string, unknown>): string`

- [ ] **Step 1: Write failing tests**

`tests/gates.test.ts`:

```ts
import { expect, test } from 'vitest'
import { evaluateGates } from '../verify/gates'
import { renderReport } from '../verify/report'

test('a metric within the target range passes, outside it fails', () => {
  const ok = evaluateGates({ areaPerLipid: 1.2, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(ok.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('passed')
  const bad = evaluateGates({ areaPerLipid: 0.7, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(bad.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('failed')
})

test('a missing metric gives "unproven", not "passed"', () => {
  const r = evaluateGates({ areaPerLipid: 1.2 })
  expect(r.find((g) => g.id === 'bending-modulus')!.verdict).toBe('unproven')
  expect(r.find((g) => g.id === 'bending-modulus')!.value).toBeNull()
})

test('rank-D gates never come out passed', () => {
  const r = evaluateGates({ chainToBeadMapping: 1 })
  expect(r.find((g) => g.id === 'chain-to-bead-mapping')!.verdict).toBe('unproven')
})

test('the report contains the verdict, the value, the target range, and the reference', () => {
  const html = renderReport(evaluateGates({ areaPerLipid: 1.2 }), { commit: 'abc123' })
  expect(html).toContain('1.1')
  expect(html).toContain('cond-mat/0509218')
  expect(html).toContain('unproven')
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/gates.test.ts`
Expected: FAIL - `verify/gates` not found.

- [ ] **Step 3: Implement the targets and verdicts**

`data/literature.json`:

```json
{
  "gates": [
    {
      "id": "area-per-lipid",
      "title": "Area per lipid",
      "metric": "areaPerLipid",
      "unit": "sigma^2",
      "target": { "min": 1.1, "max": 1.5 },
      "rank": "A",
      "source": "Cooke & Deserno 2005, arXiv:cond-mat/0509218",
      "conditions": "kT/eps = 1.1, w_c = 1.6 sigma, zero tension"
    },
    {
      "id": "bilayer-thickness",
      "title": "Bilayer thickness",
      "metric": "thickness",
      "unit": "sigma",
      "target": { "min": 4.0, "max": 6.0 },
      "rank": "A",
      "source": "Cooke & Deserno 2005, arXiv:cond-mat/0509218",
      "conditions": "from the head-bead density maxima"
    },
    {
      "id": "bending-modulus",
      "title": "Bending modulus",
      "metric": "bendingModulus",
      "unit": "kT",
      "target": { "min": 5, "max": 50 },
      "rank": "A",
      "source": "Cooke & Deserno 2005, arXiv:cond-mat/0509218",
      "conditions": "from the undulation spectrum at q <= 8 modes"
    },
    {
      "id": "closure",
      "title": "Closed cavity",
      "metric": "enclosedVolume",
      "unit": "sigma^3",
      "target": { "min": 1 },
      "rank": "A",
      "source": "outside flood-fill detector, spec section 7",
      "conditions": "0.5 sigma grid, 0.6 sigma bead radius"
    },
    {
      "id": "chain-to-bead-mapping",
      "title": "Correspondence of carbon-atom count to bead count",
      "metric": "chainToBeadMapping",
      "unit": "-",
      "target": {},
      "rank": "D",
      "source": "our assumption, spec section 7",
      "conditions": "accepted only via the area and thickness gates"
    }
  ]
}
```

`verify/gates.ts` reads this file and, for each entry, looks up the metric by name: no metric means `unproven` and `value: null`; rank `D` is always `unproven`; otherwise it is compared against the target range. `verify/report.ts` renders a static HTML page: a table with columns "gate", "value", "target range", "rank", "verdict", "source", "conditions", plus a performance-measurements block and the commit. `verify/run.ts` starts Vite, launches Chrome, runs the measurement scenarios from Tasks 5-8, and writes `verify/out/gates.json` and `verify/out/report.html`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/gates.test.ts`
Expected: PASS all four.

- [ ] **Step 5: Run the full exam**

Run: `npm run verify`
Expected: `verify/out/report.html` is created, showing four rank-A gates with numbers and one rank-D gate marked "unproven". Give the user the full path to the file and do not offer your own quality judgment of the picture.

- [ ] **Step 6: Commit**

```bash
git add data/literature.json verify/gates.ts verify/report.ts verify/run.ts tests/gates.test.ts
git commit -m "feat: literature targets, gate verdicts and verification report"
```

---

### Task 10: Viewer

**Files:**
- Create: `viewer/index.html`, `viewer/main.ts`, `viewer/panel.ts`
- Test: `tests/viewer.test.ts`

**Interfaces:**
- Consumes: `createSystem`, `System.step`, `System.positions`, `loadParams`.
- Produces: the `/viewer/index.html` page, publishing `window.viewer = {frames: number, kT: number, wc: number, setKT(v: number): void, setWc(v: number): void, timeScaleBadge: string}`.

- [ ] **Step 1: Write a failing test**

`tests/viewer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('the viewer draws frames and shows the time-scale badge', async () => {
  const page = await gpuPage()
  await page.goto(new URL('viewer/index.html', page.url()).href, { waitUntil: 'load' })
  await page.waitForFunction('window.viewer && window.viewer.frames > 5')
  const state = await page.evaluate(() => ({
    frames: (window as any).viewer.frames,
    badge: (window as any).viewer.timeScaleBadge,
    kT: (window as any).viewer.kT,
  }))
  expect(state.frames).toBeGreaterThan(5)
  expect(state.badge).toContain('tau')
  expect(state.kT).toBe(1.1)

  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)
})
```

- [ ] **Step 2: Run and confirm the test fails**

Run: `npx vitest run tests/viewer.test.ts`
Expected: FAIL - the `viewer/index.html` page is missing.

- [ ] **Step 3: Implement the viewer**

`viewer/main.ts`: three.js's `WebGPURenderer` is not needed; the scene is drawn with an ordinary `WebGLRenderer` from positions read off the simulation, via an `InstancedMesh` of spheres (heads and tails in different materials), `OrbitControls`, and a frame counter in `window.viewer.frames`. The simulation advances 20 steps per frame. The time-scale badge shows "step dt = 0.01tau, N tau shown" and the caption "beads, not atoms," per the spec's requirement not to confuse a bead with an atom.

`viewer/panel.ts`: two sliders - `kT/epsilon` within the range given in `params.json` under the field `kTRange`, and `w_c` from 1.0 to 2.0 sigma; changing either overwrites the uniform buffer without recreating the system.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `npx vitest run tests/viewer.test.ts`
Expected: PASS.

- [ ] **Step 5: Measure performance and record it in the report**

Add a measurement to `verify/run.ts`: a system of 4000 lipids (12,000 beads), 1000 steps; record steps per second in `gates.json` under the field `stepsPerSecond` and the bead count under the field `beads`. No claims about speed in the text, only the number in the report.

Run: `npm run verify`
Expected: `verify/out/gates.json` contains `stepsPerSecond`, `beads`, `neighborBuildMs`.

- [ ] **Step 6: Commit**

```bash
git add viewer/index.html viewer/main.ts viewer/panel.ts verify/run.ts tests/viewer.test.ts
git commit -m "feat: interactive viewer with parameter panel and time-scale badge"
```

---

## What this plan deliberately does not do

Stages A and B (chemistry and the aggregation gates), the `thermo/`, `qm/`, `ref/` layers, gates 1-5, and the phase map are the subject of separate plans. This plan ends with a working stage C with gate 6 and an automated report, on which everything else is later verified.
