# L0–L2: движок мембраны, метрики и автоматический экзамен — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Довести до работающего состояния ступень C симуляции — самосборку липидного бислоя и замкнутой везикулы в WebGPU — вместе с метриками, которые сверяются с опубликованными величинами, и отчётом, который выносит вердикт по воротам 6.

**Architecture:** Физика живёт в WGSL и исполняется одним и тем же кодом в интерактивном просмотре и в проверке. Числовые константы модели не вписаны в код, а читаются из `data/params.json`. Проверка запускается через системный Chrome в headless-режиме под управлением puppeteer-core; метрики читаются вызовом функции на странице, а не из дампа DOM. Чистая математика (спектр ундуляций, заливка полостей, вердикты ворот) реализована на TypeScript и тестируется без GPU, а её GPU-версии сверяются с этой эталонной реализацией.

**Tech Stack:** TypeScript, Vite, Vitest, WGSL (WebGPU), three.js, puppeteer-core с системным Chrome. Node 22 (в системе). Без UI-фреймворка.

**Spec:** `docs/superpowers/specs/2026-08-15-protocell-genesis-design.md`

## Global Constraints

- В `engine/` не должно быть ни одной вписанной числовой константы модели: всё читается из `data/params.json`. Задача 2 добавляет тест-сторож, который это проверяет сканированием файлов.
- Параметры модели Cooke & Deserno 2005 используются точно в опубликованных значениях: `b_hh = b_ht = 0.95σ`, `b_tt = σ`, WCA с обрезкой `r_c = 2^(1/6)·b`, FENE `k_bond = 30ε/σ²` и `r_∞ = 1.5σ`, изгиб `k_bend = 10ε/σ²` вокруг `4σ`, притяжение хвостов `−ε cos²[π(r−r_c)/(2w_c)]`, термостат Ланжевена `Γ = 1/τ`, шаг `δt = 0.01τ`, рабочая точка `kT/ε = 1.1` и `w_c = 1.6σ`.
- Единицы приведённые: `σ = 1`, `ε = 1`, `m = 1`, `τ = σ√(m/ε) = 1`.
- Литературные цели ворот 6: площадь на липид **1.1–1.5 σ²**, толщина бислоя **≈5σ**, модуль изгиба **5–50 kT**.
- Проверка обязана читать результат вызовом на странице (CDP `page.evaluate`). Использовать `--dump-dom` запрещено: замерено, что он отдаёт содержимое до завершения асинхронной работы GPU и живой GPU выглядит отсутствующим.
- Chrome запускается с флагом `--enable-unsafe-webgpu`; путь к системному Chrome — `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`. Замерено: адаптер `apple/metal-3`, compute-проход исполняется.
- Каждое утверждение о характеристиках сопровождается замером, попадающим в `verify/out/gates.json`. Формулировки «работает быстро», «выглядит правильно» в отчёт не попадают.
- Редукции на GPU зависят от порядка суммирования, поэтому все сравнения GPU и CPU идут с допуском, а размеры групп фиксированы.

---

## Структура файлов

Создаётся в этом плане:

- `package.json`, `tsconfig.json`, `vite.config.ts` — сборка и тесты.
- `data/params.json` — параметры модели, единственный источник констант.
- `data/literature.json` — цели ворот со ссылками и условиями.
- `engine/src/gpu.ts` — получение устройства WebGPU, загрузка шейдеров, вспомогательные буферы.
- `engine/src/params.ts` — чтение и проверка `data/params.json`, производные величины.
- `engine/src/forces.ts` — сборка pipeline силовых ядер, точка входа для тестов-зондов.
- `engine/src/sim.ts` — состояние системы, шаг интегрирования, MC-ход по площади бокса.
- `engine/src/metrics.ts` — площадь на липид, профиль плотности, толщина.
- `engine/src/spectrum.ts` — высотное поле, дискретное преобразование, оценка κ.
- `engine/src/closure.ts` — сетка занятости, заливка снаружи, объём полости (эталон на TypeScript).
- `engine/src/index.ts` — публичный фасад: всё, что вызывается из тестов и из просмотрщика.
- `engine/wgsl/forces.wgsl` — потенциалы и их производные.
- `engine/wgsl/neighbor.wgsl` — сетка соседей.
- `engine/wgsl/integrate.wgsl` — шаг Ланжевена.
- `engine/wgsl/closure.wgsl` — заливка на GPU.
- `verify/gates.ts` — вычисление вердиктов ворот из метрик и `literature.json`.
- `verify/report.ts` — генерация `verify/out/report.html`.
- `verify/run.ts` — прогон полного экзамена одной командой.
- `viewer/index.html`, `viewer/main.ts` — просмотрщик на three.js, панель, бейдж масштаба времени.
- `tests/helpers/gpu.ts` — запуск Vite и Chrome, страница-раннер.
- `tests/runner.html` — страница, публикующая фасад движка в `window.api`.
- `tests/*.test.ts` — тесты по задачам.

---

### Task 1: Каркас проекта и мост к GPU

**Files:**
- Create: `package.json`, `tsconfig.json`, `vite.config.ts`
- Create: `engine/src/gpu.ts`, `engine/src/index.ts`
- Create: `tests/runner.html`, `tests/helpers/gpu.ts`
- Test: `tests/gpu-smoke.test.ts`

**Interfaces:**
- Consumes: ничего.
- Produces: `gpuSmoke(): Promise<{vendor: string, architecture: string, doubled: number[]}>` из `engine/src/index.ts`; `gpuPage(): Promise<Page>` и `shutdownGpu(): Promise<void>` из `tests/helpers/gpu.ts`.

- [ ] **Step 1: Завести зависимости и конфигурацию**

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

Импортировать `defineConfig` именно из `vitest/config`, а не из `vite`: у версии из `vite` нет поля `test`, и настройки тестов будут молча проигнорированы вместе с таймаутами.

Run: `npm install`

- [ ] **Step 2: Написать падающий тест моста к GPU**

`tests/gpu-smoke.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('headless Chrome отдаёт адаптер WebGPU и исполняет compute-проход', async () => {
  const page = await gpuPage()
  const info = await page.evaluate(() => (window as any).api.gpuSmoke())
  expect(info.vendor).toBe('apple')
  expect(info.doubled).toEqual([2, 4, 6, 8])
})
```

- [ ] **Step 3: Запустить тест и убедиться, что он падает**

Run: `npx vitest run tests/gpu-smoke.test.ts`
Expected: FAIL — модуль `./helpers/gpu` не найден.

- [ ] **Step 4: Реализовать страницу-раннер и хелпер запуска**

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

- [ ] **Step 5: Реализовать `gpuSmoke`**

`engine/src/gpu.ts`:

```ts
export interface Gpu {
  device: GPUDevice
  adapterInfo: { vendor: string; architecture: string }
}

export async function getGpu(): Promise<Gpu> {
  if (!navigator.gpu) throw new Error('navigator.gpu отсутствует')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('адаптер WebGPU не выдан')
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

- [ ] **Step 6: Запустить тест и убедиться, что он проходит**

Run: `npx vitest run tests/gpu-smoke.test.ts`
Expected: PASS. Если падает на `adapter не выдан` — проверить, что флаг `--enable-unsafe-webgpu` дошёл до Chrome, и не пытаться читать результат через `--dump-dom`.

- [ ] **Step 7: Коммит**

```bash
git add package.json tsconfig.json vite.config.ts engine/src/gpu.ts engine/src/index.ts tests/runner.html tests/helpers/gpu.ts tests/gpu-smoke.test.ts package-lock.json
git commit -m "feat: WebGPU bridge and headless test harness"
```

---

### Task 2: Параметры модели как единственный источник констант

**Files:**
- Create: `data/params.json`, `engine/src/params.ts`
- Test: `tests/params.test.ts`

**Interfaces:**
- Consumes: ничего.
- Produces: `loadParams(): Params`, тип `Params` с полями `sigma, epsilon, beadSizes.{head_head,head_tail,tail_tail}, fene.{k,rInf}, bend.{k,r0}, attraction.wc, thermostat.{gamma,kT}, integrator.dt`, и `wcaCutoff(b: number): number`.

- [ ] **Step 1: Написать падающие тесты**

`tests/params.test.ts`:

```ts
import { expect, test } from 'vitest'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { loadParams, wcaCutoff } from '../engine/src/params'

test('параметры совпадают с опубликованными значениями Cooke & Deserno 2005', () => {
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

test('обрезка WCA равна 2^(1/6)·b', () => {
  expect(wcaCutoff(1)).toBeCloseTo(1.1224620483, 9)
  expect(wcaCutoff(0.95)).toBeCloseTo(0.95 * 2 ** (1 / 6), 9)
})

test('в движке нет вписанных констант модели', () => {
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

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/params.test.ts`
Expected: FAIL — `engine/src/params` не найден.

- [ ] **Step 3: Реализовать данные и загрузчик**

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
      throw new Error(`data/params.json: отсутствует поле ${key}`)
    }
  }
  return p
}

export function wcaCutoff(b: number): number {
  return Math.pow(2, 1 / 6) * b
}

/** Плоский массив для униформ-буфера WGSL. Порядок обязан совпадать со struct Params в forces.wgsl. */
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

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/params.test.ts`
Expected: PASS все три.

- [ ] **Step 5: Коммит**

```bash
git add data/params.json engine/src/params.ts tests/params.test.ts
git commit -m "feat: model parameters as the single source of constants"
```

---

### Task 3: Силовые ядра и сверка с аналитикой

**Files:**
- Create: `engine/wgsl/forces.wgsl`, `engine/src/forces.ts`
- Modify: `engine/src/index.ts` (экспортировать `probeForces`)
- Test: `tests/forces.test.ts`

**Interfaces:**
- Consumes: `loadParams`, `paramsToUniform`, `getGpu`, `storageBuffer`, `readBack`.
- Produces: `probeForces(kind: 'wca' | 'fene' | 'bend' | 'attr', b: number, radii: number[]): Promise<number[]>` — возвращает `dV/dr` в заданных точках. Знак: положительное значение — отталкивание.

- [ ] **Step 1: Написать падающие тесты**

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

test('WCA совпадает с аналитической производной и обрезается', async () => {
  const page = await gpuPage()
  const radii = [0.8, 0.9, 1.0, 1.05, 1.2]
  const got = await page.evaluate((r) => (window as any).api.probeForces('wca', 1.0, r), radii)
  radii.forEach((r, i) => expect(got[i]).toBeCloseTo(wcaAnalytic(r, 1.0), 4))
  expect(got[4]).toBe(0)
})

test('FENE тянет к центру и растёт у предела растяжения', async () => {
  const page = await gpuPage()
  const radii = [0.0, 0.5, 1.0, 1.4]
  const got = await page.evaluate((r) => (window as any).api.probeForces('fene', 1.0, r), radii)
  radii.forEach((r, i) => {
    const want = (p.fene.k * r) / (1 - (r / p.fene.rInf) ** 2)
    expect(got[i]).toBeCloseTo(want, 4)
  })
  expect(got[3]).toBeGreaterThan(got[2])
})

test('изгибный потенциал линеен вокруг r0', async () => {
  const page = await gpuPage()
  const radii = [3.0, 4.0, 5.0]
  const got = await page.evaluate((r) => (window as any).api.probeForces('bend', 1.0, r), radii)
  expect(got[0]).toBeCloseTo(p.bend.k * (3 - p.bend.r0), 5)
  expect(got[1]).toBeCloseTo(0, 6)
  expect(got[2]).toBeCloseTo(p.bend.k * (5 - p.bend.r0), 5)
})

test('притяжение хвостов гладко сходит к нулю на обоих концах', async () => {
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

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/forces.test.ts`
Expected: FAIL — `api.probeForces` не функция.

- [ ] **Step 3: Реализовать ядра**

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

Добавить в `engine/src/index.ts`: `export { probeForces } from './forces'`.

Импорт `../wgsl/forces.wgsl?raw` работает в Vite без плагинов и без правки конфигурации. Не добавлять `assetsInclude: ['**/*.wgsl']`: это превратит шейдер в ассет, и `?raw` начнёт отдавать путь вместо текста.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/forces.test.ts`
Expected: PASS все четыре. Если WCA расходится на пятом знаке — это ожидаемо для f32, допуск `toBeCloseTo(..., 4)` учитывает это; расхождение в первом знаке означает ошибку в формуле, а не точность.

- [ ] **Step 5: Коммит**

```bash
git add engine/wgsl/forces.wgsl engine/src/forces.ts engine/src/index.ts vite.config.ts tests/forces.test.ts
git commit -m "feat: force kernels verified against analytic derivatives"
```

---

### Task 4: Интегратор Ланжевена, сетка соседей, равнораспределение

**Files:**
- Create: `engine/wgsl/integrate.wgsl`, `engine/wgsl/neighbor.wgsl`, `engine/src/sim.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/sim.test.ts`

**Interfaces:**
- Consumes: `probeForces` не нужен; используются `getGpu`, `paramsToUniform`, `loadParams`.
- Produces: из `engine/src/sim.ts`:
  - `createSystem(opts: {lipids: number, box: [number, number, number], seed: number, layout: 'random' | 'bilayer' | 'vesicle', gamma?: number}): Promise<System>` — `gamma` перекрывает трение из `params.json` (нужно для проверки сохранения энергии при `gamma: 0`), `layout: 'vesicle'` описан в задаче 8
  - `System.step(n: number): Promise<void>`
  - `System.positions(): Promise<Float32Array>` — по 4 float на бид (x, y, z, type; type 0 = голова, 1 = хвост)
  - `System.kineticEnergyPerDof(): Promise<number>`
  - `System.totalEnergy(): Promise<number>`
  - `System.forcesBruteForce(): Promise<Float32Array>` — те же силы без сетки соседей, для сверки
  - `System.forces(): Promise<Float32Array>`

- [ ] **Step 1: Написать падающие тесты**

`tests/sim.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadParams } from '../engine/src/params'

afterAll(shutdownGpu)
const p = loadParams()

test('сетка соседей даёт те же силы, что и полный перебор', async () => {
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

test('термостат выводит систему на заданную температуру', async () => {
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

test('без трения полная энергия дрейфует слабо', async () => {
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

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/sim.test.ts`
Expected: FAIL — `api.createSystem` не функция.

- [ ] **Step 3: Реализовать сетку соседей**

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

Порядок работы: `count_main` считает заполнение, на CPU или отдельным ядром берутся префиксные суммы в буфер начал ячеек, `counts` обнуляется, `fill_main` раскладывает индексы. Префиксную сумму на первом шаге считать на CPU по вычитанному буферу — это допустимо, потому что размер сетки мал (тысячи ячеек), и заменяется на GPU-версию только если замер покажет, что это узкое место. Замер записать в `verify/out/gates.json` полем `neighborBuildMs`.

- [ ] **Step 4: Реализовать шаг Ланжевена**

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

Периодические границы применяются в отдельном коротком ядре после шага, чтобы сила и координаты не расходились в одном проходе: `x = x - floor(x / box) * box` покомпонентно по x и y, по z — стенки не применяются, бокс берётся достаточно большим.

`engine/src/sim.ts` собирает три pipeline (сетка, силы, интегратор), хранит буферы позиций, скоростей и сил, реализует перечисленные в Interfaces методы. `layout: 'bilayer'` раскладывает липиды в два слоя по сетке в плоскости xy с головами наружу; `layout: 'random'` рассыпает липиды случайно с ориентацией по случайному направлению; оба используют `seed` для воспроизводимости.

- [ ] **Step 5: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/sim.test.ts`
Expected: PASS все три. Диагностика: если температура выходит выше цели — проверить, что шум масштабирован как `sqrt(2·Γ·kT·dt)`, а не `sqrt(2·Γ·kT/dt)`; если силы не совпадают с полным перебором — проверить, что радиус ячейки сетки не меньше максимального радиуса взаимодействия `r_c + w_c`.

- [ ] **Step 6: Коммит**

```bash
git add engine/wgsl/neighbor.wgsl engine/wgsl/integrate.wgsl engine/src/sim.ts engine/src/index.ts tests/sim.test.ts
git commit -m "feat: Langevin integrator and neighbor grid with equipartition test"
```

---

### Task 5: Нулевое натяжение, площадь на липид, толщина бислоя

**Files:**
- Create: `engine/src/metrics.ts`
- Modify: `engine/src/sim.ts` (добавить MC-ход по площади), `engine/src/index.ts`
- Test: `tests/metrics.test.ts`

**Interfaces:**
- Consumes: `System` из задачи 4.
- Produces:
  - `System.areaMove(trials: number): Promise<number>` — Монте-Карло ход по площади бокса: `L_x` и `L_y` умножаются на `√s`, `L_z` не меняется; возвращает долю принятых ходов.
  - `areaPerLipid(box: [number, number, number], lipids: number): number`
  - `densityProfileZ(positions: Float32Array, box: [number, number, number], bins: number): {z: number[], head: number[], tail: number[]}`
  - `bilayerThickness(profile: ReturnType<typeof densityProfileZ>): number` — расстояние между максимумами плотности голов.
  - `measureBilayer(sys: System): Promise<{areaPerLipid: number, thickness: number, box: [number, number, number], lipids: number, steps: number}>` — фасад в `engine/src/index.ts`.

- [ ] **Step 1: Написать падающие тесты чистой математики**

`tests/metrics.test.ts`:

```ts
import { expect, test } from 'vitest'
import { areaPerLipid, bilayerThickness, densityProfileZ } from '../engine/src/metrics'

test('площадь на липид считается по половине числа липидов на слой', () => {
  expect(areaPerLipid([20, 20, 30], 1000)).toBeCloseTo(0.8, 12)
})

test('толщина берётся как расстояние между пиками плотности голов', () => {
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

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/metrics.test.ts`
Expected: FAIL — `engine/src/metrics` не найден.

- [ ] **Step 3: Реализовать метрики**

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

MC-ход по площади в `sim.ts`: предложить множитель `s = exp(u)`, `u ~ U(−0.005, 0.005)`, масштабировать `L_x, L_y` на `s` и `1/s`? Нет — при нулевом натяжении меняется именно площадь, поэтому масштабировать `L_x` и `L_y` оба на `√s`, координаты x и y частиц — тоже на `√s`, `L_z` не менять. Принять с вероятностью `min(1, exp(−ΔU/kT))`, где `ΔU` — разница полной потенциальной энергии до и после (натяжение γ = 0, поэтому член `γΔA` отсутствует). Вызывать ход раз в 100 шагов интегрирования.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/metrics.test.ts`
Expected: PASS оба.

- [ ] **Step 5: Написать падающий тест ворот 6 по бислою**

`tests/gate6-bilayer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('готовый бислой при нулевом натяжении держит площадь и толщину из литературы', async () => {
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

- [ ] **Step 6: Реализовать `measureBilayer` и добиться прохождения**

`measureBilayer(sys)` в `engine/src/index.ts`: снять позиции, посчитать `areaPerLipid` по текущему боксу, построить профиль на 200 бинов, вернуть `{areaPerLipid, thickness, box, lipids, steps}`.

Run: `npx vitest run tests/gate6-bilayer.test.ts`
Expected: PASS. Диагностика при провале: площадь заметно выше коридора означает, что MC-ход не принимается (проверить долю принятых — она должна быть 0.2–0.6; если 0, ошибка знака в `ΔU`); площадь ниже коридора при `kT/ε = 1.1` означает гелевую фазу — проверить, что `w_c = 1.6σ` дошло до шейдера.

- [ ] **Step 7: Коммит**

```bash
git add engine/src/metrics.ts engine/src/sim.ts engine/src/index.ts tests/metrics.test.ts tests/gate6-bilayer.test.ts
git commit -m "feat: zero-tension area move, area per lipid and thickness metrics"
```

---

### Task 6: Самосборка из случайного раствора

**Files:**
- Create: `engine/src/aggregate.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/self-assembly.test.ts`

**Interfaces:**
- Consumes: `System`, `densityProfileZ`.
- Produces: `clusters(positions: Float32Array, box: [number, number, number], cutoff: number): number[]` — размеры кластеров по связности хвостов; `largestClusterFraction(positions, box, cutoff): number`; фасад `largestClusterFractionOf(sys: System): Promise<number>` в `engine/src/index.ts`, снимающий позиции и зовущий `largestClusterFraction` с `cutoff = r_c + w_c`.

- [ ] **Step 1: Написать падающий тест кластеризации на синтетике**

`tests/self-assembly.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { clusters, largestClusterFraction } from '../engine/src/aggregate'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('кластеризация различает две далёкие группы', () => {
  const pos: number[] = []
  for (let i = 0; i < 10; i++) pos.push(1 + i * 0.2, 1, 1, 1)
  for (let i = 0; i < 6; i++) pos.push(20 + i * 0.2, 20, 20, 1)
  const sizes = clusters(new Float32Array(pos), [40, 40, 40], 0.5).sort((a, b) => b - a)
  expect(sizes).toEqual([10, 6])
})

test('из случайного раствора вырастает один крупный агрегат', async () => {
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

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/self-assembly.test.ts`
Expected: FAIL — `engine/src/aggregate` не найден.

- [ ] **Step 3: Реализовать кластеризацию**

`engine/src/aggregate.ts`: объединение по расстоянию хвостовых бидов с учётом периодичности по x и y, союз-поиск (union-find) с сеткой ячеек размера `cutoff`, возврат размеров компонент; `largestClusterFraction` делит максимальную компоненту на число хвостовых бидов.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/self-assembly.test.ts`
Expected: PASS оба. Если доля крупнейшего кластера застряла около 0.2 — это стадия мицелл; увеличить число шагов и записать замеренное время выхода в `gates.json` полем `assemblySteps`, а не «подкручивать» притяжение.

- [ ] **Step 5: Коммит**

```bash
git add engine/src/aggregate.ts engine/src/index.ts tests/self-assembly.test.ts
git commit -m "feat: cluster analysis and self-assembly from random solution"
```

---

### Task 7: Спектр ундуляций и модуль изгиба

**Files:**
- Create: `engine/src/spectrum.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/spectrum.test.ts`

**Interfaces:**
- Consumes: позиции из `System`.
- Produces:
  - `heightField(positions: Float32Array, box: [number, number, number], n: number): Float32Array` — срединная поверхность на сетке `n × n`.
  - `spectrum(h: Float32Array, n: number, box: [number, number, number]): {q: number[], hq2: number[]}`
  - `fitBendingModulus(s: {q: number[], hq2: number[]}, kT: number, area: number, qMax: number): number`
  - `synthesizeHeightField(n: number, box: [number, number, number], kappa: number, kT: number, seed: number): Float32Array` — генератор поля с известным κ для круговой проверки.
  - фасад `measureBendingModulus(sys: System, opts: {grid: number, modes: number, samples: number}): Promise<number>` в `engine/src/index.ts`.

- [ ] **Step 1: Написать падающий тест круговой проверки**

`tests/spectrum.test.ts`:

```ts
import { expect, test } from 'vitest'
import { fitBendingModulus, spectrum, synthesizeHeightField } from '../engine/src/spectrum'

test('оценка κ восстанавливает величину, заложенную в синтетическое поле', () => {
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

- [ ] **Step 2: Запустить и убедиться, что тест падает**

Run: `npx vitest run tests/spectrum.test.ts`
Expected: FAIL — `engine/src/spectrum` не найден.

- [ ] **Step 3: Реализовать спектр**

Соглашение о нормировке фиксируется один раз и используется и в генераторе, и в оценке: `h_q = (1/N)·Σ_r h(r)·exp(−i q·r)`, где `N = n²`, и `⟨|h_q|²⟩ = kT/(A·κ·q⁴)`. Генератор `synthesizeHeightField` заполняет амплитуды `|h_q| = sqrt(kT/(A·κ·q⁴))` со случайными фазами и делает обратное преобразование; `fitBendingModulus` берёт линейную регрессию `log⟨|h_q|²⟩` против `log q` в области `q ≤ qMax`, проверяет, что наклон близок к −4 (иначе бросает ошибку с замеренным наклоном), и извлекает κ из пересечения.

Дискретное преобразование делается прямым двойным циклом по модам с `q ≤ qMax`: сетка 64×64 и десяток мод — это доли секунды, быстрое преобразование не нужно.

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

Run: `npx vitest run tests/spectrum.test.ts`
Expected: PASS для всех трёх значений κ. Если восстановленное κ систематически вдвое-вчетверо отличается — это ошибка нормировки, а не физики: сверить множитель `1/N` в обе стороны.

- [ ] **Step 5: Написать тест ворот 6 по модулю изгиба**

`tests/gate6-kappa.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('модуль изгиба бислоя попадает в измеренный диапазон 5–50 kT', async () => {
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

- [ ] **Step 6: Реализовать `measureBendingModulus` и добиться прохождения**

Функция усредняет `|h_q|²` по `samples` конфигурациям, снимаемым через равные промежутки шагов, затем зовёт `fitBendingModulus`.

Run: `npx vitest run tests/gate6-kappa.test.ts`
Expected: PASS.

- [ ] **Step 7: Коммит**

```bash
git add engine/src/spectrum.ts engine/src/index.ts tests/spectrum.test.ts tests/gate6-kappa.test.ts
git commit -m "feat: undulation spectrum and bending modulus measurement"
```

---

### Task 8: Детектор замкнутой везикулы заливкой

**Files:**
- Create: `engine/src/closure.ts`, `engine/wgsl/closure.wgsl`
- Modify: `engine/src/index.ts`
- Test: `tests/closure.test.ts`

**Interfaces:**
- Consumes: позиции из `System`.
- Produces:
  - `occupancy(positions: Float32Array, box: [number, number, number], cell: number, radius: number): Uint8Array`
  - `enclosedVolume(occ: Uint8Array, dims: [number, number, number], cell: number): number` — эталон на TypeScript, заливка от границы, объём непролитых пустых ячеек.
  - фасады в `engine/src/index.ts`: `enclosedVolumeCpu(sys: System, opts: {cell: number, radius: number}): Promise<number>` — снимает позиции и считает эталоном; `enclosedVolumeGpu(sys: System, opts: {cell: number, radius: number}): Promise<number>` — считает ядром `closure.wgsl`.

- [ ] **Step 1: Написать падающие тесты на синтетике**

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

test('замкнутая оболочка даёт полость близкую к объёму шара', () => {
  const box: [number, number, number] = [40, 40, 40]
  const occ = occupancy(shell(8, 1.5, 40_000), box, 0.5, 0.6)
  const v = enclosedVolume(occ, [80, 80, 80], 0.5)
  const ideal = (4 / 3) * Math.PI * 7.5 ** 3
  expect(v).toBeGreaterThan(ideal * 0.7)
  expect(v).toBeLessThan(ideal * 1.3)
})

test('плоский лист полости не даёт', () => {
  const box: [number, number, number] = [40, 40, 40]
  const pos: number[] = []
  for (let i = 0; i < 20_000; i++) pos.push(Math.random() * 40, Math.random() * 40, 20 + (Math.random() - 0.5), 1)
  const occ = occupancy(new Float32Array(pos), box, 0.5, 0.6)
  expect(enclosedVolume(occ, [80, 80, 80], 0.5)).toBe(0)
})

test('версия на GPU совпадает с эталоном на TypeScript', async () => {
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

Задача 4 получает дополнение: `layout` принимает третье значение `'vesicle'` — липиды раскладываются двумя концентрическими слоями на сфере радиуса, подобранного под число липидов при площади 1.2 σ² на липид. Это стартовая конфигурация только для тестов детектора; сюжетная везикула получается самосборкой в задаче 6.

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/closure.test.ts`
Expected: FAIL — `engine/src/closure` не найден.

- [ ] **Step 3: Реализовать эталон и GPU-версию**

`closure.ts`: сетка занятости отмечает ячейки, центры которых ближе `radius` к любому биду (обход по ячейкам вокруг бида, без полного перебора); заливка — обход в ширину из всех пустых граничных ячеек; объём считается по непосещённым пустым ячейкам.

`closure.wgsl`: занятость одним проходом по бидам с `atomicOr`; заливка — итеративное распространение метки «снаружи» по соседям с флагом изменения в буфере, повторяется до отсутствия изменений (число итераций ограничить длиной диагонали сетки и записать замер в `gates.json`).

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/closure.test.ts`
Expected: PASS все три.

- [ ] **Step 5: Коммит**

```bash
git add engine/src/closure.ts engine/wgsl/closure.wgsl engine/src/sim.ts engine/src/index.ts tests/closure.test.ts
git commit -m "feat: closed-vesicle detection by outside flood fill"
```

---

### Task 9: Литературные цели, вердикты ворот, отчёт

**Files:**
- Create: `data/literature.json`, `verify/gates.ts`, `verify/report.ts`, `verify/run.ts`
- Test: `tests/gates.test.ts`

**Interfaces:**
- Consumes: метрики из задач 5–8.
- Produces:
  - `evaluateGates(metrics: Record<string, number>): GateResult[]`, где `GateResult = {id: string, title: string, value: number | null, target: {min?: number, max?: number}, unit: string, rank: 'A'|'B'|'C'|'D', verdict: 'passed'|'failed'|'unproven', source: string}`
  - `renderReport(results: GateResult[], meta: Record<string, unknown>): string`

- [ ] **Step 1: Написать падающие тесты**

`tests/gates.test.ts`:

```ts
import { expect, test } from 'vitest'
import { evaluateGates } from '../verify/gates'
import { renderReport } from '../verify/report'

test('метрика в коридоре проходит, вне коридора падает', () => {
  const ok = evaluateGates({ areaPerLipid: 1.2, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(ok.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('passed')
  const bad = evaluateGates({ areaPerLipid: 0.7, thickness: 5.0, bendingModulus: 20, enclosedVolume: 1800 })
  expect(bad.find((g) => g.id === 'area-per-lipid')!.verdict).toBe('failed')
})

test('отсутствующая метрика даёт «недоказано», а не «пройдено»', () => {
  const r = evaluateGates({ areaPerLipid: 1.2 })
  expect(r.find((g) => g.id === 'bending-modulus')!.verdict).toBe('unproven')
  expect(r.find((g) => g.id === 'bending-modulus')!.value).toBeNull()
})

test('ворота ранга D никогда не выходят пройденными', () => {
  const r = evaluateGates({ chainToBeadMapping: 1 })
  expect(r.find((g) => g.id === 'chain-to-bead-mapping')!.verdict).toBe('unproven')
})

test('отчёт содержит вердикт, число, коридор и ссылку', () => {
  const html = renderReport(evaluateGates({ areaPerLipid: 1.2 }), { commit: 'abc123' })
  expect(html).toContain('1.1')
  expect(html).toContain('cond-mat/0509218')
  expect(html).toContain('недоказано')
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/gates.test.ts`
Expected: FAIL — `verify/gates` не найден.

- [ ] **Step 3: Реализовать цели и вердикты**

`data/literature.json`:

```json
{
  "gates": [
    {
      "id": "area-per-lipid",
      "title": "Площадь на липид",
      "metric": "areaPerLipid",
      "unit": "sigma^2",
      "target": { "min": 1.1, "max": 1.5 },
      "rank": "A",
      "source": "Cooke & Deserno 2005, arXiv:cond-mat/0509218",
      "conditions": "kT/eps = 1.1, w_c = 1.6 sigma, нулевое натяжение"
    },
    {
      "id": "bilayer-thickness",
      "title": "Толщина бислоя",
      "metric": "thickness",
      "unit": "sigma",
      "target": { "min": 4.0, "max": 6.0 },
      "rank": "A",
      "source": "Cooke & Deserno 2005, arXiv:cond-mat/0509218",
      "conditions": "по максимумам плотности головных бидов"
    },
    {
      "id": "bending-modulus",
      "title": "Модуль изгиба",
      "metric": "bendingModulus",
      "unit": "kT",
      "target": { "min": 5, "max": 50 },
      "rank": "A",
      "source": "Cooke & Deserno 2005, arXiv:cond-mat/0509218",
      "conditions": "из спектра ундуляций при q <= 8 мод"
    },
    {
      "id": "closure",
      "title": "Замкнутая полость",
      "metric": "enclosedVolume",
      "unit": "sigma^3",
      "target": { "min": 1 },
      "rank": "A",
      "source": "детектор заливкой снаружи, спецификация раздел 7",
      "conditions": "сетка 0.5 sigma, радиус бида 0.6 sigma"
    },
    {
      "id": "chain-to-bead-mapping",
      "title": "Соответствие числа атомов углерода числу бидов",
      "metric": "chainToBeadMapping",
      "unit": "-",
      "target": {},
      "rank": "D",
      "source": "наше допущение, спецификация раздел 7",
      "conditions": "принимается только через ворота площади и толщины"
    }
  ]
}
```

`verify/gates.ts` читает этот файл, для каждой записи берёт метрику по имени: нет метрики — `unproven` и `value: null`; ранг `D` — всегда `unproven`; иначе сравнение с коридором. `verify/report.ts` рендерит статический HTML: таблица с колонками «ворота», «значение», «коридор», «ранг», «вердикт», «источник», «условия», плюс блок замеров производительности и коммит. `verify/run.ts` поднимает Vite, запускает Chrome, прогоняет измерительные сценарии задач 5–8, пишет `verify/out/gates.json` и `verify/out/report.html`.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/gates.test.ts`
Expected: PASS все четыре.

- [ ] **Step 5: Прогнать полный экзамен**

Run: `npm run verify`
Expected: создан `verify/out/report.html`, в нём четыре ворот ранга A с числами и одни ворота ранга D как «недоказано». Отдать пользователю полный путь к файлу и не давать своих оценок качества картинки.

- [ ] **Step 6: Коммит**

```bash
git add data/literature.json verify/gates.ts verify/report.ts verify/run.ts tests/gates.test.ts
git commit -m "feat: literature targets, gate verdicts and verification report"
```

---

### Task 10: Просмотрщик

**Files:**
- Create: `viewer/index.html`, `viewer/main.ts`, `viewer/panel.ts`
- Test: `tests/viewer.test.ts`

**Interfaces:**
- Consumes: `createSystem`, `System.step`, `System.positions`, `loadParams`.
- Produces: страница `/viewer/index.html`, публикующая `window.viewer = {frames: number, kT: number, wc: number, setKT(v: number): void, setWc(v: number): void, timeScaleBadge: string}`.

- [ ] **Step 1: Написать падающий тест**

`tests/viewer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('просмотрщик рисует кадры и показывает бейдж масштаба времени', async () => {
  const page = await gpuPage()
  await page.goto(new URL('viewer/index.html', page.url()).href, { waitUntil: 'load' })
  await page.waitForFunction('window.viewer && window.viewer.frames > 5')
  const state = await page.evaluate(() => ({
    frames: (window as any).viewer.frames,
    badge: (window as any).viewer.timeScaleBadge,
    kT: (window as any).viewer.kT,
  }))
  expect(state.frames).toBeGreaterThan(5)
  expect(state.badge).toContain('τ')
  expect(state.kT).toBe(1.1)

  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)
})
```

- [ ] **Step 2: Запустить и убедиться, что тест падает**

Run: `npx vitest run tests/viewer.test.ts`
Expected: FAIL — страница `viewer/index.html` отсутствует.

- [ ] **Step 3: Реализовать просмотрщик**

`viewer/main.ts`: three.js `WebGPURenderer` не требуется — сцена рисуется обычным `WebGLRenderer` из позиций, снятых с симуляции, через `InstancedMesh` со сферами (головы и хвосты разными материалами), `OrbitControls`, счётчик кадров в `window.viewer.frames`. Симуляция шагает по 20 шагов на кадр. Бейдж масштаба времени показывает `шаг δt = 0.01τ, показано N τ` и подпись «биды, не атомы» — по требованию спецификации о неспутывании бида с атомом.

`viewer/panel.ts`: два ползунка — `kT/ε` в пределах, указанных в `params.json` полем `kTRange`, и `w_c` от 1.0 до 2.0 σ; изменение перезаписывает униформ-буфер и не пересоздаёт систему.

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

Run: `npx vitest run tests/viewer.test.ts`
Expected: PASS.

- [ ] **Step 5: Замерить производительность и записать в отчёт**

Добавить в `verify/run.ts` замер: система на 4000 липидов (12 000 бидов), 1000 шагов, записать шагов в секунду в `gates.json` полем `stepsPerSecond` и число бидов полем `beads`. Никаких утверждений о скорости в тексте — только число в отчёте.

Run: `npm run verify`
Expected: в `verify/out/gates.json` присутствуют `stepsPerSecond`, `beads`, `neighborBuildMs`.

- [ ] **Step 6: Коммит**

```bash
git add viewer/index.html viewer/main.ts viewer/panel.ts verify/run.ts tests/viewer.test.ts
git commit -m "feat: interactive viewer with parameter panel and time-scale badge"
```

---

## Что план сознательно не делает

Ступени A и B (химия и ворота агрегации), слои `thermo/`, `qm/`, `ref/`, ворота 1–5 и фазовая карта — предмет отдельных планов. Этот план заканчивается работающей ступенью C с воротами 6 и автоматическим отчётом, на котором всё остальное потом проверяется.
