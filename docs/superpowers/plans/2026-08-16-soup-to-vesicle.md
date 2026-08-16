# Из бульона в везикулу: один непрерывный прогон — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Один прогон без швов: сцена стартует смесью простых молекул, связи образуются правилами, из них возникают амфифилы, амфифилы сами собираются в бислой и замыкаются в везикулу — и всё это проверяется воротами, включая честное «нет» там, где везикулы не возникают.

**Architecture:** Одна огрублённая шкала на весь прогон. Частица — либо мономер-носитель (углеродное звено, кислородная голова, водородный донор), либо каталитический центр, либо готовое звено цепи. Связи создаются и рвутся стохастическими правилами с детальным балансом, поэтому равновесие определяется разностью энергий, а не произволом. Несвязанные взаимодействия берутся из проверенной модели ступени C, поэтому собранный бислой обязан воспроизводить её измеренные площадь на липид и толщину. Замкнутость определяется заливкой снаружи — тем же детектором, что уже проверен.

**Tech Stack:** TypeScript, WGSL (WebGPU), three.js, Vitest, puppeteer-core с системным Chrome. Переиспользуются `engine/wgsl/forces.wgsl`, `engine/wgsl/neighbor.wgsl`, `engine/src/closure.ts`, `chem/src/species.ts`, `chem/src/backmap.ts`, механика `verify/`.

**Spec:** `docs/superpowers/specs/2026-08-15-protocell-genesis-design.md`, раздел 1a (поправка о непрерывном прогоне), разделы 3 и 8

## Global Constraints

- **Один прогон, одна шкала времени.** Ни одна величина не переносится между отдельными симуляциями; сцена от бульона до везикулы идёт непрерывно. Единственный множитель времени `κ_t` выводится числом на экран и в отчёт.
- Несвязанные взаимодействия — параметры Cooke & Deserno 2005 из `data/params.json`, без изменений: `b_hh = b_ht = 0.95σ`, `b_tt = σ`, `w_c = 1.6σ`, `kT/ε = 1.1`, WCA с обрезкой `2^(1/6)b`, притяжение `−ε cos²`. Собранный бислой обязан попадать в измеренные коридоры: площадь на липид **1.1–1.5 σ²**, толщина **4–6 σ**.
- Связи, образованные правилами, используют ту же FENE-связь и тот же изгибный потенциал, что проверены аналитически в ступени C. Новых потенциалов не вводится.
- **Скорости реакций несут ранг D** — литературных константы для условий ранней Земли не существует. Любые ворота, зависящие только от них, публикуются как недоказанные. Обратные скорости связаны с прямыми детальным балансом `k_f/k_r = exp(−ΔU/kT)`, поэтому равновесие корректно при неизвестном времени.
- **Точные инварианты:** число мономеров каждого сорта сохраняется, число связей меняется только на события образования и разрыва, суммарный заряд сохраняется. Нарушение — падение теста, не предупреждение.
- Никаких вписанных химических констант в коде: правила и скорости живут в `data/soup.json`. Сторож из `tests/params.test.ts` расширяется на `soup/`.
- Результаты читаются через `page.evaluate`; `--dump-dom` запрещён.
- Длинные прогоны — в фоновом режиме инструмента с логом и ожиданием через `Monitor` в том же ходу. Один прогон на вызов. Не держать dev-сервер во время GPU-набора: замерено, что конкуренция убивает воркер vitest, а не роняет тест.

---

## Структура файлов

- `data/soup.json` — сорта мономеров, правила связывания и разрыва с рангами, стартовый состав, окно параметров для фазовой карты.
- `soup/src/rules.ts` — загрузка правил, проверка балансов, детальный баланс, вероятности на шаг.
- `soup/src/sim.ts` — состояние прогона: частицы, связи, каталитические центры, счётчики событий, инварианты.
- `soup/wgsl/bond.wgsl` — образование и разрыв связей по сетке соседей с атомарным захватом участников.
- `soup/wgsl/step.wgsl` — интегратор Ланжевена поверх связей и несвязанных сил (переиспользует функции `engine/wgsl/forces.wgsl`).
- `soup/src/amphiphile.ts` — распознавание амфифила в графе связей: цепь с полярным концом, длина, счёт по длинам.
- `soup/src/stages.ts` — определение стадий прогона по замерам: мономеры, амфифилы, мицеллы, бислой, замкнутая везикула.
- `viewer/soup.html`, `viewer/soup.ts` — сцена непрерывного прогона: поатомный вид через `chem/src/backmap.ts` для распознанных амфифилов, шкала стадий, счётчики, бейдж `κ_t` и рангов.
- `verify/soup.ts` — сценарии и ворота непрерывного прогона, включая фазовую карту.
- `tests/soup-*.test.ts` — тесты по задачам.

---

### Task 1: Правила связывания и инварианты

**Files:**
- Create: `data/soup.json`, `soup/src/rules.ts`
- Modify: `tests/params.test.ts` (сторож на `soup/src`, `soup/wgsl`)
- Test: `tests/soup-rules.test.ts`

**Interfaces:**
- Produces: `loadSoup(): Soup` где `Soup = {monomers: Monomer[], rules: Rule[], start: Record<string, number>, sweep: Sweep}`, `Monomer = {id: string, kind: 'carbon' | 'head' | 'donor' | 'catalyst', radiusSigma: number, polar: boolean}`, `Rule = {id: string, kind: 'bond' | 'break', a: string, b: string, requiresCatalyst: boolean, energyKT: number, attemptRate: number, rank: 'A'|'B'|'C'|'D', basis: string}`; `forwardBackwardRatio(r: Rule): number` — `exp(−energyKT)`; `assertRulesConsistent(s: Soup): void` — бросает, если у правила образования нет парного разрыва, если ранг не D для скоростей синтеза, или если обоснование пустое.

- [ ] **Step 1: Написать падающие тесты**

`tests/soup-rules.test.ts`:

```ts
import { expect, test } from 'vitest'
import { assertRulesConsistent, forwardBackwardRatio, loadSoup } from '../soup/src/rules'

test('каждое правило образования имеет парный разрыв и детальный баланс', () => {
  const s = loadSoup()
  expect(() => assertRulesConsistent(s)).not.toThrow()
  for (const r of s.rules.filter((x) => x.kind === 'bond')) {
    const back = s.rules.find((x) => x.kind === 'break' && x.a === r.a && x.b === r.b)
    expect(back).toBeDefined()
    expect(forwardBackwardRatio(r)).toBeCloseTo(Math.exp(-r.energyKT), 10)
  }
})

test('все скорости синтеза объявлены рангом D с обоснованием', () => {
  for (const r of loadSoup().rules) {
    expect(r.rank).toBe('D')
    expect(r.basis.length).toBeGreaterThan(10)
  }
})

test('стартовый состав содержит только мономеры и не содержит готовых амфифилов', () => {
  const s = loadSoup()
  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) expect(ids.has(k)).toBe(true)
  expect(Object.keys(s.start).length).toBeGreaterThan(2)
})

test('несогласованный набор правил выявляется', () => {
  const s = loadSoup()
  const broken = { ...s, rules: s.rules.filter((r) => r.kind !== 'break') }
  expect(() => assertRulesConsistent(broken)).toThrow(/разрыв/)
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/soup-rules.test.ts`
Expected: FAIL — `soup/src/rules` не найден.

- [ ] **Step 3: Реализовать правила**

`data/soup.json`:

```json
{
  "kappaT": 1.0,
  "monomers": [
    { "id": "C", "kind": "carbon",   "radiusSigma": 1.0,  "polar": false },
    { "id": "O", "kind": "head",     "radiusSigma": 0.95, "polar": true  },
    { "id": "H", "kind": "donor",    "radiusSigma": 0.8,  "polar": false },
    { "id": "M", "kind": "catalyst", "radiusSigma": 1.2,  "polar": false }
  ],
  "start": { "C": 6000, "O": 900, "H": 6000, "M": 200 },
  "rules": [
    { "id": "cc_bond",  "kind": "bond",  "a": "C", "b": "C", "requiresCatalyst": true,  "energyKT": 6.0, "attemptRate": 0.05, "rank": "D",
      "basis": "оценка: рост углеродной цепи идёт только на каталитическом центре; глубина 6 kT задаёт устойчивость цепи к тепловому разрыву" },
    { "id": "cc_break", "kind": "break", "a": "C", "b": "C", "requiresCatalyst": false, "energyKT": 6.0, "attemptRate": 0.05, "rank": "D",
      "basis": "парный разрыв к cc_bond, связан детальным балансом" },
    { "id": "co_bond",  "kind": "bond",  "a": "C", "b": "O", "requiresCatalyst": false, "energyKT": 8.0, "attemptRate": 0.05, "rank": "D",
      "basis": "оценка: присоединение полярной головы к концу цепи, глубже связи C-C, поэтому голова не отваливается первой" },
    { "id": "co_break", "kind": "break", "a": "C", "b": "O", "requiresCatalyst": false, "energyKT": 8.0, "attemptRate": 0.05, "rank": "D",
      "basis": "парный разрыв к co_bond, связан детальным балансом" }
  ],
  "sweep": { "kT": [0.9, 1.1, 1.3], "carbonDensity": [0.03, 0.06, 0.09], "catalystCount": [50, 200, 800] }
}
```

`soup/src/rules.ts` загружает файл, проверяет парность правил, ранги и обоснования, и выводит вероятность попытки на шаг как `attemptRate·dt`, а вероятность принятия — из `energyKT` по Метрополису, чтобы прямое и обратное направления автоматически удовлетворяли детальному балансу.

Расширить сторож в `tests/params.test.ts`: каталоги становятся `['engine/src', 'engine/wgsl', 'chem/src', 'soup/src', 'soup/wgsl']`.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/soup-rules.test.ts tests/params.test.ts`
Expected: PASS все пять.

- [ ] **Step 5: Коммит**

```bash
git add data/soup.json soup/src/rules.ts tests/soup-rules.test.ts tests/params.test.ts
git commit -m "feat: bonding rules with detailed balance and exact invariants"
```

---

### Task 2: Образование и разрыв связей в динамике

**Files:**
- Create: `soup/wgsl/bond.wgsl`, `soup/wgsl/step.wgsl`, `soup/src/sim.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/soup-bonds.test.ts`

**Interfaces:**
- Consumes: `loadSoup`, вероятности из задачи 1; `getGpu`, `readBack`, `storageBuffer`; функции потенциалов из `engine/wgsl/forces.wgsl`; сетка из `engine/wgsl/neighbor.wgsl`.
- Produces: `createSoup(opts: {box: [number,number,number], seed: number, kT: number, start?: Record<string, number>, catalystCount?: number}): Promise<SoupSystem>`; `SoupSystem.step(n: number): Promise<void>`; `SoupSystem.particles(): Promise<Float32Array>` — по 4 float (x, y, z, сорт); `SoupSystem.bonds(): Promise<Uint32Array>` — пары индексов; `SoupSystem.events(): Promise<Record<string, number>>`; `SoupSystem.invariants(): Promise<{monomers: Record<string, number>, bonds: number, charge: number}>`.

- [ ] **Step 1: Написать падающие тесты**

`tests/soup-bonds.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('связи образуются только на каталитическом центре там, где правило это требует', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const withM = await api.createSoup({ box: [30, 30, 30], seed: 4, kT: 1.1, catalystCount: 200 })
    await withM.step(50000)
    const a = await withM.events()
    const noM = await api.createSoup({ box: [30, 30, 30], seed: 4, kT: 1.1, catalystCount: 0 })
    await noM.step(50000)
    const b = await noM.events()
    return { withCatalyst: a['cc_bond'] ?? 0, without: b['cc_bond'] ?? 0 }
  })
  expect(r.withCatalyst).toBeGreaterThan(100)
  expect(r.without).toBe(0)
})

test('число мономеров каждого сорта и заряд сохраняются при работающих реакциях', async () => {
  const page = await gpuPage()
  const inv = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSoup({ box: [30, 30, 30], seed: 7, kT: 1.1 })
    const before = await sys.invariants()
    await sys.step(50000)
    const after = await sys.invariants()
    return { before, after }
  })
  expect(inv.after.monomers).toEqual(inv.before.monomers)
  expect(inv.after.charge).toBe(inv.before.charge)
  expect(inv.after.bonds).toBeGreaterThan(0)
})

test('при высокой температуре связей меньше, чем при низкой — равновесие определяется энергией', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const out: Record<string, number> = {}
    for (const kT of [0.9, 1.8]) {
      const sys = await api.createSoup({ box: [30, 30, 30], seed: 11, kT })
      await sys.step(80000)
      out[String(kT)] = (await sys.invariants()).bonds
    }
    return out
  })
  expect(r['1.8']).toBeLessThan(r['0.9'])
}, 1_800_000)
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/soup-bonds.test.ts`
Expected: FAIL — `api.createSoup` не функция.

- [ ] **Step 3: Реализовать динамику связей**

`soup/wgsl/bond.wgsl`: по сетке соседей ищутся пары подходящих сортов на расстоянии меньше контактного радиуса. Оба участника захватываются атомарно, чтобы одна частица не получила две связи за шаг сверх допустимой валентности (углерод — не больше двух связей в цепи плюс одна голова; голова — одна связь). Попытка образования принимается по Метрополису от `energyKT`; попытка разрыва идёт по существующим связям с той же энергией, поэтому отношение прямой и обратной вероятностей равно `exp(−ΔU/kT)` численно, а не по декларации.

`soup/wgsl/step.wgsl`: шаг Ланжевена, где силы складываются из несвязанных взаимодействий (функции из `engine/wgsl/forces.wgsl`, радиусы по сортам, притяжение только между неполярными концами) и связанных FENE плюс изгиб для троек вдоль цепи. Полярные головы не участвуют в притяжении хвостов — именно эта асимметрия делает продукт амфифилом.

`soup/src/sim.ts` держит буферы частиц, связей, валентностей, центров, счётчиков; реализует перечисленные методы; проверяет валентность как инвариант при каждом чтении связей.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/soup-bonds.test.ts`
Expected: PASS все три. Если связи образуются без катализатора — правило `requiresCatalyst` не дошло до шейдера; если число мономеров изменилось — частица потеряна при захвате, проверить откат при неудачном атомарном обмене.

- [ ] **Step 5: Коммит**

```bash
git add soup/wgsl/bond.wgsl soup/wgsl/step.wgsl soup/src/sim.ts engine/src/index.ts tests/soup-bonds.test.ts
git commit -m "feat: bond formation and breaking in dynamics, catalyst-gated"
```

---

### Task 3: Распознавание амфифила и стадий

**Files:**
- Create: `soup/src/amphiphile.ts`, `soup/src/stages.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/soup-amphiphile.test.ts`

**Interfaces:**
- Consumes: `SoupSystem.particles`, `SoupSystem.bonds`.
- Produces: `findAmphiphiles(particles: Float32Array, bonds: Uint32Array, monomers: Monomer[]): Amphiphile[]` где `Amphiphile = {headIndex: number, chain: number[], length: number}` — цепь из углеродов с ровно одной полярной головой на конце; `amphiphileHistogram(a: Amphiphile[]): Record<number, number>`; `detectStage(sys: SoupSystem): Promise<{stage: 'monomers'|'amphiphiles'|'micelles'|'bilayer'|'vesicle', evidence: Record<string, number>}>` — стадия определяется замерами: доля углерода в амфифилах (поле `amphiphileFraction`), размер крупнейшего агрегата, число пиков в профиле плотности голов, объём непролитой полости (поле `enclosedVolume`). Фасад в `engine/src/index.ts`: `stageOf(sys: SoupSystem)` — то же самое, опубликованное как `window.api.stageOf` для тестов и сцены.

- [ ] **Step 1: Написать падающие тесты**

`tests/soup-amphiphile.test.ts`:

```ts
import { expect, test } from 'vitest'
import { amphiphileHistogram, findAmphiphiles } from '../soup/src/amphiphile'
import { loadSoup } from '../soup/src/rules'

const P = (xs: number[][]) => new Float32Array(xs.flat())

test('цепь с одной полярной головой распознаётся как амфифил', () => {
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 2,3])
  const a = findAmphiphiles(parts, bonds, loadSoup().monomers)
  expect(a.length).toBe(1)
  expect(a[0].length).toBe(3)
})

test('цепь без головы и цепь с двумя головами амфифилами не считаются', () => {
  const m = loadSoup().monomers
  const noHead = findAmphiphiles(P([[0,0,0,0],[1,0,0,0],[2,0,0,0]]), new Uint32Array([0,1, 1,2]), m)
  expect(noHead.length).toBe(0)
  const twoHeads = findAmphiphiles(P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,1]]), new Uint32Array([0,1, 1,2, 2,3]), m)
  expect(twoHeads.length).toBe(0)
})

test('гистограмма длин считает цепи по числу углеродов', () => {
  const m = loadSoup().monomers
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0], [10,0,0,1],[11,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 3,4])
  expect(amphiphileHistogram(findAmphiphiles(parts, bonds, m))).toEqual({ 1: 1, 2: 1 })
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/soup-amphiphile.test.ts`
Expected: FAIL — `soup/src/amphiphile` не найден.

- [ ] **Step 3: Реализовать распознавание**

`soup/src/amphiphile.ts` обходит граф связей союз-поиском, для каждой компоненты проверяет, что она — простая цепь (все степени не больше двух), содержит ровно одну полярную частицу и что эта частица на конце; длина считается числом неполярных звеньев. `soup/src/stages.ts` вызывает `findAmphiphiles`, `largestClusterFraction` из `engine/src/aggregate.ts`, `bilayerPeaks` из `engine/src/metrics.ts` и `enclosedVolumeCpu` из фасада, и по этим четырём замерам возвращает стадию с числами-доказательствами.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/soup-amphiphile.test.ts`
Expected: PASS все три.

- [ ] **Step 5: Коммит**

```bash
git add soup/src/amphiphile.ts soup/src/stages.ts engine/src/index.ts tests/soup-amphiphile.test.ts
git commit -m "feat: amphiphile recognition and stage detection from measurements"
```

---

### Task 4: Непрерывный прогон до везикулы

**Files:**
- Modify: `soup/src/sim.ts`, `engine/src/index.ts`
- Test: `tests/soup-vesicle.test.ts`

**Interfaces:**
- Produces: `SoupSystem.runUntil(stage: string, opts: {maxSteps: number, sampleEvery: number}): Promise<{reached: boolean, steps: number, trace: {steps: number, stage: string, evidence: Record<string, number>}[]}>`.

- [ ] **Step 1: Написать падающий тест**

`tests/soup-vesicle.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('из бульона без готовых амфифилов возникает замкнутая везикула', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSoup({ box: [40, 40, 40], seed: 19, kT: 1.1, catalystCount: 400 })
    const startAmph = (await api.stageOf(sys)).evidence.amphiphileFraction
    const run = await sys.runUntil('vesicle', { maxSteps: 4_000_000, sampleEvery: 20_000 })
    return { startAmph, ...run }
  })
  expect(r.startAmph).toBe(0)
  expect(r.trace.map((t) => t.stage)).toContain('amphiphiles')
  expect(r.trace.map((t) => t.stage)).toContain('bilayer')
  expect(r.reached).toBe(true)
  expect(r.trace[r.trace.length - 1].evidence.enclosedVolume).toBeGreaterThan(1)
}, 3_600_000)
```

- [ ] **Step 2: Запустить и убедиться, что тест падает**

Run: `npx vitest run tests/soup-vesicle.test.ts`
Expected: FAIL — `runUntil` отсутствует.

- [ ] **Step 3: Реализовать прогон и добиться прохождения**

`runUntil` шагает пачками по `sampleEvery`, после каждой считает стадию и пишет в трассу, останавливается при достижении целевой стадии или при исчерпании `maxSteps`. Если везикула не возникает — **не подкручивать притяжение и не ослаблять порог**: сначала посмотреть трассу и решить по числам, чего не хватает (углерода на амфифилы, центров на скорость синтеза, времени на замыкание), затем менять состав или число шагов в `data/soup.json` и в тесте, документируя выбор. Если и это не помогает, доложить BLOCKED с полной трассой.

Run: `npx vitest run tests/soup-vesicle.test.ts`
Expected: PASS.

- [ ] **Step 4: Коммит**

```bash
git add soup/src/sim.ts engine/src/index.ts tests/soup-vesicle.test.ts
git commit -m "feat: continuous run from soup to closed vesicle"
```

---

### Task 5: Сцена непрерывного прогона

**Files:**
- Create: `viewer/soup.html`, `viewer/soup.ts`
- Test: `tests/soup-viewer.test.ts`

**Interfaces:**
- Consumes: `createSoup`, `runUntil`, `findAmphiphiles`, `backmapLipid` из `chem/src/backmap.ts`, `data/atoms.json`.
- Produces: страница `/viewer/soup.html`, публикующая `window.soup = {frames: number, steps: number, stage: string, amphiphiles: number, largestCluster: number, enclosedVolume: number, kappaT: number, badge: string}`.

- [ ] **Step 1: Написать падающий тест**

`tests/soup-viewer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('сцена бульона идёт, показывает стадию и подписывает ранги', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/soup.html', page.url()).href, { waitUntil: 'load' })
  await page.waitForFunction('window.soup && window.soup.frames > 5 && window.soup.steps > 0')
  const s = await page.evaluate(() => ({ ...(window as any).soup }))
  expect(['monomers', 'amphiphiles', 'micelles', 'bilayer', 'vesicle']).toContain(s.stage)
  expect(s.badge).toContain('κ_t')
  expect(s.badge.toLowerCase()).toContain('ранг d')
  expect(typeof s.enclosedVolume).toBe('number')
  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)
})
```

- [ ] **Step 2: Запустить и убедиться, что тест падает**

Run: `npx vitest run tests/soup-viewer.test.ts`
Expected: FAIL — страница отсутствует.

- [ ] **Step 3: Реализовать сцену**

Мономеры рисуются шарами по сортам и радиусам; распознанные амфифилы разворачиваются поатомно через `backmapLipid` для тех, что попали в срез, остальные — упрощённо; связи — цилиндрами. Сверху идёт шкала стадий с подсветкой текущей и числами-доказательствами: доля углерода в амфифилах, размер крупнейшего агрегата, число пиков плотности голов, объём полости. Бейдж говорит: множитель времени `κ_t` числом, скорости реакций ранга D, атомная детализация амфифилов — реконструкция по справочной геометрии, а сборка идёт проверенным потенциалом.

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

Run: `npx vitest run tests/soup-viewer.test.ts`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add viewer/soup.html viewer/soup.ts tests/soup-viewer.test.ts
git commit -m "feat: continuous soup-to-vesicle scene with stage evidence on screen"
```

---

### Task 6: Проверка гипотезы фазовой картой

**Files:**
- Create: `verify/soup.ts`
- Modify: `data/literature.json`, `verify/run.ts`, `verify/report.ts`
- Test: `tests/soup-gates.test.ts`

**Interfaces:**
- Produces: ворота `soup-to-vesicle`, `soup-invariants`, `soup-bilayer-corridors`, `soup-phase-window` в `data/literature.json`; `sweepPhaseMap(): Promise<{kT: number, carbonDensity: number, catalystCount: number, stage: string, enclosedVolume: number}[]>` в `verify/soup.ts`.

- [ ] **Step 1: Написать падающие тесты**

`tests/soup-gates.test.ts`:

```ts
import { expect, test } from 'vitest'
import { readFileSync } from 'node:fs'
import { evaluateGates } from '../verify/gates'

test('ворота непрерывного прогона объявлены', () => {
  const ids = JSON.parse(readFileSync('data/literature.json', 'utf8')).gates.map((g: any) => g.id)
  for (const id of ['soup-to-vesicle', 'soup-invariants', 'soup-bilayer-corridors', 'soup-phase-window']) {
    expect(ids).toContain(id)
  }
})

test('инварианты бульона проходят только при точном нуле дрейфа', () => {
  expect(evaluateGates({ soupInvariantDrift: 0 }).find((g) => g.id === 'soup-invariants')!.verdict).toBe('passed')
  expect(evaluateGates({ soupInvariantDrift: 1 }).find((g) => g.id === 'soup-invariants')!.verdict).toBe('failed')
})

test('ворота пути до везикулы имеют ранг D и выходят недоказанными', () => {
  const r = evaluateGates({ soupReachedVesicle: 1 })
  expect(r.find((g) => g.id === 'soup-to-vesicle')!.rank).toBe('D')
  expect(r.find((g) => g.id === 'soup-to-vesicle')!.verdict).toBe('unproven')
})

test('коридоры собранного бислоя те же, что у проверенной модели', () => {
  const lit = JSON.parse(readFileSync('data/literature.json', 'utf8'))
  const g = lit.gates.find((x: any) => x.id === 'soup-bilayer-corridors')
  expect(g.target).toEqual({ min: 1.1, max: 1.5 })
  expect(g.rank).toBe('A')
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/soup-gates.test.ts`
Expected: FAIL — записи ворот отсутствуют.

- [ ] **Step 3: Реализовать ворота и фазовую карту**

Записи в `data/literature.json`: `soup-to-vesicle` (метрика `soupReachedVesicle`, ранг **D**, источник — гипотеза первичного образования везикул, условия — скорости ранга D, поэтому ворота недоказаны по построению); `soup-invariants` (метрика `soupInvariantDrift`, цель `max: 0`, ранг A, точный инвариант); `soup-bilayer-corridors` (метрика `soupAreaPerLipid`, цель `1.1–1.5`, ранг A, тот же источник Cooke & Deserno 2005 — собранный из бульона бислой обязан попасть в тот же коридор, и это настоящая проверка, что правила дали настоящий амфифил); `soup-phase-window` (метрика `soupVesicleFraction` — доля точек сетки, где возникла везикула, ранг D, содержание проверки гипотезы).

`verify/soup.ts` гоняет прогон до везикулы, считает инварианты, измеряет площадь на липид собранного бислоя и прогоняет сетку параметров из `data/soup.json`, отдавая карту точек со стадиями. `verify/report.ts` печатает карту таблицей: где везикула возникла, где остановилась на бислое, где на мицеллах, где вещество осталось мономерами — и подпись, что «нет» в части карты является результатом, а не сбоем.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/soup-gates.test.ts`
Expected: PASS все четыре.

- [ ] **Step 5: Прогнать экзамен целиком**

Run: `npm run verify`
Expected: в отчёте появились ворота непрерывного прогона и таблица фазовой карты. Отдать пользователю полный путь к отчёту.

- [ ] **Step 6: Коммит**

```bash
git add verify/soup.ts data/literature.json verify/run.ts verify/report.ts tests/soup-gates.test.ts
git commit -m "feat: hypothesis test by phase map over temperature, density and catalyst"
```

---

## Что план сознательно не делает

Не привязывает время к секундам: `κ_t` остаётся единственным множителем, и абсолютные скорости не заявляются. Не считает квантовые барьеры и не повышает ранг скоростей выше D — это отдельные слои `thermo/` и `qm/`. Не строит полноатомную динамику: атомная детализация в кадре остаётся реконструкцией поверх огрублённых координат, и так подписана на экране.
