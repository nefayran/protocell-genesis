# Ступень A: атомы, реакции, рост цепи — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Настоящие атомы в трёхмерной сцене: молекулы диффундируют, сталкиваются, реагируют, на каталитической стенке растёт углеродная цепь, и из неё получается молекула алкановой кислоты — с проверкой каждого механизма против аналитического закона.

**Architecture:** Реактивная броуновская динамика в WGSL: частица — молекула с внутренней поатомной геометрией, построенной по литературным длинам связей. Реакции происходят при контакте с вероятностью, выведенной из константы скорости через диффузионный предел Смолуховского. Рост цепи идёт на дискретных активных центрах стенки как конкуренция продолжения и обрыва, поэтому распределение длин цепей возникает следствием механизма и предсказывается независимо измеренным отношением скоростей. Рендер поатомный: шар-стержень, радиусы Ван-дер-Ваальса, цвета CPK.

**Tech Stack:** TypeScript, WGSL (WebGPU), three.js, Vitest, puppeteer-core с системным Chrome. Переиспользуется вся обвязка ступени C: `engine/src/gpu.ts`, шаблон сетки соседей, стенд тестов через `gpuPage()`, механика ворот из `verify/`.

**Spec:** `docs/superpowers/specs/2026-08-15-protocell-genesis-design.md`, раздел 5 (ступень A) и раздел 3 (иерархия доводов)

## Global Constraints

- Единицы ступени A — **нанометры, наносекунды, молекулы**; они не совпадают с приведёнными единицами ступени C (σ, τ, ε). Ни одна величина не переносится между ступенями без явного пересчёта, и на экране всегда видно, в каких единицах идёт текущая ступень.
- Условия прогона: водный флюид, **175 °C = 448.15 K**, вязкость воды при этой температуре **1.53·10⁻⁴ Pa·s**. Источник углерода — CO и H₂ (продукты диспропорционирования формиата или оксалата).
- Атомные данные (ранг A, из справочников, значения проверяются тестом): длины связей CO 1.128 Å, H₂ 0.741 Å, H₂O O–H 0.9572 Å при угле 104.52°, CO₂ C=O 1.160 Å линейно, C–C 1.54 Å, C–H 1.09 Å, карбоксильные C=O 1.21 Å и C–O 1.36 Å, O–H 0.97 Å, тетраэдрический угол 109.47°. Радиусы Ван-дер-Ваальса по Бонди: H 1.20, C 1.70, N 1.55, O 1.52 Å.
- **Константы скорости реакций синтеза для условий ранней Земли не существует** (спека, раздел 2). Все они входят с рангом **D**, помечены как оценки, и любые ворота, опирающиеся только на них, публикуются как недоказанные. Ранг повышается позже слоями `thermo/` и `qm/`, не в этом плане.
- Единственный глобальный множитель времени `κ_t` выводится числом на экран и в отчёт. Ни одна другая величина не масштабируется скрытно.
- Диффузия считается по Стоксу—Эйнштейну `D = kT/(6πηa)`; диффузионный предел реакции — по Смолуховскому `k = 4πD_AB·R_AB`, где `D_AB = D_A + D_B`.
- Баланс массы по каждому элементу и баланс заряда — **точные инварианты**. Ни одна реакция не проходит проверку схемы, если они нарушены, и длинный прогон проверяется на их сохранение.
- Никаких вписанных химических констант в коде: всё читается из `data/chemistry.json` и `data/atoms.json`. Сторож из `tests/params.test.ts` расширяется на новые каталоги.
- Результаты из браузера читаются только через `page.evaluate`; `--dump-dom` запрещён (замерено: отдаёт содержимое до завершения работы GPU).
- Длинные прогоны запускаются в фоновом режиме инструмента с выводом в лог и ожидаются в том же ходу через `Monitor`; передний план обрезан на 600 с, дочерний процесс через `&` умирает с концом хода, а цепочка длинных прогонов в одном вызове убивает воркер vitest.

---

## Структура файлов

- `data/atoms.json` — элементы: масса, радиус Ван-дер-Ваальса, цвет CPK, ковалентный радиус.
- `data/molecules.json` — виды: формула, заряд, топология связей, длины и углы, источник каждого значения.
- `data/chemistry.json` — реакции: стехиометрия, тип (объёмная, адсорбция, продолжение цепи, обрыв), константа с ранга D и её обоснование, контактный радиус.
- `chem/src/species.ts` — разбор формулы, масса, заряд, число атомов по элементам.
- `chem/src/geometry.ts` — построение поатомных координат вида по длинам и углам, включая зигзаг насыщенной цепи.
- `chem/src/network.ts` — загрузка `chemistry.json`, проверка баланса массы и заряда, вывод контактных радиусов и вероятностей реакции.
- `chem/src/diffusion.ts` — Стокс—Эйнштейн, Смолуховский, перевод констант в вероятность на шаг.
- `chem/wgsl/bd.wgsl` — броуновский шаг с периодическими границами.
- `chem/wgsl/react.wgsl` — контактные реакции по сетке соседей.
- `chem/wgsl/surface.wgsl` — активные центры стенки: адсорбция, продолжение, обрыв.
- `chem/src/sim.ts` — состояние ступени A, шаг, счётчики событий, инварианты.
- `chem/src/asf.ts` — распределение длин цепей, оценка α из счётчиков и из гистограммы.
- `viewer/atoms.ts` — поатомный рендер: шар-стержень, CPK, радиусы, легенда, бейдж единиц и `κ_t`.
- `verify/stage-a.ts` — сценарии ступени A и её ворота.
- `tests/chem-*.test.ts` — тесты по задачам.

---

### Task 1: Атомы, виды, поатомная геометрия

**Files:**
- Create: `data/atoms.json`, `data/molecules.json`, `chem/src/species.ts`, `chem/src/geometry.ts`
- Test: `tests/chem-species.test.ts`

**Interfaces:**
- Produces: `parseFormula(f: string): Record<string, number>`; `molarMass(f: string): number`; `loadSpecies(): Record<string, Species>` где `Species = {id: string, formula: string, charge: number, atoms: AtomRef[], bonds: [number, number][], source: string}` и `AtomRef = {element: string, position: [number, number, number]}` в нанометрах; `buildAlkanoicAcid(n: number): Species` — н-алкановая кислота с `n` атомами углерода, зигзаг во всю длину.

- [ ] **Step 1: Написать падающие тесты**

`tests/chem-species.test.ts`:

```ts
import { expect, test } from 'vitest'
import { buildAlkanoicAcid, loadSpecies, molarMass, parseFormula } from '../chem/src/species'

const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

test('формула разбирается и даёт молярную массу', () => {
  expect(parseFormula('H2O')).toEqual({ H: 2, O: 1 })
  expect(parseFormula('C10H20O2')).toEqual({ C: 10, H: 20, O: 2 })
  expect(molarMass('H2O')).toBeCloseTo(18.015, 2)
  expect(molarMass('CO')).toBeCloseTo(28.010, 2)
})

test('геометрия воды совпадает со справочными значениями', () => {
  const w = loadSpecies()['H2O']
  const o = w.atoms.find((a) => a.element === 'O')!.position
  const hs = w.atoms.filter((a) => a.element === 'H').map((a) => a.position)
  expect(dist(o, hs[0])).toBeCloseTo(0.09572, 5)
  expect(dist(o, hs[1])).toBeCloseTo(0.09572, 5)
  const cos = ((hs[0][0] - o[0]) * (hs[1][0] - o[0]) + (hs[0][1] - o[1]) * (hs[1][1] - o[1]) + (hs[0][2] - o[2]) * (hs[1][2] - o[2])) / (0.09572 * 0.09572)
  expect((Math.acos(cos) * 180) / Math.PI).toBeCloseTo(104.52, 1)
})

test('CO и H2 двухатомны с правильной длиной связи', () => {
  const s = loadSpecies()
  expect(dist(s['CO'].atoms[0].position, s['CO'].atoms[1].position)).toBeCloseTo(0.1128, 5)
  expect(dist(s['H2'].atoms[0].position, s['H2'].atoms[1].position)).toBeCloseTo(0.0741, 5)
})

test('цепь кислоты строится зигзагом с правильными связями и составом', () => {
  const c10 = buildAlkanoicAcid(10)
  expect(parseFormula(c10.formula)).toEqual({ C: 10, H: 20, O: 2 })
  const cs = c10.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  for (let i = 1; i < cs.length; i++) expect(dist(cs[i - 1], cs[i])).toBeCloseTo(0.154, 4)
  const span = dist(cs[0], cs[cs.length - 1])
  expect(span).toBeGreaterThan(0.154 * (cs.length - 1) * 0.7)
  expect(span).toBeLessThan(0.154 * (cs.length - 1))
})

test('заряды видов взяты из данных, а не выдуманы', () => {
  const s = loadSpecies()
  expect(s['HCOO-'].charge).toBe(-1)
  expect(s['H+'].charge).toBe(1)
  expect(s['CO'].charge).toBe(0)
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/chem-species.test.ts`
Expected: FAIL — `chem/src/species` не найден.

- [ ] **Step 3: Реализовать данные атомов**

`data/atoms.json`:

```json
{
  "source": "массы IUPAC 2021; радиусы Ван-дер-Ваальса Bondi 1964; цвета CPK",
  "rank": "A",
  "elements": {
    "H":  { "mass": 1.008,  "vdw": 0.120, "covalent": 0.031, "color": "#ffffff" },
    "C":  { "mass": 12.011, "vdw": 0.170, "covalent": 0.076, "color": "#4d4d4d" },
    "N":  { "mass": 14.007, "vdw": 0.155, "covalent": 0.071, "color": "#3050f8" },
    "O":  { "mass": 15.999, "vdw": 0.152, "covalent": 0.066, "color": "#ff0d0d" },
    "Na": { "mass": 22.990, "vdw": 0.227, "covalent": 0.166, "color": "#ab5cf2" },
    "Cl": { "mass": 35.45,  "vdw": 0.175, "covalent": 0.102, "color": "#1ff01f" }
  }
}
```

Все длины в нанометрах.

- [ ] **Step 4: Реализовать данные видов и построение геометрии**

`data/molecules.json` описывает виды через длины и углы, а не через готовые координаты, чтобы источник был виден:

```json
{
  "rank": "A",
  "bondLengths": {
    "CO_triple": 0.1128, "H2": 0.0741, "OH_water": 0.09572,
    "CO2_double": 0.1160, "CC_single": 0.154, "CH": 0.109,
    "C_O_carboxyl_double": 0.121, "C_O_carboxyl_single": 0.136, "OH_carboxyl": 0.097,
    "C_O_formate": 0.127
  },
  "angles": { "water_HOH": 104.52, "tetrahedral": 109.47, "formate_OCO": 126.0 },
  "sources": {
    "CO_triple": "CRC Handbook, 97th ed.",
    "H2": "CRC Handbook, 97th ed.",
    "OH_water": "Benedict, Gailar & Plyler 1956",
    "CO2_double": "CRC Handbook, 97th ed.",
    "CC_single": "стандартная длина одинарной связи C-C в алканах",
    "CH": "стандартная длина связи C-H в алканах",
    "C_O_carboxyl_double": "стандартная карбонильная связь карбоновых кислот",
    "C_O_carboxyl_single": "стандартная гидроксильная связь карбоновых кислот",
    "OH_carboxyl": "стандартная связь O-H",
    "C_O_formate": "делокализованная связь в формиат-ионе",
    "water_HOH": "Benedict, Gailar & Plyler 1956",
    "tetrahedral": "идеальная sp3-геометрия",
    "formate_OCO": "экспериментальная геометрия формиат-иона"
  },
  "species": [
    { "id": "CO",    "formula": "CO",    "charge": 0,  "kind": "diatomic", "bond": "CO_triple" },
    { "id": "H2",    "formula": "H2",    "charge": 0,  "kind": "diatomic", "bond": "H2" },
    { "id": "H2O",   "formula": "H2O",   "charge": 0,  "kind": "bent",     "bond": "OH_water", "angle": "water_HOH" },
    { "id": "CO2",   "formula": "CO2",   "charge": 0,  "kind": "linear3",  "bond": "CO2_double" },
    { "id": "HCOO-", "formula": "CHO2",  "charge": -1, "kind": "formate" },
    { "id": "H+",    "formula": "H",     "charge": 1,  "kind": "single" },
    { "id": "OH-",   "formula": "HO",    "charge": -1, "kind": "diatomic", "bond": "OH_carboxyl" },
    { "id": "Na+",   "formula": "Na",    "charge": 1,  "kind": "single" },
    { "id": "Cl-",   "formula": "Cl",    "charge": -1, "kind": "single" }
  ]
}
```

`chem/src/species.ts` разбирает формулу регулярным выражением по парам «элемент, число», берёт массы из `atoms.json`, и собирает координаты по `kind`: `single` — один атом в начале координат; `diatomic` — второй атом по оси x на длину связи; `bent` — центральный атом в начале, два соседа под заданным углом в плоскости xy; `linear3` — центральный атом в начале, два соседа по ±x; `formate` — углерод в начале, два кислорода под углом `formate_OCO` на длину `C_O_formate`, водород на C по длине `CH`.

`buildAlkanoicAcid(n)` строит скелет из `n` углеродов зигзагом: углы `tetrahedral`, длины `CC_single`, чередование знака по оси y, так что проекция на ось цепи равна `CC_single·cos(θ/2)` на звено; на первом углероде — карбоксильная группа с `C_O_carboxyl_double`, `C_O_carboxyl_single` и `OH_carboxyl`; остальные углероды дополняются водородами до четырёх связей. Формула собирается из фактического числа атомов, а не задаётся строкой, чтобы состав нельзя было разойтись с геометрией.

- [ ] **Step 5: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/chem-species.test.ts`
Expected: PASS все пять. Если длина цепи вышла больше `0.154·(n−1)` — зигзаг выродился в прямую линию, проверить чередование знака.

- [ ] **Step 6: Коммит**

```bash
git add data/atoms.json data/molecules.json chem/src/species.ts chem/src/geometry.ts tests/chem-species.test.ts
git commit -m "feat: atomic data, species geometry from literature bond lengths"
```

---

### Task 2: Сеть реакций с точными инвариантами

**Files:**
- Create: `data/chemistry.json`, `chem/src/network.ts`
- Modify: `tests/params.test.ts` (расширить сторож констант на `chem/src` и `chem/wgsl`)
- Test: `tests/chem-network.test.ts`

**Interfaces:**
- Consumes: `loadSpecies`, `parseFormula` из задачи 1.
- Produces: `loadNetwork(): Network` где `Network = {reactions: Reaction[]}` и `Reaction = {id: string, kind: 'bulk' | 'adsorb' | 'propagate' | 'terminate', reactants: string[], products: string[], k: number, kUnits: string, rank: 'A'|'B'|'C'|'D', basis: string, contactRadius?: number}`; `checkBalance(r: Reaction): {mass: Record<string, number>, charge: number}` — возвращает разности по элементам и по заряду, нули означают баланс; `assertNetworkBalanced(n: Network): void` — бросает с перечислением нарушений.

- [ ] **Step 1: Написать падающие тесты**

`tests/chem-network.test.ts`:

```ts
import { expect, test } from 'vitest'
import { assertNetworkBalanced, checkBalance, loadNetwork } from '../chem/src/network'

test('вся сеть сбалансирована по массе и заряду', () => {
  expect(() => assertNetworkBalanced(loadNetwork())).not.toThrow()
})

test('несбалансированная реакция выявляется по элементам и заряду', () => {
  const bad = { id: 'bad', kind: 'bulk' as const, reactants: ['CO', 'H2'], products: ['H2O'], k: 1, kUnits: 'nm^3/ns', rank: 'D' as const, basis: 'тест' }
  const d = checkBalance(bad)
  expect(d.mass.C).toBe(1)
  expect(d.charge).toBe(0)

  const charged = { ...bad, id: 'charged', reactants: ['H+'], products: ['H2O'] }
  expect(checkBalance(charged).charge).toBe(1)
})

test('каждая реакция несёт ранг и обоснование, и все скорости синтеза имеют ранг D', () => {
  for (const r of loadNetwork().reactions) {
    expect(['A', 'B', 'C', 'D']).toContain(r.rank)
    expect(r.basis.length).toBeGreaterThan(10)
    if (r.kind !== 'bulk') expect(r.rank).toBe('D')
  }
})

test('каждая реакция объёмного типа имеет контактный радиус', () => {
  for (const r of loadNetwork().reactions) {
    if (r.kind === 'bulk') expect(r.contactRadius).toBeGreaterThan(0)
  }
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/chem-network.test.ts`
Expected: FAIL — `chem/src/network` не найден.

- [ ] **Step 3: Реализовать сеть**

`data/chemistry.json` — минимальная сеть, которой достаточно для роста цепи, и ни одной лишней реакции:

```json
{
  "conditions": { "temperatureK": 448.15, "viscosityPaS": 1.53e-4, "note": "175 C, водный флюид" },
  "reactions": [
    { "id": "adsorb_CO", "kind": "adsorb", "reactants": ["CO"], "products": ["*CO"], "k": 0.5, "kUnits": "1/ns на центр", "rank": "D",
      "basis": "оценка: адсорбция CO на металлическом центре быстрая относительно роста цепи; абсолютная величина неизвестна, задаёт только масштаб kappa_t" },
    { "id": "reduce_CO", "kind": "propagate", "reactants": ["*CO", "H2"], "products": ["*CH2", "H2O"], "k": 0.2, "kUnits": "1/ns на центр", "rank": "D",
      "basis": "оценка: восстановление адсорбированного CO водородом до метиленового звена, суммарная стадия" },
    { "id": "grow", "kind": "propagate", "reactants": ["*Cn", "*CH2"], "products": ["*Cn+1"], "k": 0.15, "kUnits": "1/ns на центр", "rank": "D",
      "basis": "оценка: присоединение метиленового звена к растущей цепи; отношение к обрыву задаёт alpha и проверяется распределением" },
    { "id": "terminate_acid", "kind": "terminate", "reactants": ["*Cn", "H2O"], "products": ["acid_n"], "k": 0.05, "kUnits": "1/ns на центр", "rank": "D",
      "basis": "оценка: обрыв цепи с образованием карбоксильной группы и десорбцией" },
    { "id": "formate_split", "kind": "bulk", "reactants": ["HCOO-", "H+"], "products": ["CO", "H2O"], "k": 1.2, "kUnits": "nm^3/ns", "rank": "D",
      "basis": "оценка: диспропорционирование формиата как источник CO; в этом плане нужна только как поставщик CO", "contactRadius": 0.35 }
  ]
}
```

`chem/src/network.ts` загружает файл, для видов вида `*Cn` и `acid_n` разворачивает шаблон по длине цепи при проверке баланса (звено `*CH2` даёт CH₂, кислота из `n` углеродов — `C_nH_{2n}O_2`), считает разности по элементам и заряду, и бросает с полным перечислением.

Расширить сторож в `tests/params.test.ts`: сканируемые каталоги становятся `['engine/src', 'engine/wgsl', 'chem/src', 'chem/wgsl']`, исключения — `params.ts`, `species.ts`, `network.ts`, `diffusion.ts` (они читают данные и обязаны упоминать имена полей, но не числа: проверить, что и в них нет числовых литералов из запретного списка).

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/chem-network.test.ts tests/params.test.ts`
Expected: PASS. Если баланс не сходится на `grow` — шаблон `*Cn` разворачивается неверно: звено добавляет ровно CH₂.

- [ ] **Step 5: Коммит**

```bash
git add data/chemistry.json chem/src/network.ts tests/chem-network.test.ts tests/params.test.ts
git commit -m "feat: reaction network with exact mass and charge invariants"
```

---

### Task 3: Броуновское движение и закон 6Dt

**Files:**
- Create: `chem/src/diffusion.ts`, `chem/wgsl/bd.wgsl`, `chem/src/sim.ts`
- Modify: `engine/src/index.ts` (экспортировать фасад ступени A)
- Test: `tests/chem-diffusion.test.ts`

**Interfaces:**
- Consumes: `getGpu`, `readBack`, `storageBuffer` из `engine/src/gpu.ts`; `loadSpecies` из задачи 1.
- Produces: `stokesEinstein(radiusNm: number, tempK: number, viscosityPaS: number): number` — коэффициент диффузии в нм²/нс; `smoluchowski(dA: number, dB: number, rContactNm: number): number` — константа в нм³/нс; `createChemSystem(opts: ChemOpts): Promise<ChemSystem>` где `ChemOpts = {box: [number,number,number], counts: Record<string, number>, seed: number, dtNs: number, kappaT: number, contactProbability?: number, wall?: {z: number, sites: number}}` — `contactProbability` по умолчанию 1 и используется задачей 4, `wall` добавляется задачей 5 и без него стенки нет; `ChemSystem.step(n: number): Promise<void>`; `ChemSystem.positions(): Promise<Float32Array>` — по 4 float на молекулу (x, y, z, индекс вида); `ChemSystem.msd(): Promise<number>` — среднеквадратичное смещение с начала прогона в нм²; `ChemSystem.elapsedNs: number`. Фасады в `engine/src/index.ts`: `diffusionOf(speciesId: string): number` — коэффициент диффузии вида в нм²/нс.

- [ ] **Step 1: Написать падающие тесты**

`tests/chem-diffusion.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { smoluchowski, stokesEinstein } from '../chem/src/diffusion'

afterAll(shutdownGpu)

test('Стокс-Эйнштейн даёт коэффициент диффузии нужного порядка при 175 C', () => {
  const d = stokesEinstein(0.19, 448.15, 1.53e-4)
  expect(d).toBeGreaterThan(5)
  expect(d).toBeLessThan(30)
})

test('Смолуховский складывает коэффициенты и линеен по радиусу', () => {
  const k1 = smoluchowski(10, 10, 0.35)
  const k2 = smoluchowski(10, 10, 0.70)
  expect(k2 / k1).toBeCloseTo(2, 6)
  expect(k1).toBeCloseTo(4 * Math.PI * 20 * 0.35, 6)
})

test('среднеквадратичное смещение растёт как 6Dt', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createChemSystem({ box: [40, 40, 40], counts: { CO: 4000 }, seed: 5, dtNs: 1e-4, kappaT: 1 })
    const out: { t: number; msd: number }[] = []
    for (let i = 0; i < 5; i++) { await sys.step(2000); out.push({ t: sys.elapsedNs, msd: await sys.msd() }) }
    return { out, d: api.diffusionOf('CO') }
  })
  for (const p of r.out) expect(p.msd / (6 * r.d * p.t)).toBeCloseTo(1, 1)
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/chem-diffusion.test.ts`
Expected: FAIL — `chem/src/diffusion` не найден.

- [ ] **Step 3: Реализовать диффузию**

`chem/src/diffusion.ts`: `stokesEinstein` считает `kT/(6πηa)` в СИ и переводит в нм²/нс множителем `1e18/1e9`; `smoluchowski` возвращает `4π(D_A+D_B)R`. Радиус вида берётся как максимум расстояния от центра масс до атома плюс радиус Ван-дер-Ваальса этого атома.

`chem/wgsl/bd.wgsl` — шаг сверхдемпфированной динамики: `x += sqrt(2*D*dt)*ξ` покомпонентно, где `ξ` — стандартный нормальный из PCG-хеша по индексу, шагу и семени, `D` берётся из буфера по индексу вида. Периодические границы по всем трём осям. Отдельно ведётся незавёрнутая координата, чтобы среднеквадратичное смещение не обрезалось боксом — именно она идёт в `msd()`.

`chem/src/sim.ts` держит буферы координат (завёрнутых и незавёрнутых), индексов вида, коэффициентов диффузии, и реализует перечисленные методы. `elapsedNs` увеличивается на `dtNs` за шаг и умножается на `kappaT` только при выводе на экран, не в физике.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/chem-diffusion.test.ts`
Expected: PASS все три. Если отношение к `6Dt` систематически равно 1/3 — считается смещение по одной оси вместо трёх; если вдвое меньше — в множителе шума потерян коэффициент 2.

- [ ] **Step 5: Коммит**

```bash
git add chem/src/diffusion.ts chem/wgsl/bd.wgsl chem/src/sim.ts engine/src/index.ts tests/chem-diffusion.test.ts
git commit -m "feat: Brownian dynamics verified against the 6Dt law"
```

---

### Task 4: Реакция при контакте и проверка по Смолуховскому

**Files:**
- Create: `chem/wgsl/react.wgsl`
- Modify: `chem/src/sim.ts` (реакции в шаге, счётчики событий), `engine/src/index.ts`
- Test: `tests/chem-reaction.test.ts`

**Interfaces:**
- Consumes: `ChemSystem` из задачи 3, `loadNetwork` из задачи 2.
- Produces: `ChemSystem.counts(): Promise<Record<string, number>>`; `ChemSystem.events(): Promise<Record<string, number>>` — сколько раз сработала каждая реакция; `ChemSystem.measureSecondOrderRate(reactionId: string): Promise<number>` — наблюдаемая константа в нм³/нс из убыли реагентов. Фасады в `engine/src/index.ts`: `smoluchowskiOf(reactionId: string): number` — предсказанный диффузионный предел для этой реакции; `invariantsOf(sys: ChemSystem): Promise<{elements: Record<string, number>, charge: number}>` — атомы по элементам и суммарный заряд из текущих численностей.

- [ ] **Step 1: Написать падающий тест**

`tests/chem-reaction.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('наблюдаемая скорость реакции при контакте совпадает с пределом Смолуховского', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createChemSystem({
      box: [50, 50, 50], counts: { 'HCOO-': 3000, 'H+': 3000 }, seed: 11, dtNs: 5e-5, kappaT: 1, contactProbability: 1,
    })
    await sys.step(20000)
    return { measured: await sys.measureSecondOrderRate('formate_split'), predicted: api.smoluchowskiOf('formate_split'), events: await sys.events() }
  })
  expect(r.events['formate_split']).toBeGreaterThan(200)
  expect(r.measured / r.predicted).toBeGreaterThan(0.75)
  expect(r.measured / r.predicted).toBeLessThan(1.25)
})

test('число атомов каждого элемента и суммарный заряд сохраняются', async () => {
  const page = await gpuPage()
  const inv = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createChemSystem({ box: [40, 40, 40], counts: { 'HCOO-': 2000, 'H+': 2000 }, seed: 3, dtNs: 5e-5, kappaT: 1, contactProbability: 1 })
    const before = await api.invariantsOf(sys)
    await sys.step(20000)
    const after = await api.invariantsOf(sys)
    return { before, after }
  })
  expect(inv.after.elements).toEqual(inv.before.elements)
  expect(inv.after.charge).toBe(inv.before.charge)
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/chem-reaction.test.ts`
Expected: FAIL — `api.createChemSystem` не принимает `contactProbability`, `measureSecondOrderRate` отсутствует.

- [ ] **Step 3: Реализовать реакции**

`chem/wgsl/react.wgsl`: сетка ячеек со стороной не меньше максимального контактного радиуса сети, обход соседних ячеек, и для каждой пары подходящих видов на расстоянии меньше контактного радиуса — реакция с вероятностью `contactProbability`. Чтобы одна молекула не реагировала дважды за шаг, каждая пара разрешается атомарным захватом обоих участников (`atomicCompareExchangeWeak` по флагу занятости), а проигравшие ждут следующего шага. Продукты записываются на место реагентов: изменение вида — это смена индекса вида в буфере, лишние частицы помечаются мёртвыми и не участвуют дальше.

`measureSecondOrderRate` берёт число событий `N`, объём `V`, время `t` и средние численности `n_A`, `n_B`: `k = N·V/(n_A·n_B·t)`. При `contactProbability = 1` эта величина обязана совпасть с `4πD_AB R` в пределах, заданных тестом.

`invariantsOf(sys)` пересчитывает атомы по элементам и суммарный заряд из текущих численностей видов и их формул.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/chem-reaction.test.ts`
Expected: PASS оба. Если наблюдаемая скорость систематически ниже предела — шаг слишком велик и пары успевают пролетать сквозь контактный радиус: уменьшить `dtNs` так, чтобы `sqrt(6·D·dt)` было заметно меньше контактного радиуса, и записать выбранное значение в отчёт.

- [ ] **Step 5: Коммит**

```bash
git add chem/wgsl/react.wgsl chem/src/sim.ts engine/src/index.ts tests/chem-reaction.test.ts
git commit -m "feat: contact reactions verified against the Smoluchowski limit"
```

---

### Task 5: Каталитическая стенка и рост цепи

**Files:**
- Create: `chem/wgsl/surface.wgsl`, `chem/src/asf.ts`
- Modify: `chem/src/sim.ts`, `engine/src/index.ts`
- Test: `tests/chem-growth.test.ts`

**Interfaces:**
- Consumes: `ChemSystem`, `loadNetwork`.
- Produces: `ChemSystem.sites(): Promise<{index: number, occupant: string, chainLength: number}[]>`; `ChemSystem.chainHistogram(): Promise<Record<number, number>>` — сколько молекул кислоты каждой длины получено; `alphaFromRates(net: Network): number` — `k_grow/(k_grow + k_terminate)`; `alphaFromHistogram(h: Record<number, number>): {alpha: number, r2: number}` — оценка по наклону `ln N_n` против `n`. Фасад в `engine/src/index.ts`: `longestAcid(sys: ChemSystem): Promise<{carbons: number, atoms: AtomRef[]}>` — самая длинная полученная кислота с поатомной геометрией из `buildAlkanoicAcid`.

- [ ] **Step 1: Написать падающие тесты**

`tests/chem-growth.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { alphaFromHistogram, alphaFromRates } from '../chem/src/asf'
import { loadNetwork } from '../chem/src/network'

afterAll(shutdownGpu)

test('оценка alpha по гистограмме восстанавливает заложенное значение', () => {
  const alpha = 0.72
  const h: Record<number, number> = {}
  for (let n = 1; n <= 20; n++) h[n] = Math.round(1e6 * (1 - alpha) * Math.pow(alpha, n - 1))
  const got = alphaFromHistogram(h)
  expect(got.alpha).toBeCloseTo(alpha, 2)
  expect(got.r2).toBeGreaterThan(0.99)
})

test('на стенке растут цепи, и распределение длин предсказано отношением скоростей', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createChemSystem({
      box: [30, 30, 30], counts: { CO: 6000, H2: 6000 }, seed: 17, dtNs: 1e-4, kappaT: 1,
      wall: { z: 0, sites: 256 },
    })
    await sys.step(400000)
    return { hist: await sys.chainHistogram(), events: await sys.events() }
  })
  expect(r.events['grow']).toBeGreaterThan(500)
  expect(Object.keys(r.hist).length).toBeGreaterThan(4)
  const measured = alphaFromHistogram(r.hist)
  const predicted = alphaFromRates(loadNetwork())
  expect(measured.r2).toBeGreaterThan(0.9)
  expect(Math.abs(measured.alpha - predicted)).toBeLessThan(0.1)
}, 1_800_000)

test('в кадре есть готовая молекула кислоты с правильным составом', async () => {
  const page = await gpuPage()
  const acid = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createChemSystem({ box: [30, 30, 30], counts: { CO: 6000, H2: 6000 }, seed: 23, dtNs: 1e-4, kappaT: 1, wall: { z: 0, sites: 256 } })
    await sys.step(400000)
    return api.longestAcid(sys)
  })
  expect(acid.carbons).toBeGreaterThanOrEqual(4)
  expect(acid.atoms.filter((a: any) => a.element === 'O').length).toBe(2)
  expect(acid.atoms.filter((a: any) => a.element === 'H').length).toBe(2 * acid.carbons)
}, 1_800_000)
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/chem-growth.test.ts`
Expected: FAIL — `chem/src/asf` не найден.

- [ ] **Step 3: Реализовать стенку и рост**

`chem/wgsl/surface.wgsl`: активные центры — массив фиксированных позиций в плоскости `z = wall.z`, у каждого поле занятости и длина цепи. Молекула, оказавшаяся ближе контактного радиуса к свободному центру, адсорбируется по `adsorb_CO`. Занятый центр с `*CO` и налетевшим `H2` превращается в `*CH2` по `reduce_CO`, выпуская `H2O` в объём. Центр с цепью и соседний центр с `*CH2` объединяются по `grow`, увеличивая длину цепи на единицу. Центр с цепью и налетевшая `H2O` дают по `terminate_acid` молекулу кислоты, которая уходит в объём как отдельная частица с длиной цепи в качестве параметра вида, а центр освобождается.

`chem/src/asf.ts`: `alphaFromRates` берёт отношение из сети; `alphaFromHistogram` делает линейную регрессию `ln N_n` по `n`, возвращает `alpha = exp(наклон)` и коэффициент детерминации. Первую точку `n = 1` исключить, если она искажена стадией зарождения, и сказать об этом в отчёте.

`longestAcid(sys)` находит самую длинную полученную кислоту, строит её поатомную геометрию через `buildAlkanoicAcid` и возвращает состав.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/chem-growth.test.ts`
Expected: PASS все три. Если цепи не растут дальше двух звеньев — либо центры заняты `*CO`, который не восстанавливается (мало `H2` в объёме), либо `grow` требует соседства центров, а они расставлены слишком редко: записать замеренные числа событий по каждой реакции и решать по ним, а не наугад.

- [ ] **Step 5: Коммит**

```bash
git add chem/wgsl/surface.wgsl chem/src/asf.ts chem/src/sim.ts engine/src/index.ts tests/chem-growth.test.ts
git commit -m "feat: catalytic wall chain growth with ASF distribution predicted from rate ratio"
```

---

### Task 6: Поатомный рендер

**Files:**
- Create: `viewer/atoms.ts`, `viewer/stage-a.html`
- Modify: `viewer/main.ts` (переключатель ступеней)
- Test: `tests/chem-viewer.test.ts`

**Interfaces:**
- Consumes: `ChemSystem`, `loadSpecies`, `data/atoms.json`.
- Produces: страница `/viewer/stage-a.html`, публикующая `window.stageA = {frames: number, molecules: number, atoms: number, bonds: number, unitsBadge: string, kappaT: number, speciesLegend: string[]}`.

- [ ] **Step 1: Написать падающий тест**

`tests/chem-viewer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('поатомная сцена рисует атомы и связи и подписывает единицы', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/stage-a.html', page.url()).href, { waitUntil: 'load' })
  await page.waitForFunction('window.stageA && window.stageA.frames > 5')
  const s = await page.evaluate(() => ({ ...(window as any).stageA }))
  expect(s.atoms).toBeGreaterThan(s.molecules)
  expect(s.bonds).toBeGreaterThan(0)
  expect(s.unitsBadge).toContain('нм')
  expect(s.unitsBadge).toContain('нс')
  expect(s.unitsBadge.toLowerCase()).toContain('атом')
  expect(s.speciesLegend.length).toBeGreaterThan(3)
  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)
})
```

- [ ] **Step 2: Запустить и убедиться, что тест падает**

Run: `npx vitest run tests/chem-viewer.test.ts`
Expected: FAIL — страница отсутствует.

- [ ] **Step 3: Реализовать рендер**

`viewer/atoms.ts`: два `InstancedMesh` — сферы для атомов с радиусом `vdw` из `atoms.json` (масштабированным на общий коэффициент, чтобы стержни были видны) и цилиндры для связей по топологии вида; цвет берётся из `atoms.json`. Позиция атома — позиция молекулы плюс её внутренняя координата, повёрнутая на ориентацию молекулы (ориентацию хранить кватернионом и вращать броуновски тем же кернелом, что и трансляцию, отдельным вращательным коэффициентом диффузии). Легенда перечисляет присутствующие виды с формулами.

Бейдж единиц обязан говорить: масштаб нанометры и наносекунды, **это настоящие атомы, показанные поатомно**, множитель времени `κ_t` числом, и что ступень A не соединена по времени со ступенью C. Видно без наведения.

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

Run: `npx vitest run tests/chem-viewer.test.ts`
Expected: PASS.

- [ ] **Step 5: Коммит**

```bash
git add viewer/atoms.ts viewer/stage-a.html viewer/main.ts tests/chem-viewer.test.ts
git commit -m "feat: per-atom ball-and-stick viewer for stage A"
```

---

### Task 7: Ворота ступени A в отчёте

**Files:**
- Create: `verify/stage-a.ts`
- Modify: `data/literature.json`, `verify/run.ts`, `verify/report.ts`
- Test: `tests/chem-gates.test.ts`

**Interfaces:**
- Consumes: всё предыдущее; `evaluateGates`, `renderReport` из ступени C.
- Produces: записи ворот `diffusion-6dt`, `smoluchowski-rate`, `asf-alpha`, `mass-charge-conservation` в `data/literature.json`; сценарии в `verify/stage-a.ts`, пишущие метрики в `verify/out/gates.json`.

- [ ] **Step 1: Написать падающие тесты**

`tests/chem-gates.test.ts`:

```ts
import { expect, test } from 'vitest'
import { readFileSync } from 'node:fs'
import { evaluateGates } from '../verify/gates'

test('ворота ступени A объявлены с коридорами и рангами', () => {
  const lit = JSON.parse(readFileSync('data/literature.json', 'utf8'))
  const ids = lit.gates.map((g: any) => g.id)
  for (const id of ['diffusion-6dt', 'smoluchowski-rate', 'asf-alpha', 'mass-charge-conservation']) {
    expect(ids).toContain(id)
  }
  const asf = lit.gates.find((g: any) => g.id === 'asf-alpha')
  expect(asf.rank).toBe('D')
})

test('инвариант массы и заряда проходит только при точном нуле', () => {
  const ok = evaluateGates({ 'massChargeDrift': 0 })
  expect(ok.find((g) => g.id === 'mass-charge-conservation')!.verdict).toBe('passed')
  const bad = evaluateGates({ 'massChargeDrift': 1 })
  expect(bad.find((g) => g.id === 'mass-charge-conservation')!.verdict).toBe('failed')
})

test('ворота alpha ранга D выходят недоказанными при любом значении', () => {
  const r = evaluateGates({ asfAlphaGap: 0.001 })
  expect(r.find((g) => g.id === 'asf-alpha')!.verdict).toBe('unproven')
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/chem-gates.test.ts`
Expected: FAIL — записи ворот отсутствуют.

- [ ] **Step 3: Реализовать ворота**

Добавить в `data/literature.json`:

```json
{
  "id": "diffusion-6dt", "title": "Броуновское смещение", "metric": "msdOver6Dt", "unit": "-",
  "target": { "min": 0.9, "max": 1.1 }, "rank": "A",
  "source": "закон Эйнштейна для трёхмерной диффузии, <r^2> = 6Dt",
  "conditions": "4000 молекул CO, 175 C, вязкость 1.53e-4 Pa s, D по Стоксу-Эйнштейну"
},
{
  "id": "smoluchowski-rate", "title": "Скорость реакции при контакте", "metric": "kOverSmoluchowski", "unit": "-",
  "target": { "min": 0.75, "max": 1.25 }, "rank": "A",
  "source": "диффузионный предел Смолуховского, k = 4 pi D_AB R",
  "conditions": "вероятность реакции при контакте 1, шаг подобран так, чтобы смещение за шаг было меньше контактного радиуса"
},
{
  "id": "asf-alpha", "title": "Распределение длин цепей", "metric": "asfAlphaGap", "unit": "-",
  "target": { "max": 0.1 }, "rank": "D",
  "source": "Anderson-Schulz-Flory: alpha предсказана отношением скоростей продолжения и обрыва",
  "conditions": "константы скорости ранга D (оценки); ворота публикуются недоказанными до повышения ранга слоями thermo и qm"
},
{
  "id": "mass-charge-conservation", "title": "Сохранение массы и заряда", "metric": "massChargeDrift", "unit": "-",
  "target": { "max": 0 }, "rank": "A",
  "source": "точный инвариант: атомы каждого элемента и суммарный заряд",
  "conditions": "прогон с реакциями, сверка численностей до и после"
}
```

`verify/stage-a.ts` прогоняет четыре сценария и отдаёт `msdOver6Dt`, `kOverSmoluchowski`, `asfAlphaGap`, `massChargeDrift`, а также `kappaT` и число событий по каждой реакции. `verify/run.ts` вызывает их и вносит в тот же `gates.json` с общим `runId`. `verify/report.ts` печатает раздел ступени A: единицы, `κ_t` числом, таблицу событий, гистограмму длин цепей и оговорку, что все скорости синтеза имеют ранг D.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/chem-gates.test.ts`
Expected: PASS все три.

- [ ] **Step 5: Прогнать экзамен целиком**

Run: `npm run verify`
Expected: в отчёте появились четыре ворот ступени A; `diffusion-6dt`, `smoluchowski-rate` и `mass-charge-conservation` пройдены, `asf-alpha` недоказаны по ранту D. Отдать пользователю полный путь к `verify/out/report.html`.

- [ ] **Step 6: Коммит**

```bash
git add verify/stage-a.ts data/literature.json verify/run.ts verify/report.ts tests/chem-gates.test.ts
git commit -m "feat: stage A gates in the verification report"
```

---

### Task 8: Мембрана из настоящих молекул (обратное отображение)

Выполняется ВНЕ очереди, сразу после задачи 1, потому что опирается только на неё и на готовую ступень C.

**Files:**
- Create: `chem/src/backmap.ts`, `viewer/molecular.ts`, `viewer/molecular.html`
- Test: `tests/chem-backmap.test.ts`

**Interfaces:**
- Consumes: `buildAlkanoicAcid`, `loadSpecies` из задачи 1; `createSystem`, `System.positions()` из ступени C (`engine/src/sim.ts`); `data/atoms.json`.
- Produces: `backmapLipid(head: [number,number,number], tail1: [number,number,number], tail2: [number,number,number], carbons: number, sigmaNm: number): {atoms: {element: string, position: [number,number,number]}[], bonds: [number,number][]}` — разворачивает три бида в поатомную молекулу кислоты, ориентируя её ось по направлению голова→хвост и укладывая зигзаг в плоскости, содержащей эту ось; `backmapSystem(positions: Float32Array, opts: {carbons: number, sigmaNm: number}): {atoms: …, bonds: …}` — то же для всей мембраны; страница `/viewer/molecular.html`, публикующая `window.molecular = {frames: number, molecules: number, atoms: number, bonds: number, reconstructionBadge: string}`.

- [ ] **Step 1: Написать падающие тесты**

`tests/chem-backmap.test.ts`:

```ts
import { expect, test } from 'vitest'
import { backmapLipid } from '../chem/src/backmap'

const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

test('развёрнутая молекула сохраняет состав и длины связей', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  expect(cs.length).toBe(12)
  expect(m.atoms.filter((a) => a.element === 'O').length).toBe(2)
  for (let i = 1; i < cs.length; i++) expect(d(cs[i - 1], cs[i])).toBeCloseTo(0.154, 4)
  expect(m.bonds.length).toBeGreaterThan(cs.length)
})

test('ось молекулы совпадает с направлением голова-хвост', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  const axis = [cs[cs.length - 1][0] - cs[0][0], cs[cs.length - 1][1] - cs[0][1], cs[cs.length - 1][2] - cs[0][2]]
  const len = Math.hypot(...axis)
  expect(Math.abs(axis[2] / len)).toBeGreaterThan(0.9)
})

test('карбоксильная группа сидит на головном конце, а не на хвостовом', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const os = m.atoms.filter((a) => a.element === 'O').map((a) => a.position)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  for (const o of os) expect(d(o, cs[0])).toBeLessThan(d(o, cs[cs.length - 1]))
})

test('масштаб бида в нанометры задаётся явно и меняет размер молекулы', () => {
  const a = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const b = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 16, 0.8)
  const span = (m: typeof a) => {
    const cs = m.atoms.filter((x) => x.element === 'C').map((x) => x.position)
    return d(cs[0], cs[cs.length - 1])
  }
  expect(span(b)).toBeGreaterThan(span(a))
})
```

- [ ] **Step 2: Запустить и убедиться, что тесты падают**

Run: `npx vitest run tests/chem-backmap.test.ts`
Expected: FAIL — `chem/src/backmap` не найден.

- [ ] **Step 3: Реализовать обратное отображение**

`chem/src/backmap.ts` строит молекулу через `buildAlkanoicAcid(carbons)` в её собственной системе координат, затем поворачивает так, чтобы ось цепи легла на направление от головного бида к последнему хвостовому, и переносит так, чтобы карбоксильный углерод оказался в позиции головного бида, переведённой в нанометры множителем `sigmaNm`. Число углеродов и `sigmaNm` — параметры вызова, не константы в коде.

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `npx vitest run tests/chem-backmap.test.ts`
Expected: PASS все четыре.

- [ ] **Step 5: Написать падающий тест сцены**

`tests/chem-backmap.test.ts` дополняется проверкой страницы: открыть `/viewer/molecular.html`, дождаться `window.molecular.frames > 5`, убедиться что `atoms > molecules * 10`, `bonds > atoms`, и что `reconstructionBadge` содержит слова о восстановленной атомной детализации.

- [ ] **Step 6: Реализовать сцену**

`viewer/molecular.ts` берёт координаты бидов из живой системы ступени C, разворачивает каждый липид в молекулу и рисует шар-стержень: сферы с радиусами Ван-дер-Ваальса и цветами CPK из `data/atoms.json`, цилиндры по связям. Чтобы кадр оставался живым, при большом числе липидов разворачивать поатомно только те, что попали в срез или в окно вокруг камеры, а остальные рисовать бидами — и говорить об этом на экране числом: сколько молекул показано поатомно из общего числа.

Бейдж обязан сказать прямо: положения тяжёлого скелета взяты из проверенной огрублённой динамики, атомная детализация **восстановлена по справочной геометрии**, а не досчитана независимо; и что это та же мембрана, чьи площадь на липид и толщина сверены с литературой.

- [ ] **Step 7: Коммит**

```bash
git add chem/src/backmap.ts viewer/molecular.ts viewer/molecular.html tests/chem-backmap.test.ts
git commit -m "feat: atomistic backmapping of the coarse-grained membrane"
```

---

## Что план сознательно не делает

Термодинамика (`thermo/` на pyCHNOSZ, ворота 1), эталонные решатели (`ref/` с ОДУ и Gillespie, ворота 3), квантовые барьеры (`qm/`, повышение рангов до B), ворота агрегации (CMC и pH, ворота 4 и 5) и шов со ступенью C — предмет отдельных планов. Этот план заканчивается тем, что в кадре есть настоящие атомы, они движутся по проверенному закону, реагируют с проверенной скоростью, и из них на стенке вырастает молекула алкановой кислоты.
