# From soup to vesicle: one continuous run, implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One run with no seams: the scene starts as a mixture of simple molecules, bonds form by rules, amphiphiles emerge from them, the amphiphiles self-assemble into a bilayer and close into a vesicle, and all of it is checked by gates, including an honest "no" where vesicles do not appear.

**Architecture:** One coarse-grained scale for the whole run. A particle is either a monomer carrier (a carbon unit, an oxygen head, a hydrogen donor), a catalytic center, or a finished chain unit. Bonds are created and broken by stochastic rules with detailed balance, so the equilibrium is set by the energy difference, not by arbitrary choice. Non-bonded interactions are taken from the validated stage C model, so the assembled bilayer must reproduce its measured area per lipid and thickness. Closure is determined by an outside fill, the same detector already validated.

**Tech Stack:** TypeScript, WGSL (WebGPU), three.js, Vitest, puppeteer-core with the system Chrome. Reuses `engine/wgsl/forces.wgsl`, `engine/wgsl/neighbor.wgsl`, `engine/src/closure.ts`, `chem/src/species.ts`, `chem/src/backmap.ts`, and the `verify/` machinery.

**Spec:** `docs/superpowers/specs/2026-08-15-protocell-genesis-design.md`, section 1a (amendment on the continuous run), sections 3 and 8

## Global Constraints

- **One run, one time scale.** No quantity is carried between separate simulations; the scene runs continuously from soup to vesicle. The single time multiplier `κ_t` is output as a number to the screen and to the report.
- Non-bonded interactions use the Cooke & Deserno 2005 parameters from `data/params.json`, unchanged: `b_hh = b_ht = 0.95σ`, `b_tt = σ`, `w_c = 1.6σ`, `kT/ε = 1.1`, WCA cut off at `2^(1/6)b`, attraction `−ε cos²`. The assembled bilayer must fall within the measured corridors: area per lipid **1.1–1.5 σ²**, thickness **4–6 σ**.
- Bonds formed by rules use the same FENE bond and the same bending potential already validated analytically in stage C. No new potentials are introduced.
- **Reaction rates carry rank D**: no literature constants exist for early-Earth conditions. Any gate that depends only on them is published as unproven. Reverse rates are tied to forward rates by detailed balance `k_f/k_r = exp(−ΔU/kT)`, so the equilibrium is correct even when time is unknown.
- **Exact invariants:** the number of monomers of each kind is conserved, the number of bonds changes only through formation and breaking events, total charge is conserved. A violation fails the test, not a warning.
- No chemical constants baked into code: rules and rates live in `data/soup.json`. The guard in `tests/params.test.ts` is extended to cover `soup/`.
- Results are read via `page.evaluate`; `--dump-dom` is forbidden.
- Long runs go in the background of the tool with a log, waited on via `Monitor` in the same turn. One run per call. Do not keep the dev server running during a GPU batch: measured that contention kills the vitest worker rather than failing the test.

---

## File structure

- `data/soup.json`: monomer kinds, bonding and breaking rules with ranks, starting composition, parameter window for the phase map.
- `soup/src/rules.ts`: rule loading, balance checks, detailed balance, per-step probabilities.
- `soup/src/sim.ts`: run state: particles, bonds, catalytic centers, event counters, invariants.
- `soup/wgsl/bond.wgsl`: bond formation and breaking over the neighbor grid with atomic capture of participants.
- `soup/wgsl/step.wgsl`: Langevin integrator over bonded and non-bonded forces (reuses functions from `engine/wgsl/forces.wgsl`).
- `soup/src/amphiphile.ts`: amphiphile recognition in the bond graph: a chain with a polar end, its length, and a length histogram.
- `soup/src/stages.ts`: determination of run stages from measurements: monomers, amphiphiles, micelles, bilayer, closed vesicle.
- `viewer/soup.html`, `viewer/soup.ts`: continuous-run scene: per-atom view via `chem/src/backmap.ts` for recognized amphiphiles, a stage scale, counters, a `κ_t` badge and rank labels.
- `verify/soup.ts`: continuous-run scenarios and gates, including the phase map.
- `tests/soup-*.test.ts`: tests per task.

---

### Task 1: Bonding rules and invariants

**Files:**
- Create: `data/soup.json`, `soup/src/rules.ts`
- Modify: `tests/params.test.ts` (guard for `soup/src`, `soup/wgsl`)
- Test: `tests/soup-rules.test.ts`

**Interfaces:**
- Produces: `loadSoup(): Soup` where `Soup = {monomers: Monomer[], rules: Rule[], start: Record<string, number>, sweep: Sweep}`, `Monomer = {id: string, kind: 'carbon' | 'head' | 'donor' | 'catalyst', radiusSigma: number, polar: boolean}`, `Rule = {id: string, kind: 'bond' | 'break', a: string, b: string, requiresCatalyst: boolean, energyKT: number, attemptRate: number, rank: 'A'|'B'|'C'|'D', basis: string}`; `forwardBackwardRatio(r: Rule): number` returns `exp(−energyKT)`; `assertRulesConsistent(s: Soup): void` throws if a formation rule has no matching break rule, if the rank is not D for synthesis rates, or if the basis text is empty.

- [ ] **Step 1: Write failing tests**

`tests/soup-rules.test.ts`:

```ts
import { expect, test } from 'vitest'
import { assertRulesConsistent, forwardBackwardRatio, loadSoup } from '../soup/src/rules'

test('every formation rule has a matching break rule and detailed balance', () => {
  const s = loadSoup()
  expect(() => assertRulesConsistent(s)).not.toThrow()
  for (const r of s.rules.filter((x) => x.kind === 'bond')) {
    const back = s.rules.find((x) => x.kind === 'break' && x.a === r.a && x.b === r.b)
    expect(back).toBeDefined()
    expect(forwardBackwardRatio(r)).toBeCloseTo(Math.exp(-r.energyKT), 10)
  }
})

test('all synthesis rates are declared rank D with a basis', () => {
  for (const r of loadSoup().rules) {
    expect(r.rank).toBe('D')
    expect(r.basis.length).toBeGreaterThan(10)
  }
})

test('the starting composition contains only monomers and no finished amphiphiles', () => {
  const s = loadSoup()
  const ids = new Set(s.monomers.map((m) => m.id))
  for (const k of Object.keys(s.start)) expect(ids.has(k)).toBe(true)
  expect(Object.keys(s.start).length).toBeGreaterThan(2)
})

test('an inconsistent rule set is detected', () => {
  const s = loadSoup()
  const broken = { ...s, rules: s.rules.filter((r) => r.kind !== 'break') }
  expect(() => assertRulesConsistent(broken)).toThrow(/break/)
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/soup-rules.test.ts`
Expected: FAIL, `soup/src/rules` not found.

- [ ] **Step 3: Implement the rules**

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
      "basis": "estimate: carbon chain growth occurs only at the catalytic center; a depth of 6 kT sets the chain's resistance to thermal breakage" },
    { "id": "cc_break", "kind": "break", "a": "C", "b": "C", "requiresCatalyst": false, "energyKT": 6.0, "attemptRate": 0.05, "rank": "D",
      "basis": "paired break for cc_bond, linked by detailed balance" },
    { "id": "co_bond",  "kind": "bond",  "a": "C", "b": "O", "requiresCatalyst": false, "energyKT": 8.0, "attemptRate": 0.05, "rank": "D",
      "basis": "estimate: attachment of the polar head to the chain end, deeper than the C-C bond, so the head does not fall off first" },
    { "id": "co_break", "kind": "break", "a": "C", "b": "O", "requiresCatalyst": false, "energyKT": 8.0, "attemptRate": 0.05, "rank": "D",
      "basis": "paired break for co_bond, linked by detailed balance" }
  ],
  "sweep": { "kT": [0.9, 1.1, 1.3], "carbonDensity": [0.03, 0.06, 0.09], "catalystCount": [50, 200, 800] }
}
```

`soup/src/rules.ts` loads the file, checks the pairing of rules, the ranks, and the bases, and derives the per-step attempt probability as `attemptRate·dt`, with the acceptance probability derived from `energyKT` by the Metropolis criterion, so the forward and reverse directions automatically satisfy detailed balance.

Extend the guard in `tests/params.test.ts`: the directory list becomes `['engine/src', 'engine/wgsl', 'chem/src', 'soup/src', 'soup/wgsl']`.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/soup-rules.test.ts tests/params.test.ts`
Expected: PASS, all five.

- [ ] **Step 5: Commit**

```bash
git add data/soup.json soup/src/rules.ts tests/soup-rules.test.ts tests/params.test.ts
git commit -m "feat: bonding rules with detailed balance and exact invariants"
```

---

### Task 2: Bond formation and breaking in the dynamics

**Files:**
- Create: `soup/wgsl/bond.wgsl`, `soup/wgsl/step.wgsl`, `soup/src/sim.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/soup-bonds.test.ts`

**Interfaces:**
- Consumes: `loadSoup`, probabilities from Task 1; `getGpu`, `readBack`, `storageBuffer`; potential functions from `engine/wgsl/forces.wgsl`; the grid from `engine/wgsl/neighbor.wgsl`.
- Produces: `createSoup(opts: {box: [number,number,number], seed: number, kT: number, start?: Record<string, number>, catalystCount?: number}): Promise<SoupSystem>`; `SoupSystem.step(n: number): Promise<void>`; `SoupSystem.particles(): Promise<Float32Array>`, 4 floats each (x, y, z, kind); `SoupSystem.bonds(): Promise<Uint32Array>`, index pairs; `SoupSystem.events(): Promise<Record<string, number>>`; `SoupSystem.invariants(): Promise<{monomers: Record<string, number>, bonds: number, charge: number}>`.

- [ ] **Step 1: Write failing tests**

`tests/soup-bonds.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('bonds form only at the catalytic center where the rule requires it', async () => {
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

test('the count of each monomer kind and the charge are conserved while reactions run', async () => {
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

test('at high temperature there are fewer bonds than at low temperature: equilibrium is set by energy', async () => {
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

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/soup-bonds.test.ts`
Expected: FAIL, `api.createSoup` is not a function.

- [ ] **Step 3: Implement the bond dynamics**

`soup/wgsl/bond.wgsl`: pairs of matching kinds within the contact radius are found over the neighbor grid. Both participants are captured atomically so that a particle does not gain two bonds in one step beyond its allowed valence (carbon: no more than two chain bonds plus one head; head: one bond). A formation attempt is accepted by the Metropolis criterion from `energyKT`; a breaking attempt runs over existing bonds with the same energy, so the ratio of forward to reverse probability equals `exp(−ΔU/kT)` numerically, not merely by declaration.

`soup/wgsl/step.wgsl`: a Langevin step where forces are the sum of non-bonded interactions (functions from `engine/wgsl/forces.wgsl`, radii by kind, attraction only between non-polar ends) and bonded FENE plus bending for triples along the chain. Polar heads do not participate in tail attraction: it is exactly this asymmetry that makes the product an amphiphile.

`soup/src/sim.ts` holds the buffers for particles, bonds, valences, centers, and counters; implements the listed methods; checks valence as an invariant on every read of the bonds.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/soup-bonds.test.ts`
Expected: PASS, all three. If bonds form without a catalyst, the `requiresCatalyst` rule did not reach the shader; if the monomer count changed, a particle was lost during capture, check the rollback on a failed atomic exchange.

- [ ] **Step 5: Commit**

```bash
git add soup/wgsl/bond.wgsl soup/wgsl/step.wgsl soup/src/sim.ts engine/src/index.ts tests/soup-bonds.test.ts
git commit -m "feat: bond formation and breaking in dynamics, catalyst-gated"
```

---

### Task 3: Amphiphile and stage recognition

**Files:**
- Create: `soup/src/amphiphile.ts`, `soup/src/stages.ts`
- Modify: `engine/src/index.ts`
- Test: `tests/soup-amphiphile.test.ts`

**Interfaces:**
- Consumes: `SoupSystem.particles`, `SoupSystem.bonds`.
- Produces: `findAmphiphiles(particles: Float32Array, bonds: Uint32Array, monomers: Monomer[]): Amphiphile[]` where `Amphiphile = {headIndex: number, chain: number[], length: number}`, a chain of carbons with exactly one polar head at the end; `amphiphileHistogram(a: Amphiphile[]): Record<number, number>`; `detectStage(sys: SoupSystem): Promise<{stage: 'monomers'|'amphiphiles'|'micelles'|'bilayer'|'vesicle', evidence: Record<string, number>}>`, the stage is determined from measurements: the fraction of carbon in amphiphiles (field `amphiphileFraction`), the size of the largest aggregate, the number of peaks in the head density profile, the volume of the unfilled cavity (field `enclosedVolume`). A facade in `engine/src/index.ts`: `stageOf(sys: SoupSystem)`, the same thing, published as `window.api.stageOf` for tests and the scene.

- [ ] **Step 1: Write failing tests**

`tests/soup-amphiphile.test.ts`:

```ts
import { expect, test } from 'vitest'
import { amphiphileHistogram, findAmphiphiles } from '../soup/src/amphiphile'
import { loadSoup } from '../soup/src/rules'

const P = (xs: number[][]) => new Float32Array(xs.flat())

test('a chain with one polar head is recognized as an amphiphile', () => {
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 2,3])
  const a = findAmphiphiles(parts, bonds, loadSoup().monomers)
  expect(a.length).toBe(1)
  expect(a[0].length).toBe(3)
})

test('a chain with no head and a chain with two heads are not counted as amphiphiles', () => {
  const m = loadSoup().monomers
  const noHead = findAmphiphiles(P([[0,0,0,0],[1,0,0,0],[2,0,0,0]]), new Uint32Array([0,1, 1,2]), m)
  expect(noHead.length).toBe(0)
  const twoHeads = findAmphiphiles(P([[0,0,0,1],[1,0,0,0],[2,0,0,0],[3,0,0,1]]), new Uint32Array([0,1, 1,2, 2,3]), m)
  expect(twoHeads.length).toBe(0)
})

test('the length histogram counts chains by number of carbons', () => {
  const m = loadSoup().monomers
  const parts = P([[0,0,0,1],[1,0,0,0],[2,0,0,0], [10,0,0,1],[11,0,0,0]])
  const bonds = new Uint32Array([0,1, 1,2, 3,4])
  expect(amphiphileHistogram(findAmphiphiles(parts, bonds, m))).toEqual({ 1: 1, 2: 1 })
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/soup-amphiphile.test.ts`
Expected: FAIL, `soup/src/amphiphile` not found.

- [ ] **Step 3: Implement the recognition**

`soup/src/amphiphile.ts` walks the bond graph with union-find, and for each component checks that it is a simple chain (all degrees at most two), contains exactly one polar particle, and that this particle is at an end; length is counted as the number of non-polar units. `soup/src/stages.ts` calls `findAmphiphiles`, `largestClusterFraction` from `engine/src/aggregate.ts`, `bilayerPeaks` from `engine/src/metrics.ts`, and `enclosedVolumeCpu` from the facade, and from these four measurements returns a stage with the supporting numbers.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/soup-amphiphile.test.ts`
Expected: PASS, all three.

- [ ] **Step 5: Commit**

```bash
git add soup/src/amphiphile.ts soup/src/stages.ts engine/src/index.ts tests/soup-amphiphile.test.ts
git commit -m "feat: amphiphile recognition and stage detection from measurements"
```

---

### Task 4: Continuous run to the vesicle

**Files:**
- Modify: `soup/src/sim.ts`, `engine/src/index.ts`
- Test: `tests/soup-vesicle.test.ts`

**Interfaces:**
- Produces: `SoupSystem.runUntil(stage: string, opts: {maxSteps: number, sampleEvery: number}): Promise<{reached: boolean, steps: number, trace: {steps: number, stage: string, evidence: Record<string, number>}[]}>`.

- [ ] **Step 1: Write a failing test**

`tests/soup-vesicle.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('a closed vesicle emerges from soup with no ready-made amphiphiles', async () => {
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

- [ ] **Step 2: Run and confirm the test fails**

Run: `npx vitest run tests/soup-vesicle.test.ts`
Expected: FAIL, `runUntil` is missing.

- [ ] **Step 3: Implement the run and make it pass**

`runUntil` steps in batches of `sampleEvery`, computes the stage after each batch and writes it to the trace, and stops when the target stage is reached or `maxSteps` is exhausted. If the vesicle does not appear, **do not tweak the attraction and do not weaken the threshold**: first look at the trace and decide from the numbers what is missing (carbon for amphiphiles, centers for the synthesis rate, time for closure), then change the composition or the step count in `data/soup.json` and in the test, documenting the choice. If that still does not help, report BLOCKED with the full trace.

Run: `npx vitest run tests/soup-vesicle.test.ts`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add soup/src/sim.ts engine/src/index.ts tests/soup-vesicle.test.ts
git commit -m "feat: continuous run from soup to closed vesicle"
```

---

### Task 5: The continuous-run scene

**Files:**
- Create: `viewer/soup.html`, `viewer/soup.ts`
- Test: `tests/soup-viewer.test.ts`

**Interfaces:**
- Consumes: `createSoup`, `runUntil`, `findAmphiphiles`, `backmapLipid` from `chem/src/backmap.ts`, `data/atoms.json`.
- Produces: page `/viewer/soup.html`, publishing `window.soup = {frames: number, steps: number, stage: string, amphiphiles: number, largestCluster: number, enclosedVolume: number, kappaT: number, badge: string}`.

- [ ] **Step 1: Write a failing test**

`tests/soup-viewer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('the soup scene runs, shows the stage, and labels the ranks', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/soup.html', page.url()).href, { waitUntil: 'load' })
  await page.waitForFunction('window.soup && window.soup.frames > 5 && window.soup.steps > 0')
  const s = await page.evaluate(() => ({ ...(window as any).soup }))
  expect(['monomers', 'amphiphiles', 'micelles', 'bilayer', 'vesicle']).toContain(s.stage)
  expect(s.badge).toContain('κ_t')
  expect(s.badge.toLowerCase()).toContain('rank d')
  expect(typeof s.enclosedVolume).toBe('number')
  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)
})
```

- [ ] **Step 2: Run and confirm the test fails**

Run: `npx vitest run tests/soup-viewer.test.ts`
Expected: FAIL, the page is missing.

- [ ] **Step 3: Implement the scene**

Monomers are drawn as spheres by kind and radius; recognized amphiphiles are expanded per atom via `backmapLipid` for those caught in the slice, the rest simplified; bonds as cylinders. At the top runs a stage scale highlighting the current stage with the supporting numbers: the fraction of carbon in amphiphiles, the size of the largest aggregate, the number of head-density peaks, the cavity volume. The badge states: the time multiplier `κ_t` as a number, reaction rates of rank D, that the atomic detail of amphiphiles is a reconstruction from reference geometry, while the assembly uses the validated potential.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `npx vitest run tests/soup-viewer.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add viewer/soup.html viewer/soup.ts tests/soup-viewer.test.ts
git commit -m "feat: continuous soup-to-vesicle scene with stage evidence on screen"
```

---

### Task 6: Testing the hypothesis with a phase map

**Files:**
- Create: `verify/soup.ts`
- Modify: `data/literature.json`, `verify/run.ts`, `verify/report.ts`
- Test: `tests/soup-gates.test.ts`

**Interfaces:**
- Produces: gates `soup-to-vesicle`, `soup-invariants`, `soup-bilayer-corridors`, `soup-phase-window` in `data/literature.json`; `sweepPhaseMap(): Promise<{kT: number, carbonDensity: number, catalystCount: number, stage: string, enclosedVolume: number}[]>` in `verify/soup.ts`.

- [ ] **Step 1: Write failing tests**

`tests/soup-gates.test.ts`:

```ts
import { expect, test } from 'vitest'
import { readFileSync } from 'node:fs'
import { evaluateGates } from '../verify/gates'

test('the continuous-run gates are declared', () => {
  const ids = JSON.parse(readFileSync('data/literature.json', 'utf8')).gates.map((g: any) => g.id)
  for (const id of ['soup-to-vesicle', 'soup-invariants', 'soup-bilayer-corridors', 'soup-phase-window']) {
    expect(ids).toContain(id)
  }
})

test('the soup invariants gate passes only at exactly zero drift', () => {
  expect(evaluateGates({ soupInvariantDrift: 0 }).find((g) => g.id === 'soup-invariants')!.verdict).toBe('passed')
  expect(evaluateGates({ soupInvariantDrift: 1 }).find((g) => g.id === 'soup-invariants')!.verdict).toBe('failed')
})

test('the vesicle-path gate has rank D and comes out unproven', () => {
  const r = evaluateGates({ soupReachedVesicle: 1 })
  expect(r.find((g) => g.id === 'soup-to-vesicle')!.rank).toBe('D')
  expect(r.find((g) => g.id === 'soup-to-vesicle')!.verdict).toBe('unproven')
})

test('the corridors of the assembled bilayer match the validated model', () => {
  const lit = JSON.parse(readFileSync('data/literature.json', 'utf8'))
  const g = lit.gates.find((x: any) => x.id === 'soup-bilayer-corridors')
  expect(g.target).toEqual({ min: 1.1, max: 1.5 })
  expect(g.rank).toBe('A')
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/soup-gates.test.ts`
Expected: FAIL, gate entries are missing.

- [ ] **Step 3: Implement the gates and the phase map**

Entries in `data/literature.json`: `soup-to-vesicle` (metric `soupReachedVesicle`, rank **D**, source: the primordial vesicle-formation hypothesis, conditions: rank D rates, so the gate is unproven by construction); `soup-invariants` (metric `soupInvariantDrift`, target `max: 0`, rank A, exact invariant); `soup-bilayer-corridors` (metric `soupAreaPerLipid`, target `1.1–1.5`, rank A, the same source, Cooke & Deserno 2005; the bilayer assembled from soup must fall in the same corridor, and this is a real check that the rules produced a real amphiphile); `soup-phase-window` (metric `soupVesicleFraction`, the fraction of grid points where a vesicle appeared, rank D, the substance of the hypothesis test).

`verify/soup.ts` runs to the vesicle, computes the invariants, measures the area per lipid of the assembled bilayer, and sweeps the parameter grid from `data/soup.json`, returning a map of points with their stages. `verify/report.ts` prints the map as a table: where the vesicle appeared, where it stopped at the bilayer, where at micelles, where the material stayed as monomers, with a note that a "no" in part of the map is a result, not a failure.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/soup-gates.test.ts`
Expected: PASS, all four.

- [ ] **Step 5: Run the full exam**

Run: `npm run verify`
Expected: the continuous-run gates and the phase-map table appear in the report. Hand the user the full path to the report.

- [ ] **Step 6: Commit**

```bash
git add verify/soup.ts data/literature.json verify/run.ts verify/report.ts tests/soup-gates.test.ts
git commit -m "feat: hypothesis test by phase map over temperature, density and catalyst"
```

---

### Task 7: Chemistry rigor: Flory and van't Hoff

These two tests are the answer to the requirement "a continuous run, but scientifically sound": they depend on no unknown rate constant, because they check equilibrium and its temperature dependence, not time.

**Files:**
- Create: `soup/src/equilibrium.ts`
- Test: `tests/soup-equilibrium.test.ts`

**Interfaces:**
- Consumes: `loadSoup`, `SoupSystem`, `findAmphiphiles`, `amphiphileHistogram`.
- Produces: `floryPrediction(energyKT: number, monomerVolumeFraction: number, maxLength: number): Record<number, number>`, the equilibrium geometric fraction of chains of each length from the equilibrium constant `K = exp(energyKT)` and the volume fraction; `bondFraction(sys: SoupSystem): Promise<number>`, the fraction of realized bonds out of the maximum possible; `vanHoffSlope(points: {kT: number, bondFraction: number}[]): {slope: number, r2: number}`, the slope of `ln[p/(1−p)]` against `1/kT`.

- [ ] **Step 1: Write failing tests**

`tests/soup-equilibrium.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { floryPrediction, vanHoffSlope } from '../soup/src/equilibrium'

afterAll(shutdownGpu)

test('the Flory prediction is normalized and decreases geometrically', () => {
  const p = floryPrediction(2.0, 0.05, 30)
  const vals = Object.values(p)
  expect(vals.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6)
  for (let n = 2; n <= 10; n++) expect(p[n] / p[n - 1]).toBeCloseTo(p[3] / p[2], 3)
})

test('the van\'t Hoff slope estimator recovers the built-in energy on synthetic data', () => {
  const e = 6.0
  const pts = [0.8, 1.0, 1.2, 1.5, 2.0].map((kT) => {
    const K = Math.exp(e / kT) * 0.02
    return { kT, bondFraction: K / (1 + K) }
  })
  const f = vanHoffSlope(pts)
  expect(f.slope).toBeCloseTo(e, 1)
  expect(f.r2).toBeGreaterThan(0.99)
})

test('the measured length distribution matches Flory with no fitting', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSoup({ box: [30, 30, 30], seed: 29, kT: 1.1, catalystCount: 800 })
    await sys.step(600000)
    return { hist: await api.histogramOf(sys), predicted: api.floryOf(sys) }
  })
  const ns = Object.keys(r.hist).map(Number).filter((n) => n >= 2 && n <= 8)
  const total = ns.reduce((s, n) => s + r.hist[n], 0)
  for (const n of ns) {
    const measured = r.hist[n] / total
    expect(Math.abs(measured - r.predicted[n])).toBeLessThan(0.08)
  }
}, 1_800_000)

test('bond fraction against inverse temperature gives back the built-in energy', async () => {
  const page = await gpuPage()
  const pts = await page.evaluate(async () => {
    const api = (window as any).api
    const out: { kT: number; bondFraction: number }[] = []
    for (const kT of [0.8, 1.1, 1.5, 2.0]) {
      const sys = await api.createSoup({ box: [30, 30, 30], seed: 31, kT, catalystCount: 800 })
      await sys.step(400000)
      out.push({ kT, bondFraction: await api.bondFractionOf(sys) })
    }
    return out
  })
  const f = vanHoffSlope(pts)
  expect(f.r2).toBeGreaterThan(0.9)
  expect(f.slope).toBeGreaterThan(4.5)
  expect(f.slope).toBeLessThan(7.5)
}, 3_600_000)
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/soup-equilibrium.test.ts`
Expected: FAIL, `soup/src/equilibrium` not found.

- [ ] **Step 3: Implement**

`floryPrediction` builds `p_n = (1−x)·x^(n−1)` where `x` is the continuation probability derived from the equilibrium constant and the monomer volume fraction, and normalizes over `maxLength`. `bondFraction` divides the number of bonds by the maximum possible given the valence. `vanHoffSlope` performs a linear regression of `ln[p/(1−p)]` against `1/kT` and returns the slope with the coefficient of determination. The facades `histogramOf`, `floryOf`, `bondFractionOf` are published in `engine/src/index.ts`.

If the measured distribution is systematically wider than predicted, this is NOT a reason to change the tolerance: first check that the run reached equilibrium (the bond fraction stopped growing) and that the valence in the shader matches the one used to compute the prediction.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/soup-equilibrium.test.ts`
Expected: PASS, all four. A van't Hoff slope that differs from 6 kT by a large factor means the Metropolis acceptance is not being applied to both directions.

- [ ] **Step 5: Commit**

```bash
git add soup/src/equilibrium.ts engine/src/index.ts tests/soup-equilibrium.test.ts
git commit -m "feat: Flory distribution and van't Hoff slope as rate-free rigor checks"
```

---

### Task 8: Assembly rigor: the aggregation-threshold law

**Files:**
- Create: `soup/src/cac.ts`
- Modify: `data/literature.json`, `verify/soup.ts`
- Test: `tests/soup-cac.test.ts`

**Interfaces:**
- Produces: `criticalAggregationConcentration(sys: SoupSystem, chainLength: number): Promise<number>`, the concentration of free amphiphiles at which the first stable aggregate appears; `cacSlope(points: {n: number, cac: number}[]): {slope: number, r2: number}`, the slope of `log10 CAC` against chain length.

- [ ] **Step 1: Write a failing test**

`tests/soup-cac.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { cacSlope } from '../soup/src/cac'

afterAll(shutdownGpu)

test('the slope estimator recovers the measured octanoate and decanoate points', () => {
  const f = cacSlope([{ n: 8, cac: 300 }, { n: 10, cac: 86 }])
  expect(f.slope).toBeCloseTo(-(Math.log10(300 / 86) / 2), 3)
})

test('the aggregation threshold drops with chain length at roughly the measured slope', async () => {
  const page = await gpuPage()
  const pts = await page.evaluate(async () => {
    const api = (window as any).api
    const out: { n: number; cac: number }[] = []
    for (const n of [4, 6, 8, 10]) out.push({ n, cac: await api.cacFor(n) })
    return out
  })
  const f = cacSlope(pts)
  expect(f.r2).toBeGreaterThan(0.9)
  expect(Math.abs(f.slope)).toBeGreaterThan(0.15)
  expect(Math.abs(f.slope)).toBeLessThan(0.45)
}, 3_600_000)
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/soup-cac.test.ts`
Expected: FAIL, `soup/src/cac` not found.

- [ ] **Step 3: Implement**

`cacFor(n)` assembles a system of ready-made amphiphiles of length `n` (this is acceptable: here the assembly law is being checked, not the synthesis), raises the concentration in steps, and records the concentration at which the first stable aggregate appears, stability judged by the fraction of the largest cluster holding for a given number of steps. `cacSlope` performs a regression of `log10 CAC` against `n`.

Add gate `cac-slope` to `data/literature.json`: metric `cacSlopeAbs`, corridor `min: 0.15, max: 0.45`, rank **A**, source: the measured CMC of sodium octanoate (300 mM) and decanoate (86 mM), which gives 0.27 per methylene group; conditions: reduced model units, so the slope is compared, not the absolute concentration.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/soup-cac.test.ts`
Expected: PASS, both.

- [ ] **Step 5: Commit**

```bash
git add soup/src/cac.ts data/literature.json verify/soup.ts tests/soup-cac.test.ts
git commit -m "feat: aggregation threshold law checked against measured CMC slope"
```

---

## What this plan deliberately does not do

It does not tie time to seconds: `κ_t` remains the sole multiplier, and no absolute rates are claimed. It does not account for quantum barriers and does not raise the rate rank above D; that belongs to the separate `thermo/` and `qm/` layers. It does not build full-atom dynamics: the atomic detail on screen remains a reconstruction over the coarse-grained coordinates, and is labeled as such on screen.
