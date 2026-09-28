# Stage A: Atoms, Reactions, Chain Growth. Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Real atoms in a three-dimensional scene: molecules diffuse, collide, and react; a carbon chain grows on a catalytic wall, producing an alkanoic acid molecule, with every mechanism verified against an analytical law.

**Architecture:** Reactive Brownian dynamics in WGSL: a particle is a molecule with internal per-atom geometry built from literature bond lengths. Reactions occur on contact with a probability derived from the rate constant through the Smoluchowski diffusion limit. Chain growth happens at discrete active sites on the wall as competition between propagation and termination, so the chain-length distribution emerges as a consequence of the mechanism and is predicted by an independently measured rate ratio. Rendering is per-atom: ball-and-stick, Van der Waals radii, CPK colors.

**Tech Stack:** TypeScript, WGSL (WebGPU), three.js, Vitest, puppeteer-core with the system Chrome. All of Stage C's harness is reused: `engine/src/gpu.ts`, the neighbor-grid template, the test rig through `gpuPage()`, the gate mechanics from `verify/`.

**Spec:** `docs/superpowers/specs/2026-08-15-protocell-genesis-design.md`, section 5 (Stage A) and section 3 (evidence hierarchy)

## Global Constraints

- Stage A's units are **nanometers, nanoseconds, molecules**; they do not match Stage C's reduced units (σ, τ, ε). No quantity is carried over between stages without an explicit conversion, and the screen always shows which units the current stage is using.
- Run conditions: aqueous fluid, **175 °C = 448.15 K**, water viscosity at this temperature **1.53·10⁻⁴ Pa·s**. The carbon source is CO and H₂ (products of formate or oxalate disproportionation).
- Atomic data (rank A, from reference handbooks, values checked by a test): bond lengths CO 1.128 Å, H₂ 0.741 Å, H₂O O-H 0.9572 Å at an angle of 104.52°, CO₂ C=O 1.160 Å linear, C-C 1.54 Å, C-H 1.09 Å, carboxyl C=O 1.21 Å and C-O 1.36 Å, O-H 0.97 Å, tetrahedral angle 109.47°. Van der Waals radii per Bondi: H 1.20, C 1.70, N 1.55, O 1.52 Å.
- **No rate constants exist for early-Earth synthesis reactions under these conditions** (spec, section 2). All of them enter at rank **D**, marked as estimates, and any gate relying solely on them is published as unproven. The rank is raised later by the `thermo/` and `qm/` layers, not in this plan.
- The single global time multiplier `κ_t` is displayed as a number on screen and in the report. No other quantity is scaled silently.
- Diffusion is computed from the Stokes-Einstein relation `D = kT/(6πηa)`; the reaction's diffusion limit follows Smoluchowski's `k = 4πD_AB·R_AB`, where `D_AB = D_A + D_B`.
- Mass balance for each element and charge balance are **exact invariants**. No reaction passes the scheme check if they are violated, and a long run is checked for their conservation.
- No hardcoded chemical constants in the code: everything is read from `data/chemistry.json` and `data/atoms.json`. The guard in `tests/params.test.ts` is extended to the new directories.
- Results from the browser are read only through `page.evaluate`; `--dump-dom` is forbidden (measured: it returns content before the GPU finishes working).
- Long runs are launched in the tool's background mode with output to a log and awaited in the same turn through `Monitor`; the foreground is cut off at 600 s, a child process started with `&` dies at the end of the turn, and chaining long runs in a single call kills the vitest worker.

---

## File Structure

- `data/atoms.json`: elements (mass, Van der Waals radius, CPK color, covalent radius).
- `data/molecules.json`: species (formula, charge, bond topology, lengths and angles, source for each value).
- `data/chemistry.json`: reactions: stoichiometry, type (bulk, adsorption, chain propagation, termination), a rank-D rate constant and its justification, contact radius.
- `chem/src/species.ts`: formula parsing, mass, charge, atom counts per element.
- `chem/src/geometry.ts`: builds per-atom coordinates for a species from bond lengths and angles, including the zigzag of a saturated chain.
- `chem/src/network.ts`: loads `chemistry.json`, checks mass and charge balance, derives contact radii and reaction probabilities.
- `chem/src/diffusion.ts`: Stokes-Einstein, Smoluchowski, converting rate constants into a per-step probability.
- `chem/wgsl/bd.wgsl`: Brownian step with periodic boundaries.
- `chem/wgsl/react.wgsl`: contact reactions via the neighbor grid.
- `chem/wgsl/surface.wgsl`: wall active sites: adsorption, propagation, termination.
- `chem/src/sim.ts`: Stage A state, stepping, event counters, invariants.
- `chem/src/asf.ts`: chain-length distribution, estimating alpha from event counts and from the histogram.
- `viewer/atoms.ts`: per-atom rendering: ball-and-stick, CPK, radii, legend, units badge and `κ_t`.
- `verify/stage-a.ts`: Stage A scenarios and its gates.
- `tests/chem-*.test.ts`: tests per task.

---

### Task 1: Atoms, Species, Per-Atom Geometry

**Files:**
- Create: `data/atoms.json`, `data/molecules.json`, `chem/src/species.ts`, `chem/src/geometry.ts`
- Test: `tests/chem-species.test.ts`

**Interfaces:**
- Produces: `parseFormula(f: string): Record<string, number>`; `molarMass(f: string): number`; `loadSpecies(): Record<string, Species>` where `Species = {id: string, formula: string, charge: number, atoms: AtomRef[], bonds: [number, number][], source: string}` and `AtomRef = {element: string, position: [number, number, number]}` in nanometers; `buildAlkanoicAcid(n: number): Species`, an n-alkanoic acid with `n` carbon atoms, zigzag along its full length.

- [ ] **Step 1: Write failing tests**

`tests/chem-species.test.ts`:

```ts
import { expect, test } from 'vitest'
import { buildAlkanoicAcid, loadSpecies, molarMass, parseFormula } from '../chem/src/species'

const dist = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

test('formula parses and gives the molar mass', () => {
  expect(parseFormula('H2O')).toEqual({ H: 2, O: 1 })
  expect(parseFormula('C10H20O2')).toEqual({ C: 10, H: 20, O: 2 })
  expect(molarMass('H2O')).toBeCloseTo(18.015, 2)
  expect(molarMass('CO')).toBeCloseTo(28.010, 2)
})

test('water geometry matches reference values', () => {
  const w = loadSpecies()['H2O']
  const o = w.atoms.find((a) => a.element === 'O')!.position
  const hs = w.atoms.filter((a) => a.element === 'H').map((a) => a.position)
  expect(dist(o, hs[0])).toBeCloseTo(0.09572, 5)
  expect(dist(o, hs[1])).toBeCloseTo(0.09572, 5)
  const cos = ((hs[0][0] - o[0]) * (hs[1][0] - o[0]) + (hs[0][1] - o[1]) * (hs[1][1] - o[1]) + (hs[0][2] - o[2]) * (hs[1][2] - o[2])) / (0.09572 * 0.09572)
  expect((Math.acos(cos) * 180) / Math.PI).toBeCloseTo(104.52, 1)
})

test('CO and H2 are diatomic with the correct bond length', () => {
  const s = loadSpecies()
  expect(dist(s['CO'].atoms[0].position, s['CO'].atoms[1].position)).toBeCloseTo(0.1128, 5)
  expect(dist(s['H2'].atoms[0].position, s['H2'].atoms[1].position)).toBeCloseTo(0.0741, 5)
})

test('the acid chain is built as a zigzag with correct bonds and composition', () => {
  const c10 = buildAlkanoicAcid(10)
  expect(parseFormula(c10.formula)).toEqual({ C: 10, H: 20, O: 2 })
  const cs = c10.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  for (let i = 1; i < cs.length; i++) expect(dist(cs[i - 1], cs[i])).toBeCloseTo(0.154, 4)
  const span = dist(cs[0], cs[cs.length - 1])
  expect(span).toBeGreaterThan(0.154 * (cs.length - 1) * 0.7)
  expect(span).toBeLessThan(0.154 * (cs.length - 1))
})

test('species charges come from the data, not made up', () => {
  const s = loadSpecies()
  expect(s['HCOO-'].charge).toBe(-1)
  expect(s['H+'].charge).toBe(1)
  expect(s['CO'].charge).toBe(0)
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/chem-species.test.ts`
Expected: FAIL, `chem/src/species` not found.

- [ ] **Step 3: Implement the atom data**

`data/atoms.json`:

```json
{
  "source": "IUPAC 2021 masses; Van der Waals radii Bondi 1964; CPK colors",
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

All lengths are in nanometers.

- [ ] **Step 4: Implement species data and geometry construction**

`data/molecules.json` describes species through lengths and angles rather than ready-made coordinates, so the source of each value stays visible:

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
    "CC_single": "standard C-C single bond length in alkanes",
    "CH": "standard C-H bond length in alkanes",
    "C_O_carboxyl_double": "standard carbonyl bond of carboxylic acids",
    "C_O_carboxyl_single": "standard hydroxyl bond of carboxylic acids",
    "OH_carboxyl": "standard O-H bond",
    "C_O_formate": "delocalized bond in the formate ion",
    "water_HOH": "Benedict, Gailar & Plyler 1956",
    "tetrahedral": "ideal sp3 geometry",
    "formate_OCO": "experimental geometry of the formate ion"
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

`chem/src/species.ts` parses the formula with a regular expression over "element, count" pairs, takes masses from `atoms.json`, and assembles coordinates by `kind`: `single`: one atom at the origin; `diatomic`: a second atom along the x axis at the bond length; `bent`: a central atom at the origin with two neighbors at the given angle in the xy plane; `linear3`: a central atom at the origin with two neighbors along ±x; `formate`: carbon at the origin, two oxygens at the `formate_OCO` angle at the `C_O_formate` length, and a hydrogen on C at the `CH` length.

`buildAlkanoicAcid(n)` builds a skeleton of `n` carbons as a zigzag: `tetrahedral` angles, `CC_single` lengths, alternating sign along the y axis, so the projection onto the chain axis equals `CC_single·cos(θ/2)` per unit; the first carbon carries the carboxyl group with `C_O_carboxyl_double`, `C_O_carboxyl_single`, and `OH_carboxyl`; the remaining carbons are filled out with hydrogens up to four bonds. The formula is assembled from the actual atom count rather than given as a fixed string, so the composition can never diverge from the geometry.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `npx vitest run tests/chem-species.test.ts`
Expected: PASS, all five. If the chain length comes out greater than `0.154·(n-1)`, the zigzag has degenerated into a straight line: check the sign alternation.

- [ ] **Step 6: Commit**

```bash
git add data/atoms.json data/molecules.json chem/src/species.ts chem/src/geometry.ts tests/chem-species.test.ts
git commit -m "feat: atomic data, species geometry from literature bond lengths"
```

---

### Task 2: Reaction Network with Exact Invariants

**Files:**
- Create: `data/chemistry.json`, `chem/src/network.ts`
- Modify: `tests/params.test.ts` (extend the constant guard to `chem/src` and `chem/wgsl`)
- Test: `tests/chem-network.test.ts`

**Interfaces:**
- Consumes: `loadSpecies`, `parseFormula` from Task 1.
- Produces: `loadNetwork(): Network` where `Network = {reactions: Reaction[]}` and `Reaction = {id: string, kind: 'bulk' | 'adsorb' | 'propagate' | 'terminate', reactants: string[], products: string[], k: number, kUnits: string, rank: 'A'|'B'|'C'|'D', basis: string, contactRadius?: number}`; `checkBalance(r: Reaction): {mass: Record<string, number>, charge: number}`, which returns the differences per element and per charge, zeros meaning balanced; `assertNetworkBalanced(n: Network): void`, which throws with a list of the violations.

- [ ] **Step 1: Write failing tests**

`tests/chem-network.test.ts`:

```ts
import { expect, test } from 'vitest'
import { assertNetworkBalanced, checkBalance, loadNetwork } from '../chem/src/network'

test('the whole network is balanced in mass and charge', () => {
  expect(() => assertNetworkBalanced(loadNetwork())).not.toThrow()
})

test('an unbalanced reaction is detected by element and by charge', () => {
  const bad = { id: 'bad', kind: 'bulk' as const, reactants: ['CO', 'H2'], products: ['H2O'], k: 1, kUnits: 'nm^3/ns', rank: 'D' as const, basis: 'test' }
  const d = checkBalance(bad)
  expect(d.mass.C).toBe(1)
  expect(d.charge).toBe(0)

  const charged = { ...bad, id: 'charged', reactants: ['H+'], products: ['H2O'] }
  expect(checkBalance(charged).charge).toBe(1)
})

test('every reaction carries a rank and a justification, and all synthesis rates have rank D', () => {
  for (const r of loadNetwork().reactions) {
    expect(['A', 'B', 'C', 'D']).toContain(r.rank)
    expect(r.basis.length).toBeGreaterThan(10)
    if (r.kind !== 'bulk') expect(r.rank).toBe('D')
  }
})

test('every bulk-type reaction has a contact radius', () => {
  for (const r of loadNetwork().reactions) {
    if (r.kind === 'bulk') expect(r.contactRadius).toBeGreaterThan(0)
  }
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/chem-network.test.ts`
Expected: FAIL, `chem/src/network` not found.

- [ ] **Step 3: Implement the network**

`data/chemistry.json`: the minimal network sufficient for chain growth, with no extra reaction:

```json
{
  "conditions": { "temperatureK": 448.15, "viscosityPaS": 1.53e-4, "note": "175 C, aqueous fluid" },
  "reactions": [
    { "id": "adsorb_CO", "kind": "adsorb", "reactants": ["CO"], "products": ["*CO"], "k": 0.5, "kUnits": "1/ns per site", "rank": "D",
      "basis": "estimate: CO adsorption on the metallic site is fast relative to chain growth; the absolute value is unknown, it only sets the scale of kappa_t" },
    { "id": "reduce_CO", "kind": "propagate", "reactants": ["*CO", "H2"], "products": ["*CH2", "H2O"], "k": 0.2, "kUnits": "1/ns per site", "rank": "D",
      "basis": "estimate: reduction of adsorbed CO by hydrogen to a methylene unit, a lumped step" },
    { "id": "grow", "kind": "propagate", "reactants": ["*Cn", "*CH2"], "products": ["*Cn+1"], "k": 0.15, "kUnits": "1/ns per site", "rank": "D",
      "basis": "estimate: addition of a methylene unit to the growing chain; the ratio to termination sets alpha and is checked against the distribution" },
    { "id": "terminate_acid", "kind": "terminate", "reactants": ["*Cn", "H2O"], "products": ["acid_n"], "k": 0.05, "kUnits": "1/ns per site", "rank": "D",
      "basis": "estimate: chain termination forming the carboxyl group with desorption" },
    { "id": "formate_split", "kind": "bulk", "reactants": ["HCOO-", "H+"], "products": ["CO", "H2O"], "k": 1.2, "kUnits": "nm^3/ns", "rank": "D",
      "basis": "estimate: formate disproportionation as a CO source; in this plan it is needed only as a CO supplier", "contactRadius": 0.35 }
  ]
}
```

`chem/src/network.ts` loads the file, and for species of the form `*Cn` and `acid_n` expands the template by chain length when checking balance (the `*CH2` unit gives CH₂, an acid with `n` carbons gives `C_nH_{2n}O_2`), computes the differences per element and charge, and throws with the full list.

Extend the guard in `tests/params.test.ts`: the scanned directories become `['engine/src', 'engine/wgsl', 'chem/src', 'chem/wgsl']`, with exceptions for `params.ts`, `species.ts`, `network.ts`, `diffusion.ts` (they read data and are allowed to mention field names, but not numbers: check that they too contain no numeric literals from the forbidden list).

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/chem-network.test.ts tests/params.test.ts`
Expected: PASS. If the balance does not close on `grow`, the `*Cn` template is expanding incorrectly: the unit must add exactly CH₂.

- [ ] **Step 5: Commit**

```bash
git add data/chemistry.json chem/src/network.ts tests/chem-network.test.ts tests/params.test.ts
git commit -m "feat: reaction network with exact mass and charge invariants"
```

---

### Task 3: Brownian Motion and the 6Dt Law

**Files:**
- Create: `chem/src/diffusion.ts`, `chem/wgsl/bd.wgsl`, `chem/src/sim.ts`
- Modify: `engine/src/index.ts` (export the Stage A facade)
- Test: `tests/chem-diffusion.test.ts`

**Interfaces:**
- Consumes: `getGpu`, `readBack`, `storageBuffer` from `engine/src/gpu.ts`; `loadSpecies` from Task 1.
- Produces: `stokesEinstein(radiusNm: number, tempK: number, viscosityPaS: number): number`, the diffusion coefficient in nm²/ns; `smoluchowski(dA: number, dB: number, rContactNm: number): number`, the constant in nm³/ns; `createChemSystem(opts: ChemOpts): Promise<ChemSystem>` where `ChemOpts = {box: [number,number,number], counts: Record<string, number>, seed: number, dtNs: number, kappaT: number, contactProbability?: number, wall?: {z: number, sites: number}}`; `contactProbability` defaults to 1 and is used by Task 4, `wall` is added by Task 5 and there is no wall without it; `ChemSystem.step(n: number): Promise<void>`; `ChemSystem.positions(): Promise<Float32Array>`, 4 floats per molecule (x, y, z, species index); `ChemSystem.msd(): Promise<number>`, the mean-squared displacement since the start of the run, in nm²; `ChemSystem.elapsedNs: number`. Facades in `engine/src/index.ts`: `diffusionOf(speciesId: string): number`, the species' diffusion coefficient in nm²/ns.

- [ ] **Step 1: Write failing tests**

`tests/chem-diffusion.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { smoluchowski, stokesEinstein } from '../chem/src/diffusion'

afterAll(shutdownGpu)

test('Stokes-Einstein gives a diffusion coefficient of the right order at 175 C', () => {
  const d = stokesEinstein(0.19, 448.15, 1.53e-4)
  expect(d).toBeGreaterThan(5)
  expect(d).toBeLessThan(30)
})

test('Smoluchowski adds the coefficients and is linear in radius', () => {
  const k1 = smoluchowski(10, 10, 0.35)
  const k2 = smoluchowski(10, 10, 0.70)
  expect(k2 / k1).toBeCloseTo(2, 6)
  expect(k1).toBeCloseTo(4 * Math.PI * 20 * 0.35, 6)
})

test('mean-squared displacement grows as 6Dt', async () => {
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

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/chem-diffusion.test.ts`
Expected: FAIL, `chem/src/diffusion` not found.

- [ ] **Step 3: Implement diffusion**

`chem/src/diffusion.ts`: `stokesEinstein` computes `kT/(6πηa)` in SI units and converts to nm²/ns with the factor `1e18/1e9`; `smoluchowski` returns `4π(D_A+D_B)R`. A species' radius is taken as the maximum distance from the center of mass to an atom, plus that atom's Van der Waals radius.

`chem/wgsl/bd.wgsl`: an overdamped-dynamics step: `x += sqrt(2*D*dt)*ξ` component-wise, where `ξ` is a standard normal from a PCG hash over the index, step, and seed, and `D` is taken from a buffer by species index. Periodic boundaries on all three axes. An unwrapped coordinate is tracked separately so the mean-squared displacement is not clipped by the box; it is this coordinate that feeds `msd()`.

`chem/src/sim.ts` holds the coordinate buffers (wrapped and unwrapped), species indices, diffusion coefficients, and implements the listed methods. `elapsedNs` increases by `dtNs` per step and is multiplied by `kappaT` only for on-screen display, not in the physics.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/chem-diffusion.test.ts`
Expected: PASS, all three. If the ratio to `6Dt` is systematically 1/3, the displacement is being computed along one axis instead of three; if it is half as much, the noise-factor coefficient of 2 has been lost.

- [ ] **Step 5: Commit**

```bash
git add chem/src/diffusion.ts chem/wgsl/bd.wgsl chem/src/sim.ts engine/src/index.ts tests/chem-diffusion.test.ts
git commit -m "feat: Brownian dynamics verified against the 6Dt law"
```

---

### Task 4: Contact Reaction and Verification Against Smoluchowski

**Files:**
- Create: `chem/wgsl/react.wgsl`
- Modify: `chem/src/sim.ts` (reactions in the step, event counters), `engine/src/index.ts`
- Test: `tests/chem-reaction.test.ts`

**Interfaces:**
- Consumes: `ChemSystem` from Task 3, `loadNetwork` from Task 2.
- Produces: `ChemSystem.counts(): Promise<Record<string, number>>`; `ChemSystem.events(): Promise<Record<string, number>>`, how many times each reaction fired; `ChemSystem.measureSecondOrderRate(reactionId: string): Promise<number>`, the observed constant in nm³/ns from reactant depletion. Facades in `engine/src/index.ts`: `smoluchowskiOf(reactionId: string): number`, the predicted diffusion limit for that reaction; `invariantsOf(sys: ChemSystem): Promise<{elements: Record<string, number>, charge: number}>`, atoms per element and total charge from the current counts.

- [ ] **Step 1: Write a failing test**

`tests/chem-reaction.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('the observed contact-reaction rate matches the Smoluchowski limit', async () => {
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

test('the atom count of each element and the total charge are conserved', async () => {
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

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/chem-reaction.test.ts`
Expected: FAIL, `api.createChemSystem` does not accept `contactProbability`, `measureSecondOrderRate` is missing.

- [ ] **Step 3: Implement reactions**

`chem/wgsl/react.wgsl`: a cell grid with a side no smaller than the network's largest contact radius, traversal of neighboring cells, and for every eligible species pair within the contact radius, a reaction with probability `contactProbability`. To keep one molecule from reacting twice in a step, each pair is resolved by an atomic capture of both participants (`atomicCompareExchangeWeak` on a busy flag), and the losers wait for the next step. Products are written in place of the reactants: a species change is a change of the species index in the buffer, and surplus particles are marked dead and take no further part.

`measureSecondOrderRate` takes the event count `N`, volume `V`, time `t`, and average populations `n_A`, `n_B`: `k = N·V/(n_A·n_B·t)`. At `contactProbability = 1` this value must match `4πD_AB R` within the bounds set by the test.

`invariantsOf(sys)` recomputes atoms per element and the total charge from the current species counts and their formulas.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/chem-reaction.test.ts`
Expected: PASS, both. If the observed rate is systematically below the limit, the step is too large and pairs manage to fly past the contact radius: reduce `dtNs` so that `sqrt(6·D·dt)` is noticeably smaller than the contact radius, and record the chosen value in the report.

- [ ] **Step 5: Commit**

```bash
git add chem/wgsl/react.wgsl chem/src/sim.ts engine/src/index.ts tests/chem-reaction.test.ts
git commit -m "feat: contact reactions verified against the Smoluchowski limit"
```

---

### Task 5: Catalytic Wall and Chain Growth

**Files:**
- Create: `chem/wgsl/surface.wgsl`, `chem/src/asf.ts`
- Modify: `chem/src/sim.ts`, `engine/src/index.ts`
- Test: `tests/chem-growth.test.ts`

**Interfaces:**
- Consumes: `ChemSystem`, `loadNetwork`.
- Produces: `ChemSystem.sites(): Promise<{index: number, occupant: string, chainLength: number}[]>`; `ChemSystem.chainHistogram(): Promise<Record<number, number>>`, how many acid molecules of each length were obtained; `alphaFromRates(net: Network): number`, `k_grow/(k_grow + k_terminate)`; `alphaFromHistogram(h: Record<number, number>): {alpha: number, r2: number}`, an estimate from the slope of `ln N_n` versus `n`. Facade in `engine/src/index.ts`: `longestAcid(sys: ChemSystem): Promise<{carbons: number, atoms: AtomRef[]}>`, the longest acid obtained, with per-atom geometry from `buildAlkanoicAcid`.

- [ ] **Step 1: Write failing tests**

`tests/chem-growth.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { alphaFromHistogram, alphaFromRates } from '../chem/src/asf'
import { loadNetwork } from '../chem/src/network'

afterAll(shutdownGpu)

test('the alpha estimate from the histogram recovers the seeded value', () => {
  const alpha = 0.72
  const h: Record<number, number> = {}
  for (let n = 1; n <= 20; n++) h[n] = Math.round(1e6 * (1 - alpha) * Math.pow(alpha, n - 1))
  const got = alphaFromHistogram(h)
  expect(got.alpha).toBeCloseTo(alpha, 2)
  expect(got.r2).toBeGreaterThan(0.99)
})

test('chains grow on the wall, and the length distribution is predicted by the rate ratio', async () => {
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

test('the frame contains a finished acid molecule with the correct composition', async () => {
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

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/chem-growth.test.ts`
Expected: FAIL, `chem/src/asf` not found.

- [ ] **Step 3: Implement the wall and growth**

`chem/wgsl/surface.wgsl`: active sites are an array of fixed positions in the plane `z = wall.z`, each with an occupancy field and a chain length. A molecule that ends up closer than the contact radius to a free site adsorbs via `adsorb_CO`. A site occupied by `*CO` that is hit by `H2` turns into `*CH2` via `reduce_CO`, releasing `H2O` into the bulk. A site holding a chain and a neighboring site holding `*CH2` merge via `grow`, increasing the chain length by one. A site holding a chain hit by `H2O` gives, via `terminate_acid`, an acid molecule that leaves into the bulk as a separate particle with the chain length as a species parameter, and the site is freed.

`chem/src/asf.ts`: `alphaFromRates` takes the ratio from the network; `alphaFromHistogram` performs a linear regression of `ln N_n` against `n`, returning `alpha = exp(slope)` and the coefficient of determination. Exclude the first point `n = 1` if it is skewed by the nucleation stage, and say so in the report.

`longestAcid(sys)` finds the longest acid obtained, builds its per-atom geometry via `buildAlkanoicAcid`, and returns the composition.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/chem-growth.test.ts`
Expected: PASS, all three. If chains do not grow past two units, either the sites are stuck occupied by `*CO` that never gets reduced (too little `H2` in the bulk), or `grow` requires adjacent sites that are spaced too far apart: record the measured event counts for each reaction and decide from them, not by guessing.

- [ ] **Step 5: Commit**

```bash
git add chem/wgsl/surface.wgsl chem/src/asf.ts chem/src/sim.ts engine/src/index.ts tests/chem-growth.test.ts
git commit -m "feat: catalytic wall chain growth with ASF distribution predicted from rate ratio"
```

---

### Task 6: Per-Atom Rendering

**Files:**
- Create: `viewer/atoms.ts`, `viewer/stage-a.html`
- Modify: `viewer/main.ts` (stage switcher)
- Test: `tests/chem-viewer.test.ts`

**Interfaces:**
- Consumes: `ChemSystem`, `loadSpecies`, `data/atoms.json`.
- Produces: the page `/viewer/stage-a.html`, which publishes `window.stageA = {frames: number, molecules: number, atoms: number, bonds: number, unitsBadge: string, kappaT: number, speciesLegend: string[]}`.

- [ ] **Step 1: Write a failing test**

`tests/chem-viewer.test.ts`:

```ts
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('the per-atom scene draws atoms and bonds and labels units', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/stage-a.html', page.url()).href, { waitUntil: 'load' })
  await page.waitForFunction('window.stageA && window.stageA.frames > 5')
  const s = await page.evaluate(() => ({ ...(window as any).stageA }))
  expect(s.atoms).toBeGreaterThan(s.molecules)
  expect(s.bonds).toBeGreaterThan(0)
  expect(s.unitsBadge).toContain('nm')
  expect(s.unitsBadge).toContain('ns')
  expect(s.unitsBadge.toLowerCase()).toContain('atom')
  expect(s.speciesLegend.length).toBeGreaterThan(3)
  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)
})
```

- [ ] **Step 2: Run and confirm the test fails**

Run: `npx vitest run tests/chem-viewer.test.ts`
Expected: FAIL, the page does not exist.

- [ ] **Step 3: Implement the rendering**

`viewer/atoms.ts`: two `InstancedMesh` objects: spheres for atoms with the `vdw` radius from `atoms.json` (scaled by a shared factor so the sticks stay visible) and cylinders for bonds by species topology; color comes from `atoms.json`. An atom's position is the molecule's position plus its internal coordinate, rotated by the molecule's orientation (store the orientation as a quaternion and rotate it via Brownian motion with the same kernel as the translation, using a separate rotational diffusion coefficient). The legend lists the species present with their formulas.

The units badge must state: the scale is nanometers and nanoseconds, **these are real atoms, shown per-atom**, the time multiplier `κ_t` as a number, and that Stage A is not linked in time to Stage C. Visible without hovering.

- [ ] **Step 4: Run the test and confirm it passes**

Run: `npx vitest run tests/chem-viewer.test.ts`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add viewer/atoms.ts viewer/stage-a.html viewer/main.ts tests/chem-viewer.test.ts
git commit -m "feat: per-atom ball-and-stick viewer for stage A"
```

---

### Task 7: Stage A Gates in the Report

**Files:**
- Create: `verify/stage-a.ts`
- Modify: `data/literature.json`, `verify/run.ts`, `verify/report.ts`
- Test: `tests/chem-gates.test.ts`

**Interfaces:**
- Consumes: everything above; `evaluateGates`, `renderReport` from Stage C.
- Produces: gate entries `diffusion-6dt`, `smoluchowski-rate`, `asf-alpha`, `mass-charge-conservation` in `data/literature.json`; scenarios in `verify/stage-a.ts` that write metrics to `verify/out/gates.json`.

- [ ] **Step 1: Write failing tests**

`tests/chem-gates.test.ts`:

```ts
import { expect, test } from 'vitest'
import { readFileSync } from 'node:fs'
import { evaluateGates } from '../verify/gates'

test('Stage A gates are declared with corridors and ranks', () => {
  const lit = JSON.parse(readFileSync('data/literature.json', 'utf8'))
  const ids = lit.gates.map((g: any) => g.id)
  for (const id of ['diffusion-6dt', 'smoluchowski-rate', 'asf-alpha', 'mass-charge-conservation']) {
    expect(ids).toContain(id)
  }
  const asf = lit.gates.find((g: any) => g.id === 'asf-alpha')
  expect(asf.rank).toBe('D')
})

test('the mass and charge invariant passes only at exactly zero', () => {
  const ok = evaluateGates({ 'massChargeDrift': 0 })
  expect(ok.find((g) => g.id === 'mass-charge-conservation')!.verdict).toBe('passed')
  const bad = evaluateGates({ 'massChargeDrift': 1 })
  expect(bad.find((g) => g.id === 'mass-charge-conservation')!.verdict).toBe('failed')
})

test('the rank-D alpha gate comes out unproven regardless of value', () => {
  const r = evaluateGates({ asfAlphaGap: 0.001 })
  expect(r.find((g) => g.id === 'asf-alpha')!.verdict).toBe('unproven')
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/chem-gates.test.ts`
Expected: FAIL, the gate entries do not exist.

- [ ] **Step 3: Implement the gates**

Add to `data/literature.json`:

```json
{
  "id": "diffusion-6dt", "title": "Brownian displacement", "metric": "msdOver6Dt", "unit": "-",
  "target": { "min": 0.9, "max": 1.1 }, "rank": "A",
  "source": "Einstein's law for three-dimensional diffusion, <r^2> = 6Dt",
  "conditions": "4000 CO molecules, 175 C, viscosity 1.53e-4 Pa s, D from Stokes-Einstein"
},
{
  "id": "smoluchowski-rate", "title": "Contact reaction rate", "metric": "kOverSmoluchowski", "unit": "-",
  "target": { "min": 0.75, "max": 1.25 }, "rank": "A",
  "source": "Smoluchowski diffusion limit, k = 4 pi D_AB R",
  "conditions": "contact reaction probability 1, step chosen so the per-step displacement is smaller than the contact radius"
},
{
  "id": "asf-alpha", "title": "Chain length distribution", "metric": "asfAlphaGap", "unit": "-",
  "target": { "max": 0.1 }, "rank": "D",
  "source": "Anderson-Schulz-Flory: alpha predicted by the ratio of propagation and termination rates",
  "conditions": "rank-D rate constants (estimates); the gate is published as unproven until the rank is raised by the thermo and qm layers"
},
{
  "id": "mass-charge-conservation", "title": "Mass and charge conservation", "metric": "massChargeDrift", "unit": "-",
  "target": { "max": 0 }, "rank": "A",
  "source": "exact invariant: atoms of each element and total charge",
  "conditions": "run with reactions, comparing counts before and after"
}
```

`verify/stage-a.ts` runs four scenarios and returns `msdOver6Dt`, `kOverSmoluchowski`, `asfAlphaGap`, `massChargeDrift`, plus `kappaT` and the event count for each reaction. `verify/run.ts` calls them and writes into the same `gates.json` under a shared `runId`. `verify/report.ts` prints the Stage A section: units, `κ_t` as a number, the event table, the chain-length histogram, and a note that all synthesis rates carry rank D.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/chem-gates.test.ts`
Expected: PASS, all three.

- [ ] **Step 5: Run the full exam**

Run: `npm run verify`
Expected: the report shows the four Stage A gates; `diffusion-6dt`, `smoluchowski-rate`, and `mass-charge-conservation` pass, `asf-alpha` is unproven due to rank D. Give the user the full path to `verify/out/report.html`.

- [ ] **Step 6: Commit**

```bash
git add verify/stage-a.ts data/literature.json verify/run.ts verify/report.ts tests/chem-gates.test.ts
git commit -m "feat: stage A gates in the verification report"
```

---

### Task 8: Membrane from Real Molecules (Backmapping)

Carried out out of sequence, right after Task 1, because it depends only on that and on the completed Stage C.

**Files:**
- Create: `chem/src/backmap.ts`, `viewer/molecular.ts`, `viewer/molecular.html`
- Test: `tests/chem-backmap.test.ts`

**Interfaces:**
- Consumes: `buildAlkanoicAcid`, `loadSpecies` from Task 1; `createSystem`, `System.positions()` from Stage C (`engine/src/sim.ts`); `data/atoms.json`.
- Produces: `backmapLipid(head: [number,number,number], tail1: [number,number,number], tail2: [number,number,number], carbons: number, sigmaNm: number): {atoms: {element: string, position: [number,number,number]}[], bonds: [number,number][]}`, which unfolds three beads into a per-atom acid molecule, orienting its axis along the head→tail direction and laying the zigzag in the plane containing that axis; `backmapSystem(positions: Float32Array, opts: {carbons: number, sigmaNm: number}): {atoms: …, bonds: …}`, the same for the whole membrane; the page `/viewer/molecular.html`, which publishes `window.molecular = {frames: number, molecules: number, atoms: number, bonds: number, reconstructionBadge: string}`.

- [ ] **Step 1: Write failing tests**

`tests/chem-backmap.test.ts`:

```ts
import { expect, test } from 'vitest'
import { backmapLipid } from '../chem/src/backmap'

const d = (a: number[], b: number[]) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])

test('the unfolded molecule preserves composition and bond lengths', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  expect(cs.length).toBe(12)
  expect(m.atoms.filter((a) => a.element === 'O').length).toBe(2)
  for (let i = 1; i < cs.length; i++) expect(d(cs[i - 1], cs[i])).toBeCloseTo(0.154, 4)
  expect(m.bonds.length).toBeGreaterThan(cs.length)
})

test('the molecule axis matches the head-tail direction', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  const axis = [cs[cs.length - 1][0] - cs[0][0], cs[cs.length - 1][1] - cs[0][1], cs[cs.length - 1][2] - cs[0][2]]
  const len = Math.hypot(...axis)
  expect(Math.abs(axis[2] / len)).toBeGreaterThan(0.9)
})

test('the carboxyl group sits at the head end, not the tail end', () => {
  const m = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const os = m.atoms.filter((a) => a.element === 'O').map((a) => a.position)
  const cs = m.atoms.filter((a) => a.element === 'C').map((a) => a.position)
  for (const o of os) expect(d(o, cs[0])).toBeLessThan(d(o, cs[cs.length - 1]))
})

test('the bead-to-nanometer scale is given explicitly and changes the molecule size', () => {
  const a = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 12, 0.8)
  const b = backmapLipid([0, 0, 4], [0, 0, 2], [0, 0, 0], 16, 0.8)
  const span = (m: typeof a) => {
    const cs = m.atoms.filter((x) => x.element === 'C').map((x) => x.position)
    return d(cs[0], cs[cs.length - 1])
  }
  expect(span(b)).toBeGreaterThan(span(a))
})
```

- [ ] **Step 2: Run and confirm the tests fail**

Run: `npx vitest run tests/chem-backmap.test.ts`
Expected: FAIL, `chem/src/backmap` not found.

- [ ] **Step 3: Implement the backmapping**

`chem/src/backmap.ts` builds the molecule via `buildAlkanoicAcid(carbons)` in its own coordinate system, then rotates it so the chain axis lies along the direction from the head bead to the last tail bead, and translates it so the carboxyl carbon ends up at the head bead's position, converted to nanometers by the factor `sigmaNm`. The carbon count and `sigmaNm` are call parameters, not constants in the code.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `npx vitest run tests/chem-backmap.test.ts`
Expected: PASS, all four.

- [ ] **Step 5: Write a failing test for the scene**

`tests/chem-backmap.test.ts` is extended with a check of the page: open `/viewer/molecular.html`, wait for `window.molecular.frames > 5`, confirm that `atoms > molecules * 10`, `bonds > atoms`, and that `reconstructionBadge` contains wording about the reconstructed atomic detail.

- [ ] **Step 6: Implement the scene**

`viewer/molecular.ts` takes bead coordinates from Stage C's live system, unfolds each lipid into a molecule, and draws it ball-and-stick: spheres with Van der Waals radii and CPK colors from `data/atoms.json`, cylinders along the bonds. To keep the frame responsive, when there are many lipids, unfold only those that fall inside a slice or a window around the camera per-atom, and draw the rest as beads, stating this on screen as a number: how many molecules are shown per-atom out of the total.

The badge must say plainly: the heavy-skeleton positions come from the verified coarse-grained dynamics, the atomic detail is **reconstructed from reference geometry** rather than independently computed, and that this is the same membrane whose area per lipid and thickness were checked against the literature.

- [ ] **Step 7: Commit**

```bash
git add chem/src/backmap.ts viewer/molecular.ts viewer/molecular.html tests/chem-backmap.test.ts
git commit -m "feat: atomistic backmapping of the coarse-grained membrane"
```

---

## What This Plan Deliberately Does Not Do

Thermodynamics (`thermo/` on pyCHNOSZ, gate 1), reference solvers (`ref/` with ODEs and Gillespie, gate 3), quantum barriers (`qm/`, raising ranks to B), aggregation gates (CMC and pH, gates 4 and 5), and the seam with Stage C are the subject of separate plans. This plan ends with real atoms in the frame, moving according to a verified law, reacting at a verified rate, and growing into an alkanoic acid molecule on the wall.
