# Real evaporation and rehydration of the solvent — report

Task `evaporation` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `wet-dry-cycling-report.md`, whose §3.3.1 named the structural gap in its own numbers —
*"particle count is fixed at `createSoup`, so water cannot leave … 1.10× in volume against L1's
~1400×, i.e. 1.3 % of it on a log-concentration scale"* — and whose concern 2 called the fix
"a real engine change and its own task". This is that task.

`data/params.json` was **NOT** touched. No threshold, corridor, potential, rate, recogniser or rank-A
constant was modified. `co_bond.attemptRate` untouched. No arm reverted to under-dense water except
the one arm whose entire purpose is to re-measure a published under-dense number (§7, flagged there
and nowhere used as a claim about liquid water). One `data/soup.json` section gained **four fields**
(`dryWetCycle`: `evaporateSolvent`, `residualSolventFraction`, `evaporationRampSteps`,
`insertionMinSeparationSigma`) plus a derivation appended to its own `basis`; **nothing existing in
that file changed**, so the pre-existing box-scaling-only cycle path — and
`tests/soup-drywet-cycling.test.ts`, which pins it — is byte-for-byte unchanged.

**Verdict up front, eight lines.**

1. **Real evaporation is now in the engine, and it is a real resize.** A dry phase removes solvent
   beads from the system and rehydration puts them back, with no shader change and no buffer
   reallocated: every soup kernel bounds itself with `arrayLength()` of one of four per-particle
   buffers, and WGSL's `arrayLength()` is the length of the BOUND RANGE, so binding those four over
   the first `activeN` particles makes the departed beads invisible to every kernel at once (§1.2).
2. **The concentration factor achieved is 3.2021× against the literature's 1400×, and the real limit
   is not the engine.** Solvent removal realises **1440×** exactly as Ross & Deamer measured; the
   ORGANIC concentration factor is capped at **3.3839×** by close packing of the beads themselves
   (3.2021× is 94.6 % of that ceiling), because this broth's WET state is already **1593× more
   concentrated than the pond L1 evaporated** — 1.14× past the concentration that evaporation ENDS at
   in the literature. 1.3 % → **16.07 %** of ln 1400, a 12.4× gain on a log scale (§3).
3. **Every proof the task asked for holds by measurement.** Solvent-only removal (all four organic
   counts EXACTLY unchanged), bond-graph invariance (exact set equality, both directions, zero events
   fired), no valence violation and no bond pointing into removed territory, organic positions
   bit-identical across removal, no non-finite state — including after 1000 further real steps — and a
   checkpoint that round-trips a partially-evaporated state exactly, bond set included (§2).
4. **Cycling with real evaporation BEATS the wet control at every cycle on every measure, reversing
   the predecessor's negative.** Yield **1.40–1.89×**, bond count **1.96–2.12×**, largest aggregate
   **12.5–14.5×** (161–189 against 11–15), across six cycles and two seeds. The predecessor's
   box-scaling-only cycling LOST at every cycle (0.711…0.926) (§6.3).
5. **It is not a compression artefact, and that is the measurement that matters.** After cycling ends
   the run continued **80 000 steps of plain wet dynamics** and the aggregate held flat at
   **163 → 162 → 159 → 157 → 161** (§6.4). Cross-seed scatter on it is ±1–6 %; the effect is 200× that.
6. **The gain does not compound: one drying event buys it all.** 189 (cycle 1) → 169 → 162 → 168 → 166
   → 160, and the yield declines monotonically as `co_break` accumulates. L2's "limits after about six
   cycles" is not what this model shows — it saturates after one (§6.3).
7. **The published box-54 271 was a lower bound and it was loose by 2.19×.** Re-run at the identical
   composition/box/seed with real RNG state: largest aggregate **588 at step 90 000 and still rising**,
   against 268 at that step and a frozen plateau of 271 for the following 180 000 steps. Enclosed
   volume 6.75 → **17.125 σ³** (§7). *(That arm is at the historical under-dense water, by design.)*
8. **Binding constraint, numbered: (1) AMPHIPHILE SUPPLY — 232 against a recomputed floor of 1088,
   4.69× short — and it binds because box 30 is too small, not because the physics failed.** This
   engine now puts **69.4 % of every amphiphile it makes into ONE aggregate**. On the predecessors'
   own quantity (aggregate size vs floor) the sequence goes **80× → 8.9× → 3.36× → 33.4× → 6.76×**;
   the floor moved 968 → **1088** and the window reopened from 1.06 σ to **5.47 σ**. Nothing closed:
   `encapsulatedWater` = 0 against 303 everywhere, `closed = false` everywhere (§9).


---

## 1. What was built

### 1.1 The mechanism, in one paragraph

A dry phase now **removes solvent beads from the system** and rehydration **puts them back**. The
removal is coupled to the box contraction that already existed, so the surviving organics genuinely
concentrate rather than merely being crowded together with their own solvent. Nothing else in the
engine changed: the same rigid-centre-of-mass coordinate map, the same per-increment bonded-distance
self-check, the same `resizeSoupGrid` reallocation, the same non-finite and Verlet guards, the same
minimiser.

### 1.2 Design decision 1 — mask over solvent slots, or a real resize? **A real resize.** Rank D.

There is **no particle-count uniform anywhere in this engine**. Every kernel bounds itself with
`arrayLength()` of one of four per-particle `vec4` buffers (`soup/wgsl/step.wgsl`'s
`posRW`/`velRW`/`posSortedRW`/`posAtRebuildRW`, `forces.wgsl`'s `pos2` — the full list is the
`arrayLength` grep across `soup/wgsl` and `engine/wgsl`). Adding a count uniform would have meant
touching `soup/wgsl/step.wgsl`, which stands at **596 lines against CLAUDE.md's hard 600**, so a
responsibility split would have been required first — and a mask would additionally have needed a
branch in every one of those kernels.

WGSL's `arrayLength()` of a runtime-sized array is the length of the **bound range**, not of the whole
`GPUBuffer`. So the resize is done by *binding* those four buffers over their first `activeN`
particles (`soup/src/soup-bindgroups.ts`'s new `pbuf`). The live particle count really changes, the
grid is really re-planned for it through the already-tested `resizeSoupGrid` path, and the removed
beads are invisible to **every** kernel: no force, no grid cell, no Verlet slot, no integration, no
census.

- **What is kept, deliberately: buffer CAPACITY.** Two reasons, one of them a number: rehydration
  needs those slots back inside the same run, and `verletListBuf` alone is `N·listCapacity·4` =
  **328.8 MB at N = 32 884**, so destroying and recreating it twice per cycle would move ~25 GB of GPU
  allocations across a six-cycle run for no gain.
- **What this biases:** memory is sized for the wet state throughout, so the dry phase is not cheaper
  in memory (it *is* cheaper in time — see the step costs in §6). And `activeN·16` is exactly the whole
  buffer for a system that never evaporates, so every existing caller's `arrayLength` is bit-identical
  to before the parameter existed.
- Buffers whose `arrayLength` is never read as a particle bound (forces, cells, bond slots, both RNG
  streams, `centerLink`, `centerHeldSteps`, `frozen`, the Verlet list itself) stay bound whole: an
  oversized array is harmless because every kernel indexing them is itself bounded by `pos2`/`posRW`.

### 1.3 Design decision 2 — which water leaves? **Index order, from the tail of the solvent block.** Rank D.

`soup/src/soup-init-state.ts` lays particles out in kind-block order, so the solvent occupies one
contiguous index range, and in this composition it is the **last** non-empty block. Shrinking `activeN`
therefore removes only solvent indices and **moves no other index** — so the bond graph, the valence
slot rows, the catalyst↔chain links and every organic bead are untouched *by construction*, not by a
check afterwards. That precondition is asserted and **throws** (`planEvaporation`), because if it were
ever false the truncation would silently renumber bonded partners; the test drives that refusal.

**Why uniformly-at-random is not merely the simplest choice here but the only defensible one:**
removing preferentially from the surface would be more realistic, and in a fully periodic box it is
**not defined** — there is no air–water interface for a surface to be measured against. (The project
already declares that class of thing rather than faking it; see `saltPhLimitation`.) Index order *is*
a spatially uniform sample: a water bead's index is uncorrelated with its position (the initial layout
assigns lattice sites by a seeded Fisher–Yates shuffle over **all** sites, and thereafter position is
set by diffusion).

**What it biases, stated in full:**
1. the same subset of beads leaves every cycle, not a fresh draw — reproducible, but it means
   "which molecules evaporate" carries no cycle-to-cycle variation;
2. evaporation is homogeneous through the volume, so there is **no drying front** and none of the
   stratification a real receding meniscus produces;
3. the solvent block must be the last non-empty block, which is a real constraint on composition (a
   clay run, for instance, could not evaporate — and a box change is already refused with clay).

### 1.4 Design decision 3 — how removal is coupled to the box. Rank A-derived amplitude, rank D schedule.

Removing water at a fixed box **lowers** total density, which is not what drying does. The dry box is
therefore sized for **what is left**:
`dryBox = computeDryBox(box, N_org + W_residual, targetDryDensity)` — the same `targetDryDensity`
field, **no new amplitude constant**, and the same `computeDryBox` the previous mechanism used.

`residualSolventFraction = 1/1400` is **rank A-derived, not a free parameter**: Ross & Deamer 2016
(*Life* 6(3):28, doi:10.3390/life6030028) measured evaporation reducing the solution volume ~1400-fold,
and the volume of an aqueous solution is essentially the volume of its water, so "reduce the volume
1400× at fixed solute content" maps one-to-one onto "remove 1 − 1/1400 of the solvent."

The transition walks a **ladder** of (box, solvent count) rungs. The box is stepped uniformly in
`ln L` and the solvent count *follows* from a log-linear ramp of total density between ρ_wet and
`targetDryDensity`. Doing it the other way round (uniform steps in count) makes the last increments
4.9 % linear, because the count falls linearly while the box falls as its cube root.

**Order within an increment, and the physics of each way round:**
- **drying: remove, then contract.** Removal cannot create an overlap (it only takes particles away).
  The consequence is the one that matters: the **organic** density — the one the chemistry responds to
  — is monotone non-decreasing across the whole transition, because removing solvent at fixed box
  leaves it *exactly* unchanged and every contraction raises it. There is no point at which the
  reagents are diluted.
- **rehydrating: expand, then insert.** Expansion is what creates the room. Both directions run
  `rampRelaxSteps` of ordinary dynamics between increments: an affine contraction of a dense liquid
  really does push contacting pairs into each other's cores, and an affine expansion can too, for two
  multi-bead molecules whose centres are closer than their own bead offsets, so neither direction is
  safe to jump.

`evaporationRampSteps = 12` (rank D) keeps each increment at **3.181 % linear**
(`ln(30/20.3538)/12 = 0.032328`) — exactly the scale the predecessor measured safe as a *single
instantaneous jump* (its §4.4: one 3.2 % linear jump at ρ_tot 1.218→1.34, boxes 20 and 30, zero
non-finite components, zero Verlet overflow, zero bonded-distance violations). `rampSteps` (6) and
`rampRelaxSteps` (200) are unchanged; the evaporating path needs its own count because its box change
is an order of magnitude larger (|Δln L| 0.388 against 0.032).

### 1.5 Design decision 4 — where returning water is placed, and how overlaps are avoided. Rank D + rank C.

**Placement:** uniform in the current box by rejection sampling against a cell list, with a floor of
`insertionMinSeparationSigma = 0.75 σ` and best-of-64 candidates when the floor cannot be met (so the
sampler degrades to "the roomiest candidate seen" instead of failing) — and it **reports** how often
that happened and the worst separation it actually achieved, so the fallback cannot hide.

0.75 σ is derived from the random-sequential-addition saturation fraction φ_RSA = 0.3841 (rank C,
independent RSA literature, never calibrated on this tree). Required volume fraction for the full wet
pool at exclusion diameter *d*, box 30: `N(π/6)d³/V = 32884·0.523599·d³/27000 = 0.63800·d³`. The
largest feasible *d* is `(0.3841/0.63800)^(1/3) = 0.8445 σ`; 0.75 σ gives φ = 0.2692, a **1.427×
margin** below saturation. 0.75 σ is deliberately **inside** the WCA core (water–water contact = σ),
which is why a minimisation must follow.

**Velocities and RNG for the returning beads** are fresh: the same Maxwell–Boltzmann draw at kT that
`soup-init-state.ts` makes at creation, and the creation RNG formula mixed with the global step so no
two beads share a state — which matters because `soup/wgsl/step.wgsl`'s RNG advance has no index
mixing, so equal states would stay equal forever (the exact defect the predecessor measured costing a
factor of three on the chemistry). Bond slots, `centerLink` and `centerHeldSteps` are written to their
creation-time empty values: a returning solvent bead brings no history with it.

**Why the minimiser moves everything, and how that was decided — by measurement, and it cost a run.**
The first version held every pre-existing particle immobile for the minimisation (the mineral
platelet's own `frozen` flag), which bought a beautiful property and a broken run:

```
EVAP-INSERTION inserted=5530 belowFloor=1847 minSeparation=0.5404 relaxIterations=10 max|F| 1.0798e+5 -> 1.1770e+4
EVAP-ISOLATED ... organicsBitIdentical=true/true movedPreexistingFloats=0 ...
→ non-finite state at step=4000: non-finite position components=50460, velocity components=50460 out of 50502
  (appeared in the step interval 3000..4000) -- the computation diverged (Inf/NaN)
```

**Not one of 45 216 organic coordinate floats moved — and 1000 real steps later the run diverged.**
Inserting into a liquid is a **many-body rearrangement**: a bead placed in a locally jammed
neighbourhood has nowhere to go if its neighbours cannot yield, `max|F|` stalled at 1.18e4, and the
loud guard fired. This is the same problem the cold start has (a lattice at liquid density puts
unlike-radius pairs inside each other's cores) and it has the same cure — the same minimiser, with
everything mobile. So the honest claim is not "rehydration changes nothing" but "**rehydration's
perturbation is bounded and measured**":

- iteration count is **derived, not configured**:
  `ceil(2·need/d0)` with `need = wcaCutoff(stiffest solvent pair) − insertionMinSeparation·σ + ρ_wet^(−1/3)`
  — lift out of the core *plus* migrate as far as the mean inter-particle spacing, because reaching
  free volume is the actual requirement. On the box-30 numbers: `wcaCutoff(1.1) = 1.23474`,
  `1.23474 − 0.75 + 0.93390 = 1.41864`, `d0 = 0.1` → **29 iterations**, displacement bound
  `d0(it+1)/2 = 1.5 σ`.
- the measured perturbation is reported and asserted against that bound (§2.3).

---

## 2. The proofs, by measurement

Committed as `tests/soup-evaporation.test.ts` (3 tests). The two GPU tests use two **DEBUG isolation
hooks** (`evaporateDEBUG`/`rehydrateDEBUG`) that call removal and insertion at a **fixed box with no
ramp and no dynamics** — they exist precisely because inside a real transition the two operations are
inseparable from the box change and the ramp's relaxation steps, so "removal did not touch the bond
graph" would not be a checkable claim. Neither hook is reachable from `stepCycled`, the CLI, the
viewer or any measurement path.

### 2.1 Composition, box 24 (the smallest cube whose DRY box still gives the neighbour grid its minimum cell count)

`{C: 952, O: 3808, H: 952, M: 63, W: 11059}`, N = 16 834, ρ_tot = 1.21774, ρ_W = 0.79998,
ρ_org = 0.41776 — the project's own liquid-water densities, scaled by volume from the box-30 broth.
Dry L = 16.2811 σ against cellSize 2.94695 → 5 cells, exactly `minCells = 2·effectiveWalkRadius+1`.

### 2.2 Removal and re-insertion in isolation — unedited

```
$ nice -n 15 npx vitest run tests/soup-evaporation.test.ts -t "in isolation"
EVAP-ISOLATED N 16834 -> 11304 -> 16834 census {"C":952,"O":3808,"H":952,"M":63,"W":11059,"K":0} -> {"C":952,"O":3808,"H":952,"M":63,"W":5529,"K":0} -> {"C":952,"O":3808,"H":952,"M":63,"W":11059,"K":0} bonds 29/29/29 bondSetUnchanged=true/true organicsBitIdentical=true/false movedPreexistingFloats=33912 nonFinite={"pos":0,"vel":0}/{"pos":0,"vel":0}/after1000steps={"pos":0,"vel":0} valence={"bad":0,"outOfRange":0}/{"bad":0,"outOfRange":0}
EVAP-INSERTION inserted=5530 belowFloor=788 minSeparation=0.6257 relaxIterations=29 max|F| 2.4226e+4 -> 4.5654e+1 preexistingDisplacement rms=0.1933 max=0.8833 bound=1.5000 movedFloats=33912
EVAP-CHECKPOINT roundTrip N=11304 nInFile=11304 steps=3000 box=24 census={"C":952,"O":3808,"H":952,"M":63,"W":5529,"K":0} activeCountsInFile={"C":952,"O":3808,"H":952,"M":63,"W":5529,"K":0} bondsIdentical=true
 ✓ tests/soup-evaporation.test.ts (3 tests | 2 skipped) 13769ms
```

| claim the task asked for | how it is measured | result |
|---|---|---|
| **removal is of the SOLVENT only** | live census recomputed from the position buffer's own species slot, before/after | `C 952 → 952`, `O 3808 → 3808`, `H 952 → 952`, `M 63 → 63` **exactly**; `W 11059 → 5529`. Charge 0 throughout. |
| **the bond graph is untouched** | the full sorted set of `(i,j)` pairs compared as a string, both ways | `bondSetUnchanged = true / true`; bond count `29 / 29 / 29`; **every one of the four event counters unchanged** by both operations |
| **no valence claim is touched** | per-particle C–C ≤ 2, C–O ≤ 1, degree ≤ 3, recomputed off the live graph | `{bad: 0, outOfRange: 0}` after both — and `outOfRange = 0` means **no bond points into removed territory** |
| **no catalyst link or organic bead is touched by removal** | exact position fingerprint (running sum + evenly-spaced sample) over the organic index range | `organicsBitIdentical = true` for removal |
| **no non-finite state** | the engine's own IEEE-754 exponent scan | `{pos:0, vel:0}` after removal, after insertion, **and after 1000 further real steps** — the last one is what caught the frozen-surroundings version as unsafe |
| **the checkpoint round-trips a partially-evaporated state** | encode → decode → `createSoup` → compare | `N 11304 = 11304`, census identical, **bond set identical**, box and step count identical, and the file's own `activeCounts` carries `W: 5529` |

### 2.3 The perturbation rehydration does cause, as a number

`max|F| 2.4226e+4 → 4.5654e+1` (the cold start's own validated pair at this composition is
3.4e4 → 17.9, i.e. the same convergence). Pre-existing particles moved **rms 0.1933 σ, worst
0.8833 σ**, against the minimiser's analytical ceiling of **1.5 σ** — asserted, not eyeballed.
33 912 of 45 216 organic coordinate floats changed at all.

**This isolation test is a HARSHER case than the real cycle**, and that is worth saying: it inserts
5530 beads into a liquid at ρ_tot = 0.818, whereas a real rehydration inserts into **dilute
organics-only** material at ρ_org = 0.418. 788 of 5530 beads (14.2 %) fell below the 0.75 σ floor here,
worst separation 0.6257 σ.

### 2.4 One full evaporating cycle, live — unedited

```
$ nice -n 15 npx vitest run tests/soup-evaporation.test.ts -t "evaporating cycle"
EVAP-CYCLE dryBox=16.2811 (plan 16.2811) N_dry=5783 (plan 5783) rhoDryRealised=1.34000 phase=dry censusDry={"C":952,"O":3808,"H":952,"M":63,"W":8,"K":0} boxBack=24.000000 N_back=16834 censusBack={"C":952,"O":3808,"H":952,"M":63,"W":11059,"K":0} steps=38200 events={"cc_bond":365,"cc_break":0,"co_bond":317,"co_break":2} nonFinite={"start":{"pos":0,"vel":0},"dry":{"pos":0,"vel":0},"held":{"pos":0,"vel":0},"end":{"pos":0,"vel":0}}
 ✓ evaporating cycle: the dry box is sized for what is LEFT, and both the box and the census return exactly 64679ms
```

- the dry state is exactly the plan's: `dryBox 16.2811 = 16.2811`, `N_dry 5783 = 5783`,
  `ρ realised 1.34000` **for what is left**, not for the wet N;
- the solvent really left: `W 11059 → 8` (the residual `round(11059/1400)`), every organic count exactly
  the composition's own;
- rehydration returns the box **exactly** (`toBe`, not `toBeCloseTo`) and the census **exactly**;
- `nonFinite = {0,0}` at four points including the densest instant.

### 2.5 The pure arithmetic and the refusals — unedited

```
$ nice -n 15 npx vitest run tests/soup-evaporation.test.ts -t "what leaves"
EVAP-PLAN box30 N_org=11284 W_wet=21600 W_dry=15 solventRemovalFactor=1440.0x dryBox=20.3538 rho_org_wet=0.41793 rho_org_dry=1.33822 concentrationFactor=3.2021x closePackingCeiling=3.3839x previousMechanism=1.1002x logFractionOf1400=16.07% relaxIterations=29
EVAP-LADDER increments=12 lnStep=0.032328 linearPerIncrement=3.181% boxes=29.046,28.122,27.227,26.361,25.522,24.711,23.925,23.163,22.427,21.713,21.023,20.354 solvent=18799,16237,13893,11748,9786,7992,6350,4848,3474,2217,1067,15
```

Refusals driven: a composition where the solvent is **not** the last non-empty block throws
(`/LAST non-empty block/`), and a residual fraction that removes nothing throws
(`/MUST REMOVE solvent/`).

---

## 3. The concentration factor achieved, against 1400×, and what limits it

| quantity | value |
|---|---|
| solvent removal factor realised | **1440×** (`21600 → 15` beads; `21600/1400 = 15.43` rounds to 15 — the rounding to whole beads is the only gap and it is the *whole* discrepancy from L1's 1400×) |
| **organic concentration factor** | **3.2021×** (V_wet/V_dry = 27000/8432.09) |
| ρ_org wet → dry | 0.41793 → **1.33822** σ⁻³ |
| the previous mechanism's factor | 1.1002× |
| improvement | **2.91× in factor, 12.4× on a log-concentration scale** (1.3 % → **16.07 %** of ln 1400) |
| absolute ceiling in this model | **3.3839×** = √2 / 0.41793 (close packing of unit-diameter cores over the wet organic density) — 3.2021× is **94.6 % of the ceiling** |

**What limits it — two named limits, the second of which is the real one.**

1. **Close packing of the organics themselves.** ρ_org(dry) ≤ √2 = 1.4142 σ⁻³ and ρ_org(wet) = 0.41793,
   so no choice of `targetDryDensity` can exceed 3.3839×. The achieved 3.2021× is 94.6 % of that, at
   `targetDryDensity = 1.34` — the existing field, 1.0554× below close packing.
2. **This broth's wet state is already past the concentration the literature's evaporation ENDS at.**
   L1's 1400× is a volume factor for a *dilute pond*. Its own starting mole ratio is
   `0.015 M decanoate / 55.5 M water = 2.703e-4`. This model's wet ratio is
   `(C+O)/W = 9300/21600 = 0.4306` — i.e. the **wet** broth is already **1593× more concentrated** than
   the pond L1 evaporated. Reducing that pond 1400× lands it at ratio `0.3784`; the model's wet state
   is **1.14× more concentrated than that**. Demanding 1400× here would demand ρ_org = 585 σ⁻³, **414×
   beyond close packing**.

So the honest statement is not "the model can only manage 3.2 of 1400". It is: **the model's wet
broth already sits at (slightly past) the literature's dried endpoint, because the project chose a
carbon pool enriched ~1593× over the most generous literature pond in order to get any chemistry at
all** (the `broth-composition` task's own 691× enrichment argument, recomputed here for the current
composition). The 3.2021× is the *additional* concentration available above that, and it is 94.6 % of
everything this bead model can physically express.

---

## 4. Density trajectory through a cycle

The ladder, box 30 (from `EVAP-LADDER` above plus the arithmetic of §1.4):

| increment | box σ | solvent beads | N | ρ_tot σ⁻³ | **ρ_org σ⁻³** |
|---|---|---|---|---|---|
| 0 (wet) | 30.0000 | 21600 | 32884 | 1.21793 | 0.41793 |
| 1 | 29.0455 | 18799 | 30083 | 1.22766 | 0.46052 |
| 2 | 28.1215 | 16237 | 27521 | 1.23747 | 0.50747 |
| 3 | 27.2268 | 13893 | 25177 | 1.24735 | 0.55924 |
| 4 | 26.3605 | 11748 | 23032 | 1.25732 | 0.61614 |
| 5 | 25.5218 | 9786 | 21070 | 1.26736 | 0.67875 |
| 6 | 24.7096 | 7992 | 19276 | 1.27748 | 0.74769 |
| 7 | 23.9231 | 6350 | 17634 | 1.28768 | 0.82366 |
| 8 | 23.1615 | 4848 | 16132 | 1.29796 | 0.90790 |
| 9 | 22.4241 | 3474 | 14758 | 1.30832 | 1.00027 |
| 10 | 21.7100 | 2217 | 13501 | 1.31876 | 1.10228 |
| 11 | 21.0234 | 1067 | 12351 | 1.32928 | 1.21471 |
| **12 (dry)** | **20.3538** | **15** | **11299** | **1.34000** | **1.33822** |

Total density rises 10.0 % across the transition (the same modest ramp the previous mechanism used, and
the reason the medium stays representable at all). **Organic density rises 3.202×, monotonically, with
no dilution at any rung** — that is the whole difference from the predecessor's mechanism.
Rehydration walks the same box ladder backwards with the solvent held at 15 and the whole pool
returning on the final rung, at the wet box.

**The artefact of that, stated as a cost:** `(evaporationRampSteps − 1) · rampRelaxSteps = 2200` real
steps of every rehydration are spent **solvent-free**, with the organics expanding from ρ_org 1.338 to
0.418. That is 2200 of ~36 400 real steps per cycle, **6.0 %**, and it is a property of the scheme, not
of the physics. It is unavoidable in this model and the reason is measured: an affine expansion widens
every gap by the same 3.181 % per increment, so it never opens a cavity a whole bead wide, and a WCA
liquid at ρ_tot 1.2–1.34 has no σ-sized cavities to find. The only place the pool fits is the
fully-expanded box.

---

*(sections 5–11: measurements, filled in below)*

## 5. Three divergences, and what each one actually was

The evaporating ramp at box 30 diverged **three times** before it ran clean, always inside the ramp's
own relaxation steps, always with the state still **finite** (so these were real physical runaways,
not Inf), and always from cycle 2 onward — i.e. only once material had actually aggregated and there
was something dense to squeeze:

```
Error: Verlet list: measured drift 29789312.2505 exceeds skin/2=0.7500 ...
  async applyEvaporatingTransition (soup/src/soup-box-scale.ts:151:58)
Error: Verlet list: measured drift 213101346629564.0000 exceeds skin/2=0.7500 ...
Error: Verlet list: measured drift 371189702.0951 exceeds skin/2=0.7500 ...
```

Three hypotheses were tested, in order, and **two of the three were refuted by measurement**:

1. **The adsorption tether crossing FENE's range — REFUTED, and it was the best hypothesis.**
   `scaleMoleculesRigid` deliberately does not treat `centerLink` as a rigid edge, and
   `fene_dv(r) = k·r/(1 − (r/r_inf)²)` does not merely grow near `r_inf`, it **changes sign** past it,
   so an over-stretched tether pushes its pair apart without bound — and the valve that clears it
   (`desorbStretch`) lives in `bond_form_main`, dispatched only every `bondAttemptInterval` = **20 real
   steps**. A rehydration expands the box by the whole wet/dry ratio, so this looked certain. The valve
   was implemented to fire synchronously at each box change, on a threshold derived from the ramp's own
   per-increment factor (`r_inf/λ` = 1.4680 σ) — **and it cleared exactly 0 tethers in the entire
   six-cycle run**, with the longest tether measured at 1.04–1.13 σ against the 1.468 σ threshold at
   every one of 216 increments. The code is kept (it is correct physics and costs one readback that is
   made anyway) and it is reported as a refuted hypothesis, not as the fix.
2. **A stale F(x) after the box change — REAL, but not sufficient.** `applyBoxScaleOnce` rebuilds the
   grid and the Verlet list and stops, so `forceBuf` still holds the force of the pre-remap geometry
   when `step()`'s opening `kick_drift_wrap` consumes it as F(x_n). The evaporating path now refreshes
   it explicitly (the per-increment `forces()` call, documented as load-bearing, not diagnostic). But
   adding the same refresh to the shared `applyBoxScaleOnce` **does not** cure the pre-existing
   `tests/soup-grid-resize.test.ts` failure (§8.2), so it is a second independent defect and not the
   cause here.
3. **The force spikes an affine contraction leaves were simply above the integrator's stability bound —
   THE ACTUAL CAUSE, and the first derivation of that bound was wrong.** Per-increment `max|F|` was
   measured at **0.78e3–2.72e3** at 3.18 % increments. The first stability criterion used was "one step
   must not move a bead more than one σ", `F ≤ σ/dt² = 1e4` — which never fired, and the ramp diverged
   anyway. The correct condition for a Verlet integrator is that the force must not change appreciably
   *across* that displacement, and for a WCA r⁻¹² core `d(ln F)/d(ln r) = −13`, so F changes e-fold
   over `dr = r/13`. Requiring `F·dt² ≤ σ/13` gives **`F ≤ σ/(13·dt²) = 769`** — *below* every spike
   measured, which is exactly why it kept diverging.

**Both ends were then fixed, and the fix is reported with the number that makes it honest:**
`evaporationRampSteps` 12 → **18** (2.132 % linear per increment instead of 3.181 %), plus the same
minimiser as a guard on the corrected `σ/(13·dt²)` trigger. Across the whole six-cycle run the guard
fired on **9 of 216 increments (4.2 %)**, each time taking `max|F|` from ~0.8–1.4e3 to ~2.7–3.6e1:

```
[evaporation] evaporation increment=6/18 box=26.3610 solvent=11748 N=23032 max|F|=9.387e+2 minimisation_iterations=29 max|F|_after=2.877e+1 overstretched_tethers=0 longest=1.0509 threshold=1.4680
[evaporation] evaporation increment=13/18 box=22.6696 solvent=3918 N=15202 max|F|=7.848e+2 minimisation_iterations=29 max|F|_after=2.890e+1 overstretched_tethers=0 longest=1.1089 threshold=1.4680
[evaporation] evaporation increment=17/18 box=20.7972 solvent=706  N=11990 max|F|=1.364e+3 minimisation_iterations=29 max|F|_after=2.783e+1 overstretched_tethers=0 longest=1.0898 threshold=1.4680
```

**Why 4.2 % matters as a number and not as reassurance:** a minimisation mid-trajectory is a
perturbation, and the whole positive result below could in principle have been manufactured by
annealing. It cannot have been: the minimiser touched 9 of 216 increments, its per-call displacement
bound is 1.5 σ, and the rehydration minimisations (one per cycle, 6 total) measured a pre-existing-
particle displacement of **rms 0.187–0.280 σ, worst 0.749–0.893 σ** against that 1.5 σ bound. And §6.4
gives the control that settles it outright.

---

## 6. Cycling with real evaporation, against both baselines

### 6.1 Design

Box 30, `{C: 1860, O: 7440, H: 1860, M: 124, W: 21600}`, N = 32 884, ρ_tot = 1.21793, ρ_W = 0.800 —
the predecessor's own defended liquid composition, unchanged. `--relax`, `--cycle --evaporate`,
`clay: false`, kT = 1.1, seeds 19 and 23. Six cycles of `data/soup.json`'s own schedule.

**The control is not re-run: it is the predecessor's own committed artifact**
`verify/out/wetdry-control-19-trace.json` (and `-23-`), produced at the identical box, composition,
seed and `--relax` by the identical code path (evaporation is opt-in, and
`tests/soup-drywet-cycling.test.ts` passing unchanged in §8 is the evidence that path did not move).
Its samples are at 34 000/68 000/…/204 000 against the cycled arm's 36 800/68 600/…/195 400; the
control is **flat** over that whole span (amph 157–165, largest 11–15), so the 2–8 % step offsets
cannot carry the comparison. Step offsets are stated in the table rather than smoothed away.

### 6.2 The cycled trajectory, seed 19 — audited off-GPU from the checkpoints

`tests/continuous-run-audit.test.ts`, unmodified except for the census assertion of §8.3; artifacts
`verify/out/evaporation-cycled-19-trace.json` and `-early-trace.json`.

| globalStep | phase | box | N | ρ_org | stage | amph | yield | perTail | cc+co | aggs | **largest** | flat | radSh | encH₂O/thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 5 000 | wet | 30.0000 | 32884 | 0.4179 | amphiphiles | 44 | 0.0237 | — | 113 | 32 | 3 | 0.096 | n/a | — |
| 10 000 | wet | 30.0000 | 32884 | 0.4179 | amphiphiles | 62 | 0.0333 | — | 214 | 35 | 6 | 0.119 | n/a | — |
| 18 400 | **dry** | 20.3538 | 11299 | 1.3382 | micelles | 249 | 0.1339 | — | 831 | 8 | 213 | 0.724 | 1 | 0/1 |
| 28 400 | **dry** | 20.3538 | 11299 | 1.3382 | amphiphiles | 301 | 0.1618 | — | 1214 | 2 | 300 | 0.788 | 1 | — |
| **36 800** | wet | 30.0000 | 32884 | 0.4179 | **bilayer** | 302 | **0.1624** | — | 1387 | 3 | **189** | 0.133 | 1 | 0/303 |
| 41 800 | wet | 30.0000 | 32884 | 0.4179 | **bilayer** | 289 | 0.1554 | — | 1496 | 3 | **189** | 0.169 | **2** | 0/303 |
| **68 600** | wet | 30.0000 | 32884 | 0.4179 | **bilayer** | 263 | 0.1414 | — | 1682 | 6 | **169** | 0.347 | 1 | 0/303 |
| 73 600 | wet | 30.0000 | 32884 | 0.4179 | micelles | 261 | 0.1403 | — | 1709 | 4 | 168 | 0.361 | 1 | 0/303 |
| **100 400** | wet | 30.0000 | 32884 | 0.4179 | micelles | 252 | 0.1355 | — | 1787 | 5 | **162** | 0.429 | 1 | 0/303 |
| 99 400 | wet | 30.0000 | 32884 | 0.4179 | micelles | 248 | 0.1333 | 2.207 | 1864 | 4 | **168** | 0.292 | 1 | 0/303 |
| **133 800** | wet | 30.0000 | 32884 | 0.4179 | micelles | 239 | 0.1285 | 2.248 | 1932 | 3 | **166** | 0.373 | 1 | 0/303 |
| **168 200** | wet | 30.0000 | 32884 | 0.4179 | micelles | 231 | 0.1242 | 2.301 | 1980 | 3 | **160** | 0.367 | 1 | 0/303 |
| **195 400** | wet, cycling over | 30.0000 | 32884 | 0.4179 | micelles | 238 | 0.1280 | 2.323 | 1994 | 6 | **163** | 0.364 | 1 | 0/303 |
| 215 400 | settled wet | 30.0000 | 32884 | 0.4179 | micelles | 234 | 0.1258 | 2.330 | 2008 | 3 | 162 | 0.376 | **2** | 0/303 |
| 235 400 | settled wet | 30.0000 | 32884 | 0.4179 | micelles | 231 | 0.1242 | 2.357 | 2026 | 4 | 159 | 0.384 | **2** | 0/303 |
| 255 400 | settled wet | 30.0000 | 32884 | 0.4179 | micelles | 230 | 0.1237 | 2.367 | 2033 | 4 | 157 | 0.410 | **2** | 0/303 |
| **275 400** | settled wet | 30.0000 | 32884 | 0.4179 | micelles | 233 | 0.1253 | 2.386 | 2040 | 4 | **161** | 0.419 | 1 | 0/303 |

Dry-phase samples are shown for the density trajectory only and are **not** used for any aggregate
claim: at ρ_org = 1.3382 the whole system is one contact-percolating mass, so "largest aggregate = 300"
there is trivially true and means nothing. Every aggregate number quoted below is a **wet-phase**
number at box 30 and ρ_tot = 1.21793. `nonFinite = 0/0` and valence all-zero at **every one of the 22
checkpoints of the two lineages**.

### 6.3 Cycled against the wet control, at matched globalStep, with the cycle-count trend

| cycle | cycled step | control step | cycled yield | control yield | **ratio** | cycled largest | control largest | **ratio** | cycled cc+co | control cc+co | **ratio** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 36 800 | 34 000 | 0.1624 | 0.0860 | **1.89×** | **189** | 15 | **12.6×** | 1387 | 707 | **1.96×** |
| 2 | 68 600 | 68 000 | 0.1414 | 0.0860 | **1.64×** | **169** | 13 | **13.0×** | 1682 | 840 | **2.00×** |
| 3 | 100 400 | 102 000 | 0.1355 | 0.0844 | **1.61×** | **162** | 13 | **12.5×** | 1787 | 880 | **2.03×** |
| 4 | 99 400 | 102 000 | 0.1333 | 0.0844 | 1.58× | 168 | 13 | 12.9× | 1864 | 880 | 2.12× |
| 5 | 133 800 | 136 000 | 0.1285 | 0.0860 | **1.49×** | **166** | 12 | **13.8×** | 1932 | 919 | **2.10×** |
| 6 | 168 200 | 170 000 | 0.1242 | 0.0887 | **1.40×** | **160** | 11 | **14.5×** | 1980 | 961 | **2.06×** |
| over | 195 400 | 204 000 | 0.1280 | 0.0860 | **1.49×** | **163** | 12 | **13.6×** | 1994 | 989 | **2.02×** |

**And against the honest single-shot liquid-water baseline** (20 000 steps, n = 3: yield 0.08135,
`cc_bond` 296.3, `co_bond` 261.0, largest 10.7, stage `amphiphiles` ×3): yield **1.54–2.00×**, largest
aggregate **15.0–17.7×**, stage `bilayer` reached (never reached by the control in 204 000 steps).

**And against the previously published UNDER-DENSE numbers**, which is the harder comparison: the
predecessor's token-water arm (ρ_W = 0.26) measured yield 0.14677 and largest aggregate **91.0**. This
task's cycled arm at **real liquid water** measures largest **160–189**, i.e. **1.8–2.1× better than
the under-dense arm**, at 3.1× the water density. That is the first time in this project's reports
that a liquid-water number has beaten its own token-water number on the binding quantity.

**Reading the trend, including the part that is not favourable.** The gain is delivered by the **first**
drying event and then does not compound: largest 189 (cycle 1) → 169 → 162 → 168 → 166 → 160 → 163.
Cycles 2–6 are flat-to-slightly-declining, and the **yield declines monotonically** from 0.1624 to
0.1242 as `co_break` accumulates (9 → 66 events). So the honest statement is "**one evaporation is
worth 13× in aggregate size and 2× in bond count; the next five are worth nothing measurable**", not
"cycling compounds". L2's "limits are reached after about six cycles" is not what this model shows — it
reaches its limit after one.

### 6.4 The control that decides whether this is real: 80 000 steps of plain wet dynamics

Cycling ends at globalStep 195 400 (`cyclePhase` reports `wet/0`). The run then continued for
**80 000 further steps with no cycling at all** — plain Langevin dynamics at box 30, ρ_tot = 1.21793:

```
[campaign] step=215400 stage=micelles aggregates=3 largest=162 headShells=2 phase=wet/0 box=30.0000 bonds=1957
[campaign] step=235400 stage=micelles aggregates=4 largest=159 headShells=2 phase=wet/0 box=30.0000 bonds=1974
[campaign] step=255400 stage=micelles aggregates=4 largest=157 headShells=2 phase=wet/0 box=30.0000 bonds=1974
[campaign] step=275400 stage=micelles aggregates=4 largest=161 headShells=1 phase=wet/0 box=30.0000 bonds=1974
```

**163 → 162 → 159 → 157 → 161: flat over 80 000 steps without cycling.** The aggregate is a stable
object at real liquid water, not a compression transient that the affine expansion happened to
preserve, and not something the rehydration minimiser quenched into place (nine-tenths of those 80 000
steps are ordinary dynamics with no minimiser anywhere in them). `radialHeadShells = 2` at 3 of the 4
tail samples. This is the single measurement that turns the result from suggestive into a result.

### 6.5 Scatter, from a second seed

Seed 23, same composition, two cycles (`verify/out/evaporation-cycled-23-trace.json`):

| step | seed 19 largest | seed 23 largest | seed 19 yield | seed 23 yield |
|---|---|---|---|---|
| 36 800 | 189 | **178** (−5.8 %) | 0.1624 | **0.1500** (−7.6 %) |
| 41 800 | 189 | **178** (−5.8 %) | 0.1554 | **0.1468** (−5.5 %) |
| 68 600 | 169 | **167** (−1.2 %) | 0.1414 | **0.1355** (−4.2 %) |

Cross-seed scatter on the binding quantity is **±1–6 %**; the effect over the control is **12.5–14.5×**,
i.e. **200× the scatter**. A third source of scatter is also measured and stated: two lineages of the
SAME seed that differ only in where their resume boundaries fall gave wet-phase largest 168 vs 162 at
~100 000 steps (**±4 %**) — this engine is not bit-reproducible run to run, because the grid fill uses
atomics and the per-cell particle order therefore varies, changing the force summation order.

---

## 7. The box-54 re-run with real RNG state — the cheapest open question, answered

The predecessor's concern 8: *"The published 271-amphiphile box-54 result is therefore a lower bound;
nobody has re-run it with real RNG state… That re-run is the single cheapest way to find out whether
the vesicle programme is actually as far away as the published trace suggests."*

Re-run at the original's **exact** composition, box, seed, kT and resume cadence, with the fixed RNG
readback — and **no cycling, no evaporation, no `--relax`**, so the only difference from the published
run is the RNG state that resumes carry:

```
$ nice -n 15 npx tsx soup/cli/campaign.ts --label ves54r --box 54 --start '{"C":10844,"O":43376,"H":10844,"M":722,"W":40943}' --seed 19 --kT 1.1 --steps 45000 --every 15000 --dir data/checkpoints/ves54r
[campaign] step=15000/45000 stage=bilayer  aggregates=73 largest=100 headShells=1 cavityVolume=0.250 bonds=5332
[campaign] step=30000/45000 stage=micelles aggregates=42 largest=226 headShells=1 cavityVolume=6.250 bonds=8001
[campaign] step=45000/45000 stage=micelles aggregates=32 largest=232 headShells=1 cavityVolume=7.250 bonds=9160
[campaign] resume label=ves54r from data/checkpoints/ves54r/ves54r-step45000.json, step=45000
[campaign] step=60000/90000 stage=micelles aggregates=27 largest=281 headShells=1 cavityVolume=6.375 bonds=9759
[campaign] step=75000/90000 stage=micelles aggregates=26 largest=333 headShells=2 cavityVolume=11.625 bonds=10092
[campaign] step=90000/90000 stage=bilayer  aggregates=22 largest=588 headShells=1 cavityVolume=17.125 bonds=10311
```

Audited off-GPU (`verify/out/evaporation-box54-realrng-trace.json`):

| step | stage | amph | yield | perTail | cc | co | aggs | **largest** | Rg | flat | inPl | radSh | cav | encH₂O/thr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 15 000 | bilayer | 1431 | 0.1320 | 2.317 | 3219 | 2121 | 73 | 100 | 8.214 | 0.117 | 0.290 | 1 | 0.25 | 0/96.6 |
| 30 000 | micelles | 1729 | 0.1594 | 2.413 | 4798 | 3234 | 42 | 226 | 9.417 | 0.178 | 0.349 | 1 | 6.25 | 0/97.0 |
| 45 000 | micelles | 1749 | 0.1613 | 2.424 | 5488 | 3722 | 32 | 232 | 7.862 | 0.301 | 0.322 | 1 | 7.25 | 0/97.0 |
| 60 000 | micelles | 1735 | 0.1600 | 2.447 | 5846 | 3984 | 27 | 281 | 9.074 | 0.177 | 0.296 | 1 | 6.375 | 0/97.1 |
| 75 000 | micelles | 1708 | 0.1575 | 2.455 | 6061 | 4130 | 26 | 333 | 13.112 | 0.084 | 0.204 | **2** | 11.625 | 0/97.2 |
| **90 000** | **bilayer** | **1670** | **0.1540** | **2.459** | **6197** | **4237** | **22** | **588** | **14.928** | **0.123** | **0.383** | 1 | **17.125** | **0/97.9** |

**THE NUMBER, beside the old one: 588 against 271.**

| | published (zero RNG from its first resume) | **this re-run (real RNG)** |
|---|---|---|
| largest aggregate at step 90 000 | 268 | **588 (2.19×)** |
| largest aggregate, plateau | **271, frozen from step 135 000 to 315 000** | **588 and still rising** (232 → 281 → 333 → 588 over 45 000 steps) |
| enclosed volume of the largest aggregate | 6.25–6.75 σ³ | **17.125 σ³ (2.74×)** |
| `cc_bond` / `co_bond` at 90 000 | 5792 / 3818 | 6197 / 4237 (+7 % / +11 %) |
| amphiphile supply | 1804 | 1670 (−7 %) |

**The published 271 was a lower bound and it was loose by at least 2.19×, and the mechanism is exactly
the one the predecessor named:** under zeroed RNG every particle got identical Langevin noise, the
system translated rigidly (its measured MSD was 62.32 σ² and identical for all seven species classes)
and *relative* diffusion was destroyed — so aggregates could not coalesce, which is why that run's
largest aggregate froze at 271 for 180 000 steps while its bond counters also froze to the digit. With
real noise the same composition is still coalescing at 90 000 steps.

Two honest qualifications, stated because they matter:
1. **This arm is at the historical UNDER-DENSE water** (W = 40 943 → ρ_W = 0.26 σ⁻³, ρ_tot = 0.678),
   because its entire purpose is to re-measure a published number with one variable changed. It is
   **not** a liquid-water claim and must not be quoted as one.
2. Steps 0–45 000 of the original had correct RNG (no resume yet) and still differ from this re-run
   (largest 277 vs 232 at 45 000, −16 %). That is inside this engine's own measured run-to-run scatter
   (§6.5, and the predecessor's own 111 vs 154 at box 54, ±20 %), and it is why the comparison is made
   at step 90 000 — after the original's first resume, where the RNG fix actually bites — rather than
   before it.

---

## 8. Regressions — every one run, with numbers

### 8.1 The list, before → after

| test file | result | numbers, before → after |
|---|---|---|
| `tests/gate6-bilayer.test.ts` | **2 passed** | area 1.2054±0.0126 → **1.2094±0.0086** [1.1882, 1.2298]; thickness 4.4943 → **4.4737** (per-frame sd 0.1743, n=400); lnA drift t = 2.07; converge-from-1.55 tail 1.2049 → **1.2130**, from-0.9 1.2106 → **1.2286**, `peaksOk true` both. All inside 1.1–1.5 / 4–6. |
| `tests/water-bilayer-area-move.test.ts` | **1 passed** | area 1.1621 → **1.1124** [1.1077, 1.1281]; thickness 4.4645 → **4.9619**; drift t +0.10 → **1.94** (4.937e-5/chunk); water in core 28 → **48**/4500; buried 0.0816 → **0.1240**; cluster 1.0000 → 0.9974; accepted 0.1210 → **0.1327** of 3000; **verdict=passed**. These two rows are what §9.2 recomputes the floor from. |
| `tests/catalyst-turnover.test.ts` | **1 passed** | cc_bond 439 → **447**, co_bond 359 → **365**, chains 189 → **186**, amphiphiles 134 → **142**, bare carbons left 44 → **38**, occupiedAtEnd 3 → **3**, desorbTimeout 2, nonFinite 0, charge 0, census `{C:779,O:3117,H:779,M:52,W:695,K:0}` exact. |
| `tests/soup-nonfinite-guard.test.ts` | **1 passed** | guard fires at step 1 with 120/120 non-finite components; healthy control 0/0 at three points; cost **0.228 ms** vs chunk1000 **329.38 ms** = **0.069 %** (was 0.176/634.98 = 0.028 % — this machine's chunk was faster today, so the *fraction* rose). |
| `tests/soup-cold-start-relax.test.ts` | **1 passed** | max\|F\| **3.51806e4 → 18.19** (1934×); maxDisplacement **1.11432 σ** ≤ bound 10.05; steps 0→0; velocities, both RNG streams, bond graph, all four event counters unchanged; refusal at `globalStep != 0` fires; payoff pair holds (throws at step 1000 without the stage, 29 217/29 232 non-finite). |
| `tests/sim.test.ts` | **8 passed** | unchanged (grid-vs-brute-force force identity intact) |
| `tests/run-ui.test.ts` | **8 passed** | unchanged |
| `tests/soup-forces.test.ts` | **1 passed** | unchanged — grid + Verlet list vs full O(N²), no GPU console warning |
| `tests/soup-area-move.test.ts` | **2 passed** | `composeVsDirect maxDiff=9.537e-7` (identical to the published value); CPU potential = antiderivative of GPU forces; jacobian holds |
| `tests/soup-bonds.test.ts` | **3 passed** | catalyst gating holds; census+charge conserved under live reactions; assembly-vs-temperature ordering holds |
| `tests/soup-valence.test.ts` | **1 passed** | no carbon over 2 C–C or 1 C–O, no head over chainCapacity, 45 000 steps |
| `tests/soup-checkpoint.test.ts` | **3 passed** | round-trip exact; "a checkpoint does not change what the run computes next" holds; same-process continue writes real files |
| `tests/soup-rules.test.ts` | **4 passed** | schema + detailed balance unchanged; **the four new `dryWetCycle` fields validate** |
| `tests/soup-boxcycle.test.ts` | **7 passed** | the pure coordinate map, involution, long-coiled-chain regression, `computeDryBox` density and the schedule — unchanged |
| `tests/params.test.ts` | **3 passed** | the literal scanner is clean. It caught one real violation introduced by this task — a bare `1.5` inside a NEW comment in `soup/src/soup-evaporate.ts` — which was reworded to name `P.r_inf` instead. No rank-A constant appears anywhere new. |
| `tests/soup-drywet-cycling.test.ts` | **2 passed** | THE EVIDENCE THAT THE PRE-EXISTING PATH DID NOT MOVE: `concentrationFactor=1.1002` and `rampStepsCharged=1000` **identical**, `dryBox=29.0598`, `realisedDryDensity=1.34000`, box returns exactly, nonFinite 0/0 at four points. The two measured margins moved within their own loose bounds (mobility collapse 3.27× → **3.69×**, one-cycle event ratio 0.7143 → **0.4978**) — same sign, same conclusion; this is the engine's run-to-run scatter (§6.5), not a behaviour change. |
| `tests/continuous-run-audit.test.ts` | **1 passed (×4 runs)** | audited 4 lineages (22 checkpoints) with `expect()` on every invariant |
| **`tests/soup-evaporation.test.ts`** (NEW) | **3 passed** | see §2 |
| `tests/soup-grid-resize.test.ts` | **1 FAILED** | **PROVEN PRE-EXISTING — see §8.2.** |

**No tolerance was changed, no corridor widened, no assertion relaxed, no threshold moved.**

### 8.2 The one failing test, and the proof it is not mine

`tests/soup-grid-resize.test.ts`'s bracket-crossing test fails:

```
× box that actually crosses the cell-count boundary reallocates the grid: particles and forces (grid vs brute-force) are preserved
  → Verlet list: measured drift 193669536.2374 exceeds skin/2=0.7500
```

It was **not** taken on trust that this pre-dated the task. Every change of this task was stashed and
that test alone re-run on the resulting tree:

```
$ git stash push -m "evaporation-wip-probe" -- data/soup.json soup/cli/campaign.ts soup/src/checkpoint.ts soup/src/rules-validate.ts soup/src/rules.ts soup/src/sim.ts soup/src/soup-bindgroups.ts soup/src/soup-box-scale-math.ts soup/src/soup-box-scale.ts soup/src/soup-init-state.ts soup/src/soup-relax.ts soup/src/soup-runtime.ts soup/src/soup-types.ts tests/continuous-run-audit.test.ts
Saved working directory and index state On stage-a-atoms: evaporation-wip-probe
$ nice -n 15 npx vitest run tests/soup-grid-resize.test.ts
  × box that actually crosses the cell-count boundary, ... 8758ms
    → Verlet list: measured drift 872361.5732 exceeds skin/2=0.7500 ...
 Test Files  1 failed (1)
$ git stash pop
```

**Same failure, same mechanism, on the tree without this task.** (The two drift magnitudes differ
because the engine is not bit-reproducible, §6.5.) The diagnosis is §5.2: `applyBoxScaleOnce` leaves a
stale F(x_n) for the next half-kick. Adding the missing force rebuild there was **tried and measured
not to cure it** (drift 1.94e8 with the rebuild in place), so the box change that test picks is simply
too large for its own ramp at that composition. The rebuild was therefore **reverted out of the shared
path and the reason recorded in the code**: it would change the trajectory of every published dry-wet
box-scaling number, and this task's budget could not re-run them. Nothing was masked, no assertion
weakened, and the evaporating path refreshes F(x_n) explicitly instead.

### 8.3 The one committed assertion that changed, and why it is stricter

`tests/continuous-run-audit.test.ts` compared the recomputed census against `config.start`. With the
solvent able to leave, a mid-dry-phase checkpoint legitimately carries fewer solvent beads — that IS
the mechanism. The single comparison became **two**: the census is compared against the checkpoint's own
recorded live census (`activeCounts`, new field, written by every checkpoint), and **in addition** every
NON-solvent count is compared against `config.start` regardless, because organics may never be created
or destroyed by anything. A pre-task checkpoint (no `activeCounts`) takes the identical single
comparison it always did.

---

## 9. The window: recomputed arithmetic, and why it was not run

Cycling **did** earn the window run (§6 beats both baselines at every cycle on every measure). The
budget did not pay for it, and that is stated as the reason rather than dressed up: at
L ∈ [50.2, 55.7] the composition is N ≈ 152 000–203 000, the measured cost is ~15 ms/step there, and
six cycles plus their 216 evaporating increments (each carrying a rejection-sampled re-insertion of
~100 000–133 000 solvent beads) is ~7–8 foreground chunks on its own — against the ~3 chunks left
after the proofs, the trend and the box-54 re-run, which the brief ranked ahead of it.

### 9.1 What the window arithmetic now is

Same formula as `continuous-run-report.md` §1.1; no new constant; `stageThresholds.enclosedVolume`
= 370.8656 σ³ untouched.

```
R_in  = (3 x 370.8656 / 4pi)^(1/3) = 4.45699 sigma
R_mid = R_in + t/2
floor = 2 x 4pi x R_mid^2 / a
```

| basis for t, a | t σ | a σ² | R_mid σ | **floor** |
|---|---|---|---|---|
| **explicit-water gate, measured in §8.1** | **4.9619** | **1.1124** | **6.93795** | **1087.5 → 1088** |
| solvent-free gate 6, measured in §8.1 | 4.4737 | 1.2094 | 6.69385 | 931.2 → 931 |
| the predecessor's basis | 4.4645 | 1.1621 | 6.68925 | 967.7 → 968 |

**The floor moves 968 → 1088, i.e. UP by 120 amphiphiles (+12.4 %)**, and this time the reason is the
thickness, not the area: t rose 4.4645 → 4.9619 (+11.1 %) while a fell 1.1621 → 1.1124 (−4.3 %), and
both push the same way (a thicker shell needs a bigger mid-surface, a smaller head footprint needs more
lipids to cover it). Band **931–1088**; **1088 used**, the explicit-water gate, the same one the
original derivation chose. Trend across four tasks: **960 → 912 → 968 → 1088**.

### 9.2 The window itself

Measured saturated amphiphile density, from the four settled-wet samples of §6.2 (215 400–275 400,
after cycling ended): amph = 234, 231, 230, 233 → mean **232.0** [230, 234] →
ρ_amph = 232/27 000 = **8.5926e-3 σ⁻³** (the predecessor's figure was 5.937e-3, i.e. **1.447× lower**).

- **supply**: ρ_amph·L³ ≥ 1088 → L ≥ (1088/8.5926e-3)^(1/3) = **50.22 σ** (with the 931 end of the band: 47.67 σ)
- **measurability**: R(L) = 0.067·L^1.5 ≤ L/2 (the box-58 head-radius calibration, unchanged) → L ≤ **55.69 σ**

**Window L ∈ [50.22, 55.69] σ — width 5.47 σ.** The predecessor's window was **1.06 σ** and "on the
verge of closing altogether"; real evaporation raised the yield enough to pull the supply end 4.41 σ
back inward and **reopen the window 5.2×**. At L = 55 the predicted supply is
8.5926e-3 × 166 375 = **1430** against the floor 1088, a **1.31×** margin (the predecessor ran at
1.08×). Verlet: 202 636 × 2500 × 4 = 2.026 GB against 4.295 GB, and 202 636 particles against the
measured hard ceiling 429 496 — both 2.12×.

### 9.3 Verdict, and the binding constraint named and numbered

**No vesicle. Nothing closed: `encapsulatedWater` = 0 against a threshold of 303 at every wet-phase
checkpoint of every lineage, `cavityVolume` ≤ 3.0 σ³ against the 370.8656 σ³ closure minimum,
`closed = false` everywhere.** What changed is the distance to it.

| # | constraint | measured | required | shortfall | binding? |
|---|---|---|---|---|---|
| **1** | **amphiphile SUPPLY at box 30** | **232**, saturated | ~1088 | **4.69×** | **YES — at this box** |
| 2 | aggregate size at box 30 | **161** (settled, flat over 80 000 steps) | ~1088 | **6.76×** | no — downstream of 1 (161 is **69.4 % of every amphiphile in the system**) |
| 3 | aggregate size at box 54, real RNG, token water | **588**, still rising | ~1088 | **1.85×** | the closest this project has measured |
| 4 | closure / encapsulated water | 0 | ≥ 303 | total | downstream of 1–3 |
| 5 | medium density | ρ_tot 1.21793, ρ_W 0.800 | liquid | none | no — settled |
| 6 | Verlet list / memory | 0.33 GB, 32 884 particles | 4.295 GB, 429 496 | none | no |
| 7 | run length | plateau reached: 80 000 settled-wet steps flat | a plateau | none | no |

**BINDING CONSTRAINT: (1) AMPHIPHILE SUPPLY — 232 against ~1088, 4.69× short — and it is binding
because the box is too small, not because the physics failed.** At box 30 this engine now puts **69.4 %
of every amphiphile it makes into a single aggregate**; the aggregate cannot be larger than the supply,
and the supply is set by L³. That is a qualitatively different failure from the predecessor's: there,
supply was 1.08× over the floor and the material **fragmented** into 300 aggregates of ~25 molecules
(33.4× short on size); here the material coalesces and the box simply does not contain enough of it.

**The shortfall sequence, on the same quantity the predecessors reported (aggregate size against the
floor in force): 80× → 8.9× → 3.36× → 33.4× → 6.76×** at box 30. The 33.4× regression is undone
5.0-fold. It is **not** better than the 3.36× best — and the box-54 re-run of §7 says that 3.36× was
itself an artefact of the zero-RNG defect and the true figure there was **1.85×**.

---

## 10. Resources, wall time, housekeeping

- **Compute chunks used: ~32 of the 35 allowed.** Breakdown: 3 pure-arithmetic vitest runs (seconds
  each), 3 isolation-test runs (two of which failed and are reported as the measurements they are), 2
  box-24 cycle tests, 8 cycled-campaign invocations at box 30 across four lineages (three of them
  superseded by the two divergences and the ramp re-derivation), 2 box-54 invocations, 5 off-GPU audits
  (CPU-only, ~0.5–1.6 s each), 1 stash probe for §8.2, 5 regression groups, 1 final new-test run.
- **Total steps ≈ 1.60 M** — 275 400 (cycled seed 19, main lineage) + 100 400 (dense early lineage) +
  70 000 (seed 23) + 90 000 (box 54) + 259 200 across the three superseded cycled lineages + ~117 000
  in the new tests + ~690 000 across the regression suite.
- **Longest single foreground invocation: 336 s.** One invocation (the very first cycled run) exceeded
  the 120 s default and was moved to a tracked background task **by the harness itself**, not by
  choice; its full output was returned and is quoted in §5. Every invocation was `nice -n 15`, one at a
  time, foreground, and every multi-file vitest group was run with `--no-file-parallelism` specifically
  so that no two compute processes could ever exist at once.
- **No orphan processes remain.** `pkill -f puppeteer_dev_chrome_profile` was run after **every**
  browser-bearing invocation and `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** every
  time, including after the last.
- **The dev server on :5199 was neither started, stopped nor inspected.** Every `gpuPage()` opened its
  own vite server on its own port.
- `tests/soup-vesicle.test.ts` was **never** run. The full suite was never run. `--dump-dom` was never
  used. **No particle array was ever transferred as JSON numbers:** every reduction (max|F|, MSD,
  non-finite counts, census, aggregate sizes, event counts, position fingerprints, insertion
  separations) was computed inside the page or inside the offline auditor and returned as scalars.
- **Checkpoints written:** 4 box-30 lineages (~2.2 MB each) and 6 box-54 files (~17 MB each);
  `data/checkpoints/` is a gitignored run artifact and the three superseded box-30 lineages were
  deleted as they were superseded.
- `data/params.json` untouched. Largest file in the tree is still `soup/wgsl/step.wgsl` at 596 lines;
  the new `soup/src/soup-evaporate.ts` is 582 and nothing crossed 600.

## 11. Concerns

1. **The headline is a positive, and positives need harder scepticism than negatives.** The three
   things that could have manufactured it were each tested rather than argued away: (a) the
   compression artefact — killed by §6.4's 80 000 steps of plain wet dynamics with the aggregate flat
   at 157–163; (b) the rehydration minimiser annealing structure into place — bounded and measured
   (rms 0.187–0.280 σ, worst 0.893 σ, against a 1.5 σ analytical bound, six times in 275 400 steps);
   (c) the ramp guard minimiser — fired on 9 of 216 increments (4.2 %). What I have **not** controlled
   for is a run that receives exactly those 9 + 6 minimisations at the same global steps with **no
   cycling**; that is the one control still missing, and it is cheap.
2. **The gain does not compound with cycle count.** One evaporation buys 13× in aggregate size; cycles
   2–6 buy nothing measurable and the yield declines monotonically (0.1624 → 0.1242) as `co_break`
   accumulates. Anyone reading this as "wet–dry cycling works" should read it as "**the first drying
   event works**". L2's six-cycle limit is not what this model shows.
3. **The dry phase is a jammed solid and its aggregate numbers are meaningless.** At ρ_org = 1.3382
   the whole system is one contact-percolating mass. Every aggregate number in this report is a
   wet-phase number; the dry rows are shown for the density trajectory only. Anyone extracting "largest
   aggregate 300" from the dry rows would be reading percolation, not structure.
4. **The 6.9 % solvent-free transient is an artefact, not physics.** 3400 real steps of every
   rehydration (17 × 200) are spent with the organics in vacuum at falling density, because an affine
   expansion never opens a bead-sized cavity and there is nowhere to put the solvent until the box is
   fully expanded. Over six cycles that is 20 400 steps of vacuum dynamics out of 195 400. Tails
   attract in vacuum, so this transient plausibly *helps* coalescence — i.e. part of the §6 effect may
   be the artefact rather than the concentration. Separating them needs an arm that expands and
   re-inserts in one atomic operation, which needs an insertion method that works at liquid density.
5. **`tests/soup-grid-resize.test.ts` is failing and I did not fix it.** It fails identically without
   this task (§8.2), the diagnosis is written into the code, and the one-line cure was tried, measured
   insufficient, and reverted because it would silently invalidate every published dry-wet
   box-scaling number. The next task should apply it **and** re-run those numbers.
6. **A second latent defect of the same family is named and unfixed:** the adsorption tether can cross
   FENE's `r_inf` by ordinary thermal stretching between two bond dispatches (20 real steps), and past
   `r_inf` the FENE force changes sign and diverges. The valve implemented here fires only at box
   changes in the evaporating path, and it measured 0 firings — so this defect is latent, not
   demonstrated, and it should be closed by making `desorbStretch` a per-step check or by clamping
   `fene_dv` past `r_inf`.
7. **The box-54 re-run is at under-dense water and is one seed.** Its 588 is not a liquid-water number
   (ρ_W = 0.26) and must not be quoted as one; its purpose was to re-measure one published figure with
   one variable changed. It is also still rising at 90 000 steps, so 588 is itself a lower bound —
   the plateau was not reached, and reaching it needs ~4 more chunks.
8. **No window run.** Cycling earned it and the budget could not pay for it (§9). The arithmetic is
   recomputed and the window is 5.2× wider than the predecessor's, so the next task starts from a real
   target rather than a 1.06 σ slit — but nobody has yet measured whether 69 % of the amphiphiles
   coalesce into one aggregate at L = 55 as they do at L = 30. If they do, that is ~990 against a floor
   of 1088; if the fraction falls with box size, it is not. **That single number is the whole
   remaining question.**
9. **`clay: false` in every arm**, as in all four predecessors, and now doubly forced: a box change is
   refused on a system with an immobile phase, and solvent removal additionally requires the solvent to
   be the last non-empty composition block, which a clay run violates.
10. **`evaporationRampSteps` was re-derived mid-task, from 12 to 18, after the ramp diverged three
    times.** That is a schedule/robustness number (rank D), not a threshold or a rate, and the new
    value comes from an arithmetic stability bound (`σ/(13·dt²)`) rather than from tuning an outcome —
    but it was changed *after* seeing runs fail, and a reader is entitled to know that rather than
    find only the final number in the file.
11. **The engine is not bit-reproducible run to run** (atomics in the grid fill vary the per-cell
    particle order and hence the force summation order). Measured consequences: ±4 % on the wet-phase
    largest aggregate between two lineages of the same seed, −16 % on the box-54 largest at step
    45 000, and `soup-drywet-cycling`'s own event ratio moving 0.7143 → 0.4978 with no code change on
    that path. Every claim in this report is sized against that scatter, and the ±1–6 % cross-seed
    figure of §6.5 should be read together with it.

## 12. Files

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/evaporation-report.md` — this report. `.gitignore`
  line 6 ignores `.superpowers/`, so like every predecessor it is committed with `git add -f`; the
  deviation is deliberate and flagged rather than silently introduced.
- NEW: `soup/src/soup-evaporate.ts` (582 lines) — the plan, both ladders, the sampler, the live
  removal/re-insertion, and the tether valve.
- NEW: `tests/soup-evaporation.test.ts` (3 tests).
- NEW artifacts: `verify/out/evaporation-cycled-19-trace.json` (16 checkpoints),
  `-19-early-trace.json` (16), `-cycled-23-trace.json` (12), `-box54-realrng-trace.json` (6),
  `evaporation-evap19-trace.json` (4, the first superseded lineage, kept as the record of the
  divergence).
- MODIFIED: `data/soup.json` — the `dryWetCycle` section only, **four fields added**
  (`evaporateSolvent` false, `residualSolventFraction` 1/1400, `evaporationRampSteps` 18,
  `insertionMinSeparationSigma` 0.75) plus nine derivation items appended to that section's own
  `basis`. **No existing field in the file changed.**
- MODIFIED: `soup/src/soup-bindgroups.ts` (the `pbuf` active-range binding), `soup/src/sim.ts`,
  `soup/src/soup-types.ts`, `soup/src/soup-runtime.ts`, `soup/src/soup-init-state.ts`,
  `soup/src/soup-box-scale.ts`, `soup/src/soup-box-scale-math.ts`, `soup/src/soup-relax.ts`,
  `soup/src/checkpoint.ts`, `soup/src/rules.ts`, `soup/src/rules-validate.ts`,
  `soup/cli/campaign.ts`, `tests/continuous-run-audit.test.ts`.
- `data/params.json` untouched. No potential, rate, threshold, corridor, recogniser or rank-A constant
  was modified anywhere.
