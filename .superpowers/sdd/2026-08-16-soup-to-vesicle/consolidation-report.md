# Consolidation: the gate table regenerated, the one document a reader needs, and the page brought back onto the physics

Task `consolidation` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree. Predecessor:
`final-campaign-report.md`. **Delivery step, no new science**: no campaign started, no corridor
widened, no threshold moved, no rank-A constant touched. `data/params.json` **NOT touched**.
`data/soup.json` **NOT touched, not one field** — including at the moment where it would have been
the easy fix (§3.1: the viewer needed liquid water, and it got it by deriving the count on the page
instead of editing the file's `start`). `tests/soup-vesicle.test.ts` **never run**.

---

## 0. WHAT CHANGED, TEN LINES

1. **`verify/out/gates.json` regenerated through its own pipeline** (`npm run verify`), now
   publishing **12 gates instead of 5** — including the two explicit-water gates that are a genuinely
   new result of this campaign, the FAILED closure gate carrying the encapsulated-water number, the
   soup chain-length statistics, and the final verdict gate naming **connectivity/percolation** as the
   binding constraint. Nothing hand-edited (§1).
2. **Every campaign-sourced row was re-measured today, off-GPU, from the campaign's own checkpoints**
   (5 audit checkpoints + 5 percolation checkpoints in **6.5 s** at N = 191 778) — so no number is
   carried forward stale, and each row states which artifact it came from and when (§1.3).
3. **The pipeline now refuses to launder a stale number**: a missing input artifact yields no metric,
   so the gate publishes as `unproven` **with the reason and the report to consult**, pinned by a test
   (§1.2, §1.5).
4. **`docs/soup-to-vesicle-verdict.md`** (new, committed, Russian) — the single honest summary for
   someone who was not here: what was tested, what was built, what was measured with ranks, the
   verdict and its mechanism, the five real defects with the published numbers each one moved, the
   unproven/one-seed list, and what a next model must change (§2).
5. **The run page was showing a medium no current gate is measured in.** Both size presets now derive
   the solvent count from the box at the **measured liquid density 0.8 σ⁻³**: `tiny` **570 → 3938**
   particles, `default` **~15 000 → 25 904**, with the density printed in the preview (§3.1).
6. **The page never ran the cold-start minimisation** — every campaign passes `--relax`, the page was
   the one caller that did not. At the new density that is the difference between a run and a NaN. Now
   it runs it and prints the numbers into the trace (§3.2).
7. **A silent click-eating layout bug, found by measurement and fixed at the class level.** The
   bottom-anchored honesty notes spanned the full width and covered the START button: the very first
   headless run of the page after the preview grew never started at all — state `idle`, trace empty,
   no error anywhere. Notes moved clear of the control column and stacked in a flex column; the hit
   target is now asserted in `tests/run-ui.test.ts` (§3.3).
8. **The honesty note now states the model's actual limitations** — no vesicle, no electrostatics,
   salt/pH unrepresentable, evaporation as solvent removal at 3.20× against the literature's 1400×,
   and the rank-D list — instead of only what the time axis means (§3.4).
9. **The stage readout was showing the wrong closure observable.** The per-aggregate cards printed
   `cavityVolume` (the VACUUM-cavity detector, which reads a water-filled interior as empty) and never
   `encapsulatedWater`, which is what the verdict actually rests on. Both rows added, including the
   honest `centre-untrusted` refusal (§3.5).
10. **The clay platelet: measured visible (191 instances, `visible: true`), and given the control it
    never had.** Every arm of every campaign ran `clay: false`; the page silently ran WITH a platelet.
    It is now a labelled switch that says so (§3.6).

---

## 1. TASK 1 — THE GATE TABLE

### 1.1 The regenerated table, every row

`npm run verify` → `verify/out/gates.json`, `runId` stamped once, `generatedAt`
**2026-08-20T11:49:51.254Z**, commit `5b5937c` (dirty: true — this task's own diff, which is what the
stamp is for).

| gate | value | corridor | rank | verdict | source |
|---|---|---|---|---|---|
| `area-per-lipid` | **1.200976** σ² [1.1871, 1.2204] | 1.1–1.5 | A | **passed** | this run |
| `bilayer-thickness` | **4.492472** σ | 4–6 | A | **passed** | this run |
| `bending-modulus` | **null** | 5–50 kT | A | **unproven** | this run (fit window invalid: slope −1.056, 4 shells, q_max 0.1803) |
| `area-per-lipid-water` | **1.159693** σ² [1.1558, 1.1637] | 1.1–1.5 | C | **passed** | `water-bilayer-area-move.json`, 11:49:04Z |
| `bilayer-thickness-water` | **4.531147** σ | 4–6 | C | **passed** | same artifact |
| `vesicle-closure-water` | **0** (0 beads vs **320.891**) | ≥ 1 | B | **FAILED** | `gates-campaign-trace.json`, 11:49:36Z, step 113 400 |
| `aggregate-percolation` | **3** axes of 3 (slabs 19/19/19 of 19) | ≤ 0 | B | **FAILED** | `gates-percolation.json`, 11:49:38Z |
| `vesicle-verdict` | **0** aggregates | ≥ 1 | A | **FAILED** | `gates-campaign-trace.json`, 5 checkpoints, all false |
| `chain-length-asf` | **0.151862** | — (none) | D | **unproven** | trace, step 176 400 |
| `mean-tail-length` | **3.490** beads | 2–3 | D | **unproven** | trace, step 176 400 |
| `closure` (detector fixture) | **1284.875** σ³ | ≥ 370.8656 | D | **unproven** | this run (synthetic shell) |
| `chain-to-bead-mapping` | **3** | — | D | **unproven** | this run |

**Why each `unproven` is unproven**, since the brief requires the reason to be published and not
inferred:

- `bending-modulus` — **measured this run and the measurement failed**: `valid: false`, no κ exists to
  report, so the metric is deliberately OMITTED from the metrics record rather than smuggled through
  as NaN. Not a budget problem.
- `chain-length-asf` — **rank D, and the reason is in the gate's own `conditions`**: the spec states
  gate 2's criterion as "histogram divergence" with no numeric corridor, and no independent
  literature corridor exists for α of THIS system. Inventing a tolerance here would be fitting the
  gate to the result. The value and both α estimates are published anyway (α_ev 0.7513, α_rec 0.7732,
  r² 0.928 — they agree to **2.9 %**, and the shape is geometric, which is the ASF signature).
- `mean-tail-length` — **rank D by the spec's own §4**: the carbon→bead mapping is ours. Published
  with its window and with the honest note that **3.490 is ABOVE 2–3 and would read as a failure if
  that mapping carried an independent corridor**.
- `closure` — unchanged rank D, unchanged meaning: its value is the flood-fill detector's own
  SYNTHETIC check shell, not a membrane. Left exactly as it was, deliberately (§1.4).
- `chain-to-bead-mapping` — unchanged rank D.

**No gate was published `unproven` for being too expensive to re-measure.** Every campaign-sourced row
was re-measured today (§1.3).

### 1.2 What was added to `data/literature.json`, and the rank of each

Seven new gate definitions; the five existing ones are byte-identical except for reordering. Reading
order is now: solvent-free rank-A gates → explicit-water gates → closure/percolation/verdict → chain
statistics → the two historical rank-D rows.

- `area-per-lipid-water`, `bilayer-thickness-water` — **rank C**, and this is a deliberate demotion
  from the corridor's own rank A: the corridor IS Cooke & Deserno (pinned to be literally the same
  object as its solvent-free namesake's, §1.5), but the number is produced in a medium whose pair
  depths are RATIOS taken from `martini_v2.1.itp` and not calibrated on this tree, with
  `epsilonScale = 1.0` at rank D. `hydrophobic-asymmetry-report.md` ranks that table C itself.
- `vesicle-closure-water` — **rank B (computed)**. Published as a RATIO because the threshold is
  recomputed per snapshot from the live bulk water density (measured 317.1–320.9 beads across the
  campaign), so no fixed number for it can honestly live in a literature file. The floor of 1 is the
  definition, not a tolerance.
- `aggregate-percolation` — **rank B**. Target `{max: 0}`: a finite object wraps zero axes BY
  DEFINITION. No constant is introduced; the connectivity cutoff is the project's own.
- `vesicle-verdict` — **rank A (measured)**: the project's own vesicle recogniser, on every checkpoint.
- `chain-length-asf`, `mean-tail-length` — **rank D**, see §1.1.

### 1.3 How the campaign rows were re-measured rather than quoted

The campaign itself is 176 400 steps at N = 191 778 and must never be re-run by `npm run verify` on
someone's working machine. But its **checkpoints are on disk**, and the analysis is off-GPU, so the
numbers are genuinely re-derived, not copied out of a report:

```
$ CONTINUOUS_RUN_CHECKPOINTS="data/checkpoints/zfB54/zfB54-step41400.json data/checkpoints/zfB54/zfB54-step104400.json data/checkpoints/zfB54/zfB54-step113400.json data/checkpoints/zfB54/zfB54-step140400.json data/checkpoints/zfB54/zfB54-step176400.json" \
  CONTINUOUS_RUN_ARTIFACT=verify/out/gates-campaign-trace.json \
  PERC_CHECKPOINTS="data/checkpoints/zfB54/zfB54-step41400.json data/checkpoints/zfB54/zfB54-step113400.json data/checkpoints/zfB54/zfB54-step176400.json data/checkpoints/dec54/dec54-step174400.json data/checkpoints/dec54/dec54-step41400.json" \
  PERC_ARTIFACT=verify/out/gates-percolation.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts tests/percolation-check.test.ts tests/gates.test.ts --no-file-parallelism

RUN-AUDIT artifact written: verify/out/gates-campaign-trace.json (5 checkpoints)
RUN-AUDIT-TETHER violations=0
 ✓ tests/continuous-run-audit.test.ts (1 test) 3484ms
PERC {"file":"...zfB54-step41400.json","step":41400,"box":54,"aggregates":2,"amphiphilesInLargest":2211,"particlesInLargest":11842,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":"...zfB54-step113400.json","step":113400,"box":54,"aggregates":2,"amphiphilesInLargest":2112,"particlesInLargest":13851,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":"...zfB54-step176400.json","step":176400,"box":54,"aggregates":2,"amphiphilesInLargest":2077,"particlesInLargest":14265,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":"...dec54-step174400.json","step":174400,"box":54,"aggregates":25,"amphiphilesInLargest":151,"particlesInLargest":596,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[7,7,10,19]}
PERC {"file":"...dec54-step41400.json","step":41400,"box":54,"aggregates":22,"amphiphilesInLargest":200,"particlesInLargest":704,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[12,13,12,19]}
 ✓ tests/percolation-check.test.ts (1 test) 2232ms
 ✓ tests/gates.test.ts (8 tests) 5ms
 Test Files  3 passed (3)      Tests  10 passed (10)      Duration  6.57s
```

**Every number reproduces the predecessor's to the digit** — largest 2211/2112/2077, 3 of 3 axes,
19/19 slabs, controls at 0 of 3 with 7/7/10 and 12/13/12 slabs, invariants and the tether detector
clean. The audit's own closure readings: `encapsulatedCount: 0` against
`encapsulationThresholdCount` **317.149** (step 41 400), **320.787** (104 400), **320.891**
(113 400), `closed: false` at all three; `null(centre-untrusted)` at 140 400 and 176 400 — 3 of 5
wet checkpoints reportable, and the closure gate takes the LAST reportable one (113 400), not the
most flattering.

The explicit-water gates come from a real re-measurement too, not from the predecessor's line:

```
$ nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=450 areaPerLipid(MEASURED, tail mean)=1.1597
  [min 1.1558, max 1.1637, corridor 1.1-1.5] driftPerChunk(lnA)=-3.424e-6 t=-0.73
  thickness(measured)=4.5311 [corridor 4-6] clusterFraction=1.0000 waterInCore=55/4500
  headBuriedFraction=0.1608 acceptedFraction=0.1142 of 4500 trials throughput=1139.84 steps/s at N=5700 verdict=passed
WATER-BILAYER-AREAMOVE WROTE verify/out/water-bilayer-area-move.json
 ✓ tests/water-bilayer-area-move.test.ts (1 test) 199779ms
```

Against the predecessor's own numbers (`final-campaign-report.md` §3: area 1.1777 [1.1722, 1.1828],
thickness 4.7990, 300 chunks, buried 0.0857, core 53/4500): area **1.1777 → 1.1597** (−1.5 %),
thickness **4.7990 → 4.5311** (−5.6 %), head burial **0.0857 → 0.1608**, chunks **300 → 450** (this
run's settle criterion took longer). Both still inside their untouched corridors, and the movement is
this engine's own run-to-run scatter — the same scatter `water-calibration-report.md` §8.1 measured at
±0.3–0.5 σ on thickness at identical seeds. **No corridor was touched to accommodate it.**

### 1.4 What was deliberately NOT changed

The existing `closure` gate keeps its id, metric (`enclosedVolume`), target (`min: 370.8656`), rank D
and meaning. Two reasons, in order: `soup/src/stages.ts`'s `loadStageThresholds()` reads
**`data/literature.json`'s `closure.target.min` by that exact id** and throws if
`data/soup.json`'s `stageThresholds.enclosedVolume` disagrees — that tie is the guard against the
stage ladder and the gate table disagreeing about what a vesicle is, and repurposing the id would have
quietly cut it. And the rank-D distinction it encodes (the value is the detector's own synthetic check
shell, not a membrane) is still true and still worth publishing. The FAILED closure gate the brief
asks for is therefore a **new** row, `vesicle-closure-water`, on the observable the verdict actually
rests on.

### 1.5 The guards added, so this cannot rot quietly

`tests/gates.test.ts` **5 → 8 tests**, all passing:

- the new corridors are pinned, and the two explicit-water gates are pinned to be `toEqual` their
  solvent-free namesakes' targets — so «the bilayer holds in water» can never become «…by a corridor
  of its own»;
- `mean-tail-length` and `chain-length-asf` are pinned at rank **D**, so nobody promotes them by
  editing one field;
- the three campaign gates are asserted to come out **failed** on the measured numbers
  (0/320.891, 3 axes, 0 vesicles) **and** to come out **passed** on a hypothetical finite object with
  a full lumen (1.4, 0 axes, 1) — so they are not merely wired to fail;
- a missing input artifact is asserted to produce `unproven` + `value: null` + a `note` containing the
  reason, and `provenance` is asserted to be `undefined` when the pipeline does not know one (never
  invented).

---

## 2. TASK 2 — THE ONE DOCUMENT

**`docs/soup-to-vesicle-verdict.md`** (new, 363 lines, Russian, committed — `docs/` is not
gitignored, unlike `.superpowers/`). Eight sections:

1. **what was tested** — one question, plus the three project rules that decide what counts as an
   answer (gates published passed/failed/unproven and never widened; the A/B/C/D rank table quoted
   from the spec; no hand-written chemical constant in `engine/`; hydrophobicity must be emergent).
2. **what was built** — the six layers with their paths, and the statement that the atomistic detail
   on screen is a reconstruction, not a simulation.
3. **what was MEASURED** — the full 12-row gate table with ranks and verdicts, plus the two
   subsections a reader needs to judge it: why the explicit-water gates are the campaign's genuinely
   new result (12.2–12.4 σ → 4.5311 σ, the head-burial mechanism, the MARTINI ratio table, the
   +4.5 kJ/mol exchange energy that keeps hydrophobicity emergent, and why the rank is C), and the
   chain-length statistics with both α estimates and why the gate is nevertheless unproven.
4. **the verdict and the mechanism** — the constraint table (supply 2.10× over, size 2.08× over, edge
   energy 0.583 kT, **connectivity 3 of 3 axes = binding**), how percolation is measured rather than
   argued, the three discriminating controls, and the two regimes with nothing measured between them.
5. **the five defects**, each with what was wrong, how it was diagnosed, and the exact published
   numbers it moved: the tip-choice nucleation bug (eligible triples 393 → 3 → 0; nucleations
   117 → 192, amphiphiles 56 → 129, bare carbon left 133 → 33 of 779), the zero-RNG checkpoints (all
   72 checkpoints 100 % zero RNG bytes; largest aggregate 268 → 588 = 2.19×, the "271 frozen for
   180 000 steps" plateau dissolved, enclosed volume 6.25–6.75 → 17.125 σ³, published shortfall
   3.36× → 1.85×), the stale F(x) after a box change (resident buffer wrong by 313.7 on a 513 force
   scale; largest aggregate 249.3 → 109.3 = a 2.28× shrink with non-overlapping ranges, so the honest
   cycling effect is 8.1–8.4× and not the published 12.5–14.5×; the ramp guard 9-of-216 → 0-of-36 was
   the defect, not physics), the silent divergence (every NaN comparison is false, so the guard passed
   forever; the new O(N) exponent scan costs 0.212 ms against 331.68 ms = 0.064 % and throws at step
   1000 naming 6957 non-finite components of 60 750), and the z-boundary double count (the seven-row
   table of moved numbers, including counts being OVERstated by 10–30 % and
   `radialHeadShells` 2 → 1 turning a published PASS into a FAIL).
6. **twelve items that remain unproven, one-seed or disputed** — including the one-seed campaign, the
   new instrument's first result, what `cavityVolume` actually measures on a network, κ never measured,
   that two-tailedness and tail length cannot be separated by composition, the unmeasured O:C gap, the
   deliberate `water-bilayer*` inconsistency, closure resting on 6 of 15 samples, the three known-flaky
   tests with their own spread on unchanged code, the 3.20× vs 1400× evaporation gap and the 691×
   carbon enrichment, no electrostatics/salt/pH, and the clay platelet that no published measurement used.
7. **what a next model must change** — the four requirements as `final-campaign-report.md` §9.1
   specifies them (two-tailed by construction: confirmed; a dilution axis independent of the chemistry:
   new; non-fragmenting rehydration: retired; a real CMC: open and untested), plus the fully specified
   cheapest next measurement (pre-formed lipids at ρ_amph 1.34e−2 / 6.7e−3 / 3.3e−3 / 1.7e−3 σ⁻³ at
   box 54, percolation as the read-out, and the arithmetic showing the window, if any, lies between
   the first two rungs).
8. **how to re-check it** — the three commands, the rule that a missing artifact yields `unproven`
   rather than a stale number, and how to watch a run on the page with its measured cost.

---

## 3. TASK 3 — THE PAGE

The dev server on **:5199 (PID 94131)** was neither started, stopped nor written to. Confirmed once at
the start of the task and never touched again:

```
$ lsof -i :5199 | head -5
COMMAND     PID     USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
Google    16655 user   38u  IPv6 0x87211a73908a0dd0      0t0  TCP localhost:62455->localhost:5199 (ESTABLISHED)
node      94131 user   21u  IPv6 0x59341bf9aa4fa01c      0t0  TCP localhost:5199 (LISTEN)
node      94131 user   25u  IPv6 0x46ffd970215f5964      0t0  TCP localhost:5199->localhost:62455 (ESTABLISHED)
```

Every measurement below ran through `tests/helpers/gpu.ts`'s own Vite server on its own port.

### 3.1 The presets: the page was showing a medium no gate is measured in

| preset | before | after |
|---|---|---|
| `tiny` (box 16³, the page's DEFAULT) | `W: 100` → **570 particles + 191 mineral = 761**, ρ_W = **0.024 σ⁻³** | W derived = 3277 → **3938 particles**, ρ_W = **0.800**, ρ_total 0.961 |
| `default` (box 30³) | no `start` at all → inherited `data/soup.json`'s `W: 10700` → **~15 000**, ρ_W = **0.396 σ⁻³** | W derived = 21 600 → **25 904 particles**, ρ_W = **0.800**, ρ_total 0.959 |

Both old densities are below the liquid threshold this project measured
(`data/soup.json`'s `solvent.basis` §1: 0.80 is the lowest density with no macroscopic void and
sub-Poissonian uniformity; `water-calibration-report.md` §3 measured everything above it unstable), and
both explicit-water gates only pass at 0.8. `broth-composition-report.md` §2 recorded that gap
honestly and it was never closed on the page.

The fix is **not a number typed into a preset**: `viewer/run-types.ts` gains
`LIQUID_SOLVENT_DENSITY = 0.8` with its measurement in the doc comment plus `liquidSolventCount(box)`,
and `run-control-panel.ts`'s `currentSizeSelection()` **overwrites** the count of every species
`data/soup.json` flags `solvent` with `liquidSolventCount(box)`. Consequences, both intended: the
medium tracks whatever box the user types, and the particle-scale control **no longer scales the
solvent** — doubling C/H/M must not double the water and walk the run into the regime measured as
unstable. The preview says so:

```
particles: 3938 (C:200 O:50 H:200 M:20 W:3277 K:191) · grid 5×5×5=125 · rho_solvent 0.800 (liquid 0.8, from box) · rho_total 0.961 σ⁻³
particles: 25904 (C:1500 O:500 H:1500 M:100 W:21600 K:704) · grid 10×10×10=1000 · rho_solvent 0.800 (liquid 0.8, from box) · rho_total 0.959 σ⁻³
```

**Neither preset was re-sized for cost, because measurement said neither needed it**: `tiny` runs at
**2155–2158 steps/s** at 3938 particles, i.e. inside the 2086–2302 steps/s the page's own comments
already record for the old ~13 100-particle preset. `default` at 25 904 carries the word **expensive** in
its label. `STEP_BATCH = 250` at 2155 steps/s is 0.12 s per tick, far inside the 15 s watchdog.

### 3.2 The cold-start minimisation the page never ran

`soup/src/soup-relax.ts`'s `relaxColdStart` is opt-in by construction and refuses at
`globalStep != 0`. Every campaign passes `--relax`; the page did not. At the token water counts that
was survivable, at 0.961 total density it is not — the guard test's own payoff pair measures the same
start throwing at step 1000 with 29 199/29 232 non-finite components without it. Now `startRun()`
calls it and prints what it did:

```
[start] box=16×16×16 target=vesicle step_limit=20000 clay=yes (NOT in the published measurements)
[minimization] iterations=200 max|F| 3.626e+4 -> 1.392e+1 displacement_bound=10.050σ non-finite 0/0
```

### 3.3 The layout bug that silently ate the START click — found by measurement, fixed at the class level

The first headless run of the page after §3.1's preview grew from one line to three **never started**:

```
POLL 0 t=19ms state=idle steps=0 err=-
...
POLL 39 t=58625ms state=idle steps=0 err=-
TRACE:
(empty)
```

No error, no exception, no failed run — just nothing. The ground truth, once asked for directly:

```
CLICKTARGET [{"id":"start-btn","top":461,"bottom":492,"hitId":"visibility-note"},
             {"id":"pause-btn","top":461,"bottom":492,"hitId":"visibility-note"},
             {"id":"stop-btn","top":461,"bottom":492,"hitId":"visibility-note"}]
```

`#visibility-note` is bottom-anchored, spanned the full width, comes later in the DOM at the same
`z-index`, and at the 800×600 headless viewport it sat exactly over the buttons. `viewer/run.html`'s
own CSS comment records this class of failure being hit **twice before** and bought off each time with
one more row of clearance on `#cavity`'s fixed offset. This time it is fixed at the class level: the
three bottom notes moved to `left: 288px` — **clear of the 260 px control column entirely** — and
stacked in one `#notes` flex column instead of three hand-maintained `bottom` offsets (which, once
narrowed, made each note taller and overlapped each OTHER, visible in this task's own screenshot).
`#cavity`/`#aggregates` offsets bumped 525→620 / 735→830 for this task's extra row plus headroom.

After:

```
CLICKTARGET [{"id":"start-btn","top":461,"bottom":492,"hitId":"start-btn"},
             {"id":"pause-btn","top":461,"bottom":492,"hitId":"pause-btn"},
             {"id":"stop-btn","top":461,"bottom":492,"hitId":"stop-btn"}]
```

and the same assertion is now in `tests/run-ui.test.ts`'s ocean-look test, because every other
assertion in that file happens AFTER a click it assumes landed:

```
RUN-UI CLICKTARGET [{"id":"start-btn","hitId":"start-btn"},{"id":"pause-btn","hitId":"pause-btn"},{"id":"stop-btn","hitId":"stop-btn"}]
```

### 3.4 The honesty notes

The summary line was only about the time axis. It now reads
"**What this model cannot do: no closed vesicle was obtained; no electrostatics; salt and pH are not
representable; steps are reduced τ, not seconds. ⓘ more, the full list with numbers.**" and the full
text carries six numbered items: (1) the project's verdict with the campaign's own numbers (2077–2211
amphiphiles, encapsulated water 0 against 317.1–320.9, `closed = false`, 1 head shell instead of 2,
3 of 3 wrapping axes, supply and size retired at 2.10×/2.08×); (2) no electrostatics at all, and the
head mapped to MARTINI's uncharged Na rather than the charged carboxylate Qa; (3) salt and pH declared
unrepresentable and not faked, with the real fatty-acid window named; (4) evaporation as solvent
removal, **3.20× concentration against the literature cycle's 1400×** (16.07 % of it by logarithm),
plus the **~691×** carbon-pool enrichment against the most generous literature pond (~15 mM decanoic
acid, ACS Earth Space Chem. 2023, PMC9869395); (5) the **rank-D list** — the carbon→bead mapping,
reaction rates, the 0.8 σ⁻³ solvent density, both clay surfaces (a bracket, not a mineral),
`epsilonScale = 1.0` — with pair-depth ratios at rank C and ε/w_c at rank A; (6) the original
measured-vs-estimated paragraph, kept verbatim including κ_t.

### 3.5 The stage readout

The per-aggregate cards printed `cavityVolume` — the VACUUM-cavity detector, which by
`soup/src/water-closure.ts`'s own header reads a water-filled interior as empty exactly as it reads a
true vacuum, and never the observable the verdict rests on. Two rows added: **encaps. water / threshold**
and **closed**, rendering all three honest states distinctly (`not measured` when no water indices
were supplied, `centre untrusted` when the detector REFUSES because the aggregate's periodic centre
cannot be trusted, and `count / threshold` when measured). Verified on a real run — this is the
readout of the page's own 20 000-step run, at the moment it stopped:

```
#1, amph. 1 · flatness λ0/λ2 0.000 · in-plane λ1/λ2 0.008 · head shells (radial) n/a ·
head shells (transverse) 0 · cavity, σ³ 0.000 · encaps. water / threshold  centre untrusted ·
closed  n/a (centre untrusted)
```

### 3.6 The clay platelet

**It draws.** Measured, not assumed — `sceneDebug.instanceCounts()` (new test hook) reads the counts
straight off the meshes `draw()` writes:

```
RUN-UI INSTANCES {"C":{"count":200,"visible":true,"capacity":3938},"O":{"count":50,...},
"H":{"count":200,...},"M":{"count":20,...},"W":{"count":0,"visible":false,"capacity":3938},
"K":{"count":191,"visible":true,"capacity":3938}}
```

and the screenshot shows it as a matte grey-olive slab, not a sphere and not a silhouette. What was
genuinely wrong is different and worse: **the platelet was in every run this page could start and in
NO measurement this project published** — every arm of all eight campaigns ran `clay: false`
(`final-campaign-report.md` §12). So it now has its own labelled control,
"Clay platelet (K), NOT in the published measurements", wired into `createSoup`'s `clay` option, into
the preview, into the run's own trace line, and locked during a run like every other input. The
preview agrees with what will be built, both ways:

```
CLAY checked=false :: particles: 3747 (C:200 O:50 H:200 M:20 W:3277 K:0)   · rho_total 0.915 σ⁻³
CLAY checked=true  :: particles: 3938 (C:200 O:50 H:200 M:20 W:3277 K:191) · rho_total 0.961 σ⁻³
```

`instanceCounts()` is read AFTER a frame has been drawn, deliberately: an earlier version of the
assertion sat right after `steps > 0` and failed with `K.count = 0` **on a platelet that was in fact
on screen** — the counts report what the last `draw()` wrote. That is recorded in the test.

### 3.7 Proof a run progresses and draws

The page's own default preset, its own step cap, started by clicking its own button in headless Chrome:

```
[start] box=16×16×16 target=vesicle step_limit=20000 clay=yes (NOT in the published measurements)
[minimization] iterations=200 max|F| 3.626e+4 -> 1.392e+1 displacement_bound=10.050σ non-finite 0/0
step=250   stage=monomers amph=0.0000 agg=0.0000 peaks=n/a volume=0.0000 cavities=0
step=13000 stage=monomers amph=0.0150 agg=1.0000 peaks=n/a volume=0.0000 cavities=0
step=19750 stage=monomers amph=0.0150 agg=1.0000 peaks=n/a volume=0.0000 cavities=0
[stopped] step limit reached

state=stopped steps=20000 sps=2155.17 error=null trace=27 frames=249 nonBackground=1
SCREENSHOT bytes=175632   WALL 9581 ms
```

The chemistry moves (amphiphile fraction 0 → 0.0150, one aggregate appears at step 13 000), the stage
stays `monomers` because 0.0150 is under the ladder's own untouched 0.03 threshold — correct, not a
bug, at a 16 σ box with 50 heads — 249 real frames were rendered, and the screenshot is 175 632 bytes
of actual content.

---

## 4. REGRESSIONS

| test file | result | notes |
|---|---|---|
| `tests/run-ui.test.ts` | **8 passed** (all 8, as required) | 29.7 s. Two assertions ADDED (§3.3 click target, §3.6 instance counts), none relaxed. Test count unchanged at 8. |
| `tests/gates.test.ts` | **8 passed** (was 5) | 3 new tests, §1.5. Nothing widened. |
| `tests/water-bilayer-area-move.test.ts` | **1 passed** | 199.8 s; verdict `passed`; now also writes its artifact |
| `tests/continuous-run-audit.test.ts` | **1 passed** | 5 checkpoints at N = 191 778, `RUN-AUDIT-TETHER violations=0` |
| `tests/percolation-check.test.ts` | **1 passed** | 5 checkpoints incl. 2 must-say-no controls |
| `tests/params.test.ts` | **3 passed** | the literal scanner: **no rank-A constant appears anywhere new**. `LIQUID_SOLVENT_DENSITY = 0.8` lives in `viewer/`, outside the scanned dirs (`engine/src`, `engine/wgsl`, `chem/src`, `soup/src`, `soup/wgsl`) — and it is a page-level derivation of a `data/soup.json`-documented measurement, not a physics constant |
| `tests/soup-rules.test.ts` | **4 passed** | `data/soup.json` schema unchanged |
| `tests/soup-equilibrium.test.ts` | **14 passed** | the ASF/Flory helpers the chain gate reads |
| `npx tsc --noEmit` | **17 errors, all pre-existing** | identical count before and after; every one in `engine/src/{forces,gpu,params,sim-area-move,sim-buffers,sim-measure}.ts` and `soup/src/{sim,soup-box-scale,soup-buffers}.ts` (`@webgpu/types` vs TS-lib `Float32Array<ArrayBufferLike>`). **Zero errors in any file this task touched.** |

Known-flaky, deliberately not run and not chased (all three proven pre-existing by predecessors):
`tests/soup-drywet-cycling.test.ts`, `tests/soup-grid-resize.test.ts`, 2 of 3
`tests/rim-lambda-insitu.test.ts`. **No bound was widened anywhere.**

---

## 5. RESOURCES AND HOUSEKEEPING

- **12 compute invocations**, all `nice -n 15`, one at a time, **foreground**, none backgrounded:
  5 viewer probes (one of which timed out at the harness's 120 s default and was re-run with an
  explicit timeout; two returned the `state=idle` diagnosis that found §3.3's bug), 3 `run-ui` runs
  (the middle one caught my own assertion in the wrong test), 1 water-bilayer measurement, 1 combined
  audit+percolation+gates run, 1 `npm run verify`, 1 CPU regression batch.
- **Longest single invocation: 200 s** (`water-bilayer-area-move`), then **~170 s** (`npm run verify`,
  of which 145 s is the κ scenario). Every one under the 500 s cap.
- **No orphan processes remain.** `pkill -f puppeteer_dev_chrome_profile` after every browser-bearing
  invocation, and `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** every time, including
  at the end of the task.
- **No campaign started.** `tests/soup-vesicle.test.ts` never run. The full suite never run.
  `--dump-dom` never used. macOS `timeout` never invoked. No `node_modules/.vite/deps_temp_*` cleanup
  was needed (no vitest died mid-run). No particle array was ever transferred as JSON numbers.
- **Scratch deleted**: `verify/tmp-viewer-probe.ts` and its two screenshots, same precedent as
  `tests/tmp-water-calib-probe.test.ts` (`water-calibration-report.md` §7). Its numbers are in §3.
- **File sizes, CLAUDE.md's 400–600 rule.** Nothing crossed 600 and nothing was pushed over:
  `viewer/run-control-panel.ts` 414 → **485**, `viewer/run-types.ts` 286 → **330**,
  `viewer/run-readout.ts` 174 → **229**, `viewer/run.html` 199 → **226**,
  `tests/water-bilayer-area-move.test.ts` 458 → **507**, `verify/gates.ts` 88 → **111**,
  `tests/gates.test.ts` 51 → **113**. New: `verify/campaign-gates.ts` **285**,
  `docs/soup-to-vesicle-verdict.md` **363** (prose, not a code file — the 400–600 rule is about source modules). Largest file in the tree is still `soup/wgsl/step.wgsl`
  at **596**, untouched.
- **`data/params.json` NOT touched. `data/soup.json` NOT touched, not one field.** No threshold,
  corridor, potential, rate, recogniser or rank-A constant modified anywhere.

---

## 6. CONCERNS

1. **The explicit-water gates moved by more than I would like on a re-measurement with nothing
   changed**: thickness 4.7990 → 4.5311 (−5.6 %), head burial 0.0857 → 0.1608 (+88 %), chunks to
   settle 300 → 450. Both gates still pass their untouched corridors and the direction is inside the
   scatter two predecessors measured, but a gate whose diagnostic doubles between identical runs is a
   gate resting on one seed, and neither this task nor its predecessors ever ran it more than once per
   task. The published number is one draw from a distribution nobody has characterised.
2. **The chain-length gate publishes 15.2 % disagreement and calls it unproven, which reads as
   evasive and is not.** The ASF mean from the event ratio (4.020) sits well above the measured mean
   per tail (3.490), and the honest reason is structural, not statistical: chains here are already 2
   carbons long before a head can cap them (`propagateOnCenter` nucleates a PAIR), so the n = 1 bin
   the ASF frame predicts cannot exist and the measured mean is biased low by construction. That is a
   known limitation of the mapping, not a fresh finding, and it is exactly why no corridor can be
   invented for this gate — but a reader could reasonably say the deviation should be quoted against a
   truncated ASF instead. I did not build that, because building it inside a delivery task would be
   new science.
3. **`mean-tail-length` = 3.490 against a 2–3 window is published as `unproven` on a rank-D
   technicality, and it would be a FAILURE if the mapping carried an independent corridor.** The
   document and the gate's own `conditions` both say so in as many words, but the table's verdict
   column says "unproven", and a reader who skims only that column will read this as neutral when it
   is not.
4. **The campaign rows rest on 5 re-audited checkpoints, not 20.** Every number reproduced the
   predecessor's to the digit, the tether and invariant assertions all ran, and the closure gate takes
   the LAST reportable checkpoint rather than the friendliest — but the plateau statistics quoted in
   the summary document (2087.2 ± 7.7 over 15 wet samples, cavity 50.5 ± 1.6 over 8) come from
   `final-campaign-report.md`'s own 20-checkpoint trace, not from this task's 5. They are cited as
   that report's numbers, and `verify/out/zfix-campaign-B54-trace.json` is still on disk, so nothing
   is unverifiable — but the gate table and the prose draw on two overlapping traces rather than one.
5. **The percolation gate reduces a 3-axis boolean to `wrappingAxes: 3` against `{max: 0}`.** That is
   the honest summary, and the per-axis booleans and slab counts survive in `gates.json`'s `campaign`
   block — but a future edit that starts publishing, say, only `wrapsZ` would look identical in the
   verdict column. The gate has no guard against being narrowed that way; only its `conditions` text
   describes what the number is.
6. **`viewer/run-control-panel.ts` is at 485 lines and gained a control this task.** Inside the
   400–600 band, so no split was required, but it is the file that will cross 600 next, and the next
   person adding a row to `#controls` should split it first rather than after.
7. **The bottom notes now sit to the right of the control column, which narrows them.** With the
   honesty note's full text expanded on a narrow window that column will be tall and will cover a lot
   of the scene. It is collapsed by default and the summary line carries the verdict, so nothing
   important is hidden — but the trade I made is "notes never eat a click" against "notes can cover
   more of the view when opened", and I made it deliberately in that direction.
8. **The `default` preset is now 25 904 particles and I never started it.** Its cost is stated in its
   own label and its neighbour grid previews cleanly at 10×10×10, and the guard that refuses an
   impossible selection is unchanged — but the only preset this task actually ran on the page is
   `tiny`. If 25 904 particles behaves differently at cold start on a busier machine, nobody has
   measured it, and the honest thing is that the label says expensive rather than that the number is
   proven safe.
9. **The clay control ships default-ON, matching `data/soup.json`, which means the page still starts
   runs in a configuration no published measurement used.** I chose to keep the file's own composition
   as the default and make it visible rather than to silently flip the page's default to the campaigns'
   `clay: false`, because flipping it would make the page disagree with `data/soup.json` instead. Both
   choices are defensible and I would not argue hard for mine.
10. **Nothing in this task re-measured the solvent-free κ gate's failure.** It is published `unproven`
    with its own fresh invalid fit (slope −1.056), which is honest, but the bending modulus has now
    been unproven across every task in this project's history and no one has ever fixed the fit
    window. It is the one rank-A gate that has never produced a number, and it is quietly load-bearing:
    the edge-energy argument in §7.3 of the predecessor leans on κ's spread.

## 7. FILES

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/consolidation-report.md` — this report (`.gitignore`
  line 6 ignores `.superpowers/`, so committed with `git add -f`, like every predecessor).
- **NEW, committed and readable**: `docs/soup-to-vesicle-verdict.md` (task 2).
- **NEW**: `verify/campaign-gates.ts` — reads the three input artifacts, attaches provenance, turns a
  missing artifact into an `unproven` reason.
- MODIFIED: `data/literature.json` (7 gates added, 5 unchanged and reordered), `verify/gates.ts`
  (`provenance`/`note` + `GateContext`), `verify/run.ts` (wires the campaign inputs, publishes a
  `campaign` block), `verify/report.ts` (new "number source" column).
- MODIFIED, the page: `viewer/run-types.ts` (`LIQUID_SOLVENT_DENSITY`, `liquidSolventCount`, both
  presets, `instanceCounts` hook), `viewer/run-control-panel.ts` (solvent derived from the box, cold-
  start minimisation, clay control, preview), `viewer/run-readout.ts` (honesty note, encapsulated-water
  rows), `viewer/run-render.ts` (`instanceCounts` implementation), `viewer/run.html` (clay checkbox,
  `#notes` flex column, offsets).
- MODIFIED, tests: `tests/run-ui.test.ts` (2 assertions added, still 8 tests),
  `tests/gates.test.ts` (5 → 8 tests), `tests/water-bilayer-area-move.test.ts` (writes its artifact).
- REGENERATED artifacts: `verify/out/gates.json`, `verify/out/report.html`,
  `verify/out/kappa-measurement.json`. NEW artifacts:
  `verify/out/water-bilayer-area-move.json`, `verify/out/gates-campaign-trace.json`,
  `verify/out/gates-percolation.json`.
- `data/params.json` and `data/soup.json` **untouched**.
