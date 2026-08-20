# The decisive run — the missing control, the stale-F(x) defect, and the box-54 vesicle verdict

Task `decisive-run` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `evaporation-report.md`, whose own **concern 1** named the missing control ("a run that
receives exactly those 9 + 6 minimisations at the same global steps with **no cycling**; that is the
one control still missing") and whose **concerns 5 and 6** named two unfixed defects (a stale F(x)
after a box change, and an adsorption tether able to cross FENE's `r_inf` thermally). Those were
tasks 1 and 2 here; the box-54 run was task 3.

`data/params.json` was **NOT** touched. No threshold, corridor, potential, rate, recogniser or rank-A
constant was modified. `co_bond.attemptRate` untouched. `data/soup.json` was **NOT** touched at all
this task — not one field. No arm reverted to under-dense water: every arm in this report runs at
ρ_W = 0.800 σ⁻³ (measured 0.8023 in the decisive run's own closure readout). Nothing started from
anything pre-made: the decisive lineage begins at a monomers-only lattice (`RUN-AUDIT-START` shows
its step-3000 checkpoint carrying the run's own first 329 bonds and nothing inherited).

---

## Verdict up front, nine lines

1. **The stale-F(x) defect was real, it sat directly under the headline, and correcting it cut the
   cycling result by 2.28× on the one quantity the campaign is about.** With the fix the largest
   aggregate at box 30, step 36 800 is **109.3 [71, 154]** (n = 3) against **249.3 [210, 273]**
   (n = 3) on the identical arms with the fix stashed — non-overlapping ranges. Yield and bond count
   moved not at all (0.1516 → 0.1525; 1436 → 1392) (§2).
2. **The published 12.5–14.5× in largest aggregate is honestly 8.1–8.4×.** Yield 1.68–1.80×, bonds
   1.76–1.93×, largest aggregate **8.41×** against the minimisation-only control and **8.10×**
   against the plain-wet control (§1.4).
3. **Minimisation explains none of it.** The minimisation-only control (n = 3, one 29-iteration
   minimisation at global step 35 400, no cycle) gives largest aggregate **13.0 [12, 14]** — inside
   the plain-wet control's own 11–18 — yield 1.07× and bonds 1.09× of plain wet, both at the edge of
   the arms' combined scatter. **0 % of the aggregate effect, ≤ 9 % of the bond effect** (§1).
4. **And the defect was also manufacturing the ramp guard.** With F(x) correct, per-increment `max|F|`
   is **1.5e2–7.2e2** against the σ/(13·dt²) = 769 trigger, so the guard fired on **0 of 36
   increments** in every cycle-1 transition measured at box 30 and **0 of 36** at box 54. The
   predecessor's 9-of-216 guard minimisations were a consequence of the defect, not of the physics —
   which retires its own concern (c) about annealing (§2.3).
5. **The tether defect did not fire in any run kept.** Measured off-GPU on **every checkpoint of
   every lineage** (19 box-54 + 9 box-30 = 28 checkpoints), on ordinary wet dynamics and not only at
   box changes: **0 tethers over `r_inf` = 1.5 σ anywhere**, longest 1.1631 σ (decisive run),
   1.1778 σ (worst across all lineages), 224 live tethers at the end. Latent, still unfixed, now with
   a detector committed (§2.4).
6. **The floor moved DOWN, 1088 → 920, and the window is the widest this project has had.** From the
   re-measured explicit-water gate (t = 4.5472 σ, a = 1.2372 σ²): floor **920**; band 920–933, the
   tightest ever (the two independent gates now agree to 1.4 %). Window **L ∈ [46.55, 55.69] σ,
   width 9.14 σ** against the predecessor's 5.47 σ and its predecessor's 1.06 σ (§3.2).
7. **SUPPLY IS SOLVED.** The decisive run at box 54 measures a settled amphiphile supply of
   **1435.8 [1420, 1471]** against the floor 920 — **1.56× OVER**, where every predecessor was
   short. That constraint is retired (§3.3).
8. **NO VESICLE, and the curve was NOT rising when it stopped.** Largest aggregate over 133 000 steps
   of settled wet dynamics after the drying event: 153 → 149 → 143 → 138 → 138 → 139 → 141 → 136 →
   138 → 137 → 136 → 135 → **135**. Flat, then monotonically declining over the last five samples.
   `encapsulatedWater` = **0** against a threshold of **297.6**, `closed = false`,
   `cavityVolume = 1.75 σ³` against the 370.8656 σ³ closure minimum, `radialHeadShells` 1–2 (§3.4).
9. **BINDING CONSTRAINT, numbered: (2) AGGREGATE SIZE — COALESCENCE, 136.2 against 920, 6.75× short —
   and the number that names the mechanism is this: going from box 30 to box 54 multiplied SUPPLY by
   5.06× (283.7 → 1435.8) and the largest aggregate by only 1.24× (109.3 → 135).** The
   single-aggregate share collapses from **38.5 %** at box 30 to **9.5 %** at box 54: 1423
   amphiphiles sitting in 34 aggregates. The predecessor's own "whole remaining question" — whether
   69 % of amphiphiles coalesce into one aggregate at L ≈ 55 as they appeared to at L = 30 — is
   answered **NO** (§4). Shortfall sequence: 80× → 8.9× → 3.36× → 33.4× → 6.76× → **6.75×**.

---

## 1. TASK 1 — the minimisation-only control

### 1.1 What had to be isolated, and how

The predecessor compared cycling against a control that did **not** receive the same energy
minimisations. A minimisation removes overlaps, and could plausibly deliver part of the benefit on
its own. Isolating it needs an arm that takes exactly the cycled arm's minimisations, at exactly the
cycled arm's global steps, with **no evaporation cycle**.

That needed one new capability, because the existing minimiser refuses to run mid-trajectory by
design (`relaxColdStart` throws at `globalStep != 0`, and that refusal is what makes the cold-start
stage provably outside every measurement). So `SoupSystem.minimiseNowDEBUG(iterations)` was added:
**the same function** the cold start, the solvent insertion and the evaporating ramp's guard all use
(`soup/src/soup-relax.ts`'s `relaxIterations`) — not a second minimiser — applied at the current box
with no box change and no solvent movement. It is reachable only from `soup/cli/campaign.ts`'s new
`--minimiseAt` flag and from `tests/soup-stale-force.test.ts`. Its neutrality is pinned by test, not
asserted: it advances no step counter, changes no box, and leaves census, charge, bond set and all
four event counters identical (§5, `MIDRUN-MINIMISE`).

### 1.2 How many minimisations, and at which step — measured from the cycled arm, not assumed

With the stale-F(x) fix in place the cycled arm's cycle 1 receives **exactly one** minimisation: the
ramp guard fires on 0 of its 36 box increments (§2.3), and the solvent re-insertion minimises once,
29 iterations, at the end of the rehydration ramp. Global step of that event:
`32 000 + (18 − 1)·200 = 35 400`. So the control is: `--minimiseAt 35400 --minimiseIterations 29`,
no `--cycle`, no `--evaporate`. 29 is not chosen — it is `EvaporationPlan.relaxIterations`, derived
from the composition (`planEvaporation`), and printed by the cycled arm's own log line.

### 1.3 The three observables, three arms, with scatter

Box 30, `{C: 1860, O: 7440, H: 1860, M: 124, W: 21600}`, N = 32 884, ρ_tot = 1.21793, ρ_W = 0.800,
kT = 1.1, `--relax`, `clay: false`. Seeds 19/23/29. All samples at **global step 36 800**, audited
off-GPU (`verify/out/decisive-arms-box30-trace.json`). "Yield" is the predecessor's own definition —
amphiphile count / total carbon — so these numbers are directly comparable with its tables; the
audit's `amphiphileFraction` (carbon-in-amphiphile / total carbon) is given beside it.

| arm | n | yield (amph/C) | cc+co bonds | **largest aggregate** | aggregates | `amphiphileFraction` |
|---|---|---|---|---|---|---|
| **plain wet** (predecessor's committed artifact, step 34 000, seeds 19/23) | 2 | 0.08495 [0.08387, 0.08602] | 722.5 [707, 738] | **13.5 [12, 15]** | 53–54 | 0.2078 |
| **minimisation-only control** (1 × 29 iterations at step 35 400, no cycle) | 3 | **0.09086 [0.08172, 0.09839]** | **788.7 [752, 823]** | **13.0 [12, 14]** | 47–53 | 0.2263 |
| **cycled + real evaporation, post-fix** | 3 | **0.15251 [0.14194, 0.16828]** | **1392 [1353, 1448]** | **109.3 [71, 154]** | 5–7 | 0.3636 |
| cycled + real evaporation, pre-fix (fix stashed) | 3 | 0.15161 [0.14892, 0.15376] | 1436 [1401, 1466] | 249.3 [210, 273] | 3–5 | 0.3502 |

Per-seed, so the scatter is not hidden behind a mean:

| seed | min-only largest | min-only yield | cycled(post-fix) largest | cycled(post-fix) yield | cycled(pre-fix) largest |
|---|---|---|---|---|---|
| 19 | 14 | 0.09839 | 103 | 0.16828 | 265 |
| 23 | 12 | 0.08172 | 71 | 0.14731 | 273 |
| 29 | 13 | 0.09247 | 154 | 0.14194 | 210 |

### 1.4 Does minimisation explain a material share? No — and the cycling effect restated

Minimisation-only against plain wet: **yield 1.07×, bonds 1.09×, largest aggregate 0.96×.**

- On the binding quantity — largest aggregate — the minimisation contributes **nothing**: 13.0
  against 13.5, and every one of the three min-only values (12, 13, 14) sits inside the published
  plain-wet control's own range across its whole 204 000-step trace (11–18).
- On bonds the 1.09× is real in sign but is **at the edge of the arms' combined scatter** (plain wet
  707–738 against min-only 752–823) and cannot be attributed with n = 2 vs n = 3. Stated as an upper
  bound: **≤ 9 %**. It is *not* the 2800-step offset: the published control's own bond count drifts
  707 → 961 over 170 000 steps, i.e. ~4 bonds per 2800 steps, against the 66-bond gap.
- On yield the 1.07× carries the same caveat, ≤ 7 %.

**The cycling effect at its honest size** (post-fix cycled, n = 3, step 36 800):

| measure | vs minimisation-only control | vs plain-wet control | **published claim** |
|---|---|---|---|
| yield | **1.68×** | **1.80×** | 1.40–1.89× |
| cc+co bonds | **1.76×** | **1.93×** | 1.96–2.12× |
| **largest aggregate** | **8.41×** (5.1×–12.8× per seed) | **8.10×** | **12.5–14.5×** |

So: the missing control does **not** explain the effect away, and the effect on yield and bonds
survives essentially intact. What shrinks the aggregate multiplier from 12.5–14.5× to ~8.4× is not
the minimisation — it is task 2's defect.

---

## 2. TASK 2 — the stale-F(x) defect, and the tether defect

### 2.1 What the defect was

`soup/src/soup-box-scale.ts`'s `applyBoxScaleOnce` applies a box change by rescaling every molecule's
centre of mass, then rebuilds the neighbour grid and the Verlet list — and stopped there.
`soup/src/soup-integrate.ts`'s `step()` opens with `kick_drift_wrap`, which consumes `forceBuf` as
F(x_n). So the first half-kick after **every** box change applied a force computed for a geometry
that no longer existed. The evaporating transition worked around it with a per-increment `forces()`
readback; the shared path — the dry-wet ramp, `growBoxTo`, `scaleBoxTo`, and
`soup/src/soup-area-move.ts`'s accepted-chain application — did not.

### 2.2 The fix, and the pin that has teeth

The fix is three lines: the force dispatch is encoded in the same pass as the grid/Verlet rebuild, for
every caller. Cost is one dispatch per box change (not per step).

Observing it needed a new readback, because `SoupSystem.forces()` **recomputes** before reading back,
which makes a stale buffer invisible through it. `SoupSystem.forcesNoRebuildDEBUG()` reads `forceBuf`
exactly as it stands — the very F(x_n) the next kick will consume — and encodes no dispatch.

`tests/soup-stale-force.test.ts` (NEW, 2 tests) pins the claim in **both** directions, and was run
with the fix stashed to prove it is not vacuous:

```
$ nice -n 15 npx vitest run tests/soup-stale-force.test.ts --no-file-parallelism
STALE-FORCE {"n":20880,"box":[15,15,15],"maxPreFresh":195.05258178710938,"maxPostFresh":513.0130004882812,"preResidentVsFresh":0,"postResidentVsFresh":0.000152587890625,"postResidentVsPreFresh":444.15340995788574}
 ✓ tests/soup-stale-force.test.ts (2 tests) 8937ms

$ git stash push -m "staleforce-fix-probe" -- soup/src/soup-box-scale.ts
$ nice -n 15 npx vitest run tests/soup-stale-force.test.ts --no-file-parallelism -t "изменение бокса"
STALE-FORCE {"n":20880,"box":[15,15,15],"maxPreFresh":204.99313354492188,"maxPostFresh":518.7139892578125,"preResidentVsFresh":0,"postResidentVsFresh":313.7208557128906,"postResidentVsPreFresh":0}
  × box change must leave F(x_n) for the NEW geometry
    → expected 313.7208557128906 to be less than 5.187139892578125
$ git stash pop
```

| | resident F vs fresh F at the NEW box | resident F vs the PRE-change F |
|---|---|---|
| **with the fix** | **1.5e-4** against a force scale of 513 (3e-7 relative) | **444** — emphatically not the old force |
| without the fix | **313.7** — a third of the force scale, wrong | **0** — bit for bit the old force |

The `0` in the bottom-right is the whole defect: before the fix the buffer the next kick consumed was
*exactly* the pre-remap force. `preResidentVsFresh = 0` in both runs shows the resident buffer does
track a fresh compute when no box change intervened, so the comparison isolates the box change.

### 2.3 Re-measured cycling benefit after the fix — the number shrank, so it is reported as the result

Identical arms, identical flags, identical seeds, the only difference being the three lines of §2.2:

| | largest aggregate @36 800 (n = 3) | yield | cc+co |
|---|---|---|---|
| pre-fix (fix stashed) | **249.3 [210, 273]** | 0.15161 | 1436 |
| **post-fix** | **109.3 [71, 154]** | 0.15251 | 1392 |
| ratio | **0.44× (a 2.28× shrink)** | 1.006× | 0.97× |

The ranges do not overlap (210–273 against 71–154). Yield and bond count are unchanged inside their
own scatter. **The defect was inflating exactly one quantity — aggregate size — and by ~2.3×.**

The mechanism is visible in the ramp's own logs and is the second half of the finding. A spurious
half-kick at each box increment injects momentum (`dv = F·dt/2` with F wrong by ~300–450 in these
units), which acts as repeated local annealing and accelerates coalescence. With F(x) correct the
per-increment force spikes fall by ~5×:

| | per-increment `max|F|` | guard firings (σ/(13·dt²) = 769) |
|---|---|---|
| predecessor's published arm | 0.78e3 – 2.72e3 | 9 of 216 |
| **post-fix, box 30, cycle 1** | **2.8e2 – 5.1e2** | **0 of 36** |
| **post-fix, box 54, cycle 1** | **3.8e2 – 7.2e2** | **0 of 36** |
| pre-fix, box 30, cycle 1 (this task) | 3.0e2 – 3.9e2 | 0 of 36 |

So the predecessor's ramp-guard minimisations were themselves produced by the defect: with a correct
integrator the evaporating ramp needs **no minimiser at all**, and the only minimisation left in a
whole cycle is the one the solvent insertion structurally requires. That retires the predecessor's own
concern (c) — "the ramp guard minimiser could have annealed the structure into place" — by removing the
minimiser rather than by bounding it.

`tests/soup-drywet-cycling.test.ts`, which pins the pre-existing box-scaling-only path, still passes
with `concentrationFactor = 1.1002` and `rampStepsCharged = 1000` identical, and its one-cycle event
ratio moved **0.4978 → 0.7171**, i.e. back to within 0.4 % of the value originally published (0.7143)
before the defect perturbed it (§5).

### 2.4 The second named defect — the tether crossing `r_inf`

An adsorption tether (`centerLink`) can be stretched past FENE's `r_inf` by ordinary thermal motion
between two bond dispatches (`bondAttemptInterval` = 20 real steps), and past `r_inf`
`fene_dv(r) = k·r/(1 − (r/r_inf)²)` changes **sign** and diverges. The predecessor measured it only at
box changes (its own valve, 0 firings in 216 increments) and left it "latent, not demonstrated".

A detector was added to `tests/continuous-run-audit.test.ts` — off-GPU, on every checkpoint, on
**ordinary wet dynamics** and not only at box changes: the longest live tether against `r_inf`,
recorded per checkpoint and asserted after the artifact is written so a violation lands in the trace
rather than destroying it. Across **28 checkpoints of 4 lineages**:

```
RUN-AUDIT-TETHER нарушений=0
final decisive checkpoint: {"live":224,"longest":1.1046,"rInf":1.5,"overRInf":0}
```

Longest tether anywhere: **1.1778 σ** (box-30 seed 29) against `r_inf` = 1.5 σ — a 1.27× margin. Over
the decisive run's 19 checkpoints the longest ranges 1.0847–1.1631 σ. **It did not fire in any run
kept, in any phase, at either box.** Status: still **unfixed** and still latent — the cure is a
per-step `desorbStretch` check or a clamp on `fene_dv` past `r_inf`, both engine changes — but it is
now measured on the trajectory rather than only at box boundaries, and the detector is committed so
the next run cannot miss it.

---

## 3. TASK 3 — the decisive run

### 3.1 Box, composition, water — the arithmetic

Box **54 σ**, cube. Chosen at the task's stated minimum and inside the recomputed window, for a
reason that is a measurement and not a preference: the number this campaign has to beat — the 588
still-rising aggregate — was measured at box 54, so 54 makes the comparison direct. It is also
2.24× inside both hard memory ceilings.

```
L = 54            V = 157 464 σ³
ratios kept exactly as the defended liquid broth's (C:O:H:M = 15:60:15:1)
M = round(124 · 5.832) = 723  ->  C = H = 15·723 = 10 845,  O = 60·723 = 43 380
W = round(0.800 · 157 464) = 125 971
N_org = 65 793   N = 191 764
rho_tot = 191 764 / 157 464 = 1.21783   (box-30 broth: 1.21793)
rho_org =  65 793 / 157 464 = 0.41783   (box-30 broth: 0.41793)
rho_W   = 125 971 / 157 464 = 0.79 999 87   -- LIQUID, and measured live at 0.8023 by the
                                              closure readout's own bulkWaterDensity
```

Evaporation plan, from the same `residualSolventFraction = 1/1400`:
```
W_residual = round(125 971/1400) = 90
N_dry = 65 883      dryBox = (65 883/1.34)^(1/3) = 36.6344 sigma   (realised rho 1.34000)
concentration factor = 157 464 / 36.6344^3 = 3.2027x   (box 30 gave 3.2021x -- same physics)
grid at the dry box: 36.6344 / 2.94695 = 12.43 -> 12 cells, against minCells = 5. VALID.
ladder: 18 increments, 2.179 % linear each
Verlet: 191 764 x 2500 x 4 = 1.9176 GB against 4.2950 GB  (2.24x)
particles: 191 764 against the measured hard ceiling 429 496  (2.24x)
```

### 3.2 The window and the floor, recomputed from THIS task's measured thickness and area

Both gates were re-measured on the fixed code (they are regressions of §5, and the fix touches the
area move's own box application, so their numbers had to be re-taken rather than reused):

```
$ nice -n 15 npx vitest run tests/gate6-bilayer.test.ts --no-file-parallelism
GATE6 area 1.2075 +/- 0.0073 (min 1.1912, max 1.2235)  thickness 4.4762 (per-frame mean 4.4817, sd 0.2173, n=400) ...

$ nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=450 areaPerLipid(MEASURED, tail mean)=1.2372 [min 1.2325, max 1.2413, corridor 1.1-1.5] driftPerChunk(lnA)=7.891e-6 t=1.38 thickness(measured)=4.5472 [corridor 4-6] clusterFraction=1.0000 waterInCore=50/4500 headBuriedFraction=0.0870 acceptedFraction=0.1151 of 4500 trials throughput=1212.08 steps/s at N=5700 verdict=passed
```

Same formula as `continuous-run-report.md` §1.1; no new constant; `stageThresholds.enclosedVolume`
= 370.8656 σ³ untouched.

```
R_in  = (3 x 370.8656 / 4pi)^(1/3) = 4.45700 sigma
R_mid = R_in + t/2
floor = 2 x 4pi x R_mid^2 / a
```

| basis for t, a | t σ | a σ² | R_mid σ | **floor** |
|---|---|---|---|---|
| **explicit-water gate, measured here** | **4.5472** | **1.2372** | **6.73060** | **920.3 → 920** |
| solvent-free gate 6, measured here | 4.4762 | 1.2075 | 6.69510 | 933.0 → **933** |
| the predecessor's explicit-water basis | 4.9619 | 1.1124 | 6.93795 | 1087.5 → 1088 |

**The floor moves 1088 → 920, DOWN by 168 (−15.4 %)**, and both inputs move toward each other rather
than one dominating: t 4.9619 → 4.5472 (−8.4 %) and a 1.1124 → 1.2372 (+11.2 %), both inside their
untouched corridors (4–6 and 1.1–1.5), both passing. Band **920–933** — the two independent gates now
agree to **1.4 %**, against 14.4 % apart in the predecessor (931 vs 1088); the explicit-water area has
converged onto the solvent-free one (1.2372 vs 1.2075) where before it sat 8.7 % below it. That
convergence is the area move's own box change no longer handing the following steps a wrong force.
**920 used**, the explicit-water gate, the same choice the original derivation made.
Trend across five tasks: 960 → 912 → 968 → 1088 → **920**.

Window, with ρ_amph measured **at the box the window is about** rather than extrapolated from box 30
(9 settled wet samples of the decisive run itself, steps 90 400–174 400: amph = 1471, 1453, 1443,
1432, 1431, 1422, 1420, 1427, 1423 → mean **1435.8**, ρ_amph = 1435.8/157 464 = **9.1181e-3 σ⁻³**):

- **supply**: ρ_amph·L³ ≥ 920 → L ≥ (920/9.1181e-3)^(1/3) = **46.55 σ** (with the 933 end: 46.77 σ)
- **measurability**: R(L) = 0.067·L^1.5 ≤ L/2 → L ≤ **55.69 σ** (box-58 head-radius calibration, unchanged)

**Window L ∈ [46.55, 55.69] σ — width 9.14 σ.** The predecessor's was 5.47 σ, its predecessor's
1.06 σ. At L = 54 the measured supply is **1435.8 against 920 — a 1.56× margin**, where the
predecessor ran at 1.08× and predicted 1.31×.

### 3.3 Schedule — one drying event, justified from the trend and not by preference

The predecessor measured that the gain does **not** compound: largest aggregate 189 (cycle 1) → 169 →
162 → 168 → 166 → 160, with the yield declining **monotonically** 0.1624 → 0.1242 as `co_break`
accumulated 9 → 66. So cycles 2–6 buy nothing measurable on aggregate size and cost yield. The
schedule that trend justifies is therefore **exactly one** wet→dry→wet cycle — the event that
delivered everything — which also maximises yield, since yield is highest at the first rehydration.

Expressing that needed a flag rather than a data-file edit, because `data/soup.json`'s cycle count is
read by every other lineage: `--cycles 1` (and `CreateSoupOpts.dryWetCycles`), an experiment-design
number in exactly the sense `--steps` and `--every` already are. It is part of the resume signature —
a one-cycle and a six-cycle run are two experiments — added by conditional spread so a run that
passes neither new flag produces a **byte-identical** signature and no existing checkpoint lineage is
orphaned. Period (32 000), dry fraction (0.65), target dry density (1.34) and both ramps are the
file's own, untouched: the drying event is byte-for-byte the one the trend measured.

### 3.4 The stage trace — the full run, audited off-GPU

`verify/out/decisive-run-54-trace.json`, 19 checkpoints, produced by `tests/continuous-run-audit.test.ts`
(the same functions the live run's own stage detection uses — there is no second implementation of any
measurement here). "share%" is the largest aggregate as a percentage of the whole amphiphile supply.

| globalStep | box | stage | amph | amph/C | perTail | cc+co | aggs | **largest** | share% | flat | inPl | radSh | cav σ³ | encH₂O/thr | closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 000 | 54.0000 | monomers | 153 | 0.0141 | 2.007 | 329 | 125 | 5 | 3.3 | 0.021 | 0.076 | n/a | 0 | n/a | n/a |
| 6 000 | 54.0000 | amphiphiles | 315 | 0.0291 | 2.056 | 793 | 213 | 7 | 2.2 | 0.035 | 0.158 | n/a | 0 | 0/296.7 | false |
| 15 400 | **36.6344** | amphiphiles | 1150 | 0.1060 | 2.044 | 3738 | 23 | *1107* | *96.3* | *0.866* | *0.908* | 1 | 0.875 | — | — |
| 21 400 | **36.6344** | amphiphiles | 1574 | 0.1451 | 2.044 | 5936 | 8 | *1567* | *99.6* | *0.873* | *0.905* | 1 | 1.875 | — | — |
| 24 000 | **36.6344** | amphiphiles | 1615 | 0.1489 | 2.045 | 6495 | 8 | *1599* | *99.0* | *0.921* | *0.932* | 1 | 2.375 | — | — |
| 31 000 | **36.6344** | amphiphiles | **1649** | **0.1521** | 2.056 | 7404 | 8 | *1632* | *99.0* | *0.884* | *0.942* | 1 | 4.5 | — | — |
| **41 400** | 54.0000 | micelles | 1607 | 0.1482 | 2.124 | 8836 | **31** | **153** | **9.5** | 0.152 | 0.209 | 1 | 0.375 | **0/297.5** | **false** |
| 55 400 | 54.0000 | micelles | 1563 | 0.1441 | 2.209 | 9566 | 31 | 149 | 9.5 | 0.172 | 0.193 | 1 | 1.625 | 0/297.6 | false |
| 69 400 | 54.0000 | micelles | 1520 | 0.1402 | 2.250 | 9939 | 30 | 143 | 9.4 | 0.172 | 0.199 | 1 | 2.125 | 0/297.5 | false |
| 79 900 | 54.0000 | micelles | 1488 | 0.1372 | 2.268 | 10145 | 32 | 138 | 9.3 | 0.209 | 0.243 | 1 | 1.5 | 0/297.5 | false |
| **90 400** | 54.0000 | micelles | 1471 | 0.1356 | 2.295 | 10290 | 30 | **138** | 9.4 | 0.211 | 0.230 | 1 | 1.25 | 0/297.6 | false |
| 100 900 | 54.0000 | micelles | 1453 | 0.1340 | 2.309 | 10433 | 31 | 139 | 9.6 | 0.207 | 0.227 | 1 | 1.625 | 0/297.6 | false |
| 111 400 | 54.0000 | micelles | 1443 | 0.1331 | 2.319 | 10564 | 31 | 141 | 9.8 | 0.202 | 0.226 | 1 | 1.375 | 0/297.6 | false |
| 121 900 | 54.0000 | micelles | 1432 | 0.1320 | 2.335 | 10659 | 30 | 136 | 9.5 | 0.210 | 0.229 | **2** | 2.0 | 0/297.6 | false |
| 132 400 | 54.0000 | micelles | 1431 | 0.1320 | 2.346 | 10744 | 31 | 138 | 9.6 | 0.212 | 0.242 | 1 | 1.75 | 0/297.6 | false |
| 142 900 | 54.0000 | micelles | 1422 | 0.1311 | 2.354 | 10812 | 31 | 137 | 9.6 | 0.204 | 0.244 | 1 | 2.25 | 0/297.6 | false |
| 153 400 | 54.0000 | micelles | 1420 | 0.1309 | 2.365 | 10858 | 32 | 136 | 9.6 | 0.205 | 0.246 | **2** | 2.0 | 0/297.6 | false |
| 163 900 | 54.0000 | micelles | 1427 | 0.1316 | 2.370 | 10911 | 32 | 135 | 9.5 | 0.199 | 0.249 | **2** | 1.5 | 0/297.6 | false |
| **174 400** | 54.0000 | micelles | **1423** | **0.1312** | **2.372** | **10953** | **34** | **135** | **9.5** | **0.201** | **0.245** | **2** | **1.75** | **0/297.6** | **false** |

*Italic rows are DRY-PHASE samples and their aggregate numbers are shown for the density trajectory
only.* At ρ_org = 1.34 the whole system is one contact-percolating mass, so "largest = 1632, flatness
0.88" there is trivially true and means nothing — the predecessor's concern 3, restated because at
box 54 the number is large enough to be tempting. Every structural claim below is a **wet-phase**
number at box 54 and ρ_tot = 1.21783.

**Invariants, per checkpoint, asserted and not eyeballed** — `expect()` in the auditor, all 19
checkpoints: non-finite positions **0**, non-finite velocities **0**, and all six valence/placement
counters **0** (`carbonCCover2`, `carbonCOover1`, `headOverChainCapacity`, `nonBondableBonded`,
`degreeOver3`, `headNotTerminal`). Census matches the checkpoint's own live `activeCounts` exactly
and every **non-solvent** count matches `config.start` exactly (organics may never be created or
destroyed): `{C:10845, O:43380, H:10845, M:723}` at every one of the 19, including the six mid-dry
ones carrying `W: 90`. `cc_break = 0` throughout; `co_break` 0 → 167. Charge 0 throughout.

### 3.5 The largest aggregate, against the closure threshold

Final checkpoint, step 174 400, full readout:

```
{"amphiphileCount":135, "particleCount":525, "radiusOfGyration":7.42,
 "principalMoments":[7.94,9.71,38.24], "flatnessRatio":0.2008, "inPlaneSymmetry":0.2454,
 "radialHeadShells":2, "transverseHeadShells":0, "cavityVolume":1.75,
 "encapsulatedWater":{"encapsulatedCount":0, "totalWater":125971,
   "bulkWaterDensity":0.8023, "encapsulationThresholdCount":297.6, "closed":false,
   "seedDistFromCentre":46.49, "theoreticalMaxDist":46.77,
   "totalEmptyCells":1256075, "unreachedCells":8}}
```

| quantity | threshold | measured | verdict |
|---|---|---|---|
| **encapsulated water** | ≥ **297.6** beads (from the live bulk density 0.8023 × the closure volume) | **0** | **fails, totally** |
| **`closed`** | true | **false** | fails |
| enclosed / cavity volume | ≥ 370.8656 σ³ | **1.75 σ³** (box-wide 7.875 σ³) | fails by 212× |
| size | ≥ 920 (this task's floor) | **135** | fails by 6.81× |
| flatness λ₁/λ₃ | ≤ 0.35 for lamellar | 0.2008 | **passes** |
| **in-plane symmetry λ₂/λ₃** | ≥ 0.50 for lamellar | **0.2454** | **fails** |
| `radialHeadShells` | 2 for a bilayer | **2** (at 4 of the last 5 samples) | passes |

**Shape, read from the moments rather than from the words**: [7.94, 9.71, 38.24] is *one large and
two small* — a **rod**, a cylindrical micelle. A bilayer sheet is two large and one small. The
flatness ratio passes for exactly the wrong reason (a rod is thin in one direction too); the in-plane
symmetry, which is the test that distinguishes a sheet from a rod, fails at 0.245 against 0.50. So the
object is not a proto-membrane that merely failed to close: it is a 135-molecule cylindrical micelle,
one of 34. `transverseHeadShells = 0` at every wet sample says the same thing.

**Nothing closed at any checkpoint of the run: `encapsulatedWater` = 0 against 297.5–297.6 and
`closed = false` at every one of the 14 wet-phase samples.**

### 3.6 Was the curve still rising when it stopped? No.

The whole point of the run length. Largest aggregate across **133 000 steps** of settled wet dynamics
after cycling ended (`cyclePhase` = wet/0 from step 41 400 on — one cycle, and it is over):

```
153, 149, 143, 138, 138, 139, 141, 136, 138, 137, 136, 135, 135
mean 138.4  [135, 153]  sd 5.3   last five: 138, 137, 136, 135, 135
```

Flat for 90 000 steps, then **monotonically declining** over the last five samples. Amphiphile supply
does the same (1607 → 1423, a slow decline as `co_break` accumulates 0 → 167). Aggregate count is
flat-to-rising (31 → 34): the material is not consolidating, it is very slightly dispersing.
The bond count is still creeping (10 744 → 10 953, +1.9 % over the last 42 000 steps), so the
chemistry has **not** stopped — this is a structural plateau at a live chemistry, not a dead run.

For scale: the under-dense box-54 arm's whole coalescence episode (232 → 588) took 45 000 steps. This
arm has had **three times that** with no growth whatsoever.

---

## 4. Verdict, and the binding constraint named and numbered

**NO VESICLE.** Proved by the closure measurement and the head-shell structure, not by shape:
`encapsulatedWater` = **0** against **297.6** and `closed = false` at every wet checkpoint;
`cavityVolume` 1.75 σ³ against the 370.8656 σ³ minimum; the largest object is a cylindrical micelle
(λ₂/λ₃ = 0.245 against the 0.50 a sheet needs), not an unclosed sheet. **There is no step at which
anything closed.**

| # | constraint | measured | required | shortfall | binding? |
|---|---|---|---|---|---|
| 1 | amphiphile **supply** at box 54 | **1435.8** [1420, 1471], settled | ~920 | **1.56× OVER** | **no — RETIRED** |
| **2** | **aggregate size / COALESCENCE** | **136.2** (last-5 mean; final 135), flat-to-declining over 133 000 steps | ~920 | **6.75×** | **YES** |
| 3 | closure / encapsulated water | 0 | ≥ 297.6 | total | downstream of 2 |
| 4 | in-plane symmetry of the largest object | 0.245 | ≥ 0.50 | 2.04× | downstream of 2 (it is a rod, not a sheet) |
| 5 | medium density | ρ_tot 1.21783, ρ_W 0.8023 measured live | liquid | none | no — settled |
| 6 | Verlet list / memory / particle count | 1.918 GB, 191 764 | 4.295 GB, 429 496 | none (2.24×) | no |
| 7 | run length | plateau reached: 133 000 settled steps, last five declining | a plateau | none | no |
| 8 | integrator correctness at box changes | fixed and pinned (§2) | — | none | no — RETIRED |

**BINDING CONSTRAINT: (2) AGGREGATE SIZE — COALESCENCE. 136.2 against 920, 6.75× short.** And the
number that names the mechanism rather than the symptom:

| | box 30 (post-fix, n = 3) | box 54 (decisive) | ratio |
|---|---|---|---|
| amphiphile **supply** | 283.7 | **1435.8** | **5.06×** |
| **largest aggregate** | 109.3 | **135** | **1.24×** |
| single-aggregate **share** | **38.5 %** | **9.5 %** | **0.25×** |
| aggregate count | 5–7 | **30–34** | ~5× |

**Supply scales with L³. The largest aggregate does not scale at all.** Multiplying the box volume
5.83× multiplied the supply 5.06× and the largest aggregate 1.24×; every extra amphiphile went into
*another* aggregate, not into the biggest one. That is a qualitatively different failure from every
predecessor's: supply is over the floor for the first time in this project, and the material simply
will not gather.

**This answers the predecessor's own closing question exactly.** Its concern 8: *"nobody has yet
measured whether 69 % of the amphiphiles coalesce into one aggregate at L = 55 as they do at L = 30.
If they do, that is ~990 against a floor of 1088; if the fraction falls with box size, it is not.
**That single number is the whole remaining question.**"* Measured: the fraction **falls**, from
38.5 % (the honest box-30 figure once the integrator is correct; the 69.4 % was itself inflated by the
stale-F(x) defect) to **9.5 %**. The extrapolation was optimistic by 7.3×.

**Shortfall sequence on the predecessors' own quantity (aggregate size against the floor in force):
80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75×.** A tie with the predecessor on the headline number, and
that tie is worth reading precisely: the predecessor's 6.76× was 161 against 1088, where the 161 was
inflated ~2.3× by the defect of §2 and the 1088 was 18 % above this task's re-measured floor. The
same box-30 arm with a correct integrator and the current floor is **8.42×** (109.3 against 920).
Moving to box 54 bought **1.25×** on that shortfall — real, and an order of magnitude less than the
5.06× more material it put in the box.

---

## 5. Regressions — every one run, with before → after

Every group run with `--no-file-parallelism`, `nice -n 15`, one invocation at a time.

| test file | result | numbers, before → after |
|---|---|---|
| `tests/gate6-bilayer.test.ts` | **2 passed** | area 1.2094±0.0086 → **1.2075±0.0073** [1.1912, 1.2235]; thickness 4.4737 → **4.4762** (per-frame sd 0.2173, n=400); lnA drift t 2.07 → **0.75**; accepted 0.250; escapedMax 0; converge-from-1.55 tail 1.2130 → **1.2217**, from-0.9 1.2286 → **1.2097**, `peaksOk true` both, cluster 1.0000/0.9975. All inside 1.1–1.5 / 4–6. |
| `tests/water-bilayer-area-move.test.ts` | **1 passed** | area 1.1124 → **1.2372** [1.2325, 1.2413]; thickness 4.9619 → **4.5472**; drift t 1.94 → **1.38** (7.891e-6/chunk); water in core 48 → **50**/4500; buried 0.1240 → **0.0870**; cluster 0.9974 → **1.0000**; accepted 0.1327 → **0.1151** of 4500; throughput 1212.08 steps/s at N=5700; **verdict=passed**. These two rows are what §3.2 recomputes the floor from. |
| `tests/catalyst-turnover.test.ts` | **1 passed** | cc_bond 447 → **432**, co_bond 365 → **354**, chains 186 → **181**, amphiphiles 142 → **132**, bare carbons left 38 → **47**, occupiedAtEnd 3 → **2**, desorbTimeout 2 → **5**, nonFinite 0, charge 0, census `{C:779,O:3117,H:779,M:52,W:695,K:0}` exact. Turnover ordering (ccOverLast15k=5 > 0, occupied falls 13→2) holds. |
| `tests/soup-nonfinite-guard.test.ts` | **1 passed** | guard fires at step 1 with 120/120 non-finite components; healthy control 0/0 at three points; cost **0.2420 ms** vs chunk1000 **323.50 ms** = **0.075 %** (1.00075× throughput). |
| `tests/soup-cold-start-relax.test.ts` | **1 passed** | max\|F\| **3.51806e4 → 18.359** (1916×); maxDisplacement **1.13886 σ** ≤ bound 10.05; steps 0→0; velocities, both RNG streams, bond graph and all four event counters unchanged; refusal at `globalStep != 0` fires; payoff pair holds (throws at step 1000 without the stage, 29 199/29 232 non-finite). |
| `tests/soup-evaporation.test.ts` | **3 passed** | `EVAP-PLAN` identical to the predecessor's to every digit (1440.0×, dryBox 20.3538, concentrationFactor 3.2021×, relaxIterations 29, logFractionOf1400 16.07 %); `EVAP-LADDER` 18 increments, 2.132 % linear, identical boxes/solvent; isolation: census solvent-only (`W 11059→5529`, all four organic counts exact), bondSetUnchanged true/true, valence `{bad:0,outOfRange:0}`, nonFinite 0 including after 1000 further steps; insertion `belowFloor 788→771`, minSeparation 0.6257→**0.6158**, preexisting displacement rms 0.1933→**0.1919** max 0.8833→**0.7572** against bound 1.5000; checkpoint round-trip exact with `activeCounts` carrying W:5529; full cycle `dryBox=16.2811 (plan 16.2811) N_dry=5783 (plan 5783) rhoDryRealised=1.34000`, box and census return **exactly**, nonFinite 0 at four points. |
| `tests/soup-drywet-cycling.test.ts` | **2 passed** | THE EVIDENCE THE PRE-EXISTING BOX-SCALING PATH STILL BEHAVES: `concentrationFactor=1.1002` and `rampStepsCharged=1000` **identical**, `dryBox=29.0598`, `realisedDryDensity=1.34000`, box returns exactly, nonFinite 0/0 at four points. Mobility collapse 3.69× → **3.79×**. One-cycle event ratio **0.4978 → 0.7171**, i.e. back to within 0.4 % of the originally published 0.7143 — the fix moved this margin toward its pre-defect value, not away. |
| `tests/sim.test.ts` | **8 passed** | unchanged (grid-vs-brute-force force identity, thermostat, frictionless energy drift) |
| `tests/run-ui.test.ts` | **8 passed** | unchanged |
| `tests/soup-forces.test.ts` | **1 passed** | unchanged — grid + Verlet list vs full O(N²), no GPU console warning |
| `tests/soup-area-move.test.ts` | **2 passed** | `composeVsDirect maxDiff=9.537e-7` — identical to the published value; CPU potential = antiderivative of GPU forces; jacobian holds |
| `tests/soup-bonds.test.ts` | **3 passed** | catalyst gating holds; census+charge conserved under live reactions; assembly-vs-temperature ordering holds |
| `tests/soup-valence.test.ts` | **1 passed** | no carbon over 2 C–C or 1 C–O, no head over chainCapacity, 45 000 steps |
| `tests/soup-checkpoint.test.ts` | **3 passed** | round-trip exact; "a checkpoint does not change what the run computes next" holds; same-process continue writes real files |
| `tests/soup-rules.test.ts` | **4 passed** | schema + detailed balance unchanged; the four `dryWetCycle` evaporation fields still validate |
| `tests/soup-boxcycle.test.ts` | **7 passed** | pure coordinate map, involution, long-coiled-chain regression, `computeDryBox` density, schedule — unchanged |
| `tests/params.test.ts` | **3 passed** | the literal scanner is clean; no rank-A constant appears anywhere new |
| `tests/continuous-run-audit.test.ts` | **1 passed (×2 invocations)** | 19 + 9 = 28 checkpoints audited with `expect()` on every invariant, plus the NEW tether assertion |
| **`tests/soup-stale-force.test.ts`** (NEW) | **2 passed** | see §2.2 and §5.1 |
| `tests/soup-grid-resize.test.ts` | **1 FAILED** | **PRE-EXISTING — attribution in §5.2.** |

**No tolerance was changed, no corridor widened, no assertion relaxed, no threshold moved.** The only
committed assertion added is *stricter* (the tether check, §2.4).

### 5.1 The new test's own numbers, across three separate runs

```
STALE-FORCE {"n":20880,"box":[15,15,15],"maxPreFresh":195.05,"maxPostFresh":513.01,"preResidentVsFresh":0,"postResidentVsFresh":0.000152587890625,"postResidentVsPreFresh":444.15}
STALE-FORCE {"n":20880,"box":[15,15,15],"maxPreFresh":257.50,"maxPostFresh":645.92,"preResidentVsFresh":0,"postResidentVsFresh":0.0001220703125,"postResidentVsPreFresh":388.42}
MIDRUN-MINIMISE {"m":{"iterations":29,"globalStep":2000,"maxForceBefore":209.59,"maxForceAfter":12.32,"displacementBound":1.5},"stepsBefore":2000,"stepsAfter":2000,"boxBefore":[16,16,16],"boxAfter":[16,16,16],"censusSame":true,"chargeSame":true,"bondSetSame":true,"eventsSame":true,"nonFinite":{"pos":0,"vel":0}}
```

### 5.2 `tests/soup-grid-resize.test.ts` — which side of the diff it belongs to

**PRE-EXISTING. It fails identically with and without this task's change, and the drift magnitude
carries no attribution because it scatters over four orders of magnitude on both sides.** Not taken
on trust — the fix was stashed and the test re-run, three times each side:

```
$ nice -n 15 npx vitest run tests/soup-grid-resize.test.ts    # WITH the fix
измеренный дрейф 33973244.2808   /  5297475.5409  /  5781.5508
$ git stash push -- soup/src/soup-box-scale.ts                # WITHOUT the fix
измеренный дрейф 437.7922        /  15889.3232    /  176622.7122
$ git stash pop
```

Same test, same assertion (`assertVerletSafety`, `skin/2 = 0.7500`), same mechanism, both sides;
ranges overlap (5.8e3–3.4e7 against 4.4e2–1.8e5), and the predecessor's own pre-task measurements of
the *same* code were 8.7e5 and 1.9e8. The engine is not bit-reproducible (atomics in the grid fill
vary the summation order), so the magnitude is noise. The diagnosis is unchanged and now
*demonstrated* rather than argued: the missing force rebuild was never the cause — the box change
that test picks is simply too large for its own ramp at that composition. My work neither fixes nor
worsens it. **It belongs on the pre-existing side.** The next task should either give that test a
finer ramp or accept it as a known-too-large box jump; what it must not do is widen the drift bound.

---

## 6. Every command, with real unedited output

Only the commands whose output is a *measurement* are reproduced in full above (§2.2, §3.2, §5.1,
§5.2). The full command list, in order:

```
nice -n 15 npx vitest run tests/soup-stale-force.test.ts --no-file-parallelism
git stash push -m "staleforce-fix-probe" -- soup/src/soup-box-scale.ts
  nice -n 15 npx vitest run tests/soup-stale-force.test.ts --no-file-parallelism -t "изменение бокса"
git stash pop

nice -n 15 npx tsx soup/cli/campaign.ts --label evapFIX19 --box 30 \
  --start '{"C":1860,"O":7440,"H":1860,"M":124,"W":21600}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --steps 40000 --every 10000 --dir data/checkpoints/evapFIX19
  (and --label evapFIX23 --seed 23, --label evapFIX29 --seed 29)

git stash push -- soup/src/soup-box-scale.ts
  nice -n 15 npx tsx soup/cli/campaign.ts --label evapPRE19 --seed 19   (and PRE23/PRE29)
git stash pop

nice -n 15 npx tsx soup/cli/campaign.ts --label minonly19 --box 30 \
  --start '{"C":1860,"O":7440,"H":1860,"M":124,"W":21600}' --seed 19 --kT 1.1 --relax \
  --minimiseAt 35400 --minimiseIterations 29 --steps 36800 --every 10000 \
  --dir data/checkpoints/minonly19            (and minonly23/minonly29)

# THE DECISIVE RUN -- one lineage, six resumable invocations of the same command
nice -n 15 npx tsx soup/cli/campaign.ts --label dec54 --box 54 \
  --start '{"C":10845,"O":43380,"H":10845,"M":723,"W":125971}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --steps <n> --every <n/2> --dir data/checkpoints/dec54
  # --steps 6000 --every 3000   -> globalStep   6 000
  # --steps 18000 --every 6000  -> globalStep  24 000   (the drying event, 18 increments)
  # --steps 14000 --every 7000  -> globalStep  41 400   (the rehydration, 18 increments + insertion)
  # --steps 28000 --every 14000 -> globalStep  69 400
  # --steps 21000 --every 10500 -> globalStep  90 400 / 111 400 / 132 400 / 153 400 / 174 400

CONTINUOUS_RUN_PREFIX=dec54-step CONTINUOUS_RUN_DIRS=data/checkpoints/dec54 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/decisive-run-54-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism
CONTINUOUS_RUN_CHECKPOINTS="<9 box-30 step36800 files>" \
  CONTINUOUS_RUN_ARTIFACT=verify/out/decisive-arms-box30-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism

nice -n 15 npx vitest run tests/gate6-bilayer.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/sim.test.ts tests/run-ui.test.ts tests/soup-forces.test.ts \
  tests/soup-area-move.test.ts tests/soup-bonds.test.ts tests/soup-valence.test.ts \
  tests/soup-checkpoint.test.ts tests/soup-boxcycle.test.ts tests/soup-rules.test.ts \
  tests/params.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/catalyst-turnover.test.ts tests/soup-nonfinite-guard.test.ts \
  tests/soup-cold-start-relax.test.ts tests/soup-stale-force.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/soup-evaporation.test.ts tests/soup-drywet-cycling.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/soup-grid-resize.test.ts --no-file-parallelism   (x3, both sides)
```

The `[campaign]` and `[evaporation]` lines of every one of those runs are quoted verbatim in §1–§3
where they carry a number, and the full traces are the two committed artifacts.

---

## 7. Resources, wall time, housekeeping

- **Compute chunks used: 29 of the 40 allowed.** Breakdown: 2 stale-force test runs (one of them the
  deliberate stashed failure), 6 box-30 cycled runs (3 post-fix + 3 pre-fix), 3 minimisation-only
  control runs, **6 decisive-run invocations at box 54**, 2 off-GPU audits, 2 bilayer gates, 3
  regression groups, 3 grid-resize attribution runs (batched into 2 invocations), 1 arithmetic check.
- **Total steps ≈ 1.06 M**: 174 400 (decisive, box 54, N = 191 764 — ~200 M particle-steps, by far
  the largest single lineage this project has run) + 6 × 46 800 (box-30 cycled arms) + 3 × 36 800
  (control arms) + ~600 000 across the regression suite.
- **Longest single foreground invocation: 624 s.** One invocation (decisive chunk 4, `--steps 28000`)
  exceeded the 600 s cap and was moved to a tracked background task **by the harness itself**, not by
  choice; its full output was returned and is quoted in §3.4. Every subsequent decisive chunk was
  re-sized to 21 000 steps (~470 s measured) so it would not recur, and it did not. Every invocation
  was `nice -n 15`, one at a time, foreground; every multi-file vitest group ran with
  `--no-file-parallelism` specifically so that no two compute processes could exist at once.
- **Measured step cost at the decisive scale: 19.5–22.3 ms/step at N = 191 764** (dropping to
  ~6 ms/step in the dry phase at N = 65 883). Checkpoint encode+write 284–514 ms for a 17 MB file;
  progress readout (full stage detection over 191 764 particles) 511–1224 ms.
- **No orphan processes remain.** `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** after
  every browser-bearing invocation, including the last. Every `git stash push` was matched by a
  `git stash pop` and `git status` verified after each — including after the one Bash call that hit
  its own tool timeout mid-sequence, where the stash was recovered explicitly before anything else ran.
- **The dev server on :5199 (PID 94131) was neither started, stopped nor inspected.** Every
  `gpuPage()` opened its own vite server on its own port.
- `tests/soup-vesicle.test.ts` was **never** run. The full suite was never run. `--dump-dom` was
  never used. **No particle array was ever transferred as JSON numbers:** every reduction (max|F|,
  non-finite counts, census, aggregate sizes, event counts, tether lengths, insertion separations)
  was computed inside the page or inside the offline auditor and returned as scalars.
- **Checkpoints written:** the decisive lineage's 19 files (~17 MB each, 320 MB) plus 9 box-30
  lineages (~2.2 MB each). `data/checkpoints/` is a gitignored run artifact.
- **File sizes, CLAUDE.md's 400–600 rule.** `soup/cli/campaign.ts` stood at 579 and this task needed
  three more flags, so it was **split by responsibility first, then extended**: the flag surface,
  usage text, resume signature and checkpoint file I/O moved verbatim to
  `soup/cli/campaign-config.ts` (261 lines), leaving campaign.ts at **410** with the run itself. Not
  one default, usage line, signature field or write discipline changed in the move — proven by
  `tests/soup-checkpoint.test.ts` and every campaign invocation in this report resuming its own
  lineage correctly. Largest file in the tree is still `soup/wgsl/step.wgsl` at 596; nothing crossed 600.

---

## 8. Concerns

1. **The headline is a negative, and this task's own positive findings deserve the harder scepticism.**
   The two positives are (a) the floor fell 15 % and (b) supply is finally over it. Both trace to the
   same re-measurement (a = 1.2372 instead of 1.1124), and that number moved *because of my own code
   change*. It is defensible — the two independent gates converged from 14.4 % apart to 1.4 % apart,
   which is what a fix looks like rather than what a drift looks like — but a reader is entitled to
   know that the floor moved in the direction that flatters this task, and that a single gate
   re-measurement is n = 1.
2. **The decisive run is ONE seed.** Every box-30 arm here is n = 3 with its scatter printed; the box-54
   run is seed 19 only, because a second seed is ~6 more chunks. Its plateau is internally consistent
   over 13 samples (sd 5.3 on 138.4, i.e. 3.8 %), and the qualitative finding — 30-plus aggregates,
   9.5 % share, no closure — is far outside any scatter this project has measured. But "136.2" as a
   number is one lineage.
3. **The box-30 scatter is much larger than the predecessor reported, and that matters for §1–§2.**
   Its §6.5 claimed ±1–6 % cross-seed on the largest aggregate; measured here at n = 3 per arm it is
   **71–154 (±38 %)** post-fix and 210–273 (±13 %) pre-fix. The 2.28× attribution in §2.3 survives
   that comfortably (non-overlapping ranges), but any *smaller* claim about this quantity at box 30
   needs n ≥ 3, and several published ones were n = 1.
4. **The minimisation-only control is one minimisation, not fifteen.** The predecessor's arm received
   9 + 6; with the defect fixed the cycled arm receives **1 per cycle**, so the honest control for the
   *current* arm is 1, which is what was run. A control matching the *published* arm's 15 was not run,
   and would now be a control for a code path that no longer exists.
5. **The dry phase's percolating mass is more tempting at box 54 than it was at box 30.** "Largest
   aggregate 1632, flatness 0.884, in-plane symmetry 0.942" at step 31 000 looks exactly like a
   near-perfect lamella and is nothing of the kind: at ρ_org = 1.34 the whole system is one contact
   cluster. Every structural number in §3.5 is a wet-phase number, and the dry rows are marked. Anyone
   quoting 1632 would be quoting percolation.
6. **The 6.9 % solvent-free transient is still an artefact and is now bigger in absolute terms.**
   3400 real steps of the rehydration run with the organics in vacuum at falling density (17 × 200),
   because an affine expansion never opens a bead-sized cavity. Tails attract in vacuum, so part of the
   cycling effect may still be that transient rather than the concentration. This task did not separate
   them either; doing so needs an insertion method that works at liquid density.
7. **The tether defect is measured-clean but still unfixed.** 0 crossings in 28 checkpoints with a
   1.27× margin at the worst. That is reassurance, not a proof: the detector samples checkpoints, and
   a crossing that happened and was resolved between two of them would be invisible. The cure is a
   per-step check or a `fene_dv` clamp past `r_inf`.
8. **`tests/soup-grid-resize.test.ts` is still failing and I did not fix it.** §5.2 proves it is
   pre-existing and proves the force rebuild was never its cause. It should be closed by giving that
   test a finer ramp, not by widening its drift bound.
9. **Why the amphiphiles do not coalesce is not answered, only measured.** The most likely reading is
   that at ρ_W = 0.80 the hydration shell around a head group is a real barrier to two micelles
   merging, and that the model's amphiphile — mean per-tail length **2.372 beads**, one or two tails —
   has too little hydrophobic surface to pay for breaking it. Both are testable: a micelle-fusion
   free-energy measurement at two water densities, and a per-tail-length sweep at fixed supply. Neither
   was run. **What is measured is that supply is no longer the wall and coalescence is** — which is a
   different programme from every predecessor's, and the next task should be that programme rather than
   a bigger box.
10. **`clay: false` in every arm**, as in all five predecessors, and doubly forced: a box change is
    refused on a system with an immobile phase, and solvent removal additionally requires the solvent
    to be the last non-empty composition block, which a clay run violates.
11. **The engine is not bit-reproducible run to run** (atomics in the grid fill vary the per-cell
    particle order and hence the force summation order). Measured here: the cold-start minimiser's own
    `max|F|` after 200 iterations differed 16.76 vs 28.36 between two runs of the *identical* fresh
    seed-19 configuration, and `soup-grid-resize`'s drift scatters over four orders of magnitude on
    unchanged code. Every claim in this report is sized against that.

## 9. Files

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/decisive-run-report.md` — this report. `.gitignore`
  line 6 ignores `.superpowers/`, so like every predecessor it is committed with `git add -f`; the
  deviation is deliberate and flagged rather than silently introduced.
- NEW: `tests/soup-stale-force.test.ts` (2 tests) — the stale-F(x) pin and the mid-run minimiser's
  neutrality pin.
- NEW: `soup/cli/campaign-config.ts` (261 lines) — the campaign CLI's flag surface, usage text,
  resume signature and checkpoint file I/O, moved verbatim out of `campaign.ts` (file-size rule)
  and then extended with `--cycles`, `--minimiseAt`, `--minimiseIterations`.
- NEW artifacts: `verify/out/decisive-run-54-trace.json` (19 checkpoints, the decisive run),
  `verify/out/decisive-arms-box30-trace.json` (9 checkpoints, the three box-30 arms at step 36 800).
- MODIFIED: `soup/src/soup-box-scale.ts` (**the fix**: the force dispatch in `applyBoxScaleOnce`'s
  rebuild pass), `soup/src/soup-readback.ts` (`forcesNoRebuild`), `soup/src/sim.ts`
  (`forcesNoRebuildDEBUG`, `minimiseNowDEBUG`), `soup/src/soup-types.ts` (their declarations plus
  `CreateSoupOpts.dryWetCycles`), `soup/src/soup-box-scale-math.ts` (`dryWetCycles` honoured in
  `deriveCycleConfig`, with its own throwing validation), `soup/src/checkpoint.ts`
  (`CheckpointConfig.dryWetCycles`/`minimiseAt`), `soup/cli/campaign.ts` (the split, plus the
  mid-run minimisation in the step loop), `tests/continuous-run-audit.test.ts` (the tether detector
  and its assertion — added fields and one *stricter* assertion, nothing existing changed).
- `data/params.json` and `data/soup.json` **untouched**. No potential, rate, threshold, corridor,
  recogniser or rank-A constant was modified anywhere.
