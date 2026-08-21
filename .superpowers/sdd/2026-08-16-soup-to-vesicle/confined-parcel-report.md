# The periodic boundary is gone: a finite parcel of water with a soft neutral wall, and what the vesicle question answers there

Task `confined-parcel` (2026-08-21). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `closeout-report.md`, whose **concern §1** and whose verdict-document §9 item 3 both named
this as the one measurement left: *"the measurement that would matter is item 3 of the verdict
document's §9 (a non-periodic domain), not more of this one"* — *"a non-periodic or large-enough domain
so a spanning object is not the ground state (named as the root)"*.

User's instruction, verbatim: «так сделай как в океане, в этом и цель».

`data/params.json` rank-A constants: **NOT touched.** `data/soup.json`: **NOT touched, not one field**
— confinement is a per-run CLI flag with its basis in `soup/cli/campaign-config.ts`, exactly the
precedent `--expandRampSteps` set. `co_bond.attemptRate`: **NOT touched** (refused an eleventh time).
`stageThresholds.enclosedVolume` = 370.8656: **read, not moved.** No threshold widened, no corridor
relaxed, no rank-A/B/C/D constant re-fitted, nothing pre-made, no fusion probe,
`tests/soup-vesicle.test.ts` **never run**.

---

## 1. THE CONFINEMENT DESIGN, EVERY DECISION, ITS RANK AND WHAT IT BIASES

### 1.1 The shape and the wall

| decision | what | rank | what it biases |
|---|---|---|---|
| shape | a **sphere** of radius R, centred in the cubic box | experiment design | a sphere is the only shape with no corner and no preferred axis, so nothing in the container can template a flat or a rod-like object. A cube or a slab would. |
| wall law | **harmonic penetration**: U(s) = 0 for s ≤ R, ½k(s−R)² for s > R; F = −k(s−R)·ŝ | experiment design | soft, bounded, non-singular at every s (a 1/s¹² wall is not: a bead that ever finds itself far outside produces Inf). Its gradient is trivially checkable against a numerical one — §3.1. |
| stiffness | **k = 100 ε/σ²**, default in `soup/cli/campaign-config.ts` with arithmetic, not a fit: ω·dt = √100·0.01 = **0.10** (an order of magnitude inside stability, and far softer than the WCA core this integrator already handles at the same dt); thermal penetration √(2kT/k) = **0.148 σ** | experiment design (rank D in this project's scale) | a *container*, not an interface: sharp on the scale of the structures being looked for, soft on the scale of the integrator. Measured penetration at the real campaign: 0.385 σ (§5). |
| neutrality | **the same repulsion for water, heads, tails, catalyst and mineral.** `soup/wgsl/wall.wgsl` never reads the species slot `pos2[i].w` — no per-species offset (not even by bead radius), no depth, no attraction | THE central choice | a hydrophilic wall would behave like a mineral surface (this project already measured what that does: `clay-surface-report.md`), a hydrophobic one would nucleate a film. Neutral is the choice that **tests** closure rather than staging it. What it still biases, stated rather than discovered: **any** repulsive container depletes and layers the liquid next to it, so an enrichment of amphiphile material in the outermost shell is a real possible outcome — which is why it is measured at every checkpoint against the solvent's own shell occupancy (§2, §5). |
| the parcel follows the box | R_live = R_wet · L_live/L_wet, written by the SAME `resizeSoupGrid` call that writes the box | structural | `scaleMoleculesRigid` maps a molecule's centre of mass by the box ratio about the origin, so a sphere centred at box/2 maps to one centred at newBox/2 with radius scaled identically. **Evaporation therefore shrinks the parcel: it is literally a drying droplet**, and it reuses the existing machinery rather than a second copy (§1.4). |

### 1.2 How periodicity is removed — and why not by branching every force path

Eleven places in this engine apply the minimum-image convention or a position wrap. Adding a
`periodic: false` flag to each is exactly the class of silent error this project has been bitten by
seven times. **So the periodic operations are made the IDENTITY, provably, by geometry** instead:
the parcel of radius R sits centred in a cubic box of side **L ≥ 4R + margin**, giving two conditions:

1. **the position wrap never engages.** Every particle stays within R + δ of the centre, i.e. inside
   `[L/2 − R − δ, L/2 + R + δ]`, at least `L/2 − R − δ` from every face, so `x − floor(x/box)·box` is
   the identity. Measured: face clearance **43.89 σ** at the wet box against the longest interaction
   reach of 16.70 σ.
2. **the minimum image never folds any pair.** Every pair separation is at most 2(R+δ) < L/2, so
   `d − round(d/box)·box` is the identity for **every pair in the system**, not merely for interacting
   ones. Measured: max pair separation **72.77 σ** against L/2 = **80**.

Both are **asserted at creation** (`resolveConfine` throws, with the box size it would need) and
**measured at every checkpoint** (`wallStats`). The price is paid in empty box volume — 20.96× the
parcel's, i.e. 157 464 grid cells instead of 7 512 — and **not in particles**, which is what the
953 589-particle ceiling is denominated in.

### 1.3 Every consumer of periodicity, checked one at a time

| consumer | file | what was done |
|---|---|---|
| position wrap, fused into the integrator | `soup/wgsl/step.wgsl` `kick_drift_wrap_main` | left alone; condition (1) makes it the identity. Measured `max|x − wrap(x)| = 0` over every coordinate of every particle. |
| standalone wrap | `soup/wgsl/step.wgsl` `soup_wrap_main` | same. |
| minimiser's wrap | `soup/wgsl/relax.wgsl` | same. The minimiser descends the SAME force the steps use, so it now descends the wall too — a bead that started outside the parcel is pulled in by the cold-start minimisation, not left there. |
| `mi3` in the bonded walk, the nonbonded pair term, the O(N²) reference and the candidate-stats probe | `soup/wgsl/step.wgsl` | left alone; condition (2) makes it the identity. Instrumented and measured: 0 of 12 777 sampled pair components folded (§3.2). |
| `mi3` in the Verlet list build and the drift guard | `soup/wgsl/verlet.wgsl` | same. |
| `mi3` in the long-range electrostatic list and its brute-force reference | `soup/wgsl/electrostatics-long.wgsl` | same — and the ES cutoff is the longest reach in the system (16.70 σ with skin), so it is what condition (1) is checked against. |
| `mi3` in the bond kernels | `soup/wgsl/bond-common.wgsl` | same. |
| cell-index wrap in every neighbour walk | `engine/wgsl/forces.wgsl` `wrap_axis` | left alone: a cell reached across a face is ≥ 2·(L/2 − R) = 88 σ away, so every candidate found there is beyond every cutoff and contributes exactly zero. |
| CPU Metropolis energy | `soup/src/soup-potential.ts` | **not given a wall term, on purpose** — its only caller is the zero-tension area move, which is **REFUSED** under confinement (§1.5). The bond and protonation Monte Carlo do not move particles, so no bond or charge move changes the wall energy. |
| rigid-COM box scaling | `soup/src/soup-box-scale-math.ts` | left alone; it maps about the origin, which is what makes R ∝ L consistent (§1.1). |
| the closure detector's periodic flood, occupancy, centre and seed | `soup/src/water-closure.ts` | **one real defect found and fixed.** `bulkWaterDensity` = reached-empty-water / (reached-empty-cells · cell³) counts the vacuum outside the parcel as bulk, which dilutes the measured density by V_box/V_parcel = **20.96×** and therefore **lowers** the derived `encapsulationThresholdCount` by the same factor. That is widening a closure threshold by a factor of twenty-one through an accounting accident. Fixed by restricting the reached-empty-cell count to cells whose centre lies inside the parcel — the threshold moves in the STRICT direction, never the loose one. Threaded through `analyzeAggregates` → `shapeOfAggregate` → `encapsulatedWaterVolume`, with the parcel read live off the system in `soup/src/stages.ts` so a dry-phase sample measures the dry parcel. |
| the vacuum-cavity detector | `soup/src/aggregates.ts` `localCavityVolume` | left alone: it floods the aggregate's own padded local bounding region with open boundaries and assumes no periodicity at all. |
| aggregate clustering, unwrapping, gyration tensor | `soup/src/aggregates.ts`, `engine/src/aggregate.ts` | left alone; every cutoff involved is ≤ 2.947 σ and condition (2) holds, so the unwrap is the identity and the tensor is the true one. |
| the axis-wrapping instrument | `tests/percolation-check.test.ts` | left alone — it is the setup check (§4). `wrapsOnAxis` was **moved verbatim** into `tests/helpers/periodic-geometry.ts` so the confinement test uses the SAME criterion rather than a second copy. |
| **the Verlet list capacity** | `soup/src/soup-plan.ts` `densestDensityOf` | **second real defect.** Derived from a box-average density, which is 20.96× below the parcel's. Fixed by an optional parcel-volume function. Without it: capacity from ρ = 0.0525 → the `capacityFloor` 1126 would have caught it here, but only by luck. |
| **the long-range electrostatic list capacity** | `soup/src/electrostatics.ts` `makeEsBasis` | **third real defect, and the dangerous one.** Head density taken over `min(box)³` = 0.00901 σ⁻³ instead of over the dry parcel = 0.1889 σ⁻³, **21.0× low**, deriving a per-head capacity of **281 instead of 5898**. An overflowing head list is a silently dropped interaction — the exact defect the long-range task removed, reappearing one level down. Fixed by `EsOverrides.densityVolumeSigma3`. |
| the zero-tension area move | `soup/src/soup-area-move.ts` | **REFUSED** (§1.5). |
| the mineral platelet | `soup/src/sim.ts` | **REFUSED**: a platelet plus a wall is two containers, and the platelet independently forbids the box changes a drying event needs. |

### 1.4 Evaporation becomes physical, and it reuses the existing machinery

The existing `planEvaporation`/`computeDryBox`/`applyEvaporatingTransition`/`rehydrateSolventTo` path
is used unchanged except that **every volume in it is the parcel's**: `computeDryBox` takes an
`occupiedVolumeOf` function, so the dry box is the one at which
(organics + residual solvent)/V_parcel(L_dry) equals `dryWetCycle.targetDryDensity` = 1.34. Computing it
over the box instead would have left the confined dry phase **20.96× less dense** than the number the
run claims to have reached. **The radius follows the water count through the box**: removing water
shrinks the box by the cube-root of the volume ratio and the parcel by the same factor — R_dry =
21.895, L_dry = 97.312, scale 0.6082 — which is exactly a drying droplet, and it keeps both no-wrap
conditions (dry: L/2 − R = 26.76 > 16.70; 2R = 43.79 < L/2 = 48.66).

Rehydration needed one change: `sampleInsertionPositions` accepts candidates by "farthest from
anything already there", and in a confined box the vacuum outside the parcel is farther from
everything than any point inside it — so an unrestricted draw would put **every** rehydrated water
bead in the vacuum, on its first candidate, every time. It now draws from the parcel
(`sampleInParcel`).

### 1.5 The thermostat and the pressure caveat, stated up front

- **The wall does work on the system.** The thermostat is a per-particle Langevin bath (`P.gamma`,
  `P.kT`, `soup/wgsl/step.wgsl`'s `kick_thermostat_main`), which is local and dissipative: it absorbs
  that work exactly as it absorbs the work of every other force. No claim of energy conservation is
  made anywhere in this project, and none is made here. What the wall DOES change is that kinetic
  energy is removed at the boundary as well as in the bulk, so a temperature measured in the outermost
  shell is not the bulk temperature; no temperature is quoted per-shell in this report.
- **No pressure-like quantity is reported.** In particular the zero-tension area move is **refused**
  rather than run: it proposes a box change at fixed volume and reads a lateral tension off the
  acceptance statistics, but in a confined run the box is not what sets the system's volume (the wall
  is), the wall does work as the box moves, and `soup/src/soup-potential.ts` — the energy that move's
  Metropolis criterion is built on — carries no wall term. **There is no zero-tension area to quote
  inside a parcel**, and the published `area-per-lipid-water` gate is and stays a periodic-box
  measurement.
- **The periodic path is intact and default.** `CreateSoupOpts.confine` absent ⇒ radius 0 ⇒ the wall
  kernel is never dispatched at all (not dispatched-and-returning-early: `encodeWall` returns before
  `setPipeline`), every density is taken over the box exactly as before, and every optional parameter
  added in §1.3 defaults to the pre-task behaviour. Proof that the periodic numbers did not move: §9.

---

## 2. PREDICTIONS, STATED BEFORE THE CAMPAIGN WAS RUN

Written before chunk 2 of the campaign; the arithmetic below is pure Python over
`data/soup.json` + `verify/out/water-bilayer-area-move.json`, no GPU.

**P1 — the setup check.** 0 of 3 wrapped axes, by construction, at every checkpoint. *Confidence:
certain* — it was already proved in `tests/soup-confine.test.ts` before any campaign step, on the
whole system's contact network (a set strictly larger than any aggregate).

**P2 — the competing sink.** A neutral wall gives shell occupancy indistinguishable from the
solvent's: |enrichment-vs-water − 1| < 0.3 for C, O, H and M at every checkpoint. *Risk named:* this
was measured at the monomer stage (1.02/0.97/0.99/0.77 in the unit test, 0.985/1.013/0.981/0.959 at
campaign step 2000); the question is whether it survives once real amphiphiles exist, because an
amphiphile has a hydrophobic part and the wall removes water from one side of it.

**P3 — the question. I predict NO CLOSURE, and the reason is arithmetic rather than pessimism.**
Confinement removes the *box-spanning* escape — that is measured, 3 of 3 axes → 0 of 3. But the
cheapest **edge-free** object in *any* geometry, periodic or not, is a **capped cylindrical micelle**,
and its cost has no lower bound tied to the parcel:

| object available in a finite parcel of R = 36 | amphiphiles | against the closure floor 932–1001 |
|---|---|---|
| **spherical micelle** (a capped cylinder of zero length) | **17.3 – 19.8** | **0.017–0.021×** — the band is empty, inverted by **47–58×** |
| capped cylinder, per σ of length | 5.806 – 6.343 | — |
| capped cylinder spanning the parcel diameter (2R = 72 σ) | 435 – 477 | **0.43–0.51×** — still cheaper than closing |
| bilayer disc across the diametral plane | 6752 – 7015 | 6.7–7.5× — **more expensive** than closure, and it has a real rim at the wall (the wall is not a periodic identification, so touching it buys nothing) |
| **closure floor** (enclose 370.8656 σ³: R_in = 4.457, one leaflet out, two leaflets) | **932 – 1001** | — |

So the honest reading of the brief's own sentence *"in a confined volume there is no spanning
competitor, so the floor is the only bound that matters"* is: **it is not true in this geometry.** A
wall-spanning object does exist (a capped cylinder through the parcel, 435–477 amphiphiles, 0.43–0.51×
the floor), and worse, the *unbounded-from-below* competitor exists too — a free capped micelle costs
17–20 amphiphiles and needs no boundary at all. What the periodic box was providing was not
edge-freeness as such but the ability to be edge-free at ARBITRARY SIZE; a capped rod is edge-free at
arbitrary size in a parcel as well, paying only ~19 amphiphiles per free end.
**Therefore the prediction is: a finite branched network of capped rods, `encapsulatedWater` = 0,
flatness ≫ 0.35, `radialHeadShells` oscillating 1↔2 — the same reading as the periodic box, minus the
wrapping.** If that is what happens, the periodic boundary was the *symptom* and the packing parameter
a/t of this molecule is the *cause*.

**P4 — settling.** The size plateau should reach this project's own 0.11–0.25 % standard within
~150 000–200 000 settled steps; the cavity, on every precedent, will still be rising.

**Supply placement, and why.** 2009 amphiphiles expected = **2.01–2.16× the recomputed floor**
(scaled from box-76's measured yield of 4512 in 438 976 σ³ at the identical composition ratios and the
identical water density). Above the floor so closure is *possible* with room for two vesicles; not far
above it, because the periodic campaigns already tested 1.50× and 4.82× and the wrapping verdict did
not move — the supply axis is the one this project has most thoroughly excluded, so the cheap
parcel (2.25× fewer particles than box 76, hence 4.4× more settled steps per unit wall time) buys
more than a bigger supply would.

**Composition** (box-76's census scaled by V_parcel/V_box76 = 0.445200, O:C preserved to 5 decimals):
`{C:24926, O:8307, H:24926, M:646, W:156346}`, N = **215 151**, ρ_total = 1.1009 σ⁻³,
ρ_water = **0.8000 σ⁻³ exactly** (liquid water, `broth-composition-report.md`), pH 7.0, I = 0.01 M,
one drying event, charge on at the calibrated 4.000 λ_D.

**Wall-time projection, made before spending it.** Measured in chunk 1: **19.68 ms/step** at
N = 215 151 (2000 steps in 39.4 s; whole invocation 62 s including the 200-iteration cold-start
minimisation). 500 s cap ⇒ 25 400 steps; **20 000-step chunks** chosen (≈394 s stepping) with smaller
chunks across the two phase transitions. 14 campaign chunks ⇒ step ≈234 000, settled wet length
≈200 000 steps — **1.8× box-76's settled length**.

---

## 3. THE FOUR "THE BOX IS NOT THE VOLUME" DEFECTS — AND WHICH PUBLISHED NUMBERS THEY TOUCH

This is the most valuable engineering result of the task and it is easy to lose behind the verdict, so
it is stated first and separately.

A fully periodic box has one volume: the box. Every density in this engine was therefore written as
`N / (box[0]*box[1]*box[2])`, and **in a periodic box every one of those expressions is correct.** A
finite parcel breaks that identity: the material occupies V_parcel = (4/3)πR³ = 195 432 σ³ while the
box is 4 096 000 σ³, a ratio of **20.959**. Four separate sites then read a density 20.959× too low.

| # | site | what it derives | what it read confined, before the fix | consequence |
|---|---|---|---|---|
| 1 | `soup/src/soup-plan.ts` `densestDensityOf` | the Verlet list's per-particle capacity | ρ = 0.0525 instead of 1.1009 σ⁻³ | an under-sized neighbour list — a **silently dropped neighbour**. Here `verletList.capacityFloor` = 1126 would have absorbed it *by luck*; at a larger parcel it would not. |
| 2 | `soup/src/electrostatics.ts` `makeEsBasis` | the long-range electrostatic head list's per-head capacity | head density 0.00901 instead of **0.1889** σ⁻³, i.e. **21.0× low** → capacity **281 instead of 5898** | **the dangerous one.** An overflowing head list is a silently dropped interaction — the exact defect the `long-range-electrostatics` task existed to remove, reappearing one level down. Nothing would have thrown. |
| 3 | `soup/src/water-closure.ts` `encapsulatedWaterVolume` | `bulkWaterDensity`, and from it `encapsulationThresholdCount` = 370.8656 × that density | the vacuum outside the parcel counted as bulk → density **20.959× low** | **the threshold itself would have been WIDENED 21-fold** (313 → ~15 water beads) by an accounting accident. That is the one thing this project does not do, and it would have been invisible: a "closure" could have been declared on 20 trapped beads. |
| 4 | `soup/src/soup-evaporate.ts` `evaporationLadder` | the wet density the drying ramp starts from and each rung's solvent target | ρ_wet = 0.0525 instead of 1.1009 → ln(1.34/ρ_wet) = 3.24 instead of 0.197 → every rung's target above the pool → the clamp held the solvent at its **full wet count for all 17 intermediate rungs** while the parcel contracted by the whole wet/dry ratio | **caught LOUDLY**: the parcel reached 2.32 σ⁻³ by rung 9 and 614 769 of 645 453 position components went non-finite between steps 12 800 and 13 000. `soup/wgsl/health.wgsl`'s finiteness guard fired with the interval named. This is the only one of the four that this task found by a crash rather than by reading, and it is the one that proves the guard earns its keep. |

### 3.1 Which previously published numbers were computed with a wrong density: NONE

Stated plainly, because "we found four density defects" invites the question. **All four expressions
were correct for every run this project has ever published**, because every one of those runs was a
fully periodic box in which the box volume *is* the occupied volume. The defects are **latent**, not
wrong-in-the-past: they are activated only by a system whose material occupies less than its box,
which did not exist before this task.

- No gate value, no campaign row, no shortfall in the sequence 80× → … → 0.21×, and no conclusion in
  `docs/soup-to-vesicle-verdict.md` was computed through any of them at a wrong density.
- In particular `encapsulatedWater` = **exactly 0 at 23 of 23 wet checkpoints** of the box-76 campaign
  is unaffected in both directions: the count is a flood-fill result that does not use the density at
  all, and the threshold it was compared against (312.64–315.41) was derived from a bulk density
  measured over a box that *was* the system.
- The fix is an **optional parameter defaulting to the previous behaviour** at all four sites, so the
  periodic path is not merely argued to be unchanged, it is unchanged by construction. The numerical
  proof is §9 (the periodic rank-A gates re-measured).

**The general lesson, which is the transferable part:** `N / box³` is not a density, it is a density
*and an assumption*. The assumption held for four years of this project's runs and was never written
down anywhere, which is why breaking it broke four places at once rather than one.

---

## 4. THE COMPETING SINK: THE AMPHIPHILES DO NOT COAT THE CONTAINER

This is the way the experiment most plausibly fails, and it is answered — measured at every
checkpoint, in the outermost shell of thickness 4.687 σ (the project's own measured bilayer thickness,
read from `verify/out/water-bilayer-area-move.json`, i.e. exactly the depth an adsorbed film would
occupy), against two nulls: the uniform-sphere fraction, and — the one that matters — **the solvent's
own shell occupancy**, which cancels the geometric factor that a soft-walled liquid's slightly
relaxed outer surface introduces for every species alike.

| phase | step | C in shell / total | H in shell | enrichment vs water: C / O / H / M | reading |
|---|---|---|---|---|---|
| monomers, wet | 2 000 | 7 962 / 24 926 | 7 928 | 0.985 / 1.013 / 0.981 / 0.959 | indifferent to the wall — the wall is neutral, measured, not assumed |
| dry | 15 400 | 12 973 / 24 926 | 12 284 | 1.023 / 1.075 / 0.968 / 1.043 | still indifferent (the solvent reference is 112 beads here and therefore noisy; the absolute enrichments 1.011/1.063/0.958/1.032 say the same) |
| dry | 31 800 | 13 490 | 7 885 | 0.854 / 1.017 / 0.499 / 0.896 | solvent reference is 71 of 112 beads — **not a usable denominator**; the absolute enrichments are 1.052/1.253/0.615/1.104 |
| **wet, aggregated** | **37 200** | **1 / 24 926** | **0** | **1e-4 / 0.684 / 0.000 / 0.021** | **the material has left the wall entirely** |
| **wet, aggregated** | **55 200** | **1 / 24 926** | **0** | **1e-4 / 0.705 / 0.000 / 0.027** | unchanged |

**One carbon of 24 926 and zero hydrogens in the wall shell, at an enrichment of 1e-4 against water.**
The aggregate's radius of gyration is **19.7 σ against a parcel radius of 36 σ**: the material
collapsed *inward*, to a compact object sitting well clear of the container, not outward onto it.

Two consequences, both worth their own line:

1. **The neutral wall did exactly what it was designed to do.** It is a container, not an interface:
   it neither wets nor nucleates. The design decision in §1.1 is therefore not just declared, it is
   verified on the real system at the real density.
2. **A positive result in this geometry could not be dismissed as a wall-adsorbed film**, and a
   negative one cannot be blamed on the wall having eaten the supply. That is precisely what made the
   competing-sink measurement worth taking before the verdict rather than after it.

(The residual O enrichment of 0.68–0.71 is the *heads that are still free monomers* distributing
themselves through the parcel including its outer shell — 1 985–2 110 of 8 307 heads — while the
2 209–2 276 heads bound into the aggregate are all in its compact core. It is a statement about the
free-monomer population, not about adsorption; the free-amphiphile fraction is reported per checkpoint
in §6.)

---

## 5. THE IMPLEMENTATION PROOFS (`tests/soup-confine.test.ts`, 3 of 3 passed)

### 5.1 The wall force against a numerical gradient

Twelve particles on twelve Fibonacci directions at radii straddling R = 20 in a box of 60, placed
through a hand-built `resume` payload (the fresh-creation lattice never puts a bead outside the
parcel, so this is the only way to make the wall force measurable), **minimum pairwise separation
18.79 σ against an interaction range of 2.947 σ — asserted, so "the force reported IS the wall force"
is measured, not assumed.** k = 137 (a deliberately non-round value, so a hardcoded 100 could not pass).

```
WALL-GRADIENT R=20 k=137 minPairSep=18.786 (interactionRange=2.947)
  s=17.0000 F_r(GPU)=0.000000 F_r(численный)=0.000000 F_r(аналит)=0.000000
  s=17.6000 F_r(GPU)=0.000000 F_r(численный)=0.000000 F_r(аналит)=0.000000
  s=18.2000 F_r(GPU)=0.000000 F_r(численный)=0.000000 F_r(аналит)=0.000000
  s=18.8000 F_r(GPU)=0.000000 F_r(численный)=0.000000 F_r(аналит)=0.000000
  s=19.4000 F_r(GPU)=0.000000 F_r(численный)=0.000000 F_r(аналит)=0.000000
  s=20.0000 F_r(GPU)=0.000000 F_r(численный)=-0.034250 F_r(аналит)=-0.000000
  s=20.6000 F_r(GPU)=-82.200312 F_r(численный)=-82.200000 F_r(аналит)=-82.200000
  s=21.2000 F_r(GPU)=-164.399858 F_r(численный)=-164.400000 F_r(аналит)=-164.400000
  s=21.8000 F_r(GPU)=-246.599919 F_r(численный)=-246.600000 F_r(аналит)=-246.600000
  s=22.4000 F_r(GPU)=-328.799722 F_r(численный)=-328.800000 F_r(аналит)=-328.800000
  s=23.0000 F_r(GPU)=-411.000009 F_r(численный)=-411.000000 F_r(аналит)=-411.000000
  s=23.6000 F_r(GPU)=-493.200287 F_r(численный)=-493.200000 F_r(аналит)=-493.200000
WALL-GRADIENT max|GPU-аналит|=3.0056e-4 (относит. 3.8020e-6) max|GPU-численный|=3.1216e-4 (относит. 3.7976e-6)
  на_изломе(s=R)|GPU-численный|=3.4250e-2 (граница шаблона k*h/4=3.4250e-2)
  max|F| внутри=0.0000e+0 max|сетка-перебор|=0.0000e+0
```

- **The kernel implements the gradient of the stated potential: 3.80e-6 relative**, against a
  float32 single-op epsilon of 1.19e-7 — i.e. ~32 epsilons over a handful of operations, and ~1e5
  times tighter than any wrong sign, factor, radius or centre convention could land.
- **Inside the parcel the force is EXACTLY zero** (`max|F| внутри = 0`), so the wall is a wall and not
  a background field.
- **The one disagreement is the STENCIL's, not the kernel's, and it is accounted rather than
  excluded.** At s = R exactly, a central difference straddles the kink where U's second derivative
  jumps, and returns −(½kh²)/(2h) = **−k·h/4 = −0.034250** however small h is. The measured
  disagreement there is **3.4250e-2 = the bound exactly**, and the kernel's own value at that point
  (0, the correct one-sided limit from inside) agrees with the analytic form like every other point.
- **`max|сетка-перебор| = 0`**: the wall is applied bit-identically on the grid/Verlet path and on the
  O(N²) reference. That is the trap a one-body term added inside three of the four force kernels — or
  at seven of the eight call sites — would have fallen into, and it is why `encodeWall` lives inside
  the three `encodeSoupForce*` functions and not at their call sites.

The same gate at full physics, with charge on at the campaign's own 10 mM, in a real confined broth
(`tests/soup-forces.test.ts`, extended per the brief):

```
SOUP-FORCES-CONFINED maxDiff(сетка+Верле+список_голов против перебора)=7.6294e-5 meanAbsRef=14.7985 N=18888 снаружи_парцеллы=329 max|F_стенки|=25.4035
SOUP-FORCES-CONFINED R=16.0000 V_парцеллы=17157.3 V_бокса/V_парцеллы=29.842 rc_es=15.2000 ёмкость_списка_голов=729 ёмкость_Верле=1126 плотнейшая_плотность=1.1009
SOUP-FORCES-CONFINED заполнение: главный max=439/1126 головы max=728/729
```

329 beads outside the parcel carrying up to 25.4 in wall force, `densestDensity` = **1.1009** (the
parcel's, not the box-average 0.0369 the un-fixed code would have used), and **neither derived list
overflowed** — the failure mode defect #1 and #2 in §3 would have produced.

### 5.2 No wrap, with the instrument's own positive control

```
NO-WRAP box=100 R=20 N=29814 maxR=20.0760 продавливание=0.0760 зазор_до_грани=30.3077 (наибольший радиус взаимодействия 6.3067) maxПара=40.152 L/2=50.000 сильное=true
NO-WRAP склеек_при_реальном_боксе=0/12777 maxDelta=0 | КОНТРОЛЬ при боксе 40: склеек=1140/12777 maxDelta=40.000
NO-WRAP max|сдвиг обёртки позиции|=0 осей_обёртки(вся система)=0/3 [false,false,false]
```

- **0 of 12 777 sampled pair components folded at the real box; maxDelta exactly 0.**
- **The same instrument, on the SAME positions, sees 1 140 folds and a maxDelta of 40.000 when the box
  is shrunk to 2R** — so "no folding" is a measurement and not a tautology. A no-wrap check that
  cannot detect a wrap is not a check.
- **`max|x − wrap(x)| = 0`** over every coordinate of every one of 29 814 particles: the position wrap
  is the identity, not merely harmless.
- Wall penetration 0.076 σ against the predicted thermal √(2kT/k) = 0.148 σ.

### 5.3 The checkpoint round-trip carries the parcel, and the wall is provably still there

```
CONFINE-ROUNDTRIP шаг=300 в_файле={"radiusSigma":16,"stiffness":100} R_live до=16.000000 после=16.000000 max|dPos|=0 max|dVel|=0 различий_в_связях=0 max|dF|=0.00005340576171875 max|F_стенки| в этой конфигурации=26.1424
```

Positions, velocities and the bond graph come back **bit-identical**; the parcel and the stiffness are
in `config.confine` and reconstruct to the same live radius. The force field agrees to **5.34e-5
against a wall force of 26.14 in the same configuration** — ~1e-6 relative, and deliberately NOT
asserted as bit-identical, because the force is a sum over a neighbour list whose per-cell append
order is set by atomics and a rebuild reorders the summation. Five orders of magnitude below the size
of the term being checked for, so a *dropped* wall would be unmissable.

### 5.4 The refusals

```
CONFINE-REFUSALS
  тесный бокс: confine: зазор до грани бокса L/2-R=1.0000 не превосходит наибольшего радиуса взаимодействия 6.3067 при box=[42.0000, 42.0000, 42.0000], R=20.0000 -- частица у поверхности парцеллы чувствовала бы СВОЙ ЖЕ образ через границу, то есть периодичность не снята. Возьмите бокс не меньше 54.6
  глина: confine: удержание в парцелле несовместимо с минеральной пластиной (clay) -- это две стенки сразу; используйте clay:false
  areaMove: areaMove: MC-ход по площади не имеет смысла в удерживаемой парцелле -- объём задаёт стенка, а не бокс, стенка совершает работу над системой при изменении бокса, и soupPotential не содержит её вклада. Нулевое натяжение измеряется в периодическом боксе (CreateSoupOpts.confine отсутствует)
```

Each of the three would otherwise have produced a number that does not mean what its name says. The
box refusal even names the box size it would need.

---

## 6. THE SETUP CHECK: 0 OF 3 AXES, AT ALL 20 CONFINED CHECKPOINTS

This is the free, non-negotiable measurement, and it is the project's own discriminating instrument —
the one that read **3 of 3 at 23 of 23 checkpoints** in the box-76 campaign — run unchanged on the
confined checkpoints (`wrapsOnAxis` was moved verbatim into `tests/helpers/periodic-geometry.ts` so
the unit test and the campaign instrument use one criterion, not two copies).

| role | file | box | aggregates | amphiphiles in largest | wrapsX/Y/Z | **wrapping axes** | slabs touched |
|---|---|---|---|---|---|---|---|
| **campaign** | cpR36-step15400 (dry) | 97.31 | 3 | 1053 | f/f/f | **0** | 17,17,17 of 32 |
| **campaign** | cpR36-step31800 (dry) | 97.31 | 1 | 2293 | f/f/f | **0** | 17,17,17 of 32 |
| **campaign** | cpR36-step37200 … step225200 (18 wet) | 160 | 1 | 2276 → 2092 | f/f/f | **0** at every one | 17–19 of 58 |
| control | bbB76-step148200 (the periodic campaign) | 76 | 4 | 4512 | **t/t/t** | **3** | 27,27,27 of 27 |
| control | swB54-step144400 (positive) | 54 | 1 | 1542 | **t/t/t** | **3** | 19,19,19 of 19 |
| control | dec54-step174400 (negative) | 54 | 25 | 151 | f/f/f | 0 | 7,7,10 of 19 |

**0 of 3 at 20 of 20 confined checkpoints, while the two positive controls still read 3 of 3 on the
same day with the same code.** The instrument is not stuck on `false`, and it is not stuck on the slab
count either (58 slabs at box 160, 27 at box 76, 19 at box 54). Everything downstream is therefore not
void — which was the precondition for reading any of it.

The consequence for the published table is direct and it is the one unambiguously positive result of
this task: **`aggregate-percolation` = 0, verdict `passed`, corridor `inside` — the first time that
gate has ever passed in this project.** It has read `3 / failed / outside` in every campaign since it
was created.

### 6.1 The wall-adsorbed fraction versus the interior, at every checkpoint

Recomputed off-GPU from the checkpoints themselves (`verify/out/confined-wall-adsorption.json`), shell
= 4.6871 σ (the bilayer thickness that was current when the campaign was sized; the gate has since
re-measured to 4.9061, which changes no conclusion here).

| step | box | R | maxR | **penetration** | face clearance | max pair | L/2 | strong | shell fraction C / O / H / M / W | **enrichment vs water C / O / H / M** | C,H in shell |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 000 | 160 | 36 | 36.385 | 0.385 | 43.89 | 72.77 | 80 | ✓ | .319/.329/.318/.311/.324 | **0.985 / 1.013 / 0.981 / 0.959** | 7962, 7928 |
| 15 400 | 97.31 | 21.895 | 22.312 | 0.416 | 26.56 | 44.62 | 48.7 | ✓ | .521/.547/.493/.531/.509 | 1.023 / 1.075 / 0.968 / 1.043 | 12973, 12284 |
| 31 800 | 97.31 | 21.895 | 22.276 | 0.381 | 26.53 | 44.55 | 48.7 | ✓ | .541/.645/.316/.568/.634 | *(0.854 / 1.017 / 0.499 / 0.896 — the solvent reference is 71 of 112 beads here; not a usable denominator)* | 13490, 7885 |
| 37 200 | 160 | 36 | 36.125 | 0.125 | 44.07 | 72.25 | 80 | ✓ | **.0000**/.254/**.0000**/.008/.372 | **1e-4 / 0.684 / 0.000 / 0.021** | **1, 0** |
| 55 200 | 160 | 36 | 36.156 | 0.156 | 44.19 | 72.31 | 80 | ✓ | .0000/.239/.0000/.009/.339 | 1e-4 / 0.705 / 0.000 / 0.027 | 1, 0 |
| 73 200 | 160 | 36 | 36.111 | 0.111 | 44.09 | 72.22 | 80 | ✓ | .0000/.238/.0000/.009/.335 | 1e-4 / 0.709 / 0.000 / 0.028 | 1, 0 |
| 84 200 | 160 | 36 | 36.055 | **0.055** | 44.34 | 72.11 | 80 | ✓ | .0000/.236/.0000/.009/.334 | 1e-4 / 0.708 / 0.000 / 0.028 | 1, 0 |
| 95 200 | 160 | 36 | 36.169 | 0.169 | 44.12 | 72.34 | 80 | ✓ | .0000/.236/.0000/.009/.332 | 1e-4 / 0.709 / 0.000 / 0.028 | 1, 0 |
| 105 200 | 160 | 36 | 36.149 | 0.149 | 44.26 | 72.30 | 80 | ✓ | .0000/.237/.0000/.009/.331 | 1e-4 / 0.716 / 0.000 / 0.028 | 1, 0 |
| 115 200 | 160 | 36 | 36.132 | 0.132 | 44.47 | 72.26 | 80 | ✓ | .0000/.237/.0000/.009/.331 | 1e-4 / 0.715 / 0.000 / 0.028 | 1, 0 |
| 125 200 | 160 | 36 | 36.114 | 0.114 | 44.53 | 72.23 | 80 | ✓ | .0000/.237/.0000/.009/.331 | 1e-4 / 0.717 / 0.000 / 0.028 | 1, 0 |
| 135 200 | 160 | 36 | 36.196 | **0.196** | 44.35 | 72.39 | 80 | ✓ | .0000/.237/.0000/.009/.331 | 1e-4 / 0.718 / 0.000 / 0.028 | 1, 0 |
| 145 200 | 160 | 36 | 36.101 | 0.101 | 43.92 | 72.20 | 80 | ✓ | .0000/.238/.0000/.009/.330 | 1e-4 / 0.722 / 0.000 / 0.028 | 1, 0 |
| 155 200 | 160 | 36 | 36.137 | 0.137 | 44.37 | 72.27 | 80 | ✓ | .0000/.237/.0000/.009/.330 | 1e-4 / 0.720 / 0.000 / 0.028 | 1, 0 |
| 165 200 | 160 | 36 | 36.107 | 0.107 | 44.32 | 72.21 | 80 | ✓ | .0000/.237/.0000/.009/.329 | 1e-4 / 0.722 / 0.000 / 0.028 | 1, 0 |
| 175 200 | 160 | 36 | 36.111 | 0.111 | 44.17 | 72.22 | 80 | ✓ | .0000/.238/.0000/.009/.329 | 1e-4 / 0.723 / 0.000 / 0.028 | 1, 0 |
| 185 200 | 160 | 36 | 36.155 | 0.155 | 44.16 | 72.31 | 80 | ✓ | .0000/.237/.0000/.009/.329 | 1e-4 / 0.720 / 0.000 / 0.028 | 1, 0 |
| 195 200 | 160 | 36 | 36.161 | 0.161 | 44.07 | 72.32 | 80 | ✓ | .0000/.238/.0000/.009/.329 | 1e-4 / 0.722 / 0.000 / 0.028 | 1, 0 |
| 205 200 | 160 | 36 | 36.143 | 0.143 | 44.31 | 72.29 | 80 | ✓ | .0000/.238/.0000/.009/.328 | 1e-4 / 0.725 / 0.000 / 0.028 | 1, 0 |
| 215 200 | 160 | 36 | 36.167 | 0.167 | 44.22 | 72.33 | 80 | ✓ | .0000/.238/.0000/.009/.328 | 1e-4 / 0.726 / 0.000 / 0.028 | 1, 0 |
| **225 200** | 160 | 36 | **36.198** | **0.198** | **44.12** | **72.395** | 80 | ✓ | **.0000/.239/.0000/.009/.329** | **1e-4 / 0.727 / 0.000 / 0.028** | **1, 0** |

**Every geometric guarantee holds by measurement at all 21 checkpoints**: face clearance 26.53–44.53
against the longest interaction reach of 16.70; max pair separation 44.55–72.77 against L/2 = 48.66
(dry) / 80 (wet); the strong no-wrap condition true at every one; wall penetration 0.055–0.416 σ,
i.e. never more than 1.2 % of the parcel radius.

**And the competing sink is answered: the amphiphiles do NOT coat the container.** In the settled wet
phase, **1 carbon of 24 926 and 0 hydrogens of 24 926 are in the wall shell — an enrichment against
water of 1e-4 and 0.000, at 18 of 18 checkpoints** — while the aggregate's radius of gyration is
**19.54–19.78 σ inside a parcel of radius 36**. The material collapsed inward, to a compact object
sitting well clear of the wall, and it stayed there for 188 000 steps. Before aggregation (step 2 000)
every species read 0.96–1.01 against water, which is the wall's neutrality measured directly rather
than declared.

Two consequences, and both are load-bearing for the verdict: **a positive result here could not have
been dismissed as a wall-adsorbed film, and this negative one cannot be blamed on the wall having
eaten the supply.**

---

## 7. THE CAMPAIGN, PER CHECKPOINT, AGAINST THE PRE-FIXED CRITERIA

`verify/out/confined-campaign-R36-trace.json`, 21 checkpoints, measured off-GPU by the run's own
auditor. Non-finite positions/velocities **0** and all six valence counters **0** at every one.
*(Italic rows are the monomer stage or the dry phase; no closure claim rests on them.)*

| step | box | R | N | aggs | amph | **largest** | r_g | flat | inPl | **radSh** | **cavity σ³** | **encH₂O** | **thresh** | closed | ρ_water | bonds | m/tail |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *2 000* | *160* | *36* | *215151* | *85* | *95* | *2* | *1.77* | *0.092* | *0.248* | *n/a* | *0* | *centre-untrusted* | — | — | — | *478* | *2.11* |
| *15 400* | *97.3* | *21.9* | *58917* | *3* | *1055* | *1053* | *16.49* | *0.861* | *0.937* | *2* | *2.875* | ***0*** | *1.05* | *false* | *0.0028* | *5088* | *2.99* |
| *31 800* | *97.3* | *21.9* | *58917* | *1* | *2293* | *2293* | *17.49* | *0.911* | *0.992* | *2* | *43.75* | ***0*** | *1.20* | *false* | *0.0032* | *11760* | *2.75* |
| 37 200 | 160 | 36 | 215151 | 1 | 2276 | **2276** | 19.778 | 0.8169 | 0.8970 | 2 | 30.625 | **0** | 312.67 | false | 0.8431 | 12785 | 2.78 |
| 55 200 | 160 | 36 | 215151 | 1 | 2209 | **2209** | 19.663 | 0.8420 | 0.9270 | 2 | 46.125 | **0** | 313.33 | false | 0.8449 | 14913 | 2.89 |
| 73 200 | 160 | 36 | 215151 | 1 | 2185 | **2185** | 19.623 | 0.8552 | 0.9286 | 2 | 52.375 | **0** | 313.85 | false | 0.8463 | 16147 | 2.98 |
| 84 200 | 160 | 36 | 215151 | 1 | 2172 | **2172** | 19.618 | 0.8450 | 0.9256 | 2 | 59.625 | **0** | 314.20 | false | 0.8472 | 16718 | 3.05 |
| 95 200 | 160 | 36 | 215151 | 1 | 2164 | **2164** | 19.580 | 0.8545 | 0.9231 | 2 | 68.250 | **0** | 314.38 | false | 0.8477 | 17176 | 3.10 |
| 105 200 | 160 | 36 | 215151 | 1 | 2165 | **2165** | 19.578 | 0.8469 | 0.9216 | 2 | 60.500 | **0** | 314.55 | false | 0.8482 | 17543 | 3.12 |
| 115 200 | 160 | 36 | 215151 | 1 | 2158 | **2158** | 19.586 | 0.8445 | 0.9245 | 2 | 71.625 | **0** | 314.61 | false | 0.8483 | 17837 | 3.14 |
| 125 200 | 160 | 36 | 215151 | 1 | 2140 | **2140** | 19.581 | 0.8338 | 0.9193 | 1 | 66.375 | **0** | 314.57 | false | 0.8482 | 18136 | 3.16 |
| 135 200 | 160 | 36 | 215151 | 1 | 2133 | **2133** | 19.587 | 0.8303 | 0.9239 | 1 | 70.250 | **0** | 314.65 | false | 0.8484 | 18397 | 3.18 |
| 145 200 | 160 | 36 | 215151 | 1 | 2131 | **2131** | 19.583 | 0.8324 | 0.9274 | 1 | 68.375 | **0** | 314.69 | false | 0.8485 | 18614 | 3.20 |
| 155 200 | 160 | 36 | 215151 | 1 | 2131 | **2131** | 19.569 | 0.8452 | 0.9337 | 1 | 70.625 | **0** | 314.77 | false | 0.8487 | 18797 | 3.22 |
| 165 200 | 160 | 36 | 215151 | 1 | 2122 | **2122** | 19.559 | 0.8413 | 0.9261 | 1 | 67.750 | **0** | 314.81 | false | 0.8489 | 18974 | 3.24 |
| 175 200 | 160 | 36 | 215151 | 1 | 2111 | **2111** | 19.556 | 0.8465 | 0.9296 | 1 | 77.125 | **0** | 314.73 | false | 0.8486 | 19134 | 3.26 |
| 185 200 | 160 | 36 | 215151 | 1 | 2103 | **2103** | 19.562 | 0.8462 | 0.9305 | 1 | 68.875 | **0** | 314.73 | false | 0.8486 | 19276 | 3.26 |
| 195 200 | 160 | 36 | 215151 | 1 | 2095 | **2095** | 19.549 | 0.8468 | 0.9346 | 1 | 77.500 | **0** | 314.75 | false | 0.8487 | 19422 | 3.29 |
| 205 200 | 160 | 36 | 215151 | 1 | 2093 | **2093** | 19.543 | 0.8483 | 0.9349 | 1 | 75.625 | **0** | 314.76 | false | 0.8487 | 19540 | 3.29 |
| 215 200 | 160 | 36 | 215151 | 1 | 2094 | **2094** | 19.539 | 0.8370 | 0.9294 | 1 | 73.250 | **0** | 314.87 | false | 0.8490 | 19636 | 3.32 |
| **225 200** | 160 | 36 | 215151 | 1 | 2092 | **2092** | 19.548 | 0.8379 | 0.9237 | 1 | 69.500 | **0** | 314.83 | **false** | 0.8489 | 19710 | 3.32 |

Aggregate-size distribution: **one aggregate holding 100 % of the amphiphile material** at every wet
checkpoint from 37 200 on (`sizeHistogram` = `[2092]` at the end); 85 aggregates of 2 at the monomer
stage; 3 (1053 + 1 + 1) during the first half of the dry phase. `hasVesicleAggregate` **false** at all
21; `hasLamellarAggregate` **false** at all 21; free-amphiphile fraction **0** at 19 of 20 (0.0019 at
step 15 400).

**A cross-check that the parcel-restricted bulk density (§3, defect #3) is right, not merely
stricter:** the confined threshold reads **312.67–314.87** water beads, against the box-76 periodic
campaign's own **312.64–315.41** — the same number, because both are now
370.8656 σ³ × the density of real liquid water (0.843–0.849 here, ~0.85 there). The un-fixed code
would have published **~15**.

Coalescence and chemistry (`verify/out/confined-audit.json`): **merges 0, fissions 0,
`exchangedFraction` 0 at all 19 intervals**; largest covalent component **15 → 35** beads of 13 239
(`covalentSpanFraction` 0.00298 → **0.00264**), so the object is a **contact** network of 2092
chemically separate molecules, exactly as in the periodic box; α_all 0.695–0.705, pKa_app 6.62–6.64,
interfacial shift +0.026…+0.172, `unlikeExcessRatio` 1.01–1.13.

### 7.1 Settling evidence, to this project's own standard

| quantity | last four (195 200 → 225 200) | mean ± sd | **relative** | slope /1000 steps (all 18 → last 4) |
|---|---|---|---|---|
| **largest aggregate** | 2095, 2093, 2094, 2092 | **2093.50 ± 1.29** | **0.062 %** | −0.806 → **−0.080** |
| r_g | 19.549, 19.543, 19.539, 19.548 | 19.5448 ± 0.0046 | **0.024 %** | −0.0008 → −0.0001 |
| flatness λ₁/λ₃ | 0.8468, 0.8483, 0.8370, 0.8379 | 0.8425 ± 0.0059 | 0.697 % | +0.0000 → −0.0004 |
| in-plane λ₂/λ₃ | 0.9346, 0.9349, 0.9294, 0.9237 | 0.9306 ± 0.0053 | 0.567 % | +0.0001 → −0.0004 |
| wrapping axes | 0, 0, 0, 0 | **0** (unanimous at 20 of 20) | — | — |
| **cavity** | 77.500, 75.625, 73.250, 69.500 | 73.97 ± 3.45 | **4.664 %** | +0.172 → **−0.264** |

- **THE SIZE IS SETTLED, AND IT IS THE TIGHTEST PLATEAU THIS PROJECT HAS EVER PUBLISHED: 0.062 %**
  over the last four checkpoints spanning 30 000 steps, against the project's own 0.11–0.25 %
  standard. The discriminating fact is not the scatter but the slope: it **falls tenfold**, −0.806 →
  −0.566 (last 8) → −0.354 (last 6) → **−0.080** (last 4), which is what approaching a plateau looks
  like. Settled wet length **188 000 steps**, 1.68× the box-76 campaign's 111 800.
- r_g at **0.024 %** and flatness/in-plane under 0.7 % say the same thing about the shape.
- **THE CAVITY IS NOT A PLATEAU BY THIS PROJECT'S STANDARD, AND THAT IS STATED PLAINLY: 4.664 % on
  the last four, 19× outside the 0.25 % bar.** But it is qualitatively different from box 76, and the
  difference is the sign: box 76's cavity slope did **not** fall (+0.695 over all 23 → +0.586 over its
  last 13, the same straight line), whereas here the slope **turns over**: +0.172 (all 18) → +0.028
  (last 8) → −0.077 (last 6) → **−0.264 (last 4)**. The cavity rose to ~77 σ³ and is now falling. In a
  finite parcel the interstitial void is bounded; in a periodic box it was not. **Best available
  reading, labelled as such:** final 69.5 → **5.34× short** of 370.8656; last-four mean 73.97 →
  **5.01× short**. Both are WORSE than box 76's 2.66×/2.80×, for the honest reason that this object is
  2.16× smaller — which is what a supply of 2× the floor instead of 4.7× buys.
- **Two things are still moving, exactly as in every predecessor**: bonds +54.2 % (12 785 → 19 710)
  and `meanPerTail` +19.5 % (2.78 → 3.32). The settled object is a steady state of a **still-growing
  molecule**, and `radialHeadShells` tracks that: it read **2 for the first seven wet checkpoints and
  1 for the last eleven** — a one-way transition at ~120 000 steps, not the 1↔2 oscillation box 76
  showed, and it moved in the WRONG direction for a vesicle.

---

## 8. THE VERDICT

### 8.1 It did not close, and the criterion is the pre-fixed one

**`encapsulatedWater` = EXACTLY 0 at 19 of 19 measurable checkpoints**, against a derived threshold of
**312.67–314.87** water beads in the wet phase (1.05–1.20 in the dry phase, where there are only 112
water beads to trap). `closed` **false** at all 19. `hasVesicleAggregate` **false** at all 21.
Published gate: **`vesicle-closure-water` = 0, rank B, failed, outside**; **`vesicle-verdict` = 0,
rank A, failed, outside.**

Judged by the pre-fixed criterion and never by shape. For completeness the shape says the same:
flatness **0.8379** against ≤ 0.35 (2.4× outside), in-plane 0.9237, `radialHeadShells` **1**.

### 8.2 Which prediction failed, and where it sits in the sequence

**The prediction that failed is the FIRST of the three the brief named: no wrapping, but no closure.**
Not a wall-coated film — §6.1 measured 1 carbon of 24 926 in the wall shell, so that competing
explanation is excluded by measurement, not by argument. Not "something else" either.

**The hypothesis that failed is the experiment's own**, and it is worth naming precisely because it
was this project's last named root cause (`docs/soup-to-vesicle-verdict.md` §9 item 3): *that the
periodic box's topology was what prevented closure.* The topology is now gone — wrapping axes **3 of 3
→ 0 of 3**, `aggregate-percolation` **failed → passed**, the first time in the project — **and closure
did not follow.** So the periodic boundary was the **symptom**, not the cause.

My own §2 prediction P3 said exactly this, before the campaign, and gave the arithmetic reason, which
the fresh gate basis has since only confirmed: the cheapest **edge-free** object in a finite parcel is
a **capped cylindrical micelle costing 17.5–21.3 amphiphiles**, with no lower bound tied to the parcel,
against a closure floor of **937–1014** — a band inverted by **44–58×**. A wall-spanning capped
cylinder costs 439–490, still **0.43–0.52×** the floor. What the periodic box provided was never
edge-freeness as such but the ability to be edge-free **at arbitrary size**; a capped rod is edge-free
at arbitrary size in a parcel too, paying only ~19 amphiphiles per free end. The measured object is
exactly that: one contact network of 2092 chemically separate molecules, flat-ish, with a bounded
interstitial void and not one trapped water bead.

**Its place in the shortfall sequence** (largest aggregate against the closure floor, the project's own
convention): floor 937–1014 against a settled 2092 → **0.45×**, i.e. the supply is **2.06–2.23× OVER**
the floor and size is not the binding constraint.

**80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75× → 0.48× → 0.42× → 0.67× → 0.21× → 0.45×**

It lands between 0.42× (the charge campaign, 2.36× over) and 0.48× (the two-tailed box-54 campaign,
2.08× over) — at deliberately the same supply multiple as those two arms, so it is the *topology* axis
that moved and nothing else. **What this step retires: the periodic-box topology.** What survives, and
is now the only thing left standing, is **the molecule's own packing parameter a/t** — a statement
about the amphiphile, not about the box, the supply, the charge, the temperature or the boundary.

### 8.3 The honest framing that would have been owed on a success, stated anyway

Had it closed, this is what the result would have had to carry, and it is worth writing down because
it is what makes the negative result informative rather than merely disappointing:

1. **The wall contributes.** It is a soft neutral repulsive container — measured neutral (§6.1: every
   species within 4 % of water's own shell occupancy before aggregation), but it is still an
   idealisation that does not exist in the ocean. A real air/water interface is hydrophobic and would
   nucleate a film (the aerosol case, prebiotically real in its own right); a mineral surface is
   hydrophilic and this project has already measured what that does. Neutral was chosen to *test*
   closure rather than stage it, and that choice is what a positive result would have had to be
   qualified by.
2. **The parcel is 36 σ.** Closure in a 36 σ droplet is not closure in the ocean; it is closure in a
   droplet whose radius is 1.6× the object's own diameter.
3. **The periodic-box result stands as its own finding, not as an error this corrects.** In a fully
   periodic box below L\* the ground state is a spanning object and the measured object was one:
   3 of 3 axes at 23 of 23 checkpoints, cavity ×2.90 with `encapsulatedWater` exactly 0. That is a
   true statement about periodic boundary conditions, and the published table still carries it as its
   own row (§9).
4. **What would be needed to show the same unconfined:** a domain large enough that a spanning object
   is not the ground state, i.e. L > L\* = **146.9–172.5 σ** (this task's own re-derived range, printed
   by the `supply-window` pin), which is **2.53e6 particles against a measured ceiling of 953 589** —
   still structurally out of reach, because a per-particle neighbour list costs L³ρ². Confinement is
   the *only* way this engine can ask the question at all, which is exactly why the wall's neutrality
   had to be measured rather than assumed.

**And a wall-adsorbed film is not a vesicle.** Nothing in this report calls one the other; §6.1 exists
so that distinction is a measurement.

---

## 9. THE PERIODIC RANK-A GATES ARE UNMOVED — THE NUMERICAL PROOF

The confinement code is inert on the periodic path **by construction** (radius 0 ⇒ `encodeWall`
returns before `setPipeline`, so not one extra dispatch is issued; every parameter added in §1.3 and
§3 defaults to the pre-task expression). The proof that it is inert in fact is the table, regenerated
through `npm run verify`, never hand-edited:

| gate | rank | closeout's published value | **this task, regenerated** | verdict | corridor | how it is obtained |
|---|---|---|---|---|---|---|
| area-per-lipid | **A** | 1.2004636 | **1.1981189** | passed → **passed** | inside → **inside** | re-measured in-process, fresh stochastic draw (−0.20 %) |
| bilayer-thickness | **A** | 4.4307669 | **4.4853148** | passed → **passed** | inside → **inside** | re-measured in-process, fresh stochastic draw (+1.23 %) |
| bending-modulus | **A** | null | **null** | unproven → **unproven** | none → **none** | κ fit still invalid |
| **vesicle-verdict** | **A** | 0 | **0** | failed → **failed** | outside → **outside** | from the periodic campaign artifact |
| area-per-lipid-water | C | 1.1607494 | **1.1834503** | passed → **passed** | inside | artifact re-measured by its own test (+1.96 %) |
| bilayer-thickness-water | C | 4.6871310 | **4.9060532** | passed → **passed** | inside | artifact re-measured by its own test (+4.67 %) |
| **vesicle-closure-water** | B | 0 | **0** | failed → **failed** | outside | periodic campaign artifact |
| **aggregate-percolation** | B | 3 | **3** | failed → **failed** | outside | periodic campaign artifact |
| chain-length-asf | D | 0.1594033 | **0.1594033** | unproven | none | byte-identical |
| mean-tail-length | D | 3.419 | **3.419** | unproven | outside | byte-identical |
| closure | D | 1284.875 | **1284.875** | unproven | inside | byte-identical |
| chain-to-bead-mapping | D | 3 | **3** | unproven | none | byte-identical |

**Every rank, every verdict and every corridor is identical.** The only values that moved are the four
that are *measured again* rather than read from an artifact — the two in-process rank-A bilayer gates
(−0.20 %, +1.23 %, both well inside their corridors, and of the same size as the closeout task's own
re-measurement drift of −0.39 % / −0.27 %) and the two rank-C water gates whose artifact was
re-generated by `tests/water-bilayer-area-move.test.ts` in this task's regression run (+1.96 %,
+4.67 %, both inside 1.1–1.5 and 4–6). Everything read from the periodic campaign's own artifacts —
including `aggregate-percolation` = **3** — is byte-identical. **The confined arm reads 0 for that same
gate from its own artifact, through the same code: one implementation, two experiments.**

### 9.1 Both rows published

`verify/campaign-gates.ts`'s `collectCampaignGateInputs` now takes its two artifact paths as
parameters defaulting to the pre-existing ones, so `npm run verify` publishes the periodic table
byte-for-byte as before, and `verify/confined-gates.ts` publishes the confined arm **beside** it
(`verify/out/gates-confined.json`) through the same collection and the same `evaluateGates`. No
corridor in `data/literature.json` was touched: those corridors were established in a periodic box, and
a confined run is a different experiment, not a correction of one.

```
$ nice -n 15 npx tsx verify/confined-gates.ts
GATE-CONFINED area-per-lipid-water: value=1.1834503086016825 rank=C verdict=passed corridor=inside
GATE-CONFINED bilayer-thickness-water: value=4.906053163496518 rank=C verdict=passed corridor=inside
GATE-CONFINED vesicle-closure-water: value=0 rank=B verdict=failed corridor=outside
GATE-CONFINED aggregate-percolation: value=0 rank=B verdict=passed corridor=inside
GATE-CONFINED vesicle-verdict: value=0 rank=A verdict=failed corridor=outside
GATE-CONFINED chain-length-asf: value=0.18753762793497886 rank=D verdict=unproven corridor=none
GATE-CONFINED mean-tail-length: value=3.322 rank=D verdict=unproven corridor=outside
GATE-CONFINED written verify/out/gates-confined.json
```

| gate | rank | **periodic box (published table)** | **confined parcel (this arm)** | what the difference means |
|---|---|---|---|---|
| **aggregate-percolation** | B | **3 — failed, outside** | **0 — PASSED, inside** | the wrap-around is gone; the object is genuinely finite. **The first pass of this gate in the project.** |
| vesicle-closure-water | B | 0 — failed, outside | **0 — failed, outside** | closure did not follow |
| vesicle-verdict | A | 0 — failed, outside | **0 — failed, outside** | no vesicle in either |
| chain-length-asf | D | 0.1594 — unproven | **0.1876 — unproven** | different chain statistics at 2× vs 4.7× the floor |
| mean-tail-length | D | 3.419 — unproven, outside | **3.322 — unproven, outside** | still above the 2–3 window in both |

---

## 10. REGRESSIONS — EVERY ONE RUN, BEFORE → AFTER

| file | before (closeout) | **after** | note |
|---|---|---|---|
| `tests/soup-forces.test.ts` | 3 passed | **4 passed** | +1: the confined brute-force gate (§5.1) |
| `tests/soup-checkpoint.test.ts` | 3 passed | **4 passed** | +1: the confinement round-trip (§5.3) |
| **`tests/soup-confine.test.ts`** | — | **3 passed** | new: gradient, no-wrap + control, setup check, refusals |
| `tests/verlet-capacity.test.ts` | 2 passed | **2 passed** | both guards fired: `VERLET-OVERFLOW` at capacity 64, `VERLET-DERIVED-OK {"cap":1126,"max":322,"atCapacity":0,"mean":293.94}`, `VERLET-CEILING … 17280000000 байт … WebGPU отказал бы МОЛЧА` |
| `tests/gates.test.ts` | 8 passed | **8 passed** | |
| `tests/supply-window.test.ts` | 1 passed | **1 passed** | `WINDOW-SCALING floor=[932, 1001] … cheapestSpanning=[5.806, 6.343]*L L*=[146.9, 172.5]sigma` — the pin's own floor brackets §2's recomputation exactly |
| `tests/coalescence-mechanism-pin.test.ts` | 5 passed | **5 passed** | |
| `tests/coalescence-mechanisms.test.ts` | 1 passed | **1 passed** | |
| `tests/es-range-calibration.test.ts` | skipped | **skipped** | needs `ES_CAL_CHECKPOINTS` |
| `tests/params.test.ts` | 3 passed | **3 passed** | the literal scanner: **no numeric model constant entered `soup/src` or `soup/wgsl`** |
| **`tests/soup-evaporation.test.ts`** | (not in closeout's list) | **3 passed** | the split's proof: `planEvaporation`/both ladders/`sampleInsertionPositions` behave identically from their new home |
| `tests/soup-boxcycle.test.ts` | — | **7 passed** | same |
| `tests/water-closure.test.ts` | — | **3 passed** | the parcel parameter is a no-op when absent |
| `tests/soup-aggregates.test.ts` | — | **4 passed** | same |
| `tests/periodic-measurement.test.ts` | — | **5 passed** | `wrapsOnAxis`'s move did not change it |
| `tests/closure.test.ts` | — | **8 passed** | |
| `tests/percolation-check.test.ts` | 1 passed | **1 passed** | 20 confined rows at 0 axes + 3 controls (2 at 3 axes) |
| `tests/continuous-run-audit.test.ts` | 1 passed | **1 passed** | 21 checkpoints, tether violations 0, invariants clean |
| `tests/electrostatics-audit.test.ts` | 1 passed | **1 passed** | re-pointed at this campaign's 20 checkpoints |
| `tests/gate6-bilayer.test.ts` | 3 passed (group) | **2 + 1 passed** | `GATE6 area 1.2021 ± 0.0075 thickness 4.4777 … escapedMax 0 box 24.442 steps 83000` |
| `tests/water-bilayer-area-move.test.ts` | 1 passed | **1 passed** | `settled=true chunks=450 areaPerLipid=1.1835 [1.1785, 1.1880] driftPerChunk=-4.670e-6 t=-0.71 thickness=4.9061 clusterFraction=1.0000` |
| `tests/soup-electrostatics.test.ts` | 8 passed | **8 passed** | |
| **`tests/run-ui.test.ts`** | 8 passed | **8 passed** | the page the user actually opens; puppeteer profile killed after, `pgrep` clean |
| `tests/soup-grid-resize.test.ts` | **1 failed** | **1 failed** | **HEAD's own side, PROVED not asserted**: `git stash -u` (the whole diff, including the two new files) then re-run → **still 1 failed**, same drift class. Not chased, per the brief |
| `npx tsc --noEmit` | 21 errors | **21 errors** | the pre-existing `ArrayBufferLike`/`Record<string,unknown>` class, **identical SET** (compared line by line with the file/column stripped), so the three new source files and the eleven edited ones add **zero** |

Known flaky, **not chased**, per the brief: `soup-drywet-cycling` (margin-flaky), `rim-lambda-insitu`
(2 of 3 pre-existing). `tests/soup-vesicle.test.ts` **never run**.

### 10.1 CLAUDE.md's file-size rule

`soup/src/soup-evaporate.ts` reached **615 lines** when the ladder-density fix (§3 defect #4) landed —
over the hard 600. Per the rule ("split first, then add") it was **split by responsibility** along the
line the file's own section comment already drew: the pure, GPU-free half (the plan, both ramps, the
insertion sampler) moved verbatim to **`soup/src/soup-evaporate-plan.ts` (304 lines)**, and the live
half that mutates a running system stayed in **`soup/src/soup-evaporate.ts` (335 lines)**, which
re-exports every moved name so no importer changed. Proved a pure move by
`tests/soup-evaporation.test.ts` 3/3 and `tests/soup-boxcycle.test.ts` 7/7. **Nothing in
`soup/ engine/ chem/ viewer/ verify/ tests/` is now over 600 lines** (largest: `soup-buffers.ts` 589,
`engine/src/closure.ts` 587, `sim.ts` 584).

---

## 11. EVERY COMMAND, IN ORDER, WITH REAL OUTPUT

Outputs quoted in full above: §5 (the three proof tests), §6 (percolation), §6.1 (the wall scan), §7
(the trace and the audit), §9 (both gate tables), §10 (the regressions).

```
# --- implementation proofs, BEFORE any campaign step
nice -n 15 npx vitest run tests/soup-confine.test.ts --no-file-parallelism            # 3 passed (3 iterations to get the tolerances honest -- S5.1)
nice -n 15 npx vitest run tests/soup-forces.test.ts tests/soup-checkpoint.test.ts --no-file-parallelism

# --- the pre-run arithmetic (pure python, no GPU): floor, competitors, composition, ladder, capacities
python3 -c "<S2 and S3 arithmetic>"

# --- the campaign: 21 foreground invocations of ONE command
nice -n 15 npx tsx soup/cli/campaign.ts --label cpR36 --box 160 \
  --confineRadius 36 --confineStiffness 100 \
  --start '{"C":24926,"O":8307,"H":24926,"M":646,"W":156346}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --charge --pH 7.0 --ionicStrength 0.01 \
  --steps <n> --every <n> --dir data/checkpoints/cpR36
  # --steps  2000 ->   2 000   62 s  monomers, 85 aggregates of 2, bonds 478   <- the monomers-only proof
  # --steps 10000 -> (12 000)  363 s DIVERGED at 12 800..13 000: 614 769 of 645 453 components non-finite
  #                                  -- defect S3 #4, caught by the finiteness guard, fixed, re-run
  # --steps 10000 ->  15 400   374 s the drying event: 18 rungs, water 156 346 -> 112, parcel 36 -> 21.895
  # --steps 16400 ->  31 800   191 s dry, one aggregate of 2293
  # --steps  2000 ->  37 200   123 s rehydration: 18 rungs, 156 234 beads inserted, min sep 0.6457
  # --steps 18000 ->  55 200   665 s  <- 656 s of stepping, OVER the 500 s cap (see S12)
  # --steps 18000 ->  73 200   650 s  <- same invocation (a two-chunk loop; the harness backgrounded it)
  # --steps 11000 ->  84 200   388 s  cavity 59.625  largest 2172
  # --steps 11000 ->  95 200   444 s  cavity 68.250  largest 2164
  # --steps 10000 -> 105 200   340 s  cavity 60.500  largest 2165
  # --steps 10000 -> 115 200   322 s  cavity 71.625  largest 2158
  # --steps 10000 -> 125 200   319 s  cavity 66.375  largest 2140   radialHeadShells 2 -> 1
  # --steps 10000 -> 135 200   321 s  cavity 70.250  largest 2133
  # --steps 10000 -> 145 200   323 s  cavity 68.375  largest 2131
  # --steps 10000 -> 155 200   301 s  cavity 70.625  largest 2131
  # --steps 10000 -> 165 200   296 s  cavity 67.750  largest 2122
  # --steps 10000 -> 175 200   301 s  cavity 77.125  largest 2111
  # --steps 10000 -> 185 200   293 s  cavity 68.875  largest 2103
  # --steps 10000 -> 195 200   281 s  cavity 77.500  largest 2095
  # --steps 10000 -> 205 200   293 s  cavity 75.625  largest 2093
  # --steps 10000 -> 215 200   295 s  cavity 73.250  largest 2094
  # --steps 10000 -> 225 200   287 s  cavity 69.500  largest 2092   <- size plateau 0.062 %

# --- the instruments, all off-GPU on the campaign's own checkpoints
CONTINUOUS_RUN_PREFIX=cpR36-step CONTINUOUS_RUN_DIRS=data/checkpoints/cpR36 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/confined-campaign-R36-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism        # 117 s, 21 checkpoints
PERC_CAMPAIGN_LABEL=cpR36 PERC_CHECKPOINTS="<20 confined + bbB76 + dec54 + swB54>" \
  PERC_ARTIFACT=verify/out/confined-percolation.json \
  nice -n 15 npx vitest run tests/percolation-check.test.ts --no-file-parallelism           # 7 s
ES_AUDIT_CHECKPOINTS="<20>" ES_AUDIT_ARTIFACT=verify/out/confined-audit.json \
  nice -n 15 npx vitest run tests/electrostatics-audit.test.ts --no-file-parallelism        # 8 s
nice -n 15 npx tsx /tmp/wallscan.ts                                    # the S6.1 table -> verify/out/confined-wall-adsorption.json

# --- the plateau arithmetic (pure python over the artifact, no GPU)
python3 -c "<least-squares slope over four windows>"

# --- regressions, four groups, each --no-file-parallelism so no two compute processes coexist
nice -n 15 npx vitest run tests/soup-forces.test.ts tests/soup-checkpoint.test.ts \
  tests/soup-confine.test.ts tests/verlet-capacity.test.ts tests/gates.test.ts \
  tests/supply-window.test.ts tests/coalescence-mechanism-pin.test.ts \
  tests/coalescence-mechanisms.test.ts tests/es-range-calibration.test.ts \
  tests/params.test.ts --no-file-parallelism                    # 31 passed, 1 skipped, 67 s
nice -n 15 npx vitest run tests/soup-evaporation.test.ts tests/soup-boxcycle.test.ts \
  tests/water-closure.test.ts tests/soup-aggregates.test.ts tests/periodic-measurement.test.ts \
  tests/closure.test.ts tests/percolation-check.test.ts \
  tests/continuous-run-audit.test.ts --no-file-parallelism      # 32 passed, 198 s (the split's proof)
nice -n 15 npx vitest run tests/gate6-bilayer.test.ts tests/water-bilayer-area-move.test.ts \
  --no-file-parallelism                                          # 3 passed, 276 s
nice -n 15 npx vitest run tests/soup-electrostatics.test.ts tests/run-ui.test.ts \
  tests/soup-grid-resize.test.ts --no-file-parallelism            # 16 passed, 1 failed (grid-resize), 45 s
pkill -f puppeteer_dev_chrome_profile ; pgrep -fl puppeteer_dev_chrome_profile   # -> NO ORPHANS (after every browser group)

# --- the attribution of the one failure, PROVED
git stash -u && nice -n 15 npx vitest run tests/soup-grid-resize.test.ts --no-file-parallelism ; git stash pop
                                                                 # -> 1 failed on HEAD's own side

# --- both published gate tables
nice -n 15 npm run verify                                        # the periodic table, S9, 174 s
nice -n 15 npx tsx verify/confined-gates.ts                       # the confined row, S9.1
nice -n 15 npx tsc --noEmit                                      # 21 errors, identical SET to HEAD's
```

---

## 12. TOTALS, RESOURCES, HOUSEKEEPING

- **Compute invocations: 38 against a budget of 40 — under by 2.** 21 campaign (one of which
  diverged and one of which was a two-chunk loop), 6 implementation-proof runs, 4 instrument passes,
  4 regression groups, 2 `npm run verify`, 1 confined-gate publish, and the stash attribution proof.
  The pure-Python arithmetic and the trace extraction ran over artifacts already on disk and are not
  counted.
- **ONE RESOURCE-RULE BREACH, REPORTED RATHER THAN HIDDEN.** A two-chunk loop was issued as a single
  Bash call; the harness moved it to the background (violating "foreground only"), and the condensed
  phase turned out **1.9× dearer per step than the monomer phase** (35.6 vs 19.68 ms/step), so that
  invocation ran **656 s and 650 s of stepping against the 500 s cap**. Cause: the wall-time
  projection in §2 was built from a step-2000 measurement taken while the system was still monomers,
  which is the wrong basis for a condensed run. Corrected immediately — 11 000 then 10 000-step
  chunks, one per invocation, foreground, longest subsequent invocation **444 s** — and every chunk
  after it stayed under the cap. The lesson is the same one as §3's: a rate measured in one regime is
  a rate *and an assumption*.
- `nice -n 15` on every invocation. Every multi-file vitest group ran `--no-file-parallelism`, so **no
  two compute processes ever existed at once**. **No dev server started or killed** — :5199 was never
  touched. `timeout` never used. `--dump-dom` never used; **no whole-particle array ever crossed the
  CDP boundary** (the one place a full array was needed off-GPU, §6.1, went through the checkpoint's
  own base64 payload and ran in Node).
- **Orphans: none.** `pkill -f puppeteer_dev_chrome_profile` after every browser group, each time
  confirmed by `pgrep -fl` printing nothing (`NO ORPHANS`). No `node_modules/.vite/deps_temp_*` cleanup
  was needed — no vitest died mid-run.
- **New trajectory: 225 200 steps**, of which **188 000 settled wet** at N = 215 151 ≈ 4.05 × 10¹⁰
  particle-steps, plus 16 400 dry steps at N = 58 917 and two 18-rung phase transitions.
- **Memory:** Verlet list 215 151 × 1284 × 4 = **1.105 GB** (0.257× the 4.294967292 GB binding limit),
  ES head list 8307 × 5898 × 4 = **0.196 GB**, per-particle state ≈0.014 GB, grid 157 464 cells ×
  3 × 4 B = 0.002 GB ⇒ ≈**1.32 GB**. No allocation was refused. The 20.96× empty box costs 0.002 GB and
  a serial prefix scan whose share of a step was too small to isolate.
- **New source: three files** — `soup/src/soup-confine.ts` (316), `soup/wgsl/wall.wgsl` (56),
  `soup/src/soup-evaporate-plan.ts` (304, a pure split), plus `verify/confined-gates.ts` (57) and
  `tests/soup-confine.test.ts` (302). All inside CLAUDE.md's rule.
- **Never done:** no threshold widened (one was *narrowed* 21-fold back to correct, §3), no corridor
  relaxed, no rank-A/B/C/D constant re-fitted, `co_bond.attemptRate` refused an eleventh time,
  `data/soup.json` and `data/params.json` and `data/literature.json` untouched, no pre-made start, no
  under-dense water (0.8000 σ⁻³ exactly), no fusion probe, `tests/soup-vesicle.test.ts` not run.
- **Artifacts written:** `verify/out/confined-campaign-R36-trace.json` (21),
  `verify/out/confined-percolation.json` (23 rows), `verify/out/confined-audit.json` (20),
  `verify/out/confined-wall-adsorption.json` (21), `verify/out/gates-confined.json`,
  `verify/out/gates-campaign-trace-confined.json`, `verify/out/gates-percolation-confined.json`,
  `verify/out/gates.json`, `verify/out/report.html`, `verify/out/water-bilayer-area-move.json`,
  `verify/out/kappa-measurement.json`. Checkpoints: `data/checkpoints/cpR36/` 21 files, ~2.0 GB
  (gitignored, as every campaign's are).

---

## 13. CONCERNS

1. **The result is a NEGATIVE about the boundary and a POSITIVE about the molecule, and the second
   half is not yet measured.** Removing the periodic topology moved the wrapping from 3 of 3 to 0 of 3
   and moved closure by nothing. That leaves the packing parameter a/t as the only surviving
   explanation — but "the only one left" is not the same as "measured". The cheapest test is now
   explicit: raise the head area or shorten the tail so a/t crosses the cylinder→bilayer boundary, in
   the SAME confined parcel, and see whether the object flattens into a disc before it closes. That is
   a composition axis, not a boundary axis, and this task did not touch it.
2. **The cavity is not settled, and it is now falling.** 4.664 % on the last four against a 0.25 %
   bar, with the slope having turned from +0.172 to −0.264 σ³/1000 steps. The turn-over is the
   interesting part (in a periodic box the same quantity never stopped rising), but it means the
   number quoted — 69.5, 5.34× short — is a sample of an oscillation, not a converged value. Since
   §7 also shows `encapsulatedWater` is exactly 0 while that quantity swings by 12 %, the cavity is
   again measuring interstitial void rather than a lumen; **quoting it as a closure shortfall is
   generous to the model, not harsh**, exactly as the closeout report concluded for box 76.
3. **One seed, seed 19, again.** A single trajectory at 215 151 particles is not a sample. The
   wrapping verdict survives this (a boolean, unanimous at 20 of 20, with two live positive controls),
   and so does `encapsulatedWater` = 0 (unanimous at 19 of 19); "cavity slope −0.264" and "size
   plateau 0.062 %" do not.
4. **`radialHeadShells` went 2 → 1 and stayed there for the last eleven checkpoints.** In box 76 it
   oscillated 1↔2 with no trend and the closeout report was careful to call that an oscillation. Here
   it is a one-way transition at ~120 000 steps, in the direction AWAY from a vesicle, coincident with
   `meanPerTail` passing ~3.16. Whether that is the object genuinely losing its second shell or the
   detector becoming bistable on this geometry is **unmeasured**, and it is the one structural number
   in §7 I would not publish without a second seed.
5. **The wall's stiffness was chosen by arithmetic and never swept.** k = 100 gives ω·dt = 0.10 and a
   measured penetration of 0.055–0.416 σ, which is defensible, but nobody has measured whether k = 30
   or k = 300 changes the aggregate. A softer wall is a more compressible container and therefore a
   different thermodynamic boundary condition; the claim "the wall is a container, not an interface"
   rests on neutrality (measured) and not on stiffness-independence (not measured).
6. **The parcel volume is an exact sphere, and the real thing is not.** A drying droplet is held by
   surface tension and deforms; this one is a rigid spherical constraint that shrinks affinely. That
   is a modelling choice with no measurement behind it, and it is the least physical part of the
   design.
7. **`tests/soup-grid-resize.test.ts` still fails and was still not chased**, now proved against a
   stash of this whole diff. The defect is real and it lives in `applyBoxScale` — the same code path
   the evaporation cycle uses, and the same code path §3's defect #4 blew up in. Nobody has measured
   whether it can bite a campaign rather than only that test; this task's own divergence is
   circumstantial evidence that the area deserves attention.
8. **The 20.96× empty box is a cost nobody has optimised.** It buys the strong no-wrap condition
   (mi3 is the identity for EVERY pair, not only interacting ones) for 0.002 GB and an unmeasurable
   slice of a step, so it was the right trade here — but the closure flood grid scales as (L/0.5)³,
   which is 32.8 M cells at L = 160 against 3.5 M at L = 76, and that showed up as a
   4.1–24.7 s per-checkpoint measurement cost. A confined run in a box of 2R + 40 instead of 4R + 16
   would keep the physics identical and pay a ninth of that; it would only weaken the *statement*, not
   the result.
9. **`verify/confined-gates.ts` is a script, not a test.** Nothing fails if the confined row stops
   being reproducible; only a human running it notices. Folding it into `npm run verify` was
   deliberately not done, because the published table must stay the periodic measurement.
