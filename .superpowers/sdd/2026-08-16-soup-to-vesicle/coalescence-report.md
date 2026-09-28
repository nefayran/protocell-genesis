# Why the amphiphiles do not coalesce — mechanism, and whether closure is reachable

Task `coalescence` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `decisive-run-report.md`, whose **concern 9** is the whole subject here — *"why the
amphiphiles do not coalesce is not answered, only measured"* — and whose verdict named the binding
constraint **(2) AGGREGATE SIZE — COALESCENCE, 136.2 against 920, 6.75× short**, with the number that
made a bigger box the wrong next move: box 30 → 54 multiplied supply **5.06×** and the largest
aggregate **1.24×**, and the single-aggregate share FELL 38.5 % → 9.5 %.

`data/params.json` rank-A constants: **NOT touched.** `co_bond.attemptRate`: **NOT touched.**
No threshold widened, no corridor relaxed, no tolerance changed.

---

## 0. THE THREE PREDICTIONS, STATED BEFORE ANY MEASUREMENT

Written into this file and committed before the first measurement ran, so that "which survived" is a
result and not a reading-back. Each prediction is chosen so that its own signature is **incompatible**
with the other two, not merely consistent with itself.

### P1 — FREE-HEAD POISONING would show

The composition runs O:C = 4:1, so tens of thousands of unreacted single polar head beads sit in the
box (43 380 O against 10 845 C at box 54). In real surfactant chemistry free polar monomers adsorbing
on an aggregate surface cap growth and stabilise small aggregates. If that is the mechanism here:

1. **Surface enrichment.** The free-head number density in an aggregate's first contact shell must be
   *enriched* over its bulk value, and enriched **relative to water**, which is the competing
   adsorbate: the normalisation-free ratio `g_O(r)/g_W(r)` (both species probing the identical
   geometry, so no shell-volume normalisation is needed at all) must be **> 1** for r inside the
   first shell, and rise as r falls.
2. **Material coverage.** Free heads in contact per surface amphiphile of the aggregate must be of
   order 0.1–1 — a capping layer, not a stray visitor. A coverage of ~0 cannot cap anything.
3. **The matched arm.** At matched amphiphile count, cutting the free-head concentration ~4× must
   RAISE the largest aggregate and the single-aggregate share.

### P2 — A HYDRATION / FUSION BARRIER would show

Two aggregates that meet fail to merge because the water between their polar surfaces must be
expelled first. If that is the mechanism:

1. **Contacts frequent, merges rare.** Aggregate–aggregate surface contacts must be common per
   snapshot while merge events over the same time must be ~0. Barrier = arrivals without reactions.
2. **Sizes frozen, not exchanging.** Merges AND fissions must both be near zero: a kinetically
   trapped distribution does not reshuffle its material.
3. **The probe.** Two equilibrated aggregates placed at contact must keep a persistent water layer in
   the gap and must NOT merge over a time long compared with their contact time.
4. **A memory.** The size distribution must carry a memory of how the aggregates were made (it is
   frozen), not a stationary shape the run keeps returning to.

### P3 — SPONTANEOUS CURVATURE / PACKING would show

Mean per-tail length is 2.372 beads with a strongly hydrated head (water–head depth 1.1429 of the
tail–tail reference), which biases the preferred aggregate towards micelles and rods. If that is the
mechanism:

1. **Merges AND fissions, balanced.** Both must occur and roughly cancel: the size distribution is
   *stationary* because it is the equilibrium one, with material exchanging freely.
2. **Stationary peaked distribution.** The largest aggregate must not grow even where contacts are
   frequent, and the whole distribution must be reproducible sample to sample.
3. **Rod shape at ALL sizes.** `inPlaneSymmetry` must sit well below the 0.50 a sheet needs for every
   aggregate, not just the largest — a preferred *shape*, not a large object that failed to flatten.
4. **The tail sweep.** At matched amphiphile count, a longer mean tail must move `inPlaneSymmetry` UP
   toward 0.50 and raise the largest aggregate; short tails must keep both low.

**How they are told apart.** P2 and P3 make OPPOSITE predictions about merges and fissions (frozen vs
freely exchanging and balanced) — one measurement separates them. P1 is separated from both by the
surface-coverage number, which is a static property of one snapshot and cannot be argued away.

---

## 1. VERDICT UP FRONT, TEN LINES

1. **P1 (free-head poisoning) is REFUTED, structurally and by measurement.** `data/soup.json`'s own
   pair-depth table gives a free polar head **zero** attraction to any bead of an aggregate
   (`polarPolar = polarApolar = 0.0`), while water attracts a head at **1.1429×** the tail–tail
   reference. Measured over 13 settled wet snapshots at box 54: free heads in the contact shell are
   **2.2–2.6× DEPLETED relative to water** (`enrichmentVsWater` **0.389–0.456**), and
   g_O(r)/g_W(r) in the first occupied bins is **0.40–0.57**, reaching 1.0 only at ~2.2 σ. A species
   excluded from a surface cannot cap it (§2).
2. **P2 (a barrier) SURVIVES in its dynamics half.** 34–44 aggregate pairs sit within 8 σ per
   snapshot; **51 encounter-intervals with a surface gap ≤ 4 σ, followed across 10 500 steps each,
   gave 1 merge, 34 still-in-contact, 16 drifted-apart** — a merge probability of **2.0 % per
   encounter-interval**. Over 126 000 settled steps: **1 merge, 0 fissions**, amphiphile-level
   exchange **exactly 0 at 9 of 12 intervals**. The closest two surfaces in the box never come below
   **2.9 σ** with **10–35 water beads bridging** them (§3).
3. **P3 (curvature/packing) SURVIVES in its shape half, FAILS in its dynamics half, and its CAUSAL
   claim is refuted by this project's own gate.** Shape is a clean sequence with size: spheres below
   ~30 amphiphiles, and above ~110 **rods whose cross-section is FIXED — R⊥ = 4.006–4.165 σ,
   constant to ±2 %, while the long axis grows** (exponent of R⊥ vs n = **0.188**, of √λ₃ = **0.748**).
   Zero lamellar objects at 11 of 13 snapshots. But merges and fissions are not balanced (1 and 0),
   and the causal story cannot be "the tail is too short": the dominant amphiphile is head + 2
   carbons, which **is** the Cooke & Deserno lipid, and this project's own
   `tests/water-bilayer-area-move.test.ts` holds a stable bilayer of it in explicit water at
   ρ_W = 0.8 (§4).
4. **WHAT SURVIVES IS BOTH, JOINED BY ONE NUMBER — EDGE ENERGY.** With this project's own measured
   line tension λ = **10.77 ε/σ** and area per lipid a = **1.2098 σ²**, a finite bilayer *disc* of
   N = 151 pays **2.42 ε = 2.20 kT of edge energy per amphiphile**. That is why the object is a rod:
   at the sizes this run reaches, a disc is thermodynamically excluded and a capped cylinder is the
   correct equilibrium shape. The rod is not a failed sheet (§5).
5. **AND THE SIZE IS FROZEN AT THE REHYDRATION FRAGMENTATION.** 22 aggregates at the first settled
   wet sample (step 41 400) and **22 at the last** (174 400), 133 000 steps apart; mean size 73.0 →
   64.0; largest 200 → 151. The dry phase made ONE percolating mass of 1632; the affine rehydration
   tore it into 22 pieces, and with fusion at 2 % per encounter and monomer exchange at ~0
   (**99.0 % of amphiphiles are inside an aggregate**), nothing since has changed (§5.2).
6. **A BIGGER BOX IS ARITHMETICALLY DEAD, with the number.** The largest of n aggregates grows only
   like ln n (measured: 64.05·ln 22 = 198 predicted, 151 observed). Reaching 920 that way needs
   **n = exp(920/64.05) = 1.73 × 10⁶ aggregates → L = 2314 σ** — **42× over** the measurability
   ceiling of 55.69 σ and **1.5 × 10¹⁰ particles** against the Verlet ceiling of 429 496 (§5.3).
7. **A MEASUREMENT DEFECT FOUND AND SIZED, not fixed in the engine.** `engine/src/aggregate.ts`
   treats **z as an open boundary** (its own comment says so — it was written for a bilayer in
   vacuum), while the soup wraps all three axes every step. Every aggregate straddling the z face is
   counted twice. Measured over the decisive lineage: largest aggregate **135 → 151 (+11.9 %)** at the
   final checkpoint, 153 → 200 at the first; aggregate count 23–25 → 22–23. Pinned by a synthetic
   test. It does not change the verdict (151 against 920 is still **6.09×** short) but every
   aggregate-size number published in this lineage is ~12 % low (§6).
8. **THE FUSION PROBE WAS BUILT AND THE ENGINE REFUSED IT.** Four constructions, three shipped; the
   clean one had **worst contact 0.6957 σ = the parent's own, on the same untouched pair**, all 10 734
   bond lengths unchanged to **3.07e-6**, all 130 tethers ≤ 1.1046 σ against r_inf = 1.5, census
   exact — and still went **565 914/575 292 components non-finite inside 1000 steps**, with a
   200-iteration minimisation taking max|F| **2.2261e4 → 1.4835e1** and diverging anyway. One cause is
   diagnosed (a species-blind clearance test: the catalyst's bead radius is 1.2 against water's 1.0),
   and with it fixed the construction becomes over-constrained. **No merge statistic came from the
   probe.** It is reported as a probe that failed, never as a run (§7).
9. **THE ONE LEVER THAT MOVED THE NUMBER: tail length and the two-tailed fraction.** At matched box,
   matched N, matched ρ_tot and matched ρ_W, lowering O:C from 4.000 to 1.000 to 0.333 moved
   per-tail length **2.11 → 2.38 → 2.75**, the two-tailed head fraction **12 % → 30 % → 48 %**, and
   the **single-aggregate share 0.47 → 0.997 → 1.000** while supply moved only 1.32–1.38× (§8).
   Literature basis: Israelachvili's critical packing parameter (rank B citation, no constant changed).
10. **IS CLOSURE REACHABLE? NOT AS THE MODEL STANDS, and the honest answer to "under what condition"
    is a measurement that has not been made.** The tail-length lever's own single aggregate at box 30
    has **r_g = 13.15 σ (arm B) and 15.31 σ (arm C) in a 30 σ box, against 15.00 σ for a uniformly
    FILLED box** — i.e. at box 30 "one aggregate holding 100 %" is indistinguishable from percolation,
    which is the predecessor's own concern 5 applying to my own positive result. The next measurement
    is that composition at box 54, where the recomputed floor is **998** and the window is
    **L ∈ [40.9, 55.69] σ** with a supply margin of **2.30×** (§9). If that object is a network and
    not a compact aggregate, then this model cannot reach closure without a change that would not be
    defensible, and §9.3 names what a next model would need.

---

## 2. MECHANISM 1 — FREE-HEAD POISONING: REFUTED

### 2.1 The structural half: there is no adsorption term to poison with

`data/soup.json`'s `solvent.attractionScale.pairEpsilon.levels`, read from the file, not assumed:

| class pair | model meaning | ε kJ/mol | multiplier of the tail–tail reference |
|---|---|---|---|
| `apolarApolar` | tail–tail | 3.5 | **1.0 (reference)** |
| `solventSolvent` | water–water | 5.0 | 1.4286 |
| `solventPolar` | **water–head** | 4.0 | **1.1429** |
| `solventApolar` | water–tail | 2.0 | 0.5714 |
| **`polarPolar`** | **head–head** | **0.0** | **0.0** |
| **`polarApolar`** | **head–tail** | **0.0** | **0.0** |

A free unreacted head is a polar bead. An aggregate is polar heads plus apolar tails. **Both of those
pair classes have depth zero**, so the only interaction a free head has with any bead of an aggregate
is the WCA core — pure exclusion — while water attracts it at 1.1429× the tail–tail reference. Free
heads are therefore *preferentially solvated away from* the surface by construction. Real surfactant
chemistry's capping monomer works through an attractive head–surface term; this model has none.

### 2.2 The measured half: the surface coverage numbers

Measured off-GPU on all 13 settled wet checkpoints of the decisive box-54 lineage
(`tests/coalescence-mechanisms.test.ts`, artifact `verify/out/coalescence-dec54-trace.json`).
"Contact" is the WCA contact separation of the head–tail bead pair `data/params.json` itself declares
(`beadSizes.head_tail = 0.95` → **1.0663 σ**), not a round number.

| step | free heads in pool | free heads in contact | per aggregate head site | water in contact | water per head site | **enrichment vs water** |
|---|---|---|---|---|---|---|
| 41 400 | 40 072 | 654 | 0.408 | 5 256 | 3.295 | **0.389** |
| 90 400 | — | — | 0.400 | — | — | **0.455** |
| 174 400 | — | — | 0.374 | — | — | **0.436** |
| range over 13 | — | — | **0.369–0.410** | — | — | **0.389–0.456** |

The 0.37–0.41 free heads per surface head site is not a capping layer — it is exactly what a
40 000-bead bulk pool delivers to any surface it does not bind to, and **less than water delivers**:
1.63 % of the free-head pool is in contact against 4.17 % of the water pool.

The normalisation-free profile, `g_O(r)/g_W(r)` in 0.25 σ bins (both species probe the identical
geometry from the identical bulk, so the shell volume cancels exactly), last snapshot:

```
bin (sigma):  0.75  1.00  1.25  1.50  1.75  2.00  2.25  2.50
g_O/g_W:     0.519 0.403 0.571 0.793 0.870 0.987 1.038 ...
```

Free heads are depleted **hardest in the first shell** and reach parity only at ~2.2 σ. That is the
signature of exclusion, and it is the opposite of adsorption.

**P1's prediction (1) fails outright and (2) fails as a mechanism.** Its prediction (3) — the matched
arm — is measured in §8, and it cannot separate P1 from P3 in this model, because composition cannot
move the free-head concentration without moving tail length: heads ARE the chain terminators, so
fewer heads means longer tails, necessarily. That is a structural limitation of the model, stated
rather than worked around. What decides against P1 anyway is §2.1 plus the depletion above, and the
fact that the low-free-head arms also change the aggregate SHAPE — which P1 predicts nothing about.

---

## 3. MECHANISM 2 — A HYDRATION / FUSION BARRIER: SURVIVES ON THE DYNAMICS

All numbers on the **fully periodic** clustering (§6), which is the physically correct one for a bulk
soup, with the project's own z-open rule reported beside it.

### 3.1 Contacts are common

| step | aggregates | pairs ≤ 8 σ | pairs ≤ 4 σ | closest gap σ | water bridging the closest pair | box-wide bridging water |
|---|---|---|---|---|---|---|
| 41 400 | 22 | 37 | 6 | 2.916 | 35 | 82 |
| 90 400 | 23 | 41 | 5 | 3.209 | 12 | 42 |
| 132 400 | 22 | 35 | 5 | 3.445 | 10 | 46 |
| 174 400 | 22 | 34 | 3 | 3.189 | 14 | 36 |
| range | 21–23 | **34–44** | **2–6** | **2.916–3.580** | **10–35** | 27–82 |

**The caveat that must be stated, because it limits what "closest gap" can mean:** two aggregates
closer than the clustering cutoff (2.7225 σ) are, by the project's own definition, ONE aggregate. So
the observed floor of 2.9 σ cannot be distinguished from the definitional floor of 2.72 σ. What the
number does say is that no pair is ever found *at* the definitional edge for long, and that the pairs
that do approach carry 10–35 water beads within 2.5 σ of both surfaces.

### 3.2 And contacts almost never proceed — the discriminating statistic

Every pair whose surfaces were within 4 σ at one checkpoint was followed to the next (10 500 steps
later) and classified by whether both aggregates' amphiphile heads ended up in the same object. Head
particle indices are stable for the whole run — organics are never created or destroyed — so
chemistry cannot confuse this bookkeeping.

```
12 intervals, 126 000 settled wet steps, box 54:
  encounters followed (gap <= 4 sigma):  51
    -> MERGED:                            1     (2.0%)
    -> still in contact 10 500 steps on: 34     (66.7%)
    -> drifted apart:                    16     (31.4%)
  merges, whole box:                       1
  fissions, whole box:                     0
  amphiphile-level exchange fraction:      0.0000 at 9 of 12 intervals; max 0.0610
```

Arrivals without reactions, and long-lived ones: two thirds of encounters are still in contact 10 500
steps later without merging. **P2's predictions 1, 2 and 4 all hold.** Its prediction 3 — the probe —
could not be measured; see §7.

---

## 4. MECHANISM 3 — CURVATURE / PACKING: THE SHAPE HALF SURVIVES, THE CAUSAL CLAIM DOES NOT

### 4.1 Shape as a function of size, over every aggregate, not the largest five

112 aggregate observations (the last 5 snapshots × 21–23 objects). R⊥ = √(λ₁+λ₂) is the
cross-sectional gyration radius; √λ₃ is the long-axis one.

| size band | n obs | R⊥ σ | √λ₃ σ | flatness λ₁/λ₃ | in-plane λ₂/λ₃ | reads as |
|---|---|---|---|---|---|---|
| 13–30 | 38 | 2.12–3.09 | 2.01–2.57 | 0.32–0.82 | 0.44–0.96 | **spheres/spheroids** |
| 39–70 | 24 | 3.10–3.74 | 2.55–5.05 | 0.18–0.85 | 0.30–0.97 | transition |
| 86–103 | 20 | 3.43–4.30 | 4.15–4.96 | 0.15–0.41 | 0.18–0.55 | prolate |
| **111–155** | **10** | **4.006–4.165** | **6.188–6.739** | **0.165–0.199** | **0.204–0.250** | **rods, fixed cross-section** |

Summary statistics: `corr(n, R⊥) = 0.877`, `corr(n, √λ₃) = 0.936`; power-law exponents fitted over
n ≥ 40: **R⊥ ∝ n^0.188**, **√λ₃ ∝ n^0.748**. Over the top band (n = 111 → 155, a 40 % size increase)
R⊥ moves **4.006 → 4.165 (±2 %)** while λ₃ moves **38.3 → 44.3**.

**A size-independent preferred cross-section is a thermodynamic signature, and it is one a
kinetically frozen ensemble cannot manufacture** — a frozen population would carry whatever radii it
froze at, uncorrelated with size. So the SHAPE at each size is the equilibrium shape. `lamellarCount`
(flatness ≤ 0.35 **and** in-plane ≥ 0.50, the project's own lamellar gate) is **0 at 11 of the 13
snapshots** under both clusterings: no object at any size is a sheet.

Note what this corrects: quoting the largest aggregate's in-plane symmetry alone (0.245) reads as "a
sheet that failed to flatten". The distribution says otherwise — mean in-plane symmetry over all
objects is **0.58–0.61** with a max of **0.95–0.98**, because the *small* ones are spheres, for which
λ₂/λ₃ → 1. In-plane symmetry alone cannot tell a sphere from a sheet; flatness is what separates them,
and no object passes both.

### 4.2 Where P3 fails

- **Its dynamics prediction fails.** It required merges and fissions both to occur and roughly
  cancel. Measured: **1 merge, 0 fissions** over 126 000 steps, and zero monomer exchange at 9 of 12
  intervals. The distribution is not maintained by exchange; it does not move at all.
- **Its causal claim is refuted by this project's own passing gate.** The per-tail histogram at step
  174 400 is `{2: 1333, 3: 278, 4: 86, 5: 32, 6: 14, 7: 6, 8: 1, 9: 2}` — **76 % of all tails are
  exactly 2 beads**, and 857 of 1423 amphiphiles are head + 2 carbons on a single tail. That is
  precisely the Cooke & Deserno lipid's geometry, and `tests/water-bilayer-area-move.test.ts` holds a
  stable bilayer of it **in explicit water at ρ_W = 0.8**, re-measured this task at
  a = **1.2098 σ²**, t = **4.9421 σ**, `waterInCore` 54/4500, `verdict=passed`. The molecule can make
  a bilayer. What it cannot make is a *small* one — §5.

---

## 5. WHAT SURVIVES: EDGE ENERGY AT A FROZEN SIZE — AND THE ARITHMETIC

### 5.1 Why a 151-molecule aggregate is a rod and not a disc

The passing bilayer gate is a **periodic** slab: it has no edge. A finite patch does. Using this
project's own measured numbers — line tension λ = **10.77 ± 1.06 ε/σ**
(`verify/out/line-tension.json`, `kappa-tightening-report.md` §2) and this task's own re-measured
a = **1.2098 σ²** — a bilayer disc of N amphiphiles has R = √(N·a/2π) and edge energy 2πRλ:

| N | disc radius σ | edge energy ε | **per amphiphile** | in kT (kT = 1.1) |
|---|---|---|---|---|
| **151** (measured largest) | 5.392 | 364.9 | **2.4164 ε** | **2.197 kT** |
| 394 (arm B, §8) | 8.710 | 589.4 | 1.4959 ε | 1.360 kT |
| **729** | 11.848 | 801.7 | 1.0998 ε | **1.000 kT** |
| 920 (decisive floor) | 13.309 | 900.7 | 0.9790 ε | 0.890 kT |
| 4 480 (= R_c) | 29.370 | 1987.5 | 0.4436 ε | 0.403 kT |

At the size this run reaches, a disc costs **2.2 kT per molecule** in edge energy alone. A capped
cylinder does not — its end caps are a favourable geometry for this packing, which is exactly what the
fixed cross-section of §4.1 measures. So the rod is the correct equilibrium object at N = 151, and the
size at which a disc becomes competitive (edge ≲ 1 kT per molecule) is **N ≈ 729**. Spontaneous
closure of such a disc needs, by the project's own criterion R_c = 4κ/λ = **29.37 σ**
(κ = 79.07 ± 7.32 ε, n = 10), **N = 2πR_c²/a = 4479**.

Three sizes, in order: **rod → disc at ~729; the closure gate's own volume floor at 920–998;
disc → closed vesicle at ~4480.** Measured largest: **151**.

### 5.2 Why the size does not grow: the distribution is born at rehydration and frozen

| step | phase | aggregates (3D) | largest | mean size | amphiphiles |
|---|---|---|---|---|---|
| 31 000 | **dry**, ρ_org = 1.34 | 8 (percolating, meaningless) | 1632 | — | 1649 |
| **41 400** | first settled wet | **22** | 200 | 73.0 | 1607 |
| 90 400 | wet | 23 | 162 | 64.0 | 1471 |
| 132 400 | wet | 22 | 155 | 64.3 | 1431 |
| **174 400** | wet, +133 000 steps | **22** | 151 | 64.0 | 1423 |

The dry phase makes ONE contact-percolating mass. The affine rehydration ramp — which rescales
molecular centres of mass rigidly — **tears it into 22 pieces**, and the size distribution the run
ends with is that fragmentation spectrum. Nothing since has changed it: the aggregate count is
identical 133 000 steps later, and the decline in the largest (200 → 151) tracks the amphiphile
supply's own chemical decline (1607 → 1423, `co_break` 0 → 167) rather than any dispersal.

The reason it cannot relax is that **both** relaxation channels are shut:
- **aggregate fusion**: 2.0 % per encounter-interval (§3.2);
- **monomer exchange**: the sum of qualifying aggregate sizes at step 174 400 is 1409 of 1423
  amphiphiles, so **99.0 % of the population is locked inside an aggregate** and the free-monomer
  reservoir a Aniansson–Wall exchange would need is effectively empty.

This also explains the box-scaling result the predecessor could not explain. The number of fragments
scales with volume (6 at box 30, 22 at box 54 — 3.7× for a 5.83× volume) while the *mean* fragment
size barely moves (78–91 at box 30, 64 at box 54). Supply scales as L³; the largest fragment does not.

### 5.3 So a bigger box is arithmetically dead — the number

For a broad (near-exponential) fragment distribution of mean m, the largest of n fragments grows like
m·ln n. Measured at box 54: m = 64.05, n = 22 → predicted 198, observed **151** (the same relation
predicts 96–109 at box 30 against 108–154 observed). Inverting it for a largest of 920:

```
n = exp(920 / 64.05) = 1.732e6 aggregates
box volume = (1.732e6 / 22) * 54^3 = 1.240e10 sigma^3
L = 2314 sigma        against the measurability ceiling 55.69 sigma  -> 42x over
N = 1.5e10 particles  against the Verlet ceiling      429 496        -> 35 000x over
```

**That is the number that closes the box road.** It is also why this task's first move was not a
bigger box.

---

## 6. A MEASUREMENT DEFECT, FOUND AND SIZED

`engine/src/aggregate.ts`'s `buildClusterUnionFind` — which `clusterComponents`,
`largestClusterFraction` and therefore `soup/src/aggregates.ts` and every stage detection all go
through — says in its own comment: *"z is unbounded (open boundary) so it gets a plain integer cell
index of width `cutoff` with no wrapping"*, and computes `ddz = zi - positions[j*4+2] // z open: no
periodic image`. That is correct for the membrane engine (a bilayer patch in vacuum has no image
across z) and **wrong for the soup**, whose own header states it wraps all three axes every step. An
aggregate straddling the z face is split and counted twice.

Sized, not argued (`tests/helpers/coalescence-geometry.ts`'s `clusterComponents3D` — the identical
connectivity rule with the z image restored, same cutoff, same union-find):

| step | z-open aggregates | z-open largest | **3D aggregates** | **3D largest** | largest understated by |
|---|---|---|---|---|---|
| 41 400 | 25 | 153 | 22 | **200** | 30.7 % |
| 90 400 | 25 | 142 | 23 | **162** | 14.1 % |
| 132 400 | 23 | 138 | 22 | **155** | 12.3 % |
| 174 400 | 23 | 135 | 22 | **151** | **11.9 %** |

Pinned in both directions by `tests/coalescence-mechanism-pin.test.ts` on synthetic geometry (one
8-bead blob straddling z = 0: `zOpen = 4+4`, `periodic3d = 8`).

**Not fixed in the engine here, deliberately.** Changing `clusterComponents` changes the number every
stage gate, every published aggregate count and every regression in this project reads, and this task
is a measurement task; it would need its own before/after across the whole suite. It is reported,
pinned, and its size is known. **It does not change the verdict:** 151 against 920 is 6.09× short
where 135 against 920 was 6.81×.

---

## 7. THE FUSION PROBE — BUILT, SHIPPED, AND REFUSED BY THE ENGINE

It is a probe and it is reported as one. It produced **no merge statistic**.

### 7.1 What was built

`tests/helpers/fusion-probe-state.ts` + `tests/fusion-probe.test.ts`. The construction is a
**permutation of positions inside the parent box**, not a new system: an aggregate is displaced
rigidly, and every free molecule its new footprint would overlap is lifted out and put back — rigidly,
site and orientation — somewhere clear. So box, composition, ρ_tot, ρ_W, free-monomer concentrations,
bond graph, velocities and both RNG streams are the parent's own, bit for bit.

The state that shipped (from `dec54-step174400`, one forced contact, 59 + 19 amphiphiles brought from
their natural 3.189 σ down to **1.8 σ** — inside the project's own 2.7225 σ clustering cutoff):

```
FUSE-BUILD-PAIRS [{"anchorN":59,"moverN":19,"gap":3.189}]
FUSE-BUILD-PARENT-WORST minSeparation=0.6957 pair=56480-62945
FUSE-BUILD {"moved":60,"displacement":[1.5131,-0.0621,-0.1836],"gapBefore":3.1889,"gapAfter":1.8,
            "intruders":123,"removedUnits":91,"relaxedPlacements":1,"leftover":0}
FUSE-BUILD-PROBE-WORST minSeparation=0.6957 pair=56480-62945 parent=0.6957
FUSE-BUILD-BONDS maxBondedDistanceChange=3.073e-6 bonds=10734
FUSE-BUILD-TETHER desorbedByDisplacement=0 live=130 longest=1.1046 rInf=1.5
```

Independently re-audited off the written file: max bond length **1.2030** (parent 1.2030), max tether
**1.1046** (parent 1.1046), 130 live tethers (parent 130), 0 over `r_inf`, 0 positions outside
[0, 54), census exact, 183 of 191 764 beads moved (`{C:70,O:38,H:34,M:2,W:39}`).

### 7.2 What happened, three times

```
$ nice -n 15 npx tsx soup/cli/campaign.ts --label fuse54 --box 54 --start '{...}' --seed 19 --kT 1.1 \
    --steps 20000 --every 5000 --dir data/checkpoints/fuse
[campaign] resume label=fuse54 from data/checkpoints/fuse/fuse54-step174400.json, step=174400
[campaign] system ready N=191764 startStep=174400 target=194400
Error: non-finite state at step=175400: non-finite position components=565914, velocities=565914
       of 575292 (appeared within the step interval 174400..175400) -- the computation diverged
```

With a mid-run minimisation asked for at the first step:

```
[campaign] minimization MID-RUN at step=174401 iterations=200 max|F| 2.2261e+4 -> 1.4835e+1
Error: non-finite state at step=174901: ... 572436 of 575292
```

**The control that exonerates the resume path**: the unmodified parent, copied into its own directory
with the identical no-cycling config, resumed and ran clean —

```
[campaign] step=175400 stage=micelles aggregates=33 largest=135 headShells=2 phase=none/0 box=54.0000
           bonds=10736 census={"C":10845,"O":43380,"H":10845,"M":723,"W":125971,"K":0}
[campaign] step=176400 stage=micelles aggregates=33 largest=136 ... bonds=10739
```

So the divergence is the surgery, not the resume, not the config.

### 7.3 One cause diagnosed, and why the construction then closed

`max|F| = 2.2261e4` is the fingerprint. My clearance test was **species-blind** — a flat 0.75 σ — but
this engine's pair bead size is `sigma*(r_i + r_j)/2` (`soup/src/soup-clay.ts:143`) and the catalyst's
`radiusSigma` is **1.2** against water's 1.0. A catalyst placed 0.75 σ from a water bead sits at 0.61
of *its own* contact distance, where 24ε[2(b/r)¹²−(b/r)⁶]/r ≈ 1.8e4. Two catalyst beads moved in that
build. Fixed: the clearance is now required in units of each pair's own b_ij.

With the corrected criterion the construction becomes over-constrained: at every one of eight target
gaps from 1.2 σ to 2.6 σ, **one free carbon chain in the swept volume has no clear site anywhere** —
site searches over the molecule's own neighbourhood, over the 45–130 vacancies the lift itself created,
over the volume the aggregate vacated, and over the whole box (60 000 draws each, with random
orientation) all fail, because a uniformly random site clears at p ≈ 0.116 per bead at ρ_tot = 1.218
and a six-bead chain therefore at ~10⁻⁶. A probe that ships an unplaced molecule is not a probe, so
none was shipped.

**Status, stated plainly: the probe is unfinished. What stands for P2 is §3.2's in-situ encounter
statistic.** The residual suspicion, which the next task should test in one run, is that some piece of
resumed state I am not reproducing consistently is position-dependent — the state passes every static
check this project owns (worst contact, bond lengths, tether lengths, census, in-box positions) and
still diverges, which is itself worth knowing.

---

## 8. THE TAIL-LENGTH ARMS — P3'S PREDICTION 4, AT MATCHED DENSITY

Box 30, **N = 32 884, ρ_tot = 1.21793, ρ_W = 0.8000, M = 124, W = 21 600, seed 19, kT = 1.1** in all
three arms — the composition is redistributed among C/O/H at *constant* particle count and *constant*
water density, so nothing but the head:carbon ratio changes. Flags identical to the decisive report's
own box-30 arms: `--relax --cycle --evaporate --steps 40000 --every 10000`. Arm A **is** those arms
(seeds 19/23/29, already on disk), re-audited here with the same code as B and C.

```
A  O:C = 4.0000   C=1860 O=7440 H=1860   N=32884  rho_tot=1.21793  rho_W=0.8000
C  O:C = 1.0000   C=3720 O=3720 H=3720   N=32884  rho_tot=1.21793  rho_W=0.8000
B  O:C = 0.3333   C=4783 O=1594 H=4783   N=32884  rho_tot=1.21793  rho_W=0.8000
```

| arm | O:C | amphiphiles | per-tail | two-tailed heads | aggregates (3D) | **largest** | **share** | free heads / head site | enrichment vs water | lamellar |
|---|---|---|---|---|---|---|---|---|---|---|
| **A** | 4.000 | 264 / 274 / 313 | 2.079–2.137 | 31–41 (**12 %**) | 3–4 | **108 / 147 / 154** | 0.394 / 0.470 / 0.583 | 0.431–0.506 | 0.365–0.451 | 0–1 |
| **C** | 1.000 | 371–376 | 2.363–2.394 | 106–116 (**30 %**) | **1** | **370–375** | **0.997** | 0.116–0.141 | 0.249–0.267 | 0 |
| **B** | 0.333 | 392–394 | 2.705–2.793 | 185–190 (**48 %**) | **1** | **392–394** | **1.000** | 0.025–0.041 | 0.162–0.223 | 0 |

Read against the supply change, which is what "matched count" is for: amphiphile supply moves
**1.32–1.38×** (284 mean → 374 → 393) while the largest aggregate moves **2.7–2.9×** and the
single-aggregate share moves **0.47 → 1.000**. The lever is not supply.

**Shape of the largest, at the last sample of each arm:**

```
arm A (154 amphiphiles):  rg=12.50  flat=0.1066  inPl=0.5135     + 65 and 42 amphiphile satellites
arm C (370 amphiphiles):  rg=15.31  flat=0.5188  inPl=0.7019     one object, 99.7% of supply
arm B (394 amphiphiles):  rg=13.15  flat=0.3900  inPl=0.6369     one object, 100% of supply
```

The shape moves off the rod (flatness 0.107 → 0.39–0.52, in-plane 0.51 → 0.64–0.70) but **not onto a
sheet**: `lamellarCount = 0` in every arm. And the confound is stated: this arm moves free-head
concentration (0.46 → 0.03 per head site) and tail length together, because in this model heads *are*
the chain terminators and the two cannot be separated by composition. P1 is refuted independently
(§2), and the shape change is something P1 predicts nothing about — but the arm on its own does not
separate them, and it is not presented as if it did.

### 8.1 The percolation caveat, with the number that makes it one

The r_g of a **uniformly filled** 30 σ box is √(3·30²/12) = **15.000 σ**. Arm C's single aggregate has
r_g = **15.31 σ** and arm B's **13.15 σ**; arm A's largest has **12.50 σ**. At box 30, "one aggregate
holding 100 % of the supply" is therefore **not distinguishable from a box-spanning percolating
network** — which is precisely the predecessor's concern 5, here applying to my own positive result.
The lever's *aggregate-count* claim (4 discrete objects → 1 connected object) survives that caveat;
its *size* claim does not, and must be re-measured at a box where the object is small compared with
the box.

---

## 9. IS A CLOSED VESICLE REACHABLE, AND UNDER WHAT CONDITION

### 9.1 Not as the model stands. Four numbers.

1. **Growth is barred.** 1 merge per 51 encounter-intervals (2.0 %), 0 fissions in 126 000 steps,
   monomer exchange exactly 0 at 9 of 12 intervals, and 99.0 % of amphiphiles locked inside an
   aggregate — there is no channel by which the size distribution can relax.
2. **The distribution is not an equilibrium one to begin with.** 22 aggregates at the first settled
   wet sample and 22 at the last, 133 000 steps apart: it is the rehydration ramp's fragmentation
   spectrum, frozen.
3. **A bigger box needs L = 2314 σ** — 42× the measurability ceiling, 35 000× the Verlet particle
   ceiling (§5.3).
4. **And even a sheet of the required size would not close spontaneously below N ≈ 4480** by the
   project's own R_c = 4κ/λ = 29.37 σ, against a measured largest of 151.

### 9.2 The condition, stated as the measurement that would settle it

The only lever measured to move the number by more than its supply change is **tail length and the
two-tailed head fraction** (§8), and it has a literature criterion behind it rather than a fit:

> **Israelachvili's critical packing parameter** p = v/(a₀·l_c): p < 1/3 → spherical micelles;
> 1/3 < p < 1/2 → **cylindrical micelles (rods)**; 1/2 < p < 1 → **vesicles and flexible bilayers**;
> p ≈ 1 → planar bilayers. — J. N. Israelachvili, D. J. Mitchell & B. W. Ninham, *Theory of
> self-assembly of hydrocarbon amphiphiles into micelles and bilayers*, J. Chem. Soc. Faraday Trans.
> II **72** (1976) 1525; and Israelachvili, *Intermolecular and Surface Forces*, 3rd ed., ch. 20.
> **Rank B** — a literature *criterion*, not a constant imported into this model. Doubling v at fixed
> l_c (a second tail on the same head) doubles p; lengthening one tail raises v and l_c together and
> so buys much less. The measured aggregate shape at O:C = 4 (rods, fixed cross-section, §4.1) places
> this amphiphile squarely in the 1/3–1/2 band, which is exactly where the criterion says a rod is
> the correct object.

**No constant was changed to exercise it.** `data/params.json` untouched; `data/soup.json` untouched;
`co_bond.attemptRate` untouched. The arms are `--start` flags — an experiment-design number in exactly
the sense `--steps` is (rank D, measured here).

**The one measurement that would settle reachability, with its arithmetic done.** Run arm B's
composition at box 54, where the object can be small compared with the box, and ask whether the single
aggregate is compact or a network. Floor and window recomputed from **this task's own** measured
thickness and area per lipid (`tests/water-bilayer-area-move.test.ts`, t = 4.9421 σ, a = 1.2098 σ²;
`stageThresholds.enclosedVolume` = 370.8656 σ³ untouched):

```
R_in  = (3 * 370.8656 / 4pi)^(1/3) = 4.45700 sigma
R_mid = R_in + t/2 = 4.45700 + 2.47105 = 6.92805 sigma
floor = 2 * 4pi * R_mid^2 / a = 2 * 4pi * 48.0019 / 1.2098 = 997.1  ->  998
  (solvent-free gate 6 this task: t=4.4761, a=1.2025 -> 937; band 937-998, the two gates 6.1% apart)

supply:        rho_amph(arm B, measured) = 394 / 27 000 = 0.014593 sigma^-3
               L >= (998 / 0.014593)^(1/3) = 40.89 sigma
measurability: R(L) = 0.067 * L^1.5 <= L/2  ->  L <= 55.69 sigma   (unchanged, box-58 calibration)
WINDOW: L in [40.89, 55.69] sigma, width 14.80 sigma  -- the widest this project has had
At L = 54: supply = 0.014593 * 157 464 = 2298 against 998 -> 2.30x margin
Verlet at that composition/box: scale arm B's 32 884 by 54^3/30^3 = 5.832 -> N ~ 191 800
               191 800 * 2500 * 4 = 1.918 GB against 4.295 GB; 191 800 against 429 496.  VALID.
```

**That campaign was not run** — it is ~6 chunks and this task's budget went into the three
discriminating measurements and the failed probe, as instructed. It is the single next measurement,
and it is now fully specified.

### 9.3 If that object is a network: what a next model would need

Then the honest answer is that **this model cannot reach closure without a change that would not be
defensible**, because the remaining levers are all illegitimate: lowering the closure gate's
enclosed-volume minimum (a threshold), re-fitting `co_bond.attemptRate` to manufacture two-tailed
heads (refused three times), weakening the head's hydration (`solventPolar` 1.1429 → a less polar
MARTINI bead) purely because it flatters the result, or a box 42× over the measurability ceiling.

**What a next model would need to be different**, named from the numbers above:

1. **A two-tailed amphiphile by construction, not by emergence.** p = v/(a₀l_c) must sit in the
   1/2–1 band *by topology*, so that the preferred object is a bilayer — which has no finite
   equilibrium size and therefore no aggregate-size wall at all. Emergent two-tailing gives 12 % at
   the defended composition and 48 % only at a head:carbon ratio that also triples the tail length.
2. **A non-zero monomer solubility — a real CMC the run sits above.** With 99.0 % of amphiphiles
   locked inside aggregates, the Aniansson–Wall exchange channel that lets a real surfactant
   population find its equilibrium distribution does not exist here, so whatever the rehydration ramp
   tears the dry mass into is final. That is a property of the tail–tail depth against kT, not of the
   box.
3. **A rehydration that does not fragment.** The affine ramp rigidly rescales centres of mass and
   therefore tears a percolating mass along its weakest links (1632 → 22 pieces in 3600 steps). A
   dilution protocol that lets the mass reorganise between increments — or a route to the aggregate
   that never goes through a percolating dry state — would not need fusion to work at all.

---

## 10. REGRESSIONS — every one run, with before → after

**The whole diff is TEST-ONLY.** No file under `soup/`, `engine/`, `chem/`, `viewer/`, `verify/src`
or `data/` was modified — `git diff --name-only` is four files under `tests/` plus this report plus
four `verify/out/*.json` artifacts. So every regression below is a proof that the measurement code
added here did not disturb anything, and any movement in a number is the engine's own
non-reproducibility (atomics in the grid fill vary the force summation order — the predecessor's own
concern 11).

| test file | result | numbers, predecessor → this task |
|---|---|---|
| `tests/gate6-bilayer.test.ts` | **2 passed** | area **1.2075±0.0073 → 1.2025±0.0125** [1.1768, 1.2300]; thickness **4.4762 → 4.4761** (per-frame sd 0.2289, n=400); lnA drift/move 5.702e-5 (t=3.97); accepted 0.297; escapedMax 0; box 24.562, 83 000 steps. Both inside the untouched corridors 1.1–1.5 and 4–6. |
| `tests/water-bilayer-area-move.test.ts` | **1 passed** | area **1.2372 → 1.2098** [1.2026, 1.2145]; thickness **4.5472 → 4.9421**; drift t 1.38 → **1.21** (1.278e-5/chunk); water in core 50 → **54**/4500; buried 0.0870 → **0.0921**; cluster **1.0000**; accepted 0.1283 of 3000; throughput 1215.11 steps/s at N=5700; **verdict=passed**. This row is what §9.2 recomputes the floor from. |
| `tests/soup-stale-force.test.ts` | **2 passed** | `postResidentVsFresh` **1.98e-4** against a force scale of 568 (3.5e-7 relative); `postResidentVsPreFresh` **344.19**; `preResidentVsFresh` **0**. `MIDRUN-MINIMISE` neutrality holds: census/charge/bondSet/events identical, steps 2000→2000, box unchanged, nonFinite 0. |
| `tests/catalyst-turnover.test.ts` | **1 passed** | cc_bond 432 → **447**, co_bond 354 → **326**, chains 181 → **189**, amphiphiles 132 → **135**, bare carbons left 47 → **29**, occupiedAtEnd 2 → **3**, desorbTimeout 5 → **3**, nonFinite 0, charge 0, census `{C:779,O:3117,H:779,M:52,W:695,K:0}` exact. Turnover ordering (ccOverLast15k=2 > 0, occupied falls 21→3) holds. |
| `tests/soup-nonfinite-guard.test.ts` (the loud-failure guard) | **1 passed** | fires at step 1 with **120/120** non-finite components; healthy control 0/0 at three points; cost **0.2060 ms** vs chunk1000 **333.10 ms** = **0.062 %** (1.00062× throughput). |
| `tests/soup-cold-start-relax.test.ts` | **1 passed** | max\|F\| **3.51806e4 → 19.104** (1842×); maxDisplacement **1.11838 σ** ≤ bound 10.05; steps 0→0; velocities, both RNG streams, bond graph and all four event counters unchanged; refusal at `globalStep != 0` fires; payoff pair holds (throws at step 1000 without the stage, 29 232/29 232 non-finite). |
| `tests/soup-evaporation.test.ts` | **3 passed** | `EVAP-PLAN` identical to the predecessor's to every digit (1440.0×, dryBox 20.3538, concentrationFactor **3.2021×**, relaxIterations 29, logFractionOf1400 16.07 %); full cycle `dryBox=16.2811 (plan 16.2811) N_dry=5783 (plan 5783) rhoDryRealised=1.34000`, box and census return exactly, nonFinite 0 at four points, events `{cc_bond:398, co_bond:348, co_break:3}`. |
| `tests/soup-drywet-cycling.test.ts` | **2 passed** (see note) | `concentrationFactor=1.1002` and `rampStepsCharged=1000` **identical**; `dryBox=29.0598`, `realisedDryDensity=1.34000`, box returns exactly, nonFinite 0/0 at four points. Mobility collapse 3.79× → **2.99×**. One-cycle event ratio **0.7171 → 0.6961**. **NOTE:** this file FAILED on its first invocation here and PASSED on an immediate re-run of the identical command with no change of any kind in between — its assertion sits at the margin of the engine's own run-to-run scatter. Both runs' numbers are above. My diff is test-only and cannot be the cause. |
| `tests/sim.test.ts` | **8 passed** | unchanged (grid-vs-brute-force force identity, thermostat, frictionless energy drift) |
| `tests/run-ui.test.ts` | **8 passed** | unchanged |
| `tests/soup-forces.test.ts` | **1 passed** | unchanged — grid + Verlet list vs full O(N²), no GPU console warning |
| `tests/soup-area-move.test.ts` | **2 passed** | unchanged — compose-vs-direct, CPU potential = antiderivative of GPU forces, jacobian |
| `tests/soup-bonds.test.ts` | **3 passed** | catalyst gating holds; census+charge conserved under live reactions; assembly-vs-temperature ordering holds |
| `tests/soup-valence.test.ts` | **1 passed** | no carbon over 2 C–C or 1 C–O, no head over chainCapacity, 45 000 steps |
| `tests/soup-checkpoint.test.ts` | **3 passed** | round-trip exact; "a checkpoint does not change what the run computes next"; same-process continue writes real files |
| `tests/soup-boxcycle.test.ts` | **7 passed** | pure coordinate map, involution, long-coiled-chain regression, `computeDryBox` density, schedule — unchanged |
| `tests/soup-rules.test.ts` | **4 passed** | schema + detailed balance unchanged; the four `dryWetCycle` evaporation fields still validate |
| `tests/params.test.ts` | **3 passed** | the literal scanner is clean; **no rank-A constant appears anywhere new** — which is the machine check that this task changed no constant |
| `tests/soup-grid-resize.test.ts` | **1 PASSED** | The predecessor recorded this as "PRE-EXISTING — FAILED". **It PASSED here.** That is the same non-reproducibility the predecessor itself documented: it measured the drift scattering over four orders of magnitude on unchanged code (4.4e2 – 3.4e7 across six runs, both sides of its own stash). Its attribution is unchanged and now doubly safe: **my diff is test-only, so this file belongs on the PRE-EXISTING side whichever way it lands.** No bound was widened. |
| **`tests/coalescence-mechanism-pin.test.ts`** (NEW, a gate) | **3 passed** | see §11 |
| `tests/coalescence-mechanisms.test.ts` (NEW, harness) | skips unsent | 13 + 3 + 2 + 2 snapshots measured, four committed artifacts |
| `tests/fusion-probe.test.ts` (NEW, harness) | skips unsent | build passes; the GPU half is §7 |

**No tolerance was changed, no corridor widened, no assertion relaxed, no threshold moved.**
`tests/soup-vesicle.test.ts` was **never** run. The full suite was never run. `--dump-dom` was never
used. No particle array was ever transferred as JSON numbers — every reduction ran inside the offline
auditor or inside the page and came back as scalars.

### 10.1 The new gate's own numbers

```
$ nice -n 15 npx vitest run tests/coalescence-mechanism-pin.test.ts --no-file-parallelism
COAL-PIN enrichmentVsWater over 13 snapshots: [0.3894, 0.4549]
COAL-PIN merges=1 fissions=0 encounters=51 ofWhichMerged=1
COAL-PIN clustering: zOpen=4+4 periodic3d=8
 ✓ tests/coalescence-mechanism-pin.test.ts (3 tests) 3ms
      Tests  3 passed (3)
```

It asserts, as a gate with no env and no GPU: (a) `polarPolar` and `polarApolar` are exactly 0 and
`solventPolar` exceeds the tail–tail reference — the structural half of §2 — and that the measured
free-head enrichment is **below 1 at every snapshot and in every first-shell bin**; (b) **zero
fissions and at most one merge per interval**, with the merge-per-encounter probability below 10 % —
the frozen-distribution claim; (c) the z-open clustering defect, in both directions, on synthetic
geometry.

---

## 11. EVERY COMMAND, with real unedited output

Measurement commands whose output is quoted in full above: §2.2, §3.1–3.2, §4.1, §7.1–7.2, §8, §10.1.
The full list, in order:

```
# --- the three discriminating measurements, off-GPU on the decisive lineage's own checkpoints
COAL_CHECKPOINTS="<13 dec54 wet checkpoints>" COAL_ARTIFACT=verify/out/coalescence-dec54-trace.json \
  nice -n 15 npx vitest run tests/coalescence-mechanisms.test.ts --no-file-parallelism

# --- the fusion probe: build (off-GPU), then run (GPU), then the control
FUSE_BUILD=data/checkpoints/dec54/dec54-step174400.json FUSE_DIR=data/checkpoints/fuse \
  FUSE_LABEL=fuse54 FUSE_GAP=1.2 nice -n 15 npx vitest run tests/fusion-probe.test.ts --no-file-parallelism
nice -n 15 npx tsx soup/cli/campaign.ts --label fuse54 --box 54 \
  --start '{"C":10845,"O":43380,"H":10845,"M":723,"W":125971}' --seed 19 --kT 1.1 \
  --steps 20000 --every 5000 --dir data/checkpoints/fuse                       # DIVERGED, see 7.2
nice -n 15 npx tsx soup/cli/campaign.ts --label fuse54 ... --minimiseAt 174401 \
  --minimiseIterations 200 --steps 1500 --every 500 --dir data/checkpoints/fuse # DIVERGED, see 7.2
nice -n 15 npx tsx soup/cli/campaign.ts --label fusectl --box 54 --start '{...}' --seed 19 --kT 1.1 \
  --steps 2000 --every 1000 --dir data/checkpoints/fusectl                      # CONTROL, ran clean

# --- the tail-length arms (arm A already on disk from the decisive report)
nice -n 15 npx tsx soup/cli/campaign.ts --label ocB19 --box 30 \
  --start '{"C":4783,"O":1594,"H":4783,"M":124,"W":21600}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --steps 40000 --every 10000 --dir data/checkpoints/ocB19
nice -n 15 npx tsx soup/cli/campaign.ts --label ocC19 --box 30 \
  --start '{"C":3720,"O":3720,"H":3720,"M":124,"W":21600}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --steps 40000 --every 10000 --dir data/checkpoints/ocC19
COAL_CHECKPOINTS=... COAL_ARTIFACT=verify/out/coalescence-tail-arm{A,B,C}.json \
  nice -n 15 npx vitest run tests/coalescence-mechanisms.test.ts --no-file-parallelism   # x3

# --- regressions
nice -n 15 npx vitest run tests/gate6-bilayer.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/sim.test.ts tests/run-ui.test.ts tests/soup-forces.test.ts \
  tests/soup-area-move.test.ts tests/soup-bonds.test.ts tests/soup-valence.test.ts \
  tests/soup-checkpoint.test.ts tests/soup-boxcycle.test.ts tests/soup-rules.test.ts \
  tests/params.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/catalyst-turnover.test.ts tests/soup-nonfinite-guard.test.ts \
  tests/soup-cold-start-relax.test.ts tests/soup-stale-force.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/soup-evaporation.test.ts tests/soup-drywet-cycling.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/soup-drywet-cycling.test.ts --no-file-parallelism      # the re-run, 10
nice -n 15 npx vitest run tests/soup-grid-resize.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/coalescence-mechanism-pin.test.ts --no-file-parallelism
```

Arm B's campaign line, verbatim, since it carries the headline of §8:

```
[evaporation] rehydration: inserted=21585 below_threshold=984 min_distance=0.6286
              minimization_iterations=29 max|F| 2.5431e+4 -> 6.2736e+1 previous_displacement rms=0.2116 max=0.7397 bound=1.5000
[campaign] step=36800/40000 stage=amphiphiles aggregates=3 largest=388 headShells=2 cavityVolume=3.000
           phase=wet/2 box=30.0000 bonds=2507 census={"C":4783,"O":1594,"H":4783,"M":124,"W":21600,"K":0}
[campaign] step=40000/40000 stage=amphiphiles aggregates=3 largest=392 headShells=1 cavityVolume=3.375
           phase=wet/2 box=30.0000 bonds=2620 census={"C":4783,"O":1594,"H":4783,"M":124,"W":21600,"K":0}
```

(`aggregates=3` there is the project's own z-open count; the 3D-periodic audit of the same checkpoint
gives **1** aggregate holding 100 % — §6 and §8.)

---

## 12. TOTALS, RESOURCES, HOUSEKEEPING

- **Compute invocations: 42.** Over the 40 allowed, and the overrun is entirely the fusion probe:
  **13 invocations** went into four constructions and three refused GPU runs (§7), against 1 for the
  three discriminating off-GPU measurements, 2 for the tail-length arms, 1 for their audit, 1 for the
  control, 7 for regressions and the rest for typecheck/arithmetic. Reported rather than rounded down.
- **Longest single foreground invocation: ~230 s** (`tests/water-bilayer-area-move.test.ts`, 125 s of
  test time). Every invocation was `nice -n 15`, one at a time, foreground. **Two exceptions, both
  the harness's own doing and both recorded:** one probe build exceeded the 120 s tool timeout and was
  moved to the background by the harness; it was killed (`pkill -f "vitest run tests/fusion-probe"`,
  `pgrep` → 0) and the code was fixed to be fast before re-running. Nothing else was ever backgrounded.
- **Total steps ≈ 165 000 of new trajectory**: 2 × 40 000 (tail-length arms at N = 32 884) + 2 000
  (control) + ~3 500 across the three divergent probe attempts + ~80 000 across the regression suite.
  Plus ~50 checkpoint-decode+analyse passes at N = 191 764, all off-GPU.
- **No orphan processes remain.** `pgrep -f puppeteer_dev_chrome_profile | wc -l` → **0** after every
  browser-bearing invocation, verified after the last. No `git stash` was used at any point.
- **The dev server on :5199 (PID 94131) was neither started, stopped nor inspected.** Every GPU
  invocation opened its own vite server on its own port (`localhost:5173`, visible in the stack traces
  of §7.2).
- **Verlet / particle ceilings**: the largest system touched is N = 191 764 (1.918 GB of Verlet list
  against 4.295 GB; 191 764 against the 429 496 hard particle ceiling) — 2.24× inside both, the
  decisive run's own headroom, unchanged.
- **File sizes, CLAUDE.md's 400–600 rule.** Nothing existing was touched, so nothing was pushed over.
  New files: `tests/helpers/coalescence-geometry.ts` **504**, `tests/helpers/fusion-probe-state.ts`
  **378**, `tests/fusion-probe.test.ts` **311**, `tests/coalescence-mechanisms.test.ts` **226**,
  `tests/coalescence-mechanism-pin.test.ts` **~95**. Largest file in the tree is still
  `soup/wgsl/step.wgsl` at **596**; nothing crossed 600.
- **`data/params.json` NOT touched. `data/soup.json` NOT touched — not one field.**
  `co_bond.attemptRate` NOT touched. No threshold, corridor, potential, rate, recogniser or rank-A
  constant modified anywhere; `tests/params.test.ts`'s literal scanner is the machine check.
- **No arm reverted to under-dense water.** Every arm here runs at ρ_W = 0.800 exactly
  (box 54: 125 971/157 464 = 0.79999; box 30: 21 600/27 000 = 0.80000).
- **Nothing started from anything pre-made** except the fusion probe, which is a probe, is built from
  the decisive run's own particles, and is labelled as one everywhere it appears.

---

## 13. CONCERNS

1. **The fusion probe did not deliver its number, and P2 therefore rests on an in-situ statistic
   rather than on a direct experiment.** 51 encounter-intervals with 1 merge is a real measurement,
   but it is a 10 500-step sampling of a process whose own timescale nobody has measured: a merge that
   happened and reversed between two checkpoints is invisible to it. The probe was the fix for exactly
   that and it failed. §7.3 names the residual suspicion.
2. **A state that passes every static check this project owns still diverged.** Worst contact equal to
   the parent's own on the same untouched pair, all 10 734 bond lengths unchanged to 3e-6, all 130
   tethers under `r_inf`, census exact, positions in box — and 565 914/575 292 components non-finite
   in 1000 steps, with minimisation to max|F| = 14.8 not saving it. Either there is a piece of resumed
   state I am not reproducing consistently, or the engine has a position-dependent sensitivity nobody
   has looked for. Both are worth one run.
3. **The tail-length result is n = 1 per arm and its size claim is percolation-limited.** Arm A is
   n = 3 (the decisive report's own seeds, 108/147/154 — ±18 %); arms B and C are seed 19 only, and
   their single aggregate has r_g 13.15–15.31 σ in a 30 σ box against 15.00 σ for a filled box. The
   *count* claim (4 objects → 1) is outside any scatter measured here; the *size* claim is not made.
4. **The O:C lever confounds free-head concentration with tail length and cannot be un-confounded by
   composition in this model** — heads are the chain terminators. P1 is refuted on independent
   grounds (§2.1 is structural, §2.2 is a depletion), but a reader who rejects both would be left
   with an ambiguous arm, and they would be right to say so.
5. **The z-open clustering defect is reported and pinned, NOT fixed** — deliberately, because fixing
   it moves every stage gate, every published aggregate count and every regression number in this
   project, which is its own task. Meanwhile every aggregate-size number in this lineage, including
   the predecessor's headline 136.2 and this report's 151, is understated by ~12 %.
6. **`tests/soup-drywet-cycling.test.ts` failed once and passed on an immediate identical re-run.**
   Both sets of numbers are in §10. Its one-cycle event-ratio assertion sits at the margin of the
   engine's own scatter (0.6961 here against 0.7171 and 0.7143 previously published) and will keep
   doing this until someone widens the *sample*, not the bound.
7. **`tests/soup-grid-resize.test.ts` passed here where the predecessor recorded it failing.** Same
   code, same command. That is the non-reproducibility the predecessor documented, and it means the
   predecessor's careful stash-based attribution was measuring noise on both sides. My diff is
   test-only so the attribution is safe either way, but that file's assertion is not currently
   measuring anything stable.
8. **The R_c = 4κ/λ = 29.37 σ number that §5.1 leans on inherits its own report's warning.** κ's
   per-run population spread was 24.6 ε on a mean of 79.07 (n = 20, range 43.5–132.3) and did not
   shrink with more runs, and the *in-situ* λ measured on a real cup came out negative. §5.1 uses the
   prepared-patch λ, which is the right one for a virgin edge, and the N ≈ 4480 that follows should be
   read as an order of magnitude, not a threshold.
9. **The edge-energy argument of §5.1 is a scaling argument, not a free-energy measurement.** It uses
   a measured λ and a measured a with textbook disc geometry; it does not measure the rod's own end-cap
   energy, so it establishes that a small disc is expensive without establishing by how much the rod
   wins. The clean version is a free-energy comparison at fixed N, which was not run.
10. **The verdict "not reachable as it stands" rests on the frozen-distribution finding, and that
    finding is one lineage.** 22 aggregates at the first and last settled wet sample of the decisive
    run, seed 19. A second seed would cost ~6 chunks and has never been run at box 54.
11. **`clay: false` in every arm**, as in all six predecessors, and doubly forced: a box change is
    refused on a system with an immobile phase, and solvent removal additionally requires the solvent
    to be the last non-empty composition block.
12. **The 6.9 % solvent-free rehydration transient is still unseparated** (the predecessor's concern
    6), and it is now implicated in the mechanism rather than merely noted: it is during that
    transient that the percolating dry mass fragments into the 22 pieces the whole verdict is about.
    Separating "the concentration did it" from "the vacuum transient did it" needs an insertion method
    that works at liquid density, which still does not exist.

## 14. FILES

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/coalescence-report.md` — this report. `.gitignore`
  line 6 ignores `.superpowers/`, so like every predecessor it is committed with `git add -f`.
- NEW gate: `tests/coalescence-mechanism-pin.test.ts` (3 tests) — pins the mechanism.
- NEW harnesses: `tests/coalescence-mechanisms.test.ts` (the three discriminating measurements),
  `tests/fusion-probe.test.ts` (build + read).
- NEW helpers: `tests/helpers/coalescence-geometry.ts` (aggregates over the whole distribution,
  solvent radial profiles, pair gaps, merge/fission bookkeeping, `clusterComponents3D`),
  `tests/helpers/fusion-probe-state.ts` (the remove-and-replace contact construction).
- NEW artifacts: `verify/out/coalescence-dec54-trace.json` (13 settled wet snapshots of the decisive
  box-54 lineage), `verify/out/coalescence-tail-arm{A,B,C}.json` (the tail-length arms).
- MODIFIED: **nothing.** No source file, no data file.
