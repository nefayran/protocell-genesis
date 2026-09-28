# Soup to vesicle: what was tested, what was measured, how it ended

Final summary of the **protocell-genesis** project, branch `stage-a-atoms`, state as of **2026-08-21**
(task `confined-parcel` is the last one in the project). Written for someone who was not here: reading it
should be enough to understand the result, its boundaries, and what to do next. All numbers are measured,
each with a reference to the artifact or report where it can be rechecked. The full protocols, 51 reports, live in
`.superpowers/sdd/2026-08-16-soup-to-vesicle/` (a directory in `.gitignore`, committed via `git add -f`);
read the last five in order: `long-range-electrostatics-report.md` → `supply-window-report.md` →
`big-box-report.md` → `closeout-report.md` → **`confined-parcel-report.md`**.

The machine-readable version of the gate table is `verify/out/gates.json`, the human-readable one is
`verify/out/report.html`. Both are rebuilt by the `npm run verify` command and are **never edited by
hand**.

> **What changed in this document on 2026-08-21.** The previous revision was written BEFORE electrostatics
> and claimed "there is no electrostatics at all" (its item 11), which is no longer true: charge, Debye
> screening and a protonation equilibrium are in the model, and that is exactly where the project's only
> **independent** validation lives (a salt-dependent apparent pKa shift of **+0.709** against a literature
> value of ≈0.7). Added: the mechanism of the failure in its current form (the spanning cylindrical micelle
> and the L\* threshold), the lifting of the memory ceiling, two new defects, a sixth class of silent
> failures, and an answer to the "cavity or lumen" question.
>
> **What changed LATER the same day (task `confined-parcel`).** Item 3 of the §9 list, "the aperiodic
> region is the root cause", has been **tested and REFUTED as the root cause**: the system was run in a
> finite parcel of water with a soft neutral wall, the number of wrapping axes went to **0 of 3 instead of
> 3 of 3** (the `aggregate-percolation` gate **passed for the first time in the project**), and closure is
> still absent: `encapsulatedWater` is EXACTLY 0 on 19 of 19 snapshots against a threshold of
> 312.67-314.87. The periodic boundary was a CONSEQUENCE, not the cause. A new §11 covers the whole
> experiment; §5.2 and §9.3 are marked with what has now been lifted from them. The published gate table
> remains the PERIODIC one (its corridors were set in the periodic box), and the confined arm is published
> as a SEPARATE line in `verify/out/gates-confined.json`; both results matter.

---

## 1. What was tested

One question: **will a self-assembled CLOSED vesicle form out of a coarse-grained "soup" of monomers**,
that is, an object that has an interior inaccessible from outside and a bilayer wall around it.

Project rules that determined what counts as an answer:

- **A gate is a statement with a model number, a literature interval, a verdict, and an evidence rank**
  (spec `docs/superpowers/specs/2026-08-15-protocell-genesis-design.md`, section 8). Gates are published
  as **passed / failed / unproven** and **are never widened** to make them pass.
- **Evidence ranks** (spec, section 3): **A**, measured (reference, conditions, error bar);
  **B**, computed; **C**, bounded by thermodynamics / ratios taken from an independent source;
  **D**, an estimate, and **a gate resting only on D comes out unproven by definition**.
- `engine/` must contain not a single hand-entered chemical constant: the engine reads `data/`.
- Hydrophobicity must be **emergent** (from water exclusion), not set by direct tail-tail attraction.
- `co_bond.attemptRate` is not tuned. Over the project's lifetime it was requested nine times and refused
  nine times.

**In one honest sentence, what was built and how it ended.** A working coarse-grained WebGPU engine was
built, with emergent hydrophobicity, catalyst-grown chain chemistry, solvent evaporation, explicit liquid
water, charge with a protonation equilibrium, and a full set of instruments for detecting closure, and it
**consistently assembles not a vesicle but a single percolating network** that spans the periodic box along
all three axes at every measurement of every campaign. The project answers its own question in the
negative and, more importantly, names the measured cause.

---

## 2. What was built

| layer | what it is | where |
|---|---|---|
| "soup" | GPU bond chemistry: C-C chain growth on a catalyst, head termination (C-O), valence, dehydration/rehydration, solvent evaporation, a clay plate | `soup/` |
| explicit water | water as a separate bead species; attraction by CLASS (solvent/polar/apolar), depth ratios taken from `martini_v2.1.itp` | `soup/wgsl/step.wgsl`, `data/soup.json` |
| **electrostatics** | screened Coulomb (Debye-Huckel) between heads; charge is a **dynamic variable**, chosen by MC at fixed pH; its own cutoff radius in units of its own screening length (4 λ_D) and its own neighbor list | `soup/src/electrostatics.ts`, `soup/wgsl/es-*.wgsl`, `data/soup.json` §`electrostatics` |
| measurements | amphiphile recognition, per-component aggregate breakdown, shapes (inertia tensor), head layers, cavity, encapsulated water, box wrapping | `soup/src/aggregates.ts`, `soup/src/water-closure.ts`, `tests/percolation-check.test.ts` |
| gates | `data/literature.json` (definitions) → `verify/gates.ts` (verdict rules) → `verify/run.ts` (run) → `verify/out/gates.json` + `report.html` | `verify/` |
| viewer | run page: start/pause/stop, live measurement, stage ladder, aggregate breakdown, cavity highlighting, honesty notes | `viewer/run.html` + `viewer/run-*.ts` |
| long runs | CLI with checkpoints and resume: a campaign proceeds as dozens of separate calls, each under 500 s | `soup/cli/campaign.ts` |

Atomistic detail in a frame is a **reconstruction** from reference geometry laid over the coarse-grained
coordinates, not full-atom dynamics. This is stated on the page itself.

---

## 3. What was measured: the published gates

Regenerated by `npm run verify` at **2026-08-20T23:23:30Z** (task `closeout`). Rank and verdict are from
`verify/out/gates.json`; the `provenance` field of each row names the artifact and the time it was
measured. **The published campaign block is box-76 run (`bbB76`), 148,200 steps** (see §6): a reader
comparing these rows against numbers from older reports will find a different box and 2.9 times more feed.

| gate | value | corridor | rank | verdict |
|---|---|---|---|---|
| area per lipid (solvent-free) | **1.2005 σ²** | 1.1-1.5 | A | **passed** (inside) |
| bilayer thickness (solvent-free) | **4.4308 σ** | 4-6 | A | **passed** (inside) |
| bending modulus κ | - | 5-50 kT | A | **unproven** (fit window invalid) |
| area per lipid **IN EXPLICIT WATER** | **1.1607 σ²** [1.1519, 1.1674] | 1.1-1.5 | C | **passed** (inside) |
| bilayer thickness **IN EXPLICIT WATER** | **4.6871 σ** | 4-6 | C | **passed** (inside) |
| **closure: encapsulated water / threshold** | **0** (0 beads against 315.409) | ≥ 1 | B | **FAILED** (outside) |
| **largest aggregate is finite** | **3 of 3 axes** (27/27/27 of 27 shells) | ≤ 0 | B | **FAILED** (outside) |
| **RESULT: self-assembled closed vesicle** | **0 aggregates** | ≥ 1 | A | **FAILED** (outside) |
| agreement with ASF (spec gate 2) | discrepancy **0.1594** | - | D | unproven (no corridor) |
| mean beads per tail | **3.419** | 2-3 | D | unproven (outside) |
| detector cavity (synthetic shell) | 1284.875 σ³ | ≥ 370.8656 | D | unproven (inside) |
| carbon-atom-to-bead mapping | 3 | - | D | unproven |

`mean-tail-length` = **3.419** is the final row of the `bbB76` campaign for this task, and it is also
proof that the gates read exactly that trace and not an old one;
`aggregate-percolation` = 3, collected from **23** rows with `role:"campaign"`.

### 3.1 A bilayer in liquid water: a real result, and how it was obtained

A bilayer **in liquid explicit water at 0.8 σ⁻³** holds both area and thickness: **1.1607 σ² / 4.6871 σ**,
a dry core (33 water beads out of 4500 inside), a buried-head fraction of **0.0656**, a cluster fraction of
1.0000, and settled area (ln A drift of 3.70e-5 per sample, t = 1.40 over 300 samples). Area here is
**measured** via an area-move step at zero tension, not fixed by construction.

This matters because before the task `hydrophobic-asymmetry` (2026-08-19) the same patch in water gave
**12.2-12.4 σ** against a 4-6 corridor, and area was an assumption of 1.3 σ²
(`broth-composition-report.md` §4). The task `water-calibration` found the failure mechanism: not water
penetrating the core (the core was dry) and not the absence of a core, but **buried heads**: 40-49% of head
beads sat inside the hydrophobic zone, because removing the old "tail-tail" term left the leaflet with no
lateral cohesion at all. The fix is not tuning but **pair depth ratios from MARTINI 2.1**
(`martini_v2.1.itp`, `[nonbond_params]`, ε = C6²/(4·C12)): water-water 5.0, water-head 4.0,
tail-tail 3.5 (the reference, multiplier exactly 1.0), water-tail 2.0 kJ/mol. The exchange
(Flory-Huggins) energy is then positive: 5.0 + 3.5 - 2·2.0 = **+4.5 kJ/mol**, meaning mixing is still
unfavorable and hydrophobicity remains emergent. The absolute depth was not refit: tail-tail matches
bit-for-bit the rank-A ε from `data/params.json`.

Rank **C**, not A: the corridor is literature-derived (rank A), but the number was obtained in an
environment whose depth ratios were taken from someone else's force field and are not calibrated on this
tree, and `epsilonScale = 1.0` carries rank D.

### 3.2 Chain-length statistics (Flory / Anderson-Schulz-Flory)

On the final snapshot of the `bbB76` campaign: event-based **α_ev = cc_bond/(cc_bond+co_bond)**, the
ASF mean `1/(1-α_ev)` against a measured mean tail length of **3.419**, a discrepancy of **0.1594**. An
independent estimate of the same α from the length histogram itself converges with the event-based one to
within a few percent, and the shape of the distribution is geometric, which is the signature of
Anderson-Schulz-Flory
(van der Laan & Beenackers, *Catal. Rev. Sci. Eng.* **41** (1999) 255-318).

The gate is still **unproven**, and the reason is stated directly: the spec defines the criterion as
"histogram discrepancy" without giving a numerical corridor, and no independent literature corridor for α
exists for this specific system. Inventing a corridor here would mean fitting the gate to the result.

The tail length **3.419** sits ABOVE the 2-3 window implied by our own mapping
"C8 → 2, C12 → 3, C16 → 4" (spec, section 4). The mapping carries rank D and is only accepted through the
area and thickness gates, so this gate comes out unproven rather than failed. An honest caveat: if the
mapping had an independent corridor, this would be a **failure**. And tail length here is not a single
number but a **trend within one run at constant composition**: 2.158 → 2.911 → 3.207
→ **3.419** over the course of `bbB76`. The molecule keeps growing after the structure has already
plateaued.

### 3.3 Evaporation: a concentration factor, not thermodynamics

Evaporation here is **solvent removal**, not a phase transition. The concentration factor that the model
actually achieves is **3.20×** against **1400×** for a literature dry-wet cycle (16.07% of it on a log
scale), and the carbon pool is on top of that enriched **~691×** against the most generous literature pool
(~15 mM decanoic acid, ACS Earth Space Chem. 2023, PMC9869395). The dry-wet cycle does **actually work**,
and this is measured: yield up 1.68-1.80×, bonds up 1.76-1.93×, the largest aggregate up 8.1-8.4×
(these numbers are AFTER fixing the defect in §7.3; before the fix, inflated numbers of 12.5-14.5× were
published).

### 3.4 Electrostatics: what was entirely absent from the previous revision of this document

Head charge, screened Coulomb (Debye-Huckel), and a **protonation equilibrium**: the charge of each head is
a dynamic variable, chosen by discrete MC at fixed pH from
ΔG = kT·ln10·(pKa_intr − pH) + ΔU_es. Ionic strength and pH are real parameters,
λ_D = 0.304 nm/√(I[M]). `data/soup.json`'s `saltPhLimitation.represented` was flipped **false → true**,
and this is the only field that changed value.

**The force was checked against a numerical gradient twice.** At the pair level, the worst relative
residual is **5.145e-9** over r ∈ [0.6, 2.7] σ; over the whole field, GPU force against the CPU −∇ of the
potential is **1.736e-5** at mean|F| = 0.0470, against a control with charge disabled of **1.303e-3** at
mean|F| = 81.31 on the same tolerance.

**The protonation sampling was checked against something independent.** At ΔU_es = 0, the measured α
reproduces the Henderson-Hasselbalch relation to within **≤ 2.08e-3** at five pH values (each deviation
within 4 binomial standard errors). With ΔU_es included, the four occupancies of a two-head system match
the **exact enumerated Boltzmann weights** to within **3.140e-3** over 200,000 sweeps, and the interaction
is indeed active (the "both charged" state is suppressed at 0.21049 against 0.25000).

**And here is the project's only independent validation.** The apparent-pKa shift between 10 and 100 mM
salt, literature ≈ **0.7**:

| attempt | charge cutoff radius | measured salt-dependent pKa shift | fraction of interaction dropped |
|---|---|---|---|
| first (`electrostatics`) | shared with LJ, 2.7224620 σ | **+0.124** (5.6× too small) | 0.8385 at 10 mM against 0.3389 at 100 mM: **asymmetric** |
| second (`long-range-electrostatics`) | **own, 4 λ_D = 15.2000 σ** | **+0.709** | **0.0916 on BOTH sides** |

That is, the first attempt was wrong not in the physics but in the range: a cutoff expressed in units of
its own screening length makes the dropped fraction exp(−x)(1+x) the same on each side, and the shift
naturally lands on the literature value. No constant was tuned for this.
**The apparent pKa itself is also emergent**: the model is given only a monomer pKa of 4.9, and the
interfacial shift measured is **+0.08 / +0.32 / +0.67 / +0.97** at pH 4/5/6/7, monotonic in α.

**An acid-soap pair has appeared**, as a correlation of alternating charge, not as a dimer: the paired
fraction is 0.058-0.080, the ratio of excess unlike contacts is 1.23-1.37 against a random null of zero.
Hydrogen bonding was NOT added in the process (`polarPolar` remains 0). The pair survival fraction grows
monotonically **0.033 → 0.244**: pairs become long-lived, and this is not a closure signal.

**What electrostatics did NOT do: it did not break the network. At any pH from 4 to 9.** The wrapping
instrument reports 3 of 3 axes on all six arms of box 30 (a neutral control and five charged arms, α from
0.110 to 1.000, indistinguishable), on the box-54 campaign, and on the box-76 campaign. Meanwhile
head-head repulsion was brought up to **0.7215 kT** against 0.909 kT of an apolar contact: **only 1.26×
weaker** (and at the strong end of the σ→nm range it actually EXCEEDS the apolar contact), over 16.2% of
beads. This is the refutation of the energetic explanation: **the energy gap closed from 2.86× to 1.26×,
and the number of wrapping axes moved by EXACTLY ZERO.**

---

## 4. Verdict

**THERE IS NO CLOSED VESICLE. Not on a single snapshot of any campaign across the entire project.**
`hasVesicleAggregate` = false, `closed` = false, `encapsulatedWater` = 0, everywhere, always.

The last and largest campaign (`bbB76`, tasks `big-box` + `closeout`): box **76 σ**,
**N = 483,268**, liquid water ρ_W = 351181/438976 = **0.8000005**, ρ_org = 0.3008980, composition
O:C = **0.333265**, seed 19, kT 1.1, charge on at pH 7.0 / I = 0.01 M, one drying event,
**148,200 steps**, of which **111,800 were settled wet steps**. Start: a lattice of monomers alone at its
own step 0 (270 aggregates, largest of 4 amphiphiles), nothing pre-built.

| quantity | measured | needed | verdict |
|---|---|---|---|
| amphiphile feed | **4519.5 ± 5.2** (0.11% over the last four) | ~930-955 | **4.73× OVER the threshold: cleared** |
| largest-aggregate size | the same object, 99.93% of feed | ~930-955 | **cleared** |
| rim energy per molecule | 0.583 kT at N = 2087 (measured on the box-54 campaign) | ≲ 1 kT | **cleared** |
| **connectivity: wrapping axes** | **3 of 3**, shells 27/27, on **23 of 23** wet snapshots | **0** | **BINDING CONSTRAINT** |
| encapsulated water | **0** on all 23 | ≥ 315.4 beads | downstream |
| radial head shells | **1-2**, oscillating | 2 (bilayer wall) | downstream |
| cavity | **139.25 σ³, STILL GROWING** | ≥ 370.8656 σ³ | 2.66× short (the most favorable reading) |
| flatness λ₁/λ₃ | 0.7571 | ≤ 0.35 | 2.16× too large |
| in-plane symmetry λ₂/λ₃ | 0.9742 | ≥ 0.50 | passed |
| monomer exchange, fissions, fusions | **0, 0, 0** on all 22 intervals | nonzero | **not tested by this model** |

**Radius of gyration 37.43 σ against 38.000 σ for a uniformly filled box of 76 σ (98.5%).** The object
fills the box just as fully as the object in box 54 filled its box.

---

## 5. The MECHANISM, as it is understood now

The previous revision of this document explained the failure this way: "at the two-tailed-head fraction
where a bilayer becomes the preferred shape, amphiphile concentration puts the system above its own
gelation threshold." That holds as a description, but **three competing explanations of the SAME fact were
subsequently tested and ruled out by measurement**, and one remains, which has arithmetic behind it.

### 5.1 What was ruled out

| explanation | how it was tested | result |
|---|---|---|
| **feed**: too much material, and it merges into one | feed reduced to 1.50× the threshold (`supply-window`), then increased to 4.82× (`big-box`) | **3 of 3 axes both times.** Ruled out |
| **rim energy**: too costly for a finite object to have an edge | 0.583 kT/molecule at N = 2087, against ≲1 kT | Ruled out |
| **pair energy**: heads repel too weakly | gap closed from 2.86× → **1.26×** (see §3.4) | **The number of wrapping axes moved by 0.** REFUTED by measurement |
| **box size**: box too small, finite object has nowhere to fit | box volume increased by **2.79×** at constant density (54 → 76 σ) | **3 of 3 axes, 27/27 shells, on all 23 snapshots.** Refuted within the engine's reach |

### 5.2 What remained: a spanning cylindrical micelle is cheaper than a closed vesicle

The project owner's claim was this: in a fully periodic box, an object spanning the box has no
edge at all, so it beats a closed vesicle, which has to pay for curvature. Therefore a vesicle is preferred
only when there is ENOUGH material for a closed shell but NOT ENOUGH for the cheapest spanning object. The
mechanism is correct, and it turns the question into arithmetic.

**The arithmetic was tested and turned out wrong in its choice of competitor.** The cheapest edgeless
spanning object for this amphiphile is not a bilayer tube of radius ≈4 σ, but a **cylindrical micelle with
radius equal to the leaflet thickness** (t/2). This was identified not by argument but by measurement:
inverting the measured radii of gyration through the measured volume gives a 151-molecule rod a physical
radius of **2.353-2.560 σ**, against t/2 = **2.229-2.484 σ**: what the project has been measuring all
along IS a cylindrical micelle.

| object | cost, in amphiphiles | depends on L? |
|---|---|---|
| closed vesicle of minimum volume (threshold 370.8656 σ³) | **930-1030** | **NO**, box-independent |
| cylindrical micelle spanning the box | **(5.806-6.635)·L** | yes, linearly |

The band [closure threshold, spanning cost] is **EMPTY and INVERTED** at any box this engine can compute:
at L = 54 the spanning object costs 314-358 against a threshold of 932-1030; at L = 76 it costs
440-450 against 930-955 (an inversion of 2.07-2.17×). The band only opens at

**L\* = 160.6-161.3 σ** (from the `big-box` task's own gates; an independent pin,
`tests/supply-window.test.ts`, gives [157.4, 164.4] σ and brackets this value),

that is, at **3.29e6 particles**. Below L\*, the spanning micelle is the ground state, and no amount of
material, no repulsion energy, and no box size within the reachable range changes that. This is the
mechanism.

> **SUBSEQUENT TEST (task `confined-parcel`, §11).** This section's conclusion, "one constraint remains
> that nothing can move: the topology of the periodic box below L\*", **was tested by direct experiment and
> did not hold up as the CAUSE**. The topology was removed (a finite parcel, wrapping axes 0 of 3 on 20 of
> 20 snapshots), and closure still did not occur. The arithmetic of this section is correct and remains
> valid for the periodic box; what was wrong was the CONCLUSION that removing the spanning competitor is
> sufficient. The reason is that in a finite region the cheapest edgeless object is not a spanning micelle
> but a **cylindrical micelle WITH END CAPS**, costing 17.5-21.3 amphiphiles, with no lower bound tied to
> region size: the band [closure floor 937-1014, cheapest edgeless object 17.5-21.3] is inverted by
> **44-58×** even inside the parcel. The periodic box was not providing edgelessness as such but
> edgelessness AT ANY SIZE, and a capped micelle is edgeless at any size, including inside the parcel.

### 5.3 Why L\* is unreachable, and why that is structural, not "need a bigger machine"

The particle ceiling was **one unmeasured number**: `verletList.listCapacity = 2500` in
`data/soup.json` was not a measurement, it was "with headroom." A measurement (eight configurations,
`soup/cli/measure-neighbours.ts`) gives a maximum neighbor count of **866**, and the binding configuration
is not the dry phase (749, dense but uniform) but a **settled percolating aggregate** (866, local density
1.94× the box average). Capacity is now derived from the density the system actually reaches:

| ceiling | basis | particles | box at ρ_W = 0.8 |
|---|---|---|---|
| 2500 | old guess | 429,496 | 81.27 σ |
| **1126** | derived, floor-binding | **953,589** | **106.03 σ** |
| 866 | **zero headroom**, the largest measured value at all | 1,239,886 | 115.73 σ |

**And even at zero headroom the box tops out at 115.73 σ, 1.39× short of L\*.** The reason is structural:
a flat per-particle list costs `N · capacity · 4 = (ρ_wet L³)·(f·(4π/3)R³·ρ_dry)·4`, meaning **L³ times the
SQUARE of density**: the capacity that physics demands grows together with the number of particles the box
demands, and one cannot be traded for the other.

**An honest alternative without a per-particle array** (cell traversal, O(N) memory, ceiling
~2.7e8 particles) was measured on the same settled state and **rejected by measurement: 122.28 ms/step
against 17.47**: 7.00× slower. Sizing the list correctly gave a 1.34× speedup on its own (23.47 → 17.47
ms/step).

**After lifting the memory ceiling, TIME became the binding constraint.** Measured ms/step ∝ N^1.35;
at L\* this is ≈1013 ms/step, and a campaign of the predecessor's shape would take **41 hours = 296 calls
of 500 s**. Overall position at the end of the project: **memory is short by 3.45×, time by roughly 15×.**
The strong hypothesis is untestable in this engine, and the reason has been named, not swept aside.

### 5.4 A sequence of shortfalls: how each stage ended

Published shortfalls of the largest aggregate's size from the closure threshold, in order of appearance:

**80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75× → 0.48× → 0.42× → 0.67× → 0.21× → 0.45×**

- **80× → 8.9×**: early soups; "chemistry does not run at all" (nucleation, §7.1) and feed were ruled out.
- **3.36×**: **this number is wrong**: under the zero-RNG-seed fix (§7.2) it is 1.85×.
- **33.4×**: an arm where composition drifted into finite rods; the "only volume is needed" hypothesis was ruled out.
- **6.76× → 6.75×**: the decisive run; under the z-axis double-counting fix (§7.5), **6.02×**.
- **0.48×**: the two-tailed box-54 campaign (2.08× OVER the threshold): **for the first time in the
  project, aggregate SIZE was ruled out as the binding constraint**, and percolation was exposed.
- **0.42×**: the charge campaign (2.36× over): "there is no charge in the model" was ruled out, and
  refutation of the energetic explanation began.
- **0.67×**: `supply-window` (1.50× over, feed DELIBERATELY reduced): **the explanation via material
  budget was ruled out**: feed was beaten at both 2.36× and 1.50×, wrapping axes at 3 both times.
- **0.21×**: `big-box` + `closeout` (4.73-4.82× over, box 76): **the explanation via box size was ruled
  out** within the reachable range.
- **0.45×**: `confined-parcel` (2.06-2.23× over, finite parcel R = 36 σ with a soft neutral wall in a box
  of 160 σ): **the TOPOLOGY of the periodic box was ruled out**: wrapping axes 3 of 3 → **0 of
  3** on 20 of 20 snapshots, the `aggregate-percolation` gate **passed for the first time in the
  project**, and closure still did not occur (`encapsulatedWater` exactly 0 on 19 of 19 against a threshold
  of 312.67-314.87). Feed was deliberately kept at the same multiplier as arms 0.42× and 0.48×, so that
  exactly one axis moved.

The last six stages lie PAST the end of the sequence (the threshold is exceeded, not unreached). That is,
the project sequentially ruled out every constraint it was able to move, including the last one, the
topology of the periodic box (`confined-parcel`, §11). After that, no explanation via BOUNDARY CONDITIONS
or MATERIAL BUDGET remains. Exactly one survived, and it concerns the molecule, not the box:
**this amphiphile's packing parameter a/t** (§9, item 1).

---

## 6. Cavity or lumen: the answer from task `closeout`

The `big-box` campaign left a cavity at **97.16 ± 7.54 σ³** against a threshold of 370.8656 σ³ and itself
warned that this was **not a plateau but a lower bound**: the cavity grew on each of the last four
snapshots. The task `closeout` resumed the same campaign from its checkpoints (**the guarantee is the same
ENSEMBLE, not the same trajectory**: chunk boundaries cut Langevin noise at different points) and added
**13 calls, 61,700 steps**: 86,500 → **148,200**, settled wet steps 50,100 → **111,800 = 2.23× more**.

**Cavity by checkpoint (σ³):** 48.1 → 64.6 → 79.1 → 74.9 → 80.0 → 91.9 → 87.6 → 94.9 → 101.5 →
104.6 → **101.3 → 109.3 → 111.4 → 111.9 → 111.8 → 121.5 → 121.8 → 118.5 → 128.8 → 125.3 → 136.0 →
130.1 → 139.3** (bold: 13 new ones).

**A plateau has NOT been reached, and this is a quantitative statement.** This project's plateau standard
is 0.25% over tens of thousands of steps (that is how size was measured). Cavity:

| window | mean | sd | slope |
|---|---|---|---|
| all 23 settled wet steps (41,100-148,200) | 104.08 | 23.62 (**22.7%**) | **+0.695 σ³/1000 steps** |
| only the 13 new ones (91,500-148,200) | 120.51 | 11.23 (**9.3%**) | **+0.586 σ³/1000 steps** |
| last 4 (129,400-148,200) | 132.66 | 6.22 (**4.7%**) | +0.769 σ³/1000 steps |

Doubling the settled-run length **did not change the slope**: 0.586 against 0.695, the same straight line,
not saturation. For comparison, over the same window **size settled to a plateau harder than ever:
4519.5 ± 5.2 = 0.11%**, and flatness 0.7541 ± 0.0059 = 0.78%. That is, the structure stands still while the
cavity keeps going. At the observed slope, reaching the threshold of 370.8656 would need another
**≈333,000 steps ≈ 71 calls ≈ 8.5 hours**, and this is a linear extrapolation of a quantity with no reason
to be linear.

**And now the reason the question was even asked.** While the cavity grew **2.90×** (48.1 → 139.3),
encapsulated water stayed **EXACTLY ZERO on all 23 wet snapshots**, `closed` = false on all
23, head shells 1-2 (oscillating: 2 → 1 → 2, no trend), wrapping axes **3 of 3 on all 23**.

**ANSWER: a growing cavity inside a percolating object does NOT become a lumen.** These are two
unrelated quantities, and here they diverge by measurement, not by argument: a 2.9× growth in cavity did
not move closure by a single bead. `cavityVolume` is a fill of interstitial voids within the object's own
bounding region; for a network the interstitial void **is not the interior of a protovesicle**. So citing
"139.25 against 370.8656, 2.66× short" as the closure shortfall is **generous** to the model, not strict.
The object is a **network with cavities, not a closed compartment**, and after doubling the run length
this statement now rests on 23 measurements instead of 11.

**And the context without which the number cannot be presented as a headline.** A box of 76 σ is **0.47
of** L\* = 160.6-161.3 σ, below the threshold where closure becomes thermodynamically preferred. If
something had closed here, it would need exactly that explanation, not a headline. Nothing closed.

---

## 7. Actual defects found along the way, and which published numbers they shifted

This is not cosmetic: each of these made a published number wrong, and each has been fixed with a pinning
test that fails if the defect returns.

### 7.1 Chain-end selection during nucleation: `soup/wgsl/bond-adsorption.wgsl`

Chain nucleation on the catalyst required a **closed triangle**: contact between bare carbon `i` and `j`,
catalyst contact with `i`, AND catalyst contact with `j`. But by that point `bondFormWalk` had already
established a catalyst contact with `i` specifically, while the nucleation branch declared `j` the endpoint
and required a contact with a DIFFERENT carbon. The diagnosis came not from reasoning but from a counter
(`verify/center-nucleation.ts`, pure CPU): the number of eligible triples dropped **393 → 3 → 0** by step
120,000 and stayed at zero through step 300,000, with 5125 free bare carbons and 1021 free catalysts.
Chemistry was not stalling for lack of raw material, it was stalling because the geometric starting
condition was unsatisfiable.
**Shifted numbers** (box 30, seed 20): nucleations **117 → 192**, amphiphiles **56 → 129**,
unreacted bare carbon **133 → 33 of 779**. Report:
`catalyst-turnover-and-window-report.md`.

### 7.2 Checkpoints with a zero RNG: `soup/src/checkpoint.ts` / `readBack`

The `bondRngBuf` / `thermoRngBuf` buffers lacked the `COPY_SRC` flag, so read-back copies were a validation
error, command submission was a no-op, and a **zero-initialized staging buffer** was returned instead. All
**72 checkpoints on disk** contained 100% zero RNG bytes: every resumed run in the project's history got
seed 0 for ALL N particles, meaning identical Langevin noise on every particle. The system consequently
translated rigidly as a whole (measured MSD 62.32 σ², the same for all seven species classes), and
**relative** diffusion was destroyed, which is why aggregates could not merge.
**Shifted numbers** (box 54, same composition/seed/steps, one parameter changed): largest aggregate at
step 90,000 **268 → 588** (2.19×), and instead of a plateau of "271, frozen from step 135,000 to
315,000" the result was growth of 232 → 281 → 333 → 588; the enclosed volume of the largest aggregate
**6.25-6.75 → 17.125 σ³** (2.74×). The published sequence's size shortfall turned out to be not
"3.36×" but **1.85×**. `readBack` now THROWS on a buffer without `COPY_SRC`. Reports:
`loud-failure-and-liquid-water-report.md`, `evaporation-report.md` §7.

### 7.3 Stale F(x) after a box-size change: `applyBoxScaleOnce`

A box change (drying/rehydration) rebuilt the neighbor grid and the Verlet list, and stopped there, so
`kick_drift_wrap` used as F(xₙ) a force computed BEFORE the coordinate rescaling. Pinned in both directions
(`tests/soup-stale-force.test.ts`): with the fix, the resident buffer agrees with a fresh calculation to
within 1.5e-4 at a force scale of 513; without the fix it is bit-for-bit equal to the old force and is
**off by 313.7**.
**Shifted numbers** (box 30, step 36,800, n = 3 on each side): largest aggregate
**249.3 [210, 273] → 109.3 [71, 154]**, a **2.28×** contraction with non-overlapping ranges, while
yield (0.1516 → 0.1525) and bond count (1436 → 1392) did not move. The honest effect of cycling is
a 1.68-1.80× yield increase, a 1.76-1.93× bond increase, an 8.1-8.4× largest-aggregate increase, not
the **published 12.5-14.5×**. Incidentally, the ramp guard that fired 9 times out of 216 was a symptom of
this same defect, not physics: after the fix, **0 of 36**. Report:
`decisive-run-report.md` §2.3.

### 7.4 Silent divergence: `soup/wgsl/health.wgsl`

The `assertVerletSafety` guard compared `sqrt(maxDriftSq) > skin/2`, and **any** comparison against NaN is
false: as soon as positions became NaN, the drift read as NaN and the check passed **forever**. The guard
was running and was blind by construction. An O(N) scan of IEEE-754 exponents was added on the same tick,
**before** the old check. The cost was measured: **0.212 ms against 331.68 ms per chunk of 1000 steps =
0.064%**. On the composition that had been diverging silently (ρ_tot 0.75, box 30), it now THROWS at step
1000, naming **6957 non-finite components out of 60,750**. Report:
`loud-failure-and-liquid-water-report.md`.

### 7.5 Double counting across the z boundary: `engine/src/aggregate.ts`

`buildClusterUnionFind`, through which **every stage definition in this project** passes, treated the z
axis as open (correct for the membrane engine: a bilayer patch in vacuum indeed has no image through z),
but the "soup" wraps all three axes every step. Any aggregate crossing the z boundary was cut in two and
counted twice. The fix is a `periodicZ` flag defaulting to `false`, so the membrane engine is bit-for-bit
unchanged, and `true` in every "soup" call.
**Shifted numbers** (remeasured on checkpoints already on disk, by the same auditor):

| published number | was | became | shift |
|---|---|---|---|
| decisive run's largest aggregate, final | 135 | **151** | +11.9% |
| same, first settled wet snapshot | 153 | **200** | +30.7% |
| headline number of the binding constraint | 136.2 | **152.8** | +12.2% |
| aggregate-size shortfall | 6.75× | **6.02×** | -10.8% |
| largest aggregate, cycled arms of box 30 | 109.3 | **136.3** | +24.7% |
| number of aggregates, decisive run, final | 34 | **25** | -26.5% (**had been inflated**) |
| radial head shells of the largest aggregate | 2 | **1** | a published "passed" became a failure |

Two consequences deserve close reading. **Aggregate counts had been INFLATED** (the cut manufactured extra
objects), while sizes were understated. And **`radialHeadShells` = 2 was an artifact**: the decisive report
read two head layers on 4 of the last 5 snapshots and recorded the row as passing; under the fix it is
**1**, meaning there was no bilayer wall at all. Report: `final-campaign-report.md` §1-2.

### 7.6 Percolation gates recognized their own input by run LABEL: `verify/campaign-gates.ts`

The campaign label (`zfB54`) was hardcoded, and the very first campaign with a different label silently
flipped the gate to `unproven`. Fixed: rows carry their own `role` field, with the substring left as a
fallback. **The class matters more than the instance: a gate pipeline that recognizes its inputs by run
LABEL goes silent exactly when someone runs a new experiment**, that is, exactly when it is needed most.
The cost that time was 2 wasted calls and one incorrect published table.

### 7.7 Page layout silently swallowed the START click: `viewer/run.html`

The first headless run of the page after it grew from one row to three **never started**:
`POLL 0…39 state=idle steps=0 err=-`, no error, no exception. Asking directly, though, revealed:
`CLICKTARGET [{"id":"start-btn","hitId":"visibility-note"}, …]`, the bottom note, stretched full width,
sat exactly on top of the buttons at a headless window of 800×600. The page's own CSS comment records that
this failure class had occurred **twice before** and had been bought off twice with one more line of
padding. Fixed at the class level: the three bottom notes moved to `left: 288px`, entirely clear of the
260 px control column, and stacked into a single flex column instead of three hand-maintained offsets.

### 7.8 SIX classes of silent failure: worth reading even if everything else is forgotten

Each looked like success. This is the project's main technical lesson.

1. **`readBack` on a buffer without `COPY_SRC`** returns a zero-initialized staging buffer (72
   checkpoints with a zero RNG, §7.2). Now throws.
2. **A comparison against NaN is always false**, so the divergence guard passed forever (§7.4). Now, an
   explicit exponent scan runs before the check.
3. **`requestDevice()` without `requiredLimits`** silently caps every pipeline to default storage limits
   (the device supports up to 10, the default is lower). Now requested explicitly, and it throws if not
   granted.
4. **Stale F(x)** after a box change: not zeros, but a quietly wrong number, an error of 313.7 at a force
   scale of 513 (§7.3).
5. **A gate recognizing its input by run label** goes silent on a new experiment (§7.6).
6. **NEW, the nastiest one. A device-limit failure returns zeros at 0.04 ms/step and looks like a
   brilliant success.** A buffer request for 5,484,474,528 bytes at box 85 **raised not a single JS
   error**: WebGPU reports the `createBuffer` validation failure as a console warning, every subsequent
   dispatch against that bind group becomes a no-op, and the run returned
   `max|F| = 0.0000e+0`, `nonFinite = 0`, `neighbors max = 0`, **0.0405 ms/step**: a 500× speedup and a
   perfectly clean health report. `createSoup` now checks every N-scaled buffer against the device's own
   limits and names the byte count that would have been refused.

**A seventh, found already in the `closeout` task and recorded here because it is the same class.**
The "page is drawing" instrument, written as a WebGPU-canvas read via `drawImage` into a 2d context,
returned **an angle of [0,0,0] and 0 differing pixels out of 30,000** on a page that at that moment was
demonstrably drawing (119 frames, 661 drawn instances). The instrument was wrong, not the page. It was
replaced with the scene's own counters (`sceneDebug.instanceCounts()`) plus a real
`page.screenshot()`, which is what `tests/run-ui.test.ts` now measures. The moral is exactly the same:
**zeros from an instrument are, first and foremost, a reading about the instrument.**

---

## 8. What remains unproven, disputed, or measured from a single seed

1. **Every campaign is a SINGLE seed** (19). The plateau is internally consistent (0.11% at 4519.5 over
   111,800 steps), and the main conclusion, wrapping along all 3 of 3 axes, is a topological binary rather
   than a number near a threshold. But "4519.5" as a number is one trajectory, and the engine is not
   bit-reproducible.
2. **The cavity did NOT settle**, the one quantity task `closeout` set out to finish measuring and did not
   finish: slope +0.586 σ³/1000 steps after doubling the run length. The published "2.66× short" is the
   **most favorable reading, not the result**.
3. **`cavityVolume` on a percolating network is not a lumen**, and this is no longer a caveat but a
   measurement (§6).
4. **`radialHeadShells` oscillates between 1 and 2** and did so throughout the project. 2 appeared after
   rehydration and did not survive; in the new 13 snapshots, 2 occurs on 4 of 13 with no trend. There is
   no bilayer wall.
5. **The bending modulus κ was never measured** (rank A, unproven): the undulation-spectrum fit window is
   invalid, log-log slope -1.056 against a theoretical -4. Spread of κ between runs is 24.6 ε at a mean of
   79.07 and did not shrink with more runs.
6. **Monomer exchange, fission, and fusion are EXACTLY ZERO** over all 22 intervals of the campaign,
   free fraction 0 … 6.6e-4. That is, this model never tested real monomer solubility / CMC at all. It
   cannot be cited as ruled out.
7. **Two-tailedness and tail length cannot be separated by composition in this model**: the head IS the
   chain terminator. Length per tail grew 2.158 → 3.419 WITHIN a single run at fixed composition. Anyone
   who believes percolation is caused by tail length rather than by two tails is entitled to say these
   campaigns do not distinguish the two.
8. **Nothing was measured between O:C = 4 and O:C = 1.** If a compact-but-large regime exists, it lives
   in that gap.
9. **The percolation test was never run against a SYNTHETICALLY BUILT object of known topology.** Its
   defenses: it reproduces "does not percolate" on three independent systems (0 axes, 7-13 of 19 shells),
   its own internal control never wraps on a single snapshot, it agrees with two independent signals that
   were not written for this purpose, and it does not lock onto a new shell count (27 at box 76, 19 at
   54). But it has no synthetic check.
10. **`centre-untrusted` is NOT a verdict about wrapping**, and earlier reports read it wrong. It is a
    **Rayleigh test for circular uniformity on each axis** (`trusted = R ≥ √(−ln α / n)`, α = 1e-6): it
    asks whether the mass distribution is far enough from uniform for a circular mean to be a meaningful
    center, that is, HOW EVENLY the object fills the box, not whether it wraps the box. The box-76
    campaign is the cleanest demonstration: **22 of 23 wet snapshots yield a NUMBER**
    (`encapsulatedWater` 0 of 315.4) **with 3 of 3 wrapping axes**. The correction is published in
    `long-range-electrostatics-report.md`, where the error was made.
11. **One inconsistency has been left in the tree deliberately**: three `water-bilayer*` tests cluster with
    z open inside the "soup" engine, which wraps z. Its size is bounded by their own printed
    `clusterFraction` at 0.26%, and switching it would shift three published gate rows with no measurable
    benefit.
12. **Known test flakiness, proven not ours**: `tests/soup-grid-resize.test.ts` fails on unchanged code
    (proven twice via `git stash`, and in the `closeout` task by the fact that no file in
    `soup/`, `engine/`, or `data/` had changed at all); 2 of 3 `rim-lambda-insitu` tests were failing before
    us; `tests/soup-drywet-cycling.test.ts` fails and passes on identical repeats: the history of its
    event ratio is 0.7143, 0.4978, 0.7171, 0.6961, 0.5941, 0.7116, a spread of 0.43× on unchanged
    code. No bound has been widened; the fix should be a larger SAMPLE, not a looser threshold.
13. **`npx tsc --noEmit` gives 21 errors**, all of one pre-existing class,
    `ArrayBufferLike`/`SharedArrayBuffer`. The count has not changed across the last few tasks and is the
    same here.
14. **`capacitySafetyFactor` 2.6 and `capacityFloor` 1126 are rank D and a judgment about the TAIL, not the
    mean.** The risk is a composition whose local condensate density exceeds 866/368.35 = 2.351 σ⁻³. The
    consequence is a loud throw, not corrupted physics, but the throw costs a run, and someone will hit it.
15. **The throughput exponent of 1.35 was measured over only a 3.9-fold range of N** and is noisy at the
    top end (1.39 / 1.56 / 0.65 by segment). The projection of 1013 ms/step at L\* extrapolates 6.8×
    beyond the largest measured point and reads as an order-of-magnitude figure. The conclusion it
    supports ("time is short by roughly 15×, not marginally") does not depend on the third digit.
16. **The neighbor-structure decision is correct for THIS device.** The 7× loss of cell traversal is a
    number specific to apple/metal-3, driven by memory behavior in its traversal kernel; on hardware where
    traversal coalesces better, or with a parallel prefix sum instead of the serial
    `@workgroup_size(1)` in `engine/wgsl/neighbor.wgsl`, the comparison could reverse. Remeasure before
    porting.
17. **The clay plate did not appear in a single published measurement**: every arm of every campaign ran
    with `clay: false`. On the run page it is now a separate toggle with that same label, not a silent
    default.

---

## 9. What the next model needs to change

A concrete list, drawn from these numbers, not from general considerations. Ordered by expected payoff.

1. **A two-tailed amphiphile BY CONSTRUCTION (by topology), with a fixed SHORT tail.** Getting the
   Israelachvili packing parameter p = v/(a₀·l_c) into the ½-1 band by composition required
   O:C = 0.333, which simultaneously raises the mean tail length 2.158 → 3.419 and, along with it, the
   connectivity of the organic phase. A fixed two-tailed topology at a fixed short tail (the Cooke &
   Deserno geometry, whose stable bilayer the project already has) gives the needed p **without** raising
   the number of covalent neighbors per molecule, and it is exactly that count which percolates. This is a
   change to the molecule, not to a constant.
2. **A dilution axis, INDEPENDENT of chemistry.** The percolation threshold is a function of ρ_org, and
   ρ_org is pinned high because `co_bond` does not fire at a prebiotically honest dilution (0 events over
   150,000 steps at a 691× enrichment). This model is structurally unable to separate "enough chemistry"
   from "above the gelation point." What is needed is either chemistry that works at low ρ_org (a real
   catalytic rate, not an enrichment coefficient), or **pre-built** amphiphiles that can be freely diluted,
   and the latter is cheaper.
3. ~~**An aperiodic or sufficiently LARGE region, so that the spanning object is not the ground state.**
   By current understanding, this is the root cause (§5.2).~~ **DONE AND RULED OUT (§11):** path
   (a) was carried out, a finite parcel with a soft neutral wall, wrapping axes 0 of 3 on 20 of 20, the
   `aggregate-percolation` gate passed for the first time, and closure did NOT occur. This is no longer the
   root cause. What remains is item 1 (the molecule's packing parameter), now the sole surviving
   explanation. The original wording is kept below as what was tested: Two paths: (a) leave periodicity
   behind, on one axis or all (a droplet in vacuum, walls, a reservoir), which gives the spanning micelle
   an edge and erases its advantage; (b) cross L\* = 160.6-161.3 σ, which requires
   3.29e6 particles against a ceiling of 953,589 and a neighbor structure that does not cost L³ρ². Path
   (a) is cheaper by orders of magnitude and changes the setup, not the machine.
4. **Real evaporation beyond 3.20×.** The literature cycle concentrates 1400-fold; the model achieves
   3.20 and makes up the rest with a pool-enrichment coefficient (~691×), which is not the same thing.
5. **Monomer exchange / solubility equilibrium.** Exchange, fission, and fusion are EXACTLY ZERO on all 22
   intervals: the channel is not being limited, it simply does not work. Without it there is no CMC, no
   Ostwald ripening, and no real fusion, that is, none of the processes by which a real vesicle grows and
   closes.
6. **Co-surfactant mixtures.** Every campaign is a single chemical species with a single head type. Real
   prebiotic fatty-acid vesicles are stable in a narrow pH window precisely because acid and soap coexist
   there; an acid-soap correlation has already appeared on its own in the model (§3.4), but as a charge
   correlation, not as a mixture of two species with different geometry.

**The cheapest next measurement, fully specified.** Pre-built two-tailed lipids (the same construction as
`tests/water-bilayer-*`, which already holds a stable bilayer at ρ_W = 0.8), without chemistry, without
evaporation, at four amphiphile densities spanning the percolation threshold: ρ_amph = 1.34e-2
(the current run), 6.7e-3, 3.3e-3, 1.7e-3 σ⁻³ on box 54, with `tests/percolation-check.test.ts` as the
readout and closure as the endpoint. This is exactly the question this model could never ask: does there
exist, at two-tailed topology, a density BELOW the gelation point and ABOVE the threshold molecule count?
The threshold on box 54 is ~1000 amphiphiles = ρ_amph 6.4e-3 σ⁻³, meaning the window, if it exists, is
narrow, and the arithmetic says it lies between the first two steps. **And it needs to be run in an
aperiodic or provably large region, or else item 3 will eat the result the same way it ate this project.**

---

## 10. How to recheck this

```
# the full gate table (about 3 minutes) -- writes verify/out/gates.json and report.html
npm run verify

# input artifacts that npm run verify reads but cannot recompute itself:
nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism   # ~135 s

CONTINUOUS_RUN_PREFIX=bbB76-step CONTINUOUS_RUN_DIRS=data/checkpoints/bbB76 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/gates-campaign-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism    # 34 s over 29 snapshots
PERC_CAMPAIGN_LABEL=bbB76 PERC_CHECKPOINTS="<23 wet bbB76 + controls dec54/swB54>" \
  PERC_ARTIFACT=verify/out/gates-percolation.json \
  nice -n 15 npx vitest run tests/percolation-check.test.ts --no-file-parallelism       # 16 s
# all off-GPU, from checkpoints already on disk. There is no need, and no reason, to rerun the campaign.

# continue a campaign (the same ENSEMBLE, not the same trajectory -- chunk boundaries cut noise differently):
nice -n 15 npx tsx soup/cli/campaign.ts --label bbB76 --box 76 \
  --start '{"C":55988,"O":18659,"H":55988,"M":1452,"W":351181}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --charge --pH 7.0 --ionicStrength 0.01 \
  --steps 4700 --every 4700 --dir data/checkpoints/bbB76        # ~405 s per call at N = 483,268

# the page live, headless: the run advances and draws
nice -n 15 npx tsx verify/viewer-smoke.ts
```

If an input artifact is missing, the corresponding gate comes out **unproven, with a reason**: the old
number is not substituted in. This rule lives in `verify/campaign-gates.ts` and is pinned by a test in
`tests/gates.test.ts`.

To see it with your own eyes: `npm run dev`, page `viewer/run.html`. The "small" preset has 3938 particles
(470 organic + 3277 water at a measured liquid density of 0.8 σ⁻³, plus a clay plate of 191), a measured
speed of **2049-2257 steps/s**, 119 frames over an 8000-step run. The "standard" preset has
25,904 particles, and its name carries the word EXPENSIVE. Water is not drawn by default (it outnumbers
everything else, it hides the chemistry, and it costs frames); the clay plate is drawn as tiles, not
spheres, and can be turned off.

---

## 11. A finite water parcel: "like in the ocean" (task `confined-parcel`, 2026-08-21)

The project's last task. The project owner's framing: "make it like in the ocean, that is the goal." Item 3 of the §9 list was tested, the only one this project had called the ROOT CAUSE: that
closure is absent because of the **topology of the periodic box**, in which an object spanning the box
wraps onto itself across the boundary and so has no edge at all, whereas a vesicle must pay for curvature.

### 11.1 What was built

The system is confined inside a **sphere of radius R** by a soft **neutral** repulsive wall:
U(s) = ½k(s−R)² for s > R, zero inside; k = 100 ε/σ² (ω·dt = 0.10, thermal penetration
√(2kT/k) = 0.148 σ). The wall **does not read particle species at all**: it is identical for water, heads,
tails, catalyst, and mineral: no offset by bead radius, no depth, no attraction. This is a CONTAINER, not
an interface. A hydrophilic wall would behave like the mineral one (see the clay section), a hydrophobic
one would nucleate a film; neutral is the only choice that **tests** closure rather than biasing it.

**Periodicity was removed by geometry, not by a flag.** Eleven places in the engine apply the minimum-image
convention or coordinate wrapping; adding a branch to each is exactly the class of silent bug this project
has been burned by seven times. Instead, the parcel was placed inside a cube of side **L ≥ 4R**, so that:
(1) no particle comes closer to a face than L/2 − R − δ, so the position wrap is the identity; (2) no
distance between ANY pair exceeds 2(R+δ) < L/2, so the minimum image is the identity for **every** pair,
not just the interacting ones. Both conditions are **checked with a throw at creation** and **measured at
every snapshot**: gap to the face 26.53-44.53 against the largest interaction radius of 16.70; largest pair
44.55-72.77 against L/2 = 48.66 (dry) / 80 (wet); wall penetration 0.055-0.416 σ. The instrument was
checked against a positive control: on the SAME coordinates at box 2R it finds 1140 splices out of 12,777,
against the real setting, **0 of 12,777**.

**Evaporation became physical.** Removing water shrinks the parcel: R_live = R_wet · L_live/L_wet, that is,
literally a drying droplet, on the existing dry-wet cycling machinery. Measured: water
156,346 → 112, parcel 36 → 21.895 σ, density 1.1009 → 1.34 σ⁻³ over 18 steps.

**The wall does work on the system.** The thermostat is Langevin and local, and it absorbs that work just
like the work of any other force; no energy-conservation claim is made here. **No quantity of the
pressure kind is published**: an area MC move at zero tension **FAILS** in the confined run (volume is set
by the wall, not the box; `soupPotential` carries no wall-contribution term): zero tension does not exist
inside the parcel. The `area-per-lipid-water` gate was, and remains, a measurement in the periodic box.

### 11.2 Four "box is not volume" defects, and which published numbers they touched: NONE

In a fully periodic box there is one volume, the box. So every density in the engine is written as
`N / box³`, and **in a periodic box every such expression is correct**. The parcel breaks this identity:
material occupies 195,432 σ³ within a box of 4,096,000 σ³, a ratio of **20.959**. Four places read a
density **20.959× too low**:

| place | what it drives | what would have happened |
|---|---|---|
| `densestDensityOf` | Verlet-list capacity | underestimated list = **a silently lost neighbor** (rescued here by `capacityFloor`, but by accident) |
| `makeEsBasis` | far-list capacity for heads | head density **21.0× too low** → capacity **281 instead of 5898**: a silently dropped interaction, exactly the defect the long-range task had removed |
| `encapsulatedWaterVolume` | `bulkWaterDensity` and the closure threshold | **the threshold would have been WIDENED 21×** (313 → ~15 beads) by bookkeeping accident |
| `evaporationLadder` | the density that drives the drying ramp | **caught LOUDLY**: the parcel reached 2.32 σ⁻³, 614,769 of 645,453 components went non-finite over steps 12,800…13,000, the finiteness guard fired naming the interval |

**Not a single previously published number was computed from a wrong density.** All four expressions were
correct for every run this project has published, because they all ran in a fully periodic box, where the
box volume IS the occupied volume. The defects are **latent**, not previously wrong: only a system whose
material occupies less than its box triggers them, and no such system existed before this task. In
particular, `encapsulatedWater` = **exactly 0 on 23 of 23** wet snapshots of the box-76 campaign is
unaffected either way: the count itself does not use fill or density, and its threshold of 312.64-315.41
was derived from a density measured in a box that WAS the system. Every fix is an **optional parameter
defaulting to the previous behavior**, so the periodic path did not "provably not change", it did not
change, by construction.

The portable lesson: **`N / box³` is not a density, it is a density AND an assumption.** The assumption
held throughout the project's history and was never written down, which is why breaking it broke four
places at once, not one.

### 11.3 The competing trap: amphiphiles do NOT plaster the container

This is the most plausible way this experiment could have failed, and it is measured on every snapshot: in
an outer shell of thickness 4.687 σ (the measured bilayer thickness, precisely the depth an adhering film
would occupy), against two null hypotheses: a uniform sphere and, most importantly, **the same shell being
occupied by SOLVENT**, which cancels the geometric factor common to every species.

- Before aggregation (step 2000): enrichment relative to water is **C 0.985, O 1.013, H 0.981, M 0.959**:
  the wall is neutral, measured, not asserted.
- In the settled wet phase: **1 carbon out of 24,926 and 0 hydrogens out of 24,926** in the shell, an
  enrichment relative to water of **1e-4** and **0.000**, on 18 of 18 snapshots, at an aggregate radius of
  gyration of **19.54-19.78 σ inside a parcel of radius 36**.

The material collapsed INWARD, into a compact object sitting far from the wall, and stayed there for
188,000 steps. Both implications carry weight: **a positive result here could not have been dismissed as
a film stuck to the wall, and this negative result cannot be dismissed as the wall having eaten the feed.**

### 11.4 What was measured: 0 of 3 axes, and closure is still absent

Campaign: R = 36 σ in a box of 160 σ, box-76 campaign composition rescaled to the parcel's volume
(`{C:24926, O:8307, H:24926, M:646, W:156346}`, N = **215,151**, ρ = 1.1009, water exactly **0.8000 σ⁻³**),
seed 19, kT 1.1, pH 7.0, I = 0.01 M, charge at a calibrated **4.000 λ_D (rc_es = 15.2000, dropped fraction
0.0916, the same numbers as in the periodic campaign)**, one drying event, starting from MONOMERS.
**225,200 steps, of which 188,000 were settled wet steps.**

| what | periodic box 76 (`closeout` campaign) | **finite parcel R = 36** |
|---|---|---|
| wrapping axes | **3 of 3** on 23 of 23 | **0 of 3 on 20 of 20** |
| `aggregate-percolation` gate | 3, **failed** | **0, PASSED** (first time in the project) |
| aggregates / largest | 4 / 4512 | 1 / **2092** |
| feed over closure floor | 4.73-4.82× | 2.06-2.23× |
| `encapsulatedWater` | **exactly 0** on 23 of 23 | **exactly 0 on 19 of 19** |
| closure threshold | 312.64-315.41 | **312.67-314.87** |
| `closed` / `hasVesicleAggregate` | false / false | **false / false** |
| flatness (need ≤ 0.35) | 0.7277-0.7610 | **0.8303-0.8552** |
| `radialHeadShells` | oscillates 1↔2 | **2 → 1** and stayed at 1 (moving AWAY from a vesicle) |
| cavity σ³ against 370.8656 | 139.25, slope **did not drop** (+0.586/1000) | **69.5**, slope **changed sign** (+0.172 → −0.264/1000) |
| size plateau (last four) | 0.11% | **0.062%**, the tightest in the project |
| fusions / fissions / exchange | 0 / 0 / 0 | **0 / 0 / 0** |
| largest covalent component | 35 of 30,361 | **35 of 13,239** (`covalentSpanFraction` 0.00264) |

**Threshold 312.67-314.87 against periodic 312.64-315.41: this is a cross-check of the §11.2 fix**: both
are 370.8656 σ³ × the density of real liquid water (0.843-0.849 here, ~0.85 there). Without the fix,
**~15** would have been published.

### 11.5 Verdict and its honest frame

**There is no closure.** `encapsulatedWater` **exactly 0 on 19 of 19** measurable snapshots against a
threshold of 312.67-314.87; `closed` false; `hasVesicleAggregate` false on all 21; gates
`vesicle-closure-water` = 0 (rank B, failed) and `vesicle-verdict` = 0 (rank A, failed). Judged by a
pre-set criterion and NEVER by shape (shape says the same thing: flatness 0.8379 against ≤ 0.35).

**The first of three named hypotheses failed: there is no wrapping, and there is still no closure.** Not a
film on the wall: §11.3 measured 1 carbon out of 24,926, so this explanation is excluded by measurement,
not by argument. **And the hypothesis of the experiment itself failed, the one this document had called
the root cause: that closure fails to occur because of the topology of the periodic box.** The topology is
gone, and closure is still absent. The periodic boundary was a **consequence, not a cause**.

Why, by arithmetic, not by a sigh: in a finite region the cheapest object WITHOUT AN EDGE is not a
spanning micelle but a **cylindrical micelle with end caps**, costing **17.5-21.3 amphiphiles** with no
lower bound tied to region size, against a closure floor of **937-1014**: the band is inverted by
**44-58×**. A micelle with caps spanning the parcel costs 439-490, that is, **0.43-0.52×** of the floor,
also cheaper than closure. The periodic box was not providing edgelessness as such but **edgelessness at
any size**; a capped micelle is edgeless at any size, including inside the parcel, at a cost of ~19
amphiphiles for a free end. The measured object is exactly that: one CONTACT network of 2092 chemically
distinct molecules, with bounded interstitial void and not a single trapped water bead.

**What is still standing after this step.** Ruled out: chemistry, feed (at 1.50×, 2.08×, 2.36×, and 4.82×
of the floor), pair energy, box size, temperature, charge/salt/pH, and now **boundary topology**. Exactly
one explanation survives, and it concerns the molecule: **this amphiphile's packing parameter a/t**
(§9, item 1: two-tailed topology BY CONSTRUCTION with a fixed short tail). This has stopped being the
first item on an expected-payoff list and become the only one.

**A frame that would have been owed on success too, and is given for that same reason.** (1) The wall is
neutral, MEASURED, but it is an idealization the ocean does not have: a real air/water boundary is
hydrophobic and would nucleate a film (the aerosol case, itself prebiotically real), a mineral one is
hydrophilic. (2) A parcel of 36 σ: closure inside a droplet whose radius is 1.6× the diameter of the object
itself is not closure in the ocean. (3) **The periodic result stands as its own conclusion, not as an
error that was corrected here**: in a periodic box below L\*, the ground state is a spanning object, and
the measured object was exactly that. The published gate table still carries that row, and the confined
arm is published as a SEPARATE row (`verify/out/gates-confined.json`). (4) To show the same thing without
confinement would need a box L > L\* = **146.9-172.5 σ** = **2.53e6 particles against a ceiling of
953,589**, still structurally unreachable, because the per-particle neighbor list costs L³ρ².
Confinement is the only way this engine can even ask the question, which is exactly why the wall's
neutrality had to be measured, not assumed.

Full protocol, all commands and all numbers: `.superpowers/sdd/2026-08-16-soup-to-vesicle/confined-parcel-report.md`.
To reproduce (there is no need to rerun the campaign):

```
# gates: the periodic table and the SEPARATE confined-arm row
nice -n 15 npm run verify
nice -n 15 npx tsx verify/confined-gates.ts

# implementation proofs (wall against a NUMERICAL gradient, absence of wrapping with a control,
# verification of the 0-of-3 setup, failure modes) -- about 32 s
nice -n 15 npx vitest run tests/soup-confine.test.ts --no-file-parallelism

# tools over checkpoints already on disk
CONTINUOUS_RUN_PREFIX=cpR36-step CONTINUOUS_RUN_DIRS=data/checkpoints/cpR36 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/confined-campaign-R36-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism
```
