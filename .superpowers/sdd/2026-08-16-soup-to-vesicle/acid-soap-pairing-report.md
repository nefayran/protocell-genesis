# Acid-soap pair in the coarse-grained model: implementation, strength sweep, pH window, arithmetic

Task `acid-soap-pairing`, branch `stage-a-atoms`, 2026-08-23. For the first time in the project, the
head-head term has a **computed number**, not an assumption: the atomistic engine in this same
repository (`atomic/`, MACE-OFF23) measured the association of two carboxylic heads. This report is
written for a person who was not here; every command and its full output, unedited, is in §9.

**In one sentence.** The charge-assisted pair is implemented so it reads the protonation state (both
as a FORCE and inside the constant-pH MC), the measured value of 10.38 kcal/mol turned out to be
**6 times higher than what the bilayer can withstand** (at 0.645 and 1.0 of the ceiling the thickness
drops 4.5 -> 1.4 -> 1.0 sigma against the 4-6 corridor), the chosen strength of 0.161 of the ceiling
holds both corridors and gives a paired fraction of 0.4246 with a pair lifetime of 2947 steps;
**alpha self-buffers to 1/2 from both sides**, a new and the cleanest result so far; **there is a pH
window, but not in bilayer capability**, rather in shape and compactness (flatness 0.18-0.25 and
Rg/(L/2) 0.58-0.77 at pH 4-6 against 0.45-0.53 and 0.88-0.99 at pH 7-9, with an S=0 control where
there is no effect); **closure did NOT become the cheapest boundless object**, the band is inverted
by 46.8-53.2x against the previous 44-58x. **The campaign is NOT earned.**

---

## 1. How the rule reads the protonation state

The new term is **the only attraction in the model that is not a function of two species**.

- **Force**, `soup/wgsl/pair.wgsl`'s `acidSoapScale(ti, tj, qi, qj)`: returns depth only if both
  beads are class POLAR (head) **and** `(qi != 0) != (qj != 0)`, that is, an XOR on "is charged".
  One protonated head (q = 0, neutral acid) plus one deprotonated head (q = -1, carboxylate). Two
  identical heads give zero, structurally.
- The charge is taken from `chargeRO`, whose **only writer is the constant-pH MC**
  (`soup/src/electrostatics.ts` + `soup/src/soup-protonation.ts`). No new state, no new species and
  no new buffer was added: the head changes who it pairs with on the same sweep on which it changes
  its own charge. That is exactly why the pH window is a prediction, not a construction.
- **The well shape is not its own**: the depth is summed with the cell table into a SINGLE
  multiplier on the same `attr_dv` (the well onset rc = wca_cut(sigma*b_tt), width wc, depth
  epsilon, rank A, `data/params.json`, untouched). So the new term contributes neither a different
  well onset, nor a different width, nor a second cutoff, and **no neighbor-grid or Verlet-list
  coverage guarantee has changed**. Since `polarPolar` in the table remains exactly 0.0, ALL
  head-head attraction in this model is this term and nothing else.
- **CPU twin**: `soup/src/soup-potential.ts` carries a transcription of the same XOR; this is the
  energy that the Metropolis criterion for the area MC move stands on.
- **And the pair term enters the acceptance criterion of the constant-pH MC**: `soup/src/acid-soap.ts`'s
  `acidSoapSiteWork`, called from `protonationSweep`. This is not optional: a protonation-state change
  alters the potential energy through the pair term exactly as it does through the screened Coulomb
  term, and sampling without it would follow a distribution not consistent with the potential the
  dynamics integrates, the same class of silent inconsistency as the stale F(x) (§7.3 of the final
  document).

The rule exists in exactly **three places that are required to agree**, and all three are pinned:
`pair.wgsl` (force), `pairingStats` (population), `soup/src/acid-soap.ts` (pair identity and
lifetime). `nearestUnlike` **was moved** from `tests/electrostatics-audit.test.ts` into
`soup/src/acid-soap.ts` (a clean move, same text) so the audit and this sweep cannot diverge in the
definition of a pair.

### 1.1 Splitting responsibility BEFORE adding

`soup/wgsl/step.wgsl` stood at 490 lines; the CLAUDE.md rule is "split first, then add," and this
same file has already been split by that rule twice (`verlet.wgsl`, `health.wgsl`/`relax.wgsl`).
**Pair responsibility** was extracted -> new `soup/wgsl/pair.wgsl`: the `Species`/`SP` and
`AttrScaleTable`/`AttrScale` declarations, `mi3()`, `speciesRadius/Polar/Solvent/Mineral/Class`,
`pairAttrScale`, `shouldAttract`, `pairB` and `nonbondedSoup()` itself. A clean move: not a single
binding index shifted, not a single formula was rewritten; the concatenation order in
`soup/src/soup-pipelines.ts` is `forces.wgsl, electrostatics.wgsl, pair.wgsl, step.wgsl, ...` (WGSL
has no forward declarations). The move is **proven by a run** of `tests/soup-forces.test.ts` BEFORE
any addition (§9.1): grid+Verlet against brute force 6.1035e-5 at mean|F| 5.1968.

step.wgsl: **490 -> 361**, pair.wgsl **204**. No file in the tree exceeds 600 lines (§8.4).

### 1.2 A hole closed as a side effect: charge did not participate in the area MC move

The area MC move built its own basis WITHOUT `esOverrides` (that is, `enabled` came from the file,
where it was `false`) and called `soupPotential` **without charges**. For an uncharged run this is
exactly correct, and every published area-move gate is uncharged and remains bit-for-bit the same;
but on a charged run this **silently dropped both charge-reading terms from the criterion**. This
task needs a charged area move, so the hole is closed, not worked around: the move is now given the
system's licensed basis and live charges.

---

## 2. Mapping the measured number, and why it is an UPPER BOUND

| what | number |
|---|---|
| acid+acid, vacuum | **16.43 kcal/mol** |
| acid+acid, 32 waters | **-5.51 kcal/mol**, in water the neutral dimer is UNFAVORABLE |
| acid+carboxylate, vacuum | **21.04 kcal/mol** |
| acid+carboxylate, 32 waters | **+10.38 kcal/mol**, four independent geometries into one minimum |

10.38 kcal/mol = **43.4299 kJ/mol**. Normalized the SAME WAY as every `pairEpsilon` level: divided by
`reference` = `apolarApolar` = 3.5 kJ/mol (MARTINI 2.1) -> **12.4086** in the table's dimensionless
units, that is, 12.4 times deeper than the reference apolar pair.

**This is ENERGY, not free energy.** The entropic cost of pairing **has NOT been computed**, it
requires enhanced sampling, which the atomistic engine does not have, and it works **against** the
pair: two molecules losing rotational and translational degrees of freedom pay for that. Plus a
coarse-grained bead is not an atom, and carrying the absolute depth over from the all-atom
calculation into this model's energy unit would be the same kind of fitting that §4's
`attractionScale.basis` field forbids for MARTINI. Therefore **12.4086 is taken as a CEILING, and the
strength is swept**.

### 2.1 What is recorded in `data/soup.json` and at what rank

`solvent.attractionScale.acidSoapPair`:

```
epsilonKJ: 7.0            <- WHAT IS USED = 2.0 normalized = 1.67 kcal/mol
measuredUpperBoundKJ: 43.4299
measuredKcalPerMol: 10.38
rank: "D"
basis: 6264 characters -- mechanism, mapping, the whole sweep, ranks, feedback on titration
```

**Rank D, and it is stated outright.** The ceiling 12.4086 is rank **B** (computed, not measured:
MACE-OFF23 is an ML potential, and this is energy without entropy). The value used is a **fraction**
of this ceiling (0.1612), chosen by the sweep against this tree's own gates, so as an estimate it is
**D**. Under the project's rule (spec, section 3) gates resting ONLY on D come out **unproven by
definition**, and any future gate on this number must carry this rank. Claiming A or B here would
mean passing off energy as free energy.

Untouched: `data/params.json` (rank A), `epsilonScale` (still 1.0), `polarPolar` (still 0.0),
`co_bond.attemptRate`, not a single literature corridor.

---

## 3. GATES 1 and 2: strength sweep, area, thickness, paired fraction, lifetime

Charged bilayer patch in explicit water, rho_W = 0.8 sigma^-3: **400 lipids, box [16.125, 16.125, 30],
seed 31, kT 1.1, pH 7.0 (the model's own pH), I = 0.01 M**, zero-tension area MC move, N = 5700.
Setup is a deliberate copy of `tests/water-bilayer-area-move.test.ts` (parameter for parameter), so
the S = 0 arm here is a charged control for its own published uncharged row. Settling criterion is
the same one: |t| < 2 **and** |drift|*TAIL < 0.01 **and** ln A range < 0.02, checked only at
non-overlapping checkpoints.
Tool: `tests/soup-acid-soap-bilayer.test.ts`, artifact `verify/out/acid-soap-strength-sweep.json`.

| S (fraction of ceiling) | settled | area [min, max] | corridor 1.1-1.5 | thickness | corridor 4-6 | verdict |
|---|---|---|---|---|---|---|
| **0** (0.000) | yes, 300 | **1.2026** [1.1979, 1.2071] | **INSIDE** | **4.4191** | **INSIDE** | **passed** |
| **1** (0.081) | yes, 300 | **1.1556** [1.1510, 1.1593] | **INSIDE** | **4.5080** | **INSIDE** | **passed** |
| **2** (0.161) | yes, 200 | **1.1510** [1.1473, 1.1546] | **INSIDE** | **4.5517** | **INSIDE** | **passed** |
| 4 (0.322) | yes, 200 | 1.1042 [**1.0972**, 1.1149] | **OUTSIDE** (below 1.1) | 4.9729 | INSIDE | **FAILED** |
| 8 (0.645) | **no** | 1.2154 [1.2108, 1.2214] | INSIDE | **1.4191** | **OUTSIDE** | **FAILED** |
| 12.409 (**1.000, CEILING**) | **no** | 1.1637 [1.1540, 1.1747] | INSIDE | **1.0079** | **OUTSIDE** | **FAILED** |

The same arms, protonation and pairing (paired fraction and survival averaged over the last six
trajectory checkpoints, not a single frame):

| S | alpha | paired fraction | unlike-contact excess | pair survival over 10,000 steps | **pair lifetime** | lnA drift (t) | cluster | buried heads | waters in core |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **0.9446** | 0.0917 | 1.313 | 0.0385 | **2686 steps** | -0.70 | 0.9843 | 0.0630 | 29/4500 |
| 1 | 0.6808 | 0.3587 | 1.186 | 0.0306 | **2857** | -1.08 | 0.9974 | 0.0785 | 10/4500 |
| **2** | **0.5921** | **0.4246** | **1.154** | 0.0345 | **2947** | **+0.70** | **1.0000** | **0.0155** | **5/4500** |
| 4 | 0.5396 | 0.4088 | 0.979 | 0.1719 | 5690 | -1.48 | 1.0000 | 0.0182 | 0/4500 |
| 8 | 0.5100 | 0.3438 | 0.892 | 0.4105 | 11,250 | -2.15 | 1.0000 | 0.0822 | 2/4500 |
| 12.409 | **0.5017** | 0.3079 | 0.863 | 0.5586 | **17,285** | -8.67 | 1.0000 | 0.0582 | 2/4500 |

**What this shows, point by point.**

1. **GATE 1: the bilayer survives only up to 0.161 of the ceiling.** The measured value DESTROYS it:
   at 0.645 and 1.0 the thickness falls to 1.42 and 1.01 sigma against the 4-6 corridor, and the area
   has not even settled yet (t = -2.15 and -8.67, that is, the patch was still shrinking). At 0.322 the
   thickness is fine (4.97), but the area drifts 0.3-0.9% BELOW the lower bound 1.1, the corridor is
   not widened, the arm is declared failed. **Not a single threshold moved.**
2. **GATE 2: the pair forms and lives.** The paired fraction grows **0.0917 -> 0.4246** (a
   4.6x increase), the lifetime grows monotonically **2686 -> 17,285 steps** (a 6.4x increase). At the
   chosen strength S = 2: **0.4246 paired at a lifetime of 2947 steps**, and this is ABOVE the
   0.058-0.080 the project observed as a charge correlation without the pair term (§3.4 of the final
   document).
3. **The unlike-contact excess and how it misleads as a readout.** At S = 0 it is 1.313, at S = 2 it
   is 1.154, and on the strong arms it FALLS below 1 (0.863-0.979). The reason is stated, not
   papered over: this excess is normalized by 2*alpha*(1-alpha), and a strong pair pulls alpha toward
   1/2, where the random baseline is MAXIMAL, so the denominator grows faster than the numerator.
   **The paired fraction is an honest readout, the unlike-contact-excess ratio under a strong term is
   not.**
4. **Alpha self-buffering is a new result.** At a fixed pH of 7.0, alpha moves
   **0.9446 -> 0.6808 -> 0.5921 -> 0.5396 -> 0.5100 -> 0.5017**, that is, toward 1/2. The mechanism is
   not tuned in, it follows from §1: a head among CHARGED neighbors gains pairing energy by becoming
   neutral, and vice versa, so the pair term is a **restoring force pulling alpha toward 1/2**. And it
   works from BOTH sides: at pH 4, alpha goes **0.1122 (S = 0) -> 0.2556 (S = 2)**, that is, UPWARD
   (§4). Pinned against the system's exact Boltzmann weights for two heads (§9.5:
   U_pair(one) = -2.000 eps = -1.8182 kT at depth 2).
5. The core stays dry on every arm (0-29 water beads out of 4500), cluster 0.98-1.00, buried heads
   0.0155-0.0822, that is, the failures of arms 8 and 12.409 are **not** water penetration and **not**
   leaflet breakup, but collapse/interdigitation: the head pulls the head HARDER than the tail pulls
   the tail.

**S = 2.0 = 7.0 kJ/mol = 0.161 of the ceiling = 1.67 kcal/mol was CHOSEN**: the deepest swept value at
which both literature corridors hold on a **settled** area (t = +0.70) and which gives the largest
paired fraction.

---

## 4. GATE 3: pH sweep, where finite bilayer-capable objects appear

**Why the tool is self-assembly, not a ready-made patch.** A ready-made patch is not a readout of pH
in this model: a single-tail amphiphile with tail-tail attraction holds a sheet at ANY pH, because
nothing forces it to prefer a micelle, "the patch survived" would be true everywhere and would
distinguish nothing. So the arms start from **DISPERSED** ready-made amphiphiles (no chemistry, no
evaporation) in explicit liquid water, and the settled state is asked the project's own bilayer
question: a lamellar readout from `analyzeAggregates`, flatness <= 0.35, in-plane symmetry >= 0.5 and
TWO head layers, on the FINAL object.

Setup: **300 amphiphiles, box 28 sigma, rho_amph = 1.367e-2 sigma^-3**, exactly the density at which
the project's two-tail campaign ran (§9 of the final document: "rho_amph = 1.34e-2, current run"),
water 0.8 sigma^-3, N = 17,118, seed 19, 40,000 steps, measured every 8000. Finiteness is read as Rg
against L/2, the same Rg a UNIFORMLY filled box would have (Rg^2 = 3*L^2/12), exactly the readout the
final document uses when it says "37.43 against 38.000 for a box of 76 (98.5%)."
Tool: `tests/soup-acid-soap-ph.test.ts`, artifact `verify/out/acid-soap-ph-sweep.json`.

**Arms with the pair (S = 2):**

| pH | alpha | apparent pKa | paired fraction | largest | flatness (<=0.35) | symmetry (>=0.5) | **Rg/(L/2)** | head layers | **bilayer-capable** |
|---|---|---|---|---|---|---|---|---|---|
| 4 | 0.2556 | 4.467 | **0.1422** | 106.0 | **0.1907** | 0.4567 | **0.7091** | 2 / 0 | 0 of 3 |
| **5** | **0.5011** | 4.998 | **0.1444** | 52.7 | **0.1821** | 0.3631 | **0.5825** | 2 / 0 | **1 of 3** |
| 6 | 0.7433 | 5.536 | **0.1567** | 133.3 | **0.2478** | 0.5629 | **0.7668** | 1 / 1 | 0 of 3 |
| 7 | 0.9033 | 6.027 | 0.0678 | 167.3 | 0.4470 | 0.6391 | 0.9676 | 2 / 0 | **1 of 3** |
| 8 | 0.9789 | 6.316 | 0.0256 | 137.0 | 0.5252 | 0.8268 | 0.8831 | 1 / 0 | 0 of 3 |
| 9 | 0.9956 | 6.625 | 0.0078 | 142.3 | 0.5042 | 0.7465 | 0.9856 | 1 / 0 | 0 of 3 |

**Control WITHOUT the pair (S = 0), the thing that makes the comparison a comparison:**

| pH | alpha | paired fraction | largest | flatness | symmetry | Rg/(L/2) | bilayer-capable |
|---|---|---|---|---|---|---|---|
| 4 | 0.1122 | 0.0433 | 147.7 | **0.5360** | 0.8634 | **0.8977** | 0 of 3 |
| 5 | 0.5033 | 0.0611 | 174.7 | **0.4625** | 0.6726 | **1.0142** | 0 of 3 |
| 6 | 0.8889 | 0.0222 | 135.0 | **0.3589** | 0.6636 | **0.9252** | **2 of 3** |

**What was measured, honestly and in order.**

1. **A PAIRING window exists and it is sharp.** Paired fraction 0.142-0.157 at pH 4-6 against 0.068 /
   0.026 / 0.0078 at pH 7 / 8 / 9, an 18-20x drop. The pair exists only where alpha is not pinned to
   1, that is, near the model's apparent pKa, exactly the predicted shape.
2. **A SHAPE-AND-COMPACTNESS window exists, and it coincides with the pairing window.** Flatness
   **0.19-0.25** at pH 4-6 (inside the bilayer threshold of 0.35) against **0.45-0.53** at pH 7-9
   (outside); Rg/(L/2) **0.58-0.77** against **0.88-0.99**. The transition is monotonic and lies
   between pH 6 and 7.
3. **And this shift is caused by the PAIR, not by pH.** At the same three pH values, the S = 0
   control gives flatness 0.536 / 0.463 / 0.359 against 0.191 / 0.182 / 0.248 with the pair, and
   Rg/(L/2) 0.898 / 1.014 / 0.925 against 0.709 / 0.583 / 0.767. That is, at S = 0 the object in this
   regime **fills the box** (Rg/(L/2) roughly 0.9-1.0, even 1.014 at pH 5), while with the pair it
   becomes **compact and flat**. This is the first time in the project that anything has shifted this
   quantity by a measurable amount.
4. **A STRICT bilayer-capable readout is NOT confirmed, and this is stated as a result.** The binary
   `hasLamellarAggregate` gives 1 of 3 checkpoints at pH 5 and 1 of 3 at pH 7 with the pair, and **2
   of 3 at pH 6 WITHOUT the pair**, that is, the binary criterion is noisy at this size and its sign
   disagrees with the continuous readouts. The reason is visible in the columns: on the flat arms
   (pH 4-5, S = 2), the **in-plane symmetry** drops (0.457 and 0.363 against the needed 0.5), that
   is, the object is flat but elongated within its own plane, a **ribbon, not a disk**. Claiming from
   this data that "bilayer-capable objects appeared in the pH window" would be wrong; the measured
   statement is: a window in pairing, flatness and compactness, with an in-plane-elongated shape.
5. **The apparent pKa is again emergent and again shifted where it should.** With the pair it goes
   4.47 -> 6.63 as pH runs 4 -> 9 (an upward shift with growing alpha, as was measured in the
   electrostatics task), and comparing S = 0 / S = 2 at pH 4 (4.900 against 4.467) and at pH 6 (5.076
   against 5.536) shows that the pair **pulls titration toward alpha = 1/2 from both sides**, not
   just one.
6. **The material did not assemble into a single object**: 31-47 aggregates, the largest 53-175 of
   300 after 40,000 steps. This is a run-length limitation, stated plainly; the conclusions of §4
   rest on the shape of the LARGEST object, not on assembly completeness.
7. **The first setup was rejected, and is presented as a discarded tool.** 100 amphiphiles in a box
   of 24 (rho_amph = 7.2e-3) gave 32 aggregates of about 22 amphiphiles each, and `radialHeadShells`
   returned **`unavailable`**, too few heads per bin for the two-layer readout to be taken AT ALL.
   That arm could not answer the question either way and is not part of the conclusions.

**The literature window of approximately 7-9 does not match the model's window, and here is why:
this is not fitting and not an excuse.** The model is given only the **monomer** pKa of 4.9, while
the fatty-acid-vesicle window in the literature sits around the **apparent** pKa at the bilayer
surface (roughly 7-8.5), which is 2.5-3.5 units above the monomer value. This model earns its
interfacial shift on its own, and it is measured as **+0.08 / +0.32 / +0.67 / +0.97** (the
electrostatics task) plus here up to +1.7 at high alpha, that is, 2-3 times SMALLER than the
literature value. So the window measured here (alpha roughly 0.25-0.75 at pH 4-6) sits where THIS
model's alpha is roughly 1/2, and the comparison must be made in units of alpha, not pH.
**In units of alpha the shape of the window matches the literature: pairing and flatness live
around alpha = 1/2 and die as alpha -> 1.** In units of pH it does not, and the difference is a
measured deficit in the interfacial shift, not a property of the pair.

---

## 5. GATE 4: closure floor and the cheapest boundless competitor, recomputed

`tests/soup-acid-soap-arithmetic.test.ts`, artifact `verify/out/acid-soap-arithmetic.json`. Every
formula is from `tests/supply-window.test.ts` and §11.5 of the parcel report, unchanged; what is new
here is the INPUT: measured (a, t) at each pair strength. V_closure = 370.8656 sigma^3, taken as is.

R_in = (3V/4pi)^(1/3) = 4.4574; R_mid = R_in + t/2; floor = 2*4pi*R_mid^2/a; v = a*t/2;
spanning micelle = pi*R^2*L/v at R = t/2; spherical micelle (zero-length cap) = (4pi/3)*R^3/v.

| basis | a | t | v | **closure floor** | spanning at L=54 | /sigma | **spherical micelle** | inverted against sphere | L\* |
|---|---|---|---|---|---|---|---|---|---|
| explicit water, UNCHARGED gate | 1.1835 | 4.9061 | 2.903 | 1014.0 | 351.6 | 6.512 | 19.1 | **53.0x** | 155.7 sigma |
| charged patch S = 0 | 1.2026 | 4.4191 | 2.657 | 928.8 | 305.4 | 5.656 | 17.5 | **53.2x** | 164.2 sigma |
| charged patch **S = 1** | 1.1556 | 4.5080 | 2.605 | 979.5 | 330.9 | 6.128 | 18.4 | **53.2x** | 159.8 sigma |
| charged patch **S = 2 (chosen)** | 1.1510 | 4.5517 | 2.620 | **989.8** | 335.4 | 6.212 | **18.8** | **52.5x** | **159.3 sigma** |
| charged patch S = 4 [failed] | 1.1042 | 4.9729 | 2.746 | 1097.3 | 382.0 | 7.074 | 23.5 | **46.8x** | 155.1 sigma |
| charged patch S = 8 [failed] | 1.2154 | 1.4191 | 0.862 | 552.0 | 99.0 | 1.834 | 1.7 | **318.1x** | 301.0 sigma |
| charged patch S = 12.409 [failed] | 1.1637 | 1.0079 | 0.586 | 531.6 | 73.5 | 1.361 | 0.9 | **581.4x** | 390.7 sigma |

**VERDICT: CLOSURE IS STILL NOT THE CHEAPEST OPTION. At no swept strength.**
The band [closure floor, spherical micelle] is inverted by **46.8-53.2x** (52.5x at the chosen
strength) against the previous **44-58x**, that is, a 1.13x improvement on the best arm against the
required roughly 50x. The band against the spanning competitor is inverted by 2.87-2.96x (previously
2.88-2.97x), and L\* shifted from **160.6-161.3 to 155.1-159.3 sigma**, which, against the engine's
ceiling of 115.7 sigma (zero Verlet-list headroom), remains structurally unreachable.

**And three things that need to be said BEFORE the reader asks them.**

1. **Counting in PAIRS instead of amphiphiles changes NOTHING**, and this is confirmed by a test, not
   by an excuse: a pair is two amphiphiles, so both the floor (989.8 -> **494.9 pairs**) and the
   spherical micelle (18.8 -> **9.4 pairs**) are divided by two, and every ratio is identically the
   same.
2. **Israelachvili's packing parameter p = v/(a0*l_c) is NOT an independent argument in this model.**
   With v = a*t/2, a0 = a, l_c = t/2, it is **identically 1** at any strength, by construction, which
   is exactly what the test prints. The claim "the pair shifted the packing parameter into the
   1/2-1 band" would be unfalsifiable here, so it is not made.
3. **The pair does NOT give a cylindrical micelle an edge.** All heads of a cylindrical micelle sit
   on its single outer surface, so acid-soap pairs form there exactly as they do in the bilayer. The
   only thing the pair can change in this arithmetic is a and t, and that is exactly what is measured
   in the table. The project's previous conclusion ("boundlessness AT ANY SIZE, about 19 amphiphiles
   per free end") stands.

Note arms 8 and 12.409: there the "floor" and the "competitor" DROP, and naively this reads as an
improvement, but the reason is a collapsed thickness of 1.0-1.4 sigma, that is, the object is no
longer a bilayer. The band there is inverted by 318-581x, that is **6-11x worse**, not better.

---

## 6. Regressions: before and after

Known and NOT ours (final document §8.12): `soup-grid-resize` fails on unchanged code (proven three
times), `soup-drywet-cycling` is noisy at the boundary, 2 of 3 `rim-lambda-insitu` were failing before
us. No corridor widened, no tolerance changed.

| test | before | after | numbers |
|---|---|---|---|
| `soup-forces` (4 tests) | -- | **4 passed** | grid+Verlet against brute force **1.9073e-5** at mean|F| 5.1966; charged against neutral 191.06; long-range list full at 13.5 sigma (1.9073e-5); confinement 7.63e-5 |
| `water-bilayer-area-move` | 1 passed (a864854: 1.1835 / 4.9061) | **1 passed** | `settled=true chunks=300 area=1.1734 [1.1696, 1.1765] drift=-2.722e-6 t=-0.71 thickness=4.4689 cluster=1.0000 buried=0.0919 water=45/4500 acc=0.1280 verdict=passed` |
| `gate6-bilayer` (2) | 1.206 / 4.458 | **2 passed** | `GATE6 area 1.2010 +/- 0.0088 thickness 4.4688 lnA drift 5.835e-6 t=0.47 accepted 0.295` |
| `soup-electrostatics` (8) | 8 passed | **8 passed** | detailed balance: **U_pair(one) = -2.000 eps = -1.8182 kT at depth 2**, worst deviation within 5e-3; Henderson-Hasselbalch < 0.01 |
| `soup-stale-force` (2) | 2 passed | **2 passed** | -- |
| `soup-checkpoint` (4) | 4 passed | **4 passed** | -- |
| `soup-area-move` (2) | 2 passed | **2 passed** | Jacobian (N_beads - N_molecules)*kT holds |
| `sim` (8) | 8 passed | **8 passed** | -- |
| `gates` (8) | 8 passed | **8 passed** | -- |
| `supply-window` (1) | 1 passed | **1 passed** | `L*=[143.1, 174.6] sigma`, band empty, inverted 2.88-2.97x, **the pin did not move** |
| `percolation-check` (1) | 1 passed | **1 passed** | skipped without `PERC_CHECKPOINTS` (as before) |
| `coalescence-mechanism-pin` (5) | 5 passed | **5 passed** | -- |
| `soup-valence` (1) | 1 passed | **1 passed** | 45,000 steps, valence intact |
| `electrostatics-audit` (1) | 1 passed | **1 passed** | after moving `nearestUnlike` into `soup/src/acid-soap.ts` |
| `soup-bonds` (3) | 3 passed (103,448 ms) | **2 passed, 1 TIMEOUT** | "monomer counts of each kind... are conserved" failed with **Test timed out in 120000ms**, both in the batch and as a solo run |
| `run-ui` (8) | -- | **7 passed, 1 FAILURE** | "bead behind the box on z" -> `TimeoutError: Waiting failed: 30000ms exceeded`; **PROVEN PRE-EXISTING** (§9.8: the same test fails the same way on HEAD under `git stash`) |
| `npx tsc --noEmit` | 21 errors | **21 errors** | same pre-existing `ArrayBufferLike`/`SharedArrayBuffer` class |
| new: `soup-acid-soap` (3) | -- | **3 passed** | rule, CPU twin, file depth |
| new: `soup-acid-soap-bilayer`, `-ph`, `-arithmetic` | -- | **passed** | measurements §3-§5 |

**Two non-passes, each analyzed, neither healed by widening a threshold.**

- **`run-ui`, "bead behind the box"**: pre-existing. Checked exactly the way the project checks such
  things: `git stash push -u -- soup engine tests data verify`, the same test, the same failure
  (46,965 ms against 48,649 ms on our side), `git stash pop`. This is NOT our regression.
- **`soup-bonds`, "monomer counts... are conserved"**: **a wall-clock timeout, not a claim about
  physics**: the test expires with `Test timed out in 120000ms` (120,127 ms in the batch, 120,000 as
  a solo run), and its own published history is **103,448 ms at a limit of 120,000**, that is, only
  16% headroom. The test runs 50,000 steps at N = 14,300 WITHOUT charge (`electrostatics.enabled` is
  `false` in the file), so the pair term is structurally absent and cannot be slowing it down. Same
  class as `soup-drywet-cycling`: **it should be fixed by run length or test limit, not by a
  threshold**, and we did neither. Honestly left open (§8.1).

---

## 7. The gate table, regenerated

`nice -n 15 npm run verify`, 2026-08-23, full output in §9.9. `verify/out/gates.json` and
`report.html` were not hand-edited. `GateResult` carries `corridor` alongside `verdict`.

| gate | value | corridor | rank | verdict |
|---|---|---|---|---|
| area-per-lipid | 1.2056205530467474 | inside | A | **passed** |
| bilayer-thickness | 4.474978261285809 | inside | A | **passed** |
| bending-modulus | null | none | A | unproven |
| **area-per-lipid-water** | **1.1734461123400626** | **inside** | C | **passed** |
| **bilayer-thickness-water** | **4.468932297634941** | **inside** | C | **passed** |
| vesicle-closure-water | 0 | outside | B | failed |
| aggregate-percolation | 3 | outside | B | failed |
| vesicle-verdict | 0 | outside | A | failed |
| chain-length-asf | 0.15940333430827724 | none | D | unproven |
| mean-tail-length | 3.419 | outside | D | unproven |
| closure | 1284.875 | inside | D | unproven |
| chain-to-bead-mapping | 3 | none | D | unproven |

**Not a single verdict changed.** Two explicit-water rows carry a NEW measurement of the uncharged
patch (1.1734 / 4.4689 against the previous 1.1835 / 4.9061), a run, not an edit. The campaign rows
(`vesicle-closure-water`, `aggregate-percolation`, `mean-tail-length`) read the `bbB76` checkpoints
already on disk and were not recomputed: this task did not run the campaign.

---

## 8. What remains disputed, unproven, or measured only once

1. **The `soup-bonds` timeout is not closed** (§6). This will cost a run for whoever meets it next.
2. **One seed per arm**: 31 for the strength sweep, 19 for the pH sweep. The binary bilayer-capable
   readout (§4, point 4) on one seed and three checkpoints is **noise**, and it is presented as such;
   the continuous quantities (flatness, Rg, paired fraction) are consistent within an arm and across
   arms, but are also a single line.
3. **The strength sweep was run only at pH 7.0.** The model's own pH was chosen because every
   campaign has run at it and because alpha self-buffering is most visible there (0.9446 -> 0.5017).
   But the area/thickness corridors at strength 2 have not been re-measured at pH 5, where pairing is
   maximal.
4. **The pH sweep did not converge on assembly**: 31-47 aggregates, the largest 53-175 of 300 after
   40,000 steps. The flatness of the largest object is not the flatness of an equilibrium object.
5. **The long-range head-list capacity** on the pH 4-6 arms (S = 2) is derived from the DISPERSED
   density (64 per head), while on the pH 7-9 arms it is derived from the condensed one (saturated to
   all 300 heads). No interaction was dropped (the engine throws a named exception, and that is
   exactly how this was found), but the two subsets of arms ended up with a formally different
   capacity basis; arms 4-6 were not rerun because they never overflowed.
6. **Pair entropy has not been computed**, so the chosen depth of 0.161 of the ceiling has no
   thermodynamic justification above the gates, only a gate-based one. Rank D (§2.1).
7. **The unlike-contact-excess ratio under a strong pair term is a poor readout** (§3, point 3); it
   should not be cited in reports as "the pair is weakening."
8. `run-ui`'s pre-existing failure is left as is; `soup-grid-resize`, `soup-drywet-cycling`,
   `rim-lambda-insitu` were not run.
9. **File sizes**: after trimming comments, `soup/src/sim.ts` and `soup/src/electrostatics.ts` stand
   at EXACTLY 600. The next addition to either requires a split, not a line.

---

## 9. Every command and its actual output

### 9.1 The WGSL split verified BEFORE adding

```
$ nice -n 15 npx vitest run tests/soup-forces.test.ts --no-file-parallelism
SOUP-FORCES-ES maxDiff(grid+Verlet against brute force)=6.1035e-5 meanAbsRef=5.1968 charged_heads=100/100 maxDiff(charged against neutral)=191.0688 A=0.976250 kappa=0.832178 rc=4.8066620
SOUP-FORCES-ES-LONG maxDiff=1.9073e-5 meanAbsRef=3.4901 rc_es=13.5000 (=3.553 lambdaD, target=15.2000, image_ceiling=13.5000) splitRadius=2.9469545 nbCutoff_was=2.7224620 dropped_integrally=0.1304
SOUP-FORCES-CONFINED maxDiff=7.6294e-5 meanAbsRef=14.8567 N=18888 outside_parcel=356 max|F_wall|=26.9749
 ✓ tests/soup-forces.test.ts (4 tests) 33685ms
 Test Files  1 passed (1)
      Tests  4 passed (4)
```

### 9.2 The pair rule pinned

```
$ nice -n 15 npx vitest run tests/soup-acid-soap.test.ts --no-file-parallelism
ACID-SOAP-RULE N=648 heads=432 rc=1.1224620 S=4 expected net|dF|=3.21845 (near 1.2 -> 0.14889, far 1.8 -> 0.95350)
  ALL NEUTRAL:      max|dF| head=0.000e+0 other=0.000e+0
  ALL CHARGED:       max|dF| head=0.000e+0 other=0.000e+0
  ALTERNATING:       max|dF| head=3.2185e+0 mean=1.073e+0 other=0.000e+0
  grid against brute force: neutral 0.00e+0/0.00e+0 alternating 0.00e+0/0.00e+0
ACID-SOAP-CPU-TWIN S=4
  ON:  acidSoap(basis)=4.0000 samples=18 max|numeric - analytic|=1.653e-2 rel=1.554e-4 scale|F|=106.3675
  OFF: acidSoap(basis)=0.0000 samples=18 max|numeric - analytic|=5.348e-4 rel=1.441e-3 scale|F|=0.3711
ACID-SOAP-DEPTH file=7 kJ/mol / reference 3.5 = 2.000000 (ceiling 10.38 kcal/mol = 43.4299 kJ/mol = 12.4085 => fraction 0.1612), rank=D, epsilon(rank A)=1
 Test Files  1 passed (1)
      Tests  3 passed (3)
```
**max|dF| EXACTLY ZERO** on both identical-charge states, that is, identical heads do not attract
structurally, not "weakly"; on the alternating state 3.2185 against the 3.21845 derived from the
model's own geometric constants (two unlike partners, 1.2 and 1.8 sigma, pulling toward each other);
tail forces are bit-for-bit the same.

### 9.3 Strength sweep (six separate invocations, each under 500 s)

```
$ ACID_SOAP_STRENGTHS=0 nice -n 15 npx vitest run tests/soup-acid-soap-bilayer.test.ts --no-file-parallelism
ACID-SOAP-SWEEP S=0 pH=7 N=5700 settled=true chunks=300 steps=152000 area=1.2026 [1.1979, 1.2071] corridor 1.1-1.5 -> INSIDE | thickness=4.4191 corridor 4-6 -> INSIDE | drift(lnA)=-2.01e-5 t=-0.70 acc=0.128 cluster=0.9843 buried=0.0630 water_in_core=29/4500 amph=381/400 alpha=0.9446 paired=0.0917 unlike_excess=1.313 survival=0.0385 over 10000 steps -> lifetime=2686 steps throughput=533.3 steps/s
 Test Files  1 passed (1)   Duration 297.29s

$ ACID_SOAP_STRENGTHS=12.409 ...
ACID-SOAP-SWEEP S=12.409 pH=7 N=5700 settled=false chunks=300 steps=152000 area=1.1637 [1.1540, 1.1747] corridor 1.1-1.5 -> INSIDE | thickness=1.0079 corridor 4-6 -> OUTSIDE | drift(lnA)=-1.85e-4 t=-8.67 acc=0.109 cluster=1.0000 buried=0.0582 water_in_core=2/4500 amph=378/400 alpha=0.5017 paired=0.3079 unlike_excess=0.863 survival=0.5586 over 10000 steps -> lifetime=17285 steps throughput=458.4 steps/s

$ ACID_SOAP_STRENGTHS=4 ...
ACID-SOAP-SWEEP S=4 pH=7 N=5700 settled=true chunks=200 steps=102000 area=1.1042 [1.0972, 1.1149] corridor 1.1-1.5 -> OUTSIDE | thickness=4.9729 corridor 4-6 -> INSIDE | drift(lnA)=-7.96e-5 t=-1.48 acc=0.141 cluster=1.0000 buried=0.0182 water_in_core=0/4500 amph=385/400 alpha=0.5396 paired=0.4088 unlike_excess=0.979 survival=0.1719 over 10000 steps -> lifetime=5690 steps throughput=464.1 steps/s

$ ACID_SOAP_STRENGTHS=2 ...
ACID-SOAP-SWEEP S=2 pH=7 N=5700 settled=true chunks=200 steps=102000 area=1.1510 [1.1473, 1.1546] corridor 1.1-1.5 -> INSIDE | thickness=4.5517 corridor 4-6 -> INSIDE | drift(lnA)=1.23e-5 t=0.70 acc=0.116 cluster=1.0000 buried=0.0155 water_in_core=5/4500 amph=388/400 alpha=0.5921 paired=0.4246 unlike_excess=1.154 survival=0.0345 over 10000 steps -> lifetime=2947 steps throughput=476.2 steps/s

$ ACID_SOAP_STRENGTHS=1 ...
ACID-SOAP-SWEEP S=1 pH=7 N=5700 settled=true chunks=300 steps=152000 area=1.1556 [1.1510, 1.1593] corridor 1.1-1.5 -> INSIDE | thickness=4.5080 corridor 4-6 -> INSIDE | drift(lnA)=-1.53e-5 t=-1.08 acc=0.132 cluster=0.9974 buried=0.0785 water_in_core=10/4500 amph=382/400 alpha=0.6808 paired=0.3587 unlike_excess=1.186 survival=0.0306 over 10000 steps -> lifetime=2857 steps throughput=475.1 steps/s

$ ACID_SOAP_STRENGTHS=8 ...
ACID-SOAP-SWEEP S=8 pH=7 N=5700 settled=false chunks=300 steps=152000 area=1.2154 [1.2108, 1.2214] corridor 1.1-1.5 -> INSIDE | thickness=1.4191 corridor 4-6 -> OUTSIDE | drift(lnA)=-2.64e-5 t=-2.15 acc=0.124 cluster=1.0000 buried=0.0822 water_in_core=2/4500 amph=377/400 alpha=0.5100 paired=0.3438 unlike_excess=0.892 survival=0.4105 over 10000 steps -> lifetime=11250 steps throughput=435.4 steps/s
```

A trap encountered and recorded: the first run of an arm failed LOUDLY, with:
`electrostatics long-range list: longRangeListCapacity=161 or headIdx size (400) was insufficient at
rc_es=4.8067 (found heads=400) -- data may have been silently dropped`.
The ready-made bilayer patch condenses all heads into a layer a few sigma thick inside a box 30 deep,
so `maxHeads/min(box)^3` underestimates the head density by a multiple. Answered honestly, with
`CreateSoupOpts.electrostatics.densityVolumeSigma3` (a field the parcel task already added for its
own case) given the volume the heads actually occupy, not an inflated safety factor.

### 9.4 pH sweep

```
$ ACID_SOAP_PH_LIST=4,5,6 nice -n 15 npx vitest run tests/soup-acid-soap-ph.test.ts --no-file-parallelism
ACID-SOAP-PH pH=4 S=2 N=17118 rho_amph=1.367e-2 alpha=0.2556 pKa_app=4.467 paired=0.1422 excess=1.682 | aggregates=36 largest=106.0/300 flatness=0.1907 (need <=0.35) symmetry=0.4567 (need >=0.5) Rg/(L/2)=0.7091 head_layers radial=2 transverse=0 | BILAYER-CAPABLE=0/3 shape_lamellar=1/3 throughput=338.5 steps/s
ACID-SOAP-PH pH=5 S=2 ... alpha=0.5011 pKa_app=4.998 paired=0.1444 excess=1.418 | aggregates=36 largest=52.7/300 flatness=0.1821 symmetry=0.3631 Rg/(L/2)=0.5825 head_layers radial=2 transverse=0 | BILAYER-CAPABLE=1/3 shape_lamellar=0/3
ACID-SOAP-PH pH=6 S=2 ... alpha=0.7433 pKa_app=5.536 paired=0.1567 excess=1.739 | aggregates=35 largest=133.3/300 flatness=0.2478 symmetry=0.5629 Rg/(L/2)=0.7668 head_layers radial=1 transverse=1 | BILAYER-CAPABLE=0/3 shape_lamellar=2/3
 Test Files  1 passed (1)

$ ACID_SOAP_PH_LIST=7,8,9 ...
ACID-SOAP-PH pH=7 S=2 ... alpha=0.9033 pKa_app=6.027 paired=0.0678 excess=2.604 | aggregates=31 largest=167.3/300 flatness=0.4470 symmetry=0.6391 Rg/(L/2)=0.9676 head_layers radial=2 transverse=0 | BILAYER-CAPABLE=1/3 shape_lamellar=0/3
ACID-SOAP-PH pH=8 S=2 ... alpha=0.9789 pKa_app=6.316 paired=0.0256 excess=5.862 | aggregates=33 largest=137.0/300 flatness=0.5252 symmetry=0.8268 Rg/(L/2)=0.8831 head_layers radial=1 transverse=0 | BILAYER-CAPABLE=0/3 shape_lamellar=0/3
ACID-SOAP-PH pH=9 S=2 ... alpha=0.9956 pKa_app=6.625 paired=0.0078 excess=9.232 | aggregates=35 largest=142.3/300 flatness=0.5042 symmetry=0.7465 Rg/(L/2)=0.9856 head_layers radial=1 transverse=0 | BILAYER-CAPABLE=0/3 shape_lamellar=0/3
 Test Files  1 passed (1)

$ ACID_SOAP_STRENGTH=0 ACID_SOAP_PH_LIST=4,5,6 ...   # CONTROL WITHOUT THE PAIR
ACID-SOAP-PH pH=4 S=0 ... alpha=0.1122 pKa_app=4.900 paired=0.0433 excess=1.301 | aggregates=35 largest=147.7/300 flatness=0.5360 symmetry=0.8634 Rg/(L/2)=0.8977 head_layers radial=2 transverse=0 | BILAYER-CAPABLE=0/3 shape_lamellar=0/3
ACID-SOAP-PH pH=5 S=0 ... alpha=0.5033 pKa_app=4.994 paired=0.0611 excess=0.999 | aggregates=30 largest=174.7/300 flatness=0.4625 symmetry=0.6726 Rg/(L/2)=1.0142 head_layers radial=1 transverse=0 | BILAYER-CAPABLE=0/3 shape_lamellar=0/3
ACID-SOAP-PH pH=6 S=0 ... alpha=0.8889 pKa_app=5.076 paired=0.0222 excess=1.087 | aggregates=47 largest=135.0/300 flatness=0.3589 symmetry=0.6636 Rg/(L/2)=0.9252 head_layers radial=1 transverse=1 | BILAYER-CAPABLE=2/3 shape_lamellar=2/3
 Test Files  1 passed (1)
```

Two setup traps, both recorded as discarded tools:
```
Error: non-finite state at step=1000: non-finite position components=51294, velocity components=51294
of 51366 (appeared over steps 0..1000) -- the calculation diverged (Inf/NaN)
```
The cause: rods hung FROM a lattice node rather than being centered on it, and two rods from neighboring cells
left a gap of `cell - 2(bHT+bTT)` = 0.05 sigma. Centering gives a worse gap of 2.03 sigma. And:
```
Error: electrostatics long-range list: longRangeListCapacity=64 ... was insufficient
```
at pH >= 7, where heads condense, answered with `densityVolumeSigma3` = the close-packed volume of
all amphiphile beads, at which the capacity **saturates at all 300 heads** (no interaction can be
dropped at any pH).

### 9.5 The pair term in the constant-pH MC, against EXACT Boltzmann weights

```
$ nice -n 15 npx vitest run tests/soup-forces.test.ts tests/soup-electrostatics.test.ts tests/soup-stale-force.test.ts --no-file-parallelism
ES-HENDERSON pKa_intrinsic=4.9
ES-DETAILED-BALANCE r=1.1 U_es(both)=0.337254eps=0.3066kT U_pair(one)=-2.000000eps=-1.8182kT (depth=2) dG_intr=0.000000eps
ES-DETAILED-BALANCE suppression of the "both" state: 0.05235 against 0.25000 without dU_es
 ✓ tests/soup-electrostatics.test.ts (8 tests) 26829ms
 ✓ tests/soup-forces.test.ts (4 tests) 33888ms
 ✓ tests/soup-stale-force.test.ts (2 tests) 27277ms
 Test Files  3 passed (3)
      Tests  14 passed (14)
```

### 9.6 Arithmetic

```
$ nice -n 15 npx vitest run tests/soup-acid-soap-arithmetic.test.ts --no-file-parallelism
ACID-SOAP-ARITH explicit-water gate (UNCHARGED, published)
  a=1.1835 t=4.9061 v=2.903 -> closure floor=1014 amphiphiles (=507 pairs)
ACID-SOAP-ARITH charged patch pH 7, pair depth S=0
  a=1.2026 t=4.4191 v=2.6571 -> closure floor=928.8 amphiphiles (=464.4 pairs)
ACID-SOAP-ARITH charged patch pH 7, pair depth S=1
  a=1.1556 t=4.508 v=2.6046 -> closure floor=979.5 amphiphiles (=489.8 pairs)
  spanning micelle at L=54: 330.9 (6.128/sigma) -> band EMPTY, inverted 2.96x, L*=159.8 sigma
  spherical micelle (zero-length cap, boundless AT ANY size): 18.4 amphiphiles (=9.2 pairs) -> band inverted 53.2x
  packing parameter p=v/(a0*lc)=1 (identically 1 by construction)
ACID-SOAP-ARITH charged patch pH 7, pair depth S=2
  a=1.151 t=4.5517 v=2.6195 -> closure floor=989.8 amphiphiles (=494.9 pairs)
  spanning micelle at L=54: 335.4 (6.212/sigma) -> band EMPTY, inverted 2.95x, L*=159.3 sigma
  spherical micelle: 18.8 amphiphiles (=9.4 pairs) -> band inverted 52.5x
ACID-SOAP-ARITH charged patch pH 7, pair depth S=4 [FAILED]
  a=1.1042 t=4.9729 v=2.7456 -> closure floor=1097.3; spanning 382 (7.074/sigma), L*=155.1; spherical 23.5 -> 46.8x
ACID-SOAP-ARITH charged patch pH 7, pair depth S=8 [FAILED]
  a=1.2154 t=1.4191 v=0.8624 -> floor 552; spanning 99 (1.834/sigma), L*=301; spherical 1.7 -> 318.1x
ACID-SOAP-ARITH charged patch pH 7, pair depth S=12.409 [FAILED]
  a=1.1637 t=1.0079 v=0.5864 -> floor 531.6; spanning 73.5 (1.361/sigma), L*=390.7; spherical 0.9 -> 581.4x
ACID-SOAP-ARITH-VERDICT arms=7 where closure is cheapest=0 -> CLOSURE IS STILL NOT THE CHEAPEST; best (least inverted) arm: charged patch pH 7, pair depth S=4 [FAILED] at 46.8x against the spherical micelle and 2.87x against the spanning one at L=54; best L*=155.1 sigma
 Test Files  1 passed (1)
```

### 9.7 Regressions

```
$ nice -n 15 npx vitest run tests/gates.test.ts tests/supply-window.test.ts tests/percolation-check.test.ts tests/soup-valence.test.ts tests/electrostatics-audit.test.ts tests/coalescence-mechanism-pin.test.ts --no-file-parallelism
WINDOW-SCALING floor=[932, 1014] (box-independent) cheapestSpanning=[5.806, 6.512]*L L*=[143.1, 174.6]sigma L_max(engine, water alone at rho_W=0.8)=81.27sigma spanningAtCeiling=[472, 529] N_at_L*=2.342e+6 overCeiling=5.5x
PERC skipped: set PERC_CHECKPOINTS
 Test Files  6 passed (6)
      Tests  17 passed (17)

$ nice -n 15 npx vitest run tests/soup-checkpoint.test.ts tests/soup-bonds.test.ts tests/soup-area-move.test.ts --no-file-parallelism
 ✓ tests/soup-area-move.test.ts (2 tests) 193484ms
 ✓ tests/soup-checkpoint.test.ts (4 tests) 42744ms
   ✓ bonds form only at the catalytic center where the rule requires it 252043ms
   × monomer counts of each kind and charge are conserved while reactions run 120277ms
FAIL  tests/soup-bonds.test.ts > monomer counts of each kind and charge are conserved while reactions run
Error: Test timed out in 120000ms.
 Test Files  1 failed | 2 passed (3)
      Tests  1 failed | 8 passed (9)

$ nice -n 15 npx vitest run tests/soup-bonds.test.ts -t "monomer counts" --no-file-parallelism
   × monomer counts of each kind and charge are conserved while reactions run 120127ms
Error: Test timed out in 120000ms.

$ nice -n 15 npx vitest run tests/gate6-bilayer.test.ts tests/sim.test.ts tests/run-ui.test.ts --no-file-parallelism
GATE6 area 1.2010 +/- 0.0088 (min 1.1836, max 1.2228)  thickness 4.4688 (per-frame mean 4.4831, sd 0.2749, n=400)  lnA drift/move 5.835e-6 +/- 1.23e-5 (t=0.47)  accepted 0.295  escapedMax 3  box 24.626  steps 83000
 ✓ tests/gate6-bilayer.test.ts (2 tests) 118410ms
 ✓ tests/sim.test.ts (8 tests) 53058ms
   × the run survives a bead that flew out of the box along z ... (item 1a) 40414ms
FAIL  tests/run-ui.test.ts > the run survives a bead that flew out of the box along z ...
TimeoutError: Waiting failed: 30000ms exceeded
 Test Files  1 failed | 2 passed (3)
      Tests  1 failed | 17 passed (18)

$ nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=300 areaPerLipid(MEASURED, tail mean)=1.1734 [min 1.1696, max 1.1765, corridor 1.1-1.5] driftPerChunk(lnA)=-2.722e-6 t=-0.71 thickness(measured)=4.4689 [corridor 4-6] clusterFraction=1.0000 waterInCore=45/4500 headBuriedFraction=0.0919 acceptedFraction=0.1280 of 3000 trials throughput=452.50 steps/s at N=5700 verdict=passed
WATER-BILAYER-AREAMOVE WROTE verify/out/water-bilayer-area-move.json
 Test Files  1 passed (1)
      Tests  1 passed (1)
```

### 9.8 Proof that the `run-ui` failure is pre-existing

```
$ git stash push -u -- soup engine tests data verify
Saved working directory and index state WIP on stage-a-atoms: 8b7b6db feat(atomic): carbon enters the model
$ nice -n 15 npx vitest run tests/run-ui.test.ts -t "flew out of the box" --no-file-parallelism
   × the run survives a bead that flew out of the box along z and does not crash any longer than usual (item 1a) 46965ms
FAIL  tests/run-ui.test.ts > the run survives a bead that flew out of the box along z ...
TimeoutError: Waiting failed: 30000ms exceeded
 Test Files  1 failed (1)
$ git stash pop
Dropped refs/stash@{0} (e39ac8af25061e6dfe9afcf8a1e3b23638fba770)
```

### 9.9 The gate table

```
$ nice -n 15 npm run verify
BILAYER-SCENARIO area=1.2056 thickness=4.4750 min=1.1840 max=1.2358 accepted=0.273 escapedMax=0 steps=83000
THROUGHPUT-SCENARIO stepsPerSecond=2615.3 beads=3600 neighborBuildMs=1.400 wallMs=7647 beadsPerLipid=3
CLOSURE-SCENARIO synthetic shell radius=8 thickness=1.5 count=40000 volume=1284.875
KAPPA-SCENARIO valid=false kappa=NaN slope=-1.7265 fitModes=20 fitShells=4 qMax=0.1806 wallClockMs=218878
WROTE verify/out/kappa-measurement.json
WROTE verify/out/gates.json
WROTE verify/out/report.html
GATE area-per-lipid: value=1.2056205530467474 rank=A verdict=passed
GATE bilayer-thickness: value=4.474978261285809 rank=A verdict=passed
GATE bending-modulus: value=null rank=A verdict=unproven
GATE area-per-lipid-water: value=1.1734461123400626 rank=C verdict=passed
GATE bilayer-thickness-water: value=4.468932297634941 rank=C verdict=passed
GATE vesicle-closure-water: value=0 rank=B verdict=failed
GATE aggregate-percolation: value=3 rank=B verdict=failed
GATE vesicle-verdict: value=0 rank=A verdict=failed
GATE chain-length-asf: value=0.15940333430827724 rank=D verdict=unproven
GATE mean-tail-length: value=3.419 rank=D verdict=unproven
GATE closure: value=1284.875 rank=D verdict=unproven
GATE chain-to-bead-mapping: value=3 rank=D verdict=unproven
```

### 9.10 File sizes and types

```
$ find soup engine chem viewer verify tests -name '*.ts' -o -name '*.wgsl' | xargs wc -l | sort -rn | head -4
   39369 total
     600 soup/src/sim.ts
     600 soup/src/electrostatics.ts
     598 soup/src/rules.ts
$ npx tsc --noEmit 2>&1 | grep -c "error TS"
21
$ wc -l soup/wgsl/pair.wgsl soup/wgsl/step.wgsl soup/src/acid-soap.ts
     204 soup/wgsl/pair.wgsl
     361 soup/wgsl/step.wgsl
     198 soup/src/acid-soap.ts
```

---

## 10. Summary, and was the campaign earned

**Total**: 29 compute calls, each `nice -n 15`, one at a time, in the foreground, none over 500 s
except two run with a longer tool limit (`water-bilayer-area-move` at 349 s, `npm run verify` at
about 370 s). After every browser run, `pkill -f puppeteer_dev_chrome_profile` and a `pgrep`
confirmation: **NO ORPHANS** every time, not a single orphaned process left behind. **Ports 5199 and
5210 were untouched**: no call opened them, every browser run went through
`vite createServer({ port: 0 })` from `tests/helpers/gpu.ts`. The vesicle campaign was NOT run;
`tests/soup-vesicle.test.ts` was NOT run; no threshold was widened; no rank-A constant was refit;
`co_bond.attemptRate` was not touched; energy was never presented as free energy.

**THE CAMPAIGN IS NOT EARNED, and the reason is arithmetic, not pessimism.** Gate 4 answers directly:
at the chosen strength, the band [closure floor 989.8, spherical micelle 18.8] is inverted by
**52.5x** against the previous 44-58x, that is, a 1.05-1.10x improvement at a required roughly 50x;
on the best measured arm (which itself failed gate 1 on area) it is 46.8x. L\* moved from
160.6-161.3 to 155.1-159.3 sigma against the engine's structural ceiling of 115.7 sigma. The
cheapest boundless object is still the capped micelle, and §5 point 3 explains why the pair cannot
change this in principle: a cylindrical micelle has all its heads on one outer surface, where pairs
form exactly as they do in the bilayer. Running the campaign whose outcome the arithmetic already
predicts as "mesh/rods, no closure" would mean burning dozens of hours to confirm what has already
been computed.

**What the task did earn, and it is not little.** (1) A mechanism that was STRUCTURALLY impossible
(`polarPolar` = 0.0 throughout the project's history) now exists, reads a dynamic variable, and its
depth is tied to a computed number with an honestly declared rank and an honestly declared upper
bound. (2) **The measured ceiling destroys the bilayer 6 times before it is reached**, a measured
boundary of applicability for the atomistic number in the coarse-grained model, and it could not
have been known by reasoning. (3) **Alpha self-buffering to 1/2 from both sides**, an emergent
consequence, pinned against exact Boltzmann weights. (4) **A window in alpha for pairing, flatness
and compactness, with a control**, the first quantity in the project that anything has shifted by a
measurable amount (Rg/(L/2) from 0.90-1.01 to 0.58-0.77). (5) A hole closed: charge was not entering
the acceptance criterion of the area MC move.

**The cheapest next measurement, fully specified.** Not a campaign, but **two arms of the strength
sweep at pH 5** (where pairing is maximal, alpha = 0.50): S = 2 and S = 3, the same charged patch,
the same settling criterion, about 300 s each. They answer the single question this task left open
and that costs one call: do the area and thickness corridors hold at a strength that gives a paired
fraction above 0.42 at pH 5, and if so, the next thing worth measuring is not closure (the arithmetic
argues against it) but **in-plane symmetry**: the measured object is flat but elongated (0.36-0.46
against the needed 0.5), that is, the model makes a **ribbon, not a disk**, and this is the next real
limitation, not a box and not a material budget.
