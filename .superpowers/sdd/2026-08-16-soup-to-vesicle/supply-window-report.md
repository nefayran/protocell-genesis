# The supply window: the band where a closed vesicle would win is EMPTY at every box this engine can run — and inside the band the hypothesis itself names, the object STILL wraps all three axes

Task `supply-window` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `long-range-electrostatics-report.md`, whose §9.2 refuted the pair-energy explanation of
percolation (closing the head-repulsion gap from 2.86× to 1.26× moved the wrapping-axis count by
exactly zero) and named what is left: **the AMOUNT of organic material**, ρ_org = 0.418.

`data/params.json` rank-A constants: **NOT touched.** `co_bond.attemptRate`: **NOT touched** (refused an
eighth time). `stageThresholds.enclosedVolume` = 370.8656: **read, not moved.** No threshold widened, no
corridor relaxed, no tolerance changed, no assertion weakened, no rank-A or rank-B constant re-fitted.
No arm reverted to under-dense water: the campaign runs at ρ_W = 125 971/157 464 = **0.80000**. Nothing
started from anything pre-made — step 0 is a monomers-only lattice, proved from the run's own first
checkpoint in §6.3. No fusion probe. `tests/soup-vesicle.test.ts` never run. **One** new source file, a
test, and it computes arithmetic; the engine is byte-identical to the predecessor's.

---

## 0. VERDICT UP FRONT, TEN LINES

1. **THE HYPOTHESIS'S ARITHMETIC IS WRONG IN ITS CHOICE OF COMPETITOR, AND THE CORRECT COMPETITOR MAKES
   THE BAND EMPTY.** The cheapest box-spanning, edge-free object this amphiphile can build is not a
   bilayer tube of radius ≈ 4 σ (≈ 2260 amphiphiles at L = 54) — it is a **cylindrical micelle of radius
   equal to the leaflet thickness**, which costs **314–358** amphiphiles at L = 54, i.e. **2.87–2.97×
   LESS than the closure floor** (§2). The band [floor, cheapest spanning] is therefore **empty at box
   54, inverted**, not narrow.
2. **AND THE BAND IS EMPTY AT EVERY BOX THIS ENGINE CAN RUN.** The spanning cost is linear in L
   (5.806–6.635 amphiphiles per σ of box side); the floor is box-independent (932–1030). The band opens
   only at **L\* = 140.4–177.3 σ**. The largest box the Verlet ceiling allows at liquid-water density is
   **81.27 σ**, where the cheapest spanning object costs 472–539 against a floor of 932 — still short by
   1.7–2.2×. L\* needs **2.214e6 particles = 5.2× over the hard ceiling of 429 496** (§3). *That is the
   general answer to "would a bigger box help": yes in principle, at 2.6–3.3× this box, which is 5.2×
   beyond what the engine can hold.*
3. **THE MEASURED ROD CONFIRMS THE IDENTIFICATION.** Inverting the coalescence report's own gyration
   numbers through the measured volume gives the 151-molecule rod a physical radius of **2.353–2.560 σ**
   against a leaflet thickness t/2 of **2.229–2.484 σ** — the thing this project has been measuring all
   along IS a cylindrical micelle, so it is a legal spanning object and it is the cheapest (§2.3).
4. **THE CAMPAIGN WAS RUN INSIDE THE BAND THE HYPOTHESIS ITSELF NAMES** ([932–1030, 2251–2308],
   midpoint 1630), because that band is non-empty and the discriminating measurement is cheap. Supply
   sized by scaling the last two campaigns' organics by **0.72** with O:C held at 0.33325 (§5).
5. **THE SUPPLY LANDED AT 1546.0 ± 3.7 (0.24 %)**, verified from the run's own trace, not assumed:
   1.50× the floor, 0.67× the band top, **0.68× the predecessor's 2271.6**, at ρ_org **0.30091** against
   0.41792 (§5.2).
6. **THE DISCRIMINATING PREDICTION FAILED. 3 of 3 axes, 19 of 19 slabs, at ALL EIGHT wet checkpoints
   over 99 000 settled steps.** The hypothesis predicted 0 of 3 for the first time. It did not happen
   (§7).
7. **`encapsulatedWater` IS REPORTABLE AGAIN — 0 against a threshold of 314.36 — AND THAT DOES NOT MEAN
   WHAT THE PREDECESSOR SAID IT MEANT.** The centre refusal is a **Rayleigh test** for circular
   uniformity per axis, R ≥ √(−ln 1e-6 / n); this object cleared a *stricter* bar (0.0366 at n = 10 305)
   than the predecessor's object failed (0.0318 at n = 13 690) while wrapping the same 3 axes. So
   reportability measures how uniformly the object fills the box, **not whether it wraps** (§8.2).
8. **NO VESICLE.** `hasVesicleAggregate` **false at all 12 checkpoints**, `closed` never true,
   `radialHeadShells` **1** at all 8 wet, `transverseHeadShells` **0**, cavity **37.03 ± 3.34 σ³**
   against 370.8656 (**10.0× short**), flatness **0.6452–0.7789** against ≤ 0.35 (§8).
9. **IT IS SETTLED THIS TIME, AND THAT IS NEW.** 99 000 settled wet steps (**2.02×** the predecessor's
   49 000); largest 1551 → 1546 → 1545 → 1542 (**0.24 %** over the last 42 000 steps); and the cavity
   **turned over** for the first time in this lineage (40.0 → 38.0), so it is a plateau and not an upper
   bound (§9).
10. **WHERE IT LANDS: 1030/1546 = 0.67× — the FOURTH consecutive landing off the end of the sequence**
    80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75× → 0.48× → 0.42× → **0.67×**. Supply has now been beaten
    at 2.36× over the floor and at 1.50× over it, with the wrapping-axis count **3** both times. **What
    this eliminates: the supply-budget explanation of percolation, in the form the hypothesis states
    it** (§10).

---

## 1. THE HYPOTHESIS, AND WHAT CHECKING IT MEANS

Stated by the customer: *in a fully periodic box, an aggregate that spans the box has no edge at all,
so it beats a closed vesicle, which must pay curvature. A closed vesicle is therefore the preferred
object only when the supply is ABOVE the closure floor and BELOW the cheapest spanning object's cost.*

The mechanism is sound and it is worth saying why, because it is the reason this is a *supply* question
and not an energy question. Compare the two edge-free objects at equal mid-surface radius R: a sphere
pays 8πκ of bending (2κ/R² per unit area), a cylinder pays κ/2R² per unit area — **four times less**.
Both pay zero edge. So a spanning tube is *always* per-area cheaper than a closed vesicle of the same
radius, and no amount of head repulsion changes that ordering. The only thing that can make a vesicle
the preferred object is that **there is not enough material to build the cheapest spanning object**.
That makes the hypothesis exactly a budget statement, and budget statements are arithmetic.

Which is why the arithmetic has to be checked rather than accepted, and there are two places it can go
wrong: the identity of the cheapest spanning object (§2), and whether the resulting band exists at any
runnable box (§3). Both went wrong.

The arithmetic is committed as **`tests/supply-window.test.ts`** — a real gate, no GPU, no env var, no
checkpoint: it reads the measured gate artifact and `data/soup.json`'s own threshold, recomputes the
whole window, and asserts each conclusion, so the window moves if and only if the gates move.

---

## 2. THE WINDOW ARITHMETIC, FROM THIS TASK'S OWN RE-MEASURED GATES

### 2.1 The two measured bases, re-measured today

```
$ nice -n 15 npx vitest run tests/gate6-bilayer.test.ts tests/water-bilayer-area-move.test.ts --no-file-parallelism
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=300 areaPerLipid(MEASURED, tail mean)=1.1763
  [min 1.1712, max 1.1806, corridor 1.1-1.5] driftPerChunk(lnA)=-1.700e-5 t=-1.55
  thickness(measured)=4.9686 [corridor 4-6] clusterFraction=1.0000 waterInCore=57/4500
  headBuriedFraction=0.0964 acceptedFraction=0.1243 of 3000 trials throughput=1201.93 steps/s
  at N=5700 verdict=passed
GATE6 area 1.2060 +/- 0.0110 (min 1.1823, max 1.2303)  thickness 4.4580 (per-frame mean 4.4704,
  sd 0.2442, n=400)  lnA drift/move 4.197e-5 +/- 1.35e-5 (t=3.11)  accepted 0.307  escapedMax 0
  box 24.538  steps 83000
```

| basis | a σ² | t σ | v = a·t/2 σ³ |
|---|---|---|---|
| **explicit-water gate** (rank C, ρ_W = 0.8) | **1.1763** [1.1712, 1.1806] | **4.9686** | 2.9222 |
| **solvent-free gate 6** (rank A corridor) | **1.2060** ± 0.0110 | **4.4580** | 2.6882 |

The two thicknesses disagree by 11.4 %, which is wider than the predecessor's 5.6 %, and that spread is
carried through everything below rather than averaged away.

### 2.2 The closure floor — same formula as `continuous-run-report.md` §1.1, no new constant

```
R_in  = (3 * 370.8656 / 4pi)^(1/3) = 4.45700 sigma        (stageThresholds.enclosedVolume, untouched)
R_mid = R_in + t/2 ;  floor = 2 * 4pi * R_mid^2 / a
```

| basis | t/2 | R_mid | **floor** |
|---|---|---|---|
| explicit-water gate | 2.4843 | 6.9413 | **1029.5 → 1030** |
| solvent-free gate 6 | 2.2290 | 6.6860 | **931.6 → 932** |

**Floor band 932–1030 (box-independent).** Trend across ten tasks: 960 → 912 → 968 → 1088 → 920 → 998
→ 1004 → 947 → 953 → **1030**.

### 2.3 The spanning objects, each costed from the SAME two measured numbers

An amphiphile occupies v = a·t/2 (a leaflet is t/2 thick and each lipid holds a of its face). Costing by
**volume** where the surface is curved avoids having to guess the splayed area-per-head of a curved
monolayer, which is the one number nothing here measures.

| candidate | geometry | cost at L = 54, explicit basis | gate-6 basis |
|---|---|---|---|
| **(1) cylindrical micelle**, one head surface, tail core radius R = t/2 | N = πR²L/v = **πtL/2a** | **358.3** | **313.5** |
| (2) bilayer tube, mid radius R = 4 σ (*the hypothesis's own competitor*) | N = 2·2πRL/a | 2307.5 | 2250.7 |
| (2′) bilayer tube, mid radius R = t | " | 2866.3 | 2508.4 |
| (2″) bilayer tube at the vesicle's own R_mid | " | 4004.3 | 3762.0 |
| (3) flat bilayer sheet (wraps **2** axes, not 3) | N = 2L²/a | 4958.0 | 4835.8 |

**(1) is the cheapest, by 6.3–7.2×, on both bases** — asserted in the test, not eyeballed. The reason is
structural: a micelle uses **one** splayed surface where a tube uses **two**, and its radius is capped
at the leaflet thickness because a tail cannot reach further than it reaches inside a bilayer.

**Is (1) a real object for THIS molecule, or a geometric fiction?** It is real, and this project has
been measuring it for two tasks. `coalescence-report.md` §4.1 reports rods of fixed cross-section:
N = 151, cross-sectional gyration radius R⊥ = 4.006–4.165 σ, long-axis √λ₃ = 6.188–6.739. A gyration
radius is not a physical radius, so invert it properly: for a uniform cylinder λ₃ = ℓ²/12, so
ℓ = √12·√λ₃ = 21.44–23.35 σ, and the measured volume N·v = πR²ℓ gives

| basis | rod radius from its own volume | leaflet thickness t/2 |
|---|---|---|
| explicit-water | **2.453–2.560 σ** | 2.484 |
| gate 6 | **2.353–2.455 σ** | 2.229 |

The two agree to within 0.1–0.23 σ on both bases. **The measured rod is a cylindrical micelle of radius
equal to the leaflet thickness** — candidate (1) with the numbers filled in. (And this also explains the
inflated R⊥: 4.0 σ of cross-sectional gyration would need a uniform radius of 5.7 σ, which at the
measured volume would put the rod's interior at bead density 0.36 — a third of a condensed phase. The
rod is a spherocylinder with diffuse ends, not a uniform cylinder, so R⊥ overstates its radius. Reading
R⊥ as a radius is what makes a ≈ 4 σ spanning tube look like the cheapest competitor.)

### 2.4 The band, with the uncertainty carried

```
WINDOW basis=explicit-water gate a=1.1763 t=4.9686 v=2.9222 floor=1029.5 rodR(measured)=2.560..2.453 leafletT/2=2.484
WINDOW-SPAN basis=explicit-water gate L=54 micelle(R=t/2)=358.3 perSigma=6.635 tube(R=4)=2307.5 tube(R=t)=2866.3 tube(R=R_mid_vesicle)=4004.3 sheet=4958.0
WINDOW-BAND basis=explicit-water gate band=[1029, 358] empty=true inverted_by=2.87x L*=155.2sigma (band exists only for L >= L*)
WINDOW basis=solvent-free gate 6 a=1.2060 t=4.4580 v=2.6882 floor=931.6 rodR(measured)=2.455..2.353 leafletT/2=2.229
WINDOW-SPAN basis=solvent-free gate 6 L=54 micelle(R=t/2)=313.5 perSigma=5.806 tube(R=4)=2250.7 tube(R=t)=2508.4 tube(R=R_mid_vesicle)=3762.0 sheet=4835.8
WINDOW-BAND basis=solvent-free gate 6 band=[932, 314] empty=true inverted_by=2.97x L*=160.4sigma (band exists only for L >= L*)
```

**At box 54 the band is EMPTY on both bases, and it is not marginal — it is inverted by 2.87–2.97×.**
Any supply large enough to close a vesicle (≥ 932) is already 2.6–3.3× more than a spanning cylindrical
micelle costs (314–358). A vesicle is therefore never the budget-preferred object at this box, and the
near-coincidence the brief noticed (2·2πRL/a ≈ 2260 against a measured 2254) is **accidental**: it
compares the supply against a competitor 6.4× more expensive than the cheapest one.

---

## 3. THE BOX-SIZE SCALING — THE GENERAL ANSWER TO "WOULD A BIGGER BOX HELP"

```
WINDOW-SCALING floor=[932, 1029] (box-independent) cheapestSpanning=[5.806, 6.635]*L
  L*=[140.4, 177.3]sigma L_max(engine, water alone at rho_W=0.8)=81.27sigma
  spanningAtCeiling=[472, 539] N_at_L*=2.214e+6 overCeiling=5.2x
```

- **The floor does not scale with the box.** It is fixed by `stageThresholds.enclosedVolume` and the
  measured bilayer geometry: 932–1030 at every box.
- **Every spanning cost scales as L¹** (the micelle and the tube) or **L²** (the sheet). The cheapest is
  **(5.806–6.635)·L** amphiphiles.
- **The band exists iff L ≥ L\* = floor/(cheapest per σ) = 140.4–177.3 σ**, central estimate 156 σ =
  **2.9× the current box**.
- **The engine cannot go there.** The Verlet list is N·`listCapacity`(2500)·4 bytes against 4 294 967 292
  → hard max **429 496 particles**. Water alone at the measured liquid density 0.8 σ⁻³ caps the box at
  (429 496/0.8)^(1/3) = **81.27 σ**. At L\* the water alone would need **2.214e6 particles, 5.2× over**.
- **And the ceiling box is not close enough to matter:** at L = 81.27 the cheapest spanning object costs
  472–539 against a floor of 932–1030 — still short by **1.7–2.2×**. Halving the box's water would not
  help either: the floor is set by an *instrument threshold*, not by the box.

**So the hypothesis, in its strict and correct form, is not testable by simulation in this engine — not
because the physics is unclear but because the box it needs is 5.2× beyond the memory ceiling. That is a
computable statement and it is the honest headline of the arithmetic.** Two ways it could ever be
opened, both stated as consequences and neither taken here: raise the closure threshold's volume (a
threshold change — refused), or make the amphiphile fatter so a micelle stops being cheap (a molecule
change, requirement 1 of `final-campaign-report.md`).

---

## 4. WHAT WAS RUN INSTEAD, AND WHY IT IS STILL THE RIGHT TEST

The band the hypothesis *itself* names is non-empty, because its competitor is the ≈ 4 σ bilayer tube:

```
WINDOW-HYPOTHESIS-BAND L=54 band=[932..1029, 2251..2308] midpoint=1630
```

That band is worth one campaign for three reasons. (i) It is the hypothesis as stated, and the brief
asks for it to be *tested*, not only recomputed. (ii) Its discriminating measurement is free — the
axis-wrapping instrument already exists and runs off-GPU on checkpoints. (iii) It is the only supply
band anyone can actually enter at this box, so a failure inside it is a strictly stronger result than a
failure outside it: if the object percolates even at a supply that cannot afford the hypothesis's own
competitor, the supply-budget explanation is finished at every band width one could argue for.

Note what the empirical record already brackets. `dec54` (box 54, liquid water, one drying event, the
short-tail O:C = 4 line) carried **1423–1607** amphiphiles and wrapped **0 of 3 axes** with a largest
aggregate of 151; the last two campaigns carried **2087–2272** on the arm-B two-tailed line and wrapped
**3 of 3**. Those two facts are often read as a supply threshold between 1607 and 2087. They are not
comparable that way — the composition differs — and the run below is the controlled version: **the
arm-B line, everything else identical, supply alone reduced.**

---

## 5. HOW THE SUPPLY WAS SIZED, AND WHERE IT ACTUALLY LANDED

### 5.1 Sizing — one scalar on the organics, O:C held fixed

Supply is emergent here: the CLI takes monomer counts, and how many amphiphiles the chemistry makes is
an outcome. The knob chosen is **a single multiplier f on every organic species**, which holds O:C — and
therefore the *molecule* (tail length is set by O:C, `tail-length-and-window-report.md`) — fixed and
changes only the concentration. Cutting O alone would have made a different molecule.

Calibration, from the last campaign's own numbers: 2271.6 amphiphiles from **9296** heads, a
head→amphiphile yield of **0.2444**; at box 30 the same composition density gave 412 of 1594 = 0.2585,
so the yield is set by density and not by box. Target = the band midpoint **1630** → f = 1630/2271.6 =
**0.7176 → 0.72**.

| | last two campaigns (arm B) | **this campaign, f = 0.72** |
|---|---|---|
| `--start` | C 27894 / O 9296 / H 27894 / M 723 / W 125971 | **C 20084 / O 6693 / H 20084 / M 521 / W 125971** |
| O:C | 0.33326 | **0.33325** |
| N | 191 778 | **173 353** |
| ρ_W | 0.79999 | **0.80000** |
| ρ_org | 0.41792 | **0.30091** |
| ρ_tot | 1.21792 | 1.10091 |
| dry box (computeDryBox, ρ = 1.34) | 36.6370 | **32.8431** |
| main Verlet list | 1.9178 GB | **1.7335 GB** of 4.294967292 |
| particles | 191 778 | **173 353** of 429 496 |

**f = 0.72 was also chosen over f = 0.70 for an electrostatics reason, stated before running.** The
charge's range is 4 λ_D capped by the minimum-image ceiling 0.45·min(box), and min(box) is the *dry*
box, which shrinks when there is less organic material. f = 0.70 would have given L_dry =
32.53 → rc_es = 14.64 σ = 3.853 λ_D; f = 0.72 gives L_dry = 32.8431 → **rc_es = 14.7794 σ = 3.889 λ_D**,
which keeps the range closest to the calibrated 4.000 λ_D that still fits. **The rule is unchanged and
no constant was re-fitted** — the cap is the rule's own ceiling doing its job, printed in every progress
line. Its price, quantified from the predecessor's own convergence curve (0.7162 kT at 3.42 λ_D →
0.7215 at 4.00): at 3.889 λ_D the head–head contact repulsion is **0.7205 kT, 0.14 % below** the value
the calibration certified, and the discarded integrated fraction is **0.1000** against 0.0916.

**pH 7.0, I = 0.01 M — identical to the predecessor, deliberately.** Justification: it is the
strongest-repulsion arm available (10 mM is the long Debye length; pH 7 sits 0.5 units above the measured
apparent pKa, so α = 0.765–0.780 of heads carry −1), *and* keeping it identical makes **supply the only
changed variable in the whole configuration**, which is what the comparison needs.

### 5.2 Where it landed, from the run's own trace

| wet checkpoint | 45 400 | 60 400 | 74 400 | 88 400 | 102 400 | 116 400 | 130 400 | 144 400 |
|---|---|---|---|---|---|---|---|---|
| `amphiphileCount` | 1596 | 1568 | 1565 | 1555 | 1551 | 1546 | 1545 | **1542** |

**Plateau over the last four: 1546.0 ± 3.7 (0.24 %).** Predicted 1636, landed 1546 — **5.5 % low**,
because the yield did drop slightly with density (1542/6693 = **0.2304** against arm B's 0.2444). Not
tuned, not re-run: the landing is inside the band and that is all the sizing had to achieve.

| against | value | verdict |
|---|---|---|
| closure floor 932–1030 | **1.50–1.66× OVER** | affordable: a vesicle is buildable |
| hypothesis band top 2251–2308 | **0.67× of it** | **inside the band** |
| band midpoint 1630 | 0.95× | mid-band |
| **cheapest spanning object 314–358 (§2)** | **4.3–4.9× OVER** | the strict arithmetic says spanning is still affordable |
| predecessor's supply 2271.6 | **0.68×** | |

---

## 6. THE CAMPAIGN

### 6.1 Configuration

Box **54 σ**, composition above, seed 19, kT 1.1, `--relax --cycle --evaporate --cycles 1` (**one drying
event**), real RNG, chunked and checkpointed, `clay: false`, charge on at **pH 7.0 / I = 0.01 M**,
**rc_es = 14.7794 σ = 3.889 λ_D**, discarded 0.1000 — printed in every progress line. Cold start:
`max|F| 1.4598e+4 → 1.7684e+1`, non-finite 0 before and after.

**Run length: 144 400 steps, of which 99 000 are settled wet steps after the rehydration completed at
step 45 400** — against the predecessor's 90 600 / 49 000. **2.02× the settled length.**

### 6.2 The stage trace, audited off-GPU with the run's own functions

`verify/out/supply-window-campaign-B54-trace.json` (copied verbatim to
`verify/out/gates-campaign-trace.json`, which is the path `verify/campaign-gates.ts` reads), 12
checkpoints, `tests/continuous-run-audit.test.ts`. *(Italic rows are dry-phase, one contact-percolating
mass by construction; no structural claim rests on them.)*

| step | box | stage | amph | perTail | 2-tail | bonds | aggs | **largest** | r_g | flat | inPl | **radSh** | cav σ³ | **encH₂O** | closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 000 | 54.0000 | **monomers** | 123 | 2.183 | 3 | 712 | **111** | **2** | 1.55 | 0.142 | 0.631 | n/a | 0 | n/a | false |
| 6 000 | 54.0000 | monomers | 191 | 2.410 | 26 | 1366 | 137 | 3 | 2.15 | 0.186 | 0.256 | n/a | 0 | 0/296.7 | false |
| *18 400* | *32.8431* | *amphiphiles* | *1240* | *2.957* | *418* | *6419* | *1* | *1240* | *16.62* | *0.886* | *0.954* | *1* | *10.6* | *centre-untrusted* | — |
| *24 000* | *32.8431* | *amphiphiles* | *1480* | *2.899* | *575* | *8233* | *1* | *1480* | *16.63* | *0.934* | *0.995* | *1* | *16.8* | *centre-untrusted* | — |
| **45 400** | 54.0000 | amphiphiles | 1596 | 2.952 | 874 | 11529 | **1** | **1596** | **27.186** | 0.7789 | 0.9104 | **1** | 18.9 | **0 / 311.9** | false |
| **60 400** | 54.0000 | amphiphiles | 1568 | 3.050 | 946 | 12670 | **2** | **1567** | **27.072** | 0.7045 | 0.9213 | **1** | 22.9 | **0 / 312.5** | false |
| **74 400** | 54.0000 | amphiphiles | 1565 | 3.142 | 978 | 13433 | **1** | **1565** | **26.966** | 0.6683 | 0.8986 | **1** | 27.8 | **0 / 313.0** | false |
| **88 400** | 54.0000 | amphiphiles | 1555 | 3.224 | 1005 | 14102 | **2** | **1554** | **27.051** | 0.6513 | 0.8873 | **1** | 29.6 | **0 / 313.5** | false |
| **102 400** | 54.0000 | amphiphiles | 1551 | 3.294 | 1032 | 14614 | **1** | **1551** | **27.087** | 0.6656 | 0.8924 | **1** | 32.3 | **0 / 313.9** | false |
| **116 400** | 54.0000 | amphiphiles | 1546 | 3.327 | 1036 | 15020 | **2** | **1545** | **27.049** | 0.6494 | 0.8803 | **1** | 37.9 | **0 / 314.1** | false |
| **130 400** | 54.0000 | amphiphiles | 1545 | 3.357 | 1037 | 15365 | **1** | **1545** | **27.149** | 0.6557 | 0.8839 | **1** | 40.0 | **0 / 314.2** | false |
| **144 400** | 54.0000 | amphiphiles | **1542** | **3.397** | **1038** | **15649** | **1** | **1542** | **27.118** | **0.6452** | **0.8785** | **1** | **38.0** | **0 / 314.4** | **false** |

### 6.3 Monomers-only start, proved from the run's own first checkpoint

`swB54-step3000.json`: `stage=monomers`, **111 aggregates whose largest holds 2 amphiphiles**, 123
amphiphiles of 6693 possible heads, `qualifyingAggregateCount` **0**, `bondSlotsUsed` 1424, events
`{cc_bond:584, cc_break:0, co_bond:128, co_break:0}`, `largestAggregateFraction` 0.0226. **Nothing
pre-made entered this run.**

**Invariants, every checkpoint, asserted not eyeballed:** non-finite positions **0**, velocities **0**,
all six valence counters **0** across all 12; census exactly `{C:20084, O:6693, H:20084, M:521,
W:125971}` at the ten wet-box checkpoints and the same organics with `W:90` at the two dry ones;
`RUN-AUDIT-TETHER violations=0`. Final events `{cc_bond:11926, cc_break:0, co_bond:3885, co_break:162}`.
Final per-tail histogram `{2:1359, 3:475, 4:249, 5:170, 6:108, 7:60, 8:55, 9:26, 10:17, 11:11, 12:15,
13:7, 14:6, 15:4, 16:6, 17:4, 18:2, 19:3, 22:2, 25:1}`.

---

## 7. THE DISCRIMINATING MEASUREMENT: AXIS WRAPPING — THE PREDICTION FAILED

**The hypothesis predicted 0 of 3 axes, for the first time in this project. Measured: 3 of 3, at every
one of eight wet checkpoints, 19 of 19 slabs on every axis.**

```
PERC {"file":"data/checkpoints/swB54/swB54-step45400.json","step":45400,"box":54,"aggregates":1,"amphiphilesInLargest":1596,"particlesInLargest":8887,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/swB54/swB54-step60400.json","step":60400,"box":54,"aggregates":2,"amphiphilesInLargest":1567,"particlesInLargest":9232,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/swB54/swB54-step74400.json","step":74400,"box":54,"aggregates":1,"amphiphilesInLargest":1565,"particlesInLargest":9554,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/swB54/swB54-step88400.json","step":88400,"box":54,"aggregates":2,"amphiphilesInLargest":1554,"particlesInLargest":9805,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/swB54/swB54-step102400.json","step":102400,"box":54,"aggregates":1,"amphiphilesInLargest":1551,"particlesInLargest":10059,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/swB54/swB54-step116400.json","step":116400,"box":54,"aggregates":2,"amphiphilesInLargest":1545,"particlesInLargest":10134,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/swB54/swB54-step130400.json","step":130400,"box":54,"aggregates":1,"amphiphilesInLargest":1545,"particlesInLargest":10212,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/swB54/swB54-step144400.json","step":144400,"box":54,"aggregates":1,"amphiphilesInLargest":1542,"particlesInLargest":10305,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"campaign"}
PERC {"file":"data/checkpoints/dec54/dec54-step41400.json","step":41400,"box":54,"aggregates":22,"amphiphilesInLargest":200,"particlesInLargest":704,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[12,13,12,19],"role":"control"}
PERC {"file":"data/checkpoints/dec54/dec54-step174400.json","step":174400,"box":54,"aggregates":25,"amphiphilesInLargest":151,"particlesInLargest":596,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[7,7,10,19],"role":"control"}
PERC {"file":"data/checkpoints/lrB54pH70/lrB54pH70-step90600.json","step":90600,"box":54,"aggregates":2,"amphiphilesInLargest":2223,"particlesInLargest":13690,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"control"}
```

- **The must-say-no controls still say no**, same day, same code: `dec54`'s 200- and 151-molecule rods at
  **0 axes**, 7–13 of 19 slabs. The instrument is not stuck on "true".
- **The predecessor's own final checkpoint is carried as a positive control** and still reads 3 of 3.
- **The rows carry `role: "campaign"` for `swB54`**, set through `PERC_CAMPAIGN_LABEL`, so
  `verify/campaign-gates.ts` reads *this* campaign's rows — the hardcoded-label defect found once here
  cannot recur silently, and §12's gate table proves it read them.
- r_g **26.966–27.186** against **27.000 for a uniformly filled 54 σ box (99.9–100.7 %)**, i.e. the
  object is as box-filling as the predecessor's at **0.68× the material**.

---

## 8. THE LARGEST AGGREGATE, AND CLOSURE, AGAINST THE SAME PRE-FIXED CRITERIA

### 8.1 The table

| quantity | threshold | **measured, reduced supply** | predecessor (long-range, 2254) | verdict |
|---|---|---|---|---|
| **wrapping axes** | **0** for a finite object | **3 of 3, 8 of 8 wet** | 3 of 3, 5 of 5 | **fails — the hypothesis's prediction** |
| aggregate size | ≥ 932–1030 (this task's floor) | **1546.0 ± 3.7** | 2254.3 | **passes, 1.50× over** |
| **`encapsulatedWater`** | ≥ ~312–314 (live bulk density × closure volume) | **0 at all 8 wet — REPORTABLE** | `null(centre-untrusted)` | **fails, but the instrument answers** |
| **`closed`** | true | **false at all 8**; `hasVesicleAggregate` **false at all 12** | false | fails |
| **`radialHeadShells`** | **2** for a bilayer wall | **1 at all 8 wet** | 1 at all 5 | fails |
| `transverseHeadShells` | 2 | **0** throughout | 0 | fails |
| enclosed / cavity volume | ≥ 370.8656 σ³ | **37.03 ± 3.34**, final 38.0, **turned over** | 36.2 ± 6.7, still rising | fails by **10.0×** |
| flatness λ₁/λ₃ | ≤ 0.35 | **0.6452–0.7789** (final 0.6452) | 0.7313–0.7668 | **fails, 1.84× over** |
| in-plane symmetry λ₂/λ₃ | ≥ 0.50 | **0.8785–0.9213** | 0.8667–0.9050 | passes |
| `hasLamellarAggregate` | true | **false, all 12** | false | fails |

**Aggregate-size distribution.** `sizeHistogramTop` is `[1596]`, `[1567,1]`, `[1565]`, `[1554,1]`,
`[1551]`, `[1545,1]`, `[1545]`, `[1542]` — **one or two objects, and the second is always a single free
amphiphile**. `qualifyingAggregateCount` **1** at all eight, `amphiphileShareInQualifying`
**0.9994–1.0000**. There is no distribution to speak of: one object holds everything, exactly as at
2254. Reducing the supply by 32 % did not fragment it.

`tests/electrostatics-audit.test.ts` on the same eight checkpoints: **merges 0, fissions 0,
`exchangedFraction` 0.000000 at all seven intervals**, `freeAmphiphileFraction` **0 … 6.5e-4** (against
the predecessor's 4.3e-4 … 8.8e-4), `largestCovalentComponent` **18–28 beads of 8887–10 305**
(`covalentSpanFraction` **0.00203–0.00286**) — the percolating object remains a **contact** network of
1542 chemically separate molecules whose largest covalent piece is 28 beads. Charge behaved as
calibrated: α 0.7645–0.7796, pKa_app 6.451–6.489, `unlikeExcessRatio` 1.236–1.415.

### 8.2 The closure instrument CAN report again — and that does NOT mean what the predecessor said

The brief expected this, and it happened: `encapsulatedWater` is a number at every wet checkpoint
(0 of 311.9–314.4, `closed:false`, `bulkWaterDensity` 0.8410–0.8476, `seedDistFromCentre` 46.53–46.72,
`totalEmptyCells` ≈ 1.19e6, `unreachedCells` 188–371), where the predecessor's was
`null(centre-untrusted)` at all five. **But the object still wraps all three axes**, so reportability
cannot be the wrapping verdict "spoken by the closure instrument".

What the refusal actually is, read from `soup/src/water-closure.ts`: a **Rayleigh test for circular
uniformity, per axis** — `trusted = R ≥ √(−ln α / n)` with α = 1e-6 (`data/periodic-measurement.json`,
whose own basis explains why a fixed R bar was rejected). The bar is **stricter for fewer points**:

| | n (particles in largest) | bar R_min | outcome |
|---|---|---|---|
| predecessor, 2254 amphiphiles | 13 690 | **0.0318** | refused on at least one axis |
| **this run, 1546 amphiphiles** | **10 305** | **0.0366** | **cleared on all three** |

So this object cleared a **harder** bar: with 32 % less material spread over the same box, each axis'
mass distribution is measurably less uniform, enough for a circular mean to be significant — while the
contact network still spans. **Correction to the predecessor's §7.3 reading: `centre-untrusted` measures
how uniformly an object fills the box, not whether it wraps, and the two can and here do disagree.** The
`0 of 314` is a real measurement, but the centre it floods from clears its significance bar by a small
margin, and that is the honest caveat on the number.

---

## 9. SETTLING — A PLATEAU THIS TIME, NOT AN UPPER BOUND

The predecessor's own flagged weakness was 49 000 settled steps with the cavity still rising. Fixed:

| quantity | over the last four wet samples (42 000 steps) | over all eight (99 000 steps) |
|---|---|---|
| largest aggregate | 1551, 1546, 1545, 1542 → **1546.0 ± 3.7 (0.24 %)** | 1553.9 ± 17.0 (1.09 %) |
| cavity σ³ | 32.25, 37.88, 40.00, 38.00 → **37.03 ± 3.34 (9.0 %)** | 30.9 ± 7.3 |
| flatness | 0.6656, 0.6494, 0.6557, 0.6452 → **0.6540 ± 0.0087** | 0.6452–0.7789, monotone down |

**The cavity TURNED OVER (40.00 → 38.00) — the first time in this lineage that the cavity is not a
rising edge.** Together with a 0.24 % size plateau and a monotone flatness that has stopped moving, the
structural numbers are a plateau, not an upper bound, and the wrapping verdict is a topological boolean
that was unanimous at all eight samples anyway.

**What has NOT stopped:** bonds still creep (14 614 → 15 649, **+7.1 %** over the last 42 000 steps) and
`meanPerTail` still grows (3.294 → 3.397). The chemistry is live; the object's size and shape are not
moving with it. Stated, not smoothed.

---

## 10. VERDICT, WHAT IT ELIMINATES, AND THE PLACE IN THE SEQUENCE

**WHICH PREDICTION FAILED: "wrapping even inside the band."** Not the weaker failure mode (no wrapping
but no closure) — the strong one. At 1546 amphiphiles, 1.50× the closure floor, 0.67× the top of the
band the hypothesis itself defines, at ρ_org 0.30091 against 0.41792, the largest aggregate wraps
**3 of 3 axes and 19 of 19 slabs on every axis at every one of eight settled samples**, and closure is
no nearer: cavity 10.0× short, `radialHeadShells` 1, flatness 1.84× over.

**What that eliminates.**

1. **The supply-budget explanation of percolation is refuted, in the form the hypothesis states it.** The
   claim was that percolation happens because the supply is at or above the cost of a box-spanning
   object. The supply was cut to 0.68× and put mid-band; the wrapping-axis count moved by **zero** —
   exactly what happened when the predecessor cut the head-repulsion gap from 2.86× to 1.26×. **Two
   independent levers (pair energy, then material budget) have now each been moved by a large factor
   with no change to the topology at all.**
2. **The near-coincidence that motivated the task is arithmetically accidental.** 2254 ≈ 2·2πRL/a at
   R ≈ 4 σ compares the supply with a competitor **6.3–7.2× more expensive** than the cheapest one; the
   cheapest spanning object costs 314–358 here, so the supply has been 4–7× above the spanning cost in
   every campaign that percolated *and* in `dec54`, which did not.
3. **It also eliminates the reverse reading of `dec54`.** `dec54` did not span at 1607 while this run
   spans at 1546, on the same box, same water, same drying schedule. **Supply is therefore not even the
   controlling variable in that comparison** — what differs is composition (arm-B two-tailed,
   `meanPerTail` 3.4, versus `dec54`'s O:C = 4 short-tail line) and the fragmentation spectrum
   rehydration leaves, with merges pinned at 0 in both. The frozen distribution, not the budget, is what
   makes `dec54` finite.
4. **And the strict form of the hypothesis is eliminated as testable here at all** (§3): the band is
   empty at every box within the engine's particle ceiling, by 1.7–2.2× even at the ceiling box.

**No vesicle. No finite-size caveat is owed, because nothing closed** — the caveat the brief prepared
for a closure ("a vesicle found inside the band is real but its preference over a spanning object
depends on the box") does not apply. The finite-size statement that *does* apply is the one in §3, and
it cuts the other way: at this box no supply can make a vesicle preferred, so the run inside the band
was testing the hypothesis's own arithmetic rather than a real thermodynamic window.

| # | constraint | measured | required | shortfall | binding? |
|---|---|---|---|---|---|
| 1 | amphiphile **supply** | 1546.0 ± 3.7 | ~932–1030 | 1.50× OVER | no — retired |
| 2 | **aggregate size** | 1546.0 ± 3.7 over 42 000 steps (0.24 %) | ~932–1030 | 1.50× OVER | no — retired |
| **3** | **CONNECTIVITY — the object PERCOLATES** | **3 of 3 axes, 8 of 8 wet, 19/19 slabs; r_g 26.97–27.19 against 27.000 for a filled box; at 0.68× the predecessor's supply** | a finite object: **0** axes | **total** | **YES** |
| 4 | closure / encapsulated water | **0 of 314.4** (now reportable) | ≥ ~312–314 | total | downstream of 3 |
| 5 | head-shell structure | `radialHeadShells` **1** at all wet | 2 | total | downstream of 3 |
| 6 | cavity / enclosed volume | **37.03 ± 3.34 σ³, plateaued** | ≥ 370.8656 | **10.0×** | downstream of 3 |
| 7 | flatness | 0.6452 final | ≤ 0.35 | 1.84× | downstream of 3 |
| **8** | **SUPPLY BUDGET vs the cheapest spanning object** | supply 1546 against a spanning cost of **314–358**; band **empty at L ≤ 81.27 σ**, opens at **L\* = 140–177 σ** | band non-empty | **5.2× over the particle ceiling** | **NO — retired as an explanation, and as a testable one** |
| 9 | monomer exchange / a real CMC | exchange **0**, fissions **0**, free fraction 0 … 6.5e-4 | non-zero, macroscopic | total | still open |
| 10 | run length / settling | **99 000 settled steps**, size plateau 0.24 %, **cavity turned over** | a plateau | none | **no — retired** |
| 11 | electrostatic range | rc_es **14.7794 σ = 3.889 λ_D** (ceiling-bound), repulsion 0.14 % below the certified value | ≥ ~3 λ_D | none | no |
| 12 | medium density | ρ_W 0.80000, ρ_org 0.30091 | liquid | none | no |
| 13 | Verlet / memory / particles | 1.7335 GB, 173 353 | 4.294967292 GB, 429 496 | none (2.48×, 2.48×) | no |

**BINDING CONSTRAINT: STILL (3) CONNECTIVITY — and it is now measured to survive a 32 % cut in the
material, after having survived a 2.27× rise in head repulsion.**

**Where this lands in the sequence.** Published shortfalls of aggregate size against the floor in force:
**80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75× → 0.48× → 0.42× → 0.67×**. This run: **1546.0 against 1030
→ 0.67× (1.50× OVER)** — the **fourth consecutive landing off the end of the sequence**, this time
*deliberately closer to the floor*, with the wall unmoved. (On the gate-6 floor of 932 it reads 0.60×;
both are quoted because the two gates' thicknesses disagree by 11 %.)

---

## 11. REGRESSIONS — EVERY ONE RUN, BEFORE → AFTER

The diff adds **one** file to the repository's source tree, `tests/supply-window.test.ts`, plus
artifacts under `verify/out/` and this report. **No engine, soup, viewer, verify or data file changed**
(`git status` in §13 shows it). So each row's "before" is the predecessor's published PASS, and every
one was re-run to confirm the state now.

| file | before | **after** | note |
|---|---|---|---|
| `tests/gate6-bilayer.test.ts` | 3 passed | **3 passed** | a 1.2060, t 4.4580, escapedMax 0 |
| `tests/water-bilayer-area-move.test.ts` | 1 passed | **1 passed** | a 1.1763, t 4.9686, settled, verdict=passed |
| `tests/soup-forces.test.ts` | 3 passed | **3 passed** | includes the charged grid/Verlet/brute agreement |
| `tests/soup-checkpoint.test.ts` | 3 passed | **3 passed** | "a checkpoint does not change what a run computes next" |
| `tests/run-ui.test.ts` | 8 passed | **8 passed** | puppeteer; profile killed after, `pgrep` clean |
| `tests/gates.test.ts` | 8 passed | **8 passed** | |
| `tests/soup-electrostatics.test.ts` | 8 passed | **8 passed** | ES-RANGE lines unchanged: 4.000 λ_D, discarded 0.0916 |
| `tests/es-range-calibration.test.ts` | env-gated | **skipped (1)** | a harness needing `ES_CAL_CHECKPOINTS`; no code it touches changed |
| `tests/electrostatics-audit.test.ts` | 1 passed | **1 passed** | re-pointed at this campaign's checkpoints |
| `tests/coalescence-mechanism-pin.test.ts` | 5 passed | **5 passed** | the coalescence pins |
| `tests/coalescence-mechanisms.test.ts` | 1 passed | **1 passed** | |
| `tests/percolation-check.test.ts` | 1 passed | **1 passed** | **read `swB54` rows via `PERC_CAMPAIGN_LABEL`, proved by the `role:"campaign"` fields in §7** |
| **`tests/supply-window.test.ts`** (new) | — | **1 passed** | the window arithmetic, all conclusions asserted |
| `npx tsc --noEmit` | 21 errors | **21 errors**, **0 in the new file** | pre-existing `ArrayBufferLike`/`SharedArrayBuffer` class |

Known flaky, **not run and not chased** as instructed: `soup-drywet-cycling`, `soup-grid-resize`,
`rim-lambda-insitu`. `tests/soup-vesicle.test.ts` **never run**.

---

## 12. THE GATE TABLE, REGENERATED THROUGH `npm run verify` (never hand-edited)

```
$ nice -n 15 npm run verify
GATE area-per-lipid: value=1.2032317798086902 rank=A verdict=passed
GATE bilayer-thickness: value=4.5091584271079626 rank=A verdict=passed
GATE bending-modulus: value=null rank=A verdict=unproven
GATE area-per-lipid-water: value=1.176289404631726 rank=C verdict=passed
GATE bilayer-thickness-water: value=4.968568267611434 rank=C verdict=passed
GATE vesicle-closure-water: value=0 rank=B verdict=failed
GATE aggregate-percolation: value=3 rank=B verdict=failed
GATE vesicle-verdict: value=0 rank=A verdict=failed
GATE chain-length-asf: value=0.1981159846923758 rank=D verdict=unproven
GATE mean-tail-length: value=3.397 rank=D verdict=unproven
GATE closure: value=1284.875 rank=D verdict=unproven
GATE chain-to-bead-mapping: value=3 rank=D verdict=unproven
```

| gate | value | rank | verdict | corridor |
|---|---|---|---|---|
| area-per-lipid | 1.2032 | A | passed | inside |
| bilayer-thickness | 4.5092 | A | passed | inside |
| bending-modulus | null | A | **unproven** | none (κ scenario `valid=false`, unchanged) |
| area-per-lipid-water | 1.1763 | C | passed | inside |
| bilayer-thickness-water | 4.9686 | C | passed | inside |
| **vesicle-closure-water** | **0** | B | **failed** | outside |
| **aggregate-percolation** | **3** | B | **failed** | outside |
| vesicle-verdict | 0 | A | failed | outside |
| chain-length-asf | 0.1981 | D | unproven | none |
| mean-tail-length | 3.397 | D | unproven | outside |
| closure | 1284.875 | D | unproven | inside |
| chain-to-bead-mapping | 3 | D | unproven | none |

Two things changed in the table and both are this campaign: **`aggregate-percolation` = 3 read from
`swB54`'s own eight rows** (not a label match), and **`vesicle-closure-water` is a number, 0, where the
predecessor's run made it a refusal** — the closure gate can answer again (§8.2). `bending-modulus`
remains `unproven` exactly as before, untouched. **The published campaign block in `gates.json` is now
this reduced-supply run**, which is stated here so no reader mistakes it for the arm-B campaign.

---

## 13. EVERY COMMAND, WITH REAL OUTPUT

Outputs quoted in full above: §2.1 (both gates), §2.4/§3 (the arithmetic), §6.2 (the trace), §7 (the
percolation), §8.1 (the audit), §12 (the gates). The full list, in order:

```
# --- the measured bases the whole window rests on
nice -n 15 npx vitest run tests/gate6-bilayer.test.ts tests/water-bilayer-area-move.test.ts --no-file-parallelism   # 180 s, 3 passed

# --- the window arithmetic, stated BEFORE the campaign, as a committed test
nice -n 15 npx vitest run tests/supply-window.test.ts --no-file-parallelism                                        # 0.1 s, 1 passed

# --- the campaign, ONE lineage, ten resumable invocations of the same command
nice -n 15 npx tsx soup/cli/campaign.ts --label swB54 --box 54 \
  --start '{"C":20084,"O":6693,"H":20084,"M":521,"W":125971}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --charge --pH 7.0 --ionicStrength 0.01 \
  --steps <n> --every <n> --dir data/checkpoints/swB54
  # --steps  6000 --every 3000   -> globalStep   6 000    (15.8 ms/step at 1366 bonds)
  # --steps 18000 --every 9000   -> globalStep  24 000    (the drying event, 18 increments, dry box 32.8431)
  # --steps 18000 --every 18000  -> globalStep  45 400    (the rehydration + insertion)
  # --steps 15000 --every 15000  -> globalStep  60 400    390 s  (25.8 ms/step at 12 670 bonds)
  # --steps 14000 --every 14000  -> globalStep  74 400    357 s
  # --steps 14000 --every 14000  -> globalStep  88 400    359 s
  # --steps 14000 --every 14000  -> globalStep 102 400    365 s
  # --steps 14000 --every 14000  -> globalStep 116 400    363 s
  # --steps 14000 --every 14000  -> globalStep 130 400    365 s
  # --steps 14000 --every 14000  -> globalStep 144 400    363 s

# --- the instruments, all off-GPU on the campaign's own checkpoints
CONTINUOUS_RUN_PREFIX=swB54-step CONTINUOUS_RUN_DIRS=data/checkpoints/swB54 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/supply-window-campaign-B54-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism
PERC_CAMPAIGN_LABEL=swB54 PERC_CHECKPOINTS="<8 swB54 wet + 2 dec54 controls + lrB54pH70 step90600>" \
  PERC_ARTIFACT=verify/out/gates-percolation.json \
  nice -n 15 npx vitest run tests/percolation-check.test.ts --no-file-parallelism
ES_AUDIT_CHECKPOINTS="<the same 8 swB54 wet>" ES_AUDIT_ARTIFACT=verify/out/supply-window-audit.json \
  nice -n 15 npx vitest run tests/electrostatics-audit.test.ts --no-file-parallelism

# --- regressions, three groups, then the published gates
nice -n 15 npx vitest run tests/gates.test.ts tests/coalescence-mechanism-pin.test.ts \
  tests/coalescence-mechanisms.test.ts tests/es-range-calibration.test.ts tests/supply-window.test.ts \
  --no-file-parallelism                                              # 15 passed, 1 skipped
nice -n 15 npx vitest run tests/soup-forces.test.ts tests/soup-checkpoint.test.ts --no-file-parallelism   # 9 s, 6 passed
nice -n 15 npx vitest run tests/soup-electrostatics.test.ts tests/run-ui.test.ts --no-file-parallelism    # 34 s, 16 passed
pkill -f puppeteer_dev_chrome_profile ; pgrep -fl puppeteer_dev_chrome_profile   # -> NO ORPHANS
nice -n 15 npx tsc --noEmit                                          # 21 errors, 0 in the new file
cp verify/out/supply-window-campaign-B54-trace.json verify/out/gates-campaign-trace.json
nice -n 15 npm run verify                                            # the gate table above
```

`git status --porcelain` before the commit — the whole diff, and the proof that no engine file moved:

```
 M verify/out/gates-campaign-trace.json
 M verify/out/gates-percolation.json
 M verify/out/gates.json
 M verify/out/kappa-measurement.json
 M verify/out/report.html
 M verify/out/water-bilayer-area-move.json
?? tests/supply-window.test.ts
?? verify/out/supply-window-audit.json
?? verify/out/supply-window-campaign-B54-trace.json
```

**One mistake is recorded rather than tidied away.** The first `--steps 15000` chunk ran 390 s, close
enough to the 500 s cap that the remaining eight were cut to 14 000 (357–365 s). No invocation
overran, nothing had to be re-run, but the margin was thinner than it should have been on the first
wet chunk and the sizing should have been measured on chunk 1 rather than assumed. **Cost: 0
invocations.**

---

## 14. TOTALS, RESOURCES, HOUSEKEEPING

- **Compute invocations: 20, against a budget of 30 — UNDER by 10.** 1 gate pair, 1 arithmetic test,
  10 campaign chunks, 3 instrument audits, 3 regression groups, 1 `tsc`, 1 `npm run verify`. Ten
  campaign chunks were the whole cost centre and none was wasted: no lineage was deleted, no chunk
  re-run, no configuration abandoned.
- **Longest single foreground invocation: 390 s** (campaign chunk 4). **Every invocation was under the
  500 s cap**, `nice -n 15`, one at a time, foreground, never backgrounded; every multi-file vitest
  group ran `--no-file-parallelism`, so **no two compute processes ever existed at once**. No dev server
  was started or killed; :5199 was never touched. `timeout` was never used.
- **Orphans: none.** `pkill -f puppeteer_dev_chrome_profile` after the only browser group, confirmed by
  `pgrep -fl puppeteer_dev_chrome_profile` printing nothing (`NO ORPHANS`). No `deps_temp_*` cleanup was
  needed — no vitest died mid-run.
- **Ceilings, checked before the first invocation and true after:** 173 353 particles against 429 496
  (**2.48× inside**), 1.7335 GB of main Verlet list against 4.294967292 GB (**2.48× inside**), plus a
  head-only ES list of 6693 heads (≈ 33 MB).
- **Never done:** no threshold widened, no rank-A or rank-B constant re-fitted, `co_bond.attemptRate`
  refused an eighth time, no pre-made start, no under-dense water, no `--dump-dom`, no whole-particle
  JSON transfer, `tests/soup-vesicle.test.ts` not run.
- Artifacts written: `verify/out/supply-window-campaign-B54-trace.json`,
  `verify/out/supply-window-audit.json`, `verify/out/gates-percolation.json`,
  `verify/out/gates-campaign-trace.json`, `verify/out/gates.json`, `verify/out/report.html`.
  Checkpoints: `data/checkpoints/swB54/` (12 files, gitignored by `data/checkpoints/` as every campaign's
  are).

---

## 15. CONCERNS

1. **The strict hypothesis is untestable in this engine, and that limits what "refuted" can mean.** §3
   shows the band opens only at L ≈ 140–177 σ, 5.2× over the particle ceiling. What §7 refutes is the
   hypothesis's own *arithmetic* (its band, its competitor) and the general claim that reducing the
   supply toward the floor stops percolation. It does **not** refute the underlying thermodynamics —
   that a spanning tube beats a vesicle in a periodic box — which remains sound and untested.
2. **The cheapest-spanning cost is a thermodynamic bound and the system is not at equilibrium.**
   Merges are **0** in every campaign, so which object appears is set by the fragmentation spectrum
   rehydration leaves, not by a free-energy comparison. `dec54` sat 4–5× above the spanning cost and
   stayed finite; this run sits 4.3–4.9× above it and spans. **A budget argument cannot decide between
   those two, which is itself the strongest reason to stop pursuing budgets.**
3. **The floor band widened to 932–1030 (10.5 %)** because the two gates' thicknesses now disagree by
   11.4 % (4.4580 vs 4.9686), against 5.6 % last task. The shortfall figure quoted in the sequence
   depends on which basis (0.67× vs 0.60×); both are published. The explicit-water thickness is also
   visibly noisy chunk to chunk (3.13–4.97 in the trajectory print), so the 4.9686 tail value carries
   real uncertainty that the gate's corridor (4–6) does not express.
4. **`rc_es` was 3.889 λ_D, not the calibrated 4.000**, because a smaller organic pool makes a smaller
   dry box and the rule's own minimum-image ceiling binds. Quantified at 0.14 % on the contact
   repulsion and 0.1000 vs 0.0916 discarded, so it cannot carry the verdict — but it means this
   campaign's charge is not *bit-identical* in range to the predecessor's, and the one variable I set
   out to change alone (supply) dragged this second one with it. Unavoidable without changing the rule.
5. **The `centre-untrusted` interpretation in the predecessor's report is wrong and should be corrected
   there too** (§8.2). Anyone reading `long-range-electrostatics-report.md` §7.3 will take
   `null(centre-untrusted)` as a wrapping verdict; it is a Rayleigh significance verdict, and this run
   shows the two coming apart.
6. **The cavity plateau is 9 % noisy on four samples** and rests on a single turnover (40.0 → 38.0). It
   is better evidence than the predecessor had, and it is not strong evidence.
7. **`gates.json`'s published campaign block is now a deliberately reduced-supply run.** That is the
   convention (the newest campaign publishes), but a reader comparing the gate table against
   `final-campaign-report.md`'s numbers will find a different supply behind it. Flagged in §12.
8. **The chemistry has not stopped** (+7.1 % bonds, `meanPerTail` 3.294 → 3.397 over the last 42 000
   steps). The structure has plateaued *while the molecule is still growing*, which means the settled
   object is a steady state of a moving composition, not an equilibrium of a fixed one.
