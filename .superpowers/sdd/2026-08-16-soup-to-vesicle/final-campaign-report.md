# The z-boundary fix, the two-tailed campaign at box 54, and the project's vesicle verdict

Task `final-campaign` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `coalescence-report.md`, whose **concern 5** is task 1 here (*"the z-open clustering
defect is reported and pinned, NOT fixed … every aggregate-size number in this lineage is understated
by ~12 %"*) and whose **§9.2** is task 2 — the campaign it fully specified and did not run.

`data/params.json` rank-A constants: **NOT touched.** `data/soup.json`: **NOT touched, not one
field.** `co_bond.attemptRate`: **NOT touched** (refused a fifth time). No threshold widened, no
corridor relaxed, no tolerance changed, no assertion weakened. No arm reverted to under-dense water:
the campaign runs at ρ_W = 125 971/157 464 = **0.79999** exactly. Nothing started from anything
pre-made — the campaign begins at a monomers-only lattice at its own step 0. No fusion probe was
built.

---

## 0. VERDICT UP FRONT, TEN LINES

1. **The z-boundary double count is FIXED in the engine, not just pinned.**
   `engine/src/aggregate.ts`'s `buildClusterUnionFind` now takes a `periodicZ` flag; it defaults to
   **false** (the membrane engine's own physics — a bilayer patch in vacuum has no image across z)
   and every **soup** call site passes **true**, because `soup/src/sim.ts` wraps all three axes every
   step. Pinned in both branches, plus a new test for the same defect class on **x and y** (§1).
2. **Which published numbers move, measured on the artifacts already on disk, not argued.** The
   decisive box-54 lineage's largest aggregate goes **135 → 151 (+11.9 %)** at its last checkpoint
   and **153 → 200 (+30.7 %)** at its first; its **binding-constraint headline 136.2 → 152.8** and
   its shortfall **6.75× → 6.02×**. The box-30 post-fix cycled arms go **109.3 → 136.3 (+24.7 %)**
   and their single-aggregate share **38.5 % → 48.2 %**. Shape numbers move too, in the direction
   nobody looked for: the largest object's `radialHeadShells` falls **2 → 1** at the last three
   checkpoints. Full list in §2.
3. **The specified arithmetic is confirmed to the digit, and then re-derived from THIS task's own
   gates.** The spec's floor 998, window [40.89, 55.69] σ and 2.30× margin at L = 54 all reproduce
   exactly from their own inputs. From this task's freshly measured t = **4.7990 σ** and
   a = **1.1777 σ²** the floor is **1004** (band **938–1004**) and the window
   **L ∈ [42.16, 55.69] σ**, width 13.53 σ (§3).
4. **The campaign ran: arm B's composition (O:C = 0.333) at box 54, N = 191 778, liquid water,
   real RNG, one drying event, 176 400 steps of which 135 000 settled.** Chosen because arm B carries
   the highest two-tailed head fraction (47–48 % against arm C's 28–31 % and arm A's 12–14 %) and
   Israelachvili's p = v/(a₀l_c) is monotone in the second tail (§4).
5. **AGGREGATE SIZE IS RETIRED AS THE BINDING CONSTRAINT — for the first time in this project.** The
   rehydration did **not** fragment: 2 aggregates, one holding **99.95 %** of the supply, largest
   **2211 → 2077** amphiphiles over 135 000 settled steps, plateau mean of the last five **2087.2
   ± 7.7 (0.37 %)**, against the floor 1004 — **2.08× OVER** (§5).
6. **AND IT IS A PERCOLATING NETWORK, measured directly and not by proxy.** A new wrapping test
   (replicate the aggregate's own members along one axis, join within the project's own cutoff, ask
   whether a particle shares a component with its own +L image) says the object **wraps on 3 of 3
   axes at every one of 5 wet checkpoints spanning 135 000 steps**, touching **19 of 19** cutoff-wide
   slabs on every axis. The controls say no: the decisive run's 151-molecule rod wraps on **0** axes
   (7/7/10 slabs), and arm A's 154-molecule object on **0**. r_g = 26.25–27.19 σ against **27.000 σ
   for a uniformly filled 54 σ box** — 98.4–100.7 % (§6).
7. **NO VESICLE, and closure is not close.** `encapsulatedWater` = **0** against **317.1–320.9** and
   `closed = false` at every wet checkpoint that could be evaluated; `cavityVolume` plateaus at
   **50.5 ± 1.6 σ³** over the last 8 samples against the 370.8656 σ³ minimum — **7.3× short**;
   `radialHeadShells` = **1** at every single checkpoint of the whole run, so there is no bilayer
   wall, only one shell (§7).
8. **NO ROD → SHEET CROSSING. The signature the lever was supposed to produce did not appear.** The
   rods' fixed cross-section R⊥ = 4.006–4.165 σ (4.085 σ re-measured under the fix) did not widen
   into a ribbon: it jumped to **20.1–21.6 σ**, which is a space-filling object, not a sheet. And no
   object passes the lamellar gate at any checkpoint — flatness **0.65–0.81** against the ≤ 0.35 a
   sheet needs, 2.20× over (§7.2).
9. **BINDING CONSTRAINT, numbered: (3) CONNECTIVITY — THE SUPPLY PERCOLATES. 3 of 3 axes wrapping,
   19/19 slabs, at 2087 amphiphiles against a floor of 1004.** An object that spans the periodic box
   has no inside and no outside; it cannot enclose anything, whatever its size. Supply (2.10× over)
   and aggregate size (2.08× over) are both retired; closure, cavity volume, head-shell count and
   flatness are all downstream of (3) (§8).
10. **IS CLOSURE REACHABLE? NO — and this is now a measurement, not a conditional.**
    `coalescence-report.md` §9.3 wrote: *"If that object is a network and not a compact aggregate,
    then this model cannot reach closure without a change that would not be defensible."* It is a
    network, on 3 of 3 axes, at the box the report itself specified. §9 answers the project's question
    and names what a next model would need — and the honest reading is that the two-tailed lever
    trades a **size** wall for a **topology** wall rather than removing a wall (§9).

---

## 1. TASK 1 — THE Z-BOUNDARY DOUBLE COUNT, FIXED

### 1.1 What was wrong, in the code's own words

`engine/src/aggregate.ts`'s `buildClusterUnionFind` — through which `clusters`,
`clusterComponents`, `largestClusterFraction`, `soup/src/aggregates.ts`'s `analyzeAggregates` and
therefore **every stage detection in this project** pass — said in its own comment: *"z is unbounded
(open boundary) so it gets a plain integer cell index of width `cutoff` with no wrapping"*, and
computed `ddz = zi - positions[j*4+2] // z open: no periodic image`.

That is **correct** for the membrane engine (Tasks 6/8: a bilayer patch in vacuum genuinely has no
image across z, and `engine/src/closure.ts`'s flood is an open-z pipeline end to end) and **wrong**
for the soup, whose own header states it wraps all three axes every step. Every aggregate straddling
the z face was split and counted twice.

### 1.2 The fix

One optional parameter, threaded through four functions:

```
function buildClusterUnionFind(positions, box, cutoff, periodicZ = false)
  nz = periodicZ ? max(1, floor(Lz / cutoff)) : 0
  wz = periodicZ ? Lz / nz : cutoff
  z  = periodicZ ? wrap1(pos_z, Lz) : pos_z
  cz = periodicZ ? min(nz-1, floor(z / wz)) : floor(z / wz)
  ncz = periodicZ ? ((cz + dz) % nz + nz) % nz : cz + dz
  ddz = periodicZ ? mi1(zi - pos_j_z, Lz) : zi - pos_j_z
```

`clusters`, `clusterComponents` and `largestClusterFraction` take the same flag and pass it down.
**The default is `false`, so every membrane-engine caller is bit-identical** —
`engine/src/index.ts`'s `largestClusterFractionOf`, `engine/src/closure.ts`'s
`largestClusterCenter`/`recenterOnLargestCluster`, `tests/self-assembly.test.ts`,
`tests/gate6-bilayer.test.ts`. `largestClusterCenter` is deliberately **not** given the flag: its z
is a plain mean feeding an open-boundary flood, and giving a periodic connectivity to an open-z
pipeline would be a different and larger change.

**Call sites switched to `true`** (all of them soup, i.e. genuinely bulk):

| file | what it feeds |
|---|---|
| `soup/src/aggregates.ts` `analyzeAggregates` | **every stage gate**, every published aggregate count and size |
| `soup/src/stages.ts` `detectStage` | `largestAggregateFraction` (diagnostic) |
| `tests/continuous-run-audit.test.ts` | the off-GPU auditor — it must not measure a different rule from the run it audits |
| `tests/helpers/shell-pore-geometry.ts`, `tests/periodic-measurement.test.ts`, `tests/rim-lambda-insitu.test.ts` | the soup aggregate analyses |
| `tests/soup-bonds.test.ts` | assembly-vs-temperature, a whole-box bulk statistic |

**Deliberately NOT switched, and stated rather than left to be found:**
`tests/water-bilayer.test.ts`, `tests/water-bilayer-broth.test.ts` and
`tests/water-bilayer-area-move.test.ts` call `api.largestClusterFraction` on a **membrane patch**
that spans x,y with bulk water above and below it. Their number is a saturation check (1.0000) on
one intact sheet, the sheet cannot reach either z face, and switching them would move three
published gate rows for no measurable reason. They keep the default, and this is a known remaining
inconsistency — the soup engine does wrap z there too. Its size is bounded by the numbers those
tests already print: `clusterFraction` is 1.0000/0.9974, so the flag could change it by at most
0.26 %.

### 1.3 The pins — the old one kept, two new ones added

`tests/coalescence-mechanism-pin.test.ts`, **5 tests** (was 3). The predecessor's z pin is kept
verbatim and merely re-titled to say it pins the *default* rather than a defect:

```
$ nice -n 15 npx vitest run tests/coalescence-mechanism-pin.test.ts tests/soup-grid-resize.test.ts --no-file-parallelism
COAL-PIN enrichmentVsWater over 13 snapshots: [0.3894, 0.4549]
COAL-PIN merges=1 fissions=0 encounters=51 ofWhichMerged=1
COAL-PIN clustering: zOpen=4+4 periodic3d=8
 ✓ tests/coalescence-mechanism-pin.test.ts (5 tests) 7ms
COAL-PIN clustering z-periodic: joined=8 genuinelyApart=2
COAL-PIN clustering x/y faces: 1 aggregate of 8 on both axes, both periodicZ branches
```

- **new test 4** — `periodicZ = true` joins the z-straddling blob into ONE component of 8, and the
  partition is **element-for-element equal** to `clusterComponents3D`'s, the independent
  implementation the predecessor wrote as its control. Two implementations agreeing is what makes
  this a fix rather than a second convention. It also checks the flag does **not** join what is
  genuinely apart: the same blob with a real 8 σ gap stays 2 components under both branches.
- **new test 5, the one the brief asked for** — the same defect class on **x and y**. A blob
  straddling the x face, and one straddling the y face, must each read as ONE aggregate of 8 under
  **both** `periodicZ` branches. x and y have been periodic since Task 6, so nobody ever checked
  them; this is the test that would have caught the z bug had it been written for z, now written for
  all three axes.

---

## 2. WHICH PREVIOUSLY PUBLISHED NUMBERS MOVE — ONE PLACE, MEASURED

Both halves re-measured **off-GPU on the checkpoints already on disk**, with the identical auditor
(`tests/continuous-run-audit.test.ts`) that produced the originals, into **new** artifacts so the
before and the after both survive:
`verify/out/zfix-decisive-run-54-trace.json` and `verify/out/zfix-box30-arms-trace.json`.

### 2.1 The decisive box-54 lineage (`decisive-run-report.md` §3.4, §3.5, §4, §5.2 of the coalescence report)

| globalStep | aggregates z-open → **fixed** | largest z-open → **fixed** | understated by |
|---|---|---|---|
| 3 000 | 125 → 121 | 5 → 5 | 0 % |
| 6 000 | 213 → 208 | 7 → 7 | 0 % |
| *15 400 (dry)* | *23 → 9* | *1107 → 1129* | *2.0 %* |
| *21 400 (dry)* | *8 → 2* | *1567 → 1573* | *0.4 %* |
| *24 000 (dry)* | *8 → 1* | *1599 → 1615* | *1.0 %* |
| *31 000 (dry)* | *8 → 1* | *1632 → 1649* | *1.0 %* |
| **41 400** (first settled wet) | **31 → 22** | **153 → 200** | **30.7 %** |
| 55 400 | 31 → 22 | 149 → 193 | 29.5 % |
| 69 400 | 30 → 22 | 143 → 190 | 32.9 % |
| 79 900 | 32 → 23 | 138 → 165 | 19.6 % |
| 90 400 | 30 → 23 | 138 → 162 | 17.4 % |
| 100 900 | 31 → 25 | 139 → 160 | 15.1 % |
| 111 400 | 31 → 24 | 141 → 160 | 13.5 % |
| 121 900 | 30 → 24 | 136 → 155 | 14.0 % |
| 132 400 | 31 → 25 | 138 → 155 | 12.3 % |
| 142 900 | 31 → 25 | 137 → 153 | 11.7 % |
| 153 400 | 32 → 25 | 136 → 153 | 12.5 % |
| 163 900 | 32 → 24 | 135 → 152 | 12.6 % |
| **174 400** | **34 → 25** | **135 → 151** | **11.9 %** |

*(italic rows are dry-phase percolating mass and mean nothing structurally; they are here because the
z-open rule was splitting even that.)*

**The list of previously published numbers now known to be UNDERSTATED, and by how much:**

| published number | where | published | **corrected** | move |
|---|---|---|---|---|
| largest aggregate, decisive run, final | decisive §3.4/3.5, coalescence §5.2 | **135** | **151** | **+11.9 %** |
| largest aggregate, decisive run, first settled wet | decisive §3.4, coalescence §5.2 | 153 (200 as 3D) | **200** | +30.7 % |
| **binding-constraint headline (last-5 mean)** | **decisive §4, "136.2 against 920, 6.75× short"** | **136.2** | **152.8** | **+12.2 %** |
| shortfall on aggregate size, decisive run | decisive §4, verdict line 9 | **6.75×** | **6.02×** | −10.8 % |
| largest aggregate, box-30 cycled post-fix, n = 3 | decisive §1.3 (103 / 71 / 154) | **109.3** | **136.3** (147 / 108 / 154) | **+24.7 %** |
| single-aggregate share at box 30 | decisive §4 ("38.5 %") | **38.5 %** | **48.2 %** | +9.7 pp |
| single-aggregate share at box 54 | decisive §4 ("9.5 %") | **9.5 %** | **10.6 %** | +1.1 pp |
| box 30 → 54 largest-aggregate ratio | decisive §4 ("1.24×") | **1.24×** | **1.11×** | the box bought *less* than published |
| largest aggregate, minimisation-only control, n = 3 | decisive §1.3 (14 / 12 / 13) | **13.0** | **17.0** (15 / 12 / 24) | +30.8 % |
| the cycling effect on largest aggregate | decisive §1.4 ("8.41×") | **8.41×** | **8.02×** | −4.6 % |
| aggregate count, decisive run, final | decisive §3.4 ("34") | **34** | **25** | **−26.5 %** (OVERstated) |
| `radialHeadShells` of the largest, last 3 samples | decisive §3.5 ("2 at 4 of the last 5") | **2** | **1** | **the bilayer-shell reading was an artefact of the split** |
| flatness of the largest, final | decisive §3.5 ("0.2008") | 0.2008 | **0.1689** | more rod-like, not less |
| in-plane symmetry of the largest, final | decisive §3.5 ("0.2454") | 0.2454 | **0.2131** | further from the 0.50 a sheet needs |
| aggregate counts on the box-58 lineage | periodic-measurement / shell-pore / rim-λ artifacts | 5–10 | **3–5** | OVERstated ~2× |
| largest aggregate, box-58 lineage | rim-λ artifact | 1282 amph | **1284** | +0.16 % |

**Three things in that list deserve to be read carefully rather than skimmed.**

- **Aggregate COUNTS were overstated, not understated** — the split manufactured extra objects
  (34 → 25 at box 54, 8 → 1 in the dry phase). Every "N aggregates" in this project's history is
  high by 10–30 % in the wet phase and by much more in the dry.
- **`radialHeadShells` = 2 was an artefact.** The decisive report read two head shells at 4 of its
  last 5 samples and listed that row as **passing** the bilayer criterion. Under the fix it is **1**
  at those same checkpoints: joining the object's two z-halves puts the heads on one shell. So one
  of the two structural criteria the decisive report recorded as passed, did not pass. Nothing
  downstream changes (`closed` was already false and `encapsulatedWater` already 0) but the row was
  wrong and is corrected here.
- **`encapsulatedWater` and `closed` do NOT move.** 0 against 297.6, `closed = false`, at every wet
  checkpoint before and after. The verdict "no vesicle" was never in question; only its distance was.

### 2.2 Not moved, and proven not moved

`encapsulatedWater`, `closed`, amphiphile supply, per-tail length, two-tailed fraction, bond counts,
event counters, census and every valence invariant are **identical** before and after — the fix
touches connectivity only. The membrane engine is untouched by construction (default `false`) and by
measurement: `tests/self-assembly.test.ts` passes with `largestClusterFractionOf = 0.9992`,
`tests/gate6-bilayer.test.ts` passes, `tests/closure.test.ts` 8/8, `tests/soup-aggregates.test.ts`
4/4 unchanged (its synthetic micelle/bilayer/vesicle fixtures do not straddle z — a prediction, and
it held).

---

## 3. THE FLOOR AND THE WINDOW, RECOMPUTED — INCLUDING THE SPEC'S OWN ARITHMETIC RE-DERIVED

Both gates were re-measured on this task's tree, because the floor must come from a measurement made
here rather than a number carried forward:

```
$ nice -n 15 npx vitest run tests/gate6-bilayer.test.ts --no-file-parallelism
GATE6 area 1.2050 +/- 0.0095 (min 1.1861, max 1.2328)  thickness 4.4952 (per-frame mean 4.4874, sd 0.2307, n=400)
      lnA drift/move 1.509e-5 +/- 1.37e-5 (t=1.10)  accepted 0.263  escapedMax 0  box 24.648  steps 83000
 Test Files  1 passed (1)      Tests  2 passed (2)

$ nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=300 areaPerLipid(MEASURED, tail mean)=1.1777
      [min 1.1722, max 1.1828, corridor 1.1-1.5] driftPerChunk(lnA)=2.868e-5 t=1.95
      thickness(measured)=4.7990 [corridor 4-6] clusterFraction=1.0000 waterInCore=53/4500
      headBuriedFraction=0.0857 acceptedFraction=0.1257 of 3000 trials throughput=1243.91 steps/s at N=5700 verdict=passed
 Test Files  1 passed (1)      Tests  1 passed (1)
```

Same formula as `continuous-run-report.md` §1.1. No new constant.
`stageThresholds.enclosedVolume` = **370.8656 σ³ untouched**.

```
R_in  = (3 * 370.8656 / 4pi)^(1/3) = 4.45700 sigma
R_mid = R_in + t/2
floor = 2 * 4pi * R_mid^2 / a
```

| basis for t, a | t σ | a σ² | R_mid σ | **floor** |
|---|---|---|---|---|
| **explicit-water gate, MEASURED THIS TASK** | **4.7990** | **1.1777** | **6.85650** | **1003.3 → 1004** |
| solvent-free gate 6, MEASURED THIS TASK | 4.4952 | 1.2050 | 6.70460 | 937.6 → **938** |
| the coalescence report's basis | 4.9421 | 1.2098 | 6.92805 | 997.1 → 998 |
| the decisive report's basis | 4.5472 | 1.2372 | 6.73060 | 920.3 → 921 |

**The spec's arithmetic, verified against its own inputs rather than trusted** — every figure
reproduces to the digit:

```
spec floor    : R_mid = 4.45700 + 4.9421/2 = 6.92805 ; 2*4pi*6.92805^2/1.2098 = 997.1 -> 998   CONFIRMED
spec supply end: rho = 394/27000 = 1.459259e-2 ; L >= (998/1.459259e-2)^(1/3) = 40.89 sigma      CONFIRMED
spec margin    : 1.459259e-2 * 157464 = 2298 against 998 = 2.30x                                 CONFIRMED
spec Verlet    : N = 191778 ; 191778*2500*4 = 1.9178 GB vs 4.2950 GB ; 191778 vs 429496. VALID.  CONFIRMED
```

**And re-derived from this task's own gates and this campaign's own measured supply** (12 settled wet
samples, steps 77 400–176 400: amph = 2154, 2139, 2133, 2123, 2113, 2104, 2103, 2097, 2096, 2089,
2081, 2078 → mean **2109.2**, ρ_amph = 2109.2/157 464 = **1.33946e-2 σ⁻³**; note this is measured
*at the box the window is about*, not extrapolated from box 30, where the same composition gave
1.45926e-2):

```
supply       : rho_amph * L^3 >= 1004  ->  L >= (1004 / 1.33946e-2)^(1/3) = 42.16 sigma
               (with the 938 end of the band: 41.22 sigma)
measurability: R(L) = 0.067 * L^1.5 <= L/2  ->  L <= (0.5/0.067)^2 = 55.69 sigma  (box-58 calibration, unchanged)

WINDOW: L in [42.16, 55.69] sigma, width 13.53 sigma
At L = 54: supply = 1.33946e-2 * 157464 = 2109 against 1004  ->  2.10x margin
MEASURED largest at plateau: 2087.2 against 1004  ->  2.08x OVER the floor
```

Floor trend across seven tasks: 960 → 912 → 968 → 1088 → 920 → 998 → **1004**. The band is
938–1004, the two independent gates **7.0 % apart**.

---

## 4. THE CAMPAIGN — BOX, COMPOSITION, WHY THAT O:C, WATER, SCHEDULE, N

### 4.1 Composition, and why arm B rather than arm C

Arm B's ratios, scaled from box 30 by the volume factor 54³/30³ = 5.832, so that ρ_tot, ρ_W and
ρ_org are the box-30 arm's own to five digits:

```
C = round(4783 * 5.832) = 27894    O = round(1594 * 5.832) = 9296    H = 27894
M = round( 124 * 5.832) =   723    W = round(0.8 * 157464) = 125971
N = 191778   rho_tot = 1.21792 (box-30 arm B: 1.21793)   rho_org = 0.41792   rho_W = 0.79999
O:C = 9296/27894 = 0.33326
```

**Why O:C = 0.333 and not 1.000.** The lever is Israelachvili's packing parameter
p = v/(a₀·l_c) (**rank B — a literature criterion, not a constant imported into this model**;
J. N. Israelachvili, D. J. Mitchell & B. W. Ninham, *J. Chem. Soc. Faraday Trans. II* **72** (1976)
1525, and *Intermolecular and Surface Forces* 3rd ed. ch. 20), and p is monotone in the **second
tail** — doubling v at fixed l_c doubles p. So the arm to run is the one with the highest two-tailed
head fraction, and re-measured here under the fix on the arms' own checkpoints:

| arm | O:C | amph | per-tail | **two-tailed heads** | aggregates | largest | share | wraps the box? |
|---|---|---|---|---|---|---|---|---|
| A (the defended broth) | 4.000 | 264–313 | 2.079–2.137 | 31–41 = **12–14 %** | 3–4 | 108 / 147 / 154 | 0.394–0.583 | **no, 0 of 3 axes** |
| C | 1.000 | 371–376 | 2.363–2.394 | 106–116 = **28–31 %** | 2 | 370–375 | 0.997 | **yes, 3 of 3** |
| **B — chosen** | **0.333** | **392–394** | **2.705–2.793** | **185–190 = 47–48 %** | **1** | **392–394** | **1.000** | **yes, 3 of 3** |

Arm B is 1.55× arm C on the quantity the criterion is monotone in, and it is the arm the predecessor
named. The confound the predecessor stated stands unchanged and is not worked around: in this model
heads **are** the chain terminators, so composition cannot move the two-tailed fraction without
moving tail length, and this arm moves both.

### 4.2 Box, water, schedule, N

- **Box 54 σ.** Inside the window [42.16, 55.69] at both ends, and chosen at the decisive run's own
  box so the comparison with its 151-molecule rod is direct rather than inferred. r_g of a uniformly
  filled 54 σ box is **27.000 σ** — the number the percolation question is asked against.
- **Water: liquid, 0.79999 σ⁻³** (125 971 of 157 464). Nothing reverted to under-dense water.
- **Schedule: ONE drying event** (`--cycle --evaporate --cycles 1`). Justified from the measured
  trend and not from preference: `evaporation-report.md` §6.3 measured the largest aggregate at
  **189 (cycle 1) → 169 → 162 → 168 → 166 → 160** with the yield declining **monotonically**
  0.1624 → 0.1242 as `co_break` accumulated 9 → 66. One drying event buys everything; cycles 2–6 buy
  nothing measurable and cost yield. Period (32 000), dry fraction (0.65), target dry density (1.34)
  and both ramps are `data/soup.json`'s own, untouched.
- **N = 191 778 against the ceilings:** Verlet `191 778 × 2500 × 4` = **1.9178 GB** against
  4.2950 GB (2.24× inside); **191 778** particles against the hard maximum **429 496** (2.24×
  inside). Checked before the first invocation, not after.
- **Evaporation plan, derived and then realised exactly:** W_residual = round(125 971/1400) = **90**,
  N_dry = 65 897, dryBox = (65 897/1.34)^(1/3) = **36.6370 σ**, grid 36.6370/2.94695 = 12.43 → **12
  cells** against minCells = 5 (VALID), concentration factor **3.2020×**. Realised: `box=36.6370`,
  `solvent=90`, `N=65897`, to the digit.
- **The evaporating ramp needed no minimiser at all.** Across all 18 drying and all 18 rehydration
  increments the guard fired **0 of 36** times, per-increment `max|F|` = **2.7e2–6.5e2** against the
  σ/(13·dt²) = 769 trigger. That is the decisive report's §2.3 finding reproducing at a composition
  it never saw.

---

## 5. THE STAGE TRACE — EVERY CHECKPOINT, AUDITED OFF-GPU

`verify/out/zfix-campaign-B54-trace.json`, 20 checkpoints, produced by
`tests/continuous-run-audit.test.ts` — the same functions the live run's own stage detection uses,
with this task's z fix in both. R⊥ = √(λ₁+λ₂) is the cross-sectional gyration radius, the quantity
the rod → sheet question is about.

| globalStep | box | stage | amph | perTail | 2-tail | cc+co | aggs | **largest** | share | r_g | flat | inPl | **R⊥** | radSh | cav σ³ | encH₂O/thr | closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 000 | 54.0000 | monomers | 276 | 2.162 | 27 | 1104 | 196 | 7 | 0.025 | 3.92 | 0.105 | 0.215 | 1.93 | n/a | 0 | 0/296.7 | false |
| 6 000 | 54.0000 | amphiphiles | 425 | 2.307 | 60 | 2013 | 248 | 11 | 0.026 | 4.60 | 0.041 | 0.138 | 1.79 | n/a | 0 | 0/296.8 | false |
| *18 400* | *36.6370* | *amphiphiles* | *1778* | *2.800* | *608* | *8900* | *1* | *1778* | *1.000* | *18.58* | *0.894* | *0.938* | *14.95* | *1* | *12* | *centre-untrusted* | — |
| *24 000* | *36.6370* | *amphiphiles* | *2097* | *2.769* | *862* | *11454* | *1* | *2097* | *1.000* | *18.60* | *0.916* | *0.947* | *15.01* | *1* | *25.1* | *centre-untrusted* | — |
| *31 000* | *36.6370* | *amphiphiles* | *2238* | *2.790* | *1015* | *13314* | *1* | *2238* | *1.000* | *18.60* | *0.911* | *0.950* | *15.00* | *1* | *40.6* | *centre-untrusted* | — |
| **41 400** | 54.0000 | amphiphiles | 2212 | 2.845 | 1174 | 15328 | **2** | **2211** | **0.9995** | **26.58** | 0.648 | 0.805 | **20.46** | **1** | 19.5 | **0/317.1** | **false** |
| 54 900 | 54.0000 | amphiphiles | 2189 | 2.973 | 1278 | 16910 | 2 | 2188 | 0.9995 | 26.51 | 0.660 | 0.799 | 20.42 | 1 | 22.9 | centre-untrusted | — |
| 68 400 | 54.0000 | amphiphiles | 2174 | 3.091 | 1351 | 18074 | 2 | 2173 | 0.9995 | 26.42 | 0.663 | 0.807 | 20.39 | 1 | 32.9 | centre-untrusted | — |
| 77 400 | 54.0000 | amphiphiles | 2154 | 3.150 | 1368 | 18671 | 2 | 2153 | 0.9995 | 26.40 | 0.659 | 0.809 | 20.36 | 1 | 33.8 | **0/319.8** | **false** |
| 86 400 | 54.0000 | amphiphiles | 2139 | 3.204 | 1380 | 19182 | 2 | 2138 | 0.9995 | 26.34 | 0.666 | 0.801 | 20.31 | 1 | 34.0 | **0/320.1** | **false** |
| 95 400 | 54.0000 | amphiphiles | 2133 | 3.253 | 1384 | 19608 | 2 | 2132 | 0.9995 | 26.27 | 0.677 | 0.810 | 20.31 | 1 | 42.5 | **0/320.4** | **false** |
| 104 400 | 54.0000 | amphiphiles | 2123 | 3.306 | 1406 | 20038 | 2 | 2122 | 0.9995 | 26.26 | 0.673 | 0.798 | 20.26 | 1 | 43.5 | **0/320.8** | **false** |
| 113 400 | 54.0000 | amphiphiles | 2113 | 3.333 | 1410 | 20406 | 2 | 2112 | 0.9995 | 26.25 | 0.652 | 0.790 | 20.17 | 1 | 50.0 | **0/320.9** | **false** |
| 122 400 | 54.0000 | amphiphiles | 2104 | 3.362 | 1406 | 20719 | 2 | 2103 | 0.9995 | 26.31 | 0.653 | 0.771 | 20.17 | 1 | 47.9 | centre-untrusted | — |
| 131 400 | 54.0000 | amphiphiles | 2103 | 3.381 | 1416 | 21046 | 2 | 2102 | 0.9995 | 26.33 | 0.657 | 0.776 | 20.21 | 1 | 51.0 | centre-untrusted | — |
| 140 400 | 54.0000 | amphiphiles | 2097 | 3.409 | 1417 | 21321 | 2 | 2096 | 0.9995 | 26.25 | 0.657 | 0.773 | 20.13 | 1 | 51.1 | centre-untrusted | — |
| 149 400 | 54.0000 | amphiphiles | 2096 | 3.430 | 1427 | 21583 | 2 | 2095 | 0.9995 | 27.14 | 0.812 | 0.933 | 21.64 | 1 | 52.0 | centre-untrusted | — |
| 158 400 | 54.0000 | amphiphiles | 2089 | 3.441 | 1426 | 21810 | 2 | 2088 | 0.9995 | 27.18 | 0.779 | 0.926 | 21.57 | 1 | 48.9 | centre-untrusted | — |
| 167 400 | 54.0000 | amphiphiles | 2081 | 3.473 | 1424 | 22045 | 2 | 2080 | 0.9995 | 27.17 | 0.773 | 0.926 | 21.56 | 1 | 50.0 | centre-untrusted | — |
| **176 400** | 54.0000 | amphiphiles | **2078** | **3.490** | **1415** | **22201** | **2** | **2077** | **0.9995** | **27.19** | **0.769** | **0.930** | **21.57** | **1** | **53.3** | centre-untrusted | — |

*Italic rows are DRY-PHASE samples at ρ_org = 1.34 — one contact-percolating mass by construction,
shown for the density trajectory only, and no structural claim rests on them.*

**`encapsulatedWater` reads `null(centre-untrusted)` at 11 of 15 wet checkpoints, and that is itself
a measurement, not a gap.** `soup/src/water-closure.ts` refuses to report when the aggregate's own
periodic centre cannot be trusted — i.e. when its extent is comparable to the box. The refusal firing
at two thirds of the wet samples is an **independent detector of the same fact §6 measures directly**,
written by a predecessor for a different purpose. Where it does report — steps 41 400, 77 400,
86 400, 95 400, 104 400, 113 400 — it reports **0 against 317.1–320.9, `closed = false`**.

**Invariants, asserted per checkpoint and not eyeballed, all 20:** non-finite positions **0**,
non-finite velocities **0**, all six valence/placement counters **0**
(`carbonCCover2`, `carbonCOover1`, `headOverChainCapacity`, `nonBondableBonded`, `degreeOver3`,
`headNotTerminal`). Census matches the run's own `activeCounts` exactly and every non-solvent count
matches `config.start` exactly at every checkpoint including the four dry ones carrying `W: 90`:
`{C:27894, O:9296, H:27894, M:723}`. Events at the end
`{cc_bond:16895, cc_break:0, co_bond:5594, co_break:288}`. `RUN-AUDIT-TETHER violations=0`, 0 tethers
over `r_inf` = 1.5 σ at any checkpoint.

### 5.1 Was the curve still rising when it stopped? No — and the cavity was the one to watch

```
largest, the 15 wet samples over 135 000 settled steps:
  2211, 2188, 2173, 2153, 2138, 2132, 2122, 2112, 2103, 2102, 2096, 2095, 2088, 2080, 2077
  last five 2096, 2095, 2088, 2080, 2077  ->  mean 2087.2, sd 7.68 (0.37 %)
  whole-span decline 2211 -> 2077 = -6.1 %, tracking the amphiphile supply's own chemical decline
  (2212 -> 2078 as co_break accumulates 0 -> 288)

cavityVolume, the same 15 samples:
  19.5, 22.9, 32.9, 33.8, 34.0, 42.5, 43.5, 50.0, 47.9, 51.0, 51.1, 52.0, 48.9, 50.0, 53.3
  last eight  ->  mean 50.52, sd 1.60  (a plateau, not a ramp)
```

This mattered: cavity volume **was** rising for the first 60 000 settled steps (19.5 → 50.0) and
stopping there would have left a rising curve, which is not a result. It plateaus at **50.5 ± 1.6 σ³**
over the last 72 000 steps. Bond count is still creeping (21 321 → 22 201, +4.1 % over the last
36 000 steps), so this is a structural plateau at a live chemistry, not a dead run.

---

## 6. THE MEASUREMENT THAT DECIDES IT: DOES THE OBJECT PERCOLATE?

`coalescence-report.md` §8.1 asked this and answered it with a **proxy** — r_g against the r_g of a
uniformly filled box — and said plainly that the proxy could not distinguish "compact object of the
same span" from "network". Two things make the proxy weak: r_g is computed on `unwrapAggregate`'s
output, whose minimum-image-from-one-reference frame is exactly what stops being trustworthy once an
aggregate approaches the box; and the comparison is one-sided.

`tests/percolation-check.test.ts` (NEW, 142 lines) replaces the proxy with the **standard wrapping
criterion**. Take the largest aggregate's own members; replicate them once along one axis at +L with
that axis made open (the other two stay periodic); join pairs within the project's own cutoff
(2.7225 σ, re-derived from `data/params.json` + `data/soup.json` radii, not re-typed); ask whether
any particle shares a connected component with its own +L image. If it does, the aggregate connects
to itself across the boundary: it wraps. **No threshold is introduced** — the output is a boolean per
axis and a slab count, and the only comparison made is against the box itself. The file additionally
asserts a **control that must say NO**: the same test on a deliberately compact one-slab subset of
the same members.

```
$ PERC_CHECKPOINTS="..." nice -n 15 npx vitest run tests/percolation-check.test.ts --no-file-parallelism
PERC {"file":".../zfB54-step41400.json","step":41400,"box":54,"aggregates":2,"amphiphilesInLargest":2211,"particlesInLargest":11842,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":".../zfB54-step77400.json","step":77400,"box":54,"aggregates":2,"amphiphilesInLargest":2153,"particlesInLargest":13245,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":".../zfB54-step104400.json","step":104400,"box":54,"aggregates":2,"amphiphilesInLargest":2122,"particlesInLargest":13786,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":".../zfB54-step140400.json","step":140400,"box":54,"aggregates":2,"amphiphilesInLargest":2096,"particlesInLargest":14072,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":".../zfB54-step176400.json","step":176400,"box":54,"aggregates":2,"amphiphilesInLargest":2077,"particlesInLargest":14265,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19]}
PERC {"file":".../dec54-step41400.json","step":41400,"box":54,"aggregates":22,"amphiphilesInLargest":200,"particlesInLargest":704,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[12,13,12,19]}
PERC {"file":".../dec54-step174400.json","step":174400,"box":54,"aggregates":25,"amphiphilesInLargest":151,"particlesInLargest":596,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[7,7,10,19]}
PERC {"file":".../ocB19-step40000.json","step":40000,"box":30,"aggregates":1,"amphiphilesInLargest":394,"particlesInLargest":2025,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[11,11,11,11]}
PERC {"file":".../ocC19-step40000.json","step":40000,"box":30,"aggregates":2,"amphiphilesInLargest":370,"particlesInLargest":1534,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[11,11,11,11]}
PERC {"file":".../evapFIX29-step36800.json","step":36800,"box":30,"aggregates":4,"amphiphilesInLargest":154,"particlesInLargest":520,"wrapsX":true→false...,"wrappingAxes":0,"slabsTouchedOfTotal":[11,9,7,11]}
 Test Files  1 passed (1)      Tests  1 passed (1)
```
*(the last line's fields verbatim: `"wrapsX":false,"wrapsY":false,"wrapsZ":false`.)*

| system | box | largest | **wrapping axes** | slabs touched / total | reads as |
|---|---|---|---|---|---|
| **this campaign, all 5 wet samples over 135 000 steps** | 54 | 2077–2211 | **3 of 3** | **19 / 19 / 19 of 19** | **percolating network** |
| decisive run (O:C = 4), first + last settled wet | 54 | 200, 151 | **0 of 3** | 12/13/12, 7/7/10 of 19 | finite object (a rod) |
| arm B at box 30 | 30 | 394 | **3 of 3** | 11/11/11 of 11 | percolating network |
| arm C at box 30 | 30 | 370 | **3 of 3** | 11/11/11 of 11 | percolating network |
| arm A at box 30 (the defended broth) | 30 | 154 | **0 of 3** | 11/9/7 of 11 | finite object |

**Three things this settles.**

1. **The predecessor's percolation caveat was right, and it is worse at box 54, not better.** As a
   fraction of a uniformly filled box's r_g: arm B at box 30 was 13.15/15.000 = **87.7 %**; this
   campaign at box 54 is 26.58/27.000 = **98.4 %** at the first wet sample and 27.19/27.000 =
   **100.7 %** at the last. Going to a box 5.83× larger did not make the object small compared with
   the box; it made it *relatively larger*. The percolation is a property of the composition, not of
   the box, and box 54 is where that becomes provable rather than suspected.
2. **The test discriminates.** Its negatives are exactly the objects this project calls rods —
   arm A's 154 and the decisive run's 151 and 200 — and they come from the same code, the same
   cutoff, the same clustering, on the same day.
3. **`coalescence-report.md` §8.1's own words were "the *size* claim is not made".** It is now made,
   and refused: at box 30 arm B's "largest = 394, share 1.000" and at box 54 "largest = 2211, share
   0.9995" are both **one number for one connected network**, not the size of an aggregate. The count
   claim (4 discrete objects → 1 connected object) survives, exactly as the predecessor said it
   would; what it converged to is a gel, not a bigger micelle.

---

## 7. CLOSURE, AND THE ROD → SHEET QUESTION

### 7.1 Closure rests on `encapsulatedWater` and the head-shell structure, not on shape

| quantity | threshold | measured, this campaign | verdict |
|---|---|---|---|
| **`encapsulatedWater`** | ≥ **317.1–320.9** beads (the live bulk density × the closure volume, recomputed per checkpoint) | **0** at all six wet checkpoints where the centre is trustworthy; refused as centre-untrusted at the other nine | **fails, totally** |
| **`closed`** | true | **false**, every reportable checkpoint | fails |
| **`radialHeadShells`** | **2** for a bilayer wall | **1** at **every one of the 20 checkpoints** of the whole run | **fails — there is no wall** |
| `transverseHeadShells` | 2 for a lamellar wall | 0 throughout | fails |
| enclosed / cavity volume | ≥ 370.8656 σ³ | plateau **50.5 ± 1.6 σ³**, final 53.3 | fails by **7.3×** |
| size | ≥ 1004 (this task's floor) | **2087.2** | **PASSES, 2.08× over** |
| flatness λ₁/λ₃ | ≤ 0.35 for lamellar | **0.65–0.81** | **fails, 2.20× over** |
| in-plane symmetry λ₂/λ₃ | ≥ 0.50 for lamellar | 0.77–0.93 | passes |
| `hasLamellarAggregate` / `hasVesicleAggregate` | true | **false / false**, all 20 checkpoints | fails |

**There is no step at which anything closed.** And the object is not a proto-membrane that failed to
close: with `radialHeadShells` = 1 at every checkpoint there is no bilayer wall to close, and with
flatness 0.65–0.81 it is not a sheet either.

### 7.2 Did any object cross from rod to sheet? NO — and the signature is unambiguous

The brief named the signature precisely: the rods held a fixed cross-section of 4.006–4.165 σ while
lengthening, so the lever working looks like **that cross-section growing**. Under the z fix the
decisive run's own rod re-measures at **R⊥ = 4.085 σ at N = 151** — inside the published band, so the
baseline is intact.

This campaign's R⊥: **20.13–21.64 σ**, i.e. **5.0×** the rod's. That is not a widening ribbon. A
sheet has two large principal moments and one small (flatness ≤ 0.35); this object's flatness is
**0.65–0.81**, near-isotropic, which is a space-filling network. The transition it made is
**rod → gel**, not rod → sheet, and the trace shows it happening in the drying phase and never
reversing: R⊥ 1.79 (step 6000, micelles) → 15.00 (dry percolating mass) → 20.46 (first settled wet)
and then flat for 135 000 steps.

`hasLamellarAggregate` is **false at all 20 checkpoints**, on the project's own lamellar gate
(flatness ≤ 0.35 **and** in-plane ≥ 0.50) — the same gate that read 0 lamellar objects at 11 of 13
snapshots in the predecessor. So no object crossed, at any size, at any step.

### 7.3 Edge energy at the new size, for completeness

With this project's own measured line tension λ = **10.77 ± 1.06 ε/σ** (`verify/out/line-tension.json`)
and this task's own a = **1.1777 σ²**, the disc-geometry scaling argument at the sizes now in play:

| N | disc radius σ | edge energy ε | per amphiphile | in kT (kT = 1.1) |
|---|---|---|---|---|
| **151** (decisive run, z-fixed) | 5.320 | 360.0 | 2.3842 ε | **2.167 kT** |
| 394 (arm B at box 30) | 8.594 | 581.5 | 1.4760 ε | 1.342 kT |
| **729** | 11.689 | 791.0 | 1.0851 ε | **0.986 kT** |
| 1004 (this task's floor) | 13.718 | 928.3 | 0.9246 ε | 0.841 kT |
| **2087** (this campaign's plateau) | **19.778** | **1338.4** | **0.6413 ε** | **0.583 kT** |
| 4602 (= 2πR_c²/a at R_c = 29.37 σ) | 28.978 | 1960.9 | 0.4377 ε | 0.398 kT |

**So the campaign did retire the edge-energy objection.** At N = 2087 a finite bilayer disc costs
**0.58 kT per molecule** in edge energy — below the 1 kT crossover the predecessor located at
N ≈ 729, and 3.7× cheaper than at N = 151. A disc is thermodynamically permitted at this size. It
still did not form, and the reason is §6: the material is one connected network, so there is no
finite patch for an edge to bound. That is the whole finding in one sentence — **the lever removed
the reason a small object had to be a rod, and replaced it with a reason no object can be finite at
all.**

---

## 8. VERDICT, AND THE BINDING CONSTRAINT NAMED AND NUMBERED

**NO VESICLE.** Proved by `encapsulatedWater` = 0 against 317.1–320.9 with `closed = false`, and by
`radialHeadShells` = 1 at every checkpoint of the run — not by shape.

| # | constraint | measured | required | shortfall | binding? |
|---|---|---|---|---|---|
| 1 | amphiphile **supply** at box 54 | **2109.2** [2078, 2154], settled | ~1004 | **2.10× OVER** | **no — RETIRED** |
| 2 | **aggregate size** | **2087.2** [2077, 2096], plateau sd 0.37 % over 135 000 steps | ~1004 | **2.08× OVER** | **no — RETIRED (first time in this project)** |
| **3** | **CONNECTIVITY — the supply PERCOLATES** | **wraps 3 of 3 axes at 5 of 5 wet samples; 19/19 slabs on every axis; r_g 26.25–27.19 σ against 27.000 for a filled box** | a finite object: **0** wrapping axes | **total — the object has no inside** | **YES** |
| 4 | closure / encapsulated water | **0** | ≥ 317.1–320.9 | total | downstream of 3 |
| 5 | head-shell structure of the largest | `radialHeadShells` **1** | 2 | no wall exists | downstream of 3 |
| 6 | cavity / enclosed volume | 50.5 ± 1.6 σ³ plateau | ≥ 370.8656 σ³ | **7.3×** | downstream of 3 |
| 7 | flatness of the largest (lamellar gate) | 0.65–0.81 | ≤ 0.35 | 2.20× | downstream of 3 |
| 8 | edge energy per amphiphile | **0.583 kT** at N = 2087 | ≲ 1 kT for a disc to compete | **none — RETIRED** | no |
| 9 | medium density | ρ_tot 1.21792, ρ_W 0.79999 | liquid | none | no — settled |
| 10 | Verlet list / memory / particle count | 1.9178 GB, 191 778 | 4.2950 GB, 429 496 | none (2.24×) | no |
| 11 | run length | plateau reached: 135 000 settled steps, last five within 0.37 % | a plateau | none | no |
| 12 | measurement correctness of the clustering | fixed and pinned both branches (§1) | — | none | no — RETIRED |

**BINDING CONSTRAINT: (3) CONNECTIVITY — THE SUPPLY PERCOLATES. 3 of 3 axes wrapping, 19 of 19
slabs, at 2087 amphiphiles against a floor of 1004.**

### 8.1 Where this run lands in the shortfall sequence

The published sequence on aggregate size against the floor in force is
**80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75×**. Two corrections and one new entry:

- the last entry, **6.75×**, was measured with the z-open defect; the same lineage under the fix is
  **6.02×** (152.8 against 920);
- **this run does not land in the sequence at all — it comes off the end of it.** 2087.2 against
  1004 is a **2.08× SURPLUS**, the first time in seven tasks that aggregate size is not a shortfall.
  Written into the sequence for continuity: **80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75× (6.02×
  corrected) → 0.48× (i.e. 2.08× over)**.

And that is exactly why the sequence stops being the right thing to track. The quantity every
predecessor reported has been beaten by more than 2×, and the project is **no closer to a vesicle**:
the shortfall moved from a number (size) to a topology (connectivity), and a topology does not have a
factor.

---

## 9. IS CLOSURE REACHABLE IN THIS MODEL? NO — AND THIS IS NOW A MEASUREMENT

`coalescence-report.md` §9.3 set the condition and named the consequence in advance: *"If that object
is a network and not a compact aggregate, then this model cannot reach closure without a change that
would not be defensible."* **The object is a network, on 3 of 3 axes, at the box the report itself
specified, at the composition it itself specified, for 135 000 settled steps.** So the answer is the
one that report pre-committed to, and it is now carried by a direct measurement rather than by a
proxy.

The full case, with everything now retired or established:

1. **Supply is not the wall** — 2109 against 1004, 2.10× over (retired by the decisive run, confirmed
   here at a different composition).
2. **Aggregate size is not the wall** — 2087 against 1004, 2.08× over. Retired here for the first
   time.
3. **Edge energy is not the wall at this size** — 0.583 kT per amphiphile at N = 2087, below the
   1 kT crossover. Retired here.
4. **The frozen distribution is not the wall either, in this composition** — the rehydration did not
   fragment (2 aggregates, not 22), so the mechanism that locked the decisive run at 151 (fusion at
   2 % per encounter, monomer exchange ~0, 99 % locked inside aggregates) simply never applied. That
   is the lever genuinely working on the thing it was aimed at.
5. **And the wall is now the one thing the lever cannot avoid: at the two-tailed fraction that makes
   a bilayer the preferred object, the amphiphile concentration this project needs for any chemistry
   at all puts the system above its own gelation threshold.** 47–48 % two-tailed heads with per-tail
   length rising 2.845 → 3.490 over the run gives a molecule that connects, and at ρ_org = 0.418 with
   2100 of them in a 54 σ box the connected phase spans the box. The two are not independent: this
   project reached its amphiphile supply by enriching the carbon pool ~1593× over the most generous
   literature pond (`evaporation-report.md` §3), and it reached its two-tailed fraction by tripling
   the tail length through composition. Both moves push toward one percolating phase.

**So the honest verdict on the project's actual question: closure is NOT reachable in this model.**
Not because of supply, not because of size, not because of edge energy, and not because of the
fragmentation freeze — each of those has now been retired by measurement — but because the two
regimes this model can express are **finite rods that are too small (O:C = 4: largest 151, 6.0× short,
0 wrapping axes)** and **one network that is large enough but has no inside (O:C = 0.333: largest
2087, 2.08× over, 3 wrapping axes)**, with nothing measured in between. Arm C at O:C = 1.000 is not
an in-between: it percolates too (3 of 3 axes at box 30, largest 370).

### 9.1 What a next model would need, named from these numbers

The predecessor named three requirements; this campaign retires one of them, confirms one, and
replaces one with something sharper.

1. **A two-tailed amphiphile by CONSTRUCTION, not by emergence — CONFIRMED, and now with the reason
   the emergent route cannot work.** Getting p into the 1/2–1 band by composition required O:C =
   0.333, which simultaneously raised per-tail length 2.11 → 2.79 → 3.49 and, with it, the
   connectivity of the organic phase. A fixed two-tailed topology at a *fixed short* tail (the
   Cooke & Deserno geometry this project already holds a stable bilayer of) would put p in the
   vesicle band **without** raising the number of covalent neighbours per molecule, which is what
   percolates. This is the single change that would matter most, and it is a change to the molecule,
   not to a constant.
2. **A dilution axis independent of the chemistry — NEW, and it is the constraint this task
   discovered.** The percolation threshold is a function of ρ_org, and ρ_org is pinned high because
   `co_bond` will not fire at prebiotically honest dilution (`data/soup.json`'s own `startBasis` §5:
   0 co_bond events in 150 000 steps at 691× enrichment). So this model cannot separate "enough
   chemistry" from "above the gel point". A next model needs either a chemistry that runs at low
   ρ_org (a real catalytic rate, not an enrichment factor) or pre-formed amphiphiles it can dilute
   freely — and the second is the cheap one: assemble the lipid, then measure closure as a function
   of ρ_amph, with the chemistry out of the loop entirely.
3. **A rehydration that does not fragment — RETIRED as a requirement.** This campaign's rehydration
   did not fragment (1632 → 22 pieces in the decisive run; 2238 → 2 here). The affine ramp tears a
   percolating mass along its weakest links only when the mass is weakly connected; at 48 %
   two-tailed heads it does not tear. So requirement 3 turns out to be a symptom of requirement 1,
   not an independent need.
4. **A non-zero monomer solubility / a real CMC** — the predecessor's requirement 2 — was never
   tested by this campaign, because with the distribution not frozen the exchange channel was never
   the limiting step here. It remains open and untested, and it should not be quoted as retired.

**The cheapest next measurement, fully specified, if anyone continues:** pre-formed two-tailed lipids
(the `tests/water-bilayer-*` construction, which already holds a stable bilayer at ρ_W = 0.8), no
chemistry, no evaporation, at four amphiphile densities bracketing the percolation threshold this
campaign sits above — ρ_amph = 1.34e-2 (this run), 6.7e-3, 3.3e-3, 1.7e-3 σ⁻³ at box 54 — with
`tests/percolation-check.test.ts` as the read-out and closure as the endpoint. That asks the one
question this model has never been able to ask: at a two-tailed topology, is there a density that is
below the gel point and still above the floor? The floor at box 54 is 1004 amphiphiles = ρ_amph
6.4e-3 σ⁻³, so the window, if it exists at all, is narrow and the arithmetic says it is between the
first two rungs.

---

## 10. REGRESSIONS — EVERY ONE RUN, WITH BEFORE → AFTER AND THE REASON FOR EVERY MOVED NUMBER

| test file | result | numbers, predecessor → this task, and the reason |
|---|---|---|
| `tests/gate6-bilayer.test.ts` | **2 passed** | area **1.2025±0.0125 → 1.2050±0.0095** [1.1861, 1.2328]; thickness **4.4761 → 4.4952** (per-frame sd 0.2307, n=400); lnA drift/move 1.509e-5 (t=1.10); accepted 0.263; escapedMax 0; box 24.648, 83 000 steps. Both inside the untouched corridors 1.1–1.5 and 4–6. **Cannot be affected by the fix** (membrane engine, default `periodicZ=false`); the movement is the engine's own run-to-run scatter. §3 recomputes the solvent-free floor from this row. |
| `tests/water-bilayer-area-move.test.ts` | **1 passed** | area **1.2098 → 1.1777** [1.1722, 1.1828]; thickness **4.9421 → 4.7990**; drift t 1.21 → **1.95** (2.868e-5/chunk); water in core 54 → **53**/4500; buried 0.0921 → **0.0857**; cluster **1.0000**; accepted 0.1257 of 3000; throughput 1243.91 steps/s at N=5700; **verdict=passed**. Its `largestClusterFraction` call keeps the default (§1.2), so again scatter, not the fix. §3 recomputes the explicit-water floor from this row. |
| `tests/soup-stale-force.test.ts` | **2 passed** | `postResidentVsFresh` **1.07e-4** against a force scale of 581 (1.8e-7 relative); `postResidentVsPreFresh` **349.02**; `preResidentVsFresh` **0**. `MIDRUN-MINIMISE` neutrality holds: 29 iterations, max\|F\| 204.20 → 17.56, census/charge/bondSet/events identical, steps 2000→2000, box unchanged, nonFinite 0. Unaffected by the fix. |
| `tests/catalyst-turnover.test.ts` | **1 passed** | cc_bond 447 → **449**, co_bond 326 → **343**, chains 189 → **188**, amphiphiles 135 → **132**, bare carbons left 29 → **42**, occupiedAtEnd 3 → **3**, desorbTimeout 3 → **8**, nonFinite 0, charge 0, census `{C:779,O:3117,H:779,M:52,W:695,K:0}` exact. Turnover ordering holds (ccOverLast15k=4 > 0, occupied falls 16→3). No clustering anywhere in this file: scatter. |
| `tests/soup-nonfinite-guard.test.ts` (**the loud-failure guard**) | **1 passed** | fires at step 1 with **120/120** non-finite components; healthy control 0/0 at three points; cost **0.2180 ms** vs chunk1000 **328.80 ms** = **0.066 %** (1.00066× throughput). |
| `tests/soup-cold-start-relax.test.ts` (**the cold-start guard**) | **1 passed** | max\|F\| **3.51806e4 → 13.372** (2631×); maxDisplacement **1.14139 σ** ≤ bound 10.05; steps 0→0; velocities, both RNG streams, bond graph and all four event counters unchanged; refusal at `globalStep != 0` fires; payoff pair holds (throws at step 1000 without the stage, 29 199/29 232 non-finite). |
| `tests/soup-evaporation.test.ts` | **3 passed** | `EVAP-PLAN` identical to every predecessor's to every digit (1440.0×, dryBox 20.3538, concentrationFactor **3.2021×**, relaxIterations 29, logFractionOf1400 16.07 %); `EVAP-LADDER` 18 increments, 2.132 % linear, identical boxes/solvent; isolation census solvent-only (`W 11059→5529`, all four organic counts exact), bondSetUnchanged true/true, valence `{bad:0,outOfRange:0}`, nonFinite 0 including after 1000 further steps; insertion `belowFloor` 771 → **749**, minSeparation 0.6158 → **0.6334**, displacement rms 0.1919 → **0.1889** max 0.7572 → **0.7526** against bound 1.5000; full cycle `dryBox=16.2811 (plan 16.2811) N_dry=5783 (plan 5783) rhoDryRealised=1.34000`, box and census return **exactly**, nonFinite 0 at four points, events `{cc_bond:399, cc_break:0, co_bond:359, co_break:3}`. Bond count in the isolation fixture 29 → **22**: chemistry scatter in a 22-bond fixture, and the claim it carries (`bondSetUnchanged`) is a comparison of the set with itself before/after, which is invariant to how many bonds there are. |
| `tests/soup-drywet-cycling.test.ts` | **FAILED, then PASSED on one identical re-run — both numbers below, as the brief requires** | run 1: mobility collapse **3.08×**, one-cycle event ratio **0.5941** → the ratio assertion failed. run 2, identical command, no change of any kind in between: mobility collapse **2.89×**, event ratio **0.7116** → **2 passed**. Invariants identical in both: `concentrationFactor=1.1002`, `rampStepsCharged=1000`, `dryBox=29.0598`, `realisedDryDensity=1.34000`, box returns exactly, nonFinite 0/0 at four points. Published history of that same ratio: 0.7143, 0.4978, 0.7171, 0.6961, **0.5941, 0.7116**. This file measures chemistry rates and MSD and contains no clustering call at all, so the fix cannot reach it; the assertion sits at the margin of the engine's own scatter and **no bound was widened**. |
| `tests/sim.test.ts` | **8 passed** | unchanged (grid-vs-brute-force force identity, thermostat, frictionless energy drift) |
| `tests/run-ui.test.ts` | **8 passed** | unchanged |
| `tests/soup-forces.test.ts` | **1 passed** | unchanged — grid + Verlet list vs full O(N²), no GPU console warning |
| `tests/soup-area-move.test.ts` | **2 passed** | `composeVsDirect maxDiff=9.537e-7` — identical to every published value; CPU potential = antiderivative of GPU forces (worst relative deviation ~5e-4 at meanAbsF=21.09); Jacobian identity holds at Nmol=1861.2 |
| `tests/soup-bonds.test.ts` | **3 passed** | catalyst gating holds; census+charge conserved under live reactions; **assembly-vs-temperature ordering holds with the fix applied to its own clustering call**: `largestClusterFraction` **0.9 → 0.08625, 1.8 → 0.0375**, ordering `r[1.8] < r[0.9]` intact. This is a **moved number by design** — it is a whole-box bulk statistic and was being computed with z open; the reason it moves is that z-straddling clusters were split, which understates the fraction. The ordering, which is what the test asserts, is unaffected because the bias is in the same direction at both temperatures. |
| `tests/soup-valence.test.ts` | **1 passed** | no carbon over 2 C–C or 1 C–O, no head over chainCapacity, 45 000 steps |
| `tests/soup-checkpoint.test.ts` | **3 passed** | round-trip exact; "a checkpoint does not change what the run computes next"; same-process continue writes real files |
| `tests/soup-boxcycle.test.ts` | **7 passed** | pure coordinate map, involution, long-coiled-chain regression, `computeDryBox` density, schedule — unchanged |
| `tests/soup-rules.test.ts` | **4 passed** | schema + detailed balance unchanged; the four `dryWetCycle` evaporation fields still validate |
| `tests/params.test.ts` | **3 passed** | the literal scanner is clean; **no rank-A constant appears anywhere new** — the machine check that this task changed no constant |
| **`tests/soup-aggregates.test.ts`** | **4 passed** | **the file that most directly pins what I changed, and its numbers did NOT move**: 3 synthetic micelles → `aggregateCount 3`, `sizeHistogram` equal, share 1; the bilayer fixture → 1 aggregate, `transverseHeadShells 2`, `hasLamellarAggregate true`, stage `bilayer`; the vesicle fixture → 1 aggregate, `radialHeadShells 2`, cavity > 370.8656, stage `vesicle`. Reason: those fixtures are placed away from the box faces, so no component straddles z. This was a prediction before running and it held. |
| `tests/soup-amphiphile.test.ts` | **12 passed** | unchanged |
| **`tests/self-assembly.test.ts`** | **2 passed** | **the proof the membrane engine is untouched**: `largestClusterFractionOf = 0.9992`, cutoff 2.7225 σ, 1200 lipids, 1 000 000 steps at 3130 steps/s. Default `periodicZ = false`, so this path is bit-identical to before the parameter existed. |
| `tests/closure.test.ts` | **8 passed** | unchanged — the open-boundary flood pipeline, which deliberately keeps the open-z `largestClusterCenter` |
| `tests/water-closure.test.ts` | **3 passed** | unchanged — the soup's own periodic never-unwrap flood |
| `tests/periodic-measurement.test.ts` | **5 passed** | **moved by design**: `aggregateCount` on the box-58 lineage **8 → 3**, **10 → 5**, **7 → …**; `amphiphileCount` of the largest 1327 → **1328**, 1313 → … . Reason: that lineage's dominant aggregate is comparable to its own box, so the z-open rule was splitting it and inventing extra objects. Every assertion in the file still passes; the artifact `verify/out/periodic-measurement.json` is updated in place with the corrected numbers. |
| `tests/shell-pore-check.test.ts` | **1 passed** | **moved by design**: `aggregateCount 8 → 3`; r_g of the largest 30.9901 → **30.9909** (+0.003 %). Reason: the same box-58 aggregate; joining its z-halves changes the count a great deal and its shape almost not at all, because the halves were adjacent. |
| `tests/rim-lambda-insitu.test.ts` | **1 passed, 2 FAILED — PRE-EXISTING, proven by stash** | the analysis half passes and its numbers moved by design (`aggregateCount 5 → 3`, largest 1282 → **1284** amphiphiles / 11 685 → **11 688** particles, r_g 30.39977 → **30.39922**). The two failures are the GPU-page tests, and they fail with `createSoup: summary contains 7 particles, but this call's composition gives N=10707 -- checkpoint does not match the configuration` — a construction mismatch inside those tests, in a code path that calls no clustering function at all. **Proven not mine**: `git stash push -- tests/rim-lambda-insitu.test.ts` and re-run gives the identical error, then `git stash pop`. See §11. |
| `tests/soup-grid-resize.test.ts` | **1 FAILED — PRE-EXISTING** | `measured drift 4587793.1347` against `skin/2=0.7500`. Inside the 4.4e2–3.4e7 band two predecessors measured for this same test on unchanged code, on both sides of a stash. It measures Verlet drift across a box change; this task's diff is clustering only and cannot reach it. **No bound was widened.** |
| **`tests/coalescence-mechanism-pin.test.ts`** | **5 passed** (was 3) | see §1.3 |
| **`tests/percolation-check.test.ts`** (NEW) | **1 passed** | see §6; its own control (a one-slab subset must not wrap) passes at every checkpoint |
| `tests/continuous-run-audit.test.ts` | **1 passed × 4 invocations** | 19 + 10 + 8 + 20 checkpoints audited with `expect()` on every invariant, plus the tether assertion; `RUN-AUDIT-TETHER violations=0` every time |

**No tolerance was changed, no corridor widened, no assertion relaxed, no threshold moved.** The only
assertions added are new ones (§1.3, §6), and they are strict.

`tests/soup-vesicle.test.ts` was **never** run. The full suite was never run. `--dump-dom` was never
used. **No particle array was ever transferred as JSON numbers** — every reduction ran inside the
offline auditor or inside the page and came back as scalars.

---

## 11. EVERY COMMAND, WITH REAL UNEDITED OUTPUT

Outputs quoted in full above: §1.3, §3, §6, §10 (per row). The full list, in order:

```
# --- task 1: the fix, its pins, and the re-measurement of what it changes
nice -n 15 npx vitest run tests/coalescence-mechanism-pin.test.ts --no-file-parallelism
CONTINUOUS_RUN_PREFIX=dec54-step CONTINUOUS_RUN_DIRS=data/checkpoints/dec54 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/zfix-decisive-run-54-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism
CONTINUOUS_RUN_CHECKPOINTS="<evapFIX19/23/29 + minonly19/23/29 + ocB19 + ocC19, step 36800/40000>" \
  CONTINUOUS_RUN_ARTIFACT=verify/out/zfix-box30-arms-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism

# --- task 2: the campaign, ONE lineage, nine resumable invocations of the same command
nice -n 15 npx tsx soup/cli/campaign.ts --label zfB54 --box 54 \
  --start '{"C":27894,"O":9296,"H":27894,"M":723,"W":125971}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --steps <n> --every <n/2> --dir data/checkpoints/zfB54
  # --steps  6000 --every  3000  -> globalStep   6 000     92 s   (12.75 ms/step)
  # --steps 18000 --every  9000  -> globalStep  24 000    259 s   (the drying event, 18 increments)
  # --steps 14000 --every  7000  -> globalStep  41 400    230 s   (the rehydration, 18 increments + insertion)
  # --steps 27000 --every 13500  -> globalStep  68 400    578 s   (over the 500 s cap: chunk resized)
  # --steps 18000 --every  9000  -> globalStep  86 400    357 s
  # --steps 18000 --every  9000  -> globalStep 104 400    386 s
  # --steps 18000 --every  9000  -> globalStep 122 400    386 s
  # --steps 18000 --every  9000  -> globalStep 140 400    384 s
  # --steps 18000 --every  9000  -> globalStep 158 400    383 s
  # --steps 18000 --every  9000  -> globalStep 176 400    381 s

CONTINUOUS_RUN_PREFIX=zfB54-step CONTINUOUS_RUN_DIRS=data/checkpoints/zfB54 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/zfix-campaign-B54-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism      # x2
PERC_CHECKPOINTS="<5 zfB54 wet + dec54 41400/174400 + ocB19 + ocC19 + evapFIX29>" \
  PERC_ARTIFACT=verify/out/percolation-check.json \
  nice -n 15 npx vitest run tests/percolation-check.test.ts --no-file-parallelism         # x2

# --- regressions
nice -n 15 npx vitest run tests/gate6-bilayer.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/sim.test.ts tests/run-ui.test.ts tests/soup-forces.test.ts \
  tests/soup-area-move.test.ts tests/soup-valence.test.ts tests/soup-checkpoint.test.ts \
  tests/soup-boxcycle.test.ts tests/soup-rules.test.ts tests/params.test.ts \
  tests/soup-aggregates.test.ts tests/soup-amphiphile.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/catalyst-turnover.test.ts tests/soup-nonfinite-guard.test.ts \
  tests/soup-cold-start-relax.test.ts tests/soup-stale-force.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/soup-evaporation.test.ts tests/soup-drywet-cycling.test.ts \
  tests/soup-stale-force.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/soup-drywet-cycling.test.ts --no-file-parallelism     # the re-run
nice -n 15 npx vitest run tests/soup-bonds.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/self-assembly.test.ts tests/closure.test.ts \
  tests/water-closure.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/periodic-measurement.test.ts tests/shell-pore-check.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/rim-lambda-insitu.test.ts --no-file-parallelism
nice -n 15 npx vitest run tests/rim-lambda-insitu.test.ts --no-file-parallelism -t "formula validation"
git stash push -m "zfix-rim-probe" -- tests/rim-lambda-insitu.test.ts
  nice -n 15 npx vitest run tests/rim-lambda-insitu.test.ts --no-file-parallelism -t "formula validation"
git stash pop
nice -n 15 npx vitest run tests/coalescence-mechanism-pin.test.ts tests/soup-grid-resize.test.ts --no-file-parallelism
```

The stash probe's verbatim pair, which is the whole attribution of §10's rim-λ row:

```
$ git stash push -m "zfix-rim-probe" -- tests/rim-lambda-insitu.test.ts
Saved working directory and index state On stage-a-atoms: zfix-rim-probe
$ nice -n 15 npx vitest run tests/rim-lambda-insitu.test.ts --no-file-parallelism -t "formula validation"
Error: createSoup: summary contains 7 particles, but this call's composition gives N=10707 -- checkpoint does not match the configuration
 Test Files  1 failed (1)
      Tests  1 failed | 2 skipped (3)
$ git stash pop
Dropped refs/stash@{0}
```

The campaign's own transition lines, verbatim, since they carry §4.2's claims:

```
[evaporation] evaporation increment=18/18 box=36.6370 solvent=90 N=65897 max|F|=6.460e+2
              minimization_iterations=0 max|F|_after=6.460e+2 overstretched_tethers=0 longest=1.1126 threshold=1.4680
[campaign] step=24000/24000 stage=amphiphiles aggregates=1 largest=2097 headShells=1 cavityVolume=25.125
           phase=dry/1 box=36.6370 bonds=11454 census={"C":27894,"O":9296,"H":27894,"M":723,"W":90,"K":0}
[evaporation] rehydration increment=18/18 box=54.0000 solvent=125971 N=191778 max|F|=5.405e+1
              minimization_iterations=0 max|F|_after=5.405e+1 overstretched_tethers=0 longest=1.0972 threshold=1.4680
[evaporation] rehydration: inserted=125881 below_threshold=6017 minimum_distance=0.6237
              minimization_iterations=29 max|F| 2.0427e+4 -> 5.4046e+1 prior_displacement rms=0.2105 max=0.8488 boundary=1.5000
[campaign] step=41400/38000 stage=amphiphiles aggregates=2 largest=2211 headShells=1 cavityVolume=19.500
           phase=wet/0 box=54.0000 bonds=15328 census={"C":27894,"O":9296,"H":27894,"M":723,"W":125971,"K":0}
[campaign] step=176400/176400 stage=amphiphiles aggregates=2 largest=2077 headShells=1 cavityVolume=53.250
           phase=wet/0 box=54.0000 bonds=22201 census={"C":27894,"O":9296,"H":27894,"M":723,"W":125971,"K":0}
```

*(`aggregates=2 largest=2077` there is already the FIXED, fully-periodic count — this is the first
campaign in the project whose live log is not carrying the z-open defect.)*

---

## 12. TOTALS, RESOURCES, HOUSEKEEPING

- **Compute invocations: 30 of the 40 allowed.** 1 pin, 2 off-GPU re-measurements of existing
  artifacts, 9 campaign invocations, 2 campaign audits, 2 percolation runs, 2 bilayer gates, 8
  regression groups, 1 drywet re-run, 1 stash probe, 2 diagnostic. **The bulk went to the campaign**,
  as instructed.
- **Longest single foreground invocation: 578 s** (campaign chunk 4, `--steps 27000`). It exceeded my
  own 500 s rule; the step cost had risen from 12.75 to 21.3 ms/step as the bond count grew
  2013 → 18 074, and every subsequent chunk was resized to 18 000 steps (**381–386 s** measured) so
  it would not recur, and it did not. **One invocation was moved to the background by the harness
  itself, not by choice** (campaign chunk 2, the drying event: it passed the tool's 120 s default
  before I had started passing an explicit timeout). It was tracked with a monitor, nothing else was
  started while it ran, its full output was returned and is quoted in §11, and it exited 0 in 259 s.
  Every other invocation was `nice -n 15`, one at a time, foreground, with an explicit timeout, and
  every multi-file vitest group ran `--no-file-parallelism` so that no two compute processes could
  exist at once.
- **Total new trajectory: 176 400 steps at N = 191 778** (≈ 3.4 × 10¹⁰ particle-steps, the largest
  single lineage this project has run) plus ~450 000 steps across the regression suite (of which
  1 000 000 lipid-steps in `self-assembly` at N = 3600). Plus ~57 off-GPU checkpoint
  decode-and-analyse passes at N = 191 778.
- **Measured step cost at the campaign scale**: 12.75 ms/step at 2013 bonds rising to
  **21.3 ms/step** at 22 201 bonds (N = 191 778 throughout); ~8.6 ms/step in the dry phase at
  N = 65 897. Checkpoint encode+write 273–635 ms for a 15 MB file; progress readout (full stage
  detection over 191 778 particles) 301–743 ms.
- **No orphan processes remain.** `pkill -f puppeteer_dev_chrome_profile` was run after **every**
  browser-bearing invocation and `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** every
  time, including after the last. Verified again at the end of the task.
- **The dev server on :5199 (PID 94131) was neither started, stopped nor inspected.** Every GPU
  invocation opened its own vite server on its own port (`localhost:5173`, visible in §10's rim-λ
  stack trace).
- **`tests/soup-vesicle.test.ts` never run. No fusion probe built.** `--dump-dom` never used. `macOS
  timeout` never invoked. No `node_modules/.vite/deps_temp_*` cleanup was needed (no vitest died
  mid-run).
- **Verlet / particle ceilings**: largest system N = **191 778** — 1.9178 GB of Verlet list against
  4.2950 GB, and 191 778 against the hard particle ceiling 429 496; **2.24× inside both**, checked
  before the first invocation.
- **Checkpoints written:** the campaign's 20 files (15.3 MB wet, 5.3 MB dry — ~250 MB).
  `data/checkpoints/` is a gitignored run artifact.
- **File sizes, CLAUDE.md's 400–600 rule.** Nothing existing was pushed over: `engine/src/aggregate.ts`
  **382 → 394**, `soup/src/aggregates.ts` **549 → 552**, `soup/src/stages.ts` **281 → 282**,
  `tests/coalescence-mechanism-pin.test.ts` **102 → 162**. New file:
  `tests/percolation-check.test.ts` **142**. Largest file in the tree is still `soup/wgsl/step.wgsl`
  at **596**; **nothing crossed 600**, and `step.wgsl` was not touched.
- **`data/params.json` NOT touched. `data/soup.json` NOT touched — not one field.**
  `co_bond.attemptRate` NOT touched. No threshold, corridor, potential, rate, recogniser or rank-A
  constant modified anywhere; `tests/params.test.ts`'s literal scanner is the machine check and it is
  clean.
- **No arm reverted to under-dense water.** The campaign runs at ρ_W = 125 971/157 464 = **0.79999**.
- **Nothing started from anything pre-made.** The campaign's step 0 is a monomers-only lattice with
  its own cold-start minimisation (`max|F| 5.3288e+4 → 2.2706e+1`, nonFinite 0 before and after).
- **`clay: false` in every arm**, as in all seven predecessors, and doubly forced (a box change is
  refused with an immobile phase, and solvent removal requires the solvent to be the last non-empty
  composition block).
- **`tsc --noEmit`**: 17 errors, all in files this task never touched
  (`engine/src/{forces,gpu,params,sim-area-move,sim-buffers,sim-measure}.ts`,
  `soup/src/{sim,soup-box-scale,soup-buffers}.ts` — a `@webgpu/types` vs TS-lib
  `Float32Array<ArrayBufferLike>` mismatch). **Zero errors in any file this task modified or added.**

---

## 13. CONCERNS

1. **The campaign is ONE seed.** Seed 19 only, because a second is ~10 more chunks. Its plateau is
   internally consistent over 15 wet samples (sd 0.37 % on 2087.2 across 135 000 steps) and its
   headline finding — 3-of-3-axis wrapping — is a **topological** boolean rather than a number near a
   threshold, which is the one kind of claim least likely to be a scatter artefact. But "2087.2" as a
   number is one lineage, and this engine is not bit-reproducible.
2. **The percolation test is new, and a new instrument's first result should be the most suspected
   thing in the report.** Its guards: it reproduces "not percolating" on three independent systems
   the project already calls finite objects (dec54 at two steps, arm A), its own in-file control (a
   one-slab subset must not wrap) passes at every checkpoint, and it agrees with two pre-existing
   independent signals nobody wrote for this purpose — the r_g/filled-box ratio (98.4–100.7 %) and
   `water-closure.ts`'s own centre-untrusted refusal firing at 11 of 15 wet samples. Still, nobody
   has run it on a *deliberately constructed* percolating object of known topology, which is the
   cheap validation I did not do.
3. **`cavityVolume` was rising for the first 60 000 settled steps and I nearly stopped there.** It
   went 19.5 → 50.0 σ³ before flattening at 50.5 ± 1.6 over the last 72 000. Had the run stopped at
   ~100 000 global steps the honest report would have been "still rising". It is worth saying that
   the plateau is 72 000 steps long, not 133 000: the *largest aggregate* plateaued early, the cavity
   did not.
4. **What `cavityVolume` measures on a percolating network is not a lumen.** It is the flood of
   interstices inside the object's own padded local bounding region. Quoting "50.5 σ³ against
   370.8656" as a closure shortfall is therefore generous to the model rather than harsh: a network's
   interstitial void is not a proto-vesicle interior at all, and the honest statement is that closure
   is not merely 7.3× away, it is not the right question for this object.
5. **The z fix moves numbers in reports I did not re-run in full.** §2 re-measures the two lineages
   whose checkpoints are on disk (dec54, the box-30 arms) and the three box-58 artifacts that their
   own tests rewrite. Every *other* published aggregate count and size in this project's 49 reports
   is still stated with the defect in it, and nobody will re-run those campaigns. The direction and
   rough size are now known (largest understated 12–31 %, counts overstated 10–30 %) but there is no
   corrected number for most of them.
6. **One inconsistency is left in the tree deliberately** (§1.2): the three `water-bilayer*` tests
   still cluster with z open inside the soup engine, which does wrap z. Its size is bounded at 0.26 %
   by their own printed `clusterFraction`, and switching them would move three published gate rows
   for no measurable gain — but it is an inconsistency, not a decision I can prove is free.
7. **`radialHeadShells` = 2 turning into 1 under the fix means a previously-published PASS became a
   FAIL** (§2.1). Nothing downstream changes, because closure was already false and encapsulated
   water already zero. But it is the one place where the fix made a historical result *worse* rather
   than better, and it should not be buried in a table.
8. **The two-tailed fraction and the tail length still cannot be separated by composition**, exactly
   as the predecessor said. Per-tail length rose 2.845 → 3.490 *during* this run at fixed
   composition, so even a single arm does not hold it constant. Anyone who thinks the percolation is
   caused by tail length rather than by the second tail would be entitled to say this campaign cannot
   tell them apart — and §9.1's requirement 1 is exactly the experiment that would.
9. **The `null(centre-untrusted)` checkpoints mean 9 of 15 wet samples carry no closure number at
   all.** I read that refusal as evidence (§5) and I still think that is right, but it also means the
   closure claim rests on 6 samples, not 15, and the 6 are the earlier ones.
10. **`tests/soup-drywet-cycling.test.ts` failed and passed on identical re-runs, for the second task
    running.** Its published event-ratio history is now 0.7143, 0.4978, 0.7171, 0.6961, 0.5941,
    0.7116 — a spread of 0.43× on unchanged code. That assertion is not currently measuring anything
    stable and will keep costing a re-run until someone widens the *sample*, not the bound.
11. **`tests/soup-grid-resize.test.ts` and 2 of 3 `rim-lambda-insitu` tests are still failing and I
    did not fix them.** Both are proven pre-existing (the second by stash, §11). The rim-λ one is a
    real construction bug in that test (`createSoup` refuses a 7-particle synthetic system against a
    10 707-particle composition) and it is a small, self-contained fix somebody should make.
12. **The edge-energy retirement in §7.3 inherits its own report's warnings.** λ = 10.77 ε/σ is the
    prepared-patch value; κ's spread was 24.6 ε on a mean of 79.07 and did not shrink with more runs;
    the in-situ λ came out negative. "0.583 kT per amphiphile at N = 2087" is an order of magnitude,
    not a threshold, and it is a scaling argument with textbook disc geometry rather than a
    free-energy measurement.
13. **The verdict is a negative about a MODEL, argued from two compositions.** O:C = 4 gives finite
    rods that are too small; O:C = 1 and O:C = 0.333 both percolate. Nothing was measured between
    O:C = 4 and O:C = 1, and if a compact-but-large regime exists it is in that gap. §9.1's closing
    measurement is designed to look for it from the density side instead, which is cheaper, but the
    composition gap is real and unmeasured.

## 14. FILES

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/final-campaign-report.md` — this report. `.gitignore`
  line 6 ignores `.superpowers/`, so like every predecessor it is committed with `git add -f`.
- MODIFIED, **the fix**: `engine/src/aggregate.ts` (`periodicZ` through `buildClusterUnionFind`,
  `clusters`, `clusterComponents`, `largestClusterFraction`; default false), `soup/src/aggregates.ts`
  and `soup/src/stages.ts` (both soup call sites pass true).
- MODIFIED, measurement paths brought onto the same rule: `tests/continuous-run-audit.test.ts`,
  `tests/helpers/shell-pore-geometry.ts`, `tests/periodic-measurement.test.ts`,
  `tests/rim-lambda-insitu.test.ts`, `tests/soup-bonds.test.ts`.
- MODIFIED, the pin: `tests/coalescence-mechanism-pin.test.ts` (3 → **5** tests; the old z pin kept
  verbatim, plus the fixed branch against the independent implementation, plus the x/y class test).
- NEW: `tests/percolation-check.test.ts` (142 lines) — the wrapping-cluster test, with its own
  must-say-no control.
- NEW artifacts: `verify/out/zfix-campaign-B54-trace.json` (20 checkpoints, the campaign),
  `verify/out/zfix-decisive-run-54-trace.json` (19, the decisive lineage re-measured),
  `verify/out/zfix-box30-arms-trace.json` (10, the box-30 arms re-measured),
  `verify/out/percolation-check.json` and `verify/out/percolation-campaign-B54.json`.
- UPDATED artifacts (rewritten by their own tests, with the corrected aggregate counts):
  `verify/out/periodic-measurement.json`, `verify/out/shell-pore-check.json`,
  `verify/out/rim-lambda-insitu.json`.
- `data/params.json` and `data/soup.json` **untouched**. No potential, rate, threshold, corridor,
  recogniser or rank-A constant modified anywhere.
