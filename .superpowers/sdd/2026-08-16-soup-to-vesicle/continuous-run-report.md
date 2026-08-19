# The continuous run: soup → ? — one run, one time scale, and the verdict

Task `continuous-run` (2026-08-19). Branch `stage-a-atoms`, in place, no worktree. The experiment this
project exists for: ONE continuous run from a mixture of monomers, on one time scale, checkpointed,
to whatever it actually reaches.

**Verdict up front, one line: VESICLE NOT REACHED. The run reached `amphiphiles` and stopped there.
The binding constraint is AGGREGATE SIZE, and it is not close: the largest aggregate holds 12
amphiphiles where the project's own closure threshold needs ~960, an 80× shortfall, and that number
has been flat for the last 84 000 steps. Amphiphile SUPPLY (742, saturated) is only 1.29× short of the
same floor, so supply is not what binds; closure energetics never comes into play at all (the largest
aggregate's flat-disc-equivalent radius is 5.0 % of R_c).**

Nothing was tuned. No threshold was moved, no corridor widened, no rate re-fitted, no pre-made
amphiphile or pre-built patch was used, and the monomers-only start is proven from the run's own trace
(§3), not assumed.

---

## 1. Box and composition: the argument, with its arithmetic

### 1.1 What the target actually costs, in amphiphiles

The smallest object this project's own detector will call a closed vesicle is fixed by three numbers
the project already owns, and by nothing I chose:

- `data/soup.json` `stageThresholds.enclosedVolume` = `data/literature.json` `closure.target.min` =
  **370.8656 σ³** (cross-checked at load time by `loadStageThresholds()`, so the ladder and the gate
  cannot disagree). An enclosed sphere of that volume has inner radius
  `R_in = (3·370.8656/4π)^(1/3) = ` **4.4570 σ**.
- the wall is one bilayer of this engine's own measured thickness. Three measured values are on
  record — 4.3441 σ (the brief), 4.4565 σ (`periodic-measurement-report.md`'s reference), 4.7619 σ
  (re-measured today, §7) — so the floor is given as a band, not a single digit.
- the wall's own area per lipid, measured at zero tension in explicit water: **1.1510 σ²** today
  (§7), against the literature corridor [1.1, 1.5] whose midpoint is 1.30 σ².

With `areaPerLipid ≡ box_area/(lipids/2)` (this codebase's own definition, `engine/src/metrics.ts`),
a shell of mid-surface radius `R_mid = R_in + t/2` needs `N = 2·4π·R_mid²/a` amphiphiles:

| wall thickness t, σ | R_mid, σ | R_out, σ | N at a=1.1510 | N at a=1.30 |
|---|---|---|---|---|
| 4.3441 | 6.6291 | 8.8011 | **960** | 850 |
| 4.4565 | 6.6853 | 8.9135 | **976** | 864 |
| 4.7619 (today) | 6.8380 | 9.2189 | **1021** | 904 |

**Geometric floor: ~960 amphiphiles in ONE aggregate** (band 850–1021). Call it 960 below.

The *thermodynamic* threshold is much higher. `kappa-tightening-report.md` measured
`R_c = 4κ/λ = 29.37 ± 3.97 σ`, i.e. `N = 2πR_c²/a₀` = **4483** amphiphiles (3353 … 5777 over its own
error band) before an open patch prefers to curl shut. (Its own §3.2 records the opposite reading from
the in-situ λ, which came out negative and would make closure favourable at any curved radius — that
disagreement is unresolved and is why the geometric floor, not R_c, is used as the target here: the
geometric floor is a *detector* threshold and cannot be argued with.)

### 1.2 The box^1.5 trap, restated and then measured from the other side

At fixed density, carbon ∝ L³, so amphiphiles ∝ L³ at fixed yield; a bilayer aggregate holding N of
them has radius R ∝ √N ∝ L^1.5, while the measurability limit (the periodic minimum-image trust
radius) is exactly L/2 ∝ L. So `R/(L/2) ∝ L^0.5` — **growing the box at constant density makes the
largest aggregate strictly LESS measurable.** This is not a derivation on paper: the dense campaign
(box 58, C=40000, no water) measured mean head radius **29.6 σ against box/2 = 29.0 σ**, ratio
**1.021**, which is exactly the 27–28 % head / 22–24 % tail trust-limit exclusion
`periodic-measurement-report.md` §3 had to report at every one of its ten checkpoints, and which its
own verdict named as *the one remaining thing* that would leave the reading undecided. Extrapolating
that ratio at constant density to L=90 gives 1.021·√(90/58) = **1.271** — worse, as predicted.

The prescription that follows is the opposite of "use a bigger box at the same density": **the box
must grow FASTER than the amphiphile supply.** So:

### 1.3 What was chosen, and why it can in principle produce the structure

- **Box: 90 σ, cubic.** Exactly 3× the shipped 30 σ box, so the volume is exactly 27× and the
  composition scales with no rounding at all.
- **Composition: the shipped ratios of `data/soup.json` `start`, multiplied by 27 — nothing else.**
  `C 40500, O 13500, H 40500, M 2700, W 288900`, N = **386 100**, total density
  386100/729000 = **0.529630 σ⁻³** — bit-identical to the shipped box-30 density, the regime this
  engine has run safely throughout the project (`broth-composition-report.md` §2). Water is 74.8 % of
  the particles, exactly as shipped; its sub-density 0.396 σ⁻³ is the same known gap that report
  already records, untouched.
- **Why THIS box can in principle produce the structure.** C = 40500 is deliberately the *same carbon
  count* as the one long campaign this project has already run to coarsening (C = 40000, box 58), which
  produced **1331 amphiphiles with 1282 of them (96.3 %) in a single aggregate**. So the amphiphile
  supply at this composition is not an extrapolation — it is measured, and 1331 ≥ 960, i.e. it clears
  the geometric floor with 39 % to spare. What box 90 changes is *only* the room the object has:
  box/2 = 45 σ against that object's own 29.6 σ head radius — a 1.52× margin where box 58 had 1.021×,
  which is precisely the fix `periodic-measurement-report.md` and `expanded-box-report.md` both asked
  for and which `expanded-box-report.md` proved cannot be obtained by expanding an existing checkpoint
  (FENE divergence on wraparound-dependent covalent bonds).
- **Why not bigger.** The Verlet list is a flat `N·listCapacity·4` bytes (`soup/src/soup-buffers.ts`),
  `listCapacity = 2500`, and this adapter reports
  `maxStorageBufferBindingSize = maxBufferSize = 4 294 967 292` (measured, §8), i.e. a hard ceiling of
  **429 496 particles**. Box 90 sits at 386 100 → 3.86 GB, 90 % of the ceiling. Box 92.6 would need
  4.27 GB and would not allocate.
- **Why not smaller.** At box 58 the object is unmeasurable (§1.2). At box 72 the carbon count falls to
  ~21 000 and the measured yield gives ~700 amphiphiles — below the 960 floor before the run starts.
- **Clay: off, and NOT by my choice.** `soup/cli/campaign.ts` line 264 hard-pins `clay: false`, with
  its own written reason (it must be able to resume pre-clay checkpoints). Reported rather than worked
  around, and the consequence is *favourable to the hypothesis*, so it cannot manufacture a positive:
  `clay-surface-chemistry-report.md` measures the shipped hydrophilic platelet as **−23.8 % chain
  growth** (62 % of which is just freezing a quarter of the catalysts), as nucleating **nothing**
  (in-band fraction 0.075 vs 0.133 uniform; aggregates further from the surface than in any other
  arm), and as carrying a **1-in-43** rigid-wall divergence rate that a 184 000-step run would very
  likely have hit. A clay-free run is therefore the hypothesis's best measured shot, which makes the
  negative below a stronger negative, not a weaker one.
- **Dry-wet cycling: off** (the shipped default). The requirement was one continuous run on one time
  scale; cycling is a second mechanism and is left for the next experiment (§9).
- **kT = 1.1** (`data/params.json`'s rank-A thermostat value, the top of the Cooke & Deserno range and
  the value every gate in this project is measured at). **Seed 19** — the dense campaign's own seed,
  so nothing about the initial draw was shopped for.

---

## 2. Stage criteria, fixed in advance

Taken verbatim from `data/soup.json` `stageThresholds` and the per-aggregate machinery in
`soup/src/aggregates.ts` / `soup/src/stages.ts`. Nothing here was written or altered by this task; the
three previously-invalidated measurement methods (box-wide thresholds, a flood fill fooled by
position, an aggregate spanning its periodic box) are all avoided because the ladder is decided on
**per-aggregate** quantities and closure on **encapsulated water** via
`soup/src/water-closure.ts`'s never-unwrap periodic flood.

| stage | condition (all of it) |
|---|---|
| `monomers` | anything that fails the line below |
| `amphiphiles` | `amphiphileFraction ≥ 0.03` — fraction of carbon bound into a recognised chain (simple chain, ≤ 2 C-C per carbon, exactly one terminal head, head capacity 2) |
| `micelles` | amphiphiles, **and** `qualifyingAggregateCount ≥ 2` (aggregates with ≥ `minAmphiphilesPerAggregate = 13`, itself the derived `ceil(4πr²/a_mid)`, checked against its formula at load), **and** `amphiphileShareInQualifying ≥ 0.5` |
| `bilayer` | amphiphiles, **and** some detailed aggregate with `flatnessRatio ≤ 0.35` **and** `inPlaneSymmetry ≥ 0.5` **and** two head layers on the radial **or** transverse profile |
| `vesicle` | amphiphiles, **and** some detailed aggregate with `radialHeadShells == 2` **and** closure — with water present that means **encapsulated water**, `closed == true` (trapped water count above the bulk-density-derived threshold), never the vacuum-cell count; `null` (untrusted periodic centre) counts as NOT closed |

Reported alongside, as diagnostics only: box-wide `enclosedVolume` (threshold 370.8656 σ³),
box-wide `headPeaks`, `largestAggregateFraction`.

---

## 3. The monomers-only start, from the run's own trace

Not assumed. `data/checkpoints/trace/soup2ves90-step1.json` is a real checkpoint of this exact
configuration one step in (bond attempts fire every `bondAttemptInterval.steps = 20`, so step 1 is
before the first attempt round), decoded on the CPU by `tests/continuous-run-audit.test.ts`:

```
{
 "step": 1, "N": 386100, "box": [90,90,90], "stage": "monomers",
 "species": {"C": 40500, "O": 13500, "H": 40500, "M": 2700, "W": 288900},
 "bonds": 0,
 "events": {"cc_bond": 0, "cc_break": 0, "co_bond": 0, "co_break": 0},
 "invariants": {"nonFinitePositions": 0, "nonFiniteVelocities": 0,
   "valence": {"carbonCCover2": 0, "carbonCOover1": 0, "headOverChainCapacity": 0,
     "nonBondableBonded": 0, "degreeOver3": 0, "headNotTerminal": 0,
     "headChainCapacity": 2, "headsWithTwoTails": 0}},
 "amphiphileCount": 0, "amphiphileFraction": 0, "meanTailLength": 0, "lengthHistogram": {},
 "aggregateCount": 0, "hasLamellarAggregate": false, "hasVesicleAggregate": false,
 "boxWideEnclosedVolume": 0, "largest": []
}
RUN-AUDIT-START data/checkpoints/trace/soup2ves90-step1.json step=1 bondSlotsUsed=0 events={"cc_bond":0,"cc_break":0,"co_bond":0,"co_break":0}
```

Zero bonds, zero bond events, zero amphiphiles, zero recognised aggregates. Every particle is a
monomer of one of the five declared species. **There is no pre-made amphiphile and no pre-built patch
in this run's initial state, and this is the run's own recorded state, not a claim.**

---

## 4. The stage trace

Two independent readings of the same run are given, because they are produced by different code paths
and agree: the campaign CLI's own live progress lines (in-page, GPU, `api.stageOf`), and the offline
CPU audit (`tests/continuous-run-audit.test.ts`, decoding the checkpoints with the *same* functions —
`findAmphiphiles`, `analyzeAggregates`, `stageFromEvidence` — so there is no second implementation of
any measurement).

### 4.1 Offline audit trace (full evidence)

`amph` = recognised amphiphiles, `frac` = amphiphileFraction, `tail` = mean tail length, `aggs` =
aggregate count, `qual` = aggregates ≥ 13 amphiphiles, `L` = largest aggregate's amphiphile count,
`Rg`/`flat`/`inPl` = its radius of gyration / `principalMoments[0]/[2]` / `[1]/[2]`, `encH₂O` =
encapsulated water count / its own threshold.

| step | stage | amph | frac | tail | cc_bond | co_bond | aggs | qual | shareQ | L | Rg | flat | inPl | radShells | cavity | encH₂O | closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | monomers | 0 | 0.00000 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — |
| 40001 | amphiphiles | 682 | 0.08417 | 4.999 | 10581 | 920 | 397 | 1 | 0.0205 | 14 | 8.795 | 0.0763 | 0.1647 | unavailable | 0 | 0/147.0 | false |
| 52001 | amphiphiles | 704 | 0.08914 | 5.128 | 11284 | 966 | 409 | 0 | 0 | 7 | 4.741 | 0.0694 | 0.1906 | unavailable | 0 | 0/147.0 | false |
| 64001 | amphiphiles | 715 | 0.09195 | 5.208 | 11676 | 989 | 407 | 0 | 0 | 10 | 5.601 | 0.0807 | 0.6430 | unavailable | 0 | 0/147.0 | false |
| 76001 | amphiphiles | 729 | 0.09474 | 5.263 | 11925 | 1006 | 404 | 0 | 0 | 11 | 5.728 | 0.0795 | 0.5950 | unavailable | 0 | 0/147.0 | false |
| 88001 | amphiphiles | 732 | 0.09575 | 5.298 | 12101 | 1010 | 400 | 0 | 0 | 11 | 5.727 | 0.0794 | 0.5936 | unavailable | 0 | 0/147.0 | false |
| 100001 | amphiphiles | 737 | 0.09659 | 5.308 | 12201 | 1016 | 406 | 0 | 0 | 12 | 5.849 | 0.0945 | 0.4782 | unavailable | 0 | 0/147.0 | false |
| 112001 | amphiphiles | 737 | 0.09679 | 5.319 | 12269 | 1017 | 397 | 0 | 0 | 12 | 5.843 | 0.0949 | 0.4780 | unavailable | 0.125 | 0/147.0 | false |
| 124001 | amphiphiles | 740 | 0.09731 | 5.326 | 12331 | 1023 | 398 | 0 | 0 | 12 | 5.842 | 0.0953 | 0.4779 | unavailable | 0.125 | 0/147.0 | false |
| 136001 | amphiphiles | 741 | 0.09760 | 5.335 | 12348 | 1025 | 402 | 0 | 0 | 12 | 5.841 | 0.0954 | 0.4783 | unavailable | 0.125 | 0/147.0 | false |
| 148001 | amphiphiles | 742 | 0.09778 | 5.337 | 12378 | 1027 | 404 | 0 | 0 | 12 | 5.835 | 0.0954 | 0.4789 | unavailable | 0 | 0/147.0 | false |
| 160001 | amphiphiles | 742 | 0.09785 | 5.341 | 12396 | 1027 | 413 | 0 | 0 | 12 | 5.835 | 0.0955 | 0.4778 | unavailable | 0 | 0/147.0 | false |
| 172001 | amphiphiles | 742 | 0.09785 | 5.341 | 12401 | 1027 | 405 | 0 | 0 | 12 | 5.835 | 0.0955 | 0.4778 | unavailable | 0 | 0/147.0 | false |
| 180001 | amphiphiles | 742 | 0.09785 | 5.341 | 12404 | 1027 | 394 | 0 | 0 | 12 | 5.835 | 0.0954 | 0.4781 | unavailable | 0 | 0/147.0 | false |
| **184001** | **amphiphiles** | **742** | **0.09785** | **5.341** | **12405** | **1027** | **409** | **0** | **0** | **12** | **5.835** | **0.0954** | **0.4781** | **unavailable** | **0** | **0/147.0** | **false** |

`cc_break = 0` and `co_break = 0` at **every** step. `cc_break` is zero by the file's own design
(`data/soup.json` `cc_break.attemptRate = 0`, the documented irreversible-C-C decision of
`kinetic-growth-report.md`); `co_break` at `accept = exp(−8/1.1) ≈ 6.8e-4` on 1027 bonds is a
low-probability channel and zero is consistent with it, not a violation — flagged in §9 anyway.

### 4.2 Live CLI trace (the run's own progress lines, unedited)

Every `--every` boundary of every chunk, first and last of each invocation shown, verbatim:

```
[campaign] шаг=1/1 stage=monomers агрегатов=0 крупнейший=0 headShells=n/a cavityVolume=0.000 stepMs=76 checkpointMs=527 progressMs=48
[campaign] шаг=4001/8001 stage=monomers агрегатов=137 крупнейший=3 headShells=unavailable cavityVolume=0.000 stepMs=72317
[campaign] шаг=8001/8001 stage=monomers агрегатов=283 крупнейший=6 headShells=unavailable cavityVolume=0.000 stepMs=250776
[campaign] шаг=12001/16001 stage=amphiphiles агрегатов=350 крупнейший=4 headShells=unavailable cavityVolume=0.000 stepMs=165019
[campaign] шаг=16001/16001 stage=amphiphiles агрегатов=371 крупнейший=6 headShells=unavailable cavityVolume=0.000 stepMs=179647
[campaign] шаг=20001/28001 stage=amphiphiles агрегатов=381 крупнейший=6 ... stepMs=176465
[campaign] шаг=24001/28001 stage=amphiphiles агрегатов=384 крупнейший=7 ... stepMs=177325
[campaign] шаг=28001/28001 stage=amphiphiles агрегатов=391 крупнейший=9 ... stepMs=179301
[campaign] шаг=32001/40001 stage=amphiphiles агрегатов=391 крупнейший=9 ... stepMs=176458
[campaign] шаг=36001/40001 stage=amphiphiles агрегатов=400 крупнейший=14 ... stepMs=181127
[campaign] шаг=40001/40001 stage=amphiphiles агрегатов=397 крупнейший=14 ... stepMs=186628
[campaign] шаг=44001/52001 stage=amphiphiles агрегатов=407 крупнейший=11 ... stepMs=184980
[campaign] шаг=48001/52001 stage=amphiphiles агрегатов=409 крупнейший=9  ... stepMs=184989
[campaign] шаг=52001/52001 stage=amphiphiles агрегатов=409 крупнейший=8  cavityVolume=0.125 stepMs=184790
[campaign] шаг=56001/64001 stage=amphiphiles агрегатов=395 крупнейший=10 ... stepMs=123554
[campaign] шаг=60001/64001 stage=amphiphiles агрегатов=403 крупнейший=10 ... stepMs=202266
[campaign] шаг=64001/64001 stage=amphiphiles агрегатов=407 крупнейший=10 ... stepMs=170433
[campaign] шаг=68001/76001 stage=amphiphiles агрегатов=411 крупнейший=10 ... stepMs=169242
[campaign] шаг=72001/76001 stage=amphiphiles агрегатов=404 крупнейший=11 ... stepMs=168357
[campaign] шаг=76001/76001 stage=amphiphiles агрегатов=404 крупнейший=11 ... stepMs=171106
[campaign] шаг=80001/88001 stage=amphiphiles агрегатов=400 крупнейший=11 ... stepMs=165259
[campaign] шаг=84001/88001 stage=amphiphiles агрегатов=414 крупнейший=11 ... stepMs=171372
[campaign] шаг=88001/88001 stage=amphiphiles агрегатов=400 крупнейший=11 ... stepMs=172143
[campaign] шаг=92001/100001 stage=amphiphiles агрегатов=399 крупнейший=11 ... stepMs=165894
[campaign] шаг=96001/100001 stage=amphiphiles агрегатов=407 крупнейший=11 ... stepMs=174993
[campaign] шаг=100001/100001 stage=amphiphiles агрегатов=406 крупнейший=12 ... stepMs=182369
[campaign] шаг=104001/112001 stage=amphiphiles агрегатов=403 крупнейший=12 cavityVolume=0.125 stepMs=180864
[campaign] шаг=108001/112001 stage=amphiphiles агрегатов=400 крупнейший=12 cavityVolume=0.125 stepMs=189317
[campaign] шаг=112001/112001 stage=amphiphiles агрегатов=397 крупнейший=12 cavityVolume=0.125 stepMs=177844
[campaign] шаг=116001/124001 stage=amphiphiles агрегатов=418 крупнейший=10 cavityVolume=0.250 stepMs=184265
[campaign] шаг=120001/124001 stage=amphiphiles агрегатов=405 крупнейший=12 cavityVolume=0.125 stepMs=181064
[campaign] шаг=124001/124001 stage=amphiphiles агрегатов=398 крупнейший=12 cavityVolume=0.125 stepMs=180789
[campaign] шаг=128001/136001 stage=amphiphiles агрегатов=403 крупнейший=12 cavityVolume=0.125 stepMs=180023
[campaign] шаг=132001/136001 stage=amphiphiles агрегатов=401 крупнейший=12 cavityVolume=0.125 stepMs=186763
[campaign] шаг=136001/136001 stage=amphiphiles агрегатов=402 крупнейший=12 cavityVolume=0.125 stepMs=184307
[campaign] шаг=140001/148001 stage=amphiphiles агрегатов=404 крупнейший=10 ... stepMs=177843
[campaign] шаг=144001/148001 stage=amphiphiles агрегатов=398 крупнейший=12 ... stepMs=179520
[campaign] шаг=148001/148001 stage=amphiphiles агрегатов=404 крупнейший=12 ... stepMs=175963
[campaign] шаг=152001/160001 stage=amphiphiles агрегатов=395 крупнейший=12 ... stepMs=177222
[campaign] шаг=156001/160001 stage=amphiphiles агрегатов=400 крупнейший=12 ... stepMs=180430
[campaign] шаг=160001/160001 stage=amphiphiles агрегатов=413 крупнейший=12 ... stepMs=179902
[campaign] шаг=164001/172001 stage=amphiphiles агрегатов=407 крупнейший=10 ... stepMs=172311
[campaign] шаг=168001/172001 stage=amphiphiles агрегатов=402 крупнейший=12 ... stepMs=176316
[campaign] шаг=172001/172001 stage=amphiphiles агрегатов=405 крупнейший=12 ... stepMs=180811
[campaign] шаг=176001/184001 stage=amphiphiles агрегатов=398 крупнейший=12 ... stepMs=176805
[campaign] шаг=180001/184001 stage=amphiphiles агрегатов=394 крупнейший=12 ... stepMs=179340
[campaign] шаг=184001/184001 stage=amphiphiles агрегатов=409 крупнейший=12 headShells=unavailable cavityVolume=0.000 stepMs=178196
```

**Stage transitions, with step numbers:** `monomers` → `amphiphiles` between step 8001 and step 12001
(the crossing of `amphiphileFraction = 0.03`). No further transition in the remaining 172 000 steps.
`micelles`, `bilayer`, `vesicle`: **never reached.**

**Resume note, as `checkpoint-resume-report.md` requires it to be stated:** this run was assembled from
**16 resumes** across 18 CLI invocations. That report's guarantee is the **same ensemble, not the same
trajectory** — the conservation invariants are exact across a resume (verified again here at every
checkpoint, §6), but the one force recomputation each resume performs passes through `atomicAdd`
ordering, so the chemistry counts are representative of the ensemble rather than a continuation of one
numerically converged path. Everything in §4/§5 should be read that way.

---

## 5. The largest aggregate, measured against the closure threshold

Final state, step 184 001, the largest of 409 aggregates (full record, unedited, from
`tests/continuous-run-audit.test.ts`):

```
{"amphiphileCount": 12, "particleCount": 107, "radiusOfGyration": 5.835,
 "principalMoments": [2.065, 10.345, 21.64],
 "flatnessRatio": 0.0954, "inPlaneSymmetry": 0.4781,
 "radialHeadShells": "unavailable", "transverseHeadShells": 0, "cavityVolume": 0,
 "encapsulatedWater": {"encapsulatedCount": 0, "totalWater": 288900,
   "bulkWaterDensity": 0.39634624734084284, "encapsulationThresholdCount": 146.99118882781008,
   "closed": false, "seedDistFromCentre": 77.78775121670749,
   "theoreticalMaxDist": 77.94228634059948, "totalEmptyCells": 5831266, "unreachedCells": 1}}
```

| quantity | measured | threshold | verdict |
|---|---|---|---|
| amphiphiles in the largest aggregate | **12** | 960 for a closed vesicle; 13 even to *qualify* as one micelle | **80× short** |
| particles in it | 107 | — | — |
| radius of gyration | 5.835 σ | box/2 = 45 σ (trust limit) | 7.71× margin — **fully measurable** |
| principal moments | 2.065 / 10.345 / 21.64 | — | prolate-to-triaxial, not a shell |
| flatness `λ₀/λ₂` | 0.0954 | ≤ 0.35 for lamellar | passes |
| in-plane symmetry `λ₁/λ₂` | **0.4781** | ≥ 0.5 for lamellar | **fails** — a rod/ribbon, not a sheet |
| radial head shells | **unavailable** | must be exactly 2 | fails (too few heads: 10 bins × ≥2 heads needs ≥20, the object has ~12) |
| transverse head shells | 0 | 2 for a flat bilayer | fails |
| enclosed cavity (vacuum cells) | **0** | > 370.8656 σ³ | fails |
| **encapsulated water** | **0 beads** | ≥ 146.99 beads (derived from the run's own bulk water density 0.39635 σ⁻³) | **fails — `closed: false`** |
| box-wide enclosed volume (diagnostic) | 0.5 σ³ | > 370.8656 σ³ | **742× below** |

**Two integrity checks on the closure measurement itself, both passed**, so this "not closed" is a
measurement and not a failure to measure:
- the periodic flood seed sits at 77.788 σ from the aggregate's own circular-mean centre against a
  theoretical maximum of 77.942 σ — a genuine far-corner "outside" cell, the precondition
  `periodic-measurement-report.md` §4 insisted on checking per-run;
- the flood reached all but **1** of 5 831 266 empty cells, and `encapsulatedWater` is neither
  `undefined` (no water supplied) nor `null` (untrusted centre) — it is a real reading.
- and, for the first time in this project, **0 % of the object's beads sit beyond the minimum-image
  trust limit** (Rg 5.835 σ against box/2 = 45 σ), against the 27–28 % of heads and 22–24 % of tails
  `periodic-measurement-report.md` had to exclude at every checkpoint. The box argument in §1.3
  delivered exactly what it was chosen for; it is the aggregate that did not arrive.

The second and third largest aggregates are the same kind of object (10 amphiphiles, Rg 5.967,
flat 0.0697, inPl 0.1132, cavity 0, water 0/146.99, not closed; and 9 amphiphiles, Rg 6.256,
flat 0.0371, inPl 0.1347, cavity 0.125, water 0/146.99, not closed).

---

## 6. Invariants — all held, at every checkpoint

Asserted (not eyeballed) by `tests/continuous-run-audit.test.ts` on **all 38** checkpoints that survive
on disk, spanning step 1 → 184 001, and by `expect(...)` calls that fail the test if violated.
Aggregate result over those 38: **0 checkpoints with any invariant violation**, species map identical to
`{C:40500, O:13500, H:40500, M:2700, W:288900}` at every one of them, `hasVesicleAggregate` false at
every one, `hasLamellarAggregate` false at every one, largest-aggregate amphiphile count never above
**14** (its whole-run maximum, at step 36 001–40 001), box-wide enclosed volume never above
**1.125 σ³** against the 370.8656 σ³ threshold, and only two stage labels ever produced:
`monomers` and `amphiphiles`.

| invariant | how it is checked | result |
|---|---|---|
| no non-finite state | every position and velocity component of all 386 100 particles | `nonFinitePositions = 0`, `nonFiniteVelocities = 0` at **every** checkpoint |
| no Verlet overflow | `assertVerletSafety` throws inside the engine on overflow or on a drift larger than the skin; the run never threw | no throw in 18 invocations / 184 001 steps |
| monomer conservation | species census recomputed from the species slot of every particle | `{C:40500, O:13500, H:40500, M:2700, W:288900}` at **every** checkpoint, exactly |
| charge conservation | conserved by construction (no charged species in this model; the CLI's own `invariants()` reports it each step) | unchanged |
| valence: carbon ≤ 2 C-C | rebuilt from the raw `bondSlots` rows | 0 violations, every checkpoint |
| valence: carbon ≤ 1 C-O | same | 0 violations |
| valence: carbon degree ≤ 3 | same | 0 violations |
| valence: head ≤ `headPlacement.chainCapacity` (= 2, read from the file, not assumed to be 1) | same | 0 violations (159 heads legitimately carry two tails at step 184 001) |
| placement: `headPlacement.terminalOnly` — a carbon carrying a head has ≤ 1 chain bond | same | 0 violations |
| nothing but carbon/head may be bonded (H, M, W never) | same | 0 violations |
| catalyst gating: `cc_bond` requires a catalytic centre | `tests/soup-bonds.test.ts` (§7) | passes |

One correction worth recording: the audit's first version asserted "a head carries at most 1 bond" and
found 131 apparent violations. That assertion was **wrong, not the run**: `data/soup.json`
`headPlacement.chainCapacity = 2` (the `two-tails` task) makes a two-tailed head legal. The audit now
reads the cap out of the file, and additionally checks the `terminalOnly` rule the first version did
not check at all.

---

## 7. The two gate tests, after the run — unchanged, both pass

```
$ nice -n 15 npx vitest run tests/gate6-bilayer.test.ts
CONVERGE start 1.55 (first sample 1.530)  moves 400  tail mean 1.2127 min 1.1753 max 1.2537  lnA drift t=0.44 (2.43e-5/move)  accepted 0.440  clusterFraction 1.0000  peaksOk true  checkpoints 2
CONVERGE start 0.9 (first sample 0.905)  moves 400  tail mean 1.2010 min 1.1725 max 1.2274  lnA drift t=1.47 (5.04e-5/move)  accepted 0.469  clusterFraction 0.9975  peaksOk true  checkpoints 2
 ✓ tests/gate6-bilayer.test.ts (2 tests) 53020ms
   ✓ готовый бислой при нулевом натяжении держит площадь и толщину из литературы 27333ms
   ✓ площадь сходится в литературный коридор и из слишком большого, и из слишком малого бокса 25603ms
 Test Files  1 passed (1)
      Tests  2 passed (2)
```

```
$ nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=450 areaPerLipid(MEASURED, tail mean)=1.1510 [min 1.1463, max 1.1572, corridor 1.1-1.5] driftPerChunk(lnA)=-8.206e-6 t=-0.90 thickness(measured)=4.7619 [corridor 4-6] clusterFraction=1.0000 waterInCore=43/4500 headBuriedFraction=0.0782 acceptedFraction=0.1429 of 4500 trials throughput=1179.40 steps/s at N=5700 verdict=passed
 ✓ tests/water-bilayer-area-move.test.ts (1 test) 192517ms
 Test Files  1 passed (1)
      Tests  1 passed (1)
```

Both inside their corridors, both inside the run-to-run scatter the `hydrophobic-asymmetry` task
measured (area 1.1211–1.2215, thickness 4.316–5.103 over n=5): area **1.1510**, thickness **4.7619**.
No threshold, corridor or parameter was touched by this task.

Rule-level invariant confirmations, one invocation each:

```
$ nice -n 15 npx vitest run tests/soup-valence.test.ts
 ✓ ни один атом углерода не превышает 2 связей C-C и 1 связь C-O; ни одна голова не превышает configured chainCapacity связей, за 45000 шагов 65449ms
 Test Files  1 passed (1)      Tests  1 passed (1)

$ nice -n 15 npx vitest run tests/soup-bonds.test.ts -t 'каталитическом'
 ✓ связи образуются только на каталитическом центре там, где правило это требует 128750ms
 Test Files  1 passed (1)      Tests  1 passed | 2 skipped (3)

$ CONTINUOUS_RUN_CHECKPOINTS='...15 checkpoints...' nice -n 15 npx vitest run tests/continuous-run-audit.test.ts
 ✓ continuous-run checkpoints: invariants hold and the stage ladder is reproducible off-GPU 37087ms
 Test Files  1 passed (1)      Tests  1 passed (1)
```

`tests/soup-vesicle.test.ts` was **never** run (its 4 M-step timeout is the pattern this project's own
rules forbid; it stays as documentation). The full suite was **never** run. The dev server on :5199
(user-owned `node`, PID 94131) was neither started nor stopped nor inspected beyond leaving it alone.

---

## 8. The verdict, and the binding constraint

### VESICLE NOT REACHED. Stage reached: `amphiphiles`.

Four candidate constraints, each with the number that settles it:

| candidate | measured | needed | shortfall | binding? |
|---|---|---|---|---|
| **aggregate size** | largest aggregate **12** amphiphiles | ~960 | **80×** | **YES — this is the binding constraint** |
| amphiphile supply | **742**, saturated | ~960 | 1.29× | no — off by 22 %, not by a factor |
| closure energetics | largest aggregate's flat-disc-equivalent radius **1.483 σ** | R_c = 29.37 σ | 19.8× (5.0 % of R_c) | no — never reached the regime where it applies |
| reachable time | see below | — | — | no for the chemistry, **not the explanation** for the aggregation |

**Why aggregate size is binding, and why time is not the answer.** The run is in a measured steady
state, not a slow transient:

- the aggregate count is flat at **394–418 (mean ≈ 404) from step 44 001 to 184 001** — 140 000 steps
  with no coarsening at all;
- the largest aggregate is the **same object** from step 100 001 to 184 001: amphiphiles 12,
  Rg 5.849 → 5.835 σ, flatness 0.0945 → 0.0954, in-plane 0.4782 → 0.4781. 84 000 steps, no growth;
- the chemistry is **exhausted, not slow**: `amphiphileCount` is 742 at steps 148 001, 160 001,
  172 001, 180 001 and 184 001 — **Δ = 0 over the last 36 000 steps**; `co_bond` = 1027 over the same
  span, **Δ = 0**; `cc_bond` grew by **27** in 36 000 steps against **703** in the 12 000 steps around
  step 40 001, a 26× deceleration. So more steps do not buy more amphiphiles either.

**Why they do not aggregate: the amphiphiles are essentially molecularly dispersed at this
concentration, and that follows from the same box argument that made them measurable.**

- 742 amphiphiles in 729 000 σ³ = **1.018e-3 σ⁻³**, mean spacing `n^(-1/3)` = **9.94 σ**;
- they sit in **409** aggregates of mean size **1.81 amphiphiles** (size histogram of the ten biggest
  at step 184 001: 12, 10, 9, 10, 8, 10, 6, 8, 5, 6 — and a long tail of ones and twos);
- the run that DID coarsen — box 58, C = 40000, no water — had amphiphile density
  1331/195112 = **6.822e-3 σ⁻³**, **6.7× denser**, and put **1282 of 1331 (96.3 %)** into ONE
  aggregate. This run put **12 of 742 (1.6 %)** into its largest;
- a Smoluchowski check confirms the two are consistent rather than contradictory: at Γ = 1/τ and
  kT = 1.1, a 107-bead cluster has D ≈ kT/(Γ·107) = 0.0103 σ²/τ, so over this run's 1840 τ it drifts
  `√(6Dt)` = **10.65 σ** — just under the **12.13 σ** mean separation between aggregates. The run is
  right at the threshold of its **first** coalescence generation, and ~8.7 halvings would be needed to
  reach one dominant aggregate, each slower than the last.

**And this is the box^1.5 trap closing from both sides — which is the real result.** §1.2's arithmetic
said a box big enough to *measure* a ~1300-amphiphile aggregate cannot be reached at constant density.
This run took the only remaining route (grow the box faster than the supply) and it worked for the
measurement — 0 % trust-limit exclusion against the previous 27–28 % — but the same dilution that
bought the margin is what stopped the aggregate forming. **The two measured points bracket the
transition and neither is vesicle-competent:**

- **6.82e-3 σ⁻³** → one connected amphiphile phase spanning the periodic box (Rg 31 σ against a
  flat-disc-equivalent 15.71 σ, ratio 1.97: a branched/percolating sheet, and `expanded-box-report.md`
  §4 measured ~93 178 of 93 200 particles in one proximity component). That is a **lamellar/percolating
  phase, not a finite closed object**, and percolation is a property of the density, not of L — so no
  box size escapes it at that density;
- **1.02e-3 σ⁻³** → **molecularly dispersed**, mean aggregate 1.81 amphiphiles.

If a finite-closed-object regime exists in this model at kT = 1.1 with the current interaction set, it
lies in the **un-sampled interval 1.0e-3 … 6.8e-3 σ⁻³**, and this run did not locate it. Two points is
two points; that is stated as the open question, not as a proof of absence.

**The one conditional window the arithmetic does allow**, for the next experiment: at the
aggregation-competent density 6.822e-3 σ⁻³ with `R(L) = 0.067·L^1.5` (calibrated on the box-58 object's
own 29.6 σ), the supply clears 960 amphiphiles at `L ≥ 51.94 σ` and the object stays inside `L/2` at
`L ≤ 55.69 σ` — a window `L ∈ [51.9, 55.7] σ` (N_amph 956 … 1179, ≈ 74 000–86 000 particles, 0.74–0.86
GB of Verlet list, comfortably affordable). **It is conditional on a compaction this model has never
shown**: the box-58 object's extent was 1.97× its own compact-shell radius, and if extent is set by
percolation rather than by √N the window is empty. Measuring which of the two it is, is the cheapest
next experiment this run points at.

---

## 9. Concerns

1. **Two densities are two points.** The "no finite-closed-object regime" reading (§8) is a bracket, not
   a sweep. A proper density sweep between 1.0e-3 and 6.8e-3 σ⁻³ at fixed box — 4–5 runs of the size
   §8's window names — is what would turn this bracket into a phase boundary. This task did not have
   the budget for it and does not claim it.
2. **The run is clay-free, and not by choice** (§1.3): `soup/cli/campaign.ts` line 264 pins
   `clay: false`. The direction of the bias is favourable to the hypothesis, so it cannot have
   manufactured this negative — but the shipped configuration has clay ON, and this run therefore does
   not test the shipped configuration. Giving the CLI a clay flag is a small, honest follow-up.
3. **`co_break` fired zero times in 184 001 steps.** Consistent with its own rate
   (`accept ≈ 6.8e-4`, expectation of order a few events), but it means the ONE reversible channel in
   this chemistry was never exercised in this run at all. Combined with `cc_break.attemptRate = 0` by
   design, the chain-length distribution here is **entirely kinetic** — the mean tail length of
   **5.341** beads, and a histogram running out to 22, is not an equilibrium property and is well above
   the 2–3 beads a C12–C18 fatty acid is supposed to map to. A tail that long changes the packing
   parameter and therefore which aggregate shape is preferred; the rod-like `inPlaneSymmetry` of
   0.11–0.48 seen throughout §4.1 may be a direct consequence. **This is the single most likely place a
   real defect is hiding, and it is a composition/kinetics question, not a threshold question.**
4. **The medium is still two-phase.** Water sub-density is 0.396 σ⁻³ against the 0.8 σ⁻³ this project
   measured as its own liquid threshold (`broth-composition-report.md` §2) — inherited, untouched, and
   the reason every per-run null in this project is wide.
5. **16 resumes.** Per `checkpoint-resume-report.md` §3c, a resume guarantees the same *ensemble*, not
   the same *trajectory*; the conservation invariants held exactly across all of them (§6), but the
   chemistry counts in §4 carry 16 opportunities for the force-recompute nondeterminism to fire. A
   single-process 184 000-step run would remove that, and is impossible under the 600 s foreground cap.
6. **`radialHeadShells` was `unavailable` at every single checkpoint**, because the largest aggregate
   never had the ≥ 20 heads that 10 bins × `aggregateMinHeadsPerBin = 2` requires. So the vesicle
   criterion's *first* clause was never even evaluable in this run — the closure verdict rests on
   `encapsulatedWater.closed = false` (a real reading, §5) rather than on both clauses. Worth stating
   plainly: this run did not test the head-shell criterion, it never got the material to test it with.
7. **The 960-amphiphile floor is a floor, not a prediction.** It is the smallest object the *detector*
   accepts. A real vesicle this model would spontaneously form could be much larger, which would make
   the shortfall worse, not better.
8. **Throughput was not constant** and the report's own extrapolations use the steady-state value: the
   first 4 000 steps ran at 55 steps/s and everything after step ~8 000 at 22–23 steps/s, as the medium
   demixed and Verlet lists lengthened. Any wall-clock estimate for a bigger run should use 22 steps/s
   at N = 386 100, scaled as 1/N, and should be treated as a lower bound.
9. **Early checkpoints 4001–36001 were pruned by me** during the timing phase (to keep the CLI's
   config scan cheap) before I realised I wanted them for the trace. Their stage/aggregate/cavity
   numbers survive in the CLI's own progress lines (§4.2) and step 1 was regenerated as a fresh
   artifact (§3), but the full offline evidence table (§4.1) starts at step 40 001. My mistake, recorded
   rather than papered over.

---

## 10. Resources, wall time, and housekeeping

- **Total steps: 184 001** in one continuous, checkpointed run (plus 1 throwaway step for the §3
  start-proof artifact, a separate instance in its own directory).
- **Total wall time: ≈ 8 930 s = 2.48 h.** Campaign: 8 330 s over 18 invocations. Gate/rule tests:
  442 s over 4. Offline CPU audits: ≈ 141 s over 6. Adapter probe: ≈ 15 s.
- **Longest single foreground invocation: 568 s**, under the 600 s cap. Every invocation was
  `nice -n 15`, ONE at a time, foreground, never backgrounded. Chunk size was set from the measured
  22–23 steps/s, and `--every 4000` means an overrun could only ever have cost one 4 000-step interval,
  never the run.
- **Compute chunks used: 25 of the 25 allowed** (19 GPU-bearing invocations + 6 CPU-only vitest audits;
  the CPU audits cost 15–38 s each and never touched the GPU or a browser).
- **Checkpoints: 38 files**, 30.9 MB each, `data/checkpoints/` (gitignored) with the trace archived
  under `data/checkpoints/trace/` — a subdirectory, which the CLI's own `soup2ves90-step*.json` scan
  skips, so archiving the trace does not slow later resumes.
- **Adapter, measured:** `{"vendor":"apple","arch":"metal-3","limits":{"maxBufferSize":4294967292,
  "maxStorageBufferBindingSize":4294967292,"maxStorageBuffersPerShaderStage":10,
  "maxComputeWorkgroupsPerDimension":65535,"maxComputeInvocationsPerWorkgroup":1024}}`
- **No orphan processes remain.** `pkill -f puppeteer_dev_chrome_profile` was run after **every**
  browser-bearing invocation and `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** every
  single time, including after the last one. No `rm -rf node_modules/.vite/deps_temp_*` was ever
  needed: no vitest worker died mid-run. No dev server was started or killed; :5199 (PID 94131) was
  left alone. `--dump-dom` was never used, and no particle array was ever transferred as JSON numbers
  (the checkpoint's base64 encoding did all the transfers, ≤ 1.1 s each).

## 11. Files

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/continuous-run-report.md` — this report. Note: the
  project's `.gitignore` line 6 ignores `.superpowers/` and **no** sibling report in this directory is
  tracked in git; this one is committed with `git add -f` because the task explicitly asked for the
  report to be committed. The deviation from the surrounding convention is deliberate and flagged here
  rather than silently introduced.
- `verify/out/continuous-run-trace.json` (new) — the machine-readable trace: all 38 checkpoints'
  full audit records (invariants, events, amphiphile census and histogram, aggregate population, and
  the three largest aggregates' shape/head-shell/cavity/encapsulated-water detail). Same convention as
  `verify/out/periodic-measurement.json`. The checkpoints themselves (38 × 30.9 MB = 1.9 GB) stay
  gitignored under `data/checkpoints/`, per that directory's own written basis; this artifact is what
  makes the report's tables re-derivable without them.
- `tests/continuous-run-audit.test.ts` (new) — the offline, GPU-free checkpoint audit: species census,
  non-finite scan, bond graph rebuilt from raw `bondSlots`, the full valence/placement rule set read
  out of `data/soup.json`, and the stage ladder recomputed with the *same* functions the live run uses.
  Discovers `soup2ves90-step*.json` under `data/checkpoints{,/trace}` or takes explicit paths in
  `CONTINUOUS_RUN_CHECKPOINTS`; skips (does not fail) when no checkpoints exist, since they are
  gitignored run artifacts.
- No source file, data file, threshold, corridor, potential, rate or recogniser was modified by this
  task. `git diff --stat` over `soup/ engine/ chem/ viewer/ verify/ data/` is empty.
