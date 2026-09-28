# The cavity did not settle — it kept the same slope through a doubled run, while the encapsulated water stayed exactly 0: the object is a network with voids, and the record is now closed

Task `closeout` (2026-08-21). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `big-box-report.md`, whose **concern §1** named this task as the highest-value next
measurement: *"The cavity did not settle, and it is the one number that improved most. 97.16 ± 7.54 σ³
against 370.8656 is 3.6× short where box 54 was 10.0× short — but it rose at all four of the last
samples, so 3.6× is the most favourable reading, not the settled one. A longer run at box 76 is the
single most valuable next measurement, and it is a pure time cost: ~13 more invocations would double
the settled length."* That is exactly what was spent, and the answer is negative in a way that is more
informative than a plateau would have been.

`data/params.json` rank-A constants: **NOT touched.** `data/soup.json`: **NOT touched, not one field.**
`co_bond.attemptRate`: **NOT touched** (refused a tenth time). `stageThresholds.enclosedVolume` =
370.8656: **read, not moved.** No threshold widened, no corridor relaxed, no tolerance changed, no
assertion weakened, no rank-A/B/C/D constant re-fitted. **No engine file changed at all** — `git diff`
over `soup/`, `engine/`, `chem/`, `viewer/` and `data/` is empty, which is the strongest possible
attribution for the one known-failing test (§7). One new file, and it is an instrument, not physics:
`verify/viewer-smoke.ts` (30 lines). No fusion probe. `tests/soup-vesicle.test.ts` **never run**.

---

## 0. VERDICT UP FRONT, TEN LINES

1. **THE PROJECTION WAS MADE BEFORE SPENDING ANYTHING, AND IT WAS RIGHT TO 4 %.** 13 chunks × ~426 s
   projected = **92.4 min**; actual **13 chunks, 5 380 s = 89.7 min**, longest single invocation
   **438 s** against the 500 s cap (§1, §2).
2. **THE SETTLED LENGTH IS DOUBLED AND THEN SOME: 50 100 → 111 800 steps (2.23×)**, global step
   86 500 → **148 200** (§2).
3. **THE CAVITY DID NOT PLATEAU, AND THE PROOF IS THAT THE SLOPE DID NOT MOVE.** Over the 13 NEW
   checkpoints the cavity rises at **+0.586 σ³ per 1000 steps**, against **+0.695** over all 23 — the
   same straight line, not a saturation. Last-four scatter **4.69 %** against this project's own
   plateau standard of **0.25 %** (§3).
4. **AND THE SIZE PLATEAUED HARDER THAN EVER ON THE SAME DATA: 4519.5 ± 5.2 = 0.11 %** over the last
   four, flatness 0.7541 ± 0.0059 = 0.78 %. So the structure stands still while the cavity walks —
   this is not a run that failed to converge (§3.2).
5. **THE CAVITY-VERSUS-LUMEN QUESTION IS ANSWERED BY MEASUREMENT, NOT ARGUMENT. The cavity grew 2.90×
   (48.125 → 139.25 σ³) while `encapsulatedWater` stayed EXACTLY 0 at all 23 wet checkpoints.** A
   growing cavity inside a percolating object does **not** become a lumen. The two quantities came
   apart over a factor of three (§4).
6. **NO VESICLE, AT 23 OF 23.** `closed` false, `hasVesicleAggregate` false, `radialHeadShells`
   oscillates 1↔2 with no trend, flatness 0.7277–0.7610 against ≤ 0.35, and the object still wraps
   **3 of 3 axes and 27 of 27 slabs at every one of 23 wet checkpoints** (§5).
7. **EXCHANGE 0, FISSIONS 0, MERGES 0 at all 22 intervals**, free fraction 0…6.6e−4, largest covalent
   component 21 → 35 beads of 30 361 (`covalentSpanFraction` 0.00081 → 0.00115) — the percolating
   object is a **contact** network of 4512 chemically separate molecules (§5.2).
8. **THE RECORD IS CLOSED.** `docs/soup-to-vesicle-verdict.md` rewritten 363 → **650 lines**: it now
   covers the whole electrostatics era it predated (its old §11 said *"there is no electrostatics at
   all"*), the retired memory ceiling, the L\* mechanism, the shortfall sequence with what each step
   retired, **seven** defects, **six** silent-failure classes plus a seventh found in this task, and a
   six-item list of what a next model needs (§6).
9. **GATES REGENERATED THROUGH `npm run verify`, NEVER HAND-EDITED, AND THE PUBLISHED CAMPAIGN ROWS
   ARE THIS TASK'S.** `mean-tail-length` = **3.419** is this run's own final row (big-box's was
   3.207), `aggregate-percolation` = 3 from **23** `role:"campaign"` rows (§8).
10. **THE PAGE WORKS: `tests/run-ui.test.ts` 8/8**, plus one headless start — **8000 steps at
    2049–2257 steps/s, 119 rendered frames, 661 drawn instances including the clay platelet, water
    correctly not drawn, screenshot 173 415 bytes** (§9). And the "draws" instrument I first wrote
    returned zeros on a page that was demonstrably rendering — a seventh silent-failure class, mine,
    reported rather than quietly fixed (§9.1).

---

## 1. THE WALL-TIME PROJECTION, MADE BEFORE RUNNING ANYTHING

Basis: the ten 5000-step wet chunks in `big-box-report.md` §15 measured **414–439 s total each**
(mean 426.3, max 439). No new measurement was needed and none was taken — the point of a projection is
that it precedes the spend.

```
$ python3 -c "..."   # arithmetic only, no GPU
measured 5000-step wet chunk totals (big-box S15): [424, 439, 430, 425, 431, 436, 424, 418, 422, 414]
mean s/chunk 426.3 max 439
10 chunks -> 50000 steps, global 136500 settled 100100 ratio 2.0 wall 71.0 min
13 chunks -> 65000 steps, global 151500 settled 115100 ratio 2.3 wall 92.4 min
15 chunks -> 75000 steps, global 161500 settled 125100 ratio 2.5 wall 106.6 min
```

**Chosen: 13 chunks, ≈92 min, settled length 2.3×** — the predecessor's own recommendation. Budget
plan written at the same time: 13 campaign + 3 instruments + 4 regression groups + 1 verify + 1 viewer
+ 1 tsc = **23 of 25**.

**Actual: 13 chunks, 5 380 s of stepping = 89.7 min** (408, 438, 405, 409, 404, 403, 404, 408, 411,
409, 417, 423, 419 s). **The projection was 3 % long on wall time and exact on count.** Chunk 2 came
in at 438 s — 88 % of the 500 s cap — so the step budget was cut **5000 → 4700 from chunk 3 onward**
rather than gambling on a cap the bond count was walking towards (bonds 39 622 → 43 732 over the run).
That is why the run is 61 700 steps and not 65 000.

**Resource rules, kept.** `nice -n 15`, **one invocation at a time, foreground, never backgrounded**;
every multi-file vitest group ran `--no-file-parallelism`, so no two compute processes ever existed at
once. Longest invocation **438 s**. No dev server started or killed — :5199 was never touched (the
vite processes visible on :5185 and two ephemeral ports belong to other projects and were left alone).
`timeout` never used. `--dump-dom` never used, no whole-particle array ever crossed the CDP boundary.

---

## 2. THE CONTINUATION, AND WHAT "RESUME" GUARANTEES

Resumed from `data/checkpoints/bbB76/bbB76-step86500.json` with the *identical* command the predecessor
used, thirteen times:

```
nice -n 15 npx tsx soup/cli/campaign.ts --label bbB76 --box 76 \
  --start '{"C":55988,"O":18659,"H":55988,"M":1452,"W":351181}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --charge --pH 7.0 --ionicStrength 0.01 \
  --steps <n> --every <n> --dir data/checkpoints/bbB76
```

**What the guarantee is, stated plainly: the same ENSEMBLE, not the same trajectory.** The checkpoint
carries the full Langevin RNG state per particle (that is what defect §7.2 of the verdict document
fixed — before it, all 72 checkpoints on disk carried zero seeds), and the configuration signature is
checked field by field, so a mismatched file is skipped rather than silently resumed into. But chunk
boundaries fall in different places than they would in one unbroken process, so the noise is drawn at
different moments: this is the same distribution continued, not a bit-identical continuation. The CLI
says so itself on every resume — `--relax skipped: this is a resume from step=N` — and no claim here rests on
trajectory identity.

Real output of the first and last chunks, unedited:

```
[campaign] resume label=bbB76 from data/checkpoints/bbB76/bbB76-step86500.json, step=86500
[campaign] system ready N=483268 startStep=86500 target=91500
[campaign] --relax skipped: this is a resume from step=86500, minimization is only allowed on a fresh start
[campaign] step=91500/91500 stage=amphiphiles aggregates=2 largest=4570 headShells=2 cavityVolume=101.250 phase=wet/0 box=76.0000 bonds=39622 census={"C":55988,"O":18659,"H":55988,"M":1452,"W":351181,"K":0} stepMs=402017 checkpointMs=849 progressMs=1456 saved=data/checkpoints/bbB76/bbB76-step91500.json rc_es=15.2000(=4.000lD, discarded=0.0916) pH=7 I=0.01 alpha=0.7513 pKaApp=6.520 paired=0.0762 unlike=0.4782(random 0.3737) sweeps=5 last(accepted=9203/18659 dEs=3.8471kT)
[campaign] step budget fully completed: step=91500
nice -n 15 npx tsx soup/cli/campaign.ts ...  23.56s user 37.97s system 15% cpu 6:47.78 total
```

```
[campaign] --relax skipped: this is a resume from step=143500, minimization is only allowed on a fresh start
[campaign] step=148200/148200 stage=amphiphiles aggregates=4 largest=4512 headShells=2 cavityVolume=139.250 phase=wet/0 box=76.0000 bonds=43732 census={"C":55988,"O":18659,"H":55988,"M":1452,"W":351181,"K":0} stepMs=412002 checkpointMs=1470 progressMs=1879 saved=data/checkpoints/bbB76/bbB76-step148200.json rc_es=15.2000(=4.000lD, discarded=0.0916) pH=7 I=0.01 alpha=0.7511 pKaApp=6.520 paired=0.0766 unlike=0.4834(random 0.3739) sweeps=4 last(accepted=9372/18659 dEs=3.8552kT)
[campaign] step budget fully completed: step=148200
nice -n 15 npx tsx soup/cli/campaign.ts ...  26.70s user 47.95s system 17% cpu 6:59.13 total
```

| | box-76 campaign as `big-box` left it | **after this task** |
|---|---|---|
| global step | 86 500 | **148 200** |
| settled wet steps (since rehydration at 36 400) | 50 100 | **111 800 (2.23×)** |
| wet checkpoints with a reportable closure number | 10 | **22** (of 23 wet) |
| bonds | 39 105 | **43 732** (+11.8 %) |
| `meanPerTail` | 3.207 | **3.419** (+6.6 %) |
| census | `{C:55988, O:18659, H:55988, M:1452, W:351181}` | **identical, all 13 new checkpoints** |
| non-finite positions / velocities | 0 | **0** |
| valence violations (all six counters) | 0 | **0** |

**Nothing was reset and nothing was pre-made**: the run's own step-3000 checkpoint is still in the
trace and still reads `stage=monomers`, 270 aggregates whose largest holds **4** amphiphiles.

---

## 3. THE CAVITY TRAJECTORY, PER CHECKPOINT, AND THE PLATEAU VERDICT

Measured off-GPU by the run's own auditor over all 29 checkpoints in one 34 s pass
(`verify/out/closeout-campaign-B76-trace.json`). *(Italic rows are dry-phase or pre-aggregation; no
structural claim rests on them.)*

| step | box | amph | aggs | **largest** | r_g | flat | inPl | **radSh** | **cavity σ³** | **encH₂O** | thresh | closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| *3 000* | *76* | *316* | *270* | *4* | *3.09* | *0.020* | *0.375* | *n/a* | *0* | *n/a* | — | — |
| *8 000* | *76* | *593* | *407* | *5* | *2.29* | *0.179* | *0.264* | *n/a* | *0.125* | *0* | *296.71* | *false* |
| *11 000* | *76* | *751* | *466* | *11* | *5.42* | *0.136* | *0.215* | *n/a* | *0* | *0* | *296.73* | *false* |
| *23 400* | *46.22* | *4324* | *1* | *4324* | *23.21* | *0.940* | *0.963* | *1* | *54.0* | *centre-untrusted* | — | — |
| *31 900* | *46.22* | *4760* | *1* | *4760* | *23.05* | *0.960* | *0.973* | *1* | *95.375* | *centre-untrusted* | — | — |
| 36 400 | 76 | 4825 | 1 | 4825 | 37.993 | 0.8389 | 0.8620 | 2 | 43.875 | centre-untrusted | — | — |
| 41 100 | 76 | 4774 | 1 | 4774 | 38.014 | 0.8335 | 0.8593 | 2 | 48.125 | **0** | 312.64 | false |
| 46 300 | 76 | 4744 | 1 | 4744 | 37.937 | 0.8251 | 0.8551 | 2 | 64.625 | **0** | 312.91 | false |
| 51 500 | 76 | 4695 | 3 | 4693 | 37.981 | 0.8237 | 0.8459 | 1 | 79.125 | **0** | 313.15 | false |
| 56 500 | 76 | 4663 | 4 | 4660 | 37.917 | 0.8166 | 0.8265 | 1 | 74.875 | **0** | 313.31 | false |
| 61 500 | 76 | 4639 | 4 | 4636 | 37.333 | 0.7555 | 0.9806 | 1 | 80.0 | **0** | 313.55 | false |
| 66 500 | 76 | 4625 | 4 | 4622 | 37.359 | 0.7507 | 0.9711 | 1 | 91.875 | **0** | 313.76 | false |
| 71 500 | 76 | 4619 | 4 | 4616 | 37.384 | 0.7403 | 0.9599 | 1 | 87.625 | **0** | 313.99 | false |
| 76 500 | 76 | 4619 | 3 | 4617 | 37.373 | 0.7391 | 0.9581 | 1 | 94.875 | **0** | 314.22 | false |
| 81 500 | 76 | 4604 | 2 | 4603 | 37.364 | 0.7402 | 0.9568 | 1 | 101.5 | **0** | 314.26 | false |
| 86 500 | 76 | 4590 | 2 | 4589 | 37.327 | 0.7374 | 0.9587 | 1 | 104.625 | **0** | 314.37 | false |
| **91 500** | 76 | 4571 | 2 | **4570** | 37.380 | 0.7277 | 0.9392 | **2** | **101.25** | **0** | 314.38 | false |
| **96 500** | 76 | 4566 | 2 | **4565** | 37.358 | 0.7370 | 0.9548 | **2** | **109.25** | **0** | 314.48 | false |
| **101 200** | 76 | 4551 | 3 | **4548** | 37.355 | 0.7406 | 0.9607 | 1 | **111.375** | **0** | 314.55 | false |
| **105 900** | 76 | 4553 | 3 | **4551** | 37.408 | 0.7412 | 0.9560 | 1 | **111.875** | **0** | 314.68 | false |
| **110 600** | 76 | 4555 | 3 | **4553** | 37.343 | 0.7422 | 0.9550 | 1 | **111.75** | **0** | 314.81 | false |
| **115 300** | 76 | 4548 | 3 | **4546** | 37.351 | 0.7434 | 0.9501 | 1 | **121.5** | **0** | 314.90 | false |
| **120 000** | 76 | 4534 | 4 | **4531** | 37.336 | 0.7445 | 0.9440 | 1 | **121.75** | **0** | 314.98 | false |
| **124 700** | 76 | 4529 | 4 | **4526** | 37.384 | 0.7469 | 0.9454 | 1 | **118.5** | **0** | 315.03 | false |
| **129 400** | 76 | 4521 | 4 | **4518** | 37.358 | 0.7475 | 0.9473 | **2** | **128.75** | **0** | 315.09 | false |
| **134 100** | 76 | 4527 | 4 | **4524** | 37.428 | 0.7484 | 0.9429 | 1 | **125.25** | **0** | 315.23 | false |
| **138 800** | 76 | 4523 | 3 | **4521** | 37.414 | 0.7564 | 0.9675 | 1 | **136.0** | **0** | 315.27 | false |
| **143 500** | 76 | 4523 | 3 | **4521** | 37.446 | 0.7610 | 0.9693 | **2** | **130.125** | **0** | 315.38 | false |
| **148 200** | 76 | 4515 | 4 | **4512** | 37.426 | 0.7571 | 0.9742 | **2** | **139.25** | **0** | 315.41 | false |

### 3.1 The plateau verdict: NOT a plateau, and the slope is the reason

The standard this project has used elsewhere is explicit: *the size plateau was 0.25 % over tens of
thousands of steps; a rising curve is not a result.* Applied to the cavity:

| window | n | mean | sd | **relative sd** | **least-squares slope** |
|---|---|---|---|---|---|
| all 23 settled wet (41 100 → 148 200) | 23 | 104.08 | 23.62 | **22.70 %** | **+0.695 σ³/1000 steps** |
| **only the 13 NEW (91 500 → 148 200)** | 13 | 120.51 | 11.23 | **9.32 %** | **+0.586 σ³/1000 steps** |
| last 8 (115 300 → 148 200) | 8 | 127.64 | 7.31 | 5.72 % | +0.545 |
| last 6 (120 000 → 148 200) | 6 | 129.65 | 7.45 | 5.74 % | +0.721 |
| **last 4 (129 400 → 148 200)** | 4 | 132.66 | 6.22 | **4.69 %** | +0.769 |

**The discriminating number is the slope, not the scatter.** Doubling the settled length changed the
slope from +0.695 to +0.586 σ³ per 1000 steps — statistically the same line. A quantity approaching a
plateau shows a *falling* slope over its late window; this one does not. The scatter is 4.69 % on the
last four against a 0.25 % standard, i.e. **19× outside it**, and the direction is monotone-ish up
(9 rises against 4 falls over the 13 new samples).

**How far it would have to go, stated so nobody has to guess.** At the observed slope the cavity
reaches 370.8656 σ³ after another **≈333 000 steps ≈ 71 invocations ≈ 8.5 hours** — and that is a
linear extrapolation of a quantity with no reason to be linear, over 2.7× the settled length already
run. It is quoted as a scale, not a forecast.

**Best available reading, labelled as such:** final 139.25 → **2.66× short** of 370.8656; plateau-mean
of the last four 132.66 → **2.80× short**. The predecessor's was 3.6× short. So the number improved
again (10.0× → 3.6× → **2.66×**) and **it still is not a result**, for the same reason it was not one
last time.

### 3.2 What DID settle — and this is what makes the cavity's failure to settle interesting

| quantity | last four samples | mean ± sd | relative |
|---|---|---|---|
| **largest aggregate** | 4518, 4524, 4521, 4512 | **4519.5 ± 5.2** | **0.11 %** |
| flatness λ₁/λ₃ | 0.7475, 0.7484, 0.7564, 0.7610, 0.7571 (last 5) | 0.7541 ± 0.0059 | 0.78 % |
| in-plane λ₂/λ₃ | 0.9473, 0.9429, 0.9675, 0.9693, 0.9742 | 0.9602 ± 0.0143 | 1.49 % |
| r_g | 37.358 … 37.426 | 37.42 ± 0.03 | 0.08 % |
| wrapping axes | 3, 3, 3, 3 | **3** (a boolean, unanimous at 23 of 23) | — |

**0.11 % on size is the tightest plateau this project has ever published** — tighter than big-box's
0.25 % and than the box-54 campaign's 0.37 %. So the run is not under-converged in general: the
structure is standing still and the cavity is walking. Two things are still moving with it, exactly as
in every predecessor: bonds (+11.8 %) and `meanPerTail` (3.207 → 3.419). The settled object is a
steady state of a **still-growing molecule**, and the cavity is tracking the molecule, not the shape.

---

## 4. THE QUESTION THIS TASK EXISTED TO ANSWER: CAVITY, OR LUMEN?

An earlier report warned that `cavityVolume` on a network is not a lumen. That was an argument. This
run turns it into a measurement, because the two quantities were watched together across a factor of
three:

| | at 41 100 | at 148 200 | change |
|---|---|---|---|
| `cavityVolume` | 48.125 σ³ | **139.25 σ³** | **× 2.895** |
| `encapsulatedWater` | **0** of 312.64 | **0** of 315.41 | **× 1 — exactly zero throughout** |
| `closed` | false | false | no change |
| wrapping axes | 3 of 3 | 3 of 3 | no change |
| `radialHeadShells` | 2 | 2 | oscillated 2→1→2, no trend |

**THE ANSWER: NO. A growing cavity inside a percolating object does not become a lumen.** The cavity
nearly tripled and the encapsulated water did not move off zero by a single bead, at 23 of 23 wet
checkpoints spanning 107 100 steps. `encapsulatedWater` is a flood-fill question — *is there water
that cannot reach the bulk* — and `cavityVolume` is an interstitial-filling question inside the
object's own bounding region. On a network the second grows as the mesh coarsens (bonds +11.8 %,
`meanPerTail` +6.6 %: the strands are getting longer and the holes between them bigger) and the first
cannot move at all, because there is no closed surface anywhere.

**So the object is a NETWORK WITH VOIDS, not a closed compartment.** That was the reading the
predecessor offered as a caution; after this run it is a measured statement standing on 23 samples
instead of 11, and on a 2.9× excursion of the very number that was supposed to be trending towards
closure.

**The consequence for how the number should be published:** quoting "139.25 against 370.8656, 2.66×
short" as a closure shortfall is **generous to the model, not harsh** — interstitial void in a network
is not the interior of a protocell at all. That sentence is now in the committed verdict document, not
only here.

**And the honest framing, which would have been owed even on a success.** Box 76 σ is **0.47× of
L\* = 160.6–161.3 σ**, the size above which closure becomes the thermodynamically preferred object
(the band [closure floor 930–955, cheapest spanning object 440–450 at L = 76] is **empty and inverted
by 2.07–2.17×**). Anything that closed here would have needed that context rather than a headline.
Nothing closed.

---

## 5. THE OTHER INSTRUMENTS, PER CHECKPOINT

### 5.1 Axis wrapping — the discriminating measurement, failed again, at 23 of 23

```
$ PERC_CAMPAIGN_LABEL=bbB76 PERC_CHECKPOINTS="<23 wet + 3 controls>" PERC_ARTIFACT=verify/out/gates-percolation.json \
    nice -n 15 npx vitest run tests/percolation-check.test.ts --no-file-parallelism
PERC {"file":"...bbB76-step91500.json","step":91500,"box":76,"aggregates":2,"amphiphilesInLargest":4570,"particlesInLargest":28761,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[27,27,27,27],"role":"campaign"}
PERC {"file":"...bbB76-step120000.json","step":120000,"box":76,"aggregates":4,"amphiphilesInLargest":4531,"particlesInLargest":29622,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[27,27,27,27],"role":"campaign"}
PERC {"file":"...bbB76-step148200.json","step":148200,"box":76,"aggregates":4,"amphiphilesInLargest":4512,"particlesInLargest":30361,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[27,27,27,27],"role":"campaign"}
PERC {"file":"...dec54-step41400.json","step":41400,"box":54,"aggregates":22,"amphiphilesInLargest":200,"particlesInLargest":704,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[12,13,12,19],"role":"control"}
PERC {"file":"...dec54-step174400.json","step":174400,"box":54,"aggregates":25,"amphiphilesInLargest":151,"particlesInLargest":596,"wrapsX":false,"wrapsY":false,"wrapsZ":false,"wrappingAxes":0,"slabsTouchedOfTotal":[7,7,10,19],"role":"control"}
PERC {"file":"...swB54-step144400.json","step":144400,"box":54,"aggregates":1,"amphiphilesInLargest":1542,"particlesInLargest":10305,"wrapsX":true,"wrapsY":true,"wrapsZ":true,"wrappingAxes":3,"slabsTouchedOfTotal":[19,19,19,19],"role":"control"}
 Test Files  1 passed (1)     Duration  15.81s
```

**3 of 3 axes, 27 of 27 slabs, at all 23** (the 10 pre-existing rows are unchanged from `big-box`, the
13 new ones read the same). **The must-say-no controls still say no on the same day with the same
code** — `dec54`'s 200- and 151-molecule rods at 0 axes and 7–13 of 19 slabs — so the instrument is not
stuck on `true`, and it is not stuck on the slab count either (27 at box 76, 19 at box 54). The
positive control (`swB54`) still reads 3 of 3.

`particlesInLargest` grew 28 681 → **30 361** (+5.9 %) at a *falling* amphiphile count (4589 → 4512),
which is the molecule getting longer, not the object getting bigger.

### 5.2 Exchange, fission, covalent span — the network is a CONTACT network, still

```
$ ES_AUDIT_CHECKPOINTS="<23 wet>" ES_AUDIT_ARTIFACT=verify/out/closeout-audit.json \
    nice -n 15 npx vitest run tests/electrostatics-audit.test.ts --no-file-parallelism
ES-AUDIT {...step 148200 ... "amphiphiles":4515,"aggregates":1,"largest":4512,"freeAmphiphileFraction":0.00066,
  "alphaAll":0.75106,"pKaAppAll":6.5204,"interfacialShift":0.0972,"pairedFraction":0.07659,
  "unlikeContactFraction":0.48338,"unlikeContactFractionRandom":0.37394,"unlikeExcessRatio":1.2927,
  "covalentComponentsInLargest":4512,"largestCovalentComponent":35,"covalentSpanFraction":0.00115,
  "merges":0,"fissions":0,"exchangedFraction":0,"matchedHeads":4508,
  "flatness":0.7571,"inPlaneSymmetry":0.9742,"radiusOfGyration":37.4259}
 Test Files  1 passed (1)     Duration  22.20s
```

| quantity | over the 22 intervals | note |
|---|---|---|
| **`merges`** | **0 at all 22** | |
| **`fissions`** | **0 at all 22** | |
| **`exchangedFraction`** | **0.000000 at all 22** | monomer exchange is a channel this model does not have |
| `freeAmphiphileFraction` | 0 … 6.6e−4 | 1–3 dissolved molecules of ~4500 |
| `largestCovalentComponent` | 21 → **35** of 30 361 particles | `covalentSpanFraction` 0.00081 → **0.00115** |
| α_all | 0.7472–0.7550 | steady, as calibrated |
| pKa_app | 6.5146–6.5295 | steady |
| interfacial shift | 0.0532–0.1108 | |
| `unlikeExcessRatio` | 1.231–1.375 | acid–soap correlation persists |
| `pairSurvivalFraction` | 0.198 → **0.244** | pairs keep getting more persistent — **not** a closure signal |

**The percolating object is a contact network of 4512 chemically separate molecules whose largest
covalent piece is 35 beads.** Doubling the run raised the largest covalent component from 25 to 35 and
its span fraction from 0.00087 to 0.00115 — a factor 1.3 on a number that would have to reach ~1.0 for
the object to be one molecule. It is not.

---

## 6. WHAT CHANGED IN `docs/soup-to-vesicle-verdict.md`

Rewritten in place (committed, Russian, for a reader who was not here): **363 → 650 lines**. It
predated the electrostatics work entirely — its old §11 read *"there is no electrostatics at all, salt and pH
are declared unrepresentable"* — and it now covers everything since.

| section | what is new |
|---|---|
| header | a dated change-note saying exactly what the previous redaction got wrong, so a reader of a stale copy can tell |
| §1 | the project's own rules restated, plus a one-paragraph honest statement of what was built and what it does instead |
| §2 | the **electrostatics layer** added to the "what is built" table (`soup/src/electrostatics.ts`, its own 4 λ_D range and its own neighbour list) and the checkpoint/resume CLI |
| §3 | gate table **regenerated** (values, ranks, verdicts and **`corridor` beside `verdict`**, as the convention requires); the published campaign block named as box-76 so nobody mistakes it |
| **§3.4 (new)** | **the whole electrostatics era**: force verified against a numerical gradient twice (5.145e−9 pair, 1.736e−5 full field), Henderson–Hasselbalch to ≤2.08e−3, exact two-head Boltzmann to 3.140e−3, the emergent interfacial pKa shift (+0.08/+0.32/+0.67/+0.97), and **the salt shift +0.124 → +0.709 against the literature's ≈0.7 as the project's one independent validation**, with the reason it moved (a cutoff in units of its own screening length makes the discarded fraction 0.0916 in *both* salt arms instead of 0.8385 vs 0.3389) |
| §3.3 | the evaporation concentration factor **3.20× against the literature's 1400×**, the ~691× carbon enrichment, and the wet–dry result **as corrected** (1.68–1.80× yield, 8.1–8.4× largest aggregate — not the published 12.5–14.5×) |
| **§5 (new)** | **the mechanism as now understood**, as a table of what was retired and how: supply (tested at 1.50× and 4.82× the floor), edge energy (0.583 kT), **pair energy REFUTED by measurement** (gap closed 2.86× → 1.26×, wrapping-axis count moved by exactly zero), box size (2.79× volume, moved by zero) — and the survivor: the cheapest edge-free object in a periodic box is a **spanning cylindrical micelle** costing (5.806–6.635)·L against a **box-independent** closure floor of ~930–1030, so closure is preferred only above **L\* = 160.6–161.3 σ** = 3.29e6 particles against a ceiling of 953 589, structurally unreachable because a flat per-particle list costs **L³ρ²** |
| **§5.4 (new)** | the **shortfall sequence 80× → 8.9× → 3.36× → 33.4× → 6.76× → 6.75× → 0.48× → 0.42× → 0.67× → 0.21×** with what each step retired, and with the two entries that are *known wrong* flagged (3.36× → 1.85× under the RNG fix; 6.75× → 6.02× under the z fix) |
| **§6 (new)** | **this task**: the cavity trajectory, the plateau arithmetic, and the cavity-versus-lumen answer with the L\* framing |
| §7 | defects **5 → 7**: the percolation gate's hardcoded label and the click-eating viewer layout added, each with the numbers it moved |
| **§7.8 (new)** | **the six silent-failure classes** enumerated as a list, including the newest — a device-limit refusal that returned `max|F| = 0`, 0 non-finite, 0 neighbours at **0.0405 ms/step**, a 500× speed-up that looked like a clean bill of health — plus a **seventh found in this task and owned** (§9.1) |
| §8 | unproven/one-seed/flaky rewritten to 17 items: the cavity's non-settlement is now item 2, `centre-untrusted` correctly described as a **Rayleigh test for circular uniformity** (item 10, correcting a misreading two reports made), zero exchange/fission/merge as item 6, the rank-D capacity judgement, the noisy 1.35 exponent, the device-specific structure decision, the known-flaky tests with the `git stash` proof |
| §9 | **the six-item list of what a next model needs**, in expected-return order: two-tailed by **topology** at fixed short tail; a dilution axis independent of the chemistry; **a non-periodic or large-enough domain so a spanning object is not the ground state** (named as the root); real evaporation past 3.20×; monomer exchange / solubility equilibrium; co-surfactant mixtures — and the cheapest fully-specified next measurement, with the warning that it must be run in a non-periodic domain or item 3 will eat it |
| §10 | the reproduction commands updated to this task's real ones and timings, including the resume command with the ensemble-not-trajectory caveat and `verify/viewer-smoke.ts` |

Convention kept throughout: gates published as **passed / failed / unproven**, never widened, with
`corridor` printed beside `verdict`.

---

## 7. REGRESSIONS — EVERY ONE RUN, BEFORE → AFTER

The diff contains **no engine, no data and no chemistry change** (`git diff --stat` over `soup/`,
`engine/`, `chem/`, `data/` is empty), so the interesting column is "did anything move anyway".
Nothing did.

| file | before (`big-box`) | **after** | note |
|---|---|---|---|
| `tests/soup-forces.test.ts` | 3 passed | **3 passed** | grid/Verlet/brute-force agreement incl. the charged path |
| `tests/soup-checkpoint.test.ts` | 3 passed | **3 passed** | the resume path this whole task depends on |
| `tests/verlet-capacity.test.ts` | 2 passed | **2 passed** | both guards fired: `VERLET-OVERFLOW` at capacity 64, `VERLET-DERIVED-OK {"cap":1126,"max":322,"atCapacity":0}`, and `VERLET-CEILING … 17280000000 bytes … WebGPU would fail SILENTLY` |
| `tests/gates.test.ts` | 8 passed | **8 passed** | |
| `tests/supply-window.test.ts` | 1 passed | **1 passed** | `WINDOW-SCALING floor=[932, 955] cheapestSpanning=[5.806, 5.917]*L L*=[157.4, 164.4]sigma … N_at_L*=3.121e+6 overCeiling=7.3x` — the pin brackets big-box's re-derived L\* = 160.6–161.3 |
| `tests/coalescence-mechanism-pin.test.ts` | 5 passed | **5 passed** | |
| `tests/coalescence-mechanisms.test.ts` | 1 passed | **1 passed** | |
| `tests/es-range-calibration.test.ts` | skipped (1) | **skipped (1)** | needs `ES_CAL_CHECKPOINTS` |
| `tests/gate6-bilayer.test.ts` | 3 passed | **3 passed** | `GATE6 area 1.2017 ± 0.0060 thickness 4.4902 … escapedMax 0 box 24.567 steps 83000` |
| `tests/water-bilayer-area-move.test.ts` | 1 passed | **1 passed** | `a=1.1607 [1.1519, 1.1674] t=4.6871 settled=true driftT=1.40 clusterFraction=1.0000 waterInCore=33/4500 verdict=passed` |
| `tests/soup-electrostatics.test.ts` | 8 passed | **8 passed** | |
| **`tests/run-ui.test.ts`** | 8 passed | **8 passed** | puppeteer; profile killed after, `pgrep` clean |
| `tests/continuous-run-audit.test.ts` | 1 passed | **1 passed** | **29** checkpoints, tether violations 0, invariants clean |
| `tests/percolation-check.test.ts` | 1 passed | **1 passed** | 23 campaign rows + 3 controls |
| `tests/electrostatics-audit.test.ts` | 1 passed | **1 passed** | re-pointed at this campaign's 23 wet checkpoints |
| `tests/soup-grid-resize.test.ts` | **1 failed** | **1 failed** | **HEAD's own side, and this task proves it more strongly than a stash could**: not one byte of `soup/`, `engine/` or `data/` changed, so the failure cannot be attributable to this diff. Same message, same line: `Verlet list: measured drift 55675.0687 exceeds skin/2=0.7500`. Not chased, per the brief |
| `npx tsc --noEmit` | 21 errors | **21 errors** | the pre-existing `ArrayBufferLike`/`SharedArrayBuffer` class, unchanged in count — the new `verify/viewer-smoke.ts` adds zero |

Known flaky, **not chased**, per the brief: `soup-drywet-cycling` (margin-flaky, event ratio 0.43×
spread on unchanged code), `rim-lambda-insitu` (2 of 3 pre-existing).
`tests/soup-vesicle.test.ts` **never run**.

---

## 8. THE GATE TABLE, REGENERATED THROUGH `npm run verify` (never hand-edited)

```
$ nice -n 15 npm run verify
✓ built in 18ms
GATE area-per-lipid: value=1.2004636215678743 rank=A verdict=passed
GATE bilayer-thickness: value=4.430766935292141 rank=A verdict=passed
GATE bending-modulus: value=null rank=A verdict=unproven
GATE area-per-lipid-water: value=1.1607493773305628 rank=C verdict=passed
GATE bilayer-thickness-water: value=4.68713096487309 rank=C verdict=passed
GATE vesicle-closure-water: value=0 rank=B verdict=failed
GATE aggregate-percolation: value=3 rank=B verdict=failed
GATE vesicle-verdict: value=0 rank=A verdict=failed
GATE chain-length-asf: value=0.15940333430827724 rank=D verdict=unproven
GATE mean-tail-length: value=3.419 rank=D verdict=unproven
GATE closure: value=1284.875 rank=D verdict=unproven
GATE chain-to-bead-mapping: value=3 rank=D verdict=unproven
```

Read back from `verify/out/gates.json` (`generatedAt 2026-08-20T23:23:30.106Z`), with the `corridor`
field the convention requires:

| gate | value | rank | verdict | corridor |
|---|---|---|---|---|
| area-per-lipid | 1.2005 | A | passed | inside |
| bilayer-thickness | 4.4308 | A | passed | inside |
| bending-modulus | null | A | **unproven** | none |
| area-per-lipid-water | 1.1607 | C | passed | inside |
| bilayer-thickness-water | 4.6871 | C | passed | inside |
| **vesicle-closure-water** | **0** | B | **failed** | outside |
| **aggregate-percolation** | **3** | B | **failed** | outside |
| vesicle-verdict | 0 | A | failed | outside |
| chain-length-asf | 0.1594 | D | unproven | none |
| mean-tail-length | **3.419** | D | unproven | outside |
| closure | 1284.875 | D | unproven | inside |
| chain-to-bead-mapping | 3 | D | unproven | none |

**The published campaign rows are the ones this task produced**, and the proof is internal, not a
promise: `mean-tail-length` = **3.419** is the final row of `bbB76-step148200` (big-box's table read
3.207 from step 86 500), and `aggregate-percolation` = 3 is reduced from **23** rows carrying
`role:"campaign"`. `verify/out/gates-campaign-trace.json` is a verbatim copy of this task's own
`closeout-campaign-B76-trace.json` (29 checkpoints, last step 148 200). No threshold moved:
`vesicle-closure-water`, `aggregate-percolation` and `vesicle-verdict` are still **failed**, and
`bending-modulus` is still **unproven** with its κ fit still invalid.

Movements against big-box's table, all stochastic re-measurements of the same gates, all still inside
their corridors: area-per-lipid 1.2052 → 1.2005, thickness 4.4429 → 4.4308, water area 1.1637 →
1.1607, water thickness 4.3839 → **4.6871** (the largest move, +6.9 %, still well inside 4–6),
chain-length-asf 0.1416 → 0.1594, mean-tail-length 3.207 → **3.419**.

---

## 9. THE VIEWER CHECK — THE THING THE USER ACTUALLY OPENS

**`tests/run-ui.test.ts`: 8 of 8 passed**, unchanged, including the two that matter most here — the
click-target hit test (the fix for §7.7 of the verdict document holds) and the ocean-look test that
proves frames are still being produced.

```
 ✓ tests/run-ui.test.ts (8 tests) 37191ms
   ✓ run control: start keeps the count, pause stops it, stop stops it permanently 4348ms
   ✓ second run on the same page: the step count advances past one STEP_BATCH 1589ms
   ✓ the run survives a bead that flew out of the box along z, and does not crash for longer than usual (item 1a) 24758ms
   ✓ the scene does not disappear when the camera is turned across several distances and angles (item 2) 843ms
   ✓ too small a box is rejected with a clear message, rather than crashing the run (item 3) 1490ms
   ✓ composition: the head count is set directly, is visible in the preview, and actually reduces the broth (item "composition") 1232ms
   ✓ composition: a negative head count is rejected with a clear message, rather than crashing the run (item "composition") 1586ms
   ✓ ocean background: the run starts and draws frames, the honesty notes are in place, the screenshot is non-trivial (ocean-look) 1095ms
```

And one headless start of the real page, as its own committed instrument
(`verify/viewer-smoke.ts`, 30 lines):

```
$ nice -n 15 npx tsx verify/viewer-smoke.ts
POLL 0 state=running steps=250 steps/s=549 stage=monomers
POLL 1 state=running steps=1250 steps/s=2257 stage=monomers
POLL 2 state=running steps=2500 steps/s=2192 stage=monomers
POLL 3 state=running steps=3500 steps/s=2119 stage=monomers
POLL 4 state=running steps=4500 steps/s=2057 stage=monomers
POLL 5 state=running steps=5500 steps/s=2054 stage=monomers
POLL 6 state=running steps=6500 steps/s=2057 stage=monomers
POLL 7 state=running steps=7500 steps/s=2055 stage=monomers
POLL 8 state=stopped steps=8000 steps/s=2049 stage=monomers
DRAWN {"renderedFrameCount":119,"instances":{"C":{"count":200,"visible":true,"capacity":3938},"O":{"count":50,"visible":true,"capacity":3938},"H":{"count":200,"visible":true,"capacity":3938},"M":{"count":20,"visible":true,"capacity":3938},"W":{"count":0,"visible":false,"capacity":3938},"K":{"count":191,"visible":true,"capacity":3938}},"screenshotBytes":173415}
FINAL {"st":"stopped","n":8000,"stage":"monomers","trace":11}
```

**A run progresses** (idle → running → 8000 steps → stopped at its own cap, 2049–2257 steps/s, 11 trace
rows written) **and it draws** (119 rendered frames; **661 drawn instances** across five species with
the clay platelet at 191 visible, water correctly *not* drawn as the page's own default; a real
173 415-byte screenshot).

### 9.1 A seventh silent-failure class, and it was mine

The first version of that instrument measured "draws" by reading the canvas through a 2d
`drawImage()` and counting pixels differing from the corner. It reported:

```
DRAWN {"w":800,"h":600,"sampled":30000,"differing":0,"fraction":0,"corner":[0,0,0]}
FINAL {"st":"stopped","n":8000,"stage":"monomers","trace":11}
```

**Zero of thirty thousand pixels differing, on a page that was demonstrably rendering** (119 frames,
661 instances, a 173 kB screenshot). A WebGPU canvas read back through a 2d context comes back blank.
Had I published that number it would have read as "the page runs but does not draw" — a false regression
report about the one artefact the user actually opens. Replaced with `sceneDebug`'s own counters plus a
real `page.screenshot()`, which is what `tests/run-ui.test.ts` uses, and the failure is recorded rather
than quietly deleted: it is the same class as the six in the verdict document's §7.8, and the lesson is
identical — **zeros from an instrument are first of all a statement about the instrument.**

---

## 10. EVERY COMMAND, IN ORDER, WITH REAL OUTPUT

Outputs quoted in full above: §1 (the projection), §2 (the first and last campaign chunks), §3 (the
trace), §5.1 (percolation), §5.2 (the electrostatics audit), §7 (the regressions), §8 (the gates), §9
(the viewer). The full list:

```
# --- the projection, BEFORE spending anything (arithmetic only, no GPU)
date ; pgrep -fl puppeteer_dev_chrome_profile ; pgrep -fl vite ; python3 -c "<projection>"

# --- the continuation: 13 foreground invocations of ONE command, same lineage
nice -n 15 npx tsx soup/cli/campaign.ts --label bbB76 --box 76 \
  --start '{"C":55988,"O":18659,"H":55988,"M":1452,"W":351181}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --charge --pH 7.0 --ionicStrength 0.01 \
  --steps <n> --every <n> --dir data/checkpoints/bbB76
  # --steps 5000 -> 91 500   408 s   cavity 101.250  headShells 2  largest 4570  bonds 39 622
  # --steps 5000 -> 96 500   438 s   cavity 109.250  headShells 2  largest 4565  bonds 40 099  <- 88 % of the cap; cut to 4700
  # --steps 4700 -> 101 200  405 s   cavity 111.375  headShells 1  largest 4548  bonds 40 519
  # --steps 4700 -> 105 900  409 s   cavity 111.875  headShells 1  largest 4551  bonds 40 941
  # --steps 4700 -> 110 600  404 s   cavity 111.750  headShells 1  largest 4553  bonds 41 332
  # --steps 4700 -> 115 300  403 s   cavity 121.500  headShells 1  largest 4546  bonds 41 701
  # --steps 4700 -> 120 000  404 s   cavity 121.750  headShells 1  largest 4531  bonds 42 050
  # --steps 4700 -> 124 700  408 s   cavity 118.500  headShells 1  largest 4526  bonds 42 368
  # --steps 4700 -> 129 400  411 s   cavity 128.750  headShells 2  largest 4518  bonds 42 666
  # --steps 4700 -> 134 100  409 s   cavity 125.250  headShells 1  largest 4524  bonds 42 937
  # --steps 4700 -> 138 800  417 s   cavity 136.000  headShells 1  largest 4521  bonds 43 220
  # --steps 4700 -> 143 500  423 s   cavity 130.125  headShells 2  largest 4521  bonds 43 457
  # --steps 4700 -> 148 200  419 s   cavity 139.250  headShells 2  largest 4512  bonds 43 732

# --- the instruments, all off-GPU on the campaign's own checkpoints
CONTINUOUS_RUN_PREFIX=bbB76-step CONTINUOUS_RUN_DIRS=data/checkpoints/bbB76 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/closeout-campaign-B76-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism      # 34 s, 29 checkpoints
PERC_CAMPAIGN_LABEL=bbB76 PERC_CHECKPOINTS="<23 wet + 3 controls>" \
  PERC_ARTIFACT=verify/out/gates-percolation.json \
  nice -n 15 npx vitest run tests/percolation-check.test.ts --no-file-parallelism         # 16 s
ES_AUDIT_CHECKPOINTS="<23 wet>" ES_AUDIT_ARTIFACT=verify/out/closeout-audit.json \
  nice -n 15 npx vitest run tests/electrostatics-audit.test.ts --no-file-parallelism      # 22 s

# --- the plateau arithmetic (no GPU)
python3 -c "<least-squares slope over five windows>"

# --- regressions, three groups, each --no-file-parallelism so no two compute processes coexist
nice -n 15 npx vitest run tests/soup-forces.test.ts tests/soup-checkpoint.test.ts \
  tests/verlet-capacity.test.ts tests/gates.test.ts tests/supply-window.test.ts \
  tests/coalescence-mechanism-pin.test.ts tests/coalescence-mechanisms.test.ts \
  tests/es-range-calibration.test.ts --no-file-parallelism         # 23 passed, 1 skipped, 10.4 s
nice -n 15 npx vitest run tests/gate6-bilayer.test.ts tests/water-bilayer-area-move.test.ts \
  --no-file-parallelism                                            # 3 passed, 190 s
nice -n 15 npx vitest run tests/soup-electrostatics.test.ts tests/run-ui.test.ts \
  tests/soup-grid-resize.test.ts --no-file-parallelism              # 16 passed, 1 failed (grid-resize, HEAD's), 52 s
pkill -f puppeteer_dev_chrome_profile ; pgrep -fl puppeteer_dev_chrome_profile   # -> NO ORPHANS

# --- the page
nice -n 15 npx tsx verify/viewer-smoke.ts                          # 1st: the blank-canvas instrument bug (S9.1)
nice -n 15 npx tsx verify/viewer-smoke.ts                          # 2nd: corrected instrument, quoted in S9
pkill -f puppeteer_dev_chrome_profile ; pgrep -fl puppeteer_dev_chrome_profile   # -> NO ORPHANS

# --- the published gates and the typecheck
cp verify/out/closeout-campaign-B76-trace.json verify/out/gates-campaign-trace.json
nice -n 15 npm run verify                                          # the table in S8, 178 s
nice -n 15 npx tsc --noEmit                                        # 21 errors, unchanged
```

---

## 11. TOTALS, RESOURCES, HOUSEKEEPING

- **Compute invocations: 24 against a budget of 25 — UNDER by 1.** 13 campaign chunks, 3 instrument
  audits, 3 regression groups, 2 viewer smoke runs (the first found the instrument bug in §9.1), 1
  `npm run verify`, 1 `tsc --noEmit`, 1 machine-state/projection call. The plateau arithmetic and the
  trace extraction were pure Node/Python over artifacts already on disk and are not counted as compute.
- **Longest single foreground invocation: 438 s** (campaign chunk 2). **Every invocation under the
  500 s cap**, `nice -n 15`, **one at a time, foreground, never backgrounded**. Every multi-file vitest
  group ran `--no-file-parallelism`, so **no two compute processes ever existed at once**. The step
  budget was cut 5000 → 4700 after chunk 2 precisely so the cap could not be approached again as bonds
  grew.
- **No dev server started or killed.** :5199 was never touched. The vite processes seen on the machine
  belong to other projects (`cfa-registration-poc` and two ephemeral ports) and were left running.
- **Orphans: none.** `pkill -f puppeteer_dev_chrome_profile` after every browser group, each time
  confirmed by `pgrep -fl puppeteer_dev_chrome_profile` printing nothing (`NO ORPHANS`). No
  `node_modules/.vite/deps_temp_*` cleanup was needed — no vitest died mid-run. `timeout` never used.
- **New trajectory: 61 700 steps at N = 483 268** ≈ 2.98 × 10¹⁰ particle-steps, in 5 380 s of stepping
  (mean **87.2 ms/step**, against big-box's 79–88 at the same N — the +11.8 % bond growth is visible and
  small).
- **Memory:** unchanged from big-box, because nothing about the allocation changed — main Verlet list
  **2.482 GB** at capacity 1284 (0.578× the 4.294967292 GB binding limit), ES head list 0.440 GB,
  per-particle state ≈0.05 GB, ≈**2.97 GB** total. No allocation was refused in this task.
- **Never done:** no threshold widened, no corridor relaxed, no rank-A/B/C constant re-fitted,
  `co_bond.attemptRate` refused a tenth time, no pre-made start, no under-dense water, no `--dump-dom`,
  no whole-particle JSON transfer, no fusion probe, `tests/soup-vesicle.test.ts` not run.
- **Artifacts written:** `verify/out/closeout-campaign-B76-trace.json` (29 checkpoints),
  `verify/out/closeout-audit.json` (23), `verify/out/gates-percolation.json` (26 rows),
  `verify/out/gates-campaign-trace.json`, `verify/out/gates.json`, `verify/out/report.html`,
  `verify/out/water-bilayer-area-move.json`, `verify/out/kappa-measurement.json`. Checkpoints:
  `data/checkpoints/bbB76/` now 29 files, ~2.2 GB (gitignored, as every campaign's are).
- **New source: one file, `verify/viewer-smoke.ts`, 30 lines** — inside CLAUDE.md's 400–600 rule, and
  nothing else in `soup/ engine/ chem/ viewer/ verify/ tests/` changed size at all.

---

## 12. CONCERNS

1. **The cavity is unsettled and now demonstrably so, which closes the question the predecessor asked
   but does not answer the one underneath it.** We know the curve is still a straight line after
   111 800 settled steps; we do *not* know what it converges to, and finding out costs ~71 more
   invocations for a number that §4 argues is not measuring closure in the first place. **Recommendation:
   do not spend them.** The measurement that would matter is item 3 of the verdict document's §9 (a
   non-periodic domain), not more of this one.
2. **`radialHeadShells` oscillates 1↔2 with no trend and never settled in the whole project.** **2 at
   5 of the 13 new checkpoints** (91 500, 96 500, 129 400, 143 500, 148 200), 1 at the other eight, no correlation with the cavity or with anything else measured. It is reported as an
   oscillation rather than as "it reached 2", which is how an earlier report read the same pattern.
   Whether the detector is bistable on this geometry or the wall genuinely flickers is **unmeasured**.
3. **One seed, seed 19, in every campaign in the project.** A single trajectory at 483 268 particles is
   not a sample; the wrapping verdict survives that criticism because it is a boolean that is unanimous
   at 23 of 23, but "cavity slope +0.586" does not.
4. **The plateau arithmetic uses an unweighted least-squares slope on 13 points** whose sampling
   interval changed mid-run (5000 → 4700). The change is 6 % and the conclusion (the slope did not
   fall) is not close, but the number is not a fitted physical rate.
5. **The `supply-window` pin and big-box's re-derived L\* do not use the same ceiling basis.** The pin
   prints `L_max = 81.27σ` and `overCeiling = 7.3x` (the *old* 2500-capacity ceiling); big-box's
   re-derivation gives 3.45× against the measured ceiling of 953 589. Both are in the document, with
   the basis named. The pin was not touched, because touching it to agree with a report is how a pin
   stops being a pin — but a reader comparing the two numbers will be briefly confused, and that is on
   this task.
6. **`tests/soup-grid-resize.test.ts` still fails and was still not chased.** This task's attribution is
   airtight (no file it depends on changed), but the defect is real and it is a dynamics blow-up in
   `applyBoxScale`, i.e. in the same code path the evaporation cycle uses. Nobody has measured whether
   it can bite a campaign rather than only that test.
7. **The verdict document is now 650 lines and is the only thing a reader will read.** Its numbers are
   all traceable to artifacts, but it is a secondary source: if it and a report disagree, the report
   and the artifact win, and the document says so only implicitly by citing them.
8. **`verify/viewer-smoke.ts` is committed but is not a test** — nothing will fail if the page stops
   drawing; only a human running it will notice. Folding its two assertions into
   `tests/run-ui.test.ts` would fix that and was not done, because the ocean-look test already covers
   the same ground and a second copy of a slow browser test costs a chunk on every future run.
