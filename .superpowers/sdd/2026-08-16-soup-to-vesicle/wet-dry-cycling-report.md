# Wet–dry cycling at real liquid water — report

Task `wet-dry-cycling` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `loud-failure-and-liquid-water-report.md` (liquid water is now reachable, failure is now
loud, the zero-RNG checkpoint defect is fixed). `data/params.json` was **NOT** touched
(`git diff --stat data/params.json` empty). No threshold, corridor, potential, rate, recogniser or
rank-A constant was modified; `co_bond.attemptRate` untouched; no arm was quietly reverted to
under-dense water. One `data/soup.json` section was **re-derived** (`dryWetCycle`: four numbers, with
the derivation appended to its own `basis`) because the old amplitude makes cycling **structurally
impossible** at liquid water — the engine throws (§4.1). One test fixture gained one line
(`tests/runner.html`, an empty favicon — §8.2); no assertion, tolerance or corridor changed anywhere.

**Verdict up front, seven lines.**

1. **The honest liquid-water baseline reproduces the predecessor's liquid arm and is worse than every
   number published before it.** Box 30, water 0.8 σ⁻³, ρ_tot 1.21793, single-shot, n = 3, 20 000
   steps: yield **0.08135** [0.0785, 0.0844], `cc_bond` **296.3** [269, 323], `co_bond` **261.0**
   [234, 286], mean per-tail **2.101** [2.098, 2.105], largest aggregate **10.7** [8, 13], stage
   `amphiphiles` ×3. Run to saturation (204 000 steps, campaign, audited off-GPU) the amphiphile
   count settles at **160.3** [157, 165] and the largest aggregate at **12.7** [11, 15] — and the run
   **never leaves stage `amphiphiles` in 204 000 steps**. That last part *is* worse than every
   published number in this project's reports: every earlier campaign reached `micelles` or `bilayer`.
2. **The old zero-RNG systematic was a factor of THREE on the chemistry, and I measured it.** Resuming
   with the historical all-zero RNG bytes against resuming with real RNG state, identical seeds,
   n = 3, 20 000 steps: `cc_bond` **96.0 vs 288.3 (3.00× fewer)**, `co_bond` **81.3 vs 255.0 (3.14×)**,
   amphiphiles **66.7 vs 146.0 (2.19×)**, largest aggregate **5.0 vs 9.3 (1.87×)**. The mechanism is
   named and measured: under zeroed RNG the mean-square displacement over 2000 steps is
   **62.32 σ² and IDENTICAL for every species to three significant figures** — a rigid whole-system
   translation with essentially zero *relative* diffusion — against 0.40–0.98 σ² and
   species-dependent with real RNG. It is **permanent, not transient** (`soup/wgsl/step.wgsl`'s RNG
   advance has no index mixing, so identical states stay identical forever). Every historical resumed
   campaign therefore ran with its reaction rate suppressed ~3× **from its first resume onward** —
   which means the published resumed numbers are *under*-statements, not over-statements.
3. **Resume fidelity is now good.** Real-RNG resumed vs single-shot, n = 3: `cc_bond` 288.3 vs 296.3
   (−2.7 %), amphiphiles 146.0 vs 151.3 (−3.5 %), largest 9.3 vs 10.7 — all inside the seed-to-seed
   scatter (`cc_bond` spans 269–323 across seeds alone).
4. **The literature's cycling amplitude is 1400× in volume. This model can express 1.10×.** Ross &
   Deamer 2016 measured evaporation reducing volume ~1400-fold. Box scaling at fixed composition
   compresses the water with the organics, and close packing of unit-diameter cores sits at
   1.4142 σ⁻³, so the whole available run from ρ_tot = 1.21793 is to <1.4142. Chosen amplitude
   **1.34 σ⁻³** = the highest density measured still mobile, with 1.054× margin to close packing:
   concentration factor **1.1002×**, i.e. **ln(1.1002)/ln(1400) = 1.3 %** of the literature's
   evaporation on a log-concentration scale. Stated as a model limitation, not a parameter.
5. **Cycling does not recover the chemistry; it suppresses it, and the mechanism is kinetic arrest.**
   Free-carbon mobility falls **6.2×** from ρ = 1.21793 to ρ = 1.34120 (D = 4.94e-3 → 7.9e-4 σ²/τ),
   measured; in the live cycle, against a matched wet control over the same step window, the collapse
   is **3.27×** and the bond-formation rate goes **0.01100 → 0.00850 per step**. Across six cycles
   against a matched control at the same `globalStep`, cycled/control bond count is
   **0.711, 0.810, 0.863, 0.902, 0.917, 0.926** — the deficit shrinks with cycle number and **never
   closes**. Largest aggregate: cycled 11.2 [8, 15] vs control 12.7 [11, 15]. **No cycle count in
   1…6 makes cycling match the wet control, let alone beat it.**
6. **The window closed to a 1.06 σ slit and the window run reached stage `bilayer` but not vesicle.**
   Recomputed floor **968** amphiphiles (from gates measured in this task); measured saturated
   ρ_amph = 5.937e-3 σ⁻³ → supply needs L ≥ **54.63 σ**, measurability caps L ≤ **55.69 σ**, window
   width **1.06 σ**. L = **55** chosen, N = **202 636**. 127 000 steps, 7 checkpoints, 0 non-finite:
   supply **1047** (1.08× over the floor — the thinnest margin this project has ever run at), largest
   aggregate **20–31, plateau over 107 000 steps**, `radialHeadShells = 2` at 5 of 7 checkpoints,
   encapsulated water **0** of a 296.8 threshold at every checkpoint, `closed = false` everywhere.
7. **Binding constraint, numbered: (1) AGGREGATE SIZE — 29 amphiphiles in one aggregate against a
   floor of 968 = 33.4× short.** It was 8.89×, then 3.36×, now 33.4×: real liquid water made the
   binding constraint an order of magnitude worse, and cycling as this engine can express it does not
   touch it. A well-evidenced negative.

---

## 1. The honest liquid-water baseline

### 1.1 Design

Box 30, organics C 1860 / O 7440 / H 1860 / M 124 (ρ_org = 0.41793), water **21 600 = 0.8 σ⁻³**,
N = 32 884, ρ_tot = **1.21793** — the predecessor's own defended liquid composition, unchanged.
`--relax` in every arm (the cold-start minimisation; without it this composition throws, §8.1).
Seeds 19 / 23 / 29. Single-shot (no resume), so no RNG systematic of any kind can enter.
`yield = amphiphiles / C`, the project's own definition. `clay: false`.

### 1.2 Every run, unedited

```
$ nice -n 15 npx tsx scratch-wd.ts '[{"label":"BASE-liquid-s19","box":30,"start":{"C":1860,"O":7440,"H":1860,"M":124,"W":21600},"seed":19,"steps":40000,"relax":true,"sampleEvery":20000},{"label":"BASE-liquid-s23",...,"seed":23,...}]'
ARM BASE-liquid-s19 box=30 N=32884 rhoWet=1.21793 seed=19 steps=40000 relax=true cycle=false resumeEvery=0 zeroRng=false wall=83439ms
    relax maxF 3.3966e+4 -> 1.7852e+1 nonFinite 0/0
    {"steps":20000,"box":30,"rhoTot":1.21793,"phase":"none","cycleIndex":0,"events":{"cc_bond":269,"cc_break":0,"co_bond":234,"co_break":1},"amph":146,"meanTail":2.1013,"chains":1591,"meanChain":1.1691,"stage":"amphiphiles","aggs":55,"largest":11,"radialHeadShells":"unavailable","flatness":0.3037,"inPlane":0.5737,"cavity":0,"encapsulated":0,"nonFinite":{"pos":0,"vel":0}}
    {"steps":40000,...,"events":{"cc_bond":382,"cc_break":0,"co_bond":336,"co_break":6},"amph":165,"meanTail":2.1283,"chains":1478,"meanChain":1.2585,"stage":"amphiphiles","aggs":52,"largest":14,...,"nonFinite":{"pos":0,"vel":0}}
    invariants {"monomers":{"C":1860,"O":7440,"H":1860,"M":124,"W":21600,"K":0},"bonds":712,"charge":0}
ARM BASE-liquid-s23 box=30 N=32884 rhoWet=1.21793 seed=23 steps=40000 relax=true ... wall=154174ms
    relax maxF 5.2759e+4 -> 1.7412e+1 nonFinite 0/0
    {"steps":20000,...,"events":{"cc_bond":297,"cc_break":0,"co_bond":263,"co_break":1},"amph":151,"meanTail":2.0976,"chains":1563,"meanChain":1.19,"stage":"amphiphiles","aggs":58,"largest":8,...}
    {"steps":40000,...,"events":{"cc_bond":411,"cc_break":0,"co_bond":362,"co_break":3},"amph":166,"meanTail":2.1492,"chains":1449,"meanChain":1.2836,"stage":"amphiphiles","aggs":55,"largest":11,...}
    invariants {"monomers":{"C":1860,"O":7440,"H":1860,"M":124,"W":21600,"K":0},"bonds":770,"charge":0}
ARM BASE-liquid-s29 box=30 N=32884 rhoWet=1.21793 seed=29 steps=40000 relax=true ... wall=109148ms
    relax maxF 3.2154e+4 -> 2.1110e+1 nonFinite 0/0
    {"steps":20000,...,"events":{"cc_bond":323,"cc_break":0,"co_bond":286,"co_break":1},"amph":157,"meanTail":2.1047,"chains":1537,"meanChain":1.2101,"stage":"amphiphiles","aggs":53,"largest":13,...}
    {"steps":40000,...,"events":{"cc_bond":445,"cc_break":0,"co_bond":393,"co_break":3},"amph":173,"meanTail":2.1809,"chains":1415,"meanChain":1.3145,"stage":"amphiphiles","aggs":51,"largest":10,...}
    invariants {"monomers":{"C":1860,"O":7440,"H":1860,"M":124,"W":21600,"K":0},"bonds":835,"charge":0}
```

### 1.3 The baseline, with measured scatter, against what is already published

| quantity, 20 000 steps | **this task, n = 3, single-shot** | predecessor's liquid arm (n = 3) | predecessor's TOKEN-water arm (n = 3) |
|---|---|---|---|
| `cc_bond` | **296.3** [269, 323] | 302.3 [274, 320] | 650.0 [638, 661] |
| `co_bond` | **261.0** [234, 286] | 264.7 [241, 280] | 438.7 [437, 441] |
| amphiphiles | **151.3** [146, 157] | 148.3 [143, 152] | 273.0 [271, 275] |
| yield | **0.08135** [0.0785, 0.0844] | 0.07975 [0.0769, 0.0817] | 0.14677 [0.1457, 0.1479] |
| mean per-tail | **2.101** [2.098, 2.105] | 2.099 [2.073, 2.133] | 2.327 [2.288, 2.359] |
| largest aggregate | **10.7** [8, 13] | 14.3 [9, 20] | 91.0 [69, 129] |
| stage | `amphiphiles` ×3 | `amphiphiles` ×3 | `bilayer` ×2, `micelles` ×1 |

**The liquid-water result is reproduced, not softened:** every quantity sits inside the predecessor's
own range (yield +2.0 %, `cc_bond` −2.0 %, largest-aggregate ranges 8–13 vs 9–20 overlap). So the
honest baseline is **not** worse than *every* number ever published here — it is statistically
indistinguishable from the one liquid-water measurement that exists, and **1.80× worse in yield and
8.5× worse in largest aggregate than every number published before liquid water was reachable.**

### 1.4 The saturated baseline, and the one thing that IS worse than everything published

The 20 000-step number is not saturated. Run to 204 000 steps through the resumable campaign CLI and
audited off-GPU (`tests/continuous-run-audit.test.ts`, unmodified; artifact
`verify/out/wetdry-control-19-trace.json`):

```
$ nice -n 15 npx tsx soup/cli/campaign.ts --label wdctl19 --box 30 --start '{"C":1860,"O":7440,"H":1860,"M":124,"W":21600}' --seed 19 --kT 1.1 --steps 102000 --every 34000 --dir data/checkpoints/wdctl19 --relax
[campaign] минимизация холодного старта: итераций=200 шаг_первой=0.1 граница_суммарного_смещения=10.0500 max|F| 3.3966e+4 -> 1.7054e+1 нефинитных_до=0 нефинитных_после=0 шаг_системы=0
[campaign] шаг=34000/102000 stage=amphiphiles агрегатов=54 крупнейший=15 headShells=unavailable cavityVolume=0.000 stepMs=100663 ...
[campaign] шаг=68000/102000 stage=amphiphiles агрегатов=50 крупнейший=13 headShells=unavailable cavityVolume=0.250 stepMs=114239 ...
[campaign] шаг=102000/102000 stage=amphiphiles агрегатов=51 крупнейший=13 headShells=unavailable cavityVolume=0.000 stepMs=122503 ...
[campaign] резюме label=wdctl19 из data/checkpoints/wdctl19/wdctl19-step102000.json, шаг=102000
[campaign] шаг=136000/204000 stage=amphiphiles агрегатов=45 крупнейший=12 ... stepMs=117052 ...
[campaign] шаг=170000/204000 stage=amphiphiles агрегатов=46 крупнейший=11 ... stepMs=121317 ...
[campaign] шаг=204000/204000 stage=amphiphiles агрегатов=45 крупнейший=12 ... stepMs=121261 ...
[campaign] бюджет шагов выполнен полностью: шаг=204000
```

| step | stage | amph | yield | perTail | cc | co | co_brk | aggs | qualifying | largest | cavity | encH₂O/thr | nonFinite |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 34 000 | amphiphiles | 160 | 0.0860 | 2.133 | 380 | 327 | 4 | 54 | 1 | 15 | 0 | 0/297.20 | 0/0 |
| 68 000 | amphiphiles | 160 | 0.0860 | 2.166 | 451 | 389 | 12 | 50 | 2 | 13 | 0.25 | 0/297.14 | 0/0 |
| 102 000 | amphiphiles | 157 | 0.0844 | 2.168 | 473 | 407 | 16 | 51 | 1 | 13 | 0 | 0/297.14 | 0/0 |
| 136 000 | amphiphiles | 160 | 0.0860 | 2.191 | 495 | 424 | 20 | 45 | 0 | 12 | 0 | 0/297.11 | 0/0 |
| 170 000 | amphiphiles | 165 | 0.0887 | 2.207 | 518 | 443 | 23 | 46 | 0 | 11 | 0 | 0/297.10 | 0/0 |
| **204 000** | **amphiphiles** | **160** | **0.0860** | **2.216** | **535** | **454** | **28** | **45** | **0** | **12** | **0** | **0/297.12** | **0/0** |

Seed 23's control (`verify/out/wetdry-control-23-trace.json`) gives amph 156 / 171 / 175 and largest
12 / 12 / 18 at 34 000 / 68 000 / 102 000 — same picture, slightly higher supply.

- **Amphiphile supply saturates by step 34 000** and stays flat to 204 000: 160.3 ± 2.4 over six
  samples, yield **0.0862**. ρ_amph = 160.3/27 000 = **5.937e-3 σ⁻³**. This is the number §6 needs.
- **The largest aggregate never grows**: 15 → 12, and the *trend across 170 000 steps is downward*.
- **In those words: this is worse than every published number in this project's reports.** The stage
  ladder stalls at `amphiphiles` for 204 000 steps, `hasLamellarAggregate` and `hasVesicleAggregate`
  are `false` at every checkpoint, and the count of aggregates even *qualifying* (≥13 amphiphiles)
  falls to **0** from step 136 000 on. Every earlier campaign in this project reached at least
  `micelles`; the box-46 run made a 682-molecule cup and the box-54 run a 271-molecule bilayer patch.
  At real liquid water, box 30, this engine makes 45 aggregates of a dozen molecules each and stops.

---

## 2. The zero-RNG systematic, measured

### 2.1 How it was measured without touching production code

The historical defect (predecessor §1.5) was that `bondRngBuf`/`thermoRngBuf` lacked `COPY_SRC`, so
`readBack` returned a zero-filled staging buffer and **every checkpoint on disk carried all-zero RNG
state**. It is fixed. To measure what it *cost*, the defect is re-created in the data rather than in
the code: a checkpoint is taken, its `bondRngB64`/`thermoRngB64` are overwritten with the same number
of zero bytes, and the run resumes from that — bit-for-bit the historical situation, with the fixed
engine. Three arms per condition, resume every 5000 steps of a 20 000-step run, seeds 19/23/29.

```
$ nice -n 15 npx tsx scratch-wd.ts '[{"label":"RESUME-realRNG-s19",...,"steps":20000,"relax":true,"resumeEvery":5000}, s23, s29]'
ARM RESUME-realRNG-s19 ... resumeEvery=5000 zeroRng=false wall=... 
    {"resumeAt":5000,"bondRngNonzeroBytes":131037}
    {"resumeAt":10000,"bondRngNonzeroBytes":131034}
    {"resumeAt":15000,"bondRngNonzeroBytes":131035}
    {"steps":20000,...,"events":{"cc_bond":292,"cc_break":0,"co_bond":256,"co_break":1},"amph":151,"meanTail":2.1341,"chains":1568,"meanChain":1.1862,"stage":"amphiphiles","aggs":51,"largest":10,...}
ARM RESUME-realRNG-s23 ... {"steps":20000,...,"cc_bond":285,"co_bond":250,...,"amph":140,...,"largest":8,...}
ARM RESUME-realRNG-s29 ... {"steps":20000,...,"cc_bond":288,"co_bond":259,...,"amph":147,...,"largest":10,...}

$ nice -n 15 npx tsx scratch-wd.ts '[{"label":"RESUME-zeroRNG-s19",...,"resumeEvery":5000,"zeroRng":true}, s23, s29]'
ARM RESUME-zeroRNG-s19 ... resumeEvery=5000 zeroRng=true
    {"resumeAt":5000,"bondRngNonzeroBytes":0}
    {"resumeAt":10000,"bondRngNonzeroBytes":0}
    {"resumeAt":15000,"bondRngNonzeroBytes":0}
    {"steps":20000,...,"events":{"cc_bond":90,"cc_break":0,"co_bond":77,"co_break":0},"amph":66,"meanTail":2.1159,"chains":1770,"meanChain":1.0508,"stage":"amphiphiles","aggs":44,"largest":4,...}
ARM RESUME-zeroRNG-s23 ... {"steps":20000,...,"cc_bond":103,"co_bond":82,...,"amph":66,...,"largest":5,...}
ARM RESUME-zeroRNG-s29 ... {"steps":20000,...,"cc_bond":95,"co_bond":85,...,"amph":68,...,"largest":6,...}
```

`bondRngNonzeroBytes` is 131 029–131 040 of N·4 = 131 536 (99.6 %) with the fix and exactly **0**
when the defect is re-created — the two conditions are verified in the data, not assumed.

### 2.2 The size of the systematic

| quantity, 20 000 steps | single-shot (n=3) | resumed, REAL RNG (n=3) | resumed, ZEROED RNG (n=3) | **systematic (zero/real)** |
|---|---|---|---|---|
| `cc_bond` | 296.3 [269, 323] | **288.3** [285, 292] | **96.0** [90, 103] | **3.00× fewer** |
| `co_bond` | 261.0 [234, 286] | **255.0** [250, 259] | **81.3** [77, 85] | **3.14× fewer** |
| amphiphiles | 151.3 [146, 157] | **146.0** [140, 151] | **66.7** [66, 68] | **2.19× fewer** |
| yield | 0.08135 | **0.07849** | **0.03584** | **2.19×** |
| mean chain | 1.188 | **1.184** | **1.054** | — |
| largest aggregate | 10.7 [8, 13] | **9.3** [8, 10] | **5.0** [4, 6] | **1.87× smaller** |

- **Resume fidelity now: fine.** Real-RNG resumed vs single-shot differs by 2.7 % (`cc_bond`), 3.5 %
  (amphiphiles) — far inside the seed-to-seed scatter of the single-shot arm itself (269–323).
- **The old systematic: a factor of three on bond formation.** That is 1.5 orders of magnitude larger
  than any scatter this project reports.

### 2.3 The mechanism, and why the damage is permanent

```
ARM MSD-zeroRNG ... resumeEvery=5000 zeroRng=true
    {"resumeAt":5000,"bondRngNonzeroBytes":0}
    MSD {"steps":2000,"tau":20,"perSpecies":{"C":62.324,"C_bonded":62.2931,"O":62.3521,"O_bonded":62.3144,"H":62.3356,"M":62.3365,"W":62.3572}}
ARM MSD-realRNG-resumed ... resumeEvery=5000 zeroRng=false
    {"resumeAt":5000,"bondRngNonzeroBytes":131034}
    MSD {"steps":2000,"tau":20,"perSpecies":{"C":0.6423,"C_bonded":0.5871,"O":0.4822,"O_bonded":0.5761,"H":0.9776,"M":0.3986,"W":0.4529}}
```

Under zeroed RNG the mean-square displacement is **62.32 σ² and identical across all seven
species/bonding classes to three significant figures** (rms 7.89 σ over 2000 steps) — a *rigid
translation of the entire system*, with relative diffusion destroyed. With real RNG it is
0.40–0.98 σ² and species-dependent, as thermal noise must be. That is what a common-mode Langevin
kick looks like, and it is exactly why encounters — and therefore bonds — collapse 3×.

And it does not heal: `soup/wgsl/step.wgsl` advances the stream as
`var s = intRng[i]; s = pcgSoup(s); … intRng[i] = s` — **no particle index enters the update**, so
per-particle states that start equal stay equal for the rest of the run. The predecessor's window
campaign resumed 8 times, the first at step 45 000 of 315 000: **86 % of that trajectory ran under
correlated noise.** Its published numbers are lower bounds on what the fixed engine produces.

---

## 3. The literature, what it sets, and what this model can and cannot represent

### 3.1 Citations used, with ranks

| # | source | rank | what it supplies here |
|---|---|---|---|
| L1 | **Ross & Deamer, *Life* 2016, 6(3):28**, doi:10.3390/life6030028 | **A** (published measurement/derivation) | Condensation is **endoergic by ~3.3 kcal/mol**; molecular crowding / excluded volume in evaporites supplies **5–10 kcal/mol** of oligomer stabilisation; evaporation effects sum to **10–15 kcal/mol**. Their own evaporation **reduced volume by a factor ~1400** (to ~9 M monomer); evaporation 30 min, rehydration interval 20 h. → sets the **amplitude** the model is judged against. |
| L2 | **Higgs, *Life* 2016, 6(2):24**, doi:10.3390/life6020024 (PMC4931461) | **B** (kinetic model, same observable as ours) | "These limits are reached after **about six cycles** for the parameters illustrated" → `cycles = 6`. Its own schedule `t_dry = 8, t_wet = 0.5` (dry fraction 0.94). Mechanism: dry phase = bonding favoured but diffusion restricted; wet phase = *"they permit rearrangement of molecules and bring molecules together that may polymerize in the next dry phase"*, plus wet-phase hydrolysis. → sets **cycle count** and the **function of the wet phase**. |
| L3 | **Rieder et al., *ACS Cent. Sci.* 2025**, doi:10.1021/acscentsci.5c00488 (PMC12464751) | **A** (experiment) | Up to **10 cycles** at Δt = **24 h**; samples reach the dry state in **8–10 h**; 23 °C; c₀ = 50 mM; yield grows with cycle number and saturates. → sets **dry fraction ≈ 0.6–0.67**. |
| L4 | **Damer & Deamer, *Astrobiology* 2020, 20(4):429**, doi:10.1089/ast.2019.2045 | **C** (hypothesis/synthesis) | The dry phase is a **multilamellar matrix**; on rehydration water returns to the head groups, lamellae **swell and the outer layers bud off as vesicles**; the intermediate moist phase is a hydrogel "progenote". → names the structural mechanism this model would have to reproduce to close a vesicle by cycling. |
| L5 | **Rajamani et al., *Orig. Life Evol. Biosph.* 2008, 38:57**, doi:10.1007/s11084-007-9113-2 | **A** (experiment) | Phosphodiester synthesis driven by "the chemical potential of fluctuating anhydrous and hydrated conditions", lipid-assisted. → the original demonstration that the *alternation itself*, not either state, is the driver. |

### 3.2 What the model CAN represent

- **Concentration by volume contraction at fixed composition** — the excluded-volume/encounter half of
  L1's crowding argument, with a measured amplitude and an exact coordinate map (rigid centre-of-mass
  scaling per covalent molecule, with a runtime self-check on every bonded pair).
- **The alternation itself** (L5): a defined wet/dry schedule with ramped, reversible box changes; the
  box returns to the wet value **exactly** (pinned bit-for-bit by the new test).
- **L2's wet-phase repositioning**: the wet segment is sized so a free monomer's rms displacement is
  **1.82 σ**, i.e. more than one particle diameter — measured, not asserted.
- **The water-activity effect on condensation kinetics — but only as a static composition variable.**
  The predecessor's token-water (0.26 σ⁻³) vs liquid-water (0.8 σ⁻³) comparison *is* a measurement of
  the dry vs wet chemistry, and it points the way L1/L5 say it should: **1.84× more yield with less
  water**. This model just cannot *alternate* it (below).

### 3.3 What the model CANNOT represent — said plainly

1. **Evaporation.** Particle count is fixed at `createSoup`, so water cannot leave. Box contraction
   concentrates the water along with the organics (water is 65.7 % of all particles here), and close
   packing bounds the compression: **1.10× in volume against L1's ~1400×, i.e. 1.3 % of it on a
   log-concentration scale.** The dry state here is a compressed solution, not a dried film.
2. **The thermodynamic driving force.** `cc_bond`/`co_bond` carry fixed `energyKT` and `attemptRate`
   that do not depend on water activity, and water is **not a product of the bond rule**, so removing
   or adding water cannot shift the condensation equilibrium the way L1's +3.3 vs 5–10 kcal/mol
   argument requires. Only the *kinetic/encounter* channel exists in this engine.
3. **L2's hydrolysis half.** `cc_break.attemptRate = 0` by construction and `co_break` is rare
   (28 events in 204 000 steps against 535 `cc_bond`). The wet phase therefore cannot depolymerise,
   so the *form/break ratchet* that is the whole reason cycling beats a single dry phase in L2 has no
   representation here. What is left is monotonic polymerisation with a periodic density change.
4. **L4's multilamellar matrix and budding.** Nothing in this model orients amphiphiles into layers
   under drying; the dry state is a homogeneously compressed liquid. There is no lamellar stack to
   swell and no outer layer to bud off.
5. **Temperature cycling** (L1/L3's dry phase is hot; kT is held at 1.1 throughout), **pH and salt**
   (`saltPhLimitation`: no electrostatics at all), and **the air–water interface** (a fully periodic
   box has no free surface for evaporation to act on).

**Consequence, stated before the measurement rather than after it:** of the five mechanisms the
literature offers, this engine can express one and a half. That is why the cycling result below is
reported as a measurement of *this model's* box-scaling cycle, not as a test of Deamer's cycle.

---

## 4. Wet–dry cycling: parameters, one cycle's mechanism, and the results

### 4.1 The re-derivation was structurally forced, not cosmetic

`deriveCycleConfig` (`soup/src/soup-box-scale-math.ts`) throws when
`targetDryDensity <= wetDensity`: *"сухая фаза обязана концентрировать, не разбавлять"*. The old
amplitude was **0.6 σ⁻³**, derived for a water-free medium at ρ_tot = 0.4799. At real liquid water
ρ_tot = **1.21793**, so 0.6 is a *dilution* and cycling is impossible with it. This is pinned as the
first assertion of the new test, so the re-derivation cannot be mistaken for taste.

| field | old | **new** | basis |
|---|---|---|---|
| `targetDryDensity` | 0.6 | **1.34** | highest density measured **still mobile** (§4.2), 1.054× below close packing (√2 = 1.4142); concentration factor 1.1002× — reported against L1's 1400× |
| `cycles` | 5 | **6** | **L2**: "limits are reached after about six cycles" |
| `dryFraction` | 0.5 | **0.65** | **L3** (8–10 h dry within a 24 h cycle → 0.6–0.67); **L2**'s own model used 0.94, so 0.65 is the *conservative* end of the literature bracket, not a fit |
| `periodSteps` | 50 000 | **32 000** | measured mobility: at ρ = 1.3412, D = 7.9e-4 σ²/τ → t(1 σ) = 1/(6D) = 211 τ = **21 100 steps**; 0.65·32 000 = **20 800**. Wet segment 11 200 steps → rms **1.82 σ** at the wet D, which is L2's "repositioning" |
| `rampSteps` / `rampRelaxSteps` | 6 / 200 | **6 / 200 (kept)** | now with arithmetic: ln(L) step per increment **0.005307** = 0.529 % linear = **0.884×** the area move's own validated `AREA_MOVE_LOG_DELTA = 0.012` (0.6 % linear). Probed by measurement too (§4.4) |
| `enabled` | false | **false (kept)** | default-off discipline preserved: every existing test/gate/viewer path is untouched; cycling is requested per run (`--cycle` / `CreateSoupOpts.dryWetCycle`) |

### 4.2 The mobility ladder that sets the amplitude — unedited

```
$ nice -n 15 npx tsx scratch-wd.ts '[{"label":"MSD-wet-1.218","box":30,...,"steps":3000,"relax":true,"msdSteps":2000},{"label":"MSD-dry-1.28","box":29.5023,...},{"label":"MSD-dry-1.34","box":29.0512,...},{"label":"MSD-dry-1.40","box":28.6314,...}]'
ARM MSD-wet-1.218 box=30 N=32884 rhoWet=1.21793 ... 
    {"steps":3000,...,"events":{"cc_bond":26,"cc_break":0,"co_bond":24,"co_break":0},"amph":20,...,"nonFinite":{"pos":0,"vel":0}}
    MSD {"steps":2000,"tau":20,"perSpecies":{"C":0.5929,"C_bonded":0.5434,"O":0.5724,"O_bonded":0.4479,"H":0.8968,"M":0.4185,"W":0.5124}}
ARM MSD-dry-1.28 box=29.5023 N=32884 rhoWet=1.28061 ...
    {"steps":3000,...,"events":{"cc_bond":25,...,"co_bond":22,...},"amph":20,...}
    MSD {"steps":2000,"tau":20,"perSpecies":{"C":0.1827,"C_bonded":0.2656,"O":0.192,"O_bonded":0.2537,"H":0.3171,"M":0.1396,"W":0.1739}}
ARM MSD-dry-1.34 box=29.0512 N=32884 rhoWet=1.3412 ...
    {"steps":3000,...,"events":{"cc_bond":17,...,"co_bond":16,...},"amph":16,...}
    MSD {"steps":2000,"tau":20,"perSpecies":{"C":0.095,"C_bonded":0.0867,"O":0.1035,"O_bonded":0.0886,"H":0.151,"M":0.0517,"W":0.0927}}
ARM MSD-dry-1.40 box=28.6314 N=32884 rhoWet=1.40106 ...
    {"steps":3000,...,"events":{"cc_bond":17,...,"co_bond":17,...},"amph":17,...}
    MSD {"steps":2000,"tau":20,"perSpecies":{"C":0.072,"C_bonded":0.1043,"O":0.0759,"O_bonded":0.0773,"H":0.1033,"M":0.0551,"W":0.0708}}
```

| ρ_tot σ⁻³ | box σ | free-C MSD, τ = 20 (σ²) | D (σ²/τ) | rms over 2000 steps | `cc_bond` in 3000 steps | margin to close packing |
|---|---|---|---|---|---|---|
| **1.21793** (wet) | 30.0000 | 0.5929 | 4.94e-3 | 0.77 σ | 26 | 1.161× |
| 1.28061 | 29.5023 | 0.1827 | 1.52e-3 | 0.43 σ | 25 | 1.104× |
| **1.34120** (chosen) | 29.0512 | 0.0950 | 7.9e-4 | 0.31 σ | 17 | **1.054×** |
| 1.40106 | 28.6314 | 0.0720 | 6.0e-4 | 0.27 σ | 17 | 1.009× |

**This is the mechanism of the whole result, visible before any cycle was run.** A 10 % density
increase costs **6.2×** in mobility, and the bond count over an identical 3000 steps falls 26 → 17.
In this model the "dry" phase is not a concentrated reactive film — it is a **kinetic arrest**.

### 4.3 One cycle: the density trajectory and what happens

Ramp, six log-linear increments of ln(L) = −0.005307 each, 200 real steps of ordinary dynamics between
increments (5 gaps = **1000 real steps charged to the trajectory**, verified by the new test):

| increment | box σ | ρ_tot σ⁻³ |
|---|---|---|
| 0 (wet) | 30.0000 | 1.21793 |
| 1 | 29.8412 | 1.23747 |
| 2 | 29.6833 | 1.25733 |
| 3 | 29.5262 | 1.27751 |
| 4 | 29.3699 | 1.29801 |
| 5 | 29.2145 | 1.31884 |
| **6 (dry, exact)** | **29.0598** | **1.34000** |

and the mirror ramp back, landing on box 30.000000 **exactly** (`boxAfterRehydration[i] === box[i]`,
asserted, not approximated).

Live single-cycle probe, box 30, seed 19, sampled every 4000 steps (unedited):

```
ARM CYCLE-mech-s19 box=30 N=32884 rhoWet=1.21793 seed=19 steps=32000 relax=true cycle=true wall=88277ms
    {"steps":4000,"box":30,"rhoTot":1.21793,"phase":"wet","cycleIndex":1,"events":{"cc_bond":38,...,"co_bond":37,...},"amph":35,...,"largest":2,...}
    {"steps":8000,"box":30,"rhoTot":1.21793,"phase":"wet","cycleIndex":1,"events":{"cc_bond":95,...,"co_bond":86,...},"amph":66,...,"largest":8,...}
    {"steps":13000,"box":29.0598,"rhoTot":1.34,"phase":"dry","cycleIndex":1,"events":{"cc_bond":179,...,"co_bond":163,...},"amph":97,...,"largest":6,...}
    {"steps":17000,"box":29.0598,"rhoTot":1.34,"phase":"dry","cycleIndex":1,"events":{"cc_bond":207,...,"co_bond":187,...},"amph":100,...,"largest":6,...}
    {"steps":21000,"box":29.0598,"rhoTot":1.34,"phase":"dry","cycleIndex":1,"events":{"cc_bond":226,...,"co_bond":203,...},"amph":103,...,"largest":6,...}
    {"steps":25000,"box":29.0598,"rhoTot":1.34,"phase":"dry","cycleIndex":1,"events":{"cc_bond":234,...,"co_bond":210,...},"amph":104,...,"largest":6,...}
    {"steps":29000,"box":29.0598,"rhoTot":1.34,"phase":"dry","cycleIndex":1,"events":{"cc_bond":241,...,"co_bond":217,"co_break":1},"amph":108,...,"largest":6,...}
    {"steps":34000,"box":30,"rhoTot":1.21793,"phase":"wet","cycleIndex":2,"events":{"cc_bond":250,...,"co_bond":224,"co_break":2},"amph":113,...,"largest":6,...}
    invariants {"monomers":{"C":1860,"O":7440,"H":1860,"M":124,"W":21600,"K":0},"bonds":472,"charge":0}
```

- **Encounter frequency goes DOWN, not up.** `cc_bond` per step: 9.5e-3 (0–4k, wet) → 14.3e-3
  (4k–8k, wet) → **7.0e-3** (13k–17k, dry) → 4.75e-3 → 2.0e-3 → **1.75e-3** (25k–29k, dry): an **8×
  decay across the dry hold**. Some of that is monomer depletion; the matched-control measurement in
  §4.5 separates the two and the dry phase is still the slower one (0.00850 vs 0.01100 per step).
- **What survives rehydration: everything and nothing.** Every covalent bond survives exactly — the
  rigid-CoM map plus `applyBoxScaleOnce`'s runtime bonded-distance self-check (tolerance 1e-3 σ, on
  every bonded pair, at every one of the 12 increments per cycle) never threw in any run of this task.
  And there is nothing else to ask about: `cavityVolume` ≤ 0.125 σ³ at every sample against the
  370.8656 σ³ closure minimum, `encapsulated` = 0 at every sample, `radialHeadShells` never even
  measurable (too few heads per bin).
- **The loud-failure check stayed on through the densest instant.** `applyBoxScaleOnce` calls
  `assertStateFinite` **and then** `assertVerletSafety` after *every* increment (i.e. at ρ = 1.34
  immediately after the compression that created it), and `nonFinite = {pos:0, vel:0}` at every
  sample of every cycled run in this task, including the new test's explicit `afterRamp` probe.

### 4.4 The too-fast ramp: probed, not assumed

`data/soup.json` was temporarily patched to `rampSteps = 1, rampRelaxSteps = 0` — one instantaneous
jump, no relaxation — and the same cycle re-run at two scales:

```
ARM RAMP-1jump-box20 box=20 N=9743 rhoWet=1.21788 seed=19 steps=32000 relax=true cycle=true wall=38665ms
    {"steps":16000,"box":19.373,"rhoTot":1.34,"phase":"dry",...,"amph":33,...,"nonFinite":{"pos":0,"vel":0}}
    {"steps":32000,"box":20,"rhoTot":1.21788,"phase":"wet","cycleIndex":2,...,"amph":35,...,"cavity":0.125,"nonFinite":{"pos":0,"vel":0}}
ARM RAMP-1jump-box30 box=30 N=32884 rhoWet=1.21793 seed=19 steps=13000 relax=true cycle=true wall=52369ms
    {"steps":13000,"box":29.0598,"rhoTot":1.34,"phase":"dry",...,"amph":106,...,"nonFinite":{"pos":0,"vel":0}}
```

**Honest result: at this amplitude a single jump also survives.** No non-finite state, no Verlet
overflow, no bonded-distance violation, at either scale. The ramp is therefore **not load-bearing at
a 3.2 % linear contraction** — which is a statement about how small the amplitude is (§3.3.1), not a
licence to remove the ramp. `rampSteps = 6` is kept because the guard argument (§4.1) holds
independently of whether it fires today and because the amplitude would grow if the model ever gained
a way to express real drying. The file was restored to 6 / 200 immediately afterwards.

### 4.5 Cycling against a matched control, across the cycle-count trend

Six cycles, box 30, seed 19 (`CYCLE6-s19`), against the §1.4 control at the **same `globalStep`**
(cycling charges its ramp-relax steps to the trajectory, so both arms did 204 000 real steps):

| cycle | globalStep | cycled amph | cycled largest | cycled cc+co | control amph | control largest | control cc+co | **cycled/control bonds** |
|---|---|---|---|---|---|---|---|---|
| 1 | 34 000 | 124 | 8 | 503 | 160 | 15 | 707 | **0.711** |
| 2 | 68 000 | 144 | 12 | 680 | 160 | 13 | 840 | **0.810** |
| 3 | 102 000 | 146 | 14 | 759 | 157 | 13 | 880 | **0.863** |
| 4 | 136 000 | 143 | 9 | 829 | 160 | 12 | 919 | **0.902** |
| 5 | 170 000 | 149 | 9 | 881 | 165 | 11 | 961 | **0.917** |
| 6 | 204 000 | 153 | 15 | 916 | 160 | 12 | 989 | **0.926** |

Second seed (23), three cycles, against its own control:

| cycle | globalStep | cycled amph | cycled largest | cycled cc+co | control amph | control largest | control cc+co | ratio |
|---|---|---|---|---|---|---|---|---|
| 1 | 34 000 | 129 | 12 | 486 | 156 | 12 | 738 | 0.659 |
| 2 | 68 000 | 131 | 12 | 663 | 171 | 12 | 899 | 0.738 |
| 3 | 102 000 | 143 | 13 | 750 | 175 | 18 | 977 | 0.768 |

Full unedited traces are in the two `ARM CYCLE…` blocks (`CYCLE6-s19` wall 758 555 ms;
`CYCLE3-s23` wall 408 239 ms), the two campaign progress blocks, and
`verify/out/wetdry-control-{19,23}-trace.json`.

**Reading it, with the cycle-count trend as the answer rather than a single point:**

- **Cycling never reaches parity.** The bond deficit narrows monotonically (0.711 → 0.926 over six
  cycles at seed 19; 0.659 → 0.768 over three at seed 23) but the cycled arm is behind the wet control
  at **every** cycle, in **both** seeds. The narrowing is depletion, not a ratchet: the control
  saturates first, so the gap closes from the control's side.
- **The amphiphile count is suppressed too**, 124–153 (cycled) against a flat 156–175 (control).
- **Aggregate size is unchanged within scatter**: cycled 11.2 [8, 15] vs control 12.7 [11, 15] at
  seed 19; 12.3 [12, 13] vs 14.0 [12, 18] at seed 23. Cross-seed scatter at matched cycle number is
  ±2–4 molecules, i.e. the same size as the cycled/control difference. **No trend with cycle count.**
- **Number of cycles needed: none of 1…6 is enough**, and the trend does not extrapolate to a win —
  it extrapolates to the control's own saturation value from below.
- **Why, mechanistically:** §4.2. The dry phase costs 6.2× in mobility to buy 1.10× in concentration,
  and with `cc_break = 0` there is no hydrolysis for the wet phase to reverse (§3.3.3), so the model
  has no ratchet — only a periodic mobility tax.

The new test measures the same thing in a form that will fail if the sign ever flips (§7):

```
DRYWET-CONFIG rhoWet=1.21793 targetDryDensity=1.34 dryBox=29.0598 realisedDryDensity=1.34000 concentrationFactor=1.1002 cycles=6 periodSteps=32000 dryFraction=0.65 wetSegment=11200 drySegment=20800
DRYWET-CYCLE N=9743 wetBox=20.0000 dryBox=19.3730 rhoDry=1.34000 rampStepsCharged=1000 boxAfterRehydration=20.000000 phaseEnd=wet
DRYWET-MOBILITY msd(wet,cycled)=0.5762 msd(dry,cycled)=0.1376 msd(same window, wet control)=0.4501 collapse=3.27x
DRYWET-CHEMISTRY rate wet=0.00950/step dry=0.00850/step control(same window)=0.01100/step oneCycleTotal cycled=140 control=196 ratio=0.7143 steps={"cycled":34000,"control":34000} nonFinite={"wet":{"pos":0,"vel":0},"afterRamp":{"pos":0,"vel":0},"dry":{"pos":0,"vel":0},"end":{"pos":0,"vel":0}}
```

**Cycling is therefore NOT used in the window run** (§6) — step 2 did not show it helps.

---

## 5. The recomputed window arithmetic and aggregate-size floor

### 5.1 The floor, from gates measured in this task

Same formula as `continuous-run-report.md` §1.1; no new constant; the closure threshold
(`stageThresholds.enclosedVolume = 370.8656 σ³`) untouched.

```
R_in  = (3 x 370.8656 / 4pi)^(1/3)  = 4.4570 sigma
R_mid = R_in + t/2
N     = 2 x 4pi x R_mid^2 / a
```

Gate readings **from this task's own regression run** (§8.1, unedited there):

| basis for t, a | t σ | a σ² | R_mid σ | **floor** |
|---|---|---|---|---|
| **explicit-water gate, measured here** | **4.4645** | **1.1621** | **6.68925** | **967.7 → 968** |
| solvent-free gate 6, measured here | 4.4943 | 1.2054 | 6.70415 | 937.1 → 937 |
| the predecessor's basis | 4.3338 | 1.2095 | 6.6239 | 912 |
| the 960 basis before that | 4.3441 | 1.1510 | 6.6291 | 960 |

**The floor moves 912 → 968, i.e. UP by 56 amphiphiles (+6.1 %)**, and the reason is again the area
per lipid, not the thickness: t barely moved (4.3338 → 4.4645, +3.0 %) while **a fell 1.2095 → 1.1621
(−3.9 %)** — a smaller head footprint needs *more* lipids for the same shell area. Band **937–968**;
**968 used**, the explicit-water gate, the same one the original derivation chose. Trend across the
three tasks that have computed it: **960 → 912 → 968**.

### 5.2 The window

Measured saturated amphiphile density at liquid water (§1.4, box 30, RNG fix, 34 000–204 000 steps,
n = 6 samples): ρ_amph = 160.3 / 27 000 = **5.937e-3 σ⁻³** (the predecessor's token-water figure was
1.1444e-2, i.e. **1.93× higher**).

- **supply**: ρ_amph·L³ ≥ 968 → L ≥ (968 / 5.937e-3)^(1/3) = **54.63 σ** (with the 937 end of the
  band: 54.04 σ)
- **measurability**: R(L) = 0.067·L^1.5 ≤ L/2 (the box-58 head-radius calibration, unchanged) →
  L ≤ **55.69 σ**

**Window L ∈ [54.63, 55.69] σ — width 1.06 σ.** The supply end moved **11.6 σ outward** from the
predecessor's 43.03 σ, purely because liquid water halved the yield; the window has gone from 12.7 σ
wide to 1.06 σ wide and is now **on the verge of closing altogether**. Had the floor come out at the
gate-6 end (937) the window would be [54.04, 55.69]; had ρ_amph come out 8 % lower, the window would
be **empty**.

**L = 55 σ chosen.** `R(55) = 27.329 σ` against `L/2 = 27.5`, ratio 0.994.

Composition, derived from the box-30 densities, not typed:

```
V = 55^3 = 166 375
C = 0.0688889 x V = 11 462     O = 4C = 45 848     H = C = 11 462
M = (124/27000) x V = 764      W = 0.8 x V = 133 100
N = 202 636                    rho_org = 0.41807      rho_tot = 1.21795
predicted supply = 0.0862 x 11 462 = 988 >= 968   (1.02x over -- the thinnest margin ever run here)
Verlet: 202 636 x 2500 x 4 = 2 026 360 000 B = 2.026 GB  vs 4 294 967 292 B   margin 2.12x
N: 202 636 vs the measured hard ceiling 429 496 particles                     margin 2.12x
```

---

## 6. The window run

### 6.1 The monomers-only start, from the run's own trace

```
$ nice -n 15 npx tsx soup/cli/campaign.ts --label ves55w --box 55 --start '{"C":11462,"O":45848,"H":11462,"M":764,"W":133100}' --seed 19 --kT 1.1 --steps 1 --every 1 --dir data/checkpoints/trace55w --relax
[campaign] новый запуск label=ves55w (совпадающих контрольных точек в data/checkpoints/trace55w нет)
[campaign] система готова N=202636 стартовый_шаг=0 цель=1
[campaign] минимизация холодного старта: итераций=200 шаг_первой=0.1 граница_суммарного_смещения=10.0500 max|F| 2.5088e+4 -> 2.6231e+1 нефинитных_до=0 нефинитных_после=0 шаг_системы=0
[campaign] шаг=1/1 stage=monomers агрегатов=0 крупнейший=0 headShells=n/a cavityVolume=0.000 stepMs=132 checkpointMs=509 progressMs=60 сохранено=data/checkpoints/trace55w/ves55w-step1.json
```

Audited off-GPU from that file: `step=1 stage=monomers amph=0 aggs=0 cc=1 co=0 largest=None
census={'C': 11462, 'O': 45848, 'H': 11462, 'M': 764, 'W': 133100} nonFin=0/0` and
`bondSlotsUsed=2`. **No pre-made amphiphile, no pre-built patch, nothing pre-assembled** — 0
amphiphiles and 0 aggregates from the run's own recorded state.

One honest note: `tests/continuous-run-audit.test.ts`'s start-proof branch (`globalStep <= 1`)
demands **zero bonds**, and at N = 202 636 exactly **one** C–C bond forms inside the very first step,
so that branch fails on a system this large. I did **not** weaken it. The step-1 record above is the
start proof (it shows what the assertion is really for: no pre-existing *structure*); the committed
artifact `verify/out/window-run-55w-trace.json` audits the seven checkpoints of the run lineage
itself, where the branch does not apply, and passes.

### 6.2 Stage trace — the CLI's own progress lines, unedited, 7 resumable chunks

```
[campaign] шаг=20000/20000   stage=bilayer      агрегатов=313 крупнейший=23 headShells=2 cavityVolume=0.000 stepMs=594462 checkpointMs=624  progressMs=1744
[campaign] резюме label=ves55w из data/checkpoints/ves55w/ves55w-step20000.json, шаг=20000
[campaign] --relax пропущен: это резюме с шага=20000, минимизация допустима только на свежем старте
[campaign] шаг=38000/38000   stage=amphiphiles  агрегатов=309 крупнейший=24 headShells=1 cavityVolume=0.250 stepMs=546540 checkpointMs=520  progressMs=1837
[campaign] шаг=56000/56000   stage=bilayer      агрегатов=295 крупнейший=31 headShells=2 cavityVolume=0.000 stepMs=547548 checkpointMs=1005 progressMs=2104
[campaign] шаг=74000/74000   stage=bilayer      агрегатов=306 крупнейший=28 headShells=1 cavityVolume=0.000 stepMs=559743 checkpointMs=687  progressMs=1818
[campaign] шаг=92000/92000   stage=amphiphiles  агрегатов=304 крупнейший=29 headShells=2 cavityVolume=0.125 stepMs=566026 checkpointMs=576  progressMs=1786
[campaign] шаг=110000/110000 stage=amphiphiles  агрегатов=304 крупнейший=20 headShells=2 cavityVolume=0.000 stepMs=582481 checkpointMs=826  progressMs=2131
[campaign] шаг=127000/127000 stage=amphiphiles  агрегатов=302 крупнейший=29 headShells=2 cavityVolume=0.000 stepMs=551517 checkpointMs=568  progressMs=1889
[campaign] бюджет шагов выполнен полностью: шаг=127000
```

**Stage transitions:** `monomers` → `bilayer` **before step 20 000** (faster than any previous run:
the predecessor's first `bilayer` was step 45 000). Thereafter `bilayer` at 3 of 7 samples
(20 000 / 56 000 / 74 000) and `amphiphiles` at 4. **`vesicle`: never. `micelles`: never** — this run
goes straight past it, because it has 300 small aggregates rather than a few large ones.

**The plateau:** the largest aggregate has been **20–31 amphiphiles from step 20 000 to 127 000 —
107 000 steps** — and the aggregate *count* 295–313 over the same span. Amphiphile supply is frozen
to 1047–1063 from step 38 000 (89 000 steps). This is a shorter plateau than the predecessor's
285 000 steps and is reported as such: at N = 202 636 one 590 s foreground chunk buys ~18 000 steps
(29.7 ms/step measured), and the budget bought seven.

### 6.3 Offline CPU audit of all 7 checkpoints

`tests/continuous-run-audit.test.ts`, unmodified, `expect(...)` on every invariant; artifact
`verify/out/window-run-55w-trace.json`.

| step | stage | amph | frac | perTail | cc | co | co_brk | cc_brk | aggs | qual | shQ | largest | Rg | flat | inPl | radSh | transSh | cav | encH₂O/thr | closed | lamellar | vesicle | nonFin |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20 000 | bilayer | 891 | 0.1814 | 2.124 | 1777 | 1542 | 4 | 0 | 313 | 5 | 0.085 | **23** | 5.481 | 0.1684 | 0.5164 | **2** | 1 | 0 | 0/296.81 | false | true | false | 0/0 |
| 38 000 | amphiphiles | 1050 | 0.2224 | 2.157 | 2543 | 2211 | 15 | 0 | 309 | 5 | 0.086 | **24** | 4.676 | 0.1631 | 0.4012 | 1 | 0 | 0.25 | 0/296.82 | false | false | false | 0/0 |
| 56 000 | bilayer | 1053 | 0.2270 | 2.167 | 2832 | 2463 | 29 | 0 | 295 | 7 | 0.112 | **31** | 5.259 | 0.1395 | 0.7154 | **2** | 1 | 0 | 0/296.85 | false | true | false | 0/0 |
| 74 000 | bilayer | 1050 | 0.2283 | 2.161 | 3016 | 2638 | 47 | 0 | 306 | 7 | 0.114 | **28** | 5.042 | 0.1911 | 0.8322 | 1 | 2 | 0 | 0/296.83 | false | true | false | 0/0 |
| 92 000 | amphiphiles | 1063 | 0.2316 | 2.173 | 3148 | 2760 | 66 | 0 | 304 | 9 | 0.143 | **29** | 5.321 | 0.1243 | 0.4504 | **2** | 0 | 0.125 | 0/296.84 | false | false | false | 0/0 |
| 110 000 | amphiphiles | 1047 | 0.2288 | 2.175 | 3253 | 2845 | 81 | 0 | 304 | 8 | 0.127 | **20** | 6.242 | 0.0635 | 0.1849 | **2** | 0 | 0 | 0/296.80 | false | false | false | 0/0 |
| **127 000** | **amphiphiles** | **1047** | **0.2313** | **2.187** | **3328** | **2905** | **104** | **0** | **302** | **6** | **0.106** | **29** | **6.503** | **0.0510** | **0.3224** | **2** | **0** | **0** | **0/296.84** | **false** | **false** | **false** | **0/0** |

- Mean per-tail **2.124 → 2.187**, inside the 2–3 the C12–C18 mapping requires, at every checkpoint.
- `cc_break = 0` at every step (disabled by construction, not revisited). `co_break` reached 104.
- **`radialHeadShells = 2`** — the two-layer signature — at **5 of 7** checkpoints, and
  `hasLamellarAggregate` true at 3. The bilayer *local order* is there. It is there on aggregates of
  **20–31 molecules**.
- **Periodic-aware closure, the only test that matters:** `encapsulatedWater` = **0** against a
  threshold of ~296.8 at **every** checkpoint, `closed = false` at every checkpoint, and
  `cavityVolume` ≤ 0.25 σ³ against the 370.8656 σ³ minimum. Nothing ever closed.

### 6.4 The largest aggregate against every threshold, and the verdict

Final state, step 127 000, seed 19:

| threshold | required | **measured** | verdict | previous campaigns |
|---|---|---|---|---|
| **amphiphiles in largest aggregate** | **968** for closure (13 to qualify) | **29** (plateau 20–31 over 107 000 steps) | **33.4× short — BINDING** | 271 (3.36×), 108 (8.89×) |
| amphiphile SUPPLY | 968 | **1047** | 1.08× over — **not** binding | 1802 (1.98× over), 1600 (1.67×) |
| enclosed volume of the largest aggregate | 370.8656 σ³ | **0** (max 0.25 anywhere) | 1483× short | 6.75 |
| encapsulated water (periodic-aware closure) | ≥ 296.84 | **0** | not closed | 0 |
| `radialHeadShells` | 2 | **2** at 5 of 7 | **met** | 1 |
| flatness ratio | ≤ 0.35 (lamellar) | 0.051–0.191 | met | 0.2713 |
| in-plane symmetry | ≥ 0.50 (lamellar) | 0.185–0.832; ≥0.5 at the 3 `bilayer` samples | met at those samples | 0.6884 |
| mean per-tail | 2–3 | **2.187** | met | 2.437 |
| non-finite state | 0 | **0/0 at all 7 checkpoints** | met | met |
| Verlet / memory | < 429 496 particles, < 4.295 GB | 202 636 / 2.026 GB | 2.12× margin | 2.24× |

**VERDICT: the vesicle is NOT reached, and the binding constraint is again AGGREGATE SIZE — 29
amphiphiles in one aggregate at liquid water against a floor of 968. Shortfall 33.4×.**

Binding constraints, numbered as the last three campaigns did:

| # | constraint | measured | required | shortfall | binding? |
|---|---|---|---|---|---|
| **1** | **aggregate size** | **29** amphiphiles in one aggregate | ~968 | **33.4×** | **YES** |
| 2 | amphiphile supply | **1047**, saturated | 968 | none (1.08× over) | no — but the thinnest margin ever |
| 3 | closure / encapsulated water | 0 | ≥ 296.84 | total | downstream of 1 |
| 4 | medium density | ρ_tot 1.21795, ρ_W 0.800 | liquid | none | **no — settled** |
| 5 | Verlet list / memory | 2.026 GB, 202 636 particles | 4.295 GB, 429 496 | none (2.12×) | no |
| 6 | run length | 127 000 steps, plateau 107 000 | a plateau | shorter than the 285 000 precedent | no (plateau reached) |

**The direction of travel is backwards and I am not softening it.** The shortfall on the binding
constraint went 8.89× → 3.36× → **33.4×**. Real liquid water is the cause: it cuts the yield 1.80×
(which pushes the window's supply end 11.6 σ outward and raises the floor's reach) and — far worse —
it **fragments the material**, 300 aggregates of ~25 molecules instead of one of 271. Wet–dry cycling
as this engine can express it does not touch that (§4.5), and the literature mechanism that would
(L4's lamellar stack and budding) is not representable here at all (§3.3.4).

---

## 7. The new test

**`tests/soup-drywet-cycling.test.ts`** (2 tests, 244 lines), committed. Deliberately split into a
pure and a GPU half, and the GPU half runs a **matched control** because the observable it needs
(bond events per step) decays on its own with monomer depletion — "the dry phase is slower than the
wet phase" is not a claim one run can support.

- **Pure test** — pins the arithmetic that forced the re-derivation, not the config literals:
  `targetDryDensity` must exceed the liquid-water broth's own density, and the pre-liquid-water value
  0.6 is asserted to be **refused** by `deriveCycleConfig` (`/не превышает текущую/`); it must be
  below close packing (√2); the dry box realises the configured density to 6 decimals and is derived
  from N (not from a box literal); the schedule's phase boundaries and transition steps are the
  file's own numbers. Output: `DRYWET-CONFIG … concentrationFactor=1.1002 …`.
- **GPU test** — one full cycle of the file's own schedule at box 20, liquid water, against a
  cycle-free twin with the same seed. Asserts: the dry box realises ρ = 1.34; the box returns to the
  wet box **exactly** (`toBe`, not `toBeCloseTo`); the ramp really charged
  `(rampSteps−1)·rampRelaxSteps = 1000` real steps; `nonFinite` is 0/0 at **four** points including
  immediately after the ramp (the loud-failure guard through the densest instant); census and charge
  identical between the two arms; **mobility at the dry box < half the control's over the same step
  window**; and **the cycled arm ends one cycle with fewer bond events than the control**, with the
  dry-phase rate below the control's matched-window rate. Margins are set from the box-30
  measurements and are loose enough that only a **reversal** of the measured effect fails them.

---

## 8. Regressions — every one run, one invocation at a time, with numbers

### 8.1 The list, before → after

| test file | result | numbers, before → after |
|---|---|---|
| `tests/gate6-bilayer.test.ts` | **2 passed** | area 1.2113±0.0076 → **1.2054±0.0126**; thickness 4.4817 → **4.4943**; converge-from-1.55 tail 1.2070 → **1.2049**, from-0.9 tail 1.2132 → **1.2106**, `peaksOk true` both. All inside 1.1–1.5 / 4–6 and inside the published n=5 scatter (area 1.1211–1.2215, thickness 4.316–5.103). |
| `tests/water-bilayer-area-move.test.ts` | **1 passed** | area 1.1653 → **1.1621** [1.1588, 1.1669]; thickness 4.4016 → **4.4645**; drift t −1.26 → **+0.10**; water in core 22/4500 → **28/4500**; buried 0.0547 → **0.0816**; clusterFraction 1.0000; accepted 0.1210 of 3000; **verdict=passed**. These two rows are what §5.1 recomputes the floor from. |
| `tests/catalyst-turnover.test.ts` | **1 passed** | cc_bond 442 → **439**, co_bond 361 → **359**, chains 178 → **189**, amphiphiles 125 → **134**, bare carbons left 44 → **44**, occupiedAtEnd 2 → **3**, nonFinite 0, charge 0, census `{C:779,O:3117,H:779,M:52,W:695,K:0}` exact. Pin's A/B thresholds hold. |
| `tests/soup-nonfinite-guard.test.ts` | **1 passed** (after §8.2) | guard fires at step 1 with 120/120 non-finite components; healthy control 0/0 at three points; cost scan **0.176 ms** vs chunk1000 **634.98 ms** = **0.028 %** (was 0.212 ms / 331.68 ms = 0.064 % — this machine was slower per chunk today, so the *fraction* fell). |
| `tests/soup-cold-start-relax.test.ts` | **1 passed** (after §8.2) | max&#124;F&#124; 3.518e4 → **17.90** (1965×), maxDisplacement **1.148 σ** ≤ bound 10.05; steps 0→0; velocities (u32 bit patterns), both RNG streams, bond graph, all four event counters, census and charge unchanged; refusal at `globalStep != 0` fires; payoff pair holds (throws without the stage, `nonFinite=0` with it). |
| `tests/sim.test.ts` | **8 passed** | unchanged (membrane engine; grid-vs-brute-force force identity intact) |
| `tests/run-ui.test.ts` | **8 passed** | unchanged |
| `tests/soup-forces.test.ts` | **1 passed** | unchanged — grid + Verlet list vs full O(N²), no GPU console warning |
| `tests/soup-area-move.test.ts` | **2 passed** | `composeVsDirect maxDiff=9.537e-7`; CPU potential = antiderivative of GPU forces; jacobian (N_beads−N_mol)·kT holds at N=2197 |
| `tests/soup-bonds.test.ts` | **3 passed** | catalyst gating holds; census+charge conserved under live reactions; assembly-vs-temperature `{"0.9":0.07875,"1.8":0.03375}` (was `{"0.9":0.105,"1.8":0.03125}` — same ordering, inside the test's own tolerance) |
| `tests/soup-valence.test.ts` | **1 passed** | no carbon over 2 C–C or 1 C–O, no head over chainCapacity, 45 000 steps |
| `tests/soup-checkpoint.test.ts` | **3 passed** | round-trip exact; "a checkpoint does not change what the run computes next" holds; same-process continue writes real files |
| `tests/soup-rules.test.ts` | **4 passed** | schema + detailed balance unchanged; **the re-derived `dryWetCycle` section validates** |
| `tests/soup-boxcycle.test.ts` | **7 passed** | the pure box-cycle coordinate map, involution, long-coiled-chain regression, `computeDryBox` density and the schedule — all unchanged by the new numbers |
| `tests/params.test.ts` | **3 passed** | the literal scanner is clean; no rank-A constant appears anywhere new |
| `tests/continuous-run-audit.test.ts` | **1 passed** (×3 runs) | audited the two controls and the 7-checkpoint window lineage; every invariant `expect`ed |
| **`tests/soup-drywet-cycling.test.ts`** (NEW) | **2 passed** | see §7 |

**No tolerance was changed, no corridor widened, no assertion relaxed, no threshold moved.**

### 8.2 One real defect found in the test harness, fixed at the source rather than in the assertion

`tests/soup-nonfinite-guard.test.ts` and `tests/soup-cold-start-relax.test.ts` both failed on
`AssertionError: браузер сообщил об ошибке/предупреждении GPU во время теста: error: Failed to load
resource: the server responded with a status of 404 (Not Found)`. Every substantive number in both
tests was correct; the failure was the "page logged no console error" assertion. It survived clearing
`node_modules/.vite/deps_temp_*` and the whole `node_modules/.vite`, so it was **not** the
dependency-reoptimisation flake the predecessor recorded. Diagnosed by listening on
`page.on('response')`:

```
$ nice -n 15 npx tsx scratch-404.ts
FAILED 404 http://localhost:5173/favicon.ico
```

Chrome's implicit favicon request against a dev server that has no favicon. Fixed by declaring an
empty inline icon in `tests/runner.html` (one `<link rel="icon" href="data:," />` plus a comment
recording the diagnosis) — the request is no longer made, so the console is clean again. **No
assertion was weakened**; the alternative (loosening "no console messages") would have blinded the
one check whose entire purpose is to catch GPU validation warnings. Both tests pass afterwards.

---

## 9. Invariants per checkpoint

Checked by `tests/continuous-run-audit.test.ts` (with `expect`, off-GPU, from the checkpoints' own
raw bond slots) over all 7 window checkpoints, all 6 box-30 control checkpoints (seed 19) and all 3
(seed 23), plus by the engine's own guards during every run:

| invariant | how | result |
|---|---|---|
| no non-finite state | IEEE-754 exponent scan of every position/velocity component, per step-chunk **and** after every ramp increment | `nonFinitePositions = 0, nonFiniteVelocities = 0` at **every** checkpoint of every run in this task; the guard never fired except where it was made to (§8.1) |
| no Verlet overflow / drift | `assertVerletSafety`, throws | no throw in 12 campaign invocations, 127 001 window steps + 408 000 box-30 steps + 12 ramp increments per cycle × 9 cycled runs |
| every covalent bond length survives a box change | `applyBoxScaleOnce`'s runtime self-check, every bonded pair, tolerance 1e-3 σ | **never threw**, across 108 real box changes (9 cycled runs × 12 increments) — i.e. the rigid-CoM map held on live, coiled, reacting chains at ρ up to 1.34 |
| monomer conservation | census recomputed from the species slot of every particle vs the checkpoint's **own** `config.start` | `{C:11462, O:45848, H:11462, M:764, W:133100}` exactly at every window checkpoint; `{C:1860, O:7440, H:1860, M:124, W:21600}` exactly at every box-30 one; `{C:779,O:3117,H:779,M:52,W:695,K:0}` in the turnover test |
| charge | recomputed | 0 everywhere |
| valence | `carbonCCover2`, `carbonCOover1`, `headOverChainCapacity`, `nonBondableBonded`, `degreeOver3`, `headNotTerminal` | **all 0 at every checkpoint** |
| box returns exactly after rehydration | `toBe`, not `toBeCloseTo` | exact, pinned by the new test |
| RNG state really restored | nonzero-byte count of `bondRngB64` on every resume | 131 029–131 040 of 131 536 (99.6 %) — and exactly 0 in the deliberately re-created historical case |

---

## 10. Resources, wall time, housekeeping

- **Compute chunks used: 35 of the 35 allowed** (the budget is exhausted, which is why no further
  window chunk was run). Breakdown: 1 driver smoke, 3 baseline/mobility, 3 RNG-systematic (one lost
  to an `await` I omitted in my own scratch driver and immediately re-run), 1 one-cycle mechanism +
  RNG diffusion, 1 six-cycle run, 2 control seed 19, 1 seed-23 cycled, 1 seed-23 control, 1 too-fast
  ramp probe, 2 new test (one lost to an import name), 1 favicon diagnostic, 6 regression groups
  (including 3 lost to the favicon 404 before it was diagnosed), 1 window start proof, 7 window
  chunks, 4 CPU-only offline audits (seconds each).
- **Total steps ≈ 1.77 M** — 127 001 in the window campaign, 204 000 + 204 000 (cycled + control,
  seed 19), 102 000 + 102 000 (seed 23), 120 000 baseline, 120 000 RNG-systematic, 48 000 ramp probe,
  68 000 in the new test, 34 000 + 24 000 mechanism/diffusion, ~600 000 across the regression suite.
- **Longest single foreground invocation: 594 s** (window chunk 1). **Two invocations exceeded the
  600 s foreground cap and were moved to a tracked background task by the harness itself** — window
  chunk 1's 20 000-step request (594 s of stepping) and the six-cycle run (758 s). Neither was an
  explicit backgrounding choice and neither was lost: the completion notification carried the full
  real output, quoted above. After the first of these I re-sized every subsequent window chunk to
  17 000–18 000 steps so the rest finished inside the cap. Every invocation was `nice -n 15`, **one at
  a time, foreground, never deliberately backgrounded**; no two compute processes ever ran at once
  (each long run was waited out with a `pgrep` until-loop that does no work of its own).
- **No orphan processes remain.** `pkill -f puppeteer_dev_chrome_profile` was run after **every**
  browser-bearing invocation and `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** every
  time, including after the last. `node_modules/.vite` was cleared twice while chasing the 404.
- **The dev server on :5199 was neither started, stopped nor inspected.** Every `gpuPage()` opened its
  own vite server on its own port (5173 in the diagnostic above).
- `tests/soup-vesicle.test.ts` was **never** run. The full suite was never run. `--dump-dom` was never
  used. **No particle array was ever transferred as JSON numbers:** every reduction (MSD, non-finite
  counts, amphiphile census, aggregate sizes, event counts, RNG nonzero-byte counts) was computed
  inside the page or inside the offline auditor and returned as scalars.
- **Checkpoints written:** 8 window files (~16 MB each at N = 202 636), 6 + 3 box-30 control files.
  `data/checkpoints/` is a gitignored run artifact.
- **Scratch drivers deleted, not committed** (`scratch-wd.ts`, `scratch-404.ts`); their exact
  invocations are quoted in §1.2, §2.1, §2.3, §4.2, §4.3, §4.4 and §8.2 so every run is reproducible.

## 11. Concerns

1. **The headline is a negative and it is worse than the last one.** The binding constraint's
   shortfall went 8.89× → 3.36× → **33.4×**. Two things caused it and neither is a bug: real liquid
   water halves the yield and fragments the material, and the mechanism the literature offers against
   dilution is one this engine can only express at 1.3 % of its literature amplitude.
2. **This model's wet–dry cycle is not Deamer's wet–dry cycle, and the gap is structural.** Water
   cannot leave a fixed-composition box (§3.3.1); the bond rules are not coupled to water activity
   (§3.3.2); there is no hydrolysis for the wet phase to reverse (§3.3.3); and there is no lamellar
   stack to bud (§3.3.4). A negative result about *this* cycle is **not** evidence against wet–dry
   cycling as prebiotic chemistry. The honest next step is a composition-changing mechanism (water
   beads removed and reinserted between phases, particle count varying at a phase boundary), which is
   a real engine change and its own task — not a parameter.
3. **The composition route to "dry" already measured the literature-predicted direction and was not
   used here.** The predecessor's token-water arm (ρ_W = 0.26) has **1.84× the yield** of liquid water
   — i.e. dehydration helps, exactly as L1/L5 say. That is the model's honest representation of the
   dry state, and the engine simply cannot alternate it within one trajectory. If anyone reads this
   report as "cycling doesn't work", the precise statement is: **box-scaling cycling doesn't work
   here; the composition axis it cannot cycle does point the literature's way.**
4. **The window is 1.06 σ wide and was 12.7 σ wide one task ago.** It is now sensitive to numbers with
   ±5 % scatter: the floor band alone (937–968) moves the supply end by 0.59 σ, and an 8 % lower
   ρ_amph would close the window entirely. Any future task must recompute both ends before choosing a
   box, and should treat "the window is empty" as a live possibility rather than an error.
5. **The window run is 127 000 steps and one seed.** Its plateau (107 000 steps flat) is shorter than
   the 285 000-step precedent, purely because N = 202 636 costs 29.7 ms/step here. Supply and
   aggregate size are both frozen over the last 89 000–107 000 steps, so I do not expect a longer run
   to change the verdict, but I have not measured that, and no second seed was run at box 55 at all.
6. **The cycled/control comparison is n = 2 seeds, with the second seed at 3 cycles rather than 6.**
   The effect (cycled behind control in bond count at every cycle, in both seeds) is larger than the
   cross-seed scatter for bond counts but **not** for aggregate size, where cycled 11.2 vs control
   12.7 is within the ±2–4 molecule scatter. The strong claim is about chemistry; the aggregate-size
   claim is only "no measurable improvement", not "measurably worse".
7. **`targetDryDensity = 1.34` is defended as the mobility ceiling, not as a physical dry state.** At
   ρ = 1.40 the medium is 1.009× from close packing and would be a high-pressure artefact, which the
   predecessor already refused to call a liquid. There is no density in this model that is a *dried
   film*, and 1.34 should not be quoted as one.
8. **The zero-RNG systematic is now measured but the affected history is not re-run.** Every resumed
   campaign before the fix carried a ~3× suppression of bond formation from its first resume. The
   published 271-amphiphile box-54 result is therefore a **lower bound**; nobody has re-run it with
   real RNG state, and I did not (this task's budget went to liquid water, cycling and the box-55
   window). That re-run is the single cheapest way to find out whether the vesicle programme is
   actually as far away as the published trace suggests.
9. **`clay: false` in every arm here**, as in all three predecessors, because `soup/cli/campaign.ts`
   pins it (a box change is refused on a system with an immobile phase — which is doubly binding for a
   cycled run). Same direction of bias as before (clay measures −23.8 % chain growth), so it cannot
   have manufactured any negative above.
10. **The audit's start-proof assertion is scale-dependent** (§6.1): it demands zero bonds at
    `globalStep <= 1`, and at N = 202 636 one C–C bond forms within the first step. I left the
    assertion alone and worked around it by auditing the run lineage separately, but the next task at
    an even larger N will hit it again; the right fix is probably to assert "no amphiphile and no
    aggregate" rather than "no bond", and that is a change to a committed assertion, so I did not make
    it unilaterally.
11. **`radialHeadShells = 2` at 5 of 7 window checkpoints is the one encouraging number in this
    report** — the bilayer's local two-layer order forms readily at liquid water, and forms *faster*
    than before (stage `bilayer` before step 20 000 against the predecessor's 45 000). What fails is
    exclusively the **size** of the objects carrying that order. Whatever is tried next should aim at
    coalescence of ~300 small aggregates, not at making bilayer order appear.

## 12. Files

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/wet-dry-cycling-report.md` — this report.
  `.gitignore` line 6 ignores `.superpowers/`, so like every predecessor it is committed with
  `git add -f`; the deviation is deliberate and flagged rather than silently introduced.
- NEW: `tests/soup-drywet-cycling.test.ts` (244 lines).
- NEW artifacts: `verify/out/window-run-55w-trace.json` (7 window checkpoints),
  `verify/out/wetdry-control-19-trace.json` (6), `verify/out/wetdry-control-23-trace.json` (3).
- MODIFIED: `data/soup.json` — the `dryWetCycle` section only: `cycles` 5→6, `periodSteps`
  50000→32000, `dryFraction` 0.5→0.65, `targetDryDensity` 0.6→1.34 (`enabled`, `rampSteps`,
  `rampRelaxSteps` unchanged), plus the derivation appended to that section's own `basis` with the
  five citations and their ranks. **Nothing else in the file changed.**
- MODIFIED: `tests/runner.html` — one `<link rel="icon" href="data:," />` and its comment (§8.2).
- `data/params.json` untouched. No production `.ts`/`.wgsl` file was modified by this task at all.
  Largest file in the tree is still `soup/wgsl/step.wgsl` at 596 lines; nothing crossed 600.
