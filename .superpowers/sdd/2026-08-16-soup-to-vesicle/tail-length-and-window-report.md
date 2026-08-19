# Tail length by mechanism, and the conditional window — report

Task `tail-length-and-window` (2026-08-19/20). Branch `stage-a-atoms`, in place, no worktree.
Predecessor: `continuous-run-report.md` (its §9.3 named this defect). Nothing was tuned to a target:
no threshold moved, no corridor widened, no rank-A constant re-fitted, `co_bond.attemptRate` untouched,
no pre-made amphiphile or pre-built patch used, and the monomers-only start is proven from the window
run's own step-1 checkpoint (§5.2), not assumed.

**Verdict up front, three lines.**

1. **The defect is real and the mechanism fixes it.** Mean PER-TAIL length falls from **4.597 beads**
   (the continuous run, re-measured) to **2.540 beads** (the window run) — inside the 2–3 the C12–C18
   mapping requires — by COMPOSITION alone (head supply raising the termination-to-propagation ratio,
   O:C 0.333 → 4.0). Measured α fell 0.9235 → 0.6215; the ASF mean `1/(1−α)` it predicts, **2.642**,
   agrees with the measured 2.540 to **4 %**.
2. **The consequence claim in the brief is REFUTED as stated, and the packing-parameter argument
   behind it is refuted arithmetically.** For a single chain the packing parameter is
   chain-length-INDEPENDENT (v and l both scale with n): in bead units `p` went 4.219 → 4.487
   (+6.4 %) while the tail HALVED, and Tanford's own `v/l_c` changes by **1 %** over n_c = 12 → 57.
   Shorter tails did NOT turn the largest aggregate from a rod into a sheet (in-plane symmetry got
   *worse*, 0.4781 → 0.1492). What they did do is make every aggregate **9× bigger** and put a
   genuine bilayer-qualifying object in the population for the first time in this project.
3. **VESICLE STILL NOT REACHED — but the ladder moved two rungs, from `amphiphiles` to `bilayer`.**
   Binding constraint is again AGGREGATE SIZE, now **108 amphiphiles against the 960 floor = 8.9×
   short** (it was 80×). Amphiphile SUPPLY is no longer binding for the first time: **1600 ≥ 960**,
   1.67× over. `radialHeadShells` was EVALUABLE for the first time and **passes (== 2)**; the run
   fails on closure alone (**0 encapsulated water against a 32.29-bead threshold**).

---

## 1. `cc_break`: DISABLED BY CONSTRUCTION, not "merely never fires"

Asked first because the answer decides whether lever (b) is available at all.

**Finding: disabled by construction. It is arithmetically impossible for `cc_break` to fire, at any
step count, at any temperature.** Three pieces of evidence, each a real file:

```
$ python3 -c "import json; d=json.load(open('data/soup.json')); print([{k:r[k] for k in ('id','kind','energyKT','attemptRate','rank')} for r in d['rules']])"
[{'id': 'cc_bond', 'kind': 'bond', 'energyKT': 6.0, 'attemptRate': 0.05,        'rank': 'D'},
 {'id': 'cc_break','kind': 'break','energyKT': 6.0, 'attemptRate': 0,           'rank': 'D'},
 {'id': 'co_bond', 'kind': 'bond', 'energyKT': 8.0, 'attemptRate': 2.562669112, 'rank': 'D'},
 {'id': 'co_break','kind': 'break','energyKT': 8.0, 'attemptRate': 0.05,        'rank': 'D'}]
```

1. `data/soup.json` `cc_break.attemptRate = 0` **exactly**, with its own written basis: *"ОТКЛЮЧЕНО,
   необратимо (task 'kinetic-growth', 2026-08-17) … правило разрыва ОСТАВЛЕНО в файле … но НИКОГДА не
   срабатывает"*.
2. The propagation path is `attemptProbability(r, dt) = r.attemptRate * dt` (`soup/src/rules.ts:423`)
   → `packVec4(rules.map(... attemptProbability ...))` (`soup/src/soup-buffers.ts:267`) →
   `BP.attemptProbBreak`. The shader's gate is
   ```
   soup/wgsl/bond-dispatch.wgsl:187:    if (bondUniform01(rng) >= vget(BP.attemptProbBreak, r)) { continue; }
   ```
   `bondUniform01` returns a value in `[0, 1)`; `x >= 0` is true for **every** such x, including 0.
   So the `continue` is taken unconditionally and `acceptanceProbability` — the one formula that
   carries detailed balance — is never even evaluated for this rule. This is a compile-time-style
   impossibility, not a low probability.
3. The runs confirm it: `cc_break = 0` at **every one of 12** box-54 checkpoints (§5.3) and at every
   one of the box-90 checkpoints before them.

### 1.1 Why lever (b) was NOT used, said plainly

The brief's reading — "`cc_break` firing zero times means chain growth currently violates detailed
balance in practice, so restoring a working reverse reaction is a physics correction, not a knob" — is
half right and the half that is wrong matters.

- The **pairing invariant the plan requires is satisfied**: a `break` rule with the SAME `energyKT`
  (6.0) exists and `assertRulesConsistent` checks it. Detailed balance is *formally defined* for this
  bond.
- It is deliberately broken *in practice*, **with a literature basis, not by omission**: a covalent
  C–C bond does not dissociate at 175 °C on any reachable timescale (McCollom, Ritter & Simoneit,
  *PNAS* 96 (1999) 2555 — Fischer–Tropsch synthesis of hydrocarbons, 2–3 days at 175 °C, C2–C35+;
  already cited in this project's own spec). Chain length in this chemistry is a KINETIC quantity, and
  the file says so.
- Turning `cc_break` back on has **already been tried and already failed, measured**:
  `energy-calibration-report.md` held it enabled and tried to reach long chains through an equilibrium
  energy of 8.8 kT. Both rate-free checks failed — the measured length distribution gave mean 2.826
  against Flory's predicted 14.85, and the van't Hoff slope recovered **1.31 kT (r² = 0.617)** instead
  of the 8.8 kT set, a factor of ~7 — because at `accept(break) = exp(−8) ≈ 3.35e-4` fewer than 1 % of
  bonds get a single chance to break in 50 000 steps. The system *was already behaving irreversibly*;
  `attemptRate = 0` only says so honestly.

So re-enabling `cc_break` would not be a physics correction here — it would revert a documented,
literature-backed rank-D decision to a regime this project has already measured as unreachable. **It
was not used.** Lever (a) was, and it works (§3).

One thing the brief flagged is now resolved in passing: **`co_break`, the one reversible channel that
had never fired, fired in these runs** — 4 events by step 15 000, 21 by step 30 000 (seed 19), 25 by
30 000 and **51 by 45 000** (seed 23). `continuous-run-report.md` concern §9.3's "never exercised at
all" no longer holds.

---

## 2. The O:C sweep, re-measured in the corrected physics

`broth-composition-report.md` §5's "raising the head fraction changed no measurable quantity" is
**void**, and its own §5 heading says why: `co_bond` fired **0 times**, so there were no amphiphiles
whose tail length could depend on anything. `hydrophobic-asymmetry-report.md` then restored tail–tail
dispersion; `continuous-run-report.md` measured 1027 real `co_bond` events. The dependence is
measurable now.

Harness: `tests/tail-length-oc-sweep.test.ts` (new). Box 30, kT = 1.1, `clay: false` (matching
`soup/cli/campaign.ts`'s own hard pin, so sweep and window run are the same configuration), 150 000
steps, two seeds (19, 23) per point, analysis run INSIDE the page with the same
`findAmphiphiles` / `amphiphileHistogram` / `carbonChainLengths` / `stageOf` the live run uses — there
is no second implementation of any measurement, and no particle array ever crossed CDP as JSON numbers.

### 2.1 Two lengths, and which one the 2–3 target constrains

`Amphiphile.length` (this project's own `meanTailLength`) is TOTAL carbon over a head's one or two
tails; `headPlacement.chainCapacity = 2` means many heads carry two. The "C12–C18 → two tail beads"
mapping constrains the **PER-TAIL** length. Both are reported throughout; only the second is
comparable to 2–3. For the continuous run the difference is 5.341 (total) vs **4.597** (per tail).

### 2.2 The sweep, at step 150 000

`α_ev = cc_bond/(cc_bond + co_bond)` — the event-ratio `k_p/(k_p+k_t)` the ASF frame defines.
`ASF mean = 1/(1−α_ev)`. `α_rec` = `recoverAlphaFromChainLengths` on the raw C–C chain population
(`soup/src/equilibrium.ts`, the plan's Task-7 machinery, unchanged). Only `O` varies; C 1500, H 1500,
M 100, W 10700 fixed, so ρ_tot rises with O — **that confound is measured out in §2.4.**

| O | O:C | ρ_tot σ⁻³ | α_ev s19 | α_ev s23 | mean α_ev | ASF mean | per-tail s19 | per-tail s23 | mean | |Δ| seeds | yield s19/s23 | stage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 500 | 0.333 | 0.5296 | 0.9082 | 0.9219 | 0.9151 | 11.78 | 6.090 | 6.635 | **6.363** | 0.545 | .0293/.0313 | amphiphiles |
| 1000 | 0.667 | 0.5481 | 0.8722 | 0.8526 | 0.8624 | 7.27 | 5.520 | 5.165 | **5.343** | 0.355 | .0467/.0473 | micelles / amphiphiles |
| 1500 | 1.000 | 0.5667 | 0.8642 | 0.8655 | 0.8649 | 7.40 | 5.009 | 5.520 | **5.265** | 0.511 | .0493/.0453 | **bilayer** / micelles |
| 3000 | 2.000 | 0.6222 | 0.8087 | 0.7859 | 0.7973 | 4.93 | 4.803 | 4.228 | **4.516** | 0.575 | .0807/.0640 | micelles |
| 4500 | 3.000 | 0.6778 | 0.7370 | 0.7215 | 0.7293 | 3.69 | 3.752 | 3.607 | **3.680** | 0.145 | .0640/.0727 | micelles |
| 10520 | 4.000 | 0.6778¹ | 0.5949 | 0.5914 | 0.5932 | 2.46 | 2.514 | 2.534 | **2.524** | 0.020 | .0757/.0722 | micelles / **bilayer** |
| 9000 | 6.000 | 0.8444 | — | — | — | — | **RUN DIVERGED** (§2.3) | | | | | |

¹ this row is the window-candidate composition, C 2630 / H 2630 / M 175 / W 2345 (§4), not the
C = 1500 family — its O:C is 4.0 and its total density matches the O = 4500 row exactly.

**Scatter each number is judged against.** Per-tail run-to-run half-range over the six usable points:
0.010 – 0.288 σ, i.e. **±0.3 beads** is the honest resolution of one point at n = 2. Every
composition-to-composition step in the table except O = 1000 → 1500 (5.343 → 5.265, Δ = 0.078)
exceeds that; those two points are equal within scatter and are reported as equal. The overall
0.333 → 4.0 change, **6.363 → 2.524 = −3.84 beads**, is **13× the scatter**.

Sample raw output, unedited, one line of many:

```
OC-SWEEP O=10520 seed=19 box=30 start={"C":2630,"H":2630,"M":175,"W":2345,"O":10520} step=150000 stage=micelles cc=1517 co=1033 alphaEvent=0.5949 asfMean=2.469 alphaRec=0.5498 r2=0.918 amph=199 frac=0.24753 meanCarbonPerAmph=3.271 meanPerTail=2.514 twoTailed=60 ccFrac=0.5768 rCC=1.0022(n=1517) aggs=5 qual=3 shareQ=0.9799 yield=0.07567 nAmphDensity=7.370e-3 L1={"amphiphileCount":78,"particleCount":326,"radiusOfGyration":6.392,"flatnessRatio":0.5382,"inPlaneSymmetry":0.7115,"radialHeadShells":"1","transverseHeadShells":0,"cavityVolume":0.125} nonFinite=0 wallMs=253343
OC-SWEEP-HIST O=10520 seed=19 step=150000 perTail={"2":172,"3":57,"4":18,"5":10,"6":1,"8":1} amphLength={"2":92,"3":33,"4":36,"5":21,"6":7,"7":6,"8":3,"11":1} chain={"1":390,"2":278,"3":248,"4":108,"5":55,"6":18,"7":9,"8":5,"9":1,"13":1}
```

### 2.3 The measured histogram against the ASF prediction — where they agree and where they do not

Best statistics available: the window run's final state, 1977 tails (§5). ASF from the event-ratio
α_ev = 0.6215, renormalised over the same n ≥ 2 support the measurement has:

| per-tail n | measured fraction | ASF(α_ev = 0.6215) |
|---|---|---|
| 2 | **0.6697** | 0.3818 |
| 3 | 0.2114 | 0.2373 |
| 4 | 0.0673 | 0.1475 |
| 5 | 0.0319 | 0.0917 |
| 6 | 0.0106 | 0.0570 |
| 7 | 0.0035 | 0.0354 |
| 8 | 0.0030 | 0.0220 |
| 9–11 | 0.0025 | 0.0275 |

**The SHAPE is geometric — the ASF signature is there, cleanly.** An OLS fit of `ln(count)` against n
over the measured per-tail histogram (the same regression `recoverAlphaFromChainLengths` performs)
gives slope-α = **0.4514 with r² = 0.9694**; the same fit on the continuous run's own per-tail
histogram gives **0.6850, r² = 0.9579**. So the distribution is log-linear in both states and the
lever moves its slope in the right direction by the right amount.

**The SLOPE-α and the EVENT-α disagree, and that is reported rather than reconciled by fitting.**
0.4514 (histogram) vs 0.6215 (events); their means are 2.82 and 2.64 against a measured 2.540. Two
mechanisms are visible in the table and both are structural, not noise:

- **there is no per-tail length 1, in any run, ever.** Nucleation needs two bare carbons plus a free
  catalyst (`propagateOnCenter`), so a chain is already 2 carbons before a head can cap it. The ASF
  distribution's own n = 1 term therefore has nowhere to go and piles into n = 2, which is exactly
  the measured excess (0.670 vs 0.382).
- a `cc_bond` event does not always lengthen a *recognised* tail — branches, rings and never-capped
  chains are all counted by the event ratio and all rejected by `findAmphiphiles`.

Per the plan's Task-7 instruction ("if the measured distribution is systematically wider than
predicted, that is NOT grounds to change the tolerance"), no tolerance was changed and no α was
fitted to the target: the numbers are printed side by side and the disagreement is the result.

### 2.4 The density confound, measured out (two controls)

Raising O at fixed C/H/M/W raises ρ_tot. Two controls at box 30, seed 19, step 150 000 separate the
two effects:

| arm | O | W | ρ_tot | α_ev | per-tail | yield |
|---|---|---|---|---|---|---|
| baseline | 500 | 10700 | 0.5296 | 0.9082 | 6.090 | .0293 |
| **control A** — same O:C, density raised to the O = 4500 value | 500 | **14700** | **0.6778** | 0.9240 | **6.595** | .0340 |
| high-O | 4500 | 6700 → | 0.6778 | 0.7370 | 3.752 | .0640 |
| **control B** — same high O:C, density returned to baseline | 4500 | **6700** | **0.5296** | 0.6848 | **3.333** | .0667 |

- **Density alone (control A vs baseline): +0.505 beads**, i.e. within the ±0.3–0.55 seed scatter, and
  in the *lengthening* direction.
- **O:C alone, at matched density 0.5296 (control B vs baseline): 6.090 → 3.333, −2.757 beads**, 5× the
  scatter.

The effect is the head fraction, not the density; at matched density it is *larger*, not smaller.

### 2.5 The divergence at O:C = 6, reported

```
OC-SWEEP O=9000 seed=19 box=30 start={"C":1500,"H":1500,"M":100,"W":10700,"O":9000} step=50000 stage=monomers cc=0 co=0 ... amph=0 ... nonFinite=68049 ...
```

ρ_tot = 0.8444 σ⁻³ — above the regime `broth-composition-report.md` §2 measured as unstable at 0.933
and beyond anything this engine has run. 68 049 non-finite position components, zero bonds. **It did
not throw** — `assertVerletSafety` did not catch it; the harness's own non-finite scan did. Flagged as
a concern (§8.2), not worked around: every other point in the table is at ρ_tot ≤ 0.6778, and the
window run uses 0.6778, which is measured stable at both box 30 and box 54.

---

## 3. The lever, its literature, its rank, and the mean tail achieved

**Lever used: (a) COMPOSITION — head supply raising the termination-to-propagation ratio.** Nothing
else was touched. `co_bond.attemptRate` remains `2.562669112`; `cc_bond.attemptRate` remains `0.05`;
no energy, threshold or corridor moved. `git diff` over `data/` is empty (§7).

**Literature.** The project has already adopted the Anderson–Schulz–Flory frame for this chemistry
(`data/soup.json` `cc_break.basis`, `kinetic-growth-report.md`), and in that frame chain length is set
by α = k_p/(k_p+k_t) alone. That the branching ratio is controlled by the FEED composition — raise the
terminating agent relative to the monomer and α falls — is the central, heavily measured fact of
Fischer–Tropsch practice:

- M. E. Dry, *The Fischer–Tropsch process: 1950–2000*, **Catalysis Today 71 (2002) 227–241**: raising
  the H₂/CO ratio (more of the chain-terminating agent per chain-building monomer) shifts the product
  spectrum to lighter hydrocarbons, i.e. lowers α; the reverse raises it.
- G. P. van der Laan & A. A. C. M. Beenackers, *Kinetics and selectivity of the Fischer–Tropsch
  synthesis: a literature review*, **Catal. Rev. Sci. Eng. 41 (1999) 255–318**: the ASF chain-growth
  probability is expressed directly as a function of the reactant partial pressures.
- McCollom, Ritter & Simoneit, **PNAS 96 (1999) 2555**, the prebiotic FT anchor already in this
  project's spec, which fixes the mechanism as chain growth on a catalytic surface with a competing
  termination step.

The project's own encounter-frequency argument is the same argument in this engine's units, and it is
already load-bearing here: `surface-growth-report.md` fixed the gating asymmetry precisely so that
`k_p/(k_p+k_t)` would be *the ratio the system actually sees*, both channels gated by the same
catalytic-centre availability. Raising [O] raises k_t through that gate and nothing else.

**Rank.** The composition (`data/soup.json` `start`) is **not a rank-A constant and not a rate**: it is
the experiment's input, the same status as `--box` or `--steps`. `data/soup.json` was **not modified**
— the O:C used here is a CLI/`start` argument on a run, exactly as `--box 54` is. The *mechanism*
(α falls as the termination-to-propagation feed ratio rises) is rank A/B literature, above. The
specific value O:C = 4.0 is rank D — an experiment-design choice, selected by the measurement in §2.2
as the first point whose measured per-tail length lands inside 2–3.

**Mean tail achieved** (all at step 150 000 unless stated):

| quantity | before (continuous run, box 90, step 184 001) | after (window run, box 54, step 300 000) | target |
|---|---|---|---|
| per-tail length | **4.597** | **2.540** | 2–3 ✓ |
| total carbon per amphiphile | 5.341 | 3.138 | — |
| α_ev | **0.9235** | **0.6215** | — |
| ASF mean `1/(1−α_ev)` | 13.079 | **2.642** (vs 2.540 measured, 4 %) | — |
| per-tail histogram slope-α (r²) | 0.6850 (0.9579) | 0.4514 (0.9694) | — |
| two-tailed heads | 120 / 742 | 377 / 1600 | — |

The box-30 confirmation of the same composition at n = 2 seeds: **2.514 and 2.534**, scatter 0.020 —
the tightest agreement anywhere in this sweep, and the reason O:C = 4.0 was chosen rather than
interpolated.

---

## 4. The consequence — packing parameter and aggregate shape, before and after

### 4.1 The packing parameter is chain-length-INDEPENDENT. The brief's premise does not hold.

Computed from this project's own measured numbers, no new constant: tail-bead volume
`v_bead = (4/3)π r_C³` with `r_C = 1.0 σ` (`data/soup.json` monomers C `radiusSigma`); extended tail
length `l = n·⟨r_CC⟩` with `⟨r_CC⟩` **measured** at every sample (0.9929 – 1.0054 σ over all 13 runs;
1.0022 used); head area `a₀ = 1.1510 σ²`, the measured area per lipid at zero tension
(`hydrophobic-asymmetry-report.md`; re-measured 1.2019 today, §6.2 — the difference moves `p` by 4 %
and changes nothing below).

For one tail of n beads:

```
p = v/(a₀·l) = n·4.18879 / (1.1510 · n · 1.0022) = 3.6313      -- n cancels EXACTLY
```

So in this bead model the only thing that can move `p` is the number of tails per head. Measured:

| state | amphiphiles | two-tailed | tails/amph | **p** | per-tail |
|---|---|---|---|---|---|
| BEFORE — continuous run, box 90, step 184 001 | 742 | 120 | 1.1617 | **4.219** | 4.597 |
| AFTER — window run, box 54, step 300 000 | 1600 | 377 | 1.2356 | **4.487** | 2.540 |
| AFTER — window run, seed 23, step 45 000 | 1553 | 375 | 1.2415 | 4.508 | 2.396 |
| BEFORE — box 30, O:C 0.333, ρ 0.5296 | 44 | 23 | 1.5227 | 5.529 | 6.090 |
| AFTER — box 30, O:C 3.0, ρ 0.5296 | 100 | 32 | 1.3200 | 4.793 | 3.333 |
| AFTER — box 30, O:C 4.0 | 199 | 60 | 1.3015 | 4.726 | 2.514 |

**The tail halved and `p` moved by +6.4 % (box 90 → box 54), or FELL by 13–20 % in the matched box-30
pairs — the opposite sign to the brief's premise in the controlled comparison.** The absolute value
(≈ 3.6 per tail) is far above the classic 1/3 – 1 window because `a₀ = 1.151 σ²` is this model's own
bead-scale area per lipid, not a molecular head-group area; only the CHANGES above are meaningful.

Molecular cross-check with the standard formulas, using this project's own "two tail beads ↔ C12–C18"
mapping (6–9 carbons per bead) — Tanford, *The Hydrophobic Effect* (1980): `v = 27.4 + 26.9·n_c` Å³,
`l_c = 1.5 + 1.265·n_c` Å; Israelachvili, *Intermolecular and Surface Forces* 3rd ed., ch. 20 for
`p = v/(a₀ l_c)`:

| n_beads | n_c | v, Å³ | l_c, Å | **v/l_c, Å²** |
|---|---|---|---|---|
| 2.000 | 12 | 350.2 | 16.68 | **21.00** |
| 2.540 | 15.2 | 437.4 | 20.78 | **21.05** |
| 4.597 | 27.6 | 769.4 | 36.39 | **21.14** |
| 6.363 | 57.3 | 1567.9 | 73.94 | **21.20** |

`v/l_c` changes by **1.0 % over a factor 4.8 in chain length.** Both intercepts are small and both
terms are linear in n_c, so the standard theory says the same thing the bead model does: for a
single-chain amphiphile, chain length is not the knob that sets the packing parameter — the head area
and the number of chains are. The brief's chain "long tails raise the packing parameter … the rod the
run measured may be a direct consequence" is therefore **not supported**, and this report does not
lean on it.

### 4.2 Aggregate shape statistics, before and after — the claim tested

Two comparisons. The first is matched on box, total density, carbon count and seed, so only O:C
differs; the second is the two campaigns.

**Matched pair (box 30, C 1500, ρ_tot 0.5296, seed 19, step 150 000), largest aggregate:**

| | O:C = 0.333 (before) | O:C = 3.0 (after) |
|---|---|---|
| amphiphiles | **16** | **52** (3.25×) |
| radius of gyration σ | 5.469 | 5.918 |
| flatness λ₀/λ₂ (≤ 0.35 lamellar) | 0.5674 ✗ | 0.5039 ✗ |
| in-plane symmetry λ₁/λ₂ (≥ 0.5 lamellar) | 0.7315 ✓ | 0.5991 ✓ |
| radial head shells (== 2 for vesicle) | **unavailable** (< 20 heads) | **1** |
| stage | amphiphiles | micelles |

**Campaign pair, largest aggregate:**

| | continuous run, box 90, step 184 001 | window run, box 54, step 300 000 | window run, box 54, step 210 000 |
|---|---|---|---|
| amphiphiles | **12** | **108** (9.0×) | 105 |
| particles | 107 | 436 | 448 |
| Rg σ | 5.835 | 8.323 | 6.635 |
| principal moments | 2.065 / 10.345 / 21.64 | 6.230 / 8.185 / 54.854 | 7.987 / 12.985 / 23.046 |
| flatness (≤ 0.35) | 0.0954 ✓ | 0.1136 ✓ | **0.3466 ✓** |
| in-plane symmetry (≥ 0.5) | **0.4781 ✗** | **0.1492 ✗✗** | **0.5634 ✓** |
| radial head shells (== 2) | unavailable | **2 ✓** | 1 |
| transverse head shells | 0 | 0 | **2** |
| stage | amphiphiles | micelles | **bilayer** |

**Verdict on the claim, with the numbers: the claim as stated is REFUTED for the largest aggregate,
and CONFIRMED at the population level.**

- The largest aggregate's in-plane symmetry went from 0.4781 (just under the 0.5 sheet threshold —
  the rod the brief points at) to **0.1492**, i.e. it became a *more* extreme rod, not a sheet. The
  matched box-30 pair says the same thing more mildly (0.7315 → 0.5991, both above threshold, both
  failing flatness). Shorter tails did **not** move the largest object from rod toward sheet.
- What did change: **aggregate size, by 3.25× at matched conditions and 9.0× across campaigns**, and
  with it the two criteria that need material to be evaluable at all. `radialHeadShells` became
  measurable and **reads 2** — the vesicle criterion's first clause, never once evaluable in the
  continuous run, now passes. And the population now contains an object passing BOTH lamellar
  criteria simultaneously (flatness 0.3466 ≤ 0.35 **and** in-plane 0.5634 ≥ 0.5) plus two transverse
  head layers, which is why the run's stage reads `bilayer` — a rung this project's soup runs had
  never reached.
- The aggregate population is not a single object: 39–41 of ~47 aggregates *qualify* (≥ 13
  amphiphiles) and they hold **95–98 %** of all amphiphiles. Sizes of the ten biggest at step
  300 000: `108, 84, 77, 70, 57, 63, 63, 61, 53, 56`. The system is a stable dispersion of ~50 finite
  bilayer/micellar objects, not one coarsening phase.

---

## 5. The window run

### 5.1 Box and composition arithmetic

The window is the intersection of the continuous run's two inequalities, re-derived with THIS
composition's measured amphiphile density rather than the old one:

- **supply**: ρ_amph·L³ ≥ 960 (the geometric closure floor, `continuous-run-report.md` §1.1, taken
  unchanged). Measured ρ_amph at the O:C = 4.0 composition, box 30, step 150 000: **7.37e-3 / 7.04e-3
  σ⁻³** (two seeds) → L ≥ (960/7.37e-3)^{1/3} = **50.7 σ**.
- **measurability**: R(L) = 0.067·L^{1.5} ≤ L/2 (calibrated on the box-58 object's own 29.6 σ head
  radius, unchanged) → L ≤ **55.69 σ**.

So the window is **L ∈ [50.7, 55.7] σ** — the brief's own [51.9, 55.7] with the supply end moved
0.9 σ *outward* by the higher measured yield, not by any change of threshold. **L = 54 σ chosen**
(inside both ends; R(54) = 26.59 σ against L/2 = 27.0, ratio 0.985).

Composition = the box-30 calibration point scaled by the exact volume ratio 54³/30³ = 5.832, no other
change:

```
V = 157 464 σ³      C 15338   O 61353   H 15338   M 1021   W 13676     N = 106 726
ρ_tot = 106726/157464 = 0.677780 σ⁻³   (bit-identical to the O=4500 sweep row, measured stable)
Verlet list = 106 726 × 2500 × 4 = 1.067 GB   against the measured 4 294 967 292 B ceiling
N = 106 726  against the hard max 429 496 particles                        -- both clear, 4.0x margin
predicted amphiphiles = yield × C = 0.0757 × 15338 = 1161  (seed-23 yield: 1107)   -- both >= 960
```

**Realised, honestly, against that prediction:** the run produced **1600–1687** amphiphiles
(yield 0.1043, 38 % above the box-30 calibration), so ρ_amph came out **1.016e-2 σ⁻³** — **1.49× the
6.82e-3 the window was specified at**, i.e. the run overshot the target density and sits *above* the
density at which the old long-tailed composition percolated. That is stated as a miss, not smoothed
over; §5.5 argues it makes the negative stronger rather than weaker.

Water: W = 13 676 gives a water sub-density of **0.0871 σ⁻³**, far below the 0.8 σ⁻³ this project
measured as its own liquid threshold and below even the continuous run's 0.396. The organic budget at
O:C = 4 leaves no room for more at a stable total density. Inherited-and-worsened; §8.1.

`clay: false` (`soup/cli/campaign.ts` line 264's own hard pin), `--cycle` off, kT = 1.1, seed 19.

### 5.2 The monomers-only start, from the run's own trace

```
$ nice -n 15 npx tsx soup/cli/campaign.ts --label soup2ves54 --box 54 --start '{"C":15338,"O":61353,"H":15338,"M":1021,"W":13676}' --seed 19 --kT 1.1 --steps 1 --every 1 --dir data/checkpoints/trace54
[campaign] новый запуск label=soup2ves54 (совпадающих контрольных точек в data/checkpoints/trace54 нет)
[campaign] система готова N=106726 стартовый_шаг=0 цель=1
[campaign] шаг=1/1 stage=monomers агрегатов=0 крупнейший=0 headShells=n/a cavityVolume=0.000 stepMs=24 checkpointMs=141 progressMs=22 сохранено=data/checkpoints/trace54/soup2ves54-step1.json
```

decoded off-GPU by `tests/continuous-run-audit.test.ts`:

```
RUN-AUDIT {"step":1,"N":106726,"box":[54,54,54],"stage":"monomers","species":{"C":15338,"O":61353,"H":15338,"M":1021,"W":13676},"bonds":0,"events":{"cc_bond":0,"cc_break":0,"co_bond":0,"co_break":0},"invariants":{"nonFinitePositions":0,"nonFiniteVelocities":0,"valence":{"carbonCCover2":0,"carbonCOover1":0,"headOverChainCapacity":0,"nonBondableBonded":0,"degreeOver3":0,"headNotTerminal":0,"headChainCapacity":2,"headsWithTwoTails":0}},"amphiphileCount":0,"amphiphileFraction":0,"meanTailLength":0,"meanPerTail":0,...,"aggregateCount":0,...,"largest":[]}
```

Zero bonds, zero bond events, zero amphiphiles, zero aggregates. **No pre-made amphiphile and no
pre-built patch, from the run's own recorded state.**

### 5.3 Stage trace — the CLI's own progress lines, unedited

```
[campaign] шаг=15000/30000 stage=micelles агрегатов=88 крупнейший=71 headShells=1 cavityVolume=0.250 stepMs=106258
[campaign] шаг=30000/30000 stage=micelles агрегатов=61 крупнейший=113 headShells=2 cavityVolume=0.250 stepMs=147387
[campaign] шаг=45000/75000 stage=micelles агрегатов=45 крупнейший=111 headShells=1 cavityVolume=1.000 stepMs=155753
[campaign] шаг=60000/75000 stage=micelles агрегатов=49 крупнейший=110 headShells=1 cavityVolume=1.500 stepMs=161046
[campaign] шаг=75000/75000 stage=micelles агрегатов=48 крупнейший=108 headShells=1 cavityVolume=0.750 stepMs=159597
[campaign] шаг=90000/120000 stage=bilayer агрегатов=45 крупнейший=108 headShells=2 cavityVolume=0.750 stepMs=147254
[campaign] шаг=105000/120000 stage=micelles агрегатов=54 крупнейший=108 headShells=2 cavityVolume=0.375 stepMs=160498
[campaign] шаг=120000/120000 stage=bilayer агрегатов=49 крупнейший=108 headShells=2 cavityVolume=0.375 stepMs=157822
[campaign] шаг=135000/165000 stage=bilayer агрегатов=46 крупнейший=105 headShells=1 cavityVolume=0.625 stepMs=156966
[campaign] шаг=150000/165000 stage=bilayer агрегатов=48 крупнейший=105 headShells=1 cavityVolume=0.625 stepMs=156770
[campaign] шаг=165000/165000 stage=micelles агрегатов=51 крупнейший=108 headShells=2 cavityVolume=0.375 stepMs=156458
[campaign] шаг=180000/210000 stage=bilayer агрегатов=49 крупнейший=108 headShells=2 cavityVolume=0.375 stepMs=153823
[campaign] шаг=195000/210000 stage=bilayer агрегатов=45 крупнейший=105 headShells=1 cavityVolume=0.875 stepMs=155931
[campaign] шаг=210000/210000 stage=bilayer агрегатов=47 крупнейший=105 headShells=1 cavityVolume=0.875 stepMs=157772
[campaign] шаг=225000/255000 stage=micelles агрегатов=50 крупнейший=108 headShells=2 cavityVolume=0.500 stepMs=152276
[campaign] шаг=240000/255000 stage=micelles агрегатов=48 крупнейший=108 headShells=2 cavityVolume=0.500 stepMs=155894
[campaign] шаг=255000/255000 stage=bilayer агрегатов=50 крупнейший=105 headShells=1 cavityVolume=1.125 stepMs=155803
[campaign] шаг=270000/300000 stage=bilayer агрегатов=47 крупнейший=105 headShells=1 cavityVolume=1.125 stepMs=153479
[campaign] шаг=285000/300000 stage=bilayer агрегатов=45 крупнейший=108 headShells=2 cavityVolume=0.500 stepMs=156209
[campaign] шаг=300000/300000 stage=micelles агрегатов=56 крупнейший=108 headShells=2 cavityVolume=0.625 stepMs=155939
```

**Stage transitions:** `monomers` → `micelles` before step 15 000 (both `amphiphiles` and `micelles`
crossed inside the first checkpoint interval); first `bilayer` at step **90 000**; thereafter
oscillating `bilayer` ↔ `micelles` for 210 000 steps as the bilayer-qualifying aggregate is or is not
present at the sample. `vesicle`: **never**.

Offline CPU audit of 12 checkpoints (`tests/continuous-run-audit.test.ts`, the SAME
`findAmphiphiles` / `analyzeAggregates` / `stageFromEvidence` the live run uses):

| step | stage | amph | frac | perTail | cc_bond | co_bond | co_break | aggs | qual | shareQ | L | Rg | flat | inPl | radSh | transSh | cav | encH₂O | closed |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | monomers | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | — | — | — | — | — | — | — | — | — |
| 15000 | micelles | 1362 | 0.2327 | 2.294 | 3003 | 2179 | 4 | 88 | 36 | 0.855 | 65 | 5.994 | 0.5093 | 0.6161 | 1 | 0 | 0 | 0/32.3 | false |
| 30000 | micelles | 1687 | 0.3094 | 2.343 | 5209 | 3790 | 21 | 61 | 40 | 0.949 | 113 | 9.638 | 0.0830 | 0.1632 | 2 | 0 | 0.25 | 0/32.3 | false |
| 60000 | micelles | 1615 | 0.3243 | 2.511 | 6402 | 3989 | 21 | 49 | 41 | 0.977 | 110 | 6.622 | 0.3716 | 0.5788 | 1 | 0 | 1.5 | 0/32.3 | false |
| 90000 | bilayer | 1604 | 0.3271 | 2.536 | 6604 | 4035 | 21 | 45 | 39 | 0.984 | 106 | 6.636 | 0.3485 | 0.5533 | 1 | 2 | 0.75 | 0/32.3 | false |
| 120000 | bilayer | 1600 | 0.3274 | 2.540 | 6639 | 4044 | 21 | 49 | 39 | 0.958 | 105 | 6.635 | 0.3463 | 0.5644 | 1 | 2 | 1.0 | 0/32.3 | false |
| 150000 | bilayer | 1600 | 0.3274 | 2.540 | 6639 | 4044 | 21 | 48 | 40 | 0.972 | 105 | 6.635 | 0.3465 | 0.5634 | 1 | 2 | 0.625 | 0/32.3 | false |
| 180000 | bilayer | 1600 | 0.3274 | 2.540 | 6639 | 4044 | 21 | 49 | 38 | 0.951 | 105 | 6.635 | 0.3466 | 0.5634 | 1 | 2 | 1.0 | 0/32.3 | false |
| 210000 | bilayer | 1600 | 0.3274 | 2.540 | 6639 | 4044 | 21 | 47 | 41 | 0.977 | 105 | 6.635 | 0.3466 | 0.5634 | 1 | 2 | 0.875 | 0/32.3 | false |
| 240000 | micelles | 1600 | 0.3274 | 2.540 | 6639 | 4044 | 21 | 48 | 40 | 0.970 | 108 | 8.321 | 0.1136 | 0.1494 | 2 | 0 | 0.5 | 0/32.3 | false |
| 270000 | bilayer | 1600 | 0.3274 | 2.540 | 6639 | 4044 | 21 | 47 | 41 | 0.976 | 105 | 6.635 | 0.3466 | 0.5634 | 1 | 2 | 1.125 | 0/32.3 | false |
| **300000** | **micelles** | **1600** | **0.3274** | **2.540** | **6639** | **4044** | **21** | **56** | **39** | **0.953** | **108** | **8.323** | **0.1136** | **0.1492** | **2** | **0** | **0.625** | **0/32.3** | **false** |

`cc_break = 0` at every step. **`cc_bond`, `co_bond` and `co_break` are identical to the digit from
step 120 000 to step 300 000** — 180 000 steps with exactly zero chemistry. §6 shows why, and that it
is not "the reagents ran out".

Reproducibility, seed 23 (independent lineage, 45 000 steps):

```
[campaign] шаг=15000/45000 stage=micelles агрегатов=95 крупнейший=78  headShells=1 cavityVolume=0.375
[campaign] шаг=30000/45000 stage=micelles агрегатов=42 крупнейший=162 headShells=1 cavityVolume=0.375
[campaign] шаг=45000/45000 stage=micelles агрегатов=39 крупнейший=154 headShells=1 cavityVolume=0.875
```
audited: step 45 000 — amph 1553, frac 0.3012, perTail **2.396**, α_ev 0.5845, aggs 39, qual 29,
shareQ 0.9697, largest **154** amphiphiles / 627 particles, Rg 9.038, moments 9.362/16.382/55.933,
flat 0.1674, inPl 0.2929, radialHeadShells 1, cavity 0.875, encapsulated water **0/32.32, closed
false**. Non-finite 0, invariants clean.

**Run-to-run scatter on the binding quantity: largest aggregate 111 (seed 19) vs 154 (seed 23) at
step 45 000** — ±20 %. The 8.9× shortfall below 960 is 40× that scatter.

### 5.4 The largest aggregate against every threshold

Final state, step 300 000, seed 19, full unedited record:

```
{"amphiphileCount":108,"particleCount":436,"radiusOfGyration":8.323,
 "principalMoments":[6.23,8.185,54.854],"flatnessRatio":0.1136,"inPlaneSymmetry":0.1492,
 "radialHeadShells":2,"transverseHeadShells":0,"cavityVolume":0.625,
 "encapsulatedWater":{"encapsulatedCount":0,"totalWater":13676,
   "bulkWaterDensity":0.08705837418040614,"encapsulationThresholdCount":32.28695617544083,
   "closed":false,"seedDistFromCentre":46.58299971287844,
   "theoreticalMaxDist":46.76537180435968,"totalEmptyCells":1256725,"unreachedCells":5}}
```

| quantity | measured | threshold | verdict |
|---|---|---|---|
| amphiphiles in largest aggregate | **108** (154 at seed 23) | 960 for closure; 13 to qualify | **8.9× short — BINDING** |
| amphiphile SUPPLY | **1600** | 960 | **1.67× OVER — no longer binding** |
| amphiphile share in qualifying aggregates | 0.953 | ≥ 0.5 (micelles) | passes |
| qualifying aggregates | 39 | ≥ 2 (micelles) | passes |
| flatness λ₀/λ₂ | 0.1136 (0.3466 at step 210 000) | ≤ 0.35 | passes |
| in-plane symmetry λ₁/λ₂ | **0.1492** (0.5634 at step 210 000) | ≥ 0.5 | fails on the largest; **passes on the bilayer-qualifying aggregate** |
| **radial head shells** | **2** | **== 2** | **PASSES — evaluable for the first time** |
| transverse head shells | 0 (2 at step 210 000) | 2 for a flat bilayer | passes at step 210 000 |
| enclosed cavity (vacuum cells) | 0.625 σ³ | > 370.8656 σ³ | fails, 593× below |
| box-wide enclosed volume (diagnostic) | 8.75 σ³ | > 370.8656 σ³ | fails, 42× below |
| **encapsulated water** | **0 beads** | **≥ 32.29** (from the run's own bulk water density 0.08706 σ⁻³) | **fails — `closed: false`** |
| flat-disc-equivalent radius | 4.448 σ | R_c = 29.37 σ | 15.1 % of R_c — closure energetics never reached |
| Rg against periodic trust limit | 8.323 σ | box/2 = 27 σ | **3.24× margin, 0 % of beads excluded** |

**Three integrity checks on the closure measurement, all passed**, so "not closed" is a measurement
and not a failure to measure:
- the flood seed sits at 46.583 σ from the aggregate's circular-mean centre against a theoretical
  maximum of 46.765 σ — a genuine far-corner "outside" cell;
- the flood reached all but **5** of 1 256 725 empty cells;
- `encapsulatedWater` is neither `undefined` (no water supplied) nor `null` (untrusted centre) — it is
  a real reading, and this is the FIRST run in this project where the vesicle criterion's *head-shell*
  clause was also evaluable (`continuous-run-report.md` concern §9.6).

### 5.5 Verdict and binding constraint

### VESICLE NOT REACHED. Stage reached: `bilayer` (first time; the predecessor stopped at `amphiphiles`).

**Binding constraint, numbered: AGGREGATE SIZE — the largest aggregate holds 108 amphiphiles (154 at
seed 23) where closure needs ~960. Shortfall 8.9× (6.2× at seed 23).** Down from 80× and it is the
same constraint, not a new one.

Four candidates, each with the number that settles it:

| candidate | measured | needed | shortfall | binding? |
|---|---|---|---|---|
| **aggregate size** | **108** amphiphiles in one aggregate | ~960 | **8.9×** | **YES** |
| amphiphile supply | **1600**, saturated | ~960 | none (1.67× over) | **no — settled, first time** |
| closure energetics | flat-disc-equivalent R = 4.448 σ | R_c = 29.37 σ | 6.6× | no — regime never entered |
| reachable time | steady state, see below | — | — | no |

**Why size is binding and time is not.** The run is in a measured steady state:
- the largest aggregate is **105–113 amphiphiles from step 30 000 to step 300 000** — 270 000 steps
  with no growth at all (Rg, flatness and in-plane symmetry identical to 4 decimals across steps
  120 000–270 000 at the 105-amphiphile state);
- aggregate count flat at **45–56** over the same 270 000 steps; qualifying count flat at 38–41;
- the chemistry is **frozen, not slow**: `cc_bond`, `co_bond`, `co_break` and `amphiphileCount` are
  identical to the digit from step 120 000 to 300 000 (Δ = 0 over 180 000 steps).

**Why they do not coalesce.** ~47 aggregates in 157 464 σ³ → mean spacing `(V/47)^{1/3}` = **14.96 σ**.
A 436-bead cluster at Γ = 1/τ, kT = 1.1 has D = kT/(Γ·436) = 2.523e-3 σ²/τ, so over this run's 3000 τ
it drifts `√(6Dt)` = **6.74 σ** — under half the spacing. As in the predecessor, the run sits below
its FIRST coalescence generation, and ~4.5 halvings would be needed to reach one dominant object.

**What this settles that the predecessor could not, and why the density overshoot strengthens it.**
The predecessor bracketed the transition between "molecularly dispersed" (1.02e-3 σ⁻³, mean aggregate
1.81) and "percolating lamellar" (6.82e-3 σ⁻³, 96.3 % of amphiphiles in ONE box-spanning object) and
named the un-sampled interval between them as the open question. This run lands at **1.016e-2 σ⁻³ —
1.49× ABOVE the density that percolated with the old long tails — and does not percolate.** Instead it
gives ~50 finite objects of 53–108 amphiphiles each, holding 95–98 % of the material, stable for
270 000 steps. So the short-tail composition changed the *phase*, not just the length: the
percolating-lamellar regime was replaced by a finite-aggregate dispersion, at a density where the old
composition percolated. That is a real, new measured point — and it is a *stronger* negative than a
run at exactly 6.82e-3 would have been, because the extra 49 % of material did not help.

---

## 6. Why the chemistry freezes — measured, and it is NOT reagent exhaustion

The exact freeze (Δevents = 0 over 180 000 steps, on 1021 catalysts) was not taken on trust. The
checkpoint carries the buffer that gates every bond event — `soup/wgsl/bond-adsorption.wgsl`'s
`centerLink`, a catalyst's currently-held chain tip — so it was read directly
(`verify/center-occupancy.ts`, new, pure CPU):

```
$ nice -n 15 npx tsx verify/center-occupancy.ts data/checkpoints/soup2ves54-step30000.json data/checkpoints/soup2ves54-step120000.json data/checkpoints/soup2ves54-step300000.json data/checkpoints/soup2ves90-step184001.json
CENTERS data/checkpoints/soup2ves54-step30000.json  step=30000  catalysts=1021 linked=176 (17.2%) carbonsLinked=176 freeCarbonMonomers=6925  freeHeads=58441 desorb={"0":0,"1":0}    heldSteps[min,med,max]=[20,1640,17340]
CENTERS data/checkpoints/soup2ves54-step120000.json step=120000 catalysts=1021 linked=22  (2.2%)  carbonsLinked=22  freeCarbonMonomers=5125  freeHeads=58387 desorb={"0":0,"1":689}  heldSteps[min,med,max]=[5680,19780,26120]
CENTERS data/checkpoints/soup2ves54-step300000.json step=300000 catalysts=1021 linked=0   (0.0%)  carbonsLinked=0   freeCarbonMonomers=5125  freeHeads=58387 desorb={"0":0,"1":711}  heldSteps[min,med,max]=[-1,-1,undefined]
CENTERS data/checkpoints/soup2ves90-step184001.json step=184001 catalysts=2700 linked=15  (0.6%)  carbonsLinked=15  freeCarbonMonomers=24878 freeHeads=12632 desorb={"0":0,"1":3099} heldSteps[min,med,max]=[3460,22460,29020]
```

At step 300 000 **zero of 1021 catalysts hold anything**, while **5125 free carbon monomers and 58 387
free heads** are still in the box. Every centre released by `desorbTimeout` (711 timeouts,
`adsorption.maxHoldSteps = 30000`) and **not one re-nucleated**. The same signature is present in the
predecessor run (0.6 % linked, 24 878 free carbons, 3099 timeouts). So the amphiphile plateau is
**catalytic-centre starvation, not reagent exhaustion** — the propagation channel dies because
nucleation (two bare carbons + a free catalyst, all three in contact) stops occurring, not because
there is nothing left to react.

This does NOT change the verdict — supply is not the binding constraint here (1600 ≥ 960) — but it is
this run's own nomination for the next defect, in the same shape as §9.3 of the predecessor was for
this one, and it is a diagnosis with a buffer behind it rather than a hypothesis.

---

## 7. Invariants — all held, every checkpoint

Asserted by `expect(...)` inside `tests/continuous-run-audit.test.ts` (which fails the test on
violation), over all 12 box-54 checkpoints and both seed-23 checkpoints, plus by the sweep harness at
every one of its 40 samples.

| invariant | how checked | result |
|---|---|---|
| no non-finite state | every position and velocity component of all 106 726 particles | `nonFinitePositions = 0`, `nonFiniteVelocities = 0` at **every** checkpoint; sweep: `nonFinite = 0` at every sample except the deliberately-reported ρ = 0.844 divergence (§2.5) |
| no Verlet overflow | `assertVerletSafety` throws inside the engine on overflow or on drift beyond the skin | no throw in 9 campaign invocations / 345 001 steps |
| monomer conservation | census recomputed from the species slot of every particle, compared to the checkpoint's OWN `config.start` | `{C:15338, O:61353, H:15338, M:1021, W:13676}` exactly, at every checkpoint |
| charge conservation | `invariants().charge` | 0 at every sample |
| valence: carbon ≤ 2 C–C | rebuilt from raw `bondSlots` rows | 0 violations, every checkpoint |
| valence: carbon ≤ 1 C–O | same | 0 violations |
| valence: carbon degree ≤ 3 | same | 0 violations |
| valence: head ≤ `headPlacement.chainCapacity` (= 2, read from the file) | same | 0 violations (1057 heads legitimately carry two tails at step 300 000) |
| placement: `headPlacement.terminalOnly` | same | 0 violations |
| nothing but carbon/head bonded (H, M, W never) | same | 0 violations |
| catalyst gating: `cc_bond` requires a catalytic centre | `tests/soup-bonds.test.ts` (§7.2) | passes |
| Verlet ceiling checked BEFORE running | N = 106 726 against hard max 429 496; list 1.067 GB against 4 294 967 292 B | 4.0× margin |

### 7.1 The two gate tests, after — unchanged, both pass

```
$ nice -n 15 npx vitest run tests/gate6-bilayer.test.ts
CONVERGE start 1.55 (first sample 1.538)  moves 400  tail mean 1.2163 min 1.1783 max 1.2477  lnA drift t=-1.40 (-4.15e-5/move)  accepted 0.465  clusterFraction 1.0000  peaksOk true  checkpoints 2
CONVERGE start 0.9 (first sample 0.905)  moves 400  tail mean 1.2045 min 1.1528 max 1.2447  lnA drift t=-1.65 (-1.04e-4/move)  accepted 0.469  clusterFraction 0.9975  peaksOk true  checkpoints 2
 ✓ tests/gate6-bilayer.test.ts (2 tests) 52770ms
   ✓ готовый бислой при нулевом натяжении держит площадь и толщину из литературы 27151ms
   ✓ площадь сходится в литературный коридор и из слишком большого, и из слишком малого бокса 25529ms
 Test Files  1 passed (1)
      Tests  2 passed (2)
```

```
$ nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=300 areaPerLipid(MEASURED, tail mean)=1.2019 [min 1.1955, max 1.2072, corridor 1.1-1.5] driftPerChunk(lnA)=4.637e-6 t=0.95 thickness(measured)=4.5911 [corridor 4-6] clusterFraction=0.9974 waterInCore=32/4500 headBuriedFraction=0.0599 acceptedFraction=0.1290 of 3000 trials throughput=1231.14 steps/s at N=5700 verdict=passed
 ✓ tests/water-bilayer-area-move.test.ts (1 test) 123560ms
   ✓ бислойная заплатка в явной воде: площадь на липид ИЗМЕРЕНА при нулевом натяжении 123467ms
 Test Files  1 passed (1)
      Tests  1 passed (1)
```

Both inside their corridors and inside the run-to-run scatter `hydrophobic-asymmetry-report.md`
measured over n = 5 (area 1.1211–1.2215, thickness 4.316–5.103): area **1.2019**, thickness
**4.5911**. No threshold, corridor or parameter was touched by this task.

### 7.2 Rule-level invariants, one invocation each

```
$ nice -n 15 npx vitest run tests/soup-valence.test.ts
 ✓ ни один атом углерода не превышает 2 связей C-C и 1 связь C-O; ни одна голова не превышает configured chainCapacity связей, за 45000 шагов 54713ms
 Test Files  1 passed (1)      Tests  1 passed (1)

$ nice -n 15 npx vitest run tests/soup-bonds.test.ts -t 'каталитическом'
 ✓ связи образуются только на каталитическом центре там, где правило это требует 130404ms
 Test Files  1 passed (1)      Tests  1 passed | 2 skipped (3)
```

### 7.3 Regression: the audit's generalisation changed no measured number

`tests/continuous-run-audit.test.ts` was generalised (env-selectable prefix/artifact; monomer census
read from each checkpoint's own `config.start` instead of a `{C:40500,...}` literal — a stricter check,
not a looser one; per-tail and α fields ADDED). Re-run on the predecessor's own checkpoints it
reproduces its published numbers exactly:

```
$ CONTINUOUS_RUN_CHECKPOINTS="data/checkpoints/trace/soup2ves90-step1.json data/checkpoints/soup2ves90-step184001.json" CONTINUOUS_RUN_ARTIFACT=/tmp/regress90.json nice -n 15 npx vitest run tests/continuous-run-audit.test.ts
 Test Files  1 passed (1)
      Tests  1 passed (1)
1      monomers     amph 0   frac 0       meanTail 0     perTail 0     2tail 0   aEv None   asfM None   aggs 0
184001 amphiphiles  amph 742 frac 0.09785 meanTail 5.341 perTail 4.597 2tail 120 aEv 0.9235 asfM 13.079 aggs 409
  largest: {'amphiphileCount': 12, 'radiusOfGyration': 5.835, 'flatnessRatio': 0.0954, 'inPlaneSymmetry': 0.4781, 'radialHeadShells': 'unavailable', 'transverseHeadShells': 0, 'cavityVolume': 0}
```

742 / 0.09785 / 5.341 / 409 / 12 / 0.0954 / 0.4781 — identical to `continuous-run-report.md` §4.1 and
§5, and the recovered α_ev **0.9235** is the 0.923 the brief quotes.

`tests/soup-vesicle.test.ts` was **never** run. The full suite was **never** run. The dev server on
:5199 (user-owned `node`, PID 94131) was neither started, stopped nor inspected.

---

## 8. Concerns

1. **Water is now a token, not a solvent.** W = 13 676 gives a water sub-density of **0.0871 σ⁻³**
   against the 0.8 σ⁻³ this project measured as its own liquid threshold — 4.5× worse than the
   continuous run's 0.396, and it is a direct cost of the lever: at O:C = 4 the organic budget leaves
   no room at a stable total density. The encapsulated-water test self-calibrates (its threshold fell
   to 32.29 beads with the density), so the closure verdict is still a real reading — but the medium
   is further than ever from a liquid, and any statement about a *water*-filled vesicle in this run is
   weaker than the arithmetic makes it look.
2. **A divergence that did not throw.** At ρ_tot = 0.8444 the run produced 68 049 non-finite position
   components and zero bonds while `assertVerletSafety` stayed silent (§2.5). It was caught only by
   the sweep harness's own non-finite scan. A silent NaN state that looks like "stage=monomers, 0
   amphiphiles" is exactly the failure mode that reads as a clean negative; the engine should throw.
3. **The chemistry freeze is a defect, not a plateau** (§6): 0 of 1021 catalysts linked with 5125 free
   carbons and 58 387 free heads present, 711 desorption timeouts and zero re-nucleations. It caps the
   amphiphile supply at ~1600 out of a possible ~5000 and it is the next thing to attack. It does not
   affect this verdict (supply is not binding) but it does mean the run never tested what this
   composition could produce with a working catalytic cycle.
4. **The window's density was overshot by 1.49×** (1.016e-2 against the specified 6.82e-3, §5.1),
   because the box-54 amphiphile yield (0.1043) came out 38 % above the box-30 calibration (0.0757)
   that the arithmetic used. The direction is favourable to the hypothesis (more material, more
   supply, and the percolation the predecessor feared did not happen) so it cannot have manufactured
   the negative — but it is a miss against the specified window, and a run at exactly 6.82e-3 (about
   30 % less carbon) was not performed.
5. **The 960 floor is now conservative in an unquantified way.** It was derived from a bilayer
   thickness of 4.34–4.76 σ measured with the SHIPPED (≈ 4.6-bead) tails. With 2.5-bead tails the wall
   is thinner, R_mid smaller, and the floor LOWER — i.e. the true shortfall is smaller than 8.9×. The
   threshold was deliberately NOT recomputed (that would be moving a threshold mid-experiment); the
   direction is recorded instead. The bilayer thickness at the new tail length was not measured.
6. **One seed at full length.** Seed 19 ran 300 000 steps; seed 23 only 45 000, and its largest
   aggregate was 39 % bigger at the same step (154 vs 111). The steady-state numbers are therefore
   single-seed, with a ±20 % scatter estimate from one comparison point rather than an n ≥ 3 spread.
7. **Two of the three α's disagree by 38 %** (§2.3): event-ratio 0.6215 vs histogram slope 0.4514. The
   two structural reasons are identified and neither was fitted away, but it means "α" in this project
   is not one number, and any future use must say which one it means.
8. **The `bilayer` stage oscillates.** It reads `bilayer` at 8 of 11 post-15 000 samples and `micelles`
   at 3, because the criterion asks whether SOME aggregate qualifies and the qualifying one is not
   always present at a sample. The stage label is therefore genuinely intermittent here, not a
   monotone ladder — worth knowing before anyone quotes "reached bilayer" as a settled state.
9. **The sweep's O:C axis is confounded with density by construction** (§2.2), and de-confounded only
   at its two ends by the controls in §2.4, not at every point. The monotone trend in the middle rows
   carries that confound; the two controls bound its size at ≤ 0.5 beads.
10. **`clay: false`, still not by choice** — `soup/cli/campaign.ts` line 264's own pin. Unchanged from
    the predecessor, same direction of bias (clay measures as −23.8 % chain growth and nucleates
    nothing), so it cannot have manufactured this negative. Reported again because the shipped
    configuration has clay ON and this run does not test it.

---

## 9. Resources, wall time, housekeeping

- **Total steps: 2 615 001** — 2 270 000 across 13 sweep/control/calibration runs and 345 001 in the
  window campaign (300 000 seed 19 + 45 000 seed 23 + 1 start-proof), plus the gate/invariant tests'
  own internal steps.
- **Total wall time: ≈ 7 800 s = 2.17 h.** Sweep and controls 3 655 s over 6 invocations; window
  campaign 3 580 s over 9 invocations; gate/invariant tests 363 s over 2; CPU audits and the
  occupancy diagnostic ≈ 205 s over 4.
- **Longest single foreground invocation: 570 s**, under the 600 s cap. Every invocation was
  `nice -n 15`, ONE at a time, foreground. One invocation exceeded the harness's own default 120 s
  Bash timeout and was moved to the background by the tool rather than by me; it was a single
  `nice -n 15` process and it was waited on to completion before anything else started, and every
  subsequent call carried an explicit timeout. No two compute processes ever ran at once.
- **Compute chunks used: 24 of the 30 allowed** — 20 GPU-bearing invocations and 4 CPU-only
  (three checkpoint audits, one occupancy diagnostic).
- **Checkpoints: 23 files**, ~8.5 MB each, `data/checkpoints/` (gitignored) with the start-proof under
  `data/checkpoints/trace54/`.
- **No orphan processes remain.** `pkill -f puppeteer_dev_chrome_profile` was run after **every**
  browser-bearing invocation and `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** every
  single time, including after the last one. No `rm -rf node_modules/.vite/deps_temp_*` was needed: no
  vitest worker died mid-run. No dev server was started or killed. `--dump-dom` was never used, and no
  particle array was ever transferred as JSON numbers — the sweep harness computes inside the page and
  returns scalars, and the campaign's base64 checkpoint did every state transfer (≤ 0.3 s each).

## 10. Files

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/tail-length-and-window-report.md` — this report.
  `.gitignore` line 6 ignores `.superpowers/`, so like its predecessor it is committed with
  `git add -f`; the deviation is deliberate and flagged rather than silently introduced.
- `tests/tail-length-oc-sweep.test.ts` (new) — the O:C sweep harness. Skips when `SWEEP_RUNS` is
  unset, so a fresh clone never spends GPU time on it. Asserts non-finite, monomer conservation and
  charge at every sample.
- `tests/continuous-run-audit.test.ts` (modified) — generalised as described in §7.3; regression-proved
  bit-for-bit against the predecessor's published numbers.
- `verify/center-occupancy.ts` (new) — the pure-CPU catalyst-occupancy diagnostic behind §6.
- `verify/out/tail-length-oc-sweep.jsonl` (new) — every sweep sample, machine-readable.
- `verify/out/window-run-54-trace.json`, `verify/out/window-run-54-seed23-trace.json` (new) — the
  window run's full audit records, so §5's tables are re-derivable without the (gitignored)
  checkpoints.
- **No source file, data file, threshold, corridor, potential, rate or recogniser was modified.**
  `git diff --stat` over `soup/ engine/ chem/ viewer/ data/` is empty; `soup/wgsl/step.wgsl` stays at
  596 lines and needed no addition, because the lever was a composition argument on the CLI, not code.
