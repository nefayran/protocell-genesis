# Loud failure, and real liquid water — report

Task `loud-failure-and-liquid-water` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `catalyst-turnover-and-window-report.md` (its concern 3 nominated the silent divergence,
its concern 2 nominated the water gap, its §4.3 named cold-start initialisation as "the one change
that would let water be water here"). `data/params.json` was NOT touched (`git diff --stat data/params.json`
empty). No threshold, corridor, potential, rate, recogniser or rank-A constant was modified. One
`data/soup.json` section was ADDED (`coldStartRelax`, rank D, 5 lines, its own measured basis); nothing
existing in that file changed.

**Verdict up front, six lines.**

1. **A non-finite state now stops the run, and the guard caught the real historical case.** The check
   is an O(N) IEEE-754 exponent scan (`soup/wgsl/health.wgsl`) read back once per step()-chunk at the
   SAME sync point `assertVerletSafety` already uses, checked BEFORE it. Measured cost: **0.212 ms
   against a 331.68 ms 1000-step chunk = 0.064 % overhead** (throughput factor 1.00064×). Re-run at
   the predecessor's own silently-divergent composition (ρ_tot = 0.75, box 30, the exact `--start` of
   its §4.2), it **threw at step 1000 naming 6957 non-finite position components of 60 750**.
2. **The mechanism of the silence is now named and it was not "nobody looked".** `assertVerletSafety`
   tests `sqrt(maxDriftSq) > skin/2`; every comparison against NaN is false in IEEE-754 and in JS, so
   the moment positions go NaN the drift reads NaN and that guard passes **forever**. It ran, and was
   silent, in all three historical divergences.
3. **A SECOND silent failure was found on the way, and it is worse than the first.**
   `bondRngBuf`/`thermoRngBuf` were allocated without `COPY_SRC`, which makes `readBack`'s copy a
   validation error, which invalidates the command buffer, which makes the submit a no-op — and then
   the freshly-created (zero-initialised) staging buffer is returned as if it were the data. **Every
   one of the 72 checkpoint files on disk carries `bondRngB64`/`thermoRngB64` that are 100 % zero
   bytes** (verified: 81 000 and 426 904 zero bytes in two checkpoints from different campaigns), so
   every resumed run in this project's history was handed a per-particle RNG seed of 0 for **all** N
   particles — identical Langevin noise on every particle instead of independent noise. Fixed (two
   flags), and `readBack` now THROWS on a buffer without `COPY_SRC` so the whole failure class is loud.
4. **Relaxation: displacement-capped steepest descent before step 1, OPT-IN, and measurement-neutral
   by construction.** `createSoup` never calls it, so every existing caller, test, gate and checkpoint
   lineage is bit-identical to before it existed. It refuses to run at `globalStep != 0` (checked, not
   promised). Velocities, both RNG streams, the bond graph, every event counter, the census and the
   charge are **bitwise unchanged** across it (pinned by a test); no potential, dt or soft core exists
   after it returns, so there is nothing that could still be "on" during a measured step.
5. **LIQUID WATER IS REACHED. The gap to 0.8 σ⁻³ is ZERO.** Water at the project's own measured liquid
   threshold **0.800 σ⁻³** alongside the full organic pool (ρ_org = 0.41793, i.e. ρ_tot = **1.21793
   σ⁻³** — the very number the predecessor called "1.62× above the highest density this engine has
   ever survived") runs **20 000 steps with nonFinite = 0**, and throws loudly without the stage.
   Measured stable up to ρ_tot **1.61793** at 3000 steps. Density is no longer the binding constraint.
6. **AND LIQUID WATER MAKES THE CHEMISTRY MARKEDLY WORSE — that is this task's main finding.** At
   identical composition, seed set and step count (n = 3 per arm, 20 000 steps), water at 0.8 σ⁻³
   against water at 0.26 σ⁻³: amphiphile yield **0.0798 → 0.1468 (1.84× worse)**, `cc_bond` **302 →
   650 (2.15× fewer)**, `co_bond` **265 → 439**, largest aggregate **14.3 → 91.0 (6.4× smaller, ranges
   9–20 vs 69–129, no overlap)**, stage `amphiphiles` vs `bilayer`/`micelles`. An inert-crowder control
   at the SAME total density **recovers the yield (0.1396)** and gives the LARGEST aggregates of all
   (220.7) — so the collapse is caused by **water specifically**, not by crowding. A large part of this
   project's chemistry story was set by an under-dense solvent, in the favourable direction.

---

## 1. Defect 1 — where the non-finite check lives, its cadence, its cost, and the proof it fires

### 1.1 Where, and why there

| | |
|---|---|
| kernel | `soup/wgsl/health.wgsl` — `soup_reset_nonfinite_main` (workgroup 1) + `soup_scan_nonfinite_main` (workgroup 64), a new file because `soup/wgsl/step.wgsl` stands at **596** lines against CLAUDE.md's hard 600 and the rule is "split first, then add" |
| host side | `soup/src/soup-health.ts` — `scanNonFinite()` (pure measurement) and `assertStateFinite()` (throws) |
| call sites | `soup/src/soup-integrate.ts`'s stepper, immediately after `onSubmittedWorkDone()` and **before** `assertVerletSafety`; plus both of `soup/src/soup-box-scale.ts`'s existing `assertVerletSafety` sites |
| cadence | **once per step()-chunk** = `STEP_CHUNK` = 1000 steps, or `n` when `step(n)` is called with `n < 1000` |
| test detects | Inf **and** NaN, via the IEEE-754 exponent field (`(bitcast<u32>(x) >> 23) & 0xFF == 0xFF`), not `x != x` — the self-inequality trick is exactly what a fast-math-style optimisation may fold to `false`; the exponent field is not an optimisation target |
| unit reported | non-finite **components** (x,y,z), separately for positions and velocities — the same unit the three historical offline scans reported (60 495 / 366 282 / 68 049), so the guard's message is directly comparable with what is already published |

This is an EXTENSION of the existing machinery, as instructed, not a parallel one: same cadence, same
sync point, same throw-and-stop discipline, and it is deliberately checked FIRST because of §1.2.

`soup/src/soup-grid-verlet.ts` additionally gained an explicit `Number.isFinite(drift)` branch. It
should now be unreachable (finiteness is checked before it at every call site); it exists so that if
it ever IS reached the failure is loud rather than a false pass.

### 1.2 Why it was silent for three runs — the mechanism, not a guess

```
assertVerletSafety:   const drift = Math.sqrt(Math.max(0, rawDrift[0]))
                      if (drift > bound + 1e-6) throw ...
```

`Math.max(0, NaN)` is `NaN`; `Math.sqrt(NaN)` is `NaN`; `NaN > anything` is **false**. Once one
position is NaN, `soup_max_drift_main`'s `dot(d,d)` is NaN, the atomic max carries the NaN bit
pattern, and this guard reports a clean run for the rest of the campaign. It was running, every
chunk, through all three divergences. That is the whole reason a blown-up run was indistinguishable
from `stage=monomers, 0 amphiphiles` — a clean negative.

### 1.3 Measured cost of the check — unedited output

```
$ nice -n 15 npx vitest run tests/soup-nonfinite-guard.test.ts
NONFINITE-GUARD N=1728 healthy: at_start={"pos":0,"vel":0} after_2000={"pos":0,"vel":0} after_timing={"pos":0,"vel":0}
NONFINITE-GUARD-COST scan=0.2120ms (n=50) chunk1000=331.68ms (n=5) overhead=0.064% of a 1000-step chunk => throughput cost factor 1.00064x
NONFINITE-GUARD-FIRES steps_reached=1 at_start={"pos":0,"vel":0} after={"pos":120,"vel":120}
  message: non-finite state at step=1: non-finite components of positions=120, velocities=120 out of 5184 (appeared within step interval 0..1) -- the calculation diverged (Inf/NaN), any further numbers from this run are meaningless

 ✓ tests/soup-nonfinite-guard.test.ts (1 test) 2904ms
 Test Files  1 passed (1)
      Tests  1 passed (1)
```

**It is not free, and the number is what justifies the cadence.** 0.2120 ms per scan against 331.68 ms
per 1000-step chunk = **0.064 %**. Two components of that 0.2 ms do NOT scale with N (one submit, one
8-byte readback round trip) and one does (the O(N) streaming read of two `vec4` arrays), so at the
particle counts that matter the fraction only falls: this measurement is at N = 1728, and the same
scan on the box-30 broth (N = 32 884, 19× larger) is amortised over a chunk that is itself ~5× more
expensive. Per-STEP checking would have cost 1000× this, i.e. ~64 % — which is exactly why the cadence
is per-chunk and not per-step. The three historical divergences all appeared within the FIRST 1000
steps (§3.2), so a per-chunk cadence loses nothing in detection latency for the failure class that
actually occurs here.

### 1.4 Proof it fires — on the deliberately diverging fixture, and on the real historical case

The fixture is a real WCA blow-up, not an injected NaN: twenty pairs of water beads placed 1e-4 σ
apart on an otherwise clean lattice. `wca_dv` returns 0 at exactly r = 0 and is
finite-but-astronomical just off it — (b/r)^12 at 1e-4 is 1e48, which overflows f32 to Inf — so this is
the same physics as the cold-start overlap that produced every historical divergence, concentrated
into one step. `step(1)` puts the guard exactly one step in, which is the "early" half of "loud and
early". The healthy control (`bad = 0`, everything else identical) reads `{pos:0, vel:0}` at start,
after 2000 steps, and after the 5000 timing steps: **the guard does not fire on a good run.**

And on the predecessor's own silently-divergent composition, verbatim from §4.2 of that report:

```
$ nice -n 15 npx tsx scratch-ladder.ts '[{"label":"broth-0.75-norelax","box":30,"start":{"C":1860,"O":7440,"H":1860,"M":124,"W":8966},"steps":3000,"relax":false}]'
ARM broth-0.75-norelax box=30 N=20250 rho_tot=0.75000 relax=false steps=3000 wall=1518ms
    ok=false THREW at steps=1000: non-finite state at step=1000: non-finite components of positions=6957, velocities=6957 out of 60750 (appeared within step interval 0..1000) -- the calculation diverged (Inf/NaN), any further numbers from this run are meaningless
    maxForceStart=3.4360e+3 nonFiniteForceStart=0
```

The run that previously printed `[campaign] step=20000/20000 stage=monomers aggregates=0 largest=0`
and was only caught by an offline scan **now stops at step 1000 with the count in the message.**

### 1.5 The second silent failure, found by the new test's own invariant check

Writing the relaxation-invariants test (§2.4) meant asserting "the RNG streams are unchanged", which
meant calling `sys.rngState()`, which produced this:

```
warn: [Buffer (unlabeled)] usage (BufferUsage::(CopyDst|Storage)) doesn't include BufferUsage::CopySrc.
 - While validating source [Buffer (unlabeled)] usage.
 - While encoding [CommandEncoder (unlabeled)].CopyBufferToBuffer([Buffer (unlabeled)], 0, [Buffer (unlabeled)], 0, 38976).
warn: [Invalid CommandBuffer] is invalid due to a previous error.
```

38976 = N·4 at N = 9744, i.e. `bondRngBuf`/`thermoRngBuf`. Both were created
`STORAGE | COPY_DST`, without `COPY_SRC`. The consequence is not a thrown error, it is **zeros**: the
copy is a validation error, the command buffer is invalidated, the submit is dropped, and `readBack`
maps and returns the freshly-created (therefore zero-initialised) staging buffer. Verified on disk,
unedited:

```
$ python3 ... data/checkpoints/rho075/rho075-step20000.json  data/checkpoints/soup2ves54-step120000.json
=== data/checkpoints/rho075/rho075-step20000.json N= 20250 step= 20000
    bondRngB64 bytes 81000 nonzero 0 first16 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    thermoRngB64 bytes 81000 nonzero 0 first16 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    velocitiesB64 bytes 324000 nonzero 122010 first16 [0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 192, 127, 0, 0, 0, 0]
=== data/checkpoints/soup2ves54-step120000.json N= 106726 step= 120000
    bondRngB64 bytes 426904 nonzero 0 first16 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    thermoRngB64 bytes 426904 nonzero 0 first16 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    velocitiesB64 bytes 1707616 nonzero 1279590 first16 [71, 150, 145, 191, 147, 216, 4, 63, 1, 198, 134, 64, 0, 0, 0, 0]
```

**Every RNG byte in every checkpoint this project has written is zero**, across two unrelated
campaigns. `decodeCheckpointResume` feeds them straight back, so after every resume `intRng[i] = 0`
for all i, `pcgSoup(0)` is one number, and **every particle receives the same Langevin noise vector** —
a coherent whole-system kick instead of thermal noise, plus the same shared random stream for the bond
Monte Carlo. The predecessor's window campaign resumed 8 times. (As a bonus the first dump also shows
`velocitiesB64` starting `0,0,192,127` = 0x7FC00000 = **NaN**, i.e. the ρ_tot = 0.75 divergence is
visible in the file itself.)

Two fixes, both in this commit:
- `soup/src/soup-buffers.ts`: both buffers now carry `COPY_SRC`. These were the only readable buffers
  in that file missing it (`verletCountBuf` also lacks it but is never read back).
- `engine/src/gpu.ts`: `readBack` now **throws** if `(src.usage & COPY_SRC) === 0`. `usage` is a
  readable attribute of `GPUBuffer`, so converting this entire failure class from silent zeros into a
  thrown error costs one bitwise test. This is the same "make it loud" instruction applied to the
  defect the instruction itself uncovered.

`tests/soup-checkpoint.test.ts` passes 3/3 after the fix (§6), i.e. the round-trip and the
"checkpointing does not change what a run computes next" check both hold with real RNG state restored.

---

## 2. Defect 2 — the relaxation mechanism, why it was chosen, and the proof it is measurement-neutral

### 2.1 The diagnosis, restated as arithmetic

`soup/src/soup-init-state.ts` lays a cubic lattice with `jitterFrac = 0.15`, so its smallest possible
gap is `(1 − 2·0.15)·spacing = 0.7·spacing`. At ρ_tot = 0.8 σ⁻³, `spacing = 0.8^(−1/3) = 1.077 σ`, so
the minimum gap is **0.754 σ** — against a catalyst–catalyst WCA contact distance of
`σ·(1.2+1.2)/2 = 1.2 σ`. Those pairs start ~40 % deep in a repulsive core, and 1/r¹³ there is a force
no integrator at `data/params.json`'s dt can absorb. Measured `max|F|` at the cold start, before any
step:

| configuration | ρ_tot | max&#124;F&#124; at step 0 |
|---|---|---|
| pure water, box 30 | 0.800 | 1.715e+3 |
| pure water, box 30 | 1.000 | 4.215e+3 |
| full broth, box 30 | 0.750 | 3.436e+3 |
| full broth, box 30 | 1.218 | 3.397e+4 |
| full broth, box 30 | 1.418 | 4.900e+4 |
| full broth, box 30 | 1.618 | 9.843e+4 |

It is **not** a property of water: pure water at 0.8 σ⁻³ starts at 1.7e3 and runs stably with no
relaxation at all (§3.1), because its own lattice is laid at exactly its own spacing. The instability
is specific to a mixture of unlike radii sharing one lattice.

### 2.2 The mechanism chosen, and why the two alternatives were rejected

**Chosen: normalised steepest descent with a displacement cap decaying linearly to zero**
(`soup/wgsl/relax.wgsl` + `soup/src/soup-relax.ts`). Each iteration moves every mobile particle
exactly `RX.x` along its own force direction; `RX.x` decays to 0 over `iterations`. The force is
produced by the SAME `encodeSoupForceList`/`encodeSoupForce` call the real steps use.

Rejected, on the brief's own terms rather than by preference:
- **a soft-core force cap ramped off over the first steps** would have to live inside `nonbondedSoup`,
  i.e. it modifies the potential the whole project's calibration rests on, and it is ON during real
  steps — which the brief forbids for any step a measurement uses. There is no version of it that is
  provably off, only a version that is claimed to be off.
- **a smaller initial dt ramped up** is also inside the trajectory: the first thousands of steps would
  be integrated with a different dt than every published run, and it would still be integrating an
  overlap whose force is 3.4e4.

Minimisation is outside the trajectory entirely, which is the only option that makes §2.4's
invariants provable rather than argued.

Why **normalised** rather than proportional descent: at |F| ~ 3e4 a proportional step would need to be
~1e-5 to be stable and would then need ~1e5 iterations to travel one σ — the stiffness that IS the
problem sets the step size. A normalised step is scale-free (the deeper the overlap, the more exactly
the direction points out of it) and its displacement is bounded by construction. Settling is supplied
by the decay: a normalised descent with a fixed step cannot converge, it orbits the minimum at radius
~step. Two overflow guards are in the kernel: the direction is normalised through the largest
component (so no intermediate can overflow even at |F| → ∞), and a non-finite or zero force leaves the
particle where it is, after which `soup-relax.ts` re-scans and **throws** — an unrelaxable start fails
loudly rather than being smeared into a plausible-looking one.

### 2.3 The two numbers, measured

`data/soup.json` → `coldStartRelax` (rank D, its own basis in the file): `iterations = 200`,
`maxDisplacementSigma = 0.1`. Swept at the hardest configuration (full broth, box 30, water 0.8,
ρ_tot = 1.21793, `max|F|` before = 3.3966e+4), each arm then run 3000 steps:

| iterations | cap (σ) | displacement bound (σ) | max&#124;F&#124; after | reduction | 3000 steps |
|---|---|---|---|---|---|
| 20 | 0.1 | 1.05 | 58.44 | 581× | clean |
| 50 | 0.1 | 2.55 | 35.25 | 964× | clean |
| 100 | 0.1 | 5.05 | 26.88 | 1264× | clean |
| **200** | **0.1** | **10.05** | **22.89** | **1484×** | **clean** |
| 200 | 0.05 | 5.03 | 18.45 | 1841× | clean |
| 200 | 0.2 | 20.1 | 17.22 | 1972× | clean |

Every tested setting works at 3000 steps, including the cheapest. 200/0.1 was chosen because it is
comfortably inside the working region on both axes rather than at an edge of it, it is the setting
confirmed at 20 000 steps (§3.1), and it costs ~1–2 s once per run. The values are protocol numbers:
they cannot change the equilibrium distribution the subsequent Langevin dynamics samples, only the
starting point of the trajectory.

### 2.4 Proof it does not touch measured physics — four independent arguments

**(a) It is opt-in, so the default path is bit-identical.** `createSoup` does not call it. There is no
`CreateSoupOpts` field for it. It is reachable only through `SoupSystem.relaxColdStart()` or
`soup/cli/campaign.ts --relax`. Every existing test, gate, fixture and checkpoint lineage therefore
runs exactly the code it ran before this task — this is a structural statement, not a comparison.

**(b) It cannot be inside a trajectory: checked, not promised.** `relaxColdStart` throws at
`globalStep != 0`. Unedited:

```
COLD-START-RELAX-REFUSAL relaxColdStart: the system is already at step 1 -- minimization is allowed ONLY before the first step, otherwise it would fall inside the trajectory that measurements are computed from
```

**(c) Bitwise invariants, pinned by `tests/soup-cold-start-relax.test.ts`.** Unedited:

```
COLD-START-RELAX N=9744 relax={"iterations":200,"maxDisplacementStart":0.1,"displacementBound":10.05,"maxForceBefore":35180.6015625,"maxForceAfter":14.289834976196289,"nonFiniteBefore":0,"nonFiniteAfter":0} maxDisplacementMeasured=1.11804 steps 0->0 velUnchanged=true bondsUnchanged=true rng(bond/thermo)Unchanged=true/true events {"cc_bond":0,"cc_break":0,"co_bond":0,"co_break":0} -> {"cc_bond":0,"cc_break":0,"co_bond":0,"co_break":0}
```

- step counter `0 -> 0`;
- velocities compared as **u32 bit patterns**, not with a tolerance: unchanged. The Maxwell–Boltzmann
  draw IS the initial temperature, and a float comparison with a tolerance would pass a minimiser that
  quietly rescaled it;
- both per-particle RNG streams unchanged (a consumed stream would desynchronise every subsequent
  random number from what an unrelaxed run would have drawn) — and this assertion is only meaningful
  because of §1.5's fix;
- bond graph unchanged, all four event counters still 0, census and charge unchanged;
- it is **not** a no-op: measured max displacement 1.118 σ, and that is ≤ the analytic bound 10.05 σ
  the decaying cap implies, so nothing was teleported across the box;
- `max|F|` 3.518e4 → 14.29 (2462×), non-finite 0.

**(d) It does not move the chemistry, measured at n = 3.** The token-density arm was run WITH and
WITHOUT relaxation at identical composition/seeds/steps (§4.2): amphiphile yield 0.14677 vs 0.14659
(**0.12 % apart**), `cc_bond` 650.0 vs 654.7 (0.7 %), `co_bond` 438.7 vs 440.3 (0.4 %), mean tail
2.327 vs 2.375 (2.1 %, well inside the arm's own scatter), largest aggregate 91.0 vs 72.7 (each inside
the other's range). This is the strongest available form of the claim, because it tests the
observables the project actually reports rather than a proxy.

**(e) The published gates, re-run.** Both bilayer gates pass, inside the published run-to-run scatter
(§5, §6). No corridor, threshold or parameter was touched.

---

## 3. The highest stable density, with evidence, and the N-vs-ceiling arithmetic

### 3.1 The ladder — every arm unedited, box 30, seed 19, organics C1860/O7440/H1860/M124 (ρ_org = 0.41793)

```
ARM pure-water-0.8-norelax box=30 N=21600 rho_tot=0.80000 relax=false steps=3000 wall=3230ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=3000 maxForceEnd=2.5827e+2 events={"cc_bond":0,"cc_break":0,"co_bond":0,"co_break":0}
    maxForceStart=1.7154e+3 nonFiniteForceStart=0
ARM broth-0.75-norelax box=30 N=20250 rho_tot=0.75000 relax=false steps=3000 wall=1518ms
    ok=false THREW at steps=1000: non-finite state at step=1000: non-finite components of positions=6957, velocities=6957 out of 60750 ...
ARM broth-0.75-relax box=30 N=20250 rho_tot=0.75000 relax=true steps=3000 wall=3788ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=3000 maxForceEnd=1.6920e+2 events={"cc_bond":114,"cc_break":0,"co_bond":73,"co_break":0}
    relax={"iterations":200,"maxDisplacementStart":0.1,"displacementBound":10.05,"maxForceBefore":3436.0205078125,"maxForceAfter":10.697349548339844,"nonFiniteBefore":0,"nonFiniteAfter":0}
ARM broth-rhoW0.5-relax box=30 N=24784 rho_tot=0.91793 relax=true steps=3000 wall=4869ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=3000 maxForceEnd=2.3283e+2 events={"cc_bond":113,"cc_break":0,"co_bond":79,"co_break":0}
    relax={...,"maxForceBefore":13282.310546875,"maxForceAfter":9.233573913574219,...}
ARM broth-rhoW0.6-relax box=30 N=27484 rho_tot=1.01793 relax=true steps=3000 wall=5639ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=3000 maxForceEnd=2.1292e+2 events={"cc_bond":96,"cc_break":0,"co_bond":74,"co_break":0}
    relax={...,"maxForceBefore":16075.1962890625,"maxForceAfter":7.523904800415039,...}
ARM broth-rhoW0.8-relax box=30 N=32884 rho_tot=1.21793 relax=true steps=3000 wall=7913ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=3000 maxForceEnd=2.1314e+2 events={"cc_bond":28,"cc_break":0,"co_bond":26,"co_break":0}
    relax={...,"maxForceBefore":33965.953125,"maxForceAfter":22.885356903076172,...}
ARM broth-rhoW0.8-norelax box=30 N=32884 rho_tot=1.21793 relax=false steps=3000 wall=2214ms
    ok=false THREW at steps=1000: non-finite state at step=1000: non-finite components of positions=98112, velocities=98112 out of 98652 ...
ARM broth-rhoW1.0-relax box=30 N=38284 rho_tot=1.41793 relax=true steps=3000 wall=11215ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=3000 maxForceEnd=3.5378e+2 events={"cc_bond":17,"cc_break":0,"co_bond":17,"co_break":0}
    relax={...,"maxForceBefore":49002.64453125,"maxForceAfter":52.86977767944336,...}
ARM broth-rhoW1.2-relax box=30 N=43684 rho_tot=1.61793 relax=true steps=3000 wall=14030ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=3000 maxForceEnd=4.2338e+2 events={"cc_bond":10,"cc_break":0,"co_bond":10,"co_break":0}
    relax={...,"maxForceBefore":98430.75,"maxForceAfter":97.75430297851562,...}
ARM pure-water-1.0-norelax box=30 N=27000 rho_tot=1.00000 relax=false steps=3000 wall=1576ms
    ok=false THREW at steps=1000: non-finite state at step=1000: non-finite components of positions=80619, velocities=80619 out of 81000 ...
```

The 20 000-step confirmations, unedited:

```
ARM pure-water-0.8-norelax-20k box=30 N=21600 rho_tot=0.80000 relax=false steps=20000 wall=42804ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=20000 maxForceEnd=2.0317e+2 events={"cc_bond":0,"cc_break":0,"co_bond":0,"co_break":0}
    maxForceStart=1.7154e+3 nonFiniteForceStart=0
ARM pure-water-0.8-relax-20k box=30 N=21600 rho_tot=0.80000 relax=true steps=20000 wall=41224ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=20000 maxForceEnd=2.2247e+2 events={"cc_bond":0,"cc_break":0,"co_bond":0,"co_break":0}
    relax={...,"maxForceBefore":1715.4140625,"maxForceAfter":12.433786392211914,...}
ARM broth-rhoW0.8-relax-20k box=30 N=32884 rho_tot=1.21793 relax=true steps=20000 wall=78728ms
    ok=true nonFinite={"pos":0,"vel":0} stepsDone=20000 maxForceEnd=2.5138e+2 events={"cc_bond":273,"cc_break":0,"co_bond":244,"co_break":0}
    relax={...,"maxForceBefore":33965.953125,"maxForceAfter":18.1242733001709,...}
```

### 3.2 The table, and the answer

| ρ_tot σ⁻³ | ρ_W σ⁻³ | N | no relaxation | with relaxation |
|---|---|---|---|---|
| 0.6778 | 0.260 | 18 304 | stable (history + here) | stable |
| 0.750 | 0.332 | 20 250 | **THROWS at step 1000, 6957 non-finite** (silent before this task) | **stable** |
| 0.800 (pure water) | 0.800 | 21 600 | **stable, 20 000 steps, nonFinite = 0** | stable, 20 000 steps |
| 0.9179 | 0.500 | 24 784 | not run | stable |
| 1.000 (pure water) | 1.000 | 27 000 | **THROWS at step 1000, 80 619 non-finite** | not run |
| 1.0179 | 0.600 | 27 484 | not run | stable |
| **1.21793** | **0.800** | **32 884** | **THROWS at step 1000, 98 112 non-finite** | **stable, 20 000 steps, nonFinite = 0** |
| 1.41793 | 1.000 | 38 284 | not run | stable (3000 steps) |
| 1.61793 | 1.200 | 43 684 | not run | stable (3000 steps) |

- **The pure-water requirement is met:** 0.8 σ⁻³, box 30, N = 21 600, **20 000 steps, nonFinite = 0**,
  and it needs no relaxation at all.
- **The full-broth requirement is met, at the density I now defend: ρ_W = 0.800 σ⁻³**, the project's own
  measured liquid threshold, i.e. **the gap to 0.8 is ZERO**. ρ_tot = 1.21793 — the exact figure the
  predecessor computed and called unreachable — confirmed at 20 000 steps with nonFinite = 0.
- **The highest ρ_tot measured stable is 1.61793** (3000 steps). I do **not** defend it as a medium:
  1.618 σ⁻³ is above the close-packed density of unit-diameter spheres (1.414), so it is a
  high-pressure artefact of soft cores, not a liquid. Density is simply no longer the binding
  constraint; the liquid threshold is, and it is reached.
- Every historical divergence appeared in the first 1000 steps, which is why per-chunk cadence loses
  no detection latency here.

### 3.3 N against the measured Verlet ceiling

`verletList.listCapacity = 2500`, flat `N · 2500 · 4` bytes, against this adapter's measured
`maxStorageBufferBindingSize = maxBufferSize = 4 294 967 292 B`, i.e. a hard ceiling of
**429 496 particles**:

```
this task's confirmed config   N =  32 884  ->   32 884 x 2500 x 4 =   328 840 000 B = 0.329 GB   margin 13.06x
                                                 32 884 particles                    vs 429 496   margin 13.06x

the box-54 window config WITH WATER AT 0.8 (now reachable):
  C 10 844 + O 43 376 + H 10 844 + M 722 = 65 786 organics (rho_org = 0.41778)
  W = 0.8 x 157 464                      = 125 971 water
  N = 191 757                             rho_tot = 1.21781 sigma^-3
                                 N = 191 757 x 2500 x 4 = 1 917 570 000 B = 1.918 GB   margin  2.24x
                                   191 757 particles                      vs 429 496   margin  2.24x
```

The Verlet ceiling is **not** what binds, and now neither is density: the box-54 window run at real
liquid water sits at 2.24× under the particle ceiling and 2.24× under the buffer limit. The neighbour
count itself is also fine: `listRange = wcaCutoff(1.2σ) + w_c + skin = 4.447 σ`, so at ρ = 1.618 the
expected candidates per particle are `(4/3)π·4.447³·1.618 ≈ 590` against a capacity of 2500, and
`verletOverflow` stayed clear in every arm above (`assertVerletSafety` threw in none of them).

---

## 4. The chemistry at liquid vs token water density

### 4.1 Design

Box 30, 20 000 steps, organics **identical** in every arm (C 1860, O 7440, H 1860, M 124 — O:C = 4,
the composition the predecessor's §4.3 pinned), seeds **19, 23, 29** in every arm, relaxation applied
in every arm unless stated. Only the water count differs. `yield = amphiphiles / C`, matching the
predecessor's own definition (its 199/2630 = 0.0757).

A third arm exists because the first two differ in TWO things, not one: adding water raises ρ_tot from
0.678 to 1.218, so a difference could be crowding rather than solvation. The **inert-crowder control**
keeps water token (7020) and makes the total density up to exactly 1.21793 with `H` — the donor bead,
which appears in **no** bond rule (`cc_bond`/`cc_break`/`co_bond`/`co_break` are all C–C or C–O), is
non-polar and non-solvent, i.e. a chemically inert excluded-volume filler: `H = 16 440`, N = 32 884,
the same N as the liquid arm.

### 4.2 Every run, unedited

```
CHEM LIQUID-0.8 seed=19 box=30 N=32884 rho_tot=1.21793 relax=true steps=20000 wall=81844ms
    events={"cc_bond":274,"cc_break":0,"co_bond":241,"co_break":0} amph=143 yield=0.07688 meanTail=2.0728 chains=1586 meanChain=1.1728 stage=amphiphiles aggs=47 largest=14 nonFinite={"pos":0,"vel":0}
CHEM LIQUID-0.8 seed=23 box=30 N=32884 rho_tot=1.21793 relax=true steps=20000 wall=76834ms
    events={"cc_bond":320,"cc_break":0,"co_bond":280,"co_break":2} amph=150 yield=0.08065 meanTail=2.1325 chains=1540 meanChain=1.2078 stage=amphiphiles aggs=53 largest=9 nonFinite={"pos":0,"vel":0}
CHEM LIQUID-0.8 seed=29 box=30 N=32884 rho_tot=1.21793 relax=true steps=20000 wall=74764ms
    events={"cc_bond":313,"cc_break":0,"co_bond":273,"co_break":0} amph=152 yield=0.08172 meanTail=2.0915 chains=1547 meanChain=1.2023 stage=amphiphiles aggs=52 largest=20 nonFinite={"pos":0,"vel":0}
CHEM TOKEN-0.26-relax seed=19 box=30 N=18304 rho_tot=0.67793 relax=true steps=20000 wall=34259ms
    events={"cc_bond":661,"cc_break":0,"co_bond":441,"co_break":2} amph=273 yield=0.14677 meanTail=2.3588 chains=1199 meanChain=1.5513 stage=bilayer aggs=6 largest=75 nonFinite={"pos":0,"vel":0}
CHEM TOKEN-0.26-relax seed=23 box=30 N=18304 rho_tot=0.67793 relax=true steps=20000 wall=30540ms
    events={"cc_bond":638,"cc_break":0,"co_bond":437,"co_break":2} amph=271 yield=0.14570 meanTail=2.2879 chains=1222 meanChain=1.5221 stage=micelles aggs=12 largest=69 nonFinite={"pos":0,"vel":0}
CHEM TOKEN-0.26-relax seed=29 box=30 N=18304 rho_tot=0.67793 relax=true steps=20000 wall=30275ms
    events={"cc_bond":651,"cc_break":0,"co_bond":438,"co_break":3} amph=275 yield=0.14785 meanTail=2.3333 chains=1209 meanChain=1.5385 stage=bilayer aggs=9 largest=129 nonFinite={"pos":0,"vel":0}
CHEM TOKEN-0.26-norelax seed=19 box=30 N=18304 rho_tot=0.67793 relax=false steps=20000 wall=33190ms
    events={"cc_bond":662,"cc_break":0,"co_bond":435,"co_break":1} amph=269 yield=0.14462 meanTail=2.4181 chains=1198 meanChain=1.5526 stage=bilayer aggs=9 largest=51 nonFinite={"pos":0,"vel":0}
CHEM TOKEN-0.26-norelax seed=23 box=30 N=18304 rho_tot=0.67793 relax=false steps=20000 wall=30351ms
    events={"cc_bond":643,"cc_break":0,"co_bond":439,"co_break":3} amph=284 yield=0.15269 meanTail=2.3250 chains=1217 meanChain=1.5283 stage=micelles aggs=10 largest=91 nonFinite={"pos":0,"vel":0}
CHEM TOKEN-0.26-norelax seed=29 box=30 N=18304 rho_tot=0.67793 relax=false steps=20000 wall=28815ms
    events={"cc_bond":659,"cc_break":0,"co_bond":447,"co_break":2} amph=265 yield=0.14247 meanTail=2.3807 chains=1201 meanChain=1.5487 stage=micelles aggs=7 largest=76 nonFinite={"pos":0,"vel":0}
CHEM CROWDER-H-tokenwater seed=19 box=30 N=32884 rho_tot=1.21793 relax=true steps=20000 wall=82862ms
    events={"cc_bond":470,"cc_break":0,"co_bond":407,"co_break":4} amph=261 yield=0.14032 meanTail=2.0972 chains=1390 meanChain=1.3381 stage=micelles aggs=9 largest=174 nonFinite={"pos":0,"vel":0}
CHEM CROWDER-H-tokenwater seed=23 box=30 N=32884 rho_tot=1.21793 relax=true steps=20000 wall=93311ms
    events={"cc_bond":508,"cc_break":0,"co_bond":417,"co_break":2} amph=265 yield=0.14247 meanTail=2.1360 chains=1352 meanChain=1.3757 stage=amphiphiles aggs=10 largest=246 nonFinite={"pos":0,"vel":0}
CHEM CROWDER-H-tokenwater seed=29 box=30 N=32884 rho_tot=1.21793 relax=true steps=20000 wall=91755ms
    events={"cc_bond":503,"cc_break":0,"co_bond":428,"co_break":1} amph=253 yield=0.13602 meanTail=2.1086 chains=1357 meanChain=1.3707 stage=amphiphiles aggs=10 largest=242 nonFinite={"pos":0,"vel":0}
```

### 4.3 The comparison, with scatter measured rather than assumed (n = 3 per arm)

| quantity | LIQUID ρ_W = 0.80 | TOKEN ρ_W = 0.26 | TOKEN, no relaxation | CROWDER (ρ_W = 0.26, ρ_tot = 1.218) |
|---|---|---|---|---|
| `cc_bond` | **302.3** [274, 320] | **650.0** [638, 661] | 654.7 [643, 662] | 493.7 [470, 508] |
| `co_bond` | **264.7** [241, 280] | **438.7** [437, 441] | 440.3 [435, 447] | 417.3 [407, 428] |
| amphiphiles | **148.3** [143, 152] | **273.0** [271, 275] | 272.7 [265, 284] | 259.7 [253, 265] |
| yield (amph/C) | **0.07975** [0.0769, 0.0817] | **0.14677** [0.1457, 0.1479] | 0.14659 [0.1425, 0.1527] | 0.13960 [0.1360, 0.1425] |
| mean tail (beads) | **2.099** [2.073, 2.133] | **2.327** [2.288, 2.359] | 2.375 [2.325, 2.418] | 2.114 [2.097, 2.136] |
| largest aggregate | **14.3** [9, 20] | **91.0** [69, 129] | 72.7 [51, 91] | 220.7 [174, 246] |
| stage reached | amphiphiles ×3 | bilayer ×2, micelles ×1 | bilayer ×1, micelles ×2 | micelles ×1, amphiphiles ×2 |
| relative scatter (largest agg.) | ±38 % | ±33 % | ±28 % | ±16 % |

**How much of the chemistry story was set by an under-dense solvent: a lot, and favourably.**

- **Yield and chain growth are WATER effects, and water is bad for them.** Liquid water halves the
  amphiphile yield (0.0798 vs 0.1468, 1.84×) and cuts `cc_bond` by 2.15×. The crowder control at the
  SAME total density **recovers** the yield (0.1396, i.e. 95 % of the token arm and 1.75× the liquid
  arm), so this is not crowding: it is water. The mechanism is the one
  `broth-composition-report.md` §5 already proposed and never tested — a polar head, and now also the
  reaction partners, spend their time hydrated rather than at the catalytic tip — except that at O:C = 4
  and with the predecessor's tip fix the reaction is no longer dead, only **suppressed**, and the
  suppression is now a measured factor rather than a hypothesis.
- **Mean tail length is a DENSITY effect, not a water effect.** Liquid 2.099 ≈ crowder 2.114 <<
  token 2.327. Both dense arms shorten tails by the same amount whether the density is water or an
  inert filler. All three remain inside the 2–3 corridor the project reports.
- **Aggregate size is where the damage is worst, and the two effects push opposite ways.** Water
  destroys aggregation (14.3, ranges 9–20 — 6.4× below the token arm, with **no range overlap**),
  while crowding promotes it (220.7, the largest of any arm). The liquid arm never leaves stage
  `amphiphiles` in any of three seeds, where the token arm reaches `bilayer` in two of three. Since
  the predecessor's binding constraint was **aggregate size** (271 amphiphiles against a floor of
  912), liquid water makes that constraint **worse, not better**: this is a negative result for the
  vesicle programme and it is reported as one.
- **Relaxation is not the cause of any of it** (§2.4(d)): the two token arms differ by 0.1–2 %.

No corridor was widened, no rank-A constant re-fitted, `co_bond.attemptRate` untouched, and no arm was
quietly reverted to token water: the token arm is present and labelled as the comparison, not as the
answer.

---

## 5. The bilayer gates at the new water density

**A correction to the brief's premise, stated up front because it changes what "re-measure at the new
density" means.** `tests/water-bilayer-area-move.test.ts` has been running water at **0.8 σ⁻³ all
along** — its `waterDensity` argument is literally `0.8` (line 371). The token-water problem was never
in the bilayer gate; it was confined to the composition-driven soup runs (0.26–0.396 σ⁻³). The
bilayer construction escapes the cold-start blow-up because it lays its own water on its own regular
lattice, skipping the membrane slab, rather than sharing one jittered lattice with a 1.2 σ catalyst
bead. So the gates at "the new, defensible density" ARE the gates at 0.8 σ⁻³, and the honest
re-measurement is to re-run them and report.

```
$ nice -n 15 npx vitest run tests/gate6-bilayer.test.ts tests/water-bilayer-area-move.test.ts
GATE6 area 1.2113 +/- 0.0076 (min 1.1972, max 1.2303)  thickness 4.4817 (per-frame mean 4.4535, sd 0.2696, n=400)  lnA drift/move 6.284e-6 +/- 1.12e-5 (t=0.56)  accepted 0.258  escapedMax 0  box 24.597  steps 83000
CONVERGE start 1.55 (first sample 1.559)  moves 400  tail mean 1.2070 min 1.1758 max 1.2453  lnA drift t=1.06 (5.39e-5/move)  accepted 0.471  clusterFraction 1.0000  peaksOk true  checkpoints 2
CONVERGE start 0.9 (first sample 0.905)  moves 400  tail mean 1.2132 min 1.1811 max 1.2567  lnA drift t=1.45 (6.12e-5/move)  accepted 0.453  clusterFraction 0.9975  peaksOk true  checkpoints 2
WATER-BILAYER-AREAMOVE-VERDICT settled=true chunks=300 areaPerLipid(MEASURED, tail mean)=1.1653 [min 1.1618, max 1.1700, corridor 1.1-1.5] driftPerChunk(lnA)=-9.102e-6 t=-1.26 thickness(measured)=4.4016 [corridor 4-6] clusterFraction=1.0000 waterInCore=22/4500 headBuriedFraction=0.0547 acceptedFraction=0.1213 of 3000 trials throughput=1070.49 steps/s at N=5700 verdict=passed
 Test Files  2 passed (2)
      Tests  3 passed (3)
```

| gate | corridor | predecessor's last reading | **this task** | verdict |
|---|---|---|---|---|
| area per lipid, in water, at γ = 0 | 1.1–1.5 σ² | 1.2095 (tail 1.2057–1.2126) | **1.1653** (tail 1.1618–1.1700) | **PASS** |
| bilayer thickness, in water | 4–6 σ | 4.3338 | **4.4016** | **PASS** |
| area per lipid, solvent-free (gate 6) | 1.1–1.5 σ² | 1.2064 ± 0.0070 | **1.2113 ± 0.0076** | **PASS** |
| thickness, solvent-free (gate 6) | 4–6 σ | 4.4773 | **4.4817** | **PASS** |
| dry core | dry | 38/4500 = 0.84 % | **22/4500 = 0.49 %** | dry |
| head burial | small | 0.0809 | **0.0547** | small |
| cluster fraction | one sheet | 1.0000 | **1.0000** | intact |
| settled | required | 450 chunks | **300 chunks** | settled, t = −1.26 |

**Does liquid-density water move them? No — every reading is inside the published run-to-run
scatter and nothing crosses a bound.** `hydrophobic-asymmetry-report.md` §4.1/§5 measured that scatter
directly over n = 5 at identical inputs (area 1.1211–1.2215, sd 0.042; thickness 4.316–5.103, spread
0.787 σ); this task's 1.1653 and 4.4016 both sit inside those ranges, and the solvent-free gate moved
by 0.4 % in area and 0.1 % in thickness. The brief's quoted 1.2019 / 4.5911 pair is likewise inside the
same n = 5 band. The dry-core and buried-head numbers both improved slightly, in the same direction,
which is consistent with scatter rather than with a change of physics — and there is no change of
physics to attribute them to, since the relaxation stage is not called by any of these tests (§2.4(a)).

---

## 6. Regressions — every one run, one invocation at a time, with numbers

| test file | result | numbers, before → after |
|---|---|---|
| `tests/gate6-bilayer.test.ts` | **2 passed** | area 1.2064±0.0070 → **1.2113±0.0076**; thickness 4.4773 → **4.4817**; converge-from-1.55 tail 1.2215 → 1.2070, from-0.9 tail 1.2145 → 1.2132. All inside 1.1–1.5 / 4–6. |
| `tests/water-bilayer-area-move.test.ts` | **1 passed** | area 1.2095 → **1.1653**; thickness 4.3338 → **4.4016**; water in core 38/4500 → **22/4500**; buried 0.0809 → **0.0547**; chunks 450 → 300; amph 384 + co_break 16 = 400 exactly. All inside published n=5 scatter. |
| `tests/catalyst-turnover.test.ts` | **1 passed** | this run: cc_bond 442, co_bond 361, chains 178, amphiphiles 125, bare carbons left 44/779, occupiedAtEnd 2/52, nonFinite 0, charge 0, census `{C:779,O:3117,H:779,M:52,W:695,K:0}` exact. Pin's A/B thresholds (placed with ≥1.28× margin) hold. |
| `tests/sim.test.ts` | **8 passed** | unchanged (membrane engine; grid-vs-brute-force force identity intact) |
| `tests/run-ui.test.ts` | **8 passed** | unchanged |
| `tests/soup-forces.test.ts` | **1 passed** | unchanged — grid + Verlet list vs full O(N²), and **zero GPU console warnings**, which is now a stronger statement: the three new pipelines' `layout:'auto'` bind groups would have surfaced here as warnings if any entry were missing |
| `tests/soup-area-move.test.ts` | **2 passed** | fixed-z predicted (N−N_mol)kT = 368.59, measured **376.86 ± 5.70**; fixed-volume predicted 0, measured **2.95 ± 13.53**; CPU potential vs GPU forces agreeing to ~1e-4 |
| `tests/soup-bonds.test.ts` | **3 passed** | catalyst gating holds; census+charge conserved under live reactions; assembly-vs-temperature `{"0.9":0.105,"1.8":0.03125}` |
| `tests/soup-valence.test.ts` | **1 passed** | no carbon over 2 C–C or 1 C–O, no head over chainCapacity, 45 000 steps |
| `tests/soup-clay.test.ts` | **4 passed** | with-clay cc 299 / co 54 / amph 40, no-clay cc 349 / co 30 / amph 22, nonFinite 0 in both arms; box change still REFUSED on a platelet |
| `tests/soup-rules.test.ts` | **4 passed** | schema + detailed balance unchanged; the new `coldStartRelax` section validates |
| `tests/soup-checkpoint.test.ts` | **3 passed** | round-trip exact; "a checkpoint does not change what the run computes next" holds — **now with real RNG state restored instead of zeros** (§1.5) |
| `tests/params.test.ts` (added to the list by me) | **3 passed** | the literal scanner covers `soup/wgsl`, so the two new shader files had to be clean; one comment mentioning `dt = 0.01` was rewritten to name `data/params.json` instead |
| **`tests/soup-nonfinite-guard.test.ts`** (NEW) | **1 passed** | see §1.3/§1.4 |
| **`tests/soup-cold-start-relax.test.ts`** (NEW) | **1 passed** | see §2.4 |

**No tolerance was changed, no corridor widened, no assertion relaxed.** The only number that moved
outside a stated scatter band is nothing: every moved number above is inside a published range. Two
test-authoring bugs of my own were fixed during the task (a regex expecting `step=` where the message
says `at step=`, and one expecting `interval` where it says `step interval`) — both in my new test, neither in
production code.

`tests/soup-vesicle.test.ts` was **never** run. The full suite was never run. The dev server on :5199
was neither started, stopped nor inspected.

One environmental flake worth recording so the next task does not chase it: `tests/soup-forces.test.ts`
failed once on `error: Failed to load resource: 404`, which is NOT a GPU warning. Cause identified by
listening on `page.on('response')`: alternating `npx tsx` scratch runs with `npx vitest run` makes vite
re-optimise its dependency cache ("Re-optimizing dependencies because vite config has changed") and a
request can 404 into that window. `rm -rf node_modules/.vite/deps_temp_*` and a re-run passed
immediately. Nothing to do with this task's changes.

---

## 7. The two new tests

- **`tests/soup-nonfinite-guard.test.ts`** (196 lines) — pins the loud failure. A healthy control and a
  deliberately diverging arm built from the SAME builder differing only in whether 20 pairs are pulled
  to 1e-4 σ, so it is a control rather than two unrelated runs. Asserts: the healthy arm reads
  `{pos:0,vel:0}` at start, after 2000 steps and after 5000 timing steps; the diverging arm starts
  FINITE (the overlap is a legal coordinate, not an injected NaN) and `step(1)` throws; the message is
  the **finiteness** guard's and not the drift guard's (`.not.toMatch(/Verlet list/)` — that
  distinction is the whole point, since the drift guard was running and silent in all three historical
  cases); the message names the step (`at step=1`), the chunk interval (`0..1`), a nonzero count for BOTH
  arrays, and the total it is out of; the state really is non-finite afterwards; the run really stopped
  at step 1; and the browser logged no GPU warning. It also measures and prints the guard's cost.
- **`tests/soup-cold-start-relax.test.ts`** (219 lines) — pins the relaxation stage's invariants and its
  payoff: step counter unchanged, velocities compared as u32 **bit patterns**, both RNG streams, the
  bond graph, all event counters, census and charge unchanged; positions DID move but by no more than
  the analytic displacement bound; `max|F|` down by >100×; the refusal at `globalStep != 0` with its
  message; and the payoff pair — the same composition (box 20, water 0.8 σ⁻³, ρ_tot 1.218) throws
  `non-finite state` without the stage and runs 3000 steps with `nonFinite = 0` with it.

---

## 8. Resources, wall time, housekeeping

- **Compute chunks used: 21 of the 30 allowed.** 1 guard test, 3 ladder/sweep invocations, 3 chemistry
  invocations (9 arms + 3 control arms), 1 relaxation-test failure + 1 re-run, 1 404 diagnostic, 1
  literal scan + params test, 5 regression groups, 2 single-test re-runs, 1 gate6 re-capture. Nine spare.
- **Total steps ≈ 1 020 000** — 429 000 across the 12 chemistry arms, 129 000 across the ladder/sweep
  arms, ~460 000 across the regression suite, ~7000 in the two new tests.
- **Longest single foreground invocation: 372 s** (chunk 8, six chemistry arms). Every invocation was
  `nice -n 15`, ONE at a time, foreground, never backgrounded. No two compute processes ever ran at
  once. One invocation was lost to the 120 s default Bash timeout (chunk 5) and re-run with an explicit
  one; its orphan was cleaned before continuing (`pkill` then `pgrep` = 0).
- **No orphan processes remain.** `pkill -f puppeteer_dev_chrome_profile` was run after **every**
  browser-bearing invocation and `pgrep -f puppeteer_dev_chrome_profile | wc -l` printed **0** every
  time, including after the last. `rm -rf node_modules/.vite/deps_temp_*` was needed once (§6). No dev
  server was started, killed or inspected. `--dump-dom` was never used. No particle array was ever
  transferred as JSON numbers: every reduction (max|F|, non-finite counts, displacement, amphiphile
  census, aggregate sizes) was computed inside the page and returned as scalars.
- **Scratch drivers were deleted, not committed** (`scratch-ladder.ts`, `scratch-chem.ts`,
  `scratch-404.ts`). Their exact invocations are quoted in §1.4/§3.1/§4.2 so the runs are reproducible.

## 9. Concerns

1. **The headline result is negative for the vesicle programme and I did not soften it.** Real liquid
   water cuts the amphiphile yield 1.84× and the largest aggregate 6.4×, and the binding constraint of
   the last three tasks WAS aggregate size. The honest reading is that the project's chemistry numbers
   were flattered by an under-dense solvent, and that the vesicle target is now further away than the
   published trace suggested, not closer.
2. **The crowder control is not perfect.** `H` has radius 0.8 σ against water's 1.0 σ, so a
   number-matched crowder supplies ~51 % of water's excluded volume. The control therefore under-states
   crowding, which strengthens the "it is water, not crowding" conclusion for yield (a weaker crowder
   recovering the yield is the expected direction) but weakens it for the aggregate-size result (where
   the crowder arm's aggregates are the LARGEST of all, and a stronger crowder might be different). A
   radius-matched inert species does not exist in `data/soup.json`; adding one is a composition change
   and its own task.
3. **The two chemistry arms differ in total density as well as in water,** which is unavoidable
   (adding water IS adding density) and is exactly why the control exists. The decomposition in §4.3 is
   as far as three arms can take it; a full 2-D sweep over (ρ_W, ρ_tot) was not run.
4. **The liquid arm was measured at box 30 and 20 000 steps, not at the box-54 window scale.** N = 191 757
   is 2.24× under the ceiling (§3.3) and the mechanism is density-driven, so I expect it to hold, but
   the window run itself at real liquid water is NOT measured here — it is a campaign, and this task's
   budget went to making it possible and to measuring what it will cost.
5. **ρ_tot = 1.618 is reported as "stable" on 3000 steps only,** and I explicitly do not defend it as a
   medium (it is above close packing for unit-diameter cores). The defended number is water at 0.800,
   ρ_tot 1.21793, confirmed at 20 000 steps.
6. **The relaxation stage is opt-in, which is a real trade.** It is the strongest possible
   measurement-neutrality argument (§2.4(a)) and it means a future caller who forgets `--relax` at high
   density gets a thrown error rather than a relaxed run. I consider a loud failure the right default
   given this project's history, but it IS a footgun and it is not hidden: the campaign CLI prints
   whether it relaxed, and prints a skip line on resume.
7. **The RNG-zeroing defect (§1.5) invalidates something about every resumed run in this project's
   history, and I did not quantify what.** After each resume every particle got identical Langevin
   noise. The predecessor's 315 000-step window run resumed 8 times; `tests/soup-checkpoint.test.ts`
   passes both before and after the fix, so the effect is inside that test's own scatter tolerance at
   its scale, but nobody has measured what 8 resumes of correlated noise did to a 315 000-step
   trajectory. Every conclusion drawn from a resumed campaign should be treated as carrying an
   unmeasured systematic until someone re-runs one.
8. **Non-finite detection is per-chunk, so the reported step is a bracket, not the exact step.** The
   message says so explicitly (`appeared within step interval A..B`). All three historical divergences
   and both new ones appeared inside the first chunk, so this has cost nothing so far.
9. **`clay: false` in every arm here**, as in both predecessors, because `soup/cli/campaign.ts` and
   these fixtures pin it. Same direction of bias as before (clay measures −23.8 % chain growth), so it
   cannot have manufactured the negative in §4.
10. **The exponent-bit finiteness test is correct for f32 and is not portable to f16** if this engine
    ever uses half floats; it is also blind to a "finite but physically absurd" state (a coordinate of
    1e30 inside a 30 σ box passes it). `assertVerletSafety`'s drift bound is what covers that case, and
    it now cannot be silenced by NaN.

## 10. Files

- `.superpowers/sdd/2026-08-16-soup-to-vesicle/loud-failure-and-liquid-water-report.md` — this report.
  `.gitignore` line 6 ignores `.superpowers/`, so like every predecessor it is committed with
  `git add -f`; the deviation is deliberate and flagged rather than silently introduced.
- NEW: `soup/wgsl/health.wgsl` (68), `soup/wgsl/relax.wgsl` (72), `soup/src/soup-health.ts` (63),
  `soup/src/soup-relax.ts` (170), `tests/soup-nonfinite-guard.test.ts` (196),
  `tests/soup-cold-start-relax.test.ts` (219).
- MODIFIED: `soup/src/soup-pipelines.ts` (+3 pipelines, 2 files concatenated), `soup/src/soup-buffers.ts`
  (+2 buffers, **the COPY_SRC fix**), `soup/src/soup-bindgroups.ts` (+3 bind groups),
  `soup/src/soup-integrate.ts` (+1 guard call), `soup/src/soup-grid-verlet.ts` (+NaN branch),
  `soup/src/soup-box-scale.ts` (+2 guard calls), `soup/src/sim.ts` (+2 methods),
  `soup/src/soup-types.ts` (+2 API entries), `soup/src/rules.ts` + `soup/src/rules-validate.ts`
  (+`ColdStartRelax` and its validation), `soup/cli/campaign.ts` (+`--relax`),
  `engine/src/gpu.ts` (**readBack now throws instead of returning zeros**),
  `data/soup.json` (+`coldStartRelax`, 5 lines; nothing existing changed).
- `soup/wgsl/step.wgsl` **untouched at 596 lines**, which is why the two new kernels are new files:
  CLAUDE.md's rule is "split first, then add", and both are separate responsibilities anyway. Largest
  file after this task is still `soup/wgsl/step.wgsl` at 596; nothing crossed 600.
- `data/params.json` untouched. No threshold, corridor, potential, rate, recogniser or rank-A constant
  was modified.
