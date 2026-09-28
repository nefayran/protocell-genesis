# Soup to vesicle: what was tested, what was measured, how it ended

This is a summary of [`soup-to-vesicle-verdict.md`](soup-to-vesicle-verdict.md), the project's
closing document (state as of 2026-08-21), plus the one task that came after it,
the acid–soap pair of 2026-08-23 (§13), which changed no verdict. Every number here is copied from
those two documents, and they name the artifact or report behind each one. The full protocols are the
reports in `.superpowers/sdd/2026-08-16-soup-to-vesicle/`; the last five to read, in order, are
`long-range-electrostatics-report.md`, `supply-window-report.md`, `big-box-report.md`,
`closeout-report.md` and `confined-parcel-report.md`.

The machine-readable gate table is `verify/out/gates.json` and the human-readable one is
`verify/out/report.html`. Both are rebuilt by `npm run verify` and never edited by hand.

## 1. The question and the rules

One question: does a coarse-grained soup of monomers assemble itself into a closed vesicle, an object
with an inside that the outside cannot reach and a bilayer wall around it?

The rules that decided what counts as an answer:

- A gate is a claim with a model value, a literature corridor, a verdict and an evidence rank. Gates
  are published as passed, failed or unproven, and they are never widened to pass.
- Evidence ranks: A measured (source, conditions, error); B computed; C bounded by thermodynamics or
  by ratios from an independent source; D estimated. A gate that rests only on D evidence is unproven
  by definition.
- `engine/` holds no hand-written chemical constant; it reads them from `data/`.
- Hydrophobicity has to emerge from water displacement, never from a direct attraction between tails.
- `co_bond.attemptRate` is not tuned. It was requested nine times during the project and refused nine
  times.

In one sentence: a working coarse-grained WebGPU engine was built, with emergent hydrophobicity,
catalyst-grown chain chemistry, solvent evaporation, explicit liquid water, charge with a protonation
equilibrium and a full set of instruments for recognising closure, and it reliably assembles not a
vesicle but one percolating network that spans the periodic box along all three axes in every
measurement of every campaign. The project answers its question in the negative and names the
measured cause.

## 2. What was built

| layer | what it is | where |
|---|---|---|
| soup | GPU bond chemistry: C–C chain growth on a catalyst, head termination (C–O), valence, dehydration and rehydration, solvent evaporation, a clay plate | `soup/` |
| explicit water | water as its own bead type; attraction by class (solvent, polar, apolar), depth ratios from `martini_v2.1.itp` | `soup/wgsl/step.wgsl`, `data/soup.json` |
| electrostatics | screened Coulomb (Debye–Hückel) between heads; charge is a dynamic variable sampled by Monte Carlo at constant pH; its own cutoff in units of its own screening length (4 λ_D) and its own neighbour list | `soup/src/electrostatics.ts`, `soup/wgsl/es-*.wgsl` |
| measurement | amphiphile recognition, per-aggregate breakdown, shape from the gyration tensor, head layers, cavity, encapsulated water, box wrapping | `soup/src/aggregates.ts`, `soup/src/water-closure.ts`, `tests/percolation-check.test.ts` |
| gates | `data/literature.json` (definitions), `verify/gates.ts` (verdict rules), `verify/run.ts` (the run), `verify/out/gates.json` and `report.html` | `verify/` |
| viewer | run page: start, pause, stop, live measurement, stage ladder, aggregate breakdown, cavity highlight, honesty notes | `viewer/run.html`, `viewer/run-*.ts` |
| long runs | CLI with checkpoints and resume; a campaign runs as dozens of separate calls of under 500 s each | `soup/cli/campaign.ts` |

The atomic detail shown in the viewer is a reconstruction from reference geometry on top of the
coarse-grained coordinates, not all-atom dynamics, and the page says so.

## 3. Published gates

The table is the last regeneration by `npm run verify`, on 2026-09-28, as committed in
`verify/out/gates.json`. The campaign block is the box-76 run (`bbB76`, 148,200 steps). The two
solvent-free bilayer rows are re-measured on every regeneration, and the engine is not bitwise
reproducible from run to run, so they move slightly (1.2056 σ² and 4.4750 σ on 2026-08-23). The two
water rows are read from the measurement artifact of 2026-08-23, and the closing document quotes an
earlier measurement of the same patch (1.1607 σ² and 4.6871 σ). Every verdict is the same in all of
them.

| gate | value | corridor | rank | verdict |
|---|---|---|---|---|
| area per lipid, no solvent | 1.2004 σ² | 1.1–1.5 | A | passed |
| bilayer thickness, no solvent | 4.4545 σ | 4–6 | A | passed |
| bending modulus κ | – | 5–50 kT | A | unproven (invalid fit window) |
| area per lipid in explicit water | 1.1734 σ² [1.1696, 1.1765] | 1.1–1.5 | C | passed |
| bilayer thickness in explicit water | 4.4689 σ | 4–6 | C | passed |
| closure: encapsulated water over threshold | 0 (0 beads against 315.409) | ≥ 1 | B | failed |
| largest aggregate is finite | 3 axes of 3 (27/27/27 layers of 27) | ≤ 0 | B | failed |
| overall: a self-assembled closed vesicle | 0 aggregates | ≥ 1 | A | failed |
| agreement with Anderson–Schulz–Flory | mismatch 0.1594 | – | D | unproven (no corridor) |
| mean beads per tail | 3.419 | 2–3 | D | unproven (outside) |
| detector cavity on a synthetic shell | 1284.875 σ³ | ≥ 370.8656 | D | unproven (inside) |
| carbon atoms to beads mapping | 3 | – | D | unproven |

### 3.1 A bilayer in liquid water

In explicit liquid water at 0.8 σ⁻³ the bilayer holds both area and thickness, 1.1734 σ² and
4.4689 σ in the last regeneration, with a dry core (45 water beads out of 4,500 inside), a buried-head
fraction of 0.0919, and an area that settled (ln A drift −2.722e−6 per sample, t = −0.71 over 300
samples). The area is measured by an area move at zero tension, not set by construction.

Before the `hydrophobic-asymmetry` task the same patch in water measured 12.2 to 12.4 σ thick against
the 4 to 6 corridor. The cause was buried heads: 40 to 49% of head beads sat inside the hydrophobic
zone, because removing the old tail–tail term had left the leaflet with no lateral cohesion at all.
The fix was not a fit but the pair-depth ratios of MARTINI 2.1: water–water 5.0, water–head 4.0,
tail–tail 3.5 (the reference, factor exactly 1.0) and water–tail 2.0 kJ/mol. The exchange energy
5.0 + 3.5 − 2·2.0 = +4.5 kJ/mol stays positive, so mixing is still unfavourable and hydrophobicity is
still emergent. The rank is C, not A: the corridor is from the literature, but the depth ratios come
from another force field and `epsilonScale = 1.0` carries rank D.

### 3.2 Chain-length statistics

On the last `bbB76` snapshot the event-based α = cc_bond / (cc_bond + co_bond) predicts an
Anderson–Schulz–Flory mean that differs from the measured mean per tail, 3.419, by 0.1594, and the
distribution is geometric, which is the Anderson–Schulz–Flory signature. The gate is still unproven
because the specification gives no numeric corridor for it and inventing one would fit the gate to
the result. The tail length is also a trend inside one run at fixed composition: 2.158, 2.911, 3.207,
3.419. The molecule keeps growing after the structure has reached its plateau.

### 3.3 Evaporation is solvent removal

Evaporation here removes solvent; it is not a phase transition. The concentration factor the model
actually reaches is 3.20× against 1,400× for a literature dry–wet cycle (16.07% of it on a log
scale), and the carbon pool is enriched about 691× against the richest literature pond (about 15 mM
decanoic acid, ACS Earth Space Chem. 2023, PMC9869395). The dry–wet cycle does work, measured: yield
×1.68 to 1.80, bonds ×1.76 to 1.93, largest aggregate ×8.1 to 8.4, after the fix in §7.3 (12.5 to 14.5
was published before it).

### 3.4 Electrostatics, and the one independent validation

Each head's charge is sampled by discrete Monte Carlo at constant pH from
ΔG = kT·ln10·(pKa_intr − pH) + ΔU_es, with λ_D = 0.304 nm / √(I[M]).

- The force was checked against a numerical gradient twice: worst relative residual 5.145e−9 per pair
  on r ∈ [0.6, 2.7] σ, and 1.736e−5 for the whole field against 1.303e−3 for a control with charge
  switched off.
- With ΔU_es = 0 the sampled α reproduces Henderson–Hasselbalch to within 2.08e−3 at five pH values.
  With ΔU_es on, a two-head system matches exact enumerated Boltzmann weights to 3.140e−3 over 200,000
  sweeps, and the interaction is visible ("both charged" at 0.21049 against 0.25000).
- The salt shift of the apparent pKa between 10 and 100 mM, literature about 0.7, came out at +0.124
  with a cutoff shared with the Lennard-Jones terms (2.7224620 σ), where the discarded share of the
  interaction was 0.8385 in one arm and 0.3389 in the other. With a cutoff of 4 λ_D (15.2000 σ) the
  discarded share is 0.0916 in both arms and the shift is +0.709. No constant was fitted to get it.
- The apparent pKa itself is emergent: the model is given only the monomer pKa of 4.9, and the
  interfacial shift measures +0.08, +0.32, +0.67 and +0.97 at pH 4, 5, 6 and 7.
- An acid–soap correlation appeared on its own as alternating charge: paired fraction 0.058 to 0.080,
  unlike-contact excess 1.23 to 1.37 over random, with no hydrogen bond added.

What electrostatics did not do: it did not break the network at any pH from 4 to 9. Head–head
repulsion reached 0.7215 kT against 0.909 kT for an apolar contact, only 1.26× weaker, and the number
of wrapping axes moved by exactly zero. The energy gap closed from 2.86× to 1.26×; the topology did
not change.

## 4. Verdict

No closed vesicle, in any snapshot of any campaign. `hasVesicleAggregate` = false, `closed` = false,
`encapsulatedWater` = 0 everywhere.

The largest campaign (`bbB76`): a 76 σ box, N = 483,268, liquid water at ρ_W = 0.8000005, O:C =
0.333265, seed 19, kT 1.1, charge on at pH 7.0 and I = 0.01 M, one drying event, 148,200 steps of
which 111,800 settled and wet. It started from a lattice of monomers only (270 aggregates, the largest
of 4 amphiphiles).

| quantity | measured | needed | verdict |
|---|---|---|---|
| amphiphile supply | 4,519.5 ± 5.2 (0.11% over the last four) | about 930–955 | 4.73× above the threshold, ruled out |
| size of the largest aggregate | the same object, 99.93% of the supply | about 930–955 | ruled out |
| edge energy per molecule | 0.583 kT at N = 2,087 (box-54 campaign) | ≲ 1 kT | ruled out |
| connectivity: wrapping axes | 3 of 3, 27/27 layers, in 23 of 23 wet snapshots | 0 | the binding constraint |
| encapsulated water | 0 in all 23 | ≥ 315.4 beads | downstream |
| radial head layers | 1–2, jittering | 2 (a bilayer wall) | downstream |
| cavity | 139.25 σ³, still growing | ≥ 370.8656 σ³ | 2.66× short at the most favourable reading |
| flatness λ₁/λ₃ | 0.7571 | ≤ 0.35 | 2.16× too high |
| monomer exchange, fission, fusion | 0, 0, 0 in all 22 intervals | non-zero | not tested by this model |

The radius of gyration is 37.43 σ against 38.000 σ for a uniformly filled 76 σ box (98.5%).

## 5. Mechanism

### 5.1 What was ruled out

| explanation | test | result |
|---|---|---|
| supply: too much material merges into one object | supply lowered to 1.50× the threshold, then raised to 4.82× | 3 of 3 axes both times |
| edge energy: a finite object pays too much for its edge | 0.583 kT per molecule at N = 2,087 against ≲ 1 kT | ruled out |
| pair energy: heads repel too weakly | gap closed from 2.86× to 1.26× (§3.4) | wrapping axes moved by 0 |
| box size: no room for a finite object | box volume ×2.79 at constant density (54 σ to 76 σ) | 3 of 3 axes, 27/27 layers, in all 23 snapshots |

### 5.2 What was left: a spanning cylindrical micelle is cheaper than a closed vesicle

In a fully periodic box an object that spans the box has no edge at all, so it beats a closed vesicle
that has to pay for curvature. The cheapest edgeless spanning object of this amphiphile is a
cylindrical micelle whose radius is the leaflet thickness t/2: inverting the measured radii of
gyration gives a 151-molecule rod a physical radius of 2.353 to 2.560 σ against t/2 = 2.229 to
2.484 σ.

| object | cost in amphiphiles | depends on box side L? |
|---|---|---|
| closed vesicle of the minimum volume (370.8656 σ³) | 930–1,030 | no |
| cylindrical micelle spanning the box | (5.806–6.635)·L | yes, linearly |

At L = 54 the spanning micelle costs 314–358 against 932–1,030 for closure; at L = 76, 440–450 against
930–955. The window where a vesicle is preferred opens only at L* = 160.6 to 161.3 σ (an independent
test pins [157.4, 164.4] σ), which is 3.29 million particles.

The finite-parcel run (§11) later showed that this was not the whole story: with the periodic
topology removed, closure still did not happen, because in a finite region the cheapest edgeless
object is a capped cylindrical micelle of 17.5 to 21.3 amphiphiles, against a closure floor of 937 to
1,014. That window is inverted by a factor of 44 to 58.

### 5.3 Why L* was out of reach

The particle ceiling was one unmeasured number, `verletList.listCapacity = 2500`. Measured over eight
configurations, the maximum neighbour count is 866, set by the settled percolating aggregate (local
density 1.94× the box mean), not by the dry phase (749).

| list capacity | basis | particles | box at ρ_W = 0.8 |
|---|---|---|---|
| 2500 | the old guess | 429,496 | 81.27 σ |
| 1126 | derived floor | 953,589 | 106.03 σ |
| 866 | zero headroom | 1,239,886 | 115.73 σ |

Even with zero headroom the box stops at 115.73 σ, 1.39× short of L*, because a flat per-particle
list costs L³ times the square of the density. A cell traversal with no per-particle array was
measured and rejected at 122.28 ms per step against 17.47 (7.00× slower). With memory solved, time
binds: ms per step grows as N^1.35, which projects to about 1,013 ms per step at L* and a 41-hour
campaign. Memory is 3.45× short and time about 15×.

### 5.4 The sequence of shortfalls

The published ratio between the largest aggregate and the closure threshold, in order:
80×, 8.9×, 3.36×, 33.4×, 6.76×, 6.75×, 0.48×, 0.42×, 0.67×, 0.21×, 0.45×. The early ones fell as the
chemistry started working; 3.36× is really 1.85× after the zeroed-RNG fix (§7.2) and 6.75× is 6.02×
after the z double-count fix (§7.5). From 0.48× on the aggregate is larger than the threshold, and each
later campaign removed one more explanation: charge (0.42×), material budget (0.67×), box size
(0.21×) and the periodic topology itself (0.45×, the finite parcel). The one explanation left is about
the molecule, not the box: the packing parameter of this amphiphile.

## 6. Cavity or lumen

The `closeout` task resumed the box-76 campaign for 13 more calls and 61,700 steps (86,500 to
148,200), which doubled the settled wet time (50,100 to 111,800 steps). The cavity grew from 48.1 to
139.3 σ³ (2.90×) with no plateau: slope +0.695 σ³ per 1,000 steps over all 23 snapshots, +0.586 over
the 13 new ones. Over the same window the size was flatter than ever (4,519.5 ± 5.2, 0.11%).

While the cavity grew 2.9×, encapsulated water stayed exactly zero in all 23 wet snapshots, `closed`
stayed false, head layers stayed at 1–2, and the object wrapped 3 of 3 axes every time. A growing
cavity inside a percolating object does not become a lumen: `cavityVolume` floods the interstitial
space inside the object's own bounding region, and in a network that space is not the inside of a
protocell. Quoting "139.25 against 370.8656" as a closure shortfall is generous to the model.

## 7. Defects found on the way

Each of these made a published number wrong, and each is fixed with a test that fails if it returns.

1. Chain-tip choice at nucleation (`soup/wgsl/bond-adsorption.wgsl`). The start condition was
   geometrically impossible: eligible triples fell 393, 3, 0 by step 120,000. Fixed: nucleations
   117 to 192, amphiphiles 56 to 129.
2. Checkpoints with a zeroed RNG (`soup/src/checkpoint.ts`). All 72 checkpoints carried 100% zero RNG
   bytes, so every resumed run gave every particle the same Langevin noise and aggregates could not
   merge. Fixed: the largest aggregate at step 90,000 went from 268 to 588.
3. A stale force after a box-size change (`applyBoxScaleOnce`): an error of 313.7 at a force scale of
   513. Fixed: the largest aggregate in the cycled arms went from 249.3 to 109.3.
4. Silent divergence (`soup/wgsl/health.wgsl`): any comparison with NaN is false, so the guard passed
   forever. An exponent scan now throws, at a cost of 0.064% per chunk.
5. Double counting across the z boundary (`engine/src/aggregate.ts`): aggregates crossing z were cut
   in two. Fixed: the decisive run's largest aggregate went from 135 to 151, aggregate counts had been
   overstated by 26.5%, and a published "2 head layers, passed" became 1, failed.
6. The percolation gate recognised its input by the run label (`verify/campaign-gates.ts`) and went
   silent on the first campaign with a new label.
7. The page layout ate the START click (`viewer/run.html`): at 800×600 the point under the button
   belonged to a bottom note, and the run never started, with no error anywhere.

## 8. Six silent failures

Each looked like a success.

1. `readBack` from a buffer without `COPY_SRC` returns the zero-initialised staging buffer (§7.2).
2. A comparison with NaN is always false, so the divergence guard passed forever (§7.4).
3. `requestDevice()` without `requiredLimits` quietly caps every pipeline at the default storage limits.
4. A stale force after a box change is not zero but a quietly wrong number (§7.3).
5. Gates that recognise their input by the run label go silent on a new experiment (§7.6).
6. A buffer request above the device limit raised no JavaScript error. WebGPU logged a validation
   warning, every dispatch on that bind group became a no-op, and the run returned max|F| = 0, no
   non-finite values, no neighbours and 0.0405 ms per step: a 500× "speed-up" with a clean health
   check. `createSoup` now checks every buffer that scales with N against the device's own limits.

A seventh came from the test tooling: reading a WebGPU canvas through `drawImage` into a 2D context
returned 0 changed pixels out of 30,000 from a page that was drawing (119 frames, 661 instances). Zeros
from an instrument are first a reading about the instrument.

## 9. Still unproven, disputed, or measured with one seed

- All campaigns use one seed (19). The main conclusion, 3 of 3 wrapping axes, is a binary topological
  result, but a number such as 4,519.5 is one trajectory, and the engine is not bitwise reproducible.
- The cavity never settled, so "2.66× short" is the most favourable reading, not a result.
- Radial head layers jitter between 1 and 2 throughout; there is no bilayer wall.
- The bending modulus κ was never measured: the undulation-spectrum fit window is invalid (log-log
  slope −1.056 against the theoretical −4).
- Monomer exchange, fission and fusion were exactly zero in all 22 intervals, so this model never
  tested monomer solubility or a CMC.
- In this model the head is the chain terminator, so two-tailedness and tail length cannot be
  separated by composition; nothing was measured between O:C = 4 and O:C = 1.
- The percolation test was never run on a purpose-built object of known topology, although it
  reproduces "does not percolate" on three independent systems.
- Known flaky tests, shown not to come from recent changes: `tests/soup-grid-resize.test.ts` fails on
  unchanged code, 2 of 3 `rim-lambda-insitu` tests failed before, and
  `tests/soup-drywet-cycling.test.ts` passes and fails on identical repeats (event ratio from 0.4978
  to 0.7171). No threshold was widened; the cure is a larger sample. In a full sequential run on
  2026-09-27, `soup-grid-resize` passed and `soup-drywet-cycling` failed; of the `rim-lambda-insitu`
  tests, two need a checkpoint that is not in the repository and are skipped without it, and the third
  had a fixture that predated explicit water and passes once its water count is set to zero.
- `capacitySafetyFactor` 2.6 and `capacityFloor` 1126 are rank D. A composition whose condensate is
  denser than 2.351 σ⁻³ will get a loud error, not wrong physics.
- The throughput exponent 1.35 was measured over only a 3.9× range of N, so the 1,013 ms-per-step
  projection is an order of magnitude.
- The neighbour-structure decision holds for this device (Apple, Metal 3); re-measure before a port.
- The clay plate took part in no published measurement: every arm of every campaign ran with
  `clay: false`.

## 10. What the next model should change

1. A two-tailed amphiphile by construction, with a fixed short tail. Reaching a packing parameter of
   ½ to 1 by composition needed O:C = 0.333, which also raised the tail length from 2.158 to 3.419 and
   with it the connectivity of the organic phase. A fixed two-tailed topology gives the right packing
   without raising the number of covalent neighbours per molecule, and that number is what
   percolates.
2. A dilution axis independent of the chemistry. `co_bond` does not fire at prebiotically honest
   dilution (0 events in 150,000 steps at 691× enrichment), so this model cannot separate "enough
   chemistry" from "above the gel point". Pre-made amphiphiles that can be diluted freely are the
   cheaper route.
3. A non-periodic domain. Done in §11, and it did not produce closure.
4. Real evaporation beyond 3.20×.
5. Monomer exchange and a solubility equilibrium, without which there is no CMC, no Ostwald ripening
   and no real fusion.
6. Co-surfactant mixtures: every campaign used one species with one head type.

The cheapest next measurement: pre-made two-tailed lipids (the construction the `water-bilayer-*`
tests already hold stable at ρ_W = 0.8), no chemistry, no evaporation, at four amphiphile densities
around the percolation threshold (1.34e−2, 6.7e−3, 3.3e−3 and 1.7e−3 σ⁻³ at box 54), with the
percolation test as the reading and closure as the end point, in a non-periodic or clearly large
domain.

## 11. The finite water parcel

The last task tested the explanation the project had called the root: that closure fails because of
the periodic box, where a spanning object closes on itself through the boundary and has no edge.

### 11.1 Setup

The system is held in a sphere of radius R by a soft neutral wall, U(s) = ½k(s − R)² for s > R and
zero inside, with k = 100 ε/σ². The wall ignores the particle type entirely: same for water, heads,
tails, catalyst and mineral. A hydrophilic wall would act like a mineral surface and a hydrophobic one
would nucleate a film; a neutral one is the only choice that tests closure instead of arranging it.

Periodicity is removed by geometry, not by a flag: the parcel sits in a cube with L ≥ 4R, so no
particle comes near a face and every pair distance is below L/2, which makes both coordinate wrapping
and the minimum-image convention the identity. Both conditions are checked when the system is created
and measured on every snapshot. Evaporation became physical: remove water and the parcel shrinks, from
R = 36 to 21.895 σ as water went from 156,346 beads to 112.

### 11.2 Four latent defects where the box was not the volume

Every density in the engine was written as N / box³, which is correct in a fully periodic box. In the
parcel the material occupies 195,432 σ³ of a 4,096,000 σ³ box, a ratio of 20.959, and four places read
the density that much too low: the Verlet capacity, the long-range head-list capacity (281 instead of
5,898), the closure threshold (which would have widened from 313 to about 15 beads) and the
evaporation ramp (caught loudly, 614,769 of 645,453 components non-finite). No published number used a
wrong density, because every published run was periodic. N / box³ is a density plus an assumption.

### 11.3 The amphiphiles do not coat the wall

This is the most plausible way the run could have failed, so it was measured on every snapshot. Before
aggregation the enrichment next to the wall relative to water was C 0.985, O 1.013, H 0.981, M 0.959:
the wall is neutral by measurement. In the settled wet phase the 4.687 σ outer shell held 1 carbon out
of 24,926 and 0 hydrogens out of 24,926, in 18 of 18 snapshots, while the aggregate sat well inside the
parcel (radius of gyration 19.54 to 19.78 σ in a 36 σ parcel) for 188,000 steps.

### 11.4 Result

R = 36 σ in a 160 σ box, the box-76 composition scaled to the parcel volume (N = 215,151, water at
0.8000 σ⁻³), seed 19, kT 1.1, pH 7.0, I = 0.01 M, charge at 4.000 λ_D, one drying event, from monomers,
225,200 steps of which 188,000 settled and wet.

| quantity | periodic box 76 | finite parcel R = 36 |
|---|---|---|
| wrapping axes | 3 of 3 in 23 of 23 | 0 of 3 in 20 of 20 |
| `aggregate-percolation` gate | failed | passed, the first time in the project |
| aggregates / largest | 4 / 4,512 | 1 / 2,092 |
| supply over the closure floor | 4.73–4.82× | 2.06–2.23× |
| `encapsulatedWater` | exactly 0 in 23 of 23 | exactly 0 in 19 of 19 |
| closure threshold | 312.64–315.41 | 312.67–314.87 |
| `closed` / `hasVesicleAggregate` | false / false | false / false |
| flatness (≤ 0.35 needed) | 0.7277–0.7610 | 0.8303–0.8552 |
| radial head layers | 1 and 2, alternating | 2, then 1 for the rest of the run |
| cavity against 370.8656 σ³ | 139.25, slope not falling | 69.5, slope changed sign |
| fusion / fission / exchange | 0 / 0 / 0 | 0 / 0 / 0 |

### 11.5 Verdict

No closure. Without the periodic topology the aggregate stopped wrapping and still trapped no water,
so the periodic boundary was a consequence, not the cause. In a finite region the cheapest edgeless
object is a capped cylindrical micelle of 17.5 to 21.3 amphiphiles, against a closure floor of 937 to
1,014; even a capped micelle spanning the parcel costs 439 to 490, 0.43 to 0.52 of the floor.

Ruled out by now: chemistry, supply (at 1.50×, 2.08×, 2.36× and 4.82× the floor), pair energy, box
size, temperature, charge, salt and pH, and the boundary topology. One explanation is left, and it is
about the molecule: the packing parameter of this amphiphile.

The frame that would apply to a success applies here too. The wall is neutral by measurement but is an
idealisation: a real air–water interface is hydrophobic and a mineral surface is hydrophilic. A 36 σ
parcel is 1.6× the object's diameter, which is not an ocean. The periodic result stands on its own:
below L* the ground state in a periodic box is a spanning object, and the measured object was one.
Showing the same without confinement needs L > L* = 146.9 to 172.5 σ, 2.53 million particles against a
ceiling of 953,589.

## 12. How to re-check it

```
# the whole gate table (about 3 minutes): writes verify/out/gates.json and report.html
npm run verify

# inputs npm run verify reads but cannot recompute itself (need data/checkpoints/)
nice -n 15 npx vitest run tests/water-bilayer-area-move.test.ts --no-file-parallelism
CONTINUOUS_RUN_PREFIX=bbB76-step CONTINUOUS_RUN_DIRS=data/checkpoints/bbB76 \
  CONTINUOUS_RUN_ARTIFACT=verify/out/gates-campaign-trace.json \
  nice -n 15 npx vitest run tests/continuous-run-audit.test.ts --no-file-parallelism

# continue the campaign (same ensemble, not the same trajectory)
nice -n 15 npx tsx soup/cli/campaign.ts --label bbB76 --box 76 \
  --start '{"C":55988,"O":18659,"H":55988,"M":1452,"W":351181}' --seed 19 --kT 1.1 \
  --relax --cycle --evaporate --cycles 1 --charge --pH 7.0 --ionicStrength 0.01 \
  --steps 4700 --every 4700 --dir data/checkpoints/bbB76

# the finite parcel: its separate gate row, and the wall and no-wrap proofs
nice -n 15 npx tsx verify/confined-gates.ts
nice -n 15 npx vitest run tests/soup-confine.test.ts --no-file-parallelism

# the live page, headless: the run advances and draws
nice -n 15 npx tsx verify/viewer-smoke.ts
```

When an input artifact is missing, the corresponding gate comes out unproven with a reason; an old
number is never substituted.

## 13. After the verdict: the acid–soap pair (2026-08-23)

The last task gave the head–head term a computed number instead of an assumption. The atomic engine
of this repository (`atomic/`, MACE-OFF23) measured the association of two carboxyl heads: with 32
waters, acid plus carboxylate binds at +10.38 kcal/mol, from four independent starting geometries, while
two neutral acids are unfavourable (−5.51 kcal/mol). Until then `polarPolar` had been exactly 0 for the
whole project, so the fatty-acid pH-window mechanism was structurally impossible.

- The rule reads the protonation state: it fires only between one protonated and one deprotonated head,
  both as a force and inside the constant-pH Monte Carlo step.
- 10.38 kcal/mol is an energy, not a free energy (the entropy of pairing was not computed and works
  against the pair), so it was used as an upper bound and the strength was swept. At the full bound
  and at 0.645 of it the bilayer collapses, thickness 1.01 and 1.42 σ against the 4 to 6 corridor.
  The accepted strength is 0.161 of the bound (7.0 kJ/mol, rank D), the largest that keeps both
  corridors on a settled area: 1.1510 σ² and 4.5517 σ, with 0.4246 of heads paired and a pair lifetime
  of 2,947 steps.
- The degree of deprotonation buffers itself towards ½ from both sides: at pH 7 it goes from 0.9446
  to 0.5017 as the pair gets stronger, and at pH 4 it rises from 0.1122 to 0.2556. This is pinned
  against exact Boltzmann weights.
- A pH window appears in pairing, flatness and compactness (flatness 0.18 to 0.25 and Rg/(L/2) 0.58 to
  0.77 at pH 4 to 6, against 0.45 to 0.53 and 0.88 to 0.99 at pH 7 to 9, with no such shift in a
  control without the pair), but not in bilayer competence: the object is flat and elongated in its
  plane, a ribbon rather than a disc.
- Closure is still not the cheapest edgeless object at any strength. For the strengths that keep the
  bilayer within its corridors, the closure floor is 52.5 to 53.2 times the cost of a capped micelle
  (989.8 against 18.8 amphiphiles at the accepted strength), and L* moved from 160.6–161.3 σ to
  155.1–159.3 σ against the engine's ceiling of 115.7 σ. A new vesicle campaign was therefore not run.

The gates were regenerated and no verdict changed. The full protocol is
`.superpowers/sdd/2026-08-16-soup-to-vesicle/acid-soap-pairing-report.md`.
