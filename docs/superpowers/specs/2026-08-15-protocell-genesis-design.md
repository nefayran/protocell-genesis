# protocell-genesis: design

Date: 2026-08-15

## 1. Goal

An interactive three-dimensional simulation showing the path from individual atoms and
simple molecules of a hydrothermal fluid to a closed lipid vesicle, while numerically
guarding that path: each step is checked against a measured value from the literature,
and the result is output as a table of passed and failed gates.

The simulation does not prove that life arose exactly this way. It proves **sufficiency**:
under the stated conditions, amphiphile synthesis is thermodynamically allowed, the
chain-growth mechanism reproduces the observed length distribution, and molecules of that
length are bound to assemble into a bilayer and close up. This claim is testable and
falsifiable by the numbers.

Priorities set by the project owner, in order of importance:

1. Scientific accuracy.
2. Interactive three-dimensional observation.

Starting point: real individual atoms and synthesis "as in the primordial soup."
Finish line: a closed vesicle with water inside, confirmed by measurement, not by eye.

## 1a. Amendment from 2026-08-16: a single continuous run

The initial architecture split the path into stages with an explicit seam: chemistry was
computed at its own scale, assembly at its own, and only the molecule population was
carried across between them. The project owner clarified the requirement: what is needed
is a **single continuous run** from primordial soup to vesicle, testing the hypothesis of
primary vesicle formation from atoms. A seam does not meet this requirement, so the
architecture changes.

**A single reactive model.** All particles live on one coarse-grained scale: carriers of
carbon, hydrogen and oxygen, water as an implicit background with explicit reaction
participants, catalytic sites. Bonds form and break by probabilistic rules rather than
being fixed at creation. Amphiphiles emerge as a product of these rules, a chain with a
polar head, and their aggregation is governed by the same non-bonded potential that has
already passed the per-lipid area and bilayer thickness gates in stage C. One time scale
for the whole run.

**What this claim can prove.** Not "this is how it happened on Earth." Strictly testable:
given the stated ingredients and rules, the system spontaneously traverses the path
soup -> amphiphile -> micelle -> bilayer -> closed vesicle, and there exists a parameter
window in which this happens, and outside it, it does not. A phase map over temperature,
precursor concentration and catalytic-site density is the substance of the hypothesis test,
and a "no" over part of that map is a full-fledged result.

**What is lost in the process, and this is recorded in the report.** Absolute time is not
tied to seconds: what remains is a single explicit factor `κ_t`, displayed on screen.
Reaction rates carry rank D, because elementary-reaction rate constants for early-Earth
conditions do not exist. Atomic detail in the frame is a reconstruction from reference
geometry laid over coarse-grained coordinates, not full-atom dynamics.

**What is reused unchanged:** the verified assembly potential and its gates, the
flood-fill closure detector, the verdict and reporting mechanism, per-atom rendering with
back-mapping.

## 2. Boundaries of honesty

Three limitations are stated up front, shown in the interface, and repeated in the report.

**Time scales differ by orders of magnitude.** Chemical synthesis in the experiment runs
for two to three days; membrane self-assembly takes microseconds. The stages are joined
not by continuous dynamics but by an explicit "seam": only the molecule population by
chain length, pH and ionic strength is carried from the chemistry stage to the assembly
stage, nothing else. The seam is labeled on screen.

**Absolute chemical reaction rates are unknown.** The literature states directly that
elementary rate constants for early-Earth conditions do not exist, and industrial
Fischer-Tropsch microkinetics do not apply to them. Kinetics are therefore built on
detailed balance: the ratio of the forward to the reverse constant is strictly equal to
`exp(-ΔG/RT)`, and the overall time scale is set by a single global factor `κ_t`, which is
displayed on screen as a number. The consequence, and it is fundamental: **where the
system ends up is provable; how long it takes is not.**

**Water in the chemistry stage is not fully represented.** Only molecules that take part
in reactions are shown explicitly; everything else is an implicit background with the
correct viscosity and dielectric permittivity at the given temperature. This is labeled in
the interface.

## 3. Hierarchy of evidence

A rank is assigned to each reaction and each constant individually, not to the project as
a whole.

| Rank | Meaning | What must be recorded |
| --- | --- | --- |
| A | Measured | Reference, measurement conditions, uncertainty |
| B | Calculated | Method, level of theory, basis set, control against a rank-A reaction with the discrepancy stated in kJ/mol |
| C | Bounded by thermodynamics | Absolute value unknown; the ratio `k_f/k_r` is derived from ΔG, equilibrium is correct, timing is not |
| D | Estimate | Source of the estimate; gates that rely on D alone appear in the report as **unproven** |

The rank is stored in `data/network.json` next to the constant. A rank upgrade is a
separate commit to this file. A structural rule that keeps fitting visible: `engine/` must
not contain a single hand-entered chemical constant; the engine only reads artifacts from
`data/`.

## 4. Architecture

Five layers, separated by execution time, not by topic.

| Layer | When it runs | Technology | Output |
| --- | --- | --- | --- |
| `thermo/` | build time, rarely | Python + pyCHNOSZ (R, CHNOSZ, the OBIGT database, HKF equations) | ΔG°(T,P) and logK for each reaction |
| `qm/` | build time, even more rarely | Python + PySCF + geomeTRIC | ΔG‡ of transition states, Eyring rates, rank upgrade C -> B |
| `ref/` | build time and verification | Python | deterministic ODE solution and stochastic Gillespie SSA of the same network |
| `engine/` | runtime, real time | TypeScript + WGSL | three stages: spatial chemistry -> aggregation gates -> CG-MD, plus metrics |
| `verify/` | on demand, before every delivery | Node + puppeteer-core/CDP, system Chrome | `gates.json`, `report.html` |
| `viewer/` | runtime | three.js | one canvas, one camera through all stages, time-scale badge |

`data/network.json` is the single source of chemical truth: stoichiometry, mass and
charge balance, ΔG, logK, constants, ranks, references. It is generated by the `thermo/`
and `qm/` layers and stored in git as a deterministic artifact.

Validation runs **the same WGSL** as the interactive viewer. There is no second
implementation of the physics, so the two cannot drift apart.

## 5. Stage A: chemistry

**Conditions** are taken from a real experiment so there is something to check against: an
aqueous fluid at 175 °C, carbon source formate or oxalate disproportionating into H₂, CO₂
and CO, a catalytic wall. This is the McCollom, Ritter & Simoneit 1999 setpoint.

**Species** with real formulas and charges: HCOOH and HCOO⁻, oxalate, CO₂(aq), HCO₃⁻,
CO(aq), H₂(aq), H₂O, H⁺ and OH⁻, Na⁺ and Cl⁻ for ionic strength, surface intermediates
(`*CO`, `*C`, `*CH₂`, the growing chain `*Cₙ`), products: n-alkanoic acids, n-alkanols,
n-alkanes.

**Mechanism**: chain growth on an active site, propagation `*Cₙ + *CH₂ -> *Cₙ₊₁` competing
against termination via desorption. The ratio of these rates is α, and the
Anderson-Schulz-Flory distribution `wₙ = n(1−α)²αⁿ⁻¹` arises as a consequence of the
mechanism, not as a law written in by hand. α is fitted **once** at 175 °C and must
thereafter predict the distribution at a different temperature with no new parameters.

**Numerical method**: reactive Brownian dynamics in WGSL. The diffusion coefficient comes
from Stokes-Einstein with the viscosity of water at the given temperature; the reaction
probability on collision is derived from the rate constant through the Smoluchowski
diffusion limit `k_diff = 4πDR`, so the spatial model is required to reproduce bulk
kinetics, and this is checked by gate 3.

**Per-atom visualization.** Ball-and-stick, van der Waals radii, CPK colors, conformer
geometry from RDKit cached in `data/conformers.json`. The frame shows a chain growing on a
wall of CO and H₂ and a finished acid molecule drifting away.

## 6. Stage B: aggregation gate

Stage A supplies a population of acids by chain length, concentration, pH and ionic
strength. Protonation is computed by Henderson-Hasselbalch with an apparent pKa that rises
on aggregation, and the aggregation threshold `log₁₀ CMC = A − B·n`. It is determined which
components exceed the threshold and in what fractions they are passed to stage C.

This stage is allowed to fail. If the chemistry produced only short chains, there will be
no membrane; that is a correct negative result, not a reason to change the parameters.

## 7. Stage C: self-assembly and closure

The Cooke & Deserno 2005 model, solvent-free, three beads per lipid. Parameters are taken
from the paper in full:

- sizes: `b_hh = b_ht = 0.95σ`, `b_tt = σ`;
- WCA repulsion `V_rep = 4ε[(b/r)¹² − (b/r)⁶ + ¼]` for `r ≤ r_c`, `r_c = 2^(1/6)·b`;
- FENE bonds `V_bond = −½ k_bond r_∞² log[1 − (r/r_∞)²]`, `k_bond = 30ε/σ²`, `r_∞ = 1.5σ`;
- bending `V_bend = ½ k_bend (r − 4σ)²`, `k_bend = 10ε/σ²`;
- tail attraction `V_cos = −ε` for `r < r_c`, then `−ε cos²[π(r − r_c)/(2w_c)]` up to
  `r_c + w_c`;
- Langevin thermostat with friction `Γ = 1/τ`, step `δt = 0.01τ`, range `kT/ε = 0.6…1.1`.

Chain length from stage B is mapped to a number of tail beads: C8 -> 2, C12 -> 3, C16 -> 4.
A five-bead extension of the model has been published (parameterized for POPC and DPPC),
but the mapping "number of carbon atoms -> number of beads" here is our own, so it carries
rank D until verified: the mapping is accepted only if the per-lipid area and bilayer
thickness stay within the measured corridor of gate 6; otherwise the gate records the
discrepancy. The other model parameters are left untouched.

**Closure is determined by an outside flood fill.** Grid occupancy, flood fill from the
box boundary, an unfilled cavity is a closed vesicle, and its volume is a number. This test
was chosen over the Euler characteristic of the surface because it is unambiguous and
cheap on the GPU.

## 8. Six gates

A gate is a statement pairing a model number with a literature interval, a verdict and an
evidence rank. The `verify/report.html` report shows all six; the project is considered to
have proven only what passed gates ranked C or better.

**Gate 1: thermodynamics of synthesis.** ΔG_r of the reaction CO₂ + H₂ -> n-alkanoic acid
at 175 °C and fluid activities, computed by pyCHNOSZ against OBIGT. Support: Shock 1998:
when hydrothermal fluid mixes with seawater in the range 250-50 °C, it is thermodynamically
possible to reduce up to 100% of the carbon into a mixture of carboxylic acids, alcohols
and ketones. Criterion: the sign of ΔG and its magnitude in kJ per mole of carbon.

**Gate 2: chain-length distribution.** α is fitted at 175 °C against McCollom, Ritter &
Simoneit 1999 (products C2 to >C35), then predicts the histogram at a different
temperature with no new parameters. Criterion: histogram discrepancy.

**Gate 3: kinetic self-consistency.** The same network, solved three ways (ODE, Gillespie
SSA, spatial Brownian dynamics in WGSL), gives matching concentrations within statistical
error; `k_f/k_r = exp(−ΔG/RT)` is checked numerically. This gate is an engineering check,
but without it the rest are meaningless.

**Gate 4: CMC-versus-chain-length law.** The model must produce `log CMC = A − B·n` with a
slope matching measured points: sodium octanoate (C8), 300 mM; sodium decanoate (C10), 86
mM, giving B ≈ 0.27 per methylene group, corresponding to a transfer energy of about
0.63 kT per CH₂. Two independent quantities are checked against each other.

**Gate 5: pH and salt window.** Vesicles exist only near the apparent pKa: for decanoic
acid, roughly pH 7-9, and the apparent pKa drops by 0.7 as NaCl concentration rises from 10
to 100 mM. A model that assembles a membrane at any pH fails this gate.

**Gate 6: bilayer properties and closure.** Per-lipid area 1.1-1.5 σ², bilayer thickness
about 5σ, bending modulus 5-50 kT from the undulation spectrum
`⟨|h_q|²⟩ = kT/(κq⁴)`, presence of an unfilled cavity and its volume from flood fill.

## 9. Repository structure

```
data/literature.json      gate targets: value, interval, conditions, reference
data/network.json         generated: stoichiometry, charge, ΔG, logK, constants, ranks
data/conformers.json      molecule geometries from RDKit
thermo/                   Python: pyCHNOSZ/OBIGT -> ΔG(T,P), logK
qm/                       Python: PySCF + geomeTRIC -> ΔG‡ -> Eyring rates
ref/                      Python: ODE and Gillespie of the same network
engine/wgsl/              chem_bd, cgmd, neighbor_grid, floodfill
engine/src/               stageA, stageB, stageC, metrics, params
viewer/                   three.js: per-atom and per-bead modes, time-scale badge
verify/                   Node + CDP: run, gates.json, report.html
```

## 10. Technology stack

Runtime: TypeScript, Vite, three.js, physics in WGSL. No UI framework: one canvas and a
thin control panel.

Build layers: Python 3.13 in a dedicated venv (not 3.14: rdkit and pyscf wheels are
unreliable for it; `python@3.13` is already present on the system), plus R with the CHNOSZ
package via brew.

Validation: Node with puppeteer-core, system Chrome in `--headless=new
--enable-unsafe-webgpu` mode.

Measured on this machine (M5 Pro, 16 cores, 48 GB): headless Chrome exposes the
`apple/metal-3` adapter, and the compute pass executes correctly. A trap caught during
measurement: `--dump-dom` dumps the DOM before the asynchronous GPU work finishes and
returns the original content, which makes a live GPU look absent. Metrics must be read via
CDP or an outbound request, not from the DOM dump.

## 11. Implementation order

By layer; each layer is verifiable on its own.

- **L0.** Cooke-Deserno in WGSL: a random lipid solution, self-assembly, measurement of
  per-lipid area and thickness. If the 1.1-1.5 σ² and ≈5σ corridor is not reproduced, there
  is no point going further. The first spike.
- **L1.** Undulation spectrum -> κ; flood fill -> closed-vesicle detector. Gate 6 in full.
- **L2.** `literature.json`, `verify/`, the report. The exam becomes a single command.
- **L3.** `thermo/` on pyCHNOSZ -> `network.json`. Gate 1.
- **L4.** `ref/`: ODE and Gillespie. Gate 3.
- **L5.** Stage A: spatial chemistry in WGSL, per-atom rendering. Gate 2.
- **L6.** Stage B: protonation and CMC. Gates 4 and 5.
- **L7.** `qm/`: PySCF barriers, rank upgrades C -> B with a mandatory control against a
  reaction where a measurement exists.
- **L8.** Phase map from an ensemble of runs over T, pH, salt; optionally, trajectory
  export to Blender 5.2 for a cinematic render.

## 12. Risks

**Access to data for gate 2.** The McCollom 1999 product distributions sit behind a
Springer paywall. Plan: take the numbers from the open 2023 review on PMC. If the
distribution is not there, gate 2 remains in the report as "unproven, no data," and the
reference is handed to the project owner. Gates are not declared passed on indirect
grounds.

**Completeness of OBIGT for long chains.** The database may lack aqueous species for long
n-alkanoic acids. In that case, group contributions are applied with an explicit rank
downgrade to C or D, not a silent extrapolation.

**Neighbor-grid performance on the GPU.** Requires a grid plus prefix sums. Expectation:
30,000 beads in real time, but this will be a measurement, not a promise; the number is
recorded in `verify/report.html`.

**Discrepancy between per-atom visualization and per-bead physics.** Stage A is drawn by
atom, stage C is computed by bead. The mapping from one to the other is labeled in the
interface so the observer does not mistake a bead for an atom.

## 13. Sources

- Cooke & Deserno 2005, "Solvent-free model for self-assembling fluid bilayer membranes,"
  arXiv:cond-mat/0509218: CG model parameters, per-lipid area, thickness, bending modulus.
- McCollom, Ritter & Simoneit 1999, Origins of Life and Evolution of the Biosphere 29,
  153-166: lipid synthesis from formate and oxalate at 175 °C, products C2 to >C35.
- Shock 1998, J. Geophys. Res. Planets, "Organic synthesis during fluid mixing in
  hydrothermal systems": thermodynamic feasibility of carbon reduction.
- Review "Plausible Sources of Membrane-Forming Fatty Acids on the Early Earth," PMC9869395:
  absence of a rate constant for early-Earth conditions, yield estimates.
- Review "Dynamics of the vesicles composed of fatty acids and other amphiphile mixtures,"
  PMC7575682: pH window near the apparent pKa, pKa shift with rising ionic strength, CVC.
- CHNOSZ and the OBIGT database (chnosz.net): thermodynamic data, HKF equations.
