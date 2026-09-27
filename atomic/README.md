# Atomic engine: chemistry and physics at the level of atoms

English translation of [`README.ru.md`](README.ru.md).

The task as set: an engine that works at the atomic level, for any elements, and reproduces
processes, instead of the previous layer, where the chemistry came down to four invented rules with
invented rates.

The key difference from the previous engine is that there is not a single reaction here. A bond
exists if its bond order from the electronic structure exceeds a threshold; a molecule is a connected
component of the bond graph; a reaction is an event in which the connectivity changed. What happens
is decided by the Hamiltonian, and `engine/chem.py` only reads the result. So "reproduces any
processes" is not a promise here but a consequence of the processes never having been listed.

## What it consists of

| file | responsibility |
|---|---|
| `engine/units.py` | units Å/fs/amu/eV; conversion factors derived from the SI and CODATA definitions |
| `engine/state.py` | state: Z, coordinates, velocities, cell, charge, spin, frozen atoms |
| `engine/potential.py` | potential interface plus analytic potentials for checking the integrator |
| `engine/integrate.py` | NVE (velocity Verlet), NVT (BAOAB), L-BFGS geometry optimisation |
| `engine/chem.py` | connectivity from bond orders, molecule detection, reaction logging |
| `engine/md.py` | the driver loop, which fails loudly on a non-numeric state |
| `engine/backends/gfn2.py` | GFN2-xTB through tblite: elements up to Z = 86, dynamics |
| `engine/backends/scf.py` | Hartree–Fock and DFT through pyscf: reference accuracy |

The split between backends is not a matter of taste but measured (see the table below): GFN2 drives
the trajectory, the reference method computes the energetics of bond breaking and radicals.

## What was checked, and against what

Every number below comes from runs in `validate/` on this machine, not from the literature on the
method. Experimental values are from reference works (Huber & Herzberg; Benedict, Gailar & Plyler
1956; Herzberg; Harmony 1990; Klopper et al. 2000; Ruscic, Active Thermochemical Tables).

### Engine physics (an analytic potential with a known derivative)

| check | result |
|---|---|
| conversion factor amu·Å²/fs² to eV | 103.6426957, derived, not typed in |
| force against numerical gradient, harmonic | difference < 1e-6 |
| force against numerical gradient, Lennard-Jones | relative < 1e-4 |
| momentum conservation in NVE | drift < 1e-10 |
| order of accuracy in the time step | energy error falls as dt² (drift ratio 2–8 when the step is halved) |
| period of a harmonic oscillator | matches the analytic value within 1 % |
| BAOAB thermostat | holds the set T within 10 % |

### Chemistry (GFN2-xTB, equilibrium geometry)

| quantity | engine | experiment |
|---|---|---|
| r(H–H) in H₂ | 0.7767 Å | 0.7414 Å |
| r(H–H) in H₂, reference HF/def2-SVP | 0.741 Å | 0.7414 Å |
| r(O–H), H–O–H angle in H₂O | within 0.03 Å and 3° | 0.9572 Å, 104.52° |
| r(C–H), angles in CH₄ | within 0.03 Å and 2° | 1.0870 Å, 109.47° |
| r(C–C) ethane / r(C=C) ethene / r(C≡C) ethyne | within 0.03 Å | 1.5351 / 1.3390 / 1.2033 Å |
| bond multiplicity (1, 2, 3) | read from the bond order, set nowhere | — |
| binding energy of the water dimer | 3–8 kcal/mol | 5.02 kcal/mol |
| hydrogen bond | seen as a link between two molecules, not as a covalent bond | — |

### The measured limit of the cheap method

| quantity | GFN2 | reference B3LYP/def2-SVP | experiment |
|---|---|---|---|
| D(O–O) in H₂O₂ → 2 OH | 5.386 eV | 2.481 eV | 2.15 eV |
| r(O–O) in H₂O₂ | 1.4209 Å | 1.4420 Å | 1.4556 Å |

GFN2 overestimates the homolytic bond-breaking energy by a factor of 2.4: a tight-binding
semi-empirical method without spin polarisation describes radicals poorly. Hence the engine's working
rule, which a test pins down: dynamics with GFN2, the energetics of bond breaking and radicals with the
reference method. An open shell is computed with a spin-unrestricted method (U), not a restricted one
(RO); the difference is not cosmetic, it is what gives the right bond-breaking energy.

### Reactions detected by the engine

| check | result |
|---|---|
| water at 300 K, 1000 steps | not one false reaction, composition unchanged |
| Zundel ion H₅O₂⁺, 350 K | the bridging proton moves between the oxygens; the system charge is conserved exactly |
| stretching O–O | the bond disappears from the connectivity, and the system reads as two OH particles |

## Performance, measured

Water cluster, GFN2, NVT, 0.5 fs step, `nice -n 15` on this machine:

| molecules | atoms | steps/s | ms/step |
|---|---|---|---|
| 1 | 3 | 67.7 | 14.8 |
| 4 | 12 | 36.0 | 27.8 |
| 16 | 48 | 12.0 | 83.6 |
| 32 | 96 | 5.6 | 177.3 |
| 64 | 192 | 1.6 | 607.3 |

## The honesty ceiling

Read this before expecting results from the engine.

- Scale. At 192 atoms it runs 1.6 steps per second, about 0.8 fs of model time per second of compute.
  An hour of compute gives about 3 picoseconds. Membrane self-assembly needs 10⁵–10⁶ atoms and
  microseconds; that is out of reach with quantum forces on any hardware, not only on this machine.
  The atomic level is there to compute chemistry (energies, barriers, rate constants), not to run the
  assembly.
- Rare events. A barrier of a few electronvolts is not crossed by direct dynamics in any available
  time; this was measured (2000 steps at 6000 K on H₂O₂ give not one bond break). That needs enhanced
  sampling: NEB for barriers, metadynamics for free energy. The engine does not have them yet.
- Accuracy. GFN2 is semi-empirical: organic bond lengths are off by hundredths of an ångström, and
  radicals are poor (measured above). The reference DFT in a small basis gives bond-breaking energies
  with an error of about 10 % and without a zero-point correction.
- What is missing entirely: periodic DFT (only GFN2 handles periodic systems), excited states,
  solvation models, enhanced sampling, machine-learned potentials for scale.

## How to run the checks

```sh
cd atomic
.venv/bin/python -m pytest validate/ -q -m "not slow"
```

Environment: a separate venv in `atomic/.venv` (python 3.13) with numpy, scipy, ase, tblite, pyscf and
pytest. Do NOT install the `pyscf-dispersion` package: it is incompatible with numpy 2.5 and breaks the
pyscf import entirely (checked).

## Our own potential: what came out of it and what it cost

The task was set as an engine that runs our volumes and stays accurate. Below is what of that was
measured, including what did not work.

### Accuracy against the teacher (MACE-OFF23) and against experiment

| quantity | our potential | teacher | experiment |
|---|---|---|---|
| r(O–H) in water | **0.9585 Å** | 0.9589 Å | 0.9572 Å |
| H–O–H angle | 105.05° | 104.96° | 104.52° |
| hydrogen-bond energy | 4.29 kcal/mol | 4.96 | 5.02 |
| energy error (frames from dynamics) | 11.5 meV/atom | — | — |
| force error (frames from dynamics) | 65.0 meV/Å | — | — |
| force error (random frames) | 108.6 meV/Å | — | — |
| energy drift, NVE 400 steps | 6.6 % of the kinetic energy | — | — |

The model reproduces the water O–H bond length closer to experiment than its teacher does; the angle
comes out slightly further off. The weak point is named with a number: the hydrogen-bond energy is 14 %
too low.

### Speed: 11–31 times the teacher, and why not 1000

| atoms | teacher, ms | ours, ms | speed-up |
|---|---|---|---|
| 24 | 102.2 | 3.3 | **31.1×** |
| 81 | 197.6 | 17.9 | 11.1× |
| 192 | 886.8 | 55.7 | 15.9× |
| 375 | 1152.7 | 82.0 | 14.1× |

The measurement is honest: the coordinates are moved BEFORE EVERY call. Without that ASE returns its
cache and the teacher shows 0.1 ms. I fell into this trap twice, the second time with its description
already in this file, so it gets its own line here.

The path to these numbers was not direct, and the main step was dropping the intermediate tensor: the
calculation needs forces, not descriptor derivatives. Taking dE/dG first (one cheap pass through the
network), the forces are assembled directly in the loop over pairs and triples, and the N×D×N×3 tensor
(tens of millions of numbers at 192 atoms) never appears. Before this change our potential was SLOWER
than the teacher at 192 atoms (0.90×): the whole gain went into memory.

The remaining headroom is measured and named: the arithmetic at 192 atoms needs about seven million
operations, less than a millisecond, against the current 105. The difference is about a hundred kernel
launches per calculation (twelve angular combinations × three axes × three accumulations). That is
removed by a port to WGSL, not by more work on the tensors: removing the loop over atoms gave nothing
(20.3× against 21.9× before it), so the bottleneck was not there.

### Dynamics: how much model time per hour

The fused path is built into the backend `engine/backends/student.py`, so it runs through the same
integrator as the quantum calculators.

| system | our potential | GFN2 (quantum) |
|---|---|---|
| 192 atoms | **37.9 steps/s, 68.3 ps/hour** | 1.6 steps/s, 3 ps/hour |
| 648 atoms | 10.8 steps/s, 19.5 ps/hour | — |

So at close to quantum accuracy we get about **23 times more model time** than semi-empirical quantum
mechanics gives on the same system.

**Metal (MPS) turned out SLOWER than the CPU:** 18.0 steps/s against 37.9 at 192 atoms, and 8.6 against
10.8 at 648. The reason is the same as with the cloud GPU: the calculation is many small operations,
and the kernel-launch overhead eats the gain. The devices agree with each other (ΔE 1.5e-5 eV at 192
atoms, Δforces 6.0e-6 eV/Å), so this is a question of speed, not of correctness. The conclusion for
the port: the gain comes not from "moving it to the GPU" but from **one large kernel** instead of
dozens of small ones.

### Three levers that work against intuition

Each was tested on the same data, changing one quantity:

1. **A larger cutoff radius hurts.** 6 Å instead of 5 made the forces worse, from 71.5 to 90.6 meV/Å:
   with more surroundings at the same capacity and the same data, the model smears.
2. **Network capacity hardly matters.** Tripling it (48×48 to 128×128×64) gave 85.4 to 83.8 meV/Å.
3. **Angular resolution does not help.** 3 multiplicities and 2 widths give 85.2; five multiplicities
   give 90.5, four widths 90.1, and both together 91.2.

What works is one thing: **radial resolution** (8 to 16 functions: 109.1 to 71.5, then saturation) and
the **amount of data** for the energy (276 to 8.45 meV/atom as the set grew 13 times, while the forces
did not move). Symmetry functions level off at about 65 meV/Å; getting lower needs a different form of
descriptor (ACE-like, with real angular momenta), not more of the same.

### The trap I built myself

Decoupling the set of configurations from the dynamics sped the pipeline up about 2200 times, and lost
what the dynamics gave for free: **physical geometry**. Random sampling put atoms 0.394 Å apart, forces
reached 1.1e7 eV/Å, and training went to garbage (residual 2.9e15): a quarter of the frames were
ruined. The protection is now twofold: distance limits over ALL pairs during sampling (0.75 Å for bonded
pairs, 1.35 Å for non-bonded ones) and rejection of forces above 50 eV/Å after labelling, with the
number of rejected frames always printed.

### Where to compute: a measurement, not a habit

| where | teacher speed |
|---|---|
| this CPU | **460** atom-calculations/s |
| Kaggle, CPU | 103 |
| Kaggle, Tesla P100 | 22–92 |

The cloud loses outright for this task: on systems of tens of atoms the overhead is larger than the
computation itself, and the big card sits idle. Batched labelling on the CPU hardly helps either, and
with large batches it hurts (413, 523, 288 atom-calculations/s at batches of 1, 8, 64). The cloud makes
sense where the work is large and uniform, not on small clusters.
