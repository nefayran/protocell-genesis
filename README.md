# protocell-genesis

Can a coarse-grained "primordial soup" of monomers assemble itself into a closed vesicle, a bilayer membrane with
water trapped inside? This project built a WebGPU simulation to find out, fixed the pass criteria before running it,
and got a negative answer with a measured cause.

## Result

No closed vesicle formed in any snapshot of any campaign.

The largest campaign ran a 76 σ box with 483,268 particles, liquid water at density 0.80, charge at pH 7.0 and
ionic strength 0.01 M, for 148,200 steps, starting from a lattice of monomers. It grew one aggregate holding 99.93%
of the amphiphiles. That aggregate percolated through the periodic box along all three axes in 23 of 23 wet
snapshots, and it trapped no water at all.

Four explanations were tested and ruled out by measurement:

| explanation | test | outcome |
|---|---|---|
| too much material merges into one object | supply lowered to 1.50× the closure threshold, then raised to 4.82× | 3 of 3 wrapping axes both times |
| a finite object pays too much for its edge | edge energy measured at 0.583 kT per molecule (N = 2,087) | below the ≲ 1 kT that would matter |
| head groups repel too weakly | the pair-energy gap narrowed from 2.86× to 1.26× | wrapping axes moved by 0 |
| the box is too small | box volume ×2.79 at constant density (54 σ to 76 σ) | 3 of 3 axes, 27 of 27 layers, in all 23 snapshots |

A final run put the soup in a finite parcel of water with a neutral wall, which removes the periodic topology. The
aggregate stopped wrapping (0 of 3 axes, the first time that gate passed), and closure still did not happen:
encapsulated water was exactly 0 in 19 of 19 snapshots. The periodic boundary was a consequence, not the cause.

What is left is the molecule. In a finite region the cheapest object without an edge is a capped cylindrical
micelle, which costs 17.5 to 21.3 amphiphiles, while a closed vesicle needs at least 937 to 1,014. With this
two-tailed amphiphile and its fixed short tail, the packing parameter favours micelles by a factor of 44 to 58.

What did agree with the literature:

- bilayer area per lipid in explicit water, 1.1607 σ² (corridor 1.1 to 1.5), and thickness, 4.6871 σ (corridor 4 to 6);
- the one independent check in the project: the salt-induced shift of the apparent pKa, +0.709 against about 0.7.

The bending modulus stayed unproven because its fit window was invalid.

## Rules the project held itself to

- Every gate is a claim with a model value, a literature corridor, a verdict and an evidence rank: A measured,
  B computed, C bounded by thermodynamics or an independent ratio, D estimated. A gate that rests only on D evidence
  is unproven by definition. Gates are published as passed, failed or unproven, and they are never widened to pass.
- `engine/` contains no hand-written chemical constants; it reads them from `data/`.
- Hydrophobicity has to emerge from water displacement, never from a direct attraction between tails.
- The bond attempt rate is not tuned to get a result.

## What is in the repository

| layer | what it does | where |
|---|---|---|
| soup | bond chemistry on the GPU: C–C chain growth on a catalyst, head termination, valence, dehydration and rehydration, solvent evaporation, a clay plate | `soup/` |
| explicit water | water as its own bead type, with attraction by class (solvent, polar, apolar) and depth ratios taken from Martini v2.1 | `soup/wgsl/step.wgsl`, `data/soup.json` |
| electrostatics | screened Coulomb (Debye–Hückel) between heads; charge is a Monte Carlo variable at constant pH | `soup/src/electrostatics.ts` |
| measurement | amphiphile detection, aggregate decomposition, shape from the inertia tensor, head layers, cavity, encapsulated water, box wrapping | `soup/src/aggregates.ts`, `soup/src/water-closure.ts` |
| gates | definitions in `data/literature.json`, verdict rules in `verify/gates.ts`, one run of all of them in `verify/run.ts` | `verify/` |
| viewer | a run page with start, pause and stop, live measurements, the aggregate breakdown and a cavity highlight | `viewer/` |
| long runs | a campaign CLI with checkpoints and resume | `soup/cli/campaign.ts` |
| atomic engine | Python molecular dynamics where no reaction is listed: bonds come from electronic-structure bond orders and a reaction is a change in connectivity. GFN2-xTB (tblite) drives trajectories, Hartree–Fock and DFT (pyscf) give reference energies | `atomic/` |

The atomic detail shown in the viewer is a reconstruction on top of coarse-grained coordinates, not all-atom
dynamics, and the page says so.

### A learned interatomic potential

`atomic/train/` distils a fast potential from a slower teacher (MACE-OFF23 in the final runs). Against experiment it
gets the O–H bond length of water slightly closer than its teacher does (0.9585 Å and 0.9589 Å against 0.9572 Å),
the H–O–H angle slightly further (105.05° and 104.96° against 104.52°), and it underestimates the hydrogen-bond
energy by 14%. Force error is 65 meV/Å on frames from dynamics and 109 meV/Å on random frames. It runs 11 to 31
times faster than the teacher.

On a fatty acid in water the model alone breaks apart within 0.2 to 0.4 ps. A committee that calls the teacher on
uncertain frames (0.8% to 6.5% of steps) survived 1,500 steps in one run and 169 in the repeat, so its stability is
not claimed from a single run. The trained weights are not included: they were fitted to MACE-OFF23 labels, and the
licence terms of those models differ from this repository's.

## Six silent failures

Each of these looked like a success. They are specific to WebGPU and worth knowing before you trust a GPU
simulation:

1. `readBack` from a buffer without `COPY_SRC` returns the zero-initialised staging buffer. 72 checkpoints were
   saved with a zeroed random-number state before this was caught.
2. A comparison with NaN is always false, so the divergence guard passed forever.
3. `requestDevice()` without `requiredLimits` quietly caps every pipeline at the default storage limits.
4. A stale force table after a box-size change does not produce zeros; it produces a wrong number.
5. Gates that recognised their input by the run's label went silent on a new experiment.
6. A buffer request above the device limit raised no JavaScript error. WebGPU logged a validation warning, every
   dispatch on that bind group became a no-op, and the run reported zero forces, zero non-finite values and
   0.04 ms per step: a 500× "speed-up" with a clean health check.

A seventh came from the test tooling: reading a WebGPU canvas through `drawImage` returned 0 changed pixels from a
page that was visibly drawing. Zeros from an instrument are first a reading about the instrument.

## Running it

Node 18 or newer and a browser with WebGPU for the viewer.

```bash
npm install
npm test            # vitest
npm run verify      # rebuilds the gate table from the traces in verify/out/: gates.json and report.html, about 3 minutes
npm run dev         # the viewer
```

Campaign checkpoints are gigabytes and are not in the repository. A campaign can be re-run with
`soup/cli/campaign.ts`; the full command for the largest one is in `docs/soup-to-vesicle-verdict.md`, section 10.

The atomic engine uses its own Python environment (Python 3.13 with numpy, scipy, ase, tblite, pyscf and pytest).
Do not install `pyscf-dispersion`: it is incompatible with numpy 2.5 and breaks the pyscf import.

## Documents

- `docs/soup-to-vesicle-verdict.md`: the full verdict, mechanism, defects found and what to change next (Russian).
- `atomic/README.md`: the atomic engine and the learned potential (Russian).
- `.superpowers/sdd/2026-08-16-soup-to-vesicle/`: the working reports behind each step (English).
- `CLAUDE.md`: working rules for the coding agent used on the project (Russian).

## License

MIT. tblite, pyscf, ase and MACE-OFF23 are separate projects under their own licences.
