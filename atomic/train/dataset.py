"""
Training dataset: configurations plus teacher labeling.

What goes into the dataset is a decision no less important than the model's form. The
model will only reproduce what it has seen: if it is shown only equilibrium water, it will
not know either compressed contacts or stretched bonds, and in dynamics it will fail
exactly there. So the dataset is built deliberately DIVERSE:

  1. distorted monomers: bond stretching and angle change around equilibrium;
  2. dimers at various distances and orientations: hydrogen bonding across the whole range;
  3. trajectory slices at various temperatures and densities: what actually occurs
     in dynamics, including rare close contacts.

Labeling is energy and forces from the teacher. Forces give 3N equations per
configuration against one from energy, which is why they are the main value of every
expensive labeling run.
"""
from __future__ import annotations

import numpy as np

from engine.md import run
from engine.state import from_symbols

WATER_GEOM = np.array([[0.0, 0.0, 0.0], [0.0, -0.7575, 0.5865], [0.0, 0.7575, 0.5865]])


def water_molecule(rng: np.random.Generator, distortion: float = 0.0) -> tuple[list[str], np.ndarray]:
    """A water molecule with random distortion of bonds and angle."""
    pos = WATER_GEOM.copy()
    if distortion > 0:
        pos[1:] += rng.normal(scale=distortion, size=(2, 3))
    return ["O", "H", "H"], pos


def water_cluster(n_molecules: int, rng: np.random.Generator, density: float = 0.0334):
    """
    A cluster of water molecules in a cube at the density of liquid water (0.0334
    molecules/Å³ at 300 K).

    Molecules are placed on a lattice and rotated randomly, then the lattice is perturbed:
    without the rotations, the dataset would be degenerate in orientation, and the model
    would not see the various relative arrangements of hydrogen bonds.
    """
    from scipy.spatial.transform import Rotation

    side = int(np.ceil(n_molecules ** (1 / 3)))
    box = (n_molecules / density) ** (1 / 3)
    step = box / side
    symbols: list[str] = []
    positions = []
    placed = 0
    for a in range(side):
        for b in range(side):
            for c in range(side):
                if placed >= n_molecules:
                    break
                centre = np.array([a, b, c]) * step + rng.normal(scale=0.15, size=3)
                rot = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
                mol = (WATER_GEOM - WATER_GEOM[0]) @ rot.T + centre
                positions.append(mol)
                symbols += ["O", "H", "H"]
                placed += 1
    return symbols, np.vstack(positions), np.eye(3) * box


def label_frames(states, teacher) -> list[dict]:
    """Labels a list of states with the teacher: energy and forces."""
    frames = []
    for st in states:
        res = teacher.compute(st)
        frames.append(
            {
                "positions": st.positions.copy(),
                "numbers": st.numbers.copy(),
                "energy": res.energy_ev,
                "forces": res.forces_ev_per_a.copy(),
                "cell": None if st.cell is None else st.cell.copy(),
            }
        )
    return frames


def build_water_dataset(
    teacher,
    rng: np.random.Generator,
    n_monomers: int = 40,
    n_dimers: int = 60,
    cluster_sizes: tuple[int, ...] = (4, 8),
    md_frames_per_cluster: int = 20,
    md_temperature_k: float = 400.0,
    md_steps_between: int = 20,
) -> list[dict]:
    """
    Full dataset for water. The slice temperature is deliberately set ABOVE the working
    one (400 K against 300 K): the dataset must cover a wider range than the run, otherwise
    the model would be operating at the edge of its domain wherever the dynamics goes a
    bit further.
    """
    states = []

    for _ in range(n_monomers):
        sym, pos = water_molecule(rng, distortion=float(rng.uniform(0.0, 0.18)))
        states.append(from_symbols(sym, pos))

    for _ in range(n_dimers):
        from scipy.spatial.transform import Rotation

        sep = float(rng.uniform(2.4, 5.5))
        rot = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        mol_a = WATER_GEOM - WATER_GEOM[0]
        mol_b = (WATER_GEOM - WATER_GEOM[0]) @ rot.T + np.array([sep, 0.0, 0.0])
        states.append(from_symbols(["O", "H", "H", "O", "H", "H"], np.vstack([mol_a, mol_b])))

    frames = label_frames(states, teacher)

    # --- trajectory slices: what actually occurs in dynamics -------------------------
    for n_mol in cluster_sizes:
        sym, pos, cell = water_cluster(n_mol, rng)
        st = from_symbols(sym, pos, cell=cell, pbc=(True, True, True))
        st.set_maxwell_boltzmann(md_temperature_k, rng)
        for _ in range(md_frames_per_cluster):
            run(
                st, teacher, steps=md_steps_between, dt_fs=0.5,
                temperature_k=md_temperature_k, seed=int(rng.integers(1 << 30)),
                sample_every=md_steps_between, watch_chemistry=False,
            )
            frames += label_frames([st.copy()], teacher)

    return frames
