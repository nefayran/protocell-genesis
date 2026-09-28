"""
State of the atomistic system: atom species by atomic number Z, positions, velocities,
cell. Nothing about the potential and nothing about the integrator, only what physically
EXISTS.

Why Z rather than string species names: the engine must work on any elements, and Z IS
the element, with no lookup table that would need to be maintained. Masses are taken
from the ASE atomic weight table (standard IUPAC atomic weights) rather than set
manually: the atom species fully determines the mass.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from ase.data import atomic_masses, atomic_numbers, chemical_symbols

from .units import instantaneous_temperature_k, kinetic_energy_ev


def masses_for(numbers: np.ndarray) -> np.ndarray:
    """Standard atomic weights (amu) for an array of Z."""
    z = np.asarray(numbers, dtype=int)
    if z.min() < 1 or z.max() >= len(atomic_masses):
        raise ValueError(f"Z outside the element table: min={z.min()}, max={z.max()}")
    return atomic_masses[z].astype(float)


def numbers_from_symbols(symbols) -> np.ndarray:
    """['O','H','H'] -> [8,1,1]. An unknown symbol is an error, not a silent skip."""
    out = []
    for s in symbols:
        if s not in atomic_numbers:
            raise ValueError(f"unknown chemical symbol: {s!r}")
        out.append(atomic_numbers[s])
    return np.asarray(out, dtype=int)


@dataclass
class AtomicState:
    """
    Instantaneous state of the system. Positions in Å, velocities in Å/fs, cell in Å.

    `cell` and `pbc` are deliberately kept separate: a non-periodic calculation (a
    molecule in vacuum, a droplet) and a periodic one are different physical setups, and
    this project already had a measured case where a wrapped distance computed in a
    non-periodic system silently corrupted the result. Periodicity must be declared
    explicitly for each axis here.
    """

    numbers: np.ndarray                       # (N,) int, atomic numbers
    positions: np.ndarray                     # (N,3) float, Å
    velocities: np.ndarray | None = None      # (N,3) float, Å/fs; None -> zeros
    cell: np.ndarray | None = None            # (3,3) float, Å; None -> no cell
    pbc: tuple[bool, bool, bool] = (False, False, False)
    charge: int = 0                           # total system charge, e
    spin_multiplicity: int = 1                # 2S+1; 1 = closed-shell singlet
    frozen: np.ndarray | None = None          # (N,) bool, atoms with zeroed force
    masses: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.numbers = np.asarray(self.numbers, dtype=int)
        self.positions = np.asarray(self.positions, dtype=float).reshape(len(self.numbers), 3)
        if self.velocities is None:
            self.velocities = np.zeros_like(self.positions)
        else:
            self.velocities = np.asarray(self.velocities, dtype=float).reshape(self.positions.shape)
        if self.cell is not None:
            self.cell = np.asarray(self.cell, dtype=float).reshape(3, 3)
        if any(self.pbc) and self.cell is None:
            raise ValueError("periodicity declared but there is no cell")
        if self.frozen is not None:
            self.frozen = np.asarray(self.frozen, dtype=bool).reshape(len(self.numbers))
        self.masses = masses_for(self.numbers)

    # --- convenience views ----------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.numbers)

    @property
    def symbols(self) -> list[str]:
        return [chemical_symbols[z] for z in self.numbers]

    @property
    def formula(self) -> str:
        """Gross formula in order of decreasing count: for labels and logs."""
        uniq, counts = np.unique(self.numbers, return_counts=True)
        order = np.argsort(-counts)
        return "".join(
            f"{chemical_symbols[uniq[i]]}{counts[i] if counts[i] > 1 else ''}" for i in order
        )

    def copy(self) -> "AtomicState":
        return replace(
            self,
            numbers=self.numbers.copy(),
            positions=self.positions.copy(),
            velocities=self.velocities.copy(),
            cell=None if self.cell is None else self.cell.copy(),
            frozen=None if self.frozen is None else self.frozen.copy(),
        )

    # --- measurable quantities ------------------------------------------------------------
    def kinetic_energy_ev(self) -> float:
        return kinetic_energy_ev(self.masses, self.velocities)

    def temperature_k(self, n_constraints: int = 3) -> float:
        """
        By default three degrees of freedom are removed: the engine zeroes the total
        momentum when setting velocities (see `set_maxwell_boltzmann`), so the
        translational motion of the system as a whole is not a thermal degree of
        freedom.
        """
        return instantaneous_temperature_k(self.masses, self.velocities, n_constraints)

    def total_momentum(self) -> np.ndarray:
        return (self.masses[:, None] * self.velocities).sum(axis=0)

    def centre_of_mass(self) -> np.ndarray:
        return (self.masses[:, None] * self.positions).sum(axis=0) / self.masses.sum()

    def remove_net_momentum(self) -> None:
        """Zeroes the total momentum while preserving the kinetic energy of relative motion."""
        drift = self.total_momentum() / self.masses.sum()
        self.velocities -= drift

    def set_maxwell_boltzmann(self, temperature_k: float, rng: np.random.Generator) -> None:
        """
        Maxwell-Boltzmann velocities at the given T, with the total momentum zeroed and
        then RE-scaled to the target T. The order matters: zeroing the momentum removes
        part of the kinetic energy, so rescaling must happen AFTER it, otherwise the
        starting temperature is systematically below the requested one, exactly the
        error that the equipartition test in atomic/validate catches.
        """
        sigma = np.sqrt(
            temperature_k * 1.380649e-23 / (self.masses * 1.66053906892e-27)
        )  # m/s
        self.velocities = rng.normal(size=self.positions.shape) * (sigma[:, None] * 1e-5)  # Å/fs
        self.remove_net_momentum()
        current = self.temperature_k()
        if current > 0:
            self.velocities *= np.sqrt(temperature_k / current)

    # --- geometry ----------------------------------------------------------------------
    def distance(self, i: int, j: int) -> float:
        """Distance i-j with the minimum image ONLY along periodic axes."""
        d = self.positions[j] - self.positions[i]
        if self.cell is not None and any(self.pbc):
            frac = np.linalg.solve(self.cell.T, d)
            for axis in range(3):
                if self.pbc[axis]:
                    frac[axis] -= np.round(frac[axis])
            d = self.cell.T @ frac
        return float(np.linalg.norm(d))

    def angle_deg(self, i: int, j: int, k: int) -> float:
        """Angle i-j-k in degrees (j is the vertex)."""
        a = self.positions[i] - self.positions[j]
        b = self.positions[k] - self.positions[j]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def from_symbols(symbols, positions, **kwargs) -> AtomicState:
    """Build a state from symbols and coordinates in Å."""
    return AtomicState(numbers=numbers_from_symbols(symbols), positions=positions, **kwargs)
