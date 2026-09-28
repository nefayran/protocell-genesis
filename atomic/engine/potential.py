"""
Potential interface and analytical potentials for VALIDATING the engine.

The separation is fundamental. Real chemistry comes from the electronic structure
(atomic/engine/backends), but the integrator cannot be validated on it: there is no
closed-form expression there against which the force could be checked. That is why the
engine always has an analytical potential with an exact derivative: it is used to check
energy conservation, momentum conservation, and agreement of the force with the
numerical gradient. If these three checks pass on the analytical potential while the
result on the quantum backend looks wrong, the issue is in the backend, not the
integrator, and vice versa. Without this separation, any bug looks like "that's just
how the physics is."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .state import AtomicState


@dataclass
class PotentialResult:
    """
    Energy in eV, forces in eV/Å. `extra` is everything the backend can provide beyond
    that (bond orders, charges, dipole): chemistry is taken from there, not from our
    own rules.
    """

    energy_ev: float
    forces_ev_per_a: np.ndarray
    extra: dict = None

    def __post_init__(self) -> None:
        self.forces_ev_per_a = np.asarray(self.forces_ev_per_a, dtype=float)
        if self.extra is None:
            self.extra = {}
        if not np.isfinite(self.energy_ev) or not np.all(np.isfinite(self.forces_ev_per_a)):
            raise FloatingPointError(
                "potential returned a non-numeric value: "
                f"energy={self.energy_ev}, non-numeric force components="
                f"{int((~np.isfinite(self.forces_ev_per_a)).sum())}"
            )


class Potential(Protocol):
    """Anything that can compute energy and forces for a state."""

    name: str

    def compute(self, state: AtomicState) -> PotentialResult: ...


def numerical_forces(potential: Potential, state: AtomicState, h: float = 1e-4) -> np.ndarray:
    """
    Forces from the central difference of energy: -dE/dx. The only way to check a force
    without trusting its derivation. Costs 6N energy evaluations, so it is used on small
    systems in tests, not in production runs.
    """
    forces = np.zeros_like(state.positions)
    probe = state.copy()
    for i in range(state.n_atoms):
        for axis in range(3):
            saved = probe.positions[i, axis]
            probe.positions[i, axis] = saved + h
            e_plus = potential.compute(probe).energy_ev
            probe.positions[i, axis] = saved - h
            e_minus = potential.compute(probe).energy_ev
            probe.positions[i, axis] = saved
            forces[i, axis] = -(e_plus - e_minus) / (2 * h)
    return forces


class LennardJones:
    """
    Pairwise Lennard-Jones with an energy shift at the cutoff radius.

    The shift is mandatory, not cosmetic: without it, energy jumps discontinuously when
    a pair crosses the cutoff radius, and energy conservation in NVE breaks in steps;
    this is a classic bug that the NVE test must catch. The force is not shifted (the
    shift is a constant), so the numerical gradient and the analytical force agree.
    """

    name = "lennard-jones"

    def __init__(self, epsilon_ev: float, sigma_a: float, cutoff_a: float | None = None) -> None:
        self.epsilon = float(epsilon_ev)
        self.sigma = float(sigma_a)
        self.cutoff = float(cutoff_a) if cutoff_a is not None else 3.0 * self.sigma
        sr6 = (self.sigma / self.cutoff) ** 6
        self.energy_shift = 4.0 * self.epsilon * (sr6 * sr6 - sr6)

    def compute(self, state: AtomicState) -> PotentialResult:
        pos = state.positions
        n = len(pos)
        energy = 0.0
        forces = np.zeros_like(pos)
        for i in range(n - 1):
            d = pos[i + 1 :] - pos[i]
            if state.cell is not None and any(state.pbc):
                frac = np.linalg.solve(state.cell.T, d.T).T
                for axis in range(3):
                    if state.pbc[axis]:
                        frac[:, axis] -= np.round(frac[:, axis])
                d = (state.cell.T @ frac.T).T
            r = np.linalg.norm(d, axis=1)
            mask = (r < self.cutoff) & (r > 0)
            if not mask.any():
                continue
            rm = r[mask]
            sr6 = (self.sigma / rm) ** 6
            energy += float((4.0 * self.epsilon * (sr6 * sr6 - sr6) - self.energy_shift).sum())
            # dU/dr = 4e(-12 s^12/r^13 + 6 s^6/r^7); f = -dU/dr * (d/r)
            dudr = 4.0 * self.epsilon * (-12.0 * sr6 * sr6 + 6.0 * sr6) / rm
            fmag = (-dudr / rm)[:, None] * d[mask]
            forces[i] -= fmag.sum(axis=0)
            idx = np.arange(i + 1, n)[mask]
            forces[idx] += fmag
        return PotentialResult(energy, forces)


class HarmonicBonds:
    """
    A set of harmonic bonds: U = sum k/2 (r - r0)^2. The exact analytical solution for
    two atoms is known, so it is used to check both the vibration period and energy
    conservation at different step sizes: a direct test of the integrator's order of
    accuracy.
    """

    name = "harmonic-bonds"

    def __init__(self, bonds: list[tuple[int, int, float, float]]) -> None:
        # (i, j, k in eV/Å^2, r0 in Å)
        self.bonds = [(int(i), int(j), float(k), float(r0)) for i, j, k, r0 in bonds]

    def compute(self, state: AtomicState) -> PotentialResult:
        forces = np.zeros_like(state.positions)
        energy = 0.0
        for i, j, k, r0 in self.bonds:
            d = state.positions[j] - state.positions[i]
            r = float(np.linalg.norm(d))
            if r == 0.0:
                raise FloatingPointError(f"coincident atoms {i} and {j} in a harmonic bond")
            energy += 0.5 * k * (r - r0) ** 2
            f = -k * (r - r0) * d / r
            forces[j] += f
            forces[i] -= f
        return PotentialResult(energy, forces)


class SumPotential:
    """Sum of potentials: energies are added, forces are added. Nothing else."""

    def __init__(self, *parts: Potential) -> None:
        self.parts = parts
        self.name = "+".join(p.name for p in parts)

    def compute(self, state: AtomicState) -> PotentialResult:
        energy = 0.0
        forces = np.zeros_like(state.positions)
        extra: dict = {}
        for part in self.parts:
            res = part.compute(state)
            energy += res.energy_ev
            forces += res.forces_ev_per_a
            extra.update(res.extra)
        return PotentialResult(energy, forces, extra)
