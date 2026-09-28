"""
Chemistry as an OBSERVATION, not a rule.

There is not a single reaction defined here. A bond is considered to exist if its order
from the electronic structure exceeds a threshold; a molecule is a connected component of
the bond graph; a reaction is an event where connectivity changed. This is why the engine
"reproduces any process" not because we enumerated the processes, but because we did not:
what happens is decided by the Hamiltonian, and this file only reads the result.

The bond-order threshold is the single free number, and it is NOT tuned to the desired
answer. Threshold check: on a set of molecules with known structure (water, methane,
ethane, ethylene, acetylene, hydrogen peroxide, acetic acid), connectivity must match the
textbook one, including bond multiplicity. If the threshold is wrong, that shows up as an
extra or missing bond in a known molecule, not as a strange result at the end of a run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import chemical_symbols

# Bond-existence threshold on bond order. The value is chosen so that on the set of known
# molecules (test in atomic/validate/test_chem.py) connectivity matches the textbook one:
# clearly below a single bond (order around 1), clearly above the residual overlap of
# unbonded pairs (order around 0.05 or less).
BOND_ORDER_THRESHOLD = 0.5

# Multiplicity thresholds: a bond counts as double from 1.6 and triple from 2.4. These are
# NOT physical constants but boundaries for READING the bond order as an integer, needed
# only for a human-readable label; all the dynamics runs on the continuous order.
DOUBLE_BOND_MIN = 1.6
TRIPLE_BOND_MIN = 2.4


@dataclass(frozen=True)
class Bond:
    i: int
    j: int
    order: float

    @property
    def multiplicity(self) -> int:
        if self.order >= TRIPLE_BOND_MIN:
            return 3
        if self.order >= DOUBLE_BOND_MIN:
            return 2
        return 1


@dataclass
class Species:
    """Molecule: a set of atoms, gross formula, total Mulliken charge."""

    atom_indices: tuple[int, ...]
    formula: str
    charge: float

    @property
    def size(self) -> int:
        return len(self.atom_indices)


def bonds_from_orders(
    bond_orders: np.ndarray, threshold: float = BOND_ORDER_THRESHOLD
) -> list[Bond]:
    """Upper triangle of the bond-order matrix -> list of bonds."""
    bo = np.asarray(bond_orders, dtype=float)
    # tblite returns the matrix with an EXTRA spin axis (N, N, n_spin): for a closed-shell
    # system this is (N, N, 1). We sum over spin channels rather than take the first one:
    # for an open-shell system the total bond order is the sum over channels, and taking
    # a single channel would give exactly half the order and lose half the bonds.
    if bo.ndim == 3:
        bo = bo.sum(axis=2)
    if bo.ndim != 2 or bo.shape[0] != bo.shape[1]:
        raise ValueError(f"bond-order matrix must be square, got {bo.shape}")
    idx = np.argwhere(np.triu(bo, k=1) > threshold)
    return [Bond(int(i), int(j), float(bo[i, j])) for i, j in idx]


def connectivity_key(bonds: list[Bond]) -> frozenset[tuple[int, int]]:
    """Canonical representation of connectivity: used to compare states before and after."""
    return frozenset((min(b.i, b.j), max(b.i, b.j)) for b in bonds)


def find_species(numbers: np.ndarray, bonds: list[Bond], charges: np.ndarray | None = None) -> list[Species]:
    """Connected components of the bond graph = molecules. Iterative traversal, no recursion."""
    n = len(numbers)
    adjacency: dict[int, list[int]] = {i: [] for i in range(n)}
    for b in bonds:
        adjacency[b.i].append(b.j)
        adjacency[b.j].append(b.i)

    seen = np.zeros(n, dtype=bool)
    species: list[Species] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members: list[int] = []
        while stack:
            cur = stack.pop()
            members.append(cur)
            for nb in adjacency[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(nb)
        members.sort()
        uniq, counts = np.unique(np.asarray(numbers)[members], return_counts=True)
        # Hill order: C, H, then the rest alphabetically, as is conventional in chemistry
        parts = {chemical_symbols[z]: int(c) for z, c in zip(uniq, counts)}
        formula = ""
        for sym in ("C", "H"):
            if sym in parts:
                formula += f"{sym}{parts.pop(sym) if parts[sym] > 1 else ''}"
                parts.pop(sym, None)
        for sym in sorted(parts):
            formula += f"{sym}{parts[sym] if parts[sym] > 1 else ''}"
        q = float(np.asarray(charges)[members].sum()) if charges is not None else 0.0
        species.append(Species(tuple(members), formula, q))
    return species


@dataclass
class ReactionEvent:
    """Change in connectivity between two points in time."""

    step: int
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    species_before: tuple[str, ...]
    species_after: tuple[str, ...]

    def __str__(self) -> str:
        left = " + ".join(self.species_before)
        right = " + ".join(self.species_after)
        return f"step {self.step}: {left} -> {right} (bonds formed {len(self.formed)}, broken {len(self.broken)})"


class ReactionWatcher:
    """
    Watches connectivity and reports reactions. Neither forbids nor allows anything.

    An event is recorded only if the new connectivity HELD for `persistence` observations:
    the bond order near the threshold jitters from thermal vibrations, and without this
    condition the same pair would "react" back and forth every step. This is a debounce
    filter, not physics, and its value is visible in the report alongside the events.
    """

    def __init__(self, numbers: np.ndarray, persistence: int = 3, threshold: float = BOND_ORDER_THRESHOLD) -> None:
        self.numbers = np.asarray(numbers, dtype=int)
        self.persistence = int(persistence)
        self.threshold = float(threshold)
        self._committed: frozenset[tuple[int, int]] | None = None
        self._committed_species: tuple[str, ...] = ()
        self._candidate: frozenset[tuple[int, int]] | None = None
        self._candidate_count = 0
        self.events: list[ReactionEvent] = []

    def observe(self, step: int, bond_orders: np.ndarray, charges: np.ndarray | None = None) -> ReactionEvent | None:
        bonds = bonds_from_orders(bond_orders, self.threshold)
        key = connectivity_key(bonds)
        species = tuple(sorted(s.formula for s in find_species(self.numbers, bonds, charges)))

        if self._committed is None:
            self._committed, self._committed_species = key, species
            return None
        if key == self._committed:
            self._candidate, self._candidate_count = None, 0
            return None

        if key == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate, self._candidate_count = key, 1
        if self._candidate_count < self.persistence:
            return None

        event = ReactionEvent(
            step=step,
            formed=tuple(sorted(key - self._committed)),
            broken=tuple(sorted(self._committed - key)),
            species_before=self._committed_species,
            species_after=species,
        )
        self.events.append(event)
        self._committed, self._committed_species = key, species
        self._candidate, self._candidate_count = None, 0
        return event
