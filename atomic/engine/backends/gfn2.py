"""
Real chemistry: the semi-empirical tight-binding Hamiltonian GFN2-xTB via tblite.

Why this one was chosen as the engine's workhorse. First, it covers elements up to
Z=86, so "any elements" is not a promise but a property of the parameterization, the
same one across the whole table. Second, bonds are not specified in it: they emerge
from the electronic structure, so bond breaking and formation happen on their own,
without rules and without a list of reactions, exactly what was missing in this
project's previous engine, where chemistry was reduced to four hand-written rules.
Third, it is several orders of magnitude cheaper than DFT, and molecular dynamics
actually runs on it.

What it does NOT give, and this should be stated here rather than discovered later: the
accuracy of coupled-cluster methods, barriers accurate to a few kJ/mol, excited states,
heavy transition metals in complex spin states. For the numbers a conclusion relies on,
there is a second backend (pyscf, real DFT/HF), and the rule is simple: GFN2 carries the
dynamics, DFT validates the energetics at characteristic points.

References: Bannwarth, Ehlert, Grimme, JCTC 15 (2019) 1652 (GFN2-xTB);
Ehlert et al., JOSS 5 (2020) 2569 (the tblite/simple-dftd3 stack).
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState
from ..units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


class GFN2:
    """
    Wrapper around tblite.interface.Calculator.

    The calculator is recreated when the number or species of atoms changes, and
    REUSED when only the coordinates change: tblite has `update`, and it is an order of
    magnitude cheaper than a fresh initialization. This is the only optimization here;
    everything else is left to tblite.
    """

    name = "gfn2-xtb"

    def __init__(self, accuracy: float = 1.0, max_iterations: int = 250, electronic_temperature_k: float | None = None) -> None:
        self.accuracy = float(accuracy)
        self.max_iterations = int(max_iterations)
        self.electronic_temperature_k = electronic_temperature_k
        self._calc = None
        self._signature: tuple | None = None

    def _ensure(self, state: AtomicState):
        import tblite.interface as ti

        pos_bohr = state.positions / BOHR_IN_ANGSTROM
        lattice = None
        periodic = None
        if state.cell is not None and any(state.pbc):
            if not all(state.pbc):
                raise NotImplementedError(
                    "GFN2 via tblite supports either a fully periodic cell or a fully "
                    "non-periodic one; mixed periodicity is not supported here"
                )
            lattice = state.cell / BOHR_IN_ANGSTROM
            periodic = np.array([True, True, True])

        signature = (
            tuple(state.numbers.tolist()),
            int(state.charge),
            int(state.spin_multiplicity),
            lattice is None,
        )
        if self._calc is None or signature != self._signature:
            kwargs = {}
            if lattice is not None:
                kwargs["lattice"] = lattice
                kwargs["periodic"] = periodic
            self._calc = ti.Calculator(
                "GFN2-xTB",
                state.numbers,
                pos_bohr,
                charge=float(state.charge),
                # tblite expects the number of UNPAIRED electrons, not 2S+1
                uhf=int(state.spin_multiplicity - 1),
                **kwargs,
            )
            self._calc.set("verbosity", 0)
            self._calc.set("accuracy", self.accuracy)
            self._calc.set("max-iter", self.max_iterations)
            if self.electronic_temperature_k is not None:
                # electronic temperature is set in Hartree (kT), not in kelvin
                self._calc.set(
                    "temperature", self.electronic_temperature_k * 3.166811563e-6
                )
            self._signature = signature
        else:
            self._calc.update(positions=pos_bohr, lattice=lattice)
        return self._calc

    def compute(self, state: AtomicState) -> PotentialResult:
        calc = self._ensure(state)
        res = calc.singlepoint()

        energy_ev = float(res.get("energy")) * HARTREE_IN_EV
        # gradient in Hartree/Bohr -> force in eV/Å, with a minus sign
        grad = np.asarray(res.get("gradient"), dtype=float)
        forces = -grad * (HARTREE_IN_EV / BOHR_IN_ANGSTROM)

        extra = {
            # Mulliken-like bond orders: connectivity is determined FROM THEM, not from
            # distance thresholds; see atomic/engine/chem.py
            "bond_orders": np.asarray(res.get("bond-orders"), dtype=float),
            "charges": np.asarray(res.get("charges"), dtype=float),
            "dipole_au": np.asarray(res.get("dipole"), dtype=float),
            "backend": self.name,
        }
        return PotentialResult(energy_ev, forces, extra)
