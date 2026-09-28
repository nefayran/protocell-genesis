"""
Machine-learned potential: near-quantum accuracy at near-classical speed.

Why it is in the engine. The quantum backend (backends/gfn2.py, backends/scf.py)
computes the physics correctly, but at 192 atoms it delivers 1.6 steps per second,
about 3 picoseconds of model time per hour of compute (measured in the README). No
membrane assembly fits into numbers like that. The machine-learned potential is trained
ON quantum calculations and reproduces their energies and forces while being several
orders of magnitude cheaper: this is the only known way to get quantum accuracy at
scale, rather than having to choose between the two.

There is no in-house neural network here, and there will not be one: training a
general-purpose potential is a separate effort on the scale of years and supercomputer
compute time, and there is no reason to redo it. Published pretrained models are used
instead, and our responsibility is to VALIDATE them against our own quantum benchmarks
(atomic/validate/test_mlip.py) and report the measured speed.

MACE-OFF23 (Kovacs et al. 2023, arXiv:2312.15211) is trained on the SPICE dataset at the
omega-B97M-D3(BJ)/def2-TZVPPD level and covers H, C, N, O, F, P, S, Cl, Br, I, i.e. all
the organic chemistry needed for prebiotic chemistry. MACE-MP-0 (Batatia et al. 2023,
arXiv:2401.00096) covers 89 elements from the Materials Project database, i.e. minerals:
clay, salts, surfaces.

The adapter is deliberately generic: any ASE calculator plugs in the same way, so
switching models touches neither the integrator nor the measurement code.
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState


class ASECalculatorPotential:
    """
    Wrapper around any ASE calculator. ASE units are eV and eV/Å, which are ours as
    well; there is no conversion, and this is one of the reasons Å/fs/eV were chosen as
    the engine's units.
    """

    def __init__(self, calculator, name: str) -> None:
        self.calculator = calculator
        self.name = name
        self._atoms = None
        self._signature: tuple | None = None

    def _atoms_for(self, state: AtomicState):
        from ase import Atoms

        signature = (tuple(state.numbers.tolist()), state.cell is None, tuple(state.pbc))
        if self._atoms is None or signature != self._signature:
            self._atoms = Atoms(
                numbers=state.numbers,
                positions=state.positions,
                cell=state.cell if state.cell is not None else None,
                pbc=state.pbc,
            )
            self._atoms.calc = self.calculator
            self._signature = signature
        else:
            # reuse the object: recreating Atoms on every step is noticeably more
            # expensive than the calculation itself on small systems
            self._atoms.set_positions(state.positions)
            if state.cell is not None:
                self._atoms.set_cell(state.cell)
        return self._atoms

    def compute(self, state: AtomicState) -> PotentialResult:
        atoms = self._atoms_for(state)
        energy = float(atoms.get_potential_energy())
        forces = np.asarray(atoms.get_forces(), dtype=float)
        return PotentialResult(energy, forces, {"backend": self.name})


def mace_off(model: str = "medium", device: str | None = None, default_dtype: str = "float64"):
    """
    MACE-OFF23: organic chemistry (H, C, N, O, F, P, S, Cl, Br, I).

    `default_dtype` defaults to float64 rather than float32, and this is not
    over-caution: in float32 the force error accumulates into energy drift, and the
    energy-conservation test in NVE catches it. Precision is chosen by measurement (see
    test_mlip.py), not by habit.
    """
    import torch
    from mace.calculators import mace_off as _mace_off

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    calc = _mace_off(model=model, device=device, default_dtype=default_dtype)
    return ASECalculatorPotential(calc, f"mace-off23-{model}/{device}/{default_dtype}")


def mace_mp(model: str = "medium", device: str | None = None, default_dtype: str = "float64"):
    """MACE-MP-0: 89 elements, minerals and surfaces (clay, salts)."""
    import torch
    from mace.calculators import mace_mp as _mace_mp

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    calc = _mace_mp(model=model, device=device, default_dtype=default_dtype)
    return ASECalculatorPotential(calc, f"mace-mp0-{model}/{device}/{default_dtype}")
