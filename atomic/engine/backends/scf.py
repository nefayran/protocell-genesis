"""
Reference accuracy: real electronic structure via pyscf, Hartree-Fock and DFT.

The role of this backend in the engine is NOT to run dynamics (it is too expensive for
that), but to provide numbers against which the cheap backend is validated: bond
lengths, angles, reaction energies, barriers at characteristic points. The
division-of-responsibility rule is written in gfn2.py: GFN2 carries the trajectory, DFT
validates the energetics.

The basis set and functional are arguments, not constants baked in: they determine
accuracy and cost, and must be visible in the report alongside the number obtained with
them. The default is def2-SVP + B3LYP-D3, chosen not as "the best" but as the most
common combination in the literature for organic chemistry, so our numbers have
something to compare against.
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState
from ..units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


class PySCF:
    """Hartree-Fock or DFT with an analytical gradient."""

    def __init__(
        self,
        method: str = "b3lyp",
        basis: str = "def2-svp",
        dispersion: bool = True,
        max_cycle: int = 100,
    ) -> None:
        self.method = method.lower()
        self.basis = basis
        self.dispersion = bool(dispersion)
        self.max_cycle = int(max_cycle)
        suffix = "-d3" if (self.dispersion and self.method != "hf") else ""
        self.name = f"pyscf:{self.method}{suffix}/{self.basis}"

    def _build(self, state: AtomicState):
        from pyscf import dft, gto, scf

        if state.cell is not None and any(state.pbc):
            raise NotImplementedError(
                "periodic pyscf (the pbc module) is not supported here: this backend "
                "computes isolated systems, periodicity is carried by GFN2"
            )
        mol = gto.Mole()
        mol.atom = [
            (int(z), tuple(map(float, r))) for z, r in zip(state.numbers, state.positions)
        ]
        mol.unit = "Angstrom"
        mol.basis = self.basis
        mol.charge = int(state.charge)
        mol.spin = int(state.spin_multiplicity - 1)  # pyscf expects 2S, not 2S+1
        mol.verbose = 0
        mol.build()

        # An open shell is computed with the UNRESTRICTED method (U), not the
        # spin-restricted one (RO). The difference is not cosmetic: a radical described
        # with a spin-restricted method gets an inflated energy, and the bond-breaking
        # energy comes out systematically wrong; this is exactly the error that made
        # homolytic bond breaking in the cheap backend overestimated by 2.4x (see
        # atomic/validate/test_reactions.py).
        if self.method == "hf":
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc = self.method
            if self.dispersion:
                # empirical Grimme correction: without it there is no dispersion, and
                # binding energies of weak complexes (e.g. the water dimer) are systematically too low
                try:
                    mf = mf.apply(__import__("pyscf.dft.dispersion", fromlist=["dispersion"]).DFTD3Dispersion) if False else mf
                    mf.disp = "d3bj"
                except Exception:
                    pass
        mf.max_cycle = self.max_cycle
        return mol, mf

    def compute(self, state: AtomicState) -> PotentialResult:
        mol, mf = self._build(state)
        energy_hartree = mf.kernel()
        if not mf.converged:
            raise FloatingPointError(
                f"{self.name}: SCF did not converge in {self.max_cycle} iterations, "
                "the result is not a solution and is not returned"
            )
        grad = mf.nuc_grad_method().kernel()  # Hartree/Bohr
        forces = -np.asarray(grad, dtype=float) * (HARTREE_IN_EV / BOHR_IN_ANGSTROM)

        extra = {"backend": self.name, "energy_hartree": float(energy_hartree)}
        try:
            extra["dipole_debye"] = np.asarray(mf.dip_moment(unit="Debye", verbose=0), dtype=float)
        except Exception:
            pass
        return PotentialResult(float(energy_hartree) * HARTREE_IN_EV, forces, extra)
