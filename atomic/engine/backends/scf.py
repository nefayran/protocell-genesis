"""
Опорная точность: настоящая электронная структура через pyscf -- Хартри-Фок и DFT.

Роль этого бэкенда в движке -- НЕ считать динамику (он для этого слишком дорог), а давать
числа, против которых проверяется дешёвый бэкенд: длины связей, углы, энергии реакций,
барьеры в характерных точках. Правило разделения ответственности записано в gfn2.py:
GFN2 везёт траекторию, DFT проверяет энергетику.

Базис и функционал -- аргументы, а не константы внутри: они определяют точность и стоимость,
и обязаны быть видны в отчёте вместе с числом, которое ими получено. По умолчанию взят
def2-SVP + B3LYP-D3 не как «лучший», а как самый распространённый в литературе набор для
органики, чтобы наши числа было с чем сравнивать.
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState
from ..units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


class PySCF:
    """Хартри-Фок или DFT с аналитическим градиентом."""

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
                "периодический pyscf (pbc-модуль) здесь не заявлен: этот бэкенд считает "
                "изолированные системы, а периодику везёт GFN2"
            )
        mol = gto.Mole()
        mol.atom = [
            (int(z), tuple(map(float, r))) for z, r in zip(state.numbers, state.positions)
        ]
        mol.unit = "Angstrom"
        mol.basis = self.basis
        mol.charge = int(state.charge)
        mol.spin = int(state.spin_multiplicity - 1)  # pyscf ждёт 2S, а не 2S+1
        mol.verbose = 0
        mol.build()

        # Открытая оболочка считается НЕОГРАНИЧЕННЫМ методом (U), а не ограниченным по
        # спину (RO). Разница не косметическая: радикал с ограниченным по спину описанием
        # получает завышенную энергию, и энергия разрыва связи выходит систематически
        # неверной -- ровно та ошибка, из-за которой гомолитический разрыв в дешёвом
        # бэкенде завышен в 2.4 раза (см. atomic/validate/test_reactions.py).
        if self.method == "hf":
            mf = scf.RHF(mol) if mol.spin == 0 else scf.UHF(mol)
        else:
            mf = dft.RKS(mol) if mol.spin == 0 else dft.UKS(mol)
            mf.xc = self.method
            if self.dispersion:
                # эмпирическая поправка Гримме: без неё дисперсия отсутствует, и энергии
                # связывания слабых комплексов (например, димера воды) систематически низкие
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
                f"{self.name}: SCF не сошёлся за {self.max_cycle} итераций -- "
                "результат не является решением и не возвращается"
            )
        grad = mf.nuc_grad_method().kernel()  # Hartree/Bohr
        forces = -np.asarray(grad, dtype=float) * (HARTREE_IN_EV / BOHR_IN_ANGSTROM)

        extra = {"backend": self.name, "energy_hartree": float(energy_hartree)}
        try:
            extra["dipole_debye"] = np.asarray(mf.dip_moment(unit="Debye", verbose=0), dtype=float)
        except Exception:
            pass
        return PotentialResult(float(energy_hartree) * HARTREE_IN_EV, forces, extra)
