"""
Машинно-обученный потенциал: точность около квантовой при скорости около классической.

Зачем он в движке. Квантовый бэкенд (backends/gfn2.py, backends/scf.py) считает физику
правильно, но при 192 атомах даёт 1.6 шага в секунду -- около 3 пикосекунд модельного времени
за час счёта (замер в README). Никакая сборка мембраны в такие числа не влезает. Машинный
потенциал обучен НА квантовых расчётах и воспроизводит их энергии и силы, стоя при этом на
несколько порядков дешевле: это и есть единственный известный способ получить квантовую
точность на объёмах, а не выбирать между ними.

Здесь нет своей нейросети и не будет: обучать потенциал общего назначения -- это отдельная
работа масштабом в годы и в машинное время суперкомпьютера, и делать её заново незачем.
Взяты опубликованные обученные модели, а наша ответственность -- ПРОВЕРИТЬ их против наших же
квантовых замеров (atomic/validate/test_mlip.py) и назвать измеренную скорость.

MACE-OFF23 (Kovács et al. 2023, arXiv:2312.15211) обучена на наборе SPICE на уровне
ωB97M-D3(BJ)/def2-TZVPPD и покрывает H, C, N, O, F, P, S, Cl, Br, I -- то есть всю органику,
которая нужна пребиотической химии. MACE-MP-0 (Batatia et al. 2023, arXiv:2401.00096) покрывает
89 элементов из базы Materials Project, то есть минералы: глина, соли, поверхности.

Переходник намеренно общий: любой калькулятор ASE подключается одинаково, поэтому смена модели
не трогает ни интегратор, ни измерители.
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState


class ASECalculatorPotential:
    """
    Обёртка над любым калькулятором ASE. Единицы ASE -- эВ и эВ/Å, то есть наши; никакого
    пересчёта нет, и это одна из причин, по которой Å/фс/эВ выбраны единицами движка.
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
            # переиспользуем объект: пересоздание Atoms на каждом шаге заметно дороже
            # самого расчёта на малых системах
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
    MACE-OFF23 -- органика (H, C, N, O, F, P, S, Cl, Br, I).

    `default_dtype` по умолчанию float64, а не float32, и это не перестраховка: в float32
    ошибка сил накапливается в дрейф энергии, и тест сохранения энергии в NVE это ловит.
    Точность выбирается замером (см. test_mlip.py), а не привычкой.
    """
    import torch
    from mace.calculators import mace_off as _mace_off

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    calc = _mace_off(model=model, device=device, default_dtype=default_dtype)
    return ASECalculatorPotential(calc, f"mace-off23-{model}/{device}/{default_dtype}")


def mace_mp(model: str = "medium", device: str | None = None, default_dtype: str = "float64"):
    """MACE-MP-0 -- 89 элементов, минералы и поверхности (глина, соли)."""
    import torch
    from mace.calculators import mace_mp as _mace_mp

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    calc = _mace_mp(model=model, device=device, default_dtype=default_dtype)
    return ASECalculatorPotential(calc, f"mace-mp0-{model}/{device}/{default_dtype}")
