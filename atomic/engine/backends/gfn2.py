"""
Настоящая химия: полуэмпирический сильносвязанный гамильтониан GFN2-xTB через tblite.

Почему именно он взят рабочей лошадкой движка. Во-первых, он покрывает элементы до Z=86,
то есть «любые элементы» -- не обещание, а свойство параметризации, одной и той же для всей
таблицы. Во-вторых, связи в нём не задаются: они получаются из электронной структуры, поэтому
разрыв и образование связи происходят сами, без правил и без списка реакций -- ровно то, чего
не было в предыдущем движке этого проекта, где химия сводилась к четырём придуманным правилам.
В-третьих, он на несколько порядков дешевле DFT, и на нём реально идёт молекулярная динамика.

Чего он НЕ даёт, и это должно быть сказано здесь, а не обнаружено потом: точности связанных
кластеров, барьеров с ошибкой в единицы кДж/моль, возбуждённых состояний, тяжёлых переходных
металлов в сложных спиновых состояниях. Для чисел, на которые опирается вывод, есть второй
бэкенд (pyscf, настоящий DFT/HF) -- и правило простое: GFN2 считает динамику, DFT проверяет
энергетику на характерных точках.

Ссылки: Bannwarth, Ehlert, Grimme, JCTC 15 (2019) 1652 (GFN2-xTB);
Ehlert et al., JOSS 5 (2020) 2569 (tblite/simple-dftd3 стек).
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState
from ..units import BOHR_IN_ANGSTROM, HARTREE_IN_EV


class GFN2:
    """
    Обёртка над tblite.interface.Calculator.

    Калькулятор пересоздаётся, когда меняется число или сорт атомов, и ПЕРЕИСПОЛЬЗУЕТСЯ,
    когда меняются только координаты: у tblite есть `update`, и он на порядок дешевле, чем
    новая инициализация. Это единственная оптимизация здесь -- всё остальное отдано tblite.
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
                    "GFN2 через tblite поддерживает либо полностью периодическую ячейку, "
                    "либо полностью непериодическую; смешанная периодичность здесь не заявлена"
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
                # tblite ждёт число НЕСПАРЕННЫХ электронов, а не 2S+1
                uhf=int(state.spin_multiplicity - 1),
                **kwargs,
            )
            self._calc.set("verbosity", 0)
            self._calc.set("accuracy", self.accuracy)
            self._calc.set("max-iter", self.max_iterations)
            if self.electronic_temperature_k is not None:
                # электронная температура задаётся в Hartree (kT), а не в кельвинах
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
        # градиент в Hartree/Bohr -> сила в эВ/Å, со знаком минус
        grad = np.asarray(res.get("gradient"), dtype=float)
        forces = -grad * (HARTREE_IN_EV / BOHR_IN_ANGSTROM)

        extra = {
            # порядки связей Малликена-подобные: из НИХ определяется связность, а не из
            # порогов по расстоянию -- см. atomic/engine/chem.py
            "bond_orders": np.asarray(res.get("bond-orders"), dtype=float),
            "charges": np.asarray(res.get("charges"), dtype=float),
            "dipole_au": np.asarray(res.get("dipole"), dtype=float),
            "backend": self.name,
        }
        return PotentialResult(energy_ev, forces, extra)
