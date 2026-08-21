"""
Химия как НАБЛЮДЕНИЕ, а не как правило.

Здесь нет ни одной реакции. Связь считается существующей, если её порядок по электронной
структуре превышает порог; молекула -- это связная компонента графа связей; реакция -- это
событие, при котором связность изменилась. Поэтому движок «воспроизводит любой процесс» не
потому, что мы перечислили процессы, а потому, что не перечисляли: что произойдёт, решает
гамильтониан, а этот файл лишь читает результат.

Порог по порядку связи -- единственное свободное число, и оно НЕ подгоняется под желаемый
ответ. Проверка порога: на наборе молекул с известной структурой (вода, метан, этан, этен,
этин, пероксид водорода, уксусная кислота) связность обязана совпасть с учебной, включая
кратность. Если порог неверен, это видно как лишняя или потерянная связь в известной
молекуле, а не как странный результат в конце прогона.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ase.data import chemical_symbols

# Порог существования связи по порядку связи. Значение выбрано так, чтобы на наборе
# известных молекул (тест в atomic/validate/test_chem.py) связность совпадала с учебной:
# заметно ниже одинарной связи (порядок около 1), заметно выше остаточного перекрытия
# несвязанных пар (порядок около 0.05 и меньше).
BOND_ORDER_THRESHOLD = 0.5

# Порог кратности: связь считается двойной от 1.6 и тройной от 2.4. Это НЕ физические
# константы, а границы для ЧТЕНИЯ порядка связи как целого числа, и они нужны только для
# человекочитаемой подписи; вся динамика идёт по непрерывному порядку.
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
    """Молекула: набор атомов, брутто-формула, суммарный заряд по Малликену."""

    atom_indices: tuple[int, ...]
    formula: str
    charge: float

    @property
    def size(self) -> int:
        return len(self.atom_indices)


def bonds_from_orders(
    bond_orders: np.ndarray, threshold: float = BOND_ORDER_THRESHOLD
) -> list[Bond]:
    """Верхний треугольник матрицы порядков связей -> список связей."""
    bo = np.asarray(bond_orders, dtype=float)
    # tblite отдаёт матрицу с ДОПОЛНИТЕЛЬНОЙ спиновой осью (N, N, n_spin): для системы с
    # замкнутой оболочкой это (N, N, 1). Суммируем по спиновым каналам, а не берём первый:
    # в системе с открытой оболочкой полный порядок связи -- это сумма по каналам, и взятие
    # одного канала дало бы ровно вдвое меньший порядок и потерю половины связей.
    if bo.ndim == 3:
        bo = bo.sum(axis=2)
    if bo.ndim != 2 or bo.shape[0] != bo.shape[1]:
        raise ValueError(f"матрица порядков связей должна быть квадратной, получено {bo.shape}")
    idx = np.argwhere(np.triu(bo, k=1) > threshold)
    return [Bond(int(i), int(j), float(bo[i, j])) for i, j in idx]


def connectivity_key(bonds: list[Bond]) -> frozenset[tuple[int, int]]:
    """Каноническое представление связности -- по нему сравниваются состояния до и после."""
    return frozenset((min(b.i, b.j), max(b.i, b.j)) for b in bonds)


def find_species(numbers: np.ndarray, bonds: list[Bond], charges: np.ndarray | None = None) -> list[Species]:
    """Связные компоненты графа связей = молекулы. Обход в ширину, без рекурсии."""
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
        # порядок Хилла: C, H, потом остальные по алфавиту -- как принято в химии
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
    """Изменение связности между двумя моментами времени."""

    step: int
    formed: tuple[tuple[int, int], ...]
    broken: tuple[tuple[int, int], ...]
    species_before: tuple[str, ...]
    species_after: tuple[str, ...]

    def __str__(self) -> str:
        left = " + ".join(self.species_before)
        right = " + ".join(self.species_after)
        return f"шаг {self.step}: {left} -> {right} (связей образовано {len(self.formed)}, разорвано {len(self.broken)})"


class ReactionWatcher:
    """
    Следит за связностью и сообщает о реакциях. Ничего не запрещает и не разрешает.

    Событие фиксируется только если новая связность УДЕРЖАЛАСЬ `persistence` наблюдений:
    порядок связи около порога дрожит от тепловых колебаний, и без этого условия одна и та
    же пара будет «реагировать» туда-обратно каждый шаг. Это фильтр дребезга, а не физика,
    и его значение видно в отчёте вместе с событиями.
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
