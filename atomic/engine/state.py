"""
Состояние атомистической системы: сорта атомов по зарядовому числу Z, положения, скорости,
ячейка. Ничего про потенциал и ничего про интегратор -- только то, что физически ЕСТЬ.

Почему Z, а не строковые имена сортов: движок обязан работать на любых элементах, а Z --
это и есть элемент, без таблицы соответствий, которую пришлось бы поддерживать. Массы
берутся из таблицы атомных весов ASE (стандартные атомные веса IUPAC), а не задаются
вручную -- сорт атома полностью определяет массу.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from ase.data import atomic_masses, atomic_numbers, chemical_symbols

from .units import instantaneous_temperature_k, kinetic_energy_ev


def masses_for(numbers: np.ndarray) -> np.ndarray:
    """Стандартные атомные веса (а.е.м.) для массива Z."""
    z = np.asarray(numbers, dtype=int)
    if z.min() < 1 or z.max() >= len(atomic_masses):
        raise ValueError(f"Z вне таблицы элементов: min={z.min()}, max={z.max()}")
    return atomic_masses[z].astype(float)


def numbers_from_symbols(symbols) -> np.ndarray:
    """['O','H','H'] -> [8,1,1]. Неизвестный символ -- ошибка, а не молчаливый пропуск."""
    out = []
    for s in symbols:
        if s not in atomic_numbers:
            raise ValueError(f"неизвестный химический символ: {s!r}")
        out.append(atomic_numbers[s])
    return np.asarray(out, dtype=int)


@dataclass
class AtomicState:
    """
    Мгновенное состояние системы. Положения в Å, скорости в Å/фс, ячейка в Å.

    `cell` и `pbc` разделены умышленно: непериодический расчёт (молекула в пустоте, капля)
    и периодический -- это разные физические постановки, и в этом проекте уже был измеренный
    случай, когда завёрнутое расстояние, посчитанное в непериодической системе, тихо портило
    результат. Здесь периодичность обязана быть объявлена явно по каждой оси.
    """

    numbers: np.ndarray                       # (N,) int, зарядовые числа
    positions: np.ndarray                     # (N,3) float, Å
    velocities: np.ndarray | None = None      # (N,3) float, Å/фс; None -> нули
    cell: np.ndarray | None = None            # (3,3) float, Å; None -> нет ячейки
    pbc: tuple[bool, bool, bool] = (False, False, False)
    charge: int = 0                           # суммарный заряд системы, e
    spin_multiplicity: int = 1                # 2S+1; 1 = синглет с замкнутой оболочкой
    frozen: np.ndarray | None = None          # (N,) bool, атомы с обнулённой силой
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
            raise ValueError("периодичность объявлена, а ячейки нет")
        if self.frozen is not None:
            self.frozen = np.asarray(self.frozen, dtype=bool).reshape(len(self.numbers))
        self.masses = masses_for(self.numbers)

    # --- удобные представления ----------------------------------------------------------
    @property
    def n_atoms(self) -> int:
        return len(self.numbers)

    @property
    def symbols(self) -> list[str]:
        return [chemical_symbols[z] for z in self.numbers]

    @property
    def formula(self) -> str:
        """Брутто-формула в порядке убывания количества -- для подписей и логов."""
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

    # --- измеряемые величины ------------------------------------------------------------
    def kinetic_energy_ev(self) -> float:
        return kinetic_energy_ev(self.masses, self.velocities)

    def temperature_k(self, n_constraints: int = 3) -> float:
        """
        По умолчанию отнимаются три степени свободы: движок убирает суммарный импульс при
        задании скоростей (см. `set_maxwell_boltzmann`), поэтому поступательное движение
        системы как целого не является тепловой степенью свободы.
        """
        return instantaneous_temperature_k(self.masses, self.velocities, n_constraints)

    def total_momentum(self) -> np.ndarray:
        return (self.masses[:, None] * self.velocities).sum(axis=0)

    def centre_of_mass(self) -> np.ndarray:
        return (self.masses[:, None] * self.positions).sum(axis=0) / self.masses.sum()

    def remove_net_momentum(self) -> None:
        """Обнуляет суммарный импульс, сохраняя кинетическую энергию относительного движения."""
        drift = self.total_momentum() / self.masses.sum()
        self.velocities -= drift

    def set_maxwell_boltzmann(self, temperature_k: float, rng: np.random.Generator) -> None:
        """
        Максвелловские скорости при заданной T, с обнулением суммарного импульса и
        последующим ПЕРЕмасштабированием под целевую T. Порядок важен: обнуление импульса
        забирает часть кинетической энергии, поэтому масштабировать надо ПОСЛЕ него, иначе
        стартовая температура систематически ниже заданной -- ровно та ошибка, которую
        тест равнораспределения в atomic/validate и ловит.
        """
        sigma = np.sqrt(
            temperature_k * 1.380649e-23 / (self.masses * 1.66053906892e-27)
        )  # м/с
        self.velocities = rng.normal(size=self.positions.shape) * (sigma[:, None] * 1e-5)  # Å/фс
        self.remove_net_momentum()
        current = self.temperature_k()
        if current > 0:
            self.velocities *= np.sqrt(temperature_k / current)

    # --- геометрия ----------------------------------------------------------------------
    def distance(self, i: int, j: int) -> float:
        """Расстояние i-j с минимальным образом ТОЛЬКО по периодическим осям."""
        d = self.positions[j] - self.positions[i]
        if self.cell is not None and any(self.pbc):
            frac = np.linalg.solve(self.cell.T, d)
            for axis in range(3):
                if self.pbc[axis]:
                    frac[axis] -= np.round(frac[axis])
            d = self.cell.T @ frac
        return float(np.linalg.norm(d))

    def angle_deg(self, i: int, j: int, k: int) -> float:
        """Угол i-j-k в градусах (j -- вершина)."""
        a = self.positions[i] - self.positions[j]
        b = self.positions[k] - self.positions[j]
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def from_symbols(symbols, positions, **kwargs) -> AtomicState:
    """Собрать состояние из символов и координат в Å."""
    return AtomicState(numbers=numbers_from_symbols(symbols), positions=positions, **kwargs)
