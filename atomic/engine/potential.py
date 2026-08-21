"""
Интерфейс потенциала и аналитические потенциалы для ПРОВЕРКИ движка.

Разделение принципиальное. Настоящая химия приходит из электронной структуры
(atomic/engine/backends), но проверить интегратор на ней нельзя: там нет замкнутого
выражения, с которым можно сверить силу. Поэтому в движке всегда есть аналитический
потенциал с точной производной -- на нём проверяются сохранение энергии, сохранение
импульса и совпадение силы с численным градиентом. Если эти три проверки проходят на
аналитике, а результат на квантовом бэкенде странный, значит дело в бэкенде, а не в
интеграторе, и наоборот. Без такого разделения любая ошибка выглядит как "физика такая".
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .state import AtomicState


@dataclass
class PotentialResult:
    """
    Энергия в эВ, силы в эВ/Å. `extra` -- всё, что бэкенд может дать сверх этого
    (порядки связей, заряды, диполь): именно оттуда берётся химия, а не из наших правил.
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
                "потенциал вернул нечисловое значение: "
                f"energy={self.energy_ev}, нечисловых компонент силы="
                f"{int((~np.isfinite(self.forces_ev_per_a)).sum())}"
            )


class Potential(Protocol):
    """Всё, что умеет считать энергию и силы для состояния."""

    name: str

    def compute(self, state: AtomicState) -> PotentialResult: ...


def numerical_forces(potential: Potential, state: AtomicState, h: float = 1e-4) -> np.ndarray:
    """
    Силы центральной разностью энергии: -dE/dx. Единственный способ проверить силу,
    не доверяя её выводу. Стоит 6N вычислений энергии, поэтому применяется к маленьким
    системам в тестах, а не в прогонах.
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
    Парный Леннард-Джонс со сдвигом энергии на радиусе обрезания.

    Сдвиг обязателен, а не косметичен: без него энергия скачком меняется, когда пара
    проходит радиус обрезания, и сохранение энергии в NVE нарушается ступеньками --
    это классическая ошибка, которую тест NVE обязан ловить. Сила при этом не сдвигается
    (сдвиг постоянный), поэтому численный градиент и аналитическая сила совпадают.
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
    Набор гармонических связей: U = sum k/2 (r - r0)^2. Точное аналитическое решение для
    двух атомов известно, поэтому на нём проверяется и период колебания, и сохранение
    энергии при разных шагах -- прямой тест порядка точности интегратора.
    """

    name = "harmonic-bonds"

    def __init__(self, bonds: list[tuple[int, int, float, float]]) -> None:
        # (i, j, k в эВ/Å², r0 в Å)
        self.bonds = [(int(i), int(j), float(k), float(r0)) for i, j, k, r0 in bonds]

    def compute(self, state: AtomicState) -> PotentialResult:
        forces = np.zeros_like(state.positions)
        energy = 0.0
        for i, j, k, r0 in self.bonds:
            d = state.positions[j] - state.positions[i]
            r = float(np.linalg.norm(d))
            if r == 0.0:
                raise FloatingPointError(f"совпавшие атомы {i} и {j} в гармонической связи")
            energy += 0.5 * k * (r - r0) ** 2
            f = -k * (r - r0) * d / r
            forces[j] += f
            forces[i] -= f
        return PotentialResult(energy, forces)


class SumPotential:
    """Сумма потенциалов: энергии складываются, силы складываются. Больше ничего."""

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
