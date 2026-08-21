"""
Барьеры: сначала против точного ответа, потом против опыта.

Порядок проверок тот же, что и во всём движке: аналитическая задача, где ответ известен
буквально, ловит ошибку метода; настоящая химия против справочной величины ловит ошибку
применения. Обе нужны — если пройдёт только вторая, невозможно понять, сошлось ли по существу
или числа случайно совпали.

Опытные величины:
  барьер инверсии аммиака = 5.80 ккал/моль = 0.2515 эВ
  (Swalen & Ibers, JCP 36 (1962) 1914 -- микроволновая спектроскопия расщепления инверсии)
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.barrier import find_barrier
from engine.potential import PotentialResult
from engine.state import from_symbols
from engine.units import EV_IN_KCAL_PER_MOL


class DoubleWell:
    """U(x) = h(x²−1)² по оси x плюс жёсткая привязка по y и z: барьер РОВНО h, минимумы в ±1."""

    name = "double-well"

    def __init__(self, h: float) -> None:
        self.h = float(h)

    def compute(self, state):
        x = state.positions[0, 0]
        u = self.h * (x**2 - 1.0) ** 2 + 10.0 * (state.positions[0, 1] ** 2 + state.positions[0, 2] ** 2)
        f = np.zeros_like(state.positions)
        f[0, 0] = -self.h * 4.0 * x * (x**2 - 1.0)
        f[0, 1] = -20.0 * state.positions[0, 1]
        f[0, 2] = -20.0 * state.positions[0, 2]
        return PotentialResult(u, f)


@pytest.mark.parametrize("height", [0.3, 1.0])
def test_barrier_matches_analytic_double_well(height):
    """Барьер обязан совпасть с h до четвёртого знака, а реакция быть термонейтральной."""
    a = from_symbols(["H"], [[-1.0, 0.0, 0.0]])
    b = from_symbols(["H"], [[1.0, 0.0, 0.0]])
    r = find_barrier(a, b, DoubleWell(height), n_images=11, max_iterations=400,
                     climbing_after=30, step_a=0.03, force_tol_ev_per_a=1e-3)
    assert r.converged, f"лента не сошлась за {r.iterations} итераций"
    assert abs(r.forward_barrier_ev - height) < 1e-3, f"барьер {r.forward_barrier_ev:.5f} против {height}"
    assert abs(r.reaction_energy_ev) < 1e-6, "симметричная реакция обязана быть термонейтральной"
    assert r.peak_index == 5, f"вершина симметричного пути обязана быть посередине, а не {r.peak_index}"


def test_rate_spans_the_range_direct_dynamics_cannot_reach():
    """
    Смысл всего слоя, выраженный числом: при 300 K барьер 0.3 эВ даёт события на наносекундной
    шкале (динамика их видит), а 1.0 эВ -- раз в часы. Второе прямой динамикой недостижимо ни
    на каком железе, а через барьер считается за минуты.
    """
    a = from_symbols(["H"], [[-1.0, 0.0, 0.0]])
    b = from_symbols(["H"], [[1.0, 0.0, 0.0]])
    fast = find_barrier(a.copy(), b.copy(), DoubleWell(0.3), n_images=11, max_iterations=400,
                        climbing_after=30, step_a=0.03, force_tol_ev_per_a=1e-3)
    slow = find_barrier(a.copy(), b.copy(), DoubleWell(1.0), n_images=11, max_iterations=400,
                        climbing_after=30, step_a=0.03, force_tol_ev_per_a=1e-3)
    k_fast = fast.rate_per_second(300.0)
    k_slow = slow.rate_per_second(300.0)
    assert k_fast > 1e6, f"частота при барьере 0.3 эВ вышла {k_fast:.2e} 1/с"
    assert k_slow < 1e-2, f"частота при барьере 1.0 эВ вышла {k_slow:.2e} 1/с"
    assert k_fast / k_slow > 1e9, "разрыв шкал меньше девяти порядков -- проверить формулу"


@pytest.mark.slow
def test_ammonia_inversion_barrier_against_experiment():
    """
    Инверсия аммиака дешёвым бэкендом: 6.11 ккал/моль против опытных 5.80 (замер 2026-08-21,
    FIRE, сошлось за 39 итераций). Допуск 25 % задан известной точностью полуэмпирического
    метода на барьерах, а не подогнан под результат; профиль обязан быть симметричным, и это
    проверка независимая от величины -- у симметричной реакции вершина строго посередине.
    """
    from engine.backends.gfn2 import GFN2
    from engine.integrate import optimise_lbfgs

    pot = GFN2()
    a = from_symbols(["N", "H", "H", "H"],
                     [[0, 0, 0.38], [0.94, 0, -0.11], [-0.47, 0.81, -0.11], [-0.47, -0.81, -0.11]])
    optimise_lbfgs(a, pot, force_tol_ev_per_a=1e-3)
    b = a.copy()
    b.positions[:, 2] *= -1.0

    r = find_barrier(a, b, pot, n_images=7, max_iterations=150, climbing_after=15,
                     step_a=0.03, force_tol_ev_per_a=0.03)
    kcal = r.forward_barrier_ev * EV_IN_KCAL_PER_MOL
    assert r.converged, f"лента не сошлась: перпендикулярная сила {r.max_perpendicular_force:.3f} эВ/Å"
    assert abs(kcal - 5.80) / 5.80 < 0.25, f"барьер {kcal:.2f} ккал/моль против опытных 5.80"
    assert r.peak_index == 3, f"вершина симметричной инверсии обязана быть посередине, а не {r.peak_index}"
    profile = r.energies_ev - r.energies_ev[0]
    assert abs(profile[1] - profile[-2]) < 0.01, f"профиль несимметричен: {np.round(profile, 4)}"
