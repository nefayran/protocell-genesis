"""
Интеграторы. Два, и оба нужны по разным причинам.

`velocity_verlet` -- NVE: он симплектический, то есть сохраняет энергию с ограниченной
ошибкой, и именно на нём проверяется правильность сил. Если энергия уплывает -- виноваты
силы или шаг, и это видно сразу.

`langevin_baoab` -- NVT: схема BAOAB (Leimkuhler & Matthews 2013), у которой ошибка
конфигурационных средних по шагу выше порядком, чем у наивного ланжевена. Термостат нужен,
потому что реальная химия идёт при заданной температуре, а не при заданной энергии.

Заморозка атомов делается ОБНУЛЕНИЕМ СИЛЫ И СКОРОСТИ, а не большой массой: большая масса
оставляет атом медленно ползущим, и в этом проекте уже был случай, когда «неподвижная»
поверхность на самом деле дрейфовала. Неподвижность обязана быть структурной.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .potential import Potential, PotentialResult
from .state import AtomicState
from .units import accelerations_a_per_fs2


def _apply_frozen(state: AtomicState, forces: np.ndarray) -> np.ndarray:
    if state.frozen is None:
        return forces
    out = forces.copy()
    out[state.frozen] = 0.0
    state.velocities[state.frozen] = 0.0
    return out


@dataclass
class StepDiagnostics:
    """То, что после шага можно измерить, не пересчитывая потенциал."""

    potential_ev: float
    kinetic_ev: float
    temperature_k: float
    max_force_ev_per_a: float
    max_displacement_a: float

    @property
    def total_ev(self) -> float:
        return self.potential_ev + self.kinetic_ev


def _diagnostics(state: AtomicState, res: PotentialResult, max_disp: float) -> StepDiagnostics:
    return StepDiagnostics(
        potential_ev=res.energy_ev,
        kinetic_ev=state.kinetic_energy_ev(),
        temperature_k=state.temperature_k(),
        max_force_ev_per_a=float(np.abs(res.forces_ev_per_a).max()) if state.n_atoms else 0.0,
        max_displacement_a=max_disp,
    )


def velocity_verlet(
    state: AtomicState,
    potential: Potential,
    dt_fs: float,
    cached: PotentialResult | None = None,
) -> tuple[PotentialResult, StepDiagnostics]:
    """
    Один шаг NVE. Возвращает результат потенциала В НОВОЙ точке, чтобы вызывающий код
    передал его следующим шагом и не считал потенциал дважды: на квантовом бэкенде
    вычисление силы -- это вся стоимость шага, и лишний вызов удваивает время прогона.
    """
    res = cached if cached is not None else potential.compute(state)
    forces = _apply_frozen(state, res.forces_ev_per_a)
    acc = accelerations_a_per_fs2(forces, state.masses)

    before = state.positions.copy()
    state.velocities += 0.5 * dt_fs * acc
    state.positions += dt_fs * state.velocities

    new_res = potential.compute(state)
    new_forces = _apply_frozen(state, new_res.forces_ev_per_a)
    state.velocities += 0.5 * dt_fs * accelerations_a_per_fs2(new_forces, state.masses)

    disp = float(np.linalg.norm(state.positions - before, axis=1).max()) if state.n_atoms else 0.0
    return new_res, _diagnostics(state, new_res, disp)


def langevin_baoab(
    state: AtomicState,
    potential: Potential,
    dt_fs: float,
    temperature_k: float,
    friction_per_fs: float,
    rng: np.random.Generator,
    cached: PotentialResult | None = None,
) -> tuple[PotentialResult, StepDiagnostics]:
    """
    Шаг BAOAB: B (полтолчка) A (полперемещения) O (термостат) A (полперемещения) B (полтолчка).

    `friction_per_fs` -- обратное время затухания скорости, 1/фс. Значение НЕ подбирается
    под результат: слишком большое трение подавляет реальную динамику (диффузию, скорость
    встреч), слишком малое не термостатирует за разумное время, и правильная проверка --
    измерить время автокорреляции скорости и убедиться, что оно не короче интересующего
    процесса.
    """
    from .units import KB_EV_PER_K, AMU_A2_PER_FS2_IN_EV

    res = cached if cached is not None else potential.compute(state)
    forces = _apply_frozen(state, res.forces_ev_per_a)
    before = state.positions.copy()

    # B
    state.velocities += 0.5 * dt_fs * accelerations_a_per_fs2(forces, state.masses)
    # A
    state.positions += 0.5 * dt_fs * state.velocities
    # O: точное решение уравнения Орнштейна-Уленбека на шаг dt
    decay = np.exp(-friction_per_fs * dt_fs)
    # sigma^2 = kB T / m в единицах (Å/фс)^2: kB T в эВ, делим на массу и на переводной
    # множитель а.е.м.·Å²/фс² -> эВ. Это и есть равнораспределение, а не подгонка.
    sigma = np.sqrt(KB_EV_PER_K * temperature_k / (state.masses * AMU_A2_PER_FS2_IN_EV))
    noise = rng.normal(size=state.velocities.shape)
    state.velocities = decay * state.velocities + np.sqrt(1.0 - decay**2) * sigma[:, None] * noise
    if state.frozen is not None:
        state.velocities[state.frozen] = 0.0
    # A
    state.positions += 0.5 * dt_fs * state.velocities

    new_res = potential.compute(state)
    new_forces = _apply_frozen(state, new_res.forces_ev_per_a)
    # B
    state.velocities += 0.5 * dt_fs * accelerations_a_per_fs2(new_forces, state.masses)

    disp = float(np.linalg.norm(state.positions - before, axis=1).max()) if state.n_atoms else 0.0
    return new_res, _diagnostics(state, new_res, disp)


def optimise_lbfgs(
    state: AtomicState,
    potential: Potential,
    force_tol_ev_per_a: float = 1e-3,
    max_iterations: int = 400,
) -> tuple[PotentialResult, int]:
    """
    Поиск равновесной геометрии квазиньютоновским L-BFGS-B по энергии с АНАЛИТИЧЕСКИМ
    градиентом из того же потенциала. Аналитический градиент здесь не удобство, а условие
    осмысленности: численный градиент на квантовом бэкенде стоил бы 6N расчётов на итерацию.

    Замороженные атомы исключаются из переменных, а не удерживаются штрафом: штраф сдвигает
    минимум, исключение -- нет.
    """
    from scipy.optimize import minimize

    free = np.ones(state.n_atoms, dtype=bool) if state.frozen is None else ~state.frozen
    x0 = state.positions[free].ravel().copy()
    calls = {"n": 0}

    def energy_and_grad(x: np.ndarray):
        state.positions[free] = x.reshape(-1, 3)
        res = potential.compute(state)
        calls["n"] += 1
        # минимизируем энергию, значит градиент = -сила
        return res.energy_ev, (-res.forces_ev_per_a[free]).ravel()

    out = minimize(
        energy_and_grad,
        x0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations, "gtol": force_tol_ev_per_a, "ftol": 1e-14},
    )
    state.positions[free] = out.x.reshape(-1, 3)
    return potential.compute(state), calls["n"]


def steepest_descent(
    state: AtomicState,
    potential: Potential,
    max_steps: int = 200,
    max_step_a: float = 0.05,
    force_tol_ev_per_a: float = 0.01,
) -> tuple[PotentialResult, int]:
    """
    Спуск по силе с ограничением ДЛИНЫ шага, а не с фиксированным множителем: на первом
    шаге из плохой геометрии сила может быть огромной, и множитель, разумный в минимуме,
    отправит атомы в бесконечность. Ограничение по длине делает первый шаг безопасным без
    подбора множителя. Останов -- по максимальной силе, а не по числу шагов.
    """
    res = potential.compute(state)
    for step in range(max_steps):
        forces = _apply_frozen(state, res.forces_ev_per_a)
        fmax = float(np.abs(forces).max())
        if fmax < force_tol_ev_per_a:
            return res, step
        norm = np.linalg.norm(forces, axis=1).max()
        scale = min(max_step_a / norm, max_step_a)
        state.positions += forces * scale
        res = potential.compute(state)
    return res, max_steps
