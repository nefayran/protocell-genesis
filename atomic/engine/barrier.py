"""
Барьер реакции: метод упругой ленты с подъёмом изображения (CI-NEB).

Зачем он существует в этом движке. Прямая динамика измеренно НЕ преодолевает барьер в
несколько электронвольт: 2000 шагов при 6000 K на перекиси водорода не дали ни одного разрыва
(atomic/validate/test_reactions.py). Причина не в терпении, а в статистике -- при частоте
попыток около одной на 27 фс и больцмановском множителе для 2 эВ ожидаемое число событий за
доступное время меньше единицы. Значит редкие события надо не ждать, а СЧИТАТЬ: найти путь
наименьшей энергии между двумя состояниями, взять его вершину и получить частоту по формуле
переходного состояния. Это переводит доступное время из наносекунд в секунды.

Метод: Henkelman & Jonsson, JCP 113 (2000) 9978 (NEB) и Henkelman, Uberuaga & Jonsson,
JCP 113 (2000) 9901 (climbing image). Устройство:
  * цепочка промежуточных состояний («изображений») между началом и концом;
  * настоящая сила проецируется ПЕРПЕНДИКУЛЯРНО пути, пружинная -- ВДОЛЬ него; иначе цепочка
    либо соскальзывает в минимумы, либо срезает углы;
  * изображение с наибольшей энергией лезет ВВЕРХ по пути (climbing image) -- без этого
    вершина попадает между изображениями и барьер систематически занижен.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .potential import Potential
from .state import AtomicState


@dataclass
class BarrierResult:
    """Найденный путь и его вершина."""

    energies_ev: np.ndarray          # энергия каждого изображения
    images: list[np.ndarray]         # координаты каждого изображения
    forward_barrier_ev: float        # вершина минус начало
    reverse_barrier_ev: float        # вершина минус конец
    reaction_energy_ev: float        # конец минус начало
    peak_index: int
    converged: bool
    iterations: int
    max_perpendicular_force: float

    def rate_per_second(self, temperature_k: float, prefactor_hz: float = 1.0e13) -> float:
        """
        Частота переходов по формуле переходного состояния: k = A exp(-Ea / kB T).

        Предэкспонента 1e13 Гц -- это НЕ измеренная величина, а порядок типичной частоты
        колебаний связи (около 1000 см⁻¹ = 3e13 Гц); честная оценка требует расчёта
        колебательных частот в минимуме и в седле. Поэтому число возвращается вместе с явным
        аргументом: кто им пользуется, обязан знать, что множитель приблизителен, а
        экспоненциальная часть -- посчитана.
        """
        from .units import KB_EV_PER_K

        return float(prefactor_hz * np.exp(-self.forward_barrier_ev / (KB_EV_PER_K * temperature_k)))


def _tangent(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray, e_prev: float, e_cur: float, e_next: float) -> np.ndarray:
    """
    Касательная к пути по «улучшенной» схеме: направление берётся к более ВЫСОКОМУ соседу.

    Наивная центральная разность (nxt - prev) даёт изломы на крутых участках и приводит к
    появлению ложных изгибов цепочки -- это известная беда первых реализаций NEB, и улучшенная
    касательная (Henkelman & Jonsson 2000, раздел II) её устраняет.
    """
    tau_plus = nxt - cur
    tau_minus = cur - prev
    if e_next > e_cur > e_prev:
        tau = tau_plus
    elif e_next < e_cur < e_prev:
        tau = tau_minus
    else:
        d_max = max(abs(e_next - e_cur), abs(e_prev - e_cur))
        d_min = min(abs(e_next - e_cur), abs(e_prev - e_cur))
        tau = tau_plus * d_max + tau_minus * d_min if e_next > e_prev else tau_plus * d_min + tau_minus * d_max
    norm = np.linalg.norm(tau)
    return tau / norm if norm > 1e-12 else tau


def find_barrier(
    initial: AtomicState,
    final: AtomicState,
    potential: Potential,
    n_images: int = 9,
    spring_k: float = 5.0,
    max_iterations: int = 200,
    force_tol_ev_per_a: float = 0.05,
    climbing_after: int = 20,
    step_a: float = 0.02,
) -> BarrierResult:
    """
    Ищет путь наименьшей энергии между `initial` и `final`.

    Начальная цепочка -- линейная интерполяция координат. Это годится, когда конечное состояние
    получено из начального небольшим смещением (перенос протона, поворот, разрыв одной связи), и
    НЕ годится, когда между ними перестройка нескольких связей: там линейная интерполяция
    проводит атомы друг через друга, и цепочка стартует из нефизичной области. Признак беды --
    огромная энергия начальных изображений, и она видна в возвращаемом массиве, а не спрятана.

    `climbing_after` -- итерация, с которой вершина начинает лезть вверх. Раньше включать нельзя:
    пока цепочка не распрямилась, «вершиной» может оказаться случайное изображение.
    """
    if initial.n_atoms != final.n_atoms:
        raise ValueError("число атомов в начальном и конечном состоянии не совпадает")
    if not np.array_equal(initial.numbers, final.numbers):
        raise ValueError("сорта атомов в начальном и конечном состоянии не совпадают")

    # линейная интерполяция; крайние изображения фиксированы
    images = [
        initial.positions + (final.positions - initial.positions) * t
        for t in np.linspace(0.0, 1.0, n_images)
    ]
    probe = initial.copy()

    def energy_forces(pos):
        probe.positions = pos.copy()
        res = potential.compute(probe)
        return res.energy_ev, res.forces_ev_per_a

    converged = False
    iterations = 0
    worst_perp = float("inf")
    energies = np.zeros(n_images)
    forces = [None] * n_images

    # --- FIRE (Bitzek et al., PRL 97 (2006) 170201) вместо спуска с постоянным шагом --------
    # Замер, из-за которого это переписано: на инверсии аммиака простой спуск с шагом 0.03 Å
    # не сошёлся за 120 итераций (перпендикулярная сила осталась 1.64 эВ/Å), хотя барьер уже
    # был близок к опытному. FIRE добавляет инерцию и САМ подбирает шаг: пока сила и скорость
    # смотрят в одну сторону, шаг растёт; как только направление сменилось -- скорость
    # сбрасывается, а шаг уменьшается. Ни одного подбираемого руками числа сверх начального
    # шага при этом не появляется.
    velocities = [np.zeros_like(images[0]) for _ in range(n_images)]
    dt = step_a
    dt_max = step_a * 10.0
    alpha = 0.1
    n_positive = 0

    for iteration in range(max_iterations):
        iterations = iteration + 1
        for k in range(n_images):
            energies[k], forces[k] = energy_forces(images[k])

        peak = int(np.argmax(energies))
        climbing = iteration >= climbing_after
        worst_perp = 0.0
        updates = [np.zeros_like(images[0]) for _ in range(n_images)]

        for k in range(1, n_images - 1):
            tau = _tangent(
                images[k - 1], images[k], images[k + 1],
                energies[k - 1], energies[k], energies[k + 1],
            )
            f_true = forces[k]
            f_parallel_mag = float((f_true * tau).sum())
            f_perp = f_true - f_parallel_mag * tau

            if climbing and k == peak:
                # вершина лезет ВВЕРХ: истинная сила вдоль пути инвертируется, пружин нет
                total = f_perp - f_parallel_mag * tau
            else:
                spring = spring_k * (
                    np.linalg.norm(images[k + 1] - images[k]) - np.linalg.norm(images[k] - images[k - 1])
                )
                total = f_perp + spring * tau
            worst_perp = max(worst_perp, float(np.abs(f_perp).max()))
            updates[k] = total          # это СИЛА; шагом управляет FIRE ниже

        # --- шаг FIRE по всей ленте сразу: она одна система, а не набор независимых точек ---
        power = sum(float((velocities[k] * updates[k]).sum()) for k in range(1, n_images - 1))
        if power > 0.0:
            n_positive += 1
            if n_positive > 5:
                dt = min(dt * 1.1, dt_max)
                alpha *= 0.99
        else:
            n_positive = 0
            dt *= 0.5
            alpha = 0.1
            for k in range(1, n_images - 1):
                velocities[k][:] = 0.0

        f_norm = np.sqrt(sum(float((updates[k] ** 2).sum()) for k in range(1, n_images - 1)))
        v_norm = np.sqrt(sum(float((velocities[k] ** 2).sum()) for k in range(1, n_images - 1)))
        for k in range(1, n_images - 1):
            velocities[k] = velocities[k] + dt * updates[k]
            if f_norm > 1e-12:
                velocities[k] = (1.0 - alpha) * velocities[k] + alpha * v_norm * updates[k] / f_norm
            move = dt * velocities[k]
            # ограничение длины шага остаётся: оно защищает первую итерацию из плохой геометрии
            norm = np.linalg.norm(move)
            if norm > step_a:
                move *= step_a / norm
            images[k] = images[k] + move

        if climbing and worst_perp < force_tol_ev_per_a:
            converged = True
            break

    for k in range(n_images):
        energies[k], _ = energy_forces(images[k])
    peak = int(np.argmax(energies))
    return BarrierResult(
        energies_ev=energies.copy(),
        images=[im.copy() for im in images],
        forward_barrier_ev=float(energies[peak] - energies[0]),
        reverse_barrier_ev=float(energies[peak] - energies[-1]),
        reaction_energy_ev=float(energies[-1] - energies[0]),
        peak_index=peak,
        converged=converged,
        iterations=iterations,
        max_perpendicular_force=worst_perp,
    )
