"""
Единицы движка: Å, фс, а.е.м., эВ, К. Выбор не произволен -- это стандартный набор
классической молекулярной динамики, в котором шаг интегрирования выражается целым числом
фемтосекунд, а массы берутся прямо из таблицы атомных весов без пересчёта.

Все переводные множители -- определения СИ 2019 года или значения CODATA, а не подогнанные
числа: если какое-то из них окажется неверным, это видно по нарушению сохранения энергии в
atomic/validate (тест NVE), а не по расхождению с чьим-то ожиданием.
"""
from __future__ import annotations

# --- определения СИ и CODATA ------------------------------------------------------------
ELEMENTARY_CHARGE = 1.602176634e-19          # Кл, определение СИ 2019
ATOMIC_MASS_UNIT = 1.66053906892e-27         # кг, CODATA 2022
BOLTZMANN_J_PER_K = 1.380649e-23             # Дж/К, определение СИ 2019
HARTREE_J = 4.3597447222060e-18              # Дж, CODATA 2022
BOHR_M = 5.29177210903e-11                   # м, CODATA 2018

# --- производные множители, каждый выведен, а не введён ---------------------------------
# 1 а.е.м.·Å²/фс² в эВ: (1.66053906892e-27 кг)·(1e-10 м / 1e-15 с)² = 1.66053906892e-17 Дж,
# делим на заряд электрона -> эВ.
AMU_A2_PER_FS2_IN_EV = ATOMIC_MASS_UNIT * (1e-10 / 1e-15) ** 2 / ELEMENTARY_CHARGE

KB_EV_PER_K = BOLTZMANN_J_PER_K / ELEMENTARY_CHARGE   # эВ/К
HARTREE_IN_EV = HARTREE_J / ELEMENTARY_CHARGE          # эВ
BOHR_IN_ANGSTROM = BOHR_M * 1e10                       # Å
EV_IN_KJ_PER_MOL = ELEMENTARY_CHARGE * 6.02214076e23 / 1000.0  # кДж/моль
EV_IN_KCAL_PER_MOL = EV_IN_KJ_PER_MOL / 4.184           # ккал/моль


def kinetic_energy_ev(masses_amu, velocities_a_per_fs) -> float:
    """Кинетическая энергия в эВ из масс в а.е.м. и скоростей в Å/фс."""
    import numpy as np

    v2 = (np.asarray(velocities_a_per_fs) ** 2).sum(axis=1)
    return 0.5 * float((np.asarray(masses_amu) * v2).sum()) * AMU_A2_PER_FS2_IN_EV


def accelerations_a_per_fs2(forces_ev_per_a, masses_amu):
    """a = F/m с переводом эВ/(Å·а.е.м.) -> Å/фс²."""
    import numpy as np

    f = np.asarray(forces_ev_per_a, dtype=float)
    m = np.asarray(masses_amu, dtype=float)[:, None]
    return f / m / AMU_A2_PER_FS2_IN_EV


def instantaneous_temperature_k(masses_amu, velocities_a_per_fs, n_constraints: int = 0) -> float:
    """
    T из равнораспределения: E_kin = (3N - n_constraints)/2 · kB · T.

    `n_constraints` -- отнятые степени свободы. Три отнимаются, когда суммарный импульс
    системы зафиксирован нулём (обычный случай при старте из покоя центра масс); шесть --
    когда дополнительно зафиксирован полный момент. Значение НЕ угадывается вызывающим
    кодом: тест равнораспределения в atomic/validate проверяет, что заявленное число
    степеней свободы совпадает с фактическим, иначе измеренная T систематически смещена.
    """
    import numpy as np

    dof = 3 * len(np.asarray(masses_amu)) - n_constraints
    if dof <= 0:
        raise ValueError(f"неположительное число степеней свободы: {dof}")
    return 2.0 * kinetic_energy_ev(masses_amu, velocities_a_per_fs) / (dof * KB_EV_PER_K)
