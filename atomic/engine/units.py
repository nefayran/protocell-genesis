"""
Engine units: Å, fs, amu, eV, K. The choice is not arbitrary: it is the standard set
for classical molecular dynamics, where the integration step is expressed as an integer
number of femtoseconds and masses are taken directly from the atomic weight table without
recalculation.

All conversion factors are the SI 2019 definitions or CODATA values, not fitted numbers:
if any of them turns out to be wrong, that shows up as a violation of energy conservation
in atomic/validate (the NVE test), not as a mismatch with anyone's expectation.
"""
from __future__ import annotations

# --- SI and CODATA definitions -----------------------------------------------------------
ELEMENTARY_CHARGE = 1.602176634e-19          # C, SI 2019 definition
ATOMIC_MASS_UNIT = 1.66053906892e-27         # kg, CODATA 2022
BOLTZMANN_J_PER_K = 1.380649e-23             # J/K, SI 2019 definition
HARTREE_J = 4.3597447222060e-18              # J, CODATA 2022
BOHR_M = 5.29177210903e-11                   # m, CODATA 2018

# --- derived factors, each one derived rather than entered directly ---------------------
# 1 amu*Å^2/fs^2 in eV: (1.66053906892e-27 kg)*(1e-10 m / 1e-15 s)^2 = 1.66053906892e-17 J,
# divide by the electron charge -> eV.
AMU_A2_PER_FS2_IN_EV = ATOMIC_MASS_UNIT * (1e-10 / 1e-15) ** 2 / ELEMENTARY_CHARGE

KB_EV_PER_K = BOLTZMANN_J_PER_K / ELEMENTARY_CHARGE   # eV/K
HARTREE_IN_EV = HARTREE_J / ELEMENTARY_CHARGE          # eV
BOHR_IN_ANGSTROM = BOHR_M * 1e10                       # Å
EV_IN_KJ_PER_MOL = ELEMENTARY_CHARGE * 6.02214076e23 / 1000.0  # kJ/mol
EV_IN_KCAL_PER_MOL = EV_IN_KJ_PER_MOL / 4.184           # kcal/mol


def kinetic_energy_ev(masses_amu, velocities_a_per_fs) -> float:
    """Kinetic energy in eV from masses in amu and velocities in Å/fs."""
    import numpy as np

    v2 = (np.asarray(velocities_a_per_fs) ** 2).sum(axis=1)
    return 0.5 * float((np.asarray(masses_amu) * v2).sum()) * AMU_A2_PER_FS2_IN_EV


def accelerations_a_per_fs2(forces_ev_per_a, masses_amu):
    """a = F/m with conversion eV/(Å*amu) -> Å/fs^2."""
    import numpy as np

    f = np.asarray(forces_ev_per_a, dtype=float)
    m = np.asarray(masses_amu, dtype=float)[:, None]
    return f / m / AMU_A2_PER_FS2_IN_EV


def instantaneous_temperature_k(masses_amu, velocities_a_per_fs, n_constraints: int = 0) -> float:
    """
    T from equipartition: E_kin = (3N - n_constraints)/2 * kB * T.

    `n_constraints` is the number of removed degrees of freedom. Three are removed when
    the system's total momentum is pinned to zero (the usual case when starting from
    center-of-mass rest); six when the total angular momentum is additionally pinned.
    The value is NOT guessed by the calling code: the equipartition test in
    atomic/validate checks that the claimed number of degrees of freedom matches the
    actual one, otherwise the measured T is systematically biased.
    """
    import numpy as np

    dof = 3 * len(np.asarray(masses_amu)) - n_constraints
    if dof <= 0:
        raise ValueError(f"non-positive number of degrees of freedom: {dof}")
    return 2.0 * kinetic_energy_ev(masses_amu, velocities_a_per_fs) / (dof * KB_EV_PER_K)
