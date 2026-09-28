"""
Engine physics against exact statements, not against expectations.

Every test checks something that MUST hold if the code is correct: force matching the
numerical gradient of the energy, conservation of momentum, conservation of energy in NVE
and its order with respect to the step, the known period of a harmonic oscillator, the
thermostat converging to the target temperature. All checks run on an analytical potential
with a known derivative: if they pass, a strange result on the quantum backend is no longer
an integrator error, and vice versa.
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.integrate import langevin_baoab, velocity_verlet
from engine.potential import HarmonicBonds, LennardJones, numerical_forces
from engine.state import from_symbols
from engine.units import AMU_A2_PER_FS2_IN_EV, KB_EV_PER_K


def test_units_derived_not_guessed():
    assert abs(AMU_A2_PER_FS2_IN_EV - 103.6426957) < 1e-4
    assert abs(KB_EV_PER_K - 8.617333262e-5) < 1e-12


def test_force_equals_numerical_gradient_harmonic():
    st = from_symbols(["H", "H"], [[0, 0, 0], [0, 0, 0.85]])
    pot = HarmonicBonds([(0, 1, 30.0, 0.74)])
    analytic = pot.compute(st).forces_ev_per_a
    numeric = numerical_forces(pot, st)
    assert np.abs(analytic - numeric).max() < 1e-6


def test_force_equals_numerical_gradient_lj():
    rng = np.random.default_rng(7)
    st = from_symbols(["Ar"] * 6, rng.normal(scale=2.0, size=(6, 3)) + np.array([5.0, 5.0, 5.0]))
    pot = LennardJones(epsilon_ev=0.0104, sigma_a=3.40)
    analytic = pot.compute(st).forces_ev_per_a
    numeric = numerical_forces(pot, st, h=1e-5)
    scale = max(1e-12, np.abs(analytic).max())
    assert np.abs(analytic - numeric).max() / scale < 1e-4


def test_momentum_conserved_exactly_in_nve():
    rng = np.random.default_rng(11)
    st = from_symbols(["Ar"] * 8, rng.normal(scale=2.5, size=(8, 3)) + 6.0)
    st.set_maxwell_boltzmann(120.0, rng)
    pot = LennardJones(epsilon_ev=0.0104, sigma_a=3.40)
    p0 = st.total_momentum()
    cached = None
    for _ in range(200):
        cached, _ = velocity_verlet(st, pot, dt_fs=1.0, cached=cached)
    drift = float(np.abs(st.total_momentum() - p0).max())
    assert drift < 1e-10, f"momentum drifted by {drift}"


def test_energy_conserved_in_nve_and_scales_with_dt_squared():
    """A second-order symplectic integrator: the energy error falls off as dt². This checks
    not just that the drift is small but also the ORDER: that is what distinguishes a
    correct integrator from one with an accidentally small error."""

    def drift_for(dt_fs: float) -> float:
        rng = np.random.default_rng(3)
        st = from_symbols(["Ar"] * 8, rng.normal(scale=2.5, size=(8, 3)) + 6.0)
        st.set_maxwell_boltzmann(80.0, rng)
        pot = LennardJones(epsilon_ev=0.0104, sigma_a=3.40)
        res = pot.compute(st)
        e0 = res.energy_ev + st.kinetic_energy_ev()
        worst, cached = 0.0, res
        for _ in range(int(2000 / dt_fs)):
            cached, diag = velocity_verlet(st, pot, dt_fs=dt_fs, cached=cached)
            worst = max(worst, abs(diag.total_ev - e0))
        return worst

    coarse, fine = drift_for(4.0), drift_for(2.0)
    assert fine < coarse
    ratio = coarse / max(fine, 1e-18)
    assert 2.0 < ratio < 8.0, f"accuracy order is not second: drift ratio {ratio:.2f}"


def test_harmonic_period_matches_analytic():
    k, mu_amu = 30.0, 1.008 / 2.0
    omega = np.sqrt(k / (mu_amu * AMU_A2_PER_FS2_IN_EV))  # 1/fs
    period_fs = 2 * np.pi / omega

    st = from_symbols(["H", "H"], [[0, 0, 0], [0, 0, 0.84]])
    pot = HarmonicBonds([(0, 1, k, 0.74)])
    dt, steps = 0.05, 4000
    seps, cached = [], None
    for _ in range(steps):
        cached, _ = velocity_verlet(st, pot, dt_fs=dt, cached=cached)
        seps.append(st.distance(0, 1))
    seps = np.asarray(seps)
    minima = [i for i in range(1, len(seps) - 1) if seps[i] < seps[i - 1] and seps[i] < seps[i + 1]]
    measured = float(np.mean(np.diff(minima)) * dt)
    assert abs(measured - period_fs) / period_fs < 0.01, (
        f"period {measured:.3f} vs the analytical {period_fs:.3f} fs"
    )


def test_langevin_reaches_target_temperature():
    rng = np.random.default_rng(19)
    grid = np.stack(np.meshgrid(*[np.arange(3) * 3.8] * 3), -1).reshape(-1, 3)
    st = from_symbols(["Ar"] * 27, grid)
    pot = LennardJones(epsilon_ev=0.0104, sigma_a=3.40)
    target = 150.0
    cached, temps = None, []
    for step in range(4000):
        cached, diag = langevin_baoab(
            st, pot, dt_fs=2.0, temperature_k=target, friction_per_fs=0.01, rng=rng, cached=cached
        )
        if step > 1500:
            temps.append(diag.temperature_k)
    mean_t = float(np.mean(temps))
    assert abs(mean_t - target) / target < 0.10, f"thermostat is holding {mean_t:.1f} K instead of {target} K"


def test_potential_refuses_nonfinite():
    st = from_symbols(["H", "H"], [[0, 0, 0], [0, 0, 0]])
    with pytest.raises(FloatingPointError):
        HarmonicBonds([(0, 1, 30.0, 0.74)]).compute(st)
