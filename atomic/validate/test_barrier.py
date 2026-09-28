"""
Barriers: first against the exact answer, then against experiment.

The order of checks is the same as throughout the engine: an analytical problem where the
answer is known exactly catches a method error; real chemistry against a reference value
catches an application error. Both are needed: if only the second one passes, there is no
way to tell whether it converged for the right reason or the numbers just happened to match.

Experimental values:
  ammonia inversion barrier = 5.80 kcal/mol = 0.2515 eV
  (Swalen & Ibers, JCP 36 (1962) 1914 -- microwave spectroscopy of the inversion splitting)
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.barrier import find_barrier
from engine.potential import PotentialResult
from engine.state import from_symbols
from engine.units import EV_IN_KCAL_PER_MOL


class DoubleWell:
    """U(x) = h(x²−1)² along the x axis plus a stiff restraint along y and z: the barrier is EXACTLY h, minima at ±1."""

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
    """The barrier must match h to the fourth digit, and the reaction must be thermoneutral."""
    a = from_symbols(["H"], [[-1.0, 0.0, 0.0]])
    b = from_symbols(["H"], [[1.0, 0.0, 0.0]])
    r = find_barrier(a, b, DoubleWell(height), n_images=11, max_iterations=400,
                     climbing_after=30, step_a=0.03, force_tol_ev_per_a=1e-3)
    assert r.converged, f"band did not converge in {r.iterations} iterations"
    assert abs(r.forward_barrier_ev - height) < 1e-3, f"barrier {r.forward_barrier_ev:.5f} vs {height}"
    assert abs(r.reaction_energy_ev) < 1e-6, "a symmetric reaction must be thermoneutral"
    assert r.peak_index == 5, f"the peak of a symmetric path must be in the middle, not {r.peak_index}"


def test_rate_spans_the_range_direct_dynamics_cannot_reach():
    """
    The point of this whole layer, expressed as a number: at 300 K a 0.3 eV barrier gives
    events on the nanosecond scale (dynamics can see them), while 1.0 eV gives one every few
    hours. The latter is unreachable by direct dynamics on any hardware, but computing it via
    the barrier takes minutes.
    """
    a = from_symbols(["H"], [[-1.0, 0.0, 0.0]])
    b = from_symbols(["H"], [[1.0, 0.0, 0.0]])
    fast = find_barrier(a.copy(), b.copy(), DoubleWell(0.3), n_images=11, max_iterations=400,
                        climbing_after=30, step_a=0.03, force_tol_ev_per_a=1e-3)
    slow = find_barrier(a.copy(), b.copy(), DoubleWell(1.0), n_images=11, max_iterations=400,
                        climbing_after=30, step_a=0.03, force_tol_ev_per_a=1e-3)
    k_fast = fast.rate_per_second(300.0)
    k_slow = slow.rate_per_second(300.0)
    assert k_fast > 1e6, f"rate at a 0.3 eV barrier came out to {k_fast:.2e} 1/s"
    assert k_slow < 1e-2, f"rate at a 1.0 eV barrier came out to {k_slow:.2e} 1/s"
    assert k_fast / k_slow > 1e9, "the gap between scales is less than nine orders of magnitude -- check the formula"


@pytest.mark.slow
def test_ammonia_inversion_barrier_against_experiment():
    """
    Ammonia inversion with the cheap backend: 6.11 kcal/mol against the experimental 5.80
    (measured 2026-08-21, FIRE, converged in 39 iterations). The 25% tolerance is set by the
    known accuracy of the semiempirical method on barriers, not fitted to the result; the
    profile must be symmetric, and this check is independent of the magnitude: a symmetric
    reaction has its peak exactly in the middle.
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
    assert r.converged, f"band did not converge: perpendicular force {r.max_perpendicular_force:.3f} eV/Å"
    assert abs(kcal - 5.80) / 5.80 < 0.25, f"barrier {kcal:.2f} kcal/mol vs the experimental 5.80"
    assert r.peak_index == 3, f"the peak of a symmetric inversion must be in the middle, not {r.peak_index}"
    profile = r.energies_ev - r.energies_ev[0]
    assert abs(profile[1] - profile[-2]) < 0.01, f"profile is asymmetric: {np.round(profile, 4)}"
