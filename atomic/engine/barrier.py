"""
Reaction barrier: the nudged elastic band method with climbing image (CI-NEB).

Why it exists in this engine. Direct dynamics measurably does NOT cross a barrier of a
few electronvolts: 2000 steps at 6000 K on hydrogen peroxide produced not a single bond
breaking (atomic/validate/test_reactions.py). The reason is not patience but statistics:
at an attempt frequency of about one per 27 fs and the Boltzmann factor for 2 eV, the
expected number of events over the available time is below one. So rare events must not
be waited for but COMPUTED: find the minimum-energy path between two states, take its
peak, and get the rate from transition state theory. This converts the available time
from nanoseconds to seconds.

Method: Henkelman & Jonsson, JCP 113 (2000) 9978 (NEB) and Henkelman, Uberuaga & Jonsson,
JCP 113 (2000) 9901 (climbing image). Design:
  * a chain of intermediate states ("images") between the start and the end;
  * the true force is projected PERPENDICULAR to the path, the spring force ALONG it;
    otherwise the chain either slides down into minima or cuts corners;
  * the highest-energy image climbs UP along the path (climbing image); without this the
    peak falls between images and the barrier is systematically underestimated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .potential import Potential
from .state import AtomicState


@dataclass
class BarrierResult:
    """The found path and its peak."""

    energies_ev: np.ndarray          # energy of each image
    images: list[np.ndarray]         # coordinates of each image
    forward_barrier_ev: float        # peak minus start
    reverse_barrier_ev: float        # peak minus end
    reaction_energy_ev: float        # end minus start
    peak_index: int
    converged: bool
    iterations: int
    max_perpendicular_force: float

    def rate_per_second(self, temperature_k: float, prefactor_hz: float = 1.0e13) -> float:
        """
        Transition rate from transition state theory: k = A exp(-Ea / kB T).

        The prefactor 1e13 Hz is NOT a measured quantity but the order of magnitude of a
        typical bond vibration frequency (about 1000 cm^-1 = 3e13 Hz); a proper estimate
        would require computing the vibrational frequencies at the minimum and at the
        saddle point. That is why the number is returned together with an explicit
        argument: whoever uses it must know that the prefactor is approximate, while the
        exponential part is actually computed.
        """
        from .units import KB_EV_PER_K

        return float(prefactor_hz * np.exp(-self.forward_barrier_ev / (KB_EV_PER_K * temperature_k)))


def _tangent(prev: np.ndarray, cur: np.ndarray, nxt: np.ndarray, e_prev: float, e_cur: float, e_next: float) -> np.ndarray:
    """
    Tangent to the path using the "improved" scheme: the direction is taken toward the
    HIGHER-energy neighbor.

    The naive central difference (nxt - prev) produces kinks on steep sections and leads
    to spurious chain distortions; this is a known problem of early NEB implementations,
    and the improved tangent (Henkelman & Jonsson 2000, section II) removes it.
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
    Searches for the minimum-energy path between `initial` and `final`.

    The initial chain is a linear interpolation of coordinates. This works when the final
    state is obtained from the initial one by a small displacement (proton transfer,
    rotation, breaking one bond), and does NOT work when several bonds rearrange between
    them: there, linear interpolation drives atoms through each other, and the chain
    starts in an unphysical region. The telltale sign is a huge energy for the initial
    images, and it is visible in the returned array, not hidden.

    `climbing_after` is the iteration from which the peak starts climbing up. It must not
    be enabled earlier: until the chain has straightened out, the "peak" could turn out
    to be a random image.
    """
    if initial.n_atoms != final.n_atoms:
        raise ValueError("number of atoms in the initial and final state does not match")
    if not np.array_equal(initial.numbers, final.numbers):
        raise ValueError("atom species in the initial and final state do not match")

    # linear interpolation; the endpoint images are fixed
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

    # --- FIRE (Bitzek et al., PRL 97 (2006) 170201) instead of a fixed-step descent --------
    # The measurement that motivated this rewrite: on ammonia inversion, a plain descent
    # with a 0.03 Å step did not converge in 120 iterations (perpendicular force stayed at
    # 1.64 eV/Å), even though the barrier was already close to the experimental value.
    # FIRE adds inertia and picks the step itself: while force and velocity point the same
    # way, the step grows; as soon as the direction flips, the velocity is reset and the
    # step shrinks. No hand-tuned number beyond the initial step appears as a result.
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
                # the peak climbs UP: the true force along the path is inverted, no spring force
                total = f_perp - f_parallel_mag * tau
            else:
                spring = spring_k * (
                    np.linalg.norm(images[k + 1] - images[k]) - np.linalg.norm(images[k] - images[k - 1])
                )
                total = f_perp + spring * tau
            worst_perp = max(worst_perp, float(np.abs(f_perp).max()))
            updates[k] = total          # this is a FORCE; the step is handled by FIRE below

        # --- FIRE step for the whole band at once: it is one system, not a set of independent points ---
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
            # the step-length limit stays in place: it protects the first iteration from bad geometry
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
