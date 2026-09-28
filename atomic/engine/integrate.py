"""
Integrators. Two of them, and both are needed for different reasons.

`velocity_verlet` is NVE: it is symplectic, meaning it conserves energy with a bounded
error, and it is exactly the integrator used to check that the forces are correct. If
the energy drifts, either the forces or the step size are at fault, and that shows up
immediately.

`langevin_baoab` is NVT: the BAOAB scheme (Leimkuhler & Matthews 2013), whose error in
configurational averages is a higher order in the step size than a naive Langevin
scheme. A thermostat is needed because real chemistry runs at a given temperature, not
at a given energy.

Freezing atoms is done by ZEROING FORCE AND VELOCITY, not with a large mass: a large
mass leaves the atom slowly creeping, and this project already had a case where a
"stationary" surface was in fact drifting. Immobility must be structural.
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
    """What can be measured after a step without recomputing the potential."""

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
    One NVE step. Returns the potential result AT THE NEW POINT, so the calling code can
    pass it into the next step and not compute the potential twice: on a quantum
    backend, computing the force is the entire cost of a step, and an extra call doubles
    the run time.
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
    BAOAB step: B (half kick) A (half move) O (thermostat) A (half move) B (half kick).

    `friction_per_fs` is the inverse velocity-decay time, 1/fs. The value is NOT tuned to
    the result: too much friction suppresses real dynamics (diffusion, encounter rate),
    too little does not thermostat within a reasonable time, and the correct check is to
    measure the velocity autocorrelation time and confirm it is not shorter than the
    process of interest.
    """
    from .units import KB_EV_PER_K, AMU_A2_PER_FS2_IN_EV

    res = cached if cached is not None else potential.compute(state)
    forces = _apply_frozen(state, res.forces_ev_per_a)
    before = state.positions.copy()

    # B
    state.velocities += 0.5 * dt_fs * accelerations_a_per_fs2(forces, state.masses)
    # A
    state.positions += 0.5 * dt_fs * state.velocities
    # O: exact solution of the Ornstein-Uhlenbeck equation over one step dt
    decay = np.exp(-friction_per_fs * dt_fs)
    # sigma^2 = kB T / m in units of (Å/fs)^2: kB T in eV, divided by mass and by the
    # amu*Å^2/fs^2 -> eV conversion factor. This is equipartition, not a fit.
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
    Equilibrium geometry search with quasi-Newton L-BFGS-B on the energy, using the
    ANALYTICAL gradient from the same potential. The analytical gradient here is not a
    convenience but a condition for feasibility: a numerical gradient on a quantum
    backend would cost 6N calculations per iteration.

    Frozen atoms are excluded from the variables rather than held with a penalty: a
    penalty shifts the minimum, exclusion does not.
    """
    from scipy.optimize import minimize

    free = np.ones(state.n_atoms, dtype=bool) if state.frozen is None else ~state.frozen
    x0 = state.positions[free].ravel().copy()
    calls = {"n": 0}

    def energy_and_grad(x: np.ndarray):
        state.positions[free] = x.reshape(-1, 3)
        res = potential.compute(state)
        calls["n"] += 1
        # we minimize energy, so gradient = -force
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
    Force descent with a LENGTH limit on the step rather than a fixed multiplier: on the
    first step from a poor geometry the force can be huge, and a multiplier that is
    reasonable at the minimum would send the atoms to infinity. Limiting the length
    makes the first step safe without tuning a multiplier. Stopping is based on the
    maximum force, not on the number of steps.
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
