"""
Main loop: molecular dynamics with chemistry observation.

The design is deliberately simple: the loop steps the integrator, and after every step
(or less often, if configured) reads connectivity from the electronic structure and asks
the observer whether a reaction happened. Reactions are neither allowed nor forbidden
here, only recorded.

A loud crash instead of silent corruption is a separate requirement, drawn from this
project's history: six times a run continued on corrupted numbers and looked successful.
That is why non-numeric state, a per-step displacement that is too large, and an
unconverged SCF all abort with an exception carrying the step number, instead of being
skipped.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chem import ReactionEvent, ReactionWatcher
from .integrate import langevin_baoab, velocity_verlet
from .potential import Potential
from .state import AtomicState


@dataclass
class Frame:
    """What is stored about one observation. Coordinates are a copy, not a reference."""

    step: int
    time_fs: float
    potential_ev: float
    kinetic_ev: float
    total_ev: float
    temperature_k: float
    max_force_ev_per_a: float
    positions: np.ndarray
    species: tuple[str, ...] = ()


@dataclass
class TrajectoryResult:
    frames: list[Frame] = field(default_factory=list)
    events: list[ReactionEvent] = field(default_factory=list)
    steps_done: int = 0
    wall_seconds: float = 0.0

    @property
    def energy_drift_ev(self) -> float:
        """Spread of the total energy across frames: for NVE this is the integration quality measure."""
        if len(self.frames) < 2:
            return 0.0
        totals = np.array([f.total_ev for f in self.frames])
        return float(totals.max() - totals.min())

    @property
    def steps_per_second(self) -> float:
        return self.steps_done / self.wall_seconds if self.wall_seconds > 0 else float("nan")


# Maximum per-step atom displacement at which the run is still physical. The value is
# not a fit: the typical amplitude of thermal vibrations is hundredths of an angstrom per
# femtosecond, and a half-angstrom displacement per step means the step is too large or
# the forces are corrupted.
MAX_STEP_DISPLACEMENT_A = 0.5


def run(
    state: AtomicState,
    potential: Potential,
    steps: int,
    dt_fs: float = 0.5,
    temperature_k: float | None = 300.0,
    friction_per_fs: float = 0.01,
    seed: int = 0,
    sample_every: int = 10,
    watch_chemistry: bool = True,
    watcher_persistence: int = 3,
    on_frame=None,
    on_event=None,
) -> TrajectoryResult:
    """
    `temperature_k=None` -> NVE (energy conservation, for validation), otherwise NVT via BAOAB.

    `sample_every` controls only the RECORDING and reading of chemistry, not the
    integration step: connectivity changes on a scale of tens of femtoseconds, while the
    step is half a femtosecond, so there is no need to read it every step. The debounce
    observer is meanwhile tuned to the number of OBSERVATIONS, not steps: see
    ReactionWatcher.
    """
    import time

    rng = np.random.default_rng(seed)
    watcher = ReactionWatcher(state.numbers, persistence=watcher_persistence) if watch_chemistry else None
    out = TrajectoryResult()
    cached = potential.compute(state)
    started = time.perf_counter()

    for step in range(1, steps + 1):
        if temperature_k is None:
            cached, diag = velocity_verlet(state, potential, dt_fs, cached=cached)
        else:
            cached, diag = langevin_baoab(
                state, potential, dt_fs, temperature_k, friction_per_fs, rng, cached=cached
            )

        if diag.max_displacement_a > MAX_STEP_DISPLACEMENT_A:
            raise FloatingPointError(
                f"step {step}: displacement {diag.max_displacement_a:.3f} Å per step exceeds the limit "
                f"{MAX_STEP_DISPLACEMENT_A} Å (step too large or forces corrupted)"
            )
        if not np.all(np.isfinite(state.positions)) or not np.all(np.isfinite(state.velocities)):
            bad = int((~np.isfinite(state.positions)).sum() + (~np.isfinite(state.velocities)).sum())
            raise FloatingPointError(f"step {step}: {bad} non-numeric state components")

        if step % sample_every == 0 or step == steps:
            species: tuple[str, ...] = ()
            if watcher is not None and "bond_orders" in cached.extra:
                from .chem import bonds_from_orders, find_species

                bonds = bonds_from_orders(cached.extra["bond_orders"])
                species = tuple(
                    sorted(s.formula for s in find_species(state.numbers, bonds, cached.extra.get("charges")))
                )
                event = watcher.observe(step, cached.extra["bond_orders"], cached.extra.get("charges"))
                if event is not None:
                    out.events.append(event)
                    if on_event is not None:
                        on_event(event)
            frame = Frame(
                step=step,
                time_fs=step * dt_fs,
                potential_ev=diag.potential_ev,
                kinetic_ev=diag.kinetic_ev,
                total_ev=diag.total_ev,
                temperature_k=diag.temperature_k,
                max_force_ev_per_a=diag.max_force_ev_per_a,
                positions=state.positions.copy(),
                species=species,
            )
            out.frames.append(frame)
            if on_frame is not None:
                on_frame(frame)

    out.steps_done = steps
    out.wall_seconds = time.perf_counter() - started
    return out
