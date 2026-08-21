"""
Ведущий цикл: молекулярная динамика с наблюдением за химией.

Устройство простое и намеренно: цикл шагает интегратором, а после каждого шага (или реже,
если задано) читает связность из электронной структуры и спрашивает наблюдателя, не
случилось ли реакции. Реакции здесь не разрешают и не запрещают -- их регистрируют.

Громкое падение вместо тихой порчи -- отдельное требование, взятое из истории этого проекта:
шесть раз расчёт продолжался на испорченных числах и выглядел успешным. Поэтому нечисловое
состояние, слишком большое смещение за шаг и несошедшийся SCF обрываются исключением с
номером шага, а не пропускаются.
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
    """Что сохраняется об одном наблюдении. Координаты -- копия, а не ссылка."""

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
        """Разброс полной энергии по кадрам -- для NVE это мера качества интегрирования."""
        if len(self.frames) < 2:
            return 0.0
        totals = np.array([f.total_ev for f in self.frames])
        return float(totals.max() - totals.min())

    @property
    def steps_per_second(self) -> float:
        return self.steps_done / self.wall_seconds if self.wall_seconds > 0 else float("nan")


# Максимальное смещение атома за один шаг, при котором расчёт ещё физичен. Значение не
# подгонка: типичная амплитуда тепловых колебаний -- сотые доли ангстрема за фемтосекунду,
# и смещение в половину ангстрема за шаг означает, что шаг велик или силы испорчены.
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
    `temperature_k=None` -> NVE (сохранение энергии, для проверки), иначе NVT по BAOAB.

    `sample_every` управляет только ЗАПИСЬЮ и чтением химии, но не шагом интегрирования:
    связность меняется на масштабе десятков фемтосекунд, а шаг -- полфемтосекунды, поэтому
    читать её каждый шаг незачем. Наблюдатель дребезга при этом настроен на число
    НАБЛЮДЕНИЙ, а не шагов, -- см. ReactionWatcher.
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
                f"шаг {step}: смещение {diag.max_displacement_a:.3f} Å за шаг превышает предел "
                f"{MAX_STEP_DISPLACEMENT_A} Å -- шаг велик или силы испорчены"
            )
        if not np.all(np.isfinite(state.positions)) or not np.all(np.isfinite(state.velocities)):
            bad = int((~np.isfinite(state.positions)).sum() + (~np.isfinite(state.velocities)).sum())
            raise FloatingPointError(f"шаг {step}: {bad} нечисловых компонент состояния")

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
