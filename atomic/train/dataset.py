"""
Обучающий набор: конфигурации плюс разметка учителем.

Что попадает в набор -- решение не менее важное, чем вид модели. Модель воспроизведёт только
то, что видела: если показать ей лишь равновесную воду, она не будет знать ни сжатых
контактов, ни растянутых связей, и в динамике сорвётся именно там. Поэтому набор строится
намеренно РАЗНООБРАЗНЫМ:

  1. искажённые мономеры -- растяжение связей и изменение угла вокруг равновесия;
  2. димеры на разных расстояниях и поворотах -- водородная связь во всём диапазоне;
  3. срезы траекторий при разных температурах и плотностях -- то, что реально встречается
     в динамике, включая редкие тесные контакты.

Разметка -- энергия и силы от учителя. Силы дают 3N уравнений на конфигурацию против одного
от энергии, поэтому они и есть основная ценность каждой дорогой разметки.
"""
from __future__ import annotations

import numpy as np

from engine.md import run
from engine.state import from_symbols

WATER_GEOM = np.array([[0.0, 0.0, 0.0], [0.0, -0.7575, 0.5865], [0.0, 0.7575, 0.5865]])


def water_molecule(rng: np.random.Generator, distortion: float = 0.0) -> tuple[list[str], np.ndarray]:
    """Молекула воды со случайным искажением связей и угла."""
    pos = WATER_GEOM.copy()
    if distortion > 0:
        pos[1:] += rng.normal(scale=distortion, size=(2, 3))
    return ["O", "H", "H"], pos


def water_cluster(n_molecules: int, rng: np.random.Generator, density: float = 0.0334):
    """
    Кластер молекул воды в кубе с плотностью жидкой воды (0.0334 молекул/Å³ при 300 K).

    Молекулы ставятся на решётку и поворачиваются случайно, потом решётка возмущается: без
    поворотов набор был бы вырожден по ориентациям, и модель не увидела бы разных взаимных
    положений водородных связей.
    """
    from scipy.spatial.transform import Rotation

    side = int(np.ceil(n_molecules ** (1 / 3)))
    box = (n_molecules / density) ** (1 / 3)
    step = box / side
    symbols: list[str] = []
    positions = []
    placed = 0
    for a in range(side):
        for b in range(side):
            for c in range(side):
                if placed >= n_molecules:
                    break
                centre = np.array([a, b, c]) * step + rng.normal(scale=0.15, size=3)
                rot = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
                mol = (WATER_GEOM - WATER_GEOM[0]) @ rot.T + centre
                positions.append(mol)
                symbols += ["O", "H", "H"]
                placed += 1
    return symbols, np.vstack(positions), np.eye(3) * box


def label_frames(states, teacher) -> list[dict]:
    """Разметка списка состояний учителем: энергия и силы."""
    frames = []
    for st in states:
        res = teacher.compute(st)
        frames.append(
            {
                "positions": st.positions.copy(),
                "numbers": st.numbers.copy(),
                "energy": res.energy_ev,
                "forces": res.forces_ev_per_a.copy(),
                "cell": None if st.cell is None else st.cell.copy(),
            }
        )
    return frames


def build_water_dataset(
    teacher,
    rng: np.random.Generator,
    n_monomers: int = 40,
    n_dimers: int = 60,
    cluster_sizes: tuple[int, ...] = (4, 8),
    md_frames_per_cluster: int = 20,
    md_temperature_k: float = 400.0,
    md_steps_between: int = 20,
) -> list[dict]:
    """
    Полный набор для воды. Температура срезов взята ВЫШЕ рабочей (400 K против 300 K)
    умышленно: набор должен покрывать более широкий диапазон, чем прогон, иначе модель будет
    работать на границе своей области определения там, где динамика зайдёт чуть дальше.
    """
    states = []

    for _ in range(n_monomers):
        sym, pos = water_molecule(rng, distortion=float(rng.uniform(0.0, 0.18)))
        states.append(from_symbols(sym, pos))

    for _ in range(n_dimers):
        from scipy.spatial.transform import Rotation

        sep = float(rng.uniform(2.4, 5.5))
        rot = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
        mol_a = WATER_GEOM - WATER_GEOM[0]
        mol_b = (WATER_GEOM - WATER_GEOM[0]) @ rot.T + np.array([sep, 0.0, 0.0])
        states.append(from_symbols(["O", "H", "H", "O", "H", "H"], np.vstack([mol_a, mol_b])))

    frames = label_frames(states, teacher)

    # --- срезы траекторий: то, что реально встречается в динамике --------------------
    for n_mol in cluster_sizes:
        sym, pos, cell = water_cluster(n_mol, rng)
        st = from_symbols(sym, pos, cell=cell, pbc=(True, True, True))
        st.set_maxwell_boltzmann(md_temperature_k, rng)
        for _ in range(md_frames_per_cluster):
            run(
                st, teacher, steps=md_steps_between, dt_fs=0.5,
                temperature_k=md_temperature_k, seed=int(rng.integers(1 << 30)),
                sample_every=md_steps_between, watch_chemistry=False,
            )
            frames += label_frames([st.copy()], teacher)

    return frames
