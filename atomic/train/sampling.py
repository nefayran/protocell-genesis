"""
Набор конфигураций БЕЗ динамики -- то есть почти бесплатно.

Замер, из-за которого это появилось: разметка шла вперемешку с молекулярной динамикой, а
динамика по построению последовательна -- каждый шаг требует силы в предыдущей точке, и
пакетом её не собрать. Из-за этого дорогой учитель вызывался по одному кадру, и на облачной
карте это дало 22-92 атом-расчёта в секунду против 460 на процессоре: карта простаивала.

Расцепление простое: конфигурации порождаются случайной выборкой (сдвиги, повороты, плотности),
что стоит микросекунды, а учитель размечает их ПАКЕТОМ. Так делают опубликованные наборы для
нейросетевых потенциалов, и причина та же -- стоимость набора это стоимость разметки, а не
стоимость геометрии.

Чего этот способ НЕ даёт, и это надо знать: случайная выборка не воспроизводит распределение,
по которому система реально ходит при заданной температуре. Поэтому в наборе остаются и срезы
динамики -- но их доля мала, а покрытие пространства обеспечивает случайная часть. Правильная
проверка -- ошибка на кадрах ИЗ ДИНАМИКИ, отложенных отдельно: если модель хороша только на
случайных конфигурациях, это видно сразу.
"""
from __future__ import annotations

import numpy as np

from engine.state import from_symbols

WATER = np.array([[0.0, 0.0, 0.0], [0.0, -0.7575, 0.5865], [0.0, 0.7575, 0.5865]])


def _rotation(rng: np.random.Generator) -> np.ndarray:
    """Случайный поворот через кватернион: равномерно по группе вращений."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rattled_monomers(rng: np.random.Generator, count: int, max_rattle: float = 0.25):
    """Мономеры со случайным искажением: покрывают растяжение связей и изменение угла."""
    out = []
    for _ in range(count):
        scale = float(rng.uniform(0.02, max_rattle))
        pos = WATER + rng.normal(scale=scale, size=WATER.shape)
        out.append(from_symbols(["O", "H", "H"], pos))
    return out


def dimers(rng: np.random.Generator, count: int, r_range=(2.2, 6.0), rattle: float = 0.08):
    """
    Димеры на всех расстояниях водородной связи и во всех взаимных ориентациях.

    Нижняя граница 2.2 Å выбрана заметно НИЖЕ равновесного расстояния O-O (2.98 Å) умышленно:
    модель обязана знать отталкивание, иначе в динамике сблизившиеся молекулы провалятся друг
    в друга -- это самая частая причина срыва нейросетевого потенциала.
    """
    out = []
    for _ in range(count):
        sep = float(rng.uniform(*r_range))
        a = (WATER - WATER[0]) @ _rotation(rng).T + rng.normal(scale=rattle, size=WATER.shape)
        b = (WATER - WATER[0]) @ _rotation(rng).T + np.array([sep, 0.0, 0.0]) \
            + rng.normal(scale=rattle, size=WATER.shape)
        out.append(from_symbols(["O", "H", "H"] * 2, np.vstack([a, b])))
    return out


def clusters(
    rng: np.random.Generator,
    count: int,
    sizes=(4, 8, 16, 32),
    densities=(0.020, 0.0334, 0.045),
    rattle: float = 0.12,
    min_oo: float = 2.1,
):
    """
    Кластеры в периодической ячейке при разных плотностях -- от разреженной до сжатой.

    `min_oo` -- защита от нефизичных стартов: если два кислорода оказались ближе этого, кадр
    отбрасывается. Без неё в наборе появляются конфигурации с энергией в десятки эВ, и подгонка
    по наименьшим квадратам тратит всю ёмкость на их воспроизведение (одна такая точка весит
    как сотня нормальных).
    """
    out = []
    attempts = 0
    while len(out) < count and attempts < count * 20:
        attempts += 1
        n_mol = int(rng.choice(sizes))
        density = float(rng.choice(densities))
        box = (n_mol / density) ** (1 / 3)
        side = int(np.ceil(n_mol ** (1 / 3)))
        step = box / side
        positions, placed = [], 0
        for a in range(side):
            for b in range(side):
                for c in range(side):
                    if placed >= n_mol:
                        break
                    centre = (np.array([a, b, c]) + 0.5) * step + rng.normal(scale=rattle * 2, size=3)
                    mol = (WATER - WATER[0]) @ _rotation(rng).T + centre
                    mol += rng.normal(scale=rattle, size=mol.shape)
                    positions.append(mol)
                    placed += 1
        pos = np.vstack(positions)
        oxygens = pos[::3]
        # проверка минимального расстояния с учётом периодичности
        d = oxygens[:, None, :] - oxygens[None, :, :]
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=2)
        np.fill_diagonal(r, np.inf)
        if r.min() < min_oo:
            continue
        out.append(
            from_symbols(["O", "H", "H"] * n_mol, pos, cell=np.eye(3) * box, pbc=(True, True, True))
        )
    return out


def build_sampled_states(rng: np.random.Generator, n_monomers=800, n_dimers=1200, n_clusters=3000):
    """Полный набор конфигураций одним вызовом. Стоимость -- секунды, а не часы."""
    states = rattled_monomers(rng, n_monomers)
    states += dimers(rng, n_dimers)
    states += clusters(rng, n_clusters)
    rng.shuffle(states)
    return states
