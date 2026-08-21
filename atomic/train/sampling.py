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


# Пределы расстояний, ниже которых конфигурация нефизична. Числа взяты не на глаз: связь O-H
# в воде 0.958 Å, и сжатие её ниже 0.75 Å стоит уже единиц электронвольт; несвязанные пары
# ближе 1.4 Å в жидкой воде не встречаются вовсе. ИЗМЕРЕННАЯ цена отсутствия этой проверки:
# в наборе из 330 кадров 78 (24%) имели силу выше 100 эВ/Å, а худший кадр -- 1.1e7 эВ/Å при
# минимальном расстоянии 0.394 Å; в квадратичной невязке один такой кадр весит как миллион
# нормальных, и обучение уходило в мусор (невязка 2.9e15).
MIN_BONDED_A = 0.75
MIN_NONBONDED_A = 1.35


def _too_close(positions: np.ndarray, cell=None, molecule_size: int = 3) -> bool:
    """Есть ли в конфигурации пара ближе допустимого. Внутримолекулярные пары судятся мягче."""
    p = np.asarray(positions)
    d = p[:, None, :] - p[None, :, :]
    if cell is not None:
        length = float(cell[0, 0])
        d -= length * np.round(d / length)
    r = np.linalg.norm(d, axis=2)
    np.fill_diagonal(r, np.inf)
    n = len(p)
    same_molecule = (np.arange(n)[:, None] // molecule_size) == (np.arange(n)[None, :] // molecule_size)
    if (r[same_molecule] < MIN_BONDED_A).any():
        return True
    return bool((r[~same_molecule] < MIN_NONBONDED_A).any())


def rattled_monomers(rng: np.random.Generator, count: int, max_rattle: float = 0.12):
    """
    Мономеры со случайным искажением: покрывают растяжение связей и изменение угла.

    Размах искажения снижен с 0.25 до 0.12 Å по замеру: при 0.25 связь O-H уходила ниже 0.7 Å,
    то есть в область, где энергия растёт на порядки и кадр становится выбросом.
    """
    out = []
    while len(out) < count:
        scale = float(rng.uniform(0.02, max_rattle))
        pos = WATER + rng.normal(scale=scale, size=WATER.shape)
        if not _too_close(pos):
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
    attempts = 0
    while len(out) < count and attempts < count * 50:
        attempts += 1
        sep = float(rng.uniform(*r_range))
        a = (WATER - WATER[0]) @ _rotation(rng).T + rng.normal(scale=rattle, size=WATER.shape)
        b = (WATER - WATER[0]) @ _rotation(rng).T + np.array([sep, 0.0, 0.0]) \
            + rng.normal(scale=rattle, size=WATER.shape)
        pos = np.vstack([a, b])
        if not _too_close(pos):
            out.append(from_symbols(["O", "H", "H"] * 2, pos))
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
        # проверка ВСЕХ пар, а не только кислородных: замер показал, что водороды сходились
        # до 0.394 Å при целых кислородах, и именно эти кадры отравляли обучение
        if _too_close(pos, cell=np.eye(3) * box):
            continue
        oxygens = pos[::3]
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


MAX_FORCE_EV_PER_A = 50.0


def filter_outliers(frames, max_force=MAX_FORCE_EV_PER_A):
    """
    Отбраковка по силе ПОСЛЕ разметки -- вторая линия защиты после геометрической.

    Порог 50 эВ/Å: при 300-1000 K силы в воде редко превышают 20 эВ/Å, поэтому всё выше --
    это не редкая конфигурация, а нефизичная. Число выброшенных кадров ОБЯЗАНО печататься:
    молчаливая отбраковка четверти набора выглядела бы как удачное обучение.
    """
    kept, dropped = [], []
    for fr in frames:
        if np.abs(fr["forces"]).max() > max_force:
            dropped.append(fr)
        else:
            kept.append(fr)
    return kept, dropped


def build_sampled_states(rng: np.random.Generator, n_monomers=800, n_dimers=1200, n_clusters=3000):
    """Полный набор конфигураций одним вызовом. Стоимость -- секунды, а не часы."""
    states = rattled_monomers(rng, n_monomers)
    states += dimers(rng, n_dimers)
    states += clusters(rng, n_clusters)
    rng.shuffle(states)
    return states
