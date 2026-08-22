"""
Органика для набора: жирные кислоты, глицерин, их смеси с водой.

Зачем это здесь. Потенциал, обученный на воде, знает водород и кислород -- и ничего не знает о
цепочках углерода, из которых состоят мембраны. Между тем именно на них держится вся исходная
задача: жирная кислота, эстерная связь глицерина, водородная связь между карбоксильными
головами. Пока углерода нет, движок не может посчитать НИ ОДНУ из тех величин, которые в
предыдущем слое этого проекта были придуманы (энергия связи, барьер гидролиза, энергия
ассоциации голов) -- а именно они и были главной ложью старой модели.

Как строится геометрия. Молекула собирается грубо, по табличным длинам связей и углам, а затем
РЕЛАКСИРУЕТСЯ учителем -- то есть точная геометрия получается расчётом, а не подгонкой руками.
Так же поступают с любой новой молекулой: грубый скелет плюс релаксация, без ручной правки
координат.
"""
from __future__ import annotations

import numpy as np

from engine.state import from_symbols

# Табличные длины связей (Å) и углы (градусы) для грубой сборки. Значения справочные:
# C-C 1.53, C-H 1.09, C=O 1.21, C-O 1.36, O-H 0.97 (Harmony 1990; Herzberg).
BOND_CC = 1.53
BOND_CH = 1.09
BOND_CO_DOUBLE = 1.21
BOND_CO_SINGLE = 1.36
# C-O в СПИРТЕ длиннее, чем в карбоксиле: 1.43 против 1.36 (Harmony 1990). Первая версия
# ставила глицерину карбоксильное значение, и релаксация из такого старта почти не двигала
# связи -- выходили 1.336-1.388 Å, то есть стартовые числа, а не равновесные. Ошибка
# выглядела как «получился другой изомер», хотя это была несошедшаяся геометрия.
BOND_CO_ALCOHOL = 1.43
BOND_OH = 0.97
TETRAHEDRAL_DEG = 109.47


def _tetrahedral_free_directions(centre: np.ndarray, bonded: list[np.ndarray]) -> list[np.ndarray]:
    """
    Свободные тетраэдрические направления у атома, часть связей которого уже занята.

    Зачем это вместо смещений по осям. Первая версия ставила заместители сдвигами вида
    (0, 0, ±1.09), и релаксация ОТРЫВАЛА часть водородов: у глицерина после оптимизации два
    водорода оставались без единой связи, молекула превращалась в C3H6O3 плюс два свободных
    атома, а связи C-O выходили 1.31-1.40 Å вместо спиртовых 1.43. Ошибка выглядела как «не тот
    изомер», хотя это была невозможная стартовая геометрия.

    Здесь направления строятся честно: для одной занятой связи -- конус под тетраэдрическим
    углом к ней, для двух -- две оставшиеся вершины тетраэдра, для трёх -- одна.
    """
    if not bonded:
        return [np.array([0.0, 0.0, 1.0])]
    used = [(np.asarray(b) - centre) / np.linalg.norm(np.asarray(b) - centre) for b in bonded]
    cos_t = np.cos(np.radians(TETRAHEDRAL_DEG))

    if len(used) == 1:
        axis = used[0]
        # любой перпендикуляр к оси
        tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(axis, tmp)
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(axis, perp)
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t**2))
        return [
            cos_t * axis + sin_t * (np.cos(a) * perp + np.sin(a) * perp2)
            for a in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ]
    if len(used) == 2:
        bisector = used[0] + used[1]
        bisector /= np.linalg.norm(bisector)
        normal = np.cross(used[0], used[1])
        normal /= np.linalg.norm(normal)
        # две оставшиеся вершины лежат в плоскости, перпендикулярной первым двум
        half = np.radians(TETRAHEDRAL_DEG / 2.0)
        return [-bisector * np.cos(half) + normal * np.sin(half),
                -bisector * np.cos(half) - normal * np.sin(half)]
    total = sum(used)
    norm = np.linalg.norm(total)
    return [-total / norm] if norm > 1e-9 else [np.array([0.0, 0.0, 1.0])]


def _zigzag_chain(n_carbon: int, start=(0.0, 0.0, 0.0)) -> np.ndarray:
    """
    Углеродный скелет зигзагом в плоскости xy -- как в вытянутой алкановой цепи.

    Зигзаг, а не прямая линия: тетраэдрический угол при углероде равен 109.47°, и прямая цепь
    была бы геометрически невозможной. Шаг вдоль оси и поперёк выводится из длины связи и угла,
    а не подбирается.
    """
    half = np.radians(TETRAHEDRAL_DEG / 2.0)
    dx = BOND_CC * np.sin(half)
    dy = BOND_CC * np.cos(half)
    pos = []
    for k in range(n_carbon):
        pos.append([start[0] + k * dx, start[1] + (dy if k % 2 else 0.0), start[2]])
    return np.array(pos)


def alkanoic_acid(n_carbon: int):
    """
    Жирная кислота CH3-(CH2)_{n-2}-COOH грубой сборкой.

    Водороды расставляются попарно перпендикулярно плоскости скелета -- этого достаточно для
    старта релаксации; точные положения даёт учитель. Карбоксильная группа собирается на
    ПОСЛЕДНЕМ углероде: голова в реальной кислоте всегда концевая.
    """
    if n_carbon < 2:
        raise ValueError("жирная кислота начинается с двух углеродов")
    chain = _zigzag_chain(n_carbon)
    symbols = ["C"] * n_carbon
    positions = list(chain)

    # водороды на всех углеродах, кроме карбоксильного
    for k in range(n_carbon - 1):
        base = chain[k]
        for sign in (+1.0, -1.0):
            positions.append(base + np.array([0.0, 0.0, sign * BOND_CH]))
            symbols.append("H")
    # концевой метил получает третий водород вдоль цепи
    axis = chain[0] - chain[1]
    axis /= np.linalg.norm(axis)
    positions.append(chain[0] + axis * BOND_CH)
    symbols.append("H")

    # карбоксил на последнем углероде: =O и -OH
    tail = chain[-1]
    direction = tail - chain[-2]
    direction /= np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0], 0.0])
    o_double = tail + direction * BOND_CO_DOUBLE * 0.5 + perp * BOND_CO_DOUBLE * 0.87
    o_single = tail + direction * BOND_CO_SINGLE * 0.5 - perp * BOND_CO_SINGLE * 0.87
    positions += [o_double, o_single, o_single + np.array([0.0, 0.0, BOND_OH])]
    symbols += ["O", "O", "H"]
    return symbols, np.array(positions)


def glycerol():
    """
    Глицерин C3H8O3: три углерода, на каждом гидроксил.

    Именно он делает мембранный амфифил двухвостым В РЕАЛЬНОЙ ХИМИИ -- две жирные кислоты
    садятся на его гидроксилы эстерными связями. В предыдущем слое проекта двухвостость была
    разрешением валентности, а не молекулой; здесь молекула есть.
    """
    chain = _zigzag_chain(3)
    symbols = ["C", "C", "C"]
    positions = list(chain)
    for k in range(3):
        neighbours = [chain[j] for j in (k - 1, k + 1) if 0 <= j < 3]
        free = _tetrahedral_free_directions(chain[k], neighbours)
        # первое свободное направление -- гидроксилу, остальные -- водородам
        oxygen = chain[k] + free[0] * BOND_CO_ALCOHOL
        positions.append(oxygen)
        symbols.append("O")
        # водород гидроксила ставится под тетраэдрическим углом к связи C-O, а не «вбок»
        oh_dirs = _tetrahedral_free_directions(oxygen, [chain[k]])
        positions.append(oxygen + oh_dirs[0] * BOND_OH)
        symbols.append("H")
        for direction in free[1:]:
            positions.append(chain[k] + direction * BOND_CH)
            symbols.append("H")
    return symbols, np.array(positions)


def relax_with(teacher, symbols, positions, force_tol=0.005, max_iterations=800):
    """Грубый скелет -> равновесная геометрия расчётом учителя."""
    from engine.integrate import optimise_lbfgs

    state = from_symbols(symbols, positions)
    optimise_lbfgs(state, teacher, force_tol_ev_per_a=force_tol, max_iterations=max_iterations)
    return state


def rattled(state, rng: np.random.Generator, scale: float = 0.08, count: int = 1):
    """Тепловые искажения вокруг равновесия: то, что модель увидит в динамике."""
    out = []
    for _ in range(count):
        copy = state.copy()
        copy.positions = copy.positions + rng.normal(scale=scale, size=copy.positions.shape)
        out.append(copy)
    return out


def solvated(state, rng: np.random.Generator, n_water: int, box_pad: float = 4.0,
             min_distance: float = 1.6):
    """
    Молекула в воде: ячейка вокруг молекулы плюс молекулы воды на свободных местах.

    Вода ставится с проверкой минимального расстояния, а не «куда попало»: без этой проверки в
    набор попадают кадры с наложением атомов, и один такой кадр весит в квадратичной невязке как
    миллион нормальных -- это уже измерено на воде (силы до 1.1e7 эВ/Å, невязка 2.9e15).
    """
    from train.sampling import WATER, _rotation

    span = state.positions.max(axis=0) - state.positions.min(axis=0)
    box = float(max(span) + 2 * box_pad)
    centre = state.positions.mean(axis=0)
    positions = [state.positions - centre + box / 2.0]
    symbols = list(state.symbols)

    placed, attempts = 0, 0
    while placed < n_water and attempts < n_water * 200:
        attempts += 1
        origin = rng.uniform(1.0, box - 1.0, size=3)
        mol = (WATER - WATER[0]) @ _rotation(rng).T + origin
        existing = np.vstack(positions)
        d = mol[:, None, :] - existing[None, :, :]
        d -= box * np.round(d / box)
        if np.linalg.norm(d, axis=2).min() < min_distance:
            continue
        positions.append(mol)
        symbols += ["O", "H", "H"]
        placed += 1
    return from_symbols(symbols, np.vstack(positions), cell=np.eye(3) * box, pbc=(True, True, True))


def build_organic_states(teacher, rng: np.random.Generator, chain_lengths=(2, 4, 8),
                         rattles_per_molecule: int = 60, solvated_per_molecule: int = 40,
                         waters=(8, 16)):
    """
    Полный набор органических конфигураций: кислоты разной длины, глицерин, и то же в воде.

    Длины цепей взяты не случайно: C2 (уксусная) и C4 -- дешёвые и покрывают карбоксильную
    химию, C8 (октановая) -- первая, у которой в литературе есть измеренная критическая
    концентрация мицеллообразования (около 300 мМ), то есть по ней модель можно будет проверить
    против опыта, а не только против учителя.
    """
    states = []
    relaxed = {}
    for n in chain_lengths:
        sym, pos = alkanoic_acid(n)
        st = relax_with(teacher, sym, pos)
        relaxed[f"C{n}"] = st
        states += rattled(st, rng, scale=0.07, count=rattles_per_molecule)
    sym, pos = glycerol()
    gl = relax_with(teacher, sym, pos)
    relaxed["glycerol"] = gl
    states += rattled(gl, rng, scale=0.07, count=rattles_per_molecule)

    for name, st in relaxed.items():
        for _ in range(solvated_per_molecule):
            n_water = int(rng.choice(waters))
            states.append(solvated(st, rng, n_water))
    return states, relaxed
