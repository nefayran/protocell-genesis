"""
Дескрипторы окружения атома и их производные, написанные руками.

Выбор вида модели -- главное решение всей затеи, и он сделан из соображений СКОРОСТИ НА GPU,
а не из моды. Взяты симметрийные функции Белера-Паринелло (Behler & Parrinello, PRL 98 (2007)
146401; Behler, JCP 134 (2011) 074106): радиальные G2 и угловые G4. Причины:

1. Они инвариантны к переносу, вращению и перестановке одинаковых атомов -- то есть
   удовлетворяют тем же симметриям, что энергия, и модель не тратит ёмкость на их изучение.
2. Их производные по координатам выражаются в замкнутом виде через те же величины, что и
   сами дескрипторы. Значит силы считаются БЕЗ автоматического дифференцирования -- а именно
   автодифференцирование делает нейросетевые потенциалы дорогими в цикле динамики.
3. Модель по ним ЛИНЕЙНА, поэтому обучение -- это одна задача наименьших квадратов с точным
   решением, без итераций, без скорости обучения и без переобучения на шуме оптимизатора.
   Если линейной ёмкости не хватит, это будет видно как ошибка на отложенной выборке, и тогда
   поверх тех же дескрипторов встанет маленькая сеть -- дескрипторы при этом не меняются.

Стоимость на атом: (число соседей) × (число радиальных функций) для G2 плюс (число пар
соседей) × (число угловых функций) для G4. Это единицы тысяч операций против миллионов у
модели с обменом сообщениями -- и вся арифметика поэлементная, то есть идеальна для GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DescriptorSpec:
    """
    Набор дескрипторов. Параметры -- не подгонка под ответ, а покрытие пространства:
    центры радиальных гауссиан размещаются равномерно от контактного расстояния до радиуса
    обрезания, ширина берётся равной шагу сетки центров (чтобы соседние функции перекрывались
    и набор не имел слепых зон), а угловые функции берут стандартный набор кратностей.

    `cutoff` -- физическая величина: за ним взаимодействие считается нулевым. 5 Å выбран как
    расстояние, на котором водородная связь и первая координационная сфера воды уже внутри,
    а стоимость ещё умеренная (число соседей около 40 при плотности жидкой воды).
    """

    cutoff: float = 5.0
    n_radial: int = 8
    zetas: tuple[float, ...] = (1.0, 2.0, 4.0)
    lambdas: tuple[float, ...] = (1.0, -1.0)
    eta_angular: float = 0.08
    r_min: float = 0.8
    # НАБОР ширин, а не одна: узкие функции разрешают положение первого пика, широкие видят
    # вторую координационную сферу. С одной шириной (первая версия) отложенная ошибка сил
    # держалась на 1327 мэВ/Å -- набор ширин это прямое расширение ёмкости базиса без
    # изменения его вида и без роста стоимости на порядок.
    eta_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    eta_angular_scales: tuple[float, ...] = (0.5, 2.0)

    @property
    def mu(self) -> np.ndarray:
        """Центры радиальных гауссиан."""
        return np.linspace(self.r_min, self.cutoff * 0.95, self.n_radial)

    @property
    def eta_radial(self) -> float:
        """Ширина из шага сетки центров: перекрытие соседних функций около половины высоты."""
        step = (self.cutoff * 0.95 - self.r_min) / max(1, self.n_radial - 1)
        return 1.0 / (2.0 * step**2)

    @property
    def n_angular(self) -> int:
        return len(self.zetas) * len(self.lambdas) * len(self.eta_angular_scales)

    @property
    def n_radial_total(self) -> int:
        return self.n_radial * len(self.eta_scales)


def cutoff_function(r: np.ndarray, rc: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Косинусная функция обрезания f = 0.5(cos(pi r/rc) + 1) и её производная.

    Обрезание обязано зануляться ВМЕСТЕ С ПРОИЗВОДНОЙ на радиусе: иначе сила скачком меняется,
    когда сосед пересекает границу списка, и энергия в NVE не сохраняется. У косинусной формы
    и значение, и производная равны нулю в r = rc -- это её причина существования.
    """
    inside = r < rc
    f = np.where(inside, 0.5 * (np.cos(np.pi * r / rc) + 1.0), 0.0)
    df = np.where(inside, -0.5 * np.pi / rc * np.sin(np.pi * r / rc), 0.0)
    return f, df


def species_pair_index(z_a: int, z_b: int, species: tuple[int, ...]) -> int:
    """Индекс НЕупорядоченной пары сортов: (H,O) и (O,H) -- один канал."""
    ia, ib = species.index(z_a), species.index(z_b)
    lo, hi = min(ia, ib), max(ia, ib)
    n = len(species)
    return lo * n - lo * (lo - 1) // 2 + (hi - lo)


def n_pair_channels(species: tuple[int, ...]) -> int:
    n = len(species)
    return n * (n + 1) // 2


def descriptor_length(spec: DescriptorSpec, species: tuple[int, ...]) -> int:
    """Полная длина вектора дескрипторов одного атома."""
    n_pairs = n_pair_channels(species)
    return spec.n_radial_total * len(species) + spec.n_angular * n_pairs


def compute_descriptors(
    positions: np.ndarray,
    numbers: np.ndarray,
    spec: DescriptorSpec,
    species: tuple[int, ...],
    cell: np.ndarray | None = None,
    with_gradients: bool = True,
):
    """
    Дескрипторы всех атомов и, если нужно, их производные по координатам.

    Возвращает (G, dG) где G имеет форму (N, D), а dG -- (N, D, N, 3): производная
    дескриптора d атома i по координате атома k. Форма расточительна по памяти и годится
    только для обучения на малых системах; в динамике силы собираются на лету, без хранения
    полного тензора (см. atomic/train/model.py и порт в WGSL).

    Периодичность учитывается минимальным образом по всем осям, если задана ячейка.
    """
    n = len(numbers)
    d_len = descriptor_length(spec, species)
    g = np.zeros((n, d_len))
    dg = np.zeros((n, d_len, n, 3)) if with_gradients else None

    mu = spec.mu
    eta_r = spec.eta_radial
    n_sp = len(species)
    n_rad_block = spec.n_radial_total * n_sp

    for i in range(n):
        # --- список соседей внутри радиуса обрезания ---------------------------------
        d = positions - positions[i]
        if cell is not None:
            frac = np.linalg.solve(cell.T, d.T).T
            frac -= np.round(frac)
            d = (cell.T @ frac.T).T
        r = np.linalg.norm(d, axis=1)
        mask = (r < spec.cutoff) & (r > 1e-8)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        rij = r[idx]
        dij = d[idx]
        unit = dij / rij[:, None]
        fc, dfc = cutoff_function(rij, spec.cutoff)

        # --- G2: радиальные -----------------------------------------------------------
        # g2[n] = sum_j exp(-eta (r_ij - mu_n)^2) fc(r_ij), отдельным каналом на сорт соседа
        diff = rij[:, None] - mu[None, :]
        contrib_parts, dcontrib_parts = [], []
        for scale in spec.eta_scales:
            eta = eta_r * scale
            gauss = np.exp(-eta * diff**2)
            contrib_parts.append(gauss * fc[:, None])
            dcontrib_parts.append(gauss * (-2.0 * eta * diff * fc[:, None] + dfc[:, None]))
        contrib = np.concatenate(contrib_parts, axis=1)          # (n_nb, n_radial_total)
        dcontrib_dr = np.concatenate(dcontrib_parts, axis=1)

        for k, j in enumerate(idx):
            sp = species.index(int(numbers[j]))
            block = slice(sp * spec.n_radial_total, (sp + 1) * spec.n_radial_total)
            g[i, block] += contrib[k]
            if dg is not None:
                # производная по r_ij, разложенная по осям через единичный вектор
                grad = dcontrib_dr[k][:, None] * unit[k][None, :]
                dg[i, block, j, :] += grad
                dg[i, block, i, :] -= grad

        # --- G4: угловые ---------------------------------------------------------------
        # g4 = sum_{j<k} (1 + lambda cos(theta_ijk))^zeta * exp(-eta(r_ij^2+r_ik^2+r_jk^2)) * fc fc fc
        if idx.size >= 2:
            for a in range(idx.size - 1):
                for b in range(a + 1, idx.size):
                    ja, jb = idx[a], idx[b]
                    ra, rb = rij[a], rij[b]
                    va, vb = dij[a], dij[b]
                    vjk = vb - va
                    rjk = float(np.linalg.norm(vjk))
                    if rjk >= spec.cutoff or rjk < 1e-8:
                        continue
                    fc_jk, dfc_jk = cutoff_function(np.array([rjk]), spec.cutoff)
                    fc_jk, dfc_jk = float(fc_jk[0]), float(dfc_jk[0])
                    cos_t = float(va @ vb) / (ra * rb)
                    fcc = fc[a] * fc[b] * fc_jk
                    r2sum = ra**2 + rb**2 + rjk**2
                    ch = species_pair_index(int(numbers[ja]), int(numbers[jb]), species)
                    base = n_rad_block + ch * spec.n_angular

                    slot = 0
                    for eta_scale in spec.eta_angular_scales:
                      eta_a = spec.eta_angular * eta_scale
                      expo = np.exp(-eta_a * r2sum)
                      for zeta in spec.zetas:
                        for lam in spec.lambdas:
                            angular = (1.0 + lam * cos_t) ** zeta
                            g[i, base + slot] += angular * expo * fcc
                            if dg is not None:
                                # производные по трём расстояниям и по косинусу угла
                                dang_dcos = (
                                    zeta * lam * (1.0 + lam * cos_t) ** (zeta - 1.0)
                                    if abs(1.0 + lam * cos_t) > 1e-12
                                    else 0.0
                                )
                                # d cos / d va, d cos / d vb
                                dcos_dva = vb / (ra * rb) - cos_t * va / ra**2
                                dcos_dvb = va / (ra * rb) - cos_t * vb / rb**2
                                # d expo / d va, d vb, d vjk
                                dexpo_dva = -2.0 * eta_a * va * expo
                                dexpo_dvb = -2.0 * eta_a * vb * expo
                                dexpo_dvjk = -2.0 * eta_a * vjk * expo
                                # d fcc
                                dfcc_dva = dfc[a] * (va / ra) * fc[b] * fc_jk
                                dfcc_dvb = dfc[b] * (vb / rb) * fc[a] * fc_jk
                                dfcc_dvjk = dfc_jk * (vjk / rjk) * fc[a] * fc[b]

                                # полная производная по va и vb (vjk = vb - va)
                                dva = (
                                    dang_dcos * dcos_dva * expo * fcc
                                    + angular * dexpo_dva * fcc
                                    + angular * expo * dfcc_dva
                                    - angular * dexpo_dvjk * fcc
                                    - angular * expo * dfcc_dvjk
                                )
                                dvb = (
                                    dang_dcos * dcos_dvb * expo * fcc
                                    + angular * dexpo_dvb * fcc
                                    + angular * expo * dfcc_dvb
                                    + angular * dexpo_dvjk * fcc
                                    + angular * expo * dfcc_dvjk
                                )
                                dg[i, base + slot, ja, :] += dva
                                dg[i, base + slot, jb, :] += dvb
                                dg[i, base + slot, i, :] -= dva + dvb
                            slot += 1

    return (g, dg) if with_gradients else g
