"""
Те же дескрипторы, посчитанные векторно: угловая часть без цикла по парам соседей.

Зачем понадобилось. В прямой реализации (train/descriptors.py) угловые функции считаются
двойным циклом по парам соседей на питоне. Замер: 173 секунды на 336 кадров, то есть при
пяти тысячах кадров дескрипторы стали бы стоить сорок минут -- дороже, чем разметка их
дорогим учителем. Причина не в объёме арифметики, а в том, что при сорока соседях пар около
780 на атом, и каждая проходит через интерпретатор.

Здесь ровно та же математика, но все пары одного атома обрабатываются одним массивом. Формулы
не меняются -- меняется только способ их выполнения, и это проверяется побайтовым сравнением с
прямой реализацией (validate/test_descriptors_fast.py): расхождение обязано быть на уровне
машинной точности, а не «примерно совпадать».
"""
from __future__ import annotations

import numpy as np

from .descriptors import (
    DescriptorSpec,
    cutoff_function,
    descriptor_length,
    n_pair_channels,
    species_pair_index,
)


def compute_descriptors_fast(
    positions: np.ndarray,
    numbers: np.ndarray,
    spec: DescriptorSpec,
    species: tuple[int, ...],
    cell: np.ndarray | None = None,
    with_gradients: bool = True,
):
    """Векторная версия `compute_descriptors` с тем же смыслом возвращаемых величин."""
    n = len(numbers)
    d_len = descriptor_length(spec, species)
    g = np.zeros((n, d_len))
    dg = np.zeros((n, d_len, n, 3)) if with_gradients else None

    mu = spec.mu
    eta_r = spec.eta_radial
    n_sp = len(species)
    n_rad_block = spec.n_radial_total * n_sp
    sp_of = np.array([species.index(int(z)) for z in numbers])
    n_zeta, n_lam, n_eta_a = len(spec.zetas), len(spec.lambdas), len(spec.eta_angular_scales)

    for i in range(n):
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
        rij, dij = r[idx], d[idx]
        unit = dij / rij[:, None]
        fc, dfc = cutoff_function(rij, spec.cutoff)

        # --- радиальная часть: как в прямой версии, она уже векторная -------------------
        diff = rij[:, None] - mu[None, :]
        parts, dparts = [], []
        for scale in spec.eta_scales:
            eta = eta_r * scale
            gauss = np.exp(-eta * diff**2)
            parts.append(gauss * fc[:, None])
            dparts.append(gauss * (-2.0 * eta * diff * fc[:, None] + dfc[:, None]))
        contrib = np.concatenate(parts, axis=1)
        dcontrib = np.concatenate(dparts, axis=1)

        for sp in range(n_sp):
            sel = sp_of[idx] == sp
            if not sel.any():
                continue
            block = slice(sp * spec.n_radial_total, (sp + 1) * spec.n_radial_total)
            g[i, block] += contrib[sel].sum(axis=0)
        if dg is not None:
            for k, j in enumerate(idx):
                sp = sp_of[j]
                block = slice(sp * spec.n_radial_total, (sp + 1) * spec.n_radial_total)
                grad = dcontrib[k][:, None] * unit[k][None, :]
                dg[i, block, j, :] += grad
                dg[i, block, i, :] -= grad

        if idx.size < 2:
            continue

        # --- угловая часть: ВСЕ пары соседей одним массивом -----------------------------
        ia, ib = np.triu_indices(idx.size, k=1)
        va, vb = dij[ia], dij[ib]
        ra, rb = rij[ia], rij[ib]
        vjk = vb - va
        rjk = np.linalg.norm(vjk, axis=1)
        keep = (rjk < spec.cutoff) & (rjk > 1e-8)
        if not keep.any():
            continue
        ia, ib, va, vb, ra, rb, vjk, rjk = (
            ia[keep], ib[keep], va[keep], vb[keep], ra[keep], rb[keep], vjk[keep], rjk[keep]
        )
        fc_jk, dfc_jk = cutoff_function(rjk, spec.cutoff)
        cos_t = np.einsum("pa,pa->p", va, vb) / (ra * rb)
        fcc = fc[ia] * fc[ib] * fc_jk
        r2sum = ra**2 + rb**2 + rjk**2
        channels = np.array(
            [species_pair_index(int(numbers[idx[a]]), int(numbers[idx[b]]), species) for a, b in zip(ia, ib)]
        )

        # производные геометрических множителей -- те же выражения, что в прямой версии
        dcos_dva = vb / (ra * rb)[:, None] - cos_t[:, None] * va / (ra**2)[:, None]
        dcos_dvb = va / (ra * rb)[:, None] - cos_t[:, None] * vb / (rb**2)[:, None]
        dfcc_dva = (dfc[ia] * fc[ib] * fc_jk)[:, None] * (va / ra[:, None])
        dfcc_dvb = (dfc[ib] * fc[ia] * fc_jk)[:, None] * (vb / rb[:, None])
        dfcc_dvjk = (dfc_jk * fc[ia] * fc[ib])[:, None] * (vjk / rjk[:, None])

        slot = 0
        for eta_scale in spec.eta_angular_scales:
            eta_a = spec.eta_angular * eta_scale
            expo = np.exp(-eta_a * r2sum)
            dexpo_dva = -2.0 * eta_a * va * expo[:, None]
            dexpo_dvb = -2.0 * eta_a * vb * expo[:, None]
            dexpo_dvjk = -2.0 * eta_a * vjk * expo[:, None]
            for zeta in spec.zetas:
                for lam in spec.lambdas:
                    base = 1.0 + lam * cos_t
                    angular = base**zeta
                    value = angular * expo * fcc
                    col = n_rad_block + channels * spec.n_angular + slot
                    np.add.at(g[i], col, value)
                    if dg is not None:
                        with np.errstate(divide="ignore", invalid="ignore"):
                            dang = np.where(
                                np.abs(base) > 1e-12, zeta * lam * base ** (zeta - 1.0), 0.0
                            )
                        common = (angular * expo)[:, None]
                        dva = (
                            (dang * expo * fcc)[:, None] * dcos_dva
                            + (angular * fcc)[:, None] * dexpo_dva
                            + common * dfcc_dva
                            - (angular * fcc)[:, None] * dexpo_dvjk
                            - common * dfcc_dvjk
                        )
                        dvb = (
                            (dang * expo * fcc)[:, None] * dcos_dvb
                            + (angular * fcc)[:, None] * dexpo_dvb
                            + common * dfcc_dvb
                            + (angular * fcc)[:, None] * dexpo_dvjk
                            + common * dfcc_dvjk
                        )
                        ja, jb = idx[ia], idx[ib]
                        # np.add.at по трём индексам сразу: пары могут делить атомы
                        np.add.at(dg[i], (col, ja), dva)
                        np.add.at(dg[i], (col, jb), dvb)
                        np.add.at(dg[i], (col, np.full_like(ja, i)), -(dva + dvb))
                    slot += 1

    return (g, dg) if with_gradients else g
