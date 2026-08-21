"""
Линейная модель энергии по дескрипторам и её обучение методом наименьших квадратов.

Вид модели:
    E = sum_i [ e0(Z_i) + w(Z_i) . G_i ]
то есть у каждого сорта атома свой вектор коэффициентов и своя постоянная. Постоянная нужна
обязательно: учитель (MACE) считает энергию относительно своих атомных отсчётов, и без
свободного члена подгонка тратила бы ёмкость на воспроизведение этих отсчётов вместо физики.

Обучение идёт ПО СИЛАМ И ЭНЕРГИЯМ одновременно (force matching). Причина не в аккуратности, а
в количестве данных: одна конфигурация из N атомов даёт 1 уравнение по энергии и 3N по силам,
то есть силы дают в сотни раз больше связей на ту же стоимость расчёта учителем. Вес сил
относительно энергий -- единственный свободный параметр обучения, и он выбирается по ошибке на
ОТЛОЖЕННОЙ выборке, а не по ошибке на обучающей.

Решение находится точно (наименьшие квадраты с гребневой регуляризацией), без итераций и без
скорости обучения. Поэтому «переобучение» здесь проверяется одним числом: разностью ошибок на
обучающей и отложенной выборках.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .descriptors import DescriptorSpec, compute_descriptors, descriptor_length


@dataclass
class LinearPotentialModel:
    """Обученная модель: коэффициенты по сортам плюс постоянные."""

    spec: DescriptorSpec
    species: tuple[int, ...]
    weights: np.ndarray      # (n_species, D)
    offsets: np.ndarray      # (n_species,)

    def energy_and_forces(self, positions: np.ndarray, numbers: np.ndarray, cell=None):
        g, dg = compute_descriptors(positions, numbers, self.spec, self.species, cell=cell)
        sp_idx = np.array([self.species.index(int(z)) for z in numbers])
        w = self.weights[sp_idx]                       # (N, D)
        energy = float((g * w).sum() + self.offsets[sp_idx].sum())
        # F_k = -dE/dr_k = -sum_i w_i . dG_i/dr_k
        forces = -np.einsum("id,idka->ka", w, dg)
        return energy, forces

    def save(self, path: str) -> None:
        np.savez(
            path,
            weights=self.weights,
            offsets=self.offsets,
            species=np.array(self.species),
            cutoff=self.spec.cutoff,
            n_radial=self.spec.n_radial,
            zetas=np.array(self.spec.zetas),
            lambdas=np.array(self.spec.lambdas),
            eta_angular=self.spec.eta_angular,
            r_min=self.spec.r_min,
        )

    @staticmethod
    def load(path: str) -> "LinearPotentialModel":
        z = np.load(path)
        spec = DescriptorSpec(
            cutoff=float(z["cutoff"]),
            n_radial=int(z["n_radial"]),
            zetas=tuple(float(x) for x in z["zetas"]),
            lambdas=tuple(float(x) for x in z["lambdas"]),
            eta_angular=float(z["eta_angular"]),
            r_min=float(z["r_min"]),
        )
        return LinearPotentialModel(
            spec=spec,
            species=tuple(int(x) for x in z["species"]),
            weights=z["weights"],
            offsets=z["offsets"],
        )


def build_design_matrices(frames, spec: DescriptorSpec, species: tuple[int, ...]):
    """
    Собирает матрицы задачи наименьших квадратов.

    `frames` -- список словарей с ключами positions, numbers, energy, forces, cell.
    Возвращает (A_energy, b_energy, A_force, b_force), где число столбцов равно
    n_species*(D+1): D коэффициентов и одна постоянная на сорт.
    """
    n_sp = len(species)
    d = descriptor_length(spec, species)
    n_cols = n_sp * (d + 1)

    rows_e, rhs_e, rows_f, rhs_f = [], [], [], []
    for fr in frames:
        pos, num = np.asarray(fr["positions"]), np.asarray(fr["numbers"], dtype=int)
        g, dg = compute_descriptors(pos, num, spec, species, cell=fr.get("cell"))
        sp_idx = np.array([species.index(int(z)) for z in num])

        # --- энергия: одна строка ------------------------------------------------------
        row = np.zeros(n_cols)
        for s in range(n_sp):
            sel = sp_idx == s
            if sel.any():
                row[s * (d + 1) : s * (d + 1) + d] = g[sel].sum(axis=0)
                row[s * (d + 1) + d] = float(sel.sum())      # множитель постоянной = число атомов сорта
        rows_e.append(row)
        rhs_e.append(float(fr["energy"]))

        # --- силы: 3N строк ------------------------------------------------------------
        # F_k,a = -sum_i w(Z_i) . dG_i/dr_k,a  -- постоянные в силы не входят вовсе
        n = len(num)
        block = np.zeros((n * 3, n_cols))
        for s in range(n_sp):
            sel = sp_idx == s
            if not sel.any():
                continue
            # (D, N, 3) -> (N*3, D)
            contrib = -dg[sel].sum(axis=0)                    # (D, N, 3)
            block[:, s * (d + 1) : s * (d + 1) + d] = contrib.transpose(1, 2, 0).reshape(n * 3, d)
        rows_f.append(block)
        rhs_f.append(np.asarray(fr["forces"], dtype=float).reshape(-1))

    return (
        np.asarray(rows_e),
        np.asarray(rhs_e),
        np.vstack(rows_f) if rows_f else np.zeros((0, n_cols)),
        np.concatenate(rhs_f) if rhs_f else np.zeros(0),
    )


def fit_linear_model(
    frames,
    spec: DescriptorSpec,
    species: tuple[int, ...],
    force_weight: float = 1.0,
    ridge: float = 1e-8,
) -> LinearPotentialModel:
    """
    Точное решение задачи наименьших квадратов по энергиям и силам.

    Гребневая добавка нужна не «для устойчивости вообще», а потому что дескрипторы заведомо
    линейно зависимы (перекрывающиеся гауссианы): без неё матрица вырождена и решение зависит
    от численного шума. Значение подбирается по отложенной выборке.
    """
    a_e, b_e, a_f, b_f = build_design_matrices(frames, spec, species)
    a = np.vstack([a_e, force_weight * a_f])
    b = np.concatenate([b_e, force_weight * b_f])

    # нормальные уравнения с гребнем: (A^T A + lambda I) x = A^T b
    ata = a.T @ a
    ata[np.diag_indices_from(ata)] += ridge * np.trace(ata) / ata.shape[0]
    x = np.linalg.solve(ata, a.T @ b)

    n_sp = len(species)
    d = descriptor_length(spec, species)
    weights = np.zeros((n_sp, d))
    offsets = np.zeros(n_sp)
    for s in range(n_sp):
        weights[s] = x[s * (d + 1) : s * (d + 1) + d]
        offsets[s] = x[s * (d + 1) + d]
    return LinearPotentialModel(spec, species, weights, offsets)


def evaluate(model: LinearPotentialModel, frames) -> dict:
    """Ошибки против учителя: на атом по энергии и покомпонентно по силам."""
    de, df = [], []
    for fr in frames:
        e, f = model.energy_and_forces(
            np.asarray(fr["positions"]), np.asarray(fr["numbers"], dtype=int), fr.get("cell")
        )
        n = len(fr["numbers"])
        de.append((e - float(fr["energy"])) / n)
        df.append((f - np.asarray(fr["forces"], dtype=float)).reshape(-1))
    de = np.asarray(de)
    df = np.concatenate(df)
    return {
        "energy_mae_mev_per_atom": float(np.abs(de).mean() * 1000),
        "energy_rmse_mev_per_atom": float(np.sqrt((de**2).mean()) * 1000),
        "force_mae_mev_per_a": float(np.abs(df).mean() * 1000),
        "force_rmse_mev_per_a": float(np.sqrt((df**2).mean()) * 1000),
        "n_frames": len(frames),
    }
