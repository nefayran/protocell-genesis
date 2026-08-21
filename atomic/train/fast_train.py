"""
Обучение пакетами на карте: кадры одного размера считаются одним тензором.

Замер, из-за которого это появилось: в облаке один вариант обучения занимал от 1957 до 4369
секунд на 465 кадрах. Причина не в объёме арифметики, а в том, что цикл шёл ПО КАДРАМ на
питоне, и на каждый кадр приходился отдельный вызов сети и отдельное автодифференцирование --
карта при этом простаивала так же, как при поштучной разметке.

Здесь кадры группируются по числу атомов (тензор обязан быть прямоугольным) и внутри группы
считаются одним пакетом: сеть применяется к (B, N, D) сразу, силы собираются одним einsum.
Математика та же, что в train/nn.py -- это проверяется сравнением предсказаний обеих версий на
одних и тех же весах (validate/test_fast_train.py).

Память -- главное ограничение пакета: тензор производных имеет размер B×N×D×N×3, то есть растёт
как КВАДРАТ числа атомов. Поэтому размер пакета выводится из числа атомов и заданного предела
памяти, а не назначается одним числом на все размеры.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FastTrainingConfig:
    hidden: tuple[int, ...] = (96, 96)
    force_weight: float = 30.0
    epochs: int = 2000
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    seed: int = 0
    memory_budget_mb: float = 512.0
    dtype: str = "float32"
    lr_final_fraction: float = 0.05   # к концу обучения шаг падает до этой доли начального


def _batch_size_for(n_atoms: int, d_len: int, budget_mb: float, bytes_per_number: int) -> int:
    """Сколько кадров такого размера влезает в заданный предел памяти."""
    per_frame = n_atoms * d_len * n_atoms * 3 * bytes_per_number
    return max(1, int(budget_mb * 1e6 / max(per_frame, 1)))


def group_by_size(frames, precomputed):
    """Группы кадров одинакового размера: {n_atoms: (индексы, G, dG)}."""
    groups: dict[int, list[int]] = {}
    for k, fr in enumerate(frames):
        groups.setdefault(len(fr["numbers"]), []).append(k)
    out = {}
    for n_atoms, idxs in groups.items():
        g = np.stack([precomputed[i][0] for i in idxs])
        dg = np.stack([precomputed[i][1] for i in idxs])
        out[n_atoms] = (idxs, g, dg)
    return out


def train_fast(frames, spec, species, cfg: FastTrainingConfig, precomputed, device: str = "cpu", verbose=True):
    """
    Обучение по энергиям и силам, пакетами, на заданном устройстве.

    Затухание шага обучения по косинусу -- не украшение: при постоянном шаге невязка на
    последних эпохах колебалась (замер: 2190 -> 3824 между эпохами 222 и 296), то есть
    оптимизатор ходил вокруг минимума, не садясь в него.
    """
    import torch

    from .descriptors import descriptor_length
    from .nn import SpeciesNetwork, fit_atomic_baseline, baseline_energy, standardise

    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    bytes_per = 4 if cfg.dtype == "float32" else 8
    d_len = descriptor_length(spec, species)

    mean, std = standardise([g for g, _ in precomputed])
    baseline = fit_atomic_baseline(frames, species)
    net = SpeciesNetwork(d_len, species, cfg.hidden, cfg.seed)
    for z in net.nets:
        net.nets[z] = net.nets[z].to(device=device, dtype=dtype)

    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg.epochs, eta_min=cfg.learning_rate * cfg.lr_final_fraction
    )

    mean_t = torch.tensor(mean, device=device, dtype=dtype)
    std_t = torch.tensor(std, device=device, dtype=dtype)

    # --- подготовка групп: один раз, на устройстве ------------------------------------
    prepared = []
    for n_atoms, (idxs, g, dg) in group_by_size(frames, precomputed).items():
        batch = _batch_size_for(n_atoms, d_len, cfg.memory_budget_mb, bytes_per)
        for start in range(0, len(idxs), batch):
            part = idxs[start : start + batch]
            sl = slice(start, start + len(part))
            numbers = torch.tensor(
                np.stack([np.asarray(frames[i]["numbers"], dtype=int) for i in part]), device=device
            )
            target_e = torch.tensor(
                [float(frames[i]["energy"]) - baseline_energy(baseline, frames[i]["numbers"], species)
                 for i in part],
                device=device, dtype=dtype,
            )
            target_f = torch.tensor(
                np.stack([np.asarray(frames[i]["forces"], dtype=float) for i in part]),
                device=device, dtype=dtype,
            )
            prepared.append(
                {
                    "g": (torch.tensor(g[sl], device=device, dtype=dtype) - mean_t) / std_t,
                    "dg": torch.tensor(dg[sl], device=device, dtype=dtype) / std_t[None, None, :, None, None],
                    "numbers": numbers,
                    "energy": target_e,
                    "forces": target_f,
                    "n_atoms": n_atoms,
                }
            )
    if verbose:
        print(f"    пакетов {len(prepared)}, размеры {[p['g'].shape[0] for p in prepared][:12]}", flush=True)

    # Группировка по размеру -- требование прямоугольного тензора, но НЕ способ обучения:
    # если каждый шаг оптимизатора видит кадры одного размера, шаг смещён в сторону этого
    # размера. Замер: обучение по однородным пакетам дало 548 мэВ/атом против 246 у прежней
    # версии, которая смешивала размеры в одном пакете. Поэтому градиент НАКАПЛИВАЕТСЯ по всем
    # группам, и только потом делается шаг: тензоры остаются прямоугольными, а шаг -- смешанным.
    rng = np.random.default_rng(cfg.seed)
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(prepared))
        total = 0.0
        opt.zero_grad()
        for bi in order:
            item = prepared[bi]
            g = item["g"].detach().clone().requires_grad_(True)
            b, n = g.shape[0], g.shape[1]
            flat_e = net.atom_energies(g.reshape(b * n, -1), item["numbers"].reshape(-1))
            atom_e = flat_e.reshape(b, n)
            energy = atom_e.sum(dim=1)
            (de_dg,) = torch.autograd.grad(energy.sum(), g, create_graph=True)
            forces = -torch.einsum("bid,bidka->bka", de_dg, item["dg"])
            loss = ((energy - item["energy"]) / n).pow(2).mean() + cfg.force_weight * (
                (forces - item["forces"]).pow(2).mean()
            )
            (loss / len(prepared)).backward()
            total += float(loss.item())
        opt.step()
        sched.step()
        if verbose and (epoch + 1) % max(1, cfg.epochs // 10) == 0:
            print(f"    эпоха {epoch + 1:5d}/{cfg.epochs}: невязка {total / len(prepared):.5f}, "
                  f"шаг {sched.get_last_lr()[0]:.2e}", flush=True)

    return {
        "net": net, "mean": mean, "std": std, "baseline": baseline,
        "spec": spec, "species": species, "device": device, "dtype": cfg.dtype,
    }


def evaluate_fast(model, frames, precomputed, device: str | None = None) -> dict:
    """Ошибки против учителя, пакетами. Тот же смысл величин, что у `train.nn.evaluate`."""
    import torch

    from .nn import baseline_energy

    device = device or model.get("device", "cpu")
    dtype = torch.float32 if model.get("dtype", "float32") == "float32" else torch.float64
    mean_t = torch.tensor(model["mean"], device=device, dtype=dtype)
    std_t = torch.tensor(model["std"], device=device, dtype=dtype)

    de, df = [], []
    for n_atoms, (idxs, g, dg) in group_by_size(frames, precomputed).items():
        gt = ((torch.tensor(g, device=device, dtype=dtype) - mean_t) / std_t).requires_grad_(True)
        dgt = torch.tensor(dg, device=device, dtype=dtype) / std_t[None, None, :, None, None]
        numbers = torch.tensor(
            np.stack([np.asarray(frames[i]["numbers"], dtype=int) for i in idxs]), device=device
        )
        b, n = gt.shape[0], gt.shape[1]
        atom_e = model["net"].atom_energies(gt.reshape(b * n, -1), numbers.reshape(-1)).reshape(b, n)
        energy = atom_e.sum(dim=1)
        (de_dg,) = torch.autograd.grad(energy.sum(), gt)
        forces = -torch.einsum("bid,bidka->bka", de_dg, dgt)
        for row, i in enumerate(idxs):
            total = float(energy[row].item()) + baseline_energy(
                model["baseline"], frames[i]["numbers"], model["species"]
            )
            de.append((total - float(frames[i]["energy"])) / n)
            df.append((forces[row].detach().cpu().numpy() - np.asarray(frames[i]["forces"])).reshape(-1))
    de = np.asarray(de)
    df = np.concatenate(df)
    return {
        "energy_mae_mev_per_atom": float(np.abs(de).mean() * 1000),
        "force_mae_mev_per_a": float(np.abs(df).mean() * 1000),
        "force_rmse_mev_per_a": float(np.sqrt((df**2).mean()) * 1000),
        "n_frames": len(frames),
    }
