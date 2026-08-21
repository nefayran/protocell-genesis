"""
Маленькая сеть поверх тех же дескрипторов -- потенциал Белера-Паринелло.

Почему сеть, а не линейная модель: линейная подгонка по этим дескрипторам измерена и даёт
350-900 мэВ/атом при цели 5 (см. отчёт). При этом синтетическая проверка показала, что матрица
задачи верна и линейно представимые данные восстанавливаются с точностью 4e-6 мэВ/атом --
значит дело не в обвязке, а в том, что энергия воды не линейна в этом базисе. Нелинейный
считыватель поверх ТЕХ ЖЕ дескрипторов -- стандартное решение (Behler & Parrinello 2007), и
оно сохраняет главное свойство затеи: стоимость на атом остаётся единицами тысяч операций,
потому что сеть крошечная и одинаковая для всех атомов одного сорта.

Устройство: своя сеть на каждый сорт атома, два скрытых слоя, гладкая нелинейность. Гладкость
обязательна: сила -- это производная энергии, поэтому изломы нелинейности (как у ReLU) дают
разрывную силу и портят сохранение энергии. Взят softplus.

Обучение идёт по энергиям И силам. Силы получаются автоматическим дифференцированием ПО
ДЕСКРИПТОРАМ с последующим умножением на аналитические производные дескрипторов -- то есть
цепное правило разрывается на две части, и тяжёлая геометрическая часть считается один раз
заранее, а не на каждом шаге обучения.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingConfig:
    hidden: tuple[int, ...] = (32, 32)
    force_weight: float = 10.0     # силы важнее: их 3N против 1 на кадр
    epochs: int = 400
    batch_frames: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 1e-6
    seed: int = 0


def _torch():
    import torch
    return torch


class SpeciesNetwork:
    """Набор маленьких сетей: по одной на сорт атома."""

    def __init__(self, n_features: int, species: tuple[int, ...], hidden=(32, 32), seed: int = 0):
        torch = _torch()
        torch.manual_seed(seed)
        self.species = species
        self.n_features = n_features
        self.nets = {}
        for z in species:
            layers = []
            dim = n_features
            for h in hidden:
                layers += [torch.nn.Linear(dim, h), torch.nn.Softplus()]
                dim = h
            layers += [torch.nn.Linear(dim, 1)]
            self.nets[z] = torch.nn.Sequential(*layers).double()

    def parameters(self):
        for net in self.nets.values():
            yield from net.parameters()

    def atom_energies(self, features, numbers):
        """Энергии атомов: (N,) из (N, D) и сортов."""
        torch = _torch()
        out = torch.zeros(len(numbers), dtype=features.dtype)
        for z, net in self.nets.items():
            sel = numbers == z
            if bool(sel.any()):
                out[sel] = net(features[sel]).squeeze(-1)
        return out

    def state_dict(self):
        return {int(z): {k: v.detach().numpy() for k, v in net.state_dict().items()} for z, net in self.nets.items()}

    def load_state(self, state: dict) -> None:
        torch = _torch()
        for z, net in self.nets.items():
            net.load_state_dict({k: torch.tensor(v) for k, v in state[int(z)].items()})


def standardise(frames_features: list[np.ndarray]):
    """
    Стандартизация дескрипторов по обучающему набору: среднее ноль, разброс единица.

    Это не косметика. Дескрипторы различаются по величине на порядки (радиальные суммы против
    угловых), и без приведения к одному масштабу обучение идёт по самому крупному признаку, а
    остальные фактически не участвуют. Множители сохраняются вместе с моделью и применяются в
    расчёте -- иначе перенос в WGSL даст другие числа.
    """
    stacked = np.vstack(frames_features)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def fit_atomic_baseline(frames, species: tuple[int, ...]) -> np.ndarray:
    """
    Атомные отсчёты по составу: E ≈ sum_Z n_Z · e0(Z), решается наименьшими квадратами.

    Без этого шага обучение проваливается, и это ИЗМЕРЕНО, а не предположено: у учителя в
    полной энергии сидит около -700 эВ на атом атомного отсчёта, и сеть со случайной
    инициализацией сначала тратит всю ёмкость на воспроизведение этой постоянной. Замер:
    сеть без вычитания основы дала 11570 мэВ/атом против 350 у линейной модели со свободным
    членом. После вычитания сеть учит только взаимодействие -- величину порядка единиц эВ.
    """
    a = np.zeros((len(frames), len(species)))
    b = np.zeros(len(frames))
    for k, fr in enumerate(frames):
        nums = np.asarray(fr["numbers"], dtype=int)
        for s, z in enumerate(species):
            a[k, s] = int((nums == z).sum())
        b[k] = float(fr["energy"])
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    return sol


def baseline_energy(baseline: np.ndarray, numbers, species: tuple[int, ...]) -> float:
    nums = np.asarray(numbers, dtype=int)
    return float(sum(baseline[s] * int((nums == z).sum()) for s, z in enumerate(species)))


def train(frames, spec, species, cfg: TrainingConfig, precomputed=None, verbose=True):
    """
    Обучение по энергиям и силам. `precomputed` -- список (G, dG) на кадр, чтобы не считать
    дескрипторы заново на каждой эпохе (они не зависят от параметров сети).
    """
    torch = _torch()
    from .descriptors import compute_descriptors, descriptor_length

    if precomputed is None:
        precomputed = [
            compute_descriptors(
                np.asarray(f["positions"]), np.asarray(f["numbers"], dtype=int), spec, species,
                cell=f.get("cell"),
            )
            for f in frames
        ]

    mean, std = standardise([g for g, _ in precomputed])
    baseline = fit_atomic_baseline(frames, species)
    d = descriptor_length(spec, species)
    net = SpeciesNetwork(d, species, cfg.hidden, cfg.seed)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    # заранее готовим тензоры: (G-mean)/std и dG/std
    prepared = []
    for (g, dg), fr in zip(precomputed, frames):
        gt = torch.tensor((g - mean) / std, requires_grad=True)
        dgt = torch.tensor(dg / std[None, :, None, None])
        prepared.append(
            {
                "g": gt,
                "dg": dgt,
                "numbers": torch.tensor(np.asarray(fr["numbers"], dtype=int)),
                # цель сети -- ОСТАТОК после вычитания атомной основы, см. fit_atomic_baseline
                "energy": torch.tensor(
                    float(fr["energy"]) - baseline_energy(baseline, fr["numbers"], species),
                    dtype=torch.float64,
                ),
                "forces": torch.tensor(np.asarray(fr["forces"], dtype=float)),
                "n": len(fr["numbers"]),
            }
        )

    rng = np.random.default_rng(cfg.seed)
    history = []
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(prepared))
        for start in range(0, len(order), cfg.batch_frames):
            batch = [prepared[i] for i in order[start : start + cfg.batch_frames]]
            opt.zero_grad()
            loss = torch.zeros((), dtype=torch.float64)
            for item in batch:
                g = item["g"].detach().clone().requires_grad_(True)
                atom_e = net.atom_energies(g, item["numbers"])
                energy = atom_e.sum()
                # dE/dG -- производная по дескрипторам, а геометрия уже в dG
                (de_dg,) = torch.autograd.grad(energy, g, create_graph=True)
                forces = -torch.einsum("id,idka->ka", de_dg, item["dg"])
                loss = loss + ((energy - item["energy"]) / item["n"]) ** 2
                loss = loss + cfg.force_weight * ((forces - item["forces"]) ** 2).mean()
            (loss / len(batch)).backward()
            opt.step()
        if verbose and (epoch + 1) % max(1, cfg.epochs // 8) == 0:
            history.append(float(loss.item()) / max(1, len(batch)))
            print(f"    эпоха {epoch + 1:4d}/{cfg.epochs}: невязка {history[-1]:.5f}", flush=True)

    return {
        "net": net,
        "mean": mean,
        "std": std,
        "spec": spec,
        "species": species,
        "baseline": baseline,
    }


def predict(model: dict, positions, numbers, cell=None):
    """Энергия и силы обученной моделью."""
    torch = _torch()
    from .descriptors import compute_descriptors

    g, dg = compute_descriptors(
        np.asarray(positions), np.asarray(numbers, dtype=int), model["spec"], model["species"], cell=cell
    )
    gt = torch.tensor((g - model["mean"]) / model["std"], requires_grad=True)
    dgt = torch.tensor(dg / model["std"][None, :, None, None])
    atom_e = model["net"].atom_energies(gt, torch.tensor(np.asarray(numbers, dtype=int)))
    energy = atom_e.sum()
    (de_dg,) = torch.autograd.grad(energy, gt)
    forces = -torch.einsum("id,idka->ka", de_dg, dgt)
    total = float(energy.item()) + baseline_energy(model["baseline"], numbers, model["species"])
    return total, forces.detach().numpy()


def evaluate(model: dict, frames) -> dict:
    de, df = [], []
    for fr in frames:
        e, f = predict(model, fr["positions"], fr["numbers"], fr.get("cell"))
        n = len(fr["numbers"])
        de.append((e - float(fr["energy"])) / n)
        df.append((f - np.asarray(fr["forces"], dtype=float)).reshape(-1))
    de = np.asarray(de)
    df = np.concatenate(df)
    return {
        "energy_mae_mev_per_atom": float(np.abs(de).mean() * 1000),
        "force_mae_mev_per_a": float(np.abs(df).mean() * 1000),
        "force_rmse_mev_per_a": float(np.sqrt((df**2).mean()) * 1000),
        "n_frames": len(frames),
    }
