"""
Batched GPU training: frames of the same size are computed as a single tensor.

The measurement that led to this: in the cloud, one training run took between 1957 and
4369 seconds on 465 frames. The cause was not the amount of arithmetic but the fact that
the loop ran PER FRAME in Python, with a separate network call and separate autodiff for
every frame; the GPU was idle just as much as with per-item labeling.

Here frames are grouped by number of atoms (the tensor must be rectangular) and computed
as a single batch within a group: the network is applied to (B, N, D) at once, and forces
are gathered with a single einsum. The math is the same as in train/nn.py; this is verified
by comparing the predictions of both versions on the same weights
(validate/test_fast_train.py).

Memory is the main constraint on the batch: the derivative tensor has size B×N×D×N×3, i.e.
it grows as the SQUARE of the number of atoms. So the batch size is derived from the number
of atoms and a given memory budget, rather than fixed to one number for all sizes.
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
    lr_final_fraction: float = 0.05   # by the end of training, the step drops to this fraction of the initial one


def _batch_size_for(n_atoms: int, d_len: int, budget_mb: float, bytes_per_number: int) -> int:
    """How many frames of this size fit within a given memory budget."""
    per_frame = n_atoms * d_len * n_atoms * 3 * bytes_per_number
    return max(1, int(budget_mb * 1e6 / max(per_frame, 1)))


def group_by_size(frames, precomputed):
    """Groups of frames of the same size: {n_atoms: (indices, G, dG)}."""
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
    Training on energies and forces, in batches, on a given device.

    Cosine decay of the learning rate is not decoration: with a constant step, the loss
    oscillated in the final epochs (measured: 2190 -> 3824 between epochs 222 and 296),
    i.e. the optimizer was circling the minimum without settling into it.
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

    # --- prepare groups: once, on the device -------------------------------------------
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
        print(f"    batches {len(prepared)}, sizes {[p['g'].shape[0] for p in prepared][:12]}", flush=True)

    # Grouping by size is a requirement of the rectangular tensor, but NOT a training
    # method: if every optimizer step only sees frames of one size, the step is biased
    # toward that size. Measured: training on homogeneous batches gave 548 meV/atom against
    # 246 for the previous version, which mixed sizes within one batch. So the gradient is
    # ACCUMULATED across all groups, and only then is a step taken: the tensors stay
    # rectangular, while the step is mixed.
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
            print(f"    epoch {epoch + 1:5d}/{cfg.epochs}: loss {total / len(prepared):.5f}, "
                  f"lr {sched.get_last_lr()[0]:.2e}", flush=True)

    return {
        "net": net, "mean": mean, "std": std, "baseline": baseline,
        "spec": spec, "species": species, "device": device, "dtype": cfg.dtype,
    }


def evaluate_fast(model, frames, precomputed, device: str | None = None) -> dict:
    """Errors against the teacher, in batches. Same meaning of quantities as `train.nn.evaluate`."""
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
