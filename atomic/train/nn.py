"""
A small network on top of the same descriptors: a Behler-Parrinello potential.

Why a network and not a linear model: a linear fit on these descriptors was measured and
gives 350-900 meV/atom against a target of 5 (see the report). At the same time a synthetic
check showed that the task matrix is correct and linearly representable data is recovered
with an accuracy of 4e-6 meV/atom, so the problem is not in the plumbing but in the fact
that the energy of water is not linear in this basis. A nonlinear readout on top of the
SAME descriptors is the standard fix (Behler & Parrinello 2007), and it keeps the main
property of the scheme: the cost per atom stays at thousands of operations, because the
network is tiny and identical for all atoms of a given species.

Design: a separate network per atomic species, two hidden layers, a smooth nonlinearity.
Smoothness is mandatory: force is a derivative of energy, so kinks in the nonlinearity
(like ReLU) produce a discontinuous force and break energy conservation. Softplus is used.

Training runs on energies AND forces. Forces are obtained by automatic differentiation
WITH RESPECT TO DESCRIPTORS, followed by multiplication by the analytic derivatives of the
descriptors, i.e. the chain rule is split into two parts, and the expensive geometric part
is computed once in advance rather than at every training step.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrainingConfig:
    hidden: tuple[int, ...] = (32, 32)
    force_weight: float = 10.0     # forces matter more: 3N of them against 1 per frame
    epochs: int = 400
    batch_frames: int = 16
    learning_rate: float = 3e-3
    weight_decay: float = 1e-6
    seed: int = 0


def _torch():
    import torch
    return torch


class SpeciesNetwork:
    """A set of small networks: one per atomic species."""

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
        """Atomic energies: (N,) from (N, D) and species."""
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
    Standardisation of descriptors over the training set: zero mean, unit spread.

    This is not cosmetic. Descriptors differ in magnitude by orders of magnitude (radial
    sums against angular ones), and without bringing them to a common scale, training is
    dominated by the largest feature while the rest effectively do not participate. The
    scaling factors are saved with the model and applied at inference time; otherwise a
    port to WGSL would give different numbers.
    """
    stacked = np.vstack(frames_features)
    mean = stacked.mean(axis=0)
    std = stacked.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean, std


def fit_atomic_baseline(frames, species: tuple[int, ...]) -> np.ndarray:
    """
    Atomic baselines from composition: E ≈ sum_Z n_Z · e0(Z), solved by least squares.

    Without this step training fails, and this was MEASURED, not assumed: the teacher's
    total energy carries about -700 eV per atom of atomic baseline, and a network with a
    random initialization first spends all of its capacity reproducing that constant.
    Measured: a network without subtracting the baseline gave 11570 meV/atom against 350
    for a linear model with a free constant term. After subtraction, the network only
    learns the interaction, a quantity of the order of a few eV.
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
    Training on energies and forces. `precomputed` is a list of (G, dG) per frame, so the
    descriptors are not recomputed at every epoch (they do not depend on the network's
    parameters).
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

    # pre-build the tensors: (G-mean)/std and dG/std
    prepared = []
    for (g, dg), fr in zip(precomputed, frames):
        gt = torch.tensor((g - mean) / std, requires_grad=True)
        dgt = torch.tensor(dg / std[None, :, None, None])
        prepared.append(
            {
                "g": gt,
                "dg": dgt,
                "numbers": torch.tensor(np.asarray(fr["numbers"], dtype=int)),
                # the network's target is the RESIDUAL after subtracting the atomic baseline,
                # see fit_atomic_baseline
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
                # dE/dG is the derivative with respect to descriptors; the geometry is already in dG
                (de_dg,) = torch.autograd.grad(energy, g, create_graph=True)
                forces = -torch.einsum("id,idka->ka", de_dg, item["dg"])
                loss = loss + ((energy - item["energy"]) / item["n"]) ** 2
                loss = loss + cfg.force_weight * ((forces - item["forces"]) ** 2).mean()
            (loss / len(batch)).backward()
            opt.step()
        if verbose and (epoch + 1) % max(1, cfg.epochs // 8) == 0:
            history.append(float(loss.item()) / max(1, len(batch)))
            print(f"    epoch {epoch + 1:4d}/{cfg.epochs}: loss {history[-1]:.5f}", flush=True)

    return {
        "net": net,
        "mean": mean,
        "std": std,
        "spec": spec,
        "species": species,
        "baseline": baseline,
    }


def predict(model: dict, positions, numbers, cell=None):
    """Energy and forces from the trained model."""
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
