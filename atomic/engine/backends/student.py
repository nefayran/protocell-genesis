"""
Our own trained potential as an ordinary engine calculator.

The point of this file is to put our model in the same slot as GFN2 and DFT, so it can
be validated with the same tests and driven by the same integrator. No special
treatment: if the model is bad, the same checks that caught bugs in the other backends
will show it.

The per-atom cost here is the descriptors plus two or three matrix operations of a tiny
network. That is exactly what is supposed to give the advantage over a message-passing
model, and that is exactly what must be MEASURED, not asserted: the benchmark lives in
atomic/validate/test_student.py alongside the accuracy check.
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState


class StudentPotential:
    """Trained network on top of symmetry descriptors, loaded from a training file."""

    name = "student"

    def __init__(self, npz_path: str, device: str = "cpu") -> None:
        import torch

        from train.descriptors import DescriptorSpec

        data = np.load(npz_path)
        self.species = tuple(int(z) for z in data["species"])
        self.spec = DescriptorSpec(
            n_radial=int(data["n_radial"]),
            cutoff=float(data["cutoff"]) if "cutoff" in data else 5.0,
        )
        self.mean = data["mean"]
        self.std = data["std"]
        self.baseline = data["baseline"]
        self.device = device
        self.dtype = torch.float32

        # the network is reconstructed from the saved weights by their own shape: the
        # number of layers and their sizes are not set here again, otherwise the file
        # and the code could silently diverge
        self.nets = {}
        for z in self.species:
            weights, biases = [], []
            i = 0
            while f"{z}_{i}.weight" in data:
                weights.append(data[f"{z}_{i}.weight"])
                biases.append(data[f"{z}_{i}.bias"])
                i += 2      # in the Sequential there is a nonlinearity between linear layers
            if not weights:
                raise ValueError(f"file {npz_path} has no weights for species Z={z}")
            self.nets[z] = [
                (torch.tensor(w, device=device, dtype=self.dtype),
                 torch.tensor(b, device=device, dtype=self.dtype))
                for w, b in zip(weights, biases)
            ]
        # The fused computation path is not an implementation detail but the reason this
        # backend makes sense at all: benchmarking against the teacher gives an 11-31x
        # speedup precisely on this path, while on the previous path (through the full
        # derivative tensor) our potential was SLOWER than the teacher at 192 atoms.
        # Details and numbers are in atomic/README.md.
        from train.descriptors_torch import TorchDescriptors

        self.mean_t = torch.tensor(self.mean, device=device, dtype=self.dtype)
        self.std_t = torch.tensor(self.std, device=device, dtype=self.dtype)
        self.descriptors = TorchDescriptors(
            self.spec, self.species, device=device,
            dtype="float32" if self.dtype == torch.float32 else "float64",
        )
        self.name = f"student/D={len(self.mean)}/{device}"

    def _de_dg(self, g, numbers):
        """Energy and the derivative of energy with respect to descriptors: everything needed for forces."""
        import torch

        gt = ((g - self.mean_t) / self.std_t).detach().requires_grad_(True)
        atom_e = torch.zeros(gt.shape[0], device=self.device, dtype=self.dtype)
        for z, layers in self.nets.items():
            sel = np.flatnonzero(numbers == z)
            if sel.size == 0:
                continue
            idx = torch.as_tensor(sel, device=self.device)
            x = gt[idx]
            for k, (w, b) in enumerate(layers):
                x = x @ w.T + b
                if k < len(layers) - 1:
                    x = torch.nn.functional.softplus(x)
            atom_e[idx] = x.squeeze(-1)
        energy = atom_e.sum()
        (grad,) = torch.autograd.grad(energy, gt)
        # chain rule through normalization: dE/dG = (dE/dG_norm) / std
        return float(energy.item()), grad / self.std_t

    def compute(self, state: AtomicState) -> PotentialResult:
        numbers = np.asarray(state.numbers, dtype=int)
        energy, forces = self.descriptors.energy_forces_fused(
            state.positions, state.numbers,
            lambda g: self._de_dg(g, numbers),
            cell=state.cell,
        )
        offset = sum(
            float(self.baseline[s]) * int((numbers == z).sum())
            for s, z in enumerate(self.species)
        )
        return PotentialResult(energy + offset, forces, {"backend": self.name})
