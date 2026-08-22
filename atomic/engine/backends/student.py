"""
Свой обученный потенциал как обычный вычислитель движка.

Смысл этого файла -- поставить нашу модель на то же место, где стоят GFN2 и DFT, чтобы её можно
было проверять теми же тестами и гонять тем же интегратором. Никаких поблажек: если модель
плоха, это покажут те же проверки, что ловили ошибки у остальных бэкендов.

Стоимость на атом здесь -- дескрипторы плюс две-три матричные операции крошечной сети. Именно
это и должно дать выигрыш против модели с обменом сообщениями, и именно это надо ЗАМЕРИТЬ, а не
объявить: замер лежит в atomic/validate/test_student.py рядом с проверкой точности.
"""
from __future__ import annotations

import numpy as np

from ..potential import PotentialResult
from ..state import AtomicState


class StudentPotential:
    """Обученная сеть поверх симметрийных дескрипторов, загруженная из файла обучения."""

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

        # сеть восстанавливается из сохранённых весов по их же форме: число слоёв и их размеры
        # не задаются здесь заново, иначе файл и код могли бы разойтись молча
        self.nets = {}
        for z in self.species:
            weights, biases = [], []
            i = 0
            while f"{z}_{i}.weight" in data:
                weights.append(data[f"{z}_{i}.weight"])
                biases.append(data[f"{z}_{i}.bias"])
                i += 2      # в Sequential между линейными слоями стоит нелинейность
            if not weights:
                raise ValueError(f"в файле {npz_path} нет весов для сорта Z={z}")
            self.nets[z] = [
                (torch.tensor(w, device=device, dtype=self.dtype),
                 torch.tensor(b, device=device, dtype=self.dtype))
                for w, b in zip(weights, biases)
            ]
        # Слитый путь расчёта -- не деталь реализации, а причина, по которой этот бэкенд
        # вообще имеет смысл: замер против учителя даёт 11-31 раз именно на нём, а на прежнем
        # пути (через полный тензор производных) наш потенциал был МЕДЛЕННЕЕ учителя на 192
        # атомах. Подробности и числа -- в atomic/README.md.
        from train.descriptors_torch import TorchDescriptors

        self.mean_t = torch.tensor(self.mean, device=device, dtype=self.dtype)
        self.std_t = torch.tensor(self.std, device=device, dtype=self.dtype)
        self.descriptors = TorchDescriptors(
            self.spec, self.species, device=device,
            dtype="float32" if self.dtype == torch.float32 else "float64",
        )
        self.name = f"student/D={len(self.mean)}/{device}"

    def _de_dg(self, g, numbers):
        """Энергия и производная энергии по дескрипторам -- всё, что нужно для сил."""
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
        # цепное правило по нормировке: dE/dG = (dE/dG_norm) / std
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
