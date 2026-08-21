"""
Пакетная разметка учителем: много конфигураций одним вызовом модели.

Зачем. Разметка -- главная стоимость обучения, и она шла по одному кадру за вызов через
интерфейс ASE. Замер показал, во что это обходится: на облачной Tesla P100 учитель дал
22-92 атом-расчёта в секунду против 460 на нашем процессоре, то есть карта оказалась В РАЗЫ
МЕДЛЕННЕЕ. Причина не в карте: на системе из 3-96 атомов накладные расходы на запуск ядра и
перегонку данных больше самого счёта, и карта простаивает. Лечится это не выбором железа, а
пакетом: если за один вызов идёт сто конфигураций, работа наконец заполняет карту.

Здесь берётся внутренний интерфейс MACE (mace.data.AtomicData + собственный загрузчик
пакетов), а не ASE: ASE по устройству однокадровый и пакет через него не выразить.

Побочная выгода, не менее важная: пакет одинаково ускоряет и процессор -- меньше вызовов
питона на кадр, лучше используется многопоточность в матричных операциях.
"""
from __future__ import annotations

import time

import numpy as np


class BatchedMACE:
    """
    Учитель, размечающий пакетами. Возвращает энергии в эВ и силы в эВ/Å -- те же единицы,
    что у остального движка, без пересчёта.

    `batch_size` подбирается ЗАМЕРОМ (см. `benchmark`), а не назначается: слишком большой
    пакет упирается в память карты, слишком малый не заполняет её работой.
    """

    def __init__(self, model_path: str, device: str = "cpu", dtype: str = "float64") -> None:
        import torch
        from mace.calculators import MACECalculator

        # калькулятор ASE нужен только как загрузчик модели и таблицы сортов: сама разметка
        # идёт мимо него
        self._calc = MACECalculator(model_paths=model_path, device=device, default_dtype=dtype)
        self.model = self._calc.models[0]
        self.z_table = self._calc.z_table
        self.r_max = float(self._calc.r_max)
        self.device = device
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        self.name = f"mace-batched/{device}/{dtype}"

    def _to_data(self, positions, numbers, cell):
        from ase import Atoms
        from mace import data as mace_data

        atoms = Atoms(
            numbers=np.asarray(numbers, dtype=int),
            positions=np.asarray(positions, dtype=float),
            cell=None if cell is None else np.asarray(cell, dtype=float),
            pbc=cell is not None,
        )
        config = mace_data.config_from_atoms(atoms)
        return mace_data.AtomicData.from_config(config, z_table=self.z_table, cutoff=self.r_max)

    def label(self, states, batch_size: int = 64):
        """
        Размечает список состояний (у каждого positions/numbers/cell) пакетами.

        Возвращает список словарей в том же виде, что ждёт обучение. Кадры РАЗНОГО размера в
        одном пакете допустимы: представление графовое, число атомов в пакет не входит.
        """
        import torch
        from mace.tools.torch_geometric import DataLoader

        out = []
        for start in range(0, len(states), batch_size):
            chunk = states[start : start + batch_size]
            dataset = [
                self._to_data(st.positions, st.numbers, st.cell) for st in chunk
            ]
            loader = DataLoader(dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
            for batch in loader:
                batch = batch.to(self.device)
                result = self.model(batch.to_dict(), compute_force=True)
                energies = result["energy"].detach().cpu().numpy()
                forces = result["forces"].detach().cpu().numpy()
                # силы приходят одним массивом на весь пакет -- разрезаем по числу атомов
                offset = 0
                for st, energy in zip(chunk, energies):
                    n = st.n_atoms
                    out.append(
                        {
                            "positions": st.positions.copy(),
                            "numbers": st.numbers.copy(),
                            "energy": float(energy),
                            "forces": forces[offset : offset + n].copy(),
                            "cell": None if st.cell is None else st.cell.copy(),
                        }
                    )
                    offset += n
                assert offset == sum(st.n_atoms for st in chunk), (
                    f"силы разрезаны неверно: использовано {offset} строк из {len(forces)}"
                )
        return out

    def benchmark(self, states, batch_sizes=(1, 8, 32, 128)) -> dict:
        """
        Замер: сколько атом-расчётов в секунду даёт каждый размер пакета.

        Первый вызов прогревается и в замер не идёт: на карте первый запуск ядра включает
        компиляцию, и без прогрева пакет 1 выглядел бы намного хуже, чем он есть.
        """
        self.label(states[: min(4, len(states))], batch_size=2)
        report = {}
        for bs in batch_sizes:
            probe = states[: min(len(states), max(bs * 2, 8))]
            t0 = time.perf_counter()
            self.label(probe, batch_size=bs)
            took = time.perf_counter() - t0
            atoms = sum(st.n_atoms for st in probe)
            report[bs] = {
                "frames": len(probe),
                "seconds": took,
                "frames_per_second": len(probe) / took,
                "atom_calcs_per_second": atoms / took,
            }
        return report
