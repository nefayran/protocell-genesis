"""
Batched labeling by the teacher: many configurations in a single model call.

Why. Labeling is the main cost of training, and it ran one frame per call through the ASE
interface. A measurement showed what this costs: on a cloud Tesla P100 the teacher gave
22-92 atom-calcs per second against 460 on our CPU, i.e. the GPU turned out to be MANY TIMES
SLOWER. The cause is not the GPU: for systems of 3-96 atoms, kernel-launch and data-transfer
overhead exceeds the computation itself, and the GPU sits idle. The fix is not a hardware
choice but batching: if a hundred configurations go in a single call, the work finally
fills the GPU.

Here the internal MACE interface is used (mace.data.AtomicData plus its own batch loader)
rather than ASE: ASE's calculator is single-frame by design and a batch cannot be expressed
through it.

A side benefit, no less important: batching speeds up the CPU too, in the same way, fewer
Python calls per frame, better use of multithreading in matrix operations.
"""
from __future__ import annotations

import time

import numpy as np


class BatchedMACE:
    """
    A teacher that labels in batches. Returns energies in eV and forces in eV/Å, the same
    units as the rest of the engine, without conversion.

    `batch_size` is chosen by MEASUREMENT (see `benchmark`), not assigned arbitrarily: too
    large a batch runs into GPU memory limits, too small a one does not fill it with work.
    """

    def __init__(self, model_path: str, device: str = "cpu", dtype: str = "float64") -> None:
        import torch
        from mace.calculators import MACECalculator

        # the ASE calculator is only used to load the model and the species table; the
        # actual labeling bypasses it
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
        Labels a list of states (each with positions/numbers/cell) in batches.

        Returns a list of dicts in the same shape that training expects. Frames of
        DIFFERENT sizes within one batch are allowed: the representation is graph-based,
        the number of atoms is not part of the batch's shape.
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
                # forces come as one array for the whole batch; slice them by atom count
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
                    f"forces sliced incorrectly: used {offset} rows out of {len(forces)}"
                )
        return out

    def benchmark(self, states, batch_sizes=(1, 8, 32, 128)) -> dict:
        """
        Measures how many atom-calcs per second each batch size gives.

        The first call is a warm-up and is not part of the measurement: on the GPU the
        first kernel launch includes compilation, and without warm-up, batch size 1 would
        look much worse than it actually is.
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
