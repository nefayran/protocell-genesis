"""
A full training run on this machine.

Why here, and not in the cloud, by measurement, not convenience: the teacher on our CPU
gives 460 atom-calcs per second against 103 on Kaggle's CPU and 22-92 on a cloud Tesla
P100. For systems of 3-96 atoms, overhead exceeds the computation itself, so the cloud
GPU sits idle, while the weak cloud CPU is simply slower. Plus, here there is 48 GB of
memory against the cloud limit that killed the previous run: descriptor derivatives for
7000 frames take 14.5 GB in double precision and 7.26 GB in single precision.

Storage was deliberately switched to SINGLE precision: this is not a loss of physics,
because training runs in single precision anyway, and checking descriptors for agreement
with the direct implementation (1e-14) is done separately and in double precision.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backends.mlip import ASECalculatorPotential
from engine.md import run as md_run
from train.batched_teacher import BatchedMACE
from train.descriptors import DescriptorSpec, descriptor_length
from train.descriptors_fast import compute_descriptors_fast
from train.fast_train import FastTrainingConfig, evaluate_fast, train_fast
from train.sampling import build_sampled_states, clusters, filter_outliers

OUT = "models/local_run"
os.makedirs(OUT, exist_ok=True)
MODEL = "models/MACE-OFF23_medium.model"
SPECIES = (1, 8)
CONFIG = {
    "n_monomers": 900, "n_dimers": 1500, "n_clusters": 4600,
    "holdout_md_frames": 120, "label_batch": 8,
    # BASIS RESOLUTION was checked with a separate experiment on the same data (models/
    # resolution_test.json): doubling the radial functions at a radius of 5 Å improved
    # forces from 109.1 to 71.5 meV/Å, while increasing the radius to 6 Å made them WORSE
    # (71.5 -> 90.6): more environment at the same capacity and the same data smears out
    # the model. So the radius stays at 5 Å, and the sweep goes over the number of radial
    # functions; the cost of descriptors barely grows because of this (27-31 s).
    "sweep": [
        {"radial": 16, "hidden": (128, 128, 64), "force_weight": 50.0, "lr": 2e-3, "epochs": 4000},
        {"radial": 24, "hidden": (128, 128, 64), "force_weight": 50.0, "lr": 2e-3, "epochs": 4000},
    ],
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save(name, obj):
    with open(os.path.join(OUT, name), "w") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)


def main():
    rng = np.random.default_rng(20260821)
    log("=== configurations ===")
    t0 = time.perf_counter()
    states = build_sampled_states(rng, CONFIG["n_monomers"], CONFIG["n_dimers"], CONFIG["n_clusters"])
    log(f"{len(states)} configurations in {time.perf_counter() - t0:.1f} s")

    log("=== labeling ===")
    teacher = BatchedMACE(MODEL, device="cpu", dtype="float32")
    frames, t0 = [], time.perf_counter()
    for start in range(0, len(states), 500):
        frames += teacher.label(states[start : start + 500], batch_size=CONFIG["label_batch"])
        done = time.perf_counter() - t0
        log(f"  {len(frames)}/{len(states)} in {done:.0f} s ({len(frames) / done:.1f} frames/s)")
        save("progress.json", {"labelled": len(frames), "seconds": done})

    # second line of defense: geometry was checked at sampling time, but only the teacher knows the force
    kept, dropped = filter_outliers(frames)
    log(f"force-based filtering: kept {len(kept)}, dropped {len(dropped)} "
        f"({100 * len(dropped) / max(len(frames), 1):.2f}%)")
    save("filtering.json", {"kept": len(kept), "dropped": len(dropped),
                            "worst_kept_force": max((float(abs(f["forces"]).max()) for f in kept), default=0.0)})
    frames = kept

    log("=== held-out set from dynamics ===")
    single = ASECalculatorPotential(
        __import__("mace.calculators", fromlist=["MACECalculator"]).MACECalculator(
            model_paths=MODEL, device="cpu", default_dtype="float32"),
        "mace-single")
    st = clusters(rng, 1, sizes=(8,))[0]
    st.set_maxwell_boltzmann(330.0, rng)
    md_states = []
    for _ in range(CONFIG["holdout_md_frames"]):
        try:
            md_run(st, single, steps=10, dt_fs=0.5, temperature_k=330.0,
                   seed=int(rng.integers(1 << 30)), sample_every=10, watch_chemistry=False)
            md_states.append(st.copy())
        except FloatingPointError as exc:
            log(f"  dynamics crashed: {exc}")
            break
    md_frames = teacher.label(md_states, batch_size=CONFIG["label_batch"]) if md_states else []
    log(f"frames from dynamics: {len(md_frames)}")

    log("=== descriptors are computed inside the sweep: each variant has its own basis ===")

    def descriptors(batch, label, spec):
        t = time.perf_counter()
        out = []
        for k, fr in enumerate(batch):
            g, dg = compute_descriptors_fast(np.asarray(fr["positions"]),
                                             np.asarray(fr["numbers"], dtype=int),
                                             spec, SPECIES, cell=fr.get("cell"))
            # storage in single precision: 7.26 GB instead of 14.5 for 7000 frames
            out.append((g.astype(np.float32), dg.astype(np.float32)))
            if (k + 1) % 1000 == 0:
                log(f"  {label}: {k + 1}/{len(batch)} in {time.perf_counter() - t:.0f} s")
        log(f"  {label}: done in {time.perf_counter() - t:.0f} s")
        return out

    order = np.random.default_rng(1).permutation(len(frames))
    n_test = int(0.1 * len(frames))
    test_idx, train_idx = order[:n_test], order[n_test:]
    test = [frames[i] for i in test_idx]
    train_frames = [frames[i] for i in train_idx]
    log(f"train {len(train_frames)}, random held-out {len(test)}, from dynamics {len(md_frames)}")

    log("=== training ===")
    results, best = [], None
    for variant in CONFIG["sweep"]:
        spec = DescriptorSpec(n_radial=variant["radial"], cutoff=5.0)
        log(f"basis: {variant['radial']} radial, radius 5.0 Å, D = {descriptor_length(spec, SPECIES)}")
        pre_all = descriptors(frames, f"dataset/D{descriptor_length(spec, SPECIES)}", spec)
        pre_md = descriptors(md_frames, "dynamics", spec) if md_frames else []
        pre_test = [pre_all[i] for i in test_idx]
        pre_train = [pre_all[i] for i in train_idx]
        cfg = FastTrainingConfig(hidden=tuple(variant["hidden"]), force_weight=variant["force_weight"],
                                 epochs=variant["epochs"], learning_rate=variant["lr"], seed=0,
                                 memory_budget_mb=256.0, dtype="float32", lr_final_fraction=0.05)
        log(f"variant {variant}")
        t0 = time.perf_counter()
        model = train_fast(train_frames, spec, SPECIES, cfg, pre_train, device="cpu", verbose=True)
        took = time.perf_counter() - t0
        metrics = {
            "variant": variant,
            "test_random": evaluate_fast(model, test, pre_test),
            "test_md": evaluate_fast(model, md_frames, pre_md) if md_frames else None,
            "seconds": took, "n_train": len(train_frames),
        }
        results.append(metrics)
        log(f"  random:   E {metrics['test_random']['energy_mae_mev_per_atom']:.2f} meV/atom, "
            f"F {metrics['test_random']['force_mae_mev_per_a']:.1f} meV/Å")
        if metrics["test_md"]:
            log(f"  dynamics: E {metrics['test_md']['energy_mae_mev_per_atom']:.2f} meV/atom, "
                f"F {metrics['test_md']['force_mae_mev_per_a']:.1f} meV/Å  ({took:.0f} s)")
        save("training_results.json", results)
        score = (metrics["test_md"] or metrics["test_random"])["force_mae_mev_per_a"]
        if best is None or score < best[0]:
            best = (score, variant)
            np.savez(os.path.join(OUT, "best_model.npz"),
                     mean=model["mean"], std=model["std"], baseline=model["baseline"],
                     species=np.array(SPECIES), n_radial=variant["radial"], cutoff=5.0,
                     **{f"{z}_{k}": v for z, sd in model["net"].state_dict().items() for k, v in sd.items()})
            save("best_variant.json", {"force_mae_mev_per_a": best[0], "variant": variant})
    log(f"=== done. best: {best}")


if __name__ == "__main__":
    main()
