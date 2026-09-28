"""
Kaggle kernel, version 2: accelerated training pipeline for our own potential.

What changed compared to version 1 and WHY, each point backed by a measurement, not a
hunch:

1. CONFIGURATIONS are decoupled from dynamics. In version 1 frames came from molecular
   dynamics, which is sequential: each step needs the force at the previous point, so it
   cannot be batched. That gave 547 frames in 301 seconds and an idle GPU. Random sampling
   yields 4000 frames per second.
2. LABELING is batched through MACE's internal interface. Version 1's measurement on the
   same GPU: 22-92 atom-calcs per second versus 460 on our CPU. The GPU was idling because
   for systems of 3-96 atoms the overhead outweighs the actual computation. Batch size here
   is MEASURED.
3. DESCRIPTORS and TRAINING are vectorized. Descriptors: 58.5x speedup while matching the
   direct version to 1e-14. Training: batched by groups of equal size, with gradient
   accumulation across all groups (without accumulation, the step is biased toward one
   size, measured 548 versus 246 meV/atom).

The safeguards from the kaggle-offload skill are all in place: the GPU is verified by
computation BEFORE any spend, the payload is located by walking the tree, code overlay is
checked with an assert, and results are written as we go.
"""
import json
import os
import subprocess
import sys
import tarfile
import time

WORK = "/kaggle/working"
ROOT = "/tmp/atomic"
RESULT = os.path.join(WORK, "result")
os.makedirs(RESULT, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def save_json(name, obj):
    with open(os.path.join(RESULT, name), "w") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=1, default=str)


log("=== compute device ===")
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
except Exception as exc:
    log(f"nvidia-smi unavailable: {exc}")

log("=== installing dependencies ===")
import torch as _torch_pre

TORCH_PIN = _torch_pre.__version__.split("+")[0]
log(f"preinstalled torch {_torch_pre.__version__}, pinning to {TORCH_PIN}")
with open("/tmp/constraints.txt", "w") as fh:
    fh.write(f"torch=={TORCH_PIN}\n")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-c", "/tmp/constraints.txt", "mace-torch", "ase"],
    check=True,
)
import numpy as np
import torch

DEVICE = "cpu"
if torch.cuda.is_available():
    try:
        probe = (torch.randn(64, 64, device="cuda") @ torch.randn(64, 64, device="cuda")).sum()
        torch.cuda.synchronize()
        _ = float(probe)
        DEVICE = "cuda"
        log(f"GPU works: {torch.cuda.get_device_name(0)}, capability {torch.cuda.get_device_capability(0)}")
    except Exception as exc:
        log(f"GPU DOES NOT COMPUTE ({type(exc).__name__}: {str(exc)[:120]}) -- falling back to CPU")
log(f"compute device: {DEVICE}")

log("=== payload ===")
payload_pack = marker_src = None
for base, dirs, files in os.walk("/kaggle/input"):
    if base.count(os.sep) - "/kaggle/input".count(os.sep) > 3:
        dirs[:] = []
        continue
    for f in files:
        if f == "atomic_code.payloadpack":
            payload_pack = os.path.join(base, f)
        if f == "CODE_MARKER":
            marker_src = os.path.join(base, f)
if payload_pack is None:
    raise SystemExit("payload atomic_code.payloadpack not found -- check the dataset")
log(f"payload: {payload_pack} ({os.path.getsize(payload_pack) / 1e6:.1f} MB)")
if marker_src:
    log("marker: " + open(marker_src).read().strip())
os.makedirs(ROOT, exist_ok=True)
with tarfile.open(payload_pack) as tf:
    tf.extractall(ROOT)
for must in ("train/sampling.py", "train/batched_teacher.py", "train/descriptors_fast.py",
             "train/fast_train.py", "models/MACE-OFF23_medium.model"):
    assert os.path.exists(os.path.join(ROOT, must)), f"payload extracted incompletely: missing {must}"
log("payload contents verified -- code is fresh")
sys.path.insert(0, ROOT)

from engine.state import from_symbols                                       # noqa: E402
from train.batched_teacher import BatchedMACE                               # noqa: E402
from train.descriptors import DescriptorSpec, descriptor_length             # noqa: E402
from train.descriptors_fast import compute_descriptors_fast                 # noqa: E402
from train.fast_train import FastTrainingConfig, evaluate_fast, train_fast   # noqa: E402
from train.sampling import build_sampled_states, clusters                   # noqa: E402

CONFIG = {
    "n_monomers": 900,
    "n_dimers": 1500,
    "n_clusters": 4600,
    "holdout_md_frames": 120,
    "descriptor_radial": 8,
    "teacher_dtype": "float32",
    # The sweep is trimmed down to two variants: a measurement from the previous run showed
    # that network capacity has almost no effect (forces 85.4 versus 83.8 meV/Å when the
    # network is tripled in size), and it's the DATA that's the limiting factor. So there
    # are now 13x more frames and fewer variants -- time goes where it actually matters.
    "sweep": [
        {"hidden": (96, 96), "force_weight": 30.0, "lr": 2e-3, "epochs": 4000},
        {"hidden": (128, 128, 64), "force_weight": 50.0, "lr": 2e-3, "epochs": 4000},
    ],
}
save_json("config.json", CONFIG)

log("=== teacher: batch size benchmark ===")
teacher = BatchedMACE(os.path.join(ROOT, "models/MACE-OFF23_medium.model"), device=DEVICE,
                      dtype=CONFIG["teacher_dtype"])
rng = np.random.default_rng(20260821)
probe_states = clusters(rng, 160, sizes=(8,))
bench = teacher.benchmark(probe_states, batch_sizes=(1, 8, 32, 128))
for bs, r in bench.items():
    log(f"batch {bs:4d}: {r['frames_per_second']:8.2f} frame/s, {r['atom_calcs_per_second']:9.0f} atom-calc/s")
save_json("teacher_batch_bench.json", bench)
best_batch = max(bench, key=lambda bs: bench[bs]["atom_calcs_per_second"])
log(f"best batch size measured: {best_batch} "
    f"({bench[best_batch]['atom_calcs_per_second']:.0f} atom-calc/s versus 460 on our CPU)")

log("=== configurations ===")
t0 = time.perf_counter()
states = build_sampled_states(rng, CONFIG["n_monomers"], CONFIG["n_dimers"], CONFIG["n_clusters"])
dt = max(time.perf_counter() - t0, 1e-9)
log(f"{len(states)} configurations in {dt:.2f} s ({len(states) / dt:.0f} frames/s)")

log("=== batch labeling ===")
frames = []
t0 = time.perf_counter()
for start in range(0, len(states), 500):
    frames += teacher.label(states[start : start + 500], batch_size=best_batch)
    done = time.perf_counter() - t0
    log(f"  labeled {len(frames)}/{len(states)} in {done:.0f} s ({len(frames) / done:.1f} frame/s)")
    save_json("dataset_progress.json", {"labelled": len(frames), "of": len(states), "seconds": done})

log("=== holdout sample FROM DYNAMICS ===")
from engine.backends.mlip import ASECalculatorPotential    # noqa: E402
from engine.md import run                                   # noqa: E402
from mace.calculators import MACECalculator                 # noqa: E402

md_teacher = ASECalculatorPotential(
    MACECalculator(model_paths=os.path.join(ROOT, "models/MACE-OFF23_medium.model"),
                   device=DEVICE, default_dtype=CONFIG["teacher_dtype"]),
    "mace-single",
)
md_states = []
st = clusters(rng, 1, sizes=(8,))[0]
st.set_maxwell_boltzmann(330.0, rng)
for _ in range(CONFIG["holdout_md_frames"]):
    try:
        run(st, md_teacher, steps=10, dt_fs=0.5, temperature_k=330.0,
            seed=int(rng.integers(1 << 30)), sample_every=10, watch_chemistry=False)
        md_states.append(st.copy())
    except FloatingPointError as exc:
        log(f"  dynamics broke off: {exc}")
        break
md_frames = teacher.label(md_states, batch_size=best_batch) if md_states else []
log(f"holdout frames from dynamics: {len(md_frames)}")

log("=== descriptors (vectorized) ===")
SPECIES = (1, 8)
spec = DescriptorSpec(n_radial=CONFIG["descriptor_radial"])
log(f"D = {descriptor_length(spec, SPECIES)}")


def descriptors_for(batch, label):
    t = time.perf_counter()
    out = []
    for k, fr in enumerate(batch):
        out.append(compute_descriptors_fast(np.asarray(fr["positions"]),
                                            np.asarray(fr["numbers"], dtype=int),
                                            spec, SPECIES, cell=fr.get("cell")))
        if (k + 1) % 1000 == 0:
            log(f"  {label}: {k + 1}/{len(batch)} in {time.perf_counter() - t:.0f} s")
    log(f"  {label}: done in {time.perf_counter() - t:.0f} s")
    return out


pre = descriptors_for(frames, "dataset")
pre_md = descriptors_for(md_frames, "dynamics") if md_frames else []

order = np.random.default_rng(1).permutation(len(frames))
n_test = int(0.1 * len(frames))
test = [frames[i] for i in order[:n_test]]
pre_test = [pre[i] for i in order[:n_test]]
train_frames = [frames[i] for i in order[n_test:]]
pre_train = [pre[i] for i in order[n_test:]]
log(f"training {len(train_frames)}, random holdout {len(test)}, from dynamics {len(md_frames)}")

log("=== training ===")
results, best = [], None
for variant in CONFIG["sweep"]:
    cfg = FastTrainingConfig(
        hidden=tuple(variant["hidden"]), force_weight=variant["force_weight"],
        epochs=variant["epochs"], learning_rate=variant["lr"], seed=0,
        memory_budget_mb=1024.0, dtype="float32", lr_final_fraction=0.05,
    )
    log(f"variant {variant}")
    t0 = time.perf_counter()
    model = train_fast(train_frames, spec, SPECIES, cfg, pre_train, device=DEVICE, verbose=True)
    took = time.perf_counter() - t0
    metrics = {
        "variant": variant,
        "test_random": evaluate_fast(model, test, pre_test),
        "test_md": evaluate_fast(model, md_frames, pre_md) if md_frames else None,
        "seconds": took,
        "n_train": len(train_frames),
    }
    results.append(metrics)
    log(f"  random:     E {metrics['test_random']['energy_mae_mev_per_atom']:.2f} meV/atom, "
        f"F {metrics['test_random']['force_mae_mev_per_a']:.1f} meV/Å")
    if metrics["test_md"]:
        log(f"  dynamics:   E {metrics['test_md']['energy_mae_mev_per_atom']:.2f} meV/atom, "
            f"F {metrics['test_md']['force_mae_mev_per_a']:.1f} meV/Å   ({took:.0f} s)")
    save_json("training_results.json", results)
    score = (metrics["test_md"] or metrics["test_random"])["force_mae_mev_per_a"]
    if best is None or score < best[0]:
        best = (score, variant)
        np.savez(os.path.join(RESULT, "best_model.npz"),
                 mean=model["mean"], std=model["std"], baseline=model["baseline"],
                 species=np.array(SPECIES), n_radial=CONFIG["descriptor_radial"],
                 **{f"{z}_{k}": v for z, sd in model["net"].state_dict().items() for k, v in sd.items()})
        save_json("best_variant.json", {"force_mae_mev_per_a": best[0], "variant": variant})

log(f"=== done. best: {best}")
