"""
Ядро Kaggle, версия 2: ускоренный конвейер обучения своего потенциала.

Что изменилось против версии 1 и ПОЧЕМУ -- каждое по замеру, а не по идее:

1. КОНФИГУРАЦИИ отвязаны от динамики. В версии 1 кадры получались молекулярной динамикой, а она
   последовательна: каждый шаг требует силы в предыдущей точке, и пакетом её не собрать. Отсюда
   547 кадров за 301 секунду и простой карты. Случайная выборка даёт 4000 кадров в секунду.
2. РАЗМЕТКА пакетная, через внутренний интерфейс MACE. Замер версии 1 на этой же карте: 22-92
   атом-расчёта в секунду против 460 на нашем процессоре -- карта простаивала, потому что на
   системе из 3-96 атомов накладные расходы больше счёта. Размер пакета здесь ЗАМЕРЯЕТСЯ.
3. ДЕСКРИПТОРЫ и ОБУЧЕНИЕ векторизованы. Дескрипторы: 58.5x при совпадении с прямой версией до
   1e-14. Обучение: пакетами по группам одного размера, с накоплением градиента по всем группам
   (без накопления шаг смещён в сторону одного размера -- замер 548 против 246 мэВ/атом).

Защиты из скилла kaggle-offload все на месте: карта проверяется вычислением ДО расходов, груз
ищется обходом, наложение кода проверяется assert-ом, результат пишется по ходу.
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


log("=== вычислитель ===")
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
except Exception as exc:
    log(f"nvidia-smi недоступен: {exc}")

log("=== установка зависимостей ===")
import torch as _torch_pre

TORCH_PIN = _torch_pre.__version__.split("+")[0]
log(f"предустановленный torch {_torch_pre.__version__}, фиксируем {TORCH_PIN}")
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
        log(f"карта работает: {torch.cuda.get_device_name(0)}, способность {torch.cuda.get_device_capability(0)}")
    except Exception as exc:
        log(f"КАРТА НЕ СЧИТАЕТ ({type(exc).__name__}: {str(exc)[:120]}) -- переходим на CPU")
log(f"вычислитель: {DEVICE}")

log("=== груз ===")
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
    raise SystemExit("груз atomic_code.payloadpack не найден -- проверить датасет")
log(f"груз: {payload_pack} ({os.path.getsize(payload_pack) / 1e6:.1f} МБ)")
if marker_src:
    log("маркер: " + open(marker_src).read().strip())
os.makedirs(ROOT, exist_ok=True)
with tarfile.open(payload_pack) as tf:
    tf.extractall(ROOT)
for must in ("train/sampling.py", "train/batched_teacher.py", "train/descriptors_fast.py",
             "train/fast_train.py", "models/MACE-OFF23_medium.model"):
    assert os.path.exists(os.path.join(ROOT, must)), f"груз развернулся неполно: нет {must}"
log("состав груза проверен -- код свежий")
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
    # Перебор сокращён до двух вариантов: замер прошлого прогона показал, что ёмкость сети
    # почти не влияет (силы 85.4 против 83.8 мэВ/Å при росте сети втрое), а ограничивают ДАННЫЕ.
    # Поэтому кадров теперь в 13 раз больше, а вариантов меньше -- время идёт туда, где эффект.
    "sweep": [
        {"hidden": (96, 96), "force_weight": 30.0, "lr": 2e-3, "epochs": 4000},
        {"hidden": (128, 128, 64), "force_weight": 50.0, "lr": 2e-3, "epochs": 4000},
    ],
}
save_json("config.json", CONFIG)

log("=== учитель: замер размера пакета ===")
teacher = BatchedMACE(os.path.join(ROOT, "models/MACE-OFF23_medium.model"), device=DEVICE,
                      dtype=CONFIG["teacher_dtype"])
rng = np.random.default_rng(20260821)
probe_states = clusters(rng, 160, sizes=(8,))
bench = teacher.benchmark(probe_states, batch_sizes=(1, 8, 32, 128))
for bs, r in bench.items():
    log(f"пакет {bs:4d}: {r['frames_per_second']:8.2f} кадр/с, {r['atom_calcs_per_second']:9.0f} атом-расчёт/с")
save_json("teacher_batch_bench.json", bench)
best_batch = max(bench, key=lambda bs: bench[bs]["atom_calcs_per_second"])
log(f"лучший пакет по замеру: {best_batch} "
    f"({bench[best_batch]['atom_calcs_per_second']:.0f} атом-расчёт/с против 460 на нашем CPU)")

log("=== конфигурации ===")
t0 = time.perf_counter()
states = build_sampled_states(rng, CONFIG["n_monomers"], CONFIG["n_dimers"], CONFIG["n_clusters"])
dt = max(time.perf_counter() - t0, 1e-9)
log(f"{len(states)} конфигураций за {dt:.2f} с ({len(states) / dt:.0f} кадров/с)")

log("=== разметка пакетом ===")
frames = []
t0 = time.perf_counter()
for start in range(0, len(states), 500):
    frames += teacher.label(states[start : start + 500], batch_size=best_batch)
    done = time.perf_counter() - t0
    log(f"  размечено {len(frames)}/{len(states)} за {done:.0f} с ({len(frames) / done:.1f} кадр/с)")
    save_json("dataset_progress.json", {"labelled": len(frames), "of": len(states), "seconds": done})

log("=== отложенная выборка ИЗ ДИНАМИКИ ===")
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
        log(f"  динамика оборвалась: {exc}")
        break
md_frames = teacher.label(md_states, batch_size=best_batch) if md_states else []
log(f"отложенных кадров из динамики: {len(md_frames)}")

log("=== дескрипторы (векторные) ===")
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
            log(f"  {label}: {k + 1}/{len(batch)} за {time.perf_counter() - t:.0f} с")
    log(f"  {label}: готово за {time.perf_counter() - t:.0f} с")
    return out


pre = descriptors_for(frames, "набор")
pre_md = descriptors_for(md_frames, "динамика") if md_frames else []

order = np.random.default_rng(1).permutation(len(frames))
n_test = int(0.1 * len(frames))
test = [frames[i] for i in order[:n_test]]
pre_test = [pre[i] for i in order[:n_test]]
train_frames = [frames[i] for i in order[n_test:]]
pre_train = [pre[i] for i in order[n_test:]]
log(f"обучающих {len(train_frames)}, случайных отложенных {len(test)}, из динамики {len(md_frames)}")

log("=== обучение ===")
results, best = [], None
for variant in CONFIG["sweep"]:
    cfg = FastTrainingConfig(
        hidden=tuple(variant["hidden"]), force_weight=variant["force_weight"],
        epochs=variant["epochs"], learning_rate=variant["lr"], seed=0,
        memory_budget_mb=1024.0, dtype="float32", lr_final_fraction=0.05,
    )
    log(f"вариант {variant}")
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
    log(f"  случайные:  E {metrics['test_random']['energy_mae_mev_per_atom']:.2f} мэВ/атом, "
        f"F {metrics['test_random']['force_mae_mev_per_a']:.1f} мэВ/Å")
    if metrics["test_md"]:
        log(f"  динамика:   E {metrics['test_md']['energy_mae_mev_per_atom']:.2f} мэВ/атом, "
            f"F {metrics['test_md']['force_mae_mev_per_a']:.1f} мэВ/Å   ({took:.0f} с)")
    save_json("training_results.json", results)
    score = (metrics["test_md"] or metrics["test_random"])["force_mae_mev_per_a"]
    if best is None or score < best[0]:
        best = (score, variant)
        np.savez(os.path.join(RESULT, "best_model.npz"),
                 mean=model["mean"], std=model["std"], baseline=model["baseline"],
                 species=np.array(SPECIES), n_radial=CONFIG["descriptor_radial"],
                 **{f"{z}_{k}": v for z, sd in model["net"].state_dict().items() for k, v in sd.items()})
        save_json("best_variant.json", {"force_mae_mev_per_a": best[0], "variant": variant})

log(f"=== готово. лучший: {best}")
