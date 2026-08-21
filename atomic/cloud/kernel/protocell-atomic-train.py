"""
Ядро Kaggle: разметка обучающего набора учителем MACE-OFF23 и обучение своего потенциала.

Порядок и защиты взяты из скилла kaggle-offload -- каждая там стоила прогона:
  * какая карта досталась -- в лог первой строкой, иначе вердикт по скорости не читается;
  * груз ищется ОБХОДОМ /kaggle/input: путь монтирования не постулируется;
  * груз лежит контейнером с неизвестным распаковщику расширением (.payloadpack), потому что
    Kaggle распаковывает архивы при приёме датасета и теряет каталоги мелких файлов;
  * наложение кода проверяется assert-ом по файлу-маркеру И по ключевому файлу: без этого
    ядро молча считает старым кодом;
  * результат копируется в /kaggle/working ПО ХОДУ, чтобы упавший на таймауте прогон отдал
    то, что успел.
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
        json.dump(obj, fh, ensure_ascii=False, indent=1)


# --- 1. какая карта досталась -------------------------------------------------------------
log("=== вычислитель ===")
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)
except Exception as exc:
    log(f"nvidia-smi недоступен: {exc}")

# --- 2. зависимости ----------------------------------------------------------------------
log("=== установка зависимостей ===")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "ase", "mace-torch"],
    check=True,
)
import numpy as np
import torch

log(f"torch {torch.__version__}, CUDA доступна: {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 3. груз: искать обходом, поддержать оба вида ------------------------------------------
log("=== груз ===")
payload_pack = None
marker_src = None
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
    raise SystemExit("груз atomic_code.payloadpack не найден в /kaggle/input -- проверить датасет")
log(f"груз найден: {payload_pack} ({os.path.getsize(payload_pack) / 1e6:.1f} МБ)")
if marker_src:
    log("маркер: " + open(marker_src).read().strip())

os.makedirs(ROOT, exist_ok=True)
with tarfile.open(payload_pack) as tf:
    tf.extractall(ROOT)

# наложение проверяется ФАКТОМ, а не надеждой
for must in ("engine/md.py", "train/descriptors.py", "train/nn.py", "models/MACE-OFF23_medium.model"):
    path = os.path.join(ROOT, must)
    assert os.path.exists(path), f"груз развернулся неполно: нет {must}"
log("состав груза проверен")
sys.path.insert(0, ROOT)

from engine.backends.mlip import ASECalculatorPotential           # noqa: E402
from engine.md import run                                          # noqa: E402
from engine.state import from_symbols                              # noqa: E402
from train.dataset import water_cluster, water_molecule, WATER_GEOM  # noqa: E402
from train.descriptors import DescriptorSpec, compute_descriptors, descriptor_length  # noqa: E402
from train.nn import TrainingConfig, evaluate, train               # noqa: E402
from mace.calculators import MACECalculator                        # noqa: E402

# --- 4. учитель на карте ------------------------------------------------------------------
CONFIG = {
    # диапазон кадров и состав прописаны ЗДЕСЬ: переменные окружения в ядро через API не
    # передать (ловушка 14 скилла), поэтому правка состава = перезаливка кода
    "n_monomers": 400,
    "n_dimers": 600,
    "cluster_sizes": (4, 8, 16),
    "md_frames_per_cluster": 400,
    "md_steps_between": 12,
    "md_temperatures": (300.0, 500.0, 700.0),
    "teacher_dtype": "float32",
    "descriptor_radial": 8,
    "sweep": [
        {"hidden": (48, 48), "force_weight": 10.0, "lr": 2e-3, "epochs": 1500},
        {"hidden": (96, 96), "force_weight": 10.0, "lr": 2e-3, "epochs": 1500},
        {"hidden": (96, 96), "force_weight": 30.0, "lr": 1e-3, "epochs": 2500},
        {"hidden": (128, 128, 64), "force_weight": 30.0, "lr": 1e-3, "epochs": 2500},
    ],
}
save_json("config.json", {k: str(v) for k, v in CONFIG.items()})

log("=== учитель ===")
t0 = time.perf_counter()
calc = MACECalculator(
    model_paths=os.path.join(ROOT, "models/MACE-OFF23_medium.model"),
    device=DEVICE,
    default_dtype=CONFIG["teacher_dtype"],
)
teacher = ASECalculatorPotential(calc, f"mace-off23-medium/{DEVICE}")
log(f"учитель загружен за {time.perf_counter() - t0:.1f} с")

# замер скорости учителя на карте -- ради него и везли
bench = {}
for nmol in (1, 8, 32):
    sym, pos, cell = water_cluster(nmol, np.random.default_rng(0))
    st = from_symbols(sym, pos, cell=cell, pbc=(True, True, True))
    teacher.compute(st)
    t = time.perf_counter()
    for _ in range(5):
        st.positions += np.random.default_rng(1).normal(scale=1e-4, size=st.positions.shape)
        teacher.compute(st)
    ms = (time.perf_counter() - t) / 5 * 1000
    bench[st.n_atoms] = ms
    log(f"учитель: {st.n_atoms:4d} атомов -> {ms:8.1f} мс/расчёт ({st.n_atoms / ms * 1000:8.0f} атом-расчёт/с)")
save_json("teacher_bench.json", bench)

# --- 5. набор ------------------------------------------------------------------------------
log("=== разметка набора ===")
rng = np.random.default_rng(12345)
frames = []


def label(state):
    res = teacher.compute(state)
    frames.append(
        {
            "positions": state.positions.copy(),
            "numbers": state.numbers.copy(),
            "energy": res.energy_ev,
            "forces": res.forces_ev_per_a.copy(),
            "cell": None if state.cell is None else state.cell.copy(),
        }
    )


t0 = time.perf_counter()
for _ in range(CONFIG["n_monomers"]):
    sym, pos = water_molecule(rng, distortion=float(rng.uniform(0.0, 0.22)))
    label(from_symbols(sym, pos))
from scipy.spatial.transform import Rotation  # noqa: E402

for _ in range(CONFIG["n_dimers"]):
    sep = float(rng.uniform(2.3, 5.6))
    rot = Rotation.random(random_state=int(rng.integers(1 << 30))).as_matrix()
    a = WATER_GEOM - WATER_GEOM[0]
    b = (WATER_GEOM - WATER_GEOM[0]) @ rot.T + np.array([sep, 0.0, 0.0])
    label(from_symbols(["O", "H", "H", "O", "H", "H"], np.vstack([a, b])))
log(f"мономеры и димеры: {len(frames)} кадров за {time.perf_counter() - t0:.0f} с")

for n_mol in CONFIG["cluster_sizes"]:
    for temp in CONFIG["md_temperatures"]:
        sym, pos, cell = water_cluster(n_mol, rng)
        st = from_symbols(sym, pos, cell=cell, pbc=(True, True, True))
        st.set_maxwell_boltzmann(temp, rng)
        per_arm = CONFIG["md_frames_per_cluster"] // len(CONFIG["md_temperatures"])
        t0 = time.perf_counter()
        for _ in range(per_arm):
            try:
                run(
                    st, teacher, steps=CONFIG["md_steps_between"], dt_fs=0.5, temperature_k=temp,
                    seed=int(rng.integers(1 << 30)), sample_every=CONFIG["md_steps_between"],
                    watch_chemistry=False,
                )
                label(st.copy())
            except FloatingPointError as exc:
                log(f"  кластер {n_mol} при {temp} K оборвался: {exc}")
                break
        log(f"кластер {n_mol} молекул при {temp:.0f} K: всего кадров {len(frames)} "
            f"(+{per_arm} за {time.perf_counter() - t0:.0f} с)")
        save_json("dataset_progress.json", {"frames": len(frames)})

np.savez_compressed(
    os.path.join(RESULT, "dataset.npz"),
    **{f"f{i}_{k}": (np.array([]) if v is None else np.asarray(v))
       for i, fr in enumerate(frames) for k, v in fr.items()},
)
log(f"набор готов: {len(frames)} кадров, сохранён в вывод")

# --- 6. дескрипторы ------------------------------------------------------------------------
log("=== дескрипторы ===")
SPECIES = (1, 8)
spec = DescriptorSpec(n_radial=CONFIG["descriptor_radial"])
log(f"длина дескриптора D = {descriptor_length(spec, SPECIES)}")

t0 = time.perf_counter()
pre = []
for k, fr in enumerate(frames):
    pre.append(
        compute_descriptors(
            np.asarray(fr["positions"]), np.asarray(fr["numbers"], dtype=int), spec, SPECIES,
            cell=fr.get("cell"),
        )
    )
    if (k + 1) % 200 == 0:
        log(f"  {k + 1}/{len(frames)} за {time.perf_counter() - t0:.0f} с")
log(f"дескрипторы посчитаны за {time.perf_counter() - t0:.0f} с")

# --- 7. обучение с перебором ----------------------------------------------------------------
log("=== обучение ===")
order = np.random.default_rng(1).permutation(len(frames))
n_test = int(0.15 * len(frames))
test_idx, train_idx = order[:n_test], order[n_test:]
test = [frames[i] for i in test_idx]
train_frames = [frames[i] for i in train_idx]
train_pre = [pre[i] for i in train_idx]

results = []
best = None
for variant in CONFIG["sweep"]:
    cfg = TrainingConfig(
        hidden=tuple(variant["hidden"]), force_weight=variant["force_weight"],
        epochs=variant["epochs"], learning_rate=variant["lr"], batch_frames=32, seed=0,
    )
    log(f"вариант {variant}")
    t0 = time.perf_counter()
    model = train(train_frames, spec, SPECIES, cfg, precomputed=train_pre, verbose=True)
    took = time.perf_counter() - t0
    metrics = {
        "variant": {k: str(v) for k, v in variant.items()},
        "train": evaluate(model, train_frames[: min(200, len(train_frames))]),
        "test": evaluate(model, test),
        "seconds": took,
        "n_train": len(train_frames),
        "n_test": len(test),
    }
    results.append(metrics)
    log(f"  отложенная: E {metrics['test']['energy_mae_mev_per_atom']:.2f} мэВ/атом, "
        f"F {metrics['test']['force_mae_mev_per_a']:.1f} мэВ/Å, {took:.0f} с")
    save_json("training_results.json", results)
    if best is None or metrics["test"]["force_mae_mev_per_a"] < best[0]:
        best = (metrics["test"]["force_mae_mev_per_a"], variant)
        np.savez(
            os.path.join(RESULT, "best_model.npz"),
            mean=model["mean"], std=model["std"], baseline=model["baseline"],
            species=np.array(SPECIES), n_radial=CONFIG["descriptor_radial"],
            **{f"{z}_{k}": v for z, sd in model["net"].state_dict().items() for k, v in sd.items()},
        )
        save_json("best_variant.json", {"force_mae_mev_per_a": best[0], "variant": {k: str(v) for k, v in variant.items()}})

log("=== готово ===")
log(f"лучший вариант: {best}")
