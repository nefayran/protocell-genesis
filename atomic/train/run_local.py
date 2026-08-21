"""
Полный прогон обучения на этой машине.

Почему здесь, а не в облаке -- по замеру, а не по удобству: учитель на нашем процессоре даёт
460 атом-расчётов в секунду против 103 на процессоре Kaggle и 22-92 на облачной Tesla P100.
Для систем из 3-96 атомов накладные расходы больше самого счёта, поэтому облачная карта
простаивает, а слабый облачный процессор просто медленнее. Плюс здесь 48 ГБ памяти против
облачного предела, на котором прошлый прогон и был убит: производные дескрипторов для 7000
кадров занимают 14.5 ГБ в двойной точности и 7.26 ГБ в одинарной.

Хранение переведено в ОДИНАРНУЮ точность умышленно: это не потеря физики, потому что обучение
и так идёт в одинарной, а проверка дескрипторов на совпадение с прямой реализацией (1e-14)
делается отдельно и в двойной.
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
    # РАЗРЕШЕНИЕ БАЗИСА проверено отдельным опытом на одних и тех же данных (models/
    # resolution_test.json): удвоение радиальных функций при радиусе 5 Å дало силы 109.1 -> 71.5
    # мэВ/Å, а увеличение радиуса до 6 Å их УХУДШИЛО (71.5 -> 90.6) -- больше окружения при той
    # же ёмкости и тех же данных размазывает модель. Поэтому радиус остаётся 5 Å, а перебор идёт
    # по числу радиальных функций; стоимость дескрипторов при этом почти не растёт (27-31 с).
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
    log("=== конфигурации ===")
    t0 = time.perf_counter()
    states = build_sampled_states(rng, CONFIG["n_monomers"], CONFIG["n_dimers"], CONFIG["n_clusters"])
    log(f"{len(states)} конфигураций за {time.perf_counter() - t0:.1f} с")

    log("=== разметка ===")
    teacher = BatchedMACE(MODEL, device="cpu", dtype="float32")
    frames, t0 = [], time.perf_counter()
    for start in range(0, len(states), 500):
        frames += teacher.label(states[start : start + 500], batch_size=CONFIG["label_batch"])
        done = time.perf_counter() - t0
        log(f"  {len(frames)}/{len(states)} за {done:.0f} с ({len(frames) / done:.1f} кадр/с)")
        save("progress.json", {"labelled": len(frames), "seconds": done})

    # вторая линия защиты: геометрия проверена при выборке, но силу знает только учитель
    kept, dropped = filter_outliers(frames)
    log(f"отбраковка по силе: оставлено {len(kept)}, выброшено {len(dropped)} "
        f"({100 * len(dropped) / max(len(frames), 1):.2f}%)")
    save("filtering.json", {"kept": len(kept), "dropped": len(dropped),
                            "worst_kept_force": max((float(abs(f["forces"]).max()) for f in kept), default=0.0)})
    frames = kept

    log("=== отложенная выборка из динамики ===")
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
            log(f"  динамика оборвалась: {exc}")
            break
    md_frames = teacher.label(md_states, batch_size=CONFIG["label_batch"]) if md_states else []
    log(f"кадров из динамики: {len(md_frames)}")

    log("=== дескрипторы считаются внутри перебора: у каждого варианта свой базис ===")

    def descriptors(batch, label, spec):
        t = time.perf_counter()
        out = []
        for k, fr in enumerate(batch):
            g, dg = compute_descriptors_fast(np.asarray(fr["positions"]),
                                             np.asarray(fr["numbers"], dtype=int),
                                             spec, SPECIES, cell=fr.get("cell"))
            # хранение в одинарной точности: 7.26 ГБ вместо 14.5 на 7000 кадров
            out.append((g.astype(np.float32), dg.astype(np.float32)))
            if (k + 1) % 1000 == 0:
                log(f"  {label}: {k + 1}/{len(batch)} за {time.perf_counter() - t:.0f} с")
        log(f"  {label}: готово за {time.perf_counter() - t:.0f} с")
        return out

    order = np.random.default_rng(1).permutation(len(frames))
    n_test = int(0.1 * len(frames))
    test_idx, train_idx = order[:n_test], order[n_test:]
    test = [frames[i] for i in test_idx]
    train_frames = [frames[i] for i in train_idx]
    log(f"обучающих {len(train_frames)}, случайных отложенных {len(test)}, из динамики {len(md_frames)}")

    log("=== обучение ===")
    results, best = [], None
    for variant in CONFIG["sweep"]:
        spec = DescriptorSpec(n_radial=variant["radial"], cutoff=5.0)
        log(f"базис: {variant['radial']} радиальных, радиус 5.0 Å, D = {descriptor_length(spec, SPECIES)}")
        pre_all = descriptors(frames, f"набор/D{descriptor_length(spec, SPECIES)}", spec)
        pre_md = descriptors(md_frames, "динамика", spec) if md_frames else []
        pre_test = [pre_all[i] for i in test_idx]
        pre_train = [pre_all[i] for i in train_idx]
        cfg = FastTrainingConfig(hidden=tuple(variant["hidden"]), force_weight=variant["force_weight"],
                                 epochs=variant["epochs"], learning_rate=variant["lr"], seed=0,
                                 memory_budget_mb=256.0, dtype="float32", lr_final_fraction=0.05)
        log(f"вариант {variant}")
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
        log(f"  случайные: E {metrics['test_random']['energy_mae_mev_per_atom']:.2f} мэВ/атом, "
            f"F {metrics['test_random']['force_mae_mev_per_a']:.1f} мэВ/Å")
        if metrics["test_md"]:
            log(f"  динамика:  E {metrics['test_md']['energy_mae_mev_per_atom']:.2f} мэВ/атом, "
                f"F {metrics['test_md']['force_mae_mev_per_a']:.1f} мэВ/Å  ({took:.0f} с)")
        save("training_results.json", results)
        score = (metrics["test_md"] or metrics["test_random"])["force_mae_mev_per_a"]
        if best is None or score < best[0]:
            best = (score, variant)
            np.savez(os.path.join(OUT, "best_model.npz"),
                     mean=model["mean"], std=model["std"], baseline=model["baseline"],
                     species=np.array(SPECIES), n_radial=variant["radial"], cutoff=5.0,
                     **{f"{z}_{k}": v for z, sd in model["net"].state_dict().items() for k, v in sd.items()})
            save("best_variant.json", {"force_mae_mev_per_a": best[0], "variant": variant})
    log(f"=== готово. лучший: {best}")


if __name__ == "__main__":
    main()
