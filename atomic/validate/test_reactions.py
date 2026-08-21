"""
Реакции, которые движок ОБНАРУЖИВАЕТ, а не которые ему заданы, и цена дешёвого бэкенда,
названная числом.

В этих проверках нет ни одного правила реакции, ни одной ставки, ни одного порога энергии
реакции. Есть гамильтониан, температура и время; что произойдёт -- решает электронная
структура, а наше дело прочитать связность.

Три проверки намеренно разного смысла:
  1. При комнатной температуре молекула ОБЯЗАНА остаться целой -- проверка на отсутствие
     ложных реакций (движок, где связи рвутся сами, бесполезен).
  2. Перенос протона в ионе Цунделя ОБЯЗАН происходить -- проверка на способность к реакции
     с переносом заряда (движок, где не происходит ничего, бесполезен ровно так же).
  3. Разрыв связи при растяжении ОБЯЗАН читаться как исчезновение связи, а энергия разрыва --
     совпадать с опытной. Здесь же измерена и записана ГРАНИЦА ПРИМЕНИМОСТИ дешёвого метода.

Про редкие события сказано прямо: прямая динамика при 300 K не преодолевает барьер в
несколько электронвольт ни за какое доступное время, и это правильно физически. Такие
события требуют либо высокой температуры, либо методов ускоренной выборки (NEB,
метадинамика) -- их в движке пока НЕТ, и это ограничение, а не недосмотр.

Опытные значения:
  D0(H2O2 -> 2 OH) = 2.15 эВ = 207 кДж/моль (Ruscic, Active Thermochemical Tables)
  r(O-O) в H2O2   = 1.4556 Å, r(O-H) в OH = 0.9697 Å (Herzberg; Huber & Herzberg)
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.backends.gfn2 import GFN2
from engine.backends.scf import PySCF
from engine.chem import bonds_from_orders, find_species
from engine.integrate import optimise_lbfgs
from engine.md import run
from engine.state import from_symbols

GFN = GFN2()
H2O2_GEOM = [[0.84, 0.60, 0.36], [0.0, 0.73, -0.05], [0.0, -0.73, -0.05], [-0.84, -0.60, 0.36]]


def test_water_stays_intact_at_room_temperature():
    """Ложных реакций нет: 1000 шагов по 0.5 фс при 300 K, связность неизменна."""
    st = from_symbols(["O", "H", "H"], [[0, 0, 0], [0, -0.96, 0.30], [0, 0.96, 0.30]])
    res = run(st, GFN, steps=1000, dt_fs=0.5, temperature_k=300.0, seed=1, sample_every=25)
    assert res.events == [], f"при 300 K зафиксированы реакции, которых быть не должно: {res.events}"
    assert all(f.species == ("H2O",) for f in res.frames), (
        f"состав менялся: {sorted({f.species for f in res.frames})}"
    )


def test_proton_transfer_in_protonated_water_dimer():
    """
    H5O2+ (ион Цунделя): мостиковый протон переходит между кислородами практически без
    барьера. Это настоящая реакция с ПЕРЕНОСОМ ЗАРЯДА, и в предыдущем движке этого проекта
    она была невозможна в принципе -- там протонирование было ходом Монте-Карло по внешнему
    параметру pH, а не следствием электронной структуры.
    """
    st = from_symbols(
        ["O", "H", "H", "H", "O", "H", "H"],
        [
            [0.00, 0.00, 0.00], [-0.55, 0.78, 0.00], [-0.55, -0.78, 0.00],
            [1.05, 0.00, 0.00],                      # мостиковый протон
            [2.45, 0.00, 0.00], [3.00, 0.78, 0.00], [3.00, -0.78, 0.00],
        ],
        charge=1,
    )
    res = run(st, GFN, steps=1500, dt_fs=0.25, temperature_k=350.0, seed=3, sample_every=10)

    d_left = np.array([np.linalg.norm(f.positions[3] - f.positions[0]) for f in res.frames])
    d_right = np.array([np.linalg.norm(f.positions[3] - f.positions[4]) for f in res.frames])
    assert (d_left < d_right).any() and (d_right < d_left).any(), (
        f"мостиковый протон не переходил: d(O1-H) {d_left.min():.2f}-{d_left.max():.2f}, "
        f"d(O2-H) {d_right.min():.2f}-{d_right.max():.2f} Å"
    )
    final = GFN.compute(st)
    assert abs(float(np.asarray(final.extra["charges"]).sum()) - 1.0) < 1e-6


def test_stretching_breaks_the_bond_in_the_connectivity():
    """
    Разрыв связи виден как ИСЧЕЗНОВЕНИЕ связи из связности, а не как большое расстояние, и
    система читается как две частицы OH. Спин при этом обязателен: гомолитический разрыв даёт
    два радикала, то есть открытую оболочку, и без неё расчёт либо не сходится, либо даёт
    неверную энергию -- это измеренное свойство, см. следующий тест.
    """
    st = from_symbols(["H", "O", "O", "H"], H2O2_GEOM, spin_multiplicity=3)
    axis = np.array([0.0, -1.0, 0.0])
    st.positions[2] += axis * 2.5
    st.positions[3] += axis * 2.5
    res = GFN.compute(st)
    bonds = bonds_from_orders(res.extra["bond_orders"])
    assert not [b for b in bonds if {b.i, b.j} == {1, 2}], "связь O-O при 3.9 Å всё ещё считается"
    species = find_species(st.numbers, bonds, res.extra["charges"])
    assert len(species) == 2 and all(s.formula == "HO" for s in species), (
        f"разорванная система прочитана как {[s.formula for s in species]}"
    )


def test_dissociation_energy_dft_matches_experiment_and_gfn2_does_not():
    """
    Измеренная граница применимости, названная числом, а не спрятанная.

    D(O-O) в H2O2 -> 2 OH: опыт 2.15 эВ. Наш опорный бэкенд (B3LYP/def2-SVP, НЕограниченный
    для радикала) даёт около 2.5 эВ -- это согласие в пределах ожидаемого для такого базиса и
    без поправки на нулевые колебания. Дешёвый GFN2 даёт около 5.4 эВ, то есть завышает более
    чем вдвое: полуэмпирический сильносвязанный метод без спиновой поляризации плохо описывает
    радикалы. Отсюда рабочее правило движка: GFN2 везёт динамику, но энергетику разрывов и
    радикалов считает опорный метод.

    Тест закрепляет ОБА утверждения. Если новая версия tblite починит радикалы, этот тест
    упадёт -- и это правильно: правило разделения ролей надо будет пересмотреть по числам.
    """
    dft = PySCF(method="b3lyp", basis="def2-svp", dispersion=False)

    def dissociation(potential) -> float:
        h2o2 = from_symbols(["H", "O", "O", "H"], H2O2_GEOM)
        r_mol, _ = optimise_lbfgs(h2o2, potential, force_tol_ev_per_a=2e-3)
        oh = from_symbols(["O", "H"], [[0, 0, 0], [0, 0, 0.97]], spin_multiplicity=2)
        r_rad, _ = optimise_lbfgs(oh, potential, force_tol_ev_per_a=2e-3)
        return 2 * r_rad.energy_ev - r_mol.energy_ev

    d_dft = dissociation(dft)
    d_gfn = dissociation(GFN)
    assert abs(d_dft - 2.15) / 2.15 < 0.25, (
        f"опорный метод даёт D(O-O) = {d_dft:.3f} эВ против опытных 2.15"
    )
    assert d_gfn > 1.8 * d_dft, (
        f"известное завышение GFN2 исчезло: GFN2 {d_gfn:.3f} эВ, опорный {d_dft:.3f} эВ -- "
        "правило разделения ролей надо пересмотреть по числам"
    )


@pytest.mark.slow
def test_rare_event_needs_more_than_patience():
    """
    Прямая динамика НЕ преодолевает высокий барьер, и это записано как измеренный факт, а не
    как неудача: 2000 шагов при 6000 K на H2O2 не дают ни одного разрыва (замерено), потому
    что при частоте попыток около одной на 27 фс и больцмановском множителе для 2.2 эВ
    ожидаемое число событий меньше единицы. Ускоренная выборка в движке пока не реализована.
    """
    st = from_symbols(["H", "O", "O", "H"], H2O2_GEOM)
    st.set_maxwell_boltzmann(6000.0, np.random.default_rng(4))
    res = run(st, GFN, steps=2000, dt_fs=0.25, temperature_k=6000.0,
              friction_per_fs=0.002, seed=4, sample_every=10)
    assert res.events == [], (
        f"при 6000 K за 500 фс зафиксировано {len(res.events)} событий -- оценка редкости "
        "события неверна и её надо пересчитать"
    )
