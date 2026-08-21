"""
Химия движка против ИЗМЕРЕННЫХ величин, а не против самой себя.

Здесь нет ни одной подгонки: каждая проверка сравнивает то, что выдал гамильтониан
GFN2-xTB после поиска равновесной геометрии, с опытной величиной из справочника. Допуски
взяты не «чтобы прошло», а по известной точности метода: GFN2 воспроизводит длины связей
органики с точностью порядка сотых долей ангстрема и валентные углы порядка градуса. Если
допуск приходится расширять, значит либо неверна обвязка, либо метод применён не по
назначению -- и то и другое должно быть видно как провал, а не как «примерно сошлось».

Опытные значения и источники:
  H2   r(H-H)  = 0.7414 Å      (Huber & Herzberg, спектроскопия двухатомных молекул)
  H2O  r(O-H)  = 0.9572 Å, угол H-O-H = 104.52°   (Benedict, Gailar & Plyler 1956)
  CH4  r(C-H)  = 1.0870 Å      (Herzberg, ИК-спектр)
  C2H6 r(C-C)  = 1.5351 Å      (Harmony 1990, вращательный спектр)
  C2H4 r(C=C)  = 1.3390 Å      (Herzberg)
  C2H2 r(C≡C)  = 1.2033 Å      (Herzberg)
  димер воды: энергия связывания 5.02 ± 0.05 ккал/моль (Klopper et al. 2000, CCSD(T)/CBS)
"""
from __future__ import annotations

import numpy as np
import pytest

from engine.backends.gfn2 import GFN2
from engine.chem import bonds_from_orders, find_species
from engine.integrate import optimise_lbfgs
from engine.potential import numerical_forces
from engine.state import from_symbols
from engine.units import EV_IN_KCAL_PER_MOL

GFN = GFN2()


def _optimised(symbols, positions):
    st = from_symbols(symbols, positions)
    res, _ = optimise_lbfgs(st, GFN, force_tol_ev_per_a=1e-4)
    return st, res


def test_gfn2_force_matches_numerical_gradient():
    """Обвязка бэкенда верна: сила из градиента tblite совпадает с производной его же
    энергии. Это проверка ПЕРЕВОДА единиц и знака, без которой всё ниже бессмысленно."""
    st = from_symbols(["O", "H", "H"], [[0, 0, 0], [0, -0.76, 0.59], [0, 0.76, 0.59]])
    analytic = GFN.compute(st).forces_ev_per_a
    numeric = numerical_forces(GFN, st, h=1e-4)
    scale = np.abs(analytic).max()
    rel = np.abs(analytic - numeric).max() / scale
    assert rel < 2e-3, f"относительное расхождение силы и градиента {rel:.2e}"


def test_h2_bond_length_gfn2_deviation_is_known_and_dft_fixes_it():
    """
    Измеренная слабость дешёвого бэкенда, названная числом, а не спрятанная допуском.

    GFN2 даёт для H2 около 0.777 Å против опытных 0.7414 -- ошибка 0.035 Å, то есть H2 для
    этого метода выброс (по органике его типичная ошибка длин связей порядка 0.01-0.02 Å).
    Поэтому здесь проверяются ДВА утверждения: что отклонение GFN2 остаётся в известных для
    метода пределах, и что настоящая электронная структура (HF/def2-SVP через pyscf) кладёт
    ту же связь на опытное значение. Это и есть заявленное разделение ролей: GFN2 везёт
    динамику, DFT/HF проверяет энергетику и геометрию в характерных точках.
    """
    from engine.backends.scf import PySCF

    st_gfn, _ = _optimised(["H", "H"], [[0, 0, 0], [0, 0, 0.80]])
    r_gfn = st_gfn.distance(0, 1)
    assert abs(r_gfn - 0.7414) < 0.05, f"GFN2 r(H-H) = {r_gfn:.4f} Å против опытных 0.7414"

    st_hf = from_symbols(["H", "H"], [[0, 0, 0], [0, 0, 0.80]])
    optimise_lbfgs(st_hf, PySCF(method="hf", basis="def2-svp", dispersion=False), force_tol_ev_per_a=1e-4)
    r_hf = st_hf.distance(0, 1)
    assert abs(r_hf - 0.7414) < 0.015, f"HF r(H-H) = {r_hf:.4f} Å против опытных 0.7414"
    assert abs(r_hf - 0.7414) < abs(r_gfn - 0.7414), (
        f"опорный метод обязан быть точнее дешёвого: HF {r_hf:.4f}, GFN2 {r_gfn:.4f}"
    )


def test_water_geometry():
    st, _ = _optimised(["O", "H", "H"], [[0, 0, 0], [0, -0.76, 0.59], [0, 0.76, 0.59]])
    r1, r2 = st.distance(0, 1), st.distance(0, 2)
    angle = st.angle_deg(1, 0, 2)
    assert abs(r1 - 0.9572) < 0.03 and abs(r2 - 0.9572) < 0.03, f"r(O-H) = {r1:.4f}, {r2:.4f} Å"
    assert abs(angle - 104.52) < 3.0, f"угол H-O-H = {angle:.2f}° против опытных 104.52"


def test_methane_geometry():
    d = 0.63
    st, _ = _optimised(
        ["C", "H", "H", "H", "H"],
        [[0, 0, 0], [d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]],
    )
    lengths = [st.distance(0, i) for i in range(1, 5)]
    angles = [st.angle_deg(1, 0, j) for j in (2, 3, 4)]
    assert max(abs(x - 1.0870) for x in lengths) < 0.03, f"r(C-H) = {lengths}"
    assert max(abs(a - 109.471) for a in angles) < 2.0, f"углы H-C-H = {angles}"


@pytest.mark.parametrize(
    "name, symbols, positions, i, j, reference, tol, multiplicity",
    [
        (
            "этан C-C",
            ["C", "C", "H", "H", "H", "H", "H", "H"],
            [[0, 0, 0], [0, 0, 1.53], [0.63, 0.63, -0.36], [-0.86, 0.13, -0.36],
             [0.23, -0.76, -0.36], [0.63, -0.63, 1.89], [-0.86, -0.13, 1.89], [0.23, 0.76, 1.89]],
            0, 1, 1.5351, 0.03, 1,
        ),
        (
            "этен C=C",
            ["C", "C", "H", "H", "H", "H"],
            [[0, 0, 0], [0, 0, 1.34], [0, 0.93, -0.55], [0, -0.93, -0.55],
             [0, 0.93, 1.89], [0, -0.93, 1.89]],
            0, 1, 1.3390, 0.03, 2,
        ),
        (
            "этин C≡C",
            ["C", "C", "H", "H"],
            [[0, 0, 0], [0, 0, 1.20], [0, 0, -1.06], [0, 0, 2.26]],
            0, 1, 1.2033, 0.03, 3,
        ),
    ],
)
def test_bond_orders_recover_multiplicity(name, symbols, positions, i, j, reference, tol, multiplicity):
    """Длина связи против опыта И кратность, прочитанная из порядка связи. Второе важнее:
    кратность мы НЕ задавали нигде -- она вышла из электронной структуры."""
    st, res = _optimised(symbols, positions)
    r = st.distance(i, j)
    assert abs(r - reference) < tol, f"{name}: {r:.4f} Å против опытных {reference}"
    bonds = bonds_from_orders(res.extra["bond_orders"])
    pair = [b for b in bonds if {b.i, b.j} == {i, j}]
    assert pair, f"{name}: связь {i}-{j} не обнаружена вовсе"
    assert pair[0].multiplicity == multiplicity, (
        f"{name}: кратность прочитана как {pair[0].multiplicity} (порядок {pair[0].order:.3f}), "
        f"ожидалась {multiplicity}"
    )


def test_connectivity_of_known_molecules():
    """Связность известных молекул обязана совпасть с учебной -- это проверка порога по
    порядку связи, единственного свободного числа в atomic/engine/chem.py."""
    cases = [
        (["O", "H", "H"], [[0, 0, 0], [0, -0.76, 0.59], [0, 0.76, 0.59]], "H2O", 2),
        (["H", "O", "O", "H"],
         [[0.84, 0.60, 0.36], [0.0, 0.73, -0.05], [0.0, -0.73, -0.05], [-0.84, -0.60, 0.36]],
         "H2O2", 3),
    ]
    for symbols, positions, formula, n_bonds in cases:
        st, res = _optimised(symbols, positions)
        bonds = bonds_from_orders(res.extra["bond_orders"])
        species = find_species(st.numbers, bonds, res.extra["charges"])
        assert len(species) == 1, f"{formula}: распалось на {len(species)} частиц"
        assert species[0].formula == formula, f"формула прочитана как {species[0].formula}"
        assert len(bonds) == n_bonds, f"{formula}: связей {len(bonds)}, ожидалось {n_bonds}"


def test_water_dimer_binding_energy():
    """Слабое взаимодействие: водородная связь. Именно она в предыдущем движке этого проекта
    отсутствовала полностью (голова с головой не взаимодействовала), и именно она даёт пару
    «кислота-мыло». Здесь она никак не задана -- она следствие электронной структуры."""
    monomer, _ = _optimised(["O", "H", "H"], [[0, 0, 0], [0, -0.76, 0.59], [0, 0.76, 0.59]])
    e_monomer = GFN.compute(monomer).energy_ev

    dimer_start = [
        [0.0, 0.0, 0.0], [0.0, -0.76, 0.59], [0.0, 0.76, 0.59],
        [2.98, 0.0, 0.0], [3.55, -0.30, 0.68], [3.55, 0.30, -0.68],
    ]
    dimer, res = _optimised(["O", "H", "H", "O", "H", "H"], dimer_start)
    binding_kcal = (2 * e_monomer - res.energy_ev) * EV_IN_KCAL_PER_MOL
    assert 3.0 < binding_kcal < 8.0, (
        f"энергия связывания димера воды {binding_kcal:.2f} ккал/моль против опытных 5.02"
    )
    # и водородная связь ОБЯЗАНА быть видна как связность из двух молекул, а не одной
    bonds = bonds_from_orders(res.extra["bond_orders"])
    species = find_species(dimer.numbers, bonds, res.extra["charges"])
    assert len(species) == 2 and all(s.formula == "H2O" for s in species), (
        f"димер прочитан как {[s.formula for s in species]} -- водородная связь не должна "
        "считаться ковалентной"
    )
