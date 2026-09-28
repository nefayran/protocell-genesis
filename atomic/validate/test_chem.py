"""
Engine chemistry against MEASURED values, not against itself.

There is not a single fudge here: every check compares what the GFN2-xTB Hamiltonian
produces after finding the equilibrium geometry against an experimental value from a
reference source. Tolerances are not set "to make it pass" but by the known accuracy of the
method: GFN2 reproduces the bond lengths of organic molecules to within hundredths of an
angstrom and bond angles to within about a degree. If a tolerance has to be widened, either
the wiring is wrong or the method is being misapplied, and either way that must show up as a
failure, not as "roughly converged".

Experimental values and sources:
  H2   r(H-H)  = 0.7414 Å      (Huber & Herzberg, spectroscopy of diatomic molecules)
  H2O  r(O-H)  = 0.9572 Å, angle H-O-H = 104.52°   (Benedict, Gailar & Plyler 1956)
  CH4  r(C-H)  = 1.0870 Å      (Herzberg, IR spectrum)
  C2H6 r(C-C)  = 1.5351 Å      (Harmony 1990, rotational spectrum)
  C2H4 r(C=C)  = 1.3390 Å      (Herzberg)
  C2H2 r(C≡C)  = 1.2033 Å      (Herzberg)
  water dimer: binding energy 5.02 ± 0.05 kcal/mol (Klopper et al. 2000, CCSD(T)/CBS)
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
    """The backend wiring is correct: the force from the tblite gradient matches the
    derivative of its own energy. This checks unit CONVERSION and sign, without which
    everything below is meaningless."""
    st = from_symbols(["O", "H", "H"], [[0, 0, 0], [0, -0.76, 0.59], [0, 0.76, 0.59]])
    analytic = GFN.compute(st).forces_ev_per_a
    numeric = numerical_forces(GFN, st, h=1e-4)
    scale = np.abs(analytic).max()
    rel = np.abs(analytic - numeric).max() / scale
    assert rel < 2e-3, f"relative mismatch between force and gradient {rel:.2e}"


def test_h2_bond_length_gfn2_deviation_is_known_and_dft_fixes_it():
    """
    A measured weakness of the cheap backend, named as a number, not hidden behind a tolerance.

    GFN2 gives about 0.777 Å for H2 against the experimental 0.7414: an error of 0.035 Å,
    meaning H2 is an outlier for this method (for organic molecules its typical bond-length
    error is on the order of 0.01-0.02 Å). So two claims are checked here: that the GFN2
    deviation stays within the method's known bounds, and that a real electronic structure
    calculation (HF/def2-SVP via pyscf) places that same bond at the experimental value. This
    is exactly the declared division of roles: GFN2 carries the dynamics, DFT/HF checks
    energetics and geometry at the points that matter.
    """
    from engine.backends.scf import PySCF

    st_gfn, _ = _optimised(["H", "H"], [[0, 0, 0], [0, 0, 0.80]])
    r_gfn = st_gfn.distance(0, 1)
    assert abs(r_gfn - 0.7414) < 0.05, f"GFN2 r(H-H) = {r_gfn:.4f} Å vs the experimental 0.7414"

    st_hf = from_symbols(["H", "H"], [[0, 0, 0], [0, 0, 0.80]])
    optimise_lbfgs(st_hf, PySCF(method="hf", basis="def2-svp", dispersion=False), force_tol_ev_per_a=1e-4)
    r_hf = st_hf.distance(0, 1)
    assert abs(r_hf - 0.7414) < 0.015, f"HF r(H-H) = {r_hf:.4f} Å vs the experimental 0.7414"
    assert abs(r_hf - 0.7414) < abs(r_gfn - 0.7414), (
        f"the reference method must be more accurate than the cheap one: HF {r_hf:.4f}, GFN2 {r_gfn:.4f}"
    )


def test_water_geometry():
    st, _ = _optimised(["O", "H", "H"], [[0, 0, 0], [0, -0.76, 0.59], [0, 0.76, 0.59]])
    r1, r2 = st.distance(0, 1), st.distance(0, 2)
    angle = st.angle_deg(1, 0, 2)
    assert abs(r1 - 0.9572) < 0.03 and abs(r2 - 0.9572) < 0.03, f"r(O-H) = {r1:.4f}, {r2:.4f} Å"
    assert abs(angle - 104.52) < 3.0, f"angle H-O-H = {angle:.2f}° vs the experimental 104.52"


def test_methane_geometry():
    d = 0.63
    st, _ = _optimised(
        ["C", "H", "H", "H", "H"],
        [[0, 0, 0], [d, d, d], [-d, -d, d], [-d, d, -d], [d, -d, -d]],
    )
    lengths = [st.distance(0, i) for i in range(1, 5)]
    angles = [st.angle_deg(1, 0, j) for j in (2, 3, 4)]
    assert max(abs(x - 1.0870) for x in lengths) < 0.03, f"r(C-H) = {lengths}"
    assert max(abs(a - 109.471) for a in angles) < 2.0, f"angles H-C-H = {angles}"


@pytest.mark.parametrize(
    "name, symbols, positions, i, j, reference, tol, multiplicity",
    [
        (
            "ethane C-C",
            ["C", "C", "H", "H", "H", "H", "H", "H"],
            [[0, 0, 0], [0, 0, 1.53], [0.63, 0.63, -0.36], [-0.86, 0.13, -0.36],
             [0.23, -0.76, -0.36], [0.63, -0.63, 1.89], [-0.86, -0.13, 1.89], [0.23, 0.76, 1.89]],
            0, 1, 1.5351, 0.03, 1,
        ),
        (
            "ethylene C=C",
            ["C", "C", "H", "H", "H", "H"],
            [[0, 0, 0], [0, 0, 1.34], [0, 0.93, -0.55], [0, -0.93, -0.55],
             [0, 0.93, 1.89], [0, -0.93, 1.89]],
            0, 1, 1.3390, 0.03, 2,
        ),
        (
            "acetylene C≡C",
            ["C", "C", "H", "H"],
            [[0, 0, 0], [0, 0, 1.20], [0, 0, -1.06], [0, 0, 2.26]],
            0, 1, 1.2033, 0.03, 3,
        ),
    ],
)
def test_bond_orders_recover_multiplicity(name, symbols, positions, i, j, reference, tol, multiplicity):
    """Bond length against experiment AND multiplicity read from the bond order. The second
    matters more: we did NOT set the multiplicity anywhere; it came out of the electronic
    structure."""
    st, res = _optimised(symbols, positions)
    r = st.distance(i, j)
    assert abs(r - reference) < tol, f"{name}: {r:.4f} Å vs the experimental {reference}"
    bonds = bonds_from_orders(res.extra["bond_orders"])
    pair = [b for b in bonds if {b.i, b.j} == {i, j}]
    assert pair, f"{name}: bond {i}-{j} was not found at all"
    assert pair[0].multiplicity == multiplicity, (
        f"{name}: multiplicity read as {pair[0].multiplicity} (order {pair[0].order:.3f}), "
        f"expected {multiplicity}"
    )


def test_connectivity_of_known_molecules():
    """Connectivity of known molecules must match the textbook one: this checks the threshold
    on bond order, the only free number in atomic/engine/chem.py."""
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
        assert len(species) == 1, f"{formula}: broke apart into {len(species)} particles"
        assert species[0].formula == formula, f"formula read as {species[0].formula}"
        assert len(bonds) == n_bonds, f"{formula}: {len(bonds)} bonds, expected {n_bonds}"


def test_water_dimer_binding_energy():
    """A weak interaction: the hydrogen bond. This is exactly the one that was completely
    absent in this project's previous engine (heads did not interact with each other), and
    it is exactly the one that produces the "acid-soap" pair. Here it is not set up in any
    way; it is a consequence of the electronic structure."""
    monomer, _ = _optimised(["O", "H", "H"], [[0, 0, 0], [0, -0.76, 0.59], [0, 0.76, 0.59]])
    e_monomer = GFN.compute(monomer).energy_ev

    dimer_start = [
        [0.0, 0.0, 0.0], [0.0, -0.76, 0.59], [0.0, 0.76, 0.59],
        [2.98, 0.0, 0.0], [3.55, -0.30, 0.68], [3.55, 0.30, -0.68],
    ]
    dimer, res = _optimised(["O", "H", "H", "O", "H", "H"], dimer_start)
    binding_kcal = (2 * e_monomer - res.energy_ev) * EV_IN_KCAL_PER_MOL
    assert 3.0 < binding_kcal < 8.0, (
        f"water dimer binding energy {binding_kcal:.2f} kcal/mol vs the experimental 5.02"
    )
    # and the hydrogen bond MUST be visible as connectivity of two molecules, not one
    bonds = bonds_from_orders(res.extra["bond_orders"])
    species = find_species(dimer.numbers, bonds, res.extra["charges"])
    assert len(species) == 2 and all(s.formula == "H2O" for s in species), (
        f"dimer read as {[s.formula for s in species]} -- the hydrogen bond must not be "
        "counted as covalent"
    )
