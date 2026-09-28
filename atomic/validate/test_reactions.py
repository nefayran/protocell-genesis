"""
Reactions that the engine DETECTS, rather than ones it is given, and the price of the cheap
backend, named as a number.

There is not a single reaction rule in these checks, not a single wager, not a single
reaction energy threshold. There is a Hamiltonian, a temperature, and time; what happens is
decided by the electronic structure, and our job is only to read out the connectivity.

Three checks, deliberately of different meaning:
  1. At room temperature the molecule MUST stay intact: a check for the absence of spurious
     reactions (an engine where bonds break on their own is useless).
  2. Proton transfer in the Zundel ion MUST happen: a check for the ability to undergo a
     charge-transfer reaction (an engine where nothing happens at all is equally useless).
  3. Bond breaking under stretching MUST be read as the disappearance of a bond, and the
     dissociation energy must match the experimental one. This is also where the APPLICABILITY
     LIMIT of the cheap method is measured and recorded.

About rare events, stated plainly: direct dynamics at 300 K does not overcome a barrier of a
few electronvolts in any available time, and that is physically correct. Such events require
either high temperature or enhanced-sampling methods (NEB, metadynamics); those are NOT in
the engine yet, and that is a limitation, not an oversight.

Experimental values:
  D0(H2O2 -> 2 OH) = 2.15 eV = 207 kJ/mol (Ruscic, Active Thermochemical Tables)
  r(O-O) in H2O2   = 1.4556 Å, r(O-H) in OH = 0.9697 Å (Herzberg; Huber & Herzberg)
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
    """No spurious reactions: 1000 steps of 0.5 fs at 300 K, connectivity unchanged."""
    st = from_symbols(["O", "H", "H"], [[0, 0, 0], [0, -0.96, 0.30], [0, 0.96, 0.30]])
    res = run(st, GFN, steps=1000, dt_fs=0.5, temperature_k=300.0, seed=1, sample_every=25)
    assert res.events == [], f"reactions were recorded at 300 K that should not have happened: {res.events}"
    assert all(f.species == ("H2O",) for f in res.frames), (
        f"composition changed: {sorted({f.species for f in res.frames})}"
    )


def test_proton_transfer_in_protonated_water_dimer():
    """
    H5O2+ (the Zundel ion): the bridging proton hops between the two oxygens with
    essentially no barrier. This is a real reaction with CHARGE TRANSFER, and in this
    project's previous engine it was impossible in principle: protonation there was a Monte
    Carlo move driven by an external pH parameter, not a consequence of the electronic
    structure.
    """
    st = from_symbols(
        ["O", "H", "H", "H", "O", "H", "H"],
        [
            [0.00, 0.00, 0.00], [-0.55, 0.78, 0.00], [-0.55, -0.78, 0.00],
            [1.05, 0.00, 0.00],                      # bridging proton
            [2.45, 0.00, 0.00], [3.00, 0.78, 0.00], [3.00, -0.78, 0.00],
        ],
        charge=1,
    )
    res = run(st, GFN, steps=1500, dt_fs=0.25, temperature_k=350.0, seed=3, sample_every=10)

    d_left = np.array([np.linalg.norm(f.positions[3] - f.positions[0]) for f in res.frames])
    d_right = np.array([np.linalg.norm(f.positions[3] - f.positions[4]) for f in res.frames])
    assert (d_left < d_right).any() and (d_right < d_left).any(), (
        f"the bridging proton did not transfer: d(O1-H) {d_left.min():.2f}-{d_left.max():.2f}, "
        f"d(O2-H) {d_right.min():.2f}-{d_right.max():.2f} Å"
    )
    final = GFN.compute(st)
    assert abs(float(np.asarray(final.extra["charges"]).sum()) - 1.0) < 1e-6


def test_stretching_breaks_the_bond_in_the_connectivity():
    """
    Bond breaking is seen as the DISAPPEARANCE of a bond from the connectivity, not as a
    large distance, and the system is read as two OH particles. Spin is mandatory here:
    homolytic cleavage produces two radicals, i.e. an open shell, and without it the
    calculation either fails to converge or gives the wrong energy: that is a measured
    property, see the next test.
    """
    st = from_symbols(["H", "O", "O", "H"], H2O2_GEOM, spin_multiplicity=3)
    axis = np.array([0.0, -1.0, 0.0])
    st.positions[2] += axis * 2.5
    st.positions[3] += axis * 2.5
    res = GFN.compute(st)
    bonds = bonds_from_orders(res.extra["bond_orders"])
    assert not [b for b in bonds if {b.i, b.j} == {1, 2}], "O-O bond at 3.9 Å is still counted as a bond"
    species = find_species(st.numbers, bonds, res.extra["charges"])
    assert len(species) == 2 and all(s.formula == "HO" for s in species), (
        f"the broken-apart system was read as {[s.formula for s in species]}"
    )


def test_dissociation_energy_dft_matches_experiment_and_gfn2_does_not():
    """
    A measured applicability limit, named as a number rather than hidden.

    D(O-O) in H2O2 -> 2 OH: experiment gives 2.15 eV. Our reference backend (B3LYP/def2-SVP,
    UNrestricted for the radical) gives about 2.5 eV: agreement within what is expected for
    this basis set and without a zero-point correction. The cheap GFN2 gives about 5.4 eV,
    i.e. overshoots by more than a factor of two: a tightly-bound semiempirical method
    without spin polarization describes radicals poorly. Hence the engine's working rule:
    GFN2 carries the dynamics, but the reference method computes the energetics of bond
    breaking and radicals.

    The test locks in BOTH claims. If a new tblite version fixes radicals, this test will
    fail, and that is correct: the role-division rule will need to be revisited based on the
    numbers.
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
        f"reference method gives D(O-O) = {d_dft:.3f} eV vs the experimental 2.15"
    )
    assert d_gfn > 1.8 * d_dft, (
        f"the known GFN2 overestimate has disappeared: GFN2 {d_gfn:.3f} eV, reference {d_dft:.3f} eV -- "
        "the role-division rule needs to be revisited based on the numbers"
    )


@pytest.mark.slow
def test_rare_event_needs_more_than_patience():
    """
    Direct dynamics does NOT overcome a high barrier, and this is recorded as a measured
    fact rather than a failure: 2000 steps at 6000 K on H2O2 produce no dissociation at all
    (measured), because with an attempt frequency of about one per 27 fs and the Boltzmann
    factor for 2.2 eV, the expected number of events is less than one. Enhanced sampling is
    not yet implemented in the engine.
    """
    st = from_symbols(["H", "O", "O", "H"], H2O2_GEOM)
    st.set_maxwell_boltzmann(6000.0, np.random.default_rng(4))
    res = run(st, GFN, steps=2000, dt_fs=0.25, temperature_k=6000.0,
              friction_per_fs=0.002, seed=4, sample_every=10)
    assert res.events == [], (
        f"at 6000 K over 500 fs, {len(res.events)} events were recorded -- the rare-event "
        "estimate is wrong and needs to be recomputed"
    )
