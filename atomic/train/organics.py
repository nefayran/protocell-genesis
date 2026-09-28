"""
Organics for the dataset: fatty acids, glycerol, and their mixtures with water.

Why this is here. A potential trained on water knows hydrogen and oxygen, and knows
nothing about the carbon chains that membranes are made of. Yet it is exactly these that
the whole original task rests on: a fatty acid, the ester bond of glycerol, the hydrogen
bond between carboxyl heads. As long as there is no carbon, the engine cannot compute A
SINGLE ONE of the quantities that the previous layer of this project made up (bond energy,
hydrolysis barrier, head association energy), and those were exactly the main lie of the
old model.

How the geometry is built. The molecule is assembled roughly, from tabulated bond lengths
and angles, and is then RELAXED by the teacher, i.e. the exact geometry is obtained by
computation, not by hand-tuning. Any new molecule is treated the same way: a rough
skeleton plus relaxation, without manual correction of coordinates.
"""
from __future__ import annotations

import numpy as np

from engine.state import from_symbols

# Tabulated bond lengths (Å) and angles (degrees) for the rough assembly. Reference values:
# C-C 1.53, C-H 1.09, C=O 1.21, C-O 1.36, O-H 0.97 (Harmony 1990; Herzberg).
BOND_CC = 1.53
BOND_CH = 1.09
BOND_CO_DOUBLE = 1.21
BOND_CO_SINGLE = 1.36
# C-O in an ALCOHOL is longer than in a carboxyl: 1.43 against 1.36 (Harmony 1990). The
# first version gave glycerol the carboxyl value, and relaxation from that starting point
# barely moved the bonds; the result was 1.336-1.388 Å, i.e. the starting numbers, not the
# equilibrium ones. The error looked like "a different isomer formed", though it was
# actually unconverged geometry.
BOND_CO_ALCOHOL = 1.43
BOND_OH = 0.97
TETRAHEDRAL_DEG = 109.47


def _tetrahedral_free_directions(centre: np.ndarray, bonded: list[np.ndarray]) -> list[np.ndarray]:
    """
    Free tetrahedral directions at an atom, part of whose bonds are already occupied.

    Why this instead of axis-aligned offsets. The first version placed substituents with
    shifts like (0, 0, ±1.09), and relaxation TORE OFF some of the hydrogens: for glycerol,
    after optimization two hydrogens were left with no bond at all, the molecule turned
    into C3H6O3 plus two free atoms, and the C-O bonds came out at 1.31-1.40 Å instead of
    the alcohol's 1.43. The error looked like "the wrong isomer", though it was actually an
    impossible starting geometry.

    Here the directions are built properly: for one occupied bond, a cone at the
    tetrahedral angle to it; for two, the two remaining vertices of the tetrahedron; for
    three, one vertex.
    """
    if not bonded:
        return [np.array([0.0, 0.0, 1.0])]
    used = [(np.asarray(b) - centre) / np.linalg.norm(np.asarray(b) - centre) for b in bonded]
    cos_t = np.cos(np.radians(TETRAHEDRAL_DEG))

    if len(used) == 1:
        axis = used[0]
        # any perpendicular to the axis
        tmp = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = np.cross(axis, tmp)
        perp /= np.linalg.norm(perp)
        perp2 = np.cross(axis, perp)
        sin_t = np.sqrt(max(0.0, 1.0 - cos_t**2))
        return [
            cos_t * axis + sin_t * (np.cos(a) * perp + np.sin(a) * perp2)
            for a in (0.0, 2.0 * np.pi / 3.0, 4.0 * np.pi / 3.0)
        ]
    if len(used) == 2:
        bisector = used[0] + used[1]
        bisector /= np.linalg.norm(bisector)
        normal = np.cross(used[0], used[1])
        normal /= np.linalg.norm(normal)
        # the two remaining vertices lie in the plane perpendicular to the first two
        half = np.radians(TETRAHEDRAL_DEG / 2.0)
        return [-bisector * np.cos(half) + normal * np.sin(half),
                -bisector * np.cos(half) - normal * np.sin(half)]
    total = sum(used)
    norm = np.linalg.norm(total)
    return [-total / norm] if norm > 1e-9 else [np.array([0.0, 0.0, 1.0])]


def _zigzag_chain(n_carbon: int, start=(0.0, 0.0, 0.0)) -> np.ndarray:
    """
    A zigzag carbon skeleton in the xy plane, as in an extended alkane chain.

    A zigzag, not a straight line: the tetrahedral angle at carbon is 109.47°, and a
    straight chain would be geometrically impossible. The step along and across the axis
    is derived from the bond length and angle, not fitted.
    """
    half = np.radians(TETRAHEDRAL_DEG / 2.0)
    dx = BOND_CC * np.sin(half)
    dy = BOND_CC * np.cos(half)
    pos = []
    for k in range(n_carbon):
        pos.append([start[0] + k * dx, start[1] + (dy if k % 2 else 0.0), start[2]])
    return np.array(pos)


def alkanoic_acid(n_carbon: int):
    """
    A fatty acid CH3-(CH2)_{n-2}-COOH, roughly assembled.

    Hydrogens are placed in pairs perpendicular to the skeleton's plane; this is enough to
    start relaxation, and the teacher gives the exact positions. The carboxyl group is
    assembled on the LAST carbon: the head is always terminal in a real acid.
    """
    if n_carbon < 2:
        raise ValueError("a fatty acid starts with at least two carbons")
    chain = _zigzag_chain(n_carbon)
    symbols = ["C"] * n_carbon
    positions = list(chain)

    # hydrogens on all carbons except the carboxyl one
    for k in range(n_carbon - 1):
        base = chain[k]
        for sign in (+1.0, -1.0):
            positions.append(base + np.array([0.0, 0.0, sign * BOND_CH]))
            symbols.append("H")
    # the terminal methyl gets a third hydrogen along the chain
    axis = chain[0] - chain[1]
    axis /= np.linalg.norm(axis)
    positions.append(chain[0] + axis * BOND_CH)
    symbols.append("H")

    # carboxyl on the last carbon: =O and -OH
    tail = chain[-1]
    direction = tail - chain[-2]
    direction /= np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0], 0.0])
    o_double = tail + direction * BOND_CO_DOUBLE * 0.5 + perp * BOND_CO_DOUBLE * 0.87
    o_single = tail + direction * BOND_CO_SINGLE * 0.5 - perp * BOND_CO_SINGLE * 0.87
    positions += [o_double, o_single, o_single + np.array([0.0, 0.0, BOND_OH])]
    symbols += ["O", "O", "H"]
    return symbols, np.array(positions)


def glycerol():
    """
    Glycerol C3H8O3: three carbons, each with a hydroxyl.

    It is exactly this molecule that makes the membrane amphiphile two-tailed IN REAL
    CHEMISTRY: two fatty acids attach to its hydroxyls via ester bonds. In the previous
    layer of the project, being two-tailed was a valence resolution, not an actual
    molecule; here the molecule exists.
    """
    chain = _zigzag_chain(3)
    symbols = ["C", "C", "C"]
    positions = list(chain)
    for k in range(3):
        neighbours = [chain[j] for j in (k - 1, k + 1) if 0 <= j < 3]
        free = _tetrahedral_free_directions(chain[k], neighbours)
        # the first free direction goes to the hydroxyl, the rest to hydrogens
        oxygen = chain[k] + free[0] * BOND_CO_ALCOHOL
        positions.append(oxygen)
        symbols.append("O")
        # the hydroxyl's hydrogen is placed at the tetrahedral angle to the C-O bond, not "off to the side"
        oh_dirs = _tetrahedral_free_directions(oxygen, [chain[k]])
        positions.append(oxygen + oh_dirs[0] * BOND_OH)
        symbols.append("H")
        for direction in free[1:]:
            positions.append(chain[k] + direction * BOND_CH)
            symbols.append("H")
    return symbols, np.array(positions)


def relax_with(teacher, symbols, positions, force_tol=0.005, max_iterations=800):
    """Rough skeleton -> equilibrium geometry via teacher computation."""
    from engine.integrate import optimise_lbfgs

    state = from_symbols(symbols, positions)
    optimise_lbfgs(state, teacher, force_tol_ev_per_a=force_tol, max_iterations=max_iterations)
    return state


def rattled(state, rng: np.random.Generator, scale: float = 0.08, count: int = 1):
    """Thermal distortions around equilibrium: what the model will see in dynamics."""
    out = []
    for _ in range(count):
        copy = state.copy()
        copy.positions = copy.positions + rng.normal(scale=scale, size=copy.positions.shape)
        out.append(copy)
    return out


def solvated(state, rng: np.random.Generator, n_water: int, box_pad: float = 4.0,
             min_distance: float = 1.6):
    """
    A molecule in water: a cell around the molecule plus water molecules at free spots.

    Water is placed with a minimum-distance check, not "wherever": without this check the
    dataset picks up frames with overlapping atoms, and one such frame weighs, in the
    quadratic loss, as much as a million normal ones; this was already measured on water
    (forces up to 1.1e7 eV/Å, loss 2.9e15).
    """
    from train.sampling import WATER, _rotation

    span = state.positions.max(axis=0) - state.positions.min(axis=0)
    box = float(max(span) + 2 * box_pad)
    centre = state.positions.mean(axis=0)
    positions = [state.positions - centre + box / 2.0]
    symbols = list(state.symbols)

    placed, attempts = 0, 0
    while placed < n_water and attempts < n_water * 200:
        attempts += 1
        origin = rng.uniform(1.0, box - 1.0, size=3)
        mol = (WATER - WATER[0]) @ _rotation(rng).T + origin
        existing = np.vstack(positions)
        d = mol[:, None, :] - existing[None, :, :]
        d -= box * np.round(d / box)
        if np.linalg.norm(d, axis=2).min() < min_distance:
            continue
        positions.append(mol)
        symbols += ["O", "H", "H"]
        placed += 1
    return from_symbols(symbols, np.vstack(positions), cell=np.eye(3) * box, pbc=(True, True, True))


def build_organic_states(teacher, rng: np.random.Generator, chain_lengths=(2, 4, 8),
                         rattles_per_molecule: int = 60, solvated_per_molecule: int = 40,
                         waters=(8, 16)):
    """
    A full set of organic configurations: acids of varying length, glycerol, and the same
    in water.

    The chain lengths are not chosen at random: C2 (acetic) and C4 are cheap and cover the
    carboxyl chemistry, C8 (octanoic) is the first one for which the literature reports a
    measured critical micelle concentration (around 300 mM), so the model can be checked
    against experiment for it, not only against the teacher.
    """
    states = []
    relaxed = {}
    for n in chain_lengths:
        sym, pos = alkanoic_acid(n)
        st = relax_with(teacher, sym, pos)
        relaxed[f"C{n}"] = st
        states += rattled(st, rng, scale=0.07, count=rattles_per_molecule)
    sym, pos = glycerol()
    gl = relax_with(teacher, sym, pos)
    relaxed["glycerol"] = gl
    states += rattled(gl, rng, scale=0.07, count=rattles_per_molecule)

    for name, st in relaxed.items():
        for _ in range(solvated_per_molecule):
            n_water = int(rng.choice(waters))
            states.append(solvated(st, rng, n_water))
    return states, relaxed
