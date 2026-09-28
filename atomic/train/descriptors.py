"""
Descriptors of an atom's environment and their derivatives, hand-written.

The choice of the model's form is the main decision of the whole scheme, and it was made
for reasons of GPU SPEED, not fashion. Behler-Parrinello symmetry functions are used
(Behler & Parrinello, PRL 98 (2007) 146401; Behler, JCP 134 (2011) 074106): radial G2 and
angular G4. Reasons:

1. They are invariant under translation, rotation and permutation of identical atoms,
   i.e. they satisfy the same symmetries as energy, and the model does not spend capacity
   learning them.
2. Their derivatives with respect to coordinates are expressed in closed form through the
   same quantities as the descriptors themselves. So forces are computed WITHOUT automatic
   differentiation, and it is exactly automatic differentiation that makes neural-network
   potentials expensive in the dynamics loop.
3. The model on top of them is LINEAR, so training is a single least-squares problem with
   an exact solution, without iterations, without a learning rate, and without overfitting
   to optimizer noise. If linear capacity is not enough, it will show up as error on the
   held-out set, and then a small network is placed on top of the same descriptors; the
   descriptors themselves do not change.

Cost per atom: (number of neighbors) × (number of radial functions) for G2 plus (number
of neighbor pairs) × (number of angular functions) for G4. This is thousands of
operations against millions for a message-passing model, and all the arithmetic is
element-wise, i.e. ideal for the GPU.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DescriptorSpec:
    """
    A set of descriptors. The parameters are not a fit to the answer but coverage of the
    space: the centers of the radial Gaussians are placed evenly from the contact
    distance to the cutoff radius, the width is taken equal to the grid step of the
    centers (so neighboring functions overlap and the set has no blind spots), and the
    angular functions use a standard set of multiplicities.

    `cutoff` is a physical quantity: beyond it, the interaction is treated as zero. 5 Å is
    chosen as the distance at which the hydrogen bond and the first coordination shell of
    water are already inside, while the cost is still moderate (about 40 neighbors at the
    density of liquid water).
    """

    cutoff: float = 5.0
    n_radial: int = 8
    zetas: tuple[float, ...] = (1.0, 2.0, 4.0)
    lambdas: tuple[float, ...] = (1.0, -1.0)
    eta_angular: float = 0.08
    r_min: float = 0.8
    # A SET of widths, not one: narrow functions resolve the position of the first peak,
    # wide ones see the second coordination shell. With a single width (the first
    # version), the held-out force error stayed at 1327 meV/Å; a set of widths is a
    # direct expansion of the basis's capacity without changing its form and without
    # raising the cost by an order of magnitude.
    eta_scales: tuple[float, ...] = (0.5, 1.0, 2.0)
    eta_angular_scales: tuple[float, ...] = (0.5, 2.0)

    @property
    def mu(self) -> np.ndarray:
        """Centers of the radial Gaussians."""
        return np.linspace(self.r_min, self.cutoff * 0.95, self.n_radial)

    @property
    def eta_radial(self) -> float:
        """Width from the grid step of the centers: neighboring functions overlap at about half height."""
        step = (self.cutoff * 0.95 - self.r_min) / max(1, self.n_radial - 1)
        return 1.0 / (2.0 * step**2)

    @property
    def n_angular(self) -> int:
        return len(self.zetas) * len(self.lambdas) * len(self.eta_angular_scales)

    @property
    def n_radial_total(self) -> int:
        return self.n_radial * len(self.eta_scales)


def cutoff_function(r: np.ndarray, rc: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Cosine cutoff function f = 0.5(cos(pi r/rc) + 1) and its derivative.

    The cutoff must vanish TOGETHER WITH ITS DERIVATIVE at the radius: otherwise force
    jumps discontinuously when a neighbor crosses the list boundary, and energy is not
    conserved in NVE. For the cosine form, both the value and the derivative are zero at
    r = rc; this is its reason for existing.
    """
    inside = r < rc
    f = np.where(inside, 0.5 * (np.cos(np.pi * r / rc) + 1.0), 0.0)
    df = np.where(inside, -0.5 * np.pi / rc * np.sin(np.pi * r / rc), 0.0)
    return f, df


def species_pair_index(z_a: int, z_b: int, species: tuple[int, ...]) -> int:
    """Index of an UNordered pair of species: (H,O) and (O,H) are one channel."""
    ia, ib = species.index(z_a), species.index(z_b)
    lo, hi = min(ia, ib), max(ia, ib)
    n = len(species)
    return lo * n - lo * (lo - 1) // 2 + (hi - lo)


def n_pair_channels(species: tuple[int, ...]) -> int:
    n = len(species)
    return n * (n + 1) // 2


def descriptor_length(spec: DescriptorSpec, species: tuple[int, ...]) -> int:
    """Total length of a single atom's descriptor vector."""
    n_pairs = n_pair_channels(species)
    return spec.n_radial_total * len(species) + spec.n_angular * n_pairs


def compute_descriptors(
    positions: np.ndarray,
    numbers: np.ndarray,
    spec: DescriptorSpec,
    species: tuple[int, ...],
    cell: np.ndarray | None = None,
    with_gradients: bool = True,
):
    """
    Descriptors of all atoms and, if needed, their derivatives with respect to
    coordinates.

    Returns (G, dG) where G has shape (N, D), and dG has shape (N, D, N, 3): the derivative
    of descriptor d of atom i with respect to the coordinate of atom k. The shape is
    wasteful in memory and is suited only for training on small systems; in dynamics
    forces are gathered on the fly, without storing the full tensor (see
    atomic/train/model.py and the WGSL port).

    Periodicity is handled in a minimal way along all axes, if a cell is given.
    """
    n = len(numbers)
    d_len = descriptor_length(spec, species)
    g = np.zeros((n, d_len))
    dg = np.zeros((n, d_len, n, 3)) if with_gradients else None

    mu = spec.mu
    eta_r = spec.eta_radial
    n_sp = len(species)
    n_rad_block = spec.n_radial_total * n_sp

    for i in range(n):
        # --- list of neighbors within the cutoff radius -------------------------------
        d = positions - positions[i]
        if cell is not None:
            frac = np.linalg.solve(cell.T, d.T).T
            frac -= np.round(frac)
            d = (cell.T @ frac.T).T
        r = np.linalg.norm(d, axis=1)
        mask = (r < spec.cutoff) & (r > 1e-8)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        rij = r[idx]
        dij = d[idx]
        unit = dij / rij[:, None]
        fc, dfc = cutoff_function(rij, spec.cutoff)

        # --- G2: radial ------------------------------------------------------------------
        # g2[n] = sum_j exp(-eta (r_ij - mu_n)^2) fc(r_ij), in a separate channel per neighbor species
        diff = rij[:, None] - mu[None, :]
        contrib_parts, dcontrib_parts = [], []
        for scale in spec.eta_scales:
            eta = eta_r * scale
            gauss = np.exp(-eta * diff**2)
            contrib_parts.append(gauss * fc[:, None])
            dcontrib_parts.append(gauss * (-2.0 * eta * diff * fc[:, None] + dfc[:, None]))
        contrib = np.concatenate(contrib_parts, axis=1)          # (n_nb, n_radial_total)
        dcontrib_dr = np.concatenate(dcontrib_parts, axis=1)

        for k, j in enumerate(idx):
            sp = species.index(int(numbers[j]))
            block = slice(sp * spec.n_radial_total, (sp + 1) * spec.n_radial_total)
            g[i, block] += contrib[k]
            if dg is not None:
                # derivative with respect to r_ij, decomposed onto axes via the unit vector
                grad = dcontrib_dr[k][:, None] * unit[k][None, :]
                dg[i, block, j, :] += grad
                dg[i, block, i, :] -= grad

        # --- G4: angular -------------------------------------------------------------------
        # g4 = sum_{j<k} (1 + lambda cos(theta_ijk))^zeta * exp(-eta(r_ij^2+r_ik^2+r_jk^2)) * fc fc fc
        if idx.size >= 2:
            for a in range(idx.size - 1):
                for b in range(a + 1, idx.size):
                    ja, jb = idx[a], idx[b]
                    ra, rb = rij[a], rij[b]
                    va, vb = dij[a], dij[b]
                    vjk = vb - va
                    rjk = float(np.linalg.norm(vjk))
                    if rjk >= spec.cutoff or rjk < 1e-8:
                        continue
                    fc_jk, dfc_jk = cutoff_function(np.array([rjk]), spec.cutoff)
                    fc_jk, dfc_jk = float(fc_jk[0]), float(dfc_jk[0])
                    cos_t = float(va @ vb) / (ra * rb)
                    fcc = fc[a] * fc[b] * fc_jk
                    r2sum = ra**2 + rb**2 + rjk**2
                    ch = species_pair_index(int(numbers[ja]), int(numbers[jb]), species)
                    base = n_rad_block + ch * spec.n_angular

                    slot = 0
                    for eta_scale in spec.eta_angular_scales:
                      eta_a = spec.eta_angular * eta_scale
                      expo = np.exp(-eta_a * r2sum)
                      for zeta in spec.zetas:
                        for lam in spec.lambdas:
                            angular = (1.0 + lam * cos_t) ** zeta
                            g[i, base + slot] += angular * expo * fcc
                            if dg is not None:
                                # derivatives with respect to the three distances and to the cosine of the angle
                                dang_dcos = (
                                    zeta * lam * (1.0 + lam * cos_t) ** (zeta - 1.0)
                                    if abs(1.0 + lam * cos_t) > 1e-12
                                    else 0.0
                                )
                                # d cos / d va, d cos / d vb
                                dcos_dva = vb / (ra * rb) - cos_t * va / ra**2
                                dcos_dvb = va / (ra * rb) - cos_t * vb / rb**2
                                # d expo / d va, d vb, d vjk
                                dexpo_dva = -2.0 * eta_a * va * expo
                                dexpo_dvb = -2.0 * eta_a * vb * expo
                                dexpo_dvjk = -2.0 * eta_a * vjk * expo
                                # d fcc
                                dfcc_dva = dfc[a] * (va / ra) * fc[b] * fc_jk
                                dfcc_dvb = dfc[b] * (vb / rb) * fc[a] * fc_jk
                                dfcc_dvjk = dfc_jk * (vjk / rjk) * fc[a] * fc[b]

                                # full derivative with respect to va and vb (vjk = vb - va)
                                dva = (
                                    dang_dcos * dcos_dva * expo * fcc
                                    + angular * dexpo_dva * fcc
                                    + angular * expo * dfcc_dva
                                    - angular * dexpo_dvjk * fcc
                                    - angular * expo * dfcc_dvjk
                                )
                                dvb = (
                                    dang_dcos * dcos_dvb * expo * fcc
                                    + angular * dexpo_dvb * fcc
                                    + angular * expo * dfcc_dvb
                                    + angular * dexpo_dvjk * fcc
                                    + angular * expo * dfcc_dvjk
                                )
                                dg[i, base + slot, ja, :] += dva
                                dg[i, base + slot, jb, :] += dvb
                                dg[i, base + slot, i, :] -= dva + dvb
                            slot += 1

    return (g, dg) if with_gradients else g
