"""
Descriptors on tensors: the same computation without a Python loop over atoms.

Why. A measurement showed that the entire benefit of our model is eaten up by the
implementation: at 3 atoms we are 57x faster than the teacher, at 24 atoms 3x, at 81 atoms
1.2x, and at 192 atoms already SLOWER (0.90x). The cause is not the model: at 192 atoms it
needs about seven million operations to compute forces, i.e. less than a millisecond even
at modest throughput, against a measured 618 ms. The difference is the Python loop over
atoms and over neighbor pairs.

Here the same set of symmetry functions is computed batch-wise: all atoms at once, all
neighbor pairs at once, on CPU or GPU. The formulas do not change, only the method of
computing them does, and this is verified by comparison with the reference implementation
(validate/test_descriptors_torch.py): the discrepancy must be at the level of single
precision, not "approximately matching".

Computation design. Neighbors are searched for once per frame and stored as flat lists of
pairs: this way periodicity is handled by a single subtraction rather than a search over
cells. For the angular part, all TRIPLETS (center, neighbor a, neighbor b) are built; their
count grows as the square of the number of neighbors, so they are assembled in blocks to
stay within memory.
"""
from __future__ import annotations

import numpy as np

from .descriptors import DescriptorSpec, descriptor_length, n_pair_channels


def _pair_channel_matrix(species: tuple[int, ...]) -> np.ndarray:
    """Channel table for a pair of species: (n_sp, n_sp) -> index of an unordered pair."""
    n = len(species)
    table = np.zeros((n, n), dtype=np.int64)
    idx = 0
    for a in range(n):
        for b in range(a, n):
            table[a, b] = table[b, a] = idx
            idx += 1
    return table


class TorchDescriptors:
    """
    Batched computation of descriptors and their derivatives.

    `triplet_block` is how many triplets to process at a time. The value is chosen by
    MEASUREMENT against available memory: triplets are the main cost, their count per frame
    is of order N·nb²/2.
    """

    def __init__(self, spec: DescriptorSpec, species: tuple[int, ...], device: str = "cpu",
                 dtype: str = "float32", triplet_block: int = 200_000) -> None:
        import torch

        self.spec = spec
        self.species = species
        self.device = device
        self.torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        self.triplet_block = int(triplet_block)
        self.d_len = descriptor_length(spec, species)
        self.n_sp = len(species)
        self.n_rad = spec.n_radial_total
        self.n_rad_block = self.n_rad * self.n_sp
        self.n_ang = spec.n_angular

        self.mu = torch.tensor(spec.mu, device=device, dtype=self.torch_dtype)
        self.etas = torch.tensor(
            [spec.eta_radial * s for s in spec.eta_scales], device=device, dtype=self.torch_dtype
        )
        self.zetas = torch.tensor(list(spec.zetas), device=device, dtype=self.torch_dtype)
        self.lambdas = torch.tensor(list(spec.lambdas), device=device, dtype=self.torch_dtype)
        self.eta_ang = torch.tensor(
            [spec.eta_angular * s for s in spec.eta_angular_scales], device=device, dtype=self.torch_dtype
        )
        self.pair_table = torch.tensor(_pair_channel_matrix(species), device=device)
        self.species_index = {int(z): i for i, z in enumerate(species)}

    # --- helpers -----------------------------------------------------------------------
    def _cutoff(self, r):
        import torch

        rc = self.spec.cutoff
        inside = r < rc
        f = torch.where(inside, 0.5 * (torch.cos(np.pi * r / rc) + 1.0), torch.zeros_like(r))
        df = torch.where(inside, -0.5 * np.pi / rc * torch.sin(np.pi * r / rc), torch.zeros_like(r))
        return f, df

    def _triplets(self, i_idx, n_atoms):
        """
        Neighbor pairs of a single center WITHOUT a loop over atoms.

        Neighbors in the flat pair list come in groups by center, so all within-group
        pairs are built with a single piece of arithmetic: for a group of length c that is
        c(c-1)/2 pairs, and offsets are computed as a cumulative sum. Measurement that led
        to this rewrite: the Python loop over atoms becomes noticeably slow already at a
        hundred atoms, and all its work is just index bookkeeping.
        """
        import torch

        # The index bookkeeping is computed ON THE CPU in double precision and only then
        # moved to the device. Two reasons, both measured: Metal does not support float64
        # at all (the backend crashed on it with "Cannot convert a MPS Tensor to float64"),
        # and the inverse numbering of the upper triangle goes through a square root, where
        # single precision is not enough; at 200 thousand triplets, an off-by-one error in
        # floor gives the wrong neighbor pair. The cost of this bookkeeping is negligible
        # against the computation itself, so keeping it on the CPU costs nothing.
        counts_cpu = torch.bincount(i_idx.cpu(), minlength=n_atoms)
        starts = torch.cumsum(counts_cpu, 0) - counts_cpu
        pairs_per = counts_cpu * (counts_cpu - 1) // 2
        total = int(pairs_per.sum())
        if total == 0:
            return None
        centre_of = torch.repeat_interleave(torch.arange(n_atoms), pairs_per)
        offsets = torch.cumsum(pairs_per, 0) - pairs_per
        rank = torch.arange(total) - offsets[centre_of]
        c = counts_cpu[centre_of].to(torch.float64)
        rf = rank.to(torch.float64)
        a = torch.floor(c - 0.5 - torch.sqrt((c - 0.5) ** 2 - 2.0 * rf)).to(torch.long)
        used = (a.to(torch.float64) * (2.0 * c - a.to(torch.float64) - 1.0) / 2.0).to(torch.long)
        b = (rank - used) + a + 1
        base = starts[centre_of]
        return torch.stack([base + a, base + b]).to(i_idx.device)

    def compute(self, positions, numbers, cell=None):
        """
        Returns (G, dG) as tensors: G of shape (N, D), dG of shape (N, D, N, 3).

        The shape of dG is the same as in the reference implementation, so that training
        and checks work unchanged. It is wasteful in memory; in production forces are
        gathered on the fly, but here comparability with the reference matters.
        """
        import torch

        pos = torch.as_tensor(np.asarray(positions), device=self.device, dtype=self.torch_dtype)
        nums = np.asarray(numbers, dtype=int)
        sp_of = torch.tensor([self.species_index[int(z)] for z in nums], device=self.device)
        n = len(nums)
        cell_t = None if cell is None else torch.as_tensor(
            np.asarray(cell), device=self.device, dtype=self.torch_dtype
        )

        g = torch.zeros((n, self.d_len), device=self.device, dtype=self.torch_dtype)
        dg = torch.zeros((n, self.d_len, n, 3), device=self.device, dtype=self.torch_dtype)

        # --- all pairs within the radius, in a single pass ---------------------------------
        delta = pos[None, :, :] - pos[:, None, :]
        if cell_t is not None:
            inv = torch.linalg.inv(cell_t.T)
            frac = torch.einsum("ab,ijb->ija", inv, delta)
            frac = frac - torch.round(frac)
            delta = torch.einsum("ab,ijb->ija", cell_t.T, frac)
        dist = delta.norm(dim=2)
        eye = torch.eye(n, device=self.device, dtype=torch.bool)
        within = (dist < self.spec.cutoff) & (~eye)
        i_idx, j_idx = torch.nonzero(within, as_tuple=True)
        if i_idx.numel() == 0:
            return g, dg

        vij = delta[i_idx, j_idx]                       # (P, 3)
        rij = dist[i_idx, j_idx]                        # (P,)
        unit = vij / rij[:, None]
        fc, dfc = self._cutoff(rij)

        # --- radial part -------------------------------------------------------------------
        diff = rij[:, None] - self.mu[None, :]                                  # (P, n_radial)
        gauss = torch.exp(-self.etas[:, None, None] * diff[None] ** 2)          # (S, P, n_radial)
        value = (gauss * fc[None, :, None]).permute(1, 0, 2).reshape(len(rij), -1)
        dvalue = (
            gauss * (-2.0 * self.etas[:, None, None] * diff[None] * fc[None, :, None] + dfc[None, :, None])
        ).permute(1, 0, 2).reshape(len(rij), -1)

        base_col = sp_of[j_idx] * self.n_rad                                    # channel by NEIGHBOR species
        cols = base_col[:, None] + torch.arange(self.n_rad, device=self.device)[None, :]
        g.index_put_((i_idx[:, None].expand_as(cols), cols), value, accumulate=True)

        grad = dvalue[:, :, None] * unit[:, None, :]                            # (P, n_rad, 3)
        for axis in range(3):
            dg.index_put_(
                (i_idx[:, None].expand_as(cols), cols, j_idx[:, None].expand_as(cols),
                 torch.full_like(cols, axis)),
                grad[:, :, axis], accumulate=True,
            )
            dg.index_put_(
                (i_idx[:, None].expand_as(cols), cols, i_idx[:, None].expand_as(cols),
                 torch.full_like(cols, axis)),
                -grad[:, :, axis], accumulate=True,
            )

        # --- angular part: triplets in blocks -----------------------------------------------
        # neighbors of each center are contiguous, so triplets are built by group boundaries
        counts = torch.bincount(i_idx, minlength=n)
        starts = torch.cumsum(counts, 0) - counts
        triples = []
        for centre in range(n):
            c = int(counts[centre])
            if c < 2:
                continue
            offset = int(starts[centre])
            a, b = torch.triu_indices(c, c, offset=1, device=self.device)
            triples.append(torch.stack([a + offset, b + offset]))
        if not triples:
            return g, dg
        triples = torch.cat(triples, dim=1)

        for block_start in range(0, triples.shape[1], self.triplet_block):
            pa, pb = triples[:, block_start : block_start + self.triplet_block]
            centre = i_idx[pa]
            ja, jb = j_idx[pa], j_idx[pb]
            va, vb = vij[pa], vij[pb]
            ra, rb = rij[pa], rij[pb]
            vjk = vb - va
            rjk = vjk.norm(dim=1)
            keep = (rjk < self.spec.cutoff) & (rjk > 1e-8)
            if not bool(keep.any()):
                continue
            pa, pb, centre, ja, jb = pa[keep], pb[keep], centre[keep], ja[keep], jb[keep]
            va, vb, ra, rb, vjk, rjk = va[keep], vb[keep], ra[keep], rb[keep], vjk[keep], rjk[keep]
            fc_jk, dfc_jk = self._cutoff(rjk)
            cos_t = (va * vb).sum(dim=1) / (ra * rb)
            fcc = fc[pa] * fc[pb] * fc_jk
            r2sum = ra**2 + rb**2 + rjk**2
            channel = self.pair_table[sp_of[ja], sp_of[jb]]

            dcos_dva = vb / (ra * rb)[:, None] - cos_t[:, None] * va / (ra**2)[:, None]
            dcos_dvb = va / (ra * rb)[:, None] - cos_t[:, None] * vb / (rb**2)[:, None]
            dfcc_dva = (dfc[pa] * fc[pb] * fc_jk)[:, None] * (va / ra[:, None])
            dfcc_dvb = (dfc[pb] * fc[pa] * fc_jk)[:, None] * (vb / rb[:, None])
            dfcc_dvjk = (dfc_jk * fc[pa] * fc[pb])[:, None] * (vjk / rjk[:, None])

            slot = 0
            for eta_a in self.eta_ang:
                expo = torch.exp(-eta_a * r2sum)
                dexpo_dva = -2.0 * eta_a * va * expo[:, None]
                dexpo_dvb = -2.0 * eta_a * vb * expo[:, None]
                dexpo_dvjk = -2.0 * eta_a * vjk * expo[:, None]
                for zeta in self.zetas:
                    for lam in self.lambdas:
                        base = 1.0 + lam * cos_t
                        angular = base**zeta
                        col = self.n_rad_block + channel * self.n_ang + slot
                        g.index_put_((centre, col), angular * expo * fcc, accumulate=True)

                        dang = torch.where(
                            base.abs() > 1e-12, zeta * lam * base ** (zeta - 1.0), torch.zeros_like(base)
                        )
                        common = (angular * expo)[:, None]
                        af = (angular * fcc)[:, None]
                        dva = (
                            (dang * expo * fcc)[:, None] * dcos_dva + af * dexpo_dva
                            + common * dfcc_dva - af * dexpo_dvjk - common * dfcc_dvjk
                        )
                        dvb = (
                            (dang * expo * fcc)[:, None] * dcos_dvb + af * dexpo_dvb
                            + common * dfcc_dvb + af * dexpo_dvjk + common * dfcc_dvjk
                        )
                        for axis in range(3):
                            dg.index_put_((centre, col, ja, torch.full_like(ja, axis)),
                                          dva[:, axis], accumulate=True)
                            dg.index_put_((centre, col, jb, torch.full_like(jb, axis)),
                                          dvb[:, axis], accumulate=True)
                            dg.index_put_((centre, col, centre, torch.full_like(centre, axis)),
                                          -(dva[:, axis] + dvb[:, axis]), accumulate=True)
                        slot += 1

        return g, dg


    def energy_forces_direct(self, positions, numbers, de_dg_fn, cell=None):
        """
        Forces WITHOUT building the descriptor-derivative tensor.

        The key idea behind this method: the computation needs forces, not descriptor
        derivatives. If dE/dG is obtained first (one cheap network pass over all atoms),
        the force can be assembled directly while walking pairs and triplets:

            F_k = - sum_i sum_d (dE/dG_i,d) * (dG_i,d / dr_k)

        and each term is accounted for immediately, so the (N, D, N, 3) tensor never
        appears. Measurement that made this necessary: with that tensor, the tensor
        version gave only 2.1-2.5x against the reference, because all the work went into
        writing to memory rather than computing (81 atoms, D=132 is 2.6 million numbers
        per frame).

        `de_dg_fn` is a function that, given (N, D) descriptors, returns (E, dE/dG). This
        separation is needed so that this method knows nothing about the model's form: a
        network, a linear model, or anything else is plugged in from outside.
        """
        import torch

        pos = torch.as_tensor(np.asarray(positions), device=self.device, dtype=self.torch_dtype)
        nums = np.asarray(numbers, dtype=int)
        sp_of = torch.tensor([self.species_index[int(z)] for z in nums], device=self.device)
        n = len(nums)
        cell_t = None if cell is None else torch.as_tensor(
            np.asarray(cell), device=self.device, dtype=self.torch_dtype
        )

        delta = pos[None, :, :] - pos[:, None, :]
        if cell_t is not None:
            inv = torch.linalg.inv(cell_t.T)
            frac = torch.einsum("ab,ijb->ija", inv, delta)
            frac = frac - torch.round(frac)
            delta = torch.einsum("ab,ijb->ija", cell_t.T, frac)
        dist = delta.norm(dim=2)
        eye = torch.eye(n, device=self.device, dtype=torch.bool)
        within = (dist < self.spec.cutoff) & (~eye)
        i_idx, j_idx = torch.nonzero(within, as_tuple=True)

        g = torch.zeros((n, self.d_len), device=self.device, dtype=self.torch_dtype)
        forces = torch.zeros((n, 3), device=self.device, dtype=self.torch_dtype)
        if i_idx.numel() == 0:
            energy, _ = de_dg_fn(g)
            return float(energy), forces.cpu().numpy()

        vij = delta[i_idx, j_idx]
        rij = dist[i_idx, j_idx]
        unit = vij / rij[:, None]
        fc, dfc = self._cutoff(rij)

        # --- FIRST pass: descriptors only (no derivatives) ----------------------------------
        diff = rij[:, None] - self.mu[None, :]
        gauss = torch.exp(-self.etas[:, None, None] * diff[None] ** 2)
        value = (gauss * fc[None, :, None]).permute(1, 0, 2).reshape(len(rij), -1)
        dvalue = (
            gauss * (-2.0 * self.etas[:, None, None] * diff[None] * fc[None, :, None] + dfc[None, :, None])
        ).permute(1, 0, 2).reshape(len(rij), -1)
        cols_rad = (sp_of[j_idx] * self.n_rad)[:, None] + torch.arange(self.n_rad, device=self.device)[None, :]
        g.index_put_((i_idx[:, None].expand_as(cols_rad), cols_rad), value, accumulate=True)

        triples = self._triplets(i_idx, n)

        angular_cache = []
        if triples is not None:
            for block_start in range(0, triples.shape[1], self.triplet_block):
                pa, pb = triples[:, block_start : block_start + self.triplet_block]
                centre = i_idx[pa]
                ja, jb = j_idx[pa], j_idx[pb]
                va, vb = vij[pa], vij[pb]
                ra, rb = rij[pa], rij[pb]
                vjk = vb - va
                rjk = vjk.norm(dim=1)
                keep = (rjk < self.spec.cutoff) & (rjk > 1e-8)
                if not bool(keep.any()):
                    continue
                pa, pb, centre, ja, jb = pa[keep], pb[keep], centre[keep], ja[keep], jb[keep]
                va, vb, ra, rb, vjk, rjk = va[keep], vb[keep], ra[keep], rb[keep], vjk[keep], rjk[keep]
                fc_jk, dfc_jk = self._cutoff(rjk)
                cos_t = (va * vb).sum(dim=1) / (ra * rb)
                fcc = fc[pa] * fc[pb] * fc_jk
                r2sum = ra**2 + rb**2 + rjk**2
                channel = self.pair_table[sp_of[ja], sp_of[jb]]
                slot = 0
                for eta_a in self.eta_ang:
                    expo = torch.exp(-eta_a * r2sum)
                    for zeta in self.zetas:
                        for lam in self.lambdas:
                            base = 1.0 + lam * cos_t
                            col = self.n_rad_block + channel * self.n_ang + slot
                            g.index_put_((centre, col), base**zeta * expo * fcc, accumulate=True)
                            slot += 1
                angular_cache.append(
                    (pa, pb, centre, ja, jb, va, vb, ra, rb, vjk, rjk, fc_jk, dfc_jk, cos_t, fcc, r2sum, channel)
                )

        # --- network: energy and dE/dG ------------------------------------------------------
        energy, de_dg = de_dg_fn(g)

        # --- SECOND pass: forces, with dE/dG already known ----------------------------------
        w_rad = torch.gather(de_dg[i_idx], 1, cols_rad)                   # (P, n_rad)
        radial_scalar = (w_rad * dvalue).sum(dim=1)                       # (P,)
        contrib = radial_scalar[:, None] * unit
        forces.index_add_(0, j_idx, -contrib)
        forces.index_add_(0, i_idx, contrib)

        for (pa, pb, centre, ja, jb, va, vb, ra, rb, vjk, rjk, fc_jk, dfc_jk,
             cos_t, fcc, r2sum, channel) in angular_cache:
            dcos_dva = vb / (ra * rb)[:, None] - cos_t[:, None] * va / (ra**2)[:, None]
            dcos_dvb = va / (ra * rb)[:, None] - cos_t[:, None] * vb / (rb**2)[:, None]
            dfcc_dva = (dfc[pa] * fc[pb] * fc_jk)[:, None] * (va / ra[:, None])
            dfcc_dvb = (dfc[pb] * fc[pa] * fc_jk)[:, None] * (vb / rb[:, None])
            dfcc_dvjk = (dfc_jk * fc[pa] * fc[pb])[:, None] * (vjk / rjk[:, None])
            slot = 0
            for eta_a in self.eta_ang:
                expo = torch.exp(-eta_a * r2sum)
                dexpo_dva = -2.0 * eta_a * va * expo[:, None]
                dexpo_dvb = -2.0 * eta_a * vb * expo[:, None]
                dexpo_dvjk = -2.0 * eta_a * vjk * expo[:, None]
                for zeta in self.zetas:
                    for lam in self.lambdas:
                        base = 1.0 + lam * cos_t
                        angular = base**zeta
                        col = self.n_rad_block + channel * self.n_ang + slot
                        w = de_dg[centre, col]                                  # (T,)
                        dang = torch.where(base.abs() > 1e-12,
                                           zeta * lam * base ** (zeta - 1.0),
                                           torch.zeros_like(base))
                        common = (angular * expo)[:, None]
                        af = (angular * fcc)[:, None]
                        dva = ((dang * expo * fcc)[:, None] * dcos_dva + af * dexpo_dva
                               + common * dfcc_dva - af * dexpo_dvjk - common * dfcc_dvjk)
                        dvb = ((dang * expo * fcc)[:, None] * dcos_dvb + af * dexpo_dvb
                               + common * dfcc_dvb + af * dexpo_dvjk + common * dfcc_dvjk)
                        forces.index_add_(0, ja, -w[:, None] * dva)
                        forces.index_add_(0, jb, -w[:, None] * dvb)
                        forces.index_add_(0, centre, w[:, None] * (dva + dvb))
                        slot += 1
        return float(energy), forces.cpu().numpy()


    def energy_forces_fused(self, positions, numbers, de_dg_fn, cell=None):
        """
        The same as `energy_forces_direct`, but all angular combinations are computed in a
        SINGLE pass.

        Measurement that made this necessary. At 192 atoms: 8906 pairs, 202831 triplets,
        46.4 neighbors on average, angular part 48.7 million operations. The direct path
        spent about 180 kernel launches on it (12 combinations x 3 axes x 3 accumulations
        plus intermediates), and at a measured 105 ms this came out to less than half a
        gigaflop: the machine was busy with overhead, not computation.

        Here the multiplicities, parities and widths are unrolled into ONE axis of length
        n_angular, so instead of 180 launches about ten remain. The formulas do not change;
        agreement with the direct path is verified separately and must be at the level of
        single precision.
        """
        import torch

        pos = torch.as_tensor(np.asarray(positions), device=self.device, dtype=self.torch_dtype)
        nums = np.asarray(numbers, dtype=int)
        sp_of = torch.tensor([self.species_index[int(z)] for z in nums], device=self.device)
        n = len(nums)
        cell_t = None if cell is None else torch.as_tensor(
            np.asarray(cell), device=self.device, dtype=self.torch_dtype
        )

        delta = pos[None, :, :] - pos[:, None, :]
        if cell_t is not None:
            inv = torch.linalg.inv(cell_t.T)
            frac = torch.einsum("ab,ijb->ija", inv, delta)
            frac = frac - torch.round(frac)
            delta = torch.einsum("ab,ijb->ija", cell_t.T, frac)
        dist = delta.norm(dim=2)
        eye = torch.eye(n, device=self.device, dtype=torch.bool)
        within = (dist < self.spec.cutoff) & (~eye)
        i_idx, j_idx = torch.nonzero(within, as_tuple=True)

        g = torch.zeros((n, self.d_len), device=self.device, dtype=self.torch_dtype)
        forces = torch.zeros((n, 3), device=self.device, dtype=self.torch_dtype)
        if i_idx.numel() == 0:
            energy, _ = de_dg_fn(g)
            return float(energy), forces.cpu().numpy()

        vij = delta[i_idx, j_idx]
        rij = dist[i_idx, j_idx]
        unit = vij / rij[:, None]
        fc, dfc = self._cutoff(rij)

        # --- radial part (same as in the direct path) ---------------------------------------
        diff = rij[:, None] - self.mu[None, :]
        gauss = torch.exp(-self.etas[:, None, None] * diff[None] ** 2)
        value = (gauss * fc[None, :, None]).permute(1, 0, 2).reshape(len(rij), -1)
        dvalue = (
            gauss * (-2.0 * self.etas[:, None, None] * diff[None] * fc[None, :, None] + dfc[None, :, None])
        ).permute(1, 0, 2).reshape(len(rij), -1)
        cols_rad = (sp_of[j_idx] * self.n_rad)[:, None] + torch.arange(self.n_rad, device=self.device)[None, :]
        g.index_put_((i_idx[:, None].expand_as(cols_rad), cols_rad), value, accumulate=True)

        triples = self._triplets(i_idx, n)
        cache = []
        if triples is not None:
            for start in range(0, triples.shape[1], self.triplet_block):
                pa, pb = triples[:, start : start + self.triplet_block]
                centre, ja, jb = i_idx[pa], j_idx[pa], j_idx[pb]
                va, vb = vij[pa], vij[pb]
                ra, rb = rij[pa], rij[pb]
                vjk = vb - va
                rjk = vjk.norm(dim=1)
                keep = (rjk < self.spec.cutoff) & (rjk > 1e-8)
                if not bool(keep.any()):
                    continue
                pa, pb, centre, ja, jb = pa[keep], pb[keep], centre[keep], ja[keep], jb[keep]
                va, vb, ra, rb, vjk, rjk = va[keep], vb[keep], ra[keep], rb[keep], vjk[keep], rjk[keep]
                fc_jk, dfc_jk = self._cutoff(rjk)
                cos_t = (va * vb).sum(dim=1) / (ra * rb)
                fcc = fc[pa] * fc[pb] * fc_jk
                r2sum = ra**2 + rb**2 + rjk**2
                channel = self.pair_table[sp_of[ja], sp_of[jb]]

                # --- ALL combinations on a single axis ----------------------------------
                # expo: (T, n_eta); base/angular: (T, n_zeta, n_lam) -> combined n_angular axis
                expo = torch.exp(-self.eta_ang[None, :] * r2sum[:, None])            # (T, E)
                base = 1.0 + self.lambdas[None, :] * cos_t[:, None]                  # (T, L)
                zeta = self.zetas
                angular = base[:, None, :] ** zeta[None, :, None]                    # (T, Z, L)
                dang = torch.where(
                    base[:, None, :].abs() > 1e-12,
                    zeta[None, :, None] * self.lambdas[None, None, :] * base[:, None, :] ** (zeta[None, :, None] - 1.0),
                    torch.zeros_like(angular),
                )
                # the axis order matches the fill order in the reference: eta -> zeta -> lambda
                ang_flat = (angular[:, None, :, :] * expo[:, :, None, None]).reshape(len(cos_t), -1)
                dang_flat = (dang[:, None, :, :] * expo[:, :, None, None]).reshape(len(cos_t), -1)
                ang_only = angular[:, None, :, :].expand(-1, len(self.eta_ang), -1, -1).reshape(len(cos_t), -1)
                expo_only = expo[:, :, None, None].expand(-1, -1, len(zeta), len(self.lambdas)).reshape(len(cos_t), -1)
                eta_col = self.eta_ang[None, :, None, None].expand(
                    len(cos_t), -1, len(zeta), len(self.lambdas)
                ).reshape(len(cos_t), -1)

                cols = (self.n_rad_block + channel * self.n_ang)[:, None] + torch.arange(
                    self.n_ang, device=self.device
                )[None, :]
                g.index_put_((centre[:, None].expand_as(cols), cols),
                             ang_flat * fcc[:, None], accumulate=True)
                cache.append((centre, ja, jb, va, vb, ra, rb, vjk, rjk, fc[pa], fc[pb], fc_jk,
                              dfc[pa], dfc[pb], dfc_jk, cos_t, fcc, cols,
                              ang_flat, dang_flat, ang_only, expo_only, eta_col))

        energy, de_dg = de_dg_fn(g)

        # --- forces: radial part -------------------------------------------------------------
        w_rad = torch.gather(de_dg[i_idx], 1, cols_rad)
        contrib = (w_rad * dvalue).sum(dim=1)[:, None] * unit
        forces.index_add_(0, j_idx, -contrib)
        forces.index_add_(0, i_idx, contrib)

        # --- forces: angular part, all combinations at once -----------------------------
        for (centre, ja, jb, va, vb, ra, rb, vjk, rjk, fc_a, fc_b, fc_jk,
             dfc_a, dfc_b, dfc_jk, cos_t, fcc, cols,
             ang_flat, dang_flat, ang_only, expo_only, eta_col) in cache:
            w = torch.gather(de_dg[centre], 1, cols)                       # (T, A)
            dcos_dva = vb / (ra * rb)[:, None] - cos_t[:, None] * va / (ra**2)[:, None]
            dcos_dvb = va / (ra * rb)[:, None] - cos_t[:, None] * vb / (rb**2)[:, None]
            dfcc_dva = (dfc_a * fc_b * fc_jk)[:, None] * (va / ra[:, None])
            dfcc_dvb = (dfc_b * fc_a * fc_jk)[:, None] * (vb / rb[:, None])
            dfcc_dvjk = (dfc_jk * fc_a * fc_b)[:, None] * (vjk / rjk[:, None])

            # Reduction over the combinations axis BEFORE multiplying by vectors: (T, A) -> (T,).
            # `ang_flat` does not contain the cutoff function, so the term with its
            # derivative is JUST a product, without division. The first version divided by
            # fcc and multiplied back, and at the radius boundary, where fcc = 0, this gave
            # 0/0 = NaN: forces matched the direct path at 192 atoms and were NaN at 24,
            # i.e. the bug did not show up consistently, the nastiest kind of bug, and only
            # cross-checking the two paths caught it.
            s_dang = (w * dang_flat).sum(dim=1) * fcc
            s_ang = (w * ang_flat).sum(dim=1)
            s_expo_ang = (w * ang_only * expo_only * eta_col).sum(dim=1) * fcc

            dva = (s_dang[:, None] * dcos_dva
                   - 2.0 * s_expo_ang[:, None] * va
                   + s_ang[:, None] * dfcc_dva
                   + 2.0 * s_expo_ang[:, None] * vjk
                   - s_ang[:, None] * dfcc_dvjk)
            dvb = (s_dang[:, None] * dcos_dvb
                   - 2.0 * s_expo_ang[:, None] * vb
                   + s_ang[:, None] * dfcc_dvb
                   - 2.0 * s_expo_ang[:, None] * vjk
                   + s_ang[:, None] * dfcc_dvjk)
            forces.index_add_(0, ja, -dva)
            forces.index_add_(0, jb, -dvb)
            forces.index_add_(0, centre, dva + dvb)

        return float(energy), forces.cpu().numpy()
