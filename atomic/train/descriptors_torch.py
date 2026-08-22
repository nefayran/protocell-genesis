"""
Дескрипторы на тензорах: тот же расчёт без питоновского цикла по атомам.

Зачем. Замер показал, что вся выгода нашей модели съедается реализацией: на 3 атомах мы
быстрее учителя в 57 раз, на 24 -- в 3, на 81 -- в 1.2, а на 192 атомах уже МЕДЛЕННЕЕ (0.90).
Причина не в модели: на 192 атомах ей нужно около семи миллионов операций на расчёт сил, то
есть меньше миллисекунды при скромной производительности, против замеренных 618 мс. Разница --
это питоновский цикл по атомам и по парам соседей.

Здесь тот же самый набор симметрийных функций считается пакетно: все атомы разом, все пары
соседей разом, на процессоре или на карте. Формулы не меняются -- меняется только способ, и это
проверяется сравнением с эталонной реализацией (validate/test_descriptors_torch.py): расхождение
обязано быть на уровне одинарной точности, а не «примерно совпадать».

Устройство расчёта. Соседи ищутся один раз на кадр и хранятся плоскими списками пар: так
периодичность учитывается одним вычитанием, а не поиском по ячейкам. Для угловой части строятся
все ТРОЙКИ (центр, сосед a, сосед b) -- их число растёт как квадрат числа соседей, поэтому они
собираются блоками, чтобы не выйти за память.
"""
from __future__ import annotations

import numpy as np

from .descriptors import DescriptorSpec, descriptor_length, n_pair_channels


def _pair_channel_matrix(species: tuple[int, ...]) -> np.ndarray:
    """Таблица канала для пары сортов: (n_sp, n_sp) -> индекс неупорядоченной пары."""
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
    Пакетный расчёт дескрипторов и их производных.

    `triplet_block` -- сколько троек обрабатывать за раз. Значение подбирается ЗАМЕРОМ под
    доступную память: тройки это главный расход, их число на кадр порядка N·nb²/2.
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

    # --- вспомогательное ------------------------------------------------------------------
    def _cutoff(self, r):
        import torch

        rc = self.spec.cutoff
        inside = r < rc
        f = torch.where(inside, 0.5 * (torch.cos(np.pi * r / rc) + 1.0), torch.zeros_like(r))
        df = torch.where(inside, -0.5 * np.pi / rc * torch.sin(np.pi * r / rc), torch.zeros_like(r))
        return f, df

    def compute(self, positions, numbers, cell=None):
        """
        Возвращает (G, dG) как тензоры: G размера (N, D), dG размера (N, D, N, 3).

        Форма dG та же, что у эталонной реализации, чтобы обучение и проверки работали без
        изменений. Она расточительна по памяти -- в производственном расчёте силы собираются
        на лету, но здесь важна сверяемость с эталоном.
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

        # --- все пары внутри радиуса, одним проходом --------------------------------------
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

        # --- радиальная часть -------------------------------------------------------------
        diff = rij[:, None] - self.mu[None, :]                                  # (P, n_radial)
        gauss = torch.exp(-self.etas[:, None, None] * diff[None] ** 2)          # (S, P, n_radial)
        value = (gauss * fc[None, :, None]).permute(1, 0, 2).reshape(len(rij), -1)
        dvalue = (
            gauss * (-2.0 * self.etas[:, None, None] * diff[None] * fc[None, :, None] + dfc[None, :, None])
        ).permute(1, 0, 2).reshape(len(rij), -1)

        base_col = sp_of[j_idx] * self.n_rad                                    # канал по сорту СОСЕДА
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

        # --- угловая часть: тройки блоками ------------------------------------------------
        # соседи каждого центра идут подряд, поэтому тройки строятся по границам групп
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
        Силы БЕЗ построения тензора производных дескрипторов.

        Ключевая мысль, из-за которой этот метод существует: в расчёте нужны силы, а не
        производные дескрипторов. Если сначала получить dE/dG (один дешёвый проход сети по
        всем атомам), то силу можно собирать прямо в обходе пар и троек:

            F_k = - sum_i sum_d (dE/dG_i,d) * (dG_i,d / dr_k)

        и каждое слагаемое учитывается сразу, а тензор (N, D, N, 3) никогда не появляется.
        Замер, из-за которого это понадобилось: с этим тензором тензорная версия давала лишь
        2.1-2.5x против эталонной, потому что вся работа уходила в запись в память, а не в счёт
        (81 атом, D=132 -- это 2.6 миллиона чисел на кадр).

        `de_dg_fn` -- функция, которая по (N, D) дескрипторам возвращает (E, dE/dG). Разделение
        нужно, чтобы этот метод ничего не знал о виде модели: сеть, линейная модель или что
        угодно ещё подставляется снаружи.
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

        # --- ПЕРВЫЙ проход: только дескрипторы (без производных) --------------------------
        diff = rij[:, None] - self.mu[None, :]
        gauss = torch.exp(-self.etas[:, None, None] * diff[None] ** 2)
        value = (gauss * fc[None, :, None]).permute(1, 0, 2).reshape(len(rij), -1)
        dvalue = (
            gauss * (-2.0 * self.etas[:, None, None] * diff[None] * fc[None, :, None] + dfc[None, :, None])
        ).permute(1, 0, 2).reshape(len(rij), -1)
        cols_rad = (sp_of[j_idx] * self.n_rad)[:, None] + torch.arange(self.n_rad, device=self.device)[None, :]
        g.index_put_((i_idx[:, None].expand_as(cols_rad), cols_rad), value, accumulate=True)

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
        triples = torch.cat(triples, dim=1) if triples else None

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

        # --- сеть: энергия и dE/dG --------------------------------------------------------
        energy, de_dg = de_dg_fn(g)

        # --- ВТОРОЙ проход: силы, с уже известным dE/dG -----------------------------------
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
