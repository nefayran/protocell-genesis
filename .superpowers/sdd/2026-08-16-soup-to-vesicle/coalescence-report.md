# Why the amphiphiles do not coalesce — mechanism, and whether closure is reachable

Task `coalescence` (2026-08-20). Branch `stage-a-atoms`, work in place, no worktree.
Predecessor: `decisive-run-report.md`, whose **concern 9** is the whole subject here — *"why the
amphiphiles do not coalesce is not answered, only measured"* — and whose verdict named the binding
constraint **(2) AGGREGATE SIZE — COALESCENCE, 136.2 against 920, 6.75× short**, with the number that
made a bigger box the wrong next move: box 30 → 54 multiplied supply **5.06×** and the largest
aggregate **1.24×**, and the single-aggregate share FELL 38.5 % → 9.5 %.

`data/params.json` rank-A constants: **NOT touched.** `co_bond.attemptRate`: **NOT touched.**
No threshold widened, no corridor relaxed, no tolerance changed.

---

## 0. THE THREE PREDICTIONS, STATED BEFORE ANY MEASUREMENT

Written into this file and committed before the first measurement ran, so that "which survived" is a
result and not a reading-back. Each prediction is chosen so that its own signature is **incompatible**
with the other two, not merely consistent with itself.

### P1 — FREE-HEAD POISONING would show

The composition runs O:C = 4:1, so tens of thousands of unreacted single polar head beads sit in the
box (43 380 O against 10 845 C at box 54). In real surfactant chemistry free polar monomers adsorbing
on an aggregate surface cap growth and stabilise small aggregates. If that is the mechanism here:

1. **Surface enrichment.** The free-head number density in an aggregate's first contact shell must be
   *enriched* over its bulk value, and enriched **relative to water**, which is the competing
   adsorbate: the normalisation-free ratio `g_O(r)/g_W(r)` (both species probing the identical
   geometry, so no shell-volume normalisation is needed at all) must be **> 1** for r inside the
   first shell, and rise as r falls.
2. **Material coverage.** Free heads in contact per surface amphiphile of the aggregate must be of
   order 0.1–1 — a capping layer, not a stray visitor. A coverage of ~0 cannot cap anything.
3. **The matched arm.** At matched amphiphile count, cutting the free-head concentration ~4× must
   RAISE the largest aggregate and the single-aggregate share.

### P2 — A HYDRATION / FUSION BARRIER would show

Two aggregates that meet fail to merge because the water between their polar surfaces must be
expelled first. If that is the mechanism:

1. **Contacts frequent, merges rare.** Aggregate–aggregate surface contacts must be common per
   snapshot while merge events over the same time must be ~0. Barrier = arrivals without reactions.
2. **Sizes frozen, not exchanging.** Merges AND fissions must both be near zero: a kinetically
   trapped distribution does not reshuffle its material.
3. **The probe.** Two equilibrated aggregates placed at contact must keep a persistent water layer in
   the gap and must NOT merge over a time long compared with their contact time.
4. **A memory.** The size distribution must carry a memory of how the aggregates were made (it is
   frozen), not a stationary shape the run keeps returning to.

### P3 — SPONTANEOUS CURVATURE / PACKING would show

Mean per-tail length is 2.372 beads with a strongly hydrated head (water–head depth 1.1429 of the
tail–tail reference), which biases the preferred aggregate towards micelles and rods. If that is the
mechanism:

1. **Merges AND fissions, balanced.** Both must occur and roughly cancel: the size distribution is
   *stationary* because it is the equilibrium one, with material exchanging freely.
2. **Stationary peaked distribution.** The largest aggregate must not grow even where contacts are
   frequent, and the whole distribution must be reproducible sample to sample.
3. **Rod shape at ALL sizes.** `inPlaneSymmetry` must sit well below the 0.50 a sheet needs for every
   aggregate, not just the largest — a preferred *shape*, not a large object that failed to flatten.
4. **The tail sweep.** At matched amphiphile count, a longer mean tail must move `inPlaneSymmetry` UP
   toward 0.50 and raise the largest aggregate; short tails must keep both low.

**How they are told apart.** P2 and P3 make OPPOSITE predictions about merges and fissions (frozen vs
freely exchanging and balanced) — one measurement separates them. P1 is separated from both by the
surface-coverage number, which is a static property of one snapshot and cannot be argued away.

