// The zero-tension Metropolis Monte Carlo area move for engine/src/sim.ts — split out by
// responsibility (file-size rule, root CLAUDE.md). Moved verbatim from the original sim.ts,
// including the configurational-Jacobian entropic term and the rigid lipid-COM scaling; neither
// touched (per the task's explicit instruction not to touch the MC area move's entropic term).

import { readBack } from './gpu'
import { rebindGridDependent } from './sim-bindgroups'
import { resizeGrid } from './sim-buffers'
import { gridInvariantsHold, computeDims } from './sim-grid-geometry'
import { encodeForceGrid } from './sim-forces'
import { rebuildGridUntimed } from './sim-integrate'
import { scaleLateralRigid } from './sim-area-math'
import type { EngineRuntime } from './sim-runtime'

// Proposal half-width for the area move's log-area step u ~ U(-AREA_MOVE_LOG_DELTA,
// +AREA_MOVE_LOG_DELTA). This is a Monte Carlo move-size tuning knob, not a physical model
// parameter (it does not appear in any Cooke & Deserno formula and has no effect on the
// equilibrium distribution, only on how fast the chain explores it), so it lives here rather
// than in data/params.json. Tuned (see task-5-report.md) so the accepted fraction lands in the
// 0.2-0.6 range: measured 0.54 / 0.36 / 0.23 at half-widths 0.006 / 0.010 / 0.016 once the
// rigid-lipid displacement below became exact. Earlier, much smaller values were forced by the
// periodic-image defect in that displacement, which put a spurious quadratic cost on every
// proposal and pushed acceptance under 0.2 for anything but a tiny step; with the defect fixed a
// step three to four times larger is accepted at the same rate, and the chain reaches its area
// plateau in ~150 moves (before the fix it never reached one — it slid out of the corridor).
const AREA_MOVE_LOG_DELTA = 0.012

async function totalPotentialGPU(rt: EngineRuntime): Promise<number> {
  await rebuildGridUntimed(rt)
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeForceGrid(rt, pass)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  const u = await readBack(rt.device, rt.potentialBuf, rt.N * 4)
  let sum = 0
  for (let i = 0; i < rt.N; i++) sum += u[i]
  return sum
}

export async function areaMove(rt: EngineRuntime, trials: number): Promise<number> {
  let accepted = 0
  for (let t = 0; t < trials; t++) {
    const oldBox: [number, number, number] = [rt.liveBox[0], rt.liveBox[1], rt.liveBox[2]]

    const u = (rt.rng() * 2 - 1) * AREA_MOVE_LOG_DELTA
    const sq = Math.sqrt(Math.exp(u))
    const proposedBox: [number, number, number] = [oldBox[0] * sq, oldBox[1] * sq, oldBox[2]]

    // Reject up front, with no GPU work at all, if the proposal would breach either grid
    // invariant createSystem() enforces at construction time (dims>=3 on the periodic axes,
    // min(box.x,box.y)/2 > bend.r0). AREA_MOVE_LOG_DELTA is tuned small enough that this should
    // essentially never fire once the chain is anywhere near the literature corridor — it is a
    // safety net, not the normal path.
    if (!gridInvariantsHold(proposedBox, computeDims(proposedBox, rt.cellSize), rt.p.bend.r0)) continue

    const potentialBefore = await totalPotentialGPU(rt)
    const before = await readBack(rt.device, rt.posBuf, rt.N * 16)

    // Rigid lipid displacement: scale each LIPID's center of mass (x,y) by sqrt(s) and REBUILD its
    // three beads around the scaled center from their unchanged internal offsets. Bead coordinates
    // are never rescaled themselves, so every intramolecular distance (FENE head-tail1, FENE
    // tail1-tail2, bend head-tail2) survives the move exactly and ΔU below reflects only
    // intermolecular structure — which is what the area coordinate is supposed to couple to.
    //
    // Both the center of mass and the offsets must be handled in the periodic sense, and BOTH
    // matter (each of the two was measured to break the move on its own):
    //  - the center is built from the head coordinate plus minimum-image offsets (mi1) and then
    //    WRAPPED into [0, oldBox). Only a wrapped center makes this move the identity in scaled
    //    coordinates s = R/L (s stays in [0,1), so s' = sqrt(s)R/(sqrt(s)L) = s), which is what
    //    makes the configurational Jacobian exactly (A'/A)^N_lipids and the move an involution
    //    under u -> -u. An unwrapped center displaces its lipid by up to L*(sqrt(s)-1) away from
    //    the homogeneous value every other lipid gets.
    //  - each bead is then placed at comScaled + offset rather than at storedCoordinate +
    //    comX*(sqrt(s)-1). Those two differ by exactly L*(sqrt(s)-1) ~ 0.04 sigma for any bead
    //    whose stored coordinate sits on the far side of a periodic boundary from its own center
    //    (about a tenth of the lipids at these box sizes): translating the stored coordinate and
    //    re-wrapping lands such a bead one box length off its lipid, i.e. STRETCHES or COMPRESSES
    //    that bond by 0.04 sigma. The measured damage was severe and not obvious: the bond
    //    perturbation is quadratic in u and one-signed, which showed up as a quenched curvature of
    //    d2U/dlnA2 of order a million (so a u = 0.003 proposal cost ~13 kT whichever direction it
    //    went), a per-configuration scatter of +/-1500 in dU/dlnA, and a Metropolis chain that
    //    walked the area monotonically out of the literature corridor while every aggregate
    //    diagnostic (acceptance fraction, energy conservation) looked healthy.
    // Explicit re-wrap into [0, newBox) inside scaleLateralRigid() mirrors wrap_main in
    // integrate.wgsl — so a bead landing just past the edge sits in the cell cell_coord()
    // actually expects (it clamps rather than wraps) instead of waiting for the next step()'s
    // wrap pass.
    const proposed = scaleLateralRigid(before, oldBox, proposedBox, rt.opts.lipids)
    rt.device.queue.writeBuffer(rt.posBuf, 0, proposed)
    rt.liveBox = proposedBox
    await resizeGrid(rt, proposedBox, () => rebindGridDependent(rt))

    const potentialAfter = await totalPotentialGPU(rt)
    const dU = potentialAfter - potentialBefore
    // Zero-tension Metropolis with the configurational Jacobian from rescaling N=lipids
    // centers of mass by sqrt(A'/A): accept with min(1, exp(-(dU - lipids*kT*ln(A'/A))/kT)) =
    // min(1, exp(lipids*u - dU/kT)), since ln(A'/A) = ln(s) = u exactly. The entropic term
    // favors expansion (positive u) and is what balances the tail-tail attraction's pull
    // toward smaller area — see task-5-report.md for the measurement that showed this term is
    // required (its absence produces a monotonic collapse with a perfectly healthy aggregate
    // acceptance fraction, not a near-zero one).
    // livep.thermostat.kT, not p.thermostat.kT: p is fixed at creation time, but setLiveParams()
    // can retune kT on a running system (viewer/'s slider), rewriting the GPU-side thermostat's
    // uniform buffer without recreating the system. Using the creation-time p here would judge
    // every proposal against a kT the dynamics no longer sample at, silently decoupling this
    // Metropolis criterion from the physics it is supposed to match the moment a caller retunes.
    const acceptProb = Math.min(1, Math.exp(rt.opts.lipids * u - dU / rt.livep.thermostat.kT))
    if (rt.rng() < acceptProb) {
      accepted++
      // Proposed state kept; forceBuf/potentialBuf/grid already reflect it from the
      // totalPotentialGPU() call above — nothing left to resync.
    } else {
      rt.device.queue.writeBuffer(rt.posBuf, 0, before)
      rt.liveBox = oldBox
      await resizeGrid(rt, oldBox, () => rebindGridDependent(rt))
      // Resync forceBuf/potentialBuf/grid with the reverted positions/box so the next step()
      // kicks off of F(x) at the state actually being kept, not the discarded proposal.
      await totalPotentialGPU(rt)
    }
  }
  return accepted / trials
}
