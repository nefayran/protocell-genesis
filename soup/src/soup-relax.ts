// Cold-start energy minimisation, driven from the CPU side: iterate soup/wgsl/relax.wgsl's
// displacement-capped steepest-descent kernel with a decaying cap, then hand back a valid F(x0) for
// the first Verlet half-kick.
//
// WHAT THIS IS AND IS NOT, stated as flatly as possible because the whole value of the stage rests on
// it:
//  - it runs ONLY when a caller asks (CreateSoupOpts has no relaxation field at all; SoupSystem
//    exposes `relaxColdStart()` and soup/cli/campaign.ts exposes `--relax`). No default path anywhere
//    in this codebase changed, so every existing test, gate, checkpoint lineage and published number
//    is bit-identical to before this task BY CONSTRUCTION, not by comparison;
//  - it REFUSES to run on a system that has already taken a step (`rt.live.globalStep !== 0` throws).
//    That is what makes "the relaxation happens before step 1" a checked property rather than a
//    promise: it is impossible for any trajectory statistic to contain a minimisation iteration;
//  - it advances no step counter, consumes no RNG stream (neither the bond Monte Carlo's nor the
//    thermostat's -- the kernel touches neither buffer), attempts no bond, and does not write
//    velocities: the Maxwell-Boltzmann draw soup/src/soup-init-state.ts made survives untouched, so
//    the initial temperature is exactly what it was;
//  - it modifies no potential, no rate, no threshold, no corridor and no constant. The force it
//    descends is produced by the SAME encodeSoupForceList/encodeSoupForce call the real steps use.
//    There is no soft core, no capped potential and no modified dt, so there is nothing that could
//    still be "on" during a measured step -- the mechanism does not exist after this function
//    returns.
//
// The two numbers it needs live in data/soup.json's `coldStartRelax` (rank D, with its own measured
// basis), never as a literal here -- same discipline as every other tunable in this engine.

import { encodeGridRebuild, encodeVerletRebuild, assertVerletSafety } from './soup-grid-verlet'
import { encodeSoupForce, encodeSoupForceList } from './soup-integrate'
import { assertStateFinite } from './soup-health'
import type { SoupRuntime } from './soup-runtime'

export interface RelaxColdStartResult {
  iterations: number
  /** The displacement cap of the FIRST iteration, in sigma; it decays linearly to 0 over
   * `iterations`, so this is also the largest jump any single particle can have taken. */
  maxDisplacementStart: number
  /** Upper bound on how far ONE particle can have moved in total: sum of the decaying caps. Not the
   * measured displacement -- a bound, so a reader can see the stage cannot have teleported anything
   * across the box. */
  displacementBound: number
  /** max over all particles and all three components of |F|, from the same force kernel the real
   * steps use, immediately BEFORE the first iteration and immediately AFTER the last. This pair is
   * the whole proof that the stage did what it exists to do. */
  maxForceBefore: number
  maxForceAfter: number
  nonFiniteBefore: number
  nonFiniteAfter: number
}

/** max|F| over every component, and how many components are non-finite -- computed from a plain
 * force readback INSIDE the page, returning two scalars. Never transfers the array anywhere. */
function forceStats(f: Float32Array, n: number): { max: number; nonFinite: number } {
  let max = 0
  let nonFinite = 0
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = f[i * 4 + c]
      if (!Number.isFinite(v)) {
        nonFinite++
        continue
      }
      const a = Math.abs(v)
      if (a > max) max = a
    }
  }
  return { max, nonFinite }
}

/** Number of iterations to await the GPU on. Purely a memory-pressure knob on the SUBMISSION queue
 * (each iteration is its own tiny command buffer because the displacement cap changes between them,
 * and an unbounded pile of un-awaited submits is what this bounds) -- it cannot change what is
 * computed, exactly like soup/src/soup-integrate.ts's own STEP_CHUNK. */
const RELAX_SYNC_EVERY = 25

/** The minimisation loop itself: `iterations` displacement-capped steepest-descent steps with the
 * cap decaying linearly from `d0` (sigma) to zero, then one final grid/list/force rebuild so the
 * caller's next kick sees the force of the RELAXED configuration, then the loud finiteness/Verlet
 * checks. Extracted from relaxColdStart (below) verbatim -- not one dispatch, order or check changed
 * -- because task 'evaporation' (2026-08-20) needs the SAME minimiser after re-inserting solvent
 * beads mid-run (soup/src/soup-evaporate.ts's rehydrateSolventTo), and duplicating a minimiser is
 * exactly how two of them drift apart.
 *
 * This function carries NO precondition of its own: relaxColdStart keeps the "globalStep must be 0"
 * refusal that makes cold-start minimisation provably outside the trajectory, and the rehydration
 * caller instead makes its own use provably harmless a different way (it marks every pre-existing
 * particle immobile, so only the freshly inserted beads can move at all). */
export async function relaxIterations(rt: SoupRuntime, iterations: number, d0: number): Promise<void> {
  for (let k = 0; k < iterations; k++) {
    // Linear decay to zero over the iteration count -- see soup/wgsl/relax.wgsl's header for why a
    // normalised descent needs the cap to close, and why the alternative (an adaptive step-size
    // search) needs a global energy reduction this engine has no kernel for.
    const d = d0 * (1 - k / iterations)
    rt.device.queue.writeBuffer(rt.buf.relaxUniform, 0, new Float32Array([d, 0, 0, 0]))
    const enc = rt.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    // Rebuild the neighbour structure EVERY iteration: a minimisation displacement is bounded by
    // the cap, not by the Verlet skin, so a list reused across iterations could miss a pair -- and
    // the pairs at stake here are precisely the overlapping ones. The rebuild also re-takes the
    // drift snapshot and zeroes the drift counter, which is why assertVerletSafety below reads a
    // clean slate rather than the accumulated minimisation travel.
    if (rt.verlet.enabled) {
      encodeVerletRebuild(rt, pass)
      encodeSoupForceList(rt, pass)
    } else {
      encodeGridRebuild(rt, pass)
      encodeSoupForce(rt, pass)
    }
    pass.setPipeline(rt.pipe.relaxStep)
    pass.setBindGroup(1, rt.bind.relaxStepBind)
    pass.dispatchWorkgroups(rt.wgN)
    pass.end()
    rt.device.queue.submit([enc.finish()])
    if ((k + 1) % RELAX_SYNC_EVERY === 0) await rt.device.queue.onSubmittedWorkDone()
  }

  // The positions moved, so the caller's priming F(x0) is stale: rebuild the grid/list and the
  // force one final time, so the first kick after this sees the force of the RELAXED configuration.
  // Without this the first half-kick would use the force of the last pre-minimisation geometry --
  // i.e. the very overlap spike this stage exists to remove.
  {
    const enc = rt.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    if (rt.verlet.enabled) {
      encodeVerletRebuild(rt, pass)
      encodeSoupForceList(rt, pass)
    } else {
      encodeGridRebuild(rt, pass)
      encodeSoupForce(rt, pass)
    }
    pass.end()
    rt.device.queue.submit([enc.finish()])
  }
  await rt.device.queue.onSubmittedWorkDone()
  // Fail loudly here too: an unrelaxable configuration (a pair at exactly r = 0, whose force
  // direction is undefined -- see relax.wgsl's guard) must not be handed to the step loop as if it
  // were fine.
  await assertStateFinite(rt)
  if (rt.verlet.enabled) await assertVerletSafety(rt)
}

export function makeRelaxColdStart(
  rt: SoupRuntime,
  forces: () => Promise<Float32Array>,
): (opts?: { iterations?: number; maxDisplacementSigma?: number }) => Promise<RelaxColdStartResult> {
  return async function relaxColdStart(opts?: { iterations?: number; maxDisplacementSigma?: number }): Promise<RelaxColdStartResult> {
    const cfg = rt.soup.coldStartRelax
    if (!cfg && (opts?.iterations === undefined || opts?.maxDisplacementSigma === undefined)) {
      throw new Error(
        'relaxColdStart: data/soup.json has no coldStartRelax section, and the call did not set iterations and maxDisplacementSigma explicitly',
      )
    }
    const iterations = opts?.iterations ?? cfg!.iterations
    const maxDisplacementSigma = opts?.maxDisplacementSigma ?? cfg!.maxDisplacementSigma
    if (!Number.isInteger(iterations) || iterations < 1) {
      throw new Error(`relaxColdStart: iterations=${iterations} must be an integer >= 1`)
    }
    if (!(maxDisplacementSigma > 0)) {
      throw new Error(`relaxColdStart: maxDisplacementSigma=${maxDisplacementSigma} must be positive`)
    }
    // The one hard precondition, checked rather than documented: a minimisation applied after the
    // trajectory started would BE part of the trajectory.
    if (rt.live.globalStep !== 0) {
      throw new Error(
        `relaxColdStart: the system is already at step ${rt.live.globalStep} -- minimisation is allowed only before the first step, ` +
          `otherwise it lands inside the trajectory on which the measurements are taken`,
      )
    }
    const d0 = maxDisplacementSigma * rt.p.sigma

    const before = forceStats(await forces(), rt.N)
    await relaxIterations(rt, iterations, d0)
    const after = forceStats(await forces(), rt.N)
    // sum_{k=0}^{it-1} d0*(1 - k/it) = d0*(it+1)/2
    const displacementBound = (d0 * (iterations + 1)) / 2
    return {
      iterations,
      maxDisplacementStart: d0,
      displacementBound,
      maxForceBefore: before.max,
      maxForceAfter: after.max,
      nonFiniteBefore: before.nonFinite,
      nonFiniteAfter: after.nonFinite,
    }
  }
}
