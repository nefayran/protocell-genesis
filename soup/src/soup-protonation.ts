// Task 'electrostatics' (2026-08-20): the GPU-side glue for the constant-pH Monte Carlo. The scheme,
// its detailed-balance argument and every literature citation live in soup/src/electrostatics.ts's
// header and in data/soup.json's electrostatics.basis; this file is only the plumbing:
// read positions+charges back, run ONE pure sweep, upload the new charges, and RECOMPUTE THE FORCE.
//
// WHY THE FORCE RECOMPUTE IS NOT OPTIONAL. The integrator's first half-kick of the next step uses
// F(x_n) as it stands in forceBuf. Changing a charge changes F without moving a particle, so skipping
// the recompute would leave exactly the stale-F(x) defect tests/soup-stale-force.test.ts exists for
// (this project measured a 2.28x inflation of one statistic from that class of bug after a box
// change -- see its own report). One force dispatch per sweep, i.e. one per `sweepEverySteps` steps,
// is the same thing soup/src/soup-box-scale.ts's applyBoxScaleOnce already does for the same reason.
//
// WHERE IT IS CALLED FROM: soup/src/soup-integrate.ts's stepper, at the chunk boundary it ALREADY
// synchronises on for the non-finite guard -- so a sweep adds no extra pipeline stall.

import { protonationSweep, headIndices, hendersonAlpha, pcgNext, type EsBasis, type PcgState, type SweepResult } from './electrostatics'
import { encodeGridRebuild, encodeVerletRebuild } from './soup-grid-verlet'
import { encodeSoupForce, encodeSoupForceList } from './soup-integrate'
import type { SoupRuntime } from './soup-runtime'
import * as readback from './soup-readback'

/** Mutable protonation state carried on the runtime: the Monte Carlo's own RNG (checkpointed, so a
 * resumed run continues the same chain rather than restarting it), when the next sweep is due, and
 * the last sweep's statistics for the run log / the trace. */
export interface ProtonationState {
  es: EsBasis
  rng: PcgState
  nextSweepAt: number
  sweeps: number
  last?: SweepResult
}

/** The initial charge array. On a resume it is the checkpoint's own array verbatim. On a fresh run
 * every bead of the titratable species is drawn INDEPENDENTLY from the Henderson-Hasselbalch
 * equilibrium at this run's pH -- correct at step 0 by construction, because step 0 is a
 * monomers-only lattice where no head has a neighbour close enough to matter, so the interaction-free
 * equilibrium IS the right distribution there. Everything else stays 0. */
export function initialCharges(positions: Float32Array, es: EsBasis, seed: number): { charges: Float32Array; rng: PcgState } {
  const n = positions.length / 4
  const charges = new Float32Array(n)
  const rng: PcgState = { state: (seed ^ 0x9e3779b9) >>> 0 }
  if (!es.enabled) return { charges, rng }
  const alpha = hendersonAlpha(es.pH, es.pKaIntrinsic)
  const heads = headIndices(positions, es.chargedKind)
  for (let h = 0; h < heads.length; h++) {
    if (pcgNext(rng) < alpha) charges[heads[h]] = es.chargeDeprotonated
  }
  return { charges, rng }
}

/** Runs a sweep if one is due at the current global step, and returns what it did (or null). */
export async function maybeProtonationSweep(rt: SoupRuntime): Promise<SweepResult | null> {
  const pr = rt.protonation
  if (!pr || !pr.es.enabled) return null
  if (rt.live.globalStep < pr.nextSweepAt) return null
  pr.nextSweepAt = rt.live.globalStep + pr.es.sweepEverySteps
  const [positions, charges] = await Promise.all([readback.particles(rt), readback.charges(rt)])
  const box: [number, number, number] = [rt.live.liveBox[0], rt.live.liveBox[1], rt.live.liveBox[2]]
  const result = protonationSweep(positions, charges, box, pr.es, pr.rng)
  rt.device.queue.writeBuffer(rt.buf.chargeBuf, 0, charges, 0, rt.N)
  // F(x) is now stale with respect to the charges just written -- recompute it, see this file's header.
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
  await rt.device.queue.onSubmittedWorkDone()
  pr.sweeps++
  pr.last = result
  return result
}
