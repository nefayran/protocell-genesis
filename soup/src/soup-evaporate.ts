// Task 'evaporation' (2026-08-20): the solvent LEAVES the system in the dry phase and COMES BACK on
// rehydration. This is the mechanism the predecessor task named as structurally missing
// (.superpowers/sdd/2026-08-16-soup-to-vesicle/wet-dry-cycling-report.md §3.3.1: "particle count is
// fixed at createSoup, so water cannot leave ... 1.10x in volume against L1's ~1400x, i.e. 1.3 % of
// it on a log-concentration scale"). Every number this file uses comes from data/soup.json's
// dryWetCycle section (see its basis, items 1-8, for each derivation) or is derived here from
// data/params.json's rank-A sigma -- no numeric model constant is written in this file.
//
// HOW THE PARTICLE COUNT CHANGES, and why this is a real resize rather than an active/inactive mask
// bolted onto the shaders. Every soup kernel bounds itself with `arrayLength()` of one of four
// per-particle vec4 buffers (soup/wgsl/step.wgsl's posRW/velRW/posSortedRW/posAtRebuildRW,
// forces.wgsl's pos2) -- there is no particle-count uniform anywhere, and soup/wgsl/step.wgsl stands
// at 596 lines against CLAUDE.md's hard 600, so adding one would have required a responsibility split
// first. WGSL's arrayLength() of a runtime-sized array is the length of the BOUND RANGE, so binding
// those four buffers over their first `activeN` particles (soup/src/soup-bindgroups.ts's pbuf) makes
// the removed beads invisible to EVERY kernel -- no force, no grid cell, no Verlet slot, no
// integration, no census -- while reallocating nothing. That is the honest "real resize": the live
// particle count really changes and the grid is really re-planned for it (the tested resizeSoupGrid
// path), and the only thing kept is BUFFER CAPACITY, deliberately, because (a) rehydration needs
// those slots back within the same run and (b) verletListBuf alone is N*listCapacity*4 = 329 MB at
// N = 32884, so destroying and recreating it twice per cycle would move 25 GB of GPU allocations for
// no gain. Buffers whose arrayLength is never read as a particle bound (forces, cells, bond slots,
// the two RNG streams, centerLink, centerHeldSteps, frozen, the Verlet list itself) stay bound whole:
// an oversized array is harmless, since every kernel that indexes them is itself already bounded by
// pos2/posRW.
//
// WHY THE SOLVENT BLOCK MUST BE LAST, and what that buys. soup/src/soup-init-state.ts lays particles
// out in kind-block order, so the solvent occupies one contiguous index range. When it is the LAST
// non-empty block, shrinking activeN removes ONLY solvent indices and moves NO other index -- so the
// bond graph, the valence slot rows, the catalyst<->chain links and every organic bead are untouched
// BY CONSTRUCTION, not by a check afterwards. That precondition is asserted (and throws), because if
// it were ever false the truncation would silently renumber bonded partners.

import { NONE_U32 } from './soup-types'
import { mi3Distance } from './soup-box-scale-math'
import { liveRadius, sampleInParcel } from './soup-confine'
import { resizeSoupGrid } from './soup-buffers'
import { rebindGridDependent } from './soup-bindgroups'
import { encodeGridRebuild, encodeVerletRebuild, assertVerletSafety } from './soup-grid-verlet'
import { encodeSoupForce, encodeSoupForceList } from './soup-integrate'
import { assertStateFinite } from './soup-health'
import { relaxIterations } from './soup-relax'
import { buildBindGroups } from './soup-bindgroups'
import type { SoupRuntime } from './soup-runtime'
// Task 'confined-parcel' (2026-08-21): the PURE half of this module now lives in
// soup/src/soup-evaporate-plan.ts (CLAUDE.md's file-size rule -- see that file's header). Re-exported
// under the SAME names, so every existing importer of soup/src/soup-evaporate.ts is unaffected.
import { evaporationLadder, rehydrationLadder, sampleInsertionPositions, type EvaporationPlan } from './soup-evaporate-plan'
export { planEvaporation, evaporationLadder, rehydrationLadder, sampleInsertionPositions } from './soup-evaporate-plan'
export type { EvaporationPlan } from './soup-evaporate-plan'

// --- the live half: changing the active particle count on a running system -----------------------

/** Applies a new active particle count: re-plans the neighbour grid for `newBox`, rebinds every bind
 * group over the new active range, rebuilds the grid/Verlet list and the force, and re-checks
 * finiteness and Verlet safety -- i.e. exactly the post-conditions applyBoxScaleOnce already
 * establishes after a box change, for the same reasons. */
async function setActiveCount(rt: SoupRuntime, newN: number, newBox: [number, number, number]): Promise<void> {
  rt.N = newN
  rt.wgN = Math.ceil(newN / 64)
  rt.live.liveBox = newBox
  Object.assign(rt.bind, buildBindGroups(rt.device, rt.pipe, rt.buf, newN))
  resizeSoupGrid(rt.device, rt.buf, rt.grid, rt.startCounts, newBox, rt.effectiveWalkRadius, () =>
    rebindGridDependent(rt.device, rt.pipe, rt.buf, rt.bind, rt.N),
  )
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
  await assertStateFinite(rt)
  if (rt.verlet.enabled) await assertVerletSafety(rt)
}

/** Removes solvent beads by shrinking the active range to `targetSolvent` survivors. Removal alone
 * can never create an overlap (it only takes particles away), so this needs no relaxation and no
 * ramp of its own -- the ramp exists for the BOX change that accompanies it. */
export async function evaporateSolventTo(rt: SoupRuntime, plan: EvaporationPlan, targetSolvent: number, newBox: [number, number, number]): Promise<void> {
  if (targetSolvent > rt.N - plan.solventBlockStart) {
    throw new Error(`evaporation: ${targetSolvent} solvent beads requested, but ${rt.N - plan.solventBlockStart} are live now`)
  }
  await setActiveCount(rt, plan.solventBlockStart + targetSolvent, newBox)
}


export interface RehydrationReport {
  inserted: number
  shortOfFloor: number
  minAchieved: number
  relaxIterations: number
  maxForceBefore: number
  maxForceAfter: number
  /** Root-mean-square and worst minimum-image displacement of the particles that were ALREADY present
   * -- the measured perturbation rehydration causes, reported instead of claimed away. */
  preexistingRmsDisplacement: number
  preexistingMaxDisplacement: number
  /** The minimiser's own analytical ceiling on any single particle's total travel, d0*(it+1)/2 --
   * what the two numbers above are checked against. */
  displacementBound: number
}

/** Puts `targetSolvent - live` solvent beads back, at `newBox`.
 *
 * Placement, velocities and RNG state for the returning beads are all FRESH (see the sampler above
 * for placement; velocities are the same Maxwell-Boltzmann draw at kT soup/src/soup-init-state.ts
 * makes at creation; the two RNG streams get the creation formula mixed with the global step, so no
 * two beads share a state -- which matters because soup/wgsl/step.wgsl's RNG advance has no index
 * mixing, so equal states would stay equal forever, the exact defect the predecessor measured costing
 * a factor of three on the chemistry). Bond slots, the catalyst link and the hold clock are written
 * to their creation-time empty values: a returning solvent bead brings no history back with it.
 *
 * THE MINIMISATION AFTERWARDS MOVES EVERY PARTICLE, and that was MEASURED to be necessary rather than
 * assumed. The first version of this function held every pre-existing particle immobile for the
 * duration (the mineral platelet's own `frozen` flag), which gave a beautiful property -- the
 * non-solvent positions came back bit-identical, 0 of 45216 floats changed -- and then blew the run
 * up: inserting into a liquid is a MANY-BODY rearrangement, a bead placed in a locally jammed
 * neighbourhood has nowhere to go if its neighbours cannot yield, max|F| stalled at 1.18e4 after the
 * minimisation, and the next 1000 real steps diverged (the loud guard fired at step 4000 with 50460
 * of 50502 position components non-finite). This is the SAME problem the cold start has -- a lattice
 * at liquid density puts unlike-radius pairs inside each other's cores -- and it has the same cure,
 * the same minimiser with everything mobile. The price is that rehydration DOES perturb the organics,
 * so this function measures that perturbation (rms and max displacement over every pre-existing
 * particle) and hands it back to be reported, instead of a property it cannot honestly claim. The
 * perturbation is bounded, not merely small: the normalised descent's own total displacement bound is
 * d0*(iterations+1)/2. */
export async function rehydrateSolventTo(
  rt: SoupRuntime,
  plan: EvaporationPlan,
  targetSolvent: number,
  newBox: [number, number, number],
  particles: () => Promise<Float32Array>,
  forces: () => Promise<Float32Array>,
  seed: number,
): Promise<RehydrationReport> {
  if (rt.frozenCount > 0) {
    throw new Error(`rehydration: the system carries ${rt.frozenCount} immobile beads -- a cycle with evaporation requires clay=false/frozenBulkCatalysts=0`)
  }
  const liveSolvent = rt.N - plan.solventBlockStart
  const count = targetSolvent - liveSolvent
  if (count < 0) throw new Error(`rehydration: the target ${targetSolvent} is less than the live ${liveSolvent}`)
  const insertStart = rt.N
  if (count === 0) {
    await setActiveCount(rt, rt.N, newBox)
    return {
      inserted: 0,
      shortOfFloor: 0,
      minAchieved: 0,
      relaxIterations: 0,
      maxForceBefore: 0,
      maxForceAfter: 0,
      preexistingRmsDisplacement: 0,
      preexistingMaxDisplacement: 0,
      displacementBound: 0,
    }
  }
  const active = await particles()
  // mulberry32, the same generator soup/src/soup-init-state.ts uses for the initial layout -- kept
  // here rather than imported because that one is module-private there; a seeded generator makes
  // every rehydration reproducible from (seed, globalStep) alone.
  let a = (seed ^ Math.imul(rt.live.globalStep + 1, 0x9e3779b9)) >>> 0
  const rng = () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
  const gaussian = (): number => {
    const u1 = Math.max(rng(), 1e-9)
    const u2 = rng()
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
  }
  const sampled = sampleInsertionPositions(
    active,
    rt.N,
    newBox,
    count,
    plan.minSeparationSigma * rt.p.sigma,
    rng,
    rt.buf.confine ? () => sampleInParcel(rng, newBox, liveRadius(rt.buf.confine!, newBox)) : undefined,
  )
  const pos = new Float32Array(count * 4)
  const vel = new Float32Array(count * 4)
  const s = Math.sqrt(rt.p.thermostat.kT)
  for (let k = 0; k < count; k++) {
    pos[k * 4] = sampled.positions[k * 4]
    pos[k * 4 + 1] = sampled.positions[k * 4 + 1]
    pos[k * 4 + 2] = sampled.positions[k * 4 + 2]
    pos[k * 4 + 3] = plan.solventKind
    vel[k * 4] = s * gaussian()
    vel[k * 4 + 1] = s * gaussian()
    vel[k * 4 + 2] = s * gaussian()
  }
  const bondRng = new Uint32Array(count)
  const thermoRng = new Uint32Array(count)
  for (let k = 0; k < count; k++) {
    const i = insertStart + k
    bondRng[k] = (seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9 ^ (rt.live.globalStep >>> 0)
    thermoRng[k] = (seed >>> 0) ^ Math.imul(i + 1, 3266489917) ^ 0x85ebca6b ^ (rt.live.globalStep >>> 0)
  }
  const q = rt.device.queue
  q.writeBuffer(rt.buf.posBuf, insertStart * 16, pos)
  q.writeBuffer(rt.buf.velBuf, insertStart * 16, vel)
  q.writeBuffer(rt.buf.bondSlotsBuf, insertStart * 3 * 4, new Uint32Array(count * 3).fill(NONE_U32))
  q.writeBuffer(rt.buf.centerLinkBuf, insertStart * 4, new Uint32Array(count).fill(NONE_U32))
  q.writeBuffer(rt.buf.centerHeldStepsBuf, insertStart * 4, new Uint32Array(count))
  q.writeBuffer(rt.buf.bondRngBuf, insertStart * 4, bondRng)
  q.writeBuffer(rt.buf.thermoRngBuf, insertStart * 4, thermoRng)
  await setActiveCount(rt, plan.solventBlockStart + targetSolvent, newBox)
  const maxForceBefore = maxAbs(await forces(), rt.N)
  await relaxIterations(rt, plan.relaxIterations, (rt.soup.coldStartRelax?.maxDisplacementSigma ?? 0.1) * rt.p.sigma)
  const maxForceAfter = maxAbs(await forces(), rt.N)
  // How much the minimisation moved what was ALREADY here -- the honest replacement for the
  // bit-identical property this operation cannot have (see the doc comment above). Minimum-image, so
  // a particle that merely wrapped is not counted as having crossed the box.
  const after = await particles()
  let sumSq = 0
  let maxDisp = 0
  for (let i = 0; i < insertStart; i++) {
    let d2 = 0
    for (let ax = 0; ax < 3; ax++) {
      const d = after[i * 4 + ax] - active[i * 4 + ax]
      const m = d - Math.round(d / newBox[ax]) * newBox[ax]
      d2 += m * m
    }
    sumSq += d2
    if (d2 > maxDisp) maxDisp = d2
  }
  return {
    inserted: count,
    shortOfFloor: sampled.shortOfFloor,
    minAchieved: sampled.minAchieved,
    relaxIterations: plan.relaxIterations,
    maxForceBefore,
    maxForceAfter,
    preexistingRmsDisplacement: Math.sqrt(sumSq / Math.max(1, insertStart)),
    preexistingMaxDisplacement: Math.sqrt(maxDisp),
    displacementBound: ((rt.soup.coldStartRelax?.maxDisplacementSigma ?? 0.1) * rt.p.sigma * (plan.relaxIterations + 1)) / 2,
  }
}

/** Fires the adsorption tether's OWN desorption valve synchronously, for every link a box change has
 * stretched past FENE's range.
 *
 * WHY THIS EXISTS -- diagnosed from two real divergences, not anticipated. `scaleMoleculesRigid`
 * deliberately does NOT treat the adsorption tether (soup/wgsl/bond-adsorption.wgsl's centerLink) as a
 * rigid edge: it is a soft, desorbable association, and data/soup.json's adsorption.basis says its
 * length is ALLOWED to change under a box move because "it already has a desorption valve for exactly
 * this kind of stretch". The valve is `desorbStretch`, and it lives inside `bond_form_main`, which is
 * dispatched only every `bondAttemptInterval.steps` = 20 REAL STEPS. Meanwhile the force this tether
 * exerts is `fene_dv(r) = k*r/(1 - (r/r_inf)^2)` -- which does not merely grow as r approaches
 * P.r_inf, it CHANGES SIGN past it, so an over-stretched tether pushes its pair apart without
 * bound. A rehydration expands the box by the whole wet/dry ratio, so any tether longer than r_inf
 * divided by that ratio crosses r_inf during the ramp, and then up to 20 real steps of unbounded force run before the valve
 * gets a chance. That is exactly what was measured, twice, at box 30 from cycle 2 onward (once the
 * material had actually aggregated and catalysts were holding tips): assertVerletSafety threw drifts
 * of 2.98e7 and 2.13e14 sigma inside the ramp's own relaxation steps, with the state still FINITE, so
 * this was a real physical runaway and not an Inf.
 *
 * WHAT THIS CHANGES, AND WHAT IT DOES NOT. It fires the SAME valve on the SAME criterion (tether
 * beyond FENE's bonded range) at the moment the box change creates the condition, instead of up to 20
 * steps later, and it counts each one into the SAME `desorbEvents[0]` (stretch) counter the GPU valve
 * increments. It is called ONLY from the evaporating transition, so the pre-existing box-scaling-only
 * path is byte-for-byte unchanged. The threshold is DERIVED, not chosen: `r_inf / lambda`, where
 * lambda is the ramp's own per-increment expansion factor -- i.e. clear exactly those tethers the NEXT
 * increment could push past r_inf, and no others.
 *
 * The general case is a latent defect of the engine that this task did NOT fix (a tether can also
 * cross r_inf by ordinary thermal stretching between two bond dispatches); it is reported as a
 * concern rather than patched in a shared path whose published numbers would change. */
export async function desorbOverstretchedTethers(
  rt: SoupRuntime,
  particles: () => Promise<Float32Array>,
  centerLinks: () => Promise<Uint32Array>,
  box: [number, number, number],
  maxLength: number,
): Promise<{ cleared: number; maxLength: number; longest: number }> {
  const links = await centerLinks()
  const pos = await particles()
  const next: Uint32Array<ArrayBuffer> = Uint32Array.from(links)
  let cleared = 0
  let longest = 0
  for (let i = 0; i < rt.N; i++) {
    const owner = links[i]
    if (owner === NONE_U32 || owner >= rt.N) continue
    const r = mi3Distance(pos, box, i, owner)
    if (r > longest) longest = r
    if (r >= maxLength) {
      next[i] = NONE_U32
      next[owner] = NONE_U32
      cleared++
    }
  }
  if (cleared > 0) {
    rt.device.queue.writeBuffer(rt.buf.centerLinkBuf, 0, next.subarray(0, rt.N))
    // The hold clock must go with the link, exactly as soup/wgsl/bond-adsorption.wgsl resets it on
    // its own desorption, or a later re-adsorption would inherit a stale age.
    const held = new Uint32Array(new ArrayBuffer(rt.N * 4))
    rt.device.queue.writeBuffer(rt.buf.centerHeldStepsBuf, 0, held)
    // Counted into the SAME cumulative counter the GPU valve uses, so a run's desorption total stays
    // the true total. cleared/2 is not used: `cleared` counts links seen from one side only, since the
    // partner is blanked in the same pass and skipped as NONE afterwards.
    const raw = await readBackDesorb(rt)
    raw[0] += cleared
    rt.device.queue.writeBuffer(rt.buf.desorbEventsBuf, 0, raw)
  }
  return { cleared, maxLength, longest }
}

async function readBackDesorb(rt: SoupRuntime): Promise<Uint32Array<ArrayBuffer>> {
  const { readBack } = await import('../../engine/src/gpu')
  const raw = await readBack(rt.device, rt.buf.desorbEventsBuf, 8)
  const out = new Uint32Array(new ArrayBuffer(8))
  out.set(new Uint32Array(raw.buffer, raw.byteOffset, 2))
  return out
}

function maxAbs(f: Float32Array, n: number): number {
  let m = 0
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < 3; c++) {
      const v = Math.abs(f[i * 4 + c])
      if (!Number.isFinite(v)) return Number.POSITIVE_INFINITY
      if (v > m) m = v
    }
  }
  return m
}

