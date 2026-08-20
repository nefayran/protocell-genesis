// Measurement and readback helpers. Split out of soup/src/sim.ts's createSoup (file-size rule in
// CLAUDE.md) -- every GPU readback (particles/velocities/forces/bonds/etc, the checkpoint
// "capture" half soup/src/checkpoint.ts's encodeCheckpoint consumes) plus the two debug-only
// profiling helpers (stepPhasesDEBUG, forceCandidateStatsDEBUG), moved verbatim.

import { readBack } from '../../engine/src/gpu'
import { NONE_U32 } from './soup-types'
import type { Soup } from './rules'
import { encodeGridRebuild, encodeVerletRebuild } from './soup-grid-verlet'
import { encodeSoupForce, encodeSoupForceBrute, encodeSoupForceList, encodeOneIntegrationStep } from './soup-integrate'
import type { SoupRuntime } from './soup-runtime'

export async function particles(rt: SoupRuntime): Promise<Float32Array> {
  return readBack(rt.device, rt.buf.posBuf, rt.N * 16)
}

// Checkpoint/resume (task 'checkpoint-resume'): mirrors particles() exactly, the other half of
// the Langevin state.
export async function velocities(rt: SoupRuntime): Promise<Float32Array> {
  return readBack(rt.device, rt.buf.velBuf, rt.N * 16)
}

// perf2-report.md correctness gate: mirrors engine/src/sim.ts's forces()/forcesBruteForce() pair
// (checked by tests/sim.test.ts's "сетка соседей даёт те же силы, что и полный перебор") for the
// soup's own dynamic-topology force kernel. forces() rebuilds fresh for the CURRENT positions
// first -- when verlet.enabled, a FULL Verlet rebuild (never relying on a possibly-stale list
// from whatever step count the caller happens to be at) then the list-based force kernel;
// otherwise the same encodeGridRebuild/encodeSoupForce cell walk every real step already uses
// (candidate (a)'s walk radius and candidate (b)'s gather both feed through it).
// forcesBruteForce() needs neither -- it is the O(N^2) reference soup_force_brute_main runs
// directly off pos2/bondSlots.
export async function forces(rt: SoupRuntime): Promise<Float32Array> {
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
  return readBack(rt.device, rt.buf.forceBuf, rt.N * 16)
}

/** The force buffer EXACTLY AS IT STANDS, with no grid/list/force rebuild first -- i.e. the very
 * F(x_n) that soup/src/soup-integrate.ts's next kick_drift_wrap will consume. Task 'decisive-run'
 * (2026-08-20): forces() above recomputes before reading back, which makes a STALE forceBuf
 * unobservable through it -- so a test could not tell "the buffer holds the force of the current
 * geometry" from "the buffer holds the force of the geometry two box sizes ago". This readback is
 * the only way to see that difference, and it exists for tests/soup-stale-force.test.ts to pin the
 * applyBoxScaleOnce fix with. It changes no state at all (no dispatch is encoded). */
export async function forcesNoRebuild(rt: SoupRuntime): Promise<Float32Array> {
  return readBack(rt.device, rt.buf.forceBuf, rt.N * 16)
}

export async function forcesBruteForce(rt: SoupRuntime): Promise<Float32Array> {
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeSoupForceBrute(rt, pass)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  return readBack(rt.device, rt.buf.forceBuf, rt.N * 16)
}

export async function readBondSlots(rt: SoupRuntime): Promise<Uint32Array> {
  const raw = await readBack(rt.device, rt.buf.bondSlotsBuf, rt.N * 3 * 4)
  return new Uint32Array(raw.buffer, raw.byteOffset, rt.N * 3)
}

export async function bonds(rt: SoupRuntime): Promise<Uint32Array> {
  const slots = await readBondSlots(rt)
  const out: number[] = []
  for (let i = 0; i < rt.N; i++) {
    for (let s = 0; s < 3; s++) {
      const j = slots[i * 3 + s]
      if (j !== NONE_U32 && j > i) out.push(i, j)
    }
  }
  return new Uint32Array(out)
}

export async function centerLinks(rt: SoupRuntime): Promise<Uint32Array> {
  const raw = await readBack(rt.device, rt.buf.centerLinkBuf, rt.N * 4)
  return new Uint32Array(raw.buffer, raw.byteOffset, rt.N)
}

export async function desorbEvents(rt: SoupRuntime): Promise<{ stretch: number; timeout: number }> {
  const raw = await readBack(rt.device, rt.buf.desorbEventsBuf, 8)
  const u32 = new Uint32Array(raw.buffer, raw.byteOffset, 2)
  return { stretch: u32[0], timeout: u32[1] }
}

export async function centerHeldSteps(rt: SoupRuntime): Promise<Uint32Array> {
  const raw = await readBack(rt.device, rt.buf.centerHeldStepsBuf, rt.N * 4)
  return new Uint32Array(raw.buffer, raw.byteOffset, rt.N)
}

/** Task 'clay-surface' (2026-08-19): the per-particle immobility flag exactly as the GPU holds it --
 * 1 for a bead of the rigid mineral platelet (clay bead or catalyst surface site), 0 otherwise. Read
 * back rather than recomputed on the CPU so a test proving immobility is proving what the SHADER
 * actually saw, not what a CPU-side plan intended. */
/** Task 'electrostatics' (2026-08-20): the per-particle charge in units of e (soup/wgsl/
 * electrostatics.wgsl's chargeRO). Read by the protonation Monte Carlo before every sweep, by the
 * checkpoint, and by every off-GPU measurement of the deprotonated fraction / acid-soap pairing. */
export async function charges(rt: SoupRuntime): Promise<Float32Array> {
  return readBack(rt.device, rt.buf.chargeBuf, rt.N * 4)
}

export async function frozen(rt: SoupRuntime): Promise<Uint32Array> {
  const raw = await readBack(rt.device, rt.buf.frozenBuf, rt.N * 4)
  return new Uint32Array(raw.buffer, raw.byteOffset, rt.N)
}

export async function rngState(rt: SoupRuntime): Promise<{ bond: Uint32Array; thermo: Uint32Array }> {
  const rawBond = await readBack(rt.device, rt.buf.bondRngBuf, rt.N * 4)
  const rawThermo = await readBack(rt.device, rt.buf.thermoRngBuf, rt.N * 4)
  return {
    bond: new Uint32Array(rawBond.buffer, rawBond.byteOffset, rt.N),
    thermo: new Uint32Array(rawThermo.buffer, rawThermo.byteOffset, rt.N),
  }
}

export async function events(rt: SoupRuntime): Promise<Record<string, number>> {
  const raw = await readBack(rt.device, rt.buf.eventsBuf, rt.rules.length * 2 * 4)
  const u32 = new Uint32Array(raw.buffer, raw.byteOffset, rt.rules.length * 2)
  const out: Record<string, number> = {}
  for (let r = 0; r < rt.rules.length; r++) {
    out[rt.eventRuleIds[r][0]] = u32[r * 2 + 0]
    out[rt.eventRuleIds[r][1]] = u32[r * 2 + 1]
  }
  return out
}

export async function invariants(
  rt: SoupRuntime,
  soup: Soup,
): Promise<{ monomers: Record<string, number>; bonds: number; charge: number }> {
  const pos = await particles(rt)
  const monomers: Record<string, number> = {}
  for (const m of soup.monomers) monomers[m.id] = 0
  let charge = 0
  for (let i = 0; i < rt.N; i++) {
    const kind = Math.round(pos[i * 4 + 3])
    const m = soup.monomers[kind]
    monomers[m.id] = (monomers[m.id] ?? 0) + 1
    charge += (m as unknown as { charge?: number }).charge ?? 0
  }
  const bondCount = (await bonds(rt)).length / 2
  return { monomers, bonds: bondCount, charge }
}

// Profiling helper for the perf task (report:
// .superpowers/sdd/2026-08-16-soup-to-vesicle/perf-report.md). No timestamp-query use even
// though the adapter supports the feature: WebGPU only exposes timestampWrites at COMPUTE-PASS
// granularity, and this engine deliberately fuses all 9 dispatches of one integration step into
// ONE pass (see soup/src/sim.ts's own header and step.wgsl's -- interleaving many small
// dispatches/passes measurably dominated wall time before that fuse). Splitting per-stage
// passes to get per-stage GPU timestamps would reintroduce exactly the overhead the fuse
// removed, so it would not be measuring the thing production actually runs. Uses the
// difference-of-variants method instead, exactly as the task brief's fallback describes.
//
// A first, buggy version of this ran EACH stage alone for n steps in sequence on one shared
// system, in an order (kickDriftWrap first) that let one unsafe isolation corrupt every
// measurement after it: kickDriftWrap alone applies +0.5*dt*F from a force that is NEVER
// recomputed (soupForce is a separate phase), so velocity grows by a fixed increment every
// single iteration with nothing to damp it -- after 3000 iterations that is a large,
// accumulating, UNBOUNDED drift, and every phase measured afterward (including the final
// "fullStepEquivalent") inherited that already-deranged state. Measured symptom: the isolated
// stages summed to ~0.9ms/step but "fullStepEquivalent" (the very same 9 dispatches, just
// bundled) came out at 21.4ms/step -- 23x its own parts, and 10x the documented ~2.1ms/step
// (480 steps/s) baseline for this exact configuration. That gap is the corruption, not a real
// cost; it never got used for anything.
//
// Fix: only ever isolate GROUPS that stay physically bounded no matter how large n is.
//  - 'full': the real, unmodified per-step dispatch sequence, timed FIRST (before anything else
//    touches the buffers) -- this is just step()'s own physics, proven stable over 50_000 steps,
//    and is the number this whole profile is checked against.
//  - 'gridBuild', 'force': read positions, WRITE only their own output buffer (cellStart/cells,
//    forceBuf) -- positions never move during these, so running either alone for any n computes
//    the identical answer every iteration. No corruption possible.
//  - 'bondAttempts' (form+break together): mutates only bondSlots/events, never positions or
//    velocities, and valence is capped at 3 -- self-limiting, not unbounded.
//  - 'integration' (kickDriftWrap+kickThermostat together, exactly as they alternate in a real
//    step): unlike kickDriftWrap ALONE, this pairs every kick with the thermostat's own
//    Ornstein-Uhlenbeck damping (v -= gamma*dt*v, before adding noise) in the SAME iteration, so
//    velocity relaxes to a stationary distribution instead of growing linearly -- bounded for
//    any n, even though the force it integrates against is a stale snapshot (not recomputed
//    inside this isolated group, same as every other bucket here).
export async function stepPhasesDEBUG(rt: SoupRuntime, n: number): Promise<Record<string, number>> {
  async function timePhase(label: string, fn: (pass: GPUComputePassEncoder) => void): Promise<[string, number]> {
    const enc = rt.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    for (let k = 0; k < n; k++) fn(pass)
    pass.end()
    const t0 = performance.now()
    rt.device.queue.submit([enc.finish()])
    await rt.device.queue.onSubmittedWorkDone()
    return [label, (performance.now() - t0) / n]
  }
  const out: Record<string, number> = {}
  const encodeIntegration = (pass: GPUComputePassEncoder) => {
    pass.setPipeline(rt.pipe.kickDriftWrap)
    pass.setBindGroup(0, rt.bind.kickDriftWrapGroup0)
    pass.setBindGroup(1, rt.bind.kickDriftWrapGroup1)
    pass.dispatchWorkgroups(rt.wgN)
    pass.setPipeline(rt.pipe.kickThermostat)
    pass.setBindGroup(0, rt.bind.kickThermostatGroup0)
    pass.setBindGroup(1, rt.bind.kickThermostatGroup1)
    pass.dispatchWorkgroups(rt.wgN)
  }
  const encodeBondAttempts = (pass: GPUComputePassEncoder) => {
    pass.setPipeline(rt.pipe.bondForm)
    pass.setBindGroup(0, rt.bind.bondFormGroup0)
    pass.setBindGroup(1, rt.bind.bondFormGroup1)
    pass.setBindGroup(2, rt.bind.bondFormGroup2)
    pass.dispatchWorkgroups(rt.wgN)
    pass.setPipeline(rt.pipe.bondBreak)
    pass.setBindGroup(1, rt.bind.bondBreakGroup1)
    pass.setBindGroup(2, rt.bind.bondBreakGroup2)
    pass.dispatchWorkgroups(rt.wgN)
  }
  const phases: [string, (pass: GPUComputePassEncoder) => void][] = [
    // Worst case (every step does bonds too) -- the number to compare against gridBuild/
    // force/bondAttempts/integration's sum; the REAL amortized cost (bonds only every
    // bondAttemptInterval.steps steps) is what step()'s own timing reports.
    ['full', (pass) => encodeOneIntegrationStep(rt, pass, true, true)],
    ['gridBuild', (pass) => encodeGridRebuild(rt, pass)],
    ['force', (pass) => encodeSoupForce(rt, pass)],
    ['bondAttempts', encodeBondAttempts],
    ['integration', encodeIntegration],
  ]
  for (const [label, fn] of phases) {
    const [l, ms] = await timePhase(label, fn)
    out[l] = ms
  }
  return out
}

// perf2-report.md, STEP 1 diagnosis (and re-checked after each grid change): rebuilds the grid
// for the CURRENT position state (so counts describe the config this call sees, not a stale
// one), zeroes the stats buffer, then runs soup_force_stats_main -- the exact same ±walkRadius
// cell walk soup_force_main runs -- once, and reads its two counters back. Never part of the
// real step loop; a diagnostic only, same role stepPhasesDEBUG plays for timing.
export async function forceCandidateStatsDEBUG(rt: SoupRuntime): Promise<{ candidatesExamined: number; pairsWithinRange: number; ratio: number }> {
  rt.device.queue.writeBuffer(rt.buf.statsBuf, 0, new Uint32Array([0, 0]))
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeGridRebuild(rt, pass)
  pass.setPipeline(rt.pipe.forceStats)
  pass.setBindGroup(0, rt.bind.forceStatsGroup0)
  pass.setBindGroup(1, rt.bind.forceStatsGroup1)
  pass.setBindGroup(3, rt.bind.forceStatsGroup3)
  pass.dispatchWorkgroups(rt.wgN)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  const raw = await readBack(rt.device, rt.buf.statsBuf, 8)
  const u32 = new Uint32Array(raw.buffer, raw.byteOffset, 2)
  const candidatesExamined = u32[0]
  const pairsWithinRange = u32[1]
  return { candidatesExamined, pairsWithinRange, ratio: candidatesExamined / Math.max(1, pairsWithinRange) }
}

// Task 'big-box' (2026-08-20): THE MEASUREMENT data/soup.json's verletList.listCapacity was never
// based on. The capacity was set as a "safe upper bound" (2500) and the whole particle ceiling
// (N*listCapacity*4 bytes against the device's 4 294 967 292-byte storage-binding limit, i.e.
// 429 496 particles) followed from that one unmeasured number. verletCountBuf already holds exactly
// what is needed -- the number of candidates within listRange each particle actually found at the
// last rebuild -- so the distribution is one readback away, and the reduction happens HERE, inside
// the page, so a 2.2e6-element u32 array never crosses the CDP boundary (this project's own measured
// page-killer: checkpoint-resume-report.md).
//
// Rebuilds fresh first, for the same reason forces() does: the count must describe the configuration
// this call sees, never a list left over from whatever step count the caller happens to sit at.
// Returns null for the main list on a system with verlet.enabled=false -- there is no per-particle
// list at all in that configuration, which is the entire point of it.
export interface ListOccupancy {
  n: number
  capacity: number
  max: number
  mean: number
  sd: number
  p50: number
  p99: number
  p999: number
  /** How many entries sat at the capacity, i.e. how many were (or were about to be) truncated. */
  atCapacity: number
  /** max/mean -- the inhomogeneity factor a uniform-density estimate has to be multiplied by. */
  inhomogeneity: number
  /** The upper tail as [count, howMany] pairs, coarse-binned in 25s from p99 upward, so a report can
   * show the SHAPE of the tail rather than only its extreme. */
  tail: [number, number][]
}

function occupancyOf(counts: Uint32Array, capacity: number): ListOccupancy {
  const n = counts.length
  let max = 0
  let sum = 0
  let atCapacity = 0
  for (let i = 0; i < n; i++) {
    const c = counts[i]
    sum += c
    if (c > max) max = c
    if (c >= capacity) atCapacity++
  }
  const mean = sum / Math.max(1, n)
  let varSum = 0
  for (let i = 0; i < n; i++) varSum += (counts[i] - mean) ** 2
  const sd = Math.sqrt(varSum / Math.max(1, n))
  // Histogram over the whole integer range (max is a few hundred to a few thousand, so this is a
  // small dense array) -- exact quantiles, no sort of an N-element array.
  const hist = new Uint32Array(max + 2)
  for (let i = 0; i < n; i++) hist[counts[i]]++
  const quantile = (q: number): number => {
    const target = q * n
    let acc = 0
    for (let c = 0; c <= max; c++) {
      acc += hist[c]
      if (acc >= target) return c
    }
    return max
  }
  const p99 = quantile(0.99)
  const tail: [number, number][] = []
  for (let lo = Math.floor(p99 / 25) * 25; lo <= max; lo += 25) {
    let cnt = 0
    for (let c = lo; c < lo + 25 && c <= max; c++) cnt += hist[c]
    if (cnt > 0) tail.push([lo, cnt])
  }
  return { n, capacity, max, mean, sd, p50: quantile(0.5), p99, p999: quantile(0.999), atCapacity, inhomogeneity: max / Math.max(1e-9, mean), tail }
}

export async function listOccupancyDEBUG(rt: SoupRuntime): Promise<{
  main: ListOccupancy | null
  es: ListOccupancy | null
  listRange: number
  esListRange: number
  box: [number, number, number]
  N: number
  density: number
  /** The uniform-density expectation (4/3)*pi*listRange^3*rho -- what a DERIVED capacity is a
   * multiple of, printed beside the measured mean so the two can be compared directly. */
  uniformExpected: number
}> {
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  if (rt.verlet.enabled) encodeVerletRebuild(rt, pass)
  else encodeGridRebuild(rt, pass)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  await rt.device.queue.onSubmittedWorkDone()

  let main: ListOccupancy | null = null
  if (rt.verlet.enabled) {
    const raw = await readBack(rt.device, rt.buf.verletCountBuf, rt.N * 4)
    main = occupancyOf(new Uint32Array(raw.buffer, raw.byteOffset, rt.N), rt.verlet.listCapacity)
  }
  let es: ListOccupancy | null = null
  if (rt.protonation?.es.enabled && rt.esHeads > 0) {
    const raw = await readBack(rt.device, rt.buf.esCountBuf, rt.esHeads * 4)
    es = occupancyOf(new Uint32Array(raw.buffer, raw.byteOffset, rt.esHeads), rt.protonation.es.listCapacity)
  }
  const box = rt.live.liveBox
  const density = rt.N / (box[0] * box[1] * box[2])
  const listRange = rt.listRange
  return {
    main,
    es,
    listRange,
    esListRange: rt.protonation?.es.listRange ?? 0,
    box: [box[0], box[1], box[2]],
    N: rt.N,
    density,
    uniformExpected: ((4 * Math.PI) / 3) * listRange ** 3 * density,
  }
}
