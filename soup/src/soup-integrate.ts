// The integration step and its chunked submission. Split out of soup/src/sim.ts's createSoup
// (file-size rule in CLAUDE.md) -- the per-step dispatch sequence (one kick+drift+wrap, the
// force/grid/Verlet phase, bond chemistry, one kick+thermostat, all fused into ONE compute pass by
// design -- see this file's own STEP_CHUNK comment and sim.ts's original header for why) and
// step()'s own multi-submission chunking, moved verbatim.

import { encodeGridRebuild, encodeVerletRebuild, encodeMaxDrift, assertVerletSafety } from './soup-grid-verlet'
import { encodeBondChemistry } from './soup-bond-chemistry'
import { assertStateFinite } from './soup-health'
// Task 'electrostatics' (2026-08-20): the constant-pH Monte Carlo sweep. Imported lazily-by-name (not
// at module top with a value import that would create a cycle: soup-protonation.ts imports
// encodeSoupForce/encodeSoupForceList from THIS file) -- see the call site below.
import type { SoupRuntime } from './soup-runtime'

// Task 'long-range-electrostatics' (2026-08-20): the FAR half of the screened-Coulomb term, added
// from the dedicated head-only list. Dispatched from inside the three encodeSoupForce* functions
// below, not at their call sites -- there are eight call sites (step, box scale, evaporation,
// rehydration, cold-start relax, protonation recompute, two readbacks) and a term that is only added
// at seven of them is a silent physics bug, exactly the class tests/soup-stale-force.test.ts exists
// for. It ACCUMULATES into outForce, so it must follow the kernel that wrote it, in the same pass:
// WebGPU orders dispatches within a compute pass and inserts the barrier between them, which is what
// the whole existing grid-rebuild-then-force sequence already relies on.
function encodeEsFar(rt: SoupRuntime, pass: GPUComputePassEncoder, brute: boolean): void {
  if (!rt.protonation?.es.enabled || rt.esHeads === 0) return
  pass.setPipeline(brute ? rt.pipe.esForceFarBrute : rt.pipe.esForceFar)
  pass.setBindGroup(1, brute ? rt.bind.esForceFarBruteBind : rt.bind.esForceFarBind)
  pass.dispatchWorkgroups(brute ? rt.wgN : rt.wgEsHeads)
}

export function encodeSoupForce(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  pass.setPipeline(rt.pipe.soupForce)
  pass.setBindGroup(0, rt.bind.soupForceGroup0)
  pass.setBindGroup(1, rt.bind.soupForceGroup1)
  pass.dispatchWorkgroups(rt.wgN)
  encodeEsFar(rt, pass, false)
}

export function encodeSoupForceBrute(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  pass.setPipeline(rt.pipe.soupForceBrute)
  pass.setBindGroup(0, rt.bind.soupForceBruteGroup0)
  pass.setBindGroup(1, rt.bind.soupForceBruteGroup1)
  pass.dispatchWorkgroups(rt.wgN)
  // The BRUTE reference uses the brute far pass too -- an O(N^2) sum with no list at all, so that
  // tests/soup-forces.test.ts's grid+Verlet-vs-brute comparison actually tests the head-only list's
  // completeness over the new range instead of comparing a list against itself.
  encodeEsFar(rt, pass, true)
}

export function encodeSoupForceList(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  pass.setPipeline(rt.pipe.soupForceList)
  pass.setBindGroup(0, rt.bind.soupForceListGroup0)
  pass.setBindGroup(1, rt.bind.soupForceListGroup1)
  pass.dispatchWorkgroups(rt.wgN)
  encodeEsFar(rt, pass, false)
}

// doBonds: perf fix (b), perf-report.md. bond_form_main/bond_break_main measured ~42% of a full
// step's cost (0.69 of 1.64ms at N=13100, 30^3 box) while the events they exist to catch are
// rare (attemptProbability per real step is 0.0005 at this rule set's attemptRate/dt) -- most of
// that cost is the 3x3x3 neighbour walk run for nothing, every step, whether or not this step is
// one of the rare ones that actually attempts anything. Dispatching the pair only every
// bondAttemptInterval.steps real steps (data/soup.json, with attemptProbForm/attemptProbBreak
// already built from dt*bondAttemptInterval in soup/src/soup-buffers.ts) cuts that cost by
// ~bondAttemptInterval.steps while leaving the average attempt rate per real step, and therefore
// the equilibrium bond count detailed balance sets, unchanged (see bondAttemptInterval's basis and
// tests/soup-bonds.test.ts's before/after equilibrium check).
export function encodeOneIntegrationStep(rt: SoupRuntime, pass: GPUComputePassEncoder, doBonds: boolean, doListRebuild: boolean): void {
  // First Verlet half-kick + drift + 3-axis periodic wrap, fused into one dispatch (see
  // soup/wgsl/step.wgsl's kick_drift_wrap_main header) -- exactly kick_main+drift_main+a
  // 3-axis wrap from engine/wgsl/integrate.wgsl's own formulas, not a new integrator.
  pass.setPipeline(rt.pipe.kickDriftWrap)
  pass.setBindGroup(0, rt.bind.kickDriftWrapGroup0)
  pass.setBindGroup(1, rt.bind.kickDriftWrapGroup1)
  pass.dispatchWorkgroups(rt.wgN)

  // perf2-report.md, candidate (c): when enabled, the cell walk that dominates both force and
  // bond-attempt cost runs only on scheduled rebuild steps (doListRebuild); every other step
  // reads the list built at the last rebuild instead of re-walking cells at all.
  if (rt.verlet.enabled) {
    if (doListRebuild) encodeVerletRebuild(rt, pass)
    encodeSoupForceList(rt, pass)
    encodeMaxDrift(rt, pass)
  } else {
    encodeGridRebuild(rt, pass)
    encodeSoupForce(rt, pass)
  }

  if (doBonds) encodeBondChemistry(rt, pass)

  // Second Verlet half-kick + Langevin thermostat, fused into one dispatch.
  pass.setPipeline(rt.pipe.kickThermostat)
  pass.setBindGroup(0, rt.bind.kickThermostatGroup0)
  pass.setBindGroup(1, rt.bind.kickThermostatGroup1)
  pass.dispatchWorkgroups(rt.wgN)
}

// Steps per submitted command buffer. A single createSoup+step(50_000) in one page works fine
// (measured), but a SECOND system's step(50_000) in the SAME page hung forever at
// onSubmittedWorkDone(), regardless of the second system's parameters (reproduced with
// catalystCount 0 AND with identical params to the first system) -- narrowed by bisection to the
// SIZE of the second submission specifically: the same second system's step(100) (a ~900-command
// pass) completed instantly right after the first system's step(50_000), where step(50_000) (a
// ~450,000-command pass -- 9 dispatchWorkgroups calls x 50,000 iterations) hung. So it is not
// "two systems in one page" (engine/src/sim.ts's own gate6-bilayer.test.ts does that, successfully,
// by calling step() many times with small n) and not catalystCount -- it is a single compute pass
// grown too large, which some part of the browser's WebGPU stack fails to ever signal complete once
// it is the SECOND such giant submission in one device session. Splitting step(n) into bounded
// chunks, each its own submit()+onSubmittedWorkDone() round trip, keeps every single command buffer
// at the same order of magnitude gate6-bilayer's proven-safe per-submission size (900 commands for
// n=100) while still letting step(n) advance any n in one call -- the fix is entirely inside step()
// and changes nothing about what gets computed.
const STEP_CHUNK = 1000

/** Builds the real step() function -- runs continuously across separate step(n) calls (rt.live.
 * globalStep is not reset per call) so the bondAttemptInterval schedule stays regular regardless of
 * how a caller chunks its own n's -- e.g. step(30) then step(70) attempts bonds on the same global
 * step indices step(100) would. Checkpoint/resume: rt.live.globalStep starts at
 * opts.resume.globalStep (or 0), set by soup/src/sim.ts before this is called, so the SAME schedule
 * lines up exactly where a single, uninterrupted run would have been at this step count. */
export function buildStepper(rt: SoupRuntime): (n: number) => Promise<void> {
  return async function step(n: number): Promise<void> {
    let done = 0
    while (done < n) {
      const chunk = Math.min(STEP_CHUNK, n - done)
      const chunkStart = rt.live.globalStep
      const enc = rt.device.createCommandEncoder()
      const pass = enc.beginComputePass()
      for (let k = 0; k < chunk; k++) {
        encodeOneIntegrationStep(
          rt,
          pass,
          rt.live.globalStep % rt.bondAttemptInterval === 0,
          rt.verlet.enabled && rt.live.globalStep % rt.verlet.rebuildEvery === 0,
        )
        rt.live.globalStep++
      }
      pass.end()
      rt.device.queue.submit([enc.finish()])
      await rt.device.queue.onSubmittedWorkDone()
      // Task 'loud-failure-and-liquid-water' (2026-08-20): the non-finite guard, at the SAME cadence
      // and the SAME sync point assertVerletSafety already uses, and deliberately BEFORE it -- once
      // positions are NaN the measured drift is NaN, `NaN > skin/2` is false, and assertVerletSafety
      // reports a clean run forever (three measured instances; see soup/wgsl/health.wgsl's header).
      // Checking finiteness first is what makes the thrown message name the real cause.
      await assertStateFinite(rt, chunkStart)
      if (rt.verlet.enabled) await assertVerletSafety(rt)
      // Task 'electrostatics' (2026-08-20): the constant-pH Monte Carlo sweep, at the SAME chunk
      // boundary the two guards above already synchronise on -- so a sweep costs no extra pipeline
      // stall, only its own readback. A no-op (one integer comparison) on any system without
      // electrostatics, so every pre-task run's dispatch sequence is unchanged. Dynamic import breaks
      // an otherwise circular module reference (soup-protonation.ts needs encodeSoupForce from here).
      if (rt.protonation?.es.enabled) {
        const { maybeProtonationSweep } = await import('./soup-protonation')
        await maybeProtonationSweep(rt)
      }
      done += chunk
    }
  }
}
