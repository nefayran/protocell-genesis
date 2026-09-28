// Pins what task 'wet-dry-cycling' (2026-08-20) MEASURED about dry-wet cycling at real liquid water
// (.superpowers/sdd/2026-08-16-soup-to-vesicle/wet-dry-cycling-report.md). Two tests, deliberately
// split: a pure one (no GPU) for the arithmetic that made the re-derivation structurally necessary,
// and one GPU test for the mechanism -- both against a MATCHED control, because the observable that
// matters (reaction events per step) decays with time on its own as monomers are consumed, so
// "the dry phase is slower than the wet phase" is not a claim a single run can support.
//
// What is pinned, and why each assertion exists rather than being a restatement of the config:
//  1. data/soup.json's dryWetCycle.targetDryDensity must EXCEED the density of the project's own
//     liquid-water broth. deriveCycleConfig throws otherwise ("the dry phase must concentrate"),
//     so this is the arithmetic that forced the re-derivation: the previous value (0.6) is BELOW
//     that density and is asserted here to be refused, which is the evidence that 0.6 -> 1.34 was
//     structural and not cosmetic.
//  2. The dry box realises the configured density, and the schedule's phase boundaries are what the
//     file's own numbers say.
//  3. Through a full wet -> dry -> wet cycle at liquid water: the box returns EXACTLY to the wet box,
//     the state stays finite at every sample (the loud-failure guard runs inside applyBoxScaleOnce,
//     i.e. through the densest instant of the cycle), and the census and charge are conserved.
//  4. The measured mechanism: at the dry box the medium's mobility COLLAPSES (mean-square
//     displacement over an identical step window, against the matched wet control), and the cycled
//     run therefore ends one cycle with FEWER bond events than the control -- cycling by box
//     contraction suppresses this engine's chemistry rather than driving it. Margins are set from
//     the box-30 measurements in the report (mobility 6.2x down at rho=1.34; events ~1.2-1.4x down
//     over one cycle) and left loose enough that only a reversal of the effect fails them.
import { expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadSoup } from '../soup/src/rules'
import { computeDryBox, cyclePhaseAt, nextCycleTransition } from '../soup/src/sim'
// deriveCycleConfig is not among the names soup/src/sim.ts re-exports (only the coordinate map and
// the schedule helpers are), so it is imported from the module that owns it.
import { deriveCycleConfig } from '../soup/src/soup-box-scale-math'

// The project's own liquid-water broth, at the box the report's baseline used: water at the measured
// liquid threshold 0.8 sigma^-3 plus the full organic pool (rho_org = 0.41793).
const BOX30: [number, number, number] = [30, 30, 30]
const START30 = { C: 1860, O: 7440, H: 1860, M: 124, W: 21600 }
const N30 = Object.values(START30).reduce((a, b) => a + b, 0)

test('dry-wet cycling: the dry phase must concentrate, and the re-derived amplitude is the one that can', () => {
  const soup = loadSoup()
  const dwc = soup.dryWetCycle
  const rhoWet = N30 / (BOX30[0] * BOX30[1] * BOX30[2])
  expect(N30).toBe(32884)
  expect(rhoWet).toBeCloseTo(1.21793, 5)

  // (1) The configured amplitude concentrates at liquid water; the pre-liquid-water value does not.
  expect(dwc.targetDryDensity).toBeGreaterThan(rhoWet)
  const stale = { ...soup, dryWetCycle: { ...dwc, targetDryDensity: 0.6 } }
  expect(() => deriveCycleConfig(stale as never, { box: BOX30, dryWetCycle: true } as never, BOX30, N30, START30)).toThrow(
    /does not exceed the current/,
  )

  // Below the close-packed density of unit-diameter cores -- above it the "dry" medium would be a
  // high-pressure artefact rather than a liquid (loud-failure-and-liquid-water-report.md §3.2).
  expect(dwc.targetDryDensity).toBeLessThan(Math.SQRT2)

  // (2) The dry box realises the configured density, and cycling derives it from N, not from a box
  // literal, so the same config value means the same physics at any box.
  const { cycleCfg, dryBox } = deriveCycleConfig(soup, { box: BOX30, dryWetCycle: true } as never, BOX30, N30, START30)
  expect(cycleCfg).toBeDefined()
  const dryDensity = N30 / (dryBox[0] * dryBox[1] * dryBox[2])
  expect(dryDensity).toBeCloseTo(dwc.targetDryDensity, 6)
  expect(dryBox[0]).toBeCloseTo(computeDryBox(BOX30, N30, dwc.targetDryDensity)[0], 10)
  // Concentration factor -- the number the report states against the literature's ~1400x.
  expect(dryDensity / rhoWet).toBeCloseTo(dwc.targetDryDensity / rhoWet, 10)

  // (3) The schedule: wet first (step 0 already IS the wet box), dry for the last dryFraction of the
  // period, and one transition at each boundary.
  const wetLen = dwc.periodSteps * (1 - dwc.dryFraction)
  expect(cyclePhaseAt(0, cycleCfg!)).toEqual({ phase: 'wet', cycleIndex: 1 })
  expect(cyclePhaseAt(wetLen - 1, cycleCfg!).phase).toBe('wet')
  expect(cyclePhaseAt(wetLen, cycleCfg!).phase).toBe('dry')
  expect(cyclePhaseAt(dwc.periodSteps * dwc.cycles, cycleCfg!)).toEqual({ phase: 'wet', cycleIndex: 0 })
  expect(nextCycleTransition(0, cycleCfg!)).toBe(wetLen)
  expect(nextCycleTransition(wetLen, cycleCfg!)).toBe(dwc.periodSteps)
  expect(nextCycleTransition(dwc.periodSteps * dwc.cycles, cycleCfg!)).toBe(Infinity)

  console.log(
    `DRYWET-CONFIG rhoWet=${rhoWet.toFixed(5)} targetDryDensity=${dwc.targetDryDensity} dryBox=${dryBox[0].toFixed(4)} ` +
      `realisedDryDensity=${dryDensity.toFixed(5)} concentrationFactor=${(dryDensity / rhoWet).toFixed(4)} ` +
      `cycles=${dwc.cycles} periodSteps=${dwc.periodSteps} dryFraction=${dwc.dryFraction} wetSegment=${wetLen} drySegment=${dwc.periodSteps - wetLen}`,
  )
})

test('dry-wet cycling at liquid water: the box returns exactly, the state stays finite, and the dry phase SUPPRESSES the chemistry against a matched control', async () => {
  const soup = loadSoup()
  const dwc = soup.dryWetCycle
  // A smaller box than the report's baseline, at the SAME densities (water 0.8 sigma^-3,
  // rho_org 0.41793, rho_tot 1.2179) -- one full cycle of the file's own schedule has to fit in a
  // test, and the mechanism is a density effect, not a box-size effect.
  const box: [number, number, number] = [20, 20, 20]
  const start = { C: 551, O: 2204, H: 551, M: 37, W: 6400 }
  const wetLen = dwc.periodSteps * (1 - dwc.dryFraction)
  const page = await gpuPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  try {
    const out = await page.evaluate(
      async (boxArg: number[], startArg: Record<string, number>, wetLenArg: number, periodArg: number, msdWindow: number) => {
        const api = (window as any).api
        const box3 = boxArg as [number, number, number]
        const N = Object.values(startArg).reduce((a, b) => a + b, 0)

        const msd = (p0: Float32Array, p1: Float32Array, bx: number[]): number => {
          let s = 0
          const n = p0.length / 4
          for (let i = 0; i < n; i++) {
            let acc = 0
            for (let ax = 0; ax < 3; ax++) {
              const d = p1[i * 4 + ax] - p0[i * 4 + ax]
              const m = d - Math.round(d / bx[ax]) * bx[ax]
              acc += m * m
            }
            s += acc
          }
          return s / n
        }
        const totalEvents = (ev: Record<string, number>) => ev.cc_bond + ev.co_bond

        // Two systems, identical seed/composition/box, differing ONLY in whether cycling is on.
        const cycled = await api.createSoup({ box: box3, seed: 19, kT: 1.1, start: startArg, dryWetCycle: true, clay: false })
        const control = await api.createSoup({ box: box3, seed: 19, kT: 1.1, start: startArg, clay: false })
        await cycled.relaxColdStart()
        await control.relaxColdStart()

        // --- wet segment: run to just short of the transition, sampling mobility over msdWindow ---
        await cycled.stepCycled(wetLenArg - msdWindow - 200)
        await control.stepCycled(wetLenArg - msdWindow - 200)
        const wetP0 = await cycled.particles()
        const wetEv0 = totalEvents(await cycled.events())
        await cycled.stepCycled(msdWindow)
        const wetP1 = await cycled.particles()
        const wetEv1 = totalEvents(await cycled.events())
        const wetBoxLive = cycled.box as [number, number, number]
        const msdWet = msd(wetP0, wetP1, wetBoxLive)
        const nfWet = await cycled.nonFiniteCount()

        // --- cross the wet->dry transition (applyBoxScale's ramp runs inside stepCycled) ---
        const stepsBefore = cycled.steps
        await cycled.stepCycled(200)
        const dryBoxLive = cycled.box as [number, number, number]
        const rampSteps = cycled.steps - stepsBefore - 200
        const nfAfterRamp = await cycled.nonFiniteCount()

        // --- dry segment: the SAME mobility window, at the SAME schedule position in the control ---
        const dryP0 = await cycled.particles()
        const dryEv0 = totalEvents(await cycled.events())
        await cycled.stepCycled(msdWindow)
        const dryP1 = await cycled.particles()
        const dryEv1 = totalEvents(await cycled.events())
        const msdDry = msd(dryP0, dryP1, dryBoxLive)
        const nfDry = await cycled.nonFiniteCount()

        await control.stepCycled(cycled.steps - control.steps)
        const ctlP0 = await control.particles()
        const ctlEv0 = totalEvents(await control.events())
        await control.stepCycled(msdWindow)
        const ctlP1 = await control.particles()
        const ctlEv1 = totalEvents(await control.events())
        const msdCtl = msd(ctlP0, ctlP1, control.box as [number, number, number])

        // --- finish the cycle: rehydrate, then compare the whole cycle's chemistry ---
        await cycled.stepCycled(periodArg - cycled.steps + rampSteps)
        const boxAfterRehydration = cycled.box as [number, number, number]
        await control.stepCycled(cycled.steps - control.steps)
        const evCycledEnd = totalEvents(await cycled.events())
        const evControlEnd = totalEvents(await control.events())
        const invCycled = await cycled.invariants()
        const invControl = await control.invariants()
        const nfEnd = await cycled.nonFiniteCount()
        const stepsEnd = { cycled: cycled.steps, control: control.steps }
        const phaseEnd = cycled.cyclePhase
        cycled.dispose()
        control.dispose()
        return {
          N,
          wetBoxLive,
          dryBoxLive,
          boxAfterRehydration,
          rampSteps,
          msdWet,
          msdDry,
          msdCtl,
          wetRate: (wetEv1 - wetEv0) / msdWindow,
          dryRate: (dryEv1 - dryEv0) / msdWindow,
          ctlRate: (ctlEv1 - ctlEv0) / msdWindow,
          evCycledEnd,
          evControlEnd,
          invCycled,
          invControl,
          nf: { wet: nfWet, afterRamp: nfAfterRamp, dry: nfDry, end: nfEnd },
          stepsEnd,
          phaseEnd,
        }
      },
      box as unknown as number[],
      start,
      wetLen,
      dwc.periodSteps,
      2000,
    )

    console.log(
      `DRYWET-CYCLE N=${out.N} wetBox=${out.wetBoxLive[0].toFixed(4)} dryBox=${out.dryBoxLive[0].toFixed(4)} ` +
        `rhoDry=${(out.N / out.dryBoxLive[0] ** 3).toFixed(5)} rampStepsCharged=${out.rampSteps} ` +
        `boxAfterRehydration=${out.boxAfterRehydration[0].toFixed(6)} phaseEnd=${out.phaseEnd}`,
    )
    console.log(
      `DRYWET-MOBILITY msd(wet,cycled)=${out.msdWet.toFixed(4)} msd(dry,cycled)=${out.msdDry.toFixed(4)} ` +
        `msd(same window, wet control)=${out.msdCtl.toFixed(4)} collapse=${(out.msdCtl / out.msdDry).toFixed(2)}x`,
    )
    console.log(
      `DRYWET-CHEMISTRY rate wet=${out.wetRate.toFixed(5)}/step dry=${out.dryRate.toFixed(5)}/step control(same window)=${out.ctlRate.toFixed(5)}/step ` +
        `oneCycleTotal cycled=${out.evCycledEnd} control=${out.evControlEnd} ratio=${(out.evCycledEnd / out.evControlEnd).toFixed(4)} ` +
        `steps=${JSON.stringify(out.stepsEnd)} nonFinite=${JSON.stringify(out.nf)}`,
    )

    // Geometry: the dry box realises the configured density and rehydration returns EXACTLY.
    expect(out.N / out.dryBoxLive[0] ** 3).toBeCloseTo(dwc.targetDryDensity, 5)
    expect(out.boxAfterRehydration[0]).toBe(box[0])
    expect(out.boxAfterRehydration[1]).toBe(box[1])
    expect(out.boxAfterRehydration[2]).toBe(box[2])
    // The ramp really ran and really charged its relaxation steps to the trajectory.
    expect(out.rampSteps).toBe((dwc.rampSteps - 1) * dwc.rampRelaxSteps)

    // The loud-failure guard is on through the densest instant of the cycle.
    for (const [where, c] of Object.entries(out.nf)) {
      expect((c as { pos: number; vel: number }).pos, `nonFinite positions at ${where}`).toBe(0)
      expect((c as { pos: number; vel: number }).vel, `nonFinite velocities at ${where}`).toBe(0)
    }
    // Nothing was created or destroyed by the box changes.
    expect(out.invCycled.monomers).toEqual(out.invControl.monomers)
    expect(out.invCycled.charge).toBe(0)
    expect(out.invControl.charge).toBe(0)

    // The mechanism: mobility collapses at the dry box, measured against the SAME step window in the
    // wet control (so this is not the run's own slowing-down with time).
    expect(out.msdDry).toBeLessThan(out.msdCtl / 2)
    // And the chemistry over one whole cycle is SUPPRESSED, not driven.
    expect(out.evCycledEnd).toBeLessThan(out.evControlEnd)
    expect(out.dryRate).toBeLessThan(out.ctlRate)
  } finally {
    await shutdownGpu()
  }
}, 600_000)
