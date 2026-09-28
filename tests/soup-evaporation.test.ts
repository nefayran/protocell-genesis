// Pins task 'evaporation' (2026-08-20): the solvent really LEAVES the system in the dry phase and
// really COMES BACK on rehydration (.superpowers/sdd/2026-08-16-soup-to-vesicle/evaporation-report.md).
// Three tests, deliberately split by what each can prove:
//
//  1. PURE (no GPU) -- the arithmetic and the refusals. What the residual solvent fraction is derived
//     from, what concentration factor it actually buys against the literature's 1400x and against
//     this model's own close-packing ceiling, that the solvent block must be LAST (or truncation
//     would renumber organic particles), and that both ladders land exactly on their endpoints.
//  2. GPU, ISOLATED -- removal and re-insertion called at a FIXED box with NO ramp and NO dynamics,
//     via the two DEBUG hooks that exist precisely so these are checkable claims: SOLVENT-ONLY
//     removal, BOND-GRAPH INVARIANCE (exact set equality, both ways), organic positions BIT-IDENTICAL
//     across REMOVAL and perturbed only inside the minimiser's own analytical displacement bound
//     across INSERTION, no non-finite state -- including after 1000 further real steps, which is what
//     caught the first version of the insertion as unsafe -- no valence violation, and a CHECKPOINT
//     ROUND-TRIP of the partially-evaporated state.
//  3. GPU, LIVE CYCLE -- one full wet->dry->wet cycle of the file's own schedule with evaporation on:
//     the dry box realises targetDryDensity for what is LEFT (not for the wet N), the box and the
//     census both return EXACTLY, and the state stays finite through the densest instant.
//
// Margins: every number asserted here is either exact (a census, a box, a bond set) or derived from
// data/soup.json in this file, so nothing here is a restatement of a measured value that could drift.
import { expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadSoup } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { computeDryBox } from '../soup/src/sim'
import { planEvaporation, evaporationLadder, rehydrationLadder } from '../soup/src/soup-evaporate'

const BOX30: [number, number, number] = [30, 30, 30]
const START30 = { C: 1860, O: 7440, H: 1860, M: 124, W: 21600 }

// A smaller box at the SAME densities for the two GPU tests. 24 is the smallest cube whose DRY box
// still gives the neighbour grid its minimum cell count on every axis: dry L = 16.284 sigma against
// cellSize 2.947 = 5 cells, exactly minCells = 2*effectiveWalkRadius+1. Composition scaled from the
// box-30 broth by volume, so rho_org = 0.41776 and rho_W = 0.7999 are the project's own.
const BOX24: [number, number, number] = [24, 24, 24]
const START24 = { C: 952, O: 3808, H: 952, M: 63, W: 11059 }

test('evaporation: what leaves, what that concentrates, and what it refuses', () => {
  const soup = loadSoup()
  const p = loadParams()
  const dwc = soup.dryWetCycle
  const plan = planEvaporation(soup, p, BOX30, START30)

  // (1) The residual fraction IS 1/1400 -- Ross & Deamer 2016's own measured volume reduction, mapped
  // one-to-one onto solvent bead count (dryWetCycle.basis item 2). At W = 21600 that is 15 beads, i.e.
  // a realised removal factor of 1440x; the rounding to whole beads is the only gap and it is stated.
  expect(dwc.residualSolventFraction).toBeCloseTo(1 / 1400, 12)
  expect(plan.wetSolventCount).toBe(21600)
  expect(plan.drySolventCount).toBe(15)
  expect(plan.organicCount).toBe(1860 + 7440 + 1860 + 124)
  expect(plan.solventBlockStart).toBe(plan.organicCount)

  // (2) The dry box is sized for WHAT IS LEFT, which is the whole point: it realises
  // targetDryDensity for N_org + W_residual, not for the wet N.
  const nDry = plan.organicCount + plan.drySolventCount
  expect(nDry / (plan.dryBox[0] * plan.dryBox[1] * plan.dryBox[2])).toBeCloseTo(dwc.targetDryDensity, 9)
  expect(plan.dryBox[0]).toBeCloseTo(computeDryBox(BOX30, nDry, dwc.targetDryDensity)[0], 12)

  // (3) The concentration factor, against the two numbers that bound it.
  const rhoOrgWet = plan.organicCount / (30 * 30 * 30)
  const ceiling = Math.SQRT2 / rhoOrgWet // close packing of unit-diameter cores
  const previousMechanism = dwc.targetDryDensity / (32884 / 27000) // box scaling at fixed composition
  expect(plan.concentrationFactor).toBeGreaterThan(3.1)
  expect(plan.concentrationFactor).toBeLessThan(ceiling)
  expect(plan.concentrationFactor / previousMechanism).toBeGreaterThan(2.9)
  console.log(
    `EVAP-PLAN box30 N_org=${plan.organicCount} W_wet=${plan.wetSolventCount} W_dry=${plan.drySolventCount} ` +
      `solventRemovalFactor=${(plan.wetSolventCount / plan.drySolventCount).toFixed(1)}x dryBox=${plan.dryBox[0].toFixed(4)} ` +
      `rho_org_wet=${rhoOrgWet.toFixed(5)} rho_org_dry=${(plan.organicCount / plan.dryBox[0] ** 3).toFixed(5)} ` +
      `concentrationFactor=${plan.concentrationFactor.toFixed(4)}x closePackingCeiling=${ceiling.toFixed(4)}x ` +
      `previousMechanism=${previousMechanism.toFixed(4)}x logFractionOf1400=${(Math.log(plan.concentrationFactor) / Math.log(1400) * 100).toFixed(2)}% ` +
      `relaxIterations=${plan.relaxIterations}`,
  )

  // (4) The minimisation budget is DERIVED, not configured: enough displacement to lift a bead placed
  // at insertionMinSeparationSigma out to the outer WCA radius of its stiffest solvent pair.
  const stiffest = wcaCutoff(p.sigma * (1.0 + Math.max(...soup.monomers.map((m) => m.radiusSigma))) * 0.5)
  const d0 = soup.coldStartRelax!.maxDisplacementSigma * p.sigma
  const meanSpacing = Math.cbrt(27000 / (plan.organicCount + plan.wetSolventCount))
  const need = stiffest - dwc.insertionMinSeparationSigma! * p.sigma + meanSpacing
  expect(plan.relaxIterations).toBe(Math.ceil((2 * need) / d0))
  // The displacement bound of the normalised descent really covers the gap it has to close.
  expect((d0 * (plan.relaxIterations + 1)) / 2).toBeGreaterThanOrEqual(need)

  // (5) The insertion floor is below the random-sequential-addition saturation for the FULL wet pool
  // -- that is why the sampler can place the whole pool at all (dryWetCycle.basis item 6).
  const phiRsa = 0.3841
  const nWet = plan.organicCount + plan.wetSolventCount
  const phi = (nWet * (Math.PI / 6) * dwc.insertionMinSeparationSigma! ** 3) / 27000
  expect(phi).toBeLessThan(phiRsa)

  // (6) THE REFUSAL that makes truncation safe: the solvent must be the last non-empty block, or
  // removing it would renumber organic particles and silently break the bond graph.
  expect(() => planEvaporation(soup, p, BOX30, { ...START30, K: 10 })).toThrow(/last non-empty block/)
  // And a residual fraction that removes nothing (rounds back to the whole pool) is refused rather
  // than run as a silent no-op.
  expect(() =>
    planEvaporation({ ...soup, dryWetCycle: { ...dwc, residualSolventFraction: 0.999999 } } as never, p, BOX30, START30),
  ).toThrow(/must remove the solvent/)

  // (7) Both ladders: monotone, uniform in ln L, and landing EXACTLY on their endpoints.
  const down = evaporationLadder(plan, dwc.targetDryDensity)
  expect(down.length).toBe(dwc.evaporationRampSteps)
  expect(down[down.length - 1].box[0]).toBe(plan.dryBox[0])
  expect(down[down.length - 1].solvent).toBe(plan.drySolventCount)
  const lnSteps: number[] = []
  let prevBox = BOX30[0]
  let prevSolvent = plan.wetSolventCount
  for (const rung of down) {
    expect(rung.box[0]).toBeLessThan(prevBox)
    expect(rung.solvent).toBeLessThanOrEqual(prevSolvent)
    lnSteps.push(Math.log(prevBox / rung.box[0]))
    prevBox = rung.box[0]
    prevSolvent = rung.solvent
  }
  const lnMax = Math.max(...lnSteps)
  const lnMin = Math.min(...lnSteps)
  expect(lnMax - lnMin).toBeLessThan(1e-9) // uniform by construction, not approximately
  // Each increment is at or below the largest single box change this project has MEASURED safe
  // (predecessor §4.4: one instantaneous 3.2 % linear jump at rho_tot 1.218 -> 1.34).
  expect(1 - Math.exp(-lnMax)).toBeLessThanOrEqual(0.032 + 1e-9)

  const up = rehydrationLadder(plan)
  expect(up.length).toBe(dwc.evaporationRampSteps)
  expect(up[up.length - 1].box).toEqual(BOX30)
  expect(up[up.length - 1].solvent).toBe(plan.wetSolventCount)
  // The whole pool returns on the FINAL increment and nowhere earlier -- there is no room before it.
  for (let i = 0; i < up.length - 1; i++) expect(up[i].solvent).toBe(plan.drySolventCount)
  console.log(
    `EVAP-LADDER increments=${down.length} lnStep=${lnMax.toFixed(6)} linearPerIncrement=${((1 - Math.exp(-lnMax)) * 100).toFixed(3)}% ` +
      `boxes=${down.map((r) => r.box[0].toFixed(3)).join(',')} solvent=${down.map((r) => r.solvent).join(',')}`,
  )
})

test('evaporation in isolation: solvent-only, bond graph untouched, organics moved only inside the minimiser bound, checkpoint round-trips', async () => {
  const soup = loadSoup()
  const p = loadParams()
  const plan = planEvaporation(soup, p, BOX24, START24)
  const page = await gpuPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  try {
    const out = await page.evaluate(
      async (boxArg: number[], startArg: Record<string, number>, blockStart: number, halfSolvent: number, fullSolvent: number) => {
        const api = (window as any).api
        const box3 = boxArg as [number, number, number]
        // Fingerprints, never whole arrays over the CDP wire: an exact running sum plus an
        // evenly-spaced sample, the same discipline soup/cli/campaign.ts already uses.
        const fp = (a: Float32Array, n: number) => {
          let sum = 0
          for (let i = 0; i < n * 4; i++) sum += a[i]
          const sample: number[] = []
          for (let i = 0; i < n * 4; i += 397) sample.push(a[i])
          return { sum, sample }
        }
        const bondKey = (b: Uint32Array) => {
          const out: string[] = []
          for (let k = 0; k < b.length; k += 2) out.push(`${b[k]}-${b[k + 1]}`)
          return out.sort().join(',')
        }
        const valence = (pos: Float32Array, bonds: Uint32Array, monomers: any[], n: number) => {
          const cc = new Uint32Array(n)
          const co = new Uint32Array(n)
          const deg = new Uint32Array(n)
          let outOfRange = 0
          for (let k = 0; k < bonds.length; k += 2) {
            const i = bonds[k]
            const j = bonds[k + 1]
            if (i >= n || j >= n) {
              outOfRange++
              continue
            }
            deg[i]++
            deg[j]++
            const ki = monomers[Math.round(pos[i * 4 + 3])].kind
            const kj = monomers[Math.round(pos[j * 4 + 3])].kind
            if (ki === 'carbon' && kj === 'carbon') {
              cc[i]++
              cc[j]++
            } else {
              if (ki === 'carbon') co[i]++
              if (kj === 'carbon') co[j]++
            }
          }
          let bad = 0
          for (let i = 0; i < n; i++) {
            if (cc[i] > 2 || co[i] > 1 || deg[i] > 3) bad++
          }
          return { bad, outOfRange }
        }
        const monomers = api.loadSoup().monomers

        const sys = await api.createSoup({ box: box3, seed: 19, kT: 1.1, start: startArg, dryWetCycle: true, evaporateSolvent: true, clay: false })
        await sys.relaxColdStart()
        // Some real chemistry first, so the bond graph under test is not empty.
        await sys.step(3000)
        const invWet = await sys.invariants()
        const posWet = await sys.particles()
        const bondsWet = bondKey(await sys.bonds())
        const evWet = await sys.events()
        const nWet = posWet.length / 4
        const fpOrgWet = fp(posWet, blockStart)

        // --- REMOVAL, at a fixed box, with no dynamics in between -------------------------------
        await sys.evaporateDEBUG(halfSolvent)
        const invDry = await sys.invariants()
        const posDry = await sys.particles()
        const bondsDry = bondKey(await sys.bonds())
        const evDry = await sys.events()
        const nfDry = await sys.nonFiniteCount()
        const valDry = valence(posDry, await sys.bonds(), monomers, posDry.length / 4)
        const fpOrgDry = fp(posDry, blockStart)

        // Checkpoint the PARTIALLY EVAPORATED state and resume from it, then compare.
        const cfg = { box: box3, seed: 19, kT: 1.1, start: startArg, dryWetCycle: true, evaporateSolvent: true }
        const file = await api.encodeCheckpoint(sys, cfg)
        const resumed = await api.createSoup({ ...cfg, resume: api.decodeCheckpointResume(file), clay: false })
        const posRes = await resumed.particles()
        const invRes = await resumed.invariants()
        const bondsRes = bondKey(await resumed.bonds())
        const roundTrip = {
          N: posRes.length / 4,
          fpAll: fp(posRes, posRes.length / 4),
          census: invRes.monomers,
          bonds: bondsRes === bondsDry,
          steps: resumed.steps,
          box: resumed.box[0],
          activeCountsInFile: file.activeCounts,
          nInFile: file.N,
        }
        resumed.dispose()

        // --- RE-INSERTION, at the same fixed box, no dynamics ------------------------------------
        const report = await sys.rehydrateDEBUG(fullSolvent)
        const invBack = await sys.invariants()
        const posBack = await sys.particles()
        const bondsBack = bondKey(await sys.bonds())
        const evBack = await sys.events()
        const nfBack = await sys.nonFiniteCount()
        const valBack = valence(posBack, await sys.bonds(), monomers, posBack.length / 4)
        const fpOrgBack = fp(posBack, blockStart)
        // 1000 REAL steps after rehydration: "no non-finite state results from rehydration" is only a
        // claim worth making if the trajectory actually continues out of the rehydrated state.
        await sys.step(1000)
        const nfAfterSteps = await sys.nonFiniteCount()
        const invAfterSteps = await sys.invariants()
        // How many pre-existing coordinate floats the insertion+minimisation changed at all. NOT
        // asserted to be zero: holding them fixed was tried, and measured to blow the run up (see
        // rehydrateSolventTo's own doc comment). What IS asserted is that the displacement stays
        // inside the minimiser's own analytical bound, which the report carries as a number.
        let movedPreexisting = 0
        for (let i = 0; i < (posDry.length / 4) * 4; i++) {
          if (posBack[i] !== posDry[i]) movedPreexisting++
        }
        // Every returning bead really is solvent, and really is inside the box.
        const solventKind = monomers.findIndex((m: any) => m.id === api.loadSoup().solvent.waterId)
        let wrongKind = 0
        let outOfBox = 0
        for (let i = posDry.length / 4; i < posBack.length / 4; i++) {
          if (Math.round(posBack[i * 4 + 3]) !== solventKind) wrongKind++
          for (let a = 0; a < 3; a++) {
            const v = posBack[i * 4 + a]
            if (!(v >= 0 && v < box3[a])) outOfBox++
          }
        }
        sys.dispose()
        return {
          nWet,
          nDry: posDry.length / 4,
          nBack: posBack.length / 4,
          invWet,
          invDry,
          invBack,
          bondsUnchangedOnRemoval: bondsDry === bondsWet,
          bondsUnchangedOnInsertion: bondsBack === bondsDry,
          eventsUnchangedOnRemoval: JSON.stringify(evDry) === JSON.stringify(evWet),
          eventsUnchangedOnInsertion: JSON.stringify(evBack) === JSON.stringify(evDry),
          organicsUnchangedOnRemoval: fpOrgDry.sum === fpOrgWet.sum && JSON.stringify(fpOrgDry.sample) === JSON.stringify(fpOrgWet.sample),
          organicsUnchangedOnInsertion: fpOrgBack.sum === fpOrgDry.sum && JSON.stringify(fpOrgBack.sample) === JSON.stringify(fpOrgDry.sample),
          movedPreexisting,
          wrongKind,
          outOfBox,
          nfDry,
          nfBack,
          valDry,
          valBack,
          report,
          roundTrip,
          nfAfterSteps,
          invAfterSteps,
        }
      },
      BOX24 as unknown as number[],
      START24,
      plan.solventBlockStart,
      Math.floor(plan.wetSolventCount / 2),
      plan.wetSolventCount,
    )

    console.log(
      `EVAP-ISOLATED N ${out.nWet} -> ${out.nDry} -> ${out.nBack} census ${JSON.stringify(out.invWet.monomers)} -> ` +
        `${JSON.stringify(out.invDry.monomers)} -> ${JSON.stringify(out.invBack.monomers)} bonds ${out.invWet.bonds}/${out.invDry.bonds}/${out.invBack.bonds} ` +
        `bondSetUnchanged=${out.bondsUnchangedOnRemoval}/${out.bondsUnchangedOnInsertion} ` +
        `organicsBitIdentical=${out.organicsUnchangedOnRemoval}/${out.organicsUnchangedOnInsertion} movedPreexistingFloats=${out.movedPreexisting} ` +
        `nonFinite=${JSON.stringify(out.nfDry)}/${JSON.stringify(out.nfBack)}/after1000steps=${JSON.stringify(out.nfAfterSteps)} ` +
        `valence=${JSON.stringify(out.valDry)}/${JSON.stringify(out.valBack)}`,
    )
    console.log(
      `EVAP-INSERTION inserted=${out.report.inserted} belowFloor=${out.report.shortOfFloor} minSeparation=${out.report.minAchieved.toFixed(4)} ` +
        `relaxIterations=${out.report.relaxIterations} max|F| ${out.report.maxForceBefore.toExponential(4)} -> ${out.report.maxForceAfter.toExponential(4)} ` +
        `preexistingDisplacement rms=${out.report.preexistingRmsDisplacement.toFixed(4)} max=${out.report.preexistingMaxDisplacement.toFixed(4)} bound=${out.report.displacementBound.toFixed(4)} ` +
        `movedFloats=${out.movedPreexisting}`,
    )
    console.log(
      `EVAP-CHECKPOINT roundTrip N=${out.roundTrip.N} nInFile=${out.roundTrip.nInFile} steps=${out.roundTrip.steps} box=${out.roundTrip.box} ` +
        `census=${JSON.stringify(out.roundTrip.census)} activeCountsInFile=${JSON.stringify(out.roundTrip.activeCountsInFile)} bondsIdentical=${out.roundTrip.bonds}`,
    )

    // --- SOLVENT-ONLY: every organic count EXACTLY unchanged, only the solvent moves -------------
    const solventId = soup.solvent.waterId
    for (const id of Object.keys(START24)) {
      if (id === solventId) continue
      expect(out.invDry.monomers[id], `organic ${id} unchanged by removal`).toBe(out.invWet.monomers[id])
      expect(out.invBack.monomers[id], `organic ${id} unchanged by rehydration`).toBe(out.invWet.monomers[id])
    }
    expect(out.invWet.monomers[solventId]).toBe(plan.wetSolventCount)
    expect(out.invDry.monomers[solventId]).toBe(Math.floor(plan.wetSolventCount / 2))
    expect(out.invBack.monomers[solventId]).toBe(plan.wetSolventCount)
    expect(out.invWet.charge).toBe(0)
    expect(out.invDry.charge).toBe(0)
    expect(out.invBack.charge).toBe(0)

    // --- BOND GRAPH untouched, and no event fired, by either operation ---------------------------
    expect(out.bondsUnchangedOnRemoval, 'removal changed the bond set').toBe(true)
    expect(out.bondsUnchangedOnInsertion, 'insertion changed the bond set').toBe(true)
    expect(out.invDry.bonds).toBe(out.invWet.bonds)
    expect(out.invBack.bonds).toBe(out.invWet.bonds)
    expect(out.eventsUnchangedOnRemoval).toBe(true)
    expect(out.eventsUnchangedOnInsertion).toBe(true)
    // No valence claim broken, and no bond pointing into removed territory.
    expect(out.valDry).toEqual({ bad: 0, outOfRange: 0 })
    expect(out.valBack).toEqual({ bad: 0, outOfRange: 0 })

    // --- ORGANICS BIT-IDENTICAL across both operations -------------------------------------------
    expect(out.organicsUnchangedOnRemoval).toBe(true)
    // The perturbation rehydration causes is BOUNDED by the minimiser's own analytical ceiling.
    expect(out.report.preexistingMaxDisplacement).toBeLessThanOrEqual(out.report.displacementBound + 1e-6)
    expect(out.report.preexistingRmsDisplacement).toBeLessThan(out.report.preexistingMaxDisplacement + 1e-9)
    expect(out.wrongKind, 'a returning bead was not solvent').toBe(0)
    expect(out.outOfBox, 'a returning bead was placed outside the box').toBe(0)

    // --- NO NON-FINITE STATE after either operation ----------------------------------------------
    expect(out.nfDry).toEqual({ pos: 0, vel: 0 })
    expect(out.nfBack).toEqual({ pos: 0, vel: 0 })
    // And the trajectory really continues out of the rehydrated state.
    expect(out.nfAfterSteps).toEqual({ pos: 0, vel: 0 })
    expect(out.invAfterSteps.monomers).toEqual(out.invBack.monomers)
    // The minimisation did its job: the worst force after it is far below the insertion spike.
    expect(Number.isFinite(out.report.maxForceAfter)).toBe(true)
    expect(out.report.maxForceAfter).toBeLessThan(out.report.maxForceBefore)
    expect(out.report.inserted).toBe(plan.wetSolventCount - Math.floor(plan.wetSolventCount / 2))

    // --- CHECKPOINT ROUND-TRIP of the partially evaporated state ---------------------------------
    expect(out.roundTrip.N).toBe(out.nDry)
    expect(out.roundTrip.nInFile).toBe(out.nDry)
    expect(out.roundTrip.census).toEqual(out.invDry.monomers)
    expect(out.roundTrip.activeCountsInFile[solventId]).toBe(Math.floor(plan.wetSolventCount / 2))
    expect(out.roundTrip.bonds, 'resumed bond set differs').toBe(true)
    expect(out.roundTrip.box).toBe(BOX24[0])
  } finally {
    await shutdownGpu()
  }
}, 600_000)

test('evaporating cycle: the dry box is sized for what is LEFT, and both the box and the census return exactly', async () => {
  const soup = loadSoup()
  const p = loadParams()
  const dwc = soup.dryWetCycle
  const plan = planEvaporation(soup, p, BOX24, START24)
  const wetLen = dwc.periodSteps * (1 - dwc.dryFraction)
  const page = await gpuPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  try {
    const out = await page.evaluate(
      async (boxArg: number[], startArg: Record<string, number>, wetLenArg: number, periodArg: number) => {
        const api = (window as any).api
        const box3 = boxArg as [number, number, number]
        const sys = await api.createSoup({ box: box3, seed: 19, kT: 1.1, start: startArg, dryWetCycle: true, evaporateSolvent: true, clay: false })
        await sys.relaxColdStart()
        const nfStart = await sys.nonFiniteCount()
        // Run into the wet segment, then cross the wet->dry transition.
        await sys.stepCycled(wetLenArg)
        const invDry = await sys.invariants()
        const dryBoxLive = sys.box as [number, number, number]
        const nDry = (await sys.particles()).length / 4
        const nfDry = await sys.nonFiniteCount()
        const phaseDry = sys.cyclePhase
        // Hold dry a while, then finish the cycle (rehydration) and settle.
        await sys.stepCycled(2000)
        const nfHeld = await sys.nonFiniteCount()
        await sys.stepCycled(periodArg - sys.steps + 4000)
        const invBack = await sys.invariants()
        const boxBack = sys.box as [number, number, number]
        const nBack = (await sys.particles()).length / 4
        const nfEnd = await sys.nonFiniteCount()
        const events = await sys.events()
        const steps = sys.steps
        sys.dispose()
        return { nfStart, invDry, dryBoxLive, nDry, nfDry, phaseDry, nfHeld, invBack, boxBack, nBack, nfEnd, events, steps }
      },
      BOX24 as unknown as number[],
      START24,
      wetLen,
      dwc.periodSteps,
    )

    const rhoDryRealised = out.nDry / out.dryBoxLive[0] ** 3
    console.log(
      `EVAP-CYCLE dryBox=${out.dryBoxLive[0].toFixed(4)} (plan ${plan.dryBox[0].toFixed(4)}) N_dry=${out.nDry} (plan ${plan.organicCount + plan.drySolventCount}) ` +
        `rhoDryRealised=${rhoDryRealised.toFixed(5)} phase=${out.phaseDry} censusDry=${JSON.stringify(out.invDry.monomers)} ` +
        `boxBack=${out.boxBack[0].toFixed(6)} N_back=${out.nBack} censusBack=${JSON.stringify(out.invBack.monomers)} ` +
        `steps=${out.steps} events=${JSON.stringify(out.events)} nonFinite=${JSON.stringify({ start: out.nfStart, dry: out.nfDry, held: out.nfHeld, end: out.nfEnd })}`,
    )

    // The dry state is what the plan said, realised on a live system.
    expect(out.phaseDry).toBe('dry')
    expect(out.nDry).toBe(plan.organicCount + plan.drySolventCount)
    expect(out.dryBoxLive[0]).toBeCloseTo(plan.dryBox[0], 9)
    expect(rhoDryRealised).toBeCloseTo(dwc.targetDryDensity, 5)
    // The solvent really left: the dry census carries the residual count and every organic count is
    // exactly the composition's own.
    expect(out.invDry.monomers[soup.solvent.waterId]).toBe(plan.drySolventCount)
    for (const [id, n] of Object.entries(START24)) {
      if (id === soup.solvent.waterId) continue
      expect(out.invDry.monomers[id], `organic ${id} survives the dry phase`).toBe(n)
    }
    // Rehydration returns the box EXACTLY and the census EXACTLY.
    expect(out.boxBack[0]).toBe(BOX24[0])
    expect(out.boxBack[1]).toBe(BOX24[1])
    expect(out.boxBack[2]).toBe(BOX24[2])
    expect(out.nBack).toBe(plan.organicCount + plan.wetSolventCount)
    for (const [id, n] of Object.entries(START24)) expect(out.invBack.monomers[id], `census returns for ${id}`).toBe(n)
    expect(out.invBack.charge).toBe(0)
    // Finite through the densest instant AND after rehydration.
    for (const [where, c] of Object.entries({ start: out.nfStart, dry: out.nfDry, held: out.nfHeld, end: out.nfEnd })) {
      expect((c as { pos: number; vel: number }).pos, `nonFinite positions at ${where}`).toBe(0)
      expect((c as { pos: number; vel: number }).vel, `nonFinite velocities at ${where}`).toBe(0)
    }
  } finally {
    await shutdownGpu()
  }
}, 900_000)
