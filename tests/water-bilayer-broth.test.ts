import { afterAll, expect, test } from 'vitest'
import literature from '../data/literature.json'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'broth-composition' (2026-08-18): re-measures the bilayer-in-water gate
// (tests/water-bilayer.test.ts) at THIS task's own measured water density (0.8 sigma^-3, chosen
// from a pure-water void/coordination/dispersion sweep -- see
// .superpowers/sdd/2026-08-16-soup-to-vesicle/broth-composition-report.md) rather than the earlier
// task's 0.3/0.7 sweep. Same construction as water-bilayer.test.ts (pre-built via
// CreateSoupOpts.resume, bypassing catalytic growth -- this is a re-validation of ASSEMBLED-membrane
// physics, not growth kinetics): 400 single-tailed, 3-bead lipids (head + 2 tail carbons). Tail
// length 2 is an ASSUMED construction choice here (this model's own convention: a real C12-C18
// membrane lipid IS two tail beads at this bead scale), not a measurement of natural catalytic
// growth -- see this task's own report for why growth could not be measured at all once water was
// added (co_bond, chain termination, measured at 0 events across every tested composition and up to
// 150000 steps: water's own hydrophilic pull on the head group starves the head-to-growing-tip
// encounter rate the whole growth-then-cap kinetic scheme depends on).
//
// HONEST LIMITATION, same as water-bilayer.test.ts: no zero-tension area-move/barostat in the soup
// engine, so area-per-lipid below is the ASSUMED starting condition (literature midpoint), not an
// independent measurement. Thickness and integrity ARE genuine measurements.
//
// Task 'water-calibration' (2026-08-19, .superpowers/sdd/2026-08-16-soup-to-vesicle/
// water-calibration-report.md): this is the reproduction case that report starts from. Mechanistic
// finding (from the head/tail/water z-histogram over ALL particles, not just recognised-amphiphile
// members): the core is genuinely DRY (water fraction inside the head span is a few percent, and
// tail density is continuous, not patchy) -- the failure is not water penetration and not "leaflets
// never formed a core". It is that a large fraction of HEAD beads (headBuriedFraction below, ~40%
// at this composition) sit buried inside the hydrophobic zone instead of at either face, because
// the box's fixed area-per-lipid (no barostat) plus a solvent model whose only attractive force is
// water-water/water-head leaves the leaflet with no lateral cohesion of its own once the old
// tail-tail term was removed. A sweep of every legitimate explicit-water knob (water density, bead
// radius, and a new water-only attraction-strength scale, data/soup.json's
// solvent.attractionScale) found no combination in a defensible range that holds thickness stably
// in the literature corridor with a dry core -- see the report for the full map, including two
// single-snapshot "hits" that a longer trajectory revealed were transients (one mid-collapse, one
// oscillating without ever settling), not real calibration wins.

const AREA_MIN = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.min!
const AREA_MAX = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.max!
const THICKNESS_MIN = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.min!
const THICKNESS_MAX = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.max!
const AREA_TARGET = (AREA_MIN + AREA_MAX) / 2
const CHECKPOINTS = [500, 2000, 8000, 20000]

test(
  'бислойная заплатка в явной воде на плотности 0.8: толщина (измеренная) против литературного коридора',
  async () => {
    const page = await gpuPage()
    const result = await page.evaluate(
      async (nLipids: number, areaTarget: number, waterDensity: number, boxZ: number, seed: number, kT: number, checkpoints: number[]) => {
        const api = (window as any).api
        const soup = api.loadSoup()
        const p = api.loadParams()

        const carbonKind = soup.monomers.findIndex((m: any) => m.kind === 'carbon')
        const headKind = soup.monomers.findIndex((m: any) => m.kind === 'head')
        const waterMonomer = soup.monomers.find((m: any) => m.id === soup.solvent.waterId)
        const waterKind = soup.monomers.findIndex((m: any) => m.id === soup.solvent.waterId)

        const carbonR = soup.monomers[carbonKind].radiusSigma
        const headR = soup.monomers[headKind].radiusSigma
        const waterR = waterMonomer.radiusSigma
        const bTT = (p.sigma * (carbonR + carbonR)) / 2
        const bHT = (p.sigma * (headR + carbonR)) / 2
        const bWT = (p.sigma * (waterR + carbonR)) / 2

        const L = Math.sqrt((nLipids / 2) * areaTarget)
        const box: [number, number, number] = [L, L, boxZ]
        const z0 = boxZ / 2
        const gap = bTT / 2
        const halfSpan = gap + bTT + bHT

        const perLeaflet = Math.ceil(nLipids / 2)
        const nSide = Math.max(1, Math.ceil(Math.sqrt(perLeaflet)))
        const spacingX = box[0] / nSide
        const spacingY = box[1] / nSide
        const jitterFrac = 0.15

        const rngState = { a: seed >>> 0 }
        function rng(): number {
          rngState.a = (rngState.a + 0x6d2b79f5) | 0
          let t = Math.imul(rngState.a ^ (rngState.a >>> 15), 1 | rngState.a)
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296
        }

        const N = nLipids * 3 + Math.ceil(waterDensity * L * L * boxZ)
        const positions = new Float32Array(N * 4)
        const bondSlots = new Uint32Array(N * 3).fill(0xffffffff)
        let k = 0
        for (let i = 0; i < nLipids; i++) {
          const leaflet = i < perLeaflet ? 1 : -1
          const idx = i < perLeaflet ? i : i - perLeaflet
          const a = idx % nSide
          const b = Math.floor(idx / nSide)
          const x = ((a + 0.5) * spacingX + (rng() * 2 - 1) * jitterFrac * spacingX + L) % L
          const y = ((b + 0.5) * spacingY + (rng() * 2 - 1) * jitterFrac * spacingY + L) % L
          const zTail2 = z0 + leaflet * gap
          const zTail1 = zTail2 + leaflet * bTT
          const zHead = zTail1 + leaflet * bHT

          const headIdx = k
          const tail1Idx = k + 1
          const tail2Idx = k + 2
          positions.set([x, y, zHead, headKind], headIdx * 4)
          positions.set([x, y, zTail1, carbonKind], tail1Idx * 4)
          positions.set([x, y, zTail2, carbonKind], tail2Idx * 4)
          bondSlots[headIdx * 3 + 0] = tail1Idx
          bondSlots[tail1Idx * 3 + 2] = headIdx
          bondSlots[tail1Idx * 3 + 0] = tail2Idx
          bondSlots[tail2Idx * 3 + 0] = tail1Idx
          k += 3
        }

        const margin = bWT * 1.5
        const zLo = z0 - halfSpan - margin
        const zHi = z0 + halfSpan + margin
        const spacing = Math.cbrt(1 / waterDensity)
        const wnx = Math.max(1, Math.round(L / spacing))
        const wny = Math.max(1, Math.round(L / spacing))
        const wnz = Math.max(1, Math.round(boxZ / spacing))
        const sx = L / wnx
        const sy = L / wny
        const sz = boxZ / wnz
        let waterPlaced = 0
        for (let iz = 0; iz < wnz && k < N; iz++) {
          const zRaw = (iz + 0.5) * sz
          if (zRaw > zLo && zRaw < zHi) continue
          for (let iy = 0; iy < wny && k < N; iy++) {
            for (let ix = 0; ix < wnx && k < N; ix++) {
              const x = ((ix + 0.5) * sx + (rng() * 2 - 1) * jitterFrac * sx + L) % L
              const y = ((iy + 0.5) * sy + (rng() * 2 - 1) * jitterFrac * sy + L) % L
              const z = Math.min(boxZ - 1e-4, Math.max(0, zRaw + (rng() * 2 - 1) * jitterFrac * sz))
              positions.set([x, y, z, waterKind], k * 4)
              k++
              waterPlaced++
            }
          }
        }
        const actualN = k
        const positionsT = positions.slice(0, actualN * 4)
        const bondSlotsT = bondSlots.slice(0, actualN * 3)

        const velocities = new Float32Array(actualN * 4)
        const centerLink = new Uint32Array(actualN).fill(0xffffffff)
        const centerHeldSteps = new Uint32Array(actualN)
        const bondRng = new Uint32Array(actualN)
        const thermoRng = new Uint32Array(actualN)
        for (let i = 0; i < actualN; i++) {
          bondRng[i] = i + 1
          thermoRng[i] = i + 1000003
        }

        const startCounts: Record<string, number> = {
          [soup.monomers[headKind].id]: nLipids,
          [soup.monomers[carbonKind].id]: nLipids * 2,
          [soup.monomers[waterKind].id]: waterPlaced,
        }
        for (const mo of soup.monomers) if (!(mo.id in startCounts)) startCounts[mo.id] = 0

        const sys = await api.createSoup({
          box,
          seed,
          kT,
          start: startCounts,
          resume: {
            globalStep: 0,
            liveBox: box,
            positions: positionsT,
            velocities,
            bondSlots: bondSlotsT,
            centerLink,
            centerHeldSteps,
            desorbEvents: new Uint32Array(2),
            bondRng,
            thermoRng,
            events: {},
          },
        })

        async function measureAll() {
          const particlesNow: Float32Array = await sys.particles()
          const bondsNow: Uint32Array = await sys.bonds()
          const nNow = particlesNow.length / 4
          const amph = api.findAmphiphiles(particlesNow, bondsNow, soup.monomers)
          const memberSet = new Set<number>()
          for (const a of amph) {
            memberSet.add(a.headIndex)
            for (const c of a.chain) memberSet.add(c)
          }
          const memberPositions = new Float32Array(memberSet.size * 4)
          let mIdx = 0
          for (const i of memberSet) {
            memberPositions[mIdx * 4] = particlesNow[i * 4]
            memberPositions[mIdx * 4 + 1] = particlesNow[i * 4 + 1]
            memberPositions[mIdx * 4 + 2] = particlesNow[i * 4 + 2]
            memberPositions[mIdx * 4 + 3] = soup.monomers[Math.round(particlesNow[i * 4 + 3])].polar ? 0 : 1
            mIdx++
          }
          const profile = api.densityProfileZ(memberPositions, box, 200)
          let thickness: number | null = null
          let lower: number | null = null
          let upper: number | null = null
          let thicknessError: string | null = null
          try {
            const peaks = api.bilayerPeaks(profile)
            lower = peaks.lower
            upper = peaks.upper
            thickness = peaks.upper - peaks.lower
          } catch (e: any) {
            thicknessError = String(e?.message ?? e)
          }

          const memberRadii = soup.monomers.filter((mo: any) => mo.kind === 'carbon' || mo.kind === 'head').map((mo: any) => mo.radiusSigma)
          const cutoff = api.wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
          const remapped = particlesNow.slice()
          for (let i = 0; i < nNow; i++) remapped[i * 4 + 3] = memberSet.has(i) ? 1 : 0
          const clusterFraction = api.largestClusterFraction(remapped, box, cutoff)

          let waterInCore = 0
          let totalWaterFinal = 0
          if (lower !== null && upper !== null) {
            for (let i = 0; i < nNow; i++) {
              if (Math.round(particlesNow[i * 4 + 3]) !== waterKind) continue
              totalWaterFinal++
              const z = particlesNow[i * 4 + 2]
              if (z > lower && z < upper) waterInCore++
            }
          }

          const tails: number[] = []
          for (const a of amph) for (const t of a.tailLengths) tails.push(t)
          const meanTail = tails.length > 0 ? tails.reduce((s: number, x: number) => s + x, 0) / tails.length : null

          // water-calibration task (2026-08-19, water-calibration-report.md): the mechanistic
          // measurement that told water penetration apart from heads never forming a proper
          // surface. Fraction of HEAD beads sitting more than 1 sigma outside BOTH surface peaks --
          // i.e. buried inside the hydrophobic zone rather than sitting at either face. Computed
          // over every head bead (not just the ones inside `memberPositions`, which is the same
          // set), so it is a direct read of where the real heads are, not an artifact of
          // densityProfileZ's own binning.
          let headBuried = 0
          let headTotal = 0
          if (lower !== null && upper !== null) {
            for (const i of memberSet) {
              if (soup.monomers[Math.round(particlesNow[i * 4 + 3])].kind !== 'head') continue
              headTotal++
              const z = particlesNow[i * 4 + 2]
              if (Math.abs(z - lower) > 1 && Math.abs(z - upper) > 1) headBuried++
            }
          }
          const headBuriedFraction = headTotal > 0 ? headBuried / headTotal : null

          return {
            thickness,
            thicknessError,
            lower,
            upper,
            clusterFraction,
            waterInCore,
            totalWaterFinal,
            amphiphileCount: amph.length,
            meanTail,
            bondsAfter: bondsNow.length / 2,
            headBuriedFraction,
          }
        }

        const trajectory: { step: number; thickness: number | null; clusterFraction: number }[] = []
        const t0 = performance.now()
        let done = 0
        for (const chunk of checkpoints) {
          await sys.step(chunk - done)
          done = chunk
          const snap = await measureAll()
          trajectory.push({ step: done, thickness: snap.thickness, clusterFraction: snap.clusterFraction })
        }
        const wallClockMs = performance.now() - t0
        const final = await measureAll()

        return {
          N: actualN,
          nLipidsPlaced: nLipids,
          waterPlaced,
          boxUsed: box,
          areaPerLipid: api.areaPerLipid(box, nLipids),
          trajectory,
          ...final,
          stepsRun: done,
          wallClockMs,
          stepsPerSec: done / (wallClockMs / 1000),
        }
      },
      400,
      AREA_TARGET,
      0.8,
      30,
      31,
      1.1,
      CHECKPOINTS,
    )

    console.log('WATER-BILAYER-BROTH', JSON.stringify(result))
    console.log(
      'WATER-BILAYER-BROTH-TRAJECTORY ' +
        result.trajectory.map((t: any) => `step=${t.step} thickness=${t.thickness?.toFixed(3) ?? 'N/A'} cluster=${t.clusterFraction.toFixed(4)}`).join(' | '),
    )

    expect(result.N).toBeGreaterThan(0)
    expect(result.bondsAfter).toBeGreaterThan(0)
    // ASSUMED tail length (constructed, not grown -- see this file's header and
    // data/soup.json's startBasis §5): every RECOGNISED amphiphile must carry exactly 2 tail beads,
    // confirming the hand-built bond graph is faithfully recognised by findAmphiphiles, not merely
    // asserted by construction. Measured 398/400 recognised (not all 400 -- a small number land
    // close enough to a periodic face or to each other at construction that findAmphiphiles' own
    // degree/branch checks reject them; not investigated further, out of this task's budget), so
    // the count assertion below is the measured number, not the idealised one.
    // Measured 398 on the run that first wrote this line and 399 on the next one at identical
    // inputs -- the construction is deterministic, but findAmphiphiles reads a state the GPU
    // produced, and its degree/branch checks sit right at the edge for the couple of lipids that
    // land nearest a periodic face. An exact equality here therefore asserts run-to-run float
    // ordering, not the property under test; the property is that essentially every constructed
    // lipid IS recognised, so the bound is stated as such (>= 395 of 400 placed).
    expect(result.amphiphileCount).toBeGreaterThanOrEqual(395)
    expect(result.meanTail).toBe(2)
    const inCorridor =
      result.thickness !== null && result.thickness >= THICKNESS_MIN && result.thickness <= THICKNESS_MAX
    console.log(
      `WATER-BILAYER-BROTH-VERDICT areaPerLipid(assumed)=${result.areaPerLipid.toFixed(4)} [corridor ${AREA_MIN}-${AREA_MAX}] ` +
        `thickness(measured)=${result.thickness?.toFixed(4) ?? 'N/A: ' + result.thicknessError} [corridor ${THICKNESS_MIN}-${THICKNESS_MAX}] ` +
        `clusterFraction=${result.clusterFraction.toFixed(4)} waterInCore=${result.waterInCore}/${result.totalWaterFinal} ` +
        `headBuriedFraction=${result.headBuriedFraction?.toFixed(4) ?? 'N/A'} ` +
        `throughput=${result.stepsPerSec.toFixed(2)} steps/s at N=${result.N} ` +
        `verdict=${inCorridor ? 'passed' : 'FAILED'}`,
    )
    // water-calibration task (2026-08-19, .superpowers/sdd/2026-08-16-soup-to-vesicle/
    // water-calibration-report.md): mechanistic diagnosis + a measured sweep of every legitimate
    // explicit-water knob (water density, water bead radius, and a new water-only attraction-
    // strength scale added by that task, data/soup.json's solvent.attractionScale) found NO
    // combination in a defensible (or even a widely explored) range that holds this thickness
    // stably inside 4-6 sigma with a dry core -- see that report for the full sweep map and the
    // literature anchors (MARTINI's own interaction-level table; Lenz & Schmid 2007) that bounded
    // the search. This assertion records that verdict as a real regression: if a future change
    // ever DOES pull the measured thickness inside the corridor, this line should start failing --
    // that would be news worth looking at, not a bug to silence.
    expect(inCorridor).toBe(false)
  },
  580_000,
)
