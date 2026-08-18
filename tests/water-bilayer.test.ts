import { afterAll, expect, test } from 'vitest'
import literature from '../data/literature.json'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'explicit-water' (2026-08-18): re-validation gate for the SOUP engine (dynamic topology,
// catalytic growth) against the same literature corridor tests/gate6-bilayer.test.ts already checks
// for the MEMBRANE engine (fixed 3-bead lipid, no water) -- this is a genuinely different model
// (this task's own brief: "these are a DIFFERENT model from the solvent-free one whose parameters
// were validated"), and had never itself been checked against area-per-lipid/bilayer-thickness with
// a controlled patch before this task, water or no water.
//
// HONEST LIMITATION, stated up front rather than glossed over: the soup engine has NO zero-tension
// area-move/barostat mechanism (unlike the membrane engine's own areaMove(), which is what lets
// gate6-bilayer.test.ts measure a CONVERGED, equilibrium area). The box here is fixed for the whole
// run, at an area-per-lipid this test CHOOSES (the literature corridor's own midpoint) -- so
// `areaPerLipid` below is not an independently measured equilibrium property, it is the assumed
// starting condition. What IS a genuine, unbiased measurement, not dictated by that choice: the
// bilayer THICKNESS (head-peak separation, from the relaxed configuration's own density profile) and
// whether the patch survives as one intact, hydrophobically SEALED sheet (largestClusterFraction,
// and water penetration into the hydrophobic core) at that assumed area -- if the soup engine's own
// species sizes/interactions were wildly incompatible with the literature area, this would show up
// as rupture, interdigitation, or a leaky core, not merely as a genuine measurement it happens to
// disagree with. Per this task's own brief: "no statement about membranes in water has any force"
// beyond what these two checks (a not-independently-verified area, and a genuinely measured
// thickness/integrity) actually support.
//
// SMALL, per the resource rule: 400 single-tailed (head + 2 tail carbons, the SAME 3-bead Cooke &
// Deserno topology the literature corridor's own "conditions" describe, matching gate6-bilayer's own
// convention rather than this soup's separately-motivated two-tailed chainCapacity=2 default) lipids
// (1200 membrane beads) pre-wired directly via CreateSoupOpts.resume (bondSlots/centerLink built by
// hand, bypassing the slow catalytic-growth path entirely -- this task is about re-validating
// ASSEMBLED-MEMBRANE physics in water, not about re-running the growth chemistry, which is untouched
// by this task and already exercised by the regression tests) plus a few thousand explicit water
// beads filling the rest of the box -- thousands of particles total, not hundreds of thousands.

const AREA_MIN = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.min!
const AREA_MAX = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.max!
const THICKNESS_MIN = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.min!
const THICKNESS_MAX = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.max!
const AREA_TARGET = (AREA_MIN + AREA_MAX) / 2 // corridor midpoint -- the same convention
// soup/src/aggregates.ts's derivedMinAmphiphilesPerAggregate() already uses for an
// already-literature-anchored starting point, not a new assumption.
const CHECKPOINTS = [500, 1000, 2000, 4000, 8000, 15000, 25000, 40000]

test(
  'бислойная заплатка в явной воде: площадь на липид (принятая) и толщина (измеренная) против литературного коридора',
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

        // Lattice patch, same construction as engine/src/sim.ts's own layoutBilayer (Task 5):
        // square footprint sized to `areaTarget`, two leaflets, tails meeting near the midplane.
        const L = Math.sqrt((nLipids / 2) * areaTarget)
        const box: [number, number, number] = [L, L, boxZ]
        const z0 = boxZ / 2
        const gap = bTT / 2
        const halfSpan = gap + bTT + bHT // mid-to-head radial offset, one leaflet

        const perLeaflet = Math.ceil(nLipids / 2)
        const nSide = Math.max(1, Math.ceil(Math.sqrt(perLeaflet)))
        const spacingX = box[0] / nSide
        const spacingY = box[1] / nSide
        const jitterFrac = 0.15 // soup/src/sim.ts's own default-layout jitter fraction, reused verbatim

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
          // Head's own claim (role 2): slot 0 -> tail1.
          bondSlots[headIdx * 3 + 0] = tail1Idx
          // tail1: slot 2 (head slot, role 1) -> head; slot 0 (chain pool, role 0) -> tail2.
          bondSlots[tail1Idx * 3 + 2] = headIdx
          bondSlots[tail1Idx * 3 + 0] = tail2Idx
          // tail2: slot 0 (chain pool) -> tail1.
          bondSlots[tail2Idx * 3 + 0] = tail1Idx
          k += 3
        }

        // Water: a jittered lattice over the WHOLE box at `waterDensity`, skipping any site whose z
        // falls within the membrane's own excluded band (half-span plus a contact-distance margin so
        // no water bead spawns already overlapping a freshly-placed tail bead) -- same jitter
        // philosophy as soup/src/sim.ts's own default lattice (jitterFrac=0.15), sized for water's own
        // spacing rather than reusing the lipid lattice's.
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
          if (zRaw > zLo && zRaw < zHi) continue // inside the membrane's own excluded band
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
        // Any remaining pre-allocated slots (N was an upper-bound estimate; the excluded-band skip
        // means fewer water sites survive than N implies) collapse to the ACTUAL count used --
        // report the real total, not the pre-allocation, and trim every per-particle buffer to match.
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

        // createSoup MERGES this over data/soup.json's own default `start` (does not replace it),
        // so every monomer id this test does not want (donor H, catalyst M -- no catalytic growth
        // happens in this pre-assembled patch) must be explicitly zeroed here, not merely omitted.
        const startCounts: Record<string, number> = { [soup.monomers[headKind].id]: nLipids, [soup.monomers[carbonKind].id]: nLipids * 2, [soup.monomers[waterKind].id]: waterPlaced }
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

        // measureAll(): one full snapshot's worth of derived observables, factored out so the
        // trajectory loop below and the final measurement share exactly one implementation.
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

          return { thickness, thicknessError, lower, upper, clusterFraction, waterInCore, totalWaterFinal, amphiphileCount: amph.length, bondsAfter: bondsNow.length / 2 }
        }

        // Trajectory: sample the full measurement at several checkpoints along the way, not just at
        // the end -- needed to tell "still relaxing toward the corridor" apart from "found a
        // different equilibrium" before spending more GPU budget on either explanation.
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
      0.7,
      30,
      31,
      1.1,
      CHECKPOINTS,
    )

    console.log('WATER-BILAYER', JSON.stringify(result))
    console.log(
      'WATER-BILAYER-TRAJECTORY ' +
        result.trajectory.map((t: any) => `step=${t.step} thickness=${t.thickness?.toFixed(3) ?? 'N/A'} cluster=${t.clusterFraction.toFixed(4)}`).join(' | '),
    )

    expect(result.N).toBeGreaterThan(0)
    expect(result.bondsAfter).toBeGreaterThan(0)
    // Report against the corridor -- see this file's header for why areaPerLipid is the ASSUMED
    // starting condition (no area-move mechanism), not an independent measurement, while thickness
    // and integrity genuinely are.
    console.log(
      `WATER-BILAYER-VERDICT areaPerLipid(assumed)=${result.areaPerLipid.toFixed(4)} [corridor ${AREA_MIN}-${AREA_MAX}] ` +
        `thickness(measured)=${result.thickness?.toFixed(4) ?? 'N/A: ' + result.thicknessError} [corridor ${THICKNESS_MIN}-${THICKNESS_MAX}] ` +
        `clusterFraction=${result.clusterFraction.toFixed(4)} waterInCore=${result.waterInCore}/${result.totalWaterFinal} ` +
        `throughput=${result.stepsPerSec.toFixed(2)} steps/s at N=${result.N}`,
    )
  },
  590_000,
)
