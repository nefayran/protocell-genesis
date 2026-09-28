import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { afterAll, expect, test } from 'vitest'
import literature from '../data/literature.json'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'acid-soap-pairing' (2026-08-23): GATE 1 AND GATE 2 OF THIS TASK, in one instrument.
//
// GATE 1 -- DOES THE BILAYER SURVIVE THE NEW TERM. A head-head attraction changes the whole force
// field, so the explicit-water bilayer's own area per lipid and thickness have to be RE-MEASURED at
// every strength swept, against the SAME literature corridors (1.1-1.5 sigma^2 and 4-6 sigma, read
// from data/literature.json, never widened).
//
// GATE 2 -- DOES THE PAIR ACTUALLY FORM AND LIVE. Paired fraction (soup/src/electrostatics.ts's
// pairingStats, the same rule the force uses) and pair LIFETIME (soup/src/acid-soap.ts's
// pairSurvival over a fixed step interval), both against strength.
//
// WHY THE CHARGED PATCH RATHER THAN THE PUBLISHED GATE'S OWN RUN. The published row
// (tests/water-bilayer-area-move.test.ts) is UNCHARGED: every charge is 0 there, so the acid-soap
// term is structurally absent and that run is bit-identical before and after this task -- which is
// worth proving but proves nothing about the term. The term only exists where the protonation state
// is a live variable, so this file runs the SAME construction with `electrostatics.enabled` on at the
// model's own pH, and sweeps the depth. Construction, lipid count, water density, starting area, seed
// and settle criterion are deliberately identical to that file's, so the s = 0 arm here is the
// charged control for its uncharged row and the two are comparable.
//
// SPLIT ACROSS INVOCATIONS: ACID_SOAP_STRENGTHS selects which depths this invocation measures, so the
// sweep fits the 500 s per-invocation ceiling; each arm appends to the artifact rather than replacing
// it.

const AREA_MIN = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.min!
const AREA_MAX = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.max!
const THICKNESS_MIN = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.min!
const THICKNESS_MAX = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.max!
const AREA_START = (AREA_MIN + AREA_MAX) / 2

const ARTIFACT = process.env.ACID_SOAP_ARTIFACT ?? 'verify/out/acid-soap-strength-sweep.json'
/** The depths measured by THIS invocation, already normalised (1.0 = the apolar-apolar reference). */
const STRENGTHS = (process.env.ACID_SOAP_STRENGTHS ?? '0')
  .split(',')
  .map((s) => Number(s.trim()))
  .filter((s) => Number.isFinite(s))
const PH = Number(process.env.ACID_SOAP_PH ?? '7.0')
// Same schedule shape as tests/water-bilayer-area-move.test.ts, at a shorter cap: this is a SWEEP of
// seven arms, not the published single row, and the settle criterion below is what decides whether an
// arm's number is a plateau. TAIL/BLOCKS keep that file's own statistic.
const EQUILIBRATE_STEPS = 2000
const CHUNK_STEPS = 500
const TRIALS_PER_CHUNK = 10
const CAP = Number(process.env.ACID_SOAP_CAP ?? '300')
const TAIL = Number(process.env.ACID_SOAP_TAIL ?? '100')
const BLOCKS = 6

test(
  'acid-soap pair: a strength sweep -- bilayer area/thickness, paired fraction and pair lifetime',
  async () => {
    const page = await gpuPage()
    const rows: any[] = []
    for (const S of STRENGTHS) {
      const result = await page.evaluate(
        async (
          nLipids: number,
          areaStart: number,
          waterDensity: number,
          boxZ: number,
          seed: number,
          equilibrateSteps: number,
          chunkSteps: number,
          trialsPerChunk: number,
          cap: number,
          tail: number,
          blocks: number,
          acidSoap: number,
          pH: number,
        ) => {
          const api = (window as any).api
          const soup = api.loadSoup()
          const p = api.loadParams()
          const kT = p.thermostat.kT

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

          const L = Math.sqrt((nLipids / 2) * areaStart)
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
            clay: false,
            start: startCounts,
            // The heads sit in a bilayer SLAB, not spread through the box: tell the long-range list
            // sizer the volume they actually occupy (leaflet span plus the attraction reach on each
            // side) instead of letting it divide by the whole box and come out several-fold short.
            // The engine throws by name when the list is short, which is how this was found.
            electrostatics: { enabled: true, pH, densityVolumeSigma3: L * L * (2 * halfSpan + 2 * bWT) },
            acidSoapScaleOverride: acidSoap,
            resume: {
              globalStep: 0,
              liveBox: box,
              positions: positions.slice(0, actualN * 4),
              velocities: new Float32Array(actualN * 4),
              bondSlots: bondSlots.slice(0, actualN * 3),
              centerLink: new Uint32Array(actualN).fill(0xffffffff),
              centerHeldSteps: new Uint32Array(actualN),
              desorbEvents: new Uint32Array(2),
              bondRng: Uint32Array.from({ length: actualN }, (_, i) => i + 1),
              thermoRng: Uint32Array.from({ length: actualN }, (_, i) => i + 1000003),
              events: {},
            },
          })

          // The head-head contact radius, from data/params.json's rank-A beadSizes -- pairingStats'
          // own definition of "touching", not a chosen number.
          const contact = api.wcaCutoff(p.sigma * p.beadSizes.head_head)
          let prevPairs: Map<number, number> | null = null
          let prevStep = 0

          async function measureAll(step: number): Promise<any> {
            const particlesNow: Float32Array = await sys.particles()
            const bondsNow: Uint32Array = await sys.bonds()
            const charges: Float32Array = await sys.charges()
            const liveBox: [number, number, number] = sys.box
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
            const profile = api.densityProfileZ(memberPositions, liveBox, 200)
            let thickness: number | null = null
            let lower: number | null = null
            let upper: number | null = null
            try {
              const peaks = api.bilayerPeaks(profile)
              lower = peaks.lower
              upper = peaks.upper
              thickness = peaks.upper - peaks.lower
            } catch (e: any) {
              thickness = null
            }
            const memberRadii = soup.monomers
              .filter((mo: any) => mo.kind === 'carbon' || mo.kind === 'head')
              .map((mo: any) => mo.radiusSigma)
            const cutoff = api.wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
            const remapped = particlesNow.slice()
            for (let i = 0; i < nNow; i++) remapped[i * 4 + 3] = memberSet.has(i) ? 1 : 0
            const clusterFraction = api.largestClusterFraction(remapped, liveBox, cutoff)
            let waterInCore = 0
            let totalWater = 0
            let headBuried = 0
            let headTotal = 0
            if (lower !== null && upper !== null) {
              for (let i = 0; i < nNow; i++) {
                if (Math.round(particlesNow[i * 4 + 3]) !== waterKind) continue
                totalWater++
                const z = particlesNow[i * 4 + 2]
                if (z > lower && z < upper) waterInCore++
              }
              for (const i of memberSet) {
                if (soup.monomers[Math.round(particlesNow[i * 4 + 3])].kind !== 'head') continue
                headTotal++
                const z = particlesNow[i * 4 + 2]
                if (Math.abs(z - lower) > 1 && Math.abs(z - upper) > 1) headBuried++
              }
            } else {
              for (let i = 0; i < nNow; i++) if (Math.round(particlesNow[i * 4 + 3]) === waterKind) totalWater++
            }
            // --- GATE 2: pairing population and pair lifetime -------------------------------
            const es = sys.electrostatics()
            const pr = api.pairingStats(particlesNow, charges, liveBox, es, contact)
            const heads: number[] = []
            for (let i = 0; i < nNow; i++) if (Math.round(particlesNow[i * 4 + 3]) === headKind) heads.push(i)
            const pairsNow = api.nearestUnlike(particlesNow, charges, heads, liveBox, contact)
            const surv = prevPairs === null ? null : api.pairSurvival(prevPairs, pairsNow, step - prevStep)
            prevPairs = pairsNow
            prevStep = step
            return {
              step,
              box: liveBox,
              areaPerLipid: api.areaPerLipid(liveBox, nLipids),
              thickness,
              clusterFraction,
              waterInCore,
              totalWater,
              headBuriedFraction: headTotal > 0 ? headBuried / headTotal : null,
              amphiphileCount: amph.length,
              alpha: pr.alpha,
              pairedFraction: pr.pairedFraction,
              unlikeFraction: pr.unlikeFraction,
              unlikeFractionRandom: pr.unlikeFractionRandom,
              pairsNow: surv ? surv.now : pairsNow.size,
              pairSurvivalFraction: surv ? surv.survivalFraction : null,
              pairRepartnered: surv ? surv.repartnered : null,
              pairLifetimeSteps: surv ? (Number.isFinite(surv.lifetimeSteps) ? surv.lifetimeSteps : null) : null,
              pairLifetimeCensored: surv ? !Number.isFinite(surv.lifetimeSteps) : null,
              survivalInterval: surv ? surv.stepsElapsed : null,
            }
          }

          function blockDrift(ys: number[], nb: number): { perStep: number; stdErr: number; t: number } {
            const per = Math.floor(ys.length / nb)
            const xs: number[] = []
            const ms: number[] = []
            for (let b = 0; b < nb; b++) {
              let acc = 0
              for (let i = b * per; i < (b + 1) * per; i++) acc += ys[i]
              ms.push(acc / per)
              xs.push(b * per + (per - 1) / 2)
            }
            const xm = xs.reduce((a, b) => a + b, 0) / nb
            const ym = ms.reduce((a, b) => a + b, 0) / nb
            let sxy = 0
            let sxx = 0
            for (let b = 0; b < nb; b++) {
              sxy += (xs[b] - xm) * (ms[b] - ym)
              sxx += (xs[b] - xm) ** 2
            }
            const slope = sxy / sxx
            let sse = 0
            for (let b = 0; b < nb; b++) sse += (ms[b] - ym - slope * (xs[b] - xm)) ** 2
            const se = Math.sqrt(sse / (nb - 2) / sxx)
            return { perStep: slope, stdErr: se, t: slope / se }
          }

          await sys.step(equilibrateSteps)
          const series: number[] = []
          const trajectory: any[] = []
          const t0 = performance.now()
          let acceptedTotal = 0
          let trialsTotal = 0
          let chunksUsed = cap
          let settled = false
          for (let i = 0; i < cap; i++) {
            await sys.step(chunkSteps)
            const mv = await sys.areaMove(trialsPerChunk)
            acceptedTotal += mv.accepted
            trialsTotal += mv.trials
            series.push(api.areaPerLipid(sys.box, nLipids))
            const idx = i + 1
            if (idx % 20 === 0) trajectory.push(await measureAll(equilibrateSteps + idx * chunkSteps))
            if (series.length >= tail && idx % tail === 0) {
              const w = series.slice(-tail)
              const drift = blockDrift(w.map((a) => Math.log(a)), blocks)
              const span = Math.log(Math.max(...w) / Math.min(...w))
              if (Math.abs(drift.t) < 2 && Math.abs(drift.perStep) * tail < 0.01 && span < 0.02) {
                settled = true
                chunksUsed = idx
                break
              }
            }
          }
          const wallClockMs = performance.now() - t0
          const tailWindow = series.slice(-tail)
          const drift = blockDrift(tailWindow.map((a) => Math.log(a)), blocks)
          const final = await measureAll(equilibrateSteps + chunksUsed * chunkSteps)
          const esFinal = sys.electrostatics()
          sys.dispose()
          // Average the pairing statistics over the trajectory samples rather than reading the last
          // frame: the paired fraction is a fluctuating population, and one frame is its wander.
          const tailSamples = trajectory.slice(Math.max(0, trajectory.length - 6))
          const mean = (f: (t: any) => number | null): number | null => {
            const vs = tailSamples.map(f).filter((v): v is number => v !== null && Number.isFinite(v))
            return vs.length > 0 ? vs.reduce((a, b) => a + b, 0) / vs.length : null
          }
          return {
            acidSoap,
            pH,
            N: actualN,
            waterPlaced,
            settled,
            chunksUsed,
            stepsRun: equilibrateSteps + chunksUsed * chunkSteps,
            areaTailMean: tailWindow.reduce((a, b) => a + b, 0) / tailWindow.length,
            areaTailMin: Math.min(...tailWindow),
            areaTailMax: Math.max(...tailWindow),
            driftPerChunk: drift.perStep,
            driftT: drift.t,
            acceptedFraction: acceptedTotal / trialsTotal,
            trialsTotal,
            stepsPerSec: (chunksUsed * chunkSteps) / (wallClockMs / 1000),
            esCutoff: esFinal.cutoff,
            esDebyeSigma: esFinal.debyeSigma,
            trajectory,
            final,
            alphaMean: mean((t) => t.alpha),
            pairedFractionMean: mean((t) => t.pairedFraction),
            unlikeExcessMean: mean((t) => (t.unlikeFractionRandom > 0 ? t.unlikeFraction / t.unlikeFractionRandom : null)),
            pairSurvivalMean: mean((t) => t.pairSurvivalFraction),
            pairLifetimeMean: mean((t) => t.pairLifetimeSteps),
            survivalInterval: tailSamples.length > 0 ? tailSamples[tailSamples.length - 1].survivalInterval : null,
          }
        },
        400,
        AREA_START,
        0.8,
        30,
        31,
        EQUILIBRATE_STEPS,
        CHUNK_STEPS,
        TRIALS_PER_CHUNK,
        CAP,
        TAIL,
        BLOCKS,
        S,
        PH,
      )

      const inArea = result.areaTailMin >= AREA_MIN && result.areaTailMax <= AREA_MAX
      const inThickness =
        result.final.thickness !== null && result.final.thickness >= THICKNESS_MIN && result.final.thickness <= THICKNESS_MAX
      console.log(
        `ACID-SOAP-SWEEP S=${result.acidSoap} pH=${result.pH} N=${result.N} settled=${result.settled} ` +
          `chunks=${result.chunksUsed} steps=${result.stepsRun} ` +
          `area=${result.areaTailMean.toFixed(4)} [${result.areaTailMin.toFixed(4)}, ${result.areaTailMax.toFixed(4)}] ` +
          `corridor ${AREA_MIN}-${AREA_MAX} -> ${inArea ? 'inside' : 'outside'} | ` +
          `thickness=${result.final.thickness === null ? 'N/A' : result.final.thickness.toFixed(4)} ` +
          `corridor ${THICKNESS_MIN}-${THICKNESS_MAX} -> ${inThickness ? 'inside' : 'outside'} | ` +
          `drift(lnA)=${result.driftPerChunk.toExponential(2)} t=${result.driftT.toFixed(2)} ` +
          `acc=${result.acceptedFraction.toFixed(3)} cluster=${result.final.clusterFraction.toFixed(4)} ` +
          `buried=${result.final.headBuriedFraction === null ? 'N/A' : result.final.headBuriedFraction.toFixed(4)} ` +
          `water_in_core=${result.final.waterInCore}/${result.final.totalWater} ` +
          `amph=${result.final.amphiphileCount}/400 ` +
          `alpha=${result.alphaMean === null ? 'N/A' : result.alphaMean.toFixed(4)} ` +
          `paired=${result.pairedFractionMean === null ? 'N/A' : result.pairedFractionMean.toFixed(4)} ` +
          `unlike_excess=${result.unlikeExcessMean === null ? 'N/A' : result.unlikeExcessMean.toFixed(3)} ` +
          `survival=${result.pairSurvivalMean === null ? 'N/A' : result.pairSurvivalMean.toFixed(4)} ` +
          `over ${result.survivalInterval} steps -> lifetime=${result.pairLifetimeMean === null ? 'N/A(censored)' : result.pairLifetimeMean.toFixed(0)} steps ` +
          `throughput=${result.stepsPerSec.toFixed(1)} steps/s`,
      )
      console.log(
        'ACID-SOAP-SWEEP-TRAJECTORY S=' + result.acidSoap + '\n  ' +
          result.trajectory
            .map(
              (t: any) =>
                `step=${t.step} area=${t.areaPerLipid.toFixed(4)} thick=${t.thickness?.toFixed(3) ?? 'N/A'} ` +
                `buried=${t.headBuriedFraction?.toFixed(4) ?? 'N/A'} cluster=${t.clusterFraction.toFixed(4)} ` +
                `alpha=${t.alpha.toFixed(4)} paired=${t.pairedFraction.toFixed(4)} ` +
                `surv=${t.pairSurvivalFraction === null ? 'N/A' : t.pairSurvivalFraction.toFixed(4)} ` +
                `pairs=${t.pairsNow}`,
            )
            .join('\n  '),
      )
      rows.push({
        ...result,
        trajectory: result.trajectory.map((t: any) => ({ ...t, box: t.box.map((v: number) => Number(v.toFixed(4))) })),
        corridors: { area: [AREA_MIN, AREA_MAX], thickness: [THICKNESS_MIN, THICKNESS_MAX] },
        inArea,
        inThickness,
        verdict: result.settled && inArea && inThickness ? 'passed' : 'FAILED',
      })
    }

    // Append rather than replace: the sweep is split across invocations to respect the per-invocation
    // wall-clock ceiling, and an arm measured in an earlier invocation must not be lost.
    mkdirSync(dirname(ARTIFACT), { recursive: true })
    const prev = existsSync(ARTIFACT) ? JSON.parse(readFileSync(ARTIFACT, 'utf8')) : { arms: [] }
    const arms = (prev.arms ?? []).filter((a: any) => !rows.some((r) => r.acidSoap === a.acidSoap && r.pH === a.pH))
    writeFileSync(
      ARTIFACT,
      JSON.stringify(
        {
          measuredBy: 'tests/soup-acid-soap-bilayer.test.ts',
          generatedAt: new Date().toISOString(),
          lipids: 400,
          waterDensity: 0.8,
          protocol: { EQUILIBRATE_STEPS, CHUNK_STEPS, TRIALS_PER_CHUNK, CAP, TAIL, BLOCKS },
          arms: [...arms, ...rows].sort((a: any, b: any) => a.pH - b.pH || a.acidSoap - b.acidSoap),
        },
        null,
        2,
      ),
    )
    console.log(`ACID-SOAP-SWEEP WROTE ${ARTIFACT} (${arms.length + rows.length} arms)`)
    expect(rows.length).toBe(STRENGTHS.length)
    // Every arm must be a MEASUREMENT, not a snapshot of a wandering coordinate: the box moved, the
    // acceptance is in a usable Metropolis band. Whether it stays inside the literature corridors is
    // the RESULT and is reported, not asserted here -- this file's job is to measure the sweep, and
    // the strength this task settles on is pinned separately (tests/soup-acid-soap.test.ts).
    for (const r of rows) {
      expect(r.acceptedFraction).toBeGreaterThan(0.01)
      expect(r.acceptedFraction).toBeLessThan(0.99)
      expect(r.trialsTotal).toBeGreaterThan(0)
    }
  },
  1_500_000,
)
