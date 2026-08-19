import { afterAll, expect, test } from 'vitest'
import literature from '../data/literature.json'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'hydrophobic-asymmetry' (2026-08-19), stage 2 of the re-measurement: the SAME
// bilayer-in-water construction tests/water-bilayer-broth.test.ts measures at a FIXED box (400
// single-tailed 3-bead lipids, 4500 water beads at 0.8 sigma^-3, seed 31, kT from data/params.json),
// but with the zero-tension Monte Carlo area move running, so area-per-lipid is a MEASUREMENT
// instead of the construction's own assumption.
//
// Why a second file rather than a flag on the first: water-broth's own value is that it pins the
// FIXED-box behaviour (its box never moves, so its thickness reading is not entangled with a
// relaxing area), and this task's report compares the two stages against each other. The patch
// construction below is therefore a deliberate copy of that file's own, kept parameter-identical
// (same lipid count, same water density, same starting area, same seed) -- if the two ever diverge
// the comparison stops meaning anything, so any edit to one must be mirrored.
//
// The move itself, what was ported from engine/src/sim-area-move.ts and how the port is verified
// (CPU potential vs GPU forces, and the generalised Jacobian identity), lives in
// soup/src/soup-area-move.ts and tests/soup-area-move.test.ts.

const AREA_MIN = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.min!
const AREA_MAX = literature.gates.find((g) => g.id === 'area-per-lipid')!.target.max!
const THICKNESS_MIN = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.min!
const THICKNESS_MAX = literature.gates.find((g) => g.id === 'bilayer-thickness')!.target.max!
const AREA_START = (AREA_MIN + AREA_MAX) / 2
// Sampling schedule. The area coordinate is not read at one instant: it is SAMPLED (one reading per
// chunk) and judged on a window, exactly as tests/gate6-bilayer.test.ts judges the solvent-free
// gate's own area move -- reading a Metropolis coordinate once is reading its wander, not its
// equilibrium value, and this task's own predecessor (water-calibration-report.md §3.1) was burned
// by exactly that. CAP bounds the search; TAIL is the window; the settle test (block-mean drift
// |t| < 2 AND |drift| * TAIL < 0.05 in ln(area)) is applied only at NON-OVERLAPPING TAIL-sized
// checkpoints, so it is not an uncorrected sequential test.
const EQUILIBRATE_STEPS = 2000
const CHUNK_STEPS = 500
// TRIALS_PER_CHUNK is a PROTOCOL knob (how fast the area chain is allowed to move relative to the
// configurational relaxation between chunks), not a model parameter: it cannot change the
// equilibrium distribution the Metropolis criterion samples, only the path taken to it. Measured,
// not guessed: at 40 trials per 500 MD steps, four runs at IDENTICAL inputs settled -- each with a
// block drift statistically indistinguishable from zero -- at four DIFFERENT areas (1.0858, 1.0864,
// 1.1181, 1.2048), with head burial rising from 0.07 to 0.39 as the area fell, i.e. the chain
// compressed the sheet faster than the sheet could rearrange and then froze into whichever
// collapsed state it happened to reach. See the task report for that table.
const TRIALS_PER_CHUNK = 10
const CAP = 900
const TAIL = 150
const BLOCKS = 6

test(
  'бислойная заплатка в явной воде: площадь на липид ИЗМЕРЕНА при нулевом натяжении',
  async () => {
    const page = await gpuPage()
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
          // Task 'clay-surface' (2026-08-19): the CLAY-FREE control. data/soup.json's shipped composition
          // now carries the mineral platelet (clay.enabled), and this fixture must not: it is the bilayer-in-water area gate; the area move refuses an immobile phase, and its `resume` fixture was built at a clay-free particle count.
          // A clay-free system is byte-identical to the pre-task engine, so every number in this file is
          // unchanged by that task -- which is exactly what makes it a usable reference.
          clay: false,
          start: startCounts,
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

        async function measureAll(): Promise<any> {
          const particlesNow: Float32Array = await sys.particles()
          const bondsNow: Uint32Array = await sys.bonds()
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
          }
          const tails: number[] = []
          for (const a of amph) for (const t of a.tailLengths) tails.push(t)
          return {
            box: liveBox,
            areaPerLipid: api.areaPerLipid(liveBox, nLipids),
            thickness,
            clusterFraction,
            waterInCore,
            totalWater,
            headBuriedFraction: headTotal > 0 ? headBuried / headTotal : null,
            amphiphileCount: amph.length,
            meanTail: tails.length > 0 ? tails.reduce((s: number, x: number) => s + x, 0) / tails.length : null,
          }
        }

        // Block-mean OLS drift on ln(area) -- the SAME statistic tests/gate6-bilayer.test.ts uses
        // for the solvent-free gate (reimplemented there too, because engine/src/index.ts does not
        // export it; this is not a different statistic).
        function blockDrift(ys: number[], blocks: number): { perStep: number; stdErr: number; t: number } {
          const per = Math.floor(ys.length / blocks)
          const xs: number[] = []
          const ms: number[] = []
          for (let b = 0; b < blocks; b++) {
            let acc = 0
            for (let i = b * per; i < (b + 1) * per; i++) acc += ys[i]
            ms.push(acc / per)
            xs.push(b * per + (per - 1) / 2)
          }
          const xm = xs.reduce((a, b) => a + b, 0) / blocks
          const ym = ms.reduce((a, b) => a + b, 0) / blocks
          let sxy = 0
          let sxx = 0
          for (let b = 0; b < blocks; b++) {
            sxy += (xs[b] - xm) * (ms[b] - ym)
            sxx += (xs[b] - xm) ** 2
          }
          const slope = sxy / sxx
          let sse = 0
          for (let b = 0; b < blocks; b++) sse += (ms[b] - ym - slope * (xs[b] - xm)) ** 2
          return { perStep: slope, stdErr: Math.sqrt(sse / (blocks - 2) / sxx), t: slope / Math.sqrt(sse / (blocks - 2) / sxx) }
        }

        // EQUILIBRATION: fixed box, no area move at all -- the same shape gate6 uses (nothing to
        // tune, nothing frozen, because nothing moved yet).
        await sys.step(equilibrateSteps)
        const areaAfterEquilibration = api.areaPerLipid(sys.box, nLipids)

        const series: number[] = []
        const trajectory: any[] = []
        const t0 = performance.now()
        let acceptedTotal = 0
        let trialsTotal = 0
        let chunksUsed = cap
        let settled = false
        let checkpointsChecked = 0
        for (let i = 0; i < cap; i++) {
          await sys.step(chunkSteps)
          const mv = await sys.areaMove(trialsPerChunk)
          acceptedTotal += mv.accepted
          trialsTotal += mv.trials
          series.push(api.areaPerLipid(sys.box, nLipids))
          const idx = i + 1
          if (idx % 20 === 0) {
            const snap = await measureAll()
            trajectory.push({
              chunk: idx,
              step: equilibrateSteps + idx * chunkSteps,
              ...snap,
              acceptedFraction: acceptedTotal / trialsTotal,
            })
          }
          if (series.length >= tail && idx % tail === 0) {
            checkpointsChecked++
            const w = series.slice(-tail)
            const drift = blockDrift(w.map((a) => Math.log(a)), blocks)
            // Three conditions, all required. |t| < 2 and the absolute drift bound are gate6's own
            // criterion; the two numeric bounds here are TIGHTER than gate6's, and the window-SPAN
            // condition is added, because the loose form was measured to accept a window that was
            // still sliding: one run stopped at its very first checkpoint with the window spanning
            // 1.2201-1.2939 in area (ln span 0.059) at |t| = 1.84 and |drift|*tail = 0.021 -- a
            // 6% slide called flat, and the head-burial assertion below then failed on it, which is
            // how it was caught. A 6-block OLS slope has little power against a slow, smooth creep,
            // so the span is bounded directly instead of only through the slope's significance.
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
        const final = await measureAll()
        const events = await sys.events()
        sys.dispose()
        return {
          N: actualN,
          waterPlaced,
          startBox: box,
          areaStart: api.areaPerLipid(box, nLipids),
          areaAfterEquilibration,
          series,
          trajectory,
          settled,
          chunksUsed,
          checkpointsChecked,
          tailMean: tailWindow.reduce((a, b) => a + b, 0) / tailWindow.length,
          tailMin: Math.min(...tailWindow),
          tailMax: Math.max(...tailWindow),
          driftPerChunk: drift.perStep,
          driftT: drift.t,
          acceptedFraction: acceptedTotal / trialsTotal,
          trialsTotal,
          events,
          ...final,
          stepsRun: equilibrateSteps + chunksUsed * chunkSteps,
          stepsPerSec: (chunksUsed * chunkSteps) / (wallClockMs / 1000),
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
    )

    console.log(
      'WATER-BILAYER-AREAMOVE ' +
        JSON.stringify({ ...result, series: undefined, trajectory: undefined }),
    )
    console.log(
      'WATER-BILAYER-AREAMOVE-TRAJECTORY\n  ' +
        result.trajectory
          .map(
            (t: any) =>
              `chunk=${t.chunk} step=${t.step} Lx=${t.box[0].toFixed(3)} Lz=${t.box[2].toFixed(3)} ` +
              `area=${t.areaPerLipid.toFixed(4)} thick=${t.thickness?.toFixed(3) ?? 'N/A'} ` +
              `buried=${t.headBuriedFraction?.toFixed(4) ?? 'N/A'} water=${t.waterInCore}/${t.totalWater} ` +
              `cluster=${t.clusterFraction.toFixed(4)} acc=${t.acceptedFraction.toFixed(3)}`,
          )
          .join('\n  '),
    )
    console.log(
      'WATER-BILAYER-AREAMOVE-AREA-SERIES ' +
        result.series.map((v: number, i: number) => (i % 5 === 0 ? v.toFixed(4) : null)).filter(Boolean).join(' '),
    )
    const inArea = result.tailMin >= AREA_MIN && result.tailMax <= AREA_MAX
    const inThickness = result.thickness !== null && result.thickness >= THICKNESS_MIN && result.thickness <= THICKNESS_MAX
    console.log(
      `WATER-BILAYER-AREAMOVE-VERDICT settled=${result.settled} chunks=${result.chunksUsed} ` +
        `areaPerLipid(MEASURED, tail mean)=${result.tailMean.toFixed(4)} [min ${result.tailMin.toFixed(4)}, ` +
        `max ${result.tailMax.toFixed(4)}, corridor ${AREA_MIN}-${AREA_MAX}] ` +
        `driftPerChunk(lnA)=${result.driftPerChunk.toExponential(3)} t=${result.driftT.toFixed(2)} ` +
        `thickness(measured)=${result.thickness?.toFixed(4) ?? 'N/A'} [corridor ${THICKNESS_MIN}-${THICKNESS_MAX}] ` +
        `clusterFraction=${result.clusterFraction.toFixed(4)} waterInCore=${result.waterInCore}/${result.totalWater} ` +
        `headBuriedFraction=${result.headBuriedFraction?.toFixed(4) ?? 'N/A'} ` +
        `acceptedFraction=${result.acceptedFraction.toFixed(4)} of ${result.trialsTotal} trials ` +
        `throughput=${result.stepsPerSec.toFixed(2)} steps/s at N=${result.N} ` +
        `verdict=${result.settled && inArea && inThickness ? 'passed' : 'FAILED'}`,
    )

    expect(result.meanTail).toBe(2)
    // Recognised-lipid count falls slowly over a run this long (372-386 of 400 placed, against 396
    // at 20000 steps in tests/water-bilayer-broth.test.ts). This is NOT an area-move artifact and
    // NOT a recognition artifact: co_break is a live reaction on the head-carbon bond every
    // hand-built lipid carries, so every lost lipid is one fired event. Asserted as that identity
    // rather than as a run-length-dependent floor -- measured exactly on the nose (372 recognised +
    // 28 co_break = 400 placed), which is also why no lipid can go missing unexplained.
    expect(result.amphiphileCount + result.events.co_break).toBeGreaterThanOrEqual(395)
    expect(result.amphiphileCount).toBeGreaterThan(0.85 * 400)
    expect(result.clusterFraction).toBeGreaterThan(0.9)
    // The machinery: the box actually MOVED, the acceptance is in a usable Metropolis band, and the
    // chain SETTLED by the block-drift criterion (not merely stopped at the cap). Without this last
    // one the numbers below would be a snapshot of a wandering coordinate -- exactly the trap
    // water-calibration-report.md §3.1 documented.
    expect(Math.abs(result.box[0] - result.startBox[0])).toBeGreaterThan(1e-3)
    expect(result.acceptedFraction).toBeGreaterThan(0.05)
    expect(result.acceptedFraction).toBeLessThan(0.95)
    expect(result.settled).toBe(true)

    // THICKNESS GATE: PASSES. Measured 4.76 sigma inside 4-6 at the model's OWN zero-tension area.
    expect(result.thickness).not.toBeNull()
    expect(result.thickness!).toBeGreaterThanOrEqual(THICKNESS_MIN)
    expect(result.thickness!).toBeLessThanOrEqual(THICKNESS_MAX)
    // The core stays dry -- the failure below is not water penetration.
    expect(result.waterInCore / result.totalWater).toBeLessThan(0.02)

    // AREA GATE: PASSES, and it is now a MEASUREMENT. At 10 trials per 500 MD steps the chain does
    // not outrun the sheet: the area falls from the assumed 1.3, transiently overshoots into a
    // collapsed state around chunk 120-180 (area ~1.128 with head burial ~0.39-0.42), then RECOVERS
    // -- burial back to 0.06-0.09 while the area settles at 1.13. The window asserted below is the
    // settled one, and `settled` above is what guarantees it is a plateau rather than a crossing.
    // The corridor is NOT widened and data/params.json is NOT re-fitted.
    expect(result.tailMin).toBeGreaterThanOrEqual(AREA_MIN)
    expect(result.tailMax).toBeLessThanOrEqual(AREA_MAX)
    // Head burial -- the defect water-calibration-report.md identified (0.40-0.49 there) -- must be
    // small at the MEASURED area too, not only at the assumed one. This is the assertion that fails
    // if the sheet settles into one of the collapsed states the faster protocol produced.
    expect(result.headBuriedFraction).not.toBeNull()
    expect(result.headBuriedFraction!).toBeLessThan(0.2)
  },
  580_000,
)
