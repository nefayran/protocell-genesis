import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'loud-failure-and-liquid-water' (2026-08-20). THE DEFECT THIS PINS: a run whose state has gone
// non-finite used to keep stepping and keep printing plausible progress lines. Three measured
// instances, all silent -- rho_tot = 0.8444 (68 049 non-finite components), 0.75 (60 495) and 0.80
// (366 282) -- each found only afterwards by an offline scan of a checkpoint file, each with
// `stage=monomers, 0 amphiphiles` on screen, which is exactly what a clean negative looks like.
//
// The mechanism was not "nobody checked". assertVerletSafety (soup/src/soup-grid-verlet.ts) DID run
// at every chunk boundary and IS blind to it by construction: it tests `sqrt(maxDriftSq) > skin/2`,
// and every comparison against NaN is false, so once positions are NaN the drift reads NaN and the
// guard passes forever. soup/src/soup-health.ts now runs an O(N) exponent scan at the SAME cadence
// and the SAME sync point, BEFORE that check, and throws.
//
// The diverging fixture is a real WCA blow-up, not an injected NaN: twenty pairs of water beads
// placed 1e-4 sigma apart. wca_dv (engine/wgsl/forces.wgsl) returns 0 at exactly r = 0 and is
// finite-but-astronomical just off it -- (b/r)^12 at r = 1e-4 is 1e48, which overflows f32 to Inf --
// so this is the same physics as the cold-start overlap that produced all three historical
// divergences, just concentrated into one step instead of thousands.

const WATER_LATTICE = 12 // 12^3 = 1728 lattice sites, box 20 -> spacing 1.667 sigma, no overlaps
const OVERLAP_PAIRS = 20
const OVERLAP_SEPARATION = 1e-4

test(
  'a non-finite state stops the run with a message that names the step and the count',
  async () => {
    const page = await gpuPage()
    const consoleWarnings: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
    })

    const result = await page.evaluate(
      async (nSide: number, overlapPairs: number, sep: number) => {
        const api = (window as any).api
        const soup = api.loadSoup()
        const p = api.loadParams()
        const waterKind = soup.monomers.findIndex((m: any) => m.id === soup.solvent.waterId)
        const box: [number, number, number] = [20, 20, 20]

        /** A clean water lattice at a spacing far wider than any contact distance, then `bad` pairs
         * of it pulled to `sep` apart. `bad = 0` is the HEALTHY control and `bad > 0` the diverging
         * arm; everything else about the two systems is identical, which is what makes the pair a
         * control rather than two unrelated runs. */
        function build(bad: number) {
          const n = nSide * nSide * nSide
          const positions = new Float32Array(n * 4)
          const s = box[0] / nSide
          let k = 0
          for (let iz = 0; iz < nSide; iz++)
            for (let iy = 0; iy < nSide; iy++)
              for (let ix = 0; ix < nSide; ix++) {
                positions.set([(ix + 0.5) * s, (iy + 0.5) * s, (iz + 0.5) * s, waterKind], k * 4)
                k++
              }
          for (let q = 0; q < bad; q++) {
            const a = q * 2
            const b = q * 2 + 1
            positions[b * 4 + 0] = positions[a * 4 + 0] + sep
            positions[b * 4 + 1] = positions[a * 4 + 1]
            positions[b * 4 + 2] = positions[a * 4 + 2]
          }
          const startCounts: Record<string, number> = {}
          for (const mo of soup.monomers) startCounts[mo.id] = 0
          startCounts[soup.monomers[waterKind].id] = n
          return {
            n,
            opts: {
              box,
              seed: 7,
              kT: p.thermostat.kT,
              clay: false,
              start: startCounts,
              resume: {
                globalStep: 0,
                liveBox: box,
                positions,
                velocities: new Float32Array(n * 4),
                bondSlots: new Uint32Array(n * 3).fill(0xffffffff),
                centerLink: new Uint32Array(n).fill(0xffffffff),
                centerHeldSteps: new Uint32Array(n),
                desorbEvents: new Uint32Array(2),
                bondRng: Uint32Array.from({ length: n }, (_, i) => i + 1),
                thermoRng: Uint32Array.from({ length: n }, (_, i) => i + 1000003),
                events: {},
              },
            },
          }
        }

        // --- arm 1: the HEALTHY control ----------------------------------------------------------
        const healthyBuild = build(0)
        const healthy = await api.createSoup(healthyBuild.opts)
        const healthyAtStart = await healthy.nonFiniteCount()
        await healthy.step(2000)
        const healthyAfter = await healthy.nonFiniteCount()

        // --- cost of the check, measured on the healthy system, not guessed ----------------------
        // t_scan: the whole guard as step() pays for it -- one O(N) dispatch, one submit, one 8-byte
        // readback. t_chunk: one full 1000-step chunk, i.e. what the guard is amortised over.
        const SCAN_REPS = 50
        const tScan0 = performance.now()
        for (let i = 0; i < SCAN_REPS; i++) await healthy.nonFiniteCount()
        const scanMs = (performance.now() - tScan0) / SCAN_REPS
        const CHUNK_REPS = 5
        const tChunk0 = performance.now()
        for (let i = 0; i < CHUNK_REPS; i++) await healthy.step(1000)
        const chunkMs = (performance.now() - tChunk0) / CHUNK_REPS
        const healthyAfterTiming = await healthy.nonFiniteCount()
        healthy.dispose()

        // --- arm 2: the DELIBERATELY DIVERGING system -------------------------------------------
        const badBuild = build(overlapPairs)
        const bad = await api.createSoup(badBuild.opts)
        const badAtStart = await bad.nonFiniteCount()
        // ONE step, not a thousand: step(n) checks once per chunk and a chunk is min(1000, n), so
        // step(1) puts the guard exactly one step in -- this is the "early" half of "loud and early".
        let threw: string | null = null
        try {
          await bad.step(1)
        } catch (e: any) {
          threw = String(e?.message ?? e)
        }
        const badAfter = await bad.nonFiniteCount()
        const badSteps = bad.steps
        bad.dispose()

        return {
          n: healthyBuild.n,
          healthyAtStart,
          healthyAfter,
          healthyAfterTiming,
          scanMs,
          chunkMs,
          scanReps: SCAN_REPS,
          chunkReps: CHUNK_REPS,
          badAtStart,
          badAfter,
          badSteps,
          threw,
        }
      },
      WATER_LATTICE,
      OVERLAP_PAIRS,
      OVERLAP_SEPARATION,
    )

    console.log(
      `NONFINITE-GUARD N=${result.n} healthy: at_start=${JSON.stringify(result.healthyAtStart)} ` +
        `after_2000=${JSON.stringify(result.healthyAfter)} after_timing=${JSON.stringify(result.healthyAfterTiming)}`,
    )
    console.log(
      `NONFINITE-GUARD-COST scan=${result.scanMs.toFixed(4)}ms (n=${result.scanReps}) ` +
        `chunk1000=${result.chunkMs.toFixed(2)}ms (n=${result.chunkReps}) ` +
        `overhead=${((100 * result.scanMs) / result.chunkMs).toFixed(3)}% of a 1000-step chunk ` +
        `=> throughput cost factor ${(1 + result.scanMs / result.chunkMs).toFixed(5)}x`,
    )
    console.log(
      `NONFINITE-GUARD-FIRES steps_reached=${result.badSteps} at_start=${JSON.stringify(result.badAtStart)} ` +
        `after=${JSON.stringify(result.badAfter)}\n  message: ${result.threw}`,
    )

    // The healthy control never trips it -- a guard that fires on a good run is worse than none.
    expect(result.healthyAtStart).toEqual({ pos: 0, vel: 0 })
    expect(result.healthyAfter).toEqual({ pos: 0, vel: 0 })
    expect(result.healthyAfterTiming).toEqual({ pos: 0, vel: 0 })

    // The diverging arm starts FINITE (the overlap is a legal coordinate, not an injected NaN) and
    // is caught after exactly one step.
    expect(result.badAtStart).toEqual({ pos: 0, vel: 0 })
    expect(result.threw).not.toBeNull()
    // It must be the FINITENESS guard that spoke, not the drift guard: that distinction is the whole
    // point -- the drift guard was running, and silent, in all three historical divergences.
    expect(result.threw).toMatch(/non-finite state/)
    expect(result.threw).not.toMatch(/Verlet list/)
    // Names the step...
    expect(result.threw).toMatch(/at step=1\b/)
    expect(result.threw).toMatch(/step interval 0\.\.1/)
    // ...and names a nonzero count for BOTH arrays, with the totals it is out of.
    const posCount = Number(/positions=(\d+)/.exec(result.threw!)![1])
    const velCount = Number(/velocities=(\d+)/.exec(result.threw!)![1])
    expect(posCount).toBeGreaterThan(0)
    expect(velCount).toBeGreaterThan(0)
    expect(result.threw).toMatch(new RegExp(`of ${result.n * 3}\\b`))
    // The state really is non-finite (the throw is not a false alarm) and the run really did stop.
    expect(result.badAfter.pos).toBeGreaterThan(0)
    expect(result.badSteps).toBe(1)

    expect(consoleWarnings, `the browser reported a GPU error/warning during the test:\n${consoleWarnings.join('\n')}`).toEqual([])
  },
  300_000,
)
