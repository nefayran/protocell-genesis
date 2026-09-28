import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'loud-failure-and-liquid-water' (2026-08-20). Pins the INVARIANTS of the cold-start energy
// minimisation (soup/src/soup-relax.ts, soup/wgsl/relax.wgsl) -- i.e. everything it must NOT touch --
// and its one payoff: a composition this engine has measured blowing up from a cold lattice runs
// clean when the lattice is minimised first.
//
// Why the invariants are the test and not a comment: the whole defence of the stage is "it cannot be
// inside any measured trajectory". That is only worth anything if it is checked. So: the step counter
// must still read 0 afterwards, every velocity must be BITWISE unchanged (the Maxwell-Boltzmann draw
// is the initial temperature, and a minimiser that quietly rescaled it would be changing the
// thermodynamic state), both per-particle RNG streams must be untouched (a consumed stream would
// desynchronise every subsequent random number from what an unrelaxed run would have drawn), the bond
// graph and every event counter must be empty, the census and charge must be identical, and no
// particle may have moved further than the analytic bound the decaying displacement cap implies.
// Positions MUST have moved -- otherwise the stage is a no-op and the payoff below would be luck.

// box 20, water at the measured liquid threshold 0.8 sigma^-3 plus the shipped organic proportions
// (O:C = 4, H:C = 1, M:C = 0.0666) scaled to this box: rho_tot = 1.218 sigma^-3, i.e. above every
// density this engine had ever survived before this task.
const BOX = 20
const START = { C: 551, O: 2205, H: 551, M: 37, W: 6400 }
const PAYOFF_STEPS = 3000

test(
  'cold-start minimisation: touches nothing that is measured and makes liquid water runnable',
  async () => {
    const page = await gpuPage()
    const consoleWarnings: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
    })

    const r = await page.evaluate(
      async (box1: number, start: Record<string, number>, payoffSteps: number) => {
        const api = (window as any).api
        const p = api.loadParams()
        const soup = api.loadSoup()
        const box: [number, number, number] = [box1, box1, box1]
        const mk = () => api.createSoup({ box, seed: 19, kT: p.thermostat.kT, start, clay: false })

        function u32eq(a: Uint32Array, b: Uint32Array): boolean {
          if (a.length !== b.length) return false
          for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
          return true
        }
        // BITWISE, not toBeCloseTo: "the velocities were not touched" is an exact claim, and a
        // float comparison with a tolerance would pass a minimiser that nudged them.
        function f32bitsEq(a: Float32Array, b: Float32Array): boolean {
          if (a.length !== b.length) return false
          const ai = new Uint32Array(a.buffer, a.byteOffset, a.length)
          const bi = new Uint32Array(b.buffer, b.byteOffset, b.length)
          for (let i = 0; i < ai.length; i++) if (ai[i] !== bi[i]) return false
          return true
        }
        function maxDisplacement(a: Float32Array, b: Float32Array, bx: [number, number, number]): number {
          let mx = 0
          for (let i = 0; i < a.length / 4; i++) {
            let d2 = 0
            for (let c = 0; c < 3; c++) {
              let d = b[i * 4 + c] - a[i * 4 + c]
              d -= Math.round(d / bx[c]) * bx[c] // minimum image: the wrap is not a displacement
              d2 += d * d
            }
            if (d2 > mx) mx = d2
          }
          return Math.sqrt(mx)
        }

        // --- the invariant system -----------------------------------------------------------------
        const sys = await mk()
        const posBefore: Float32Array = await sys.particles()
        const velBefore: Float32Array = await sys.velocities()
        const bondsBefore: Uint32Array = await sys.bondSlots()
        const rngBefore = await sys.rngState()
        const invBefore = await sys.invariants()
        const eventsBefore = await sys.events()
        const stepsBefore = sys.steps

        const relax = await sys.relaxColdStart()

        const posAfter: Float32Array = await sys.particles()
        const velAfter: Float32Array = await sys.velocities()
        const bondsAfter: Uint32Array = await sys.bondSlots()
        const rngAfter = await sys.rngState()
        const invAfter = await sys.invariants()
        const eventsAfter = await sys.events()
        const stepsAfter = sys.steps
        const moved = maxDisplacement(posBefore, posAfter, box)

        // ...and the refusal: one real step, then ask again.
        await sys.step(1)
        let refusal: string | null = null
        try {
          await sys.relaxColdStart()
        } catch (e: any) {
          refusal = String(e?.message ?? e)
        }
        const nonFiniteAfterOneStep = await sys.nonFiniteCount()
        sys.dispose()

        // --- the payoff: same composition, WITHOUT and WITH the stage ----------------------------
        const bare = await mk()
        let bareError: string | null = null
        let bareStepsAtThrow = -1
        try {
          let done = 0
          while (done < payoffSteps) {
            const n = Math.min(1000, payoffSteps - done)
            await bare.step(n)
            done += n
          }
        } catch (e: any) {
          bareError = String(e?.message ?? e)
          bareStepsAtThrow = bare.steps
        }
        bare.dispose()

        const relaxed = await mk()
        const relaxedRelax = await relaxed.relaxColdStart()
        let relaxedError: string | null = null
        let done = 0
        try {
          while (done < payoffSteps) {
            const n = Math.min(1000, payoffSteps - done)
            await relaxed.step(n)
            done += n
          }
        } catch (e: any) {
          relaxedError = String(e?.message ?? e)
        }
        const relaxedNonFinite = await relaxed.nonFiniteCount()
        const relaxedSteps = relaxed.steps
        relaxed.dispose()

        return {
          N: posBefore.length / 4,
          relax,
          moved,
          stepsBefore,
          stepsAfter,
          velUnchanged: f32bitsEq(velBefore, velAfter),
          bondsUnchanged: u32eq(bondsBefore, bondsAfter),
          rngBondUnchanged: u32eq(rngBefore.bond, rngAfter.bond),
          rngThermoUnchanged: u32eq(rngBefore.thermo, rngAfter.thermo),
          invBefore,
          invAfter,
          eventsBefore,
          eventsAfter,
          refusal,
          nonFiniteAfterOneStep,
          bareError,
          bareStepsAtThrow,
          relaxedRelax,
          relaxedError,
          relaxedNonFinite,
          relaxedSteps,
        }
      },
      BOX,
      START,
      PAYOFF_STEPS,
    )

    console.log(
      `COLD-START-RELAX N=${r.N} relax=${JSON.stringify(r.relax)} maxDisplacementMeasured=${r.moved.toFixed(5)} ` +
        `steps ${r.stepsBefore}->${r.stepsAfter} velUnchanged=${r.velUnchanged} bondsUnchanged=${r.bondsUnchanged} ` +
        `rng(bond/thermo)Unchanged=${r.rngBondUnchanged}/${r.rngThermoUnchanged} ` +
        `events ${JSON.stringify(r.eventsBefore)} -> ${JSON.stringify(r.eventsAfter)}`,
    )
    console.log(`COLD-START-RELAX-REFUSAL ${r.refusal}`)
    console.log(
      `COLD-START-RELAX-PAYOFF without minimisation: ${r.bareError ? `error at step=${r.bareStepsAtThrow}: ${r.bareError}` : 'passed without error'}\n` +
        `                     with minimisation: steps=${r.relaxedSteps} nonFinite=${JSON.stringify(r.relaxedNonFinite)} ` +
        `err=${r.relaxedError ?? 'none'} maxF ${r.relaxedRelax.maxForceBefore.toExponential(4)} -> ${r.relaxedRelax.maxForceAfter.toExponential(4)}`,
    )

    // --- INVARIANTS: everything the stage must not touch --------------------------------------
    expect(r.stepsBefore).toBe(0)
    expect(r.stepsAfter).toBe(0)
    expect(r.velUnchanged).toBe(true)
    expect(r.bondsUnchanged).toBe(true)
    expect(r.rngBondUnchanged).toBe(true)
    expect(r.rngThermoUnchanged).toBe(true)
    expect(r.invAfter.monomers).toEqual(r.invBefore.monomers)
    expect(r.invAfter.charge).toBe(r.invBefore.charge)
    expect(r.invAfter.bonds).toBe(r.invBefore.bonds)
    expect(Object.values(r.eventsAfter).every((v) => v === 0)).toBe(true)
    expect(r.eventsAfter).toEqual(r.eventsBefore)
    // It is not a no-op...
    expect(r.moved).toBeGreaterThan(0)
    // ...and it moved nobody further than the decaying cap analytically allows (sum of the caps).
    expect(r.moved).toBeLessThanOrEqual(r.relax.displacementBound + 1e-4)
    // It did the job it exists to do: three orders of magnitude off the worst force.
    expect(r.relax.maxForceBefore / r.relax.maxForceAfter).toBeGreaterThan(100)
    expect(r.relax.nonFiniteAfter).toBe(0)

    // --- REFUSAL: it is impossible to put a minimisation inside a trajectory ------------------
    expect(r.refusal).not.toBeNull()
    expect(r.refusal).toMatch(/minimisation is allowed only before the first step/)
    expect(r.refusal).toMatch(/already at step 1\b/)
    expect(r.nonFiniteAfterOneStep).toEqual({ pos: 0, vel: 0 })

    // --- PAYOFF: liquid-density water becomes reachable ---------------------------------------
    // Without the stage this exact composition dies, LOUDLY (which is the other half of this task).
    expect(r.bareError).not.toBeNull()
    expect(r.bareError).toMatch(/non-finite state/)
    // With it, the same composition runs clean.
    expect(r.relaxedError).toBeNull()
    expect(r.relaxedSteps).toBe(PAYOFF_STEPS)
    expect(r.relaxedNonFinite).toEqual({ pos: 0, vel: 0 })

    expect(consoleWarnings, `the browser reported a GPU error/warning during the test:\n${consoleWarnings.join('\n')}`).toEqual([])
  },
  300_000,
)
