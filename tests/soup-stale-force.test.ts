import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'decisive-run' (2026-08-20): THE STALE F(x) DEFECT, PINNED.
//
// What the defect was. soup/src/soup-box-scale.ts's applyBoxScaleOnce applies a box change by
// rescaling every molecule's centre of mass, then rebuilds the neighbour grid and the Verlet list --
// and, until this task, STOPPED there. soup/src/soup-integrate.ts's step() opens with
// kick_drift_wrap, which consumes forceBuf as F(x_n); so the first half-kick after every box change
// applied a force computed for a geometry that no longer existed. The evaporation task diagnosed it
// (evaporation-report.md §5.2), left it in the shared path, and worked around it in the evaporating
// transition only (a per-increment forces() readback). That made a headline result rest on a
// workaround rather than on a correct integrator, which is why this task fixes the shared path.
//
// Why this test needs a DEBUG readback to exist at all: SoupSystem.forces() REBUILDS grid, list and
// force before reading back, so through it a stale buffer and a fresh one are indistinguishable.
// SoupSystem.forcesNoRebuildDEBUG() reads forceBuf exactly as it stands -- the very F(x_n) the next
// kick will consume -- and encodes no dispatch of its own.
//
// The pinning pair, both directions, so the test cannot pass vacuously:
//  (a) AFTER a box change the resident buffer must MATCH a freshly computed force at the new
//      geometry (this is what the fix establishes);
//  (b) it must NOT match the force of the geometry that existed BEFORE the box change (this is what
//      makes (a) a real claim rather than an artefact of the two numbers being similar anyway --
//      without the fix the buffer is exactly that pre-change force, bit for bit).
test('a box change must leave F(x_n) for the new geometry, not for the previous one', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    // Small, cheap, and NOT a cycling system: scaleBoxTo drives the very same applyBoxScaleOnce the
    // dry-wet ramp, growBoxTo and soup/src/soup-area-move.ts all drive, so pinning it here pins it
    // for all of them. clay: false -- a box change is refused on a system with an immobile phase.
    const start = { C: 300, O: 1200, H: 300, M: 20, W: 3400 }
    const sys = await api.createSoup({ box: [16, 16, 16], seed: 7, kT: 1.1, start, clay: false })
    await sys.relaxColdStart()
    await sys.step(2000)

    const mx = (f: Float32Array) => {
      let m = 0
      for (let i = 0; i < f.length; i++) m = Math.max(m, Math.abs(f[i]))
      return m
    }
    const maxDiff = (a: Float32Array, b: Float32Array) => {
      let m = 0
      for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]))
      return m
    }

    // The force of the PRE-change geometry, and the buffer state that goes with it.
    const preFresh = await sys.forces()
    const preResident = await sys.forcesNoRebuildDEBUG()

    // One ramped box change (3 increments), exactly as a caller outside the cycle schedule makes it.
    // rampRelaxSteps = 0 on purpose: any real dynamics after the last increment would recompute the
    // force as a side effect and hide precisely what is being measured.
    await sys.scaleBoxTo([15, 15, 15], 3, 0)

    // What the next kick would actually consume, and what it OUGHT to be.
    const postResident = await sys.forcesNoRebuildDEBUG()
    const postFresh = await sys.forces()

    return {
      n: preFresh.length,
      box: sys.box,
      maxPreFresh: mx(preFresh),
      maxPostFresh: mx(postFresh),
      // sanity: before the box change the resident buffer already matches a fresh compute (it does,
      // because step() ends with a force pass) -- so the comparison below is about the box change.
      preResidentVsFresh: maxDiff(preResident, preFresh),
      // (a) the claim the fix establishes
      postResidentVsFresh: maxDiff(postResident, postFresh),
      // (b) the claim that makes (a) non-vacuous
      postResidentVsPreFresh: maxDiff(postResident, preFresh),
    }
  })
  console.log('STALE-FORCE ' + JSON.stringify(r))

  expect(r.box).toEqual([15, 15, 15])
  expect(r.maxPostFresh).toBeGreaterThan(0)
  // (a) the resident F(x_n) after the box change is the force of the CURRENT geometry. Tolerance is
  // relative to the force scale itself and generous by design: the grid fill uses atomics, so the
  // per-cell particle order (and therefore the float summation order) varies between two computes of
  // the SAME configuration -- this engine is not bit-reproducible (evaporation-report.md §6.5). What
  // the fix has to establish is agreement to summation-order noise, not bit equality.
  expect(r.postResidentVsFresh).toBeLessThan(0.01 * r.maxPostFresh)
  // (b) and it is emphatically NOT the pre-change force -- which is exactly what it WAS before the
  // fix, where this number would have been 0 and the one above the full force scale.
  expect(r.postResidentVsPreFresh).toBeGreaterThan(0.1 * r.maxPreFresh)
}, 600_000)

// The mid-run minimiser the minimisation-only control arm uses (SoupSystem.minimiseNowDEBUG): the
// control's whole validity rests on it being the SAME minimiser the cycled arm's ramp guard and
// solvent insertion use (soup/src/soup-relax.ts's relaxIterations), applied at the current box with
// no box change and no solvent movement. Pinned here so a future edit cannot quietly make the
// control arm receive a different perturbation from the arm it controls for.
test('minimisation mid-run: the same minimiser, on the current box, composition and bond graph intact', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const start = { C: 300, O: 1200, H: 300, M: 20, W: 3400 }
    const sys = await api.createSoup({ box: [16, 16, 16], seed: 7, kT: 1.1, start, clay: false })
    await sys.relaxColdStart()
    await sys.step(2000)
    const invBefore = await sys.invariants()
    const bondsBefore = Array.from(await sys.bonds()).join(',')
    const evBefore = await sys.events()
    const stepsBefore = sys.steps
    const boxBefore = sys.box

    const m = await sys.minimiseNowDEBUG(29)

    const invAfter = await sys.invariants()
    const bondsAfter = Array.from(await sys.bonds()).join(',')
    const evAfter = await sys.events()
    const nf = await sys.nonFiniteCount()
    return {
      m,
      stepsBefore,
      stepsAfter: sys.steps,
      boxBefore,
      boxAfter: sys.box,
      censusSame: JSON.stringify(invBefore.monomers) === JSON.stringify(invAfter.monomers),
      chargeSame: invBefore.charge === invAfter.charge,
      bondSetSame: bondsBefore === bondsAfter,
      eventsSame: JSON.stringify(evBefore) === JSON.stringify(evAfter),
      nonFinite: nf,
    }
  })
  console.log('MIDRUN-MINIMISE ' + JSON.stringify(r))

  // It advances no step counter and changes no box: it is a perturbation of positions only.
  expect(r.stepsAfter).toBe(r.stepsBefore)
  expect(r.boxAfter).toEqual(r.boxBefore)
  // It creates/destroys nothing and attempts no chemistry -- so the control arm differs from a plain
  // wet arm in exactly one respect, the minimisation itself.
  expect(r.censusSame).toBe(true)
  expect(r.chargeSame).toBe(true)
  expect(r.bondSetSame).toBe(true)
  expect(r.eventsSame).toBe(true)
  expect(r.nonFinite).toEqual({ pos: 0, vel: 0 })
  // It does what a minimiser exists to do, and reports the bound it could not have exceeded.
  expect(r.m.iterations).toBe(29)
  expect(r.m.maxForceAfter).toBeLessThan(r.m.maxForceBefore)
  expect(r.m.displacementBound).toBeGreaterThan(0)
}, 600_000)
