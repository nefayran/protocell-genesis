// Checkpoint/resume (task 'checkpoint-resume'). Three independent claims, one shared browser
// (gpuPage() memoizes it, so three test() cases here cost ONE Chrome launch, not three):
//
//  1. encode -> base64 -> decode -> createSoup(opts.resume) reconstructs EVERY field a checkpoint
//     carries exactly, with zero steps taken on either side of the copy (no GPU nondeterminism can
//     have had a chance to act yet, so this isolates round-trip fidelity from the engine's own
//     run-to-run scatter).
//  2. taking a checkpoint (pure GPU readback, see soup/src/checkpoint.ts's own header) does not
//     change what a run computes next -- checked against the engine's OWN natural run-to-run
//     scatter (two independent, checkpoint-free "control" systems with identical config), not
//     against an assumption of bit-reproducibility this engine does not have.
//  3. the "same-process continue" half of the required same-process-vs-fresh-process comparison --
//     writes real checkpoint files to data/checkpoints (gitignored) that a SEPARATE, later Node
//     process (soup/cli/campaign.ts, then a small pure-CPU comparison script -- no GPU needed for
//     either) resumes and compares against.
import { afterAll, expect, test } from 'vitest'
import { mkdirSync, writeFileSync } from 'node:fs'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

const CHECKPOINT_DIR = 'data/checkpoints'

interface Invariants {
  monomers: Record<string, number>
  bonds: number
  charge: number
  amphiphileCount: number
  meanTailLength: number
}

// Runs entirely inside the page: builds the SAME invariants/aggregate observables the task's own
// comparison asks for (particle counts per species, total charge, bond count, amphiphile count,
// mean tail length) from a live SoupSystem, using the engine's own recognisers (never a duplicate
// implementation here) so this test cannot silently drift from what soup/src/amphiphile.ts actually
// considers an amphiphile.
async function readInvariants(page: Awaited<ReturnType<typeof gpuPage>>, sysHandle: string): Promise<Invariants> {
  return page.evaluate(async (handle: string) => {
    const api = (window as any).api
    const sys = (window as any)[handle]
    const inv = await sys.invariants()
    const particles = await sys.particles()
    const bonds = await sys.bonds()
    const soup = api.loadSoup()
    const amph = api.findAmphiphiles(particles, bonds, soup.monomers)
    const tails = amph.flatMap((a: any) => a.tailLengths)
    return {
      monomers: inv.monomers,
      bonds: inv.bonds,
      charge: inv.charge,
      amphiphileCount: amph.length,
      meanTailLength: tails.length ? tails.reduce((a: number, b: number) => a + b, 0) / tails.length : 0,
    }
  }, sysHandle)
}

test('checkpoint round-trip reconstructs every field exactly at zero elapsed steps', async () => {
  const page = await gpuPage()
  const config = { box: [18, 18, 18], seed: 7, kT: 1.1, start: { C: 120, O: 40, H: 120, M: 10, W: 0 } }

  const result = await page.evaluate(async (cfg: any) => {
    const api = (window as any).api
    const sysA = await api.createSoup(cfg)
    await sysA.step(1500) // real bonds/rng-advancement to check, not the all-NONE initial state
    const checkpoint = await api.encodeCheckpoint(sysA, cfg)

    const resume = api.decodeCheckpointResume(checkpoint)
    const sysB = await api.createSoup({ ...cfg, resume })

    const [pA, pB] = await Promise.all([sysA.particles(), sysB.particles()])
    const [vA, vB] = await Promise.all([sysA.velocities(), sysB.velocities()])
    const [bsA, bsB] = await Promise.all([sysA.bondSlots(), sysB.bondSlots()])
    const [clA, clB] = await Promise.all([sysA.centerLinks(), sysB.centerLinks()])
    const [chA, chB] = await Promise.all([sysA.centerHeldSteps(), sysB.centerHeldSteps()])
    const [rngA, rngB] = await Promise.all([sysA.rngState(), sysB.rngState()])
    const [deA, deB] = await Promise.all([sysA.desorbEvents(), sysB.desorbEvents()])
    const [evA, evB] = await Promise.all([sysA.events(), sysB.events()])

    const arraysEqual = (a: ArrayLike<number>, b: ArrayLike<number>) => {
      if (a.length !== b.length) return false
      for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false
      return true
    }

    sysA.dispose()
    sysB.dispose()

    return {
      stepsA: checkpoint.globalStep,
      positionsEqual: arraysEqual(pA, pB),
      velocitiesEqual: arraysEqual(vA, vB),
      bondSlotsEqual: arraysEqual(bsA, bsB),
      centerLinksEqual: arraysEqual(clA, clB),
      centerHeldStepsEqual: arraysEqual(chA, chB),
      bondRngEqual: arraysEqual(rngA.bond, rngB.bond),
      thermoRngEqual: arraysEqual(rngA.thermo, rngB.thermo),
      desorbEqual: deA.stretch === deB.stretch && deA.timeout === deB.timeout,
      eventsEqual: JSON.stringify(evA) === JSON.stringify(evB),
      boxEqual: JSON.stringify(sysA.box) === JSON.stringify(sysB.box),
      cyclePhaseEqual: sysA.cyclePhase === sysB.cyclePhase,
      cycleIndexEqual: sysA.cycleIndex === sysB.cycleIndex,
      stepsEqual: sysA.steps === sysB.steps,
      transferBytes: checkpoint.positionsB64.length + checkpoint.velocitiesB64.length + checkpoint.bondSlotsB64.length,
    }
  }, config)

  console.log('ROUNDTRIP', JSON.stringify(result))
  expect(result.stepsA).toBe(1500)
  expect(result.positionsEqual).toBe(true)
  expect(result.velocitiesEqual).toBe(true)
  expect(result.bondSlotsEqual).toBe(true)
  expect(result.centerLinksEqual).toBe(true)
  expect(result.centerHeldStepsEqual).toBe(true)
  expect(result.bondRngEqual).toBe(true)
  expect(result.thermoRngEqual).toBe(true)
  expect(result.desorbEqual).toBe(true)
  expect(result.eventsEqual).toBe(true)
  expect(result.boxEqual).toBe(true)
  expect(result.cyclePhaseEqual).toBe(true)
  expect(result.cycleIndexEqual).toBe(true)
  expect(result.stepsEqual).toBe(true)
}, 180_000)

test('taking a checkpoint does not change what a run computes next (checked against the engine\'s own run-to-run scatter)', async () => {
  const page = await gpuPage()
  const config = { box: [18, 18, 18], seed: 11, kT: 1.1, start: { C: 250, O: 80, H: 250, M: 16, W: 0 } }
  const K1 = 2000
  const K2 = 2000

  const setup = await page.evaluate(async (cfg: any, k1: number) => {
    const api = (window as any).api
    // X: will have a checkpoint taken mid-run (pure readback, see soup/src/checkpoint.ts's header
    // for why this cannot write anything). Y, Z: two independent, checkpoint-free controls with the
    // IDENTICAL config -- their own final disagreement (if any) is this engine's natural run-to-run
    // scatter, the honest baseline "checkpointing changed nothing" is measured against, not an
    // assumed bit-reproducibility this engine does not have (soup/src/sim.ts's own header: neighbour
    // grid slots are assigned by atomicAdd).
    const sysX = await api.createSoup(cfg)
    const sysY = await api.createSoup(cfg)
    const sysZ = await api.createSoup(cfg)
    await Promise.all([sysX.step(k1), sysY.step(k1), sysZ.step(k1)])
    ;(window as any).__sysX = sysX
    ;(window as any).__sysY = sysY
    ;(window as any).__sysZ = sysZ
    return true
  }, config, K1)
  expect(setup).toBe(true)

  // The checkpoint itself -- taken on X only, between the two step() halves, exactly like a real
  // periodic save would sit inside a longer run.
  const checkpoint = await page.evaluate(async (cfg: any) => {
    const api = (window as any).api
    return api.encodeCheckpoint((window as any).__sysX, cfg)
  }, config)
  expect(checkpoint.globalStep).toBe(K1)

  await page.evaluate(
    async (k2: number) => {
      await Promise.all([
        (window as any).__sysX.step(k2),
        (window as any).__sysY.step(k2),
        (window as any).__sysZ.step(k2),
      ])
    },
    K2,
  )

  const [invX, invY, invZ] = await Promise.all([
    readInvariants(page, '__sysX'),
    readInvariants(page, '__sysY'),
    readInvariants(page, '__sysZ'),
  ])
  await page.evaluate(() => {
    ;(window as any).__sysX.dispose()
    ;(window as any).__sysY.dispose()
    ;(window as any).__sysZ.dispose()
  })

  console.log('NEUTRALITY', JSON.stringify({ invX, invY, invZ }))

  // Particle counts per species and total charge: conserved EXACTLY by construction (nothing in
  // this engine ever creates or destroys a particle) -- not a tolerance, an invariant.
  expect(invX.monomers).toEqual(invY.monomers)
  expect(invY.monomers).toEqual(invZ.monomers)
  expect(invX.charge).toBe(invY.charge)
  expect(invY.charge).toBe(invZ.charge)

  // Bond count / amphiphile count / mean tail length: the natural Y-vs-Z scatter (both
  // checkpoint-free) is the honest tolerance -- X-vs-Y must not disagree by MORE than Y-vs-Z
  // already does, which is what "checkpointing changed nothing beyond the engine's own inherent
  // non-reproducibility" actually means for a non-bit-reproducible engine.
  const yzBonds = Math.abs(invY.bonds - invZ.bonds)
  const xyBonds = Math.abs(invX.bonds - invY.bonds)
  const yzAmph = Math.abs(invY.amphiphileCount - invZ.amphiphileCount)
  const xyAmph = Math.abs(invX.amphiphileCount - invY.amphiphileCount)
  console.log('NEUTRALITY-DELTAS', JSON.stringify({ yzBonds, xyBonds, yzAmph, xyAmph }))
  // A generous margin (the natural scatter's own worst case plus a fixed pad, since the natural
  // scatter is itself a single noisy sample, not a converged bound) -- not "must be zero", which
  // this engine's own documented non-reproducibility makes an unfair, false-failing bar.
  expect(xyBonds).toBeLessThanOrEqual(yzBonds + 5)
  expect(xyAmph).toBeLessThanOrEqual(yzAmph + 3)
}, 180_000)

test('same-process continue: writes real checkpoint files for a fresh-process resume to be compared against', async () => {
  mkdirSync(CHECKPOINT_DIR, { recursive: true })
  const page = await gpuPage()
  const label = 'resumetest'
  const config = { box: [20, 20, 20], seed: 23, kT: 1.1, start: { C: 450, O: 150, H: 450, M: 30, W: 0 } }
  const K1 = 6000
  const K2 = 6000

  const crashPoint = await page.evaluate(async (cfg: any, k1: number) => {
    const api = (window as any).api
    const sys = await api.createSoup(cfg)
    await sys.step(k1)
    const checkpoint = await api.encodeCheckpoint(sys, cfg)
    ;(window as any).__sysResume = sys
    return checkpoint
  }, config, K1)
  expect(crashPoint.globalStep).toBe(K1)
  writeFileSync(`${CHECKPOINT_DIR}/${label}-step${crashPoint.globalStep}.json`, JSON.stringify(crashPoint))
  console.log(`WROTE ${CHECKPOINT_DIR}/${label}-step${crashPoint.globalStep}.json`)

  const finalSameProcess = await page.evaluate(async (cfg: any, k2: number) => {
    const api = (window as any).api
    const sys = (window as any).__sysResume
    await sys.step(k2)
    const checkpoint = await api.encodeCheckpoint(sys, cfg)
    sys.dispose()
    return checkpoint
  }, config, K2)
  expect(finalSameProcess.globalStep).toBe(K1 + K2)
  const sameProcessPath = `${CHECKPOINT_DIR}/${label}-same-process-final-step${finalSameProcess.globalStep}.json`
  writeFileSync(sameProcessPath, JSON.stringify(finalSameProcess))
  console.log(`WROTE ${sameProcessPath}`)
  console.log(
    `RESUME-SETUP ${JSON.stringify({ label, config, K1, K2, crashCheckpoint: `${label}-step${K1}.json`, sameProcessFinal: sameProcessPath })}`,
  )
}, 180_000)

// Task 'confined-parcel' (2026-08-21): THE CONFINEMENT STATE MUST ROUND-TRIP. A confined run's
// identity is its parcel: resume it as an unconfined system and the wall silently disappears
// mid-trajectory, which is the same class of silent failure as resuming a charged run without its
// charges (defect §7.2 of the verdict document, which left 72 checkpoints on disk carrying zero RNG
// seeds). So this checks all three halves of it:
//  1. encodeCheckpoint's config carries `confine` verbatim;
//  2. a resume rebuilds the SAME parcel -- radius, stiffness, and the derived live radius -- and the
//     positions/velocities/bond graph come back byte-identical;
//  3. the resumed system's wall is REALLY there: the force field it reports for the resumed
//     configuration is identical to the pre-checkpoint one, which it could not be if the wall term
//     had been dropped (the outermost beads carry a nonzero wall force, asserted separately so a
//     configuration that happened to have none could not pass this vacuously).
test('confined-parcel: confinement survives checkpoint -> resume, and the wall is really in place after the resume', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const BOX = 80
    const R = 16
    const config = {
      box: [BOX, BOX, BOX],
      seed: 13,
      kT: 1.1,
      start: { C: 2188, O: 729, H: 2188, M: 57, W: 13726 },
      confine: { radiusSigma: R, stiffness: 100 },
    }
    const a = await api.createSoup({ ...config, clay: false, confine: config.confine })
    await a.relaxColdStart()
    await a.step(300)
    const confA = a.confinement()
    const posA = await a.particles()
    const velA = await a.velocities()
    const bondsA = await a.bondSlots()
    const forceA = await a.forces()
    // How much of the force field is the wall's, at this configuration -- so "identical after resume"
    // is not a claim about a term that was zero anyway.
    let wallShare = 0
    for (let i = 0; i < posA.length / 4; i++) {
      const w = api.wallForceAt([posA[i * 4], posA[i * 4 + 1], posA[i * 4 + 2]], a.box, confA.radiusLive, confA.stiffness)
      wallShare = Math.max(wallShare, Math.abs(w[0]), Math.abs(w[1]), Math.abs(w[2]))
    }
    const file = await api.encodeCheckpoint(a, config)
    a.dispose()

    const b = await api.createSoup({ ...config, clay: false, resume: api.decodeCheckpointResume(file) })
    const confB = b.confinement()
    const posB = await b.particles()
    const velB = await b.velocities()
    const bondsB = await b.bondSlots()
    const forceB = await b.forces()
    let maxPos = 0
    let maxVel = 0
    let bondDiff = 0
    let maxForce = 0
    for (let i = 0; i < posA.length; i++) {
      maxPos = Math.max(maxPos, Math.abs(posA[i] - posB[i]))
      maxVel = Math.max(maxVel, Math.abs(velA[i] - velB[i]))
      maxForce = Math.max(maxForce, Math.abs(forceA[i] - forceB[i]))
    }
    for (let i = 0; i < bondsA.length; i++) if (bondsA[i] !== bondsB[i]) bondDiff++
    b.dispose()
    return { confA, confB, confInFile: file.config.confine, maxPos, maxVel, bondDiff, maxForce, wallShare, steps: file.globalStep }
  })
  console.log(
    `CONFINE-ROUNDTRIP step=${r.steps} in_file=${JSON.stringify(r.confInFile)} ` +
      `R_live before=${Number((r.confA as any).radiusLive).toFixed(6)} after=${Number((r.confB as any).radiusLive).toFixed(6)} ` +
      `max|dPos|=${r.maxPos} max|dVel|=${r.maxVel} bond_differences=${r.bondDiff} max|dF|=${r.maxForce} ` +
      `max|F_wall| in this configuration=${r.wallShare.toFixed(4)}`,
  )
  expect(r.confInFile).toEqual({ radiusSigma: 16, stiffness: 100 })
  expect((r.confB as any).radiusLive).toBe((r.confA as any).radiusLive)
  expect((r.confB as any).stiffness).toBe((r.confA as any).stiffness)
  expect(r.maxPos).toBe(0)
  expect(r.maxVel).toBe(0)
  expect(r.bondDiff).toBe(0)
  // The wall is a real, nonzero part of this configuration's force field...
  expect(r.wallShare).toBeGreaterThan(0)
  // ...and the resumed system reproduces the whole field to float32 agreement, so it did not lose it.
  // NOT bit-identical, and deliberately not asserted as such: positions/velocities/bonds ARE bit
  // identical above, but the force is a sum over a neighbour list whose per-cell append order is set
  // by atomics, so a rebuild reorders the summation. Measured 4.58e-5 against a wall force of 27.3 in
  // the same configuration, i.e. ~1e-6 relative -- five orders of magnitude below the size of the term
  // being checked for, so a DROPPED wall would be unmissable here.
  expect(r.maxForce).toBeLessThan(1e-3)
  expect(r.maxForce).toBeLessThan(r.wallShare / 1000)
})
