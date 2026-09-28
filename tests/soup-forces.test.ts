import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// perf2-report.md correctness gate, mirroring tests/sim.test.ts's "the neighbour grid gives the same
// forces as brute force" for the soup engine's own dynamic-topology force kernel (soup/wgsl/
// step.wgsl's soup_force_main/soup_force_list_main against soup_force_brute_main, an O(N^2)
// reference with no neighbour grid at all). COMMITTED per the task brief: a neighbour-grid bug in
// this repo once silently doubled every force and looked like success, and while writing perf2's
// candidate (c) a DIFFERENT bug class showed up -- a WebGPU bind-group mismatch ('layout: auto'
// gives every pipeline a DISTINCT layout object even for an identical uniform) that failed
// validation and made the WHOLE submitted command buffer a silent no-op: forceBuf simply kept
// its all-zero initial value, so forces() and forcesBruteForce() both read back zeros and
// "agreed" perfectly -- a false pass that would have shipped a system computing no forces at all.
// Caught only by listening on the page's console for the browser's own GPU validation warnings,
// which this test now does itself and fails on, specifically so that failure mode cannot recur
// silently: a correctness test that can itself be silently skipped is not a correctness test.
test('the neighbour grid (including the Verlet list) gives the same forces as brute force', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    // box=[16,16,16] with a smaller-than-standard composition: enough particles (~1220) and all 4 monomer
    // kinds (including the catalyst and the polar head) to exercise every branch of
    // nonbondedSoup/bondedForce, but not the full scale (13100/box 30) -- the O(N^2) brute force grows
    // quadratically, and this is a diagnostic, not a production run.
    const sys = await api.createSoup({
      box: [16, 16, 16],
      seed: 7,
      kT: 1.1,
      // W:0 (task 'broth-composition', 2026-08-18): CreateSoupOpts.start merges over
      // data/soup.json's own defaults, which now include a box-30-sized water count -- explicit
      // zero keeps this box=16 force-correctness diagnostic at its own originally-intended,
      // deliberately small scale.
      // Task 'clay-surface' (2026-08-19): the CLAY-FREE control. data/soup.json's shipped composition
      // now carries the mineral platelet (clay.enabled), and this fixture must not: it is a force-correctness diagnostic at a deliberately small hand-picked scale, not a broth.
      // A clay-free system is byte-identical to the pre-task engine, so every number in this file is
      // unchanged by that task -- which is exactly what makes it a usable reference.
      clay: false,
      start: { C: 600, O: 100, H: 500, M: 20, W: 0 },
    })
    // A few real steps before the comparison -- the forces are checked not only on the starting lattice
    // (where distances are almost ideal and many WCA/attr branches are not exercised) but on an
    // already dispersed configuration.
    await sys.step(200)
    const a = await sys.forces()
    const b = await sys.forcesBruteForce()
    let max = 0
    let sumAbs = 0
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]))
      sumAbs += Math.abs(b[i])
    }
    return { maxDiff: max, meanAbsRef: sumAbs / a.length, n: a.length }
  })
  expect(consoleWarnings, `the browser reported a GPU error/warning during the test:\n${consoleWarnings.join('\n')}`).toEqual([])
  // float32 tolerance: forces here are of the order of a few tens (WCA near contact can give
  // large repulsive spikes), so the absolute threshold is given with a margin above the typical
  // float32 accumulation error of a sum over ~30-60 neighbours per particle.
  expect(r.maxDiff).toBeLessThan(1e-2)
})

// Task 'electrostatics' (2026-08-20): THE SAME correctness gate WITH CHARGE ON. The screened-Coulomb
// term lives inside nonbondedSoup, which all three force paths reach -- the grid walk
// (soup_force_main), the Verlet-list walk (soup_force_list_main, in soup/wgsl/verlet.wgsl since this
// task's responsibility split) and the O(N^2) reference (soup_force_brute_main). So the brute-force
// comparison above is exactly the instrument that can prove the new term is applied IDENTICALLY on all
// three, and it is extended here rather than duplicated in a new file: a term added to the shared
// function but wired into only two of the three bind-group layouts would fail bind-group validation
// (caught by the same console listener), and one applied with the wrong sign or the wrong index
// convention on the sorted-gather path would show up as a maxDiff blow-up here.
//
// It also asserts the term is actually DOING something: with charge on, the force must DIFFER from the
// same configuration with charge off, by more than the float32 tolerance. Otherwise a test that
// silently ran with all-zero charges would pass and prove nothing -- which is the exact false-pass
// class the header above was written about.
test('electrostatics: the grid, the Verlet list and brute force give the same forces, and the charge changes them', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const start = { C: 600, O: 100, H: 500, M: 20, W: 0 }
    const mk = (charge: boolean) =>
      api.createSoup({
        box: [16, 16, 16],
        seed: 7,
        kT: 1.1,
        clay: false,
        start,
        // pH 7 with the file's own intrinsic pKa 4.9 puts alpha at 0.992 -- i.e. essentially every
        // head charged, which is the arm where the term is largest and therefore the arm a wiring bug
        // is most visible in.
        electrostatics: charge ? { enabled: true, pH: 7 } : undefined,
      })
    const charged = await mk(true)
    await charged.step(200)
    const a = await charged.forces()
    const b = await charged.forcesBruteForce()
    let max = 0
    let sumAbs = 0
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]))
      sumAbs += Math.abs(b[i])
    }
    const q = await charged.charges()
    let chargedBeads = 0
    for (let i = 0; i < q.length; i++) if (q[i] !== 0) chargedBeads++
    const esInfo = charged.electrostatics()

    // The SAME positions, the SAME step count, charge off: the difference between the two force
    // fields is the electrostatic term itself.
    const neutral = await mk(false)
    await neutral.step(200)
    const c = await neutral.forces()
    let maxVsNeutral = 0
    for (let i = 0; i < a.length; i++) maxVsNeutral = Math.max(maxVsNeutral, Math.abs(a[i] - c[i]))
    charged.dispose()
    neutral.dispose()
    return { maxDiff: max, meanAbsRef: sumAbs / a.length, chargedBeads, heads: 100, maxVsNeutral, esInfo }
  })
  expect(consoleWarnings, `the browser reported a GPU error/warning during the test:\n${consoleWarnings.join('\n')}`).toEqual([])
  console.log(
    `SOUP-FORCES-ES maxDiff(grid+Verlet vs brute force)=${r.maxDiff.toExponential(4)} meanAbsRef=${r.meanAbsRef.toFixed(4)} ` +
      `charged_heads=${r.chargedBeads}/${r.heads} maxDiff(charged vs neutral)=${r.maxVsNeutral.toFixed(4)} ` +
      `A=${(r.esInfo as any).coeffA.toFixed(6)} kappa=${(r.esInfo as any).kappa.toFixed(6)} rc=${(r.esInfo as any).cutoff.toFixed(7)}`,
  )
  expect(r.maxDiff).toBeLessThan(1e-2)
  // The term is real: charge changed the force field by far more than the float32 agreement tolerance.
  expect(r.maxVsNeutral).toBeGreaterThan(1e-2)
  // And it was actually switched on: essentially every head is charged at pH 7 / pKa 4.9.
  expect(r.chargedBeads).toBeGreaterThan(80)
})

// Task 'long-range-electrostatics' (2026-08-20): THE SAME GATE AT THE LONG RANGE, and it is the one
// that matters, because the range is what changed. The screened-Coulomb term is now cut at a multiple
// of the DEBYE LENGTH (4, data/soup.json's longRangeDebyeLengths) instead of at the Lennard-Jones
// nonbonded cutoff 2.7224620 sigma, and it is summed in two halves: nonbondedSoup keeps r <
// interactionRange (2.9469545) from whichever neighbour structure that path uses, and a DEDICATED
// head-only list adds interactionRange <= r < rc_es. Two things can go wrong and only this test sees
// them: (a) the head-only list can be INCOMPLETE (a missing neighbour is a silently truncated
// interaction -- exactly the defect this task exists to remove, reappearing one level down), and
// (b) the two halves can double-count or leave a gap at the split radius.
//
// So the comparison is grid+Verlet+head-list against a pure O(N^2) reference that walks EVERY pair
// with no list at all (soup_es_force_far_brute_main), at 10 mM -- the low-salt arm, where lambda_D is
// 3.80 sigma and the cutoff is the longest this engine ever runs (13.5 sigma here, the minimum-image
// ceiling 0.45*30 biting before 4*lambda_D = 15.2 does). At the short cutoff the same instrument read
// 2.2888e-5; a broken list would not read anything like it.
test('long range: the heads-only list is complete at 13.5 sigma -- brute force agrees', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSoup({
      box: [30, 30, 30],
      seed: 11,
      kT: 1.1,
      clay: false,
      start: { C: 1200, O: 400, H: 1000, M: 30, W: 3000 },
      // 10 mM: lambda_D = 3.80 sigma, so 4 lambda_D = 15.2 -- longer than 0.45*30 = 13.5, which is
      // therefore what the cutoff becomes. pH 7 puts alpha at 0.99, the arm where the term is biggest.
      electrostatics: { enabled: true, pH: 7, ionicStrengthMolar: 0.01 },
    })
    await sys.step(200)
    const a = await sys.forces()
    const b = await sys.forcesBruteForce()
    let max = 0
    let sumAbs = 0
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]))
      sumAbs += Math.abs(b[i])
    }
    const es = sys.electrostatics()
    sys.dispose()
    return { maxDiff: max, meanAbsRef: sumAbs / a.length, es }
  })
  expect(consoleWarnings, `the browser reported a GPU error/warning during the test:\n${consoleWarnings.join('\n')}`).toEqual([])
  const es = r.es as any
  console.log(
    `SOUP-FORCES-ES-LONG maxDiff(grid+Verlet+head_list vs full brute force)=${r.maxDiff.toExponential(4)} ` +
      `meanAbsRef=${r.meanAbsRef.toFixed(4)} rc_es=${es.cutoff.toFixed(4)} (=${es.debyeLengthsSpanned.toFixed(3)} lambdaD, ` +
      `target=${es.targetCutoff.toFixed(4)}, image_cap=${es.imageCap.toFixed(4)}) splitRadius=${es.splitRadius.toFixed(7)} ` +
      `nbCutoff_was=${es.nbCutoff.toFixed(7)} discarded_integrated=${es.discardedIntegratedFraction.toFixed(4)}`,
  )
  expect(r.maxDiff).toBeLessThan(1e-2)
  // The cutoff really is the long one, and really was bounded by the minimum image here.
  expect(es.cutoff).toBeCloseTo(13.5, 6)
  expect(es.cutoff).toBeGreaterThan(es.nbCutoff * 4)
})

// Task 'confined-parcel' (2026-08-21): THE SAME CORRECTNESS GATE IN A CONFINED PARCEL. This is where
// confinement can break the force paths, and it can do it in two ways this comparison sees and
// nothing else does:
//  (a) the wall is a ONE-BODY term added by its own dispatch (soup/wgsl/wall.wgsl) from inside the
//      three encodeSoupForce* functions. Wire it into the grid/Verlet path but not into the O(N^2)
//      reference -- three of four kernels, seven of eight call sites, the exact trap
//      soup/src/soup-integrate.ts's encodeEsFar comment was written about -- and this test is the
//      only thing that fails;
//  (b) a bind group that references the wall uniform on one pipeline and not another fails WebGPU
//      validation, which makes the WHOLE submitted command buffer a silent no-op and both readbacks
//      return the same zeros -- a false pass, which the console listener above exists to catch.
// Charge is ON at the campaign's own ionic strength, so the long-range head list is exercised in the
// confined geometry too: that list's capacity is derived from a head DENSITY, and in a confined box
// the box-average density is 5.8x below the real one -- an under-sized list silently drops
// interactions, and a dropped interaction shows up here as a grid-vs-brute disagreement.
test('confinement: the wall is added identically on the grid/Verlet path and on brute force (and the charge is in place)', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const BOX = 80
    const R = 16
    const sys = await api.createSoup({
      box: [BOX, BOX, BOX],
      seed: 9,
      kT: 1.1,
      clay: false,
      // The campaign's own composition (box 76's census) scaled to THIS parcel's volume, so the
      // regime is the real one: liquid water at exactly 0.8 sigma^-3 and total density 1.1009.
      start: { C: 2188, O: 729, H: 2188, M: 57, W: 13726 },
      confine: { radiusSigma: R, stiffness: 100 },
      electrostatics: { enabled: true, pH: 7, ionicStrengthMolar: 0.01 },
    })
    // The cold-start minimisation, as every real run at liquid density needs: the jittered lattice
    // puts unlike-radius pairs inside each other's WCA cores, and without this the trajectory blows up
    // in a few hundred steps (measured here: an assertVerletSafety drift of 576 sigma). Nothing about
    // confinement changes that -- the wall descends the SAME force the steps use, so a bead that
    // started outside the parcel would also be pulled in by it.
    await sys.relaxColdStart()
    await sys.step(300)
    const a = await sys.forces()
    const b = await sys.forcesBruteForce()
    let max = 0
    let sumAbs = 0
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]))
      sumAbs += Math.abs(b[i])
    }
    // The wall's own contribution to THIS configuration, computed on the CPU, so the comparison above
    // cannot pass vacuously on a configuration where no bead is outside the parcel at all.
    const pos = await sys.particles()
    const conf = sys.confinement()
    let maxWall = 0
    let outside = 0
    for (let i = 0; i < pos.length / 4; i++) {
      const w = api.wallForceAt([pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]], sys.box, conf.radiusLive, conf.stiffness)
      const m = Math.max(Math.abs(w[0]), Math.abs(w[1]), Math.abs(w[2]))
      if (m > 0) outside++
      maxWall = Math.max(maxWall, m)
    }
    const es = sys.electrostatics()
    const vc = sys.verletConfig()
    const occ = await sys.listOccupancyDEBUG()
    sys.dispose()
    return { maxDiff: max, meanAbsRef: sumAbs / a.length, maxWall, outside, n: pos.length / 4, es, vc, occ, conf }
  })
  expect(consoleWarnings, `the browser reported a GPU error/warning during the test:\n${consoleWarnings.join('\n')}`).toEqual([])
  const es = r.es as any
  const vc = r.vc as any
  const occ = r.occ as any
  console.log(
    `SOUP-FORCES-CONFINED maxDiff(grid+Verlet+head_list vs brute force)=${r.maxDiff.toExponential(4)} ` +
      `meanAbsRef=${r.meanAbsRef.toFixed(4)} N=${r.n} outside_parcel=${r.outside} max|F_wall|=${r.maxWall.toFixed(4)}\n` +
      `SOUP-FORCES-CONFINED R=${Number((r.conf as any).radiusLive).toFixed(4)} V_parcel=${Number((r.conf as any).parcelVolumeLive).toFixed(1)} ` +
      `V_box/V_parcel=${(Number((r.conf as any).boxVolumeLive) / Number((r.conf as any).parcelVolumeLive)).toFixed(3)} ` +
      `rc_es=${es.cutoff.toFixed(4)} head_list_capacity=${es.listCapacity} Verlet_capacity=${vc.listCapacity} ` +
      `densest_density=${vc.densestDensity.toFixed(4)}\n` +
      `SOUP-FORCES-CONFINED occupancy: main max=${occ.main ? occ.main.max : 'n/a'}/${vc.listCapacity} ` +
      `heads max=${occ.es ? occ.es.max : 'n/a'}/${es.listCapacity}`,
  )
  expect(r.maxDiff).toBeLessThan(1e-2)
  // The wall really is part of this configuration's force field, on more than one bead.
  expect(r.outside).toBeGreaterThan(0)
  expect(r.maxWall).toBeGreaterThan(0)
  // The densest density really was taken over the parcel: the box-average would be ~1.1/5.8 = 0.19,
  // and the dry-phase target 1.34 is not in play here (no cycling), so this must read the parcel's own.
  expect(vc.densestDensity).toBeGreaterThan(1)
  // Neither derived list overflowed -- the failure mode a box-volume density would have produced.
  if (occ.main) expect(occ.main.max).toBeLessThan(vc.listCapacity)
  if (occ.es) expect(occ.es.max).toBeLessThan(es.listCapacity)
})
