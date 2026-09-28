// Task 'confined-parcel' (2026-08-21): the implementation proofs for the finite parcel of water and
// its soft neutral wall, BEFORE any campaign step is spent on it.
//
// Four things have to be true before a confined run's numbers mean anything, and each is a test here
// rather than an argument in a report:
//  1. the wall force IS the gradient of the wall potential -- checked against a CENTRAL DIFFERENCE of
//     that potential, not against a second hand-written copy of the same formula, and measured in
//     isolation (a configuration where every pair force is exactly zero by construction);
//  2. the minimum-image convention and the position wrap are the IDENTITY in a confined run --
//     measured with an instrument shown, on the SAME positions, to detect folding when the box is
//     deliberately shrunk to where folding must happen (a no-wrap check that cannot detect a wrap is
//     not a check);
//  3. the axis-wrapping instrument -- this project's own discriminating measurement, 3 of 3 at 23 of
//     23 checkpoints in every periodic campaign -- reads 0 of 3 in a confined run;
//  4. the geometry that makes (2) true is REFUSED when it does not hold, and the two combinations
//     that would silently mean something else (a mineral platelet, the zero-tension area move) are
//     refused too.
//
// Everything after the stepping happens in NODE, on the run's own checkpoint (base64, one JSON token
// -- the transfer path soup/src/checkpoint.ts exists for), so no per-particle array ever crosses the
// CDP boundary as JSON numbers, and the measurement functions used are the very ones the campaign
// uses.
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { wrapsOnAxis } from './helpers/periodic-geometry'
import { decodeCheckpointResume, type CheckpointFile } from '../soup/src/checkpoint'
import { wallStats } from '../soup/src/soup-confine'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { loadSoup } from '../soup/src/rules'

afterAll(shutdownGpu)

// ------------------------------------------------------------------------------------------------
// 1. THE WALL FORCE AGAINST A NUMERICAL GRADIENT.
//
// 12 particles, one per direction of a Fibonacci sphere, at radii straddling the parcel surface,
// placed through a hand-built `resume` payload -- the only way to put a particle deliberately
// OUTSIDE the parcel, since the fresh-creation lattice by construction never does
// (soup/src/soup-confine.ts's sphereLatticeSites). Their mutual separations are ~14 sigma at R = 20,
// far beyond the 2.947 sigma interaction range, so EVERY pair force is exactly zero (measured, not
// assumed: minPairSep is asserted) and the force the GPU reports is the wall force alone.
// ------------------------------------------------------------------------------------------------
test('wall: the GPU force matches the numerical gradient of its own potential, and inside the parcel it is exactly zero', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const BOX = 60
    const R = 20
    const K = 137
    const N = 12
    const c = [BOX / 2, BOX / 2, BOX / 2]
    // Fibonacci directions: the most uniform 12 directions available with no special case, so the
    // minimum pairwise angle -- and therefore the minimum pairwise DISTANCE -- is as large as it can
    // be for this count.
    const dirs: [number, number, number][] = []
    const ga = Math.PI * (3 - Math.sqrt(5))
    for (let i = 0; i < N; i++) {
      const y = 1 - (2 * i) / (N - 1)
      const rad = Math.sqrt(Math.max(0, 1 - y * y))
      const th = ga * i
      dirs.push([Math.cos(th) * rad, y, Math.sin(th) * rad])
    }
    const radii: number[] = []
    for (let i = 0; i < N; i++) radii.push(R - 3 + i * 0.6) // 6 inside the parcel, 6 outside
    const positions = dirs.map((d, i) => [c[0] + d[0] * radii[i], c[1] + d[1] * radii[i], c[2] + d[2] * radii[i]])
    const pos = new Float32Array(N * 4)
    for (let i = 0; i < N; i++) {
      pos[i * 4] = positions[i][0]
      pos[i * 4 + 1] = positions[i][1]
      pos[i * 4 + 2] = positions[i][2]
      pos[i * 4 + 3] = 0 // kind 0 = carbon
    }
    const resume = {
      globalStep: 0,
      liveBox: [BOX, BOX, BOX] as [number, number, number],
      activeCounts: { C: N, O: 0, H: 0, M: 0, W: 0, K: 0 },
      positions: pos,
      velocities: new Float32Array(N * 4),
      bondSlots: new Uint32Array(N * 3).fill(0xffffffff),
      centerLink: new Uint32Array(N).fill(0xffffffff),
      centerHeldSteps: new Uint32Array(N),
      desorbEvents: new Uint32Array(2),
      bondRng: new Uint32Array(N),
      thermoRng: new Uint32Array(N),
      events: {},
      charges: new Float32Array(N),
      protonationRng: 1,
    }
    const sys = await api.createSoup({
      box: [BOX, BOX, BOX],
      seed: 3,
      kT: 1.1,
      clay: false,
      start: { C: N, O: 0, H: 0, M: 0, W: 0 },
      confine: { radiusSigma: R, stiffness: K },
      resume,
    })
    const F = await sys.forces()
    const Fb = await sys.forcesBruteForce()
    let minSep = Infinity
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const dx = positions[i][0] - positions[j][0]
        const dy = positions[i][1] - positions[j][1]
        const dz = positions[i][2] - positions[j][2]
        minSep = Math.min(minSep, Math.sqrt(dx * dx + dy * dy + dz * dz))
      }
    }
    const h = 1e-3
    const rows: any[] = []
    let maxErrAnalytic = 0
    let maxRelAnalytic = 0
    let maxRelNumeric = 0
    let maxErrNumeric = 0
    let maxErrNumericAtKink = 0
    let maxInsideForce = 0
    let maxGridVsBrute = 0
    for (let i = 0; i < N; i++) {
      const gpu = [F[i * 4], F[i * 4 + 1], F[i * 4 + 2]]
      const cpu = api.wallForceAt(positions[i], [BOX, BOX, BOX], R, K)
      const s = radii[i]
      // Central difference of the potential along the radial direction: F_r = -dU/ds.
      const numericRadial = -(api.wallPotential(s + h, R, K) - api.wallPotential(s - h, R, K)) / (2 * h)
      const gpuRadial = gpu[0] * dirs[i][0] + gpu[1] * dirs[i][1] + gpu[2] * dirs[i][2]
      const absA = Math.max(Math.abs(gpu[0] - cpu[0]), Math.abs(gpu[1] - cpu[1]), Math.abs(gpu[2] - cpu[2]))
      const scale = Math.max(1, Math.abs(cpu[0]), Math.abs(cpu[1]), Math.abs(cpu[2]))
      maxErrAnalytic = Math.max(maxErrAnalytic, absA)
      // The interesting bound is RELATIVE: the wall force reaches 493 here, and the kernel is
      // float32, so an absolute bound tight enough to be meaningful at s = R + 0.6 is unreachable at
      // s = R + 3.6 for reasons that have nothing to do with the formula.
      maxRelAnalytic = Math.max(maxRelAnalytic, absA / scale)
      // A central difference is not a valid gradient estimate AT the potential's own kink: U is C^1
      // at s = R but its second derivative jumps there, so a symmetric stencil straddling R returns
      // -(0.5*k*h^2)/(2h) = -k*h/4 no matter how small h is. That point is therefore accounted
      // separately, against exactly that bound, instead of being quietly excluded -- it is a property
      // of the stencil, not of the kernel, and the kernel's own value there (0, the correct one-sided
      // limit from inside the parcel) is checked by maxErrAnalytic like every other point.
      if (Math.abs(s - R) <= h) maxErrNumericAtKink = Math.max(maxErrNumericAtKink, Math.abs(gpuRadial - numericRadial))
      else {
        maxErrNumeric = Math.max(maxErrNumeric, Math.abs(gpuRadial - numericRadial))
        maxRelNumeric = Math.max(maxRelNumeric, Math.abs(gpuRadial - numericRadial) / Math.max(1, Math.abs(numericRadial)))
      }
      if (s <= R) maxInsideForce = Math.max(maxInsideForce, Math.abs(gpu[0]), Math.abs(gpu[1]), Math.abs(gpu[2]))
      for (let k = 0; k < 3; k++) maxGridVsBrute = Math.max(maxGridVsBrute, Math.abs(F[i * 4 + k] - Fb[i * 4 + k]))
      rows.push({ s, gpuRadial, numericRadial, analytic: cpu[0] * dirs[i][0] + cpu[1] * dirs[i][1] + cpu[2] * dirs[i][2] })
    }
    const conf = sys.confinement()
    sys.dispose()
    return { rows, maxErrAnalytic, maxRelAnalytic, maxErrNumeric, maxRelNumeric, maxErrNumericAtKink, maxInsideForce, maxGridVsBrute, minSep, conf, R, K, h }
  })
  expect(consoleWarnings, `the browser reported a GPU error/warning:\n${consoleWarnings.join('\n')}`).toEqual([])
  console.log(
    `WALL-GRADIENT R=${r.R} k=${r.K} minPairSep=${r.minSep.toFixed(3)} (interactionRange=2.947)\n` +
      r.rows
        .map(
          (x: any) =>
            `  s=${x.s.toFixed(4)} F_r(GPU)=${x.gpuRadial.toFixed(6)} F_r(numeric)=${x.numericRadial.toFixed(6)} F_r(analytic)=${x.analytic.toFixed(6)}`,
        )
        .join('\n') +
      `\nWALL-GRADIENT max|GPU-analytic|=${r.maxErrAnalytic.toExponential(4)} (rel. ${r.maxRelAnalytic.toExponential(4)}) ` +
      `max|GPU-numeric|=${r.maxErrNumeric.toExponential(4)} (rel. ${r.maxRelNumeric.toExponential(4)}) ` +
      `at_kink(s=R)|GPU-numeric|=${r.maxErrNumericAtKink.toExponential(4)} (stencil bound k*h/4=${((r.K * r.h) / 4).toExponential(4)}) ` +
      `max|F| inside=${r.maxInsideForce.toExponential(4)} max|grid-brute|=${r.maxGridVsBrute.toExponential(4)}`,
  )
  expect(r.minSep).toBeGreaterThan(3)
  expect(r.maxInsideForce).toBe(0)
  // float32 over a handful of operations: 1e-5 relative is ~80x the single-op epsilon 1.19e-7 and
  // ~1e5 times tighter than any wrong sign, factor or centre convention could possibly land.
  expect(r.maxRelNumeric).toBeLessThan(1e-5)
  expect(r.maxRelAnalytic).toBeLessThan(1e-5)
  // At the kink the disagreement is the STENCIL's own, bounded by k*h/4 and nothing else.
  expect(r.maxErrNumericAtKink).toBeLessThanOrEqual((r.K * r.h) / 4 + 1e-9)
  // Applied identically on the grid/Verlet path and on the O(N^2) reference -- the trap a one-body
  // term added inside three of the four force kernels would fall into.
  expect(r.maxGridVsBrute).toBe(0)
})

// ------------------------------------------------------------------------------------------------
// 2 + 3. NO WRAP (with the instrument's own positive control) AND THE SETUP CHECK.
// ------------------------------------------------------------------------------------------------
test('confinement: the minimum image is the identity, wrapping never fires, and wrapping axes are 0 of 3', async () => {
  const page = await gpuPage()
  const BOX = 100
  const R = 20
  const file = (await page.evaluate(
    async (box: number, radius: number) => {
      const api = (window as any).api
      const sys = await api.createSoup({
        box: [box, box, box],
        seed: 5,
        kT: 1.1,
        clay: false,
        // Liquid water at 0.8 sigma^-3 inside the parcel plus organics in the campaign's own ratios,
        // scaled to this parcel: small enough to be a test, dense enough to be the real regime.
        start: { C: 4274, O: 1424, H: 4274, M: 111, W: 26808 },
        confine: { radiusSigma: radius, stiffness: 100 },
      })
      await sys.relaxColdStart()
      await sys.step(500)
      const conf = sys.confinement()
      const cp = await api.encodeCheckpoint(sys, { box: [box, box, box], seed: 5, kT: 1.1, confine: { radiusSigma: radius, stiffness: 100 } })
      sys.dispose()
      return { cp, conf }
    },
    BOX,
    R,
  )) as { cp: CheckpointFile; conf: Record<string, number | boolean> }

  // --- everything below is pure Node, off the checkpoint's own base64 payload -------------------
  const st = decodeCheckpointResume(file.cp)
  const pos = st.positions
  const n = pos.length / 4
  const soup = loadSoup()
  const params = loadParams()
  void st.bondSlots
  const ws = wallStats(
    pos,
    n,
    [BOX, BOX, BOX],
    Number(file.conf.radiusLive),
    4.687,
    soup.monomers.map((m) => m.id),
    soup.monomers.findIndex((m) => m.id === soup.solvent.waterId),
  )

  /** Over a spread sample of pairs, how far does the minimum-image displacement differ from the true
   * one? Zero means the fold never engaged. Run twice on the SAME positions: at the real box (must be
   * 0) and at a box small enough that folding is unavoidable (must not be 0) -- the second run is
   * what proves the first is a measurement and not a tautology. */
  const foldStat = (L: number) => {
    let maxDelta = 0
    let folded = 0
    let pairs = 0
    for (let i = 0; i + 1 < n; i += 7) {
      const j = (i + 1 + ((i * 131) % (n - 1))) % n
      pairs++
      for (let a = 0; a < 3; a++) {
        const d = pos[i * 4 + a] - pos[j * 4 + a]
        const m = d - Math.round(d / L) * L
        const delta = Math.abs(m - d)
        if (delta > 0) folded++
        if (delta > maxDelta) maxDelta = delta
      }
    }
    return { maxDelta, folded, pairs }
  }
  const atRealBox = foldStat(BOX)
  const atShrunkBox = foldStat(2 * R)
  let maxWrapShift = 0
  for (let i = 0; i < n; i++) {
    for (let a = 0; a < 3; a++) {
      const x = pos[i * 4 + a]
      maxWrapShift = Math.max(maxWrapShift, Math.abs(x - (x - Math.floor(x / BOX) * BOX)))
    }
  }
  // THE SETUP CHECK, with the project's own instrument and the project's own cutoff, applied to the
  // WHOLE system's contact network -- a set strictly larger than any aggregate, so if this does not
  // wrap, no aggregate inside it can. Non-vacuous at any stage, unlike the largest recognised
  // aggregate, which does not exist yet 500 steps into a monomers-only start.
  const cutoff = wcaCutoff(params.sigma * Math.max(...soup.monomers.map((m) => m.radiusSigma))) + params.attraction.wc
  const wrapsAll = ([0, 1, 2] as const).map((a) => wrapsOnAxis(pos, [BOX, BOX, BOX], cutoff, a))

  console.log(
    `NO-WRAP box=${BOX} R=${R} N=${n} maxR=${ws.maxRadius.toFixed(4)} penetration=${ws.penetration.toFixed(4)} ` +
      `face_clearance=${ws.faceClearance.toFixed(4)} (largest interaction radius ${Number(file.conf.cutMax).toFixed(4)}) ` +
      `maxPair=${ws.maxPairSeparation.toFixed(3)} L/2=${ws.halfBox.toFixed(3)} strong=${ws.strongNoWrap}\n` +
      `NO-WRAP folds_at_real_box=${atRealBox.folded}/${atRealBox.pairs * 3} maxDelta=${atRealBox.maxDelta} | ` +
      `control at box ${2 * R}: folds=${atShrunkBox.folded}/${atShrunkBox.pairs * 3} maxDelta=${atShrunkBox.maxDelta.toFixed(3)}\n` +
      `NO-WRAP max|wrap shift of position|=${maxWrapShift} wrapping_axes(whole system)=${wrapsAll.filter(Boolean).length}/3 ${JSON.stringify(wrapsAll)}\n` +
      `WALL-SHELL shell=${ws.shell} uniform_fraction=${ws.species[0].uniformFraction.toFixed(4)} ` +
      `enrichment=${JSON.stringify(ws.species.map((s) => [s.id, Number(s.enrichment.toFixed(3))]))} ` +
      `enrichment_vs_water=${JSON.stringify(ws.species.map((s) => [s.id, Number(s.enrichmentVsSolvent.toFixed(4))]))} ` +
      `in_shell=${JSON.stringify(ws.species.map((s) => [s.id, s.inShell]))}`,
  )
  // (a) the geometry holds by measurement, not by intent
  expect(ws.faceClearance).toBeGreaterThan(Number(file.conf.cutMax))
  expect(ws.maxPairSeparation).toBeLessThan(ws.halfBox)
  expect(ws.strongNoWrap).toBe(true)
  // (b) the minimum image never folded a single component of a single sampled pair
  expect(atRealBox.folded).toBe(0)
  expect(atRealBox.maxDelta).toBe(0)
  // (c) ... and the same instrument DOES see folding once the box is shrunk to where it must happen
  expect(atShrunkBox.folded).toBeGreaterThan(0)
  expect(atShrunkBox.maxDelta).toBeGreaterThan(1)
  // (d) the position wrap is the identity for every coordinate of every particle
  expect(maxWrapShift).toBe(0)
  // (e) THE SETUP CHECK: 0 of 3 axes for the WHOLE system's contact network
  expect(wrapsAll.filter(Boolean).length).toBe(0)
  // (f) the wall contained everything: nothing is more than a fraction of sigma outside it
  expect(ws.penetration).toBeLessThan(1)
  // (h) THE WALL IS NEUTRAL, measured: every species sits in the outer shell in the same proportion
  // the water does, so nothing is preferentially adsorbed or depleted by it. This is the null the
  // competing-sink question is judged against for the rest of the task.
  for (const sp of ws.species) {
    if (sp.count === 0) continue
    expect(Math.abs(sp.enrichmentVsSolvent - 1), `species ${sp.id}: enrichment vs water ${sp.enrichmentVsSolvent}`).toBeLessThan(0.3)
  }
  // (g) the confinement round-tripped through the checkpoint's own config
  expect(file.cp.config.confine).toEqual({ radiusSigma: R, stiffness: 100 })
})

// ------------------------------------------------------------------------------------------------
// 4. THE REFUSALS. Each would otherwise produce a number that does not mean what its name says.
// ------------------------------------------------------------------------------------------------
test('confinement: a tight box, a clay sheet and the MC area move refuse instead of computing silently', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const out: Record<string, string> = {}
    const start = { C: 300, O: 100, H: 300, M: 10, W: 2000 }
    try {
      await api.createSoup({ box: [42, 42, 42], seed: 1, kT: 1.1, clay: false, start, confine: { radiusSigma: 20, stiffness: 100 } })
      out.tightBox = 'did not throw'
    } catch (e: any) {
      out.tightBox = e.message
    }
    try {
      await api.createSoup({ box: [100, 100, 100], seed: 1, kT: 1.1, clay: true, start, confine: { radiusSigma: 20, stiffness: 100 } })
      out.clay = 'did not throw'
    } catch (e: any) {
      out.clay = e.message
    }
    const sys = await api.createSoup({ box: [100, 100, 100], seed: 1, kT: 1.1, clay: false, start, confine: { radiusSigma: 20, stiffness: 100 } })
    try {
      await sys.areaMove(1)
      out.areaMove = 'did not throw'
    } catch (e: any) {
      out.areaMove = e.message
    }
    sys.dispose()
    return out
  })
  console.log(`CONFINE-REFUSALS\n  tight box: ${r.tightBox}\n  clay: ${r.clay}\n  areaMove: ${r.areaMove}`)
  expect(r.tightBox).toContain('clearance to the box face')
  expect(r.clay).toContain('incompatible with a mineral sheet')
  expect(r.areaMove).toContain('makes no sense in a confined parcel')
})
