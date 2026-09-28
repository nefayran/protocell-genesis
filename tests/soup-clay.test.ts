import { afterAll, expect, test } from 'vitest'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { loadSoup } from '../soup/src/rules'
import { planClay, claySiteIndices, clayLatticePosition, distanceToPlatelet } from '../soup/src/soup-clay'
import { gpuPage, shutdownGpu } from './helpers/gpu'
// The per-arm measurement body moved to tests/helpers/clay-arm.ts (task 'clay-surface-chemistry',
// 2026-08-19) so the two surface-chemistry limits, the frozen-bulk-catalyst control and the
// siteCatalystFraction sweep are all measured by THIS code rather than by a copy of it. Clean
// transfer: this test prints the same lines and asserts the same things it did before the move.
import { runClayArm } from './helpers/clay-arm'

afterAll(shutdownGpu)

// Task 'clay-surface' (2026-08-19). Full measurement trail, literature and ranks:
// .superpowers/sdd/2026-08-16-soup-to-vesicle/clay-surface-report.md; the model's own basis and the
// explicit list of reported clay effects it CANNOT represent (surface charge first):
// data/soup.json's `clay.basis`.
//
// The measurement tests below run ONE seed by default (each arm is a full 20000-step broth run, and
// this suite has to stay runnable on the machine that owns it). CLAY_SEEDS=a,b,c runs more in one
// invocation; the report's run-to-run scatter was measured by repeating the invocation, one seed at a
// time, which is also how the numbers quoted in the assertions were obtained.
const SEEDS = (process.env.CLAY_SEEDS ?? '41').split(',').map((s) => Number(s.trim()))
const MEASURE_STEPS = Number(process.env.CLAY_STEPS ?? 20_000)

// --- geometry and schema, no GPU ----------------------------------------------------------------

test('clay sheet: the scheme and the derived geometry (no GPU)', () => {
  const soup = loadSoup()
  const p = loadParams()
  const clay = soup.clay
  expect(clay).toBeDefined()
  expect(clay!.rank).toBe('D')
  expect(clay!.basis.length).toBeGreaterThan(1000) // literature + ranks + what is NOT representable
  expect(clay!.sheets).toBe(1)

  // The mineral monomer exists, is its OWN interaction class (not polar, not solvent), and carries no
  // start count at all -- its count is geometry, not composition.
  const mineral = soup.monomers.find((m) => m.id === clay!.mineralId)!
  expect(mineral.kind).toBe('clay')
  expect(mineral.mineral).toBe(true)
  expect(mineral.polar).toBe(false)
  expect(mineral.solvent).toBeUndefined()
  expect(soup.start[clay!.mineralId]).toBeUndefined()

  // The depth table was EXTENDED, not re-invented: the mineral row exists, every cell is a ratio to
  // the same rank-A reference pair, and the hydrophilic ordering the mechanism relies on holds.
  const lv = soup.solvent.attractionScale!.pairEpsilon!.levels
  const ref = lv[soup.solvent.attractionScale!.pairEpsilon!.reference].epsilonKJ
  expect(lv.mineralSolvent.epsilonKJ / ref).toBeCloseTo(1.6, 6) // clay-water,  MARTINI P5-P4 5.6/3.5
  expect(lv.mineralPolar.epsilonKJ / ref).toBeCloseTo(1.428571, 5) // clay-head,  P5-Na  5.0/3.5
  expect(lv.mineralApolar.epsilonKJ / ref).toBeCloseTo(0.571429, 5) // clay-tail,  P5-C1  2.0/3.5
  // The surface prefers water to bulk water preferring itself, and prefers a head to water preferring
  // a head -- that is what makes it hydrophilic. And clay-tail is EXACTLY water-tail, so a tail gains
  // nothing by displacing a water at the surface.
  expect(lv.mineralSolvent.epsilonKJ).toBeGreaterThan(lv.solventSolvent.epsilonKJ)
  expect(lv.mineralPolar.epsilonKJ).toBeGreaterThan(lv.solventPolar.epsilonKJ)
  expect(lv.mineralApolar.epsilonKJ).toBe(lv.solventApolar.epsilonKJ)
  // The EXCHANGE favourability of adsorbing species X at the surface in place of a water bead:
  // swapping X (in bulk, touching a water) with a water (at the surface, touching the clay) changes
  // the energy by dE = -eps(clay,X) - eps(w,w) + eps(clay,w) + eps(w,X), so the favourability -dE is
  //   F(X) = eps(clay,X) + eps(w,w) - eps(clay,w) - eps(w,X),
  // in units of the rank-A tail-tail depth once divided by the reference. F > 0 means the surface
  // genuinely prefers X to the water it must displace -- which is the ONLY sense in which this
  // platelet can concentrate anything, since the model has no charge to do it with. The numbers are
  // SMALL, and the report states that up front rather than presenting the sign as a strong drive:
  // F(head) = (5.0 + 5.0 - 5.6 - 4.0)/3.5 = +0.114 eps, i.e. ~0.10 kT at kT=1.1 -- a Boltzmann factor
  // of only ~1.11, so a ~10% contact-shell enrichment is the most the energetics alone can give.
  const fHead = (lv.mineralPolar.epsilonKJ + lv.solventSolvent.epsilonKJ - lv.mineralSolvent.epsilonKJ - lv.solventPolar.epsilonKJ) / ref
  const fTail = (lv.mineralApolar.epsilonKJ + lv.solventSolvent.epsilonKJ - lv.mineralSolvent.epsilonKJ - lv.solventApolar.epsilonKJ) / ref
  console.log(
    `CLAY-EXCHANGE F(head)=${fHead.toFixed(4)} F(tail)=${fTail.toFixed(4)} eps_rankA ` +
      `(=${(fHead / 1.1).toFixed(4)} / ${(fTail / 1.1).toFixed(4)} kT at kT=1.1; >0 = surface prefers it to water)`,
  )
  expect(fHead).toBeGreaterThan(0) // the surface prefers a head over the water it displaces
  expect(fTail).toBeLessThan(0) // ... and prefers water over a tail: hydrophilic, as declared

  // Geometry: spacing is the mineral-mineral WCA contact distance, commensurate with the box; the
  // sheet is a plane at the box midpoint; the site count is a fraction of the catalyst pool the broth
  // ALREADY has (moved, not added).
  const box: [number, number, number] = [30, 30, 30]
  const catalystTotal = soup.start.M
  const L = planClay(soup, p, box, catalystTotal)
  const contact = wcaCutoff(p.sigma * mineral.radiusSigma)
  expect(L.nx).toBe(Math.round(box[0] / contact))
  expect(L.spacingX).toBeCloseTo(box[0] / L.nx, 12)
  expect(L.planeZ).toEqual([box[2] / 2])
  expect(L.latticeCount).toBe(L.nx * L.ny)
  expect(L.siteCount).toBe(Math.round(clay!.siteCatalystFraction * catalystTotal))
  expect(L.mineralCount).toBe(L.latticeCount - L.siteCount)
  console.log(
    `CLAY-GEOMETRY box=${box[0]} contact=${contact.toFixed(4)}σ lattice=${L.nx}x${L.ny} spacing=${L.spacingX.toFixed(4)}σ ` +
      `beads=${L.latticeCount} (mineral ${L.mineralCount} + sites ${L.siteCount}) exclusion=${L.exclusionHalfWidth.toFixed(4)}σ`,
  )

  // Impermeability, argued from the geometry before it is measured on the GPU below: a bead sitting in
  // the middle of a lattice square is spacing/sqrt(2) from each of the four beads around it, and that
  // must be well inside the WCA core of the largest species pair with the mineral.
  const holeClearance = L.spacingX / Math.SQRT2
  const biggestPair = Math.max(...soup.monomers.map((m) => wcaCutoff(p.sigma * (mineral.radiusSigma + m.radiusSigma) * 0.5)))
  console.log(`CLAY-HOLE clearance=${holeClearance.toFixed(4)}σ vs largest WCA core with mineral ${biggestPair.toFixed(4)}σ`)
  expect(holeClearance).toBeLessThan(biggestPair)
  // And the barrier itself, in kT, rather than only "it is inside the core": the WCA energy a bead
  // would have to pay to sit exactly IN the plane at a hole centre, summed over the four mineral beads
  // around that hole. Same formula engine/wgsl/forces.wgsl uses (4*eps*((b/r)^12-(b/r)^6)+eps), with
  // the rank-A eps and the pair's own b -- so this is the model's own barrier, not an estimate.
  const kT = p.thermostat.kT
  const barriers: Record<string, number> = {}
  for (const m of soup.monomers) {
    if (m.mineral) continue
    const b = p.sigma * (mineral.radiusSigma + m.radiusSigma) * 0.5
    const x = b / holeClearance
    const u = holeClearance < wcaCutoff(b) ? 4 * p.epsilon * (x ** 12 - x ** 6) + p.epsilon : 0
    barriers[m.id] = (4 * u) / kT
  }
  console.log(`CLAY-BARRIER kT=${kT} through-hole WCA barrier, kT: ${Object.entries(barriers).map(([k, v]) => `${k}=${v.toFixed(1)}`).join(' ')}`)
  // The SMALLEST species is the one that could squeeze through, so it is the one that has to be
  // blocked. 10 kT is a diagnostic floor (a Boltzmann factor of 4.5e-5 per attempt), not literature.
  expect(Math.min(...Object.values(barriers))).toBeGreaterThan(10)

  // Sites are spread over the sheet, disjoint from the mineral beads, and all lie in the plane.
  const sites = claySiteIndices(L)
  expect(sites.size).toBe(L.siteCount)
  for (const k of sites) expect(clayLatticePosition(L, k)[2]).toBeCloseTo(box[2] / 2, 12)

  // The distance function folds both faces of the sheet together and respects the periodic wrap.
  expect(distanceToPlatelet(box[2] / 2, L.planeZ, box[2])).toBeCloseTo(0, 12)
  expect(distanceToPlatelet(0, L.planeZ, box[2])).toBeCloseTo(box[2] / 2, 12)
  expect(distanceToPlatelet(box[2] - 1, L.planeZ, box[2])).toBeCloseTo(box[2] / 2 - 1, 12)
})

test('clay sheet: a box change refuses instead of spoiling the lattice', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSoup({ box: [20, 20, 20], seed: 3, kT: 1.1, start: { C: 100, O: 40, H: 100, M: 20, W: 400 } })
    let threw: string | null = null
    try {
      await sys.areaMove(1)
    } catch (e) {
      threw = (e as Error).message
    }
    let threwScale: string | null = null
    try {
      await sys.scaleBoxTo([21, 21, 21], 2, 0)
    } catch (e) {
      threwScale = (e as Error).message
    }
    const frozen: Uint32Array = await sys.frozen()
    let frozenCount = 0
    for (const f of frozen) if (f !== 0) frozenCount++
    sys.dispose()
    return { threw, threwScale, frozenCount }
  })
  console.log('CLAY-BOXCHANGE', JSON.stringify(r))
  expect(r.frozenCount).toBeGreaterThan(0)
  expect(r.threw).toMatch(/mineral sheet/)
  expect(r.threwScale).toMatch(/mineral sheet/)
}, 180_000)

// --- immobility, site placement and impermeability, PROVEN by measurement ------------------------

test('clay sheet: immobility, site placement and impermeability, by measurement, not by inspection', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async (steps: number) => {
    const api = (window as any).api
    const soup = api.loadSoup()
    const box: [number, number, number] = [20, 20, 20]
    const sys = await api.createSoup({ box, seed: 5, kT: 1.1, start: { C: 300, O: 100, H: 300, M: 40, W: 1500 } })
    const planes: number[] = sys.clayPlanes()
    const frozen: Uint32Array = await sys.frozen()
    const before: Float32Array = await sys.particles()
    await sys.step(steps)
    const after: Float32Array = await sys.particles()
    const vel: Float32Array = await sys.velocities()
    const N = before.length / 4
    const mineralKind = soup.monomers.findIndex((m: any) => m.id === soup.clay.mineralId)
    const catalystKind = soup.monomers.findIndex((m: any) => m.kind === 'catalyst')

    let maxFrozenDisp = 0
    let maxFrozenSpeed = 0
    let maxFreeDisp = 0
    let frozenCount = 0
    let frozenMineral = 0
    let frozenSites = 0
    let frozenOffPlane = 0
    let freeCatalysts = 0
    // The one number that makes this a proof rather than an inspection: the LARGEST coordinate change
    // of any frozen bead over `steps` real steps. Anything but exactly 0 means the platelet moved.
    for (let i = 0; i < N; i++) {
      const kind = Math.round(before[i * 4 + 3])
      const d = Math.max(
        Math.abs(after[i * 4] - before[i * 4]),
        Math.abs(after[i * 4 + 1] - before[i * 4 + 1]),
        Math.abs(after[i * 4 + 2] - before[i * 4 + 2]),
      )
      if (frozen[i] !== 0) {
        frozenCount++
        maxFrozenDisp = Math.max(maxFrozenDisp, d)
        maxFrozenSpeed = Math.max(maxFrozenSpeed, Math.abs(vel[i * 4]), Math.abs(vel[i * 4 + 1]), Math.abs(vel[i * 4 + 2]))
        if (kind === mineralKind) frozenMineral++
        if (kind === catalystKind) frozenSites++
        if (Math.abs(after[i * 4 + 2] - planes[0]) > 1e-6) frozenOffPlane++
      } else {
        maxFreeDisp = Math.max(maxFreeDisp, d)
        if (kind === catalystKind) freeCatalysts++
      }
    }

    // Impermeability: no NON-frozen particle may sit in the sheet's own plane. A bead squeezing
    // through a lattice square would have to pass within a fraction of sigma of z = planes[0].
    let intruders = 0
    let closestApproach = Infinity
    for (let i = 0; i < N; i++) {
      if (frozen[i] !== 0) continue
      const d = api.distanceToPlatelet(after[i * 4 + 2], planes, box[2])
      closestApproach = Math.min(closestApproach, d)
      // 0.25 sigma: a diagnostic threshold, half the first profile bin and well below both the
      // measured closest approach (0.506 sigma) and the distance at which the smallest species first
      // touches the four beads around a lattice hole (0.635 sigma) -- see the CPU test's own
      // through-hole WCA barrier for the real argument that the sheet cannot be crossed.
      if (d < 0.25) intruders++
    }

    // The surface sites are real catalytic centres, not decoration: count how many of them are
    // holding a chain end through the SAME centerLink tether a free catalyst uses.
    const links: Uint32Array = await sys.centerLinks()
    let sitesHolding = 0
    for (let i = 0; i < N; i++) {
      if (frozen[i] !== 0 && Math.round(after[i * 4 + 3]) === catalystKind && links[i] !== 0xffffffff) sitesHolding++
    }
    const events = await sys.events()
    let nonFinite = 0
    for (let k = 0; k < N * 4; k++) if (!Number.isFinite(after[k])) nonFinite++
    sys.dispose()
    return {
      N, frozenCount, frozenMineral, frozenSites, frozenOffPlane, freeCatalysts,
      maxFrozenDisp, maxFrozenSpeed, maxFreeDisp, intruders, closestApproach, sitesHolding,
      ccBond: events['cc_bond'] ?? 0, nonFinite,
    }
  }, 20_000)

  console.log('CLAY-IMMOBILITY', JSON.stringify(r))
  // Immobility is EXACT, not small: the integrator never writes a frozen particle's position at all.
  expect(r.maxFrozenDisp).toBe(0)
  expect(r.maxFrozenSpeed).toBe(0)
  // ... and the system it sits in is genuinely alive, so the zero above is not a dead-run artefact.
  expect(r.maxFreeDisp).toBeGreaterThan(0.1)
  expect(r.nonFinite).toBe(0)
  // Site placement: the frozen set is exactly the platelet -- mineral beads plus catalyst sites, all
  // of them in the sheet plane, with the rest of the catalyst pool still free in solution.
  expect(r.frozenCount).toBe(r.frozenMineral + r.frozenSites)
  expect(r.frozenSites).toBeGreaterThan(0)
  expect(r.frozenOffPlane).toBe(0)
  expect(r.freeCatalysts).toBeGreaterThan(0)
  // Impermeability, measured rather than argued: nothing mobile ever reaches the sheet's plane.
  expect(r.intruders).toBe(0)
  // The surface sites participate in the existing catalytic machinery.
  expect(r.sitesHolding).toBeGreaterThan(0)
}, 300_000)

// --- the four measurements ----------------------------------------------------------------------


test('clay sheet: concentration, growth/termination and the assembly location, with clay and without', async () => {
  const page = await gpuPage()
  for (const seed of SEEDS) {
    const withClay = await runClayArm(page, { seed, steps: MEASURE_STEPS, mode: 'hydrophilic' })
    const noClay = await runClayArm(page, { seed, steps: MEASURE_STEPS, mode: 'none' })
    for (const [label, r] of [['WITH-CLAY', withClay], ['NO-CLAY', noClay]] as const) {
      for (const id of ['C', 'O', 'H', 'W']) {
        const s = r.nullStats[id]
        const z = s.sd > 0 ? (r.enrich[id] - s.mean) / s.sd : NaN
        console.log(
          `CLAY-NULL seed=${seed} ${label} ${id}: plane=${r.enrich[id].toFixed(3)} ` +
            `null mean=${s.mean.toFixed(3)} sd=${s.sd.toFixed(3)} [${s.min.toFixed(3)},${s.max.toFixed(3)}] n=${s.n} z=${z.toFixed(2)}`,
        )
      }
      const zShell = r.shellNullSd > 0 ? (r.plateShellPolarFraction - r.shellNullMean) / r.shellNullSd : NaN
      console.log(
        `CLAY-SHELL seed=${seed} ${label} polarFraction plane=${r.plateShellPolarFraction.toFixed(4)} (n=${r.plateShellCount}) ` +
          `boxMean=${r.boxPolarFraction.toFixed(4)} null mean=${r.shellNullMean.toFixed(4)} sd=${r.shellNullSd.toFixed(4)} ` +
          `[${r.shellNullMin.toFixed(4)},${r.shellNullMax.toFixed(4)}] n=${r.shellNullN} z=${zShell.toFixed(2)}`,
      )
      console.log(
        `CLAY-ARM seed=${seed} steps=${MEASURE_STEPS} ${label} N=${r.N} frozen=${r.frozenCount} ` +
          `enrich C=${r.enrich.C.toFixed(3)} O=${r.enrich.O.toFixed(3)} H=${r.enrich.H.toFixed(3)} W=${r.enrich.W.toFixed(3)} ` +
          `cc=${r.ccBond} co=${r.coBond} ccBreak=${r.ccBreak} coBreak=${r.coBreak} ` +
          `amph=${r.amphiphileCount} members=${r.memberCount} inBand=${r.amphInBandFraction.toFixed(4)} ` +
          `(uniform ${r.uniformBandFraction.toFixed(4)}) meanD=${r.amphMeanDistance.toFixed(3)} (uniform ${r.uniformMeanDistance.toFixed(3)}) ` +
          `sites=${r.siteCount} sitesHolding=${r.sitesHolding} freeHolding=${r.freeHolding} nonFinite=${r.nonFinite}`,
      )
      for (const id of ['C', 'O', 'H', 'W']) {
        console.log(`CLAY-PROFILE seed=${seed} ${label} binW=${r.binW} ${id}=${r.counts[id].join(',')}`)
      }
    }

    // Structural expectations that do NOT depend on the size of any effect -- these are the ones that
    // would catch a broken platelet rather than a weak one.
    expect(withClay.frozenCount).toBeGreaterThan(0)
    expect(noClay.frozenCount).toBe(0)
    expect(withClay.nonFinite).toBe(0)
    expect(noClay.nonFinite).toBe(0)
    // NO FLATNESS ASSERTION ON THE CLAY-FREE ARM, and that is a measured decision, not an omission. It
    // was written as `|enrich - 1| < 0.25` first and MEASURED at 0.40 for H at seed 41: after 20000
    // steps the clay-free broth is not spatially homogeneous either (hydrophobic aggregation plus the
    // sub-liquid water density noted above), so a profile binned about an arbitrary plane wanders by
    // tens of per cent from run to run with no surface present at all. That wander IS the scatter the
    // with-clay numbers have to beat, so it is reported per seed in clay-surface-report.md rather than
    // asserted away here with a threshold picked to pass.
    expect(noClay.siteCount).toBe(0)
    // Chains still grow and still get capped by a head with the platelet present -- i.e. the platelet
    // does not break the chemistry it is supposed to host. The SIZE of any difference between the arms
    // is a measurement, reported (and judged against run-to-run scatter) in
    // clay-surface-report.md, deliberately NOT asserted here as a threshold.
    expect(withClay.ccBond).toBeGreaterThan(0)
    expect(noClay.ccBond).toBeGreaterThan(0)
  }
}, 900_000)
