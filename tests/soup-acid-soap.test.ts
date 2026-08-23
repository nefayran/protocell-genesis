import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadSoup } from '../soup/src/rules'
import { loadParams } from '../engine/src/params'
import { CLASS_APOLAR, CLASS_POLAR, acidSoapScaleOf, attractionScaleTable } from '../soup/src/soup-attraction'

afterAll(shutdownGpu)

// Task 'acid-soap-pairing' (2026-08-23): THE RULE, pinned. The charge-assisted head-head attraction
// must fire between one PROTONATED head and one DEPROTONATED head and between NOTHING ELSE -- not
// between two protonated heads, not between two deprotonated heads, not between a head and a tail.
//
// HOW IT IS PINNED, and why this shape rather than reading the numbers off one run. The instrument is
// a DIFFERENCE of two systems that are identical in every input except the acid-soap depth
// (CreateSoupOpts.acidSoapScaleOverride): same box, same seed, same composition, same resumed
// positions, same resumed charges, same pH. That difference isolates the new term exactly, with no
// tolerance argument about the rest of the force field:
//  - with EVERY head in the same protonation state (all charges 0, then all charges -1) the two
//    systems must agree BIT FOR BIT -- max|dF| exactly 0. Anything else means the rule read something
//    other than the protonation state.
//  - with ALTERNATING charges the difference must be non-zero, and it must be non-zero ONLY on head
//    beads: a tail's force may change only through the heads it is bonded to, which this fixture
//    removes from the question by giving the heads no bonds at all.
// The all-same arms are the ones that could not be faked: a term that attracted every head pair (the
// polarPolar cell this project has kept at 0 all along) would show up there immediately.
test('кислотно-мыльная пара: притягиваются ТОЛЬКО голова протонированная + голова депротонированная', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const soup = api.loadSoup()
    const headKind = soup.monomers.findIndex((m: any) => m.kind === 'head')
    const carbonKind = soup.monomers.findIndex((m: any) => m.kind === 'carbon')
    const qDeprot = soup.electrostatics.chargeDeprotonated

    // A hand-built lattice of BARE beads: on a 6x6x6 grid of cells, THREE beads per cell -- two
    // heads 1.2 sigma apart and one tail 1.2 sigma away on another axis -- with no bonds at all.
    // Bond-free on purpose: every force difference measured below is then a NONBONDED difference, and
    // a head's neighbours are heads and tails at known distances rather than whatever a relaxed broth
    // happens to make. The tail beads are what make "the term touches ONLY head pairs" measurable.
    const box: [number, number, number] = [18, 18, 18]
    const n = 6
    const s = box[0] / n
    const nCell = n * n * n
    const N = nCell * 3
    const positions = new Float32Array(N * 4)
    let k = 0
    for (let iz = 0; iz < n; iz++) {
      for (let iy = 0; iy < n; iy++) {
        for (let ix = 0; ix < n; ix++) {
          const x = (ix + 0.5) * s
          const y = (iy + 0.5) * s
          const z = (iz + 0.5) * s
          positions.set([x, y, z, headKind], k * 4)
          positions.set([x + 1.2, y, z, headKind], (k + 1) * 4)
          positions.set([x, y + 1.2, z, carbonKind], (k + 2) * 4)
          k += 3
        }
      }
    }

    const heads: number[] = []
    for (let i = 0; i < N; i++) if (Math.round(positions[i * 4 + 3]) === headKind) heads.push(i)

    async function forcesFor(charges: Float32Array, acidSoap: number): Promise<Float32Array> {
      const sys = await api.createSoup({
        box,
        seed: 11,
        kT: 1.1,
        clay: false,
        start: { C: nCell, O: nCell * 2, H: 0, M: 0, W: 0 },
        electrostatics: { enabled: true, pH: 7 },
        acidSoapScaleOverride: acidSoap,
        resume: {
          globalStep: 0,
          liveBox: box,
          positions: positions.slice(),
          velocities: new Float32Array(N * 4),
          bondSlots: new Uint32Array(N * 3).fill(0xffffffff),
          centerLink: new Uint32Array(N).fill(0xffffffff),
          centerHeldSteps: new Uint32Array(N),
          desorbEvents: new Uint32Array(2),
          bondRng: Uint32Array.from({ length: N }, (_, i) => i + 1),
          thermoRng: Uint32Array.from({ length: N }, (_, i) => i + 1000003),
          charges,
          events: {},
        },
      })
      const f = await sys.forces()
      const fb = await sys.forcesBruteForce()
      let bruteDiff = 0
      for (let i = 0; i < f.length; i++) bruteDiff = Math.max(bruteDiff, Math.abs(f[i] - fb[i]))
      return Object.assign(f, { bruteDiff }) as any
    }

    const S = 4.0
    const allNeutral = new Float32Array(N)
    const allCharged = new Float32Array(N)
    for (const i of heads) allCharged[i] = qDeprot
    const alternating = new Float32Array(N)
    for (let h = 0; h < heads.length; h++) if (h % 2 === 1) alternating[heads[h]] = qDeprot

    const out: any = {}
    for (const [name, q] of [
      ['neutral', allNeutral],
      ['charged', allCharged],
      ['alternating', alternating],
    ] as const) {
      const f0 = await forcesFor(q, 0)
      const f1 = await forcesFor(q, S)
      let maxHead = 0
      let maxOther = 0
      let sumHeadAbs = 0
      for (let i = 0; i < N; i++) {
        for (let a = 0; a < 3; a++) {
          const d = Math.abs(f1[i * 4 + a] - f0[i * 4 + a])
          if (Math.round(positions[i * 4 + 3]) === headKind) {
            maxHead = Math.max(maxHead, d)
            sumHeadAbs += d
          } else maxOther = Math.max(maxOther, d)
        }
      }
      out[name] = {
        maxHeadDiff: maxHead,
        maxOtherDiff: maxOther,
        meanHeadDiff: sumHeadAbs / (heads.length * 3),
        bruteDiff0: (f0 as any).bruteDiff,
        bruteDiff1: (f1 as any).bruteDiff,
      }
    }
    // The magnitude the alternating arm MUST show, derived from the model's own numbers rather than
    // read off the run. This lattice gives every head exactly TWO unlike partners along x: the one in
    // its own cell at 1.2 sigma and the one in the adjacent cell at (s - 1.2) = 1.8 sigma, both inside
    // attr_dv's range rc + wc = 2.7225. The two pulls oppose, so the NET difference is
    // S * (dV/dr(1.8) - dV/dr(1.2)). Nothing else is in range: the y and z neighbours sit at s = 3.0,
    // beyond the attraction, and the term never touches a head-tail or tail-tail pair at all.
    const p = api.loadParams()
    const rc = api.wcaCutoff(p.sigma * p.beadSizes.tail_tail)
    const wc = p.attraction.wc
    // d/dr of -eps*cos^2(pi(r-rc)/(2wc)) = eps*pi/(2wc)*sin(pi(r-rc)/wc); 0 inside rc, 0 beyond rc+wc.
    const dvdr = (r: number): number =>
      r <= rc || r >= rc + wc ? 0 : ((p.epsilon * Math.PI) / (2 * wc)) * Math.sin((Math.PI * (r - rc)) / wc)
    const expectedNet = S * (dvdr(s - 1.2) - dvdr(1.2))
    return { ...out, expectedNet, near: dvdr(1.2), far: dvdr(s - 1.2), rc, S, headCount: heads.length, N }
  })
  expect(consoleWarnings, `браузер сообщил об ошибке/предупреждении GPU:\n${consoleWarnings.join('\n')}`).toEqual([])
  console.log(
    `ACID-SOAP-RULE N=${r.N} голов=${r.headCount} rc=${r.rc.toFixed(7)} S=${r.S} ожидаемая net|dF|=${r.expectedNet.toFixed(5)} (ближняя 1.2 -> ${r.near.toFixed(5)}, дальняя 1.8 -> ${r.far.toFixed(5)})\n` +
      `  ВСЕ НЕЙТРАЛЬНЫ:   max|dF| голова=${r.neutral.maxHeadDiff.toExponential(3)} прочие=${r.neutral.maxOtherDiff.toExponential(3)}\n` +
      `  ВСЕ ЗАРЯЖЕНЫ:     max|dF| голова=${r.charged.maxHeadDiff.toExponential(3)} прочие=${r.charged.maxOtherDiff.toExponential(3)}\n` +
      `  ЧЕРЕДУЮЩИЕСЯ:     max|dF| голова=${r.alternating.maxHeadDiff.toExponential(4)} среднее=${r.alternating.meanHeadDiff.toExponential(3)} прочие=${r.alternating.maxOtherDiff.toExponential(3)}\n` +
      `  сетка против перебора: нейтрали ${r.neutral.bruteDiff0.toExponential(2)}/${r.neutral.bruteDiff1.toExponential(2)} ` +
      `чередующиеся ${r.alternating.bruteDiff0.toExponential(2)}/${r.alternating.bruteDiff1.toExponential(2)}`,
  )

  // 1. SAME protonation state on every head -> the term is structurally absent. Exactly zero, not
  //    "small": the two runs differ only in a uniform that this configuration never reads.
  expect(r.neutral.maxHeadDiff).toBe(0)
  expect(r.neutral.maxOtherDiff).toBe(0)
  expect(r.charged.maxHeadDiff).toBe(0)
  expect(r.charged.maxOtherDiff).toBe(0)

  // 2. UNLIKE states -> the term fires, at exactly the magnitude this lattice's own geometry gives
  //    (two opposing unlike partners, 1.2 and 1.8 sigma). Relative, not "greater than zero": a term
  //    with the wrong depth, the wrong sign or the wrong well shape fails here.
  expect(r.alternating.maxHeadDiff).toBeCloseTo(r.expectedNet, 3)
  //    ...and it touches NOTHING but head pairs: every tail force is bit-identical.
  expect(r.alternating.maxOtherDiff).toBe(0)

  // 3. The term is applied IDENTICALLY on the grid/Verlet walk and on the O(N^2) reference -- the
  //    same brute-force gate tests/soup-forces.test.ts uses, evaluated with the term on.
  expect(r.alternating.bruteDiff1).toBeLessThan(1e-2)
  expect(r.neutral.bruteDiff1).toBeLessThan(1e-2)
})

// Task 'acid-soap-pairing' (2026-08-23): THE CPU TWIN. soup/src/soup-potential.ts is the energy the
// zero-tension area move's Metropolis criterion is built on, and this task's whole gate-1 measurement
// is an area move run WITH charge -- so the CPU energy must carry BOTH charge-reading terms (the
// screened Coulomb and the acid-soap pair) or the move would sample a different force field than the
// dynamics. Checked the way tests/soup-area-move.test.ts and tests/soup-electrostatics.test.ts check
// it: numerically differentiate soupPotential along a head's coordinate and compare against the GPU
// force, F = -grad U, on a RELAXED charged broth rather than on a lattice.
//
// The control that makes the number mean something: the SAME comparison with the acid-soap depth at 0
// on both sides. If the residual is the same size in both arms, the residual is this instrument's
// finite-difference floor and not the new term.
test('CPU-потенциал согласован с силой GPU при включённой кислотно-мыльной паре (F = -grad U)', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const soup = api.loadSoup()
    const params = api.loadParams()
    const S = 4.0

    // The SAME hand-built lattice the rule test above uses, resumed with ALTERNATING charges, then
    // stepped off the perfect lattice. A relaxed broth was tried first and rejected as an instrument:
    // at this engine's own dilute force-diagnostic composition NO head has an unlike head neighbour
    // inside the attraction range (head-head has no attraction in this model at all, so heads do not
    // cluster), and a finite-difference check on term-free particles proves nothing. 200 steps is
    // fewer than electrostatics.sweepEverySteps, so the resumed protonation state is still the one
    // measured against.
    const headKind = soup.monomers.findIndex((m: any) => m.kind === 'head')
    const carbonKind = soup.monomers.findIndex((m: any) => m.kind === 'carbon')
    const qDeprot = soup.electrostatics.chargeDeprotonated
    const box: [number, number, number] = [18, 18, 18]
    const n = 6
    const cell = box[0] / n
    const nCell = n * n * n
    const N = nCell * 3
    const positions = new Float32Array(N * 4)
    {
      let k = 0
      for (let iz = 0; iz < n; iz++) {
        for (let iy = 0; iy < n; iy++) {
          for (let ix = 0; ix < n; ix++) {
            const x = (ix + 0.5) * cell
            const y = (iy + 0.5) * cell
            const z = (iz + 0.5) * cell
            positions.set([x, y, z, headKind], k * 4)
            positions.set([x + 1.2, y, z, headKind], (k + 1) * 4)
            positions.set([x, y + 1.2, z, carbonKind], (k + 2) * 4)
            k += 3
          }
        }
      }
    }
    const charges0 = new Float32Array(N)
    {
      let h = 0
      for (let i = 0; i < N; i++) {
        if (Math.round(positions[i * 4 + 3]) !== headKind) continue
        if (h % 2 === 1) charges0[i] = qDeprot
        h++
      }
    }

    async function arm(acidSoap: number): Promise<any> {
      const sys = await api.createSoup({
        box,
        seed: 5,
        kT: 1.1,
        clay: false,
        start: { C: nCell, O: nCell * 2, H: 0, M: 0, W: 0 },
        electrostatics: { enabled: true, pH: 7 },
        acidSoapScaleOverride: acidSoap,
        resume: {
          globalStep: 0,
          liveBox: box,
          positions: positions.slice(),
          velocities: new Float32Array(N * 4),
          bondSlots: new Uint32Array(N * 3).fill(0xffffffff),
          centerLink: new Uint32Array(N).fill(0xffffffff),
          centerHeldSteps: new Uint32Array(N),
          desorbEvents: new Uint32Array(2),
          bondRng: Uint32Array.from({ length: N }, (_, i) => i + 1),
          thermoRng: Uint32Array.from({ length: N }, (_, i) => i + 1000003),
          charges: charges0.slice(),
          events: {},
        },
      })
      await sys.step(200)
      const pos = await sys.particles()
      const bonds = await sys.bonds()
      const links = await sys.centerLinks()
      const q = await sys.charges()
      const forces = await sys.forces()
      const es = sys.electrostatics()
      const basis = api.makePotentialBasis(soup, params, undefined, undefined, { enabled: true, pH: 7 }, acidSoap)
      // Pick heads that HAVE an unlike head neighbour in range -- the only particles the new term
      // touches. Without that filter most picks would be term-free and the check would prove nothing.
      const rcMax = api.wcaCutoff(params.sigma * params.beadSizes.tail_tail) + params.attraction.wc
      const heads: number[] = []
      for (let i = 0; i < pos.length / 4; i++) if (Math.round(pos[i * 4 + 3]) === headKind) heads.push(i)
      const picks: number[] = []
      for (const i of heads) {
        if (picks.length >= 6) break
        for (const j of heads) {
          if (j === i) continue
          let dx = pos[i * 4] - pos[j * 4]
          let dy = pos[i * 4 + 1] - pos[j * 4 + 1]
          let dz = pos[i * 4 + 2] - pos[j * 4 + 2]
          dx -= Math.round(dx / box[0]) * box[0]
          dy -= Math.round(dy / box[1]) * box[1]
          dz -= Math.round(dz / box[2]) * box[2]
          const rr = Math.sqrt(dx * dx + dy * dy + dz * dz)
          if (rr < rcMax && (q[i] === 0) !== (q[j] === 0)) {
            picks.push(i)
            break
          }
        }
      }
      const h = 2e-3
      let maxRel = 0
      let maxAbs = 0
      let scale = 0
      const rows: any[] = []
      for (const i of picks) {
        for (let axis = 0; axis < 3; axis++) {
          const saved = pos[i * 4 + axis]
          pos[i * 4 + axis] = saved + h
          const xUp = pos[i * 4 + axis]
          const up = api.soupPotential(pos, bonds, box, basis, links, q).total
          pos[i * 4 + axis] = saved - h
          const xDn = pos[i * 4 + axis]
          const dn = api.soupPotential(pos, bonds, box, basis, links, q).total
          pos[i * 4 + axis] = saved
          const num = -(up - dn) / (xUp - xDn)
          const ana = forces[i * 4 + axis]
          maxAbs = Math.max(maxAbs, Math.abs(num - ana))
          scale = Math.max(scale, Math.abs(ana))
          rows.push({ i, axis, num, ana })
        }
      }
      maxRel = scale > 0 ? maxAbs / scale : 0
      return { acidSoap: basis.acidSoap, picks: picks.length, maxAbs, maxRel, scale, esCutoff: es.cutoff, rows: rows.length }
    }

    return { on: await arm(S), off: await arm(0), S }
  })
  console.log(
    `ACID-SOAP-CPU-TWIN S=${r.S}\n` +
      `  ВКЛ:  acidSoap(basis)=${r.on.acidSoap.toFixed(4)} проб=${r.on.rows} max|числ - анал|=${r.on.maxAbs.toExponential(3)} ` +
      `относ=${r.on.maxRel.toExponential(3)} масштаб|F|=${r.on.scale.toFixed(4)}\n` +
      `  ВЫКЛ: acidSoap(basis)=${r.off.acidSoap.toFixed(4)} проб=${r.off.rows} max|числ - анал|=${r.off.maxAbs.toExponential(3)} ` +
      `относ=${r.off.maxRel.toExponential(3)} масштаб|F|=${r.off.scale.toFixed(4)}`,
  )
  expect(r.on.acidSoap).toBeCloseTo(r.S, 10)
  expect(r.off.acidSoap).toBe(0)
  expect(r.on.rows).toBeGreaterThan(0)
  // Same tolerance shape tests/soup-electrostatics.test.ts uses for its own finite-difference check:
  // a relative residual at the float32 finite-difference floor, and the charge-off/pair-off control
  // must sit at the same floor -- which is what proves the residual is the instrument, not the term.
  expect(r.on.maxRel).toBeLessThan(2e-2)
  expect(r.off.maxRel).toBeLessThan(2e-2)
})

// Task 'acid-soap-pairing' (2026-08-23): THE STRENGTH THIS TASK SETTLED ON, pinned -- so it cannot be
// nudged later without a test saying so, and so the MAPPING from the atomic measurement stays visible
// in a test rather than only in a data file's prose. Pure Node, no GPU.
//
// What is pinned and why each line matters:
//  - the file's depth normalises to exactly 2.0 in the same units every cell of the class table is in
//    (1.0 = the apolar-apolar MARTINI reference), which is the arm that measured area 1.1510 and
//    thickness 4.5517, both inside the literature corridors, on a settled area;
//  - the recorded UPPER BOUND is the atomic measurement, and the used value is a stated FRACTION of it
//    (0.1612) -- so if anyone ever raises the depth toward the bound, this line fails and points at
//    the sweep that showed the bound destroys the bilayer (thickness 1.0079 sigma against 4-6);
//  - the rank is D, and it must stay D: the value is a gate-constrained choice, not a measurement, and
//    the project's own rule makes any gate resting only on D unproven by definition;
//  - kcal <-> kJ is the actual conversion, not a rounded copy.
test('глубина кислотно-мыльной пары: файл несёт ровно ту силу, что прошла свип, и ранг D', () => {
  const soup = loadSoup()
  const params = loadParams()
  const as = soup.solvent.attractionScale!.acidSoapPair!
  const ref = soup.solvent.attractionScale!.pairEpsilon!.levels[soup.solvent.attractionScale!.pairEpsilon!.reference]
  const normalised = acidSoapScaleOf(soup)
  const bound = as.measuredUpperBoundKJ / ref.epsilonKJ
  console.log(
    `ACID-SOAP-DEPTH файл=${as.epsilonKJ} кДж/моль / эталон ${ref.epsilonKJ} = ${normalised.toFixed(6)} ` +
      `(потолок ${as.measuredKcalPerMol} ккал/моль = ${as.measuredUpperBoundKJ} кДж/моль = ${bound.toFixed(4)} => ` +
      `доля ${(as.epsilonKJ / as.measuredUpperBoundKJ).toFixed(4)}), ранг=${as.rank}, epsilon(ранг A)=${params.epsilon}`,
  )
  expect(normalised).toBeCloseTo(2.0, 10)
  expect(as.epsilonKJ).toBe(7.0)
  expect(as.rank).toBe('D')
  // The upper bound is the atomic number, converted exactly (1 kcal = 4.184 kJ).
  expect(as.measuredUpperBoundKJ).toBeCloseTo(as.measuredKcalPerMol * 4.184, 3)
  expect(bound).toBeCloseTo(12.4085, 3)
  expect(as.epsilonKJ / as.measuredUpperBoundKJ).toBeCloseTo(0.1612, 4)
  // The used depth must stay strictly BELOW the bound: the bound is an energy with no entropy in it.
  expect(normalised).toBeLessThan(bound)
  // The class table is untouched by this task: head-head as a SPECIES pair is still exactly 0, so the
  // whole head-head attraction in this model is the charge-assisted term and nothing else.
  expect(soup.solvent.attractionScale!.pairEpsilon!.levels.polarPolar.epsilonKJ).toBe(0)
  // And the reference pair's own multiplier is still exactly 1, i.e. tail-tail still carries the
  // rank-A Cooke & Deserno depth bit for bit (the invariant hydrophobic-asymmetry's basis section 4
  // states) -- adding a pair depth must not have rescaled anything.
  const table = attractionScaleTable(soup)
  expect(table[CLASS_APOLAR][CLASS_APOLAR]).toBe(1)
  expect(table[CLASS_POLAR][CLASS_POLAR]).toBe(0)
})
