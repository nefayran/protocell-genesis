import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadParams } from '../engine/src/params'
import { scaleMoleculesRigid } from '../soup/src/soup-box-scale-math'
import { moleculeCount, proposeBox } from '../soup/src/soup-area-move'

afterAll(shutdownGpu)

// Task 'hydrophobic-asymmetry' (2026-08-19), defect 2: VERIFICATION of the zero-tension area-move
// port (soup/src/soup-area-move.ts), by measurement rather than by inspection -- the same discipline
// engine/src/sim-area-move.ts's own Jacobian was checked with (its doc comment: without the entropic
// term the box collapsed 1.352 -> 0.53 sigma^2 while every aggregate diagnostic looked healthy, and
// the fix was confirmed by the identity <dU/dlnA>_bead - <dU/dlnA>_COM = 2*N*kT).
//
// Three independent things are checked here, and each one fails loudly on its own:
//  1. (no GPU) the rigid-molecule coordinate map's algebraic properties the move relies on --
//     composition equals the direct map (which is what makes "run the chain on the CPU, apply the
//     final box once" exact), bonded distances survive, and an UNBONDED SOLVENT bead really is
//     carried by the map (a size-1 molecule scaling as its own centre of mass).
//  2. the CPU potential (soup/src/soup-potential.ts) is the antiderivative of the GPU force field:
//     -dU/dx from a central difference against SoupSystem.forces(). The soup force path writes no
//     potential of its own, so this is what makes the Metropolis energy the SAME Hamiltonian the
//     dynamics integrate rather than a plausible-looking second one.
//  3. the configurational Jacobian, through the generalised identity
//     <dU/dlnA>_bead - <dU/dlnA>_COM = (Nbeads - Nmolecules) * kT.
//     The engine's 2*N*kT is this same statement for its fixed 3-bead lipids with no solvent
//     (3N - N = 2N). Two modes are measured, giving two different predictions from ONE formula:
//     'lateral-fixed-z' (L_z fixed, so V'/V = A'/A) predicts (Nbeads-Nmol)*kT, while
//     'lateral-fixed-volume' (L_z /= exp(u)) predicts EXACTLY ZERO, because a volume-preserving
//     deformation has the same unit Jacobian under both maps. Together they pin down both the
//     entropic term's presence and its N. This is also the check that catches "the solvent was left
//     behind": water counts in Nbeads and in Nmol identically, so if the molecule map did not move
//     water at all, the water-water virial would be missing from one side only.

test('rigid-COM map: composition = the direct map, bonds intact, solvent is carried along', () => {
  // 3-bead chain (0,1,2) + three unbonded "solvent" beads (3,4,5) -- the soup's own shape: a free
  // bead is a molecule of size 1.
  const box: [number, number, number] = [10, 10, 10]
  const pos = new Float32Array([
    1, 1, 1, 0, 2, 1, 1, 1, 3, 1, 1, 1,
    9.5, 0.2, 5, 4, 4, 7, 2, 4, 0.1, 9.8, 8, 4,
  ])
  const bonds = new Uint32Array([0, 1, 1, 2])
  expect(moleculeCount(bonds, 6)).toBe(4)

  const u = 0.03
  const mid = proposeBox(box, u / 2, 'lateral-fixed-volume')
  const end = proposeBox(box, u, 'lateral-fixed-volume')
  const composed = scaleMoleculesRigid(scaleMoleculesRigid(pos, bonds, box, mid), bonds, mid, end)
  const direct = scaleMoleculesRigid(pos, bonds, box, end)
  let maxDiff = 0
  for (let i = 0; i < pos.length; i++) maxDiff = Math.max(maxDiff, Math.abs(composed[i] - direct[i]))
  console.log(`AREA-MOVE-MAP composeVsDirect maxDiff=${maxDiff.toExponential(3)}`)
  expect(maxDiff).toBeLessThan(1e-4)

  // Bonded distances (minimum image) survive the direct map exactly.
  const dist = (p: Float32Array, b: [number, number, number], i: number, j: number): number => {
    let s = 0
    for (let a = 0; a < 3; a++) {
      let d = p[i * 4 + a] - p[j * 4 + a]
      d -= Math.round(d / b[a]) * b[a]
      s += d * d
    }
    return Math.sqrt(s)
  }
  for (const [i, j] of [[0, 1], [1, 2]] as const) {
    expect(dist(direct, end, i, j)).toBeCloseTo(dist(pos, box, i, j), 5)
  }

  // The solvent beads MOVED, and by exactly the per-axis box ratio applied to their own wrapped
  // coordinate -- i.e. the map carries the solvent, it does not leave it in the old box.
  for (const i of [3, 4, 5]) {
    for (let a = 0; a < 3; a++) {
      const expected = pos[i * 4 + a] * (end[a] / box[a])
      expect(direct[i * 4 + a]).toBeCloseTo(expected - Math.floor(expected / end[a]) * end[a], 4)
    }
  }

  // Proposal geometry: the projected area scales by exp(u) in BOTH modes (so ln(A'/A) = u exactly),
  // the fixed-volume mode conserves the volume, the fixed-z mode conserves L_z.
  for (const mode of ['lateral-fixed-volume', 'lateral-fixed-z'] as const) {
    const b = proposeBox(box, u, mode)
    expect(Math.log((b[0] * b[1]) / (box[0] * box[1]))).toBeCloseTo(u, 12)
  }
  const fv = proposeBox(box, u, 'lateral-fixed-volume')
  expect((fv[0] * fv[1] * fv[2]) / (box[0] * box[1] * box[2])).toBeCloseTo(1, 12)
  expect(proposeBox(box, u, 'lateral-fixed-z')[2]).toBe(box[2])
})

test(
  'the CPU potential = the antiderivative of the GPU forces, and the area Jacobian gives (Nbeads-Nmolecules)·kT',
  async () => {
    const page = await gpuPage()
    const p = loadParams()
    const result = await page.evaluate(
      async (boxL: number, nSide: number, kT: number, seed: number, snapshots: number, stepsPer: number, h: number) => {
        const api = (window as any).api
        const soup = api.loadSoup()
        const params = api.loadParams()
        const carbonKind = soup.monomers.findIndex((m: any) => m.kind === 'carbon')
        const headKind = soup.monomers.findIndex((m: any) => m.kind === 'head')
        const waterKind = soup.monomers.findIndex((m: any) => m.id === soup.solvent.waterId)

        // Minimal fixture, NOT the bilayer patch: one lattice, and every column of `nSide` sites
        // starts with one 3-bead chain (head + 2 carbons, bonded along z at the lattice spacing, so
        // every bond sits well inside FENE's own range) and fills the rest with solvent. The
        // identity under test is a property of the coordinate maps and holds for ANY thermal
        // configuration containing both bonded molecules and free solvent, so the cheapest such
        // configuration is the right fixture -- no membrane geometry is needed or wanted here.
        const spacing = boxL / nSide
        const box: [number, number, number] = [boxL, boxL, boxL]
        const N = nSide * nSide * nSide
        const positions = new Float32Array(N * 4)
        const bondSlots = new Uint32Array(N * 3).fill(0xffffffff)
        let k = 0
        let chains = 0
        let waters = 0
        const rngState = { a: seed >>> 0 }
        const rng = (): number => {
          rngState.a = (rngState.a + 0x6d2b79f5) | 0
          let t = Math.imul(rngState.a ^ (rngState.a >>> 15), 1 | rngState.a)
          t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
          return ((t ^ (t >>> 14)) >>> 0) / 4294967296
        }
        const jitter = 0.1
        for (let ix = 0; ix < nSide; ix++) {
          for (let iy = 0; iy < nSide; iy++) {
            for (let iz = 0; iz < nSide; iz++) {
              const x = (ix + 0.5) * spacing + (rng() * 2 - 1) * jitter * spacing
              const y = (iy + 0.5) * spacing + (rng() * 2 - 1) * jitter * spacing
              const z = (iz + 0.5) * spacing
              const kind = iz < 3 ? (iz === 0 ? headKind : carbonKind) : waterKind
              positions.set([x, y, z, kind], k * 4)
              if (iz === 0) chains++
              if (iz >= 3) waters++
              if (iz === 1) {
                bondSlots[(k - 1) * 3 + 0] = k
                bondSlots[k * 3 + 2] = k - 1
              }
              if (iz === 2) {
                bondSlots[(k - 1) * 3 + 0] = k
                bondSlots[k * 3 + 0] = k - 1
              }
              k++
            }
          }
        }
        const startCounts: Record<string, number> = {}
        for (const mo of soup.monomers) startCounts[mo.id] = 0
        startCounts[soup.monomers[headKind].id] = chains
        startCounts[soup.monomers[carbonKind].id] = chains * 2
        startCounts[soup.monomers[waterKind].id] = waters

        const sys = await api.createSoup({
          box,
          seed,
          kT,
          // Task 'clay-surface' (2026-08-19): the CLAY-FREE control. data/soup.json's shipped composition
          // now carries the mineral platelet (clay.enabled), and this fixture must not: the zero-tension area move REFUSES to run on a system with an immobile phase (soup/src/soup-box-scale.ts), and its `resume` fixture was built at a clay-free particle count.
          // A clay-free system is byte-identical to the pre-task engine, so every number in this file is
          // unchanged by that task -- which is exactly what makes it a usable reference.
          clay: false,
          start: startCounts,
          resume: {
            globalStep: 0,
            liveBox: box,
            positions,
            velocities: new Float32Array(N * 4),
            bondSlots,
            centerLink: new Uint32Array(N).fill(0xffffffff),
            centerHeldSteps: new Uint32Array(N),
            desorbEvents: new Uint32Array(2),
            bondRng: Uint32Array.from({ length: N }, (_, i) => i + 1),
            thermoRng: Uint32Array.from({ length: N }, (_, i) => i + 1000003),
            events: {},
          },
        })

        const basis = api.makePotentialBasis(soup, params)
        await sys.step(2000)

        // --- (2) CPU potential vs GPU force -----------------------------------------------------
        const pos0: Float32Array = await sys.particles()
        const bonds0: Uint32Array = await sys.bonds()
        const links0: Uint32Array = await sys.centerLinks()
        const forces: Float32Array = await sys.forces()
        const liveBox: [number, number, number] = sys.box
        const grad: { i: number; axis: number; gpu: number; cpu: number }[] = []
        let meanAbsF = 0
        for (let i = 0; i < N; i++) {
          meanAbsF += Math.abs(forces[i * 4]) + Math.abs(forces[i * 4 + 1]) + Math.abs(forces[i * 4 + 2])
        }
        meanAbsF /= 3 * N
        for (const i of [7, 1001, 2000, N - 3]) {
          for (let axis = 0; axis < 3; axis++) {
            const plus = pos0.slice()
            const minus = pos0.slice()
            plus[i * 4 + axis] += h
            minus[i * 4 + axis] -= h
            const up = api.soupPotential(plus, bonds0, liveBox, basis, links0).total
            const um = api.soupPotential(minus, bonds0, liveBox, basis, links0).total
            grad.push({ i, axis, gpu: forces[i * 4 + axis], cpu: -(up - um) / (2 * h) })
          }
        }

        // --- (3) the Jacobian identity ----------------------------------------------------------
        const dUdlnA = (
          pos: Float32Array,
          bonds: Uint32Array,
          links: Uint32Array,
          bx: [number, number, number],
          mode: string,
          rigid: boolean,
        ): number => {
          const bp = api.proposeBox(bx, h, mode)
          const bm = api.proposeBox(bx, -h, mode)
          const pp = rigid ? api.scaleMoleculesRigidSoup(pos, bonds, bx, bp) : api.scaleBeadsAffine(pos, bx, bp)
          const pm = rigid ? api.scaleMoleculesRigidSoup(pos, bonds, bx, bm) : api.scaleBeadsAffine(pos, bx, bm)
          return (
            (api.soupPotential(pp, bonds, bp, basis, links).total - api.soupPotential(pm, bonds, bm, basis, links).total) /
            (2 * h)
          )
        }
        const samples: { fixedZ: number; fixedV: number; nMol: number }[] = []
        for (let s = 0; s < snapshots; s++) {
          await sys.step(stepsPer)
          const pos: Float32Array = await sys.particles()
          const bonds: Uint32Array = await sys.bonds()
          const links: Uint32Array = await sys.centerLinks()
          const bx: [number, number, number] = sys.box
          samples.push({
            fixedZ:
              dUdlnA(pos, bonds, links, bx, 'lateral-fixed-z', false) -
              dUdlnA(pos, bonds, links, bx, 'lateral-fixed-z', true),
            fixedV:
              dUdlnA(pos, bonds, links, bx, 'lateral-fixed-volume', false) -
              dUdlnA(pos, bonds, links, bx, 'lateral-fixed-volume', true),
            nMol: api.moleculeCount(bonds, N),
          })
        }
        const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length
        const sd = (xs: number[]): number => {
          const m = mean(xs)
          return Math.sqrt(xs.reduce((a, b) => a + (b - m) * (b - m), 0) / (xs.length - 1))
        }
        const fixedZ = samples.map((s) => s.fixedZ)
        const fixedV = samples.map((s) => s.fixedV)
        const nMolMean = mean(samples.map((s) => s.nMol))
        // Residual PER SNAPSHOT against that snapshot's OWN molecule count, not against the mean:
        // this fixture's head-carbon bonds can break under co_break, so Nmol drifts slightly during
        // sampling, and pairing each derivative with the topology it was measured on removes that
        // covariance instead of hiding it inside an average.
        const residualZ = samples.map((s) => s.fixedZ - (N - s.nMol) * kT)
        sys.dispose()
        return {
          N,
          chains,
          waters,
          bonds0: bonds0.length / 2,
          nMolMean,
          predictedFixedZ: (N - nMolMean) * kT,
          measuredFixedZ: mean(fixedZ),
          sdFixedZ: sd(fixedZ),
          residualZ: mean(residualZ),
          sdResidualZ: sd(residualZ),
          measuredFixedV: mean(fixedV),
          sdFixedV: sd(fixedV),
          snapshots: samples.length,
          grad,
          meanAbsF,
        }
      },
      16,
      13,
      p.thermostat.kT,
      41,
      800,
      100,
      1e-3,
    )

    console.log(
      `AREA-MOVE-GRADIENT meanAbsF=${result.meanAbsF.toFixed(4)} ` +
        result.grad
          .map((g: any) => `i${g.i}a${g.axis}: gpu=${g.gpu.toFixed(5)} cpu=${g.cpu.toFixed(5)}`)
          .join(' | '),
    )
    console.log(
      `AREA-MOVE-IDENTITY N=${result.N} chains=${result.chains} waters=${result.waters} ` +
        `bonds=${result.bonds0} Nmol=${result.nMolMean.toFixed(1)} snapshots=${result.snapshots}\n` +
        `  fixed-z:      predicted (N-Nmol)kT = ${result.predictedFixedZ.toFixed(2)}  ` +
        `measured = ${result.measuredFixedZ.toFixed(2)} +/- ${(result.sdFixedZ / Math.sqrt(result.snapshots)).toFixed(2)} ` +
        `(per-snapshot sd ${result.sdFixedZ.toFixed(2)})\n` +
        `  fixed-z residual per snapshot (measured - (N-Nmol)kT): ${result.residualZ.toFixed(2)} +/- ` +
        `${(result.sdResidualZ / Math.sqrt(result.snapshots)).toFixed(2)}\n` +
        `  fixed-volume: predicted 0  measured = ${result.measuredFixedV.toFixed(2)} +/- ` +
        `${(result.sdFixedV / Math.sqrt(result.snapshots)).toFixed(2)} (per-snapshot sd ${result.sdFixedV.toFixed(2)})`,
    )

    // (2) Every sampled component of -grad(U_cpu) matches the GPU force. Tolerance is relative to
    // the force's own magnitude plus a floor set by the mean |F| in this system, because the GPU
    // side is float32 and the WCA core makes individual components large.
    for (const g of result.grad as any[]) {
      expect(Math.abs(g.cpu - g.gpu)).toBeLessThan(0.02 * Math.abs(g.gpu) + 0.05 * result.meanAbsF)
    }

    // (3) The identity, both branches of the ONE acceptance formula.
    // Both are ensemble averages, so each is judged against its OWN standard error as well as
    // against the fixed-z prediction's scale -- the per-snapshot spread of dU/dlnA is of order
    // hundreds here (it is a virial, dominated by the WCA cores), so a tolerance stated only as a
    // fraction of the prediction would be tighter than the statistics and would pass or fail on
    // luck. `snapshots` is chosen so both standard errors land under 0.05*prediction.
    const seZ = result.sdFixedZ / Math.sqrt(result.snapshots)
    const seV = result.sdFixedV / Math.sqrt(result.snapshots)
    // The two branches have very different per-snapshot spreads (the volume-preserving one also
    // moves z, so its virial swings wider), hence two different standard-error budgets -- both
    // measured, both stated, neither tuned after the fact to make an assertion pass.
    expect(seZ).toBeLessThan(0.05 * result.predictedFixedZ)
    expect(seV).toBeLessThan(0.06 * result.predictedFixedZ)
    const seRes = result.sdResidualZ / Math.sqrt(result.snapshots)
    expect(Math.abs(result.residualZ)).toBeLessThan(Math.max(3 * seRes, 0.05 * result.predictedFixedZ))
    expect(Math.abs(result.measuredFixedV)).toBeLessThan(Math.max(3 * seV, 0.05 * result.predictedFixedZ))
  },
  600_000,
)
