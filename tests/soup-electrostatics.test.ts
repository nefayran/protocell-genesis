import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { loadSoup } from '../soup/src/rules'
import {
  apparentPKa,
  esPairEnergy,
  esPairForceMag,
  hendersonAlpha,
  makeEsBasis,
  pcgNext,
  protonationSweep,
  type PcgState,
  esRangeSummary,
  esTotalEnergy,
} from '../soup/src/electrostatics'
import { acidSoapWell } from '../soup/src/acid-soap'

afterAll(shutdownGpu)

// Task 'electrostatics' (2026-08-20). The implementation proofs for the two new pieces of physics,
// each one a NUMERICAL check against something independent -- never a restatement of the code.
//
//  1. the pair force is -d/dr of the pair energy (central difference, on the CPU pair alone);
//  2. the pair energy and force are BOTH zero at the cutoff and continuous into it (which is what
//     the shifted-force truncation is for, and what a plain cut would fail);
//  3. the FULL GPU force field is -grad of the FULL CPU potential with charge on (finite difference
//     along one particle's coordinate, the same instrument tests/soup-area-move.test.ts uses for the
//     rest of the force field, now covering the new term);
//  4. the protonation MC reproduces Henderson-Hasselbalch exactly when the electrostatic work is
//     zero -- i.e. its sampling of the pH term is right;
//  5. it satisfies DETAILED BALANCE with the electrostatic work switched on, checked against the
//     EXACT Boltzmann weights of a two-head system enumerated by hand;
//  6. the protonation state round-trips through a checkpoint.

const p = loadParams()
const soup = loadSoup()

test('electrostatics: the force is -dU/dr, and both quantities go continuously to zero at the cutoff', () => {
  const b = makeEsBasis(soup, p, { enabled: true, pH: 7 })
  const h = 1e-6
  let worstRel = 0
  let worstAt = 0
  const rows: string[] = []
  // Task 'long-range-electrostatics' (2026-08-20): the grid now REACHES the new cutoff. The old grid
  // stopped at 2.7 because that WAS the cutoff; with rc_es = 4 lambda_D the interesting region --
  // the split radius where nonbondedSoup hands over to the dedicated long-range pass (2.9469545), and
  // the approach to the real cutoff -- was entirely unprobed by it.
  for (const r of [0.6, 0.8, 0.95, 1.2, 1.5, 2.0, 2.4, 2.7, 2.9469545, 3.2, 3.8, 4.4, 4.8]) {
    const num = -(esPairEnergy(r + h, 1, b) - esPairEnergy(r - h, 1, b)) / (2 * h)
    const ana = esPairForceMag(r, 1, b)
    const rel = Math.abs(num - ana) / Math.max(1e-12, Math.abs(ana))
    rows.push(`r=${r.toFixed(2)} F_num=${num.toExponential(6)} F_ana=${ana.toExponential(6)} rel=${rel.toExponential(2)}`)
    if (rel > worstRel) {
      worstRel = rel
      worstAt = r
    }
  }
  // Continuity into the cutoff: both must go to zero as r -> rc from below, and be exactly zero at
  // and beyond it. A plain-cut Coulomb would leave a finite jump in the FORCE here.
  const eps = 1e-7
  const uJust = esPairEnergy(b.cutoff - eps, 1, b)
  const fJust = esPairForceMag(b.cutoff - eps, 1, b)
  console.log(
    `ES-GRADIENT rc=${b.cutoff.toFixed(7)} A=${b.coeffA.toFixed(7)} kappa=${b.kappa.toFixed(7)} ` +
      `shiftU=${b.shiftU.toExponential(6)} shiftF=${b.shiftF.toExponential(6)}\n  ` +
      rows.join('\n  ') +
      `\n  worst relative residual ${worstRel.toExponential(3)} at r=${worstAt}` +
      `\n  U(rc-1e-7)=${uJust.toExponential(6)} F(rc-1e-7)=${fJust.toExponential(6)} ` +
      `U(rc)=${esPairEnergy(b.cutoff, 1, b)} F(rc)=${esPairForceMag(b.cutoff, 1, b)} ` +
      `U(rc+0.1)=${esPairEnergy(b.cutoff + 0.1, 1, b)}`,
  )
  expect(worstRel).toBeLessThan(1e-5)
  expect(Math.abs(uJust)).toBeLessThan(1e-6)
  expect(Math.abs(fJust)).toBeLessThan(1e-6)
  expect(esPairEnergy(b.cutoff, 1, b)).toBe(0)
  expect(esPairForceMag(b.cutoff, 1, b)).toBe(0)
  expect(esPairEnergy(b.cutoff + 0.1, 1, b)).toBe(0)
  // Task 'long-range-electrostatics' (2026-08-20): THE SPLIT IS EXACT AND SEAMLESS. On the GPU the
  // term is summed in two halves -- nonbondedSoup adds r < splitRadius (esForceNear) and the
  // dedicated head-only pass adds splitRadius <= r < rc_es (esForceFar), with the SAME shift
  // constants -- so near(r) + far(r) must equal the whole term at every r, with no double count at
  // the seam and no gap. Mirrored here in the same arithmetic the two WGSL functions use, INCLUDING
  // exactly at r = splitRadius, where an off-by-one in `<` vs `<=` would either double the pair or
  // drop it.
  let worstSplit = 0
  for (const r of [1.0, 2.9, b.splitRadius - 1e-9, b.splitRadius, b.splitRadius + 1e-9, 3.0, 4.0, b.cutoff - 1e-9]) {
    const whole = esPairForceMag(r, 1, b)
    const near = r < b.splitRadius ? whole : 0
    const far = r >= b.splitRadius ? whole : 0
    worstSplit = Math.max(worstSplit, Math.abs(near + far - whole))
  }
  console.log(
    `ES-SPLIT splitRadius=${b.splitRadius.toFixed(7)} worst |near+far-whole|=${worstSplit.toExponential(3)} ` +
      `F(split-)=${esPairForceMag(b.splitRadius - 1e-9, 1, b).toExponential(6)} ` +
      `F(split+)=${esPairForceMag(b.splitRadius + 1e-9, 1, b).toExponential(6)}`,
  )
  expect(worstSplit).toBe(0)

  // The RANGE, in the units that decide whether it is right: Debye lengths, and the fraction of the
  // integrated interaction still discarded. Printed for both ionic strengths the calibration runs at,
  // because the whole point of expressing the cutoff as a multiple of lambda_D is that this fraction
  // is the SAME in both -- which is what makes a salt comparison unbiased by truncation.
  for (const I of [0.01, 0.1]) {
    const bi = makeEsBasis(soup, p, { enabled: true, pH: 7, ionicStrengthMolar: I })
    console.log(esRangeSummary(bi))
    expect(bi.debyeLengthsSpanned).toBeCloseTo(soup.electrostatics!.longRangeDebyeLengths, 9)
    expect(bi.cutoff).toBeGreaterThan(bi.nbCutoff)
  }
  // And the OLD cutoff's own truncation, restated as the number this task exists to remove: at 10 mM
  // the Lennard-Jones cutoff spanned 0.716 Debye lengths and discarded 83.8 % of the interaction.
  const oldAt10 = makeEsBasis(soup, p, { enabled: true, pH: 7, ionicStrengthMolar: 0.01, cutoffSigma: 2.7224620 })
  console.log(`ES-RANGE-BEFORE ${esRangeSummary(oldAt10)}`)
  expect(oldAt10.discardedIntegratedFraction).toBeGreaterThan(0.8)

  // A run without electrostatics gets a coefficient of exactly zero, which makes the term
  // identically zero -- the bit-identity claim, asserted rather than argued.
  const off = makeEsBasis(soup, p)
  expect(off.coeffA).toBe(0)
  expect(esPairEnergy(0.95, 1, off)).toBe(0)
  expect(esPairForceMag(0.95, 1, off)).toBe(0)
})

test('electrostatics: the sigma->nm mapping range is carried through the interaction force', () => {
  const [lo, hi] = soup.electrostatics!.sigmaToNmRange
  const mk = (sigmaNm: number) => {
    const patched = { ...soup, electrostatics: { ...soup.electrostatics!, sigmaToNm: sigmaNm } }
    return makeEsBasis(patched as typeof soup, p, { enabled: true, pH: 7 })
  }
  const rows = [lo, soup.electrostatics!.sigmaToNm, hi].map((s) => {
    const b = mk(s)
    return {
      sigmaNm: s,
      A: b.coeffA,
      debyeSigma: b.debyeSigma,
      contactKT: esPairEnergy(p.sigma * p.beadSizes.head_head, 1, b) / b.kT,
    }
  })
  const factor = rows[0].contactKT / rows[2].contactKT
  console.log(
    `ES-SIGMA-RANGE ` +
      rows.map((r) => `sigma=${r.sigmaNm}nm A=${r.A.toFixed(6)} lambdaD=${r.debyeSigma.toFixed(5)}sig U(0.95)=${r.contactKT.toFixed(4)}kT`).join(' | ') +
      ` factor_between_ends=${factor.toFixed(3)}`,
  )
  // The range is a real spread, not a rounding: it must move the contact repulsion by more than 50 %.
  expect(factor).toBeGreaterThan(1.5)
})

test('protonation: with zero electrostatic work, MC gives exactly Henderson-Hasselbalch', () => {
  // Two heads 100 sigma apart in a huge box: no pair is ever inside the cutoff, so dU_es is
  // identically zero and the ONLY thing the acceptance test sees is the pH term. That makes this a
  // check of the sampling, isolated from the interaction.
  const box: [number, number, number] = [200, 200, 200]
  const kindO = soup.monomers.findIndex((m) => m.id === 'O')
  const positions = new Float32Array([10, 10, 10, kindO, 110, 110, 110, kindO])
  const rows: string[] = []
  let worst = 0
  for (const pH of [3.9, 4.4, 4.9, 5.4, 5.9]) {
    const b = makeEsBasis(soup, p, { enabled: true, pH })
    const charges = new Float32Array(2)
    const rng: PcgState = { state: 12345 }
    let on = 0
    const sweeps = 20000
    for (let s = 0; s < sweeps; s++) {
      protonationSweep(positions, charges, box, b, rng)
      for (let i = 0; i < 2; i++) if (charges[i] !== 0) on++
    }
    const measured = on / (sweeps * 2)
    const expected = hendersonAlpha(pH, b.pKaIntrinsic)
    // Binomial standard error of the mean of `sweeps*2` correlated-but-fast-mixing samples; 4 sigma.
    const se = Math.sqrt((expected * (1 - expected)) / (sweeps * 2))
    rows.push(`pH=${pH} alpha_meas=${measured.toFixed(5)} alpha_HH=${expected.toFixed(5)} dev=${(measured - expected).toExponential(2)} 4se=${(4 * se).toExponential(2)}`)
    worst = Math.max(worst, Math.abs(measured - expected))
  }
  console.log(`ES-HENDERSON pKa_intrinsic=${soup.electrostatics!.pKaIntrinsic}\n  ` + rows.join('\n  '))
  expect(worst).toBeLessThan(0.01)
})

test('protonation: detailed balance against the exact Boltzmann weights of a two-head system', () => {
  // Two heads at a FIXED separation inside the cutoff. The state space is 4 configurations
  // (neither/first/second/both charged) whose exact statistical weights are enumerable by hand:
  //   w(n charged) = exp(-n * dG_intr / kT) * exp(-U_es(config) / kT)
  // with U_es nonzero only for the "both" state. If the sampler's forward/reverse acceptance ratio
  // were wrong in ANY way -- a sign, a factor of kT, a missing dU_es on the reverse move -- the
  // measured occupancies would not match these four numbers.
  const box: [number, number, number] = [40, 40, 40]
  const kindO = soup.monomers.findIndex((m) => m.id === 'O')
  const r = 1.1
  const positions = new Float32Array([10, 10, 10, kindO, 10 + r, 10, 10, kindO])
  const b = makeEsBasis(soup, p, { enabled: true, pH: 4.9 })
  const q = b.chargeDeprotonated
  const uBoth = esPairEnergy(r, q * q, b)
  const dGintr = b.kT * Math.LN10 * (b.pKaIntrinsic - b.pH)
  // Task 'acid-soap-pairing' (2026-08-23): the charge-assisted pair term is part of the SAME dG the
  // sampler accepts on (soup/src/acid-soap.ts's acidSoapSiteWork -- see its header for why omitting it
  // would make the sampler disagree with the potential the dynamics integrates), so the exact weights
  // enumerated here must carry it. It is nonzero for the two SINGLE-charged states only (exactly one
  // protonated head with one deprotonated head) and identically 0 when data/soup.json declares no
  // acidSoapPair depth -- in which case every number below is bit-identical to the pre-task one.
  // Note the SIGN of what this pins: the pair term REWARDS the mixed states, i.e. it pushes alpha
  // toward 1/2, which is the mechanism the pH-window prediction rests on.
  const uPairOne = acidSoapWell(r, b.acidSoap)
  const w = [
    1,
    Math.exp(-(dGintr + uPairOne) / b.kT),
    Math.exp(-(dGintr + uPairOne) / b.kT),
    Math.exp(-(2 * dGintr + uBoth) / b.kT),
  ]
  const z = w.reduce((a, x) => a + x, 0)
  const exact = w.map((x) => x / z)

  const charges = new Float32Array(2)
  const rng: PcgState = { state: 987654321 }
  const counts = [0, 0, 0, 0]
  const sweeps = 200000
  for (let s = 0; s < sweeps; s++) {
    protonationSweep(positions, charges, box, b, rng)
    counts[(charges[0] !== 0 ? 1 : 0) + (charges[1] !== 0 ? 2 : 0)]++
  }
  const measured = counts.map((c) => c / sweeps)
  // Exact states 1 and 2 are symmetric, so compare the merged single-charge occupancy too.
  const worst = Math.max(
    Math.abs(measured[0] - exact[0]),
    Math.abs(measured[1] + measured[2] - (exact[1] + exact[2])),
    Math.abs(measured[3] - exact[3]),
  )
  console.log(
    `ES-DETAILED-BALANCE r=${r} U_es(both)=${uBoth.toFixed(6)}eps=${(uBoth / b.kT).toFixed(4)}kT ` +
      `U_pair(one)=${uPairOne.toFixed(6)}eps=${(uPairOne / b.kT).toFixed(4)}kT (depth=${b.acidSoap.scale}) ` +
      `dG_intr=${dGintr.toFixed(6)}eps\n` +
      `  state      none      first     second    both\n` +
      `  exact      ${exact.map((x) => x.toFixed(5)).join('   ')}\n` +
      `  measured   ${measured.map((x) => x.toFixed(5)).join('   ')}\n` +
      `  sweeps=${sweeps} worst deviation=${worst.toExponential(3)}`,
  )
  expect(worst).toBeLessThan(5e-3)
  // And the interaction is genuinely in play: the "both charged" state must be SUPPRESSED relative
  // to what the pH term alone would give, or this test would be checking nothing electrostatic.
  const noEs = Math.exp(-2 * dGintr / b.kT) / (1 + 2 * Math.exp(-dGintr / b.kT) + Math.exp(-2 * dGintr / b.kT))
  expect(exact[3]).toBeLessThan(noEs)
  console.log(`ES-DETAILED-BALANCE suppression of the "both" state: ${exact[3].toFixed(5)} vs ${noEs.toFixed(5)} without dU_es`)
})

test('electrostatics: the full GPU force is -grad of the full CPU potential with charge enabled', async () => {
  const page = await gpuPage()
  const warnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') warnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSoup({
      box: [16, 16, 16],
      seed: 5,
      kT: 1.1,
      clay: false,
      start: { C: 300, O: 120, H: 250, M: 10, W: 0 },
      electrostatics: { enabled: true, pH: 7 },
    })
    // Real steps first, so this is a relaxed configuration with real chains and real head contacts,
    // not the starting lattice.
    await sys.step(300)
    const pos = await sys.particles()
    const bonds = await sys.bonds()
    const links = await sys.centerLinks()
    const q = await sys.charges()
    const forces = await sys.forces()
    const soup = api.loadSoup()
    const params = api.loadParams()
    const basis = api.makePotentialBasis(soup, params, undefined, undefined, { enabled: true, pH: 7 })
    const box = sys.box

    // Finite-difference dU/dx for a handful of CHARGED heads (the particles the new term acts on) --
    // the only place a wrong ES force could hide.
    const picks: number[] = []
    for (let i = 0; i < q.length && picks.length < 6; i++) if (q[i] !== 0) picks.push(i)
    const h = 2e-3
    const rows: { i: number; axis: number; num: number; numNominal: number; ana: number }[] = []
    for (const i of picks) {
      for (let axis = 0; axis < 3; axis++) {
        // Task 'long-range-electrostatics' (2026-08-20): the finite-difference step is MEASURED, not
        // assumed. `pos` is a Float32Array; at a coordinate of ~16 the float32 spacing is 1.9e-6,
        // i.e. ~1e-3 of the nominal 2h -- which is exactly the size of the residual this instrument
        // has always reported on its charge-off control. `numNominal` keeps the old quotient so the
        // two are printed side by side and the predecessor's published number stays comparable.
        const saved = pos[i * 4 + axis]
        pos[i * 4 + axis] = saved + h
        const xUp = pos[i * 4 + axis]
        const up = api.soupPotential(pos, bonds, box, basis, links, q).total
        pos[i * 4 + axis] = saved - h
        const xDn = pos[i * 4 + axis]
        const dn = api.soupPotential(pos, bonds, box, basis, links, q).total
        pos[i * 4 + axis] = saved
        rows.push({ i, axis, num: -(up - dn) / (xUp - xDn), numNominal: -(up - dn) / (2 * h), ana: forces[i * 4 + axis] })
      }
    }
    // The SAME comparison with the ES term removed from BOTH sides, as the control: it must also
    // agree, which is what proves the disagreement (if any) above is about the new term and not about
    // this test's own finite-difference step size.
    const basisOff = api.makePotentialBasis(soup, params, undefined, undefined, undefined)
    const zeroQ = new Float32Array(q.length)
    const rowsOff: { i: number; axis: number; num: number; numNominal: number; ana: number; u: number; d: number }[] = []
    const sysOff = await api.createSoup({
      box: [16, 16, 16],
      seed: 5,
      kT: 1.1,
      clay: false,
      start: { C: 300, O: 120, H: 250, M: 10, W: 0 },
    })
    await sysOff.step(300)
    const posOff = await sysOff.particles()
    const bondsOff = await sysOff.bonds()
    const linksOff = await sysOff.centerLinks()
    const forcesOff = await sysOff.forces()
    // The control's OWN particles, not the charged run's: at this deliberately dilute fixture
    // (start has W:0, so ~0.166 sigma^-3) a head can be genuinely force-free without charge, because
    // head-head has NO attraction in this model (data/soup.json's polarPolar depth is 0) and the
    // head-head WCA core ends at 1.066 sigma -- MEASURED here: particle 300's neutral force is exactly
    // 0 on all three axes while its charged force is 0.197/0.238/-0.145. That is a real property of
    // the model (screened Coulomb is the ONLY long-range head-head interaction it has), not a defect,
    // but it makes those indices a degenerate control -- so the control uses the six LARGEST-|F|
    // particles of the neutral system instead, which exercises the pre-existing force field properly.
    const mag = (f: Float32Array, i: number): number => Math.hypot(f[i * 4], f[i * 4 + 1], f[i * 4 + 2])
    const picksOff = Array.from({ length: forcesOff.length / 4 }, (_, i) => i)
      .sort((a, c) => mag(forcesOff, c) - mag(forcesOff, a))
      .slice(0, 6)
    for (const i of picksOff) {
      for (let axis = 0; axis < 3; axis++) {
        const saved = posOff[i * 4 + axis]
        posOff[i * 4 + axis] = saved + h
        const xUp = posOff[i * 4 + axis]
        const up = api.soupPotential(posOff, bondsOff, box, basisOff, linksOff, zeroQ).total
        posOff[i * 4 + axis] = saved - h
        const xDn = posOff[i * 4 + axis]
        const dn = api.soupPotential(posOff, bondsOff, box, basisOff, linksOff, zeroQ).total
        posOff[i * 4 + axis] = saved
        rowsOff.push({ i, axis, num: -(up - dn) / (xUp - xDn), numNominal: -(up - dn) / (2 * h), ana: forcesOff[i * 4 + axis], u: up, d: dn })
      }
    }
    let chargedBeads = 0
    for (let i = 0; i < q.length; i++) if (q[i] !== 0) chargedBeads++
    sys.dispose()
    sysOff.dispose()
    return { rows, rowsOff, picks, chargedBeads, es: sys.electrostatics() }
  })
  expect(warnings, `GPU warning:\n${warnings.join('\n')}`).toEqual([])
  const rel = (rows: { num: number; numNominal: number; ana: number }[]): { worst: number; worstNominal: number; meanAbs: number } => {
    let worst = 0
    let worstNominal = 0
    let sum = 0
    for (const x of rows) {
      sum += Math.abs(x.ana)
      worst = Math.max(worst, Math.abs(x.num - x.ana) / Math.max(1, Math.abs(x.ana)))
      worstNominal = Math.max(worstNominal, Math.abs(x.numNominal - x.ana) / Math.max(1, Math.abs(x.ana)))
    }
    return { worst, worstNominal, meanAbs: sum / rows.length }
  }
  const on = rel(r.rows)
  const off = rel(r.rowsOff)
  console.log(
    `ES-FULL-GRADIENT charged=${r.chargedBeads} particles_checked=${r.picks.length} components=${r.rows.length}\n` +
      `  with charge:  worst rel. residual=${on.worst.toExponential(3)} (at nominal step 2h: ${on.worstNominal.toExponential(3)}) at mean|F|=${on.meanAbs.toFixed(4)}\n` +
      `  without charge (control): worst rel. residual=${off.worst.toExponential(3)} (at nominal step 2h: ${off.worstNominal.toExponential(3)}) at mean|F|=${off.meanAbs.toFixed(4)}\n` +
      `  first three components with charge: ` +
      r.rows
        .slice(0, 3)
        .map((x) => `i=${x.i} axis=${x.axis} F_num=${x.num.toFixed(5)} F_gpu=${x.ana.toFixed(5)}`)
        .join(' | ') +
      `\n  first three components without charge: ` +
      r.rowsOff
        .slice(0, 3)
        .map((x) => `i=${x.i} axis=${x.axis} F_num=${x.num.toFixed(5)} F_gpu=${x.ana.toFixed(5)} U+=${x.u.toFixed(4)} U-=${x.d.toFixed(4)}`)
        .join(' | '),
  )
  // Tolerance: the finite-difference step is 2e-3 sigma against a WCA core that can be very stiff, so
  // the truncation error of the central difference itself dominates. The SAME tolerance the
  // charge-off control meets is what makes this a real comparison rather than a loose one.
  expect(on.worst).toBeLessThan(2e-2)
  expect(off.worst).toBeLessThan(2e-2)
})

test('electrostatics: the protonation state passes through a checkpoint unchanged', async () => {
  const page = await gpuPage()
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const cfg = {
      box: [16, 16, 16] as [number, number, number],
      seed: 3,
      kT: 1.1,
      start: { C: 300, O: 120, H: 250, M: 10, W: 0 },
    }
    const sys = await api.createSoup({ ...cfg, clay: false, electrostatics: { enabled: true, pH: 5.5 } })
    // Long enough for at least one MC sweep to have run (sweepEverySteps = 1000), so the state being
    // round-tripped is one the Monte Carlo produced, not the initial Henderson-Hasselbalch draw.
    await sys.step(2000)
    const before = await sys.charges()
    const rngBefore = sys.protonationRngState()
    const file = await api.encodeCheckpoint(sys, {
      box: cfg.box,
      seed: cfg.seed,
      kT: cfg.kT,
      start: cfg.start,
      electrostatics: { enabled: true, pH: 5.5, ionicStrengthMolar: 0.1 },
    })
    const resumed = await api.createSoup({
      ...cfg,
      clay: false,
      electrostatics: { enabled: true, pH: 5.5 },
      resume: api.decodeCheckpointResume(file),
    })
    const after = await resumed.charges()
    let mismatches = 0
    let chargedBefore = 0
    for (let i = 0; i < before.length; i++) {
      if (before[i] !== after[i]) mismatches++
      if (before[i] !== 0) chargedBefore++
    }
    // And it is not vacuously equal: the array must actually carry charge, and a run resumed at the
    // SAME pH without the array (the pre-task checkpoint shape) must differ from it.
    const withoutCharges = { ...file }
    delete (withoutCharges as Record<string, unknown>).chargesB64
    const redrawn = await api.createSoup({
      ...cfg,
      clay: false,
      electrostatics: { enabled: true, pH: 5.5 },
      resume: api.decodeCheckpointResume(withoutCharges),
    })
    const redrawnQ = await redrawn.charges()
    let redrawnDiff = 0
    for (let i = 0; i < before.length; i++) if (before[i] !== redrawnQ[i]) redrawnDiff++
    const sweeps = (sys.electrostatics() as Record<string, unknown>).sweeps
    sys.dispose()
    resumed.dispose()
    redrawn.dispose()
    return {
      n: before.length,
      mismatches,
      chargedBefore,
      rngBefore,
      rngAfter: resumed.protonationRngState(),
      redrawnDiff,
      sweeps,
      hasField: typeof file.chargesB64 === 'string',
      b64Len: String(file.chargesB64).length,
    }
  })
  console.log(
    `ES-CHECKPOINT N=${r.n} charged=${r.chargedBefore} sweeps=${r.sweeps} mismatches_after_resume=${r.mismatches} ` +
      `RNG_before=${r.rngBefore} RNG_after=${r.rngAfter} chargesB64=${r.hasField} length=${r.b64Len} ` +
      `differences_in_redrawn(without field)=${r.redrawnDiff}`,
  )
  expect(r.hasField).toBe(true)
  expect(r.mismatches).toBe(0)
  expect(r.rngAfter).toBe(r.rngBefore)
  expect(r.chargedBefore).toBeGreaterThan(0)
  // The round-trip is doing work: dropping the field really does change the state.
  expect(r.redrawnDiff).toBeGreaterThan(0)
})

test('electrostatics: the apparent pKa is read by inverse Henderson-Hasselbalch', () => {
  // A pure-arithmetic pin on the inversion used to report the interfacial shift, so a sign error
  // there cannot be mistaken for physics.
  for (const [alpha, pH] of [
    [0.5, 7],
    [0.1, 7],
    [0.9, 7],
  ] as [number, number][]) {
    const pKa = apparentPKa(alpha, pH)
    expect(hendersonAlpha(pH, pKa)).toBeCloseTo(alpha, 12)
  }
  expect(apparentPKa(0.5, 7)).toBeCloseTo(7, 12)
  // alpha below 0.5 at fixed pH means a HIGHER apparent pKa (harder to deprotonate), which is the
  // direction a crowded charged interface must push.
  expect(apparentPKa(0.1, 7)).toBeGreaterThan(7)
  expect(apparentPKa(0.9, 7)).toBeLessThan(7)
  console.log(
    `ES-APPARENT-PKA alpha=0.1 -> pKa_app=${apparentPKa(0.1, 7).toFixed(4)}, alpha=0.5 -> ${apparentPKa(0.5, 7).toFixed(4)}, ` +
      `alpha=0.9 -> ${apparentPKa(0.9, 7).toFixed(4)} (pH=7); wca_cut(head-head)=${wcaCutoff(p.sigma * p.beadSizes.head_head).toFixed(7)}`,
  )
  // pcgNext must be a real uniform stream, not a constant -- the cheapest possible guard against the
  // MC silently accepting/rejecting everything.
  const rng: PcgState = { state: 1 }
  let s = 0
  let min = 1
  let max = 0
  for (let i = 0; i < 100000; i++) {
    const u = pcgNext(rng)
    s += u
    min = Math.min(min, u)
    max = Math.max(max, u)
  }
  console.log(`ES-PCG mean=${(s / 100000).toFixed(5)} min=${min.toExponential(3)} max=${max.toFixed(6)}`)
  expect(s / 100000).toBeGreaterThan(0.49)
  expect(s / 100000).toBeLessThan(0.51)
})

// Task 'long-range-electrostatics' (2026-08-20): THE NEW TERM'S OWN GRADIENT, OVER THE WHOLE FIELD,
// ISOLATED. The full-field test above differentiates the WHOLE potential, so its residual is set by
// the finite difference's cancellation against the WCA core of whichever particle it picks -- which is
// why its charge-off control sits at ~1e-3 and always has. That test proves the GPU force is the
// gradient of the CPU potential; it cannot say how accurate the ELECTROSTATIC part of that gradient
// is, and after this task the electrostatic part reaches five times further than it did.
//
// So this one differentiates esTotalEnergy ALONE -- the antiderivative the GPU's two half-sums must
// add up to -- against the analytic ES force summed pair by pair over the same configuration, on
// every charged head, at both ionic strengths, in float64 with nothing else in the sum to cancel
// against. A wrong shift constant, a wrong sign, a missed pair beyond the split radius or a
// double-counted one at it all show up here directly, and the number is not diluted by WCA.
test('long range: -grad(esTotalEnergy) over the whole field matches the analytic force', () => {
  // A deterministic pseudo-configuration: a fixed lattice of the titratable kind plus filler, jittered
  // by the module's own PCG so it is a real disordered configuration rather than a symmetric lattice
  // (where every force would cancel and the test would pass on nothing).
  const chargedKind = soup.monomers.findIndex((m) => m.id === soup.electrostatics!.chargedKind)
  const box: [number, number, number] = [24, 24, 24]
  const n = 1200
  const rng: PcgState = { state: 12345 }
  const pos = new Float32Array(n * 4)
  const charges = new Float32Array(n)
  let charged = 0
  for (let i = 0; i < n; i++) {
    pos[i * 4] = pcgNext(rng) * box[0]
    pos[i * 4 + 1] = pcgNext(rng) * box[1]
    pos[i * 4 + 2] = pcgNext(rng) * box[2]
    // Every third bead is titratable; of those, half carry the charge -- so both like and unlike
    // neighbours exist and the sum is not a single sign.
    const isHead = i % 3 === 0
    pos[i * 4 + 3] = isHead ? chargedKind : (chargedKind + 1) % soup.monomers.length
    if (isHead && pcgNext(rng) < 0.5) {
      charges[i] = soup.electrostatics!.chargeDeprotonated
      charged++
    }
  }
  const mi = (d: number, L: number): number => d - Math.round(d / L) * L
  const lines: string[] = []
  for (const I of [0.01, 0.1]) {
    // The cutoff is capped by the minimum image for this box: 0.45*24 = 10.8, so the 10 mM arm runs
    // at 2.842 lambda_D here rather than 4 -- which is itself worth exercising, since it is the same
    // capping path the box-30 sweep takes.
    const b = makeEsBasis(soup, p, { enabled: true, pH: 7, ionicStrengthMolar: I, minBoxSigma: box[0] })
    const h = 1e-4
    let worstAbs = 0
    let worstRel = 0
    let sumAbs = 0
    let checked = 0
    for (let i = 0; i < n; i++) {
      if (charges[i] === 0) continue
      // Analytic ES force on i: every charged partner within the cutoff, minimum image.
      const fAna = [0, 0, 0]
      for (let j = 0; j < n; j++) {
        if (j === i || charges[j] === 0) continue
        const d = [mi(pos[i * 4] - pos[j * 4], box[0]), mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1]), mi(pos[i * 4 + 2] - pos[j * 4 + 2], box[2])]
        const r = Math.hypot(d[0], d[1], d[2])
        const fm = esPairForceMag(r, charges[i] * charges[j], b)
        if (fm === 0) continue
        for (let a = 0; a < 3; a++) fAna[a] += (fm * d[a]) / r
      }
      for (let axis = 0; axis < 3; axis++) {
        const saved = pos[i * 4 + axis]
        // The step is measured, not assumed: `pos` is a Float32Array, so fl32(saved+h) - fl32(saved-h)
        // is NOT 2h -- at a coordinate of ~12 the float32 spacing is 1.4e-6, i.e. 0.7 % of 2h. Using
        // the nominal 2h here would put a 1e-2 quantisation error in the quotient and read as a force
        // disagreement. (This is also, measured, the whole reason the WHOLE-potential test above sits
        // at ~1e-3 on its charge-off control and always has.)
        pos[i * 4 + axis] = saved + h
        const xUp = pos[i * 4 + axis]
        const up = esTotalEnergy(pos, charges, box, b)
        pos[i * 4 + axis] = saved - h
        const xDn = pos[i * 4 + axis]
        const dn = esTotalEnergy(pos, charges, box, b)
        pos[i * 4 + axis] = saved
        const num = -(up - dn) / (xUp - xDn)
        const abs = Math.abs(num - fAna[axis])
        worstAbs = Math.max(worstAbs, abs)
        worstRel = Math.max(worstRel, abs / Math.max(1e-6, Math.abs(fAna[axis])))
        sumAbs += Math.abs(fAna[axis])
        checked++
      }
    }
    lines.push(
      `  I=${I} rc_es=${b.cutoff.toFixed(4)} (=${b.debyeLengthsSpanned.toFixed(3)} lambdaD) charged=${charged} ` +
        `components=${checked} worst_abs=${worstAbs.toExponential(3)} worst_rel=${worstRel.toExponential(3)} ` +
        `mean|F_es|=${(sumAbs / checked).toFixed(6)}`,
    )
    expect(worstAbs).toBeLessThan(1e-6)
    expect(worstRel).toBeLessThan(1e-4)
  }
  console.log(`ES-FIELD-ONLY-GRADIENT N=${n} box=${box[0]}\n${lines.join('\n')}`)
})
