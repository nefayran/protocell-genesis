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
} from '../soup/src/electrostatics'

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

test('электростатика: сила есть -dU/dr, и обе величины непрерывно обращаются в ноль на обрезке', () => {
  const b = makeEsBasis(soup, p, { enabled: true, pH: 7 })
  const h = 1e-6
  let worstRel = 0
  let worstAt = 0
  const rows: string[] = []
  for (const r of [0.6, 0.8, 0.95, 1.2, 1.5, 2.0, 2.4, 2.7]) {
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
      `\n  худшая относительная невязка ${worstRel.toExponential(3)} при r=${worstAt}` +
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
  // A run without electrostatics gets a coefficient of exactly zero, which makes the term
  // identically zero -- the bit-identity claim, asserted rather than argued.
  const off = makeEsBasis(soup, p)
  expect(off.coeffA).toBe(0)
  expect(esPairEnergy(0.95, 1, off)).toBe(0)
  expect(esPairForceMag(0.95, 1, off)).toBe(0)
})

test('электростатика: диапазон отображения sigma->нм несётся через силу взаимодействия', () => {
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
      ` фактор_между_концами=${factor.toFixed(3)}`,
  )
  // The range is a real spread, not a rounding: it must move the contact repulsion by more than 50 %.
  expect(factor).toBeGreaterThan(1.5)
})

test('протонирование: при нулевой электростатической работе MC даёт ровно Гендерсона-Хассельбальха', () => {
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
    rows.push(`pH=${pH} alpha_измер=${measured.toFixed(5)} alpha_HH=${expected.toFixed(5)} откл=${(measured - expected).toExponential(2)} 4se=${(4 * se).toExponential(2)}`)
    worst = Math.max(worst, Math.abs(measured - expected))
  }
  console.log(`ES-HENDERSON pKa_intrinsic=${soup.electrostatics!.pKaIntrinsic}\n  ` + rows.join('\n  '))
  expect(worst).toBeLessThan(0.01)
})

test('протонирование: детальный баланс против ТОЧНЫХ больцмановских весов системы из двух голов', () => {
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
  const w = [1, Math.exp(-dGintr / b.kT), Math.exp(-dGintr / b.kT), Math.exp(-(2 * dGintr + uBoth) / b.kT)]
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
    `ES-DETAILED-BALANCE r=${r} U_es(оба)=${uBoth.toFixed(6)}eps=${(uBoth / b.kT).toFixed(4)}kT dG_intr=${dGintr.toFixed(6)}eps\n` +
      `  состояние      ни одна      первая      вторая      обе\n` +
      `  точно      ${exact.map((x) => x.toFixed(5)).join('   ')}\n` +
      `  измерено   ${measured.map((x) => x.toFixed(5)).join('   ')}\n` +
      `  подметаний=${sweeps} худшее отклонение=${worst.toExponential(3)}`,
  )
  expect(worst).toBeLessThan(5e-3)
  // And the interaction is genuinely in play: the "both charged" state must be SUPPRESSED relative
  // to what the pH term alone would give, or this test would be checking nothing electrostatic.
  const noEs = Math.exp(-2 * dGintr / b.kT) / (1 + 2 * Math.exp(-dGintr / b.kT) + Math.exp(-2 * dGintr / b.kT))
  expect(exact[3]).toBeLessThan(noEs)
  console.log(`ES-DETAILED-BALANCE подавление состояния «обе»: ${exact[3].toFixed(5)} против ${noEs.toFixed(5)} без dU_es`)
})

test('электростатика: полная сила GPU есть -grad полного потенциала CPU при включённом заряде', async () => {
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
    const rows: { i: number; axis: number; num: number; ana: number }[] = []
    for (const i of picks) {
      for (let axis = 0; axis < 3; axis++) {
        const saved = pos[i * 4 + axis]
        pos[i * 4 + axis] = saved + h
        const up = api.soupPotential(pos, bonds, box, basis, links, q).total
        pos[i * 4 + axis] = saved - h
        const dn = api.soupPotential(pos, bonds, box, basis, links, q).total
        pos[i * 4 + axis] = saved
        rows.push({ i, axis, num: -(up - dn) / (2 * h), ana: forces[i * 4 + axis] })
      }
    }
    // The SAME comparison with the ES term removed from BOTH sides, as the control: it must also
    // agree, which is what proves the disagreement (if any) above is about the new term and not about
    // this test's own finite-difference step size.
    const basisOff = api.makePotentialBasis(soup, params, undefined, undefined, undefined)
    const zeroQ = new Float32Array(q.length)
    const rowsOff: { i: number; axis: number; num: number; ana: number; u: number; d: number }[] = []
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
        const up = api.soupPotential(posOff, bondsOff, box, basisOff, linksOff, zeroQ).total
        posOff[i * 4 + axis] = saved - h
        const dn = api.soupPotential(posOff, bondsOff, box, basisOff, linksOff, zeroQ).total
        posOff[i * 4 + axis] = saved
        rowsOff.push({ i, axis, num: -(up - dn) / (2 * h), ana: forcesOff[i * 4 + axis], u: up, d: dn })
      }
    }
    let chargedBeads = 0
    for (let i = 0; i < q.length; i++) if (q[i] !== 0) chargedBeads++
    sys.dispose()
    sysOff.dispose()
    return { rows, rowsOff, picks, chargedBeads, es: sys.electrostatics() }
  })
  expect(warnings, `GPU-предупреждение:\n${warnings.join('\n')}`).toEqual([])
  const rel = (rows: { num: number; ana: number }[]): { worst: number; meanAbs: number } => {
    let worst = 0
    let sum = 0
    for (const x of rows) {
      sum += Math.abs(x.ana)
      worst = Math.max(worst, Math.abs(x.num - x.ana) / Math.max(1, Math.abs(x.ana)))
    }
    return { worst, meanAbs: sum / rows.length }
  }
  const on = rel(r.rows)
  const off = rel(r.rowsOff)
  console.log(
    `ES-FULL-GRADIENT заряженных=${r.chargedBeads} проверено_частиц=${r.picks.length} компонент=${r.rows.length}\n` +
      `  С ЗАРЯДОМ:  худшая отн. невязка=${on.worst.toExponential(3)} при mean|F|=${on.meanAbs.toFixed(4)}\n` +
      `  БЕЗ ЗАРЯДА (контроль): худшая отн. невязка=${off.worst.toExponential(3)} при mean|F|=${off.meanAbs.toFixed(4)}\n` +
      `  первые три компоненты с зарядом: ` +
      r.rows
        .slice(0, 3)
        .map((x) => `i=${x.i} ось=${x.axis} F_num=${x.num.toFixed(5)} F_gpu=${x.ana.toFixed(5)}`)
        .join(' | ') +
      `\n  первые три компоненты БЕЗ заряда: ` +
      r.rowsOff
        .slice(0, 3)
        .map((x) => `i=${x.i} ось=${x.axis} F_num=${x.num.toFixed(5)} F_gpu=${x.ana.toFixed(5)} U+=${x.u.toFixed(4)} U-=${x.d.toFixed(4)}`)
        .join(' | '),
  )
  // Tolerance: the finite-difference step is 2e-3 sigma against a WCA core that can be very stiff, so
  // the truncation error of the central difference itself dominates. The SAME tolerance the
  // charge-off control meets is what makes this a real comparison rather than a loose one.
  expect(on.worst).toBeLessThan(2e-2)
  expect(off.worst).toBeLessThan(2e-2)
})

test('электростатика: состояние протонирования проходит контрольную точку без изменений', async () => {
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
    `ES-CHECKPOINT N=${r.n} заряженных=${r.chargedBefore} подметаний=${r.sweeps} несовпадений_после_резюме=${r.mismatches} ` +
      `RNG_до=${r.rngBefore} RNG_после=${r.rngAfter} chargesB64=${r.hasField} длина=${r.b64Len} ` +
      `отличий_у_перерисованного(без поля)=${r.redrawnDiff}`,
  )
  expect(r.hasField).toBe(true)
  expect(r.mismatches).toBe(0)
  expect(r.rngAfter).toBe(r.rngBefore)
  expect(r.chargedBefore).toBeGreaterThan(0)
  // The round-trip is doing work: dropping the field really does change the state.
  expect(r.redrawnDiff).toBeGreaterThan(0)
})

test('электростатика: кажущаяся pKa читается обратным Гендерсоном-Хассельбальхом', () => {
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
