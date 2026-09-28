// THE RANGE CALIBRATION -- task 'long-range-electrostatics' (2026-08-20),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/long-range-electrostatics-report.md.
//
// WHAT THIS ANSWERS, and why it is a separate measurement from the campaign. The predecessor task
// ('electrostatics', same directory) cut the screened Coulomb at the engine's Lennard-Jones nonbonded
// cutoff, 2.7224620 sigma. At 10 mM that is 0.716 Debye lengths and discards exp(-x)*(1+x) = 83.8 % of
// the integrated interaction, against 33.9 % at 100 mM -- i.e. the truncation was WORST in exactly the
// arm that should show the larger electrostatic penalty. Its measured salt shift of the apparent pKa
// came out -0.124 between 10 and 100 mM against the literature's ~0.7, and it called that a
// truncation-limited lower bound. This file tests that claim directly and quantitatively.
//
// THE INSTRUMENT: a FROZEN CONFIGURATION and a swept cutoff. Positions from a real checkpoint of the
// predecessor's own campaign are held fixed, and only the protonation Monte Carlo is re-equilibrated,
// at each of several cutoffs and at both ionic strengths. Because the configuration is identical
// across the whole sweep, the difference between two rows is the RANGE and nothing else -- which no
// pair of live runs can claim, since a live run's structure responds to the force. The cost is that
// the configuration was generated at the OLD range and is not self-consistent with the new one; the
// self-consistent number comes from the live arms in the report's own sweep, and the two are reported
// side by side rather than one standing in for the other.
//
// Off-GPU: this is the pure CPU protonation MC (soup/src/electrostatics.ts) over decoded checkpoints,
// so it costs no GPU time at all. A vitest file rather than an `npx tsx` script for the same reason
// tests/electrostatics-audit.test.ts is one, and it SKIPS when its env var is unset.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { loadSoup } from '../soup/src/rules'
import { loadParams } from '../engine/src/params'
import {
  apparentPKa,
  esPairEnergy,
  esRangeSummary,
  headIndices,
  makeEsBasis,
  protonationSweep,
  type PcgState,
} from '../soup/src/electrostatics'

const CHECKPOINTS = (process.env.ES_CAL_CHECKPOINTS ?? '').split(/[\s,]+/).filter(Boolean)
const ARTIFACT = process.env.ES_CAL_ARTIFACT ?? ''
const CUTOFFS = (process.env.ES_CAL_CUTOFFS ?? '').split(/[\s,]+/).filter(Boolean).map(Number)
const SWEEPS = Number(process.env.ES_CAL_SWEEPS ?? 20)
const IONIC = (process.env.ES_CAL_IONIC ?? '0.01,0.1').split(/[\s,]+/).filter(Boolean).map(Number)
// The pH the calibration is read at. Defaults to the FILE's own, but the predecessor measured its salt
// shift at pH 6.0, so that is the value this task compares against and passes explicitly.
const PH = process.env.ES_CAL_PH !== undefined ? Number(process.env.ES_CAL_PH) : undefined

const p = loadParams()
const soup = loadSoup()
const es0 = soup.electrostatics!

test.skipIf(CHECKPOINTS.length === 0)('long-range calibration: the salt shift of pKa as a function of the cutoff', () => {
  const out: Record<string, unknown>[] = []
  for (const path of CHECKPOINTS) {
    const file = JSON.parse(readFileSync(path, 'utf8'))
    const rs = decodeCheckpointResume(file)
    const pos = rs.positions!
    const box: [number, number, number] = rs.liveBox
    const minBox = Math.min(...box)
    const imageCap = es0.longRangeMaxBoxFraction * minBox
    const heads = headIndices(pos, soup.monomers.findIndex((m) => m.id === es0.chargedKind))
    const rho = heads.length / (box[0] * box[1] * box[2])
    console.log(
      `ES-CAL-CHECKPOINT ${path.split('/').pop()} step=${file.globalStep} box=${box[0].toFixed(4)} ` +
        `N=${pos.length / 4} heads=${heads.length} rho_heads=${rho.toFixed(6)} image_cap=${imageCap.toFixed(4)}`,
    )
    // The cutoffs to sweep: the caller's list, plus ALWAYS the predecessor's own 2.7224620 as the
    // "before" row and ALWAYS the shipped rule (longRangeDebyeLengths * lambda_D, capped) as the
    // "after" row -- so the table cannot accidentally omit either end of the comparison.
    for (const I of IONIC) {
      const shipped = makeEsBasis(soup, p, { enabled: true, pH: PH ?? es0.pH, ionicStrengthMolar: I, minBoxSigma: minBox })
      const lam = shipped.debyeSigma
      const grid = Array.from(new Set([2.7224620, ...CUTOFFS.filter((c) => c <= imageCap), shipped.cutoff])).sort((a, b) => a - b)
      for (const rc of grid) {
        const b = makeEsBasis(soup, p, {
          enabled: true, pH: PH ?? es0.pH, ionicStrengthMolar: I, minBoxSigma: minBox, cutoffSigma: rc,
        })
        // Start from the checkpoint's OWN protonation state and re-equilibrate: the first half of the
        // sweeps is discarded as equilibration, alpha is averaged over the second half. The alpha
        // trajectory is printed so convergence is visible rather than assumed.
        const q = new Float32Array(rs.charges!)
        const rng: PcgState = { state: 777 }
        const traj: number[] = []
        const t0 = Date.now()
        for (let s = 0; s < SWEEPS; s++) traj.push(protonationSweep(pos, q, box, b, rng).alpha)
        const half = traj.slice(Math.floor(SWEEPS / 2))
        const alpha = half.reduce((a, c) => a + c, 0) / half.length
        const sd = Math.sqrt(half.reduce((a, c) => a + (c - alpha) ** 2, 0) / half.length)
        // The head-head CONTACT repulsion at this range: U_sf at the head-head WCA contact distance,
        // the number the report weighs against the 0.909 kT apolar (tail-tail) well.
        // Two radii, both reported. `bead` = sigma*beadSizes.head_head = 0.95, which is where the
        // PREDECESSOR measured its 0.318 kT and therefore the only number directly comparable to it;
        // `wca` = wca_cut of the same pair = 1.0663389, the separation at which the two heads' own
        // repulsive cores stop touching. Neither is chosen: both are rank-A beadSizes arithmetic.
        const bead = p.sigma * p.beadSizes.head_head
        const wca = bead * Math.pow(2, 1 / 6)
        const uContact = esPairEnergy(bead, es0.chargeDeprotonated ** 2, b) / b.kT
        const uWca = esPairEnergy(wca, es0.chargeDeprotonated ** 2, b) / b.kT
        const row = {
          file: path, step: file.globalStep, box: box[0], I, cutoff: rc,
          debyeLengths: rc / lam, discardedIntegrated: Math.exp(-rc / lam) * (1 + rc / lam),
          alpha, alphaSd: sd, pKaApp: apparentPKa(alpha, PH ?? es0.pH), uContactKT: uContact, uWcaContactKT: uWca, pH: PH ?? es0.pH,
          sweeps: SWEEPS, ms: Date.now() - t0, heads: heads.length,
        }
        out.push(row)
        console.log(
          `ES-CAL I=${I} rc=${rc.toFixed(4)} (=${(rc / lam).toFixed(3)} lambdaD, discarded=${row.discardedIntegrated.toFixed(4)}) ` +
            `alpha=${alpha.toFixed(5)}+-${sd.toFixed(5)} pKa_app=${row.pKaApp.toFixed(4)} U(0.95)=${uContact.toFixed(4)}kT U(1.0663)=${uWca.toFixed(4)}kT ` +
            `sweeps=${SWEEPS} ${row.ms}ms  trajectory=[${traj.map((x) => x.toFixed(4)).join(' ')}]`,
        )
      }
    }
    // THE HEADLINE: the salt shift pKa_app(10 mM) - pKa_app(100 mM) at each cutoff both salts share,
    // against the literature's ~-0.7. The comparison is only meaningful between rows at the same
    // NUMBER OF DEBYE LENGTHS (where the discarded fraction is equal in both arms) and, separately,
    // at the same absolute cutoff (which is what a shared-cutoff scheme forces on you).
    const rows = out.filter((r) => r.file === path)
    const shifts: string[] = []
    for (const nd of [0.716, 1, 2, 3, 4]) {
      const lo = rows.filter((r) => r.I === Math.min(...IONIC)).sort((a, c) => Math.abs((a.debyeLengths as number) - nd) - Math.abs((c.debyeLengths as number) - nd))[0]
      const hi = rows.filter((r) => r.I === Math.max(...IONIC)).sort((a, c) => Math.abs((a.debyeLengths as number) - nd) - Math.abs((c.debyeLengths as number) - nd))[0]
      if (!lo || !hi) continue
      if (Math.abs((lo.debyeLengths as number) - nd) > 0.35 || Math.abs((hi.debyeLengths as number) - nd) > 0.35) continue
      shifts.push(
        `  at ${nd} lambdaD: pKa(${lo.I})=${(lo.pKaApp as number).toFixed(4)} rc=${(lo.cutoff as number).toFixed(3)} | ` +
          `pKa(${hi.I})=${(hi.pKaApp as number).toFixed(4)} rc=${(hi.cutoff as number).toFixed(3)} | ` +
          `shift=${((lo.pKaApp as number) - (hi.pKaApp as number)).toFixed(4)} vs the literature ~-0.7`,
      )
    }
    // And the shared-cutoff comparison the predecessor was forced into, for the same configuration.
    const loShort = rows.find((r) => r.I === Math.min(...IONIC) && Math.abs((r.cutoff as number) - 2.7224620) < 1e-6)
    const hiShort = rows.find((r) => r.I === Math.max(...IONIC) && Math.abs((r.cutoff as number) - 2.7224620) < 1e-6)
    if (loShort && hiShort) {
      shifts.push(
        `  shared cutoff 2.7224620 (as in the predecessor): pKa(${loShort.I})=${(loShort.pKaApp as number).toFixed(4)} | ` +
          `pKa(${hiShort.I})=${(hiShort.pKaApp as number).toFixed(4)} | shift=${((loShort.pKaApp as number) - (hiShort.pKaApp as number)).toFixed(4)}`,
      )
    }
    console.log(`ES-CAL-SHIFT ${path.split('/').pop()}\n${shifts.join('\n')}`)
    for (const I of IONIC) console.log(esRangeSummary(makeEsBasis(soup, p, { enabled: true, pH: PH ?? es0.pH, ionicStrengthMolar: I, minBoxSigma: minBox })))
  }
  if (ARTIFACT) {
    mkdirSync(dirname(ARTIFACT), { recursive: true })
    writeFileSync(ARTIFACT, JSON.stringify({ rows: out, sweeps: SWEEPS, ionic: IONIC }, null, 1))
    expect(existsSync(ARTIFACT)).toBe(true)
  }
  expect(out.length).toBeGreaterThan(0)
})
