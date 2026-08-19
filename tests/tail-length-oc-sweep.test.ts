// O:C composition sweep against MEAN TAIL LENGTH, in the CORRECTED physics -- task
// 'tail-length-and-window' (2026-08-19),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/tail-length-and-window-report.md.
//
// Why this file exists at all: `broth-composition-report.md` §5 concluded "raising the head fraction
// changed no measurable quantity, so there is no measured basis for changing the historical
// O:C = 1:3". That verdict is VOID, and not because its arithmetic was wrong -- it was measured in a
// regime where `co_bond` fired ZERO times (its own §5 heading: "zero events, so mean tail length is
// UNDEFINED"), i.e. there were no amphiphiles at all whose tail length could depend on anything.
// `hydrophobic-asymmetry-report.md` (2026-08-19) then restored tail-tail dispersion
// (`solvent.attractionScale.pairEpsilon`'s apolarApolar level, MARTINI C1-C1) and both bilayer
// gates began to pass; `continuous-run-report.md` measured `co_bond` = 1027 real events. So the
// dependence is measurable NOW and was not measurable THEN.
//
// It works INSIDE the Flory/ASF framework the plan's Task 7 already fixed
// (docs/superpowers/plans/2026-08-16-soup-to-vesicle.md, `soup/src/equilibrium.ts`): every alpha
// below is either an EVENT-RATIO alpha (k_p/(k_p+k_t) read off the run's own `cc_bond`/`co_bond`
// counters) or a HISTOGRAM-RECOVERED alpha (`recoverAlphaFromChainLengths`, the log-linear slope of
// the measured length distribution), and the geometric ASF prediction it implies
// (`asfPrediction`, mean `1/(1-alpha)`) is printed BESIDE the measured histogram. Nothing here
// fits a rate to a target: `co_bond.attemptRate` is untouched (that is the move this project has
// refused three times), and the only thing that varies across the sweep is the number of head
// monomers in the starting composition.
//
// The analysis runs INSIDE the page (`window.api`, engine/src/index.ts's own exports -- the same
// `findAmphiphiles`/`amphiphileHistogram`/`carbonChainLengths`/`stageOf` the live run uses, so
// there is no second implementation of any measurement); only a small JSON summary crosses CDP.
// A whole particle array is NEVER transferred as JSON numbers -- that is the transfer pattern
// `checkpoint-resume-report.md` measured killing the page.
//
// Usage (every number comes from the environment, so one invocation can be chunked under the
// 600 s foreground cap):
//   SWEEP_RUNS='[{"O":500,"seed":19},{"O":1500,"seed":19}]' SWEEP_STEPS=150000 \
//   SWEEP_SAMPLE=25000 nice -n 15 npx vitest run tests/tail-length-oc-sweep.test.ts
// With SWEEP_RUNS unset the test SKIPS (it is a measurement harness, not a gate): a fresh clone
// must not spend 10 GPU-minutes on it.
import { appendFileSync, mkdirSync } from 'node:fs'
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { asfPrediction, recoverAlphaFromChainLengths } from '../soup/src/equilibrium'

afterAll(shutdownGpu)

interface RunSpec {
  O: number
  seed: number
  /** Optional per-run override of the rest of the composition (C/H/M/W) and of the box, so ONE
   * invocation can carry both a sweep point at the shipped organic:water split and a
   * different-density calibration point -- the 600 s foreground cap makes invocations, not runs,
   * the scarce resource. Absent => SWEEP_START / SWEEP_BOX for this run, exactly as before. */
  start?: Record<string, number>
  box?: number
}

/** The per-sample record the page hands back -- deliberately all scalars, small maps and a
 * three-element aggregate list. No array whose length scales with N ever crosses this boundary. */
interface Sample {
  step: number
  stage: string
  events: Record<string, number>
  amphiphileCount: number
  carbonInAmphiphiles: number
  totalCarbon: number
  amphiphileFraction: number
  /** Mean of Amphiphile.length -- TOTAL carbon per recognised amphiphile, summed over its (one or
   * two) tails. This is the project's own `meanTailLength` (tests/continuous-run-audit.test.ts). */
  meanAmphiphileCarbon: number
  /** Mean of Amphiphile.tailLengths' entries -- PER-TAIL length, which is the quantity the
   * "C12-C18 maps to two tail beads" mapping actually constrains. Different from the above
   * whenever any head carries two tails (headPlacement.chainCapacity = 2, the 'two-tails' task). */
  meanPerTail: number
  twoTailedHeads: number
  /** Histogram of Amphiphile.length, and of PER-TAIL length, and of raw C-C chain length. */
  lengthHistogram: Record<number, number>
  perTailHistogram: Record<number, number>
  chainHistogram: Record<number, number>
  ccBondFraction: number
  /** Mean measured C-C bonded separation, minimum-image -- the `l` of the packing parameter, taken
   * from the run rather than from the FENE/WCA parameters on paper. */
  meanCCBondLength: number
  ccBondSamples: number
  aggregateCount: number
  qualifyingAggregateCount: number
  amphiphileShareInQualifying: number
  largest: {
    amphiphileCount: number
    particleCount: number
    radiusOfGyration: number
    flatnessRatio: number
    inPlaneSymmetry: number
    radialHeadShells: string
    transverseHeadShells: number
    cavityVolume: number
  }[]
  invariants: { monomers: Record<string, number>; bonds: number; charge: number }
  nonFinite: number
  boxVolume: number
}

const OUT = 'verify/out/tail-length-oc-sweep.jsonl'

test('O:C sweep: mean tail length, alpha and aggregate shape in the corrected physics', async () => {
  const runsRaw = process.env.SWEEP_RUNS
  if (!runsRaw || runsRaw.trim().length === 0) {
    console.log('OC-SWEEP skipped: SWEEP_RUNS unset (measurement harness, not a gate)')
    return
  }
  const runs = JSON.parse(runsRaw) as RunSpec[]
  const box = Number(process.env.SWEEP_BOX ?? 30)
  const steps = Number(process.env.SWEEP_STEPS ?? 150000)
  const sample = Number(process.env.SWEEP_SAMPLE ?? 25000)
  const kT = Number(process.env.SWEEP_KT ?? 1.1)
  const baseStart = JSON.parse(process.env.SWEEP_START ?? '{"C":1500,"H":1500,"M":100,"W":10700}') as Record<string, number>
  mkdirSync('verify/out', { recursive: true })

  const page = await gpuPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))

  for (const spec of runs) {
    const start = { ...(spec.start ?? baseStart), O: spec.O }
    const boxThis = spec.box ?? box
    const t0 = Date.now()
    const samples = (await page.evaluate(
      async (startJson: string, boxN: number, seed: number, kTN: number, stepsN: number, sampleN: number) => {
        const api = (window as any).api
        const soup = api.loadSoup()
        const start = JSON.parse(startJson)
        // clay: false -- matches soup/cli/campaign.ts's own hard pin (line 264, its own written
        // reason: it must be able to resume pre-clay checkpoints), so this sweep and the window run
        // it feeds are the SAME configuration, and both are directly comparable to
        // continuous-run-report.md's own numbers.
        const sys = await api.createSoup({ box: [boxN, boxN, boxN], seed, kT: kTN, start, clay: false })
        const out: unknown[] = []
        try {
          let done = 0
          while (done < stepsN) {
            const chunk = Math.min(sampleN, stepsN - done)
            await sys.step(chunk)
            done += chunk

            const p = await sys.particles()
            const b = await sys.bonds()
            const N = p.length / 4
            const amph = api.findAmphiphiles(p, b, soup.monomers)
            const kindOf = (i: number) => soup.monomers[Math.round(p[i * 4 + 3])].kind

            // Per-tail statistics and the two histograms, plus the two-tailed head count.
            let carbonInAmph = 0
            let tailSum = 0
            let tailCount = 0
            let twoTailed = 0
            const perTailHist: Record<number, number> = {}
            for (const a of amph) {
              carbonInAmph += a.length
              if (a.tailLengths.length >= 2) twoTailed++
              for (const t of a.tailLengths) {
                tailSum += t
                tailCount++
                perTailHist[t] = (perTailHist[t] ?? 0) + 1
              }
            }
            let totalCarbon = 0
            for (let i = 0; i < N; i++) if (kindOf(i) === 'carbon') totalCarbon++

            // Raw C-C chain population (the Flory/ASF framework's own subject -- see
            // soup/src/equilibrium.ts's header for why the carbon-only subgraph, not the
            // head-capped tail, is what the geometric relation describes), as a histogram.
            const chains = api.carbonChainLengths(p, b, soup.monomers)
            const chainHist: Record<number, number> = {}
            for (const n of chains) chainHist[n] = (chainHist[n] ?? 0) + 1

            // Mean measured C-C bonded separation, minimum-image, over every C-C bond.
            const boxLive = sys.box
            let bondLenSum = 0
            let bondLenN = 0
            for (let k = 0; k < b.length; k += 2) {
              const i = b[k]
              const j = b[k + 1]
              if (kindOf(i) !== 'carbon' || kindOf(j) !== 'carbon') continue
              let d2 = 0
              for (let ax = 0; ax < 3; ax++) {
                const L = boxLive[ax]
                let d = p[i * 4 + ax] - p[j * 4 + ax]
                d -= L * Math.round(d / L)
                d2 += d * d
              }
              bondLenSum += Math.sqrt(d2)
              bondLenN++
            }

            let nonFinite = 0
            for (let k = 0; k < p.length; k++) if (!Number.isFinite(p[k])) nonFinite++

            const { stage, evidence } = await api.stageOf(sys)
            const aggs = evidence.aggregateAnalysis.aggregates as any[]
            const top = aggs
              .slice()
              .sort((x, y) => y.amphiphileCount - x.amphiphileCount)
              .slice(0, 3)
              .map((s) => ({
                amphiphileCount: s.amphiphileCount,
                particleCount: s.particleCount,
                radiusOfGyration: Number(s.radiusOfGyration.toFixed(3)),
                flatnessRatio: Number(s.flatnessRatio.toFixed(4)),
                inPlaneSymmetry: Number(s.inPlaneSymmetry.toFixed(4)),
                radialHeadShells: String(s.radialHeadShells),
                transverseHeadShells: s.transverseHeadShells,
                cavityVolume: s.cavityVolume,
              }))

            out.push({
              step: sys.steps,
              stage,
              events: await sys.events(),
              amphiphileCount: amph.length,
              carbonInAmphiphiles: carbonInAmph,
              totalCarbon,
              amphiphileFraction: totalCarbon > 0 ? carbonInAmph / totalCarbon : 0,
              meanAmphiphileCarbon: amph.length ? carbonInAmph / amph.length : 0,
              meanPerTail: tailCount ? tailSum / tailCount : 0,
              twoTailedHeads: twoTailed,
              lengthHistogram: api.amphiphileHistogram(amph),
              perTailHistogram: perTailHist,
              chainHistogram: chainHist,
              ccBondFraction: api.ccBondFraction(p, b, soup.monomers),
              meanCCBondLength: bondLenN ? bondLenSum / bondLenN : 0,
              ccBondSamples: bondLenN,
              aggregateCount: evidence.aggregateAnalysis.aggregateCount,
              qualifyingAggregateCount: evidence.aggregateAnalysis.qualifyingAggregateCount,
              amphiphileShareInQualifying: evidence.aggregateAnalysis.amphiphileShareInQualifying,
              largest: top,
              invariants: await sys.invariants(),
              nonFinite,
              boxVolume: boxLive[0] * boxLive[1] * boxLive[2],
            })
          }
        } finally {
          sys.dispose()
        }
        return out
      },
      JSON.stringify(start),
      boxThis,
      spec.seed,
      kT,
      steps,
      sample,
    )) as Sample[]
    const wallMs = Date.now() - t0

    for (const s of samples) {
      const cc = s.events.cc_bond ?? 0
      const co = s.events.co_bond ?? 0
      const alphaEvent = cc + co > 0 ? cc / (cc + co) : Number.NaN
      // Histogram-recovered alpha, from the RAW C-C chain population -- the same subject
      // soup/src/equilibrium.ts's own recoverAlphaFromChainLengths documents. Reconstructed from
      // the histogram (counts, not the 1500-entry array) so nothing large crossed CDP.
      const chainLengths: number[] = []
      for (const [n, c] of Object.entries(s.chainHistogram)) for (let k = 0; k < (c as number); k++) chainLengths.push(Number(n))
      let recovered: { alpha: number; r2: number } | null = null
      try {
        const r = recoverAlphaFromChainLengths(chainLengths)
        recovered = { alpha: r.alpha, r2: r.r2 }
      } catch {
        recovered = null
      }
      const asfEvent = Number.isFinite(alphaEvent) ? asfPrediction(alphaEvent, 24) : null
      const line = {
        O: spec.O,
        seed: spec.seed,
        box: boxThis,
        start,
        ...s,
        alphaEvent,
        asfMeanFromEventAlpha: Number.isFinite(alphaEvent) ? 1 / (1 - alphaEvent) : null,
        alphaRecovered: recovered?.alpha ?? null,
        alphaRecoveredR2: recovered?.r2 ?? null,
        asfMeanFromRecoveredAlpha: recovered ? 1 / (1 - recovered.alpha) : null,
        asfPredictionFromEventAlpha: asfEvent,
        amphiphileYieldPerCarbon: s.totalCarbon > 0 ? s.amphiphileCount / s.totalCarbon : 0,
        amphiphileNumberDensity: s.amphiphileCount / s.boxVolume,
        wallMs,
      }
      appendFileSync(OUT, JSON.stringify(line) + '\n')
      console.log(
        `OC-SWEEP O=${spec.O} seed=${spec.seed} box=${boxThis} start=${JSON.stringify(start)} step=${s.step} stage=${s.stage} ` +
          `cc=${cc} co=${co} alphaEvent=${alphaEvent.toFixed(4)} asfMean=${(1 / (1 - alphaEvent)).toFixed(3)} ` +
          `alphaRec=${recovered ? recovered.alpha.toFixed(4) : 'n/a'} r2=${recovered ? recovered.r2.toFixed(3) : 'n/a'} ` +
          `amph=${s.amphiphileCount} frac=${s.amphiphileFraction.toFixed(5)} ` +
          `meanCarbonPerAmph=${s.meanAmphiphileCarbon.toFixed(3)} meanPerTail=${s.meanPerTail.toFixed(3)} ` +
          `twoTailed=${s.twoTailedHeads} ccFrac=${s.ccBondFraction.toFixed(4)} ` +
          `rCC=${s.meanCCBondLength.toFixed(4)}(n=${s.ccBondSamples}) ` +
          `aggs=${s.aggregateCount} qual=${s.qualifyingAggregateCount} shareQ=${s.amphiphileShareInQualifying.toFixed(4)} ` +
          `yield=${(s.amphiphileCount / Math.max(1, s.totalCarbon)).toFixed(5)} ` +
          `nAmphDensity=${(s.amphiphileCount / s.boxVolume).toExponential(3)} ` +
          `L1=${JSON.stringify(s.largest[0] ?? null)} nonFinite=${s.nonFinite} wallMs=${wallMs}`,
      )
      console.log(`OC-SWEEP-HIST O=${spec.O} seed=${spec.seed} step=${s.step} perTail=${JSON.stringify(s.perTailHistogram)} amphLength=${JSON.stringify(s.lengthHistogram)} chain=${JSON.stringify(s.chainHistogram)}`)
      // Invariants asserted, not eyeballed -- the same set continuous-run-audit.test.ts checks.
      expect(s.nonFinite, `O=${spec.O} seed=${spec.seed} step=${s.step}: non-finite positions`).toBe(0)
      // invariants().monomers reports EVERY monomer id data/soup.json declares, including the ones
      // this composition asked for zero of (K, the clay platelet's own species, at count 0 with
      // clay: false) -- so the conservation check is "every requested count is exactly what was
      // requested, and nothing unrequested appeared with a nonzero count", not a whole-object equal.
      const requested: Record<string, number> = {}
      for (const [id, n] of Object.entries(s.invariants.monomers)) if (n > 0) requested[id] = n
      expect(requested, `O=${spec.O} seed=${spec.seed} step=${s.step}: monomer conservation`).toEqual(start)
      expect(s.invariants.charge, `O=${spec.O} seed=${spec.seed} step=${s.step}: charge`).toBe(0)
    }
  }
  console.log(`OC-SWEEP artifact appended: ${OUT}`)
}, 3_600_000)
