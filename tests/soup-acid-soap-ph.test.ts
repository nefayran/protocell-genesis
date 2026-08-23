import { mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Task 'acid-soap-pairing' (2026-08-23): GATE 3 -- THE FALSIFIABLE PREDICTION.
//
// With a charge-assisted head-head pair the model should show a pH WINDOW: bilayer-competent
// aggregates near its own apparent pKa (where half the heads are deprotonated, so every acid has a
// soap to pair with and each pair behaves as one double-tailed amphiphile) and NOT far from it. This
// file measures where such objects appear and where they do not, across and outside the pH range.
//
// THE INSTRUMENT IS SELF-ASSEMBLY, NOT A PREBUILT SHEET. A prebuilt bilayer patch is not a usable pH
// instrument in this model: a single-tailed amphiphile with a tail-tail attraction holds a patch
// together at ANY pH, because nothing in the model makes it prefer a micelle -- so "the patch
// survived" would be true everywhere and would discriminate nothing. So the arms start from
// DISPERSED prebuilt amphiphiles (no chemistry, no evaporation, nothing to grow) in explicit liquid
// water, and the question asked of the settled state is the project's own bilayer question:
// analyzeAggregates' lamellar reading -- flat (flatnessRatio <= lamellarFlatnessRatio), extended
// roughly equally in its own plane, and TWO head layers -- on a FINITE object.
//
// THE SIZE IS DELIBERATE. 300 amphiphiles at rho_amph = 1.37e-2 sigma^-3 is exactly the density the
// project's own two-tailed campaign ran at (verdict document section 9: "rho_amph = 1.34e-2, the
// current run"), and the object it coalesces into is the size regime this project has already measured,
// and
// measured as a ROD: coalescence-report.md's own object is 111-155 amphiphiles with flatness 0.757,
// identified (supply-window-report.md) as a cylindrical micelle of radius t/2. So the comparison is
// apples to apples: at the same size and the same density, does charge-assisted pairing turn that rod
// into a bilayer-competent disc, and at which pH? It is NOT a closure experiment -- 100 amphiphiles
// is far below the closure floor and no arm here can or should close anything. A FIRST attempt at 100
// amphiphiles in a box of 24 was rejected as an instrument by its own output and is reported as such:
// the material stayed in 32 aggregates of ~22 amphiphiles each, too few heads per bin for the
// two-layer reading to be taken at all (radialHeadShells came back "unavailable"), so that run could
// not answer the question either way.
//
// FINITENESS is read as the radius of gyration against L/2, which is the Rg a UNIFORMLY FILLED box of
// side L would have (Rg^2 = 3 * L^2/12): the same reading the project's verdict document uses when it
// says "37.43 sigma against 38.000 for a uniformly filled box of 76 (98.5 %)". A percolating network
// sits near 1; a compact object sits well below.

const ARTIFACT = process.env.ACID_SOAP_PH_ARTIFACT ?? 'verify/out/acid-soap-ph-sweep.json'
const PH_LIST = (process.env.ACID_SOAP_PH_LIST ?? '5,6,7')
  .split(',')
  .map((s) => Number(s.trim()))
  .filter((s) => Number.isFinite(s))
const STRENGTH = Number(process.env.ACID_SOAP_STRENGTH ?? '2')
const STEPS = Number(process.env.ACID_SOAP_PH_STEPS ?? '40000')
const SAMPLE_EVERY = Number(process.env.ACID_SOAP_PH_SAMPLE ?? '8000')
const N_LIPIDS = Number(process.env.ACID_SOAP_PH_LIPIDS ?? '300')
const BOX = Number(process.env.ACID_SOAP_PH_BOX ?? '28')

test(
  'кислотно-мыльная пара: свип pH -- где появляются КОНЕЧНЫЕ бислой-способные объекты, а где нет',
  async () => {
    const page = await gpuPage()
    const rows: any[] = []
    for (const pH of PH_LIST) {
      const result = await page.evaluate(
        async (nLipids: number, boxL: number, waterDensity: number, pHv: number, acidSoap: number, steps: number, sampleEvery: number) => {
          const api = (window as any).api
          const soup = api.loadSoup()
          const p = api.loadParams()
          const kT = p.thermostat.kT
          const carbonKind = soup.monomers.findIndex((m: any) => m.kind === 'carbon')
          const headKind = soup.monomers.findIndex((m: any) => m.kind === 'head')
          const waterKind = soup.monomers.findIndex((m: any) => m.id === soup.solvent.waterId)
          const carbonR = soup.monomers[carbonKind].radiusSigma
          const headR = soup.monomers[headKind].radiusSigma
          const bTT = (p.sigma * (carbonR + carbonR)) / 2
          const bHT = (p.sigma * (headR + carbonR)) / 2

          const box: [number, number, number] = [boxL, boxL, boxL]
          const rngState = { a: 19 >>> 0 }
          const rng = (): number => {
            rngState.a = (rngState.a + 0x6d2b79f5) | 0
            let t = Math.imul(rngState.a ^ (rngState.a >>> 15), 1 | rngState.a)
            t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
            return ((t ^ (t >>> 14)) >>> 0) / 4294967296
          }

          // Amphiphiles DISPERSED on a coarse lattice with random orientations -- nothing preassembled,
          // no sheet, no micelle. Seed 19, the seed every campaign in this project used.
          const nSide = Math.ceil(Math.cbrt(nLipids))
          const cell = boxL / nSide
          const N = nLipids * 3 + Math.ceil(waterDensity * boxL ** 3)
          const positions = new Float32Array(N * 4)
          const bondSlots = new Uint32Array(N * 3).fill(0xffffffff)
          let k = 0
          let placed = 0
          for (let iz = 0; iz < nSide && placed < nLipids; iz++) {
            for (let iy = 0; iy < nSide && placed < nLipids; iy++) {
              for (let ix = 0; ix < nSide && placed < nLipids; ix++) {
                const cx = (ix + 0.5) * cell
                const cy = (iy + 0.5) * cell
                const cz = (iz + 0.5) * cell
                // Random unit direction for the 3-bead rod.
                const u = rng() * 2 - 1
                const phi = rng() * 2 * Math.PI
                const s = Math.sqrt(Math.max(0, 1 - u * u))
                const dir = [s * Math.cos(phi), s * Math.sin(phi), u]
                const wrap = (v: number): number => ((v % boxL) + boxL) % boxL
                const head = k
                const t1 = k + 1
                const t2 = k + 2
                // CENTRED on the lattice site, not hung off it. A first version placed the tail-end
                // bead ON the site and grew the rod outward, so two rods in adjacent cells could point
                // at each other and leave a gap of cell - 2*(bHT + bTT) = 0.05 sigma at cell = 4.0 --
                // a hard overlap, and the run diverged on its first chunk (the non-finite guard threw
                // by name at step 1000, 51294 of 51366 components). Centring halves each rod's reach,
                // so the worst-case gap is cell - (bHT + bTT) = 2.03 sigma.
                const half = (bHT + bTT) / 2
                positions.set([wrap(cx + dir[0] * half), wrap(cy + dir[1] * half), wrap(cz + dir[2] * half), headKind], head * 4)
                positions.set([wrap(cx + dir[0] * (bTT - half)), wrap(cy + dir[1] * (bTT - half)), wrap(cz + dir[2] * (bTT - half)), carbonKind], t1 * 4)
                positions.set([wrap(cx - dir[0] * half), wrap(cy - dir[1] * half), wrap(cz - dir[2] * half), carbonKind], t2 * 4)
                bondSlots[head * 3 + 0] = t1
                bondSlots[t1 * 3 + 2] = head
                bondSlots[t1 * 3 + 0] = t2
                bondSlots[t2 * 3 + 0] = t1
                k += 3
                placed++
              }
            }
          }
          // Water on its own lattice, skipping sites that land on an amphiphile bead.
          const spacing = Math.cbrt(1 / waterDensity)
          const wn = Math.max(1, Math.round(boxL / spacing))
          const sw = boxL / wn
          let waterPlaced = 0
          const minSep = 0.8 * p.sigma
          const minSep2 = minSep * minSep
          for (let iz = 0; iz < wn && k < N; iz++) {
            for (let iy = 0; iy < wn && k < N; iy++) {
              for (let ix = 0; ix < wn && k < N; ix++) {
                const x = ((ix + 0.5) * sw + (rng() * 2 - 1) * 0.15 * sw + boxL) % boxL
                const y = ((iy + 0.5) * sw + (rng() * 2 - 1) * 0.15 * sw + boxL) % boxL
                const z = ((iz + 0.5) * sw + (rng() * 2 - 1) * 0.15 * sw + boxL) % boxL
                let clash = false
                for (let a = 0; a < nLipids * 3; a++) {
                  let dx = x - positions[a * 4]
                  let dy = y - positions[a * 4 + 1]
                  let dz = z - positions[a * 4 + 2]
                  dx -= Math.round(dx / boxL) * boxL
                  dy -= Math.round(dy / boxL) * boxL
                  dz -= Math.round(dz / boxL) * boxL
                  if (dx * dx + dy * dy + dz * dz < minSep2) {
                    clash = true
                    break
                  }
                }
                if (clash) continue
                positions.set([x, y, z, waterKind], k * 4)
                k++
                waterPlaced++
              }
            }
          }
          const actualN = k
          const startCounts: Record<string, number> = {
            [soup.monomers[headKind].id]: nLipids,
            [soup.monomers[carbonKind].id]: nLipids * 2,
            [soup.monomers[waterKind].id]: waterPlaced,
          }
          for (const mo of soup.monomers) if (!(mo.id in startCounts)) startCounts[mo.id] = 0

          const sys = await api.createSoup({
            box,
            seed: 19,
            kT,
            clay: false,
            start: startCounts,
            // The long-range head list is sized from the head density in a stated volume. Left at the
            // box, it is derived from the DISPERSED density and overflows once the heads condense into
            // aggregates -- which this engine catches LOUDLY (it threw by name at pH 7, capacity 64)
            // rather than dropping interaction. Answered by naming the volume the heads occupy when the
            // material is FULLY CONDENSED (the close-packed bead volume of every amphiphile bead),
            // which makes the derived capacity saturate at maxHeads: every head is in every head's
            // list, so no interaction can be dropped at ANY pH and every arm shares one basis.
            electrostatics: {
              enabled: true,
              pH: pHv,
              densityVolumeSigma3: nLipids * 3 * (Math.PI / 6) * p.sigma ** 3,
            },
            acidSoapScaleOverride: acidSoap,
            resume: {
              globalStep: 0,
              liveBox: box,
              positions: positions.slice(0, actualN * 4),
              velocities: new Float32Array(actualN * 4),
              bondSlots: bondSlots.slice(0, actualN * 3),
              centerLink: new Uint32Array(actualN).fill(0xffffffff),
              centerHeldSteps: new Uint32Array(actualN),
              desorbEvents: new Uint32Array(2),
              bondRng: Uint32Array.from({ length: actualN }, (_, i) => i + 1),
              thermoRng: Uint32Array.from({ length: actualN }, (_, i) => i + 1000003),
              events: {},
            },
          })
          const contact = api.wcaCutoff(p.sigma * p.beadSizes.head_head)
          const thresholds = api.loadStageThresholds()

          async function sample(step: number): Promise<any> {
            const { evidence: ev } = await api.stageOf(sys)
            const pos: Float32Array = await sys.particles()
            const charges: Float32Array = await sys.charges()
            const es = sys.electrostatics()
            const pr = api.pairingStats(pos, charges, sys.box, es, contact)
            const aa = ev.aggregateAnalysis
            const big = aa.aggregates.length > 0 ? aa.aggregates[0] : null
            const lamellar = big
              ? big.flatnessRatio <= thresholds.lamellarFlatnessRatio && big.inPlaneSymmetry >= thresholds.lamellarInPlaneSymmetryMin
              : false
            return {
              step,
              aggregates: aa.aggregateCount,
              largest: aa.sizeHistogram.length > 0 ? aa.sizeHistogram[0] : 0,
              sizeHistogram: aa.sizeHistogram.slice(0, 6),
              hasLamellarAggregate: aa.hasLamellarAggregate,
              hasVesicleAggregate: aa.hasVesicleAggregate,
              flatness: big ? big.flatnessRatio : null,
              inPlaneSymmetry: big ? big.inPlaneSymmetry : null,
              radialHeadShells: big ? big.radialHeadShells : null,
              transverseHeadShells: big ? big.transverseHeadShells : null,
              rg: big ? big.radiusOfGyration : null,
              rgOverUniform: big ? big.radiusOfGyration / (sys.box[0] / 2) : null,
              lamellarShape: lamellar,
              alpha: pr.alpha,
              pairedFraction: pr.pairedFraction,
              unlikeExcess: pr.unlikeFractionRandom > 0 ? pr.unlikeFraction / pr.unlikeFractionRandom : null,
              apparentPKa: api.apparentPKa(pr.alpha, es.pH),
              amphiphileFraction: ev.amphiphileFraction,
            }
          }

          const samples: any[] = []
          const t0 = performance.now()
          let done = 0
          while (done < steps) {
            const chunk = Math.min(sampleEvery, steps - done)
            await sys.step(chunk)
            done += chunk
            samples.push(await sample(done))
          }
          const wall = performance.now() - t0
          const esFinal = sys.electrostatics()
          sys.dispose()
          return {
            pH: pHv,
            acidSoap,
            N: actualN,
            waterPlaced,
            nLipids,
            box: boxL,
            rhoAmph: nLipids / boxL ** 3,
            steps,
            stepsPerSec: steps / (wall / 1000),
            esCutoff: esFinal.cutoff,
            thresholds: {
              lamellarFlatnessRatio: thresholds.lamellarFlatnessRatio,
              lamellarInPlaneSymmetryMin: thresholds.lamellarInPlaneSymmetryMin,
            },
            samples,
          }
        },
        N_LIPIDS,
        BOX,
        0.8,
        pH,
        STRENGTH,
        STEPS,
        SAMPLE_EVERY,
      )

      const tail = result.samples.slice(Math.max(0, result.samples.length - 3))
      const meanOf = (f: (s: any) => number | null): number | null => {
        const vs = tail.map(f).filter((v): v is number => v !== null && Number.isFinite(v))
        return vs.length > 0 ? vs.reduce((a: number, b: number) => a + b, 0) / vs.length : null
      }
      const bilayerCompetent = tail.filter((s: any) => s.hasLamellarAggregate).length
      const summary = {
        ...result,
        alphaTail: meanOf((s) => s.alpha),
        pairedTail: meanOf((s) => s.pairedFraction),
        unlikeExcessTail: meanOf((s) => s.unlikeExcess),
        pKaAppTail: meanOf((s) => s.apparentPKa),
        flatnessTail: meanOf((s) => s.flatness),
        inPlaneTail: meanOf((s) => s.inPlaneSymmetry),
        rgOverUniformTail: meanOf((s) => s.rgOverUniform),
        largestTail: meanOf((s) => s.largest),
        lamellarShapeTail: tail.filter((s: any) => s.lamellarShape).length,
        bilayerCompetentSamples: bilayerCompetent,
        tailSamples: tail.length,
      }
      console.log(
        `ACID-SOAP-PH pH=${summary.pH} S=${summary.acidSoap} N=${summary.N} rho_amph=${summary.rhoAmph.toExponential(3)} ` +
          `alpha=${summary.alphaTail?.toFixed(4)} pKa_app=${summary.pKaAppTail?.toFixed(3)} ` +
          `спаренных=${summary.pairedTail?.toFixed(4)} избыток=${summary.unlikeExcessTail?.toFixed(3)} | ` +
          `агрегатов=${tail[tail.length - 1].aggregates} крупнейший=${summary.largestTail?.toFixed(1)}/${summary.nLipids} ` +
          `плоскостность=${summary.flatnessTail?.toFixed(4)} (нужно <=${summary.thresholds.lamellarFlatnessRatio}) ` +
          `симметрия=${summary.inPlaneTail?.toFixed(4)} (нужно >=${summary.thresholds.lamellarInPlaneSymmetryMin}) ` +
          `Rg/(L/2)=${summary.rgOverUniformTail?.toFixed(4)} ` +
          `слоёв_голов radial=${tail[tail.length - 1].radialHeadShells} transverse=${tail[tail.length - 1].transverseHeadShells} | ` +
          `БИСЛОЙ-СПОСОБНЫХ=${bilayerCompetent}/${tail.length} ` +
          `форма_ламеллярна=${summary.lamellarShapeTail}/${tail.length} ` +
          `throughput=${summary.stepsPerSec.toFixed(1)} шаг/с`,
      )
      console.log(
        `ACID-SOAP-PH-TRAJECTORY pH=${summary.pH} S=${summary.acidSoap}\n  ` +
          result.samples
            .map(
              (s: any) =>
                `step=${s.step} aggs=${s.aggregates} largest=${s.largest} flat=${s.flatness?.toFixed(4) ?? 'N/A'} ` +
                `sym=${s.inPlaneSymmetry?.toFixed(4) ?? 'N/A'} shells=${s.radialHeadShells}/${s.transverseHeadShells} ` +
                `Rg=${s.rg?.toFixed(3) ?? 'N/A'} alpha=${s.alpha.toFixed(4)} paired=${s.pairedFraction.toFixed(4)} ` +
                `lamellar=${s.hasLamellarAggregate}`,
            )
            .join('\n  '),
      )
      rows.push(summary)
    }

    mkdirSync(dirname(ARTIFACT), { recursive: true })
    const prev = existsSync(ARTIFACT) ? JSON.parse(readFileSync(ARTIFACT, 'utf8')) : { arms: [] }
    const keep = (prev.arms ?? []).filter((a: any) => !rows.some((r) => r.pH === a.pH && r.acidSoap === a.acidSoap))
    writeFileSync(
      ARTIFACT,
      JSON.stringify(
        {
          measuredBy: 'tests/soup-acid-soap-ph.test.ts',
          generatedAt: new Date().toISOString(),
          arms: [...keep, ...rows].sort((a: any, b: any) => a.acidSoap - b.acidSoap || a.pH - b.pH),
        },
        null,
        2,
      ),
    )
    console.log(`ACID-SOAP-PH WROTE ${ARTIFACT} (${keep.length + rows.length} плеч)`)
    expect(rows.length).toBe(PH_LIST.length)
    // The arms must be real measurements: amphiphiles recognised, charge live, aggregates found.
    for (const r of rows) {
      expect(r.samples.length).toBeGreaterThan(0)
      expect(r.largestTail).toBeGreaterThan(0)
    }
  },
  1_500_000,
)
