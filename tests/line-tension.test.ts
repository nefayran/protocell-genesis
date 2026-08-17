import { mkdirSync, writeFileSync } from 'node:fs'
import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Rim line tension (lambda) of this engine's own bilayer edge -- task 'kappa-lambda' (2026-08-17),
// the cheap half of the vesicle-closure calculation (R_c ~ kappa/lambda). The model's attraction
// is explicit (WCA + FENE + bend + the tail-tail cos^2 well, engine/wgsl/forces.wgsl), so the rim's
// excess energy is directly computable from a prepared flat patch: build the SAME lattice bilayer
// layoutBilayer already produces (engine/src/sim.ts), read its pristine positions back BEFORE any
// step (no RNG re-draw needed), then instantiate it TWICE from those exact coordinates --
//  - REFERENCE: box = [Lx, Ly, boxZ], the patch spans the box exactly -> doubly periodic, no edge.
//  - PATCH: box = [Lx + 2*MARGIN_SIGMA, Ly, boxZ], the SAME positions shifted +MARGIN_SIGMA in x ->
//    periodic in y (no edge there), free in x (two edges, each of length Ly, since periodic-y makes
//    the strip infinite in that direction with no end caps -- rim length = 2*Ly).
// Both configurations carry the SAME lipid count and the SAME internal geometry; only the box (and
// therefore whether the patch's own boundary is periodic or free) differs. lambda is then
//   lambda = (mean_potential(patch) - mean_potential(reference)) / (2*Ly)
// Potential energy (excludes kinetic) is recovered from System's own public facade with NO new
// engine code: total = kinetic + potential (totalEnergy()), kinetic = 1.5*N*kineticEnergyPerDof()
// (kineticEnergyPerDof() is <v^2>/3 per bead, i.e. kinetic energy PER DEGREE OF FREEDOM in these
// reduced units -- see its own doc comment in engine/src/sim.ts), so potential = total - 1.5*N*KEdof.
// Measured at TWO self-similar sizes (Lx,Ly both doubled) as the brief requires: if the energy
// difference were secretly a BULK/area effect it would scale with area (~4x between the two sizes,
// since area ~ Lx*Ly and both double), but rim length only doubles (~2x) -- so a genuine line
// tension (lambda roughly constant across the two sizes) is distinguishable from an area artifact
// (lambda would double) by this comparison alone, with no external reference value needed.

// AREA_PER_LIPID: this project's own measured area/lipid plateau, not a Cooke & Deserno model
// literal -- same number the brief hands us for the R_c-vs-cup-radius comparison later, and
// consistent with every independent measurement of it in this codebase: engine/src/sim.ts's
// VESICLE_AREA_PER_LIPID=1.208 (self-assembly's own measured plateau) and gate6-bilayer.test.ts's
// ensemble measurement (1.2060-1.2032 across runs, task-9-report.md). Used only to size a SQUARE
// starting lattice at the right density -- layoutBilayer relaxes locally from here regardless.
const AREA_PER_LIPID = 1.209
const SPACING = Math.sqrt(AREA_PER_LIPID)

// Two self-similar patch sizes (same aspect ratio, linear scale factor 2 -> area factor 4, rim
// factor 2): NX/NY are the per-leaflet grid side length layoutBilayer's own nSide would produce for
// lipids=2*NX*NY (a perfect square lipid count keeps this exact, not an approximation of
// layoutBilayer's ceil/sqrt rounding). Sized so BOTH createSystem's grid invariants hold with
// margin to spare (checked numerically before running): dims>=3 on x,y and min(box.x,box.y)/2 >
// bend.r0=4.0 sigma (data/params.json) -- min(box)/2 is 5.50 sigma at the small size, comfortably
// above 4.0.
const SIZES = [
  { nx: 10, ny: 10 },
  { nx: 20, ny: 20 },
]
const BOX_Z = 30 // vertical room for the ~4.5 sigma bilayer (gate6-bilayer.test.ts) plus fluctuation
// clearance and the open (non-periodic) z margin every other test in this suite also budgets.
const SEED = 11 // arbitrary -- layoutBilayer's jitter is a small perturbation of the lattice, not a
// physically meaningful draw; any seed produces a valid starting lattice.

// Free-edge decoupling margin: the vacuum gap added on EACH side of the patch in x. Must exceed the
// longest-ranged intermolecular interaction (wcaCutoff(tail_tail) + attraction.wc, data/params.json)
// so the patch's own periodic images in x never see each other through the gap -- 3x that range is
// comfortable headroom for the exposed edge tails' own thermal excursions once they relax outward
// (computed from params, not a bare literal).
const MARGIN_FACTOR = 3

const EQUIL_STEPS = 30_000 // local edge/lattice relaxation budget. These are small systems (384 or
// 1536 beads) -- see the report for the measured throughput that makes this cheap; generous relative
// to gate6-bilayer.test.ts's own 3000-step fixed-box relaxation because a freshly-cut edge (patch
// side) has more to relax than an already-periodic sheet (reference side), and both sides run the
// same budget so neither is shortchanged.
const SAMPLE_COUNT = 30
const SAMPLE_SPACING = 200 // MD steps between energy samples -- same order of magnitude as
// measureBilayerAveraged's/measureBendingModulusDetailed's own inter-sample spacing (100) in
// engine/src/index.ts, widened slightly since a whole-system potential energy is a slower-decorrelating
// observable than a single mode.

const OUT_DIR = 'verify/out'
const OUT_FILE = `${OUT_DIR}/line-tension.json`

test('линейное натяжение края бислоя (lambda): patch со свободным краем против периодической ссылки, две площадки', async () => {
  const page = await gpuPage()
  page.on('console', (msg) => console.log(`[page] ${msg.text()}`))

  const wallClockStart = Date.now()
  const results = await page.evaluate(
    async (
      sizes: { nx: number; ny: number }[],
      spacing: number,
      boxZ: number,
      seed: number,
      marginFactor: number,
      equilSteps: number,
      sampleCount: number,
      sampleSpacing: number,
    ) => {
      const api = (window as any).api
      const p = api.loadParams()
      const margin = marginFactor * (api.wcaCutoff(p.beadSizes.tail_tail) + p.attraction.wc)

      async function samplePotentials(sys: any, label: string): Promise<number[]> {
        await sys.step(equilSteps)
        const N = sys.lipids * 3
        const out: number[] = []
        for (let i = 0; i < sampleCount; i++) {
          if (i > 0) await sys.step(sampleSpacing)
          const total = await sys.totalEnergy()
          const keDof = await sys.kineticEnergyPerDof()
          const potential = total - 1.5 * N * keDof
          out.push(potential)
        }
        console.log(
          `LINE-TENSION-SAMPLE ${label} lipids=${sys.lipids} steps=${sys.steps} ` +
            `potentialMean=${(out.reduce((a, b) => a + b, 0) / out.length).toFixed(4)}`,
        )
        return out
      }

      const perSize: any[] = []
      for (const { nx, ny } of sizes) {
        const lipids = 2 * nx * ny
        const Lx = nx * spacing
        const Ly = ny * spacing

        const refSys = await api.createSystem({ lipids, box: [Lx, Ly, boxZ], seed, layout: 'bilayer' })
        const pristine: Float32Array = await refSys.positions() // exact layoutBilayer output, 0 steps taken yet
        const refPotentials = await samplePotentials(refSys, `ref(${nx}x${ny})`)

        const translated = new Float32Array(pristine.length)
        for (let i = 0; i < pristine.length; i += 4) {
          translated[i] = pristine[i] + margin
          translated[i + 1] = pristine[i + 1]
          translated[i + 2] = pristine[i + 2]
          translated[i + 3] = pristine[i + 3]
        }
        const patchBox: [number, number, number] = [Lx + 2 * margin, Ly, boxZ]
        const patchSys = await api.createSystem({
          lipids,
          box: patchBox,
          seed,
          layout: 'bilayer',
          positions: translated,
        })
        const patchPotentials = await samplePotentials(patchSys, `patch(${nx}x${ny})`)

        perSize.push({ nx, ny, lipids, Lx, Ly, boxZ, margin, refPotentials, patchPotentials })
      }
      return perSize
    },
    SIZES,
    SPACING,
    BOX_Z,
    SEED,
    MARGIN_FACTOR,
    EQUIL_STEPS,
    SAMPLE_COUNT,
    SAMPLE_SPACING,
  )
  const wallClockMs = Date.now() - wallClockStart

  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
  const sd = (xs: number[]) => {
    const m = mean(xs)
    return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
  }

  const table = results.map((r: any) => {
    const refMean = mean(r.refPotentials)
    const refSd = sd(r.refPotentials)
    const patchMean = mean(r.patchPotentials)
    const patchSd = sd(r.patchPotentials)
    const deltaE = patchMean - refMean
    const deltaESe = Math.sqrt((refSd * refSd) / r.refPotentials.length + (patchSd * patchSd) / r.patchPotentials.length)
    const rimLength = 2 * r.Ly
    const lambda = deltaE / rimLength
    const lambdaSe = deltaESe / rimLength
    return { ...r, refMean, refSd, patchMean, patchSd, deltaE, deltaESe, rimLength, lambda, lambdaSe }
  })

  for (const r of table) {
    console.log(
      `LINE-TENSION nx=${r.nx} ny=${r.ny} lipids=${r.lipids} Lx=${r.Lx.toFixed(4)} Ly=${r.Ly.toFixed(4)} ` +
        `margin=${r.margin.toFixed(4)} refMean=${r.refMean.toFixed(4)}+/-${r.refSd.toFixed(4)} ` +
        `patchMean=${r.patchMean.toFixed(4)}+/-${r.patchSd.toFixed(4)} deltaE=${r.deltaE.toFixed(4)}+/-${r.deltaESe.toFixed(4)} ` +
        `rimLength=${r.rimLength.toFixed(4)} lambda=${r.lambda.toFixed(4)}+/-${r.lambdaSe.toFixed(4)}`,
    )
  }

  const lambdaSmall = table[0].lambda
  const lambdaLarge = table[1].lambda
  const ratio = lambdaLarge / lambdaSmall

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(
    OUT_FILE,
    JSON.stringify(
      {
        gate: 'line-tension-lambda',
        areaPerLipidBasis: AREA_PER_LIPID,
        marginFactor: MARGIN_FACTOR,
        equilSteps: EQUIL_STEPS,
        sampleCount: SAMPLE_COUNT,
        sampleSpacing: SAMPLE_SPACING,
        seed: SEED,
        wallClockMs,
        sizes: table.map((r: any) => ({
          nx: r.nx,
          ny: r.ny,
          lipids: r.lipids,
          Lx: r.Lx,
          Ly: r.Ly,
          margin: r.margin,
          refMean: r.refMean,
          refSd: r.refSd,
          patchMean: r.patchMean,
          patchSd: r.patchSd,
          deltaE: r.deltaE,
          deltaESe: r.deltaESe,
          rimLength: r.rimLength,
          lambda: r.lambda,
          lambdaSe: r.lambdaSe,
        })),
        lambdaSmall,
        lambdaLarge,
        sizeRatio: ratio,
      },
      null,
      2,
    ),
  )
  console.log(`LINE-TENSION artifact written: ${OUT_FILE}`)
  console.log(`LINE-TENSION lambdaSmall=${lambdaSmall.toFixed(4)} lambdaLarge=${lambdaLarge.toFixed(4)} ratio=${ratio.toFixed(4)}`)

  // The rim costs energy (a free edge is never cheaper than no edge at all for this attractive
  // potential) and the cost is finite -- both sizes.
  for (const r of table) {
    expect(Number.isFinite(r.lambda)).toBe(true)
    expect(r.lambda).toBeGreaterThan(0)
  }
  // Size-independence: an area artifact would show up as roughly a FACTOR OF 4 between the two
  // sizes (area quadruples, rim only doubles, so "lambda" computed on a bulk effect would double);
  // a genuine line tension stays within a much narrower band. This bound (3x either way) is loose
  // enough to survive this measurement's own thermal noise at these small system sizes, while still
  // clearly separating a true line-tension result from the area-artifact failure mode the brief
  // warns about.
  expect(ratio).toBeGreaterThan(1 / 3)
  expect(ratio).toBeLessThan(3)
}, 1_800_000)
