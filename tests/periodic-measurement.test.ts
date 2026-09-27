import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import type { CheckpointFile } from '../soup/src/checkpoint'
import { loadSoup, type Monomer } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { findAmphiphiles } from '../soup/src/amphiphile'
import { clusterComponents } from '../engine/src/aggregate'
import { loadStageThresholds } from '../soup/src/stages'
import periodicParams from '../data/periodic-measurement.json'
import { periodicCentre, periodicRadialProfile, twoPeaks, periodicCavityVolume, holeSearch, headDirectionsWithinTrust, wrap1 } from './helpers/periodic-geometry'
import { angleDeg } from './helpers/geometry-primitives'
import { makeRng, buildShell, buildSolidBlob, isHeadW, allIdx, TARGET_AREA_PER_HEAD_BEAD } from './helpers/periodic-synthetic-fixtures'

// Task 'periodic-measurement' (2026-08-18). Branch stage-a-atoms, no worktree, no new simulation:
// every real-data number below comes from the ten checkpoints already on disk
// (data/checkpoints/vesicle-93k-step160000.json .. step250000.json), decoded purely in Node
// (decodeCheckpointResume) -- no GPU, no gpuPage(), no page.evaluate() anywhere in this file.
//
// WHY THIS FILE EXISTS: .superpowers/sdd/2026-08-16-soup-to-vesicle/shell-pore-report.md and
// expanded-box-report.md both establish that the dense campaign's largest aggregate (~1282
// amphiphiles) spans a box of only 58 sigma/side -- its own mean head radius (~29.5-29.8 sigma) is
// comparable to or exceeds box/2=29 sigma, and its own covalent bonds pass through the periodic
// boundary too tightly to survive any box expansion (FENE divergence on wraparound-dependent bonds,
// expanded-box-report.md SS2 run 5). Consequently EVERY geometric reading obtained via a single
// unwrapped frame -- this project's own unwrapAggregate(), or shell-pore-report.md's own corrected
// properUnwrap() -- is unreliable at this scale (shell-pore-report.md SS0): a particle on the far
// side of an object whose own diameter approaches the box can alias to the wrong periodic image no
// matter which reference particle or BFS walk order the unwrap starts from.
//
// This file's whole point is to measure the SAME object WITHOUT ever building a global unwrapped
// frame. Minimum-image distances/directions between any two points are always well-defined
// regardless of the object's own size (they involve exactly one pair, no global frame at all) --
// what is ill-defined is a single COORDINATE-VALUED centre of mass, which this file replaces with a
// periodic (circular-mean) centre, defined per axis and refused when the axis is not concentrated
// enough to support one (data/periodic-measurement.json's own written basis). Every downstream
// measurement (radial profile, cavity flood, hole search) is built ONLY from minimum-image
// displacements relative to that periodic centre -- no unwrap() call, no BFS-over-covalent-bonds
// walk, anywhere in this file. The periodic-aware machinery itself lives in
// tests/helpers/periodic-geometry.ts and tests/helpers/periodic-synthetic-fixtures.ts (moved out by
// responsibility, file-size rule, root CLAUDE.md) -- this file keeps only the checkpoint-specific
// glue and the tests themselves.
//
// Runs under vitest, not plain tsx, for the same mechanical reason shell-pore-check.test.ts already
// documents: soup/src/stages.ts (loadStageThresholds, imported below for closureCell/closureRadius/
// headDensityBins/enclosedVolume -- reused, not reinvented) transitively imports engine/src/
// closure.ts, which does a Vite-specific `?raw` import of its WGSL shader at module load time; only
// Vite's own transform (vitest included) resolves that. No test in this file calls gpuPage() --
// nothing here touches a browser or the GPU.
//
// No numeric model constant is added to soup/src or engine/src (neither is touched at all -- every
// function below is new analysis code, kept in this file/its own tests/helpers only, exactly the
// precedent shell-pore-check.test.ts's own properUnwrap() already set: "this does not touch
// engine/src/aggregate.ts -- it is new analysis code, kept in this file only"). The one genuinely
// new tunable parameter this task introduces (the circular-concentration refusal threshold) lives in
// data/periodic-measurement.json with its own written basis, per this task's own constraint; the
// Fibonacci-sphere hole-search resolution is reused verbatim from shell-pore-check.test.ts's own
// choice (same file, same basis) for direct comparability, also recorded there. Every other
// threshold this file reads (closureCell, closureRadius, headDensityBins, enclosedVolume) is the
// project's own existing, already-justified data/soup.json value (loadStageThresholds()), not a new
// number.

const NONE_U32 = 0xffffffff
const STEPS = [160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000]
// The campaign checkpoints are gigabytes and are not in the repository. Without them the
// checkpoint re-analysis below is skipped, not failed.
const HAVE_CHECKPOINTS = STEPS.every((step) => existsSync(`data/checkpoints/vesicle-93k-step${step}.json`))
const OUT_DIR = 'verify/out'
const OUT_FILE = `${OUT_DIR}/periodic-measurement.json`
const CONCENTRATION_ALPHA = periodicParams.circularConcentration.alpha
const FIB_SAMPLES = periodicParams.fibonacciSphereSamples.count

// ==================================================================================================
// SYNTHETIC VALIDATION -- run before anything real is trusted, per this task's own brief.
// ==================================================================================================

const SYN_BOX: [number, number, number] = [50, 50, 50]
const SYN_CENTRE: [number, number, number] = [25, 25, 25]
const BILAYER_THICKNESS = 4.457 // this project's own measured value (data/literature.json's closure gate, verify/out/gates.json) -- reused, not a new number
const SYN_R_IN = 15 - BILAYER_THICKNESS / 2
const SYN_R_OUT = 15 + BILAYER_THICKNESS / 2
// Tail count: generous, not fine-tuned like the head counts above -- the annulus fill is only
// needed for the head:tail radial-profile check, sealing against the flood is provided entirely by
// the two head leaflets (TARGET_AREA_PER_HEAD_BEAD's own doc comment). Scaled off the SAME area-based
// reasoning (total leaflet area / TARGET_AREA_PER_HEAD_BEAD, times 2 for headroom) purely so this
// number tracks SYN_R_IN/SYN_R_OUT if either ever changes, rather than being independently hand-picked.
const SYN_TAIL_COUNT = Math.ceil((2 * (4 * Math.PI * SYN_R_IN ** 2 + 4 * Math.PI * SYN_R_OUT ** 2)) / TARGET_AREA_PER_HEAD_BEAD)
const CLOSURE = loadStageThresholds() // closureCell/closureRadius/enclosedVolume/headDensityBins -- existing thresholds, reused verbatim

test('synthetic: circular centre refuses an axis that is uniformly spread across the box, keeps the other two', () => {
  const rng = makeRng(1)
  const n = 4000
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    // x,y: tight Gaussian-like jitter around (10,40) -- a real, localized object.
    const x = 10 + (rng() - 0.5) * 2
    const y = 40 + (rng() - 0.5) * 2
    // z: uniform across the WHOLE box -- no preferred position at all on this axis.
    const z = rng() * SYN_BOX[2]
    out.push(x, y, z, 1)
  }
  const positions = new Float32Array(out)
  const idx = allIdx(positions)
  const pc = periodicCentre(positions, idx, SYN_BOX, CONCENTRATION_ALPHA)

  console.log(
    `SYN-CONCENTRATION Rx=${pc.axes[0].R.toFixed(4)} Ry=${pc.axes[1].R.toFixed(4)} Rz=${pc.axes[2].R.toFixed(4)} ` +
      `trusted=[${pc.axes.map((a) => a.trusted)}] coords=[${pc.axes.map((a) => a.coordinate?.toFixed(2) ?? 'null')}] alpha=${CONCENTRATION_ALPHA}`,
  )

  expect(pc.axes[0].trusted).toBe(true)
  expect(pc.axes[1].trusted).toBe(true)
  expect(pc.axes[2].trusted).toBe(false) // the uniformly-spread axis must be refused
  expect(pc.axes[2].coordinate).toBeNull()
  expect(pc.axes[0].coordinate).not.toBeNull()
  expect(pc.axes[0].coordinate).toBeCloseTo(10, 0)
  expect(pc.axes[1].coordinate).toBeCloseTo(40, 0)
  expect(pc.centre).toBeNull() // overall centre refused: one bad axis is enough
})

test('synthetic: periodic tools give the SAME reading for a centred shell and the same shell straddling all three periodic faces', () => {
  const rngShared = makeRng(2)
  const centred = buildShell(rngShared, SYN_CENTRE, SYN_R_IN, SYN_R_OUT, SYN_TAIL_COUNT)
  // Wrap every coordinate through a translation that moves the centre from (25,25,25) to (0,0,0) --
  // every axis straddles its own periodic face simultaneously (the hardest case, matching
  // tests/closure.test.ts's own "corner" test convention).
  const straddling = centred.slice()
  for (let i = 0; i < straddling.length; i += 4) {
    straddling[i] = wrap1(straddling[i] - SYN_CENTRE[0], SYN_BOX[0])
    straddling[i + 1] = wrap1(straddling[i + 1] - SYN_CENTRE[1], SYN_BOX[1])
    straddling[i + 2] = wrap1(straddling[i + 2] - SYN_CENTRE[2], SYN_BOX[2])
  }
  const idxC = allIdx(centred)
  const idxS = allIdx(straddling)

  const pcC = periodicCentre(centred, idxC, SYN_BOX, CONCENTRATION_ALPHA)
  const pcS = periodicCentre(straddling, idxS, SYN_BOX, CONCENTRATION_ALPHA)
  expect(pcC.centre).not.toBeNull()
  expect(pcS.centre).not.toBeNull()
  for (let a = 0; a < 3; a++) expect(pcC.axes[a].R).toBeCloseTo(pcS.axes[a].R, 6)

  const profC = periodicRadialProfile(idxC, centred, (i) => isHeadW(centred, i), pcC.centre as [number, number, number], SYN_BOX, CLOSURE.headDensityBins)
  const profS = periodicRadialProfile(idxS, straddling, (i) => isHeadW(straddling, i), pcS.centre as [number, number, number], SYN_BOX, CLOSURE.headDensityBins)
  const sigma = loadParams().sigma
  const peaksC = twoPeaks(
    profC.bins.map((b) => b.headCount),
    profC.bins.map((b) => (b.rLo + b.rHi) / 2),
    sigma,
  )
  const peaksS = twoPeaks(
    profS.bins.map((b) => b.headCount),
    profS.bins.map((b) => (b.rLo + b.rHi) / 2),
    sigma,
  )

  const cavC = periodicCavityVolume(centred, idxC, SYN_BOX, pcC.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)
  const cavS = periodicCavityVolume(straddling, idxS, SYN_BOX, pcS.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)

  console.log(
    `SYN-EQUALITY centred: peaks=[${peaksC.primary.toFixed(3)},${peaksC.secondary?.toFixed(3)}] sep=${peaksC.separation?.toFixed(3)} cavity=${cavC.volume.toFixed(3)} | ` +
      `straddling: peaks=[${peaksS.primary.toFixed(3)},${peaksS.secondary?.toFixed(3)}] sep=${peaksS.separation?.toFixed(3)} cavity=${cavS.volume.toFixed(3)}`,
  )

  // The whole point: a periodic method must not care where in the box the object happens to sit.
  expect(cavS.volume).toBe(cavC.volume)
  expect(peaksS.primary).toBeCloseTo(peaksC.primary, 3)
  expect(peaksS.secondary as number).toBeCloseTo(peaksC.secondary as number, 3)
  expect(peaksS.separation as number).toBeCloseTo(peaksC.separation as number, 3)
  // And it must actually find a real cavity, not report zero for a genuinely closed shell.
  const idealInner = (4 / 3) * Math.PI * SYN_R_IN ** 3
  expect(cavC.volume).toBeGreaterThan(idealInner * 0.6)
  expect(cavC.volume).toBeLessThan(idealInner * 1.3)
  expect(cavC.volume).toBeGreaterThan(CLOSURE.enclosedVolume) // clears the project's own vesicle-closure minimum
})

test('synthetic: a known, full-thickness hole is detected via angular deficit with no unwrap, and the cavity leaks', () => {
  const rng = makeRng(3)
  const holeDir = (() => {
    const v: [number, number, number] = [1, 1, 1]
    const m = Math.sqrt(3)
    return [v[0] / m, v[1] / m, v[2] / m] as [number, number, number]
  })()
  const HOLE_HALF_ANGLE = 30
  const closed = buildShell(rng, SYN_CENTRE, SYN_R_IN, SYN_R_OUT, SYN_TAIL_COUNT)
  const holed = buildShell(makeRng(3), SYN_CENTRE, SYN_R_IN, SYN_R_OUT, SYN_TAIL_COUNT, { dir: holeDir, halfAngleDeg: HOLE_HALF_ANGLE })

  const idxClosed = allIdx(closed)
  const idxHoled = allIdx(holed)
  const pcClosed = periodicCentre(closed, idxClosed, SYN_BOX, CONCENTRATION_ALPHA)
  const pcHoled = periodicCentre(holed, idxHoled, SYN_BOX, CONCENTRATION_ALPHA)
  expect(pcClosed.centre).not.toBeNull()
  expect(pcHoled.centre).not.toBeNull()

  const { dirs: dirsClosed } = headDirectionsWithinTrust(idxClosed, closed, (i) => isHeadW(closed, i), pcClosed.centre as [number, number, number], SYN_BOX)
  const { dirs: dirsHoled, excluded, total } = headDirectionsWithinTrust(idxHoled, holed, (i) => isHeadW(holed, i), pcHoled.centre as [number, number, number], SYN_BOX)
  const holeOnClosed = holeSearch(dirsClosed, FIB_SAMPLES)
  const holeOnHoled = holeSearch(dirsHoled, FIB_SAMPLES)

  const cavClosed = periodicCavityVolume(closed, idxClosed, SYN_BOX, pcClosed.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)
  const cavHoled = periodicCavityVolume(holed, idxHoled, SYN_BOX, pcHoled.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)

  console.log(
    `SYN-HOLE closedShell: holeRadiusDeg=${holeOnClosed.holeRadiusDeg.toFixed(2)} cavity=${cavClosed.volume.toFixed(3)} | ` +
      `holedShell: knownHalfAngle=${HOLE_HALF_ANGLE} foundHoleRadiusDeg=${holeOnHoled.holeRadiusDeg.toFixed(2)} ` +
      `foundHoleCentre=[${holeOnHoled.holeCentre.map((v) => v.toFixed(3))}] angleToKnownDir=${angleDeg(holeOnHoled.holeCentre, holeDir).toFixed(2)}deg ` +
      `cavity=${cavHoled.volume.toFixed(3)} headsExcludedBeyondTrust=${excluded}/${total}`,
  )

  // A truly closed shell should show only the ordinary Poisson-noise-sized gaps a finite random
  // sample leaves, well under a real punched hole.
  expect(holeOnClosed.holeRadiusDeg).toBeLessThan(HOLE_HALF_ANGLE * 0.6)
  // The punched hole must be found close to where it actually is (Fibonacci-sphere resolution at
  // 800 samples plus the finite bead sample means "close", not "exact").
  expect(angleDeg(holeOnHoled.holeCentre, holeDir)).toBeLessThan(15)
  expect(holeOnHoled.holeRadiusDeg).toBeGreaterThan(HOLE_HALF_ANGLE * 0.6)
  expect(holeOnHoled.holeRadiusDeg).toBeLessThan(HOLE_HALF_ANGLE * 1.4)
  // A full-thickness channel this wide must let the flood leak substantially relative to the closed case.
  expect(cavHoled.volume).toBeLessThan(cavClosed.volume * 0.5)
})

test('synthetic: a solid blob with no interior gives no cavity', () => {
  const rng = makeRng(4)
  const blob = buildSolidBlob(rng, SYN_CENTRE, 15, 150_000)
  const idx = allIdx(blob)
  const pc = periodicCentre(blob, idx, SYN_BOX, CONCENTRATION_ALPHA)
  expect(pc.centre).not.toBeNull()
  const cavity = periodicCavityVolume(blob, idx, SYN_BOX, pc.centre as [number, number, number], CLOSURE.closureCell, CLOSURE.closureRadius)
  console.log(`SYN-BLOB cavity=${cavity.volume.toFixed(4)} totalEmpty=${cavity.totalEmpty} unreached=${cavity.unreached} closureThreshold=${CLOSURE.enclosedVolume}`)
  expect(cavity.volume).toBeLessThan(CLOSURE.enclosedVolume / 100) // same "must not be mistaken for a real cavity" bound tests/closure.test.ts's own precedent uses
})

// ==================================================================================================
// REAL DATA -- the last ten dense-campaign checkpoints, same STEPS as shell-pore-report.md for direct
// comparability. No unwrap anywhere below this line either.
// ==================================================================================================

function loadCheckpoint(step: number): { positions: Float32Array; box: [number, number, number] } {
  const path = `data/checkpoints/vesicle-93k-step${step}.json`
  const file = JSON.parse(readFileSync(path, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  if (resume.globalStep !== step) throw new Error(`checkpoint step${step}.json: globalStep=${resume.globalStep} != ${step}`)
  return { positions: resume.positions, box: file.config.box }
}

/** Original indices of the largest aggregate's own members (heads+chains of every recognised
 * amphiphile whose head sits in the largest connected component) -- same recognition pipeline
 * shell-pore-check.test.ts's own findLargest() used (findAmphiphiles + clusterComponents, same
 * cutoff derivation), simplified here since this file never needs the covalent-bond edge list (no
 * unwrap sanity check is performed anywhere in this file -- there is no unwrap to sanity-check). */
function findLargestMembers(positions: Float32Array, bonds: Uint32Array, box: [number, number, number], monomers: Monomer[]): { idx: number[]; amphiphileCount: number; aggregateCount: number } {
  const params = loadParams()
  const amphiphiles = findAmphiphiles(positions, bonds, monomers)
  const memberIdx = new Set<number>()
  for (const a of amphiphiles) {
    memberIdx.add(a.headIndex)
    for (const c of a.chain) memberIdx.add(c)
  }
  const idxArr = Array.from(memberIdx)
  const memberPositions = new Float32Array(idxArr.length * 4)
  for (let k = 0; k < idxArr.length; k++) memberPositions.set(positions.subarray(idxArr[k] * 4, idxArr[k] * 4 + 4), k * 4)
  const memberRadii = monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
  const cutoff = wcaCutoff(params.sigma * Math.max(...memberRadii)) + params.attraction.wc
  const components = clusterComponents(memberPositions, box, cutoff, true) // z periodic: bulk soup
  const largestLocal = components[0]
  const aggOriginalIdx = largestLocal.map((k) => idxArr[k])
  const aggSet = new Set(aggOriginalIdx)
  const amphiphileCount = amphiphiles.filter((a) => aggSet.has(a.headIndex)).length
  return { idx: aggOriginalIdx, amphiphileCount, aggregateCount: components.length }
}

function loadBonds(step: number): Uint32Array {
  const path = `data/checkpoints/vesicle-93k-step${step}.json`
  const file = JSON.parse(readFileSync(path, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  const N = file.N
  const edges: number[] = []
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < 3; s++) {
      const p = resume.bondSlots[i * 3 + s]
      if (p !== NONE_U32 && p > i) edges.push(i, p)
    }
  }
  return new Uint32Array(edges)
}

test.skipIf(!HAVE_CHECKPOINTS)('periodic re-analysis of the ten dense-campaign checkpoints: circular concentration, head/tail profiles, periodic cavity, hole without unwrapping', () => {
  const soup = loadSoup()
  const monomers: Monomer[] = soup.monomers
  const thresholds = CLOSURE
  const isHeadReal = (positions: Float32Array) => (i: number) => !!monomers[Math.round(positions[i * 4 + 3])]?.polar

  const perStep = STEPS.map((step) => {
    const { positions, box } = loadCheckpoint(step)
    const bonds = loadBonds(step)
    const agg = findLargestMembers(positions, bonds, box, monomers)
    const pc = periodicCentre(positions, agg.idx, box, CONCENTRATION_ALPHA)

    console.log(
      `PERIODIC-CENTRE step=${step} amphiphiles=${agg.amphiphileCount} aggregates=${agg.aggregateCount} particles=${agg.idx.length} ` +
        `Rx=${pc.axes[0].R.toFixed(4)} Ry=${pc.axes[1].R.toFixed(4)} Rz=${pc.axes[2].R.toFixed(4)} ` +
        `trusted=[${pc.axes.map((a) => a.trusted)}] centre=${pc.centre ? `[${pc.centre.map((v) => v.toFixed(3))}]` : 'REFUSED'} alpha=${CONCENTRATION_ALPHA}`,
    )

    if (!pc.centre) {
      // Refused per this file's own rule -- report it and skip the downstream measurements that
      // would need a well-defined centre, rather than silently substituting something else.
      return { step, agg, pc, radial: null, cavity: null, hole: null }
    }
    const centre = pc.centre

    const radial = periodicRadialProfile(agg.idx, positions, isHeadReal(positions), centre, box, thresholds.headDensityBins)
    const sigma = loadParams().sigma
    const peaks = twoPeaks(
      radial.bins.map((b) => b.headCount),
      radial.bins.map((b) => (b.rLo + b.rHi) / 2),
      sigma,
    )
    console.log(
      `PERIODIC-RADIAL step=${step} trustLimit=${radial.trustLimit.toFixed(3)} headTotal=${radial.headTotal} headBeyondTrust=${radial.headBeyondTrust} ` +
        `(${((100 * radial.headBeyondTrust) / radial.headTotal).toFixed(1)}%) tailTotal=${radial.tailTotal} tailBeyondTrust=${radial.tailBeyondTrust} ` +
        `(${((100 * radial.tailBeyondTrust) / radial.tailTotal).toFixed(1)}%) headPeakPrimary=${peaks.primary.toFixed(3)} ` +
        `headPeakSecondary=${peaks.secondary?.toFixed(3) ?? 'none'} separation=${peaks.separation?.toFixed(3) ?? 'none'} bilayerThicknessRef=${BILAYER_THICKNESS}`,
    )
    console.log(
      `PERIODIC-RADIAL-BINS step=${step} headCounts=[${radial.bins.map((b) => b.headCount).join(',')}] tailCounts=[${radial.bins.map((b) => b.tailCount).join(',')}]`,
    )

    const cavity = periodicCavityVolume(positions, agg.idx, box, centre, thresholds.closureCell, thresholds.closureRadius)
    console.log(
      `PERIODIC-CAVITY step=${step} volume=${cavity.volume.toFixed(4)} unreached=${cavity.unreached} totalEmpty=${cavity.totalEmpty} ` +
        `seedDistFromCentre=${cavity.seedDistFromCentre.toFixed(3)} theoreticalMaxDist=${cavity.theoreticalMaxDist.toFixed(3)} closureThreshold=${thresholds.enclosedVolume}`,
    )

    const { dirs, excluded, total } = headDirectionsWithinTrust(agg.idx, positions, isHeadReal(positions), centre, box)
    const hole = holeSearch(dirs, FIB_SAMPLES)
    console.log(
      `PERIODIC-HOLE step=${step} headsUsed=${dirs.length} headsExcludedBeyondTrust=${excluded}/${total} ` +
        `(${((100 * excluded) / total).toFixed(1)}%) holeRadiusDeg=${hole.holeRadiusDeg.toFixed(2)} solidAngleFrac=${(hole.solidAngleFraction * 100).toFixed(2)}% ` +
        `holeCentre=[${hole.holeCentre.map((v) => v.toFixed(3))}]`,
    )

    return { step, agg, pc, radial, peaks, cavity, hole, headsExcluded: excluded, headsTotal: total }
  })

  expect(perStep.length).toBe(10)
  for (const p of perStep) {
    expect(p.agg.amphiphileCount).toBeGreaterThan(1000) // sanity: same ~1282-amphiphile object at every step
    // The periodic centre must be reportable at every one of these checkpoints for the rest of this
    // test to say anything about them -- if a future checkpoint ever refuses, that itself is the
    // finding and this assertion is what would surface it instead of a silent null downstream.
    expect(p.pc.centre).not.toBeNull()
  }

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(
    OUT_FILE,
    JSON.stringify(
      {
        gate: 'periodic-measurement',
        alpha: CONCENTRATION_ALPHA,
        fibonacciSphereSamples: FIB_SAMPLES,
        steps: STEPS,
        perStep: perStep.map((p) => ({
          step: p.step,
          amphiphileCount: p.agg.amphiphileCount,
          aggregateCount: p.agg.aggregateCount,
          particleCount: p.agg.idx.length,
          circularConcentration: { Rx: p.pc.axes[0].R, Ry: p.pc.axes[1].R, Rz: p.pc.axes[2].R, trusted: p.pc.axes.map((a) => a.trusted) },
          centre: p.pc.centre,
          radial: p.radial
            ? {
                trustLimit: p.radial.trustLimit,
                headTotal: p.radial.headTotal,
                headBeyondTrust: p.radial.headBeyondTrust,
                tailTotal: p.radial.tailTotal,
                tailBeyondTrust: p.radial.tailBeyondTrust,
                headPeakPrimary: p.peaks?.primary ?? null,
                headPeakSecondary: p.peaks?.secondary ?? null,
                separation: p.peaks?.separation ?? null,
                bilayerThicknessReference: BILAYER_THICKNESS,
                bins: p.radial.bins,
              }
            : null,
          cavity: p.cavity,
          hole: p.hole ? { ...p.hole, headsUsed: p.headsExcluded !== undefined ? p.headsTotal! - p.headsExcluded! : null, headsExcludedBeyondTrust: p.headsExcluded, headsTotal: p.headsTotal } : null,
        })),
      },
      null,
      2,
    ),
  )
  console.log(`PERIODIC-MEASUREMENT artifact written: ${OUT_FILE}`)
})
