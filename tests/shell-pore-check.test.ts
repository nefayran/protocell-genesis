import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import type { CheckpointFile } from '../soup/src/checkpoint'
import { loadSoup, type Monomer } from '../soup/src/rules'
import { loadStageThresholds } from '../soup/src/stages'
import { shapeOf } from '../engine/src/aggregate'
import { angleDeg, mi1 } from './helpers/geometry-primitives'
import {
  findLargest,
  properUnwrap,
  checkUnwrapSanity,
  shellGeometry,
  compareUnwrapMethods,
  radialProfile,
  honestCavityVolume,
  localCavityVolumeWrapper,
  pluggedFlood,
  localRimVsBulk,
} from './helpers/shell-pore-geometry'

// Task 'shell-pore-check' (2026-08-18): independent test of the claim raised by
// .superpowers/sdd/2026-08-16-soup-to-vesicle/kappa-tightening-report.md's own SS2 -- that the
// campaign's largest aggregate (data/checkpoints/vesicle-93k-step*.json, label 'vesicle-93k') is a
// near-closed SPHERICAL SHELL with a single, persistent, geometrically small hole, not an open cup,
// and that the rim's in-situ line tension is small and negative (-1.5 to -4.6 eps/sigma there,
// vs +10.77 for the prepared flat patch). This file adds NO new simulation: every number below comes
// from the ten checkpoints already on disk (step160000..step250000, 10000-step spacing) via pure
// Node decode (decodeCheckpointResume, soup/src/checkpoint.ts -- no GPU, no gpuPage() call anywhere
// in this file). It reuses this project's own already-tested functions throughout (findAmphiphiles,
// clusterComponents, unwrapAggregate, shapeOf, densityProfileZ, bilayerPeaks,
// occupancy/enclosedVolume/dimsFor, analyzeAggregates) and adds no new physics, threshold, potential,
// recogniser or bond rule anywhere -- confirmed by `git diff --stat -- soup/src engine/src soup/wgsl
// engine/wgsl data/params.json data/soup.json data/literature.json` staying empty for this task (see
// shell-pore-report.md's own housekeeping section for the actual command+output). The geometry
// machinery itself now lives in tests/helpers/shell-pore-geometry.ts (moved out by responsibility,
// file-size rule, root CLAUDE.md); fibonacciSphere/angleDeg/mi1 come from
// tests/helpers/geometry-primitives.ts (verified identical to the copies in
// tests/periodic-measurement.test.ts and tests/rim-lambda-insitu.test.ts before unifying).
//
// This runs under vitest (not plain tsx) for one purely mechanical reason: engine/src/closure.ts
// (needed for occupancy/enclosedVolume, and transitively imported by soup/src/aggregates.ts's own
// analyzeAggregates) does a Vite-specific `?raw` import of its WGSL shader at module load time, which
// only Vite's own transform (vitest included) understands -- plain tsx/Node throws
// ERR_UNKNOWN_FILE_EXTENSION on it. tests/rim-lambda-insitu.test.ts's own first test hits the same
// constraint and is run the same way, by its own header note ("no GPU needed for this step at all").
// No test here calls gpuPage() -- nothing in this file touches a browser or the GPU.
//
// The Fibonacci-sphere hole search below is a byte-for-byte copy of the SAME method
// tests/rim-lambda-insitu.test.ts already used and reported (kappa-tightening-report.md SS2.2,
// attempt 2) -- repeated here on ten checkpoints instead of one, plus a consecutive-snapshot
// correlation test that report's own SS5 concern #4 asked for but did not have the budget to run.

const NONE_U32 = 0xffffffff
const STEPS = [160000, 170000, 180000, 190000, 200000, 210000, 220000, 230000, 240000, 250000]
// The campaign checkpoints are gigabytes and are not in the repository. Without them the
// checkpoint re-analysis below is skipped, not failed.
const HAVE_CHECKPOINTS = STEPS.every((step) => existsSync(`data/checkpoints/vesicle-93k-step${step}.json`))
const OUT_DIR = 'verify/out'
const OUT_FILE = `${OUT_DIR}/shell-pore-check.json`

// --- checkpoint decode (pure Node, no GPU) ------------------------------------------------------
function loadCheckpoint(step: number): { positions: Float32Array; box: [number, number, number]; bonds: Uint32Array } {
  const path = `data/checkpoints/vesicle-93k-step${step}.json`
  const file = JSON.parse(readFileSync(path, 'utf8')) as CheckpointFile
  const resume = decodeCheckpointResume(file)
  const N = file.N
  const box = file.config.box
  const edges: number[] = []
  for (let i = 0; i < N; i++) {
    for (let s = 0; s < 3; s++) {
      const p = resume.bondSlots[i * 3 + s]
      if (p !== NONE_U32 && p > i) edges.push(i, p)
    }
  }
  if (resume.globalStep !== step) throw new Error(`checkpoint step${step}.json: globalStep=${resume.globalStep} != ${step}`)
  return { positions: resume.positions, box, bonds: new Uint32Array(edges) }
}

test.skipIf(!HAVE_CHECKPOINTS)('shell-pore-check: radial profile, hole persistence across 10 checkpoints, plugged-vs-measured volume, rim head enrichment (no GPU, pure decode+analysis)', () => {
  const soup = loadSoup()
  const monomers: Monomer[] = soup.monomers
  const thresholds = loadStageThresholds()

  const perStep = STEPS.map((step) => {
    const { positions, box, bonds } = loadCheckpoint(step)
    const agg = findLargest(positions, bonds, box, monomers)
    const geom = shellGeometry(agg, positions, box)
    const honest = honestCavityVolume(positions, box, monomers, agg)
    const fixedSanity = checkUnwrapSanity(agg, geom.unwrapped)
    const { brokenSanity, brokenShape } = compareUnwrapMethods(agg, positions, box)
    console.log(
      `SHELL-PORE step=${step} aggregates=${agg.aggregateCount} amphiphiles=${agg.aggAmphiphiles.length} ` +
        `particles=${agg.aggOriginalIdx.length} Rg=${geom.radiusOfGyration.toFixed(3)} flatness=${geom.flatnessRatio.toFixed(4)} ` +
        `holeCentre=[${geom.holeCentre.map((v) => v.toFixed(3))}] holeRadiusDeg=${geom.holeRadiusDeg.toFixed(2)} ` +
        `solidAngleFrac=${(geom.solidAngleFraction * 100).toFixed(2)}% meanHeadRadius=${geom.meanHeadRadius.toFixed(2)} ` +
        `cavityVolumeHonest=${honest.toFixed(4)} unreachedByProperUnwrap=${geom.unreached}`,
    )
    console.log(
      `SHELL-PORE-UNWRAP step=${step} box=${box[0]} boxHalf=${(box[0] / 2).toFixed(1)} edges=${agg.edges.length} ` +
        `BROKEN(unwrapAggregate) meanBondLen=${brokenSanity.meanBondLength.toFixed(3)} maxBondLen=${brokenSanity.maxBondLength.toFixed(2)} broken(>${brokenSanity.brokenThreshold}sigma)=${brokenSanity.brokenCount} Rg=${brokenShape.radiusOfGyration.toFixed(3)} | ` +
        `FIXED(properUnwrap) meanBondLen=${fixedSanity.meanBondLength.toFixed(3)} maxBondLen=${fixedSanity.maxBondLength.toFixed(2)} broken(>${fixedSanity.brokenThreshold}sigma)=${fixedSanity.brokenCount} Rg=${geom.radiusOfGyration.toFixed(3)}`,
    )
    return { step, agg, geom, honest, positions, fixedSanity, brokenSanity, brokenShape }
  })

  expect(perStep.length).toBe(10)
  for (const p of perStep) {
    expect(p.agg.aggAmphiphiles.length).toBeGreaterThan(1000) // sanity: same ~1282-amphiphile cup at every step
    expect(p.geom.holeRadiusDeg).toBeGreaterThan(0)
    expect(p.geom.unreached).toBe(0) // properUnwrap must reach every member via the SAME graph that defined the aggregate
    // properUnwrap() removes MOST of unwrapAggregate()'s own single-reference wrap failures (see the
    // BROKEN vs FIXED counts logged above) but not all of them: this aggregate's own diameter (2x
    // meanHeadRadius ~59-64 sigma) is comparable to or EXCEEDS the box's own side length (58 sigma),
    // so the object cannot fit inside one periodic image of the box at all -- a genuine topological
    // wraparound, not an unwrap-algorithm defect, and no single-frame unwrap (BFS-based or otherwise)
    // can fully resolve it. The residual is real and reported, not hidden: expect it to be SMALL
    // relative to the broken method's own count, not necessarily zero.
    expect(p.fixedSanity.brokenCount).toBeLessThan(p.brokenSanity.brokenCount)
    expect(p.brokenSanity.brokenCount).toBeGreaterThan(0) // documents that unwrapAggregate() genuinely does break here
  }

  // --- Task 2: consecutive-snapshot hole-direction correlation ----------------------------------
  const consecutive = perStep.slice(1).map((p, i) => ({
    fromStep: perStep[i].step,
    toStep: p.step,
    angleDeg: angleDeg(perStep[i].geom.holeCentre, p.geom.holeCentre),
  }))
  const meanConsecutiveAngle = consecutive.reduce((s, c) => s + c.angleDeg, 0) / consecutive.length
  const netDrift = angleDeg(perStep[0].geom.holeCentre, perStep[perStep.length - 1].geom.holeCentre)
  console.log(
    `SHELL-PORE-CORR consecutive angles(deg)=[${consecutive.map((c) => c.angleDeg.toFixed(1)).join(',')}] mean=${meanConsecutiveAngle.toFixed(2)} netDriftFirstToLast=${netDrift.toFixed(2)}`,
  )
  expect(Number.isFinite(meanConsecutiveAngle)).toBe(true)

  // --- Task 1: radial head/tail profile on the LAST checkpoint (step 250000, the same one
  // kappa-tightening-report.md's own SS2 measured lambda on) and the FIRST of the ten (step 160000)
  // as a within-window robustness check. -----------------------------------------------------------
  const last = perStep[perStep.length - 1]
  const first = perStep[0]
  const radialLast = radialProfile(last.agg, last.geom, last.positions, monomers, 60)
  const radialFirst = radialProfile(first.agg, first.geom, first.positions, monomers, 60)
  const sepLast = radialLast.peaks ? radialLast.peaks.upper - radialLast.peaks.lower : null
  const sepFirst = radialFirst.peaks ? radialFirst.peaks.upper - radialFirst.peaks.lower : null
  console.log(
    `SHELL-PORE-RADIAL step=${last.step} innerPeak=${radialLast.peaks?.lower.toFixed(3)} outerPeak=${radialLast.peaks?.upper.toFixed(3)} ` +
      `separation=${sepLast?.toFixed(3)} bilayerThicknessRef=4.457 maxR=${radialLast.maxR.toFixed(2)}`,
  )
  console.log(
    `SHELL-PORE-RADIAL step=${first.step} innerPeak=${radialFirst.peaks?.lower.toFixed(3)} outerPeak=${radialFirst.peaks?.upper.toFixed(3)} ` +
      `separation=${sepFirst?.toFixed(3)} bilayerThicknessRef=4.457 maxR=${radialFirst.maxR.toFixed(2)}`,
  )
  expect(radialLast.peaks).not.toBeNull()

  // --- Robustness check: does properUnwrap's own shape reading depend on which particle it starts
  // BFS from? It should NOT, if the underlying proximity graph is topologically consistent; the
  // residual ~1.6-2% broken-bond count already found above says it is not QUITE consistent (this
  // aggregate's own diameter is comparable to the box), so this checks how much that residual
  // inconsistency actually moves the headline shape numbers, not just its bond-length symptom. -----
  const altRootLocal = Math.floor(last.agg.aggOriginalIdx.length / 2)
  const altUnwrap = properUnwrap(last.agg.aggOriginalIdx, last.positions, loadCheckpoint(last.step).box, last.agg.cutoff, altRootLocal)
  const altShape = shapeOf(altUnwrap.unwrapped)
  const altFlatness = altShape.principalMoments[2] > 1e-12 ? altShape.principalMoments[0] / altShape.principalMoments[2] : 1
  console.log(
    `SHELL-PORE-ROOTCHECK step=${last.step} root=0 Rg=${last.geom.radiusOfGyration.toFixed(3)} flatness=${last.geom.flatnessRatio.toFixed(4)} | ` +
      `root=${altRootLocal} Rg=${altShape.radiusOfGyration.toFixed(3)} flatness=${altFlatness.toFixed(4)} unreached=${altUnwrap.unreached}`,
  )

  // --- Task 3: plugged counterfactual vs the honest measurement, on step 250000 -----------------
  // Both the unplugged and plugged flood below run on the SAME frame (this file's own properUnwrap
  // output, geom.unwrapped) for a fair apples-to-apples comparison -- analyzeAggregates()'s own
  // cavityVolume (last.honest, printed above as "cavityVolumeHonest") uses the PROJECT'S own internal
  // unwrapAggregate() (the broken-at-this-scale method) and is reported separately, as the project's
  // own official reading, not mixed into this comparison.
  const innerRadius = radialLast.peaks ? radialLast.peaks.lower : last.geom.meanHeadRadius
  const analyticPlugged = (4 / 3) * Math.PI * innerRadius ** 3
  const unplugged = localCavityVolumeWrapper(last.geom.unwrapped, thresholds.closureCell, thresholds.closureRadius)
  const floodPlugged = pluggedFlood(last.geom, thresholds.closureCell, thresholds.closureRadius)
  console.log(
    `SHELL-PORE-VOLUME step=${last.step} projectOfficialHonest(brokenUnwrap)=${last.honest.toFixed(4)} ` +
      `sameFrameUnplugged(properUnwrap)=${unplugged.toFixed(4)} pluggedAnalytic(4/3*pi*r_inner^3, r_inner=${innerRadius.toFixed(3)})=${analyticPlugged.toFixed(2)} ` +
      `pluggedFlood(sameFrame)=${floodPlugged.toFixed(2)} closureThreshold=${thresholds.enclosedVolume}`,
  )
  expect(floodPlugged).toBeGreaterThanOrEqual(unplugged) // sealing the hole should never shrink the enclosed volume, same-frame comparison

  // --- Task 4: rim-vs-bulk head:tail ratio and local area-per-lipid, on step 250000, two margins --
  const rimBulk15 = localRimVsBulk(last.agg, last.geom, last.positions, monomers, 15)
  const rimBulk25 = localRimVsBulk(last.agg, last.geom, last.positions, monomers, 25)
  for (const rb of [rimBulk15, rimBulk25]) {
    console.log(
      `SHELL-PORE-RIMBULK step=${last.step} margin=${rb.marginDeg} ` +
        `rim[heads=${rb.rim.heads} tails=${rb.rim.tails} h:t=${rb.rim.headTailRatio.toFixed(3)} aLipid=${rb.rim.areaPerLipid.toFixed(3)} meanR=${rb.rim.meanRadius.toFixed(2)}] ` +
        `antipodal[heads=${rb.antipodal.heads} tails=${rb.antipodal.tails} h:t=${rb.antipodal.headTailRatio.toFixed(3)} aLipid=${rb.antipodal.areaPerLipid.toFixed(3)} meanR=${rb.antipodal.meanRadius.toFixed(2)}] ` +
        `rest[heads=${rb.rest.heads} tails=${rb.rest.tails} h:t=${rb.rest.headTailRatio.toFixed(3)} aLipid=${rb.rest.areaPerLipid.toFixed(3)} meanR=${rb.rest.meanRadius.toFixed(2)}]`,
    )
  }

  // --- Independent, unwrap-FREE confirmation that the aggregate's own footprint is comparable to
  // the box: the true minimum-image (periodic-aware) distance between any TWO given particles is
  // always well-defined, with no global-frame ambiguity at all, since it only involves one pair. A
  // random sample of many such pairwise distances approximates the aggregate's own true diameter
  // without needing any unwrap, sanity-checking the box/2=29-vs-meanHeadRadius~29.6 comparison above
  // from a completely different angle. -----------------------------------------------------------
  {
    const { positions, box } = loadCheckpoint(last.step)
    const idx = last.agg.aggOriginalIdx
    let maxD = 0
    let sumD = 0
    const samples = 20000
    for (let s = 0; s < samples; s++) {
      const i = idx[Math.floor(Math.random() * idx.length)]
      const j = idx[Math.floor(Math.random() * idx.length)]
      const dx = mi1(positions[i * 4] - positions[j * 4], box[0])
      const dy = mi1(positions[i * 4 + 1] - positions[j * 4 + 1], box[1])
      const dz = mi1(positions[i * 4 + 2] - positions[j * 4 + 2], box[2])
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz)
      if (d > maxD) maxD = d
      sumD += d
    }
    console.log(
      `SHELL-PORE-DIAMETER step=${last.step} box=${box[0]} boxHalfDiag(sqrt3*box/2)=${((Math.sqrt(3) * box[0]) / 2).toFixed(2)} ` +
        `sampledMaxPeriodicPairDist(n=${samples})=${maxD.toFixed(2)} meanPeriodicPairDist=${(sumD / samples).toFixed(2)}`,
    )
  }

  mkdirSync(OUT_DIR, { recursive: true })
  writeFileSync(
    OUT_FILE,
    JSON.stringify(
      {
        gate: 'shell-pore-check',
        steps: STEPS,
        perStep: perStep.map((p) => ({
          step: p.step,
          aggregateCount: p.agg.aggregateCount,
          amphiphiles: p.agg.aggAmphiphiles.length,
          particles: p.agg.aggOriginalIdx.length,
          radiusOfGyration: p.geom.radiusOfGyration,
          flatnessRatio: p.geom.flatnessRatio,
          holeCentre: p.geom.holeCentre,
          holeRadiusDeg: p.geom.holeRadiusDeg,
          solidAngleFraction: p.geom.solidAngleFraction,
          meanHeadRadius: p.geom.meanHeadRadius,
          cavityVolumeHonest: p.honest,
          unreachedByProperUnwrap: p.geom.unreached,
          unwrapComparison: {
            broken: { meanBondLength: p.brokenSanity.meanBondLength, maxBondLength: p.brokenSanity.maxBondLength, brokenCount: p.brokenSanity.brokenCount, edgeCount: p.agg.edges.length, radiusOfGyration: p.brokenShape.radiusOfGyration },
            fixed: { meanBondLength: p.fixedSanity.meanBondLength, maxBondLength: p.fixedSanity.maxBondLength, brokenCount: p.fixedSanity.brokenCount, radiusOfGyration: p.geom.radiusOfGyration },
          },
        })),
        consecutiveHoleAngles: consecutive,
        meanConsecutiveAngleDeg: meanConsecutiveAngle,
        netDriftFirstToLastDeg: netDrift,
        radialProfile: {
          step: last.step,
          bins: radialLast.profile.z.length,
          z: radialLast.profile.z,
          head: radialLast.profile.head,
          tail: radialLast.profile.tail,
          innerPeak: radialLast.peaks?.lower ?? null,
          outerPeak: radialLast.peaks?.upper ?? null,
          separation: sepLast,
          bilayerThicknessReference: 4.457,
        },
        radialProfileFirst: { step: first.step, innerPeak: radialFirst.peaks?.lower ?? null, outerPeak: radialFirst.peaks?.upper ?? null, separation: sepFirst },
        volumes: {
          step: last.step,
          projectOfficialHonest_brokenUnwrap: last.honest,
          sameFrameUnplugged_properUnwrap: unplugged,
          pluggedAnalytic: analyticPlugged,
          pluggedFlood_sameFrame: floodPlugged,
          innerRadiusUsed: innerRadius,
          closureThreshold: thresholds.enclosedVolume,
        },
        rimVsBulk: [rimBulk15, rimBulk25],
      },
      null,
      2,
    ),
  )
  console.log(`SHELL-PORE artifact written: ${OUT_FILE}`)
})
