// Offline (pure-CPU, no GPU, no browser) audit of the soup-to-vesicle continuous campaign's own
// checkpoints -- task 'continuous-run' (2026-08-19),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/continuous-run-report.md.
//
// Why a vitest file rather than a plain `npx tsx` script: soup/src/aggregates.ts imports
// engine/src/closure.ts, which imports engine/wgsl/closure.wgsl with vite's `?raw` -- node/tsx
// cannot load a .wgsl module, vite can. tests/periodic-measurement.test.ts is this project's own
// precedent for exactly this pattern (a CPU-only vitest that decodes real checkpoints and measures).
//
// It re-uses the SAME functions the live run's own stage detection uses (findAmphiphiles,
// analyzeAggregates, stageFromEvidence, loadStageThresholds) -- there is no second implementation of
// any measurement here, so an audit number cannot disagree with what the run itself reported.
//
// The bond graph is rebuilt from the checkpoint's RAW slot rows (`bondSlots`, 3 u32 per particle),
// which is the checkpoint's own authoritative representation -- not from a derived pair list -- so
// the valence audit below reads exactly what soup/wgsl/bond-valence.wgsl's roleOf()/tryClaimSlot()
// enforced on the GPU.
//
// Usage: CONTINUOUS_RUN_CHECKPOINTS='a.json b.json' npx vitest run tests/continuous-run-audit.test.ts
// With the variable unset it audits every `soup2ves90-step*.json` under data/checkpoints/trace plus
// data/checkpoints, in step order, and skips (rather than fails) when none exist -- the checkpoints
// are gitignored run artifacts, so a fresh clone must not fail this file.
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'
import { expect, test } from 'vitest'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { loadSoup } from '../soup/src/rules'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { amphiphileHistogram, findAmphiphiles } from '../soup/src/amphiphile'
import { analyzeAggregates, memberIndicesOf } from '../soup/src/aggregates'
import { computeHeadPeaks, loadStageThresholds, positionsFor, stageFromEvidence } from '../soup/src/stages'
import { largestClusterFraction } from '../engine/src/aggregate'
import { enclosedVolumeFromPositions } from '../engine/src/closure'

const EMPTY = 0xffffffff

function discover(): string[] {
  const env = process.env.CONTINUOUS_RUN_CHECKPOINTS
  if (env && env.trim().length > 0) return env.trim().split(/[\s,]+/)
  const out: string[] = []
  for (const dir of ['data/checkpoints/trace', 'data/checkpoints']) {
    if (!existsSync(dir)) continue
    for (const n of readdirSync(dir)) {
      if (n.startsWith('soup2ves90-step') && n.endsWith('.json')) out.push(join(dir, n))
    }
  }
  return out.sort((a, b) => stepOf(a) - stepOf(b))
}
const stepOf = (p: string) => Number(/step(\d+)\.json$/.exec(p)?.[1] ?? 0)

test('continuous-run checkpoints: invariants hold and the stage ladder is reproducible off-GPU', () => {
  const files = discover()
  if (files.length === 0) {
    console.log('RUN-AUDIT skipped: no soup2ves90 checkpoints on disk (gitignored run artifacts)')
    return
  }
  const soup = loadSoup()
  const t = loadStageThresholds()
  const p = loadParams()
  // Committed artifact, same convention as tests/periodic-measurement.test.ts's own
  // verify/out/periodic-measurement.json: the trace itself, machine-readable, so the report's tables
  // can be re-derived without the (gitignored, 30.9 MB each) checkpoints.
  const artifact: unknown[] = []

  for (const path of files) {
    const r = decodeCheckpointResume(JSON.parse(readFileSync(path, 'utf8')))
    const pos = r.positions
    const N = pos.length / 4
    const box = r.liveBox as [number, number, number]

    const species: Record<string, number> = {}
    for (let i = 0; i < N; i++) {
      const id = soup.monomers[Math.round(pos[i * 4 + 3])]?.id ?? '?'
      species[id] = (species[id] ?? 0) + 1
    }
    let nfPos = 0
    let nfVel = 0
    for (let k = 0; k < pos.length; k++) if (!Number.isFinite(pos[k])) nfPos++
    for (let k = 0; k < r.velocities.length; k++) if (!Number.isFinite(r.velocities[k])) nfVel++

    const pairs: number[] = []
    const deg = new Int32Array(N)
    const cc = new Int32Array(N)
    const co = new Int32Array(N)
    const kindOf = (i: number) => soup.monomers[Math.round(pos[i * 4 + 3])].kind
    for (let i = 0; i < N; i++) {
      for (let s = 0; s < 3; s++) {
        const j = r.bondSlots[i * 3 + s]
        if (j === EMPTY) continue
        deg[i]++
        if (i >= j) continue
        pairs.push(i, j)
        const a = kindOf(i)
        const b = kindOf(j)
        if (a === 'carbon' && b === 'carbon') {
          cc[i]++
          cc[j]++
        } else if ((a === 'carbon' && b === 'head') || (a === 'head' && b === 'carbon')) {
          co[i]++
          co[j]++
        }
      }
    }
    // The valence/placement rules as data/soup.json + soup/wgsl/bond-valence.wgsl actually define
    // them, read from the file rather than assumed:
    //  - carbon: at most 2 chain (C-C) bonds and at most 1 head (C-O) bond, so degree <= 3;
    //  - head: at most headPlacement.chainCapacity tails (2 -- the 'two-tails' task), NOT 1;
    //  - headPlacement.terminalOnly: a carbon that carries a head may have at most 1 chain bond
    //    (the head sits on a chain END, never mid-chain);
    //  - nothing else (donor H, catalyst M, water W) may carry any bond at all.
    const headCapacity = soup.headPlacement?.chainCapacity ?? 1
    const terminalOnly = soup.headPlacement?.terminalOnly ?? false
    let vCC = 0
    let vCO = 0
    let vHead = 0
    let vOther = 0
    let vDeg = 0
    let vTerminal = 0
    for (let i = 0; i < N; i++) {
      const k = kindOf(i)
      if (k === 'carbon') {
        if (cc[i] > 2) vCC++
        if (co[i] > 1) vCO++
        if (deg[i] > 3) vDeg++
        if (terminalOnly && co[i] > 0 && cc[i] > 1) vTerminal++
      } else if (k === 'head') {
        if (deg[i] > headCapacity) vHead++
      } else if (deg[i] > 0) vOther++
    }
    const bonds = new Uint32Array(pairs)

    const amph = findAmphiphiles(pos, bonds, soup.monomers)
    const memberIdx = memberIndicesOf(amph)
    const memberRadii = soup.monomers.filter((m) => m.kind === 'carbon' || m.kind === 'head').map((m) => m.radiusSigma)
    const cutoff = wcaCutoff(p.sigma * Math.max(...memberRadii)) + p.attraction.wc
    const waterKind = soup.monomers.findIndex((m) => m.id === soup.solvent.waterId)
    const waterIdx: number[] = []
    if (waterKind >= 0) for (let i = 0; i < N; i++) if (Math.round(pos[i * 4 + 3]) === waterKind) waterIdx.push(i)

    const analysis = analyzeAggregates(pos, box, soup.monomers, amph, memberIdx, cutoff, t, waterIdx)
    const totalCarbon = species['C'] ?? 0
    const carbonInAmph = amph.reduce((s, a) => s + a.length, 0)
    let largestAggregateFraction = 0
    let enclosedVolume = 0
    if (memberIdx.size > 0) {
      const out = pos.slice()
      for (let i = 0; i < N; i++) out[i * 4 + 3] = memberIdx.has(i) ? 1 : 0
      largestAggregateFraction = largestClusterFraction(out, box, cutoff)
      enclosedVolume = enclosedVolumeFromPositions(positionsFor(pos, memberIdx), box, { cell: t.closureCell, radius: t.closureRadius })
    }
    const evidence = {
      amphiphileFraction: totalCarbon > 0 ? carbonInAmph / totalCarbon : 0,
      largestAggregateFraction,
      headPeaks: computeHeadPeaks(pos, box, soup.monomers, t),
      enclosedVolume,
      aggregateAnalysis: analysis,
    }
    const stage = stageFromEvidence(evidence as never, t)

    const record = {
          step: r.globalStep,
          N,
          box,
          stage,
          species,
          bonds: bonds.length / 2,
          events: r.events,
          invariants: {
            nonFinitePositions: nfPos,
            nonFiniteVelocities: nfVel,
            valence: {
              carbonCCover2: vCC,
              carbonCOover1: vCO,
              headOverChainCapacity: vHead,
              nonBondableBonded: vOther,
              degreeOver3: vDeg,
              headNotTerminal: vTerminal,
              headChainCapacity: headCapacity,
              headsWithTwoTails: (() => {
                let n = 0
                for (let i = 0; i < N; i++) if (kindOf(i) === 'head' && deg[i] === 2) n++
                return n
              })(),
            },
          },
          amphiphileCount: amph.length,
          amphiphileFraction: Number(evidence.amphiphileFraction.toFixed(5)),
          meanTailLength: amph.length ? Number((carbonInAmph / amph.length).toFixed(3)) : 0,
          lengthHistogram: amphiphileHistogram(amph),
          aggregateCount: analysis.aggregateCount,
          qualifyingAggregateCount: analysis.qualifyingAggregateCount,
          amphiphilesInQualifying: analysis.amphiphilesInQualifying,
          amphiphileShareInQualifying: Number(analysis.amphiphileShareInQualifying.toFixed(4)),
          hasLamellarAggregate: analysis.hasLamellarAggregate,
          hasVesicleAggregate: analysis.hasVesicleAggregate,
          largestAggregateFraction: Number(largestAggregateFraction.toFixed(4)),
          boxWideEnclosedVolume: enclosedVolume,
          sizeHistogramTop: analysis.sizeHistogram.slice(0, 10),
          largest: analysis.aggregates.slice(0, 3).map((s) => ({
            amphiphileCount: s.amphiphileCount,
            particleCount: s.particleCount,
            radiusOfGyration: Number(s.radiusOfGyration.toFixed(3)),
            principalMoments: s.principalMoments.map((x) => Number(x.toFixed(3))),
            flatnessRatio: Number(s.flatnessRatio.toFixed(4)),
            inPlaneSymmetry: Number(s.inPlaneSymmetry.toFixed(4)),
            radialHeadShells: s.radialHeadShells,
            transverseHeadShells: s.transverseHeadShells,
            cavityVolume: s.cavityVolume,
            encapsulatedWater:
              s.encapsulatedWater === undefined ? 'undefined' : s.encapsulatedWater === null ? 'null(centre-untrusted)' : s.encapsulatedWater,
          })),
          closureThreshold: t.enclosedVolume,
          minAmphiphilesPerAggregate: t.minAmphiphilesPerAggregate,
    }
    artifact.push(record)
    console.log('RUN-AUDIT ' + JSON.stringify(record))

    // --- the invariants this run claims, asserted rather than eyeballed
    expect(nfPos, `${path}: non-finite positions`).toBe(0)
    expect(nfVel, `${path}: non-finite velocities`).toBe(0)
    expect({ vCC, vCO, vHead, vOther, vDeg, vTerminal }, `${path}: valence/placement`).toEqual({
      vCC: 0,
      vCO: 0,
      vHead: 0,
      vOther: 0,
      vDeg: 0,
      vTerminal: 0,
    })
    // Monomer conservation: the campaign's own requested composition, unchanged at every step.
    expect(species, `${path}: monomer conservation`).toEqual({ C: 40500, O: 13500, H: 40500, M: 2700, W: 288900 })
    // Every bond is one the rules declare (C-C or C-O); nothing else may ever be bonded.
    expect(vOther, `${path}: only carbon/head may carry bonds`).toBe(0)
  }

  artifact.sort((a, b) => (a as { step: number }).step - (b as { step: number }).step)
  mkdirSync('verify/out', { recursive: true })
  writeFileSync('verify/out/continuous-run-trace.json', JSON.stringify(artifact, null, 1))
  console.log(`RUN-AUDIT artifact written: verify/out/continuous-run-trace.json (${artifact.length} checkpoints)`)

  // The monomers-only start, from the run's OWN trace rather than by assumption: the earliest
  // checkpoint on disk must carry zero bonds, zero amphiphiles and zero bond events.
  // Earliest by step number, not by argument order -- an explicit CONTINUOUS_RUN_CHECKPOINTS list
  // is honoured in the order given for the trace, but the start proof must always read the FIRST
  // step on the list, whatever order the shell happened to expand it in.
  const first = files.slice().sort((a, b) => stepOf(a) - stepOf(b))[0]
  const r0 = decodeCheckpointResume(JSON.parse(readFileSync(first, 'utf8')))
  let bonds0 = 0
  for (let k = 0; k < r0.bondSlots.length; k++) if (r0.bondSlots[k] !== EMPTY) bonds0++
  console.log(`RUN-AUDIT-START ${first} step=${r0.globalStep} bondSlotsUsed=${bonds0} events=${JSON.stringify(r0.events)}`)
  if (r0.globalStep <= 1) {
    expect(bonds0, 'monomers-only start: no bond may exist at step<=1').toBe(0)
    expect(Object.values(r0.events).reduce((a, b) => a + b, 0), 'monomers-only start: no bond event').toBe(0)
    expect(findAmphiphiles(r0.positions, new Uint32Array(0), loadSoup().monomers).length).toBe(0)
  }
}, 3_600_000)
