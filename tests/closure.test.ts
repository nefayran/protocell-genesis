import { afterAll, expect, test } from 'vitest'
import { cavities, cavitiesFromPositions, dimsFor, enclosedVolume, enclosedVolumeFromPositions, occupancy } from '../engine/src/closure'
import { loadStageThresholds, stageFromEvidence } from '../soup/src/stages'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

function shell(radius: number, thickness: number, count: number): Float32Array {
  const out: number[] = []
  for (let i = 0; i < count; i++) {
    const u = Math.random() * 2 - 1
    const phi = Math.random() * 2 * Math.PI
    const r = radius + (Math.random() - 0.5) * thickness
    const s = Math.sqrt(1 - u * u)
    out.push(20 + r * s * Math.cos(phi), 20 + r * s * Math.sin(phi), 20 + r * u, 1)
  }
  return new Float32Array(out)
}

/** Translates every bead's x,y by (dx,dy), wrapping into [0, box) -- used to move the synthetic
 * shell's centre away from the box midpoint and toward/across a periodic face, without touching z
 * (z is never periodic in this engine, so an unbounded shift there would push beads outside the
 * domain instead of testing anything about the periodic-flood fix). */
function shiftXY(positions: Float32Array, box: [number, number, number], dx: number, dy: number): Float32Array {
  const out = positions.slice()
  for (let i = 0; i < out.length; i += 4) {
    out[i] = ((out[i] + dx) % box[0] + box[0]) % box[0]
    out[i + 1] = ((out[i + 1] + dy) % box[1] + box[1]) % box[1]
  }
  return out
}

test('замкнутая оболочка даёт полость близкую к объёму шара', () => {
  const box: [number, number, number] = [40, 40, 40]
  const occ = occupancy(shell(8, 1.5, 40_000), box, 0.5, 0.6)
  const v = enclosedVolume(occ, [80, 80, 80], 0.5)
  const ideal = (4 / 3) * Math.PI * 7.5 ** 3
  expect(v).toBeGreaterThan(ideal * 0.7)
  expect(v).toBeLessThan(ideal * 1.3)
})

test('оболочка, пересекающая границу по x, даёт ту же полость после рецентровки', () => {
  const box: [number, number, number] = [40, 40, 40]
  const centered = shell(8, 1.5, 40_000)
  const straddling = shiftXY(centered, box, -20, 0) // centre moves from x=20 to x=0: half the shell wraps to x~38-40
  const vCentered = enclosedVolumeFromPositions(centered, box, { cell: 0.5, radius: 0.6 })
  const vStraddling = enclosedVolumeFromPositions(straddling, box, { cell: 0.5, radius: 0.6 })
  // Without recentring, the open-boundary flood would seed x=0 and x=nx-1 as outside, splitting
  // this shell's two halves apart at the seam and leaking almost the whole cavity out through it
  // (this is the exact bug: v_straddling would collapse toward 0 instead of matching v_centered).
  expect(vStraddling).toBeGreaterThan(vCentered * 0.7)
  expect(vStraddling).toBeLessThan(vCentered * 1.3)
})

test('оболочка в углу коробки (x и y сразу) даёт ту же полость после рецентровки', () => {
  const box: [number, number, number] = [40, 40, 40]
  const centered = shell(8, 1.5, 40_000)
  const corner = shiftXY(centered, box, -20, -20) // centre moves from (20,20) to (0,0): both periodic faces at once
  const vCentered = enclosedVolumeFromPositions(centered, box, { cell: 0.5, radius: 0.6 })
  const vCorner = enclosedVolumeFromPositions(corner, box, { cell: 0.5, radius: 0.6 })
  expect(vCorner).toBeGreaterThan(vCentered * 0.7)
  expect(vCorner).toBeLessThan(vCentered * 1.3)
})

test('плоский лист полости не даёт', () => {
  const box: [number, number, number] = [40, 40, 40]
  const pos: number[] = []
  for (let i = 0; i < 20_000; i++) pos.push(Math.random() * 40, Math.random() * 40, 20 + (Math.random() - 0.5), 1)
  const occ = occupancy(new Float32Array(pos), box, 0.5, 0.6)
  expect(enclosedVolume(occ, [80, 80, 80], 0.5)).toBe(0)
})

// --- corrected closure criterion (task: "soup-to-vesicle" cavity-highlight): min:1 used to mean
// "the flood-fill left at least one cell unreached" -- a technical sanity check on the detector
// itself, not a physical statement about vesicles. The live run this task responds to reported
// enclosedVolume=1.1250 with 548 amphiphiles recognised: at cell=0.5sigma that is exactly 9 cells
// (1.125/0.5^3), a pocket between a few beads, and the OLD threshold called it a vesicle. These two
// tests pin the corrected threshold (data/literature.json's closure gate / data/soup.json's
// stageThresholds, kept equal by soup/src/stages.ts's loadStageThresholds() itself) against both
// ends: a real detector run on a genuinely large synthetic shell must pass, and the exact reported
// pocket number must not -- regardless of how saturated the rest of the stage ladder is.
test('исправленный порог closure: настоящая большая полость даёт vesicle, карман в 9 клеток из живого прогона — нет, независимо от насыщенности остальной лестницы', () => {
  const thresholds = loadStageThresholds()
  const ladderSaturated = { amphiphileFraction: 1, largestAggregateFraction: 1, headPeaks: 2 as const }

  // Genuine cavity: the SAME big synthetic shell as this file's very first test (radius 8, thickness
  // 1.5, 40_000 beads), run through the REAL detector (occupancy+enclosedVolume), not a fabricated
  // number. Its enclosed volume clears the new physically-grounded minimum by a wide margin.
  const box: [number, number, number] = [40, 40, 40]
  const bigVolume = enclosedVolumeFromPositions(shell(8, 1.5, 40_000), box, { cell: 0.5, radius: 0.6 })
  expect(bigVolume).toBeGreaterThan(thresholds.enclosedVolume)
  expect(stageFromEvidence({ ...ladderSaturated, enclosedVolume: bigVolume }, thresholds)).toBe('vesicle')

  // The bug this task fixes, pinned with the EXACT number the live run reported: enclosedVolume =
  // 1.1250 (9 flood-fill cells at cell=0.5sigma, cell^3=0.125). However saturated the amphiphile/
  // aggregation/head-peak ladder is, a 9-cell pocket must not read as a vesicle under the corrected
  // threshold -- and it must sit nowhere close to the new minimum, not just barely under it.
  const reportedPocketVolume = 1.125
  expect(reportedPocketVolume).toBeLessThan(thresholds.enclosedVolume / 100)
  expect(stageFromEvidence({ ...ladderSaturated, enclosedVolume: reportedPocketVolume }, thresholds)).not.toBe('vesicle')
  expect(stageFromEvidence({ ...ladderSaturated, enclosedVolume: reportedPocketVolume }, thresholds)).toBe('bilayer')
})

// cavities()/cavitiesFromPositions(): the per-cavity breakdown the viewer draws from. Pins that (a)
// the breakdown never invents or loses volume relative to the existing enclosedVolume() scalar --
// it is a labelling of the SAME unreached cells, not a second measurement -- and (b) the facade
// correctly undoes the internal recentring, so a cavity's reported centre/voxels land back in the
// SAME (possibly box-face-straddling) frame the caller's own positions were given in.
test('cavities() размечает полости без изменения суммарного объёма enclosedVolume()', () => {
  const box: [number, number, number] = [40, 40, 40]
  const positions = shell(8, 1.5, 40_000)
  const dims = dimsFor(box, 0.5)
  const occ = occupancy(positions, box, 0.5, 0.6)
  const comps = cavities(occ, dims, 0.5)
  const total = enclosedVolume(occ, dims, 0.5)

  expect(comps.length).toBeGreaterThanOrEqual(1)
  const summed = comps.reduce((s, c) => s + c.volume, 0)
  expect(summed).toBeCloseTo(total, 5)
  expect(comps[0].voxelCount).toBeGreaterThan(0)
  // this shell is centred at (20,20,20) in a 40^3 box -- its cavity's centre must sit near there.
  for (const axis of [0, 1, 2] as const) {
    expect(comps[0].centre[axis]).toBeGreaterThan(15)
    expect(comps[0].centre[axis]).toBeLessThan(25)
  }
})

test('cavitiesFromPositions переносит центр и вокселы полости обратно в исходную систему координат вызывающего', () => {
  const box: [number, number, number] = [40, 40, 40]
  const centered = shell(8, 1.5, 40_000)
  const straddling = shiftXY(centered, box, -20, 0) // same shift as this file's boundary-straddling test above
  const result = cavitiesFromPositions(straddling, box, { cell: 0.5, radius: 0.6 })

  expect(result.cavityCount).toBeGreaterThanOrEqual(1)
  const largest = result.cavities[0]
  expect(largest.voxelCentres.length).toBe(largest.voxelCount * 3)
  expect(largest.radius).toBeGreaterThan(0)
  // the shell's centre after the -20 shift on x sits near x=0/x=40 (wrapped), NOT near x=20 --
  // i.e. cavitiesFromPositions must undo its own internal recentring, not leave the cavity sitting
  // wherever the recentred flood happened to compute it.
  expect(largest.centre[0] < 3 || largest.centre[0] > 37).toBe(true)
})

test('версия на GPU совпадает с эталоном на TypeScript', async () => {
  const page = await gpuPage()
  const rel = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 2000, box: [40, 40, 40], seed: 23, layout: 'vesicle' })
    const cpu = await api.enclosedVolumeCpu(sys, { cell: 0.5, radius: 0.6 })
    const gpu = await api.enclosedVolumeGpu(sys, { cell: 0.5, radius: 0.6 })
    return Math.abs(gpu - cpu) / Math.max(cpu, 1)
  })
  expect(rel).toBeLessThan(0.01)
})
