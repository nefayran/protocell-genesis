import { getGpu, readBack, storageBuffer } from './gpu'
import {
  areaPerLipid,
  bilayerThickness,
  centerMembraneZ,
  densityProfileZ,
  dropEscapedZ,
  sumProfiles,
  type ZProfile,
} from './metrics'
import type { System } from './sim'

export { probeForces } from './forces'
export { createSystem } from './sim'
export type { CreateSystemOpts, Layout, System } from './sim'
export {
  areaPerLipid,
  bilayerPeaks,
  bilayerThickness,
  centerMembraneZ,
  densityProfileZ,
  dropEscapedZ,
  sumProfiles,
} from './metrics'
export type { ZProfile } from './metrics'

/** Facade for Task 5's structural gate: current area per lipid and bilayer thickness (200-bin
 * z-density profile of head beads), plus enough bookkeeping (box, lipids, cumulative steps) for
 * a caller to report a full trajectory without re-deriving it from raw positions each time. */
export async function measureBilayer(sys: System): Promise<{
  areaPerLipid: number
  thickness: number
  box: [number, number, number]
  lipids: number
  steps: number
}> {
  const pos = await sys.positions()
  const box = sys.box
  const lipids = sys.lipids
  const profile = densityProfileZ(pos, box, 200)
  return {
    areaPerLipid: areaPerLipid(box, lipids),
    thickness: bilayerThickness(profile),
    box,
    lipids,
    steps: sys.steps,
  }
}

/** Ordinary-least-squares slope of `ys` against index, with the standard error taken from BLOCK
 * means rather than the raw series. A Monte Carlo trajectory of ln A is strongly autocorrelated —
 * consecutive moves change it by at most the proposal width — so a naive OLS standard error is
 * far too small and reports a wildly significant drift for a perfectly stationary chain (measured:
 * |t| up to 35 on windows whose slopes alternated in sign). Averaging into blocks long enough to
 * decorrelate first, then regressing the block means, is the standard fix. */
function blockDrift(ys: number[], blocks: number): { perStep: number; stdErr: number; t: number } {
  const per = Math.floor(ys.length / blocks)
  if (per < 2 || blocks < 4) throw new Error(`blockDrift: серия из ${ys.length} точек мала для ${blocks} блоков`)
  const xs: number[] = []
  const ms: number[] = []
  for (let b = 0; b < blocks; b++) {
    let s = 0
    for (let i = b * per; i < (b + 1) * per; i++) s += ys[i]
    ms.push(s / per)
    xs.push(b * per + (per - 1) / 2)
  }
  const xm = xs.reduce((a, b) => a + b, 0) / blocks
  const ym = ms.reduce((a, b) => a + b, 0) / blocks
  let sxy = 0
  let sxx = 0
  for (let b = 0; b < blocks; b++) {
    sxy += (xs[b] - xm) * (ms[b] - ym)
    sxx += (xs[b] - xm) ** 2
  }
  const slope = sxy / sxx
  let sse = 0
  for (let b = 0; b < blocks; b++) sse += (ms[b] - ym - slope * (xs[b] - xm)) ** 2
  const stdErr = Math.sqrt(sse / (blocks - 2) / sxx)
  return { perStep: slope, stdErr, t: slope / stdErr }
}

/** Ensemble measurement of the same two structural numbers `measureBilayer` reads off one frame:
 * area per lipid and bilayer thickness, sampled over many configurations while the area coordinate
 * keeps moving. Advances the system — `samples` rounds of `stepsPerSample` integration steps plus
 * `areaTrialsPerSample` area moves — and returns
 *  - area per lipid averaged over samples, with its scatter and its extremes over the window,
 *  - thickness from the ACCUMULATED head-density histogram (peaks located by sub-bin parabolic
 *    interpolation), which is the honest estimator: one frame puts ~500 heads per leaflet into
 *    0.2-sigma bins and its peak-to-peak distance scatters by about +/-0.23 sigma frame to frame,
 *    enough to make a single-frame reading land on a literature bound by luck alone,
 *  - the per-sample thickness mean and standard deviation, so that scatter is visible rather than
 *    hidden by the averaging,
 *  - the drift of ln A per move with a block-based standard error, which is what says whether the
 *    area coordinate is at equilibrium or still relaxing,
 *  - the largest number of beads that had left the box in z at any sample (see dropEscapedZ). */
export async function measureBilayerAveraged(
  sys: System,
  opts: { samples: number; stepsPerSample: number; areaTrialsPerSample?: number; bins?: number; blocks?: number },
): Promise<{
  areaPerLipid: number
  areaPerLipidSd: number
  areaPerLipidMin: number
  areaPerLipidMax: number
  thickness: number
  thicknessMean: number
  thicknessSd: number
  perSampleThickness: number[]
  lnADriftPerMove: number
  lnADriftStdErr: number
  lnADriftT: number
  acceptedFraction: number
  escapedMax: number
  box: [number, number, number]
  lipids: number
  steps: number
  samples: number
}> {
  const bins = opts.bins ?? 200
  const trials = opts.areaTrialsPerSample ?? 1
  const profiles: ZProfile[] = []
  const perSampleThickness: number[] = []
  const lnA: number[] = []
  const areas: number[] = []
  let accepted = 0
  let escapedMax = 0

  for (let s = 0; s < opts.samples; s++) {
    await sys.step(opts.stepsPerSample)
    accepted += (await sys.areaMove(trials)) * trials
    const box = sys.box
    const raw = await sys.positions()
    const { positions, escaped } = dropEscapedZ(centerMembraneZ(raw, box), box)
    escapedMax = Math.max(escapedMax, escaped)
    const profile = densityProfileZ(positions, box, bins)
    profiles.push(profile)
    perSampleThickness.push(bilayerThickness(profile))
    lnA.push(Math.log(box[0] * box[1]))
    areas.push(areaPerLipid(box, sys.lipids))
  }

  const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length
  const sd = (xs: number[]) => {
    const m = mean(xs)
    return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
  }
  const drift = blockDrift(lnA, opts.blocks ?? 20)

  return {
    areaPerLipid: mean(areas),
    areaPerLipidSd: sd(areas),
    areaPerLipidMin: Math.min(...areas),
    areaPerLipidMax: Math.max(...areas),
    thickness: bilayerThickness(sumProfiles(profiles)),
    thicknessMean: mean(perSampleThickness),
    thicknessSd: sd(perSampleThickness),
    perSampleThickness,
    lnADriftPerMove: drift.perStep,
    lnADriftStdErr: drift.stdErr,
    lnADriftT: drift.t,
    acceptedFraction: accepted / (opts.samples * trials),
    escapedMax,
    box: sys.box,
    lipids: sys.lipids,
    steps: sys.steps,
    samples: opts.samples,
  }
}

export async function gpuSmoke() {
  const { device, adapterInfo } = await getGpu()
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
  b[i.x] = b[i.x] * 2.0;
}`,
  })
  const buf = storageBuffer(device, new Float32Array([1, 2, 3, 4]))
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } })
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: buf } }],
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bind)
  pass.dispatchWorkgroups(1)
  pass.end()
  device.queue.submit([enc.finish()])
  let doubled: number[]
  try {
    doubled = Array.from(await readBack(device, buf, 16))
  } finally {
    buf.destroy()
  }
  return { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture, doubled }
}
