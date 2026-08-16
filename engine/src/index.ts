import { clusters, largestClusterFraction } from './aggregate'
import { enclosedVolumeFromPositions, enclosedVolumeGpuDetailed, recenterOnLargestCluster } from './closure'
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
import { loadParams, wcaCutoff } from './params'
import type { System } from './sim'
import {
  columnNoiseStats,
  fitBendingModulus,
  heightField,
  logLogFit,
  selectFitWindow,
  spectrum,
  tailEndBeads,
  type Spectrum,
} from './spectrum'

export { probeForces } from './forces'
export { createSystem } from './sim'
export type { CreateSystemOpts, Layout, System } from './sim'
// Task 2: the soup's own bond-forming/breaking dynamics, re-exported here so tests/runner.html's
// `window.api` (a straight `import * as api from './index.ts'`) can reach it the same way it
// reaches createSystem.
export { createSoup } from '../../soup/src/sim'
export type { CreateSoupOpts, SoupSystem } from '../../soup/src/sim'
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
export { clusters, largestClusterFraction, largestClusterCenter } from './aggregate'
export {
  dimsFor,
  enclosedVolume,
  enclosedVolumeFromPositions,
  enclosedVolumeGpuDetailed,
  occupancy,
  recenterOnLargestCluster,
} from './closure'
export type { Dims, EnclosedVolumeGpuResult } from './closure'
export {
  columnNoiseStats,
  fitBendingModulus,
  heightField,
  logLogFit,
  selectFitWindow,
  spectrum,
  synthesizeHeightField,
  tailEndBeads,
} from './spectrum'
export type { ColumnNoiseStats, Spectrum } from './spectrum'

/** Facade for Task 6's self-assembly gate: snapshots `sys`'s current positions and box, then hands
 * them to largestClusterFraction with cutoff = r_c + w_c (the tail-tail attraction's full range —
 * the same reach the physics itself uses to pull two tails together), matching the cell size the
 * engine's own neighbor grid uses (see `cellSize` in sim.ts). */
export async function largestClusterFractionOf(sys: System): Promise<number> {
  const pos = await sys.positions()
  const p = loadParams()
  const cutoff = wcaCutoff(p.beadSizes.tail_tail) + p.attraction.wc
  return largestClusterFraction(pos, sys.box, cutoff)
}

/** Facade for Task 8's closure gate: snapshots `sys`'s current positions and box, RECENTRES on the
 * dominant connected structure (recenterOnLargestCluster -- a self-assembled vesicle forms
 * wherever it forms in a periodic box, and the flood's open boundary must not mistake "near a box
 * face" for "open"; see the periodic-flood note in closure.ts), then builds the occupancy grid and
 * floods it from the box's boundary on the CPU -- the reference implementation that
 * enclosedVolumeGpu below is checked against. */
export async function enclosedVolumeCpu(sys: System, opts: { cell: number; radius: number }): Promise<number> {
  const pos = await sys.positions()
  return enclosedVolumeFromPositions(pos, sys.box, opts)
}

/** Same measurement as enclosedVolumeCpu (including the same recentring step, so the two remain
 * comparable), computed by closure.wgsl on the GPU (occupancy via atomicOr, then iterative
 * outside-flood propagation with a change flag) -- see enclosedVolumeGpuDetailed in closure.ts for
 * the iteration count/wall-clock this task's brief asks to report, logged here rather than added
 * to this facade's return type since the interface this task specifies is `Promise<number>`. */
export async function enclosedVolumeGpu(sys: System, opts: { cell: number; radius: number }): Promise<number> {
  const pos = await sys.positions()
  const box = sys.box
  const centered = recenterOnLargestCluster(pos, box)
  const result = await enclosedVolumeGpuDetailed(centered, box, opts)
  console.log(
    `CLOSURE-GPU iterations=${result.iterations}/${result.maxIterations} ms=${result.ms.toFixed(1)} volume=${result.volume.toFixed(2)}`,
  )
  return result.volume
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

/** Ensemble measurement of the two structural numbers a single frame would read off directly --
 * area per lipid and bilayer thickness -- sampled over many configurations while the area coordinate
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

// Sampling spacing (in MD steps) between successive height-field snapshots for the bending-modulus
// ensemble, when the caller does not override it. Not a physical model parameter -- it sets how far
// apart samples are, not what the samples measure -- so it lives here rather than in
// data/params.json, same rationale as AREA_MOVE_LOG_DELTA in sim.ts. Matches the order of magnitude
// measureBilayerAveraged uses for the same purpose on this engine's bilayers.
const BENDING_STEPS_PER_SAMPLE = 100

// Number of blocks the sample window is split into for reporting kappa's scatter, mirroring
// blockDrift's block-averaging above: consecutive samples are correlated (a bending mode relaxes
// over many integration steps, not one), so a standard error needs decorrelated block MEANS, not
// the raw per-sample series.
const BENDING_BLOCKS = 10

export interface BendingModulusResult {
  kappa: number
  kappaSd: number
  slope: number
  fitModes: number
  fitShells: number
  /** false means the data-driven window never found a trustworthy fit (see selectFitWindow in
   * spectrum.ts) -- kappa is NaN in that case, on purpose: a failed measurement must read as a
   * failed measurement, not as a number that happens to fail its own range check for an unrelated
   * reason. slope/qMax/fitShells still describe the best attempt made, for diagnosis. */
  valid: boolean
  samples: number
  stepsPerSample: number
  qMax: number
  ceilingQMax: number
  spectrum: Spectrum
}

/** Ensemble measurement of the membrane's bending modulus from its own undulation spectrum (Task
 * 7's structural gate). Takes `samples` height-field snapshots spaced `stepsPerSample` MD steps
 * apart (the box is not touched here -- no areaMove -- so every snapshot shares one q grid),
 * averages |h_q|^2 mode-by-mode over all of them, then picks the fit window BY THE DATA
 * (selectFitWindow: grows from the smallest |q| shell while the slope stays near -4, up to the
 * generous ceiling `(2*pi*opts.modes)/Lx`) rather than trusting a fixed mode count -- see
 * selectFitWindow's doc comment in spectrum.ts for why: a fixed ceiling has no way to know where
 * this box's q^-4 regime actually ends before the spectrum crosses over into protrusion/tilt modes
 * (~q^-2), and this measurement DID cross that line the first time it ran with a fixed ceiling at
 * mode index 8 (task-7-report.md: slopes -3.25, -3.09, -2.72 across independent runs, never close
 * to -4). kappa itself still comes from fitBendingModulus on the selected window (its -4 check and
 * intercept->kappa conversion are the single source of truth for that step).
 *
 * Also reports kappa's SCATTER across BENDING_BLOCKS contiguous blocks of the sample window, each
 * block's own ensemble-averaged spectrum fit independently (over the SAME selected window -- the
 * window is a property of the box/grid, shared by every block, not re-selected per block) for its
 * own kappa point estimate -- mirroring measureBilayerAveraged's per-sample scatter reporting, and
 * directly answering the lesson from Tasks 5-6 that this engine is not bit-reproducible run to run
 * and a single marginal number proves nothing. A block kappa skips fitBendingModulus's -4 throw (a
 * single noisier block failing that check must widen the reported scatter, not abort the whole
 * measurement); the headline kappa above still goes through the full check. */
export async function measureBendingModulusDetailed(
  sys: System,
  opts: { grid: number; modes: number; samples: number; stepsPerSample?: number },
): Promise<BendingModulusResult> {
  const p = loadParams()
  const kT = p.thermostat.kT
  const box = sys.box
  const area = box[0] * box[1]
  // Ceiling for selectFitWindow's search, not the window itself: opts.modes now names the FURTHEST
  // mode index the data-driven search is ever allowed to grow into (a generous upper limit -- the
  // grid's own Nyquist index is the natural choice), not a fixed cutoff every measurement uses.
  const ceilingQMax = (2 * Math.PI * opts.modes) / box[0]
  const stepsPerSample = opts.stepsPerSample ?? BENDING_STEPS_PER_SAMPLE

  let qs: number[] | undefined
  const perSampleHq2: number[][] = []
  const t0 = performance.now()
  // Progress logging: this loop is the expensive part of a real gate run (hundreds of samples,
  // each stepsPerSample MD steps apart), and a silent multi-minute await gives no signal that the
  // process is alive or where it is heading. Printed every tenth of the sample budget (at least
  // every sample for small budgets) with a running kappa estimate from the mean-so-far -- caught
  // rather than thrown, since early samples may not yet satisfy fitBendingModulus's -4 sanity
  // check and that must not abort a still-accumulating ensemble.
  const logEvery = Math.max(1, Math.floor(opts.samples / 10))
  for (let s = 0; s < opts.samples; s++) {
    if (s > 0) await sys.step(stepsPerSample)
    const raw = await sys.positions()
    // tailEndBeads, not the full bead array: averaging heads AND tails per column injects their
    // ~2 sigma vertical separation divided by only sqrt(beads/column), which measured as a flat
    // noise floor dominating ~90% of the searched spectrum (task-7-report.md). Tail-end beads sit
    // at the midplane by construction and remove that leading spread.
    const h = heightField(tailEndBeads(raw), sys.box, opts.grid)
    const sp = spectrum(h, opts.grid, sys.box)
    if (!qs) qs = sp.q
    perSampleHq2.push(sp.hq2)
    if (s % logEvery === 0 || s === opts.samples - 1) {
      const running = new Array(sp.hq2.length).fill(0)
      for (const hq2 of perSampleHq2) for (let i = 0; i < hq2.length; i++) running[i] += hq2[i] / perSampleHq2.length
      // selectFitWindow never throws (see spectrum.ts) -- it returns its best attempt with
      // valid:false when the running average does not yet look like q^-4 anywhere, which is
      // expected early on; NaN-ing the reported kappa in that case is enough, no try/catch needed.
      const win = selectFitWindow({ q: qs, hq2: running }, ceilingQMax)
      const runningKappa = win.valid ? kT / (area * Math.exp(win.intercept)) : NaN
      console.log(
        `BENDING-PROGRESS sample=${s + 1}/${opts.samples} elapsedMs=${(performance.now() - t0).toFixed(0)} ` +
          `steps=${sys.steps} runningKappa=${runningKappa.toFixed(3)} runningSlope=${win.slope.toFixed(3)} ` +
          `runningShells=${win.nShells} runningValid=${win.valid}`,
      )
    }
  }
  const nModes = qs!.length

  const meanHq2 = new Array(nModes).fill(0)
  for (const hq2 of perSampleHq2) for (let i = 0; i < nModes; i++) meanHq2[i] += hq2[i] / opts.samples
  const meanSpectrum: Spectrum = { q: qs!, hq2: meanHq2 }

  // The final, ensemble-averaged window. If valid, fitBendingModulus is called for the headline
  // kappa (reusing its canonical intercept->kappa conversion, guaranteed not to throw here since
  // selectFitWindow already verified the slope is within tolerance at this exact qMax); if not,
  // kappa is NaN on purpose -- see BendingModulusResult's doc comment.
  const window = selectFitWindow(meanSpectrum, ceilingQMax)
  const kappa = window.valid ? fitBendingModulus(meanSpectrum, kT, area, window.qMax) : NaN

  const blocks = Math.min(BENDING_BLOCKS, opts.samples)
  const perBlock = Math.floor(opts.samples / blocks)
  const blockKappas: number[] = []
  for (let b = 0; b < blocks; b++) {
    const blockMean = new Array(nModes).fill(0)
    for (let k = b * perBlock; k < (b + 1) * perBlock; k++) {
      for (let i = 0; i < nModes; i++) blockMean[i] += perSampleHq2[k][i] / perBlock
    }
    const { intercept } = logLogFit({ q: qs!, hq2: blockMean }, window.qMax)
    blockKappas.push(kT / (area * Math.exp(intercept)))
  }
  const kappaMean = blockKappas.reduce((a, b) => a + b, 0) / blockKappas.length
  const kappaSd = Math.sqrt(
    blockKappas.reduce((a, b) => a + (b - kappaMean) ** 2, 0) / Math.max(1, blockKappas.length - 1),
  )

  return {
    kappa,
    kappaSd,
    slope: window.slope,
    fitModes: window.nModes,
    fitShells: window.nShells,
    valid: window.valid,
    samples: opts.samples,
    stepsPerSample,
    qMax: window.qMax,
    ceilingQMax,
    spectrum: meanSpectrum,
  }
}

/** Facade for Task 7's structural gate (gate6-kappa.test.ts): bending modulus alone, from an
 * ensemble-averaged undulation spectrum -- see measureBendingModulusDetailed for the block-averaged
 * scatter, the fitted slope, and the spectrum itself that this collapses to one number. */
export async function measureBendingModulus(
  sys: System,
  opts: { grid: number; modes: number; samples: number },
): Promise<number> {
  return (await measureBendingModulusDetailed(sys, opts)).kappa
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
