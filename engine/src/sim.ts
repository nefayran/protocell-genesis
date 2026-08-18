// Thin composition point for the validated Cooke & Deserno membrane engine (verify/out/gates.json
// carries its published gates — treat this file's own logic changes as safety-critical). Split by
// responsibility (file-size rule, root CLAUDE.md) into: sim-types.ts (public types), sim-layouts.ts
// (seeded RNG + the three initial layouts), sim-area-math.ts (the pure, GPU-free area-move
// coordinate map), sim-pipelines.ts (compute-pipeline cache), sim-grid-geometry.ts (neighbor-grid
// cell-count/invariant math), sim-runtime.ts (the shared mutable EngineRuntime context type),
// sim-buffers.ts (buffer allocation + the grid-resize path), sim-bindgroups.ts (bind-group
// construction, static and grid-dependent), sim-integrate.ts (the Langevin integration step),
// sim-forces.ts (TS-side dispatch of the bonded/non-bonded force pipelines — WCA on bonded pairs
// included, per forces.wgsl's own header; the physics itself is entirely in WGSL, not here),
// sim-area-move.ts (the zero-tension MC area move with its configurational Jacobian and rigid
// lipid-COM scaling), sim-measure.ts (measurement/readback + live-params retuning).
//
// createSystem() derives creation-time config, wires every module above in the same order the
// original single-file createSystem() did, and assembles the returned System — same shape, same
// export names, so every existing import site (engine/src/index.ts, viewer/main.ts,
// viewer/molecular.ts, tests/sim.test.ts, tests/buckling-kappa*.test.ts) needs zero changes.

import { getGpu } from './gpu'
import { loadParams, wcaCutoff, type Params } from './params'
import { layoutRandom, layoutBilayer, layoutVesicle, mulberry32, initialVelocities } from './sim-layouts'
import { getPipelines } from './sim-pipelines'
import { computeDims, gridInvariantsHold } from './sim-grid-geometry'
import { allocateBuffers, writeGridUniforms } from './sim-buffers'
import { rebindGridDependent, buildStaticBindGroups } from './sim-bindgroups'
import { rebuildGridTimed, rebuildGridUntimed, step as stepFn } from './sim-integrate'
import { encodeForceGrid, forces as forcesFn, forcesBruteForce as forcesBruteForceFn } from './sim-forces'
import { areaMove as areaMoveFn } from './sim-area-move'
import { positions as positionsFn, kineticEnergyPerDof as kineticEnergyPerDofFn, totalEnergy as totalEnergyFn, setLiveParams as setLiveParamsFn } from './sim-measure'
import type { EngineRuntime } from './sim-runtime'
import type { CreateSystemOpts, System } from './sim-types'

export type { Layout, CreateSystemOpts, System } from './sim-types'
export { scaleLateralRigid } from './sim-area-math'

export async function createSystem(opts: CreateSystemOpts): Promise<System> {
  const base = loadParams()
  const p: Params = opts.gamma === undefined ? base : { ...base, thermostat: { ...base.thermostat, gamma: opts.gamma } }

  const N = opts.lipids * 3
  const box = opts.box
  const liveBox: [number, number, number] = [box[0], box[1], box[2]]
  const rng = mulberry32(opts.seed)

  const positions0 =
    opts.positions ??
    (opts.layout === 'random'
      ? layoutRandom(opts.lipids, box, p, rng)
      : opts.layout === 'vesicle'
        ? layoutVesicle(opts.lipids, box, p)
        : layoutBilayer(opts.lipids, box, p, rng))
  const velocities0 = opts.velocities ?? initialVelocities(N, p.thermostat.kT, rng)
  const rngState0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) rngState0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9

  // The wc value the neighbor grid was actually sized for — setLiveParams (sim-measure.ts) must
  // enforce this as a ceiling, since raising wc past this after creation is exactly the
  // silently-dropped-forces failure mode sim-grid-geometry.ts's computeDims doc comment describes,
  // just triggered live instead of at construction.
  const builtForWc = Math.max(p.attraction.wc, opts.maxWc ?? p.attraction.wc)
  const cellSize = wcaCutoff(p.beadSizes.tail_tail) + builtForWc

  const dims = computeDims(box, cellSize)
  if (!gridInvariantsHold(box, dims, p.bend.r0)) {
    throw new Error(
      `сетка соседей: box=[${box[0]},${box[1]},${box[2]}] даёт cellSize=${cellSize.toFixed(4)}, dims=[${dims[0]},${dims[1]},${dims[2]}] ` +
        `и min(box.x,box.y)/2=${(Math.min(box[0], box[1]) / 2).toFixed(4)} — нужно dims>=3 на осях x,y и min(box.x,box.y)/2 > bend.r0=${p.bend.r0}`,
    )
  }
  const ncells = dims[0] * dims[1] * dims[2]

  const { device } = await getGpu()
  const pipe = getPipelines(device)

  const buffers = allocateBuffers(device, N, ncells, positions0, velocities0, rngState0, p)

  const rt: EngineRuntime = {
    device,
    pipe,
    opts,
    p,
    livep: p,
    rng,
    N,
    cellSize,
    builtForWc,
    liveBox,
    dims,
    ncells,
    totalSteps: 0,
    neighborBuildMs: 0,
    ...buffers,
    // Bind groups are assigned immediately below (rebindGridDependent/buildStaticBindGroups),
    // before any dispatch reads them — placeholders here only to satisfy EngineRuntime's shape.
    forceGridGroup0: {} as GPUBindGroup,
    forceBruteGroup0: {} as GPUBindGroup,
    forceBruteGroup1: {} as GPUBindGroup,
    kickBind: {} as GPUBindGroup,
    driftBind: {} as GPUBindGroup,
    wrapBind: {} as GPUBindGroup,
    thermostatBind: {} as GPUBindGroup,
    clearCountsBind: {} as GPUBindGroup,
    countBind: {} as GPUBindGroup,
    prefixBind: {} as GPUBindGroup,
    fillBind: {} as GPUBindGroup,
    forceGridGroup1: {} as GPUBindGroup,
    wgCells: 0,
    wgN: Math.ceil(N / 64),
  }

  writeGridUniforms(rt.device, rt.gridUniform, rt.boxUniform, box, dims)
  rebindGridDependent(rt)
  buildStaticBindGroups(rt)

  // Warm-up: pipelines are compiled lazily on first dispatch, not at createComputePipeline()
  // time, and that one-time compile latency would otherwise leak into the first measurement
  // (measured: 5.4ms for a 600-particle system built first on the page vs about a fifth of that
  // for a 3000-particle one built after pipelines were already warm). Run one untimed rebuild before
  // the timed one so `neighborBuildMs` reports the rebuild itself, not compilation.
  await rebuildGridUntimed(rt)

  // Initial grid build + force evaluation, timed for Task 9's `neighborBuildMs`, and needed as
  // F(x0) for the first kick of step().
  await rebuildGridTimed(rt)
  {
    const enc = rt.device.createCommandEncoder()
    const pass = enc.beginComputePass()
    encodeForceGrid(rt, pass)
    pass.end()
    rt.device.queue.submit([enc.finish()])
  }

  return {
    step: (n: number) => stepFn(rt, n),
    positions: () => positionsFn(rt),
    kineticEnergyPerDof: () => kineticEnergyPerDofFn(rt),
    totalEnergy: () => totalEnergyFn(rt),
    areaMove: (trials: number) => areaMoveFn(rt, trials),
    setLiveParams: (overrides: { kT?: number; wc?: number }) => setLiveParamsFn(rt, overrides),
    get box(): [number, number, number] {
      return [rt.liveBox[0], rt.liveBox[1], rt.liveBox[2]]
    },
    lipids: opts.lipids,
    get steps() {
      return rt.totalSteps
    },
    forces: () => forcesFn(rt),
    forcesBruteForce: () => forcesBruteForceFn(rt),
    get neighborBuildMs() {
      return rt.neighborBuildMs
    },
  }
}
