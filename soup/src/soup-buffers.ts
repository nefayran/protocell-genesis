// GPU buffer allocation and ownership, including the neighbour-grid resize path. Split out of
// soup/src/sim.ts's createSoup (file-size rule in CLAUDE.md) -- every buffer creation call and
// every `device.queue.writeBuffer` that seeds it moved verbatim, plus resizeSoupGrid (which
// reallocates the ncells-sized buffers when a dry-wet box change moves dims into a different
// cell-count bracket) and the dispose() list. resizeSoupGrid does NOT itself rebuild bind groups --
// that is soup/src/soup-bindgroups.ts's job (rebind-on-resize logic) -- it takes an `onResized`
// callback instead, so this file owns buffer lifetime without needing to import pipeline/bind-group
// code at all.

import type { ResolvedRule } from './soup-plan'
import { planSoupGrid } from './soup-plan'
import { storageBuffer } from '../../engine/src/gpu'
import { paramsToUniform, type Params } from '../../engine/src/params'
import { packSpeciesSlots, packVec4 } from './soup-plan'
import type { Soup } from './rules'
import { ATTR_SCALE_UNIFORM_BYTES, attractionScaleUniform } from './soup-attraction'
import { attemptProbability, acceptanceProbability } from './rules'
import type { InitialState } from './soup-init-state'

export interface SoupBuffers {
  posBuf: GPUBuffer
  velBuf: GPUBuffer
  forceBuf: GPUBuffer
  cellsBuf: GPUBuffer
  // Sized ncells (not N) -- reallocated by resizeSoupGrid() below whenever a dry-wet box change
  // moves dims to a different cell-count bracket, exactly like engine/src/sim.ts's own
  // countsBuf/cellStartBuf/cursorBuf under areaMove()'s resizeGrid.
  countsBuf: GPUBuffer
  cellStartBuf: GPUBuffer
  cursorBuf: GPUBuffer
  // perf2-report.md, candidate (b): cell-sorted GATHER of positions, rebuilt every grid rebuild
  // (i.e. every step) from the SAME cellIdx permutation fill_main already produces -- see
  // soup_gather_sorted_main in soup/wgsl/step.wgsl. Positions/velocities/bondSlots themselves stay
  // in ORIGINAL index space (never physically reordered), so bond bookkeeping is untouched by
  // construction -- see soup/src/sim.ts's header note on why that choice was made over a full resort.
  posSortedBuf: GPUBuffer
  // perf2-report.md, STEP 1 diagnosis: [candidatesExamined, pairsWithinRange], read back once per
  // forceCandidateStatsDEBUG call, never touched by the real step loop.
  statsBuf: GPUBuffer
  // perf2-report.md, candidate (c): Verlet list buffers. verletListBuf is a flat N*listCapacity
  // array (soup_build_verlet_list_main/soup_force_list_main/bond_form_list_main all index it as
  // i*listCapacity+slot); verletCountBuf is the per-particle count actually found this rebuild;
  // verletOverflowBuf is a single flag the build kernel sets (never clears) if any particle found
  // more than listCapacity candidates -- soup/src/soup-grid-verlet.ts asserts it stays clear after
  // every rebuild rather than silently trusting listCapacity was big enough. posAtRebuildBuf/
  // maxDriftSqBuf implement the drift-safety guard: a snapshot taken at every rebuild and the
  // running max squared-drift since it, read back and checked after every step() call.
  verletListBuf: GPUBuffer
  verletCountBuf: GPUBuffer
  verletOverflowBuf: GPUBuffer
  posAtRebuildBuf: GPUBuffer
  maxDriftSqBuf: GPUBuffer
  // VL: x=listRange (interactionRange+skin), y=listCapacity (as f32, cast to u32 in WGSL) -- see
  // step.wgsl/bond-verlet.wgsl's own VL declarations.
  verletUniform: GPUBuffer
  bondSlotsBuf: GPUBuffer
  // Surface growth (surface-growth-report.md): centerLink, bound at group1 binding20
  // (soup/wgsl/bond-adsorption.wgsl) into both bond-form bind groups (cell-walk and Verlet-list
  // variants) -- bond_break_main never references it (co_break/cc_break's own reversal is
  // deliberately orthogonal to this bookkeeping, see that task's own report).
  centerLinkBuf: GPUBuffer
  /** Task 'clay-surface' (2026-08-19): per-particle immobility, bound at group1 binding21 into the
   * two integrator bind groups (kickDriftWrap / kickThermostat). Not referenced by any force or bond
   * kernel: a frozen bead interacts exactly like any other, it simply never integrates. */
  frozenBuf: GPUBuffer
  // Surface growth / adsorption (adsorption-report.md): centerHeldSteps/desorbEvents, bound at
  // group1 bindings 21/22 (soup/wgsl/bond-adsorption.wgsl) into both bond-form bind groups
  // (cell-walk and Verlet-list variants) -- neither is read by soup/wgsl/step.wgsl's force kernels
  // (only centerLink itself is, for the tether force) nor by bond_break_main (desorption is decided
  // entirely inside bond_form_main's own dispatch).
  centerHeldStepsBuf: GPUBuffer
  desorbEventsBuf: GPUBuffer
  bondRngBuf: GPUBuffer
  thermoRngBuf: GPUBuffer
  eventsBuf: GPUBuffer
  paramsUniform: GPUBuffer
  gridUniform: GPUBuffer
  speciesUniform: GPUBuffer
  bondParamsUniform: GPUBuffer
  // Task 'hydrophobic-asymmetry' (2026-08-19): the per-class attraction-depth table (3 rows of
  // vec4, one row per species class) soup/wgsl/step.wgsl's pairAttrScale() reads -- built by
  // soup/src/soup-attraction.ts from data/soup.json's solvent.attractionScale, optionally with
  // CreateSoupOpts.solventAttractionScaleOverride replacing the file's global epsilonScale.
  attrScaleUniform: GPUBuffer
  /** Task 'loud-failure-and-liquid-water' (2026-08-20): two u32 counters -- [nonFinitePosComponents,
   * nonFiniteVelComponents] -- written by soup/wgsl/health.wgsl's scan kernel and read back once per
   * step()-chunk by soup/src/soup-health.ts. 8 bytes, COPY_SRC so readBack can reach it. */
  healthBuf: GPUBuffer
  /** Task 'loud-failure-and-liquid-water' (2026-08-20): RX (soup/wgsl/relax.wgsl) -- x = this
   * minimisation iteration's displacement cap in sigma, rewritten by soup/src/soup-relax.ts before
   * every iteration's submit. Untouched (and the kernel never dispatched) on any system that does not
   * ask for relaxation. */
  relaxUniform: GPUBuffer
}

export interface SoupGridState {
  dims: [number, number, number]
  ncells: number
  wgCells: number
}

export interface AllocateBuffersInput {
  device: GPUDevice
  soup: Soup
  p: Params
  N: number
  dims: [number, number, number]
  ncells: number
  effectiveWalkRadius: number
  initialLiveBox: [number, number, number]
  verlet: Soup['verletList']
  listRange: number
  rules: ResolvedRule[]
  catalystKind: number
  bondAttemptInterval: number
  kT: number
  initial: InitialState
  /** Water-calibration task (2026-08-19): overrides soup.solvent.attractionScale.epsilonScale for
   * this system only -- see soup/src/soup-types.ts's CreateSoupOpts.solventAttractionScaleOverride. */
  solventAttractionScaleOverride?: number
  /** Task 'clay-surface-chemistry' (2026-08-19): which mineral depth row the AttrScale uniform below
   * carries -- see soup/src/soup-types.ts's CreateSoupOpts.claySurfaceChemistry. */
  claySurfaceChemistry?: string
}

export function allocateSoupBuffers(input: AllocateBuffersInput): { buf: SoupBuffers; grid: SoupGridState } {
  const { device, soup, p, N, dims, ncells, effectiveWalkRadius, initialLiveBox, verlet, listRange, rules, catalystKind, bondAttemptInterval, kT, initial, solventAttractionScaleOverride, claySurfaceChemistry } = input
  const { positions0, velocities0, bondSlots0, centerLink0, centerHeldSteps0, desorbEventsInit, bondRng0, thermoRng0, eventsInit, frozen0 } = initial

  const posBuf = storageBuffer(device, positions0)
  const velBuf = storageBuffer(device, velocities0)
  const forceBuf = storageBuffer(device, new Float32Array(N * 4))
  const cellsBuf = storageBuffer(device, new Float32Array(N))
  const countsBuf = storageBuffer(device, new Float32Array(ncells))
  const cellStartBuf = storageBuffer(device, new Float32Array(ncells + 1))
  const cursorBuf = storageBuffer(device, new Float32Array(ncells))
  const posSortedBuf = storageBuffer(device, new Float32Array(N * 4))
  const statsBuf = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })

  const verletListBuf = device.createBuffer({
    size: N * verlet.listCapacity * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const verletCountBuf = device.createBuffer({ size: Math.max(4, N * 4), usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST })
  const verletOverflowBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(verletOverflowBuf, 0, new Uint32Array([0]))
  const posAtRebuildBuf = storageBuffer(device, new Float32Array(N * 4))
  const maxDriftSqBuf = device.createBuffer({
    size: 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(maxDriftSqBuf, 0, new Uint32Array([0]))
  const verletUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(verletUniform, 0, new Float32Array([listRange, verlet.listCapacity, 0, 0]))

  const bondSlotsBuf = device.createBuffer({
    size: bondSlots0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(bondSlotsBuf, 0, bondSlots0)

  // Task 'clay-surface' (2026-08-19): per-particle immobility flag, read (never written) by
  // soup/wgsl/step.wgsl's two integrator kernels -- see frozenRO's own declaration there for why
  // those two kernels are the only place immobility can be enforced. Same allocate-and-upload-once
  // pattern as bondSlots/centerLink above.
  const frozenBuf = device.createBuffer({
    size: frozen0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(frozenBuf, 0, frozen0)

  const centerLinkBuf = device.createBuffer({
    size: centerLink0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(centerLinkBuf, 0, centerLink0)

  const centerHeldStepsBuf = device.createBuffer({
    size: centerHeldSteps0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(centerHeldStepsBuf, 0, centerHeldSteps0)
  const desorbEventsBuf = device.createBuffer({
    size: desorbEventsInit.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(desorbEventsBuf, 0, desorbEventsInit)

  // COPY_SRC (task 'loud-failure-and-liquid-water', 2026-08-20): these two were the ONLY readable
  // buffers in this file missing the flag, and soup/src/soup-readback.ts's rngState() reads both --
  // so every rngState() call, and therefore the bondRngB64/thermoRngB64 field of EVERY checkpoint
  // this project has ever written, silently returned all zeros (verified on disk: 81 000 and 426 904
  // zero bytes in two checkpoints from different campaigns). decodeCheckpointResume fed those zeros
  // back in, giving every resumed run a per-particle RNG seed of 0 for all N particles -- identical
  // Langevin noise on every particle instead of independent noise. engine/src/gpu.ts's readBack now
  // THROWS on a buffer without the flag rather than returning zeros, so this cannot recur silently.
  const bondRngBuf = device.createBuffer({
    size: bondRng0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(bondRngBuf, 0, bondRng0)
  const thermoRngBuf = device.createBuffer({
    size: thermoRng0.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(thermoRngBuf, 0, thermoRng0)

  const eventsBuf = device.createBuffer({
    size: eventsInit.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(eventsBuf, 0, eventsInit)

  const paramsUniform = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(paramsUniform, 0, paramsToUniform(p))

  const gridUniform = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const bytes = new ArrayBuffer(32)
    // dims.w carries walkRadius (perf2-report.md, candidate (a)) -- unused by engine/wgsl/
    // neighbor.wgsl (its cell_of()/clear_counts_main only ever read dims.xyz), read by
    // soup/wgsl/step.wgsl's soup_force_main and soup/wgsl/bond-dispatch.wgsl's bond_form_main to
    // drive the ±walkRadius cell walk instead of a hardcoded ±1.
    new Uint32Array(bytes, 0, 4).set([dims[0], dims[1], dims[2], effectiveWalkRadius])
    // initialLiveBox, not the wet creation box: a checkpoint resumed mid dry-phase lives in the DRY
    // box, and `dims` (from planSoupGrid(initialLiveBox, ...) in soup/src/sim.ts) already reflects
    // that -- writing the wet box here instead would desync the uniform's own box floats from the
    // dims right next to them, exactly the bug resizeSoupGrid's own doc comment describes for a
    // live box change.
    new Float32Array(bytes, 16, 4).set([initialLiveBox[0], initialLiveBox[1], initialLiveBox[2], 0])
    device.queue.writeBuffer(gridUniform, 0, bytes)
  }

  // Task 'explicit-water' (2026-08-18): 96 bytes (3 fields x 2 vec4 x 4 bytes), not 32 -- see
  // soup/wgsl/step.wgsl's own Species struct, widened from a single vec4 per field (4 species) to
  // array<vec4<f32>,2> (8 species) so the 5th species (water) fits without a new binding.
  // Task 'clay-surface' (2026-08-19): 96 -> 128 bytes for the 4th per-species field (`mineral`, 8
  // slots x 4 bytes). soup/wgsl/bond-common.wgsl's own Species struct is still the 3-field, 96-byte
  // shape and stays bound to this same, now-larger buffer on purpose -- a uniform binding only needs
  // the buffer to be at least as large as the struct, and no bond rule can ever name the mineral
  // species (see step.wgsl's own note at the Species declaration).
  const speciesUniform = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const radius = packSpeciesSlots(soup.monomers.map((m) => m.radiusSigma))
    const polar = packSpeciesSlots(soup.monomers.map((m) => (m.polar ? 1 : 0)))
    const solvent = packSpeciesSlots(soup.monomers.map((m) => (m.solvent ? 1 : 0)))
    const mineral = packSpeciesSlots(soup.monomers.map((m) => (m.mineral ? 1 : 0)))
    device.queue.writeBuffer(speciesUniform, 0, new Float32Array([...radius, ...polar, ...solvent, ...mineral]))
  }

  // Task 'hydrophobic-asymmetry' (2026-08-19): the per-class attraction-depth table, see
  // soup/wgsl/step.wgsl's AttrScale declaration and soup/src/soup-attraction.ts (which builds it
  // from data/soup.json and is ALSO what the CPU-side Metropolis energy reads, so the two cannot
  // drift). `solventAttractionScaleOverride` still overrides the file's global epsilonScale for one
  // system (calibration sweeps); absent-from-both reads as unscaled.
  const attrScaleUniform = device.createBuffer({
    size: ATTR_SCALE_UNIFORM_BYTES,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(attrScaleUniform, 0, attractionScaleUniform(soup, solventAttractionScaleOverride, claySurfaceChemistry))

  // BondParams: acceptProbForm/acceptProbBreak come DIRECTLY from soup/src/rules.ts's
  // acceptanceProbability(rule, kT) -- the same function tests/soup-rules.test.ts checks against
  // forwardBackwardRatio -- and attemptProbForm/attemptProbBreak from its attemptProbability(rule,
  // dt), so this is a straight upload of Task 1's own numbers, not a re-derivation.
  //
  // Perf fix (b), perf-report.md: bond_form_main/bond_break_main are only DISPATCHED every
  // bondAttemptInterval.steps real steps (see soup/src/soup-integrate.ts), so the dt this uniform is
  // built from is multiplied by that interval -- attemptProbability's own formula (attemptRate*dt)
  // is linear in dt, so attemptRate*(dt*k) is exactly k independent per-step attempts' worth of
  // probability folded into one (Poisson thinning), keeping the average attempt rate per REAL step
  // unchanged. acceptanceProbability below is untouched -- it depends only on energyKT/kT, never on
  // dt, so this has no effect on the forward/backward ratio detailed balance is carried by.
  const bondDt = p.integrator.dt * bondAttemptInterval
  // Surface growth / adsorption (adsorption-report.md): BondParams grew one more vec4
  // (adsorptionParams -- see bond-common.wgsl's own struct comment), so its uniform buffer grows
  // from 160 to 176 bytes (11 vec4-aligned f32 groups instead of 10) -- the WRITE below is the
  // single place that size must stay in sync with bond-common.wgsl's own struct layout.
  const bondParamsUniform = device.createBuffer({ size: 176, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  {
    const kindA = packVec4(rules.map((r) => r.kindA))
    const kindB = packVec4(rules.map((r) => r.kindB))
    const attemptProbForm = packVec4(rules.map((r) => attemptProbability(r.bond, bondDt)))
    const attemptProbBreak = packVec4(rules.map((r) => attemptProbability(r.brk, bondDt)))
    const acceptProbForm = packVec4(rules.map((r) => acceptanceProbability(r.bond, kT)))
    const acceptProbBreak = packVec4(rules.map((r) => acceptanceProbability(r.brk, kT)))
    const requiresCatalyst = packVec4(rules.map((r) => (r.bond.requiresCatalyst ? 1 : 0)))
    const slotRoleA = packVec4(rules.map((r) => r.slotRoleA))
    const slotRoleB = packVec4(rules.map((r) => r.slotRoleB))
    // data/soup.json's headPlacement.terminalOnly (rank D, basis in that file): whether
    // soup/wgsl/bond-valence.wgsl's tryClaimSlot gates head/chain-slot claims on chain-end position,
    // packed as a 1/0 float the same way requiresCatalyst already is -- the flag is data, not a
    // constant written into soup/src or soup/wgsl.
    const headTerminalOnly = soup.headPlacement.terminalOnly ? 1 : 0
    // data/soup.json's headPlacement.chainCapacity (rank D, basis in that file): how many chain
    // slots a head's own claim (soup/wgsl/bond-valence.wgsl's tryClaimSlot, role==2) may try --
    // uploaded as a plain float the same way headTerminalOnly already is, replacing that struct's
    // former bpPad1 padding slot (see BondParams's own comment in bond-common.wgsl).
    // assertRulesConsistent (soup/src/rules.ts) has already checked this is an integer within the
    // architectural 1..3 range by the time loadSoup() returns it here.
    const headChainCapacity = soup.headPlacement.chainCapacity
    // Surface growth / adsorption (adsorption-report.md): data/soup.json's adsorption.maxHoldSteps
    // (real steps, x) and bondAttemptInterval.steps (real steps per bond-dispatch cycle, y -- what
    // soup/wgsl/bond-adsorption.wgsl's desorbTimeout increments centerHeldSteps by each cycle, so x
    // and the running total it is compared against stay in the SAME real-step units). Both already
    // exist as plain numbers read from data/soup.json / derived by the caller -- no new numeric
    // literal here.
    const adsorptionMaxHoldSteps = soup.adsorption.maxHoldSteps
    device.queue.writeBuffer(
      bondParamsUniform,
      0,
      new Float32Array([
        ...kindA,
        ...kindB,
        ...attemptProbForm,
        ...attemptProbBreak,
        ...acceptProbForm,
        ...acceptProbBreak,
        ...requiresCatalyst,
        ...slotRoleA,
        ...slotRoleB,
        catalystKind,
        headTerminalOnly,
        headChainCapacity,
        0,
        adsorptionMaxHoldSteps,
        bondAttemptInterval,
        0,
        0,
      ]),
    )
  }

  const healthBuf = device.createBuffer({
    size: 8,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  })
  device.queue.writeBuffer(healthBuf, 0, new Uint32Array([0, 0]))
  const relaxUniform = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(relaxUniform, 0, new Float32Array([0, 0, 0, 0]))

  const buf: SoupBuffers = {
    posBuf,
    velBuf,
    forceBuf,
    cellsBuf,
    countsBuf,
    cellStartBuf,
    cursorBuf,
    posSortedBuf,
    statsBuf,
    verletListBuf,
    verletCountBuf,
    verletOverflowBuf,
    posAtRebuildBuf,
    maxDriftSqBuf,
    verletUniform,
    bondSlotsBuf,
    centerLinkBuf,
    centerHeldStepsBuf,
    desorbEventsBuf,
    bondRngBuf,
    thermoRngBuf,
    eventsBuf,
    paramsUniform,
    gridUniform,
    frozenBuf,
    speciesUniform,
    bondParamsUniform,
    attrScaleUniform,
    healthBuf,
    relaxUniform,
  }
  const grid: SoupGridState = { dims, ncells, wgCells: Math.ceil(ncells / 64) }
  return { buf, grid }
}

/** Dry-wet cycling's own grid-rebuild capability (task 'grid-rebuild'), generalising
 * engine/src/sim.ts's own resizeGrid() (added there for the membrane's area move, reviewed) to the
 * soup engine: recomputes dims for `newBox` via planSoupGrid -- the single source of truth for
 * grid validity, the SAME derivation createSoup used at construction time -- and, if the cell
 * count actually changed, destroys and reallocates countsBuf/cellStartBuf/cursorBuf and calls
 * `onResized` (soup/src/soup-bindgroups.ts's rebindGridDependent, threaded in by soup/src/sim.ts
 * rather than imported directly here -- this file owns buffer lifetime, not bind-group rebuilding),
 * then rewrites the grid uniform (dims + box) unconditionally. Called by soup/src/soup-box-scale.ts's
 * applyBoxScaleOnce after every dry-wet transition increment -- this is what removes the blocker
 * wet-dry-cycle-report.md measured: a box change that moves dims into a different cell-count
 * bracket (the FULL-scale case at box 46, targetDryDensity=0.6: wet dims=[15,15,15] -> dry
 * dims=[14,14,14]) no longer desyncs the fixed-size grid buffers from the live box, it reallocates
 * them to match. planSoupGrid's own `valid` is the ONLY thing this cannot paper over -- a box with
 * fewer than minCells cells on some axis, or one that violates the minimum-image convention, is a
 * genuinely invalid geometry, not a cell-count change a reallocation can fix, so it still throws
 * (task requirement 2: "let it fire only on a genuinely invalid geometry ... not on a legal change
 * of cell count"). The Verlet list itself (verletListBuf, sized N*listCapacity -- independent of
 * ncells) is never touched here; its own rebuild (needed because neighbourhoods move under any box
 * change) is the caller's job, done immediately afterward by applyBoxScaleOnce's own
 * encodeVerletRebuild call, whose assertVerletSafety check afterward is what keeps task requirement
 * 3 (the overflow guard "must stay in force") true across a resize. */
export function resizeSoupGrid(
  device: GPUDevice,
  buf: SoupBuffers,
  grid: SoupGridState,
  startCounts: Record<string, number>,
  newBox: [number, number, number],
  effectiveWalkRadius: number,
  onResized: () => void,
): void {
  const newPlan = planSoupGrid(newBox, startCounts)
  if (!newPlan.valid) throw new Error(newPlan.reason!)
  if (newPlan.dims[0] !== grid.dims[0] || newPlan.dims[1] !== grid.dims[1] || newPlan.dims[2] !== grid.dims[2]) {
    buf.countsBuf.destroy()
    buf.cellStartBuf.destroy()
    buf.cursorBuf.destroy()
    grid.dims = newPlan.dims
    grid.ncells = newPlan.ncells
    grid.wgCells = Math.ceil(grid.ncells / 64)
    buf.countsBuf = storageBuffer(device, new Float32Array(grid.ncells))
    buf.cellStartBuf = storageBuffer(device, new Float32Array(grid.ncells + 1))
    buf.cursorBuf = storageBuffer(device, new Float32Array(grid.ncells))
    onResized()
  }
  const bytes = new ArrayBuffer(32)
  new Uint32Array(bytes, 0, 4).set([grid.dims[0], grid.dims[1], grid.dims[2], effectiveWalkRadius])
  new Float32Array(bytes, 16, 4).set([newBox[0], newBox[1], newBox[2], 0])
  device.queue.writeBuffer(buf.gridUniform, 0, bytes)
}

// Every GPUBuffer allocateSoupBuffers allocates above, for dispose() to destroy -- listed
// exhaustively rather than tracked via a running array at allocation time, so this list is a single
// place to audit against allocateSoupBuffers's own allocations whenever a new buffer is added there.
export function disposeSoupBuffers(buf: SoupBuffers): void {
  buf.posBuf.destroy()
  buf.velBuf.destroy()
  buf.forceBuf.destroy()
  buf.cellsBuf.destroy()
  buf.countsBuf.destroy()
  buf.cellStartBuf.destroy()
  buf.cursorBuf.destroy()
  buf.posSortedBuf.destroy()
  buf.statsBuf.destroy()
  buf.verletListBuf.destroy()
  buf.verletCountBuf.destroy()
  buf.verletOverflowBuf.destroy()
  buf.posAtRebuildBuf.destroy()
  buf.maxDriftSqBuf.destroy()
  buf.verletUniform.destroy()
  buf.bondSlotsBuf.destroy()
  buf.centerLinkBuf.destroy()
  buf.centerHeldStepsBuf.destroy()
  buf.desorbEventsBuf.destroy()
  buf.bondRngBuf.destroy()
  buf.thermoRngBuf.destroy()
  buf.eventsBuf.destroy()
  buf.paramsUniform.destroy()
  buf.gridUniform.destroy()
  buf.frozenBuf.destroy()
  buf.speciesUniform.destroy()
  buf.bondParamsUniform.destroy()
  buf.attrScaleUniform.destroy()
  buf.healthBuf.destroy()
  buf.relaxUniform.destroy()
}
