// Checkpoint/resume (task 'checkpoint-resume'): a long soup run needs to survive being killed for a
// reason that has NOTHING to do with the physics -- a test's own step timeout, the page dying while
// handing back a huge array of raw numbers, puppeteer's own CDP protocolTimeout, a background run
// vanishing when its parent turn ends. Every one of those has already thrown away a fully-computed
// state on this project (see soup/cli/campaign.ts's own header for the four measured losses this
// module exists to stop).
//
// This is the ONE place that turns a live SoupSystem's mutable state into a plain, JSON-serialisable
// object (encodeCheckpoint, called from INSIDE the browser page, where the GPU readbacks SoupSystem
// exposes actually live) and back into CreateSoupOpts.resume's own shape (decodeCheckpointResume,
// callable from either side of the CDP boundary -- the browser, to actually resume a system, or Node,
// to inspect a checkpoint file without a GPU at all). It has no fs/disk knowledge: writing the encoded
// object to a file, and reading it back, is Node's job (soup/cli/campaign.ts), never this module's or
// the browser's -- "the browser cannot write files, so the page hands the state to node and node
// writes it" is this task's own brief, and the split here is exactly that boundary.
//
// Base64, never a JSON array of numbers -- the SPECIFIC thing already measured to kill the page: a
// completed 250 000-step, 93 200-particle run died transferring its own coordinates back to Node as a
// plain number array (TargetCloseError after 1405s of correct physics, no crash in the physics
// itself). A base64 STRING is one JSON token; the same bytes as a JSON number array are on the order
// of 100 000 comma-separated tokens, and both the CDP wire protocol's own JSON framing and V8's
// (de)serialiser pay per TOKEN, not per byte -- see soup/cli/campaign.ts's own measured transfer time
// at full scale for the number this predicts.
//
// atob/btoa, not Node's Buffer: this file runs in the browser AND under plain Node (soup/cli/
// campaign.ts, via tsx, no bundler) -- both environments have had atob/btoa as globals for years
// (Node since v16, stable since v18, this project's own floor), Buffer is a Node global a browser
// bundle does not have, and TextEncoder/TextDecoder are for UTF-8 TEXT, not arbitrary binary (a raw
// Float32Array's bytes are not valid UTF-8 in general, so decoding them as one would corrupt data).

import type { CreateSoupOpts } from './sim'

// String.fromCharCode's own practical argument-count ceiling (V8's call-stack argument limit is well
// above this, but staying comfortably under it -- the same order of magnitude other engines use -- is
// what keeps this loop correct on every engine, not just the one this was tested on).
const CHUNK = 0x8000

function bytesToBase64(bytes: Uint8Array): string {
  let binary = ''
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK))
  }
  return btoa(binary)
}

function base64ToBytes(b64: string): Uint8Array {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

function encodeTyped(arr: Float32Array | Uint32Array): string {
  return bytesToBase64(new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength))
}

function decodeFloat32(b64: string): Float32Array {
  const bytes = base64ToBytes(b64)
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
}

function decodeUint32(b64: string): Uint32Array {
  const bytes = base64ToBytes(b64)
  return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
}

/** Everything createSoup(opts) needs BESIDES opts.resume itself -- persisted alongside the state so a
 * resume call is a straight `createSoup({ ...file.config, resume: decodeCheckpointResume(file) })`,
 * never a caller re-guessing what composition/box/seed produced this checkpoint's N. */
export interface CheckpointConfig {
  box: [number, number, number]
  seed: number
  kT: number
  start?: Record<string, number>
  catalystCount?: number
  dryWetCycle?: boolean
  /** Task 'evaporation' (2026-08-20): whether this lineage's dry phase removes solvent beads. Part of
   * the config (and therefore of soup/cli/campaign.ts's own resume signature) because two runs that
   * differ in it are different experiments, not two snapshots of one. */
  evaporateSolvent?: boolean
  /** Task 'decisive-run' (2026-08-20): how many wet->dry->wet cycles THIS run's schedule has,
   * overriding data/soup.json's dryWetCycle.cycles. Part of the run's identity, not a snapshot
   * detail: a one-cycle run and a six-cycle run at the same box/composition/seed are two different
   * experiments, so soup/cli/campaign-config.ts's configSignature must refuse to resume one from the
   * other. Undefined (every pre-task run) means "the file's own cycle count", and the signature omits
   * the key entirely in that case, so no existing checkpoint lineage is orphaned. */
  dryWetCycles?: number
  /** Task 'decisive-run' (2026-08-20): the global steps at which this run applied a MID-RUN energy
   * minimisation (SoupSystem.minimiseNowDEBUG) -- the minimisation-only control arm. Same identity
   * argument as dryWetCycles above: a trajectory that was minimised at step 12 000 is not a snapshot
   * of one that was not. Omitted from the signature when absent or empty. */
  minimiseAt?: number[]
}

export interface CheckpointFile {
  version: 1
  createdAt: string
  config: CheckpointConfig
  /** Total particle count -- redundant with config.start's own sum, kept explicit so a reader (or
   * soup/cli/campaign.ts's own newest-checkpoint scan) can sanity-check a file without decoding any
   * base64 payload at all. */
  N: number
  globalStep: number
  liveBox: [number, number, number]
  cyclePhase: 'wet' | 'dry' | 'none'
  cycleIndex: number
  /** Task 'evaporation' (2026-08-20): the LIVE per-monomer census, which is NOT redundant with
   * `config.start` once the solvent can leave: a checkpoint taken mid dry-phase has fewer solvent
   * beads than the composition its run was created with, and `N`/the base64 payloads below are sized
   * for THAT. Written by every checkpoint (also by non-evaporating ones, where it simply equals
   * config.start after catalystCount/clay resolution) so a reader never has to guess which. */
  activeCounts: Record<string, number>
  /** Keyed by data/soup.json rule id, mirroring SoupSystem.events()'s own return shape exactly. */
  events: Record<string, number>
  desorbEvents: { stretch: number; timeout: number }
  positionsB64: string
  velocitiesB64: string
  bondSlotsB64: string
  centerLinkB64: string
  centerHeldStepsB64: string
  bondRngB64: string
  thermoRngB64: string
}

/** The subset of SoupSystem a checkpoint needs -- named separately (rather than importing the whole
 * SoupSystem interface) so this module's own contract is exactly the fields it reads, nothing more. */
export interface CheckpointableSystem {
  particles(): Promise<Float32Array>
  velocities(): Promise<Float32Array>
  bondSlots(): Promise<Uint32Array>
  centerLinks(): Promise<Uint32Array>
  centerHeldSteps(): Promise<Uint32Array>
  desorbEvents(): Promise<{ stretch: number; timeout: number }>
  rngState(): Promise<{ bond: Uint32Array; thermo: Uint32Array }>
  events(): Promise<Record<string, number>>
  invariants(): Promise<{ monomers: Record<string, number>; bonds: number; charge: number }>
  readonly box: [number, number, number]
  readonly cyclePhase: 'wet' | 'dry' | 'none'
  readonly cycleIndex: number
  readonly steps: number
}

/** Runs where `sys` actually lives -- inside the browser page, since every field it reads is a GPU
 * readback (soup/src/sim.ts's own SoupSystem accessors). Every readback happens through
 * Promise.all -- concurrent, not sequential -- because none of them write anything (readBack is
 * copy+map, never writeBuffer, see engine/src/gpu.ts), so there is no ordering hazard between them
 * and no reason to pay for eight sequential round trips when eight concurrent ones cost the same
 * wall time as the slowest single one. This read-only property is also what makes checkpointing
 * provably trajectory-neutral: nothing in this function, or in any accessor it calls, ever issues a
 * GPUCommandEncoder compute pass or a writeBuffer call, so a caller inserting a checkpoint between
 * two step() calls submits the EXACT SAME sequence of compute dispatches either way. */
export async function encodeCheckpoint(sys: CheckpointableSystem, config: CheckpointConfig): Promise<CheckpointFile> {
  const [positions, velocities, bondSlots, centerLink, centerHeldSteps, desorb, rng, events, inv] = await Promise.all([
    sys.particles(),
    sys.velocities(),
    sys.bondSlots(),
    sys.centerLinks(),
    sys.centerHeldSteps(),
    sys.desorbEvents(),
    sys.rngState(),
    sys.events(),
    // invariants() is another pure readback (it recomputes the census from the positions buffer's own
    // species slot), so it joins the same concurrent batch and keeps this function trajectory-neutral.
    sys.invariants(),
  ])
  return {
    version: 1,
    createdAt: new Date().toISOString(),
    config,
    N: positions.length / 4,
    globalStep: sys.steps,
    liveBox: sys.box,
    cyclePhase: sys.cyclePhase,
    cycleIndex: sys.cycleIndex,
    activeCounts: inv.monomers,
    events,
    desorbEvents: desorb,
    positionsB64: encodeTyped(positions),
    velocitiesB64: encodeTyped(velocities),
    bondSlotsB64: encodeTyped(bondSlots),
    centerLinkB64: encodeTyped(centerLink),
    centerHeldStepsB64: encodeTyped(centerHeldSteps),
    bondRngB64: encodeTyped(rng.bond),
    thermoRngB64: encodeTyped(rng.thermo),
  }
}

/** The other side of encodeCheckpoint -- pure decode, no GPU, callable from Node OR the browser.
 * Returns exactly the shape createSoup(opts) expects at opts.resume, so a full resume call is
 * `createSoup({ ...file.config, resume: decodeCheckpointResume(file) })`. */
export function decodeCheckpointResume(file: CheckpointFile): NonNullable<CreateSoupOpts['resume']> {
  return {
    globalStep: file.globalStep,
    liveBox: file.liveBox,
    activeCounts: file.activeCounts,
    positions: decodeFloat32(file.positionsB64),
    velocities: decodeFloat32(file.velocitiesB64),
    bondSlots: decodeUint32(file.bondSlotsB64),
    centerLink: decodeUint32(file.centerLinkB64),
    centerHeldSteps: decodeUint32(file.centerHeldStepsB64),
    desorbEvents: new Uint32Array([file.desorbEvents.stretch, file.desorbEvents.timeout]),
    bondRng: decodeUint32(file.bondRngB64),
    thermoRng: decodeUint32(file.thermoRngB64),
    events: file.events,
  }
}
