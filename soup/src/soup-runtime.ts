// The shared, mutable context threaded through the soup engine's stateful modules -- a plain
// object playing the exact role the original monolithic sim.ts's own closure over local `let`/
// `const` variables played (device, buffers, bind groups, grid dims, live box, global step, cycle
// phase/index): every field here is one of those former locals, given a home so the functions that
// used to close over them directly can instead take `rt: SoupRuntime` as an explicit parameter --
// this is what makes the split into separate files possible without duplicating state or changing
// any read/write order. No physics, no formula, lives in this file -- it is a type plus the "live"
// sub-object's own small mutable state (grid dims/ncells/wgCells and liveBox/globalStep/cyclePhase/
// cycleIndex), nothing else.

import type { SoupPipelines } from './soup-pipelines'
import type { SoupBuffers, SoupGridState } from './soup-buffers'
import type { SoupBindGroups } from './soup-bindgroups'
import type { ProtonationState } from './soup-protonation'
import type { ResolvedRule } from './soup-plan'
import type { Soup } from './rules'
import type { Params } from '../../engine/src/params'
import type { CycleSchedule } from './soup-box-scale-math'
import type { ClayLayout } from './soup-clay'
import type { EvaporationPlan } from './soup-evaporate'

/** Mutated by soup/src/soup-box-scale.ts (liveBox, cyclePhase, cycleIndex) and
 * soup/src/soup-integrate.ts (globalStep) -- the exact fields the original sim.ts exposed through
 * SoupSystem's live getters (`box`, `cyclePhase`, `cycleIndex`, `steps`). */
export interface SoupLiveState {
  liveBox: [number, number, number]
  globalStep: number
  cyclePhase: 'wet' | 'dry' | 'none'
  cycleIndex: number
}

export interface SoupRuntime {
  device: GPUDevice
  pipe: SoupPipelines
  soup: Soup
  p: Params
  rules: ResolvedRule[]
  catalystKind: number
  eventRuleIds: [string, string][]
  startCounts: Record<string, number>
  /** LIVE active particle count. Constant for a system that does not evaporate its solvent; mutated
   * by soup/src/soup-evaporate.ts's setActiveCount when solvent beads leave or return (task
   * 'evaporation', 2026-08-20) -- every readback, dispatch size and census reads it, which is why it
   * is the single source of truth rather than being re-derived anywhere. */
  N: number
  wgN: number
  /** Task 'long-range-electrostatics' (2026-08-20): the number of titratable beads this system can
   * hold (its creation census of electrostatics.chargedKind -- the species is never created or
   * destroyed) and the workgroup count for the head-indexed long-range kernels. 0 without charge,
   * which is what makes every long-range dispatch a no-op there. */
  esHeads: number
  wgEsHeads: number
  verlet: Soup['verletList']
  effectiveWalkRadius: number
  bondAttemptInterval: number
  sortedGather: boolean
  /** Task 'clay-surface' (2026-08-19): how many particles are immobile (soup/src/soup-clay.ts's
   * platelet). Used by soup/src/soup-box-scale.ts's applyBoxScaleOnce to REFUSE a box change rather
   * than silently stretching the platelet's rigid lattice. */
  frozenCount: number
  /** The platelet's derived geometry, or null when this system carries no mineral phase. */
  clay: ClayLayout | null
  /** The immutable WET/creation box (CreateSoupOpts.box) -- never mutated after createSoup returns;
   * `live.liveBox` is the one that moves under dry-wet cycling / scaleBoxTo / growBoxTo. */
  box: [number, number, number]
  dryBox: [number, number, number]
  cycleCfg: CycleSchedule | undefined
  /** Task 'evaporation' (2026-08-20): resolved once at creation when the run asked for solvent
   * removal, `undefined` otherwise -- and `undefined` is what makes the pre-existing box-scaling-only
   * cycle path byte-for-byte unchanged. */
  evap: EvaporationPlan | undefined
  buf: SoupBuffers
  bind: SoupBindGroups
  grid: SoupGridState
  live: SoupLiveState
  /** Task 'electrostatics' (2026-08-20): the constant-pH Monte Carlo's own mutable state (its
   * checkpointed RNG, when the next sweep is due, the last sweep's statistics) plus the resolved
   * electrostatics basis. Present on EVERY system -- with `es.enabled` false when the run did not ask
   * for charge, which makes soup/src/soup-protonation.ts's sweep a no-op and the GPU term identically
   * zero. */
  protonation: ProtonationState
}
