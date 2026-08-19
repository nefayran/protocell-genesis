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
import type { ResolvedRule } from './soup-plan'
import type { Soup } from './rules'
import type { Params } from '../../engine/src/params'
import type { CycleSchedule } from './soup-box-scale-math'
import type { ClayLayout } from './soup-clay'

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
  N: number
  wgN: number
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
  buf: SoupBuffers
  bind: SoupBindGroups
  grid: SoupGridState
  live: SoupLiveState
}
