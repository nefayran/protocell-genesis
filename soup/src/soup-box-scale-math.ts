// The pure, GPU-free half of box-scaling / dry-wet cycling: the rigid-COM coordinate map plus the
// wet/dry schedule, and the creation-time cycle-config derivation built on top of them. Split out
// of soup/src/sim.ts (file-size rule in CLAUDE.md, and split again from soup/src/soup-box-scale.ts
// itself once that file also grew past 600 lines) -- every export here is still re-exported from
// soup/src/sim.ts under its original name (tests/soup-boxcycle.test.ts and
// tests/helpers/gridSearch.ts import them from `soup/src/sim` directly), and soup-box-scale.ts's
// own stateful GPU-side ramp (applyBoxScaleOnce/applyBoxScale/scaleBoxTo/growBoxTo/stepCycled)
// imports these pure functions from here rather than duplicating them.
//
// Deamer's dry-wet cycling forces closure by changing the BOX, not the membrane's own physics --
// see data/soup.json's dryWetCycle.basis for the experimental motivation and this task's amplitude/
// period reasoning. The coordinate map below reuses the discipline engine/src/sim.ts's
// scaleLateralRigid established for the membrane's fixed 3-bead lipids (its own doc comment: "cost
// four review rounds"): scale each MOLECULE's center of mass, rebuild every bead around that scaled
// center from its UNCHANGED internal offset, never rescale a bead directly. Generalised here from a
// fixed head/tail1/tail2 triple to the soup's dynamic topology (bond graph connected components,
// including singleton unbonded monomers as size-1 "molecules") and from a lateral (x,y only) scale
// to a full 3-axis isotropic one -- a bulk soup has no vacuum-facing axis the way a membrane patch
// does, so "drying" contracts every periodic axis together.

import type { CreateSoupOpts } from './soup-types'
import { planSoupGrid } from './soup-plan'
import type { Soup } from './rules'

// Minimum-image displacement of a scalar coordinate difference -- same role as engine/src/sim.ts's
// private mi1(), duplicated rather than imported (that one is a module-private helper, not exported,
// and this file's own step.wgsl/bond.wgsl mi3() is a GPU-side twin of the same formula).
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

// Wraps one coordinate into [0, box) -- the scalar form of engine/src/sim.ts's wrapXY(), generalised
// to all three axes by being called per-axis below (this soup has no open axis to skip).
function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

/** Pure per-molecule rigid coordinate map for a box change: groups particles into connected
 * components of the COVALENT bond graph (`bonds`, matching SoupSystem.bonds()'s own pair layout --
 * an unbonded monomer is its own size-1 component), scales each component's center of mass by
 * `newBox/oldBox` per axis, and rebuilds every member bead around that scaled center from its
 * UNCHANGED internal offset. No bead is ever rescaled directly, so every intramolecular (bonded)
 * distance survives the move exactly -- tests/soup-boxcycle.test.ts checks this property on
 * synthetic topologies with no GPU. The adsorption tether (soup/wgsl/bond-adsorption.wgsl's
 * centerLink) is deliberately NOT one of these edges: it is a soft, desorbable association, not a
 * covalent bond, and its own length is ALLOWED to change by this move (data/soup.json's
 * adsorption.basis already gives it a desorption valve for exactly this kind of stretch) -- see
 * soup/src/soup-box-scale.ts's applyBoxScaleOnce for how that is exercised for real.
 *
 * Offsets are accumulated by WALKING THE BOND GRAPH edge by edge (BFS from an arbitrary root in
 * each component), not by taking one minimum-image reading against a single fixed reference bead --
 * a real bug, caught by applyBoxScaleOnce's own runtime self-check on a live 250000-step run
 * (wet-dry-cycle-report.md): this project's cc_bond has no length cap, and at the elevated dry-phase
 * density this task's own cycle deliberately targets, a real carbon chain grew long/coiled enough
 * that ITS OWN two ends sat more than half the (already-shrunk) box apart -- exactly the case a
 * single-reference mi1() reading aliases (engine/src/aggregate.ts's unwrapAggregate has the SAME
 * known limitation, for the same reason, on a spatial-proximity cluster that has no explicit
 * topology to walk instead). Every individual COVALENT BOND, by construction, forms only at
 * reaction-contact distance (a couple sigma at most, soup/wgsl/bond-dispatch.wgsl) -- far under half
 * of ANY box this system ever runs at -- so accumulating one mi1() step per EDGE, never against a
 * bead that might be many bonds and sigma away, cannot alias regardless of how long or coiled the
 * chain is. */
export function scaleMoleculesRigid(
  positions: Float32Array,
  bonds: Uint32Array,
  oldBox: [number, number, number],
  newBox: [number, number, number],
): Float32Array {
  const n = positions.length / 4
  const sx = newBox[0] / oldBox[0]
  const sy = newBox[1] / oldBox[1]
  const sz = newBox[2] / oldBox[2]

  const adjacency: number[][] = Array.from({ length: n }, () => [])
  for (let k = 0; k < bonds.length; k += 2) {
    const i = bonds[k]
    const j = bonds[k + 1]
    adjacency[i].push(j)
    adjacency[j].push(i)
  }

  const visited = new Uint8Array(n)
  const ox = new Float64Array(n)
  const oy = new Float64Array(n)
  const oz = new Float64Array(n)
  const out = new Float32Array(positions.length)

  for (let root = 0; root < n; root++) {
    if (visited[root]) continue
    // BFS from `root`: ox/oy/oz[member] ends up as that member's offset relative to `root`'s own
    // (wrapped) position, accumulated one bond-length hop at a time -- see this function's own doc
    // comment for why that is the property a single-reference reading cannot guarantee.
    visited[root] = 1
    ox[root] = 0
    oy[root] = 0
    oz[root] = 0
    const queue = [root]
    const members = [root]
    let qi = 0
    while (qi < queue.length) {
      const p = queue[qi++]
      for (const c of adjacency[p]) {
        if (visited[c]) continue
        visited[c] = 1
        ox[c] = ox[p] + mi1(positions[c * 4] - positions[p * 4], oldBox[0])
        oy[c] = oy[p] + mi1(positions[c * 4 + 1] - positions[p * 4 + 1], oldBox[1])
        oz[c] = oz[p] + mi1(positions[c * 4 + 2] - positions[p * 4 + 2], oldBox[2])
        queue.push(c)
        members.push(c)
      }
    }

    let sumx = 0
    let sumy = 0
    let sumz = 0
    for (const b of members) {
      sumx += ox[b]
      sumy += oy[b]
      sumz += oz[b]
    }
    const cx = sumx / members.length
    const cy = sumy / members.length
    const cz = sumz / members.length
    const rx = positions[root * 4]
    const ry = positions[root * 4 + 1]
    const rz = positions[root * 4 + 2]
    const comX = wrap1(rx + cx, oldBox[0]) * sx
    const comY = wrap1(ry + cy, oldBox[1]) * sy
    const comZ = wrap1(rz + cz, oldBox[2]) * sz
    for (const b of members) {
      out[b * 4] = wrap1(comX + (ox[b] - cx), newBox[0])
      out[b * 4 + 1] = wrap1(comY + (oy[b] - cy), newBox[1])
      out[b * 4 + 2] = wrap1(comZ + (oz[b] - cz), newBox[2])
      out[b * 4 + 3] = positions[b * 4 + 3]
    }
  }
  return out
}

/** All particle pairs (i,j), i<j, within `cutoff` of each other by 3-axis minimum-image distance,
 * among particles flagged in `eligible` (a length-N mask; a particle with eligible[i]===0 can never
 * appear on EITHER side of a returned pair) -- used by applyBoxScaleOnce's own `cohesion` parameter
 * to widen scaleMoleculesRigid's rigid-unit grouping from "covalently bonded" to "covalently bonded
 * OR merely close, AND both eligible", so a densely-packed non-covalent aggregate (e.g. this
 * project's own vesicle candidates, held together by tail-tail attraction, never a covalent bond
 * between two different amphiphiles) scales as one rigid body rather than as many independently-
 * scaled small molecules (see applyBoxScaleOnce's own doc comment for the measured failure this
 * fixes, and for why `eligible` matters: an EARLIER version of this function searched the whole
 * system with no restriction and found ~5.5 MILLION proximity edges among 93200 particles at this
 * checkpoint's own box-average density -- enough to percolate nearly the entire system into one
 * giant rigid blob, which made the ORIGINAL overlap problem WORSE, not better, since a giant blob's
 * own single root-relative offset construction gives it zero internal freedom to redistribute at
 * all. Restricting `eligible` to just the recognised aggregate's own member particles -- the SAME
 * membership analyzeAggregates()/detectStage() already compute for measurement purposes -- keeps the
 * fix scoped to the actual densely-packed structure and leaves every dilute free monomer/catalyst
 * scaling exactly as dry-wet cycling's own already-verified per-covalent-molecule path always has).
 * Cell list of side >= cutoff, all THREE axes periodic -- mirrors engine/src/aggregate.ts's own
 * buildClusterUnionFind, generalised from that file's x,y-periodic/z-open membrane convention to
 * this soup engine's fully-periodic bulk convention, and returns edges (for scaleMoleculesRigid to
 * consume directly) rather than that file's union-find/component output, since scaleMoleculesRigid
 * already does its own BFS grouping from an edge list and there is no reason to duplicate that here. */
export function proximityPairs(positions: Float32Array, box: [number, number, number], cutoff: number, eligible: Uint8Array): Uint32Array {
  const n = positions.length / 4
  const nx = Math.max(1, Math.floor(box[0] / cutoff))
  const ny = Math.max(1, Math.floor(box[1] / cutoff))
  const nz = Math.max(1, Math.floor(box[2] / cutoff))
  const wx = box[0] / nx
  const wy = box[1] / ny
  const wz = box[2] / nz
  const cellX = new Int32Array(n)
  const cellY = new Int32Array(n)
  const cellZ = new Int32Array(n)
  const buckets = new Map<string, number[]>()
  for (let i = 0; i < n; i++) {
    if (!eligible[i]) continue
    const x = wrap1(positions[i * 4], box[0])
    const y = wrap1(positions[i * 4 + 1], box[1])
    const z = wrap1(positions[i * 4 + 2], box[2])
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = Math.min(nz - 1, Math.floor(z / wz))
    cellX[i] = cx
    cellY[i] = cy
    cellZ[i] = cz
    const key = `${cx},${cy},${cz}`
    const bucket = buckets.get(key)
    if (bucket) bucket.push(i)
    else buckets.set(key, [i])
  }
  const cutoff2 = cutoff * cutoff
  const pairs: number[] = []
  for (let i = 0; i < n; i++) {
    if (!eligible[i]) continue
    const cx = cellX[i]
    const cy = cellY[i]
    const cz = cellZ[i]
    for (let dx = -1; dx <= 1; dx++) {
      const ncx = ((cx + dx) % nx + nx) % nx
      for (let dy = -1; dy <= 1; dy++) {
        const ncy = ((cy + dy) % ny + ny) % ny
        for (let dz = -1; dz <= 1; dz++) {
          const ncz = ((cz + dz) % nz + nz) % nz
          const bucket = buckets.get(`${ncx},${ncy},${ncz}`)
          if (!bucket) continue
          for (const j of bucket) {
            if (j <= i) continue
            const ddx = mi1(positions[i * 4] - positions[j * 4], box[0])
            const ddy = mi1(positions[i * 4 + 1] - positions[j * 4 + 1], box[1])
            const ddz = mi1(positions[i * 4 + 2] - positions[j * 4 + 2], box[2])
            if (ddx * ddx + ddy * ddy + ddz * ddz <= cutoff2) pairs.push(i, j)
          }
        }
      }
    }
  }
  return new Uint32Array(pairs)
}

/** The dry box a wet `box`/`N` pair maps to under data/soup.json's dryWetCycle.targetDryDensity --
 * isotropic (volume scales as N/targetDryDensity, every axis scales by the same cube-root factor),
 * pulled out as its own pure function so a caller (soup/src/sim.ts's createSoup,
 * tests/soup-boxcycle.test.ts) can check the density it actually produces without a GPU. */
export function computeDryBox(box: [number, number, number], N: number, targetDryDensity: number): [number, number, number] {
  const wetVolume = box[0] * box[1] * box[2]
  const dryVolume = N / targetDryDensity
  const scale = Math.cbrt(dryVolume / wetVolume)
  return [box[0] * scale, box[1] * scale, box[2] * scale]
}

/** The wet/dry schedule a running system needs to know nothing about except "what step am I at" --
 * pure functions of a step count so tests/soup-boxcycle.test.ts can check the schedule with no GPU.
 * One cycle = a WET segment (this system's own creation box, `1-dryFraction` of `periodSteps`) THEN
 * a DRY segment (`dryFraction` of `periodSteps`) -- wet first because step 0 IS already the wet box
 * (CreateSoupOpts.box), so cycle 1 needs no box change at all until its own dry segment starts.
 * After `cycles` full cycles, the schedule settles at (and stays at) wet -- `cycleIndex` reports 0
 * once cycling is over, matching "did any cavity survive rehydration" needing a well-defined final
 * wet state to check, not an indefinitely repeating cycle. */
export interface CycleSchedule {
  periodSteps: number
  dryFraction: number
  cycles: number
}

function cycleTransitionSteps(cfg: CycleSchedule): number[] {
  const wetLen = cfg.periodSteps * (1 - cfg.dryFraction)
  const out: number[] = []
  for (let k = 0; k < cfg.cycles; k++) {
    out.push(k * cfg.periodSteps + wetLen) // wet -> dry, this cycle's own dry segment starts
    out.push((k + 1) * cfg.periodSteps) // dry -> wet (the LAST one is final rehydration, cycling ends)
  }
  return out
}

/** The phase ('wet'/'dry') and 1-based cycle number a given real-step count falls in -- `cycleIndex`
 * is 0 once `step` has passed every configured cycle (settled wet, cycling over). */
export function cyclePhaseAt(step: number, cfg: CycleSchedule): { phase: 'wet' | 'dry'; cycleIndex: number } {
  const total = cfg.periodSteps * cfg.cycles
  if (step >= total) return { phase: 'wet', cycleIndex: 0 }
  const within = step % cfg.periodSteps
  const wetLen = cfg.periodSteps * (1 - cfg.dryFraction)
  return { phase: within < wetLen ? 'wet' : 'dry', cycleIndex: Math.floor(step / cfg.periodSteps) + 1 }
}

/** The smallest transition step strictly AFTER `step` -- `Infinity` once cycling is over, the
 * sentinel soup/src/soup-box-scale.ts's stepCycled() uses to fall through to a plain, unbounded
 * step(). */
export function nextCycleTransition(step: number, cfg: CycleSchedule): number {
  let best = Infinity
  for (const t of cycleTransitionSteps(cfg)) {
    if (t > step && t < best) best = t
  }
  return best
}

// 3-axis minimum-image distance -- the same formula soup/wgsl/forces.wgsl's mi3() encodes on the
// GPU side, needed here only for applyBoxScaleOnce's own runtime self-check (soup-box-scale.ts): a
// plain CPU re-check that every covalent bond's length really did survive a box change, independent
// of and in addition to tests/soup-boxcycle.test.ts's own pure-function check on synthetic
// topologies.
export function mi3Distance(positions: Float32Array, box: [number, number, number], i: number, j: number): number {
  const dx = mi1(positions[i * 4] - positions[j * 4], box[0])
  const dy = mi1(positions[i * 4 + 1] - positions[j * 4 + 1], box[1])
  const dz = mi1(positions[i * 4 + 2] - positions[j * 4 + 2], box[2])
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

// --- creation-time dry-wet cycle setup, WITH its throwing validation ------------------------------

/** Dry-wet cycling (task 'wet-dry-cycle', data/soup.json's dryWetCycle.basis): resolved ONCE at
 * creation time from the file's own schedule plus this call's box/N -- never touched again unless
 * applyBoxScale changes liveBox. Returns `cycleCfg: undefined` (the guard every use site checks)
 * when cycling is not requested, so this whole feature costs nothing when off. Moved verbatim out
 * of createSoup's body (file-size rule in CLAUDE.md). */
export function deriveCycleConfig(
  soup: Soup,
  opts: CreateSoupOpts,
  box: [number, number, number],
  N: number,
  startCounts: Record<string, number>,
): { cycleCfg: CycleSchedule | undefined; dryBox: [number, number, number] } {
  const cycleEnabled = opts.dryWetCycle ?? soup.dryWetCycle.enabled
  let cycleCfg: CycleSchedule | undefined
  let dryBox: [number, number, number] = box
  if (cycleEnabled) {
    const dwc = soup.dryWetCycle
    const wetVolume = box[0] * box[1] * box[2]
    const wetDensity = N / wetVolume
    if (dwc.targetDryDensity <= wetDensity) {
      throw new Error(
        `data/soup.json: dryWetCycle.targetDryDensity=${dwc.targetDryDensity} не превышает текущую ` +
          `плотность бульона ${wetDensity.toFixed(4)} (N=${N}, box=[${box}]) -- сухая фаза обязана концентрировать, не разбавлять`,
      )
    }
    dryBox = computeDryBox(box, N, dwc.targetDryDensity)
    // Guard: the neighbour grid must stay VALID (task requirement 2 -- "fewer than three cells on a
    // periodic axis, or a box too small for the minimum-image convention" -- exactly planSoupGrid's
    // own `valid`, generalised from a fixed "3" to this soup's own minCells=2*effectiveWalkRadius+1)
    // at the smallest box the cycle visits. A box change that merely CHANGES the cell count (wet
    // dims != dry dims) is deliberately NOT rejected here: soup/src/soup-buffers.ts's
    // resizeSoupGrid destroys and reallocates the ncells-sized buffers and rebuilds every bind
    // group referencing them whenever dims actually changes, mirroring engine/src/sim.ts's own
    // resizeGrid -- the SAME capability this task adds, generalised from the membrane's area move
    // to the soup's dry-wet cycle. planSoupGrid's `valid` is the one thing a reallocation cannot
    // fix (a genuinely invalid geometry, not a merely-different-but-legal cell count), so it
    // remains the sole guard here.
    const dryPlan = planSoupGrid(dryBox, startCounts)
    if (!dryPlan.valid) throw new Error(dryPlan.reason!)
    cycleCfg = { periodSteps: dwc.periodSteps, dryFraction: dwc.dryFraction, cycles: dwc.cycles }
  }
  return { cycleCfg, dryBox }
}
