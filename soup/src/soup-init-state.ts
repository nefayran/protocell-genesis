// Initial system state: either a jittered lattice (fresh creation) or a prior checkpoint's own
// recorded state (CreateSoupOpts.resume) -- the "restore" half of checkpoint/resume (the "capture"
// half is just soup/src/soup-readback.ts's plain readbacks, which is all soup/src/checkpoint.ts's
// encodeCheckpoint ever needed). Split out of soup/src/sim.ts's createSoup (file-size rule in
// CLAUDE.md) as the one function that produces every CPU-side array createSoup uploads at creation
// time, moved verbatim -- no formula, no validation message, changed.

import type { CreateSoupOpts } from './soup-types'
import { NONE_U32 } from './soup-types'
import type { ResolvedRule } from './soup-plan'
import type { loadSoup } from './rules'
import { clayLatticePosition, claySiteIndices, type ClayLayout } from './soup-clay'

// --- seeded RNG for reproducible initial layouts -------------------------------------------------

function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function gaussian(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-9)
  const u2 = rng()
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2)
}

export interface InitialState {
  positions0: Float32Array
  velocities0: Float32Array
  bondSlots0: Uint32Array
  centerLink0: Uint32Array
  centerHeldSteps0: Uint32Array
  desorbEventsInit: Uint32Array
  bondRng0: Uint32Array
  thermoRng0: Uint32Array
  eventsInit: Uint32Array
  /** Task 'clay-surface' (2026-08-19): 1 for every particle that belongs to the rigid mineral
   * platelet (its clay beads plus the catalyst beads immobilised on it as surface sites), 0 for
   * everything else. All zeros when this system has no platelet, in which case NOTHING else in this
   * function behaves differently either -- the clay-free path is byte-identical to the pre-task one.
   *
   * Typed `Uint32Array<ArrayBuffer>` (not the bare `Uint32Array` the older fields above use) so
   * soup/src/soup-buffers.ts's writeBuffer call type-checks: the bare form widens to
   * ArrayBufferLike, which is exactly the pre-existing lib-type noise `npx tsc --noEmit` already
   * reports for every OTHER buffer upload in that file. No new noise was left behind by this task. */
  frozen0: Uint32Array<ArrayBuffer>
}

/** Produces every CPU-side array createSoup uploads at creation time, either freshly generated (a
 * jittered lattice, all bookkeeping zeroed) or loaded from `opts.resume` -- see
 * CreateSoupOpts.resume's own doc comment (soup/src/soup-types.ts) for exactly what a checkpoint
 * carries and why the coarse grid/Verlet list are deliberately NOT among these arrays. */
export function buildInitialState(
  soup: ReturnType<typeof loadSoup>,
  opts: CreateSoupOpts,
  N: number,
  countsByKind: number[],
  box: [number, number, number],
  rules: ResolvedRule[],
  eventRuleIds: [string, string][],
  clay: ClayLayout | null,
): InitialState {
  const rng = mulberry32(opts.seed)

  // --- initial state: a jittered lattice, not independent uniform placement -----------------------
  // Independent uniform placement has no minimum-separation guarantee: at data/soup.json's default
  // density (~13100 particles in a 30^3 box, ~1.27 sigma average spacing against WCA cores up to
  // ~1.35 sigma) it is only a matter of trials before two particles land near-coincident, and
  // wca_dv diverges as r -> 0 -- measured here as positions overflowing float32 range (+-2^23 and
  // beyond) after exactly ONE kick+drift, not a slow drift. A lattice with the SAME spacing has no
  // such tail risk (nearest-neighbour separation is bounded below by construction); a small jitter
  // keeps it from being a perfectly artificial starting configuration. Species are interleaved by
  // shuffling which lattice SITE each particle gets (a seeded Fisher-Yates), not by shuffling
  // positions within a kind's own block, so the soup starts well-mixed rather than segregated into
  // one spatial region per monomer kind.
  const positions0 = new Float32Array(N * 4)
  const velocities0 = new Float32Array(N * 4)
  const frozen0 = new Uint32Array(N)
  // Checkpoint/resume: a resumed system's positions/velocities are the checkpoint's own recorded
  // state, not a fresh lattice+jitter -- the whole point of resuming being "continue the SAME
  // trajectory", not "restart with the right particle count". `rng` above is still consumed further
  // down (bondRng0/thermoRng0's own fallback branch), so it is not wasted even on a resume; it is
  // simply not this block's OWN source of positions/velocities any more.
  if (opts.resume) {
    if (opts.resume.positions.length !== N * 4) {
      throw new Error(
        `createSoup: резюме содержит ${opts.resume.positions.length / 4} частиц, а состав этого вызова даёт N=${N} -- checkpoint не соответствует конфигурации`,
      )
    }
    if (opts.resume.velocities.length !== N * 4) {
      throw new Error(`createSoup: резюме содержит ${opts.resume.velocities.length / 4} скоростей, а N=${N}`)
    }
    positions0.set(opts.resume.positions)
    velocities0.set(opts.resume.velocities)
  } else {
    let nx = Math.max(1, Math.ceil(Math.cbrt(N)))
    while (nx * nx * nx < N) nx++
    const spacing: [number, number, number] = [box[0] / nx, box[1] / nx, box[2] / nx]
    const jitterFrac = 0.15
    const nSites = nx * nx * nx
    const siteOrder = Array.from({ length: nSites }, (_, i) => i)
    for (let i = siteOrder.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1))
      const tmp = siteOrder[i]
      siteOrder[i] = siteOrder[j]
      siteOrder[j] = tmp
    }
    // Task 'clay-surface' (2026-08-19): with a platelet present, the platelet's OWN beads are placed
    // on its rigid lattice (not on this jittered one), and every other particle's z is squeezed into
    // the space the sheet does not occupy. Both are no-ops when `clay` is null, so a clay-free system
    // takes the identical code path it always did -- same RNG draws in the same order, same
    // coordinates bit for bit.
    //
    // WHY z IS REMAPPED RATHER THAN RE-DRAWN. A lattice site landing inside the sheet's WCA core
    // would start the run with a genuine overlap against an IMMOBILE bead, which is the one overlap
    // the system cannot relax by moving both partners apart -- data/soup.json's own startBasis §2
    // records what a cold-start overlap did to this engine (a 24.58 sigma displacement in 10 steps).
    // Rejecting-and-redrawing would consume a seed-dependent number of RNG draws and silently change
    // every subsequent particle's position; an affine squeeze of the z coordinate into the free slab
    // keeps the draw sequence, the x/y layout and the velocities exactly as they were and only
    // compresses the axis the sheet blocks. The free slab is genuinely smaller with a platelet in the
    // box (the mineral occupies volume), so this is physics, not a workaround.
    const claySites = clay ? claySiteIndices(clay) : new Set<number>()
    const clayFree: number[] = []
    if (clay) {
      for (let k = 0; k < clay.latticeCount; k++) if (!claySites.has(k)) clayFree.push(k)
      if (claySites.size !== clay.siteCount) {
        throw new Error(`createSoup: разметка центров на пластине дала ${claySites.size} позиций вместо ${clay.siteCount}`)
      }
    }
    // The free slab runs from just above the TOP sheet plane, around through the periodic wrap, to
    // just below the BOTTOM one -- one contiguous interval of length zSpan starting at zStart.
    const zStart = clay ? clay.planeZ[clay.planeZ.length - 1] + clay.exclusionHalfWidth : 0
    const zSpan = clay ? box[2] - (clay.planeZ[clay.planeZ.length - 1] - clay.planeZ[0]) - 2 * clay.exclusionHalfWidth : box[2]
    if (clay && zSpan <= 0) {
      throw new Error(
        `createSoup: пластина глины (${clay.planeZ.length} слоёв) вместе с исключённой зоной ${clay.exclusionHalfWidth.toFixed(3)}σ ` +
          `не оставляет места в коробке высотой ${box[2]}σ`,
      )
    }
    // The mineral beads and the surface sites are consumed in lattice order as their kind comes up in
    // the composition loop below; these two cursors are what keeps that order deterministic.
    let siteCursor = 0
    let mineralCursor = 0
    const siteList = clay ? [...claySites].sort((a, b) => a - b) : []

    let idx = 0
    for (let kind = 0; kind < countsByKind.length; kind++) {
      for (let c = 0; c < countsByKind[kind]; c++) {
        const isMineral = clay !== null && kind === clay.mineralKind
        const isSite = clay !== null && kind === clay.catalystKind && siteCursor < clay.siteCount
        if (isMineral || isSite) {
          const k = isMineral ? clayFree[mineralCursor++] : siteList[siteCursor++]
          const [px, py, pz] = clayLatticePosition(clay!, k)
          // No jitter and no velocity: this bead is frozen, so a jitter would only be a permanent
          // lattice defect and a velocity would be a number the integrator refuses to use.
          positions0[idx * 4 + 0] = px
          positions0[idx * 4 + 1] = py
          positions0[idx * 4 + 2] = pz
          positions0[idx * 4 + 3] = kind
          frozen0[idx] = 1
          idx++
          continue
        }
        const site = siteOrder[idx]
        const ix = site % nx
        const iy = Math.floor(site / nx) % nx
        const iz = Math.floor(site / (nx * nx))
        positions0[idx * 4 + 0] = (ix + 0.5) * spacing[0] + (rng() * 2 - 1) * jitterFrac * spacing[0]
        positions0[idx * 4 + 1] = (iy + 0.5) * spacing[1] + (rng() * 2 - 1) * jitterFrac * spacing[1]
        const zRaw = (iz + 0.5) * spacing[2] + (rng() * 2 - 1) * jitterFrac * spacing[2]
        if (clay) {
          const u = (((zRaw / box[2]) % 1) + 1) % 1
          positions0[idx * 4 + 2] = (((zStart + u * zSpan) % box[2]) + box[2]) % box[2]
        } else {
          positions0[idx * 4 + 2] = zRaw
        }
        positions0[idx * 4 + 3] = kind
        const s = Math.sqrt(opts.kT)
        velocities0[idx * 4 + 0] = s * gaussian(rng)
        velocities0[idx * 4 + 1] = s * gaussian(rng)
        velocities0[idx * 4 + 2] = s * gaussian(rng)
        idx++
      }
    }
    if (clay && (mineralCursor !== clay.mineralCount || siteCursor !== clay.siteCount)) {
      throw new Error(
        `createSoup: пластина разложена не полностью — минеральных ${mineralCursor}/${clay.mineralCount}, центров ${siteCursor}/${clay.siteCount}`,
      )
    }
  }

  const bondSlots0 = new Uint32Array(N * 3).fill(NONE_U32)
  // Surface growth (surface-growth-report.md): one association slot per particle -- see
  // soup/wgsl/bond-adsorption.wgsl's own header ("Surface growth") for what a catalyst vs a carbon
  // stores in it. All-unassociated at creation, exactly like bondSlots0.
  const centerLink0 = new Uint32Array(N).fill(NONE_U32)
  // Surface growth / adsorption (adsorption-report.md): per-particle hold-timeout clock
  // (soup/wgsl/bond-adsorption.wgsl's centerHeldSteps) -- zero-filled, meaningful only once a centre
  // claims a chain (see that file's own comments on where it is reset/incremented/checked).
  const centerHeldSteps0 = new Uint32Array(N)
  // Surface growth / adsorption: [0]=stretch-triggered desorptions, [1]=timeout-triggered
  // desorptions -- see SoupSystem.desorbEvents()'s own doc comment.
  const desorbEventsInit = new Uint32Array(2)
  const bondRng0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) bondRng0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 2654435761) ^ 0x9e3779b9
  const thermoRng0 = new Uint32Array(N)
  for (let i = 0; i < N; i++) thermoRng0[i] = (opts.seed >>> 0) ^ Math.imul(i + 1, 3266489917) ^ 0x85ebca6b

  // Checkpoint/resume: overwrite every one of the arrays above (all already sized N/N*3/2 by their
  // own declarations, so the .set()s below cannot mismatch in length except by the explicit checks
  // here) with the checkpoint's OWN state, in place of this system's fresh-creation defaults --
  // exactly the same "resume overrides the lattice defaults" discipline positions0/velocities0
  // apply above, generalised to the rest of the mutable state CreateSoupOpts.resume documents.
  const eventsInit = new Uint32Array(rules.length * 2)
  if (opts.resume) {
    const r = opts.resume
    if (r.bondSlots.length !== N * 3) throw new Error(`createSoup: резюме содержит ${r.bondSlots.length} bondSlots-слотов, ожидалось ${N * 3}`)
    if (r.centerLink.length !== N) throw new Error(`createSoup: резюме содержит ${r.centerLink.length} centerLink-записей, ожидалось ${N}`)
    if (r.centerHeldSteps.length !== N) throw new Error(`createSoup: резюме содержит ${r.centerHeldSteps.length} centerHeldSteps-записей, ожидалось ${N}`)
    if (r.desorbEvents.length !== 2) throw new Error(`createSoup: резюме содержит ${r.desorbEvents.length} desorbEvents-счётчиков, ожидалось 2`)
    if (r.bondRng.length !== N) throw new Error(`createSoup: резюме содержит ${r.bondRng.length} bondRng-состояний, ожидалось ${N}`)
    if (r.thermoRng.length !== N) throw new Error(`createSoup: резюме содержит ${r.thermoRng.length} thermoRng-состояний, ожидалось ${N}`)
    bondSlots0.set(r.bondSlots)
    centerLink0.set(r.centerLink)
    centerHeldSteps0.set(r.centerHeldSteps)
    desorbEventsInit.set(r.desorbEvents)
    bondRng0.set(r.bondRng)
    thermoRng0.set(r.thermoRng)
    // eventRuleIds pulled forward from where it lived in the pre-split sim.ts (only needed there
    // for events()'s own readback formatting) so checkpoint/resume can use the SAME (bond id, break
    // id) -> array-position mapping to go the other way: resume.events is a Record<string,number>
    // BY RULE ID (matching events()'s own return shape, see SoupSystem.events()), not a positional
    // array, so it survives a future reordering of data/soup.json's own rules list -- looked up by
    // id here rather than trusted to already be at the right index.
    for (let ri = 0; ri < rules.length; ri++) {
      eventsInit[ri * 2 + 0] = opts.resume.events[eventRuleIds[ri][0]] ?? 0
      eventsInit[ri * 2 + 1] = opts.resume.events[eventRuleIds[ri][1]] ?? 0
    }
  }

  return { positions0, velocities0, bondSlots0, centerLink0, centerHeldSteps0, desorbEventsInit, bondRng0, thermoRng0, eventsInit, frozen0 }
}
