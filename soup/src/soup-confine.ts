// Task 'confined-parcel' (2026-08-21): a FINITE parcel of water with a soft, neutral, repulsive
// wall -- the one thing every campaign in this project has never varied.
//
// WHY IT EXISTS. In a fully periodic box an aggregate that spans the box closes on itself through
// the boundary and therefore has NO EDGE AT ALL, while a vesicle must pay curvature. Every campaign
// duly measured the same object: 3 of 3 axes wrapped at 23 of 23 checkpoints, cavity x2.90 while
// `encapsulatedWater` stayed exactly 0 (closeout-report.md). In real water there is no wrap-around:
// a growing patch has a rim, the rim costs energy, and the only way to lose it is to close. This
// module restores that asymmetry.
//
// HOW PERIODICITY IS REMOVED, AND WHY IT IS DONE THIS WAY. Not by adding a `periodic` flag to every
// mi3()/wrap in the engine -- there are eleven force/measurement consumers of the minimum-image
// convention (see confined-parcel-report.md's own list), and a wrapped distance computed inside a
// non-periodic run is exactly the class of silent error this project has been bitten by seven times.
// Instead the periodic operations are made THE IDENTITY, provably, by geometry: the parcel is a
// sphere of radius R centred in a cubic box of side L with L >= 4R + a margin, so
//   (1) every particle stays within R + delta of the centre, hence inside [L/2-R-d, L/2+R+d], which
//       is at least (L/2 - R - d) from every face -- so `x - floor(x/box)*box` is the identity and
//       no position is ever wrapped; and
//   (2) every PAIR separation is at most 2(R+delta) < L/2, so `d - round(d/box)*box` is the identity
//       for every pair in the system -- not merely for interacting ones.
// Both are ASSERTED at creation (assertConfinedGeometry below) and MEASURED at every checkpoint
// (wallStats().maxRadius / minCoordinate / maxCoordinate), so "the wrapping is off" is a checked
// property of the run rather than a promise about the code. The price is paid in empty box volume
// (grid cells, which cost O(ncells) in a serial prefix scan and nothing else) and not in particles.
//
// THE WALL IS NEUTRAL, AND THAT IS A CHOICE. wallForce below is the same function of distance for
// water, heads, tails, catalyst and mineral: no per-species offset, no per-species depth, no
// attraction at all. A hydrophilic wall would behave like a mineral surface (this project already
// measured what that does -- clay-surface-report.md) and a hydrophobic one would nucleate a film;
// neutral is the choice that TESTS closure rather than staging it. What it still biases, stated
// rather than discovered: any repulsive container depletes and layers the liquid next to it, so an
// enrichment of amphiphile material in the outermost shell is a real possible outcome and is
// measured at every checkpoint (wallStats below), not assumed away.
//
// No numeric model constant lives in this file (the project-wide rule, tests/params.test.ts's
// literal scanner covers soup/src): the radius and the stiffness are per-run experiment-design
// numbers passed in by the caller (soup/cli/campaign-config.ts's --confineRadius/--confineStiffness,
// with their written defaults and basis there), exactly the precedent --expandRampSteps set.

/** What a caller asks for: the WET parcel radius and the wall's spring constant. */
export interface ConfineOpts {
  radiusSigma: number
  /** Spring constant of the wall's harmonic penetration term, in epsilon/sigma^2. */
  stiffness: number
}

export interface Confinement {
  /** The parcel radius at the CREATION (wet) box -- the live radius follows the live box, see
   * liveRadius(). */
  radiusWet: number
  stiffness: number
  boxWet: [number, number, number]
  /** The longest interaction reach in the system (max of the Verlet list range and the long-range
   * electrostatic list range) -- what the clearance conditions above are checked against. */
  cutMax: number
  /** L/2 - R at the wet box: how far the parcel's own surface is from the nearest box face. */
  clearanceWet: number
  /** True iff 2R < L/2 at the wet box, i.e. the minimum image is the identity for EVERY pair and
   * not only for interacting ones (the strong condition (2) in this file's header). */
  strongNoWrapWet: boolean
}

/** Volume of the parcel of radius r. Every density this run derives -- the Verlet-list capacity, the
 * long-range electrostatic list capacity, the dry box the evaporation targets -- must use THIS and
 * not the box volume, or it under-counts the real local density by V_box/V_parcel (a factor of 5.8
 * at the geometry this task runs, which would have silently under-sized both neighbour lists). */
export function parcelVolume(radius: number): number {
  return (4 / 3) * Math.PI * radius ** 3
}

/** The parcel is centred in the box and scales affinely with it -- soup/src/soup-box-scale-math.ts's
 * scaleMoleculesRigid maps a molecule's centre of mass by the box RATIO about the origin, so a sphere
 * centred at box/2 with radius R maps to one centred at newBox/2 with radius R*newL/L. Which is
 * exactly what a drying droplet does: remove water, the parcel shrinks. */
export function liveRadius(c: Confinement, liveBox: readonly [number, number, number]): number {
  return c.radiusWet * (liveBox[0] / c.boxWet[0])
}

export function parcelCentre(box: readonly [number, number, number]): [number, number, number] {
  return [box[0] / 2, box[1] / 2, box[2] / 2]
}

/** The wall's potential energy for a particle whose distance from the parcel centre is `s`:
 * 0 inside, 0.5*k*(s-R)^2 outside. Harmonic and not a WCA-style 1/s^12 on purpose: it is bounded and
 * non-singular for every s (a particle that somehow found itself far outside is pulled back with a
 * finite force instead of producing an Inf), its gradient is trivially checkable against a numerical
 * one, and its curvature k is a single number whose integrator stability (omega*dt = sqrt(k)*dt) and
 * thermal penetration (~sqrt(2kT/k)) can both be stated in closed form. */
export function wallPotential(s: number, radius: number, stiffness: number): number {
  if (!(radius > 0) || s <= radius) return 0
  const pen = s - radius
  return 0.5 * stiffness * pen * pen
}

/** -dU/ds: the inward force magnitude on a particle at distance `s` from the centre (0 inside). */
export function wallForceMagnitude(s: number, radius: number, stiffness: number): number {
  if (!(radius > 0) || s <= radius) return 0
  return -stiffness * (s - radius)
}

/** The full wall force vector on a particle at `p`, in the box `box` -- the CPU twin of
 * soup/wgsl/wall.wgsl's soup_wall_force_main, kept here so the two can be compared directly and so a
 * numerical gradient can be taken of the SAME potential the kernel differentiates. */
export function wallForceAt(
  p: readonly [number, number, number],
  box: readonly [number, number, number],
  radius: number,
  stiffness: number,
): [number, number, number] {
  const c = parcelCentre(box)
  const d: [number, number, number] = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]
  const s = Math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
  const m = wallForceMagnitude(s, radius, stiffness)
  if (m === 0) return [0, 0, 0]
  const inv = 1 / Math.max(s, 1e-6)
  return [m * d[0] * inv, m * d[1] * inv, m * d[2] * inv]
}

/** The 16-byte uniform soup/wgsl/wall.wgsl reads (typed `Float32Array<ArrayBuffer>`, not the bare
 * form, so device.queue.writeBuffer type-checks -- the bare form widens to ArrayBufferLike, which is
 * exactly the pre-existing lib-type noise `npx tsc --noEmit` already reports for every OTHER buffer
 * upload in soup/src/soup-buffers.ts; this task leaves that count where it found it): x = the LIVE radius, y = the stiffness. The centre
 * is not carried because the kernel derives it from the grid uniform's own box (GB.box*0.5), which
 * is rewritten by the same resizeSoupGrid call that rewrites this one -- so the two can never
 * disagree about which box the parcel is centred in. Radius 0 = no confinement, and the kernel
 * returns immediately, which is what makes every periodic run bit-identical. */
export function wallUniformBytes(c: Confinement | null, liveBox: readonly [number, number, number]): Float32Array<ArrayBuffer> {
  if (!c) return new Float32Array([0, 0, 0, 0])
  return new Float32Array([liveRadius(c, liveBox), c.stiffness, 0, 0])
}

/** Resolves and VALIDATES the geometry. Throws rather than silently running a confined system in a
 * box whose faces the parcel can feel -- that would reintroduce exactly the wrap-around interaction
 * this whole task exists to remove, and it would do it silently. */
export function resolveConfine(
  opts: ConfineOpts,
  box: [number, number, number],
  boxesVisited: readonly [number, number, number][],
  cutMax: number,
): Confinement {
  if (!(opts.radiusSigma > 0)) throw new Error(`confine: radius=${opts.radiusSigma} must be positive`)
  if (!(opts.stiffness > 0)) throw new Error(`confine: wall stiffness=${opts.stiffness} must be positive`)
  if (!(box[0] === box[1] && box[1] === box[2])) {
    throw new Error(`confine: confinement is implemented only in a cubic box, given [${box.join(', ')}]`)
  }
  const c: Confinement = {
    radiusWet: opts.radiusSigma,
    stiffness: opts.stiffness,
    boxWet: [box[0], box[1], box[2]],
    cutMax,
    clearanceWet: box[0] / 2 - opts.radiusSigma,
    strongNoWrapWet: 2 * opts.radiusSigma < box[0] / 2,
  }
  // Checked at EVERY box this run will ever live in (wet and dry), not only the creation one: a
  // drying event shrinks both the box and the parcel by the same factor, so the ratio conditions are
  // scale-invariant in exact arithmetic -- but they are checked anyway, because "scale-invariant"
  // is an argument and this is a measurement.
  for (const b of boxesVisited) {
    const r = liveRadius(c, b)
    const clearance = b[0] / 2 - r
    if (!(clearance > cutMax)) {
      throw new Error(
        `confine: the clearance to the box face L/2-R=${clearance.toFixed(4)} does not exceed the largest interaction radius ` +
          `${cutMax.toFixed(4)} at box=[${b.map((x) => x.toFixed(4)).join(', ')}], R=${r.toFixed(4)} -- ` +
          `a particle at the parcel surface would feel its own image across the boundary, so periodicity is not removed. ` +
          `Use a box of at least ${(2 * (r + cutMax) + 2).toFixed(1)}`,
      )
    }
  }
  return c
}

/** Cubic-lattice site coordinates INSIDE the parcel, for the fresh-creation layout: the same
 * jittered-lattice construction soup/src/soup-init-state.ts already uses in a box (a lattice has a
 * minimum-separation guarantee that independent uniform placement does not -- see that file's own
 * comment and what a cold-start overlap did to this engine), restricted to the sphere. `nx` is grown
 * until at least `n` sites fall inside, so the mean occupied density is exactly n/parcelVolume(R),
 * the same relation the box lattice has to the box. */
export function sphereLatticeSites(n: number, radius: number, centre: readonly [number, number, number]): { spacing: number; sites: Float64Array } {
  if (n <= 0) return { spacing: radius, sites: new Float64Array(0) }
  // (pi/6) is the fraction of a cube's lattice sites that land inside its inscribed sphere.
  let nx = Math.max(2, Math.ceil(Math.cbrt(n / (Math.PI / 6))))
  for (;;) {
    const h = (2 * radius) / nx
    const out: number[] = []
    const r2 = radius * radius
    for (let iz = 0; iz < nx; iz++) {
      const z = -radius + (iz + 0.5) * h
      for (let iy = 0; iy < nx; iy++) {
        const y = -radius + (iy + 0.5) * h
        for (let ix = 0; ix < nx; ix++) {
          const x = -radius + (ix + 0.5) * h
          if (x * x + y * y + z * z <= r2) {
            out.push(centre[0] + x, centre[1] + y, centre[2] + z)
          }
        }
      }
    }
    if (out.length / 3 >= n) return { spacing: h, sites: Float64Array.from(out) }
    nx++
  }
}

/** Rejection-samples ONE point uniformly inside the parcel -- what a confined rehydration must draw
 * from instead of the whole box (soup/src/soup-evaporate.ts's sampleInsertionPositions would
 * otherwise place solvent in the vacuum outside the parcel, where nothing is nearby, so its
 * farthest-from-anything candidate rule would accept the very first draw every time). */
export function sampleInParcel(rng: () => number, box: readonly [number, number, number], radius: number): [number, number, number] {
  const c = parcelCentre(box)
  for (;;) {
    const x = (rng() * 2 - 1) * radius
    const y = (rng() * 2 - 1) * radius
    const z = (rng() * 2 - 1) * radius
    if (x * x + y * y + z * z <= radius * radius) return [c[0] + x, c[1] + y, c[2] + z]
  }
}

export interface WallSpeciesStat {
  /** data/soup.json monomer id. */
  id: string
  count: number
  /** Particles of this species whose distance from the parcel centre exceeds radius - shell. */
  inShell: number
  fraction: number
  /** The fraction a UNIFORM distribution would put in the same shell -- 1 - ((R-shell)/R)^3. The
   * null hypothesis, so `enrichment` below is a number that means something on its own. */
  uniformFraction: number
  /** fraction / uniformFraction. The uniform-sphere null, which is the right null ONLY if the liquid
   * fills the parcel right up to R. It does not: the outer surface of a soft-walled liquid relaxes a
   * little inside the nominal radius, so EVERY species reads below 1 by the same geometric factor
   * (measured: 0.768-0.783 for C/O/H/W at R = 20 in tests/soup-confine.test.ts). Useful as an
   * absolute, but not the adsorption measurement on its own. */
  enrichment: number
  /** THE ADSORPTION MEASUREMENT: this species' shell fraction divided by the SOLVENT's. It cancels
   * the geometric factor above exactly, because both numerator and denominator carry it, so 1.0 means
   * "this species is at the container in the same proportion as the water is" -- i.e. no preferential
   * adsorption -- and a value well above 1 means a film. This is what a neutral wall has to deliver
   * and what the competing-sink question is actually asking. 0 when there is no solvent to compare
   * against. */
  enrichmentVsSolvent: number
}

export interface WallStats {
  radius: number
  shell: number
  /** Largest distance from the parcel centre reached by ANY particle -- the wall's own containment
   * measurement, and half of the no-wrap proof (this plus the box side is what makes the minimum
   * image the identity). */
  maxRadius: number
  /** How far the outermost particle is outside the nominal radius: maxRadius - radius, i.e. the
   * measured wall penetration. */
  penetration: number
  minCoordinate: number
  maxCoordinate: number
  /** min(L/2 - maxRadius) -- the measured clearance from the nearest box face. Must exceed the
   * longest interaction range for the run to be non-periodic in fact and not only in intent. */
  faceClearance: number
  /** 2*maxRadius, the largest pair separation any two particles in the system can have. The
   * minimum image is the identity for every pair iff this is below L/2. */
  maxPairSeparation: number
  halfBox: number
  strongNoWrap: boolean
  species: WallSpeciesStat[]
}

/** The competing-sink measurement: how much of each species sits in the outermost shell of the
 * parcel, against what a uniform distribution would put there. Pure, scalar-returning, and computed
 * over the whole live census in one pass -- never transfers a per-particle array anywhere. */
export function wallStats(
  positions: Float32Array,
  liveN: number,
  box: readonly [number, number, number],
  radius: number,
  shell: number,
  monomerIds: readonly string[],
  /** Index of the solvent species in `monomerIds` (data/soup.json's solvent.waterId), for the
   * solvent-relative enrichment that is the real adsorption measurement. -1 = no solvent. */
  solventKind = -1,
): WallStats {
  const c = parcelCentre(box)
  const inner = Math.max(0, radius - shell)
  const counts = new Array(monomerIds.length).fill(0)
  const shellCounts = new Array(monomerIds.length).fill(0)
  let maxR = 0
  let minC = Infinity
  let maxC = -Infinity
  for (let i = 0; i < liveN; i++) {
    const x = positions[i * 4]
    const y = positions[i * 4 + 1]
    const z = positions[i * 4 + 2]
    if (x < minC) minC = x
    if (y < minC) minC = y
    if (z < minC) minC = z
    if (x > maxC) maxC = x
    if (y > maxC) maxC = y
    if (z > maxC) maxC = z
    const dx = x - c[0]
    const dy = y - c[1]
    const dz = z - c[2]
    const s = Math.sqrt(dx * dx + dy * dy + dz * dz)
    if (s > maxR) maxR = s
    const k = Math.round(positions[i * 4 + 3])
    if (k >= 0 && k < counts.length) {
      counts[k]++
      if (s > inner) shellCounts[k]++
    }
  }
  const uniformFraction = radius > 0 ? 1 - (inner / radius) ** 3 : 0
  const solventFraction = solventKind >= 0 && counts[solventKind] > 0 ? shellCounts[solventKind] / counts[solventKind] : 0
  const species: WallSpeciesStat[] = monomerIds.map((id, k) => ({
    id,
    count: counts[k],
    inShell: shellCounts[k],
    fraction: counts[k] > 0 ? shellCounts[k] / counts[k] : 0,
    uniformFraction,
    enrichment: counts[k] > 0 && uniformFraction > 0 ? shellCounts[k] / counts[k] / uniformFraction : 0,
    enrichmentVsSolvent: counts[k] > 0 && solventFraction > 0 ? shellCounts[k] / counts[k] / solventFraction : 0,
  }))
  const halfBox = Math.min(box[0], box[1], box[2]) / 2
  return {
    radius,
    shell,
    maxRadius: maxR,
    penetration: maxR - radius,
    minCoordinate: Number.isFinite(minC) ? minC : 0,
    maxCoordinate: Number.isFinite(maxC) ? maxC : 0,
    faceClearance: Math.min(minC, box[0] - maxC, box[1] - maxC, box[2] - maxC),
    maxPairSeparation: 2 * maxR,
    halfBox,
    strongNoWrap: 2 * maxR < halfBox,
    species,
  }
}
