// Public types for engine/src/sim.ts's createSystem/System — split out of sim.ts by responsibility
// (file-size rule, root CLAUDE.md). Pure types, no logic; moved verbatim, byte-for-byte, from the
// original sim.ts.

export type Layout = 'random' | 'bilayer' | 'vesicle'

export interface CreateSystemOpts {
  lipids: number
  box: [number, number, number]
  seed: number
  layout: Layout
  /** Overrides thermostat.gamma from params.json — needed to run the drift test at gamma=0. */
  gamma?: number
  /** Widens the neighbor-grid cell size as if attraction.wc were at least this value, WITHOUT
   * changing the actual physics wc used at creation (that still comes from params.json/gamma
   * override above). Needed by viewer/, whose w_c slider calls System.setLiveParams() after
   * creation to retune a RUNNING system — the grid's cell size is fixed at creation time (see the
   * `cellSize` comment below for why a frozen-too-small grid silently drops forces), so a caller
   * that plans to raise wc above its initial value after the fact must say so up front here. */
  maxWc?: number
  /** Test-only escape hatch: when given, replaces the layout function's output with these exact
   * bead positions (4 floats per bead: x,y,z,type), skipping RNG entirely. `layout` is still
   * required by the type but is not consulted. Lets a test build a fixed, hand-picked
   * configuration (no RNG) to check against an independently-computed reference. */
  positions?: Float32Array
  /** Paired with `positions`: fixed initial velocities (4 floats per bead: vx,vy,vz,0). Defaults
   * to the same Gaussian(kT) initialization used otherwise if omitted. */
  velocities?: Float32Array
}

export interface System {
  /** Advances n Langevin/velocity-Verlet steps. Encoded as one command buffer: no per-step
   * CPU<->GPU round trip. */
  step(n: number): Promise<void>
  /** 4 floats per bead: x, y, z, type (0 = head, 1 = tail). */
  positions(): Promise<Float32Array>
  /** Mean of squared velocity components over all beads (mass = 1) — i.e. kinetic energy PER
   * DEGREE OF FREEDOM, which in these reduced units equals kT directly (not kT/2: equipartition
   * gives <0.5*m*v_x^2> = 0.5*kT per dof, so <v_x^2> = kT). Returning kT/2 here would fail the
   * equipartition test by exactly a factor of two, since it compares this value directly against
   * params.thermostat.kT. */
  kineticEnergyPerDof(): Promise<number>
  /** Kinetic + potential energy of the whole system. */
  totalEnergy(): Promise<number>
  /** Same physics as forces(), full O(N^2) pair loop instead of the neighbor grid — for
   * cross-checking the grid result. */
  forcesBruteForce(): Promise<Float32Array>
  forces(): Promise<Float32Array>
  /** Wall-clock time (ms, GPU-inclusive) of the most recent neighbor-grid rebuild. */
  neighborBuildMs: number
  /** Zero-tension Metropolis Monte Carlo move on the box's lateral area. Proposes s = exp(u),
   * u ~ U(-delta, +delta), multiplies L_x and L_y by sqrt(s) (L_z and every bead's z untouched),
   * and displaces each LIPID rigidly: its center of mass in x,y is scaled by sqrt(s) and its three
   * beads are rebuilt around that scaled center from their unchanged internal offsets, so every
   * intramolecular distance survives the move exactly and DeltaU is purely intermolecular. Accepts
   * with
   * min(1, exp(-(DeltaU - N*kT*ln(A'/A))/kT)) = min(1, exp(N*u - DeltaU/kT)) where N is the number
   * of LIPIDS (the objects whose coordinates are being scaled, hence the count that enters the
   * configurational Jacobian) and DeltaU is the change in total POTENTIAL energy only (kinetic
   * energy is untouched by a positional move, and lateral tension is zero so there is no
   * gamma*DeltaA term). Runs `trials` such moves and returns the accepted fraction. Rebuilds the
   * neighbor grid after every trial (accepted or not) since the box the grid's cell/box uniform
   * refers to may have changed. */
  areaMove(trials: number): Promise<number>
  /** Rewrites the params uniform buffer in place (kT and/or wc), WITHOUT recreating the system or
   * touching any other engine state — the running simulation keeps its positions, velocities,
   * step count and neighbor grid exactly as they were. Only subsequent kick/force/thermostat
   * passes see the new value. Added for viewer/'s two sliders (kT/epsilon and w_c), which must
   * retune a live simulation rather than restart it.
   *
   * Validates and THROWS rather than clamping: wc above the value the neighbor grid was built for
   * (params.attraction.wc, or createSystem's `maxWc` override if given) would silently drop
   * forces beyond cell range -- the exact failure mode `cellSize`'s doc comment above describes,
   * just reached live instead of at construction; kT outside data/params.json's `kTRange` is
   * rejected the same way. A clamp would hide the caller's mistake; the throw names the requested
   * value, the value the grid was actually built for (or the configured range), and the resulting
   * cellSize, so misuse fails at the call site instead of showing up later as wrong physics. */
  setLiveParams(overrides: { kT?: number; wc?: number }): void
  /** Current box lengths — a live snapshot, since areaMove() mutates L_x, L_y in place. */
  readonly box: [number, number, number]
  /** Number of lipids the system was created with (fixed for its lifetime). */
  readonly lipids: number
  /** Cumulative count of integration steps taken via step(), across all calls so far. */
  readonly steps: number
}
