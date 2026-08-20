// Cluster analysis for the self-assembly gate (Task 6): does a random lipid solution spontaneously
// aggregate into one large cluster under the tail-tail cos^2 attraction? Pure CPU math — no GPU
// calls — same split as metrics.ts (density/thickness) vs sim.ts (GPU-touching area move).

/** Union-find (disjoint set) with path compression and union by size — the standard structure for
 * turning a stream of "these two are connected" edges into connected-component sizes without
 * revisiting every pair. */
class UnionFind {
  private readonly parent: Int32Array
  private readonly size: Int32Array

  constructor(n: number) {
    this.parent = new Int32Array(n)
    this.size = new Int32Array(n).fill(1)
    for (let i = 0; i < n; i++) this.parent[i] = i
  }

  find(x: number): number {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]]
      x = this.parent[x]
    }
    return x
  }

  union(a: number, b: number): void {
    let ra = this.find(a)
    let rb = this.find(b)
    if (ra === rb) return
    if (this.size[ra] < this.size[rb]) [ra, rb] = [rb, ra]
    this.parent[rb] = ra
    this.size[ra] += this.size[rb]
  }
}

/** Minimum-image displacement of one coordinate on a periodic axis of length `box` — matches mi()
 * in forces.wgsl / mi1() in sim.ts. */
function mi1(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

/** Wraps one coordinate into [0, box). */
function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

/** Shared connectivity pass for clusters()/largestClusterFraction() (Task 6) and
 * largestClusterCenter() (Task 8): builds the union-find over `positions`, joining any pair within
 * `cutoff` of each other. `positions` is the flat vec4-per-bead layout used throughout the engine
 * (x, y, z, type) — the type field is not consulted here; callers decide which bead subset defines
 * connectivity and pass just that subset in.
 *
 * x and y are always periodic (minimum-image, matching the engine's box). z is periodic only when
 * the caller says so, via `periodicZ`, and the default is FALSE because that is the membrane
 * engine's own physics: a bilayer patch in vacuum has no image across z (Task 6/8, engine/src/
 * closure.ts's open-boundary flood, tests/self-assembly.test.ts). `periodicZ = true` is the SOUP's
 * physics: soup/src/sim.ts wraps all three axes every step ("a bulk soup has no preferred axis"),
 * so with z left open an aggregate straddling the z face is split and counted TWICE -- the
 * measurement defect coalescence-report.md §6 sized at +11.9 % on the largest aggregate and fixed
 * here. Every soup call site therefore passes true; every membrane call site keeps the default.
 * Built on a cell list of side `cutoff` (so cell width is always >=
 * cutoff, guaranteeing any pair within cutoff shares a cell or one of its 26 neighbors) plus
 * union-find, not an O(N^2) double loop: the self-assembly test runs this on ~3600 beads every time
 * it samples the trajectory, and a double loop there would be the dominant cost of the whole test. */
function buildClusterUnionFind(positions: Float32Array, box: [number, number, number], cutoff: number, periodicZ = false): { uf: UnionFind; n: number } {
  const n = positions.length / 4
  const uf = new UnionFind(n)
  if (n === 0) return { uf, n }

  const [Lx, Ly, Lz] = box
  // Cells tile the periodic x,y plane exactly: nx/ny cells of width Lx/nx, Ly/ny, each >= cutoff
  // since nx = floor(Lx/cutoff) <= Lx/cutoff. When z is periodic it tiles the same way; when it is
  // open (the default) it gets a plain integer cell index of width `cutoff` with no wrapping and no
  // fixed cell count.
  const nx = Math.max(1, Math.floor(Lx / cutoff))
  const ny = Math.max(1, Math.floor(Ly / cutoff))
  const nz = periodicZ ? Math.max(1, Math.floor(Lz / cutoff)) : 0
  const wx = Lx / nx
  const wy = Ly / ny
  const wz = periodicZ ? Lz / nz : cutoff

  const cellX = new Int32Array(n)
  const cellY = new Int32Array(n)
  const cellZ = new Int32Array(n)
  const buckets = new Map<string, number[]>()

  for (let i = 0; i < n; i++) {
    const x = wrap1(positions[i * 4], Lx)
    const y = wrap1(positions[i * 4 + 1], Ly)
    const z = periodicZ ? wrap1(positions[i * 4 + 2], Lz) : positions[i * 4 + 2]
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = periodicZ ? Math.min(nz - 1, Math.floor(z / wz)) : Math.floor(z / wz)
    cellX[i] = cx
    cellY[i] = cy
    cellZ[i] = cz
    const key = `${cx},${cy},${cz}`
    const bucket = buckets.get(key)
    if (bucket) bucket.push(i)
    else buckets.set(key, [i])
  }

  const cutoff2 = cutoff * cutoff

  for (let i = 0; i < n; i++) {
    const cx = cellX[i], cy = cellY[i], cz = cellZ[i]
    const xi = positions[i * 4], yi = positions[i * 4 + 1], zi = positions[i * 4 + 2]
    for (let dx = -1; dx <= 1; dx++) {
      const ncx = ((cx + dx) % nx + nx) % nx
      for (let dy = -1; dy <= 1; dy++) {
        const ncy = ((cy + dy) % ny + ny) % ny
        for (let dz = -1; dz <= 1; dz++) {
          const ncz = periodicZ ? ((cz + dz) % nz + nz) % nz : cz + dz
          const bucket = buckets.get(`${ncx},${ncy},${ncz}`)
          if (!bucket) continue
          for (const j of bucket) {
            if (j <= i) continue // each unordered pair considered once, from the lower index
            const ddx = mi1(xi - positions[j * 4], Lx)
            const ddy = mi1(yi - positions[j * 4 + 1], Ly)
            const ddz = periodicZ ? mi1(zi - positions[j * 4 + 2], Lz) : zi - positions[j * 4 + 2]
            const r2 = ddx * ddx + ddy * ddy + ddz * ddz
            if (r2 <= cutoff2) uf.union(i, j)
          }
        }
      }
    }
  }

  return { uf, n }
}

/** Cluster sizes among the given beads — see buildClusterUnionFind() above for the connectivity
 * rule this reduces to component sizes. */
export function clusters(positions: Float32Array, box: [number, number, number], cutoff: number, periodicZ = false): number[] {
  const { uf, n } = buildClusterUnionFind(positions, box, cutoff, periodicZ)
  if (n === 0) return []
  const sizeByRoot = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    const r = uf.find(i)
    sizeByRoot.set(r, (sizeByRoot.get(r) ?? 0) + 1)
  }
  return Array.from(sizeByRoot.values())
}

/** Periodic-aware (x,y) centre of mass, plus plain (z) mean, of the single largest connected
 * cluster among `positions` (same connectivity rule as clusters() above, sharing one
 * buildClusterUnionFind() pass). Task 8's recenterOnLargestCluster (closure.ts) uses this to
 * relocate a self-assembled object's dominant structure to the box centre before flood-filling for
 * enclosed volume — a vesicle forms wherever it forms in a periodic box, and its own cavity must
 * not be seeded as "outside" just because the vesicle happens to sit near a box face.
 *
 * The (x,y) mean is computed the same way the area-move's rigid-lipid map does (sim.ts's
 * scaleLateralRigid): pick the first cluster member as a reference, accumulate minimum-image
 * offsets from it (mi1), average, then wrap the reference-plus-average back into [0, box) — the
 * only way to average points that may sit on opposite sides of a periodic wrap without the naive
 * mean being pulled toward whichever side happens to have more points near the seam. */
export function largestClusterCenter(
  positions: Float32Array,
  box: [number, number, number],
  cutoff: number,
): [number, number, number] {
  const { uf, n } = buildClusterUnionFind(positions, box, cutoff)
  if (n === 0) throw new Error('largestClusterCenter: пустой набор бидов')

  const sizeByRoot = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    const r = uf.find(i)
    sizeByRoot.set(r, (sizeByRoot.get(r) ?? 0) + 1)
  }
  let bestRoot = -1
  let bestSize = -1
  for (const [r, s] of sizeByRoot) {
    if (s > bestSize) { bestSize = s; bestRoot = r }
  }

  const [Lx, Ly] = box
  let refX = 0
  let refY = 0
  let sx = 0, sy = 0, sz = 0, count = 0
  for (let i = 0; i < n; i++) {
    if (uf.find(i) !== bestRoot) continue
    const x = positions[i * 4], y = positions[i * 4 + 1], z = positions[i * 4 + 2]
    if (count === 0) { refX = x; refY = y }
    sx += mi1(x - refX, Lx)
    sy += mi1(y - refY, Ly)
    sz += z
    count++
  }
  return [wrap1(refX + sx / count, Lx), wrap1(refY + sy / count, Ly), sz / count]
}

/** Cluster BREAKDOWN, not just sizes: every connected component among `positions` (same
 * connectivity rule as clusters()/buildClusterUnionFind() above -- one shared union-find pass), as
 * arrays of LOCAL indices into `positions` (0..n-1, i.e. matching the caller's own index into
 * whatever subset it passed in -- the caller maps these back to original particle indices itself,
 * the same way positionsFor()'s own caller already tracks that mapping). Sorted largest-first, the
 * order the soup-to-vesicle per-aggregate analysis (soup/src/aggregates.ts) walks them in: the
 * stage-deciding lamellar/vesicle candidate, by construction, cannot be a small aggregate, so a
 * caller that only wants to spend the expensive shape/cavity work on "the largest few" can just
 * take a prefix of this array. */
export function clusterComponents(positions: Float32Array, box: [number, number, number], cutoff: number, periodicZ = false): number[][] {
  const { uf, n } = buildClusterUnionFind(positions, box, cutoff, periodicZ)
  if (n === 0) return []
  const byRoot = new Map<number, number[]>()
  for (let i = 0; i < n; i++) {
    const r = uf.find(i)
    const arr = byRoot.get(r)
    if (arr) arr.push(i)
    else byRoot.set(r, [i])
  }
  return Array.from(byRoot.values()).sort((a, b) => b.length - a.length)
}

// --- per-aggregate shape: gyration tensor + principal moments (soup-to-vesicle per-aggregate task) -
//
// clusters()/largestClusterFraction() above default to the membrane engine's own periodicity
// convention (x,y periodic, z open -- a bilayer in vacuum has no image across z); the soup engine
// (soup/src/sim.ts) is a genuinely BULK system where every one of x,y,z wraps every step (that
// file's own header: "a bulk soup has no preferred axis") and therefore passes `periodicZ = true`
// to the CONNECTIVITY pass as well (see buildClusterUnionFind's own header for the double-count
// that flag removes). A micelle/vesicle candidate aggregate can
// straddle any of the three periodic faces, and computing a gyration tensor directly off wrapped
// coordinates would blow up (two beads of the same tight cluster reading box[axis] apart instead of
// a few sigma). unwrapAggregate() below is the fix -- the same minimum-image-relative-to-a-reference
// technique largestClusterCenter() already uses for x,y, generalised to whichever axes the caller
// says are periodic (all three, for the soup) -- producing one coherent local frame for shapeOf()
// to measure. largestClusterCenter() itself is deliberately NOT given the flag: its z is a plain
// mean feeding engine/src/closure.ts's open-boundary flood, which is an open-z pipeline end to end.

/** Minimum-image displacement of one coordinate, generalised from mi1() above (kept private to that
 * z-open convention) so this section can apply it per-axis under an explicit `periodic` flag. */
function miAxis(d: number, box: number): number {
  return d - Math.round(d / box) * box
}

/** Unwraps one already-known-connected aggregate (e.g. one clusterComponents() entry's own
 * positions) into a single coherent local frame: every particle is walked relative to the FIRST
 * particle using the minimum-image convention on whichever axes `periodic` flags, then offset back
 * by that reference so the result sits near the aggregate's own original location (not translated
 * to the origin) -- only internally coherent, not wrapped back into [0,box). Positions are NOT
 * wrapped into canonical [0,box) afterward: the point is a frame shapeOf()/a radial-from-centre
 * profile can measure directly, not a canonical box-relative position. `positions` must already be
 * ONE connected component under the same cutoff used to find it -- this function does not verify
 * that, it only removes the periodic-wrap discontinuity a connected object can still show. */
export function unwrapAggregate(
  positions: Float32Array,
  box: [number, number, number],
  periodic: [boolean, boolean, boolean],
): Float32Array {
  const n = positions.length / 4
  const out = new Float32Array(positions.length)
  if (n === 0) return out
  const ref: [number, number, number] = [positions[0], positions[1], positions[2]]
  for (let i = 0; i < n; i++) {
    for (let axis = 0; axis < 3; axis++) {
      const v = positions[i * 4 + axis]
      out[i * 4 + axis] = periodic[axis] ? ref[axis] + miAxis(v - ref[axis], box[axis]) : v
    }
    out[i * 4 + 3] = positions[i * 4 + 3]
  }
  return out
}

/** Ascending eigenvalues of a real symmetric 3x3 matrix, given as its six distinct entries --
 * Smith's (1961) closed-form trigonometric solution for the symmetric eigenvalue problem (the
 * standard closed form for this exact case; no library dependency, no iteration). Falls back to a
 * plain sort of the diagonal when every off-diagonal entry is already (numerically) zero -- the
 * general formula divides by a quantity that is exactly zero for a diagonal matrix. */
function symmetricEigenvalues3x3(
  Sxx: number, Syy: number, Szz: number, Sxy: number, Sxz: number, Syz: number,
): [number, number, number] {
  const p1 = Sxy * Sxy + Sxz * Sxz + Syz * Syz
  if (p1 < 1e-30) {
    return [Sxx, Syy, Szz].sort((a, b) => a - b) as [number, number, number]
  }
  const q = (Sxx + Syy + Szz) / 3
  const p2 = (Sxx - q) ** 2 + (Syy - q) ** 2 + (Szz - q) ** 2 + 2 * p1
  const p = Math.sqrt(p2 / 6)
  const Bxx = (Sxx - q) / p, Byy = (Syy - q) / p, Bzz = (Szz - q) / p
  const Bxy = Sxy / p, Bxz = Sxz / p, Byz = Syz / p
  const detB = Bxx * (Byy * Bzz - Byz * Byz) - Bxy * (Bxy * Bzz - Byz * Bxz) + Bxz * (Bxy * Byz - Byy * Bxz)
  const r = Math.max(-1, Math.min(1, detB / 2))
  const phi = Math.acos(r) / 3
  const largest = q + 2 * p * Math.cos(phi)
  const smallest = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3)
  const middle = 3 * q - largest - smallest
  return [smallest, middle, largest].sort((a, b) => a - b) as [number, number, number]
}

/** Unit eigenvector for ONE eigenvalue of a real symmetric 3x3 matrix, via the standard cross-
 * product-of-two-rows-of-(A-lambda*I) construction: any two rows of the singular matrix (A-lambda*I)
 * span its column space, so their cross product lies in the orthogonal complement, i.e. the null
 * space -- the eigenvector itself, for a non-repeated eigenvalue. Tries all three row pairs and
 * keeps the largest-magnitude cross product for numerical stability (whichever pair is least
 * parallel), returning null only when the eigenvalue is (numerically) repeated -- every pair of rows
 * degenerates to the same direction or to zero, meaning the "eigenvector" is not unique and this
 * axis must not be trusted (shapeOf()'s own doc comment on smallestAxis covers the caller's side of
 * this). */
function symmetricEigenvector3x3(
  Sxx: number, Syy: number, Szz: number, Sxy: number, Sxz: number, Syz: number, lambda: number,
): [number, number, number] | null {
  const r0: [number, number, number] = [Sxx - lambda, Sxy, Sxz]
  const r1: [number, number, number] = [Sxy, Syy - lambda, Syz]
  const r2: [number, number, number] = [Sxz, Syz, Szz - lambda]
  const cross = (a: [number, number, number], b: [number, number, number]): [number, number, number] => [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ]
  const mag = (v: [number, number, number]) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
  const candidates = [cross(r0, r1), cross(r0, r2), cross(r1, r2)]
  let best = candidates[0]
  let bestMag = mag(best)
  for (const c of candidates.slice(1)) {
    const m = mag(c)
    if (m > bestMag) { bestMag = m; best = c }
  }
  if (bestMag < 1e-9) return null
  return [best[0] / bestMag, best[1] / bestMag, best[2] / bestMag]
}

export interface ShapeMetrics {
  centre: [number, number, number]
  /** sqrt(sum of the gyration tensor's own three eigenvalues) -- the standard radius-of-gyration
   * definition, independent of shape (a sphere and a rod of the same Rg simply distribute that same
   * total spread differently across the three principal axes below). */
  radiusOfGyration: number
  /** Ascending eigenvalues of the (unweighted -- every particle counts once, matching this coarse
   * bead model) gyration tensor: principalMoments[0] <= [1] <= [2]. Shape reads off their RATIOS:
   * sphere ~= all three equal; oblate/lamellar (flat sheet or shell wall) has [0] much smaller than
   * [1]~=[2] (thin in one direction, extended equally in the other two); prolate (rod/fibre) has
   * [0]~=[1] much smaller than [2] (thin in two directions, extended in the third). */
  principalMoments: [number, number, number]
  /** Unit eigenvector of principalMoments[0] -- the "thin" direction for an oblate/lamellar shape
   * (the membrane normal of a flat patch). Only trustworthy when principalMoments[0] is genuinely
   * distinct from [1]/[2] (an oblate reading) -- for a near-isotropic (spherical) aggregate this
   * axis is numerically arbitrary, since any direction is an equally valid "smallest" eigenvector of
   * a tensor with three nearly-equal eigenvalues. [0,0,1] when the eigenvalue is (numerically)
   * repeated and symmetricEigenvector3x3 could not isolate a direction -- an arbitrary but stable
   * placeholder, never meant to be read in that case (see the caller-side check above). */
  smallestAxis: [number, number, number]
}

/** Mass-weighted (every bead weighted equally) shape of one already-connected, already-UNWRAPPED
 * (unwrapAggregate() above, if the source engine is periodic on any axis the aggregate could
 * straddle) set of positions: centre of mass, radius of gyration, and the three principal moments +
 * smallest-moment axis a caller uses to tell a sphere from a flat sheet from a rod. Pure geometry,
 * no box/periodicity argument needed here -- by the time positions reach this function they are
 * already one coherent local frame. */
export function shapeOf(positions: Float32Array): ShapeMetrics {
  const n = positions.length / 4
  if (n === 0) throw new Error('shapeOf: пустой набор частиц')
  let cx = 0, cy = 0, cz = 0
  for (let i = 0; i < n; i++) {
    cx += positions[i * 4]
    cy += positions[i * 4 + 1]
    cz += positions[i * 4 + 2]
  }
  cx /= n; cy /= n; cz /= n
  let Sxx = 0, Syy = 0, Szz = 0, Sxy = 0, Sxz = 0, Syz = 0
  for (let i = 0; i < n; i++) {
    const dx = positions[i * 4] - cx
    const dy = positions[i * 4 + 1] - cy
    const dz = positions[i * 4 + 2] - cz
    Sxx += dx * dx; Syy += dy * dy; Szz += dz * dz
    Sxy += dx * dy; Sxz += dx * dz; Syz += dy * dz
  }
  Sxx /= n; Syy /= n; Szz /= n; Sxy /= n; Sxz /= n; Syz /= n
  const principalMoments = symmetricEigenvalues3x3(Sxx, Syy, Szz, Sxy, Sxz, Syz)
  const axis = symmetricEigenvector3x3(Sxx, Syy, Szz, Sxy, Sxz, Syz, principalMoments[0]) ?? [0, 0, 1]
  const radiusOfGyration = Math.sqrt(Math.max(0, principalMoments[0] + principalMoments[1] + principalMoments[2]))
  return { centre: [cx, cy, cz], radiusOfGyration, principalMoments, smallestAxis: axis }
}

/** Fraction of TAIL beads (type field nonzero, i.e. w !== 0 — head is 0, tail1/tail2 are 1) that
 * belong to the single largest cluster. This is the self-assembly diagnostic from the brief: a
 * random lipid solution under the tail-tail attraction is expected to converge toward one dominant
 * aggregate (fraction -> 1), typically passing through a micelle stage where several small clusters
 * coexist (fraction stalled around ~0.2) before coarsening further with more simulation time. */
export function largestClusterFraction(positions: Float32Array, box: [number, number, number], cutoff: number, periodicZ = false): number {
  const n = positions.length / 4
  const tailIdx: number[] = []
  for (let i = 0; i < n; i++) if (positions[i * 4 + 3] !== 0) tailIdx.push(i)
  if (tailIdx.length === 0) return 0

  const tailPos = new Float32Array(tailIdx.length * 4)
  for (let k = 0; k < tailIdx.length; k++) {
    tailPos.set(positions.subarray(tailIdx[k] * 4, tailIdx[k] * 4 + 4), k * 4)
  }

  const sizes = clusters(tailPos, box, cutoff, periodicZ)
  const largest = sizes.length ? Math.max(...sizes) : 0
  return largest / tailIdx.length
}
