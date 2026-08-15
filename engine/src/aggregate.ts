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

/** Cluster sizes among the given beads, joining any pair within `cutoff` of each other. `positions`
 * is the flat vec4-per-bead layout used throughout the engine (x, y, z, type) — the type field is
 * not consulted here; callers decide which bead subset defines connectivity (the self-assembly
 * facade below uses tail beads only, per the brief's definition) and pass just that subset in.
 *
 * x and y are periodic (minimum-image, matching the engine's box); z is open — exactly the physics
 * (only x,y wrap; z does not). Built on a cell list of side `cutoff` (so cell width is always >=
 * cutoff, guaranteeing any pair within cutoff shares a cell or one of its 26 neighbors) plus
 * union-find, not an O(N^2) double loop: the self-assembly test runs this on ~3600 beads every time
 * it samples the trajectory, and a double loop there would be the dominant cost of the whole test. */
export function clusters(positions: Float32Array, box: [number, number, number], cutoff: number): number[] {
  const n = positions.length / 4
  if (n === 0) return []

  const [Lx, Ly] = box
  // Cells tile the periodic x,y plane exactly: nx/ny cells of width Lx/nx, Ly/ny, each >= cutoff
  // since nx = floor(Lx/cutoff) <= Lx/cutoff. z is unbounded (open boundary) so it gets a plain
  // integer cell index of width `cutoff` with no wrapping and no fixed cell count.
  const nx = Math.max(1, Math.floor(Lx / cutoff))
  const ny = Math.max(1, Math.floor(Ly / cutoff))
  const wx = Lx / nx
  const wy = Ly / ny

  const cellX = new Int32Array(n)
  const cellY = new Int32Array(n)
  const cellZ = new Int32Array(n)
  const buckets = new Map<string, number[]>()

  for (let i = 0; i < n; i++) {
    const x = wrap1(positions[i * 4], Lx)
    const y = wrap1(positions[i * 4 + 1], Ly)
    const z = positions[i * 4 + 2]
    const cx = Math.min(nx - 1, Math.floor(x / wx))
    const cy = Math.min(ny - 1, Math.floor(y / wy))
    const cz = Math.floor(z / cutoff)
    cellX[i] = cx
    cellY[i] = cy
    cellZ[i] = cz
    const key = `${cx},${cy},${cz}`
    const bucket = buckets.get(key)
    if (bucket) bucket.push(i)
    else buckets.set(key, [i])
  }

  const uf = new UnionFind(n)
  const cutoff2 = cutoff * cutoff

  for (let i = 0; i < n; i++) {
    const cx = cellX[i], cy = cellY[i], cz = cellZ[i]
    const xi = positions[i * 4], yi = positions[i * 4 + 1], zi = positions[i * 4 + 2]
    for (let dx = -1; dx <= 1; dx++) {
      const ncx = ((cx + dx) % nx + nx) % nx
      for (let dy = -1; dy <= 1; dy++) {
        const ncy = ((cy + dy) % ny + ny) % ny
        for (let dz = -1; dz <= 1; dz++) {
          const ncz = cz + dz
          const bucket = buckets.get(`${ncx},${ncy},${ncz}`)
          if (!bucket) continue
          for (const j of bucket) {
            if (j <= i) continue // each unordered pair considered once, from the lower index
            const ddx = mi1(xi - positions[j * 4], Lx)
            const ddy = mi1(yi - positions[j * 4 + 1], Ly)
            const ddz = zi - positions[j * 4 + 2] // z open: no periodic image
            const r2 = ddx * ddx + ddy * ddy + ddz * ddz
            if (r2 <= cutoff2) uf.union(i, j)
          }
        }
      }
    }
  }

  const sizeByRoot = new Map<number, number>()
  for (let i = 0; i < n; i++) {
    const r = uf.find(i)
    sizeByRoot.set(r, (sizeByRoot.get(r) ?? 0) + 1)
  }
  return Array.from(sizeByRoot.values())
}

/** Fraction of TAIL beads (type field nonzero, i.e. w !== 0 — head is 0, tail1/tail2 are 1) that
 * belong to the single largest cluster. This is the self-assembly diagnostic from the brief: a
 * random lipid solution under the tail-tail attraction is expected to converge toward one dominant
 * aggregate (fraction -> 1), typically passing through a micelle stage where several small clusters
 * coexist (fraction stalled around ~0.2) before coarsening further with more simulation time. */
export function largestClusterFraction(positions: Float32Array, box: [number, number, number], cutoff: number): number {
  const n = positions.length / 4
  const tailIdx: number[] = []
  for (let i = 0; i < n; i++) if (positions[i * 4 + 3] !== 0) tailIdx.push(i)
  if (tailIdx.length === 0) return 0

  const tailPos = new Float32Array(tailIdx.length * 4)
  for (let k = 0; k < tailIdx.length; k++) {
    tailPos.set(positions.subarray(tailIdx[k] * 4, tailIdx[k] * 4 + 4), k * 4)
  }

  const sizes = clusters(tailPos, box, cutoff)
  const largest = sizes.length ? Math.max(...sizes) : 0
  return largest / tailIdx.length
}
