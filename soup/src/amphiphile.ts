// Task 3: recognising an amphiphile in the bond graph, purely from structure -- no GPU calls, no
// numeric model constants (data/params.json / data/soup.json own those; params.test.ts's literal
// scanner covers this directory).
//
// AN AMPHIPHILE IS A COMPONENT WITH EXACTLY ONE POLAR PARTICLE AND NO BRANCHING ANYWHERE ELSE,
// nothing looser: the whole point of this project's claim is that amphiphiles EMERGED from
// soup/src/rules.ts's bonding rules, so a recogniser that also accepts branched clusters,
// two-headed chains, or headless chains would let the phase map (Task 6) count assembly that never
// happened. Since "two-tails" (data/soup.json's headPlacement.chainCapacity, now up to 2) a head
// may legitimately sit with degree 2 -- one bond to EACH tail -- so the polar particle is now
// allowed to be a BRANCH POINT, but only that one particle: every non-polar (carbon) particle keeps
// the same degree<=2 cap it always had (soup/wgsl/bond.wgsl's chain pool is still exactly 2 slots
// per carbon), so the only shape a valid component can take, once the polar particle's own degree
// is bounded the same way, is a PATH -- either the head sits at one end (a single tail, degree 1)
// or in the interior (two tails, degree 2, i.e. a "Y" whose branch point IS the head, one arm per
// tail). No other branch point is possible: a carbon with degree 3 (two chain bonds plus a head, or
// three chain bonds) is rejected outright, which is exactly what catches a head buried mid-chain or
// a genuinely branched tail. Checks per connected component of the bond graph, all necessary:
//  - exactly one polar particle in the component (soup/src/rules.ts's Monomer.polar) -- zero heads
//    (a bare carbon chain) or two heads (a chain capped at both ends) are both explicitly rejected
//    by tests/soup-amphiphile.test.ts, and both are real configurations this soup's C-C/C-O rules
//    can and do produce.
//  - every NON-polar particle has degree <= 2 -- a branch anywhere but the head (a T-branch through
//    a carbon, or a head buried mid-chain, which gives that carbon 2 chain bonds + 1 head bond =
//    degree 3) is rejected here, independently of what the head's own degree is.
//  - the one polar particle has degree 1 or 2 -- 0 (a bare, unbonded head) or >2 (more tails than
//    data/soup.json's own chainCapacity should ever let form) are both rejected.
//  - the component has exactly (size-1) edges -- a tree, not a cycle: with every degree already
//    bounded at <=2, a cycle is the only OTHER shape those bounds would allow (a ring has no free
//    ends for a tail to terminate at), so this check is what turns "degree-bounded" into "a path or
//    a Y with the head at the fork", not merely "no vertex of degree >2".

import type { Monomer } from './rules'

/** One recognised amphiphile: `headIndex` is the polar particle's index into the `particles`
 * array findAmphiphiles was given; `chain` is every non-polar particle index in the component,
 * ordered as tail 1 walked outward from the head followed by tail 2 (empty for a single-tailed
 * amphiphile) -- the concatenation is what soup/src/aggregates.ts's memberIndicesOf() needs to
 * attribute the WHOLE molecule (both tails) to one aggregate; `length` is chain.length, the total
 * non-polar (carbon) unit count across both tails, which is what soup/src/rules.ts's chemistry and
 * the Flory-style length distribution (a later task) both count by; `tailLengths` reports each
 * tail's own length separately (one entry for a single-tailed amphiphile, two for a two-tailed one,
 * in the same order as `chain`'s own concatenation) since asymmetric tails matter for shape. */
export interface Amphiphile {
  headIndex: number
  chain: number[]
  length: number
  tailLengths: number[]
}

/** Finds every amphiphile in a snapshot's bond graph. `particles` is the engine's flat
 * vec4-per-particle layout (x, y, z, kind-index into `monomers`), matching SoupSystem.particles();
 * `bonds` is pairs of particle indices, matching SoupSystem.bonds(); `monomers` is
 * loadSoup().monomers, whose `polar` field decides which particles can be a chain end. */
export function findAmphiphiles(particles: Float32Array, bonds: Uint32Array, monomers: Monomer[]): Amphiphile[] {
  const n = particles.length / 4
  const polarByKind = monomers.map((m) => m.polar)
  const isPolar = (i: number): boolean => {
    const kind = Math.round(particles[i * 4 + 3])
    return polarByKind[kind] ?? false
  }

  // Adjacency list, built once from the bond pairs.
  const adjacency: number[][] = Array.from({ length: n }, () => [])
  for (let k = 0; k < bonds.length; k += 2) {
    const i = bonds[k]
    const j = bonds[k + 1]
    adjacency[i].push(j)
    adjacency[j].push(i)
  }

  // Union-find over the bond graph -- connected components are the candidate chains.
  const parent = new Int32Array(n)
  for (let i = 0; i < n; i++) parent[i] = i
  const find = (x: number): number => {
    while (parent[x] !== x) {
      parent[x] = parent[parent[x]]
      x = parent[x]
    }
    return x
  }
  for (let k = 0; k < bonds.length; k += 2) {
    const ra = find(bonds[k])
    const rb = find(bonds[k + 1])
    if (ra !== rb) parent[ra] = rb
  }

  const components = new Map<number, number[]>()
  for (let i = 0; i < n; i++) {
    const root = find(i)
    const members = components.get(root)
    if (members) members.push(i)
    else components.set(root, [i])
  }

  const result: Amphiphile[] = []
  for (const members of components.values()) {
    let sumDegree = 0
    let polarCount = 0
    let polarIdx = -1
    let violated = false
    for (const idx of members) {
      const degree = adjacency[idx].length
      sumDegree += degree
      if (isPolar(idx)) {
        polarCount++
        polarIdx = idx
        if (degree < 1 || degree > 2) violated = true // bare head, or more tails than any cap allows
      } else if (degree > 2) {
        violated = true // branch anywhere but the head -- a T-branch, or a head buried mid-chain
      }
    }
    if (violated) continue
    if (polarCount !== 1) continue // no head, or more than one -- both rejected by design
    const edgeCount = sumDegree / 2 // each edge counted from both its endpoints above
    if (edgeCount !== members.length - 1) continue // a ring, the only other shape degree<=2 allows

    // Walk each of the head's own branches (one for a single tail, two for a two-tailed "Y") outward
    // to its free end -- with every non-head degree already bounded at <=2, each branch is a simple
    // path with no further forking, so a linear walk per branch is exhaustive.
    const tails: number[][] = []
    for (const start of adjacency[polarIdx]) {
      const tail: number[] = []
      let prev = polarIdx
      let cur = start
      for (;;) {
        tail.push(cur)
        const next = adjacency[cur].find((x) => x !== prev)
        if (next === undefined) break
        prev = cur
        cur = next
      }
      tails.push(tail)
    }

    const chain = tails.flat()
    result.push({ headIndex: polarIdx, chain, length: chain.length, tailLengths: tails.map((t) => t.length) })
  }

  return result
}

/** Counts amphiphiles by chain length (number of carbon units) -- the raw histogram the Flory
 * comparison (a later task) checks against a geometric prediction with no fitting. */
export function amphiphileHistogram(a: Amphiphile[]): Record<number, number> {
  const hist: Record<number, number> = {}
  for (const x of a) hist[x.length] = (hist[x.length] ?? 0) + 1
  return hist
}
