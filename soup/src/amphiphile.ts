// Task 3: recognising an amphiphile in the bond graph, purely from structure -- no GPU calls, no
// numeric model constants (data/params.json / data/soup.json own those; params.test.ts's literal
// scanner covers this directory).
//
// AN AMPHIPHILE IS A CHAIN WITH EXACTLY ONE POLAR END, nothing looser: the whole point of this
// project's claim is that amphiphiles EMERGED from soup/src/rules.ts's bonding rules, so a
// recogniser that also accepts branched clusters, two-headed chains, or headless chains would let
// the phase map (Task 6) count assembly that never happened. Three checks per connected component
// of the bond graph, all of them necessary:
//  - simple path: no particle has more than two bonds (a branch means it is not a chain at all --
//    caught by maxDegree>2 below), and the component has exactly (size-1) edges (a component with
//    max degree <=2 but size edges is a CYCLE, not a path, and a ring has no ends for a polar
//    particle to sit at -- caught by the edgeCount==size-1 check, which a maxDegree check alone
//    would miss).
//  - exactly one polar particle in the component (soup/src/rules.ts's Monomer.polar) -- zero heads
//    (a bare carbon chain) or two heads (a chain capped at both ends) are both explicitly rejected
//    by tests/soup-amphiphile.test.ts, and both are real configurations this soup's C-C/C-O rules
//    can and do produce.
//  - that one polar particle sits AT AN END of the path (degree <= 1 within the component), not
//    partway along it -- a head grafted onto the middle of a carbon chain is not "one polar end",
//    it is a T-branch through the head, which the maxDegree check already independently excludes,
//    but this check is the one that states the requirement directly rather than as a side effect.

import type { Monomer } from './rules'

/** One recognised amphiphile: `headIndex` is the polar particle's index into the `particles`
 * array findAmphiphiles was given; `chain` is the ordered sequence of non-polar particle indices
 * walked outward from the head to the chain's free end; `length` is chain.length -- the number of
 * non-polar (carbon) units, which is what soup/src/rules.ts's chemistry and the Flory-style length
 * distribution (a later task) both count by. */
export interface Amphiphile {
  headIndex: number
  chain: number[]
  length: number
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
    let maxDegree = 0
    let polarCount = 0
    let polarIdx = -1
    for (const idx of members) {
      const degree = adjacency[idx].length
      sumDegree += degree
      if (degree > maxDegree) maxDegree = degree
      if (isPolar(idx)) {
        polarCount++
        polarIdx = idx
      }
    }
    if (maxDegree > 2) continue // branch -- not a chain at all
    const edgeCount = sumDegree / 2 // each edge counted from both its endpoints above
    if (edgeCount !== members.length - 1) continue // a ring: max degree <=2 but no free end
    if (polarCount !== 1) continue // no head, or more than one -- both rejected by design
    if (adjacency[polarIdx].length > 1) continue // the one polar particle must sit at an END
    if (members.length - 1 < 1) continue // a bare, unbonded polar particle is not a "chain"

    // Walk the path outward from the polar end to build the ordered non-polar chain.
    const chain: number[] = []
    let prev = -1
    let cur = polarIdx
    for (;;) {
      const next = adjacency[cur].find((x) => x !== prev)
      if (next === undefined) break
      chain.push(next)
      prev = cur
      cur = next
    }

    result.push({ headIndex: polarIdx, chain, length: chain.length })
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
