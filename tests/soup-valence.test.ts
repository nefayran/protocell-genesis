import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadSoup } from '../soup/src/rules'

afterAll(shutdownGpu)

// Regression guard for the valence cap the bond kernel is supposed to enforce (soup/wgsl/bond.wgsl's
// tryClaimSlot: two chain-pool slots per carbon for C-C bonds, one head slot per carbon for a C-O
// bond, up to data/soup.json's headPlacement.chainCapacity slots per head). Reads real bond/species
// state back from the GPU AFTER a real run (never asserts on intent) and counts degree straight from
// the bond list -- if any particle's chain-bond count or head-bond count ever exceeds its cap, this
// must fail, not the recogniser's own (unrelated) rejection of branched components as non-amphiphiles.
//
// The carbon cap (2 chain bonds + 1 head bond) is the model's stated design (data/soup.json's
// monomers are carbon/head; the two-bond-kinds ruleset -- cc_bond, co_bond -- implies it), not a
// number pulled from soup/src: this test only encodes the SAME two integers (2, 1) the kernel's own
// slot layout (2 chain slots + 1 head slot per particle) already commits to. The head cap is READ
// from data/soup.json's own headPlacement.chainCapacity ("two-tails" task raised it from 1 to 2) --
// this test asserts no head ever exceeds WHATEVER that configured value is, not a number hardcoded
// here, so it stays a true regression guard across future capacity changes too.
test(
  'ни один атом углерода не превышает 2 связей C-C и 1 связь C-O; ни одна голова не превышает configured chainCapacity связей, за 45000 шагов',
  async () => {
    const soup = loadSoup()
    const carbonId = soup.monomers.find((m) => m.kind === 'carbon')!.id
    const headId = soup.monomers.find((m) => m.kind === 'head')!.id
    const kindIndex = new Map(soup.monomers.map((m, idx) => [m.id, idx]))
    const carbonKind = kindIndex.get(carbonId)!
    const headKind = kindIndex.get(headId)!
    const headChainCapacity = soup.headPlacement.chainCapacity

    const page = await gpuPage()
    const r = await page.evaluate(
      async (carbonKind: number, headKind: number) => {
        const api = (window as any).api
        const sys = await api.createSoup({ box: [30, 30, 30], seed: 11, kT: 1.1 })
        await sys.step(45000)
        const pos = await sys.particles()
        const edges = await sys.bonds()
        const N = pos.length / 4
        const kindOf = new Int32Array(N)
        for (let i = 0; i < N; i++) kindOf[i] = Math.round(pos[i * 4 + 3])
        const ccDeg = new Int32Array(N)
        const coDeg = new Int32Array(N)
        const headDeg = new Int32Array(N)
        for (let e = 0; e < edges.length; e += 2) {
          const i = edges[e]
          const j = edges[e + 1]
          const ki = kindOf[i]
          const kj = kindOf[j]
          if (ki === carbonKind && kj === carbonKind) {
            ccDeg[i]++
            ccDeg[j]++
          } else if (ki === carbonKind && kj === headKind) {
            coDeg[i]++
            headDeg[j]++
          } else if (kj === carbonKind && ki === headKind) {
            coDeg[j]++
            headDeg[i]++
          }
        }
        let maxCC = 0
        let maxCO = 0
        let maxHead = 0
        for (let i = 0; i < N; i++) {
          if (kindOf[i] === carbonKind) {
            if (ccDeg[i] > maxCC) maxCC = ccDeg[i]
            if (coDeg[i] > maxCO) maxCO = coDeg[i]
          } else if (kindOf[i] === headKind) {
            if (headDeg[i] > maxHead) maxHead = headDeg[i]
          }
        }
        return { N, edgeCount: edges.length / 2, maxCC, maxCO, maxHead }
      },
      carbonKind,
      headKind,
    )
    expect(r.edgeCount).toBeGreaterThan(0) // sanity: the run actually formed bonds
    expect(r.maxCC).toBeLessThanOrEqual(2)
    expect(r.maxCO).toBeLessThanOrEqual(1)
    expect(r.maxHead).toBeLessThanOrEqual(headChainCapacity)
  },
  280_000,
)
