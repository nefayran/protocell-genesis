import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// CATALYST TURNOVER, pinned (task 'catalyst-turnover-and-window', 2026-08-20,
// .superpowers/sdd/2026-08-16-soup-to-vesicle/catalyst-turnover-and-window-report.md).
//
// WHAT BROKE, measured (verify/center-nucleation.ts, run on the predecessor window run's own
// checkpoints): soup/wgsl/bond-adsorption.wgsl's propagateOnCenter nucleation branch claimed `j` --
// the arbitrary one of the two bare carbons -- as the new chain tip and then required a catalyst to
// be within reaction contact of THAT carbon, throwing away the catalyst-to-`i` contact bondFormWalk
// had already established. Nucleation therefore needed a closed TRIANGLE (carbon i - carbon j,
// carbon j - catalyst, catalyst - carbon i) at one instant. Measured on the box-54 window run: the
// triangle count fell 393 -> 172 -> 3 -> 1 -> 0 over steps 15 000 ... 120 000 and then sat at
// EXACTLY 0 for the remaining 180 000 steps, with 0 of 1021 centres holding a tip and cc_bond /
// co_bond frozen to the digit -- while 5125 bare carbons and 58 387 free heads were still in the box
// and 234 of those bare carbons were in contact with a FREE catalyst. The centres were not stuck
// (597 of 1021 were measured re-claiming a tip earlier in that same run) and the reagents were not
// gone. The fix makes the tip the carbon whose catalyst contact is already known (i).
//
// WHY THIS SHAPE OF TEST. A hand-built triangle-open fixture was tried first and does not work, for
// a measured physical reason worth recording: BOTH contacts a nucleation needs sit INSIDE the
// repulsive part of their pair potential (a cc_bond candidate needs r < wca_cut(bPairB(C,C)) =
// 1.12246 and the catalyst reach is r < wca_cut(bPairB(C,M)) = 1.23471, and WCA's own minimum is AT
// its cutoff), so a static fixture placed inside those radii is pushed out of them before the first
// bond-dispatch cycle 20 steps later -- measured: 300 open sites, 200 steps, cc_bond = 0 even WITH
// the fix. Nucleation in this engine is a collision event, so the pin is a small real soup instead.
//
// THE PIN, and its A/B. This composition is the O:C = 4.0 point tail-length-and-window-report.md
// sec 2.2 calibrated, scaled to box 20 by the exact volume ratio (20/30)^3 = 0.2963 so the total
// density is the same 0.6778 sigma^-3 measured stable at box 30 and box 54. Both arms below were
// measured on ONE implementation, at 120 000 steps, seed 19, by reverting ONLY the two-line tip
// choice in soup/wgsl/bond-adsorption.wgsl and re-running this exact file (report sec 3.1):
//
//   quantity at step 120 000        PRE-FIX   POST-FIX
//   cumulative nucleations (chains)     117        192
//   amphiphiles                          56        129
//   bare carbons left of 779            133         33
//   cc_bond / co_bond               436/304    456/351
//
// The three assertions below are placed between the two arms with >= 1.28x margin on each side, so
// they separate the two states of the code rather than sitting near run-to-run noise (GPU atomics
// make repeated runs of the same seed differ slightly: the post-fix nucleation count measured 186 at
// 40 000 steps and 192 at 120 000 steps across two invocations).
//
// NOT asserted, on purpose: "chemistry still rising at the very end". At box 20 the carbon pool is
// genuinely exhausted by step 120 000 (33 bare carbons left of 779, and the fix is what consumes
// them), so a plateau HERE is honest exhaustion, which is exactly what the predecessor run's plateau
// was NOT. The distinction is the whole point of the diagnosis and must not be asserted away.
test(
  'the catalyst turns over: in one and the same run more chains nucleate and more carbon is consumed',
  async () => {
    const page = await gpuPage()
    page.on('console', (m) => console.log(`[page] ${m.text()}`))
    const r = await page.evaluate(async () => {
      const api = (window as any).api
      // box 20 = the box-30 O:C=4.0 calibration composition scaled by (20/30)^3 = 0.2963, so the
      // total density is the same 0.6778 sigma^-3 measured stable there and at box 54.
      const sys = await api.createSoup({
        box: [20, 20, 20],
        seed: 19,
        kT: 1.1,
        clay: false, // soup/cli/campaign.ts's own pin, so this fixture matches the campaign it pins
        start: { C: 779, O: 3117, H: 779, M: 52, W: 695 },
      })
      const trace: {
        step: number
        cc: number
        co: number
        occupied: number
        chains: number
        amph: number
        desorb: { stretch: number; timeout: number }
        bareC: number
      }[] = []
      const EMPTY = 0xffffffff
      const sample = async (step: number) => {
        const [ev, links, slots, pos, des] = await Promise.all([
          sys.events(),
          sys.centerLinks(),
          sys.bondSlots(),
          sys.particles(),
          sys.desorbEvents(),
        ])
        const N = pos.length / 4
        const soup = await api.loadSoup()
        const kindOf = (i: number) => soup.monomers[Math.round(pos[i * 4 + 3])].kind
        let occupied = 0
        let bareC = 0
        const deg = new Int32Array(N)
        for (let i = 0; i < N; i++) for (let s = 0; s < 3; s++) if (slots[i * 3 + s] !== EMPTY) deg[i]++
        for (let i = 0; i < N; i++) {
          const k = kindOf(i)
          if (k === 'catalyst' && links[i] !== EMPTY) occupied++
          if (k === 'carbon' && deg[i] === 0) bareC++
        }
        // number of bonded components containing at least one carbon -- with cc_break disabled and
        // chain-chain coupling refused, this rises by exactly one per nucleation and falls by one
        // per two-tailed head, so it is a checkpoint-derivable cumulative nucleation count.
        const parent = new Int32Array(N)
        for (let i = 0; i < N; i++) parent[i] = i
        const find = (x: number): number => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x] } return x }
        for (let i = 0; i < N; i++) for (let s = 0; s < 3; s++) { const j = slots[i * 3 + s]; if (j !== EMPTY) { const a = find(i), b = find(j); if (a !== b) parent[a] = b } }
        const carbonRoots = new Set<number>()
        for (let i = 0; i < N; i++) if (kindOf(i) === 'carbon' && deg[i] > 0) carbonRoots.add(find(i))
        const amph = (await api.findAmphiphiles(pos, await sys.bonds(), soup.monomers)).length
        trace.push({ step, cc: ev['cc_bond'] ?? 0, co: ev['co_bond'] ?? 0, occupied, chains: carbonRoots.size, amph, desorb: des, bareC })
      }
      for (let s = 0; s < 8; s++) {
        await sys.step(15000)
        await sample((s + 1) * 15000)
      }
      const pos = await sys.particles()
      let nonFinite = 0
      for (let k = 0; k < pos.length; k++) if (!Number.isFinite(pos[k])) nonFinite++
      const inv = await sys.invariants()
      sys.dispose()
      return { trace, nonFinite, monomers: inv.monomers, charge: inv.charge }
    })

    for (const t of r.trace) {
      console.log(
        `TURNOVER step=${t.step} cc_bond=${t.cc} co_bond=${t.co} occupiedCentres=${t.occupied}/52 `
          + `chains=${t.chains} amphiphiles=${t.amph} bareCarbons=${t.bareC} desorbTimeout=${t.desorb.timeout}`,
      )
    }
    const last = r.trace[r.trace.length - 1]
    const prev = r.trace[r.trace.length - 2]
    console.log(
      `TURNOVER-VERDICT nonFinite=${r.nonFinite} charge=${r.charge} monomers=${JSON.stringify(r.monomers)} `
        + `ccOverLast15k=${last.cc - prev.cc} chainsOverLast15k=${last.chains - prev.chains} occupiedAtEnd=${last.occupied} `
        + `chains=${last.chains} amphiphiles=${last.amph} bareCarbonsLeft=${last.bareC}`,
    )

    expect(r.nonFinite).toBe(0)
    expect(r.monomers).toEqual({ C: 779, O: 3117, H: 779, M: 52, W: 695, K: 0 })
    // A working catalytic cycle leaves centres holding chains -- a liveness check, not the pin
    // (pre-fix measured 6 occupied here too: the pre-fix defect is that a centre stops STARTING
    // chains, not that it never holds one).
    expect(last.occupied).toBeGreaterThan(0)
    // The pin, all three measured on both sides of the two-line change (see the header table):
    // more chains nucleated, more carbon actually incorporated, more amphiphiles produced.
    expect(last.chains).toBeGreaterThanOrEqual(150)
    expect(last.bareC).toBeLessThanOrEqual(80)
    expect(last.amph).toBeGreaterThanOrEqual(100)
  },
  600_000,
)
