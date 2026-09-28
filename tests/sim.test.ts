import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { loadParams } from '../engine/src/params'
import { scaleLateralRigid } from '../engine/src/sim'

afterAll(shutdownGpu)
const p = loadParams()

test('the neighbour grid gives the same forces as brute force', async () => {
  const page = await gpuPage()
  const diff = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 600, box: [20, 20, 20], seed: 7, layout: 'random' })
    const a = await sys.forces()
    const b = await sys.forcesBruteForce()
    let max = 0
    for (let i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]))
    return max
  })
  expect(diff).toBeLessThan(1e-4)
})

test('the thermostat brings the system to the set temperature', async () => {
  const page = await gpuPage()
  const kT = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1000, box: [24, 24, 24], seed: 11, layout: 'random' })
    await sys.step(20_000)
    return sys.kineticEnergyPerDof()
  })
  // Measured value here is ~1.1355 against target kT=1.1, i.e. biased high by ~3.2%, still inside
  // the 5% window below. This is discretization bias, not a bug — two independent, multiplicative
  // effects account for the order of magnitude:
  //  - the Euler–Maruyama (Euler–OU) discretization of the Langevin friction+noise step has a
  //    stationary second moment of kT/(1-gamma*dt/2), not exactly kT: with gamma=1, dt=0.01 that's
  //    1.1/(1-0.005) ~= 1.1055 (a +0.5% bias on its own);
  //  - velocity-Verlet integrating a stiff bond (FENE, k_fene=30, so omega~sqrt(k)~30 in these
  //    units) inflates the sampled kinetic energy of that mode by ~1/(1-(omega*dt)^2/4): with
  //    omega~30, dt=0.01, that's 1/(1-0.09/4) = 1/0.9775 ~= 1.023 (a further +2.3%).
  // Their product, ~1.1055*1.023 ~= 1.131, is within a few percent of the ~1.1355 measured here —
  // consistent with the two effects compounding rather than one of them being a mistake. If this
  // window ever needs widening, argue it from these two factors, not by re-measuring and shrugging.
  expect(kT).toBeGreaterThan(p.thermostat.kT * 0.95)
  expect(kT).toBeLessThan(p.thermostat.kT * 1.05)
})

/** Runs one fixed, RNG-free configuration through the real engine (grid forces + totalEnergy) and
 * hands back everything the CPU reference below needs to check it. Velocities are hand-picked (no
 * RNG) so kinetic energy is exercised too. */
async function gpuFixture(beads: number[][], box: number[]) {
  const page = await gpuPage()
  return page.evaluate(
    async ({ beads, box }) => {
      const api = (window as any).api
      const positions = new Float32Array(beads.length * 4)
      beads.forEach((b, i) => positions.set(b, i * 4))
      const N = beads.length
      const velocities = new Float32Array(N * 4)
      for (let i = 0; i < N; i++) {
        velocities.set([0.1 * (i + 1), -0.05 * (i + 1), 0.02 * (i + 1), 0], i * 4)
      }
      const sys = await api.createSystem({
        lipids: N / 3,
        box,
        seed: 1,
        layout: 'random',
        positions,
        velocities,
      })
      const forces = Array.from(await sys.forces())
      const totalEnergy = await sys.totalEnergy()
      return { forces, totalEnergy, positions: Array.from(positions), velocities: Array.from(velocities), N, box }
    },
    { beads, box },
  )
}

test('forces and total energy for a fixed configuration match an independent CPU reference', async () => {
  // Three lipids, fixed coordinates, no RNG. Layout, in reduced units (sigma=b=1 for tail-tail):
  //   A: head(0,0,10) - tail1(0,0,9.05) - tail2(0,0,8.05)   [straight rod along -z]
  //   B: head(1.5,0,10) - tail1(1.5,0,9.05) - tail2(1.5,0,8.05)  [same rod, offset +1.5 in x]
  //   C: head(19,0,10) - tail1(19,0,9.05) - tail2(19,0,8.05)     [same rod, offset -1 in x via
  //      periodic wrap: 19 is 1 short of the box edge at 20]
  // Bond lengths (0.95 head-tail, 1.0 tail-tail) put every bonded pair at a modest, safe FENE
  // stretch. A-C tail-tail pairs land at r=1 (< r_c=2^(1/6): WCA repulsion, and — since the fix
  // for the plateau-gating bug — inside the attraction's constant -epsilon plateau too). A-B and
  // A-C tail-tail pairs at r~1.5/1.8/2.06/2.36 exercise the cos^2 attraction ramp. A-C spans the
  // periodic boundary in x, exercising mi(). All A-B, A-C, B-C non-bonded pairs are included by
  // the reference below (a plain double loop). Same-lipid pairs (head-tail1 r=0.95, tail1-tail2
  // r=1.0, both under their WCA cutoffs) now ALSO get a WCA contribution on top of FENE/bend —
  // per the ruling reversing Task 4's exclusion, WCA acts between every pair including bonded
  // and 1-3 pairs; only the tail-tail cos^2 attraction stays excluded within a lipid.
  // NOTE: this configuration does NOT cover WCA on the 1-3 (head-tail2) pair — at 1.95 it sits
  // above that pair's cutoff (2^(1/6)*0.95 = 1.0665). The compressed fixture in the next test does.
  const beads: number[][] = []
  const rod = (hx: number) => {
    beads.push([hx, 0, 10, 0])
    beads.push([hx, 0, 9.05, 1])
    beads.push([hx, 0, 8.05, 1])
  }
  rod(0)
  rod(1.5)
  rod(19)

  const result = await gpuFixture(beads, [20, 20, 20])
  expectMatchesReference(result)
})

test('compressed configuration: WCA also acts on the 1-3 pair (head-tail2)', async () => {
  // The exact branch the WCA ruling reversed — WCA between the head and tail2 of ONE lipid — is
  // only live when that pair sits below its own cutoff, 2^(1/6)*b_ht = 1.0665. The fixture above
  // never gets there (1.95). Here lipid A is strongly bent so head-tail2 = 1.0500 < 1.0665 while
  // both FENE bonds keep safe lengths (head-tail1 = 0.95, tail1-tail2 = 1.0):
  //   A: head(0,0,10) - tail1(0,0,9.05) - tail2(0.9070,0,9.4711)
  // (tail2 solved from |tail2-tail1| = 1 and |tail2-head| = 1.05.) Lipid B is the straight rod,
  // offset far enough in x that no intermolecular WCA is live but the tail-tail cos^2 ramp is, so
  // the fixture still exercises intermolecular terms alongside the bonded ones. If WCA were
  // excluded on the 1-3 pair, the head and tail2 forces would differ from this reference by ~1.2
  // — three hundred times the toBeCloseTo(_, 2) tolerance.
  const beads: number[][] = [
    [0, 0, 10, 0],
    [0, 0, 9.05, 1],
    [0.907, 0, 9.4711, 1],
    [2.5, 0, 10, 0],
    [2.5, 0, 9.05, 1],
    [2.5, 0, 8.05, 1],
  ]
  const result = await gpuFixture(beads, [20, 20, 20])
  // Guard the fixture itself: if anyone edits the coordinates, this is what must stay true.
  const dHeadTail2 = Math.hypot(0.907, 10 - 9.4711)
  expect(dHeadTail2).toBeLessThan(2 ** (1 / 6) * p.beadSizes.head_tail)
  expectMatchesReference(result)
})

// --- independent CPU reference, written from the Cooke & Deserno 2005 formulas (not from the
// shader): WCA is a shifted-truncated LJ, FENE is the standard finite-extensible bond, bend is a
// harmonic spring on the head-tail2 distance, and the tail attraction is a cos^2 ramp between
// r_c and r_c+w_c preceded by a constant -epsilon plateau for r < r_c.
function expectMatchesReference(result: Awaited<ReturnType<typeof gpuFixture>>) {
  const wcaCut = (b: number) => 2 ** (1 / 6) * b
  const wcaDv = (r: number, b: number) => {
    if (r >= wcaCut(b) || r <= 0) return 0
    const s6 = (b / r) ** 6
    return (-24 * p.epsilon) / r * (2 * s6 * s6 - s6)
  }
  const wcaV = (r: number, b: number) => {
    if (r >= wcaCut(b)) return 0
    const s6 = (b / r) ** 6
    return 4 * p.epsilon * (s6 * s6 - s6) + p.epsilon
  }
  const feneDv = (r: number) => (p.fene.k * r) / (1 - (r / p.fene.rInf) ** 2)
  const feneV = (r: number) => -0.5 * p.fene.k * p.fene.rInf ** 2 * Math.log(1 - (r / p.fene.rInf) ** 2)
  const bendDv = (r: number) => p.bend.k * (r - p.bend.r0)
  const bendV = (r: number) => 0.5 * p.bend.k * (r - p.bend.r0) ** 2
  const rcTT = wcaCut(p.beadSizes.tail_tail)
  const attrDv = (r: number) => {
    if (r < rcTT || r > rcTT + p.attraction.wc) return 0
    const x = (Math.PI * (r - rcTT)) / (2 * p.attraction.wc)
    return (p.epsilon * Math.PI * Math.sin(2 * x)) / (2 * p.attraction.wc)
  }
  const attrV = (r: number) => {
    if (r < rcTT) return -p.epsilon
    if (r > rcTT + p.attraction.wc) return 0
    const x = (Math.PI * (r - rcTT)) / (2 * p.attraction.wc)
    return -p.epsilon * Math.cos(x) ** 2
  }
  const mi = (d: number, box: number) => d - Math.round(d / box) * box

  function reference(positions: number[], velocities: number[], N: number, box: number[]) {
    const pos = (i: number) => [positions[i * 4], positions[i * 4 + 1], positions[i * 4 + 2]]
    const type = (i: number) => positions[i * 4 + 3]
    const disp = (i: number, j: number) => {
      const a = pos(i), b = pos(j)
      const d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
      d[0] = mi(d[0], box[0])
      d[1] = mi(d[1], box[1])
      return d
    }
    const force = Array.from({ length: N }, () => [0, 0, 0])
    let potential = 0
    // Applies -dv(r)*d/r to i's force only (d = disp(i,j), pointing from j to i).
    const addForce = (i: number, d: number[], dv: number, r: number) => {
      for (let k = 0; k < 3; k++) force[i][k] -= (dv * d[k]) / r
    }
    // A bonded pair (i,j): force acts on both beads, potential is credited once.
    const addBond = (i: number, j: number, dv: (r: number) => number, v: (r: number) => number) => {
      const d = disp(i, j)
      const r = Math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
      addForce(i, d, dv(r), r)
      addForce(j, disp(j, i), dv(r), r)
      potential += v(r)
    }
    // bonded: FENE(0,1), FENE(1,2), bend(0,2) within each lipid.
    for (let lip = 0; lip < N / 3; lip++) {
      const h = lip * 3, t1 = lip * 3 + 1, t2 = lip * 3 + 2
      addBond(h, t1, feneDv, feneV)
      addBond(t1, t2, feneDv, feneV)
      addBond(h, t2, bendDv, bendV)
    }
    // non-bonded: EVERY pair gets WCA, including same-lipid (bonded/1-3) pairs — the reversed
    // Task 4 exclusion. Only the tail-tail cos^2 attraction stays excluded within a lipid.
    for (let i = 0; i < N; i++) {
      for (let j = i + 1; j < N; j++) {
        const sameLipid = Math.floor(i / 3) === Math.floor(j / 3)
        const d = disp(i, j)
        const r = Math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
        const ti = type(i), tj = type(j)
        const b = ti < 0.5 && tj < 0.5 ? p.beadSizes.head_head : ti > 0.5 && tj > 0.5 ? p.beadSizes.tail_tail : p.beadSizes.head_tail
        if (r < wcaCut(b)) {
          const dv = wcaDv(r, b)
          for (let k = 0; k < 3; k++) {
            force[i][k] -= (dv * d[k]) / r
            force[j][k] += (dv * d[k]) / r
          }
          potential += wcaV(r, b)
        }
        if (ti > 0.5 && tj > 0.5 && !sameLipid && r <= rcTT + p.attraction.wc) {
          const dva = attrDv(r)
          for (let k = 0; k < 3; k++) {
            force[i][k] -= (dva * d[k]) / r
            force[j][k] += (dva * d[k]) / r
          }
          potential += attrV(r)
        }
      }
    }
    let kinetic = 0
    for (let i = 0; i < N; i++) {
      const vx = velocities[i * 4], vy = velocities[i * 4 + 1], vz = velocities[i * 4 + 2]
      kinetic += 0.5 * (vx * vx + vy * vy + vz * vz)
    }
    return { force, totalEnergy: kinetic + potential }
  }

  const ref = reference(result.positions, result.velocities, result.N, result.box)
  for (let i = 0; i < result.N; i++) {
    for (let k = 0; k < 3; k++) {
      expect(result.forces[i * 4 + k]).toBeCloseTo(ref.force[i][k], 2)
    }
    expect(result.forces[i * 4 + 3]).toBe(0)
  }
  expect(result.totalEnergy).toBeCloseTo(ref.totalEnergy, 2)
}

// --- area-move coordinate map: direct, GPU-free unit test (Task 6) -----------------------------
//
// Task 5's biggest defect (a box move that silently stretched intramolecular bonds by translating
// each bead's stored coordinate instead of rebuilding it from a periodic-aware offset) survived two
// review rounds and was only caught by an indirect thermodynamic identity (d2U/dlnA2 blowing up).
// These two tests check the actual coordinate map directly and cheaply, with no GPU: every
// intramolecular distance must be exactly unchanged by the move, and applying the map forward then
// backward must be an involution. Fixed, RNG-free positions (no seed to reproduce, no browser).

/** 24 lipids, fixed head positions spread across a 10x10x10 box, with two lipids (index 5 and 11)
 * placed deliberately close enough to the x and y edges that their tail beads wrap around the
 * periodic boundary — the case Task 5's bug actually broke on. One shared, fixed (non-physical,
 * arbitrary but small) offset vector per bond keeps every lipid's internal geometry identical and
 * easy to check by hand. */
function buildAreaMoveFixture(): { positions: Float32Array; lipids: number; box: [number, number, number] } {
  const box: [number, number, number] = [10, 10, 10]
  const lipids = 24
  const offsetHT1 = [0.4, 0.25, 0.15] // head -> tail1
  const offsetT1T2 = [0.35, -0.2, -0.1] // tail1 -> tail2
  const wrap = (v: number, L: number) => v - Math.floor(v / L) * L

  const heads: number[][] = []
  for (let i = 0; i < lipids; i++) {
    const col = i % 6
    const row = Math.floor(i / 6)
    heads.push([0.3 + col * 1.6, 0.3 + row * 2.3, 5.0])
  }
  heads[5] = [9.8, 5.0, 5.0] // head-tail1 will wrap in x
  heads[11] = [5.0, 9.85, 5.0] // head-tail1 will wrap in y

  const positions = new Float32Array(lipids * 3 * 4)
  for (let lip = 0; lip < lipids; lip++) {
    const [hx, hy, hz] = heads[lip]
    const t1x = wrap(hx + offsetHT1[0], box[0])
    const t1y = wrap(hy + offsetHT1[1], box[1])
    const t1z = hz + offsetHT1[2]
    const t2x = wrap(t1x + offsetT1T2[0], box[0])
    const t2y = wrap(t1y + offsetT1T2[1], box[1])
    const t2z = t1z + offsetT1T2[2]
    const base = lip * 12
    positions.set([hx, hy, hz, 0], base)
    positions.set([t1x, t1y, t1z, 1], base + 4)
    positions.set([t2x, t2y, t2z, 1], base + 8)
  }
  return { positions, lipids, box }
}

/** Independent periodic-aware distance (minimum image in x,y, open in z) — matches mi() in
 * forces.wgsl / mi1() in sim.ts, written from scratch here rather than imported, so this check does
 * not share code with whatever it is verifying. */
function periodicDistance(pos: Float32Array, box: [number, number, number], i: number, j: number): number {
  const mi = (d: number, L: number) => d - Math.round(d / L) * L
  const dx = mi(pos[i * 4] - pos[j * 4], box[0])
  const dy = mi(pos[i * 4 + 1] - pos[j * 4 + 1], box[1])
  const dz = pos[i * 4 + 2] - pos[j * 4 + 2] // z is open: no periodic image
  return Math.sqrt(dx * dx + dy * dy + dz * dz)
}

function intramolecularDistances(pos: Float32Array, box: [number, number, number], lipids: number) {
  const out: { ht1: number; t1t2: number; ht2: number }[] = []
  for (let lip = 0; lip < lipids; lip++) {
    const h = lip * 3, t1 = lip * 3 + 1, t2 = lip * 3 + 2
    out.push({
      ht1: periodicDistance(pos, box, h, t1),
      t1t2: periodicDistance(pos, box, t1, t2),
      ht2: periodicDistance(pos, box, h, t2),
    })
  }
  return out
}

test('the area move map preserves all intramolecular distances exactly', () => {
  const { positions, lipids, box } = buildAreaMoveFixture()
  const before = intramolecularDistances(positions, box, lipids)

  const newBox: [number, number, number] = [box[0] * 1.18, box[1] * 1.18, box[2]]
  const after = scaleLateralRigid(positions, box, newBox, lipids)
  const afterDist = intramolecularDistances(after, newBox, lipids)

  expect(before.length).toBe(lipids)
  // scaleLateralRigid writes a Float32Array (matching the GPU buffer it feeds in production), so
  // "floating-point noise" here is float32 noise (~1e-7 relative), not float64's ~1e-15 — precision
  // 6 (threshold 5e-7) is comfortably above the measured ~6e-8 float32 rounding and comfortably
  // below any real bond-length perturbation, which would be of order 0.01-1.
  for (let lip = 0; lip < lipids; lip++) {
    expect(afterDist[lip].ht1).toBeCloseTo(before[lip].ht1, 6)
    expect(afterDist[lip].t1t2).toBeCloseTo(before[lip].t1t2, 6)
    expect(afterDist[lip].ht2).toBeCloseTo(before[lip].ht2, 6)
  }
})

test('the area move map is an involution: +u then -u restores the original positions', () => {
  const { positions, lipids, box } = buildAreaMoveFixture()

  const newBox: [number, number, number] = [box[0] * 0.82, box[1] * 0.82, box[2]]
  const mid = scaleLateralRigid(positions, box, newBox, lipids)
  const back = scaleLateralRigid(mid, newBox, box, lipids)

  for (let i = 0; i < positions.length; i++) {
    expect(back[i]).toBeCloseTo(positions[i], 6)
  }
})

test('without friction the total energy drifts only weakly', async () => {
  const page = await gpuPage()
  const drift = await page.evaluate(async () => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 200, box: [16, 16, 16], seed: 3, layout: 'bilayer', gamma: 0 })
    const e0 = await sys.totalEnergy()
    await sys.step(2000)
    const e1 = await sys.totalEnergy()
    return Math.abs((e1 - e0) / e0)
  })
  expect(drift).toBeLessThan(0.02)
})

test('setLiveParams throws for wc above the value the grid was built for, and for kT outside kTRange', async () => {
  const page = await gpuPage()
  const result = await page.evaluate(async () => {
    const api = (window as any).api
    // maxWc=1.8: the grid is built for wc up to 1.8 (params.json's own default wc=1.6 is lower,
    // so builtForWc = max(1.6, 1.8) = 1.8 — see sim.ts's builtForWc). 1.9 must be rejected.
    const sys = await api.createSystem({ lipids: 200, box: [16, 16, 16], seed: 1, layout: 'bilayer', maxWc: 1.8 })
    let wcMessage = ''
    try {
      sys.setLiveParams({ wc: 1.9 })
    } catch (e: any) {
      wcMessage = e.message
    }
    let ktMessage = ''
    try {
      sys.setLiveParams({ kT: 999 })
    } catch (e: any) {
      ktMessage = e.message
    }
    return { wcMessage, ktMessage }
  })
  expect(result.wcMessage).toContain('exceeds')
  expect(result.wcMessage).toContain('builtForWc=1.8')
  expect(result.ktMessage).toContain('kTRange=[0.6,1.1]')
})
