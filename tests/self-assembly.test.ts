import { afterAll, expect, test } from 'vitest'
import { clusters, largestClusterFraction } from '../engine/src/aggregate'
import { loadParams, wcaCutoff } from '../engine/src/params'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

test('clustering tells apart two distant groups', () => {
  const pos: number[] = []
  for (let i = 0; i < 10; i++) pos.push(1 + i * 0.2, 1, 1, 1)
  for (let i = 0; i < 6; i++) pos.push(20 + i * 0.2, 20, 20, 1)
  const sizes = clusters(new Float32Array(pos), [40, 40, 40], 0.5).sort((a, b) => b - a)
  expect(sizes).toEqual([10, 6])
})

// The brief's core claim: a random lipid solution self-assembles into one large aggregate under the
// tail-tail attraction alone — no scaffolding, no seeded bilayer. Lipid count (1200) and the pass
// threshold (0.8) are the brief's fixed physics and are not touched here.
//
// Step count was measured up from the brief's baseline of 400_000, per the brief's own instruction
// (Step 4: if the fraction stalls, increase steps and record the measured number — do not tune the
// attraction). Measured evidence for the increase (task-6-report.md has the full trajectories):
//   - this engine's GPU compute is NOT bit-reproducible run to run even at a fixed seed — the
//     neighbor-grid compaction's atomic slot counter is a benign race (the resulting grid is always
//     correct) but it makes the ORDER beads are summed within a cell vary between executions, and
//     this ~3600-body system is chaotic enough (Lyapunov-unstable) to amplify that microscopic
//     float-rounding difference into qualitatively different late-time aggregation states;
//   - one observed execution at 400_000 steps got kinetically trapped with TWO large, roughly equal,
//     non-merging aggregates (fraction plateaued ~0.53-0.54 from step 80_000 to 400_000, flat for
//     320_000+ steps — this is not the brief's micelle stage, it is two-bicelle coexistence);
//   - independent longer runs (up to 2_000_000 steps, same seed, same everything) never reproduced
//     that stall: they converged to fraction > 0.99 by 100_000-200_000 steps and stayed there through
//     2_000_000. 1_000_000 steps (2.5x the brief's baseline, 3x the observed 320_000-step stall
//     window) is the step count taken forward here as a measured safety margin against that
//     metastable trap, not a physics change.
// Sampled every 40_000 steps (26 points including the t=0 baseline) so the aggregation PATHWAY is
// visible — including any stall — rather than asserting only the final number.
test('one large aggregate grows from a random solution', async () => {
  const p = loadParams()
  const cutoff = wcaCutoff(p.beadSizes.tail_tail) + p.attraction.wc // r_c + w_c, per the brief

  const page = await gpuPage()
  const result = await page.evaluate(async (cutoff: number) => {
    const api = (window as any).api
    const sys = await api.createSystem({ lipids: 1200, box: [28, 28, 28], seed: 13, layout: 'random' })

    async function sample() {
      const pos: Float32Array = await sys.positions()
      const box = sys.box
      const n = pos.length / 4
      const tail: number[] = []
      for (let i = 0; i < n; i++) {
        if (pos[i * 4 + 3] !== 0) tail.push(pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2], pos[i * 4 + 3])
      }
      const sizes: number[] = api.clusters(new Float32Array(tail), box, cutoff)
      const total = tail.length / 4
      const largest = sizes.length ? Math.max(...sizes) : 0
      const top = sizes
        .slice()
        .sort((a: number, b: number) => b - a)
        .slice(0, 8)
      return { frac: total > 0 ? largest / total : 0, nClusters: sizes.length, totalTailBeads: total, top }
    }

    const CHUNK = 40_000
    const CHUNKS = 25 // 25 * 40_000 = 1_000_000 steps — see the comment above this test for why
    const trajectory: Array<{ steps: number; wallMs: number } & Awaited<ReturnType<typeof sample>>> = []
    trajectory.push({ steps: 0, wallMs: 0, ...(await sample()) })
    for (let c = 1; c <= CHUNKS; c++) {
      const t0 = performance.now()
      await sys.step(CHUNK)
      const wallMs = performance.now() - t0
      trajectory.push({ steps: c * CHUNK, wallMs, ...(await sample()) })
    }
    // Cross-check: the production facade (index.ts's largestClusterFractionOf, which snapshots
    // positions itself and derives the same cutoff from params.json) must agree with the manual
    // sample() above on the final configuration — catches any divergence between the two call paths.
    const facadeFrac = await api.largestClusterFractionOf(sys)
    return { trajectory, facadeFrac }
  }, cutoff)

  const { trajectory, facadeFrac } = result
  const totalWallMs = trajectory.reduce((a, r) => a + r.wallMs, 0)
  const totalSteps = trajectory[trajectory.length - 1].steps
  console.log(`SELF-ASSEMBLY cutoff=r_c+w_c=${cutoff.toFixed(4)} sigma, lipids=1200, box=[28,28,28], seed=13`)
  for (const r of trajectory) {
    console.log(
      `  steps=${String(r.steps).padStart(6)}  frac=${r.frac.toFixed(4)}  nClusters=${String(r.nClusters).padStart(4)}  ` +
        `totalTailBeads=${r.totalTailBeads}  top8=[${r.top.join(',')}]  chunkWallMs=${r.wallMs.toFixed(0)}`,
    )
  }
  console.log(
    `SELF-ASSEMBLY total wall-clock ${(totalWallMs / 1000).toFixed(1)} s for ${totalSteps} steps ` +
      `=> ${(totalSteps / (totalWallMs / 1000)).toFixed(0)} steps/s ` +
      `(${((totalWallMs / totalSteps) * 1000).toFixed(1)} us/step)`,
  )
  console.log(`SELF-ASSEMBLY facade cross-check: largestClusterFractionOf=${facadeFrac.toFixed(4)}`)

  const final = trajectory[trajectory.length - 1]
  // The production facade (largestClusterFractionOf) must agree with the manual sample() computed
  // inside the page — same cutoff, same tail filter, same clusters() call — catching any divergence
  // between the two call paths rather than trusting the facade blindly.
  expect(facadeFrac).toBeCloseTo(final.frac, 6)
  expect(final.frac).toBeGreaterThan(0.8)
}, 900_000)
