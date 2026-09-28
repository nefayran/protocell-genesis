// Task 9: renders the gate verdicts (verify/gates.ts) plus run metadata into one static, self-
// contained HTML page. Every number that appears here is handed in by the caller (verify/run.ts),
// read from an artifact a real run produced (verify/out/kappa-measurement.json, or the fresh
// measurements verify/run.ts itself takes) -- this module only formats, it never measures and never
// invents a number of its own.
import type { GateResult } from './gates'

interface KappaSpectrumRow {
  q: number
  degeneracy: number
  mean: number
  degenerateSpreadSd: number
  degenerateSpreadRel: number
  inFitWindow: boolean
}

interface KappaDetail {
  valid: boolean
  kappa: number | null
  kappaSd?: number
  slope: number
  fitModes: number
  fitShells: number
  qMax: number
  ceilingQMax?: number
  spectrumTable: KappaSpectrumRow[]
  wallClockMs?: number
  samples?: number
  stepsPerSample?: number
  grid?: number
  system?: { lipids: number; box: [number, number, number]; seed: number; layout: string }
  heightFieldDefinition?: string
}

interface PerformanceMeta {
  stepsPerSecond: number
  beads: number
  neighborBuildMs: number
  scenario?: string
}

interface ClosureMeta {
  volume: number
  label: string
}

const VERDICT_RU: Record<GateResult['verdict'], string> = {
  passed: 'passed',
  failed: 'failed',
  unproven: 'unproven',
}

function escapeHtml(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;')
}

function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return '—'
  return v.toFixed(digits)
}

function corridorText(target: GateResult['target'], unit: string): string {
  if (target.min !== undefined && target.max !== undefined) return `${target.min} – ${target.max} ${unit}`
  if (target.min !== undefined) return `> ${target.min} ${unit}`
  if (target.max !== undefined) return `< ${target.max} ${unit}`
  return 'no corridor (accepted only through other gates)'
}

// Rule 4's caveats, verbatim in substance: facts already established and documented in the
// engine's own source/tests (sim.ts's non-reproducibility note, index.ts's measureBilayerAveraged
// doc comment, self-assembly.test.ts's stall record, tests/sim.test.ts's thermostat-bias
// measurement, task-5/5b/6-report.md's measured plateau and rupture behaviour) -- true regardless
// of which particular numbers this run's own scenarios land on, so they are carried as fixed prose
// rather than re-derived from this run's artifact.
const CAVEATS: string[] = [
  'The engine is not bit-reproducible from run to run: the neighbour grid places beads into cells via atomicAdd, the order of force summation within a cell changes, and the system (~3600 bodies) is chaotic and amplifies this microscopic difference into qualitatively different trajectories.',
  'The self-assembly threshold (largest-cluster fraction > 0.8) is met with an empirical margin at 1 000 000 steps, not proven as a guarantee: one run at 400 000 steps got stuck at a fraction of 0.53 with two aggregates that had not merged.',
  '`areaPerLipid` divides the box area by the created number of lipids, not the surviving one; this biases the estimate on very long runs, from which some beads leave along z and stop being part of the bilayer.',
  'The area-per-lipid plateau (≈1.208 σ²) lies in the lower half of the literature corridor 1.1–1.5 σ², not in its middle.',
  'An overstretched bilayer started from a state near 1.9 σ² per lipid sometimes ruptures; after rupture, "area per lipid" is not a meaningful quantity, because its denominator assumes one intact membrane sheet covering the whole box.',
  'The dynamics is in fact thermostatted to kT ≈ 1.1355 (a discretisation bias of ~3.2% above the nominal value, measured in tests/sim.test.ts: Euler–Maruyama friction + a stiff FENE bond under velocity-Verlet), while the Metropolis criterion in areaMove and the prefactor κ = kT/(A·exp(intercept)) in the spectrum fit use the nominal data/params.json kT=1.1; a discrepancy of about 3%, not removed in this run.',
]

export function renderReport(results: GateResult[], meta: Record<string, unknown>): string {
  // meta.commit is a {sha, dirty} stamp (verify/run.ts's gitCommit()), not a bare string: a run's
  // artifacts are committed TOGETHER with the code changes that produced them, so at the moment
  // this renders, HEAD is still the PARENT of the commit that will contain this very file -- a
  // commit cannot name its own hash before it exists. `dirty` records that the working tree
  // differed from `sha` when this ran, and the sentence below says so out loud instead of quietly
  // publishing a commit hash that reads as "these numbers belong to this commit" when they do not.
  const commitMeta = (meta.commit ?? {}) as Partial<{ sha: string; dirty: boolean }>
  const commitSha = commitMeta.sha ?? 'unknown'
  const commitDirty = commitMeta.dirty === true
  const generatedAt = typeof meta.generatedAt === 'string' ? meta.generatedAt : new Date().toISOString()
  // Rendered visibly so a reader can compare it against the same field verify/run.ts stamps onto
  // gates.json and kappa-measurement.json -- a mismatch here would mean the three artifacts came
  // from different runs (the exact staleness class a review caught once already).
  const runId = typeof meta.runId === 'string' ? meta.runId : 'unknown'
  const performance = (meta.performance ?? {}) as Partial<PerformanceMeta>
  const kappaDetail = meta.kappaDetail as KappaDetail | undefined
  const closureDetail = meta.closureDetail as ClosureMeta | undefined

  const rows = results
    .map(
      (g) => `
      <tr class="${g.verdict}">
        <td>${escapeHtml(g.title)}<br><span class="id">${escapeHtml(g.id)}</span></td>
        <td>${fmtNum(g.value)}${g.value === null ? '' : ' ' + escapeHtml(g.unit)}</td>
        <td>${escapeHtml(corridorText(g.target, g.unit))}</td>
        <td>${escapeHtml(g.rank)}</td>
        <td class="verdict">${VERDICT_RU[g.verdict]}</td>
        <td>${escapeHtml(g.provenance ?? (g.note ? `Not measured in this run: ${g.note}` : 'measured by this run (verify/run.ts)'))}</td>
        <td>${escapeHtml(g.source)}</td>
        <td>${escapeHtml(g.conditions)}</td>
      </tr>`,
    )
    .join('\n')

  // Derived from kappaDetail.spectrumTable AT RENDER TIME, not hardcoded: an earlier version of
  // this sentence quoted a fixed "58-75%" figure from a different (Task 7) run, which drifted out
  // of sync with the table rendered two lines below it the moment a later run's spread numbers
  // differed (review finding). min/max here are always the exact bounds of the rows the table
  // itself marks inFitWindow, so the sentence and the table can never disagree.
  const inWindowSpreads = kappaDetail?.spectrumTable.filter((r) => r.inFitWindow).map((r) => r.degenerateSpreadRel) ?? []
  const spreadSentence =
    inWindowSpreads.length > 0
      ? `show a spread of ${(Math.min(...inWindowSpreads) * 100).toFixed(1)}–${(Math.max(...inWindowSpreads) * 100).toFixed(1)}% between modes that lattice symmetry requires to coincide:`
      : `show a large and uneven spread between modes that lattice symmetry requires to coincide:`

  const kappaSection = kappaDetail
    ? `
    <section class="kappa">
      <h2>Bending modulus κ: why "unproven"</h2>
      <p>The bending mode relaxes at a rate of order q⁻³…q⁻⁴, so the fit needs precisely
      the slowest, least converged q shells. The shells that the fit window requires
      ${spreadSentence}
      direct, reference-free evidence that these shells have not converged yet.</p>
      <table class="meta-table">
        <tr><td>slope of the log-log fit</td><td>${fmtNum(kappaDetail.slope)}</td></tr>
        <tr><td>number of modes in the fit window</td><td>${kappaDetail.fitModes}</td></tr>
        <tr><td>number of shells in the fit window</td><td>${kappaDetail.fitShells}</td></tr>
        <tr><td>q_max of the window</td><td>${fmtNum(kappaDetail.qMax)}</td></tr>
        <tr><td>window valid</td><td>${kappaDetail.valid ? 'yes' : 'no'}</td></tr>
      </table>
      <table>
        <thead>
          <tr><th>q, σ⁻¹</th><th>degeneracy</th><th>⟨|h_q|²⟩</th><th>spread between degenerate modes</th><th>in fit window</th></tr>
        </thead>
        <tbody>
          ${kappaDetail.spectrumTable
            .map(
              (r) =>
                `<tr><td>${r.q.toFixed(4)}</td><td>${r.degeneracy}</td><td>${r.mean.toExponential(3)}</td><td>${(r.degenerateSpreadRel * 100).toFixed(1)}%</td><td>${r.inFitWindow ? 'yes' : 'no'}</td></tr>`,
            )
            .join('\n')}
        </tbody>
      </table>
    </section>`
    : ''

  const closureSection = closureDetail
    ? `
    <section class="closure-note">
      <p><strong>On the closed-cavity number:</strong> ${escapeHtml(closureDetail.label)}: this is a property
      of the detector's own synthetic flood-fill test layout, not a measured volume of the membrane's
      inner cavity.</p>
    </section>`
    : ''

  const perfSection = `
    <section class="performance">
      <h2>Performance</h2>
      <table class="meta-table">
        <tr><td>steps per second</td><td>${performance.stepsPerSecond !== undefined ? performance.stepsPerSecond.toFixed(1) : '—'}</td></tr>
        <tr><td>beads in the system</td><td>${performance.beads ?? '—'}</td></tr>
        <tr><td>neighbour grid rebuild time, ms (separate measurement)</td><td>${performance.neighborBuildMs !== undefined ? performance.neighborBuildMs.toFixed(3) : '—'}</td></tr>
        ${performance.scenario ? `<tr><td>scenario</td><td>${escapeHtml(performance.scenario)}</td></tr>` : ''}
      </table>
    </section>`

  const caveatsSection = `
    <section class="caveats">
      <h2>Caveats</h2>
      <ul>
        ${CAVEATS.map((c) => `<li>${escapeHtml(c)}</li>`).join('\n')}
      </ul>
    </section>`

  return `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>protocell-genesis: verification report</title>
<style>
  body { font: 15px/1.5 -apple-system, BlinkMacSystemFont, sans-serif; max-width: 980px; margin: 2rem auto; padding: 0 1rem; color: #1a1a1a; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; vertical-align: top; }
  th { background: #f0f0f0; }
  tr.passed .verdict { color: #147a14; font-weight: 600; }
  tr.failed .verdict { color: #a3120a; font-weight: 600; }
  tr.unproven .verdict { color: #8a6d00; font-weight: 600; }
  .id { color: #888; font-size: 0.85em; }
  .meta-table td:first-child { color: #555; width: 40%; }
  section { margin: 2rem 0; }
  footer { color: #888; font-size: 0.85em; margin-top: 3rem; }
</style>
</head>
<body>
<h1>protocell-genesis: verification report (Stage C, gate 6)</h1>
<p>Generated ${escapeHtml(generatedAt)}, commit <code>${escapeHtml(commitSha)}</code>${
    commitDirty
      ? ' <strong>(the working tree was not clean at generation time: these artifacts were generated before the commit that contains them; the commit shown is the last real one, not the one that packages this file)</strong>'
      : ' (the working tree was clean: the commit above really contains the code that produced these numbers)'
  }, run <code>${escapeHtml(runId)}</code>
(the same identifier is stamped on gates.json and kappa-measurement.json; a mismatch would mean the artifacts come from different runs).</p>
<table>
  <thead>
    <tr><th>gate</th><th>value</th><th>corridor</th><th>rank</th><th>verdict</th><th>where the number comes from</th><th>source</th><th>conditions</th></tr>
  </thead>
  <tbody>
    ${rows}
  </tbody>
</table>
${kappaSection}
${closureSection}
${perfSection}
${caveatsSection}
<footer>verify/report.ts only formats; the data is supplied by verify/run.ts from one measurement run in this same process (runId above). verify/out/gates.json and verify/out/kappa-measurement.json are written from that same measurement, not read from disk by this report, and not the other way round.</footer>
</body>
</html>`
}
