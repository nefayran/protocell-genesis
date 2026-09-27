// Live readout of measurements (steps/s, elapsed, ETA, stage ladder, evidence, per-aggregate panel,
// scrolling trace) plus the collapsible honesty-note/atom-badge notes -- split out of viewer/run.ts
// (2026-08-18). The cavity-count/voxels/volume/radius/centre numbers are painted by run-cavity.ts's
// own CavityPanel (that module owns the #cavity DOM subtree end to end); paintProgress() below
// still calls it at the exact same point in the sequence the original single paintProgress() did, so
// on-screen output and paint order are unchanged.
import type { AggregateShape } from '../soup/src/aggregates'
import { HEAD_PEAKS_UNAVAILABLE, type StageThresholds } from '../soup/src/stages'
import type { Soup } from '../soup/src/rules'
import { formatElapsed, MAX_TRACE_LINES, RunUI, STAGES, STAGE_LABEL } from './run-types'

/** Item 4 (2026-08 UI-fixes task): three "the page must never imply more than it delivers"
 * disclaimers -- honesty-note (step units are reduced τ, not seconds; sps/elapsed are measured, ETA
 * is a forecast), cavity-honesty (a flood-fill pocket is not a vesicle interior, wired up by
 * run-cavity.ts using this same helper), atom-badge (beads are not atoms; the shown atomic detail is
 * reconstructed, not independently simulated) -- used to sit on screen as full paragraphs, always
 * fully expanded. Correct, and required (this project's own honesty rule), but loud: a long-running
 * page kept three multi-sentence blocks permanently open. Collapses each into one short summary line
 * plus a small toggle that reveals the EXACT SAME full text on demand -- nothing here is deleted, and
 * nothing moves into a tooltip (which reliably never gets read, the reason the task rejects that
 * option outright). Remembers the open/closed choice for the rest of this browser SESSION via
 * sessionStorage (not localStorage: a fresh tab should default back to collapsed, not silently
 * inherit a choice made days ago in a different session). Returns a setter for the full text, since
 * one of the three (atom-badge) recomputes its full text every sampled tick -- the summary/toggle DOM
 * nodes stay put; only the hidden full-text node's content changes, so the toggle's own open/closed
 * state is never disturbed by a repaint. */
export function initCollapsibleNote(container: HTMLElement, storageKey: string, summary: string): (full: string) => void {
  container.innerHTML =
    `<div class="note-summary"><span>${summary}</span>` +
    `<button class="note-toggle" type="button" aria-expanded="false">ⓘ details</button></div>` +
    `<div class="note-full" hidden></div>`
  const toggleBtn = container.querySelector('.note-toggle') as HTMLButtonElement
  const fullEl = container.querySelector('.note-full') as HTMLElement
  const storeKey = `run-note-expanded:${storageKey}`
  const setExpanded = (expanded: boolean): void => {
    fullEl.hidden = !expanded
    toggleBtn.setAttribute('aria-expanded', String(expanded))
    toggleBtn.textContent = expanded ? 'ⓘ collapse' : 'ⓘ details'
    sessionStorage.setItem(storeKey, expanded ? '1' : '0')
  }
  toggleBtn.addEventListener('click', () => setExpanded(fullEl.hidden))
  setExpanded(sessionStorage.getItem(storeKey) === '1')
  return (full: string) => {
    fullEl.textContent = full
  }
}

export interface ProgressReadout {
  /** `elapsedMs` is the caller's already-computed run-runtime.ts's own currentElapsedMs(rt) --
   * passed in rather than this module reaching into RunRuntime itself, so this module's only
   * dependency stays RunUI/StageThresholds/Soup, none of run-control-panel.ts's lifecycle state. */
  paintProgress(runUI: RunUI, stepCap: number, bondCount: number, thresholds: StageThresholds, elapsedMs: number): void
  appendTraceLine(text: string): void
  /** Clears the scrolling trace log -- startRun()'s own `traceLogEl.textContent = ''` reset, kept
   * as a named method now that the element itself is private to this module. */
  resetTrace(): void
  /** atom-badge's full text is recomputed every sampled tick (it names how many amphiphiles got the
   * atomistic treatment THIS tick) -- see run-sim-driver.ts's own use of this setter. */
  setAtomBadgeFull: (full: string) => void
}

/** `paintCavity` is called at the exact point the original single paintProgress() painted the
 * #cavity panel's own numbers -- injected rather than imported directly so this module never has to
 * know run-cavity.ts's own DOM/thresholds shape, only that "something else repaints when progress
 * does". */
/** Renders AggregateShape.encapsulatedWater's three possible states honestly and distinctly:
 * `undefined` = the caller supplied no water indices at all (nothing was asked), `null` = asked and
 * REFUSED because the aggregate's own periodic centre is untrustworthy, otherwise the measured count
 * against the threshold measured on the same snapshot. */
function encapsulatedText(e: AggregateShape['encapsulatedWater']): string {
  if (e === undefined) return 'not measured'
  if (e === null) return 'centre untrusted'
  return `${e.encapsulatedCount} / ${e.encapsulationThresholdCount.toFixed(1)}`
}
function closedText(e: AggregateShape['encapsulatedWater']): string {
  if (e === undefined) return 'n/a'
  if (e === null) return 'n/a (centre untrusted)'
  return e.closed ? 'YES' : 'no'
}

export function createProgressReadout(paintCavity: () => void, soup: Soup): ProgressReadout {
  const stepsValue = document.getElementById('steps-value') as HTMLElement
  const spsValue = document.getElementById('sps-value') as HTMLElement
  const elapsedValue = document.getElementById('elapsed-value') as HTMLElement
  const etaValue = document.getElementById('eta-value') as HTMLElement
  const stageValue = document.getElementById('stage-value') as HTMLElement
  const ladderEl = document.getElementById('ladder') as HTMLElement
  const evidenceEl = document.getElementById('evidence') as HTMLElement
  const traceLogEl = document.getElementById('trace-log') as HTMLElement
  const honestyNoteEl = document.getElementById('honesty-note') as HTMLElement
  const atomBadgeEl = document.getElementById('atom-badge') as HTMLElement

  // Per-aggregate panel (task-3c/per-aggregate-report) -- see paintProgress()'s own use of these
  // for what each field shows and StageEvidence.aggregateAnalysis's own doc comment (soup/src/
  // stages.ts) for where the numbers come from.
  const aggCountEl = document.getElementById('agg-count') as HTMLElement
  const aggQualifyingEl = document.getElementById('agg-qualifying') as HTMLElement
  const aggShareEl = document.getElementById('agg-share') as HTMLElement
  const aggLamellarEl = document.getElementById('agg-lamellar') as HTMLElement
  const aggVesicleEl = document.getElementById('agg-vesicle') as HTMLElement
  const aggHistogramEl = document.getElementById('aggregate-histogram') as HTMLElement
  const aggDetailEl = document.getElementById('aggregate-detail') as HTMLElement

  ladderEl.innerHTML = STAGES.map((s) => `<div class="stage" data-stage="${s}">${STAGE_LABEL[s]}</div>`).join('')

  // Task 'consolidation' (2026-08-20): the honesty note said only what the TIME axis means. Every
  // other published limitation of this model was on screen nowhere, while the page is the thing a
  // person actually looks at -- so the full text now carries the whole list, each item with the
  // number that makes it checkable, and each one traceable to the artifact it was measured in
  // (data/soup.json's own basis fields, data/literature.json, and the reports named inline).
  initCollapsibleNote(
    honestyNoteEl,
    'honesty',
    'WHAT THIS MODEL CANNOT DO: no closed vesicle formed in any run of the project; this page runs ' +
      'without charge; steps are reduced τ units, not seconds. ⓘ details: the full list with numbers.',
  )(
    `1. THE PROJECT'S RESULT, not a promise: no self-assembled CLOSED vesicle formed in any run. The ` +
      `largest campaign (76 σ box, 483,268 particles, 148,200 steps, charge on at pH 7.0) grew one ` +
      `aggregate holding 99.93 % of the amphiphiles; it wrapped the periodic box along 3 of 3 axes in 23 of ` +
      `23 wet snapshots and trapped 0 water beads against a closure threshold of about 315. In a finite ` +
      `parcel of water it stopped wrapping (0 of 3 axes) and still trapped no water. What is left is the ` +
      `molecule: in a finite region a capped micelle costs about 19 amphiphiles and a closed vesicle ` +
      `about 990 (docs/verdict.md). ` +
      `2. THIS PAGE RUNS WITHOUT CHARGE (data/soup.json electrostatics.enabled = false). The published ` +
      `campaigns switched on screened Coulomb between heads, with charge sampled by Monte Carlo at ` +
      `constant pH; that is where the project's one independent validation comes from, the salt shift of ` +
      `the apparent pKa (+0.709 against about 0.7). ` +
      `3. EVAPORATION IS SOLVENT REMOVAL, not the thermodynamics of a phase transition: the ` +
      `concentration factor this model actually reaches is 3.20× against 1400× for a literature dry-wet ` +
      `cycle (16.07 % of it on a log scale), and the carbon pool is also enriched ~691× against the richest ` +
      `literature pond (~15 mM decanoic acid, ACS Earth Space Chem. 2023, PMC9869395). ` +
      `4. RANK D (an estimate; a gate that rests on it comes out UNPROVEN) covers: the carbon-atoms-to-beads ` +
      `mapping, the reaction rates (no elementary rate constants for the early Earth exist), the liquid ` +
      `solvent density 0.8 σ⁻³ (measured on THIS interaction set, not taken from the literature), both clay ` +
      `surfaces (the hydrophilic and apolar limits are a bracket, not a mineral), epsilonScale=1.0. Pair-depth ` +
      `ratios are rank C (MARTINI 2.1, martini_v2.1.itp); the absolute depth ε and w_c are rank A (Cooke & ` +
      `Deserno 2005, arXiv:cond-mat/0509218). ` +
      `5. ESTIMATED vs MEASURED: "steps/s" and "elapsed" are measured with the browser's clock; "remaining" ` +
      `is a forecast from the current measured rate, not a guarantee. Steps are in REDUCED units (τ, σ=1), ` +
      `not real-world seconds: κ_t = ${soup.kappaT} is the model's only explicit time-scale calibration ` +
      `(data/soup.json); the soup has no real calibration to seconds.`,
  )

  const setAtomBadgeFull = initCollapsibleNote(
    atomBadgeEl,
    'atom-badge',
    'Beads are not atoms; the atomic detail shown is reconstructed, not simulated separately.',
  )

  function updateLadder(reachedIdx: number): void {
    for (const el of Array.from(ladderEl.children) as HTMLElement[]) {
      const stage = el.dataset.stage as (typeof STAGES)[number]
      const idx = STAGES.indexOf(stage)
      el.classList.toggle('reached', idx <= reachedIdx)
      el.classList.toggle('current', idx === reachedIdx)
    }
  }

  function appendTraceLine(text: string): void {
    traceLogEl.textContent += (traceLogEl.textContent ? '\n' : '') + text
    const lines = traceLogEl.textContent.split('\n')
    if (lines.length > MAX_TRACE_LINES) traceLogEl.textContent = lines.slice(lines.length - MAX_TRACE_LINES).join('\n')
    traceLogEl.scrollTop = traceLogEl.scrollHeight
  }

  function paintProgress(runUI: RunUI, stepCap: number, bondCount: number, thresholds: StageThresholds, elapsedMs: number): void {
    stepsValue.textContent = String(runUI.steps)
    spsValue.textContent = runUI.stepsPerSecond > 0 ? runUI.stepsPerSecond.toFixed(0) : '–'
    elapsedValue.textContent = formatElapsed(elapsedMs)
    const remaining = stepCap - runUI.steps
    etaValue.textContent =
      runUI.stepsPerSecond > 0 && remaining > 0 ? `~${formatElapsed((remaining / runUI.stepsPerSecond) * 1000)}` : '–'
    stageValue.textContent = STAGE_LABEL[runUI.stage]
    updateLadder(STAGES.indexOf(runUI.stage))
    const ev = runUI.evidence
    evidenceEl.innerHTML =
      `<div class="row"><span>amphiphile fraction</span><span>${ev.amphiphileFraction.toFixed(4)}</span></div>` +
      `<div class="row"><span>largest aggregate fraction</span><span>${ev.largestAggregateFraction.toFixed(4)}</span></div>` +
      `<div class="row"><span>head peaks</span><span>${ev.headPeaks === HEAD_PEAKS_UNAVAILABLE ? 'n/a' : ev.headPeaks}</span></div>` +
      `<div class="row"><span>enclosed volume</span><span>${ev.enclosedVolume.toFixed(4)}</span></div>` +
      `<div class="row"><span>bonds</span><span>${bondCount}</span></div>`

    paintCavity()

    // --- per-aggregate panel (task-3c/per-aggregate-report): WHY this tick's stage did or did not
    // fire -- every number the ladder itself reads (loadStageThresholds()'s new fields), not a
    // narrative summary of them. ev.aggregateAnalysis is the SAME object stageFromEvidence() (soup/
    // src/stages.ts) decided runUI.stage from -- this panel shows its own reasoning, not a second
    // guess at it.
    const agg = ev.aggregateAnalysis
    aggCountEl.textContent = String(agg.aggregateCount)
    aggQualifyingEl.textContent = `${agg.qualifyingAggregateCount} (≥${thresholds.minAmphiphilesPerAggregate} amph.)`
    aggShareEl.textContent = agg.amphiphileShareInQualifying.toFixed(3)
    aggLamellarEl.textContent = agg.hasLamellarAggregate ? 'yes' : 'no'
    aggVesicleEl.textContent = agg.hasVesicleAggregate ? 'yes' : 'no'
    aggHistogramEl.textContent = agg.sizeHistogram.length > 0 ? `sizes: [${agg.sizeHistogram.join(', ')}]` : 'no aggregates'
    aggDetailEl.innerHTML = agg.aggregates
      .map(
        (a, i) =>
          `<div class="aggregate-card">` +
          `<div class="row"><span>#${i + 1}, amphiphiles</span><span>${a.amphiphileCount}</span></div>` +
          `<div class="row"><span>flatness λ0/λ2</span><span>${a.flatnessRatio.toFixed(3)}</span></div>` +
          `<div class="row"><span>in-plane λ1/λ2</span><span>${a.inPlaneSymmetry.toFixed(3)}</span></div>` +
          `<div class="row"><span>head layers (radial)</span><span>${a.radialHeadShells === HEAD_PEAKS_UNAVAILABLE ? 'n/a' : a.radialHeadShells}</span></div>` +
          `<div class="row"><span>head layers (transverse)</span><span>${a.transverseHeadShells === HEAD_PEAKS_UNAVAILABLE ? 'n/a' : a.transverseHeadShells}</span></div>` +
          `<div class="row"><span>cavity, σ³</span><span>${a.cavityVolume.toFixed(3)}</span></div>` +
          // Task 'consolidation' (2026-08-20): the CLOSURE observable this project now decides a
          // vesicle by -- water particles that cannot reach the bulk (soup/src/water-closure.ts),
          // against the threshold it derives from the live bulk density on the same snapshot. The
          // panel used to show only `cavityVolume`, the VACUUM-cavity detector, which reads a
          // water-filled interior as empty exactly as it reads a true vacuum (that file's own
          // header) -- so the number on screen was not the number the verdict rests on. `null`
          // here is itself a measurement, not a gap: water-closure.ts refuses to report when the
          // aggregate's own periodic centre cannot be trusted, i.e. when its extent is comparable
          // to the box, which is precisely the percolating case (final-campaign-report.md §5).
          `<div class="row"><span>water inside / threshold</span><span>${encapsulatedText(a.encapsulatedWater)}</span></div>` +
          `<div class="row"><span>closed</span><span>${closedText(a.encapsulatedWater)}</span></div>` +
          `</div>`,
      )
      .join('')
  }

  function resetTrace(): void {
    traceLogEl.textContent = ''
  }

  return { paintProgress, appendTraceLine, resetTrace, setAtomBadgeFull }
}
