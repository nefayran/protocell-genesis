// Live readout of measurements (steps/s, elapsed, ETA, stage ladder, evidence, per-aggregate panel,
// scrolling trace) plus the collapsible honesty-note/atom-badge notes -- split out of viewer/run.ts
// (2026-08-18). The cavity-count/voxels/volume/radius/centre numbers are painted by run-cavity.ts's
// own CavityPanel (that module owns the #cavity DOM subtree end to end); paintProgress() below
// still calls it at the exact same point in the sequence the original single paintProgress() did, so
// on-screen output and paint order are unchanged.
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
    `<button class="note-toggle" type="button" aria-expanded="false">ⓘ подробнее</button></div>` +
    `<div class="note-full" hidden></div>`
  const toggleBtn = container.querySelector('.note-toggle') as HTMLButtonElement
  const fullEl = container.querySelector('.note-full') as HTMLElement
  const storeKey = `run-note-expanded:${storageKey}`
  const setExpanded = (expanded: boolean): void => {
    fullEl.hidden = !expanded
    toggleBtn.setAttribute('aria-expanded', String(expanded))
    toggleBtn.textContent = expanded ? 'ⓘ свернуть' : 'ⓘ подробнее'
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

  initCollapsibleNote(
    honestyNoteEl,
    'honesty',
    'ОЦЕНКА vs ИЗМЕРЕНО: шаги — приведённые единицы (τ), не секунды; ETA — прогноз, не гарантия.',
  )(
    `ОЦЕНКА vs ИЗМЕРЕНО: «шагов/с» и «прошло (реал.)» измерены по системным часам браузера; ` +
      `«осталось (оцен.)» — прогноз из текущей измеренной скорости, не гарантия. Единицы шагов — ` +
      `ПРИВЕДЁННЫЕ (τ, σ=1), это не секунды реального мира: κ_t = ${soup.kappaT} — единственная явная ` +
      `калибровка временной шкалы модели (data/soup.json), настоящей секундной привязки для бульона нет.`,
  )

  const setAtomBadgeFull = initCollapsibleNote(
    atomBadgeEl,
    'atom-badge',
    'Бусины — не атомы; показанная атомная детализация реконструирована, не симулирована отдельно.',
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
      `<div class="row"><span>доля амфифилов</span><span>${ev.amphiphileFraction.toFixed(4)}</span></div>` +
      `<div class="row"><span>доля крупн. агрегата</span><span>${ev.largestAggregateFraction.toFixed(4)}</span></div>` +
      `<div class="row"><span>пики голов</span><span>${ev.headPeaks === HEAD_PEAKS_UNAVAILABLE ? 'н/д' : ev.headPeaks}</span></div>` +
      `<div class="row"><span>замкн. объём</span><span>${ev.enclosedVolume.toFixed(4)}</span></div>` +
      `<div class="row"><span>связей</span><span>${bondCount}</span></div>`

    paintCavity()

    // --- per-aggregate panel (task-3c/per-aggregate-report): WHY this tick's stage did or did not
    // fire -- every number the ladder itself reads (loadStageThresholds()'s new fields), not a
    // narrative summary of them. ev.aggregateAnalysis is the SAME object stageFromEvidence() (soup/
    // src/stages.ts) decided runUI.stage from -- this panel shows its own reasoning, not a second
    // guess at it.
    const agg = ev.aggregateAnalysis
    aggCountEl.textContent = String(agg.aggregateCount)
    aggQualifyingEl.textContent = `${agg.qualifyingAggregateCount} (≥${thresholds.minAmphiphilesPerAggregate} амф.)`
    aggShareEl.textContent = agg.amphiphileShareInQualifying.toFixed(3)
    aggLamellarEl.textContent = agg.hasLamellarAggregate ? 'да' : 'нет'
    aggVesicleEl.textContent = agg.hasVesicleAggregate ? 'да' : 'нет'
    aggHistogramEl.textContent = agg.sizeHistogram.length > 0 ? `размеры: [${agg.sizeHistogram.join(', ')}]` : 'нет агрегатов'
    aggDetailEl.innerHTML = agg.aggregates
      .map(
        (a, i) =>
          `<div class="aggregate-card">` +
          `<div class="row"><span>#${i + 1}, амф.</span><span>${a.amphiphileCount}</span></div>` +
          `<div class="row"><span>плоскостность λ0/λ2</span><span>${a.flatnessRatio.toFixed(3)}</span></div>` +
          `<div class="row"><span>в-плоск. λ1/λ2</span><span>${a.inPlaneSymmetry.toFixed(3)}</span></div>` +
          `<div class="row"><span>слоёв голов (радиал.)</span><span>${a.radialHeadShells === HEAD_PEAKS_UNAVAILABLE ? 'н/д' : a.radialHeadShells}</span></div>` +
          `<div class="row"><span>слоёв голов (поперечн.)</span><span>${a.transverseHeadShells === HEAD_PEAKS_UNAVAILABLE ? 'н/д' : a.transverseHeadShells}</span></div>` +
          `<div class="row"><span>полость, σ³</span><span>${a.cavityVolume.toFixed(3)}</span></div>` +
          `</div>`,
      )
      .join('')
  }

  function resetTrace(): void {
    traceLogEl.textContent = ''
  }

  return { paintProgress, appendTraceLine, resetTrace, setAtomBadgeFull }
}
