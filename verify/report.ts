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
  passed: 'пройдено',
  failed: 'провалено',
  unproven: 'недоказано',
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
  return 'нет коридора (принимается только через другие ворота)'
}

// Rule 4's caveats, verbatim in substance: facts already established and documented in the
// engine's own source/tests (sim.ts's non-reproducibility note, index.ts's measureBilayerAveraged
// doc comment, self-assembly.test.ts's stall record, tests/sim.test.ts's thermostat-bias
// measurement, task-5/5b/6-report.md's measured plateau and rupture behaviour) -- true regardless
// of which particular numbers this run's own scenarios land on, so they are carried as fixed prose
// rather than re-derived from this run's artifact.
const CAVEATS: string[] = [
  'Движок не бит-воспроизводим от запуска к запуску: сетка соседей раскладывает бидов по ячейкам через atomicAdd, порядок суммирования сил внутри ячейки меняется, а система (~3600 тел) хаотична и усиливает эту микроскопическую разницу до качественно разных траекторий.',
  'Порог самосборки (largest-cluster fraction > 0.8) выполнен с эмпирическим запасом на 1 000 000 шагов, а не доказан как гарантия: один прогон на 400 000 шагах застрял на доле 0.53 с двумя не слившимися агрегатами.',
  '`areaPerLipid` делит площадь бокса на СОЗДАННОЕ, а не выжившее число липидов — это смещает оценку на очень длинных прогонах, откуда часть бидов уходит по z и перестаёт быть частью бислоя.',
  'Плато площади на липид (≈1.208 σ²) лежит в нижней половине литературного коридора 1.1–1.5 σ², а не в его середине.',
  'Перетянутый бислой, запущенный из состояния около 1.9 σ² на липид, иногда рвётся; после разрыва «площадь на липид» не является содержательной величиной, так как её знаменатель предполагает один целый лист мембраны, покрывающий весь бокс.',
  'Динамика фактически термостатируется к kT ≈ 1.1355 (дискретизационное смещение ~3.2% над номинальным значением, измерено в tests/sim.test.ts: Euler–Maruyama трения + жёсткая FENE-связь под velocity-Verlet), тогда как критерий Метрополиса в areaMove и префактор κ = kT/(A·exp(intercept)) в подгонке спектра используют номинальное data/params.json kT=1.1 — расхождение около 3%, не устранённое в этом прогоне.',
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
        <td>${escapeHtml(g.provenance ?? (g.note ? `НЕ ИЗМЕРЯЛОСЬ В ЭТОМ ПРОГОНЕ: ${g.note}` : 'измерено этим прогоном (verify/run.ts)'))}</td>
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
      ? `наблюдается разброс ${(Math.min(...inWindowSpreads) * 100).toFixed(1)}–${(Math.max(...inWindowSpreads) * 100).toFixed(1)}% между модами, которые по симметрии решётки обязаны совпадать —`
      : `наблюдается большой и неравномерный разброс между модами, которые по симметрии решётки обязаны совпадать —`

  const kappaSection = kappaDetail
    ? `
    <section class="kappa">
      <h2>Модуль изгиба κ — почему «недоказано»</h2>
      <p>Изгибная мода релаксирует со скоростью порядка q⁻³…q⁻⁴, поэтому подгонка нуждается именно
      в самых медленных, наименее сошедшихся оболочках q. У оболочек, которые требует окно подгонки,
      ${spreadSentence}
      прямое, не требующее эталона доказательство того, что эти оболочки ещё не сошлись.</p>
      <table class="meta-table">
        <tr><td>наклон log-log подгонки</td><td>${fmtNum(kappaDetail.slope)}</td></tr>
        <tr><td>число мод в окне подгонки</td><td>${kappaDetail.fitModes}</td></tr>
        <tr><td>число оболочек в окне подгонки</td><td>${kappaDetail.fitShells}</td></tr>
        <tr><td>q_max окна</td><td>${fmtNum(kappaDetail.qMax)}</td></tr>
        <tr><td>окно валидно</td><td>${kappaDetail.valid ? 'да' : 'нет'}</td></tr>
      </table>
      <table>
        <thead>
          <tr><th>q, σ⁻¹</th><th>вырождение</th><th>⟨|h_q|²⟩</th><th>разброс между вырожденными модами</th><th>в окне подгонки</th></tr>
        </thead>
        <tbody>
          ${kappaDetail.spectrumTable
            .map(
              (r) =>
                `<tr><td>${r.q.toFixed(4)}</td><td>${r.degeneracy}</td><td>${r.mean.toExponential(3)}</td><td>${(r.degenerateSpreadRel * 100).toFixed(1)}%</td><td>${r.inFitWindow ? 'да' : 'нет'}</td></tr>`,
            )
            .join('\n')}
        </tbody>
      </table>
    </section>`
    : ''

  const closureSection = closureDetail
    ? `
    <section class="closure-note">
      <p><strong>О числе замкнутой полости:</strong> ${escapeHtml(closureDetail.label)} — это свойство
      синтетической проверочной раскладки самого детектора заливкой, а не измеренный объём внутренней
      полости мембраны.</p>
    </section>`
    : ''

  const perfSection = `
    <section class="performance">
      <h2>Производительность</h2>
      <table class="meta-table">
        <tr><td>шагов в секунду</td><td>${performance.stepsPerSecond !== undefined ? performance.stepsPerSecond.toFixed(1) : '—'}</td></tr>
        <tr><td>бидов в системе</td><td>${performance.beads ?? '—'}</td></tr>
        <tr><td>время перестройки сетки соседей, мс (отдельный замер)</td><td>${performance.neighborBuildMs !== undefined ? performance.neighborBuildMs.toFixed(3) : '—'}</td></tr>
        ${performance.scenario ? `<tr><td>сценарий</td><td>${escapeHtml(performance.scenario)}</td></tr>` : ''}
      </table>
    </section>`

  const caveatsSection = `
    <section class="caveats">
      <h2>Оговорки</h2>
      <ul>
        ${CAVEATS.map((c) => `<li>${escapeHtml(c)}</li>`).join('\n')}
      </ul>
    </section>`

  return `<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>protocell-genesis — отчёт проверки</title>
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
<h1>protocell-genesis — отчёт проверки (Ступень C, ворота 6)</h1>
<p>Сгенерировано ${escapeHtml(generatedAt)}, коммит <code>${escapeHtml(commitSha)}</code>${
    commitDirty
      ? ' <strong>(рабочее дерево на момент генерации было НЕ чистым — эти артефакты сгенерированы ДО коммита, который их содержит; указанный коммит — последний реальный, а не тот, что упаковывает этот файл)</strong>'
      : ' (рабочее дерево было чистым — коммит выше действительно содержит код, который дал эти числа)'
  }, run <code>${escapeHtml(runId)}</code>
(тот же идентификатор проставлен на gates.json и kappa-measurement.json — расхождение означало бы, что артефакты из разных прогонов).</p>
<table>
  <thead>
    <tr><th>ворота</th><th>значение</th><th>коридор</th><th>ранг</th><th>вердикт</th><th>откуда число</th><th>источник</th><th>условия</th></tr>
  </thead>
  <tbody>
    ${rows}
  </tbody>
</table>
${kappaSection}
${closureSection}
${perfSection}
${caveatsSection}
<footer>verify/report.ts только форматирует; данные подаёт verify/run.ts из ОДНОГО прогона измерения в этом же процессе (runId выше) — verify/out/gates.json и verify/out/kappa-measurement.json пишутся ИЗ ТОГО ЖЕ измерения, а не читаются этим отчётом с диска, и не наоборот.</footer>
</body>
</html>`
}
