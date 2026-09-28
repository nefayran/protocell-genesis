// Task 9: turns measured numbers into literature-referenced gate verdicts. Reads the corridor
// definitions from data/literature.json (never hardcodes a target here) and applies three rules,
// in this order of precedence:
//   1. rank D is ALWAYS unproven, whatever the metric's value. Two gates currently carry it, for
//      two different reasons:
//      - chain-to-bead-mapping (spec section 7) has no independent literature corridor to check
//        against at all; it is accepted only indirectly, through the area/thickness gates it is
//        folded into.
//      - closure: its corridor (>1 sigma^3) is real, but the only enclosedVolume this engine has
//        ever produced is the cavity of a hand-built SYNTHETIC test shell (verify/run.ts's
//        runClosureScenario) -- proof the flood-fill detector works, not proof the engine
//        self-assembles a closed vesicle. Letting that value pass a rank-A gate would publish the
//        detector's own test fixture as a measured membrane property; rank D keeps the distinction
//        in the verdict column itself -- see verify/report.ts's closure note for the full account.
//   2. a metric that is missing (undefined) or not a finite number (NaN -- fitBendingModulus
//      returns NaN when selectFitWindow never finds a valid fit window, see spectrum.ts) is
//      unproven with value:null -- never silently passed, never silently dropped from the report.
//   3. otherwise, compare the measured value against the corridor (target.min/target.max, either
//      bound optional) and report passed/failed.
import literature from '../data/literature.json'

export interface GateTarget {
  min?: number
  max?: number
}

export type GateRank = 'A' | 'B' | 'C' | 'D'
export type GateVerdict = 'passed' | 'failed' | 'unproven'
/** Where a measured value fell relative to its own window, independently of `verdict`: `outside`
 *  on a rank-D gate means the number missed its stated window even though rank D forbids calling
 *  that a failure. `none` = no window is stated, or nothing was measured. */
export type GateCorridor = 'inside' | 'outside' | 'none'

export interface GateResult {
  id: string
  title: string
  value: number | null
  target: GateTarget
  unit: string
  rank: GateRank
  verdict: GateVerdict
  /** Where the measured value fell relative to its own window -- see GateCorridor. Kept separate
   *  from `verdict` so the rank-D convention never hides an out-of-window number. */
  corridor: GateCorridor
  source: string
  /** Not in the brief's minimal Interfaces list, but the report table needs a "conditions" column and
   * data/literature.json already carries one per gate -- attaching it here is a superset, not a
   * departure from the contract (every field the four tests check is present and typed as given). */
  conditions: string
  /** WHERE this row's number came from, and WHEN it was measured. Task 'consolidation'
   * (2026-08-20): the published set now spans gates measured in three different places -- the
   * membrane-engine scenarios verify/run.ts runs itself, the explicit-water patch measured by
   * tests/water-bilayer-area-move.test.ts, and the box-54 campaign re-analysed off-GPU from its own
   * checkpoints -- and a reader cannot tell those apart from the value alone. So every gate carries
   * the artifact its metric was read from and that artifact's own timestamp. `undefined` means the
   * metric was measured by this very process, in this run. */
  provenance?: string
  /** Why an `unproven` verdict is unproven, in one sentence, and which report the number lives in
   * when there is one. Present exactly when the pipeline knows more than "the metric is absent" --
   * i.e. an input artifact was missing, so nothing fresh could be measured and NOTHING STALE WAS
   * CARRIED FORWARD in its place. */
  note?: string
}

/** Extra, per-gate context the pipeline knows and data/literature.json cannot: see GateResult's own
 * `provenance`/`note` doc comments. Both maps are keyed by gate id and are entirely optional -- an
 * `evaluateGates(metrics)` call with no second argument behaves exactly as it did before. */
export interface GateContext {
  provenance?: Record<string, string>
  notes?: Record<string, string>
}

interface LiteratureGate {
  id: string
  title: string
  metric: string
  unit: string
  target: GateTarget
  rank: GateRank
  source: string
  conditions: string
}

function loadLiteratureGates(): LiteratureGate[] {
  return (literature as { gates: LiteratureGate[] }).gates
}

export function evaluateGates(metrics: Record<string, number>, context: GateContext = {}): GateResult[] {
  return loadLiteratureGates().map((g) => {
    const raw = metrics[g.metric]
    const value = raw === undefined || Number.isNaN(raw) ? null : raw

    // Rank D stays `unproven` whatever the number says (rule 1 above) -- but a rank-D gate whose
    // measured value lies OUTSIDE its own stated window is a different thing from one that simply
    // cannot be judged, and publishing both as a bare "unproven" understates the first. The verdict
    // field keeps the convention; `corridor` records where the number actually fell, so a reader of
    // the report cannot mistake "we cannot prove this" for "this is fine". Measured case that forced
    // this: mean-tail-length 3.490 beads against its own 2-3 window (final campaign, O:C = 0.333).
    const inWindow =
      value !== null &&
      (g.target.min === undefined || value >= g.target.min) &&
      (g.target.max === undefined || value <= g.target.max)
    const hasWindow = g.target.min !== undefined || g.target.max !== undefined
    const corridor: GateCorridor = value === null || !hasWindow ? 'none' : inWindow ? 'inside' : 'outside'

    let verdict: GateVerdict
    if (g.rank === 'D') {
      verdict = 'unproven'
    } else if (value === null) {
      verdict = 'unproven'
    } else {
      const aboveMin = g.target.min === undefined || value >= g.target.min
      const belowMax = g.target.max === undefined || value <= g.target.max
      verdict = aboveMin && belowMax ? 'passed' : 'failed'
    }

    return {
      id: g.id,
      title: g.title,
      value,
      target: g.target,
      unit: g.unit,
      rank: g.rank,
      verdict,
      corridor,
      source: g.source,
      conditions: g.conditions,
      provenance: context.provenance?.[g.id],
      note: context.notes?.[g.id],
    }
  })
}
