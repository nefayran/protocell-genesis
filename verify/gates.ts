// Task 9: turns measured numbers into literature-referenced gate verdicts. Reads the corridor
// definitions from data/literature.json (never hardcodes a target here) and applies three rules,
// in this order of precedence:
//   1. rank D is ALWAYS unproven, whatever the metric's value -- our own chain-length-to-bead-count
//      mapping (spec section 7) has no independent literature corridor to check against, so no
//      value can ever make it "pass"; it is accepted only indirectly, through the area/thickness
//      gates it is folded into.
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

export interface GateResult {
  id: string
  title: string
  value: number | null
  target: GateTarget
  unit: string
  rank: GateRank
  verdict: GateVerdict
  source: string
  /** Not in the brief's minimal Interfaces list, but the report table needs a "условия" column and
   * data/literature.json already carries one per gate -- attaching it here is a superset, not a
   * departure from the contract (every field the four tests check is present and typed as given). */
  conditions: string
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

export function evaluateGates(metrics: Record<string, number>): GateResult[] {
  return loadLiteratureGates().map((g) => {
    const raw = metrics[g.metric]
    const value = raw === undefined || Number.isNaN(raw) ? null : raw

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
      source: g.source,
      conditions: g.conditions,
    }
  })
}
