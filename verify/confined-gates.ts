// Task 'confined-parcel' (2026-08-21): the campaign gates of the CONFINED arm, published as their
// OWN row rather than overwriting the periodic one.
//
// WHY A SECOND FILE AND NOT A SECOND CALL INSIDE verify/run.ts. The published gate table
// (verify/out/gates.json) is the periodic-box measurement, and it must stay exactly that: the
// rank-A/B/C corridors in data/literature.json were established in a periodic box, and a confined
// run is a DIFFERENT EXPERIMENT, not a correction of it. Both results matter -- the periodic campaign
// found a percolating network that wraps 3 of 3 axes and encapsulates exactly zero water, which is a
// finding, and this arm asks the same question with the wrap-around removed. So the periodic table is
// regenerated untouched by `npm run verify`, and this script publishes the confined arm beside it,
// through the SAME collection and evaluation code (verify/campaign-gates.ts's
// collectCampaignGateInputs, verify/gates.ts's evaluateGates) pointed at the confined artifacts --
// never a second, hand-maintained copy of the gate logic.
//
// Run: nice -n 15 npx tsx verify/confined-gates.ts
// Reads:  verify/out/gates-campaign-trace-confined.json  (tests/continuous-run-audit.test.ts)
//         verify/out/gates-percolation-confined.json     (tests/percolation-check.test.ts)
// Writes: verify/out/gates-confined.json
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { collectCampaignGateInputs } from './campaign-gates'
import { evaluateGates } from './gates'

const TRACE = 'verify/out/gates-campaign-trace-confined.json'
const PERC = 'verify/out/gates-percolation-confined.json'
const OUT = 'verify/out/gates-confined.json'

const campaign = collectCampaignGateInputs(TRACE, PERC)
// Only the CAMPAIGN metrics are evaluated here. The bilayer/closure/throughput metrics of the
// published table come from a membrane-engine scenario verify/run.ts measures in its own process and
// which has nothing to do with confinement, so re-publishing them here would be duplicating one
// measurement under two names.
const results = evaluateGates(campaign.metrics, { provenance: campaign.provenance, notes: campaign.notes })
const campaignGateIds = new Set(['vesicle-closure-water', 'aggregate-percolation', 'vesicle-verdict', 'chain-length-asf', 'mean-tail-length', 'area-per-lipid-water', 'bilayer-thickness-water'])
const rows = results.filter((r) => campaignGateIds.has(r.id))

mkdirSync(dirname(OUT), { recursive: true })
writeFileSync(
  OUT,
  JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      arm: 'confined parcel (soft neutral spherical wall, R_wet = 36 sigma in a box of 160 sigma)',
      note:
        'ОТДЕЛЬНАЯ строка, не замена периодической. Коридоры в data/literature.json установлены в ' +
        'ПЕРИОДИЧЕСКОМ боксе; удержание -- другой опыт, а не поправка к нему. Ворота бислоя в воде ' +
        'здесь читаются из того же артефакта, что и в опубликованной таблице (они измеряются в ' +
        'периодическом патче и от удержания не зависят) -- приведены только для полноты строки.',
      trace: TRACE,
      percolation: PERC,
      gates: rows,
      detail: campaign.detail,
    },
    null,
    2,
  ),
)
for (const r of rows) {
  console.log(`GATE-CONFINED ${r.id}: value=${r.value} rank=${r.rank} verdict=${r.verdict} corridor=${r.corridor}`)
}
console.log(`GATE-CONFINED written ${OUT}`)
