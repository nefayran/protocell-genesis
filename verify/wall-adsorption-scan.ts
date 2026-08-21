// Task 'confined-parcel' (2026-08-21): THE COMPETING-SINK MEASUREMENT, off-GPU, over a confined
// campaign's own checkpoints -- how much of each species sits in the parcel's outermost shell against
// what the SOLVENT puts there, plus the two geometric quantities the no-wrap argument rests on (the
// largest radius any particle reached, and therefore the largest pair separation, against L/2).
//
// Solvent-relative, not uniform-relative, on purpose: a soft-walled liquid's outer surface relaxes a
// little inside the nominal radius, so every species reads below the uniform-sphere null by the SAME
// geometric factor -- dividing by the solvent's own shell fraction cancels it exactly, and 1.0 then
// means "this species is at the container in the same proportion the water is", i.e. no preferential
// adsorption. That is the number the competing-sink question actually asks.
//
// Pure Node over the checkpoints' own base64 payloads: no GPU, no page, and no per-particle array ever
// crosses a process boundary. Run: nice -n 15 npx tsx verify/wall-adsorption-scan.ts
import { readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { wallStats } from '../soup/src/soup-confine'
import { loadSoup } from '../soup/src/rules'
const dir = 'data/checkpoints/cpR36'
const soup = loadSoup()
const ids = soup.monomers.map((m) => m.id)
const sk = soup.monomers.findIndex((m) => m.id === soup.solvent.waterId)
const wb = JSON.parse(readFileSync('verify/out/water-bilayer-area-move.json', 'utf8'))
const SHELL = 4.68713096487309 // the thickness that was current when the campaign was sized
const rows: any[] = []
for (const f of readdirSync(dir).filter((x) => x.endsWith('.json'))) {
  const raw = JSON.parse(readFileSync(`${dir}/${f}`, 'utf8'))
  const r = decodeCheckpointResume(raw)
  const box = r.liveBox as [number, number, number]
  const R = raw.config.confine.radiusSigma * (box[0] / raw.config.box[0])
  const ws = wallStats(r.positions, r.positions.length / 4, box, R, SHELL, ids, sk)
  rows.push({ step: raw.globalStep, box: +box[0].toFixed(4), R: +R.toFixed(4), ws })
}
rows.sort((a, b) => a.step - b.step)
console.log(`shell=${SHELL.toFixed(4)} (fresh gate thickness now ${wb.thickness.toFixed(4)})`)
console.log('step box R maxR penetration faceClear maxPair L/2 strong uniformFrac | fraction C/O/H/M/W | enrichVsWater C/O/H/M | inShell C/H')
for (const x of rows) {
  const s = (id: string) => x.ws.species.find((y: any) => y.id === id)
  const f = (id: string) => s(id).fraction.toFixed(4)
  const e = (id: string) => s(id).enrichmentVsSolvent.toFixed(4)
  console.log(
    `${x.step} ${x.box} ${x.R} ${x.ws.maxRadius.toFixed(4)} ${x.ws.penetration.toFixed(4)} ${x.ws.faceClearance.toFixed(3)} ${x.ws.maxPairSeparation.toFixed(3)} ${x.ws.halfBox.toFixed(1)} ${x.ws.strongNoWrap} ${x.ws.species[0].uniformFraction.toFixed(4)} | ${f('C')}/${f('O')}/${f('H')}/${f('M')}/${f('W')} | ${e('C')}/${e('O')}/${e('H')}/${e('M')} | ${s('C').inShell}/${s('H').inShell}`,
  )
}
writeFileSync('verify/out/confined-wall-adsorption.json', JSON.stringify(rows, null, 1))
