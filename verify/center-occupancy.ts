// Catalyst-centre occupancy, read straight off a checkpoint's own `centerLink` /
// `centerHeldSteps` / `desorbEvents` payload -- pure CPU, no GPU, no browser. Task
// 'tail-length-and-window' (2026-08-19),
// .superpowers/sdd/2026-08-16-soup-to-vesicle/tail-length-and-window-report.md sec 6.
//
// Exists because the box-54 window run's chemistry froze EXACTLY (cc_bond/co_bond/co_break
// identical to the digit from step 120 000 to step 300 000) while thousands of free carbon
// monomers and tens of thousands of free heads were still in the box -- so "the chemistry is
// exhausted" had to be either confirmed or refuted against the buffer that actually gates it
// (soup/wgsl/bond-adsorption.wgsl's centerLink: a catalyst holds the id of the chain tip it is
// growing, or 0xFFFFFFFF when free). It is refuted: at step 300 000 ZERO of 1021 catalysts hold
// anything, and no new chain nucleates.
//
// Usage: npx tsx verify/center-occupancy.ts <checkpoint.json> [...]
import { readFileSync } from 'node:fs'
import { decodeCheckpointResume } from '../soup/src/checkpoint'
import { loadSoup } from '../soup/src/rules'
const EMPTY = 0xffffffff
const soup = loadSoup()
for (const path of process.argv.slice(2)) {
  const r = decodeCheckpointResume(JSON.parse(readFileSync(path, 'utf8')))
  const pos = r.positions
  const N = pos.length / 4
  const kindOf = (i: number) => soup.monomers[Math.round(pos[i * 4 + 3])].kind
  let cat = 0, catLinked = 0, carbLinked = 0, freeCarbon = 0, cap = 0
  const held: number[] = []
  const deg = new Int32Array(N)
  for (let i = 0; i < N; i++) for (let s = 0; s < 3; s++) if (r.bondSlots[i * 3 + s] !== EMPTY) deg[i]++
  for (let i = 0; i < N; i++) {
    const k = kindOf(i)
    if (k === 'catalyst') { cat++; if (r.centerLink[i] !== EMPTY) { catLinked++; held.push(r.centerHeldSteps[i]) } }
    if (k === 'carbon') { if (r.centerLink[i] !== EMPTY) carbLinked++; if (deg[i] === 0) freeCarbon++ }
    if (k === 'head' && deg[i] === 0) cap++
  }
  held.sort((a, b) => a - b)
  const q = (f: number) => held.length ? held[Math.min(held.length - 1, Math.floor(f * held.length))] : -1
  console.log(`CENTERS ${path} step=${r.globalStep} catalysts=${cat} linked=${catLinked} (${(100*catLinked/cat).toFixed(1)}%) carbonsLinked=${carbLinked} freeCarbonMonomers=${freeCarbon} freeHeads=${cap} desorb=${JSON.stringify(r.desorbEvents)} heldSteps[min,med,max]=[${q(0)},${q(0.5)},${held[held.length-1]}]`)
}
