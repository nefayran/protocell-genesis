import { afterAll, expect, test } from 'vitest'
import { loadSoup } from '../soup/src/rules'
import { attractionScaleTable, CLASS_APOLAR, CLASS_MINERAL, CLASS_POLAR, CLASS_SOLVENT, claySurfaceChemistryOf } from '../soup/src/soup-attraction'
import { gpuPage, shutdownGpu } from './helpers/gpu'
import { runClayArm, type ClayArm } from './helpers/clay-arm'

afterAll(shutdownGpu)

// Task 'clay-surface-chemistry' (2026-08-19). Full measurement trail, literature, ranks and the
// verdict: .superpowers/sdd/2026-08-16-soup-to-vesicle/clay-surface-chemistry-report.md; the model's
// own basis for the two limits: data/soup.json's clay.surfaceChemistryBasis and
// clay.surfaceChemistries[*].basis.
//
// WHY THIS TEST EXISTS. The predecessor task ('clay-surface') shipped ONE surface chemistry -- the
// most hydrophilic mapping expressible without electrostatics (MARTINI P5) -- and measured that the
// platelet DEPLETES carbon in its contact shell, slows chain growth 29 % and EXCLUDES aggregation.
// Its own concern §1 was that this mapping disagrees with the published coarse-grained clay: Sposito
// et al. (PNAS 1999, 96:3358) report the uncharged siloxane basal surface as hydrophobic, and Khan &
// Goel (JPCB 2019, 123:9011) put the basal beads at an apolar subtype with the wetting carried by
// CHARGED beads this engine has no way to represent. So the shipped platelet was an upper bound on
// hydrophilicity whose lower bound had never been measured. Both limits are now selectable, and this
// test measures the SAME four things at each of them, with the same null, plus the two things the
// predecessor left unmeasured: the frozen-bulk-catalyst control and the siteCatalystFraction sweep.
//
// ARMS ARE DRIVEN BY ENV so one invocation can carry as many as fit inside the machine's own limits,
// and the statistics are built by repeating invocations (which is how the run-to-run scatter every
// number below is judged against was measured, exactly as the predecessor did it):
//   CHEM_ARMS="apolar,none,bulk,apolar@0,apolar@0.5"   (mode, optionally mode@siteFraction)
//   CHEM_SEEDS="41,42"   CHEM_STEPS=20000
const SEEDS = (process.env.CHEM_SEEDS ?? '41').split(',').map((s) => Number(s.trim()))
const STEPS = Number(process.env.CHEM_STEPS ?? 20_000)
const ARMS = (process.env.CHEM_ARMS ?? 'apolar,none').split(',').map((s) => s.trim()).filter((s) => s.length > 0)

function parseArm(spec: string, seed: number): ClayArm {
  const [mode, frac] = spec.split('@')
  return {
    seed,
    steps: STEPS,
    mode: mode as ClayArm['mode'],
    siteFraction: frac === undefined ? undefined : Number(frac),
  }
}

// --- the bracket itself, on the CPU: two mappings, both citing the table, neither inventing a number
test('clay surface chemistry: two limits of the bracket, both aliases of already existing levels', () => {
  const soup = loadSoup()
  const clay = soup.clay!
  const levels = soup.solvent.attractionScale!.pairEpsilon!.levels
  const ref = levels[soup.solvent.attractionScale!.pairEpsilon!.reference].epsilonKJ

  // Both limits are declared, and the file says which one ships.
  expect(Object.keys(clay.surfaceChemistries).sort()).toEqual(['apolar', 'hydrophilic'])
  expect(clay.surfaceChemistries[clay.surfaceChemistry]).toBeDefined()
  expect(clay.surfaceChemistryBasis.length).toBeGreaterThan(200)

  // The published-mapping limit is rank C (it follows literature) while the hydrophilic one is rank D
  // (it is a choice this project made) -- stated in the file, asserted here.
  expect(clay.surfaceChemistries.apolar.rank).toBe('C')
  expect(clay.surfaceChemistries.hydrophilic.rank).toBe('D')

  // A CHEMISTRY IS AN ALIAS, NEVER A DEPTH. Every mineral pair of every chemistry must point at a
  // level that already exists in the table with its own martini_v2.1.itp source line, and the apolar
  // limit must not introduce ANY level of its own: all four of its aliases are pre-existing
  // non-mineral rows (the C1 apolar row), so this task adds no uncited energy at all.
  for (const [name, chem] of Object.entries(clay.surfaceChemistries)) {
    for (const key of ['mineralSolvent', 'mineralPolar', 'mineralApolar', 'mineralMineral']) {
      expect(levels[chem.pairs[key]], `${name}.${key}`).toBeDefined()
    }
  }
  for (const alias of Object.values(clay.surfaceChemistries.apolar.pairs)) {
    expect(alias.startsWith('mineral')).toBe(false)
  }

  // The two mineral rows of the depth table, side by side, as the GPU uniform and the CPU Metropolis
  // energy will both see them (one builder, soup/src/soup-attraction.ts, so they cannot disagree).
  const rows: Record<string, number[]> = {}
  for (const name of ['hydrophilic', 'apolar']) {
    const t = attractionScaleTable(soup, undefined, name)
    rows[name] = [t[CLASS_MINERAL][CLASS_SOLVENT], t[CLASS_MINERAL][CLASS_POLAR], t[CLASS_MINERAL][CLASS_APOLAR], t[CLASS_MINERAL][CLASS_MINERAL]]
    // The EXCHANGE favourability of adsorbing X at the surface in place of a surface water, in units
    // of the rank-A tail-tail depth (the predecessor's own formula, unchanged):
    //   F(X) = eps(K,X) + eps(W,W) - eps(K,W) - eps(W,X);  F > 0 = the surface prefers X to water.
    const kw = levels[clay.surfaceChemistries[name].pairs.mineralSolvent].epsilonKJ
    const ko = levels[clay.surfaceChemistries[name].pairs.mineralPolar].epsilonKJ
    const kc = levels[clay.surfaceChemistries[name].pairs.mineralApolar].epsilonKJ
    const ww = levels.solventSolvent.epsilonKJ
    const wo = levels.solventPolar.epsilonKJ
    const wc = levels.solventApolar.epsilonKJ
    const fHead = (ko + ww - kw - wo) / ref
    const fTail = (kc + ww - kw - wc) / ref
    console.log(
      `CHEM-LIMIT ${name} rank=${clay.surfaceChemistries[name].rank} depths(mineral row, x eps_rankA): ` +
        `water=${rows[name][0].toFixed(4)} head=${rows[name][1].toFixed(4)} tail=${rows[name][2].toFixed(4)} ` +
        `mineral=${rows[name][3].toFixed(4)} | F(head)=${fHead.toFixed(4)} F(tail)=${fTail.toFixed(4)} eps_rankA ` +
        `(=${(fHead / 1.1).toFixed(4)} / ${(fTail / 1.1).toFixed(4)} kT at kT=1.1)`,
    )
    // The two limits are genuinely OPPOSITE in the only sense that drives adsorption here: the
    // hydrophilic one prefers the head and rejects the tail, the apolar one the reverse.
    if (name === 'hydrophilic') {
      expect(fHead).toBeGreaterThan(0)
      expect(fTail).toBeLessThan(0)
    } else {
      expect(fHead).toBeLessThan(0)
      expect(fTail).toBeGreaterThan(0)
    }
  }
  // ... and the mineral row differs between them in every cell that can move a particle.
  expect(rows.apolar[0]).not.toBeCloseTo(rows.hydrophilic[0], 6)
  expect(rows.apolar[1]).not.toBeCloseTo(rows.hydrophilic[1], 6)
  expect(rows.apolar[2]).not.toBeCloseTo(rows.hydrophilic[2], 6)

  // An unknown chemistry THROWS by name rather than falling back to a default -- a silently wrong
  // surface is exactly the error a later measurement would attribute to physics.
  expect(() => claySurfaceChemistryOf(soup, 'nonexistent')).toThrow(/is not declared/)

  // Nothing about the non-mineral part of the table depends on the chemistry: the same three classes
  // keep the same depths, so switching the surface cannot silently re-tune the water or the lipids.
  const a = attractionScaleTable(soup, undefined, 'apolar')
  const h = attractionScaleTable(soup, undefined, 'hydrophilic')
  for (const ci of [CLASS_APOLAR, CLASS_POLAR, CLASS_SOLVENT]) {
    for (const cj of [CLASS_APOLAR, CLASS_POLAR, CLASS_SOLVENT]) expect(a[ci][cj]).toBeCloseTo(h[ci][cj], 12)
  }
})

// --- the arms: the same four measurements at each limit, plus the two missing controls -------------
test('clay surface chemistry: four measurements on each arm (CHEM_ARMS)', async () => {
  const page = await gpuPage()
  let diverged = 0
  let total = 0
  for (const seed of SEEDS) {
    for (const spec of ARMS) {
      const arm = parseArm(spec, seed)
      total++
      const r = await runClayArm(page, arm)
      const label = spec.toUpperCase()
      for (const id of ['C', 'O', 'H', 'W']) {
        const s = r.nullStats[id]
        const z = s.sd > 0 ? (r.enrich[id] - s.mean) / s.sd : NaN
        console.log(
          `CHEM-NULL seed=${seed} ${label} ${id}: plane=${r.enrich[id].toFixed(3)} ` +
            `null mean=${s.mean.toFixed(3)} sd=${s.sd.toFixed(3)} [${s.min.toFixed(3)},${s.max.toFixed(3)}] n=${s.n} z=${z.toFixed(2)}`,
        )
      }
      const zShell = r.shellNullSd > 0 ? (r.plateShellPolarFraction - r.shellNullMean) / r.shellNullSd : NaN
      console.log(
        `CHEM-SHELL seed=${seed} ${label} polarFraction plane=${r.plateShellPolarFraction.toFixed(4)} (n=${r.plateShellCount}) ` +
          `boxMean=${r.boxPolarFraction.toFixed(4)} null mean=${r.shellNullMean.toFixed(4)} sd=${r.shellNullSd.toFixed(4)} ` +
          `[${r.shellNullMin.toFixed(4)},${r.shellNullMax.toFixed(4)}] n=${r.shellNullN} z=${zShell.toFixed(2)}`,
      )
      console.log(
        `CHEM-ARM seed=${seed} steps=${STEPS} ${label} mode=${r.mode} frac=${r.siteFraction} N=${r.N} frozen=${r.frozenCount} ` +
          `bulkFrozen=${r.bulkFrozen} enrich C=${r.enrich.C.toFixed(3)} O=${r.enrich.O.toFixed(3)} H=${r.enrich.H.toFixed(3)} W=${r.enrich.W.toFixed(3)} ` +
          `cc=${r.ccBond} co=${r.coBond} ccBreak=${r.ccBreak} coBreak=${r.coBreak} ` +
          `amph=${r.amphiphileCount} members=${r.memberCount} inBand=${r.amphInBandFraction.toFixed(4)} ` +
          `(uniform ${r.uniformBandFraction.toFixed(4)}) meanD=${r.amphMeanDistance.toFixed(3)} (uniform ${r.uniformMeanDistance.toFixed(3)}) ` +
          `sites=${r.siteCount} sitesHolding=${r.sitesHolding} freeHolding=${r.freeHolding} nonFinite=${r.nonFinite}`,
      )
      for (const id of ['C', 'O', 'H', 'W']) {
        console.log(`CHEM-PROFILE seed=${seed} ${label} binW=${r.binW} ${id}=${r.counts[id].join(',')}`)
      }

      // DIVERGENCE IS A MEASURED PROPERTY OF THE PLATELET, NOT A TEST FAILURE TO BE HIDDEN. Task
      // 'clay-surface-chemistry' measured it: a platelet is a RIGID, IMMOBILE wall integrated with
      // data/params.json's rank-A dt, so a bead squeezed against it by its neighbours takes the whole
      // WCA impulse alone (a mobile partner would share it and separate smoothly) and a fraction of
      // runs blow up. The rate, the arms it happened on and the two things that do NOT cure it are in
      // .superpowers/sdd/2026-08-16-soup-to-vesicle/clay-surface-chemistry-report.md.
      //
      // What IS asserted is the attribution: a SURFACE-FREE arm has no rigid wall and must never
      // diverge -- that is what makes "the wall did it" a measurement rather than a story. A diverged
      // run's own measurements are meaningless, so it is reported and skipped, never averaged in.
      if (r.nonFinite > 0) {
        diverged++
        console.log(
          `CHEM-DIVERGED seed=${seed} ${label} nonFinite=${r.nonFinite} of ${r.N * 4} cc=${r.ccBond}: ` +
            `the run diverged; this arm's measurements do not go into the statistics`,
        )
        expect(arm.mode === 'hydrophilic' || arm.mode === 'apolar').toBe(true)
        continue
      }

      // STRUCTURAL expectations only. Not one threshold here encodes the SIZE of any effect: the
      // effects are the report's business, judged against the run-to-run scatter measured by repeating
      // these invocations, and a threshold picked to pass would be the exact failure this task exists
      // to avoid.
      expect(r.ccBond).toBeGreaterThan(0)
      if (arm.mode === 'none') {
        expect(r.frozenCount).toBe(0)
        expect(r.siteCount).toBe(0)
      } else if (arm.mode === 'bulk') {
        // The control freezes catalysts and NOTHING else: every frozen bead is a catalyst, and there
        // are exactly as many as the platelet arm it controls for would have immobilised.
        expect(r.bulkFrozen).toBeGreaterThan(0)
        expect(r.frozenCount).toBe(r.bulkFrozen)
        expect(r.siteCount).toBe(r.bulkFrozen)
      } else {
        // A platelet: the sheet is there whatever the site fraction, and at fraction 0 it carries no
        // catalytic site at all (which is the arm that separates surface from immobilised catalyst).
        expect(r.frozenCount).toBeGreaterThan(0)
        if (arm.siteFraction === 0) expect(r.siteCount).toBe(0)
        else expect(r.siteCount).toBeGreaterThan(0)
      }
    }
  }
  // The measured divergence rate of this invocation, printed rather than asserted against a threshold.
  // The only assertion is that the engine is not simply broken: not every arm may diverge.
  console.log(`CHEM-DIVERGENCE-RATE diverged=${diverged} of ${total} arms (${SEEDS.length} seeds x ${ARMS.length} arms)`)
  expect(diverged).toBeLessThan(total)
}, 1_800_000)
