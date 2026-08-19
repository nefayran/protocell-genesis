import { computeDryBox, planSoupGrid } from '../../soup/src/sim'
import { loadSoup } from '../../soup/src/rules'

/** Task 'grid-rebuild': the smallest (side, start-count) pair whose dry-wet box change moves the
 * neighbour grid's cell count into a DIFFERENT bracket -- exactly the class of change
 * wet-dry-cycle-report.md measured failing outright at full scale (box 46, targetDryDensity=0.6:
 * wet dims=[15,15,15] -> dry dims=[14,14,14], "этот механизм переписывает только box-униформ, не
 * перестраивает буферы сетки"). Searches for the pair rather than hardcoding one, so this stays
 * correct if data/soup.json's species/neighborGrid/verletList settings (which set the ABSOLUTE
 * cellSize every candidate box is measured against) ever change -- a hardcoded box could silently
 * stop crossing a bracket and this fixture would then test nothing. Both
 * tests/soup-boxcycle.test.ts's pure-TS invariant check and tests/soup-grid-resize.test.ts's real
 * GPU resize call this ONE function so they exercise the identical box pair, not two independently
 * chosen ones that could drift apart from each other. */
export function findBracketCrossingBox(): {
  box: [number, number, number]
  start: Record<string, number>
  N: number
  dryBox: [number, number, number]
  wetDims: [number, number, number]
  dryDims: [number, number, number]
} {
  const soup = loadSoup()
  const dwc = soup.dryWetCycle
  for (let side = 12; side <= 30; side += 0.25) {
    const box: [number, number, number] = [side, side, side]
    // This project's own established working dilute density (kinetic-growth-report.md et al.) AND
    // its own default monomer ratio C:O:H:M=30:10:30:1 (data/soup.json's own `start`) -- matching
    // the ratio matters, not just the density: an earlier version of this search used 0.3/0.1/0.3
    // fractions that summed to only 0.7 (not 1, since it dropped the catalyst share entirely from
    // the other three), which silently made the REAL density ~0.32 instead of the intended ~0.45 --
    // the dry-phase compression needed to reach dwc.targetDryDensity from there was ~19% linear
    // instead of the ~7% this project's own ramp (rampSteps/rampRelaxSteps) is sized for and has
    // been measured stable at, and running that oversized jump on a real GPU system blew up
    // (assertVerletSafety's drift check caught a nonsense ~53000 sigma "drift" -- real physics
    // instability from a compression amplitude never validated at, not a grid-buffer bug).
    const wetDensity = 0.45
    const approxN = wetDensity * side * side * side
    // W:0 (task 'broth-composition', 2026-08-18): data/soup.json now carries a default water
    // count sized for a box-30 run; CreateSoupOpts.start MERGES over the file's defaults, so
    // without this explicit zero, this small (side 12-30) search box would silently inherit that
    // full water count and end up at an absurd, guaranteed-unstable density. This helper is about
    // the neighbour-grid bracket-crossing geometry, not about broth composition.
    const start = {
      C: Math.round(approxN * (30 / 71)),
      O: Math.round(approxN * (10 / 71)),
      H: Math.round(approxN * (30 / 71)),
      M: Math.max(4, Math.round(approxN * (1 / 71))),
      W: 0,
    }
    const N = start.C + start.O + start.H + start.M
    const wetPlan = planSoupGrid(box, start)
    if (!wetPlan.valid) continue
    const dryBox = computeDryBox(box, N, dwc.targetDryDensity)
    const dryPlan = planSoupGrid(dryBox, start)
    if (!dryPlan.valid) continue
    const changed = dryPlan.dims[0] !== wetPlan.dims[0] || dryPlan.dims[1] !== wetPlan.dims[1] || dryPlan.dims[2] !== wetPlan.dims[2]
    if (changed) return { box, start, N, dryBox, wetDims: wetPlan.dims, dryDims: dryPlan.dims }
  }
  throw new Error(
    'findBracketCrossingBox: не нашлось маленькой пары wet/dry box (side 12..30), меняющей число ' +
      'ячеек сетки соседей -- data/soup.json-параметры сетки (cellSize/effectiveWalkRadius) изменились?',
  )
}
