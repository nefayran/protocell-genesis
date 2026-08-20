// Task 'supply-window' (2026-08-20): the SUPPLY BAND in which a closed vesicle is the preferred
// object in a FULLY PERIODIC box, computed from this project's own measured geometry.
//
// The hypothesis under test (stated by the customer, checked here rather than accepted): in a periodic
// box an aggregate that SPANS the box has no edge at all, so it beats a closed vesicle, which must pay
// curvature. A closed vesicle can therefore only be the preferred object when the amphiphile supply is
//   ABOVE the closure floor  (enough material to enclose stageThresholds.enclosedVolume)  AND
//   BELOW the cost of the CHEAPEST box-spanning, edge-free object  (else spanning is affordable).
// The floor does not depend on the box; every spanning cost is LINEAR in the box side L. So the band's
// existence is itself arithmetic in L, and that is the general answer to "would a bigger box help".
//
// Nothing here runs the GPU: it is the same closed-form arithmetic every campaign report has used
// (continuous-run-report.md §1.1 for the floor), fed by THIS task's own re-measured gates, so the
// window is recomputed and re-asserted whenever those gates move. No threshold is widened and no
// rank-A constant is touched: stageThresholds.enclosedVolume is read from data/soup.json as-is.
import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'

/** Bilayer geometry measured by a gate, in sigma / sigma^2. */
type Basis = { name: string; a: number; t: number; provenance: string }

const soup = JSON.parse(readFileSync('data/soup.json', 'utf8'))
const V_ENCLOSED: number = soup.stageThresholds.enclosedVolume // 370.8656, untouched

const wb = JSON.parse(readFileSync('verify/out/water-bilayer-area-move.json', 'utf8'))

const BASES: Basis[] = [
  {
    name: 'explicit-water gate',
    a: wb.areaPerLipid,
    t: wb.thickness,
    provenance: 'verify/out/water-bilayer-area-move.json (tests/water-bilayer-area-move.test.ts)',
  },
  {
    // Printed by tests/gate6-bilayer.test.ts in this task's own run; that gate writes no artifact.
    // GATE6 area 1.2060 +/- 0.0110 (min 1.1823, max 1.2303)  thickness 4.4580 (per-frame mean 4.4704)
    name: 'solvent-free gate 6',
    a: 1.206,
    t: 4.458,
    provenance: 'tests/gate6-bilayer.test.ts stdout, this task (GATE6 line quoted in the report)',
  },
]

/** Amphiphiles needed to CLOSE a vesicle whose lumen is exactly the closure threshold volume.
 * R_in from the volume, mid-surface one leaflet out, two leaflets of area-per-lipid a. */
function closureFloor(b: Basis): number {
  const rIn = Math.cbrt((3 * V_ENCLOSED) / (4 * Math.PI))
  const rMid = rIn + b.t / 2
  return (2 * 4 * Math.PI * rMid * rMid) / b.a
}

/** Volume occupied by one amphiphile, from the same two numbers: a leaflet of thickness t/2. */
const volPerAmphiphile = (b: Basis) => (b.a * b.t) / 2

/** (1) The cheapest edge-free spanning object: a CYLINDRICAL MICELLE spanning L, one head surface,
 * tail core radius R (which packing caps at the leaflet thickness t/2 -- a tail cannot reach
 * further than it reaches inside a bilayer). Costed by VOLUME, so the splayed area-per-head of a
 * curved surface never has to be guessed: N = pi R^2 L / v. */
const spanningMicelle = (b: Basis, L: number, R = b.t / 2) => (Math.PI * R * R * L) / volPerAmphiphile(b)

/** (2) A spanning BILAYER TUBE of mid-surface radius R: two leaflets, area 2 * 2 pi R L. */
const spanningTube = (b: Basis, L: number, R: number) => (2 * (2 * Math.PI * R * L)) / b.a

/** (3) A spanning flat BILAYER SHEET: two leaflets of area L^2 (and it wraps 2 axes, not 3). */
const spanningSheet = (b: Basis, L: number) => (2 * L * L) / b.a

// The rod this project actually measures (coalescence-report.md §4.1): 111-155 amphiphiles,
// cross-sectional gyration radius R_perp 4.006-4.165 sigma, long-axis sqrt(lambda3) 6.188-6.739.
// A uniform cylinder of length l has lambda3 = l^2/12, so l = sqrt(12 * lambda3); inverting the
// MEASURED volume N*v = pi R^2 l gives the physical radius, which is the honest way to read a
// gyration number. The check that matters: that radius should agree with the leaflet thickness t/2.
const ROD = { n: 151, sqrtL3: [6.188, 6.739] as const }
function rodRadius(b: Basis, sqrtL3: number): number {
  const len = Math.sqrt(12) * sqrtL3
  return Math.sqrt((ROD.n * volPerAmphiphile(b)) / (Math.PI * len))
}

const L_CAMPAIGN = 54
const PARTICLE_CEILING = 429496 // N * listCapacity(2500) * 4 bytes vs 4294967292
const RHO_W = 0.8 // liquid water, measured (broth-composition-report.md)

describe('supply window: is a closed vesicle EVER the preferred object in a periodic box', () => {
  it('prints the window arithmetic and the box-size scaling, and says whether the band exists', () => {
    const lines: string[] = []
    const perSigma: number[] = []
    const floors: number[] = []
    for (const b of BASES) {
      const v = volPerAmphiphile(b)
      const floor = closureFloor(b)
      floors.push(floor)
      const rods = ROD.sqrtL3.map((s) => rodRadius(b, s))
      const cheap = spanningMicelle(b, L_CAMPAIGN)
      const coeff = cheap / L_CAMPAIGN
      perSigma.push(coeff)
      lines.push(
        `WINDOW basis=${b.name} a=${b.a.toFixed(4)} t=${b.t.toFixed(4)} v=${v.toFixed(4)} ` +
          `floor=${floor.toFixed(1)} rodR(measured)=${rods.map((r) => r.toFixed(3)).join('..')} ` +
          `leafletT/2=${(b.t / 2).toFixed(3)}`,
      )
      lines.push(
        `WINDOW-SPAN basis=${b.name} L=${L_CAMPAIGN} micelle(R=t/2)=${cheap.toFixed(1)} ` +
          `perSigma=${coeff.toFixed(3)} tube(R=4)=${spanningTube(b, L_CAMPAIGN, 4).toFixed(1)} ` +
          `tube(R=t)=${spanningTube(b, L_CAMPAIGN, b.t).toFixed(1)} ` +
          `tube(R=R_mid_vesicle)=${spanningTube(b, L_CAMPAIGN, Math.cbrt((3 * V_ENCLOSED) / (4 * Math.PI)) + b.t / 2).toFixed(1)} ` +
          `sheet=${spanningSheet(b, L_CAMPAIGN).toFixed(1)}`,
      )
      lines.push(
        `WINDOW-BAND basis=${b.name} band=[${floor.toFixed(0)}, ${cheap.toFixed(0)}] ` +
          `empty=${cheap <= floor} inverted_by=${(floor / cheap).toFixed(2)}x ` +
          `L*=${(floor / coeff).toFixed(1)}sigma (band exists only for L >= L*)`,
      )
    }
    const floorLo = Math.min(...floors)
    const floorHi = Math.max(...floors)
    const cLo = Math.min(...perSigma)
    const cHi = Math.max(...perSigma)
    const lStarLo = floorLo / cHi
    const lStarHi = floorHi / cLo
    const lCeiling = Math.cbrt(PARTICLE_CEILING / RHO_W)
    lines.push(
      `WINDOW-SCALING floor=[${floorLo.toFixed(0)}, ${floorHi.toFixed(0)}] (box-independent) ` +
        `cheapestSpanning=[${cLo.toFixed(3)}, ${cHi.toFixed(3)}]*L ` +
        `L*=[${lStarLo.toFixed(1)}, ${lStarHi.toFixed(1)}]sigma ` +
        `L_max(engine, water alone at rho_W=${RHO_W})=${lCeiling.toFixed(2)}sigma ` +
        `spanningAtCeiling=[${(cLo * lCeiling).toFixed(0)}, ${(cHi * lCeiling).toFixed(0)}] ` +
        `N_at_L*=${(RHO_W * lStarLo ** 3).toExponential(3)} overCeiling=${((RHO_W * lStarLo ** 3) / PARTICLE_CEILING).toFixed(1)}x`,
    )
    // The band the HYPOTHESIS itself names: its competitor is a tube of R ~ 4 sigma, not the cheapest
    // object. Reported because that is the band this task's campaign is run inside.
    const tubeLo = Math.min(...BASES.map((b) => spanningTube(b, L_CAMPAIGN, 4)))
    const tubeHi = Math.max(...BASES.map((b) => spanningTube(b, L_CAMPAIGN, 4)))
    lines.push(
      `WINDOW-HYPOTHESIS-BAND L=54 band=[${floorLo.toFixed(0)}..${floorHi.toFixed(0)}, ` +
        `${tubeLo.toFixed(0)}..${tubeHi.toFixed(0)}] midpoint=${(((floorLo + floorHi) / 2 + (tubeLo + tubeHi) / 2) / 2).toFixed(0)}`,
    )
    console.log(lines.join('\n'))

    // --- the assertions, all of them consequences of the arithmetic above -----------------------
    // The measured rod IS a cylindrical micelle: its volume-inverted radius brackets the leaflet
    // thickness. If this ever fails, the "cheapest spanning object" identification must be redone.
    for (const b of BASES) {
      const rods = ROD.sqrtL3.map((s) => rodRadius(b, s))
      expect(Math.min(...rods)).toBeLessThan(b.t / 2 + 0.4)
      expect(Math.max(...rods)).toBeGreaterThan(b.t / 2 - 0.4)
    }
    // A cylindrical micelle is the cheapest of the three spanning geometries, on both bases.
    for (const b of BASES) {
      const cheap = spanningMicelle(b, L_CAMPAIGN)
      expect(cheap).toBeLessThan(spanningTube(b, L_CAMPAIGN, b.t))
      expect(cheap).toBeLessThan(spanningSheet(b, L_CAMPAIGN))
    }
    // THE BAND IS EMPTY AT THE CAMPAIGN BOX on both bases -- the cheapest spanning object costs LESS
    // than the closure floor, so no supply at L=54 can make a vesicle preferred over spanning.
    for (const b of BASES) expect(spanningMicelle(b, L_CAMPAIGN)).toBeLessThan(closureFloor(b))
    // And the box that would open it is beyond the engine's own particle ceiling.
    expect(lStarLo).toBeGreaterThan(lCeiling)
    expect(cHi * lCeiling).toBeLessThan(floorLo)
  })
})
