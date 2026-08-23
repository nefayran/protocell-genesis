import { readFileSync, existsSync, mkdirSync, writeFileSync } from 'node:fs'
import { dirname } from 'node:path'
import { describe, expect, it } from 'vitest'

// Task 'acid-soap-pairing' (2026-08-23): GATE 4 -- THE ARITHMETIC THAT HAS BLOCKED EVERY CAMPAIGN,
// recomputed with the charge-assisted pair in play.
//
// The question is the same one tests/supply-window.test.ts asks and the same one
// confined-parcel-report.md section 11.5 answers: is a CLOSED vesicle the cheapest EDGE-FREE object,
// or is something else? Every formula below is that file's, unchanged and deliberately so -- what is
// new is the INPUT: the (area per lipid, thickness) pairs measured at each swept pair strength
// (verify/out/acid-soap-strength-sweep.json) instead of one uncharged basis. Nothing here runs the
// GPU; no threshold is widened; stageThresholds.enclosedVolume is read as-is.
//
// AND THE HONEST LIMIT OF WHAT PAIRING CAN CHANGE HERE, stated before the numbers rather than after.
// Every cost in this arithmetic is a function of exactly two measured numbers, a and t. So:
//  - Counting in PAIRS instead of amphiphiles changes NOTHING: a pair is two amphiphiles, so the
//    closure floor and every competitor's cost are both halved and every ratio is identical. That is
//    asserted below, not asserted away.
//  - Israelachvili's packing parameter p = v/(a0*lc) is NOT an independent handle in this model when
//    v is derived from the bilayer's own geometry: with v = a*t/2, a0 = a and lc = t/2, p = 1
//    identically, for every strength, by construction. A claim that pairing "moved the packing
//    parameter into the 1/2-1 band" would be unfalsifiable here, so it is not made.
//  - Pairing does NOT give a cylindrical micelle an edge. Every head of a cylindrical micelle sits on
//    its one outer surface, so acid-soap pairs can form there exactly as they do in a bilayer. What
//    pairing can change is a and t, and therefore the numbers below -- and that is the whole of what
//    this file measures.

const soup = JSON.parse(readFileSync('data/soup.json', 'utf8'))
const V_ENCLOSED: number = soup.stageThresholds.enclosedVolume

type Basis = { name: string; a: number; t: number; provenance: string; acidSoap: number | null }

function bases(): Basis[] {
  const out: Basis[] = []
  const wbPath = 'verify/out/water-bilayer-area-move.json'
  if (existsSync(wbPath)) {
    const wb = JSON.parse(readFileSync(wbPath, 'utf8'))
    out.push({
      name: 'explicit-water gate (UNCHARGED, published)',
      a: wb.areaPerLipid,
      t: wb.thickness,
      provenance: wbPath,
      acidSoap: null,
    })
  }
  const swPath = 'verify/out/acid-soap-strength-sweep.json'
  if (existsSync(swPath)) {
    const sw = JSON.parse(readFileSync(swPath, 'utf8'))
    for (const arm of sw.arms) {
      if (arm.final?.thickness === null || arm.final?.thickness === undefined) continue
      out.push({
        name: `charged patch pH ${arm.pH}, pair depth S=${arm.acidSoap}${arm.verdict === 'passed' ? '' : ' [' + arm.verdict + ']'}`,
        a: arm.areaTailMean,
        t: arm.final.thickness,
        provenance: `${swPath} (tests/soup-acid-soap-bilayer.test.ts)`,
        acidSoap: arm.acidSoap,
      })
    }
  }
  return out
}

/** Amphiphiles needed to CLOSE a vesicle whose lumen is exactly the closure threshold volume.
 * R_in from the volume, mid-surface one leaflet out, two leaflets of area-per-lipid a. Identical to
 * tests/supply-window.test.ts's closureFloor. */
function closureFloor(b: Basis): number {
  const rIn = Math.cbrt((3 * V_ENCLOSED) / (4 * Math.PI))
  const rMid = rIn + b.t / 2
  return (2 * 4 * Math.PI * rMid * rMid) / b.a
}
const volPerAmphiphile = (b: Basis) => (b.a * b.t) / 2
/** Cheapest edge-free BOX-SPANNING object: a cylindrical micelle of tail-core radius t/2, costed by
 * volume so no splayed area-per-head has to be guessed. */
const spanningMicelle = (b: Basis, L: number) => (Math.PI * (b.t / 2) ** 2 * L) / volPerAmphiphile(b)
/** Cheapest edge-free object in ANY geometry, periodic or not, and the one confined-parcel-report.md
 * identified as the real competitor: a capped cylinder of zero length, i.e. a spherical micelle of
 * radius t/2. Its cost has no lower bound tied to the region's size. */
const sphericalMicelle = (b: Basis) => (((4 * Math.PI) / 3) * (b.t / 2) ** 3) / volPerAmphiphile(b)

const L_CAMPAIGN = 54
const ARTIFACT = 'verify/out/acid-soap-arithmetic.json'

describe('кислотно-мыльная пара: пол замыкания и дешевейший бескрайний конкурент, пересчитанные', () => {
  it('печатает арифметику при каждой замеренной силе пары и даёт вердикт «стало ли замыкание дешевейшим»', () => {
    const bs = bases()
    expect(bs.length).toBeGreaterThan(0)
    const rows: any[] = []
    for (const b of bs) {
      const v = volPerAmphiphile(b)
      const floor = closureFloor(b)
      const span54 = spanningMicelle(b, L_CAMPAIGN)
      const perSigma = span54 / L_CAMPAIGN
      const sphere = sphericalMicelle(b)
      const lStar = floor / perSigma
      // p = v/(a0*lc) with v = a*t/2, a0 = a, lc = t/2 -- identically 1, printed to make the
      // circularity visible rather than to claim anything from it.
      const packing = v / (b.a * (b.t / 2))
      const row = {
        name: b.name,
        acidSoap: b.acidSoap,
        a: Number(b.a.toFixed(4)),
        t: Number(b.t.toFixed(4)),
        v: Number(v.toFixed(4)),
        closureFloor: Number(floor.toFixed(1)),
        floorInPairs: Number((floor / 2).toFixed(1)),
        spanningMicelleAtL54: Number(span54.toFixed(1)),
        spanningPerSigma: Number(perSigma.toFixed(3)),
        sphericalMicelle: Number(sphere.toFixed(1)),
        sphericalMicelleInPairs: Number((sphere / 2).toFixed(1)),
        bandVsSpanningEmpty: span54 <= floor,
        invertedVsSpanning: Number((floor / span54).toFixed(2)),
        invertedVsSphere: Number((floor / sphere).toFixed(1)),
        lStarSigma: Number(lStar.toFixed(1)),
        packingParameter: Number(packing.toFixed(6)),
        provenance: b.provenance,
        // Unrounded, for the pairs-vs-amphiphiles identity asserted below: the printed fields are
        // rounded for readability and rounding is not the thing under test.
        raw: { floor, sphere, span54 },
      }
      rows.push(row)
      console.log(
        `ACID-SOAP-ARITH ${row.name}\n` +
          `  a=${row.a} t=${row.t} v=${row.v} -> пол замыкания=${row.closureFloor} амфифилов (=${row.floorInPairs} пар)\n` +
          `  спанирующая мицелла при L=54: ${row.spanningMicelleAtL54} (${row.spanningPerSigma}/сигма) -> полоса ` +
          `${row.bandVsSpanningEmpty ? 'ПУСТА, перевёрнута в ' + row.invertedVsSpanning + 'x' : 'ОТКРЫТА'}, L*=${row.lStarSigma} сигма\n` +
          `  сферическая мицелла (шапочка нулевой длины, бескрайняя ПРИ ЛЮБОМ размере): ${row.sphericalMicelle} амфифилов ` +
          `(=${row.sphericalMicelleInPairs} пар) -> полоса перевёрнута в ${row.invertedVsSphere}x\n` +
          `  параметр упаковки p=v/(a0*lc)=${row.packingParameter} (тождественно 1 по построению, см. заголовок файла)`,
      )
    }

    // THE VERDICT, computed rather than narrated: has closure become the cheapest edge-free object at
    // ANY measured pair strength? It has iff the spherical micelle costs MORE than the closure floor.
    const closureCheapest = rows.filter((r) => r.sphericalMicelle >= r.closureFloor)
    const best = rows.reduce((x, y) => (y.invertedVsSphere < x.invertedVsSphere ? y : x))
    console.log(
      `ACID-SOAP-ARITH-VERDICT плеч=${rows.length} где замыкание дешевейшее=${closureCheapest.length} -> ` +
        `${closureCheapest.length > 0 ? 'ЗАМЫКАНИЕ СТАЛО ДЕШЕВЕЙШИМ' : 'ЗАМЫКАНИЕ ПО-ПРЕЖНЕМУ НЕ ДЕШЕВЕЙШЕЕ'}; ` +
        `наилучшее (наименее перевёрнутое) плечо: ${best.name} при ${best.invertedVsSphere}x против ` +
        `сферической мицеллы и ${best.invertedVsSpanning}x против спанирующей при L=54; ` +
        `наилучшее L*=${Math.min(...rows.map((r) => r.lStarSigma))} сигма`,
    )

    // COUNTING IN PAIRS CHANGES NOTHING -- asserted, because it is the first thing a reader will ask
    // and the honest answer is "no, and here is why": both sides of every ratio halve.
    for (const r of rows) {
      expect(r.raw.floor / r.raw.sphere).toBeCloseTo(r.raw.floor / 2 / (r.raw.sphere / 2), 9)
      expect(r.raw.floor / r.raw.span54).toBeCloseTo(r.raw.floor / 2 / (r.raw.span54 / 2), 9)
      // And the packing parameter is 1 by construction at every strength, so it is not evidence.
      expect(r.packingParameter).toBeCloseTo(1, 9)
    }

    mkdirSync(dirname(ARTIFACT), { recursive: true })
    writeFileSync(
      ARTIFACT,
      JSON.stringify(
        {
          measuredBy: 'tests/soup-acid-soap-arithmetic.test.ts',
          generatedAt: new Date().toISOString(),
          enclosedVolume: V_ENCLOSED,
          lCampaign: L_CAMPAIGN,
          rows,
          closureIsCheapestAtSomeStrength: closureCheapest.length > 0,
        },
        null,
        2,
      ),
    )
    console.log(`ACID-SOAP-ARITH WROTE ${ARTIFACT}`)
  })
})
