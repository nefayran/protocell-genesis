// ONE ARM of the clay campaign: a full broth run, then every measurement the 'clay-surface' campaign
// made, on whatever surface chemistry (or surface-free control) the arm names. Extracted VERBATIM from
// tests/soup-clay.test.ts by task 'clay-surface-chemistry' (2026-08-19) -- a clean transfer, so that
// the two limits of the hydrophilicity bracket, the frozen-bulk-catalyst control and the
// siteCatalystFraction sweep are all measured by THE SAME code with THE SAME null, which is the only
// way their numbers are comparable. tests/soup-clay.test.ts imports it back and its measurement test
// prints the identical lines it printed before.
//
// The four measurements, all in one arm because they all read the same final state:
//  1. the contact-shell concentration profile per species (0.5 sigma bins in |distance to plane|),
//     with the plane-to-plane NULL distribution measured inside the same run;
//  2. the density-independent shell COMPOSITION (polar fraction), which divides out the phase
//     structure the medium has at this composition;
//  3. cc_bond / co_bond (growth / termination) event counts;
//  4. where the recognised amphiphile aggregates sit relative to the plane, against the uniform
//     expectation.

/** What distinguishes one arm from another. Everything else -- box, kT, composition, seed handling,
 * step count, binning plane -- is identical across arms by construction. */
export interface ClayArm {
  seed: number
  steps: number
  /** 'none' = no platelet and nothing frozen (the original control);
   *  'hydrophilic' = the platelet with the P5 mapping (the 'clay-surface' campaign's surface);
   *  'apolar' = the platelet with the published uncharged-siloxane mapping (data/soup.json's
   *    clay.surfaceChemistries.apolar: Sposito PNAS 1999 96:3358 / Khan & Goel JPCB 2019 123:9011);
   *  'bulk' = NO platelet, but the same number of catalyst beads frozen where they lie in the BULK --
   *    the control that separates "a fraction of the catalysts stopped diffusing" from "the surface
   *    did it". Its count is DERIVED by asking planClay what the clay arm would have used, never typed. */
  mode: 'none' | 'hydrophilic' | 'apolar' | 'bulk'
  /** Overrides data/soup.json's clay.siteCatalystFraction for this arm (the sweep). Also sets how many
   * catalysts a 'bulk' arm freezes, so a sweep point and its control stay matched. */
  siteFraction?: number
}

/** Runs ONE arm. Every arm bins against the SAME plane (the box midplane, which is where a platelet
 * would be), so a surface-free arm is a genuine control for the binning itself and not just for the
 * physics. */
export async function runClayArm(page: any, arm: ClayArm) {
  return page.evaluate(
    async (seedArg: number, stepsArg: number, modeArg: string, fracArg: number | null) => {
      const api = (window as any).api
      const soup = api.loadSoup()
      const box: [number, number, number] = [30, 30, 30]
      const clayArg = modeArg === 'hydrophilic' || modeArg === 'apolar'
      const opts: Record<string, unknown> = { box, seed: seedArg, kT: 1.1, clay: clayArg }
      if (clayArg) opts.claySurfaceChemistry = modeArg
      if (clayArg && fracArg !== null) opts.claySiteCatalystFraction = fracArg
      // The frozen-bulk-catalyst control immobilises EXACTLY as many catalysts as the platelet arm
      // would, and that count is asked of planClay rather than written here -- so the control tracks
      // the sweep automatically and cannot drift from the arm it controls for.
      let bulkFrozen = 0
      if (modeArg === 'bulk') {
        const catalystId = soup.monomers.find((m: any) => m.kind === 'catalyst').id
        const L = api.planClay(soup, api.loadParams(), box, soup.start[catalystId], fracArg ?? undefined)
        bulkFrozen = L.siteCount
        opts.frozenBulkCatalysts = bulkFrozen
      }
      const sys = await api.createSoup(opts)
      const planes: number[] = clayArg ? sys.clayPlanes() : [box[2] / 2]
      const frozen: Uint32Array = await sys.frozen()
      await sys.step(stepsArg)
      const particles: Float32Array = await sys.particles()
      const bonds: Uint32Array = await sys.bonds()
      const events = await sys.events()
      const N = particles.length / 4
      let nonFinite = 0
      for (let k = 0; k < N * 4; k++) if (!Number.isFinite(particles[k])) nonFinite++

      // Profile: 0.5 sigma bins in |distance to plane|, FROZEN beads excluded (a profile that counts
      // the platelet's own beads would report the platelet, not what the platelet attracted).
      const binW = 0.5
      const nBins = Math.floor(box[2] / 2 / binW)
      const ids: string[] = soup.monomers.map((m: any) => m.id)
      const counts: Record<string, number[]> = {}
      for (const id of ids) counts[id] = new Array(nBins).fill(0)
      for (let i = 0; i < N; i++) {
        if (frozen[i] !== 0) continue
        const d = api.distanceToPlatelet(particles[i * 4 + 2], planes, box[2])
        const b = Math.min(nBins - 1, Math.floor(d / binW))
        counts[ids[Math.round(particles[i * 4 + 3])]][b]++
      }
      // Bin volume is the same for EVERY bin (two slabs of Lx*Ly*binW, one per face of the plane), so a
      // count ratio is already a density ratio and no volume factor can be got wrong here.
      //
      // The reference is the species' OWN box mean (total/nBins), not a "bulk" far-field window. That
      // choice is forced by a measurement, not preferred: at this composition the water sub-density is
      // 0.396 sigma^-3, BELOW the 0.8 sigma^-3 liquid threshold data/soup.json's startBasis §1 measured
      // (its §2 records that discrepancy openly), so after 20000 steps the medium is two-phase -- dense
      // regions and voids -- and there is no homogeneous far field to divide by. A far-field denominator
      // measured on such a profile reports which phase happened to land in the far bins. The box mean is
      // well defined whatever the phase structure, and the clay-free arm measures the same quantity
      // about the same plane, so it carries the scatter this number has to be judged against.
      const nearBins = Math.round(1 / binW) // within 1 sigma of the plane: the first contact shell
      // nearOverMean(plane): the contact-shell density about ANY plane, divided by this species' own box
      // mean. Written as a function of the plane because that is what turns the number into a statistic:
      // a plane in the same box that has NO surface on it gives the null value, and there are 30 of them
      // available in one run.
      function nearOverMean(id: string, planeSet: number[]): number {
        let near = 0
        let total = 0
        for (let i = 0; i < N; i++) {
          if (frozen[i] !== 0) continue
          if (ids[Math.round(particles[i * 4 + 3])] !== id) continue
          total++
          if (api.distanceToPlatelet(particles[i * 4 + 2], planeSet, box[2]) < nearBins * binW) near++
        }
        const meanPerBin = total / nBins
        return meanPerBin > 0 ? near / nearBins / meanPerBin : 0
      }
      const enrich: Record<string, number> = {}
      for (const id of ids) enrich[id] = nearOverMean(id, planes)
      // THE NULL DISTRIBUTION, measured inside the very same run rather than assumed. 30 candidate
      // planes evenly spaced across the box; in the clay arm the ones within 3 sigma of the platelet are
      // dropped (they are not surface-free). Every one of them is binned exactly like the platelet's own
      // plane, so their spread is precisely "how much does an arbitrary plane's contact shell deviate
      // from the box mean in a broth that is not spatially homogeneous" -- the scatter the platelet's own
      // number has to beat, per species, per run.
      // THE DENSITY-INDEPENDENT STATISTIC, and the reason the per-species one above is not the last word.
      // A per-species contact-shell density is confounded by the phase structure: a plane that happens to
      // fall inside a dense domain lifts EVERY species together and one that falls in a void drops every
      // species together, which is exactly the ~1.0 sd the null shows. The local COMPOSITION of the shell
      // -- what fraction of the beads in it are polar (head or water) rather than apolar (carbon or
      // donor) -- divides that common density factor out, so it measures what the surface PREFERS rather
      // than how much of the medium happens to be there.
      function shellPolarFraction(planeSet: number[]): { frac: number; n: number } {
        let polar = 0
        let n = 0
        for (let i = 0; i < N; i++) {
          if (frozen[i] !== 0) continue
          if (api.distanceToPlatelet(particles[i * 4 + 2], planeSet, box[2]) >= nearBins * binW) continue
          const m = soup.monomers[Math.round(particles[i * 4 + 3])]
          n++
          if (m.polar || m.solvent) polar++
        }
        return { frac: n > 0 ? polar / n : 0, n }
      }
      let boxPolar = 0
      let boxAll = 0
      for (let i = 0; i < N; i++) {
        if (frozen[i] !== 0) continue
        const m = soup.monomers[Math.round(particles[i * 4 + 3])]
        boxAll++
        if (m.polar || m.solvent) boxPolar++
      }

      const nullStats: Record<string, { mean: number; sd: number; min: number; max: number; n: number }> = {}
      const nPlanes = 30
      for (const id of ids) {
        const vals: number[] = []
        for (let j = 0; j < nPlanes; j++) {
          const z = (j * box[2]) / nPlanes
          if (api.distanceToPlatelet(z, planes, box[2]) < 3) continue
          vals.push(nearOverMean(id, [z]))
        }
        const mean = vals.reduce((a, b) => a + b, 0) / Math.max(1, vals.length)
        const sd = Math.sqrt(vals.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(1, vals.length - 1))
        nullStats[id] = { mean, sd, min: vals.length ? Math.min(...vals) : 0, max: vals.length ? Math.max(...vals) : 0, n: vals.length }
      }
      const plateShell = shellPolarFraction(planes)
      const shellNull: number[] = []
      for (let j = 0; j < nPlanes; j++) {
        const z = (j * box[2]) / nPlanes
        if (api.distanceToPlatelet(z, planes, box[2]) < 3) continue
        const s = shellPolarFraction([z])
        if (s.n > 0) shellNull.push(s.frac)
      }
      const shellNullMean = shellNull.reduce((a, b) => a + b, 0) / Math.max(1, shellNull.length)
      const shellNullSd = Math.sqrt(shellNull.reduce((a, b) => a + (b - shellNullMean) ** 2, 0) / Math.max(1, shellNull.length - 1))

      // Aggregate location: every particle belonging to a recognised amphiphile, and how far from the
      // plane it sits. `uniformFraction` is what a spatially uniform population would put in the same
      // band, which is the only honest baseline for "preferentially at the surface".
      // How many of the platelet's own catalytic sites are holding a chain end right now, through the
      // SAME centerLink tether a free catalyst uses -- 0 by construction in the clay-free arm.
      const links: Uint32Array = await sys.centerLinks()
      const catalystKind = soup.monomers.findIndex((m: any) => m.kind === 'catalyst')
      let sitesHolding = 0
      let siteCount = 0
      let freeHolding = 0
      for (let i = 0; i < N; i++) {
        const isCat = Math.round(particles[i * 4 + 3]) === catalystKind
        if (!isCat) continue
        if (frozen[i] !== 0) {
          siteCount++
          if (links[i] !== 0xffffffff) sitesHolding++
        } else if (links[i] !== 0xffffffff) freeHolding++
      }

      const amph = api.findAmphiphiles(particles, bonds, soup.monomers)
      const members: number[] = []
      for (const a of amph) {
        members.push(a.headIndex)
        for (const c of a.chain) members.push(c)
      }
      const band = 2.0
      let inBand = 0
      let dSum = 0
      for (const i of members) {
        const d = api.distanceToPlatelet(particles[i * 4 + 2], planes, box[2])
        dSum += d
        if (d <= band) inBand++
      }
      sys.dispose()
      return {
        N,
        frozenCount: frozen.reduce((a: number, b: number) => a + (b !== 0 ? 1 : 0), 0),
        counts,
        enrich,
        binW,
        amphiphileCount: amph.length,
        memberCount: members.length,
        amphInBandFraction: members.length > 0 ? inBand / members.length : 0,
        amphMeanDistance: members.length > 0 ? dSum / members.length : 0,
        uniformBandFraction: (2 * band) / box[2],
        uniformMeanDistance: box[2] / 4,
        ccBond: events['cc_bond'] ?? 0,
        coBond: events['co_bond'] ?? 0,
        ccBreak: events['cc_break'] ?? 0,
        coBreak: events['co_break'] ?? 0,
        nonFinite,
        nullStats,
        boxPolarFraction: boxAll > 0 ? boxPolar / boxAll : 0,
        plateShellPolarFraction: plateShell.frac,
        plateShellCount: plateShell.n,
        shellNullMean,
        shellNullSd,
        shellNullMin: shellNull.length ? Math.min(...shellNull) : 0,
        shellNullMax: shellNull.length ? Math.max(...shellNull) : 0,
        shellNullN: shellNull.length,
        siteCount,
        sitesHolding,
        freeHolding,
        mode: modeArg,
        bulkFrozen,
        siteFraction: fracArg,
      }
    },
    arm.seed,
    arm.steps,
    arm.mode,
    arm.siteFraction ?? null,
  )
}
