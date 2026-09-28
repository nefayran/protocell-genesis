// Undulation spectrum and bending modulus (Task 7 of gate 6): does the membrane's own thermal
// height fluctuations, run through the Helfrich elastic model, land kappa in the literature range
// (5-50 kT)? Pure CPU math -- no GPU calls -- same split as metrics.ts/aggregate.ts: the GPU-touching
// half (stepping the system and reading back positions) stays in sim.ts/index.ts, this module only
// consumes a snapshot of positions.
//
// NORMALISATION CONVENTION (fixed once, used identically by the generator and the estimator -- a
// mismatched factor here is invisible in the final kappa, it just yields a plausible-looking number
// that is wrong by a constant multiple):
//
//   h_q = (1/N) * sum_r h(r) * exp(-i q.r),   N = n^2
//   <|h_q|^2> = kT / (A * kappa * q^4)          (Helfrich, small-q limit)
//
// so the inverse transform (used to CONSTRUCT a field from prescribed Fourier amplitudes, in
// synthesizeHeightField below) carries no 1/N factor: h(r) = sum_q h_q * exp(+i q.r), by the
// standard forward/inverse DFT pairing that keeps sum_r exp(i(q-q').r)/N = delta_{q,q'} exact on the
// discrete grid.

/** Same seeded generator as sim.ts's mulberry32 -- copied locally rather than imported, since it is
 * an unexported implementation detail there and this module has no other reason to depend on
 * sim.ts (heightField/spectrum/fitBendingModulus are pure functions of arrays the caller already
 * has; only index.ts's facade touches a live System). */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Wraps one coordinate into [0, box) -- matches wrap1 in sim.ts/aggregate.ts. */
function wrap1(v: number, box: number): number {
  return v - Math.floor(v / box) * box
}

export interface Spectrum {
  q: number[]
  hq2: number[]
}

/** Membrane midplane height field on an n x n lateral grid, from a raw bead snapshot. For each
 * cell, h(cell) is the MEAN z of every bead (head or tail -- a lipid's three beads move together
 * with the local membrane patch, so averaging over all of them cancels the much smaller, higher-
 * frequency bond-length/thickness noise rather than injecting it into the undulation signal),
 * minus the field's own mean height ("relative to the membrane's own centre", per the brief) --
 * mathematically this offset only ever lands in the q=0 mode, which fitBendingModulus never uses,
 * but subtracting it up front keeps the returned field meaningful on its own.
 *
 * Empty columns (no bead's lateral position fell in that cell) are filled with the grid's own mean
 * height rather than left at 0 or interpolated from neighbours: measured to occur at 4000 lipids on
 * a 32x32 grid (12000 beads over 1024 cells, ~11.7/cell on average, but not perfectly uniform). A
 * flat fill at the mean adds zero deviation of its own -- no spurious long-wavelength slope, unlike
 * nearest-neighbour interpolation, which can manufacture a gradient where the true membrane has
 * none -- at the cost of a slight, documented underestimate of high-q power from cells landing
 * exactly on empty columns. Coarsening the grid or discarding the sample were the brief's other
 * sanctioned options; this one was chosen because it needs no extra bookkeeping and never discards
 * real data.
 */
export function heightField(positions: Float32Array, box: [number, number, number], n: number): Float32Array {
  const [Lx, Ly] = box
  const dx = Lx / n
  const dy = Ly / n
  const nBeads = positions.length / 4
  const sums = new Float64Array(n * n)
  const counts = new Int32Array(n * n)
  for (let i = 0; i < nBeads; i++) {
    const x = wrap1(positions[i * 4], Lx)
    const y = wrap1(positions[i * 4 + 1], Ly)
    const z = positions[i * 4 + 2]
    const cx = Math.min(n - 1, Math.floor(x / dx))
    const cy = Math.min(n - 1, Math.floor(y / dy))
    const idx = cy * n + cx
    sums[idx] += z
    counts[idx] += 1
  }

  const h = new Float64Array(n * n)
  let sumMean = 0
  let nonEmpty = 0
  for (let i = 0; i < n * n; i++) {
    if (counts[i] > 0) {
      h[i] = sums[i] / counts[i]
      sumMean += h[i]
      nonEmpty += 1
    }
  }
  if (nonEmpty === 0) throw new Error(`heightField: the ${n}x${n} grid is empty, not a single bead fell into any column`)
  const globalMean = sumMean / nonEmpty

  const out = new Float32Array(n * n)
  for (let i = 0; i < n * n; i++) out[i] = (counts[i] > 0 ? h[i] : globalMean) - globalMean
  return out
}

export interface ColumnNoiseStats {
  meanBeadsPerColumn: number
  meanColumnVariance: number
  predictedFloor: number
}

/** Predicts the flat, q-INDEPENDENT noise floor a column-averaged height field carries from
 * intra-column bead scatter (any spread of individual bead z around the local column mean, within
 * one lateral cell). For i.i.d. per-column noise of variance sigma_col^2 =
 * sigma_bead^2/beadsPerColumn, the DFT of a spatially UNCORRELATED (white) field has
 * E[|h_q|^2] = sigma_col^2/N for every q!=0 -- uncorrelated noise has no preferred wavelength, so
 * it lands as a flat plateau rather than falling off with q the way genuine bending power does.
 *
 * sigma_bead^2 is estimated directly from the data: the population variance of individual bead z
 * around each column's OWN mean, averaged across non-empty columns weighted by bead count (so
 * columns that actually have beads in them, not sparse edge cases, dominate the estimate).
 *
 * Diagnostic only -- takes the SAME positions/box/n a heightField()+spectrum() call would, but
 * predicts what a purely-noise floor in that measurement would look like, for comparison against
 * the observed high-q plateau (task-7-report.md: measured ~4.5-6e-5 with the all-beads/32x32
 * construction, motivating tailEndBeads() and a coarser grid below). Not used by
 * heightField/spectrum/fitBendingModulus themselves. */
export function columnNoiseStats(positions: Float32Array, box: [number, number, number], n: number): ColumnNoiseStats {
  const [Lx, Ly] = box
  const dx = Lx / n
  const dy = Ly / n
  const nBeads = positions.length / 4
  const sums = new Float64Array(n * n)
  const sqSums = new Float64Array(n * n)
  const counts = new Int32Array(n * n)
  for (let i = 0; i < nBeads; i++) {
    const x = wrap1(positions[i * 4], Lx)
    const y = wrap1(positions[i * 4 + 1], Ly)
    const z = positions[i * 4 + 2]
    const cx = Math.min(n - 1, Math.floor(x / dx))
    const cy = Math.min(n - 1, Math.floor(y / dy))
    const idx = cy * n + cx
    sums[idx] += z
    sqSums[idx] += z * z
    counts[idx] += 1
  }
  let totalBeads = 0
  let nonEmpty = 0
  let weightedVarSum = 0
  for (let i = 0; i < n * n; i++) {
    if (counts[i] > 0) {
      const mean = sums[i] / counts[i]
      const variance = Math.max(0, sqSums[i] / counts[i] - mean * mean)
      weightedVarSum += variance * counts[i]
      totalBeads += counts[i]
      nonEmpty += 1
    }
  }
  if (nonEmpty === 0) throw new Error(`columnNoiseStats: the ${n}x${n} grid is empty`)
  const meanBeadsPerColumn = totalBeads / nonEmpty
  const meanColumnVariance = weightedVarSum / totalBeads
  const N = n * n
  const predictedFloor = meanColumnVariance / (meanBeadsPerColumn * N)
  return { meanBeadsPerColumn, meanColumnVariance, predictedFloor }
}

/** Extracts the tail-end bead -- the third of every three, closest to the bilayer midplane BY
 * CONSTRUCTION -- from a flat positions array. Every lipid layout in sim.ts (layoutBilayer,
 * layoutRandom) writes its three beads in the fixed order head, tail1, tail2, and nothing in this
 * engine ever reorders beads within a lipid, so index%3===2 is tail2 for the whole run.
 *
 * Built for heightField's caller (measureBendingModulusDetailed in index.ts): averaging heads AND
 * tails per lateral column mixes in their ~2 sigma vertical separation, which divided by only
 * sqrt(beads per column) is large enough to dominate the column mean's own noise over most of the
 * q range -- measured as the flat ~4.5-6e-5 plateau spanning ~90% of the searched spectrum
 * (task-7-report.md, columnNoiseStats above quantifies this). Restricting to the beads that sit at
 * the midplane already removes that leading spread; heightField itself is unchanged and works on
 * whatever bead subset it is given. */
export function tailEndBeads(positions: Float32Array): Float32Array {
  const n = positions.length / 4
  if (n % 3 !== 0) {
    throw new Error(`tailEndBeads: ${n} beads is not divisible by 3, so the layout is not head/tail1/tail2 per lipid`)
  }
  const nLipids = n / 3
  const out = new Float32Array(nLipids * 4)
  for (let lip = 0; lip < nLipids; lip++) {
    out.set(positions.subarray((lip * 3 + 2) * 4, (lip * 3 + 2) * 4 + 4), lip * 4)
  }
  return out
}

/** Discrete Fourier amplitude spectrum of a height field on an n x n grid: |h_q|^2 for every mode
 * (mx, my) up to the grid's own Nyquist limit (|mx|,|my| <= floor(n/2)), skipping the DC mode
 * (mx=my=0, where q=0 and the Helfrich formula has a pole). Computed by direct summation over
 * modes -- a nested loop per mode, each doing an O(n^2) sum over the real-space grid -- per the
 * brief: a 64x64 grid and a dozen-ish low-order modes are a fraction of a second, no FFT needed,
 * and this keeps the code a direct transcription of the normalisation convention above rather than
 * an FFT implementation with its own index-ordering conventions to get wrong.
 *
 * +q and -q modes are BOTH included (their |h_q|^2 are equal by construction for a real field --
 * Hermitian symmetry -- so this duplicates points, not information); fitBendingModulus's OLS fit is
 * unaffected by exact duplicate points, and keeping the full set here means spectrum()'s contract
 * ("the discrete spectrum of this field") does not depend on an arbitrary choice of representative
 * half-plane. The result is sorted by ascending q for a readable q/hq2 pairing.
 */
export function spectrum(h: Float32Array, n: number, box: [number, number, number]): Spectrum {
  const [Lx, Ly] = box
  const N = n * n
  const dx = Lx / n
  const dy = Ly / n
  const half = Math.floor(n / 2)

  const entries: { q: number; hq2: number }[] = []
  for (let mx = -half; mx <= half; mx++) {
    const qx = (2 * Math.PI * mx) / Lx
    for (let my = -half; my <= half; my++) {
      if (mx === 0 && my === 0) continue
      const qy = (2 * Math.PI * my) / Ly
      let re = 0
      let im = 0
      for (let iy = 0; iy < n; iy++) {
        const y = iy * dy
        const rowBase = iy * n
        for (let ix = 0; ix < n; ix++) {
          const phase = qx * ix * dx + qy * y
          const hv = h[rowBase + ix]
          re += hv * Math.cos(phase)
          im -= hv * Math.sin(phase)
        }
      }
      re /= N
      im /= N
      entries.push({ q: Math.hypot(qx, qy), hq2: re * re + im * im })
    }
  }
  entries.sort((a, b) => a.q - b.q)
  return { q: entries.map((e) => e.q), hq2: entries.map((e) => e.hq2) }
}

/** Ordinary least squares fit of log(hq2) against log(q), restricted to q in (0, qMax]. Shared by
 * fitBendingModulus (which additionally checks the slope and converts the intercept to kappa) and
 * by the ensemble facade in index.ts (which fits each block of an ensemble separately to report
 * kappa's scatter across blocks) -- both need the same regression without a mandatory throw on
 * every one of the many block-level fits, so the -4 sanity check lives only in fitBendingModulus. */
export function logLogFit(s: Spectrum, qMax: number): { slope: number; intercept: number; n: number } {
  const xs: number[] = []
  const ys: number[] = []
  for (let i = 0; i < s.q.length; i++) {
    if (s.q[i] > 0 && s.q[i] <= qMax) {
      xs.push(Math.log(s.q[i]))
      ys.push(Math.log(s.hq2[i]))
    }
  }
  const n = xs.length
  if (n < 2) throw new Error(`logLogFit: only ${n} mode(s) with q<=${qMax}, not enough for a regression`)
  const mx = xs.reduce((a, b) => a + b, 0) / n
  const my = ys.reduce((a, b) => a + b, 0) / n
  let sxy = 0
  let sxx = 0
  for (let i = 0; i < n; i++) {
    sxy += (xs[i] - mx) * (ys[i] - my)
    sxx += (xs[i] - mx) ** 2
  }
  const slope = sxy / sxx
  const intercept = my - slope * mx
  return { slope, intercept, n }
}

// How far the fitted slope may sit from the Helfrich prediction (-4) before the spectrum is
// rejected as not being a bending-mode spectrum at all. Not a physical model parameter -- it does
// not appear in any Cooke & Deserno formula and has no effect on the fitted kappa itself, only on
// whether the fit is trusted -- so, like AREA_MOVE_LOG_DELTA in sim.ts, it lives here as a named
// tuning knob rather than in data/params.json. A slope more than this far from -4 means q^-4 does
// not describe the data (wrong q range, insufficient averaging, or a genuinely non-Helfrich
// membrane), and reporting a kappa from it would be meaningless.
const SLOPE_TOLERANCE = 1

// Minimum number of independent |q| SHELLS (distinct radii -- several (mx,my) pairs typically
// share one, per spectrum()'s doc comment above) selectFitWindow's grown window must contain
// before a fit is trusted enough to report at all. Below this, "the slope looks fine" is not
// distinguishable from a couple of points landing near a line by chance; this is a statistical
// floor on the OLS fit's own credibility, not a physical model parameter, so it lives here rather
// than in data/params.json.
const MIN_FIT_SHELLS = 4

export interface FitWindow {
  qMax: number
  slope: number
  intercept: number
  nModes: number
  nShells: number
  /** false means this is NOT a trustworthy window: either fewer than MIN_FIT_SHELLS independent
   * |q| shells existed at all below ceilingQMax, or even the smallest MIN_FIT_SHELLS-shell window
   * failed the -4 tolerance. slope/intercept/qMax are still populated in this case (the best -- or
   * only -- attempt made), for DIAGNOSIS, but must not be converted to a kappa: per the ruling that
   * created this function, a window this data can't support is a failed measurement, and the fix
   * is a bigger box (more shells below the q^-4/q^-2 crossover), never a looser tolerance or a
   * smaller MIN_FIT_SHELLS. */
  valid: boolean
}

/** Chooses the fit window BY THE DATA rather than by a fixed mode-count ceiling: starting from the
 * smallest non-zero |q| shell, grows the window one shell at a time for as long as the resulting
 * OLS slope (over ALL modes, duplicates included, with q<=that shell) stays within
 * SLOPE_TOLERANCE of -4, and stops at the last shell where it still did.
 *
 * This exists because a fixed mode-count ceiling has no way to know, for a GIVEN box size, where
 * the q^-4 Helfrich regime actually ends: at q*thickness ~ O(1) the spectrum crosses over into
 * protrusion/tilt fluctuations (~q^-2), and fitting across that crossover yields a slope that is
 * neither -4 nor -2 and a kappa with no physical meaning -- measured here when this module's
 * ensemble facade first fit a FIXED ceiling at mode index 8 on a ~49-52 sigma box (task-7-report.md:
 * three independent runs gave slopes -3.25, -3.09, and a fourth that outright failed the -4 check
 * at -2.72 -- not scatter around a valid answer, a systematically too-wide window every time).
 *
 * `ceilingQMax` bounds how far the search is EVER allowed to grow (a generous upper limit, not the
 * window itself -- the natural choice is the grid's own Nyquist limit).
 *
 * Never throws -- returns `valid: false` instead of a working window when the data does not
 * support one (too few shells below ceilingQMax, or even the smallest window fails tolerance),
 * WITH the best-effort slope/intercept/qMax attempted, so a caller can still show the failed
 * attempt (this is what a BLOCKED report needs: the actual numbers, not just "it threw"). Callers
 * that need a hard failure on an invalid window (this module's own ensemble facade in index.ts)
 * check `.valid` themselves and decide; fitBendingModulus's own -4 throw, used independently of
 * this function by the round-trip test, is unrelated and unchanged. */
export function selectFitWindow(s: Spectrum, ceilingQMax: number): FitWindow {
  const shells = Array.from(new Set(s.q.filter((q) => q > 0 && q <= ceilingQMax))).sort((a, b) => a - b)

  if (shells.length < MIN_FIT_SHELLS) {
    const qMax = shells.length > 0 ? shells[shells.length - 1] : ceilingQMax
    let slope = NaN
    let intercept = NaN
    let nModes = 0
    try {
      const fit = logLogFit(s, qMax)
      slope = fit.slope
      intercept = fit.intercept
      nModes = fit.n
    } catch {
      // No modes at all below ceilingQMax -- leave the NaN placeholders; nShells: 0 below already
      // says everything a caller needs to know.
    }
    return { qMax, slope, intercept, nModes, nShells: shells.length, valid: false }
  }

  let best: FitWindow | undefined
  let lastAttempt: FitWindow = { qMax: shells[0], slope: NaN, intercept: NaN, nModes: 0, nShells: 0, valid: false }
  for (let k = MIN_FIT_SHELLS; k <= shells.length; k++) {
    const qMax = shells[k - 1]
    const { slope, intercept, n } = logLogFit(s, qMax)
    const valid = Math.abs(slope - -4) <= SLOPE_TOLERANCE
    lastAttempt = { qMax, slope, intercept, nModes: n, nShells: k, valid }
    if (!valid) break
    best = lastAttempt
  }
  return best ?? lastAttempt
}

/** Extracts the bending modulus from a spectrum: OLS-fits log(hq2) vs log(q) for q<=qMax (the
 * Helfrich q^-4 law is a small-q, continuum-elastic result and is not expected to hold at large q),
 * verifies the fitted slope is within SLOPE_TOLERANCE of -4 -- throwing with the measured slope if
 * not, since a spectrum that is not q^-4 is not a bending spectrum -- and reads kappa off the
 * intercept: the fitted line predicts log(hq2) at log(q)=0, i.e. hq2(q=1) = kT/(A*kappa*1^4), so
 * kappa = kT / (A * exp(intercept)). q=1 need not (and generally will not) fall inside the fitted
 * q range -- that is an ordinary property of a linear regression's intercept, not a limitation
 * introduced here. */
export function fitBendingModulus(s: Spectrum, kT: number, area: number, qMax: number): number {
  const { slope, intercept } = logLogFit(s, qMax)
  if (Math.abs(slope - -4) > SLOPE_TOLERANCE) {
    throw new Error(
      `fitBendingModulus: the slope of log<|h_q|^2> vs log q is ${slope.toFixed(3)}, ` +
        `not about -4 (tolerance +/-${SLOPE_TOLERANCE}); this is not a spectrum of bending modes`,
    )
  }
  return kT / (area * Math.exp(intercept))
}

/** Synthetic height field with a KNOWN bending modulus, for the round-trip test that pins the
 * normalisation convention above: generating a field with a given kappa and recovering the same
 * kappa through spectrum()+fitBendingModulus() is the only check that catches a constant-factor
 * convention error, which is otherwise invisible (it just produces a different, still-plausible
 * kappa).
 *
 * Sets |h_q| DETERMINISTICALLY to the theoretical target sqrt(kT/(A*kappa*q^4)) and randomises only
 * the PHASE -- not the amplitude. A real thermally-fluctuating membrane has |h_q|^2 itself
 * exponentially distributed around that target (equipartition per mode), which is exactly why the
 * real measurement in index.ts needs an ensemble average over many configurations; this generator
 * deliberately has no such noise, so a single call is enough to pin the convention against an exact
 * target instead of convolving a normalisation check with sampling noise.
 *
 * Builds the field by summing, for one representative of each +-q pair (q.x >= 0, excluding the
 * self-conjugate Nyquist boundary |m|=floor(n/2) where q and -q alias to the same grid index and a
 * free phase would make h(r) complex), 2*|h_q|*cos(q.r + phase) -- exactly the real-valued sum of
 * h_q*exp(i q.r) + h_{-q}*exp(-i q.r) for h_{-q} = conj(h_q), which is what "a real field with
 * prescribed |h_q| and free phase" means. */
export function synthesizeHeightField(
  n: number,
  box: [number, number, number],
  kappa: number,
  kT: number,
  seed: number,
): Float32Array {
  const [Lx, Ly] = box
  const area = Lx * Ly
  const dx = Lx / n
  const dy = Ly / n
  const half = Math.floor(n / 2)
  const rng = mulberry32(seed)
  const h = new Float64Array(n * n)

  for (let mx = 0; mx <= half - 1; mx++) {
    const qx = (2 * Math.PI * mx) / Lx
    const myLo = mx === 0 ? 1 : -(half - 1)
    for (let my = myLo; my <= half - 1; my++) {
      const qy = (2 * Math.PI * my) / Ly
      const q2 = qx * qx + qy * qy
      const amp = Math.sqrt(kT / (area * kappa * q2 * q2))
      const phase = rng() * 2 * Math.PI
      for (let iy = 0; iy < n; iy++) {
        const y = iy * dy
        const rowBase = iy * n
        for (let ix = 0; ix < n; ix++) {
          h[rowBase + ix] += 2 * amp * Math.cos(qx * ix * dx + qy * y + phase)
        }
      }
    }
  }
  return Float32Array.from(h)
}
