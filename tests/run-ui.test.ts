import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// Run-control page (viewer/run.html + viewer/run.ts): the user starts/pauses/stops the run
// themselves instead of a script launching a long one for them. Kept tiny on purpose -- the
// "tiny" size preset is this page's own DEFAULT selection, and the step cap set below is small, so
// this test costs seconds, not minutes.
//
// Task 'consolidation' (2026-08-20): that preset is now 3938 particles, not ~470, because its
// solvent count is derived from the box at the measured LIQUID density (0.8 sigma^-3, viewer/
// run-types.ts's LIQUID_SOLVENT_DENSITY) instead of the token `W: 100` it used to carry -- the page
// was previewing a medium no current gate is measured in. Measured cost of the change, not
// estimated: 2155-2158 steps/s at 3938 particles against 2086-2302 at the old ~13100-particle
// "default" preset, i.e. this test still costs seconds.
test('управление прогоном: старт держит счёт, пауза останавливает его, стоп останавливает навсегда', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  // Controls exist.
  for (const id of ['size-select', 'stage-select', 'step-cap', 'start-btn', 'pause-btn', 'stop-btn']) {
    const handle = await page.$(`#${id}`)
    expect(handle, `#${id} должен существовать`).not.toBeNull()
  }

  // window.runUI exposes the required shape before anything has started.
  const initialShape = await page.evaluate(() => {
    const ui = (window as any).runUI
    return {
      hasState: 'state' in ui,
      hasSteps: 'steps' in ui,
      hasStepsPerSecond: 'stepsPerSecond' in ui,
      hasStage: 'stage' in ui,
      hasEvidence: 'evidence' in ui,
      hasTrace: 'trace' in ui,
      state: ui.state,
    }
  })
  expect(initialShape.hasState).toBe(true)
  expect(initialShape.hasSteps).toBe(true)
  expect(initialShape.hasStepsPerSecond).toBe(true)
  expect(initialShape.hasStage).toBe(true)
  expect(initialShape.hasEvidence).toBe(true)
  expect(initialShape.hasTrace).toBe(true)
  expect(initialShape.state).toBe('idle')

  // Keep the run tiny: a small step cap on top of the page's own tiny/16^3 default preset.
  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '6000'
  })

  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')

  const runningState = await page.evaluate(() => ({ state: (window as any).runUI.state, steps: (window as any).runUI.steps }))
  expect(runningState.state).toBe('running')
  expect(runningState.steps).toBeGreaterThan(0)

  // --- PAUSE: steps must stop advancing ---------------------------------------------------------
  await page.click('#pause-btn')
  await page.waitForFunction('window.runUI.state === "paused"')
  // Let any in-flight step() batch (already submitted before the click) finish settling.
  await new Promise((r) => setTimeout(r, 600))
  const pausedSteps1 = await page.evaluate(() => (window as any).runUI.steps)
  await new Promise((r) => setTimeout(r, 600))
  const pausedSteps2 = await page.evaluate(() => (window as any).runUI.steps)
  const pausedState = await page.evaluate(() => (window as any).runUI.state)
  expect(pausedState).toBe('paused')
  expect(pausedSteps2).toBe(pausedSteps1)

  // --- STOP: state becomes 'stopped' and stepping has ceased for good --------------------------
  await page.click('#stop-btn')
  await page.waitForFunction('window.runUI.state === "stopped"')
  await new Promise((r) => setTimeout(r, 600))
  const stoppedSteps1 = await page.evaluate(() => (window as any).runUI.steps)
  await new Promise((r) => setTimeout(r, 600))
  const stoppedSteps2 = await page.evaluate(() => (window as any).runUI.steps)
  const stoppedState = await page.evaluate(() => (window as any).runUI.state)
  expect(stoppedState).toBe('stopped')
  expect(stoppedSteps2).toBe(stoppedSteps1)

  // Progress numbers reported at least once, honestly (not fabricated).
  const finalUI = await page.evaluate(() => {
    const ui = (window as any).runUI
    return { stepsPerSecond: ui.stepsPerSecond, traceLength: ui.trace.length, stage: ui.stage }
  })
  expect(finalUI.stepsPerSecond).toBeGreaterThan(0)
  expect(finalUI.traceLength).toBeGreaterThan(0)
  expect(typeof finalUI.stage).toBe('string')
})

// Regression: starting a SECOND run in the same page (after the first was stopped) used to wedge
// forever after exactly one STEP_BATCH -- getGpu() memoizes ONE GPUDevice for the whole page
// (engine/src/gpu.ts), and SoupSystem had no dispose(), so the first run's ~20 GPUBuffers stayed
// alive on that shared device when the second run's createSoup() allocated its own fresh set.
// SoupSystem.dispose() (soup/src/sim.ts) plus viewer/run.ts calling it on every run-ending path
// (finishRun/failRun) and defensively before starting a new one is the fix this guards. Kept tiny
// (the page's own default "tiny" preset, a low step cap) so this costs seconds, not minutes.
test('второй прогон на той же странице: счёт шагов продвигается дальше одного STEP_BATCH', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '500'
  })

  // --- RUN 1: start it, then stop it early (covers both triggers named in the bug report --
  // "STOP then START" and "let it finish on its step cap" -- by stopping manually here and letting
  // run 2 below run to its own cap). --------------------------------------------------------------
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')
  await page.click('#stop-btn')
  await page.waitForFunction('window.runUI.state === "stopped"')

  // --- RUN 2: a fresh SoupSystem on the SAME page/device. Before the fix this got exactly one
  // STEP_BATCH in and then wedged forever (state stuck at 'running', steps stuck at 250). ---------
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')

  // Prove it is genuinely advancing, not just slow: wait for it to pass the first STEP_BATCH. (Not
  // asserting state === 'running' at this exact instant -- with a small step cap the run can
  // legitimately finish between this wait resolving and the next read, which is itself part of
  // the proof nothing is wedged.)
  await page.waitForFunction('window.runUI.steps > 250', { timeout: 30_000 })

  const run2 = await page.evaluate(() => ({
    steps: (window as any).runUI.steps,
    error: (window as any).runUI.error,
  }))
  expect(run2.error).toBeNull()
  expect(run2.steps).toBeGreaterThan(250)

  // Let it reach its own step cap and confirm it finishes cleanly (not via the watchdog/error path).
  await page.waitForFunction('window.runUI.state === "stopped"', { timeout: 30_000 })
  const final = await page.evaluate(() => ({
    steps: (window as any).runUI.steps,
    error: (window as any).runUI.error,
  }))
  expect(final.error).toBeNull()
  expect(final.steps).toBe(500)
})

// --- 2026-08 UI-fixes task: regressions for the four user-reported problems -----------------------

// Item 1a: the run crashed with `densityProfileZ: бусина с z=... вне [0, ...)` because
// computeHeadPeaks (soup/src/stages.ts) fed densityProfileZ a raw, unfiltered snapshot, and a soup
// particle can legitimately read back with z outside [0, box[2]) on a live snapshot (see that
// function's own updated doc comment). Two things are checked here, both through page.evaluate on
// this SAME production module (a dynamic import of the exact file viewer/run.ts itself imports, not
// a copy) so this is the real code, not a stand-in:
//  (a) the precise defect, reproduced deterministically (an escaped head bumped into the exact
//      two-slab profile from tests/soup-amphiphile.test.ts's own passing case) -- must not throw;
//  (b) a real run, through the actual UI, advancing a materially larger number of steps than any
//      other committed run-ui test -- every sampled tick calls this exact code path, so this is the
//      best available defense-in-depth check that nothing in it can still kill the run loop.
test('прогон переживает бусину, улетевшую за коробку по z, и не рушится дольше обычного (item 1a)', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  // (a) the exact defect, direct and deterministic. Uses an indirect `import()` (via `new
  // Function`, never a literal `import(...)` in this callback's own source) -- vitest's own vite-
  // node transform rewrites a literal dynamic import in test-file source into an internal
  // `__vite_ssr_dynamic_import__` helper before this function is ever serialised for
  // page.evaluate(), and that helper does not exist in the BROWSER page this runs in.
  const direct = await page.evaluate(async () => {
    const dynamicImport = new Function('url', 'return import(url)') as (url: string) => Promise<any>
    const { computeHeadPeaks, loadStageThresholds } = await dynamicImport('/soup/src/stages.ts')
    const { loadSoup } = await dynamicImport('/soup/src/rules.ts')
    const thresholds = loadStageThresholds()
    const monomers = loadSoup().monomers
    const box: [number, number, number] = [20, 20, 20]
    const n = 700
    const parts = new Float32Array(n * 4)
    for (let i = 0; i < n; i++) {
      parts[i * 4 + 2] = i < n / 2 ? 6 : 13 // two thin slabs, 7 sigma apart -- a trusted bilayer signal
      parts[i * 4 + 3] = 1
    }
    parts[0 * 4 + 2] = box[2] + 4 // one head walks out of the box in z
    let threw: string | null = null
    let result: number | string | null = null
    try {
      result = computeHeadPeaks(parts, box, monomers, thresholds)
    } catch (e) {
      threw = (e as Error).message
    }
    return { threw, result }
  })
  expect(direct.threw).toBeNull()
  expect(direct.result).toBe(2) // one escapee out of 700 heads still leaves a trusted bilayer signal

  // (b) a real run, through the real UI, well past this page's own default step cap (20000) and
  // past the OTHER committed run-ui tests' step caps (6000/500) -- every sampled tick (~4/s) calls
  // detectStage -> computeHeadPeaks on the run's own live snapshot.
  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '40000'
  })
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')
  await page.waitForFunction('window.runUI.state === "stopped" || window.runUI.state === "error"', { timeout: 30_000 })
  const final = await page.evaluate(() => ({
    state: (window as any).runUI.state,
    error: (window as any).runUI.error,
    steps: (window as any).runUI.steps,
    traceHasException: (document.getElementById('trace-log')?.textContent ?? '').includes('densityProfileZ'),
  }))
  expect(final.state).toBe('stopped') // NOT 'error' -- see runUI.errorKind's own doc comment
  expect(final.error).toBeNull()
  expect(final.steps).toBe(40000)
  expect(final.traceHasException).toBe(false)
})

// Item 2: the scene disappeared at some zoom levels/angles -- InstancedMesh frustum-culls against
// its GEOMETRY's own bounding sphere at the object's local origin, which knows nothing about where
// the per-instance transforms actually put anything (see buildMeshes()'s own doc comment in
// viewer/run.ts). Verified by scripting the camera through several distances and angles (via the
// sceneDebug test hook, page.evaluate only) and asserting a non-trivial fraction of non-background
// pixels at EVERY one, not just a single eyeballed shot. Confirmed against a real regression: with
// frustumCulled reverted to its (unset, so three.js default `true`) behaviour, the (d=0.4,
// theta=180, phi=120) pose below reads back EXACTLY 0 -- this is not a made-up threshold.
test('сцена не пропадает при развороте камеры по нескольким дистанциям и углам (item 2)', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.trace.length > 0') // at least one full sample -> a real snapshot exists to draw

  const sweeps = [
    { d: 0.4, theta: 0, phi: 60 },
    { d: 0.4, theta: 180, phi: 120 },
    { d: 1.0, theta: 45, phi: 45 },
    { d: 1.0, theta: 270, phi: 90 },
    { d: 3.0, theta: 90, phi: 30 },
    { d: 3.0, theta: 200, phi: 150 },
    { d: 8.0, theta: 10, phi: 80 },
  ]
  const results: { d: number; theta: number; phi: number; frac: number }[] = []
  for (const s of sweeps) {
    const frac = await page.evaluate(
      (d, theta, phi) => {
        ;(window as any).sceneDebug.setCameraOrbit(d, theta, phi)
        return (window as any).sceneDebug.nonBackgroundPixelFraction() as number
      },
      s.d,
      s.theta,
      s.phi,
    )
    results.push({ ...s, frac })
  }
  await page.click('#stop-btn')

  for (const r of results) {
    expect(r.frac, `d=${r.d} theta=${r.theta} phi=${r.phi} дал долю ${r.frac}`).toBeGreaterThan(0.0001)
  }
})

// Item 3: the user could only pick between two fixed presets, and an invalid choice would either
// crash createSoup's own neighbour-grid guard or (for a box too small to ever contain a qualifying
// cavity) run for however long the user let it, unable to ever succeed. Both guards now run BEFORE
// createSoup is even called (validateSizeSelection() in viewer/run.ts, reusing soup/src/sim.ts's own
// exported planSoupGrid rather than a second copy) -- a violating choice must refuse with a visible
// message and leave the run 'idle', never crash and never silently start anyway.
test('слишком малый бокс отклоняется с понятным сообщением, а не рушит прогон (item 3)', async () => {
  const page = await gpuPage()
  const consoleErrors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`))
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  // A box side of 3σ fails BOTH guards planSoupGrid checks (too few neighbour-grid cells, and well
  // under the minimum-image bound) and is nowhere near closureMinBoxSide's own ~13σ floor either.
  await page.$eval('#box-side-input', (el) => {
    ;(el as HTMLInputElement).value = '3'
    el.dispatchEvent(new Event('input', { bubbles: true }))
  })
  const preview = await page.evaluate(() => ({
    errorHidden: (document.getElementById('size-error') as HTMLElement).hidden,
    errorText: document.getElementById('size-error')?.textContent ?? '',
  }))
  expect(preview.errorHidden).toBe(false) // refused already visible in the live preview, before any click
  expect(preview.errorText.length).toBeGreaterThan(0)

  await page.click('#start-btn')
  // Give any (incorrect) async start path a moment to misbehave before asserting nothing happened.
  await new Promise((r) => setTimeout(r, 500))
  const after = await page.evaluate(() => ({
    state: (window as any).runUI.state,
    steps: (window as any).runUI.steps,
    errorHidden: (document.getElementById('size-error') as HTMLElement).hidden,
    errorText: document.getElementById('size-error')?.textContent ?? '',
  }))
  expect(after.state).toBe('idle') // refused before starting -- not 'error', not 'running'
  expect(after.steps).toBe(0)
  expect(after.errorHidden).toBe(false)
  expect(after.errorText.length).toBeGreaterThan(0)
  expect(consoleErrors).toEqual([]) // refused cleanly -- no uncaught exception anywhere

  // Recovery: fixing the box side back to a valid value clears the refusal and a run starts fine.
  await page.$eval('#box-side-input', (el) => {
    ;(el as HTMLInputElement).value = '16'
    el.dispatchEvent(new Event('input', { bubbles: true }))
  })
  const cleared = await page.evaluate(() => (document.getElementById('size-error') as HTMLElement).hidden)
  expect(cleared).toBe(true)
  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '500'
  })
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')
})

// --- composition control (soup-to-vesicle, tail-length task, 2026-08-17) -------------------------
// The user can now set the composition -- specifically the head count (data/soup.json's monomer
// `O`) -- directly, not just box side and a global particle scale. Two things to prove: (a) a
// composition choice actually reaches the preview and a real run (exercised, not just parsed), and
// (b) an invalid composition (a negative head count) is refused with a clear message, the same
// "refuse, don't crash" discipline item 3's box-size guard above already established, rather than
// falling through to createSoup()'s own much less specific throw.
test('состав: число голов задаётся напрямую, видно в предпросмотре и реально уменьшает бульон (item "composition")', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  for (const id of ['head-count-input']) {
    const handle = await page.$(`#${id}`)
    expect(handle, `#${id} должен существовать`).not.toBeNull()
  }

  // Default (tiny preset, its own default composition) shows O:50 in the preview, per the task's
  // own requirement that presets stay the defaults.
  const defaultPreview = await page.evaluate(() => document.getElementById('size-preview')?.textContent ?? '')
  expect(defaultPreview).toContain('O:50')

  // A quarter of the default heads (Task Two's own reduced-head-count measurement, exercised here
  // through the real control instead of a script bypassing it) -- the preview must show the new O
  // count and the new (smaller) total, before anything is started.
  await page.$eval('#head-count-input', (el) => {
    ;(el as HTMLInputElement).value = '12'
    el.dispatchEvent(new Event('input', { bubbles: true }))
  })
  const reducedPreview = await page.evaluate(() => document.getElementById('size-preview')?.textContent ?? '')
  expect(reducedPreview).toContain('O:12')
  expect(reducedPreview).not.toContain('O:50')
  const errorHiddenAfterReduce = await page.evaluate(() => (document.getElementById('size-error') as HTMLElement).hidden)
  expect(errorHiddenAfterReduce).toBe(true) // a valid, smaller head count is not a refusal

  // C/H/M are untouched by this control -- only O moved (particle-scale still scales those three).
  expect(reducedPreview).toContain('C:200')
  expect(reducedPreview).toContain('H:200')
  expect(reducedPreview).toContain('M:20')

  // The reduced composition actually starts and runs -- not just accepted by the preview.
  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '500'
  })
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')

  await page.waitForFunction('window.runUI.state === "stopped"', { timeout: 30_000 })
  const final = await page.evaluate(() => ({ steps: (window as any).runUI.steps, error: (window as any).runUI.error }))
  expect(final.error).toBeNull()
  expect(final.steps).toBe(500)
})

test('состав: отрицательное число голов отклоняется с понятным сообщением, а не рушит прогон (item "composition")', async () => {
  const page = await gpuPage()
  const consoleErrors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (e) => consoleErrors.push(`pageerror: ${e.message}`))
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  await page.$eval('#head-count-input', (el) => {
    ;(el as HTMLInputElement).value = '-5'
    el.dispatchEvent(new Event('input', { bubbles: true }))
  })
  const preview = await page.evaluate(() => ({
    errorHidden: (document.getElementById('size-error') as HTMLElement).hidden,
    errorText: document.getElementById('size-error')?.textContent ?? '',
  }))
  expect(preview.errorHidden).toBe(false) // refused already visible in the live preview, before any click
  expect(preview.errorText).toContain('cannot be negative')

  await page.click('#start-btn')
  // Give any (incorrect) async start path a moment to misbehave before asserting nothing happened.
  await new Promise((r) => setTimeout(r, 500))
  const after = await page.evaluate(() => ({
    state: (window as any).runUI.state,
    steps: (window as any).runUI.steps,
    errorHidden: (document.getElementById('size-error') as HTMLElement).hidden,
  }))
  expect(after.state).toBe('idle') // refused before starting -- not 'error', not 'running'
  expect(after.steps).toBe(0)
  expect(after.errorHidden).toBe(false)
  expect(consoleErrors).toEqual([]) // refused cleanly -- no uncaught exception anywhere

  // Recovery: a valid head count clears the refusal and a run starts fine.
  await page.$eval('#head-count-input', (el) => {
    ;(el as HTMLInputElement).value = '10'
    el.dispatchEvent(new Event('input', { bubbles: true }))
  })
  const cleared = await page.evaluate(() => (document.getElementById('size-error') as HTMLElement).hidden)
  expect(cleared).toBe(true)
  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '500'
  })
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')
})

// --- ocean-look task (2026-08) --------------------------------------------------------------------
// The scene got a depth-graded background, water-like lighting/materials and a drifting caustic
// backdrop (all in viewer/run.ts, see paintOceanBackdrop/frameCamera/buildMeshes there). None of
// that is allowed to break what already worked: the run still starts and advances real rendered
// frames (sceneDebug.renderedFrameCount, the same counter used for this task's own before/after
// frame-rate measurement), and every honesty note (#honesty-note, #cavity-honesty inside #cavity,
// #atom-badge) is still present in the DOM -- only the background/material styling changed, never
// their text or existence. A screenshot proves something real (not a blank canvas) is on screen.
test('океанский фон: прогон стартует и рисует кадры, заметки честности на месте, скриншот нетривиален (ocean-look)', async () => {
  const page = await gpuPage()
  await page.goto(new URL('/viewer/run.html', page.url()).href, { waitUntil: 'load' })

  // Honesty notes still exist in the DOM, unchanged by the new background/material styling.
  for (const id of ['honesty-note', 'cavity-honesty', 'atom-badge']) {
    const handle = await page.$(`#${id}`)
    expect(handle, `#${id} должен существовать`).not.toBeNull()
  }

  // Task 'consolidation' (2026-08-20): the buttons must actually RECEIVE their own clicks. The
  // bottom-anchored honesty notes are absolutely positioned and grow upward, and twice now a taller
  // #controls has ended up UNDER one of them at the same z-index -- which does not look broken, it
  // silently eats the click (measured: elementFromPoint over #start-btn's centre came back as
  // `visibility-note`, the run never started, state stayed 'idle', trace empty, no error anywhere).
  // Asserting the hit target is the only thing that catches that class of failure, since every
  // other assertion in this file happens AFTER a click it assumes landed.
  // The form fields above the buttons are probed the same way: when the left panels moved into one
  // column, the progress panel landed in it too and covered them, with every button still clickable.
  const hitTargets = await page.evaluate(() =>
    ['start-btn', 'pause-btn', 'stop-btn', 'size-select', 'box-side-input', 'step-cap'].map((id) => {
      const r = document.getElementById(id)!.getBoundingClientRect()
      return { id, hitId: document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2)?.id ?? 'null' }
    }),
  )
  console.log(`RUN-UI CLICKTARGET ${JSON.stringify(hitTargets)}`)
  for (const t of hitTargets) expect(t.hitId, `клик по #${t.id} должен попадать в саму кнопку`).toBe(t.id)

  await page.$eval('#step-cap', (el) => {
    ;(el as HTMLInputElement).value = '500'
  })
  await page.click('#start-btn')
  await page.waitForFunction('window.runUI.steps > 0 && window.runUI.state === "running"')

  // Real rendered frames are still being produced with the new background in place.
  const frames0 = await page.evaluate(() => (window as any).sceneDebug.renderedFrameCount)
  await page.waitForFunction(`window.sceneDebug.renderedFrameCount > ${frames0}`)

  // Task 'consolidation' (2026-08-20): WHICH species is drawn, not just "something is". The clay
  // platelet is the one most easily lost -- it is the only species drawn as flat plates rather than
  // spheres, its bead count is derived from the box instead of carried in `start`, and it sits at
  // the box floor. sceneDebug.instanceCounts() reads the counts straight off the meshes draw()
  // writes, so this is the platelet's own instances, not a pixel guess.
  const instances = await page.evaluate(() => (window as any).sceneDebug.instanceCounts())
  console.log(`RUN-UI INSTANCES ${JSON.stringify(instances)}`)
  expect(instances.K.visible, 'пластина глины должна быть видимой').toBe(true)
  expect(instances.K.count, 'у пластины глины должны быть нарисованные экземпляры').toBeGreaterThan(0)
  expect(instances.W.visible, 'растворитель по умолчанию не рисуется').toBe(false)
  // Read AFTER a real frame has been drawn, deliberately: instanceCounts() reports what the last
  // draw() wrote, so asking before the first frame reports 0 for every species -- measured, an
  // earlier version of this assertion sat right after `steps > 0` and failed for exactly that
  // reason, on a platelet that was in fact on screen.

  // A screenshot is non-trivial (not a blank/solid canvas) -- same size floor as tests/viewer.test.ts
  // uses for the same purpose.
  const shot = await page.screenshot({ encoding: 'binary' })
  expect(shot.length).toBeGreaterThan(5000)

  await page.waitForFunction('window.runUI.state === "stopped"', { timeout: 30_000 })
  const final = await page.evaluate(() => ({ steps: (window as any).runUI.steps, error: (window as any).runUI.error }))
  expect(final.error).toBeNull()
  expect(final.steps).toBe(500)
})
