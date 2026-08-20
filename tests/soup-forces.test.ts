import { afterAll, expect, test } from 'vitest'
import { gpuPage, shutdownGpu } from './helpers/gpu'

afterAll(shutdownGpu)

// perf2-report.md correctness gate, mirroring tests/sim.test.ts's "сетка соседей даёт те же силы,
// что и полный перебор" for the soup engine's own dynamic-topology force kernel (soup/wgsl/
// step.wgsl's soup_force_main/soup_force_list_main against soup_force_brute_main, an O(N^2)
// reference with no neighbour grid at all). COMMITTED per the task brief: a neighbour-grid bug in
// this repo once silently doubled every force and looked like success, and while writing perf2's
// candidate (c) a DIFFERENT bug class showed up -- a WebGPU bind-group mismatch ('layout: auto'
// gives every pipeline a DISTINCT layout object even for an identical uniform) that failed
// validation and made the WHOLE submitted command buffer a silent no-op: forceBuf simply kept
// its all-zero initial value, so forces() and forcesBruteForce() both read back zeros and
// "agreed" perfectly -- a false pass that would have shipped a system computing no forces at all.
// Caught only by listening on the page's console for the browser's own GPU validation warnings,
// which this test now does itself and fails on, specifically so that failure mode cannot recur
// silently: a correctness test that can itself be silently skipped is not a correctness test.
test('сетка соседей (в т.ч. список Верле) даёт те же силы, что и полный перебор', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    // box=[16,16,16] с составом поменьше стандартного: достаточно частиц (~1220) и всех 4 видов
    // мономеров (включая катализатор и полярную голову), чтобы упражнять каждую ветку
    // nonbondedSoup/bondedForce, но не полный масштаб (13100/box 30) -- O(N^2) перебор растёт
    // квадратично, а это диагностика, не производственный прогон.
    const sys = await api.createSoup({
      box: [16, 16, 16],
      seed: 7,
      kT: 1.1,
      // W:0 (task 'broth-composition', 2026-08-18): CreateSoupOpts.start merges over
      // data/soup.json's own defaults, which now include a box-30-sized water count -- explicit
      // zero keeps this box=16 force-correctness diagnostic at its own originally-intended,
      // deliberately small scale.
      // Task 'clay-surface' (2026-08-19): the CLAY-FREE control. data/soup.json's shipped composition
      // now carries the mineral platelet (clay.enabled), and this fixture must not: it is a force-correctness diagnostic at a deliberately small hand-picked scale, not a broth.
      // A clay-free system is byte-identical to the pre-task engine, so every number in this file is
      // unchanged by that task -- which is exactly what makes it a usable reference.
      clay: false,
      start: { C: 600, O: 100, H: 500, M: 20, W: 0 },
    })
    // Несколько реальных шагов ПЕРЕД сравнением -- проверяем силы не только на стартовой решётке
    // (где расстояния почти идеальны и многие ветки WCA/attr не задействованы), а на уже
    // разошедшейся конфигурации.
    await sys.step(200)
    const a = await sys.forces()
    const b = await sys.forcesBruteForce()
    let max = 0
    let sumAbs = 0
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]))
      sumAbs += Math.abs(b[i])
    }
    return { maxDiff: max, meanAbsRef: sumAbs / a.length, n: a.length }
  })
  expect(consoleWarnings, `браузер сообщил об ошибке/предупреждении GPU во время теста:\n${consoleWarnings.join('\n')}`).toEqual([])
  // Допуск float32: силы здесь порядка нескольких десятков (WCA близко к контакту может давать
  // большие отталкивающие пики), поэтому абсолютный порог даётся с запасом поверх типичной
  // float32-ошибки накопления суммы по ~30-60 соседям на частицу.
  expect(r.maxDiff).toBeLessThan(1e-2)
})

// Task 'electrostatics' (2026-08-20): THE SAME correctness gate WITH CHARGE ON. The screened-Coulomb
// term lives inside nonbondedSoup, which all three force paths reach -- the grid walk
// (soup_force_main), the Verlet-list walk (soup_force_list_main, in soup/wgsl/verlet.wgsl since this
// task's responsibility split) and the O(N^2) reference (soup_force_brute_main). So the brute-force
// comparison above is exactly the instrument that can prove the new term is applied IDENTICALLY on all
// three, and it is extended here rather than duplicated in a new file: a term added to the shared
// function but wired into only two of the three bind-group layouts would fail bind-group validation
// (caught by the same console listener), and one applied with the wrong sign or the wrong index
// convention on the sorted-gather path would show up as a maxDiff blow-up here.
//
// It also asserts the term is actually DOING something: with charge on, the force must DIFFER from the
// same configuration with charge off, by more than the float32 tolerance. Otherwise a test that
// silently ran with all-zero charges would pass and prove nothing -- which is the exact false-pass
// class the header above was written about.
test('электростатика: сетка, список Верле и полный перебор дают ОДНИ силы, и заряд их МЕНЯЕТ', async () => {
  const page = await gpuPage()
  const consoleWarnings: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'warn' || msg.type() === 'error') consoleWarnings.push(`${msg.type()}: ${msg.text()}`)
  })
  const r = await page.evaluate(async () => {
    const api = (window as any).api
    const start = { C: 600, O: 100, H: 500, M: 20, W: 0 }
    const mk = (charge: boolean) =>
      api.createSoup({
        box: [16, 16, 16],
        seed: 7,
        kT: 1.1,
        clay: false,
        start,
        // pH 7 with the file's own intrinsic pKa 4.9 puts alpha at 0.992 -- i.e. essentially every
        // head charged, which is the arm where the term is largest and therefore the arm a wiring bug
        // is most visible in.
        electrostatics: charge ? { enabled: true, pH: 7 } : undefined,
      })
    const charged = await mk(true)
    await charged.step(200)
    const a = await charged.forces()
    const b = await charged.forcesBruteForce()
    let max = 0
    let sumAbs = 0
    for (let i = 0; i < a.length; i++) {
      max = Math.max(max, Math.abs(a[i] - b[i]))
      sumAbs += Math.abs(b[i])
    }
    const q = await charged.charges()
    let chargedBeads = 0
    for (let i = 0; i < q.length; i++) if (q[i] !== 0) chargedBeads++
    const esInfo = charged.electrostatics()

    // The SAME positions, the SAME step count, charge off: the difference between the two force
    // fields is the electrostatic term itself.
    const neutral = await mk(false)
    await neutral.step(200)
    const c = await neutral.forces()
    let maxVsNeutral = 0
    for (let i = 0; i < a.length; i++) maxVsNeutral = Math.max(maxVsNeutral, Math.abs(a[i] - c[i]))
    charged.dispose()
    neutral.dispose()
    return { maxDiff: max, meanAbsRef: sumAbs / a.length, chargedBeads, heads: 100, maxVsNeutral, esInfo }
  })
  expect(consoleWarnings, `браузер сообщил об ошибке/предупреждении GPU во время теста:\n${consoleWarnings.join('\n')}`).toEqual([])
  console.log(
    `SOUP-FORCES-ES maxDiff(сетка+Верле против перебора)=${r.maxDiff.toExponential(4)} meanAbsRef=${r.meanAbsRef.toFixed(4)} ` +
      `заряженных_голов=${r.chargedBeads}/${r.heads} maxDiff(заряд против нейтрали)=${r.maxVsNeutral.toFixed(4)} ` +
      `A=${(r.esInfo as any).coeffA.toFixed(6)} kappa=${(r.esInfo as any).kappa.toFixed(6)} rc=${(r.esInfo as any).cutoff.toFixed(7)}`,
  )
  expect(r.maxDiff).toBeLessThan(1e-2)
  // The term is real: charge changed the force field by far more than the float32 agreement tolerance.
  expect(r.maxVsNeutral).toBeGreaterThan(1e-2)
  // And it was actually switched on: essentially every head is charged at pH 7 / pKa 4.9.
  expect(r.chargedBeads).toBeGreaterThan(80)
})
