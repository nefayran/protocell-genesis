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
