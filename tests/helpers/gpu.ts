import { createServer, type ViteDevServer } from 'vite'
import puppeteer, { type Browser, type Page } from 'puppeteer-core'

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

let server: ViteDevServer | null = null
let browser: Browser | null = null

export async function gpuPage(): Promise<Page> {
  if (!server) {
    server = await createServer({ server: { port: 0 } })
    await server.listen()
  }
  if (!browser) {
    browser = await puppeteer.launch({
      executablePath: CHROME,
      headless: true,
      args: ['--enable-unsafe-webgpu', '--no-sandbox'],
      // Puppeteer's CDP protocolTimeout defaults to 180_000ms and applies to every command,
      // including page.evaluate() — Task 6's self-assembly test runs its whole step loop inside
      // ONE evaluate() call and measured ~300s there (1_000_000 steps), so the default ceiling
      // would kill a legitimate run with a generic "Runtime.callFunctionOn timed out" unrelated to
      // vitest's own testTimeout.
      //
      // This must stay FINITE, not 0 (0 disables it entirely): protocolTimeout is also the only
      // thing that bounds a genuinely HUNG evaluate() — one that never resolves at all, as opposed
      // to one that is merely slow but progressing. Without it, a true hang is bounded only by
      // vitest's own per-test timeout racing against `browser.close()` inside the 60s `afterAll`
      // hook, and if the browser process itself is what's wedged, that close() can hang too,
      // taking the whole suite down instead of failing one test.
      //
      // 1_800_000ms (30 min): 6x the longest legitimate evaluate() measured for the committed
      // suite (self-assembly, ~300s) and 3x the longest ever measured during this task's
      // diagnostics (a 2_000_000-step exploratory run, ~596s) — comfortable headroom for a slower
      // CI machine without being long enough to leave a real hang undetected for the length of a
      // workday.
      protocolTimeout: 1_800_000,
    })
  }
  const base = server.resolvedUrls!.local[0]
  const page = await browser.newPage()
  page.on('pageerror', (e) => console.error('page error:', e.message))
  await page.goto(new URL('tests/runner.html', base).href, { waitUntil: 'load' })
  await page.waitForFunction('window.__ready === true')
  return page
}

export async function shutdownGpu(): Promise<void> {
  await browser?.close()
  await server?.close()
  browser = null
  server = null
}
