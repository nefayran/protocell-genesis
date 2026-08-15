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
