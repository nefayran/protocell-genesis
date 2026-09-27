// Renders the README images from the snapshot gallery (viewer/gallery.html): the three-panel hero and
// the assembly animation. Every image is a screenshot of real checkpoint data, never a mock-up.
//
// Needs Google Chrome (CHROME overrides the macOS default path) and ffmpeg on PATH.
// Run: node docs/media/render.mjs
import { execFileSync } from 'node:child_process'
import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createServer } from 'vite'
import puppeteer from 'puppeteer-core'

const CHROME = process.env.CHROME ?? '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
const OUT = 'docs/media'

const server = await createServer({ server: { port: 0 }, logLevel: 'error' })
await server.listen()
const base = server.resolvedUrls.local[0]
const browser = await puppeteer.launch({ executablePath: CHROME, headless: true, args: ['--no-sandbox'] })

async function shoot(query, width, height, scale = 2) {
  const page = await browser.newPage()
  await page.setViewport({ width, height, deviceScaleFactor: scale })
  const errors = []
  page.on('pageerror', (e) => errors.push(e.message))
  await page.goto(new URL(`viewer/gallery.html?shot=1&rotate=0&${query}`, base).href, { waitUntil: 'load' })
  await page.waitForFunction('window.__galleryReady === true', { timeout: 60_000 })
  if (errors.length) throw new Error(`gallery errors for ${query}: ${errors.join('; ')}`)
  const png = await page.screenshot({ type: 'png' })
  await page.close()
  return png
}

const dataUrl = (png) => `data:image/png;base64,${Buffer.from(png).toString('base64')}`

// --- hero: the start of the big periodic campaign, its end, and the end of the finite-parcel run.
const index = JSON.parse(readFileSync('viewer/gallery-data/index.json', 'utf8')).snapshots
const byId = Object.fromEntries(index.map((s) => [s.id, s]))
const fmt = (x) => x.toLocaleString('en-US')
const share = (s) => ((100 * s.amphiphilesInLargest) / s.amphiphiles).toFixed(2)
const early = byId['periodic-3000']
const late = byId['periodic-148200']
const confined = byId['confined-225200']
const panels = [
  {
    png: await shoot('snap=periodic-3000&az=38&el=20&zoom=1.04', 640, 560),
    title: `Periodic box, step ${fmt(early.step)}`,
    text: `${fmt(early.amphiphiles)} amphiphiles in ${fmt(early.aggregates)} small clusters`,
  },
  {
    png: await shoot('snap=periodic-148200&az=38&el=20&zoom=1.04', 640, 560),
    title: `Periodic box, step ${fmt(late.step)}`,
    text: `one aggregate holds ${share(late)}%, wraps ${late.wrappingAxes} of 3 axes, ${late.encapsulatedWater} water beads inside`,
  },
  {
    png: await shoot('snap=confined-225200&az=38&el=20&zoom=1.04', 640, 560),
    title: `Finite water parcel, step ${fmt(confined.step)}`,
    text: `one aggregate, wraps ${confined.wrappingAxes} of 3 axes, ${confined.encapsulatedWater} water beads inside`,
  },
]

const heroHtml = `<!doctype html><meta charset="utf-8"><style>
  body { margin: 0; background: #05080d; color: #dfe7ef; font: 22px/1.4 -apple-system, "Segoe UI", system-ui, sans-serif; }
  .wrap { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; padding: 0 0 26px; width: 1920px; }
  figure { margin: 0; }
  img { width: 640px; height: 560px; display: block; }
  figcaption { padding: 0 36px; }
  b { display: block; font-weight: 650; font-size: 26px; margin-bottom: 4px; }
  span { color: #9fb0c1; }
  .legend { display: flex; gap: 30px; justify-content: center; color: #9fb0c1; font-size: 20px; padding: 0 0 30px; width: 1920px; }
  .dot { display: inline-block; width: 15px; height: 15px; border-radius: 50%; margin-right: 9px; vertical-align: -1px; }
</style><div class="wrap">${panels
  .map((p) => `<figure><img src="${dataUrl(p.png)}"><figcaption><b>${p.title}</b><span>${p.text}</span></figcaption></figure>`)
  .join('')}</div>
<div class="legend"><span><i class="dot" style="background:#f2dfb8"></i>amphiphile head</span><span><i class="dot" style="background:#2f7f86"></i>amphiphile tail</span><span><i class="dot" style="background:#8795a3"></i>unreacted monomer</span><span>coarse-grained beads from the campaign checkpoints</span></div>`

async function composite(html, width, file) {
  const page = await browser.newPage()
  await page.setViewport({ width, height: 800, deviceScaleFactor: 1 })
  await page.setContent(html, { waitUntil: 'load' })
  const box = await page.evaluate(() => ({ width: document.body.scrollWidth, height: document.body.scrollHeight }))
  await page.screenshot({ path: join(OUT, file), clip: { x: 0, y: 0, ...box } })
  await page.close()
}
await composite(heroHtml, 1920, 'hero.png')
// The same three panels stacked, for narrow screens, where three columns would shrink the captions away.
await composite(
  heroHtml
    .replace('grid-template-columns: repeat(3, 1fr); gap: 0; padding: 0 0 26px; width: 1920px;', 'grid-template-columns: 1fr; gap: 26px; padding: 0 0 26px; width: 640px;')
    .replace('justify-content: center; color: #9fb0c1; font-size: 20px; padding: 0 0 30px; width: 1920px;', 'flex-wrap: wrap; row-gap: 6px; justify-content: flex-start; color: #9fb0c1; font-size: 20px; padding: 0 36px 30px; width: 640px; box-sizing: border-box;'),
  640,
  'hero-stacked.png',
)

// --- animation: the periodic campaign snapshot by snapshot while the camera turns.
const frames = join(tmpdir(), `protocell-frames-${process.pid}`)
mkdirSync(frames, { recursive: true })
const sequence = ['periodic-3000', 'periodic-11000', 'periodic-23400', 'periodic-41100', 'periodic-148200']
const perSnapshot = 12
let f = 0
for (const id of sequence) {
  for (let k = 0; k < perSnapshot; k++) {
    const az = 20 + (f * 90) / (sequence.length * perSnapshot)
    writeFileSync(join(frames, `f${String(f).padStart(4, '0')}.png`), await shoot(`snap=${id}&az=${az.toFixed(2)}&el=22&label=1&zoom=0.9`, 440, 440, 1))
    f++
  }
}
const palette = join(frames, 'palette.png')
const input = ['-loglevel', 'error', '-y', '-framerate', '8', '-i', join(frames, 'f%04d.png')]
execFileSync('ffmpeg', [...input, '-vf', 'palettegen=max_colors=96:stats_mode=full', palette])
execFileSync('ffmpeg', [...input, '-i', palette, '-lavfi', 'paletteuse=dither=bayer:bayer_scale=3', join(OUT, 'assembly.gif')])
// The same frames as H.264: a smaller download for the project page.
execFileSync('ffmpeg', [...input, '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', '-movflags', '+faststart', join(OUT, 'assembly.mp4')])
rmSync(frames, { recursive: true, force: true })

await browser.close()
await server.close()
console.log('wrote docs/media/hero.png, hero-stacked.png, assembly.gif and assembly.mp4')
