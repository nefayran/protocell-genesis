import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'

const page = (path: string) => fileURLToPath(new URL(path, import.meta.url))

export default defineConfig({
  root: '.',
  // GitHub Pages serves the project under /protocell-genesis/; the Pages workflow sets PAGES_BASE.
  base: process.env.PAGES_BASE ?? '/',
  server: { port: 0 },
  build: {
    rollupOptions: {
      input: {
        main: page('./index.html'),
        gallery: page('./viewer/gallery.html'),
        run: page('./viewer/run.html'),
      },
    },
  },
  test: { environment: 'node', testTimeout: 120_000, hookTimeout: 60_000 },
})
