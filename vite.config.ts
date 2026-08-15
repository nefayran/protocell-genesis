import { defineConfig } from 'vitest/config'

export default defineConfig({
  root: '.',
  server: { port: 0 },
  test: { environment: 'node', testTimeout: 120_000, hookTimeout: 60_000 },
})
