import { readdirSync, readFileSync } from 'node:fs'
import { configDefaults, defineConfig, mergeConfig } from 'vitest/config'
import base from './vite.config'

// Test files that drive a WebGPU page through tests/helpers/gpu.ts need Chrome and a GPU. The rest
// run on the CPU alone, and they are what CI runs on every push.
const gpuTests = readdirSync('tests')
  .filter((f) => f.endsWith('.test.ts'))
  .filter((f) => /helpers\/gpu|puppeteer/.test(readFileSync(`tests/${f}`, 'utf8')))
  .map((f) => `tests/${f}`)

export default mergeConfig(base, defineConfig({ test: { exclude: [...configDefaults.exclude, ...gpuTests] } }))
