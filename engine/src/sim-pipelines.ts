// Compute-pipeline creation and cache for engine/src/sim.ts — split out by responsibility
// (file-size rule, root CLAUDE.md). Moved verbatim: same shader modules, same entry points, same
// device-keyed cache.

import forcesWgsl from '../wgsl/forces.wgsl?raw'
import neighborWgsl from '../wgsl/neighbor.wgsl?raw'
import integrateWgsl from '../wgsl/integrate.wgsl?raw'

export interface Pipelines {
  device: GPUDevice
  forceGrid: GPUComputePipeline
  forceBrute: GPUComputePipeline
  clearCounts: GPUComputePipeline
  count: GPUComputePipeline
  prefix: GPUComputePipeline
  fill: GPUComputePipeline
  kick: GPUComputePipeline
  drift: GPUComputePipeline
  wrap: GPUComputePipeline
  thermostat: GPUComputePipeline
}

let cached: Pipelines | undefined

export function getPipelines(device: GPUDevice): Pipelines {
  if (cached && cached.device === device) return cached
  const forceModule = device.createShaderModule({ code: forcesWgsl })
  const neighborModule = device.createShaderModule({ code: neighborWgsl })
  const integrateModule = device.createShaderModule({ code: integrateWgsl })
  const cp = (module: GPUShaderModule, entryPoint: string) =>
    device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint } })
  cached = {
    device,
    forceGrid: cp(forceModule, 'force_main'),
    forceBrute: cp(forceModule, 'force_brute_main'),
    clearCounts: cp(neighborModule, 'clear_counts_main'),
    count: cp(neighborModule, 'count_main'),
    prefix: cp(neighborModule, 'prefix_main'),
    fill: cp(neighborModule, 'fill_main'),
    kick: cp(integrateModule, 'kick_main'),
    drift: cp(integrateModule, 'drift_main'),
    wrap: cp(integrateModule, 'wrap_main'),
    thermostat: cp(integrateModule, 'thermostat_main'),
  }
  return cached
}
