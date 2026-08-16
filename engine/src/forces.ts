import forcesWgsl from '../wgsl/forces.wgsl?raw'
import { getGpu, readBack, storageBuffer } from './gpu'
import { loadParams, paramsToUniform } from './params'

const KINDS = { wca: 0, fene: 1, bend: 2, attr: 3 } as const

// One pipeline serves all kinds: the kind is a runtime value carried in the `probe`
// uniform, not something baked into the shader module or the bind group layout. Cached
// at module level (keyed by device, in case getGpu's memoized device is ever replaced)
// so hundreds of probeForces() calls per measurement run don't each recompile a pipeline.
let cached: { device: GPUDevice; pipeline: GPUComputePipeline } | undefined

function getPipeline(device: GPUDevice): GPUComputePipeline {
  if (cached && cached.device === device) return cached.pipeline
  const module = device.createShaderModule({ code: forcesWgsl })
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'probe_main' } })
  cached = { device, pipeline }
  return pipeline
}

export async function probeForces(kind: keyof typeof KINDS, b: number, radii: number[]): Promise<number[]> {
  const { device } = await getGpu()
  const pipeline = getPipeline(device)
  const uni = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(uni, 0, paramsToUniform(loadParams()))
  const probe = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(probe, 0, new Float32Array([KINDS[kind], b, 0, 0]))
  const rin = storageBuffer(device, new Float32Array(radii))
  const rout = storageBuffer(device, new Float32Array(radii.length))
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } },
      { binding: 1, resource: { buffer: rin } },
      { binding: 2, resource: { buffer: rout } },
      { binding: 3, resource: { buffer: probe } },
    ],
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bind)
  pass.dispatchWorkgroups(Math.ceil(radii.length / 64))
  pass.end()
  device.queue.submit([enc.finish()])
  try {
    return Array.from(await readBack(device, rout, radii.length * 4))
  } finally {
    uni.destroy()
    probe.destroy()
    rin.destroy()
    rout.destroy()
  }
}
