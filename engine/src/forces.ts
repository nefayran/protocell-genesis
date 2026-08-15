import forcesWgsl from '../wgsl/forces.wgsl?raw'
import { getGpu, readBack, storageBuffer } from './gpu'
import { loadParams, paramsToUniform } from './params'

const KINDS = { wca: 0, fene: 1, bend: 2, attr: 3 } as const

export async function probeForces(kind: keyof typeof KINDS, b: number, radii: number[]): Promise<number[]> {
  const { device } = await getGpu()
  const module = device.createShaderModule({ code: forcesWgsl })
  const uni = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(uni, 0, paramsToUniform(loadParams()))
  const probe = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
  device.queue.writeBuffer(probe, 0, new Float32Array([KINDS[kind], b, 0, 0]))
  const rin = storageBuffer(device, new Float32Array(radii))
  const rout = storageBuffer(device, new Float32Array(radii.length))
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'probe_main' } })
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
  return Array.from(await readBack(device, rout, radii.length * 4))
}
