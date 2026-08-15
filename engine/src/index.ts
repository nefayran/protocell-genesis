import { getGpu, readBack, storageBuffer } from './gpu'

export { probeForces } from './forces'
export { createSystem } from './sim'
export type { CreateSystemOpts, Layout, System } from './sim'

export async function gpuSmoke() {
  const { device, adapterInfo } = await getGpu()
  const module = device.createShaderModule({
    code: `@group(0) @binding(0) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(4) fn main(@builtin(global_invocation_id) i: vec3<u32>) {
  b[i.x] = b[i.x] * 2.0;
}`,
  })
  const buf = storageBuffer(device, new Float32Array([1, 2, 3, 4]))
  const pipeline = device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } })
  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: buf } }],
  })
  const enc = device.createCommandEncoder()
  const pass = enc.beginComputePass()
  pass.setPipeline(pipeline)
  pass.setBindGroup(0, bind)
  pass.dispatchWorkgroups(1)
  pass.end()
  device.queue.submit([enc.finish()])
  let doubled: number[]
  try {
    doubled = Array.from(await readBack(device, buf, 16))
  } finally {
    buf.destroy()
  }
  return { vendor: adapterInfo.vendor, architecture: adapterInfo.architecture, doubled }
}
