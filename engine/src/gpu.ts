export interface Gpu {
  device: GPUDevice
  adapterInfo: { vendor: string; architecture: string }
}

export async function getGpu(): Promise<Gpu> {
  if (!navigator.gpu) throw new Error('navigator.gpu отсутствует')
  const adapter = await navigator.gpu.requestAdapter()
  if (!adapter) throw new Error('адаптер WebGPU не выдан')
  const device = await adapter.requestDevice()
  const info = adapter.info ?? ({} as GPUAdapterInfo)
  return {
    device,
    adapterInfo: { vendor: info.vendor ?? '', architecture: info.architecture ?? '' },
  }
}

export async function readBack(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<Float32Array> {
  const dst = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ })
  const enc = device.createCommandEncoder()
  enc.copyBufferToBuffer(src, 0, dst, 0, bytes)
  device.queue.submit([enc.finish()])
  await dst.mapAsync(GPUMapMode.READ)
  const out = new Float32Array(dst.getMappedRange().slice(0))
  dst.unmap()
  dst.destroy()
  return out
}

export function storageBuffer(device: GPUDevice, data: Float32Array): GPUBuffer {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(buf, 0, data)
  return buf
}
