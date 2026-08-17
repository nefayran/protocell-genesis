export interface Gpu {
  device: GPUDevice
  adapterInfo: { vendor: string; architecture: string }
}

let gpuPromise: Promise<Gpu> | undefined

export async function getGpu(): Promise<Gpu> {
  // Memoize at module level: all callers share one device.
  // Concurrent calls return the same in-flight promise to avoid racing two requestDevice() calls.
  if (gpuPromise) return gpuPromise

  gpuPromise = (async () => {
    if (!navigator.gpu) throw new Error('navigator.gpu отсутствует')
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('адаптер WebGPU не выдан')
    // Surface growth / adsorption (adsorption-report.md): the soup's bond-form kernels
    // (soup/wgsl/bond.wgsl) grew past the WebGPU DEFAULT per-stage storage-buffer limit (8) once
    // centerHeldSteps/desorbEvents joined pos2/cellStart/cellIdx/bondSlots/bondEvents/bondRng/
    // posSortedRO/centerLink/verletList/verletCount -- a device created with no requiredLimits
    // silently caps every pipeline at the default and 'auto' layout creation fails with only a
    // console warning (no thrown JS error), leaving every affected compute pass a no-op (0 bonds
    // ever formed) -- exactly the failure mode tests/soup-forces.test.ts's own header already
    // warns this codebase is vulnerable to. Requesting the ADAPTER's OWN reported maximum (not a
    // hardcoded number -- adapters vary) rather than a fixed limit keeps this portable across
    // hardware while still comfortably covering every kernel in this project.
    // The SIZE limits matter for the same reason the COUNT limit does, and they bite at a
    // measured particle count rather than at a kernel edit: the Verlet neighbour list is a flat
    // N * listCapacity * 4 bytes, so at listCapacity 1000 it reaches 113.6 MB at 28 400 particles
    // (runs) and 145.6 MB at 36 400 (silently does nothing) — the default
    // maxStorageBufferBindingSize is 128 MiB. As with the count limit, a device created without
    // asking for more caps every pipeline at the default and the failure is silent. Ask the
    // adapter for its own reported maxima rather than a hardcoded number.
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    })
    const info = adapter.info ?? ({} as GPUAdapterInfo)
    return {
      device,
      adapterInfo: { vendor: info.vendor ?? '', architecture: info.architecture ?? '' },
    }
  })()

  return gpuPromise
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
