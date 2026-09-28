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
    if (!navigator.gpu) throw new Error('navigator.gpu is missing')
    const adapter = await navigator.gpu.requestAdapter()
    if (!adapter) throw new Error('no WebGPU adapter was provided')
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

export async function readBack(device: GPUDevice, src: GPUBuffer, bytes: number): Promise<Float32Array<ArrayBuffer>> {
  // Task 'loud-failure-and-liquid-water' (2026-08-20). WHY THIS CHECK EXISTS, measured: a buffer
  // allocated without COPY_SRC makes the copyBufferToBuffer below a validation error, which
  // INVALIDATES the command buffer, which makes the submit a no-op -- and then the freshly created
  // (therefore zero-initialised) `dst` is mapped and returned as if it were the real data. WebGPU
  // reports this only as a console warning, so from JS it is indistinguishable from "this buffer
  // genuinely contains zeros". It cost this project every RNG stream in every checkpoint ever
  // written (soup/src/soup-buffers.ts's bondRngBuf/thermoRngBuf were missing the flag; all 72
  // checkpoint files on disk carry bondRngB64/thermoRngB64 that are 100% zero bytes), and through
  // decodeCheckpointResume that fed every resumed run a thermostat seeded to 0 for EVERY particle,
  // i.e. identical Langevin noise on all N particles instead of independent noise. `usage` is a
  // readable attribute of GPUBuffer, so the whole failure class costs one bitwise test to convert
  // from silent zeros into a thrown error.
  if ((src.usage & GPUBufferUsage.COPY_SRC) === 0) {
    throw new Error(
      `readBack: the buffer was created without GPUBufferUsage.COPY_SRC (usage=${src.usage}) -- the copy would be ` +
        `a validation error, the submit is silently dropped, and the caller would get zeros instead of data`,
    )
  }
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
  // Nothing in this project allocates a SharedArrayBuffer, so every Float32Array here is backed by a
  // plain ArrayBuffer; the cast only narrows the lib type (ArrayBufferLike) that writeBuffer rejects.
  device.queue.writeBuffer(buf, 0, data as Float32Array<ArrayBuffer>)
  return buf
}
