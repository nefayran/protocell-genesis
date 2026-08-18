// Measurement/readback helpers for engine/src/sim.ts, plus the live-params retuning setter that
// shares their "read/write runtime state without recreating the system" nature — split out by
// responsibility (file-size rule, root CLAUDE.md). Moved verbatim from the original sim.ts.

import { readBack } from './gpu'
import { paramsToUniform } from './params'
import { encodeGridRebuild } from './sim-integrate'
import { encodeForceGrid } from './sim-forces'
import type { EngineRuntime } from './sim-runtime'

export async function positions(rt: EngineRuntime): Promise<Float32Array> {
  return readBack(rt.device, rt.posBuf, rt.N * 16)
}

export async function kineticEnergyPerDof(rt: EngineRuntime): Promise<number> {
  const v = await readBack(rt.device, rt.velBuf, rt.N * 16)
  let sumSq = 0
  for (let i = 0; i < rt.N; i++) {
    const vx = v[i * 4], vy = v[i * 4 + 1], vz = v[i * 4 + 2]
    sumSq += vx * vx + vy * vy + vz * vz
  }
  return sumSq / (3 * rt.N)
}

export async function totalEnergy(rt: EngineRuntime): Promise<number> {
  const enc = rt.device.createCommandEncoder()
  const pass = enc.beginComputePass()
  encodeGridRebuild(rt, pass)
  encodeForceGrid(rt, pass)
  pass.end()
  rt.device.queue.submit([enc.finish()])
  const [v, u] = await Promise.all([readBack(rt.device, rt.velBuf, rt.N * 16), readBack(rt.device, rt.potentialBuf, rt.N * 4)])
  let kinetic = 0
  for (let i = 0; i < rt.N; i++) {
    const vx = v[i * 4], vy = v[i * 4 + 1], vz = v[i * 4 + 2]
    kinetic += 0.5 * (vx * vx + vy * vy + vz * vz)
  }
  let potential = 0
  for (let i = 0; i < rt.N; i++) potential += u[i]
  return kinetic + potential
}

/** Validates and THROWS rather than clamping — see System.setLiveParams's doc comment
 * (sim-types.ts) for why. Same style as createSystem's own gridInvariantsHold/discriminant
 * guards. */
export function setLiveParams(rt: EngineRuntime, overrides: { kT?: number; wc?: number }): void {
  if (overrides.wc !== undefined && overrides.wc > rt.builtForWc) {
    throw new Error(
      `setLiveParams: wc=${overrides.wc} превышает значение, под которое построена сетка соседей ` +
        `(builtForWc=${rt.builtForWc}, cellSize=${rt.cellSize.toFixed(4)}) — пересоздайте систему с ` +
        `опцией maxWc>=${overrides.wc}, иначе силы будут молча теряться`,
    )
  }
  if (overrides.kT !== undefined && (overrides.kT < rt.p.kTRange[0] || overrides.kT > rt.p.kTRange[1])) {
    throw new Error(
      `setLiveParams: kT=${overrides.kT} вне data/params.json kTRange=[${rt.p.kTRange[0]},${rt.p.kTRange[1]}]`,
    )
  }
  rt.livep = {
    ...rt.livep,
    thermostat: overrides.kT !== undefined ? { ...rt.livep.thermostat, kT: overrides.kT } : rt.livep.thermostat,
    attraction: overrides.wc !== undefined ? { ...rt.livep.attraction, wc: overrides.wc } : rt.livep.attraction,
  }
  rt.device.queue.writeBuffer(rt.paramsUniform, 0, paramsToUniform(rt.livep))
}
