import raw from '../../data/params.json'

export interface Params {
  source: string
  rank: string
  sigma: number
  epsilon: number
  beadSizes: { head_head: number; head_tail: number; tail_tail: number }
  fene: { k: number; rInf: number }
  bend: { k: number; r0: number }
  attraction: { wc: number }
  thermostat: { gamma: number; kT: number }
  integrator: { dt: number }
  kTRange: [number, number]
}

const REQUIRED = [
  'sigma', 'epsilon', 'beadSizes', 'fene', 'bend', 'attraction', 'thermostat', 'integrator',
] as const

export function loadParams(): Params {
  const p = raw as unknown as Params
  for (const key of REQUIRED) {
    if ((p as Record<string, unknown>)[key] === undefined) {
      throw new Error(`data/params.json: отсутствует поле ${key}`)
    }
  }
  return p
}

export function wcaCutoff(b: number): number {
  return Math.pow(2, 1 / 6) * b
}

/** Плоский массив для униформ-буфера WGSL. Порядок обязан совпадать со struct Params в forces.wgsl. */
export function paramsToUniform(p: Params): Float32Array {
  return new Float32Array([
    p.sigma, p.epsilon,
    p.beadSizes.head_head, p.beadSizes.head_tail, p.beadSizes.tail_tail,
    p.fene.k, p.fene.rInf,
    p.bend.k, p.bend.r0,
    p.attraction.wc,
    p.thermostat.kT, p.thermostat.gamma,
    p.integrator.dt,
    0, 0, 0,
  ])
}
