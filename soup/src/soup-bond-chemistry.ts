// Bond chemistry dispatch: formation (grid-walk or Verlet-list variant) then breaking, as two
// separate, ordered dispatches within the same pass -- see soup/wgsl/bond-common.wgsl's header for
// why that ordering is what makes the i<j dedupe race-free. Split out of soup/src/sim.ts's
// createSoup (file-size rule in CLAUDE.md), moved verbatim; the actual claim-and-rollback/valence/
// adsorption/desorption/head-terminal logic all lives in soup/wgsl/bond-*.wgsl, never here -- this
// TS-side file is only the pipeline/bind-group dispatch that decides WHICH bond kernel entry point
// runs this step, not what it computes.

import type { SoupRuntime } from './soup-runtime'

export function encodeBondFormList(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  const { pipe, bind } = rt
  pass.setPipeline(pipe.bondFormList)
  pass.setBindGroup(0, bind.bondFormListGroup0)
  pass.setBindGroup(1, bind.bondFormListGroup1)
  pass.setBindGroup(2, bind.bondFormListGroup2)
  pass.dispatchWorkgroups(rt.wgN)
}

/** Bond Monte Carlo: formation then breaking, on the freshly rebuilt grid/positions (or the
 * current Verlet list). Called only when this step is scheduled to attempt bonds at all
 * (soup/src/soup-integrate.ts's own bondAttemptInterval gate) -- soup/wgsl/bond-common.wgsl's
 * header explains why formation/breaking never run concurrently with each other. */
export function encodeBondChemistry(rt: SoupRuntime, pass: GPUComputePassEncoder): void {
  const { pipe, bind } = rt
  if (rt.verlet.enabled) {
    encodeBondFormList(rt, pass)
  } else {
    pass.setPipeline(pipe.bondForm)
    pass.setBindGroup(0, bind.bondFormGroup0)
    pass.setBindGroup(1, bind.bondFormGroup1)
    pass.setBindGroup(2, bind.bondFormGroup2)
    pass.dispatchWorkgroups(rt.wgN)
  }

  pass.setPipeline(pipe.bondBreak)
  pass.setBindGroup(1, bind.bondBreakGroup1)
  pass.setBindGroup(2, bind.bondBreakGroup2)
  pass.dispatchWorkgroups(rt.wgN)
}
