// Two sliders: kT/epsilon (bounded by data/params.json's kTRange) and w_c (bounded by the
// viewer-only range main.ts passes in). Neither slider recreates the system -- see setLiveParams
// in engine/src/sim.ts -- this module only owns the DOM: reading user input and painting the
// current value back, never touching engine state itself.

export interface PanelHandles {
  /** Moves the kT slider to v and repaints its label, WITHOUT firing onKT — for a caller (e.g.
   * window.viewer.setKT) that has already applied v to the engine and just needs the panel to
   * reflect it. */
  setKT(v: number): void
  setWc(v: number): void
}

export interface PanelOpts {
  kTRange: [number, number]
  wcRange: [number, number]
  initialKT: number
  initialWc: number
  onKT: (v: number) => void
  onWc: (v: number) => void
}

export function createPanel(opts: PanelOpts): PanelHandles {
  const ktSlider = document.getElementById('kt-slider') as HTMLInputElement
  const wcSlider = document.getElementById('wc-slider') as HTMLInputElement
  const ktValue = document.getElementById('kt-value') as HTMLElement
  const wcValue = document.getElementById('wc-value') as HTMLElement

  ktSlider.min = String(opts.kTRange[0])
  ktSlider.max = String(opts.kTRange[1])
  ktSlider.value = String(opts.initialKT)
  wcSlider.min = String(opts.wcRange[0])
  wcSlider.max = String(opts.wcRange[1])
  wcSlider.value = String(opts.initialWc)

  function paint(): void {
    ktValue.textContent = Number(ktSlider.value).toFixed(3)
    wcValue.textContent = Number(wcSlider.value).toFixed(2)
  }
  paint()

  ktSlider.addEventListener('input', () => opts.onKT(Number(ktSlider.value)))
  wcSlider.addEventListener('input', () => opts.onWc(Number(wcSlider.value)))

  return {
    setKT(v: number) {
      ktSlider.value = String(v)
      paint()
    },
    setWc(v: number) {
      wcSlider.value = String(v)
      paint()
    },
  }
}
