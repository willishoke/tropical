/**
 * energy_floor.ts — shared per-patch energy floor for the web differential suites.
 *
 * The differentials prove *agreement* between backends, but two backends that
 * both render silence agree perfectly — and "graceful exclusion" makes an
 * all-zero kernel a legal compile result. Every case must therefore also show
 * the rendered buffer carries real energy, or the agreement is vacuous.
 */

// Calibrated 2026-07-07 against every case in both suites: the quietest patch
// observed was web/dist clock-chord at RMS 1.14e-2 (all others 1.75e-2..1.9e2).
// The floor sits two orders of magnitude below that — loose enough for any
// legitimately quiet patch, but far above numerical zero, so an all-zero
// (gracefully excluded) kernel can never clear it.
export const RMS_FLOOR = 1e-4

export function rms(buf: Float64Array): number {
  let s = 0
  for (let i = 0; i < buf.length; i++) s += buf[i]! * buf[i]!
  return Math.sqrt(s / buf.length)
}

export function expectAudible(name: string, buf: Float64Array): void {
  const r = rms(buf)
  if (!(r > RMS_FLOOR)) {
    throw new Error(
      `${name}: silent agreement — RMS ${r.toExponential(3)} is below the floor ` +
      `${RMS_FLOOR}; both backends rendered no energy; graceful exclusion may ` +
      `have eaten the patch`,
    )
  }
}
