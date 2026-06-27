// Pure waveform → braille rendering. Each terminal cell is a 2×4 dot grid
// (Unicode U+2800–U+28FF), so `w` chars wide × `h` chars tall gives a
// 2w × 4h dot canvas — enough resolution for a clean scope trace in a
// terminal. No I/O here, so it's unit-testable headlessly.

// Braille dot bit per (col 0..1, row 0..3):  col-major layout of the standard
//   (1,4)(2,5)(3,6)(7,8) ordering →
const DOT = [
  [0x01, 0x08],
  [0x02, 0x10],
  [0x04, 0x20],
  [0x40, 0x80],
]

/**
 * Render `samples` (each in [-1, 1]) as `h` rows of `w` braille chars.
 * The signal is drawn as a connected line: for each dot-column we map a
 * sample to a dot-row and fill the vertical span to the previous column so
 * steep edges stay continuous. Out-of-range samples clamp to the band.
 */
export function renderWaveform(samples: number[], w: number, h: number): string[] {
  const DW = w * 2 // dot columns
  const DH = h * 4 // dot rows
  const cells = new Uint8Array(w * h) // braille bitmask per char cell

  const yAt = (v: number): number => {
    const c = v < -1 ? -1 : v > 1 ? 1 : v // clamp
    // v=+1 → top (row 0), v=-1 → bottom (row DH-1)
    return Math.round(((1 - c) / 2) * (DH - 1))
  }
  const plot = (dx: number, dy: number) => {
    if (dx < 0 || dx >= DW || dy < 0 || dy >= DH) return
    const cx = dx >> 1, cy = dy >> 2
    cells[cy * w + cx] |= DOT[dy & 3][dx & 1]
  }

  const n = samples.length
  let prevY: number | null = null
  for (let dx = 0; dx < DW; dx++) {
    // nearest sample for this dot-column
    const s = n <= 1 ? (samples[0] ?? 0) : samples[Math.min(n - 1, Math.round((dx / (DW - 1)) * (n - 1)))]
    const y = yAt(s)
    if (prevY === null) prevY = y
    const lo = Math.min(prevY, y), hi = Math.max(prevY, y)
    for (let dy = lo; dy <= hi; dy++) plot(dx, dy)
    prevY = y
  }

  const rows: string[] = []
  for (let cy = 0; cy < h; cy++) {
    let line = ''
    for (let cx = 0; cx < w; cx++) line += String.fromCharCode(0x2800 + cells[cy * w + cx])
    rows.push(line)
  }
  return rows
}
