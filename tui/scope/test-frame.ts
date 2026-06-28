// Headless smoke test: in-patch morph sweep + phase-trigger lock + braille.
// render_window gets a warmup lead so the session's per-wire unit delays are
// primed before the displayed window.
import { resolve } from 'node:path'
import { EngineClient, setupMorphScope } from './client'
import { renderWaveform } from './braille'
import { findPhaseTrigger } from './trigger'

const repoRoot = resolve(import.meta.dir, '../..')
const c = new EngineClient({ repoRoot, onError: (e) => { console.error('engine error:', e.message); process.exit(1) } })

const FREQS = [220, 330], SR = 44100, SEARCH = 220
const slots = await setupMorphScope(c, FREQS as [number, number], 0.15)
console.log('slots:', slots.join(', '))

const WARMUP = 64
async function rw(start: number, count: number): Promise<number[][]> {
  const renderStart = Math.max(0, start - WARMUP)
  const lead = start - renderStart
  const r = (await c.call('render_window', { start: renderStart, count: lead + count, slots })).values as number[][]
  return r.map((ch) => ch.slice(lead))
}

const W = 60, H = 4, WINDOW = 600
async function at(start: number, label: string) {
  const ch1 = (await rw(start, WINDOW))[0]
  console.log(`\n${label}  (start=${start}, peak ${Math.max(...ch1.map(Math.abs)).toFixed(3)})`)
  for (const row of renderWaveform(ch1, W, H)) console.log('  ' + row)
}
await at(73500, 'morph≈1 · sine')
await at(0, 'morph≈0.5 · blend')
await at(220500, 'morph≈0 · saw')

// Mid-morph the blend has TWO rising zero-crossings per period, so a level
// trigger flips between them frame-to-frame (the jitter); the phase trigger,
// locked to the morph-independent phasor wrap, holds across a sweep of positions.
const TRANS = 147000 // morph≈0.5 region
const match = (h: number[], g: number[]) => Math.max(...h.map((x, i) => Math.abs(x - g[i]))) < 0.05
const offs = [0, 40, 80, 120, 160]
const frames: number[][][] = []
for (const o of offs) frames.push(await rw(TRANS + o, SEARCH + 200))
const headP = (i: number) => { const t = findPhaseTrigger(FREQS[0], SR, TRANS + offs[i], SEARCH); return frames[i][0].slice(t, t + 5) }
const allMatch = (h: (i: number) => number[]) => offs.every((_, i) => match(h(i), h(0)))
const span = frames[0][0].length
let rc = 0; for (let i = 1; i < span; i++) if (frames[0][0][i - 1] < 0 && frames[0][0][i] >= 0) rc++
const periods = span / (SR / FREQS[0])
console.log('\nat the morph transition:')
console.log(`  rising zero-crossings: ${(rc / periods).toFixed(1)}/period  (>1 ⇒ a level trigger flips → the jitter)`)
console.log('  phase-trigger all-stationary:', allMatch(headP), '(the fix — solid)')
console.log('\npipeline OK')
c.kill()
process.exit(0)
