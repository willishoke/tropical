// Headless smoke test: in-patch morph sweep + per-channel trigger lock + braille.
// render_window is given a warmup lead so the session's per-wire unit delays are
// primed before the displayed window (a cold start gives garbage samples).
import { resolve } from 'node:path'
import { EngineClient, setupMorphScope } from './client'
import { renderWaveform } from './braille'
import { findTrigger } from './trigger'

const repoRoot = resolve(import.meta.dir, '..')
const c = new EngineClient({ repoRoot, onError: (e) => { console.error('engine error:', e.message); process.exit(1) } })

const slots = await setupMorphScope(c, [220, 330], 0.15)
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
// LFO @ 0.15 Hz: quarter→sine, zero→blend, 3/4→saw
await at(73500, 'morph≈1 · sine')
await at(0, 'morph≈0.5 · blend')
await at(220500, 'morph≈0 · saw')

const SEARCH = 220
function aligned(s: number[]) {
  const t = Math.max(0, findTrigger(s, 0, SEARCH))
  return { t, edge: t > 0 && s[t - 1] < 0 && s[t] >= 0, head: s.slice(t, t + 5) }
}
async function frameAt(start: number) {
  const r = await rw(start, SEARCH + 200)
  return { ch1: aligned(r[0]), ch2: aligned(r[1]) }
}
const A = await frameAt(73500), B = await frameAt(73650) // Δ≈0.75 CH1 period / 1.1 CH2 period
const match = (h: number[], g: number[]) => Math.max(...h.map((x, i) => Math.abs(x - g[i]))) < 0.05
console.log('\nper-channel trigger lock (two nearby positions):')
console.log(`  CH1: edges ${A.ch1.edge}/${B.ch1.edge}  aligned-stationary ${match(A.ch1.head, B.ch1.head)}`)
console.log(`  CH2: edges ${A.ch2.edge}/${B.ch2.edge}  aligned-stationary ${match(A.ch2.head, B.ch2.head)} (would FLIP under a shared trigger)`)
console.log('\npipeline OK')
c.kill()
process.exit(0)
