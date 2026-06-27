// Headless smoke test: in-patch morph sweep (rendered at different sample
// indices since the LFO makes morph a function of time) + trigger lock + braille.
import { resolve } from 'node:path'
import { EngineClient, setupMorphScope } from './client'
import { renderWaveform } from './braille'
import { findTrigger } from './trigger'

const repoRoot = resolve(import.meta.dir, '..')
const c = new EngineClient({ repoRoot, onError: (e) => { console.error('engine error:', e.message); process.exit(1) } })

const slots = await setupMorphScope(c, [220, 330], 0.15)
console.log('slots:', slots.join(', '))

const W = 60, H = 4, WINDOW = 600
async function at(start: number, label: string) {
  const ch1 = (await c.call('render_window', { start, count: WINDOW, slots })).values[0] as number[]
  console.log(`\n${label}  (start=${start}, peak ${Math.max(...ch1.map(Math.abs)).toFixed(3)})`)
  for (const row of renderWaveform(ch1, W, H)) console.log('  ' + row)
}
// LFO @ 0.15 Hz (period ≈ 294 k samples): quarter→sine, zero→blend, 3/4→saw
await at(73500, 'morph≈1 · sine')
await at(0, 'morph≈0.5 · blend')
await at(220500, 'morph≈0 · saw')

// trigger: nearby windows both lock to a rising crossing → stationary trace
const SEARCH = 220
async function locked(start: number) {
  const a = (await c.call('render_window', { start, count: WINDOW, slots })).values[0] as number[]
  const t = findTrigger(a, 0, SEARCH)
  return { t, edge: t > 0 && a[t - 1] < 0 && a[t] >= 0 }
}
const A = await locked(73500), B = await locked(73623)
console.log('\ntrigger lock:')
console.log(`  start=73500 trig@${A.t} risingEdge=${A.edge}`)
console.log(`  start=73623 trig@${B.t} risingEdge=${B.edge}`)
console.log('  both lock to a rising crossing → trace stationary:', A.edge && B.edge)
console.log('\npipeline OK')
c.kill()
process.exit(0)
