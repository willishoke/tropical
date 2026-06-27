// Headless smoke test of the scope data pipeline (no Ink, no terminal):
// spawn engine → build the two-voice session → render_window → braille → stdout.
// `bun tui/test-frame.ts` from anywhere (repoRoot is resolved from this file).
import { resolve } from 'node:path'
import { EngineClient, setupTwoVoice } from './client'
import { renderWaveform } from './braille'

const repoRoot = resolve(import.meta.dir, '..')
const c = new EngineClient({ repoRoot, onError: (e) => { console.error('engine error:', e.message); process.exit(1) } })

const slots = await setupTwoVoice(c, [220, 330])
console.log('session slots:', slots.join(', '))

const WINDOW = 800, w = 60, h = 4
const res = await c.call('render_window', { start: 0, count: WINDOW, slots })
const [ch1, ch2] = res.values as number[][]

const band = (label: string, samples: number[]) => {
  console.log(`\n${label}  (peak ${Math.max(...samples.map(Math.abs)).toFixed(3)})`)
  for (const row of renderWaveform(samples, w, h)) console.log('  ' + row)
}
band('CH1  220 Hz', ch1)
band('CH2  330 Hz', ch2)

const pos = await c.call('playback_position', {})
console.log('\nplayback_position:', pos.position)
console.log('pipeline OK')
c.kill()
process.exit(0)
