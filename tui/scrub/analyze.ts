// Render the patch and Fourier-analyze the output to surface artifact sources.
// Run: bun tui/analyze.ts
import { spawnSync } from 'node:child_process'
import { writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { buildProgram, PARAMS } from './patch'

const root = resolve(import.meta.dir, '..')
const DIFFCLI = './lean/.lake/build/bin/frontend' // not used; use diffcli below
const DC = './lean/.lake/build/bin/diffcli'

function render(patch: any, label: string, frames = 256): Float64Array {
  writeFileSync('/tmp/rr_patch.json', JSON.stringify(patch))
  const c = spawnSync(DC, ['compile', '/tmp/rr_patch.json'], { cwd: root, maxBuffer: 1 << 28 })
  if (c.status !== 0) throw new Error(`${label} compile: ${c.stderr}`)
  writeFileSync('/tmp/rr_plan.json', c.stdout)
  const r = spawnSync(DC, ['render-bytes', '/tmp/rr_plan.json', '--frames', String(frames), '--buffer', '256'],
    { cwd: root, maxBuffer: 1 << 28 })
  if (r.status !== 0) throw new Error(`${label} render: ${r.stderr}`)
  const buf = r.stdout as Buffer
  const n = buf.length / 8
  const out = new Float64Array(n)
  for (let i = 0; i < n; i++) out[i] = buf.readDoubleLE(i * 8)
  return out
}

// in-place iterative radix-2 FFT (re, im)
function fft(re: Float64Array, im: Float64Array) {
  const n = re.length
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) { [re[i], re[j]] = [re[j], re[i]];[im[i], im[j]] = [im[j], im[i]] }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len
    const wr = Math.cos(ang), wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = i + k + len / 2
        const tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr
        re[b] = re[a] - tr; im[b] = im[a] - ti
        re[a] += tr; im[a] += ti
        const ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr
      }
    }
  }
}

const SR = 48000
function spectrum(sig: Float64Array, start: number, N: number) {
  const re = new Float64Array(N), im = new Float64Array(N)
  for (let i = 0; i < N; i++) {
    const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (N - 1)) // Hann
    re[i] = sig[start + i] * w
  }
  fft(re, im)
  const mag = new Float64Array(N / 2)
  for (let i = 0; i < N / 2; i++) mag[i] = Math.hypot(re[i], im[i]) / N
  return mag
}

function bandEnergy(mag: Float64Array, loHz: number, hiHz: number) {
  const lo = Math.floor((loHz * mag.length * 2) / SR), hi = Math.ceil((hiHz * mag.length * 2) / SR)
  let e = 0
  for (let i = lo; i < Math.min(hi, mag.length); i++) e += mag[i] * mag[i]
  return e
}

function topPeaks(mag: Float64Array, aboveHz: number, count: number) {
  const lo = Math.floor((aboveHz * mag.length * 2) / SR)
  const idx = Array.from({ length: mag.length - lo }, (_, i) => i + lo)
  idx.sort((a, b) => mag[b] - mag[a])
  return idx.slice(0, count).map((i) => ({ hz: Math.round((i * SR) / (mag.length * 2)), db: (20 * Math.log10(mag[i] + 1e-12)).toFixed(1) }))
}

const N = 16384
const hfFrac = (sig: Float64Array, start: number) => {
  const mag = spectrum(sig, start, N)
  return (100 * bandEnergy(mag, 2000, SR / 2)) / bandEnergy(mag, 0, SR / 2)
}

// SENSITIVE high-freq probe: many events, report dB of the HF floor vs the
// fundamental (an impulse train at event_rate shows as a HF comb tens of dB
// down — inaudible in an energy %, obvious to the ear).
const sensitive = () => {
  const p = buildProgram()
  for (const d of (p.body.decls as any[])) if (d.op === 'paramDecl' && d.name === 'event_rate') d.value = 4
  const sig = render(p, 'sensitive', 1024) // ~5.5 s
  const M = 131072
  const mag = spectrum(sig, 65536, M) // window at 1.37 s in, settled, ~11 events
  const db = (e: number) => 10 * Math.log10(e + 1e-30)
  const fund = bandEnergy(mag, 60, 900)
  for (const [lo, hi] of [[2000, 4000], [4000, 8000], [8000, 16000]]) {
    const rel = db(bandEnergy(mag, lo, hi)) - db(fund)
    console.log(`  ${lo}-${hi}Hz vs fundamental: ${rel.toFixed(1)} dB   ${rel > -70 ? '◄ AUDIBLE junk' : ''}`)
  }
  console.log('  top peaks 2-8kHz:', topPeaks(mag, 2000, 8).map((p) => `${p.hz}/${p.db}`).join(' '))
  process.exit(0)
}
sensitive()

const main = () => {
  // 1) baseline: constant f0, is the steady state clean? (yes — shown earlier)
  const flat = render(buildProgram(), 'flat')
  console.log(`baseline (constant f0): >2kHz artifact = ${hfFrac(flat, flat.length - N - 1024).toFixed(3)}%  → clean\n`)

  // 2) the τ-dependence test. Bake a CONTINUOUS pitch modulation (4 Hz, ±12%)
  //    and render ~6 s. If the bug is phase = f0·τ, the artifact is ∝ τ:
  //    invisible early, exploding late. Same modulation, different τ window.
  const wob = buildProgram()
  ;(wob.body.decls as any[]).push({ op: 'instanceDecl', name: 'plfo', program: 'SinOsc', inputs: { freq: 4 } })
  for (const d of (wob.body.decls as any[])) {
    if (d.name === 'pphase') { // modulate the anchored pitch RATE (where pitch now lives)
      d.inputs.rate = { op: 'mul', args: [d.inputs.rate, { op: 'add', args: [1, { op: 'mul', args: [0.12, { op: 'ref', instance: 'plfo', output: 'sine' }] }] }] }
    }
  }
  const wobble = render(wob, 'pitch-mod', 1200) // ~6.4 s
  console.log('Same ±12% @4Hz pitch modulation, measured at growing τ:')
  console.log('  τ (s) │ >2kHz artifact energy')
  console.log('  ──────┼──────────────────────')
  for (const sec of [0.3, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    const start = Math.min(Math.floor(sec * SR), wobble.length - N - 1)
    console.log(`   ${sec.toFixed(1)}  │ ${hfFrac(wobble, start).toFixed(2)}%`)
  }
}
main()
