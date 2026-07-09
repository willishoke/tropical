// tropical demo — the instrument. One fixed circuit, everything live:
//
//   o1, o2 ──mix──► ADDRESS ──► r1 r2 r3 r4 ──modalmix──► REVERB ──► FILTER ──► out
//   (the oscillator bank is the HAND: its summed waveform is the time-address
//    that scrubs four modal rings; the rings compose through the filter's
//    conjugate pole pair by the residue calculus — no state anywhere)
//
// Four scopes read the running kernel by RANDOM ACCESS (render_window): the
// same closed-form function the audio thread evaluates, at any τ, while audio
// runs on the GPU. The knobs are live param slots — no recompile, ever.
const rpc = (m, p) => window.tropical.call(m, p)

// ── the circuit ──────────────────────────────────────────────────────────────
// The modal REVERB is back, and the compile wall is gone: the strata passes
// walk the shared DAG with memos, and stage-0 hoisting moves the composed
// amplitudes (~90% of the flops — everything that scales with mode count)
// into a one-sample coefficient kernel compiled dumb and re-run at knob
// writes (see playground/README.md). The whole circuit now loads
// cache-cold in a few seconds. Reverb BEFORE the filter —
// compose small-into-big last.
const GRAPH = {
  nodes: [
    { id: 'o1', kind: 'source', params: { freq: 0.11, morph: 0 }, sel: {}, in: {} },
    { id: 'o2', kind: 'source', params: { freq: 2.2, morph: 0.6 }, sel: {}, in: {} },
    { id: 'adr', kind: 'mix', params: {}, sel: {}, in: { in: ['o1', 'o2'] } },
    { id: 'r1', kind: 'resonator', params: { freq: 110, decay: 4 }, sel: {}, in: { addr: ['adr'] } },
    { id: 'r2', kind: 'resonator', params: { freq: 165, decay: 4 }, sel: {}, in: { addr: ['adr'] } },
    { id: 'r3', kind: 'resonator', params: { freq: 220, decay: 4 }, sel: {}, in: { addr: ['adr'] } },
    { id: 'r4', kind: 'resonator', params: { freq: 330, decay: 4 }, sel: {}, in: { addr: ['adr'] } },
    { id: 'mx', kind: 'modalmix', params: {}, sel: {}, in: { in: ['r1', 'r2', 'r3', 'r4'] } },
    { id: 'rv', kind: 'reverb', params: { rt60: 2 }, sel: {}, in: { in: ['mx'] } },
    { id: 'flt', kind: 'filter', params: { cutoff: 800, resonance: 0.5 }, sel: {}, in: { in: ['rv'] } },
    { id: 'out', kind: 'out', params: {}, sel: {}, in: { in: ['flt'] } },
  ],
  out: 'out',
  taps: true,
}

// The four wells, in signal order. Each names the tap (= node id) it watches.
const SCOPES = [
  { tap: 'adr', label: 'ADDRESS', trig: false },
  { tap: 'r1', label: 'RING I', trig: true },
  { tap: 'mx', label: 'RINGS Σ', trig: true },
  { tap: 'flt', label: 'FILTER', trig: true },
]

// The panel: rows are modules; every knob is a live `param:<node>.<name>` slot.
// `fan` writes one gesture to several slots (the shared ring decay).
const MODULES = [
  { name: 'ADDRESS', knobs: [
    { slot: 'o1.freq', label: 'drift', min: 0.02, max: 8, step: 0, log: true, value: 0.11, unit: 'Hz' },
    { slot: 'o2.freq', label: 'wobble', min: 0.02, max: 60, step: 0, log: true, value: 2.2, unit: 'Hz' },
    { slot: 'o2.morph', label: 'shape', min: 0, max: 1, step: 0.02, value: 0.6, unit: '' },
  ]},
  { name: 'RINGS', knobs: [
    { slot: 'r1.freq', label: 'I', min: 40, max: 2000, step: 0, log: true, value: 110, unit: 'Hz' },
    { slot: 'r2.freq', label: 'II', min: 40, max: 2000, step: 0, log: true, value: 165, unit: 'Hz' },
    { slot: 'r3.freq', label: 'III', min: 40, max: 2000, step: 0, log: true, value: 220, unit: 'Hz' },
    { slot: 'r4.freq', label: 'IV', min: 40, max: 2000, step: 0, log: true, value: 330, unit: 'Hz' },
    { slot: 'r1.decay', label: 'decay', min: 0.5, max: 50, step: 0, log: true, value: 4, unit: '',
      fan: ['r1.decay', 'r2.decay', 'r3.decay', 'r4.decay'] },
  ]},
  { name: 'REVERB', knobs: [
    { slot: 'rv.rt60', label: 'size', min: 0.2, max: 12, step: 0, log: true, value: 2, unit: 's' },
    { slot: 'rv.dir', label: 'dir', min: 0, max: 1, step: 0.02, value: 0, unit: '' },
  ]},
  { name: 'FILTER', knobs: [
    { slot: 'flt.cutoff', label: 'cutoff', min: 20, max: 8000, step: 0, log: true, value: 800, unit: 'Hz' },
    { slot: 'flt.resonance', label: 'reso', min: 0, max: 1, step: 0.02, value: 0.5, unit: '' },
  ]},
  { name: 'TRANSPORT', knobs: [
    { slot: 'master.velocity', label: 'time warp', min: -2, max: 2, step: 0.05, value: 1, unit: '×' },
  ]},
]
const KNOBS = MODULES.flatMap((m) => m.knobs)

// ── knob mechanics ───────────────────────────────────────────────────────────
let sel = 0
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))
function nudge(k, dir, fine) {
  if (k.log) {
    const ratio = Math.pow(k.max / k.min, (fine ? 0.004 : 0.02) * dir)
    k.value = clamp(k.value * ratio, k.min, k.max)
  } else {
    const st = k.step * (fine ? 0.2 : 1)
    k.value = clamp(k.value + st * dir, k.min, k.max)
  }
  const v = k.value
  for (const slot of (k.fan ?? [k.slot])) rpc('set_param', { name: slot, value: v })
}

function fmt(k) {
  const v = k.value
  const s = k.log || Math.abs(v) >= 100 ? (v >= 100 ? v.toFixed(0) : v.toFixed(2)) : v.toFixed(2)
  return k.unit ? `${s} ${k.unit}` : s
}

function bar(k, width = 10) {
  const r = k.log
    ? Math.log(k.value / k.min) / Math.log(k.max / k.min)
    : (k.value - k.min) / (k.max - k.min)
  const f = clamp(Math.round(r * width), 0, width)
  return '█'.repeat(f) + '░'.repeat(width - f)
}

function renderPanel() {
  const root = document.getElementById('modules')
  root.innerHTML = ''
  let idx = 0
  for (const mod of MODULES) {
    const row = document.createElement('div')
    row.className = 'modrow'
    const name = document.createElement('div')
    name.className = 'modname'
    name.textContent = mod.name
    row.appendChild(name)
    const knobs = document.createElement('div')
    knobs.className = 'knobs'
    for (const k of mod.knobs) {
      const el = document.createElement('span')
      el.className = 'knob' + (idx === sel ? ' sel' : '')
      el.innerHTML = `<span class="lbl">${k.label}</span><span class="bar">${bar(k)}</span><span class="val">${fmt(k)}</span>`
      knobs.appendChild(el)
      idx++
    }
    row.appendChild(knobs)
    root.appendChild(row)
  }
  renderTransport()
}

function renderTransport() {
  const v = KNOBS.find((k) => k.slot === 'master.velocity').value
  const g = document.getElementById('tglyph')
  const t = document.getElementById('ttext')
  if (v <= -0.025) { g.textContent = '◀◀'; t.textContent = `REVERSE ${Math.abs(v).toFixed(2)}×` }
  else if (v < 0.025) { g.textContent = '⏸'; t.textContent = 'FROZEN' }
  else { g.textContent = '▶▶'; t.textContent = `FORWARD ${v.toFixed(2)}×` }
}

window.addEventListener('keydown', (e) => {
  const k = KNOBS[sel]
  if (e.key === 'ArrowUp') sel = (sel + KNOBS.length - 1) % KNOBS.length
  else if (e.key === 'ArrowDown') sel = (sel + 1) % KNOBS.length
  else if (e.key === 'ArrowLeft') nudge(k, -1, e.shiftKey)
  else if (e.key === 'ArrowRight') nudge(k, +1, e.shiftKey)
  else if (e.key === ' ') {
    const tw = KNOBS.find((x) => x.slot === 'master.velocity')
    tw.value = tw.value === 0 ? 1 : 0
    rpc('set_param', { name: 'master.velocity', value: tw.value })
  } else if (e.key === 'r') {
    const tw = KNOBS.find((x) => x.slot === 'master.velocity')
    tw.value = tw.value <= 0 ? 1 : -1
    rpc('set_param', { name: 'master.velocity', value: tw.value })
  } else return
  e.preventDefault()
  renderPanel()
})

// ── scope wells ──────────────────────────────────────────────────────────────
// Level trigger with hysteresis (ported from tui/scope/trigger.ts): lock the
// window to a rising edge so the trace holds still while τ advances.
function findTrigger(v, level, searchLen, hyst = 0.05) {
  const lim = Math.min(searchLen, v.length)
  let armed = (v[0] ?? 0) < level - hyst
  for (let i = 1; i < lim; i++) {
    if (v[i] < level - hyst) armed = true
    else if (armed && v[i - 1] < level && v[i] >= level) return i
  }
  for (let i = 1; i < lim; i++) if (v[i - 1] < level && v[i] >= level) return i
  return 0
}

const wells = []
function buildScopes() {
  const root = document.getElementById('scopes')
  for (const s of SCOPES) {
    const well = document.createElement('div')
    well.className = 'well'
    well.innerHTML = `<div class="paneltitle"><span>${s.label}</span><span class="tapname">scope.${s.tap}</span></div>`
    const canvas = document.createElement('canvas')
    well.appendChild(canvas)
    root.appendChild(well)
    wells.push({ ...s, canvas, slot: null })
  }
}

function drawTrace(canvas, v) {
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth * dpr
  const h = canvas.clientHeight * dpr
  if (canvas.width !== w) canvas.width = w
  if (canvas.height !== h) canvas.height = h
  const ctx = canvas.getContext('2d')
  const css = getComputedStyle(document.documentElement)
  ctx.fillStyle = css.getPropertyValue('--crt')
  ctx.fillRect(0, 0, w, h)
  // centerline + faint graticule, 1-bit spirit
  ctx.strokeStyle = css.getPropertyValue('--phosphor-dim')
  ctx.lineWidth = 1
  ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke()
  for (let gx = 1; gx < 4; gx++) {
    ctx.beginPath(); ctx.moveTo((w * gx) / 4, 0); ctx.lineTo((w * gx) / 4, h); ctx.stroke()
  }
  if (!v || v.length < 2) return
  // autoscale against a slow-decaying peak so quiet tails stay visible
  let peak = 1e-6
  for (const x of v) peak = Math.max(peak, Math.abs(x))
  ctx.strokeStyle = css.getPropertyValue('--phosphor')
  ctx.lineWidth = Math.max(1.5, dpr)
  ctx.shadowColor = css.getPropertyValue('--phosphor')
  ctx.shadowBlur = 4 * dpr
  ctx.beginPath()
  for (let i = 0; i < v.length; i++) {
    const x = (i / (v.length - 1)) * w
    const y = h / 2 - (v[i] / (peak * 1.15)) * (h / 2 - 4 * dpr)
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)
  }
  ctx.stroke()
  ctx.shadowBlur = 0
}

// ── boot + the render loop ───────────────────────────────────────────────────
const DISPLAY = 1024      // samples per well
const WARMUP = 64
const FPS = 24

async function boot() {
  const status = document.getElementById('status')
  buildScopes()
  renderPanel()
  drawDiagram()
  try {
    status.textContent = 'compiling the circuit… (one closed-form kernel incl. the 32-mode room — a few seconds)'
    await rpc('load_patch_graph', GRAPH)
    let audio = 'GPU · METAL'
    try { await rpc('start_audio', {}) } catch { audio = 'NO AUDIO DEVICE' }
    // resolve taps → render_window slots
    const res = await rpc('list_scope_taps', {})
    const taps = new Map((res.taps ?? []).map((t) => [t.name, t.slot]))
    for (const wll of wells) wll.slot = taps.get(wll.tap) ?? null
    const missing = wells.filter((s) => !s.slot).map((s) => s.tap)
    status.textContent = `${audio} · 44100 · ${missing.length ? 'missing taps: ' + missing.join(',') : '4 taps live'}`
    loop()
  } catch (e) {
    status.textContent = `boot failed: ${e.message}`
  }
}

async function loop() {
  try {
    const p = (await rpc('playback_position', {})).position
    document.getElementById('tau').textContent = `τ ${p}`
    const live = wells.filter((s) => s.slot)
    const count = DISPLAY * 2 // display + trigger search
    const start = Math.max(0, p - count - WARMUP)
    const res = await rpc('render_window', {
      start, count: count + WARMUP, slots: live.map((s) => s.slot),
    })
    live.forEach((s, i) => {
      const v = (res.values[i] ?? []).slice(WARMUP)
      const off = s.trig ? findTrigger(v, 0, DISPLAY) : 0
      drawTrace(s.canvas, v.slice(off, off + DISPLAY))
    })
  } catch { /* transient (hot-swap) — skip frame */ }
  setTimeout(loop, 1000 / FPS)
}

// ── the circuit diagram (1-bit, hand-drawn once) ─────────────────────────────
function drawDiagram() {
  const canvas = document.getElementById('diagram')
  const dpr = window.devicePixelRatio || 1
  const w = canvas.clientWidth * dpr
  const h = canvas.clientHeight * dpr
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d')
  const ink = getComputedStyle(document.documentElement).getPropertyValue('--ink')
  ctx.strokeStyle = ink
  ctx.fillStyle = ink
  ctx.lineWidth = 2 * dpr
  ctx.font = `700 ${10 * dpr}px Menlo, monospace`
  ctx.textAlign = 'center'
  const u = h / 34   // vertical unit
  const cx = w / 2
  const box = (x, y, bw, bh, label) => {
    ctx.strokeRect(x - bw / 2, y - bh / 2, bw, bh)
    ctx.fillText(label, x, y + 3.5 * dpr)
    return { x, y, bw, bh }
  }
  const wire = (x1, y1, x2, y2) => {
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke()
  }
  const arrow = (x, y) => {
    ctx.beginPath()
    ctx.moveTo(x - 4 * dpr, y - 5 * dpr); ctx.lineTo(x + 4 * dpr, y)
    ctx.lineTo(x - 4 * dpr, y + 5 * dpr); ctx.closePath(); ctx.fill()
  }
  const bw = w * 0.34, bh = u * 2.2
  // oscillator bank
  const o1 = box(cx - w * 0.22, u * 2.5, bw, bh, 'OSC o1')
  const o2 = box(cx + w * 0.22, u * 2.5, bw, bh, 'OSC o2')
  const adr = box(cx, u * 6, w * 0.5, bh, 'ADDRESS Σ')
  wire(o1.x, o1.y + bh / 2, o1.x, adr.y - bh / 2 - u * 0.4); wire(o1.x, adr.y - bh / 2 - u * 0.4, adr.x - w * 0.12, adr.y - bh / 2)
  wire(o2.x, o2.y + bh / 2, o2.x, adr.y - bh / 2 - u * 0.4); wire(o2.x, adr.y - bh / 2 - u * 0.4, adr.x + w * 0.12, adr.y - bh / 2)
  // rings, addressed
  const ys = u * 11
  const ring = []
  const labels = ['R I', 'R II', 'R III', 'R IV']
  for (let i = 0; i < 4; i++) {
    const x = w * (0.16 + 0.2267 * i)
    ring.push(box(x, ys, w * 0.19, bh, labels[i]))
    wire(adr.x, adr.y + bh / 2, adr.x, ys - bh / 2 - u * 0.8)
    wire(adr.x, ys - bh / 2 - u * 0.8, x, ys - bh / 2 - u * 0.4)
    wire(x, ys - bh / 2 - u * 0.4, x, ys - bh / 2)
  }
  ctx.fillText('addr — the hand that plays the rings', cx, ys - bh / 2 - u * 1.2)
  // modal mix → reverb → filter → out (the modal island, one column)
  const mx = box(cx, u * 15.5, w * 0.5, bh, 'MODAL Σ')
  for (const r of ring) { wire(r.x, r.y + bh / 2, r.x, mx.y - bh / 2 - u * 0.4); wire(r.x, mx.y - bh / 2 - u * 0.4, mx.x, mx.y - bh / 2) }
  const rv = box(cx, u * 19.3, w * 0.5, bh, 'REVERB ∿32')
  wire(mx.x, mx.y + bh / 2, rv.x, rv.y - bh / 2); arrow(rv.x, rv.y - bh / 2 - 2 * dpr)
  const flt = box(cx, u * 23.1, w * 0.5, bh, 'FILTER ~Q')
  wire(rv.x, rv.y + bh / 2, flt.x, flt.y - bh / 2); arrow(flt.x, flt.y - bh / 2 - 2 * dpr)
  const out = box(cx, u * 27, w * 0.36, bh, 'OUT')
  wire(flt.x, flt.y + bh / 2, out.x, out.y - bh / 2); arrow(out.x, out.y - bh / 2 - 2 * dpr)
  ctx.textAlign = 'left'
  ctx.font = `${8.5 * dpr}px Menlo, monospace`
  ctx.fillText('poles compose by residue —', 8 * dpr, u * 32)
  ctx.fillText('no state, anywhere', 8 * dpr, u * 33.2)
}

window.addEventListener('resize', drawDiagram)
boot()
