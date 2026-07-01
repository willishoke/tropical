'use strict'

// ── Node vocabulary ────────────────────────────────────────────────────────
// 1:1 with the Lean arrow patch-graph `Node` cases. You patch DOWNSTREAM only;
// the frontend's lowering threads each effect's (possibly signal-modulated) warp
// onto the generators' clocks — totally composable, so effects stack freely.
const KINDS = {
  source: {
    title: 'Osc', accent: '#6cc7ff', summing: false,
    // `freq` is a CONTROL inlet: patch a Knob into it to drive pitch from a live
    // param slot (the baked `freq` knob is hidden while it's wired). Only a knob
    // may wire here — audio-rate into the pitch port is not FM (use the FM node).
    inlets: ['freq'], outlets: ['out'],
    sels: [{ name: 'voice', options: ['sine', 'saw', 'morph'], def: 'sine' }],
    knobs: [
      { name: 'freq', min: 20, max: 2000, def: 220, log: true, unit: 'Hz', anchor: true },
      { name: 'morph', min: 0, max: 1, def: 0, unit: '' },
    ],
  },
  knob: {
    title: 'Knob', accent: '#b7a4ff', summing: false,
    // A program that is nothing but a param with one output — for a value you want
    // to PATCH (fan it out, or into a `freq`/`mod` control inlet). Every knob is
    // live; this one is also a first-class wire source.
    inlets: [], outlets: ['out'], sels: [],
    knobs: [{ name: 'value', min: 0, max: 1000, def: 220, unit: '' }],
  },
  flange: {
    title: 'Flange', accent: '#ffcc66', summing: false,
    inlets: ['in'], outlets: ['out'], sels: [],
    // static δ comb — the pure plain-warp slide. `depth` is the GLIDE prototype:
    // its turns drive set_param_glide (closed-form ramp), so A/B it against any
    // other (raw set_param) knob to hear the difference.
    knobs: [{ name: 'depth', min: 0.0001, max: 0.01, def: 0.002, log: true, unit: 's', glide: true }],
  },
  sflange: {
    title: 'SwFlange', accent: '#ffd089', summing: false,
    inlets: ['in', 'mod'], outlets: ['out'], sels: [],
    // SIGNAL-modulated comb. Patch a signal into `mod` to sweep it; leave it open
    // and the built-in LFO at `rate` drives the sweep. The signal-warp composes.
    knobs: [
      { name: 'depth', min: 0.0002, max: 0.02, def: 0.005, log: true, unit: 's' },
      { name: 'rate', min: 0.02, max: 12, def: 0.3, log: true, unit: 'Hz' },
    ],
  },
  fm: {
    title: 'FM', accent: '#ff9ec7', summing: false,
    inlets: ['in'], outlets: ['out'], sels: [],
    knobs: [
      { name: 'carrier', min: 20, max: 2000, def: 330, log: true, unit: 'Hz' },
      { name: 'depth', min: 1, max: 400, def: 60, log: true, unit: '' },
    ],
  },
  warp: {
    title: 'Warp', accent: '#86e8c0', summing: false,
    inlets: ['in'], outlets: ['out'],
    sels: [{ name: 'mode', options: ['delay', 'reverse'], def: 'delay' }],
    knobs: [{ name: 'amount', min: 0.0001, max: 0.02, def: 0.004, log: true, unit: 's' }],
  },
  mix: {
    title: 'Mix', accent: '#c7ced9', summing: true,
    inlets: ['in'], outlets: ['out'], sels: [], knobs: [],
  },
  out: {
    title: 'Out · dac', accent: '#ff8a8a', summing: true, fixed: true,
    inlets: ['in'], outlets: [], sels: [], knobs: [],
  },
}

// TOPOLOGY RELOWERS, SCALARS ARE LIVE. Every continuous knob is a live
// `param:<id>.<knob>` module slot: turning it drives `set_param` on the running
// kernel, no recompile. Only structural edits — add/remove/rewire, and the
// `voice`/`mode` selectors — change the graph topology and trigger a relower +
// hot-swap. Control inlets accept a Knob node's output (a control value, not audio).
const CONTROL_INLETS = new Set(['freq', 'mod'])

// ── State ──────────────────────────────────────────────────────────────────
// node.in : { [inletPort]: [sourceNodeId, …] }
const nodes = new Map()
let counter = 0
let audioOn = false

const canvas = document.getElementById('canvas')
const svg = document.getElementById('wires')
const statusEl = document.getElementById('status')
const audioBtn = document.getElementById('btn-audio')

const allIns = (n) => Object.values(n.in).flat()

// ── Engine plumbing ──────────────────────────────────────────────────────────
async function rpc(method, params) {
  const r = await window.tropical.rpc(method, params)
  if (!r.ok) throw new Error(r.error)
  return r.data
}

let pushTimer = null
let pushInFlight = false
let pushAgain = false
function schedulePush(delay = 0) {
  clearTimeout(pushTimer)
  pushTimer = setTimeout(pushGraph, delay)
}
async function pushGraph() {
  if (pushInFlight) { pushAgain = true; return }
  pushInFlight = true
  try {
    await rpc('load_patch_graph', serialize())
    setStatus(audioOn ? 'playing' : 'compiled', false)
  } catch (e) {
    setStatus(`compile: ${e.message}`, true)
  } finally {
    pushInFlight = false
    if (pushAgain) { pushAgain = false; pushGraph() }
  }
}

// Live param drive: set the `param:<name>` slot on the running kernel via
// `set_param` — no recompile. Rapid drags coalesce to the latest value per name;
// a cold-start miss (slot not compiled yet) falls back to a relower that bakes the
// value as the slot default, after which subsequent turns are live.
const paramPending = new Map()
const paramBusy = new Set()
async function sendParam(method, name, value) {
  paramPending.set(name, value)
  if (paramBusy.has(name)) return
  paramBusy.add(name)
  try {
    while (paramPending.has(name)) {
      const v = paramPending.get(name); paramPending.delete(name)
      try { await rpc(method, { name, value: v }) }
      catch (e) { schedulePush(); break }
    }
  } finally { paramBusy.delete(name) }
}
// A raw slot write (steps at block rate) vs. a closed-form glide (the engine
// re-anchors 3 ramp slots; the kernel eases f(τ) — click-free). `glide: true` on
// a knob spec opts it into the ramp.
const setParamLive = (name, value) => sendParam('set_param', name, value)
const setParamGlide = (name, value) => sendParam('set_param_glide', name, value)
// A phase-anchored freq change: the engine bumps the oscillator's phase offset so
// the waveform stays continuous across the pitch change (no click).
const setParamFreq = (name, value) => sendParam('set_param_freq', name, value)

function serialize() {
  const out = []
  for (const n of nodes.values()) {
    const spec = KINDS[n.kind]
    const node = { id: n.id, kind: n.kind, params: {}, sel: {}, in: {} }
    for (const k of spec.knobs) node.params[k.name] = n.vals[k.name]
    for (const s of spec.sels) node.sel[s.name] = n.sels[s.name]
    for (const p of spec.inlets) node.in[p] = (n.in[p] || []).slice()
    out.push(node)
  }
  const outNode = [...nodes.values()].find((n) => n.kind === 'out')
  return { nodes: out, out: outNode ? outNode.id : null }
}

// ── Knob value ↔ angle math ───────────────────────────────────────────────
function toNorm(v, k) {
  if (k.log) return (Math.log(v) - Math.log(k.min)) / (Math.log(k.max) - Math.log(k.min))
  return (v - k.min) / (k.max - k.min)
}
function fromNorm(t, k) {
  t = Math.min(1, Math.max(0, t))
  if (k.log) return Math.exp(Math.log(k.min) + t * (Math.log(k.max) - Math.log(k.min)))
  return k.min + t * (k.max - k.min)
}
function fmtVal(v, k) {
  if (k.unit === 's') return `${(v * 1000).toFixed(v < 0.001 ? 2 : 1)}ms`
  if (k.unit === 'Hz') return v >= 100 ? `${v.toFixed(0)}Hz` : `${v.toFixed(2)}Hz`
  return v.toFixed(2)
}

// ── Node lifecycle ───────────────────────────────────────────────────────────
function addNode(kind, x, y) {
  const spec = KINDS[kind]
  if (spec.fixed && [...nodes.values()].some((n) => n.kind === kind)) return null
  const id = `${kind}${++counter}`
  const vals = {}; for (const k of spec.knobs) vals[k.name] = k.def
  const sels = {}; for (const s of spec.sels) sels[s.name] = s.def
  const inp = {}; for (const p of spec.inlets) inp[p] = []
  const n = { id, kind, x: x ?? 80 + (counter % 6) * 34, y: y ?? 90 + (counter % 6) * 30, vals, sels, in: inp }
  nodes.set(id, n)
  renderNodes()
  return n
}

function deleteNode(id) {
  const n = nodes.get(id)
  if (!n || KINDS[n.kind].fixed) return
  nodes.delete(id)
  for (const m of nodes.values())
    for (const p of Object.keys(m.in)) m.in[p] = m.in[p].filter((s) => s !== id)
  renderNodes(); schedulePush()
}

// reachability for the cycle check (across all inlets)
function reaches(a, target, seen = new Set()) {
  if (a === target) return true
  if (seen.has(a)) return false
  seen.add(a)
  const na = nodes.get(a)
  return na ? allIns(na).some((s) => reaches(s, target, seen)) : false
}

function canConnect(fromId, toId, toPort) {
  if (fromId === toId) return false
  const from = nodes.get(fromId), to = nodes.get(toId)
  if (!from || !to || !KINDS[to.kind].inlets.includes(toPort)) return false
  if ((to.in[toPort] || []).includes(fromId)) return false
  const fromIsKnob = from.kind === 'knob'
  // A knob is a control value: it may only drive a control inlet (freq/mod).
  if (fromIsKnob && !CONTROL_INLETS.has(toPort)) return false
  // `freq` is control-only — audio-rate into the pitch port is not FM.
  if (!fromIsKnob && toPort === 'freq') return false
  if (reaches(fromId, toId)) return false   // would form a cycle
  return true
}

function connect(fromId, toId, toPort) {
  if (!canConnect(fromId, toId, toPort)) return
  const to = nodes.get(toId)
  if (!KINDS[to.kind].summing) to.in[toPort] = []   // single inlet: replace
  to.in[toPort].push(fromId)
  renderNodes(); schedulePush()
}

function deleteEdge(toId, toPort, fromId) {
  const to = nodes.get(toId)
  if (!to) return
  to.in[toPort] = (to.in[toPort] || []).filter((s) => s !== fromId)
  renderNodes(); schedulePush()
}

// ── Rendering ────────────────────────────────────────────────────────────────
function renderNodes() {
  for (const el of [...canvas.querySelectorAll('.node')]) el.remove()
  for (const n of nodes.values()) canvas.appendChild(nodeEl(n))
  renderWires()
}

function nodeEl(n) {
  const spec = KINDS[n.kind]
  const el = document.createElement('div')
  el.className = 'node'; el.dataset.id = n.id
  el.style.left = `${n.x}px`; el.style.top = `${n.y}px`

  const title = document.createElement('div')
  title.className = 'title'
  title.innerHTML = `<span class="swatch" style="background:${spec.accent}"></span>` +
    `<span class="label">${spec.title}</span><span class="id">${n.id}</span>`
  startNodeDrag(title, n)
  el.appendChild(title)

  const body = document.createElement('div')
  body.className = 'body'
  for (const s of spec.sels) {
    const sel = document.createElement('select'); sel.className = 'sel'
    for (const o of s.options) {
      const opt = document.createElement('option'); opt.value = o; opt.textContent = o
      if (o === n.sels[s.name]) opt.selected = true
      sel.appendChild(opt)
    }
    sel.onchange = () => { n.sels[s.name] = sel.value; schedulePush() }
    body.appendChild(sel)
  }
  // A knob whose name matches a wired inlet is overridden by the patch cord
  // (e.g. `freq` driven by a Knob node), so hide it.
  const shownKnobs = spec.knobs.filter((k) => !((n.in[k.name] || []).length))
  if (shownKnobs.length) {
    const row = document.createElement('div'); row.className = 'row'
    for (const k of shownKnobs) row.appendChild(knobEl(n, k))
    body.appendChild(row)
  }
  el.appendChild(body)

  const jacks = document.createElement('div'); jacks.className = 'jacks'
  const inCol = document.createElement('div'); inCol.className = 'jack-col in'
  const outCol = document.createElement('div'); outCol.className = 'jack-col out'
  for (const p of spec.inlets) inCol.appendChild(jackEl(n, p, 'in'))
  for (const p of spec.outlets) outCol.appendChild(jackEl(n, p, 'out'))
  jacks.appendChild(inCol); jacks.appendChild(outCol)
  el.appendChild(jacks)

  el.ondblclick = (e) => { e.stopPropagation(); deleteNode(n.id) }
  return el
}

function knobEl(n, k) {
  const wrap = document.createElement('div'); wrap.className = 'knob-wrap'
  const knob = document.createElement('div'); knob.className = 'knob'
  const valEl = document.createElement('div'); valEl.className = 'knob-val'
  const setAngle = () => {
    const t = toNorm(n.vals[k.name], k)
    knob.style.setProperty('--ang', `${-135 + t * 270}deg`)
    valEl.textContent = fmtVal(n.vals[k.name], k)
  }
  setAngle()
  // Every continuous knob is a live param slot `<id>.<knob>` — turning it drives
  // the running kernel with no relower (only topology/selector edits do). A glided
  // knob eases via set_param_glide; the rest do a raw set_param write.
  const pname = `${n.id}.${k.name}`
  const send = k.glide ? setParamGlide : k.anchor ? setParamFreq : setParamLive
  knob.onmousedown = (e) => {
    e.preventDefault(); e.stopPropagation()
    const y0 = e.clientY, t0 = toNorm(n.vals[k.name], k)
    const move = (ev) => {
      n.vals[k.name] = fromNorm(t0 + (y0 - ev.clientY) / 180, k)
      setAngle()
      send(pname, n.vals[k.name])
    }
    const up = () => {
      window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up)
      send(pname, n.vals[k.name])
    }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }
  wrap.appendChild(knob)
  const lab = document.createElement('div'); lab.className = 'knob-label'; lab.textContent = k.name
  wrap.appendChild(lab); wrap.appendChild(valEl)
  return wrap
}

function jackEl(n, port, dir) {
  const j = document.createElement('div'); j.className = 'jack'
  const dot = document.createElement('div'); dot.className = 'dot'
  const lab = document.createElement('span'); lab.textContent = port
  dot.dataset.id = n.id; dot.dataset.dir = dir; dot.dataset.port = port
  j.appendChild(dot); j.appendChild(lab)
  if (dir === 'out') startWireDrag(dot, n)
  return j
}

function jackCenter(id, dir, port) {
  const dot = canvas.querySelector(`.dot[data-id="${id}"][data-dir="${dir}"][data-port="${port}"]`)
  if (!dot) return null
  const r = dot.getBoundingClientRect(), cr = canvas.getBoundingClientRect()
  return { x: r.left + r.width / 2 - cr.left, y: r.top + r.height / 2 - cr.top }
}

function renderWires(pending) {
  while (svg.firstChild) svg.removeChild(svg.firstChild)
  for (const n of nodes.values()) {
    for (const [port, srcs] of Object.entries(n.in)) {
      for (const src of srcs) {
        const a = jackCenter(src, 'out', 'out'), b = jackCenter(n.id, 'in', port)
        if (a && b) svg.appendChild(wirePath(a, b, false, { to: n.id, port, from: src }))
      }
    }
  }
  if (pending) svg.appendChild(wirePath(pending.a, pending.b, true))
}

function wirePath(a, b, isPending, edge) {
  const dx = Math.max(40, Math.abs(b.x - a.x) * 0.5)
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path')
  p.setAttribute('d', `M ${a.x} ${a.y} C ${a.x + dx} ${a.y}, ${b.x - dx} ${b.y}, ${b.x} ${b.y}`)
  if (isPending) p.setAttribute('class', 'pending')
  if (edge) {
    p.style.pointerEvents = 'stroke'; p.style.cursor = 'pointer'
    p.onclick = () => deleteEdge(edge.to, edge.port, edge.from)
  }
  return p
}

// ── Drag: move node ──────────────────────────────────────────────────────────
function startNodeDrag(handle, n) {
  handle.onmousedown = (e) => {
    if (e.button !== 0) return
    e.preventDefault()
    const ox = e.clientX - n.x, oy = e.clientY - n.y
    const el = handle.parentElement
    const move = (ev) => {
      n.x = Math.max(0, ev.clientX - ox); n.y = Math.max(0, ev.clientY - oy)
      el.style.left = `${n.x}px`; el.style.top = `${n.y}px`; renderWires()
    }
    const up = () => { window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up) }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }
}

// ── Drag: patch cord ─────────────────────────────────────────────────────────
function startWireDrag(dot, n) {
  dot.onmousedown = (e) => {
    e.preventDefault(); e.stopPropagation()
    dot.classList.add('armed')
    const a = jackCenter(n.id, 'out', 'out')
    const move = (ev) => {
      const cr = canvas.getBoundingClientRect()
      renderWires({ a, b: { x: ev.clientX - cr.left, y: ev.clientY - cr.top } })
    }
    const up = (ev) => {
      window.removeEventListener('mousemove', move); window.removeEventListener('mouseup', up)
      dot.classList.remove('armed')
      const tgt = document.elementFromPoint(ev.clientX, ev.clientY)
      if (tgt && tgt.classList.contains('dot') && tgt.dataset.dir === 'in')
        connect(n.id, tgt.dataset.id, tgt.dataset.port)
      else renderWires()
    }
    window.addEventListener('mousemove', move); window.addEventListener('mouseup', up)
  }
}

// ── Transport ────────────────────────────────────────────────────────────────
audioBtn.onclick = async () => {
  try {
    if (!audioOn) {
      await rpc('start_audio', {})
      audioOn = true; audioBtn.textContent = '■ stop audio'; audioBtn.classList.add('on'); setStatus('playing', false)
    } else {
      await rpc('stop_audio', {})
      audioOn = false; audioBtn.textContent = '▶ start audio'; audioBtn.classList.remove('on'); setStatus('stopped', false)
    }
  } catch (e) { setStatus(e.message, true) }
}

function setStatus(text, err) { statusEl.textContent = text; statusEl.classList.toggle('err', !!err) }

// ── Palette + boot ───────────────────────────────────────────────────────────
function buildPalette() {
  const pal = document.getElementById('palette')
  for (const [kind, spec] of Object.entries(KINDS)) {
    if (spec.fixed) continue
    const b = document.createElement('button')
    b.innerHTML = `<span class="dot" style="background:${spec.accent}"></span>${spec.title}`
    b.onclick = () => addNode(kind)
    pal.appendChild(b)
  }
}

window.tropical.onStatus((s) => {
  if (s.kind === 'exit') setStatus(s.text, true)
  else if (s.kind === 'up') setStatus('engine up', false)
  else if (s.kind === 'stderr') console.warn('[engine]', s.text)
})

buildPalette()
addNode('out', 1180, 360)
addNode('source', 120, 200)
setStatus('engine up · patch something', false)
