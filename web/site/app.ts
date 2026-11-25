/**
 * app.ts — Demo UI: patch dropdown, start/stop, patch loading.
 */

import { startHost, type TropicalHost } from '../host/context.js'
import type { CycleDisplay, Curve } from '../worker/curve.js'
import {
  kernelFrames, kernelOutputChannelCount, kernelParams, type KernelManifest,
} from '../runtime/index.js'

/** A playable knob (see `web/build_patches.ts`): a `param:` slot, named
 *  without the prefix, with its slider range. */
type Knob = { param: string; label: string; min: number; max: number; log?: boolean; unit?: string }

type ManifestEntry = {
  slug: string
  title: string
  description: string
  knobs?: Knob[]
  /** One cycle of an output channel to draw (`worker/curve.ts`). */
  display?: CycleDisplay
  wasmPath: string
  manifestPath: string
}

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id)
  if (!el) throw new Error(`#${id} not found`)
  return el as T
}

const patchSelect = $<HTMLSelectElement>('patch-select')
const playBtn = $<HTMLButtonElement>('play-btn')
const stopBtn = $<HTMLButtonElement>('stop-btn')
const patchMeta = $('patch-meta')
const statusEl = $('status')
const infoEl = $('info')
const errorEl = $('error')
const knobsEl = $('knobs')
const curveEl = $('curve')
const curveCanvas = $<HTMLCanvasElement>('curve-canvas')
const curveCaption = $('curve-caption')

let host: TropicalHost | null = null
let manifest: ManifestEntry[] = []

function setStatus(s: string): void { statusEl.textContent = s }
function setInfo(s: string): void { infoEl.textContent = s }
function setError(s: string): void { errorEl.textContent = s }

async function loadManifest(): Promise<void> {
  try {
    const res = await fetch('./patches/index.json')
    manifest = await res.json() as ManifestEntry[]
    patchSelect.innerHTML = ''
    for (const p of manifest) {
      const opt = document.createElement('option')
      opt.value = p.slug
      opt.textContent = p.title
      patchSelect.appendChild(opt)
    }
    patchSelect.dispatchEvent(new Event('change'))
    setStatus('ready')
    playBtn.disabled = false
  } catch (err) {
    setStatus('error')
    setError(`Failed to load patch manifest: ${(err as Error).message}`)
  }
}

// ── knobs ────────────────────────────────────────────────────────────────
// A slider runs 0..SLIDER_STEPS; a log knob maps it geometrically. Values are
// kept per patch, so a stop/start keeps the knobs where they were.
const SLIDER_STEPS = 1000
const knobValues = new Map<string, Map<string, number>>()

const toSlider = (k: Knob, v: number): number => {
  const t = k.log ? Math.log(v / k.min) / Math.log(k.max / k.min) : (v - k.min) / (k.max - k.min)
  return Math.round(Math.min(1, Math.max(0, t)) * SLIDER_STEPS)
}
const fromSlider = (k: Knob, s: number): number => {
  const t = s / SLIDER_STEPS
  return k.log ? k.min * Math.pow(k.max / k.min, t) : k.min + t * (k.max - k.min)
}
const fmtValue = (v: number, unit?: string): string => {
  const a = Math.abs(v)
  const s = a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : a >= 1 ? v.toFixed(2) : v.toFixed(3)
  return unit ? `${s} ${unit}` : s
}

/** Build the knob panel for `patch`, seeded from the values kept for it or
 *  from the kernel's param defaults. */
function renderKnobs(patch: ManifestEntry, kernel: KernelManifest): void {
  knobsEl.innerHTML = ''
  const knobs = patch.knobs ?? []
  if (!knobValues.has(patch.slug)) knobValues.set(patch.slug, new Map())
  const values = knobValues.get(patch.slug)!
  for (const k of knobs) {
    const name = `param:${k.param}`
    const port = kernelParams(kernel).find((p) => p.name === name)
    const value = values.get(name) ?? port?.default ?? k.min
    values.set(name, value)
    const row = document.createElement('label')
    row.className = 'knob'
    const label = document.createElement('span')
    label.textContent = k.label
    const slider = document.createElement('input')
    slider.type = 'range'
    slider.min = '0'
    slider.max = String(SLIDER_STEPS)
    slider.value = String(toSlider(k, value))
    const readout = document.createElement('output')
    readout.textContent = fmtValue(value, k.unit)
    slider.addEventListener('input', () => {
      const v = fromSlider(k, Number(slider.value))
      values.set(name, v)
      readout.textContent = fmtValue(v, k.unit)
      host?.setParam(name, v)
    })
    row.append(label, slider, readout)
    knobsEl.appendChild(row)
  }
}

/** Send the kept knob values for `patch` (after a load: its kernel starts
 *  from the manifest defaults). */
function sendKnobs(patch: ManifestEntry): void {
  const values = knobValues.get(patch.slug)
  if (!values || !host) return
  for (const [name, v] of values) host.setParam(name, v)
}

// ── the curve ────────────────────────────────────────────────────────────
// Every point is the kernel's own output at a chosen τ, evaluated in the
// renderer Worker (`WasmKernel.renderAt`); the page only draws it. The
// vertical scale follows the peak, falling back slowly so a fading envelope
// shows as a shrinking curve.
const CURVE_POINTS = 512
let curveScale = 0

function showDisplay(patch: ManifestEntry | undefined): void {
  curveScale = 0
  curveEl.hidden = !patch?.display
  if (!host) return
  host.onCurve = drawCurve
  host.setDisplay(patch?.display ?? null, CURVE_POINTS)
}

function drawCurve(c: Curve): void {
  const dpr = window.devicePixelRatio || 1
  const w = Math.round(curveCanvas.clientWidth * dpr), h = Math.round(curveCanvas.clientHeight * dpr)
  if (curveCanvas.width !== w || curveCanvas.height !== h) { curveCanvas.width = w; curveCanvas.height = h }
  const g = curveCanvas.getContext('2d')
  if (!g || w === 0 || h === 0) return
  let peak = 0
  for (const v of c.data) if (Number.isFinite(v)) peak = Math.max(peak, Math.abs(v))
  curveScale = Math.max(peak, curveScale * 0.97, 1e-6)
  const css = getComputedStyle(document.documentElement)
  g.clearRect(0, 0, w, h)
  g.strokeStyle = css.getPropertyValue('--border').trim()
  g.lineWidth = dpr
  g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke()
  g.strokeStyle = css.getPropertyValue('--accent').trim()
  g.lineWidth = 1.5 * dpr
  g.beginPath()
  const n = c.data.length
  for (let j = 0; j < n; j++) {
    const x = (j / (n - 1)) * w
    const y = h / 2 - (c.data[j]! / curveScale) * (h / 2 - 4 * dpr)
    if (j === 0) g.moveTo(x, y); else g.lineTo(x, y)
  }
  g.stroke()
  curveCaption.textContent =
    `One ${c.freq.toFixed(1)} Hz cycle, ${n} points, each the kernel itself at τ from frame ${c.frame} ` +
    `(the played frame); peak ${peak.toFixed(3)}. Slower clocks advance by the same cycle.`
}

function currentPatch(): ManifestEntry | undefined {
  return manifest.find((p) => p.slug === patchSelect.value)
}

patchSelect.addEventListener('change', async () => {
  const p = currentPatch()
  patchMeta.textContent = p?.description ?? ''
  knobsEl.innerHTML = ''
  curveEl.hidden = true
  if (p?.knobs?.length) {
    try {
      const kernel = await fetch(`./${p.manifestPath}`).then((r) => r.json() as Promise<KernelManifest>)
      if (currentPatch() === p) renderKnobs(p, kernel)
    } catch (err) {
      setError(`Failed to load ${p.title}'s knobs: ${(err as Error).message}`)
    }
  }
})

playBtn.addEventListener('click', async () => {
  try {
    setError('')
    playBtn.disabled = true
    const patch = currentPatch()
    if (!patch) return

    if (!host) {
      setStatus('initializing audio…')
      host = await startHost({ workletUrl: './worklet.js', rendererUrl: './renderer.js', outputChannels: 2 })
    }

    setStatus(`loading ${patch.title}…`)
    const [wasm, manifestJson] = await Promise.all([
      fetch(`./${patch.wasmPath}`).then((r) => r.arrayBuffer()),
      fetch(`./${patch.manifestPath}`).then((r) => r.json() as Promise<KernelManifest>),
    ])
    setInfo(
      `wasm: ${(wasm.byteLength / 1024).toFixed(1)} KB · ` +
      `frames: ${kernelFrames(manifestJson)} · params: ${kernelParams(manifestJson).length} · ` +
      `channels: ${kernelOutputChannelCount(manifestJson)} · rate: ${manifestJson.rate}Hz · ` +
      `audioCtx: ${host.context.sampleRate}Hz ${host.context.state}`,
    )

    host.loadPatch(wasm, manifestJson)
    renderKnobs(patch, manifestJson)
    sendKnobs(patch)
    showDisplay(patch)
    host.fadeIn()
    setStatus(`playing — ${patch.title}`)
    stopBtn.disabled = false
  } catch (err) {
    setStatus('error')
    setError(`${(err as Error).message}`)
    playBtn.disabled = false
  }
})

stopBtn.addEventListener('click', async () => {
  stopBtn.disabled = true
  setStatus('stopping…')
  const h = host
  host = null
  if (h) {
    // Let the fade reach the speakers before tearing down.
    await h.fadeOut()
    await h.dispose()
  }
  playBtn.disabled = false
  setStatus('stopped')
})

loadManifest()
