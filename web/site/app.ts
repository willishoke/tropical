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
  /** The kernel channels that are sound, in host order; the rest are taps. */
  audio: number[]
  /** Curves to draw: one cycle of an output channel each (`worker/curve.ts`). */
  displays?: CycleDisplay[]
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
const curvesEl = $('curves')

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

// ── the curves ───────────────────────────────────────────────────────────
// Every point is the kernel's own output at a chosen τ, evaluated in the
// renderer Worker (`WasmKernel.renderAt`); the page only draws it. Each
// curve's vertical scale follows its peak, falling back slowly.
const CURVE_POINTS = 512
type CurveView = { canvas: HTMLCanvasElement; caption: HTMLElement; display: CycleDisplay; scale: number }
let curveViews: CurveView[] = []

/** Build a canvas per display of `patch` (none: the panel is cleared) and
 *  ask the renderer for its curves. */
function showDisplays(patch: ManifestEntry | undefined): void {
  curvesEl.innerHTML = ''
  curveViews = (patch?.displays ?? []).map((display) => {
    const fig = document.createElement('figure')
    fig.className = 'curve'
    const canvas = document.createElement('canvas')
    const caption = document.createElement('figcaption')
    caption.textContent = display.label ?? `channel ${display.channel}`
    fig.append(canvas, caption)
    curvesEl.appendChild(fig)
    return { canvas, caption, display, scale: 0 }
  })
  if (!host) return
  host.onCurve = drawCurve
  host.setDisplay(patch?.displays ?? [], CURVE_POINTS)
}

function drawCurve(c: Curve): void {
  const v = curveViews[c.index]
  if (!v) return
  const { canvas } = v
  const dpr = window.devicePixelRatio || 1
  const w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr)
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h }
  const g = canvas.getContext('2d')
  if (!g || w === 0 || h === 0) return
  let peak = 0
  for (const x of c.data) if (Number.isFinite(x)) peak = Math.max(peak, Math.abs(x))
  v.scale = Math.max(peak, v.scale * 0.97, 1e-6)
  const css = getComputedStyle(document.documentElement)
  g.clearRect(0, 0, w, h)
  g.strokeStyle = css.getPropertyValue('--border').trim()
  g.lineWidth = dpr
  g.beginPath(); g.moveTo(0, h / 2); g.lineTo(w, h / 2); g.stroke()
  g.strokeStyle = css.getPropertyValue('--accent').trim()
  g.lineWidth = 1.5 * dpr
  g.beginPath()
  // Point j sits at phase j/n of the cycle, so the n points span (n−1)/n of
  // the width; the cycle's end (phase 1, which is the next cycle's phase 0)
  // is the right edge, not drawn.
  const n = c.data.length
  for (let j = 0; j < n; j++) {
    const x = (j / n) * w
    const y = h / 2 - (c.data[j]! / v.scale) * (h / 2 - 4 * dpr)
    if (j === 0) g.moveTo(x, y); else g.lineTo(x, y)
  }
  g.stroke()
  v.caption.textContent =
    `${v.display.label ?? `channel ${v.display.channel}`} — ${c.freq.toFixed(2)} Hz in kernel time, ` +
    `cycle ${c.cycle} from its phase 0; ${n} points, each the kernel itself; peak ${peak.toFixed(3)}. ` +
    'Drawn with the newest knob values at the renderer\'s estimate of the next frame to play, ' +
    'so it leads what you hear by the knob margin plus the device latency.'
}

function clearDisplays(): void {
  curvesEl.innerHTML = ''
  curveViews = []
  host?.setDisplay([], CURVE_POINTS)
}

function currentPatch(): ManifestEntry | undefined {
  return manifest.find((p) => p.slug === patchSelect.value)
}

patchSelect.addEventListener('change', async () => {
  const p = currentPatch()
  patchMeta.textContent = p?.description ?? ''
  knobsEl.innerHTML = ''
  clearDisplays()
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

    host.loadPatch(wasm, manifestJson, patch.audio)
    renderKnobs(patch, manifestJson)
    sendKnobs(patch)
    showDisplays(patch)
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
  clearDisplays()
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
