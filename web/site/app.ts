/**
 * app.ts — Demo UI: patch dropdown, start/stop, plan loading.
 */

import { startHost, type TropicalHost } from '../host/context.js'
import { compilePlanJson } from '../host/compiler.js'

type ManifestEntry = { slug: string; title: string; description: string; planPath: string }

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

function currentPatch(): ManifestEntry | undefined {
  return manifest.find((p) => p.slug === patchSelect.value)
}

patchSelect.addEventListener('change', () => {
  const p = currentPatch()
  patchMeta.textContent = p?.description ?? ''
})

playBtn.addEventListener('click', async () => {
  try {
    setError('')
    playBtn.disabled = true
    const patch = currentPatch()
    if (!patch) return

    if (!host) {
      setStatus('initializing audio…')
      host = await startHost({ workletUrl: './worklet.js', outputChannels: 2 })
    }

    setStatus(`compiling ${patch.title}…`)
    const res = await fetch(`./${patch.planPath}`)
    const planJson = await res.text()
    const t0 = performance.now()
    const loaded = await compilePlanJson(planJson, 2048)
    const dt = performance.now() - t0
    const crossOK = (window as unknown as { crossOriginIsolated?: boolean }).crossOriginIsolated
    setInfo(
      `memory: ${(loaded.layout.pageCount * 65536 / 1024).toFixed(0)} KB · ` +
      `registers: ${loaded.layout.registerCount} · arrays: ${loaded.layout.arraySlotCount} · ` +
      `params: ${loaded.layout.paramCount} · compile: ${dt.toFixed(1)} ms · ` +
      `audioCtx: ${host.context.sampleRate}Hz ${host.context.state} · ` +
      `crossOriginIsolated: ${crossOK ? 'yes' : 'NO'}`,
    )

    host.loadPlan(loaded)
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
  if (host) {
    host.fadeOut()
    // Give fade time before tearing down; keep context alive so restart is fast.
    setTimeout(() => {
      if (host) {
        host.dispose()
        host = null
      }
    }, 100)
  }
  stopBtn.disabled = true
  playBtn.disabled = false
  setStatus('stopped')
})

loadManifest()
