// tropical TUI — drive the flanger+delay patch live over JSON-RPC.
// Run from tui/:  bun start    (or: bun run app.tsx)
import React, { useEffect, useRef, useState } from 'react'
import { render, Box, Text, useApp, useInput } from 'ink'
import { EngineClient, repoRoot } from './client'
import { buildGraph, PARAMS, type ParamMode } from './patch'

const clamp = (x: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, x))

function bar(ratio: number, width = 18) {
  const f = clamp(Math.round(ratio * width), 0, width)
  return '█'.repeat(f) + '░'.repeat(width - f)
}

function fmt(step: number, unit: string, v: number) {
  if (unit === 's') return `${(v * 1000).toFixed(v < 0.001 ? 2 : 1)} ms`
  const s = step >= 1 ? Math.round(v).toString() : v.toFixed(2)
  return unit ? `${s} ${unit}` : s
}

const GROUP_LABEL: Record<string, string> = {
  transport: 'TRANSPORT — global τ scrub',
  voice: 'VOICE — plucked MorphOsc',
  space: 'SPACE — resonant comb (echo / pre-echo)',
  motion: 'MOTION — through-zero flange',
}

// The RPC that drives a knob, per its slot mode: glided knobs ease via a
// closed-form ramp, the anchored pitch stays phase-continuous, velocity scrubs
// the global master clock (re-basing τ for continuity), the rest step.
const METHOD: Record<ParamMode, string> = {
  live: 'set_param',
  glide: 'set_param_glide',
  freq: 'set_param_freq',
  velocity: 'set_param_velocity',
}

// The forward / freeze / reverse readout for the global time-warp knob.
function transport(v: number) {
  if (v <= -0.025) return { glyph: '◀◀', text: `REVERSE ${Math.abs(v).toFixed(2)}×`, color: 'redBright' as const }
  if (v < 0.025) return { glyph: '⏸ ', text: 'FROZEN', color: 'yellowBright' as const }
  return { glyph: '▶▶', text: `FORWARD ${v.toFixed(2)}×`, color: 'greenBright' as const }
}

const NO_AUDIO = !!process.env.TUI_NO_AUDIO

export function App() {
  const { exit } = useApp()
  const client = useRef<EngineClient | null>(null)
  const [vals, setVals] = useState<Record<string, number>>(() =>
    Object.fromEntries(PARAMS.map((p) => [p.slot, p.value])),
  )
  const [sel, setSel] = useState(0)
  const [running, setRunning] = useState(false)
  const [status, setStatus] = useState('starting engine…')

  useEffect(() => {
    const c = new EngineClient(repoRoot)
    client.current = c
    ;(async () => {
      try {
        await c.call('list_params')
        await c.call('load_patch_graph', buildGraph())
        if (!NO_AUDIO) await c.call('start_audio', { sample_rate: 48000, channels: 2 })
        setRunning(!NO_AUDIO)
        setStatus(NO_AUDIO ? 'loaded (audio off)' : 'playing')
      } catch (e: any) {
        setStatus('error: ' + e.message)
      }
    })()
    return () => c.kill()
  }, [])

  const drive = (slot: string, mode: ParamMode, value: number) =>
    client.current?.call(METHOD[mode], { name: slot, value }).catch(() => {})

  const quit = async () => {
    try { await client.current?.call('stop_audio') } catch {}
    client.current?.kill()
    exit()
  }

  const toggleAudio = async () => {
    try {
      if (running) { await client.current?.call('stop_audio'); setRunning(false); setStatus('stopped') }
      else { await client.current?.call('start_audio', { sample_rate: 48000, channels: 2 }); setRunning(true); setStatus('playing') }
    } catch (e: any) { setStatus('error: ' + e.message) }
  }

  const reset = () => {
    setVals(() => {
      const next: Record<string, number> = {}
      for (const p of PARAMS) { next[p.slot] = p.value; drive(p.slot, p.mode, p.value) }
      return next
    })
    setStatus('reset to defaults')
  }

  useInput((input, key) => {
    if (input === 'q' || (key.ctrl && input === 'c')) { quit(); return }
    if (input === ' ') { toggleAudio(); return }
    if (input === 'r') { reset(); return }
    if (key.upArrow) setSel((s) => (s - 1 + PARAMS.length) % PARAMS.length)
    else if (key.downArrow) setSel((s) => (s + 1) % PARAMS.length)
    else if (key.leftArrow || key.rightArrow) {
      const p = PARAMS[sel]
      const dir = key.rightArrow ? 1 : -1
      const mult = key.shift ? 10 : 1
      setVals((v) => {
        const nv = clamp(v[p.slot] + dir * p.step * mult, p.min, p.max)
        drive(p.slot, p.mode, nv)
        return { ...v, [p.slot]: nv }
      })
    }
  })

  let lastGroup = ''
  return (
    <Box flexDirection="column" paddingX={1}>
      <Text bold color="magentaBright">
        tropical · reversible τ-scrub — reverse reverb via the arrow moat
      </Text>
      <Text color="gray">
        engine: {running ? <Text color="greenBright">● {status}</Text> : <Text color="yellow">○ {status}</Text>}
      </Text>
      <Box height={1} />
      {PARAMS.map((p, i) => {
        const selected = i === sel
        const ratio = (vals[p.slot] - p.min) / (p.max - p.min || 1)
        const header = p.group !== lastGroup ? ((lastGroup = p.group), GROUP_LABEL[p.group]) : null
        return (
          <Box flexDirection="column" key={p.slot}>
            {header && (
              <Text color="cyan" dimColor>
                {'\n'}{header}
              </Text>
            )}
            <Text color={selected ? 'whiteBright' : undefined}>
              <Text color={selected ? 'greenBright' : 'gray'}>{selected ? '› ' : '  '}</Text>
              <Text>{p.label.padEnd(13)}</Text>
              <Text color={selected ? 'greenBright' : 'green'}>{bar(ratio)}</Text>
              <Text>  {fmt(p.step, p.unit, vals[p.slot]).padEnd(12)}</Text>
              {p.mode === 'velocity'
                ? (() => { const t = transport(vals[p.slot]); return <Text color={t.color} bold>{t.glyph} {t.text}</Text> })()
                : <Text color="gray" dimColor>{p.mode === 'glide' ? '~glide' : p.mode === 'freq' ? '~anchor' : ''}</Text>}
            </Text>
          </Box>
        )
      })}
      <Box height={1} />
      <Text color="gray">
        ↑↓ select · ←→ adjust (⇧ ×10) · space play/stop · r reset · q quit
      </Text>
      <Text color="gray" dimColor>
        Set a positive Tap (pre-echo, reading the future) and a slow Pluck rate, then drag
        Time warp negative — the pluck swells backward and the pre-echo tail flips to a trail.
      </Text>
    </Box>
  )
}

if (import.meta.main) render(<App />)
