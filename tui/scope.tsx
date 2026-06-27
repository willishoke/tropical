// tropical · scope — a 2-channel oscilloscope TUI with basic scope controls.
//
// Each frame: drive the morph param (auto-LFO unless grabbed), read
// playback_position (the audio-paced master clock), render_window the two voice
// slots over the window ending there, lock the trace to a rising zero-crossing
// (the trigger), and draw braille. Cadence is the app's refresh; content is the
// master position, so the trace tracks playback. Controls:
//   ↑/↓ select · ←/→ adjust · space toggle (trigger / morph-auto) · q quit
//
//   bun tui/scope.tsx                  (spawns its own engine)
//   TROPICAL_SOCK=/path bun scope.tsx  (attach to a running --serve engine)
import React, { useEffect, useReducer, useRef, useState } from 'react'
import { render, Box, Text, useApp, useInput } from 'ink'
import { resolve } from 'node:path'
import { EngineClient, setupMorphScope } from './client'
import { renderWaveform } from './braille'
import { findTrigger, slice } from './trigger'

const REPO = resolve(import.meta.dir, '..')
const FREQS: [number, number] = [220, 330]
const SR = 44100
const FPS = 30
const BAND_H = 5
const TIMES = [256, 512, 1024, 2048, 4096]
const TRIGS = ['off', 'rising']
const SEARCH = Math.ceil(SR / Math.min(...FREQS)) + 8 // ≥ one period, so a rising edge always exists

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v)

type Ctl = { sel: number; timeIdx: number; offset: number; trig: number }

function Scope() {
  const { exit } = useApp()
  const [rows1, setRows1] = useState<string[]>([])
  const [rows2, setRows2] = useState<string[]>([])
  const [status, setStatus] = useState('connecting…')
  const [pos, setPos] = useState(0)
  const [, force] = useReducer((x: number) => x + 1, 0)
  const clientRef = useRef<EngineClient | null>(null)
  const ctl = useRef<Ctl>({ sel: 0, timeIdx: 2, offset: 0, trig: 1 })

  useInput((input, key) => {
    const c = ctl.current
    if (input === 'q' || key.escape) { clientRef.current?.kill(); exit(); return }
    if (key.upArrow) c.sel = (c.sel + 2) % 3
    else if (key.downArrow) c.sel = (c.sel + 1) % 3
    else if (key.leftArrow || key.rightArrow) {
      const d = key.rightArrow ? 1 : -1
      if (c.sel === 0) c.timeIdx = clamp(c.timeIdx + d, 0, TIMES.length - 1)
      else if (c.sel === 1) c.offset = clamp(Math.round((c.offset + d * 0.1) * 10) / 10, -1, 1)
      else c.trig = clamp(c.trig + d, 0, TRIGS.length - 1)
    } else if (input === ' ' && c.sel === 2) c.trig = 1 - c.trig
    force()
  })

  useEffect(() => {
    let stopped = false
    const c = new EngineClient({ repoRoot: REPO, onError: (e) => setStatus('engine error: ' + e.message) })
    clientRef.current = c
    ;(async () => {
      let slots: string[]
      try {
        slots = await setupMorphScope(c, FREQS)
        try { await c.call('start_audio', {}); setStatus('▶ live · locked to audio') }
        catch { setStatus('◼ no audio device · static window') }
      } catch (e: any) { setStatus('setup failed: ' + e.message); return }

      const w = Math.max(20, (process.stdout.columns ?? 84) - 4)
      const tick = async () => {
        if (stopped) return
        try {
          const k = ctl.current
          const display = TIMES[k.timeIdx]
          const search = k.trig === 0 ? 0 : SEARCH
          const count = display + search
          // render the window ending at the master position, then trigger-lock.
          // (morph sweeps inside the patch via the LFO — it's already baked into
          // whatever the kernel computes at these sample indices.)
          const p = (await c.call('playback_position', {})).position as number
          const res = await c.call('render_window', { start: Math.max(0, p - count), count, slots })
          const [a, b] = res.values as number[][]
          const off = k.trig === 1 ? Math.max(0, findTrigger(a, 0, search)) : 0
          setRows1(renderWaveform(slice(a, off, display, k.offset), w, BAND_H))
          setRows2(renderWaveform(slice(b, off, display, k.offset), w, BAND_H))
          setPos(p)
        } catch { /* transient (mid hot-swap) — skip frame */ }
        if (!stopped) setTimeout(tick, 1000 / FPS)
      }
      tick()
    })()
    return () => { stopped = true; c.kill() }
  }, [])

  const k = ctl.current
  const items = [
    ['timescale', String(TIMES[k.timeIdx])],
    ['offset', (k.offset >= 0 ? '+' : '') + k.offset.toFixed(1)],
    ['trigger', TRIGS[k.trig]],
  ] as const

  const Band = ({ label, color, rows }: { label: string; color: string; rows: string[] }) => (
    <Box flexDirection="column">
      <Text color={color} bold>{label}</Text>
      {rows.map((r, i) => (<Text key={i} color={color}>{r}</Text>))}
    </Box>
  )

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box justifyContent="space-between">
        <Text backgroundColor="cyan" color="black" bold> tropical · scope </Text>
        <Text dimColor>{status}   ·   sample {pos}</Text>
      </Box>
      <Box height={1} />
      <Band label={`CH1  ${FREQS[0]} Hz`} color="green" rows={rows1} />
      <Box height={1} />
      <Band label={`CH2  ${FREQS[1]} Hz`} color="magenta" rows={rows2} />
      <Box height={1} />
      <Box>
        {items.map(([label, val], i) => (
          <Text key={label}>
            {i === k.sel
              ? <Text backgroundColor="white" color="black" bold>{` ${label} ‹${val}› `}</Text>
              : <Text dimColor>{` ${label} ‹${val}› `}</Text>}
          </Text>
        ))}
        <Text dimColor>   ↑↓ select · ←→ adjust · space toggle · q quit</Text>
      </Box>
    </Box>
  )
}

render(<Scope />)
