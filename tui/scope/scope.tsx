// tropical · scope — a tap browser. Anything wired to `scope.<name>` (a reserved
// sink, like `dac`) is observable: pick a tap, render_window its module slot over
// the window ending at the audio-paced master clock, trigger-lock, draw braille.
//
// Standalone it builds a morph demo and taps its nodes; attached (TROPICAL_SOCK)
// it taps whatever patch the other TUI has live.
//   ↑/↓ select control · ←/→ adjust (source cycles taps) · space trigger · q quit
//
//   bun tui/scope/scope.tsx                       (spawns its own engine)
//   TROPICAL_SOCK=/path bun tui/scope/scope.tsx   (attach to a running --serve engine)
import React, { useEffect, useReducer, useRef, useState } from 'react'
import { render, Box, Text, useApp, useInput } from 'ink'
import { resolve } from 'node:path'
import { EngineClient, setupTapScope, type Tap } from './client'
import { renderWaveform } from './braille'
import { findTrigger, slice } from './trigger'

const REPO = resolve(import.meta.dir, '../..')
const FPS = 30
const BAND_H = 7
const WARMUP = 64 // CF-only is cold-start-exact (no per-wire delays), but a small
                  // lead is harmless and robust if a patch ever reintroduces one.
const TIMES = [256, 512, 1024, 2048, 4096]
const TRIGS = ['off', 'rising']

const clamp = (v: number, lo: number, hi: number) => (v < lo ? lo : v > hi ? hi : v)

type Ctl = { sel: number; srcIdx: number; timeIdx: number; offset: number; trig: number }

function Scope() {
  const { exit } = useApp()
  const [rows, setRows] = useState<string[]>([])
  const [status, setStatus] = useState('connecting…')
  const [pos, setPos] = useState(0)
  const tapsRef = useRef<Tap[]>([])
  const [, force] = useReducer((x: number) => x + 1, 0)
  const clientRef = useRef<EngineClient | null>(null)
  const ctl = useRef<Ctl>({ sel: 0, srcIdx: 0, timeIdx: 2, offset: 0, trig: 1 })

  useInput((input, key) => {
    const c = ctl.current
    const nTaps = Math.max(1, tapsRef.current.length)
    if (input === 'q' || key.escape) { clientRef.current?.kill(); exit(); return }
    if (key.upArrow) c.sel = (c.sel + 3) % 4
    else if (key.downArrow) c.sel = (c.sel + 1) % 4
    else if (key.leftArrow || key.rightArrow) {
      const d = key.rightArrow ? 1 : -1
      if (c.sel === 0) c.srcIdx = (c.srcIdx + d + nTaps) % nTaps
      else if (c.sel === 1) c.timeIdx = clamp(c.timeIdx + d, 0, TIMES.length - 1)
      else if (c.sel === 2) c.offset = clamp(Math.round((c.offset + d * 0.1) * 10) / 10, -1, 1)
      else c.trig = clamp(c.trig + d, 0, TRIGS.length - 1)
    } else if (input === ' ' && c.sel === 3) c.trig = 1 - c.trig
    force()
  })

  useEffect(() => {
    let stopped = false
    const c = new EngineClient({ repoRoot: REPO, onError: (e) => setStatus('engine error: ' + e.message) })
    clientRef.current = c
    ;(async () => {
      try {
        tapsRef.current = await setupTapScope(c)
        if (tapsRef.current.length === 0) { setStatus('no taps — wire something to scope.<name>'); return }
        try { await c.call('start_audio', {}); setStatus(`▶ live · ${tapsRef.current.length} taps`) }
        catch { setStatus(`◼ no audio device · static window · ${tapsRef.current.length} taps`) }
      } catch (e: any) { setStatus('setup failed: ' + e.message); return }

      const w = Math.max(20, (process.stdout.columns ?? 84) - 4)
      const tick = async () => {
        if (stopped) return
        try {
          const k = ctl.current
          const taps = tapsRef.current
          const tap = taps[Math.min(k.srcIdx, taps.length - 1)]
          const display = TIMES[k.timeIdx]
          const search = k.trig === 0 ? 0 : display // up to one window to find a rising edge
          const count = display + search
          const p = (await c.call('playback_position', {})).position as number
          const desiredStart = Math.max(0, p - count)
          const renderStart = Math.max(0, desiredStart - WARMUP)
          const lead = desiredStart - renderStart
          const res = await c.call('render_window', { start: renderStart, count: lead + count, slots: [tap.slot] })
          const v = (res.values[0] as number[]).slice(lead)
          const off = k.trig === 1 ? Math.max(0, findTrigger(v, 0, search)) : 0
          setRows(renderWaveform(slice(v, off, display, k.offset), w, BAND_H))
          setPos(p)
        } catch { /* transient (mid hot-swap / patch reload) — skip frame */ }
        if (!stopped) setTimeout(tick, 1000 / FPS)
      }
      tick()
    })()
    return () => { stopped = true; c.kill() }
  }, [])

  const k = ctl.current
  const taps = tapsRef.current
  const tapName = taps.length ? taps[Math.min(k.srcIdx, taps.length - 1)].name : '—'
  const items = [
    ['source', `${tapName} (${k.srcIdx + 1}/${Math.max(1, taps.length)})`],
    ['timescale', String(TIMES[k.timeIdx])],
    ['offset', (k.offset >= 0 ? '+' : '') + k.offset.toFixed(1)],
    ['trigger', TRIGS[k.trig]],
  ] as const

  return (
    <Box flexDirection="column" paddingX={1}>
      <Box justifyContent="space-between">
        <Text backgroundColor="cyan" color="black" bold> tropical · scope </Text>
        <Text dimColor>{status}   ·   sample {pos}</Text>
      </Box>
      <Box height={1} />
      <Box flexDirection="column">
        <Text color="green" bold>{`scope.${tapName}`}</Text>
        {rows.map((r, i) => (<Text key={i} color="green">{r}</Text>))}
      </Box>
      <Box height={1} />
      <Box>
        {items.map(([label, val], i) => (
          <Text key={label}>
            {i === k.sel
              ? <Text backgroundColor="white" color="black" bold>{` ${label} ‹${val}› `}</Text>
              : <Text dimColor>{` ${label} ‹${val}› `}</Text>}
          </Text>
        ))}
        <Text dimColor>   ↑↓ select · ←→ adjust · space trigger · q quit</Text>
      </Box>
    </Box>
  )
}

render(<Scope />)
