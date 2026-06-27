// tropical · scope — a dead-simple 2-channel oscilloscope TUI.
//
// Each frame: read the engine's playback_position (the audio-paced master
// clock), render_window the two voice slots over the window ENDING at that
// position, and draw them as braille traces. Cadence is this app's own refresh
// (FPS); content is the master position — so with audio running the trace is
// locked to playback and doesn't drift, regardless of refresh jitter. With no
// audio device it shows a static window at sample 0.
//
//   bun tui/scope.tsx                 (spawns its own engine)
//   TROPICAL_SOCK=/path bun scope.tsx (attach to a running --serve engine)
import React, { useEffect, useState, useRef } from 'react'
import { render, Box, Text, useApp, useInput } from 'ink'
import { resolve } from 'node:path'
import { EngineClient, setupTwoVoice } from './client'
import { renderWaveform } from './braille'

const REPO = resolve(import.meta.dir, '..')
const FREQS: [number, number] = [220, 330]
const WINDOW = 1024 // samples per scope frame (~23 ms at 44.1 k)
const FPS = 30
const BAND_H = 5 // braille rows per channel

function Scope() {
  const { exit } = useApp()
  const [rows1, setRows1] = useState<string[]>([])
  const [rows2, setRows2] = useState<string[]>([])
  const [status, setStatus] = useState('connecting…')
  const [pos, setPos] = useState(0)
  const clientRef = useRef<EngineClient | null>(null)

  useInput((input, key) => {
    if (input === 'q' || key.escape) { clientRef.current?.kill(); exit() }
  })

  useEffect(() => {
    let stopped = false
    const c = new EngineClient({ repoRoot: REPO, onError: (e) => setStatus('engine error: ' + e.message) })
    clientRef.current = c
    ;(async () => {
      let slots: string[]
      try {
        slots = await setupTwoVoice(c, FREQS)
        try { await c.call('start_audio', {}); setStatus('▶ live · locked to audio') }
        catch { setStatus('◼ no audio device · static window') }
      } catch (e: any) { setStatus('setup failed: ' + e.message); return }

      const w = Math.max(20, (process.stdout.columns ?? 82) - 4)
      const tick = async () => {
        if (stopped) return
        try {
          const p = (await c.call('playback_position', {})).position as number
          const res = await c.call('render_window', { start: Math.max(0, p - WINDOW), count: WINDOW, slots })
          const [a, b] = res.values as number[][]
          setRows1(renderWaveform(a, w, BAND_H))
          setRows2(renderWaveform(b, w, BAND_H))
          setPos(p)
        } catch { /* transient (e.g. mid hot-swap) — skip this frame */ }
        if (!stopped) setTimeout(tick, 1000 / FPS)
      }
      tick()
    })()
    return () => { stopped = true; c.kill() }
  }, [])

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
        <Text dimColor>{status}   ·   sample {pos}   ·   q to quit</Text>
      </Box>
      <Box height={1} />
      <Band label={`CH1  ${FREQS[0]} Hz`} color="green" rows={rows1} />
      <Box height={1} />
      <Band label={`CH2  ${FREQS[1]} Hz`} color="magenta" rows={rows2} />
    </Box>
  )
}

render(<Scope />)
