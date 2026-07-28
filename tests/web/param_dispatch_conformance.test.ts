/**
 * param_dispatch_conformance.test.ts — the host-param-dispatch differential.
 *
 * design/host-param-dispatch.md: every runtime host dispatches param writes
 * itself from the plan's `param_disciplines` table; no client ever chooses a
 * write verb. Two implementations exist today:
 *
 *   - the Lean control plane (Engine.lean handleSetParamGlide / Freq /
 *     Velocity — the reference), reached over the socket via the alias verbs
 *     `set_param_glide` / `set_param_freq` / `set_param_velocity`;
 *   - the C++ socket data-plane (tropical_socket.cpp `set_param`, dispatching
 *     per the loaded plan's table against FlatRuntime's slots).
 *
 * This gate drives an IDENTICAL write sequence through both, on two separate
 * engine instances, and requires the companion-slot values (v0/v1/t0, #phase,
 * tau_base) and the audible output trajectory to match to near-identity.
 * These are pure closed-form re-anchorings computed from the same inputs, so
 * they should typically be EXACTLY equal; a material difference means the C++
 * math diverged from the reference (fix the C++, do not widen the tolerance).
 *
 * TIMING: neither run starts audio, so current_sample_index is frozen at 0
 * for both — `now` is identical and the trajectories are exactly comparable
 * (verified via `playback_position` before driving writes). With now = 0 the
 * anchor/velocity bumps multiply by zero by construction, so the glide's
 * smoothstep is exercised numerically by pre-seeding its companions (t0 < 0:
 * a ramp already in flight) through raw writes identical in both runs.
 */

import { describe, test, expect, setDefaultTimeout } from 'bun:test'
import { spawn, type ChildProcess } from 'node:child_process'
import { connect, type Socket } from 'node:net'
import { existsSync } from 'node:fs'
import { join, resolve, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

// Each run spawns a full engine and compiles a patch through LLVM; give the
// file integration-gate headroom, not unit-test headroom.
setDefaultTimeout(120_000)

const __dirname = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(__dirname, '../..')
// The built binary, never `lake exe` (the lake wrapper's DYLD environment
// shadows Homebrew's libLLVM by leaf name and the process dies at load).
const frontendBin = resolve(repoRoot, 'lean/.lake/build/bin/frontend')

// ── Minimal socket client (the tui/scope/client.ts pattern) ────────────────
// Control-plane replies wrap an inner {status,data} in result.content[0].text;
// data-plane replies (set_param, render_window, playback_position) are a
// plain `result`. Unwraps both.
type Pending = { resolve: (v: any) => void; reject: (e: any) => void }

class EngineClient {
  private proc: ChildProcess
  private sock: Socket | null = null
  private sockPath: string
  private buf = ''
  private pending = new Map<number, Pending>()
  private outbox: string[] = []
  private nextId = 1
  private connected = false
  private dead = false

  constructor(tag: string) {
    this.sockPath = join(tmpdir(), `tropical-conform-${process.pid}-${tag}.sock`)
    this.proc = spawn(frontendBin, ['--serve', this.sockPath], {
      cwd: repoRoot,
      stdio: ['ignore', 'ignore', 'ignore'],
    })
    this.proc.on('exit', () => this.failAll(new Error('engine exited')))
    void this.connectLoop()
  }

  // Wait for the engine to bind its socket node, then connect. Polling for
  // the file first avoids racing connect() against the bind (bun raises the
  // ENOENT synchronously, unlike node).
  private async connectLoop() {
    for (let attempt = 0; attempt < 300 && !this.dead; attempt++) {
      if (existsSync(this.sockPath)) {
        const ok = await new Promise<boolean>((res) => {
          const s = connect({ path: this.sockPath })
          s.on('connect', () => {
            this.sock = s
            this.connected = true
            for (const l of this.outbox) s.write(l)
            this.outbox = []
            res(true)
          })
          s.on('data', (c: Buffer) => this.onData(c.toString()))
          s.on('error', () => {
            try { s.destroy() } catch {}
            res(false)
          })
        })
        if (ok) return
      }
      await new Promise((r) => setTimeout(r, 100))
    }
    if (!this.dead) this.failAll(new Error(`could not connect to ${this.sockPath}`))
  }

  private failAll(e: Error) {
    for (const { reject } of this.pending.values()) reject(e)
    this.pending.clear()
  }

  private onData(s: string) {
    this.buf += s
    let i: number
    while ((i = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, i).trim()
      this.buf = this.buf.slice(i + 1)
      if (!line) continue
      let msg: any
      try { msg = JSON.parse(line) } catch { continue }
      if (typeof msg.id !== 'number') continue
      const p = this.pending.get(msg.id)
      if (!p) continue
      this.pending.delete(msg.id)
      if (msg.error) { p.reject(new Error(`RPC: ${JSON.stringify(msg.error)}`)); continue }
      const r = msg.result
      const text = r?.content?.[0]?.text
      if (typeof text === 'string') {
        try {
          const inner = JSON.parse(text)
          inner.status === 'ok'
            ? p.resolve(inner.data)
            : p.reject(new Error(`${inner.error?.code}: ${inner.error?.message}`))
        } catch (e) { p.reject(e) }
      } else p.resolve(r)
    }
  }

  call(method: string, params: any = {}): Promise<any> {
    const id = this.nextId++
    const line = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      if (this.connected && this.sock) this.sock.write(line)
      else this.outbox.push(line)
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id)
          reject(new Error(`timeout: ${method}`))
        }
      }, 60_000)
    })
  }

  kill() {
    this.dead = true
    try { this.sock?.destroy() } catch {}
    try { this.proc.kill() } catch {}
  }
}

// ── The patch under test ────────────────────────────────────────────────────
// source → sflange → out (playground vocabulary), covering all four
// disciplines in the plan's table:
//   src.freq        anchor    (#phase companion)
//   sfl.depth       glide     (#v0/#v1/#t0 companions, no base slot)
//   sfl.rate        raw
//   master.velocity velocity  (master.tau_base companion; always in the table)
const graph = {
  nodes: [
    { id: 'src', kind: 'source', params: { freq: 220, morph: 0 }, sel: {}, in: { freq: [] } },
    { id: 'sfl', kind: 'sflange', params: { depth: 0.006, rate: 0.3 }, sel: {}, in: { in: ['src'], mod: [] } },
    { id: 'out', kind: 'out', params: {}, sel: {}, in: { in: ['sfl'] } },
  ],
  out: 'out', // MANDATORY: the reachability root
  taps: true, // materialize render_window-readable taps per node
}

// The companion/base slots whose post-sequence values the two runs must agree
// on. All are param slots; render_window resolves ANY slot name via
// slot_index, so a 1-sample window reads them directly.
const observedSlots = [
  'param:sfl.depth#v0',
  'param:sfl.depth#v1',
  'param:sfl.depth#t0',
  'param:src.freq',
  'param:src.freq#phase',
  'param:master.velocity',
  'param:master.tau_base',
  'param:sfl.rate',
]

// Companion pre-seed: a glide ramp already in flight (t0 in the past — with
// now frozen at 0 that means t0 < 0), and a nonzero phase offset, so the
// smoothstep evaluation has a nontrivial value to re-anchor from. Companion
// names are absent from the discipline table, so these dispatch as raw slot
// writes — identical in both runs by construction (setup, not the gate).
const seedWrites: Array<[string, number]> = [
  ['sfl.depth#v0', 0.001],
  ['sfl.depth#v1', 0.02],
  ['sfl.depth#t0', -400],
  ['src.freq#phase', 0.7],
]

// The write sequence under test (base names + targets).
const driveValues = { 'sfl.depth': 0.012, 'src.freq': 440, 'master.velocity': -0.5, 'sfl.rate': 1.5 }

async function loadAndSeed(c: EngineClient): Promise<string> {
  const report = await c.call('load_patch_graph', graph)
  expect(report.ok).toBe(true)
  // No DAC is running, so the master clock is frozen: `now` = 0 for every
  // write in both runs, making the re-anchorings exactly comparable.
  const pos = await c.call('playback_position', {})
  expect(pos.position).toBe(0)
  for (const [name, value] of seedWrites) await c.call('set_param', { name, value })
  // The out tap: the patch's final mix, render_window-readable.
  const outTap = (report.taps as Array<{ name: string; slot: string }>).find((t) => t.name === 'out')
  expect(outTap).toBeDefined()
  return outTap!.slot
}

async function readSlots(c: EngineClient, names: string[]): Promise<number[]> {
  const r = await c.call('render_window', { start: 0, count: 1, slots: names })
  return (r.values as number[][]).map((ch) => ch[0])
}

describe('param dispatch conformance (C++ data-plane ≡ Lean reference)', () => {
  test('identical write sequence → identical companion slots and audio', async () => {
    if (!existsSync(frontendBin))
      throw new Error(`frontend binary missing: ${frontendBin} (run \`make lean\`)`)

    // Run A: the C++ socket data-plane. `set_param` over the socket never
    // reaches Lean — handle_line routes it to handle_data, which dispatches
    // per the loaded plan's param_disciplines table.
    const a = new EngineClient('data')
    // Run B: the Lean reference, via the control-plane alias verbs.
    const b = new EngineClient('ctrl')
    try {
      const [tapA, tapB] = await Promise.all([loadAndSeed(a), loadAndSeed(b)])
      expect(tapA).toBe(tapB)

      // The glided param has NO base slot — only companions. render_window
      // must reject it before AND after the write (the C++ dispatch must not
      // create or write one).
      await expect(readSlots(a, ['param:sfl.depth'])).rejects.toThrow(/unknown slot/)

      // Drive the identical sequence.
      const productionDispatches = [
        await a.call('set_param', { name: 'sfl.depth', value: driveValues['sfl.depth'] }),
        await a.call('set_param', { name: 'src.freq', value: driveValues['src.freq'] }),
        await a.call('set_param', { name: 'master.velocity', value: driveValues['master.velocity'] }),
        await a.call('set_param', { name: 'sfl.rate', value: driveValues['sfl.rate'] }),
      ]
      // The data plane reports the exact completed boundary used by each
      // production dispatch. No DAC runs in this differential, so all four
      // must have applied at the frozen zero boundary.
      for (const result of productionDispatches)
        expect(result.applied_sample_index).toBe(0)

      await b.call('set_param_glide', { name: 'sfl.depth', value: driveValues['sfl.depth'] })
      await b.call('set_param_freq', { name: 'src.freq', value: driveValues['src.freq'] })
      await b.call('set_param_velocity', { name: 'master.velocity', value: driveValues['master.velocity'] })
      // raw has no control-plane alias verb; the raw data-plane path is
      // asserted against its own written value below.
      await b.call('set_param', { name: 'sfl.rate', value: driveValues['sfl.rate'] })

      // ── The differential: companion trajectories must match ──────────────
      const [slotsA, slotsB] = await Promise.all([
        readSlots(a, observedSlots),
        readSlots(b, observedSlots),
      ])
      for (let i = 0; i < observedSlots.length; i++) {
        // Closed-form re-anchorings from the same inputs: typically exactly
        // equal; 1e-9 absorbs float noise only.
        expect(Math.abs(slotsA[i] - slotsB[i]),
          `${observedSlots[i]}: data-plane ${slotsA[i]} vs reference ${slotsB[i]}`,
        ).toBeLessThanOrEqual(1e-9)
      }

      // ── Pin the values themselves (not just agreement) ────────────────────
      const v = Object.fromEntries(observedSlots.map((n, i) => [n, slotsA[i]]))
      // glide: dur = 0.02·SR (SR = 44100 for the arrow patch plan);
      // s = clamp((0 − (−400))/882, 0, 1), curr = v0 + (v1−v0)·s²(3−2s).
      const s = Math.min(1, Math.max(0, (0 - -400) / (0.02 * 44100)))
      const expectedV0 = 0.001 + (0.02 - 0.001) * (s * s * (3 - 2 * s))
      expect(Math.abs(v['param:sfl.depth#v0'] - expectedV0)).toBeLessThanOrEqual(1e-12)
      expect(v['param:sfl.depth#v1']).toBe(driveValues['sfl.depth']) // target lands in v1
      expect(v['param:sfl.depth#t0']).toBe(0) // re-anchored to now
      // anchor: Δφ = (inc0 − inc1)·now/2³² = 0 at now = 0 — phase unchanged,
      // base written after the bump.
      expect(v['param:src.freq']).toBe(driveValues['src.freq'])
      expect(v['param:src.freq#phase']).toBe(0.7)
      // velocity: tau_base += (v0 − target)·now/SR = 0 at now = 0.
      expect(v['param:master.velocity']).toBe(driveValues['master.velocity'])
      expect(v['param:master.tau_base']).toBe(0)
      // raw dispatches as a plain base-slot write, exactly as before.
      expect(v['param:sfl.rate']).toBe(driveValues['sfl.rate'])

      // Still no base slot for the glided param after the write.
      await expect(readSlots(a, ['param:sfl.depth'])).rejects.toThrow(/unknown slot/)

      // ── Audible trajectory: identical slot state ⇒ bit-identical audio ────
      const winA = await a.call('render_window', { start: 0, count: 512, slots: [tapA] })
      const winB = await b.call('render_window', { start: 0, count: 512, slots: [tapB] })
      expect(winA.values[0]).toEqual(winB.values[0])
    } finally {
      a.kill()
      b.kill()
    }
  })
})
