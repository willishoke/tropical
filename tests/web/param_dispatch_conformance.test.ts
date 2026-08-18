/**
 * param_dispatch_conformance.test.ts — the host-param-dispatch differential.
 *
 * design/host-param-dispatch.md: every runtime host dispatches param writes
 * itself from the plan's `param_disciplines` table; no client ever chooses a
 * write verb. Two implementations exist today:
 *
 *   - the Lean control plane (Engine.Audio's discipline implementations — the
 *     reference), reached over stdio RPC through the same public `set_param`;
 *   - the C++ socket data-plane (tropical_socket.cpp `set_param`, dispatching
 *     per the loaded plan's table against FlatRuntime's slots).
 *
 * This gate drives an IDENTICAL write sequence through both, on two separate
 * engine instances, pins the data plane's companion-slot values
 * (v0/v1/t0, #phase, tau_base), and requires the rendered output trajectory
 * to be byte-identical.
 * These are pure closed-form re-anchorings computed from the same inputs, so
 * they should typically be EXACTLY equal; a material difference means the C++
 * math diverged from the reference (fix the C++, do not widen the tolerance).
 *
 * TIMING: neither run starts audio. One explicit debug-render advances both
 * runtimes to the same 512-sample boundary before the writes, so every
 * discipline exercises nonzero timing math while remaining exactly comparable.
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

interface Client {
  call(method: string, params?: any): Promise<any>
  kill(): void
}

class SocketEngineClient implements Client {
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

// ── Minimal stdio RPC client ───────────────────────────────────────────────
// This reaches Engine.handleTool directly, so public `set_param` exercises the
// Lean reference dispatcher without retaining private method aliases.
class RpcEngineClient implements Client {
  private proc: ChildProcess
  private buf = ''
  private pending = new Map<number, Pending>()
  private nextId = 1

  constructor() {
    this.proc = spawn(frontendBin, ['--rpc'], {
      cwd: repoRoot,
      stdio: ['pipe', 'pipe', 'ignore'],
    })
    this.proc.stdout!.on('data', (c: Buffer) => this.onData(c.toString()))
    this.proc.on('exit', () => this.failAll(new Error('RPC engine exited')))
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
      const text = msg.result?.content?.[0]?.text
      if (typeof text !== 'string') {
        p.reject(new Error(`missing tool envelope: ${JSON.stringify(msg.result)}`))
        continue
      }
      try {
        const inner = JSON.parse(text)
        inner.status === 'ok'
          ? p.resolve(inner.data)
          : p.reject(new Error(`${inner.error?.code}: ${inner.error?.message}`))
      } catch (e) { p.reject(e) }
    }
  }

  call(method: string, params: any = {}): Promise<any> {
    const id = this.nextId++
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.proc.stdin!.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n')
      setTimeout(() => {
        if (this.pending.has(id)) {
          this.pending.delete(id)
          reject(new Error(`timeout: ${method}`))
        }
      }, 60_000)
    })
  }

  kill() {
    try { this.proc.kill() } catch {}
  }
}

// ── The patch under test ────────────────────────────────────────────────────
// source → sflange → out (playground vocabulary), covering all four
// disciplines in the plan's table:
//   src.freq        anchor    (#phase companion)
//   sfl.depth       glide     (#v0/#v1/#t0 plus exact t0 limbs, no base slot)
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
  'param:sfl.depth#t0#u0',
  'param:sfl.depth#t0#u1',
  'param:sfl.depth#t0#u2',
  'param:sfl.depth#t0#u3',
  'param:src.freq',
  'param:src.freq#phase',
  'param:master.velocity',
  'param:master.tau_base',
  'param:sfl.rate',
]

// Companion pre-seed: a glide ramp anchored at zero and a nonzero phase
// offset. The explicit render below advances `now`, so smoothstep has a
// nontrivial value to re-anchor from. Companion
// names are absent from the discipline table, so these dispatch as raw slot
// writes — identical in both runs by construction (setup, not the gate).
const seedWrites: Array<[string, number]> = [
  ['sfl.depth#v0', 0.001],
  ['sfl.depth#v1', 0.02],
  ['sfl.depth#t0', 0],
  ['src.freq#phase', 0.7],
]

// The write sequence under test (base names + targets).
const driveValues = { 'sfl.depth': 0.012, 'src.freq': 440, 'master.velocity': -0.5, 'sfl.rate': 1.5 }

async function loadAndSeed(c: Client, verifyPosition: boolean): Promise<string> {
  const report = await c.call('load_patch_graph', graph)
  expect(report.ok).toBe(true)
  // No DAC is running, so the master clock moves only when the test asks it to.
  if (verifyPosition) {
    const pos = await c.call('playback_position', {})
    expect(pos.position).toBe(0)
  }
  for (const [name, value] of seedWrites) await c.call('set_param', { name, value })
  // The out tap: the patch's final mix, render_window-readable.
  const outTap = (report.taps as Array<{ name: string; slot: string }>).find((t) => t.name === 'out')
  expect(outTap).toBeDefined()
  return outTap!.slot
}

async function readSlots(c: Client, names: string[]): Promise<number[]> {
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
    const a = new SocketEngineClient('data')
    // Run B: the Lean reference, via stdio and the same public method.
    const b = new RpcEngineClient()
    try {
      const [tapA, tapB] = await Promise.all([
        loadAndSeed(a, true),
        loadAndSeed(b, false),
      ])
      expect(tapA).toBe(tapB)

      // One process quantum places both runtimes at the same nonzero boundary.
      const [advanceA, advanceB] = await Promise.all([
        a.call('debug_render', { frames: 1 }),
        b.call('debug_render', { frames: 1 }),
      ])
      expect(advanceA.hex).toBe(advanceB.hex)
      const posA = await a.call('playback_position', {})
      expect(posA.position).toBe(512)

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
      // The data plane reports both the completed boundary it observed and
      // the first audible sample used by the discipline math. No DAC runs in
      // this differential, so both equal the explicit render boundary.
      for (const result of productionDispatches)
      {
        expect(result.observed_sample_index).toBe(512)
        expect(result.effective_sample_index).toBe(512)
      }

      await b.call('set_param', { name: 'sfl.depth', value: driveValues['sfl.depth'] })
      await b.call('set_param', { name: 'src.freq', value: driveValues['src.freq'] })
      await b.call('set_param', { name: 'master.velocity', value: driveValues['master.velocity'] })
      await b.call('set_param', { name: 'sfl.rate', value: driveValues['sfl.rate'] })

      // ── Pin the data-plane companion values themselves ───────────────────
      const slotsA = await readSlots(a, observedSlots)
      const v = Object.fromEntries(observedSlots.map((n, i) => [n, slotsA[i]]))
      // glide: dur = 0.01·SR (SR = 44100 for the arrow patch plan);
      // s = clamp((512 − 0)/441, 0, 1), curr = v0 + (v1−v0)·s²(3−2s).
      const s = Math.min(1, Math.max(0, 512 / (0.01 * 44100)))
      const expectedV0 = 0.001 + (0.02 - 0.001) * (s * s * (3 - 2 * s))
      expect(Math.abs(v['param:sfl.depth#v0'] - expectedV0)).toBeLessThanOrEqual(1e-12)
      expect(v['param:sfl.depth#v1']).toBe(driveValues['sfl.depth']) // target lands in v1
      expect(v['param:sfl.depth#t0']).toBe(512) // re-anchored to now
      expect(v['param:sfl.depth#t0#u0']).toBe(512)
      expect(v['param:sfl.depth#t0#u1']).toBe(0)
      expect(v['param:sfl.depth#t0#u2']).toBe(0)
      expect(v['param:sfl.depth#t0#u3']).toBe(0)
      // anchor: phase is corrected at the same nonzero boundary, then the
      // base frequency is written.
      expect(v['param:src.freq']).toBe(driveValues['src.freq'])
      const inc0 = Math.floor(220 * 4294967296 / 44100)
      const inc1 = Math.floor(440 * 4294967296 / 44100)
      const expectedPhase = ((0.7 + (inc0 - inc1) * 512 / 4294967296) % 1 + 1) % 1
      expect(Math.abs(v['param:src.freq#phase'] - expectedPhase)).toBeLessThanOrEqual(1e-12)
      // velocity: tau_base += (v0 − target)·now/SR.
      expect(v['param:master.velocity']).toBe(driveValues['master.velocity'])
      const expectedTauBase = (1 - driveValues['master.velocity']) * 512 / 44100
      expect(Math.abs(v['param:master.tau_base'] - expectedTauBase)).toBeLessThanOrEqual(1e-12)
      // raw dispatches as a plain base-slot write, exactly as before.
      expect(v['param:sfl.rate']).toBe(driveValues['sfl.rate'])

      // Still no base slot for the glided param after the write.
      await expect(readSlots(a, ['param:sfl.depth'])).rejects.toThrow(/unknown slot/)

      // ── Audible trajectory: both public dispatchers ⇒ bit-identical audio ─
      const [renderA, renderB] = await Promise.all([
        a.call('debug_render', { frames: 4 }),
        b.call('debug_render', { frames: 4 }),
      ])
      expect(renderA.hex).toBe(renderB.hex)
    } finally {
      a.kill()
      b.kill()
    }
  })
})
