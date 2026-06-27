// Minimal JSON-RPC client over the engine's `frontend --serve` Unix socket.
// Control-plane replies wrap an inner {status,data} in result.content[0].text;
// data-plane replies (render_window, playback_position) are a plain `result`.
// This unwraps both. By default it spawns its own engine; set TROPICAL_SOCK to
// attach to a running one (e.g. one already playing audio).
import { spawn, type ChildProcess } from 'node:child_process'
import { connect, type Socket } from 'node:net'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

type Pending = { resolve: (v: any) => void; reject: (e: any) => void }

export class EngineClient {
  private proc: ChildProcess | null = null
  private sock: Socket | null = null
  private sockPath: string
  private ownsEngine: boolean
  private buf = ''
  private pending = new Map<number, Pending>()
  private outbox: string[] = []
  private nextId = 1
  private connected = false
  private dead = false
  private onErr: (e: Error) => void

  constructor(opts: { repoRoot?: string; onError?: (e: Error) => void } = {}) {
    this.onErr = opts.onError ?? (() => {})
    const attach = process.env.TROPICAL_SOCK
    if (attach) { this.sockPath = attach; this.ownsEngine = false }
    else {
      this.sockPath = join(tmpdir(), `tropical-scope-${process.pid}.sock`)
      this.ownsEngine = true
      const bin = process.env.TROPICAL_ENGINE ?? './lean/.lake/build/bin/frontend'
      this.proc = spawn(bin, ['--serve', this.sockPath], { cwd: opts.repoRoot ?? process.cwd(), stdio: ['ignore', 'ignore', 'ignore'] })
      this.proc.on('exit', () => this.failAll(new Error('engine exited')))
    }
    this.tryConnect(0)
  }

  private tryConnect(attempt: number) {
    if (this.dead) return
    const s = connect({ path: this.sockPath })
    s.on('connect', () => { this.sock = s; this.connected = true; for (const l of this.outbox) s.write(l); this.outbox = [] })
    s.on('data', (c: Buffer) => this.onData(c.toString()))
    s.on('error', () => {
      try { s.destroy() } catch {}
      if (this.connected || this.dead) return
      if (attempt < 200) setTimeout(() => this.tryConnect(attempt + 1), 100)
      else this.failAll(new Error(`could not connect to ${this.sockPath}`))
    })
  }

  private failAll(e: Error) { for (const { reject } of this.pending.values()) reject(e); this.pending.clear(); if (!this.dead) this.onErr(e) }

  private onData(s: string) {
    this.buf += s
    let i: number
    while ((i = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, i).trim(); this.buf = this.buf.slice(i + 1)
      if (!line) continue
      let msg: any; try { msg = JSON.parse(line) } catch { continue }
      if (typeof msg.id !== 'number') continue
      const p = this.pending.get(msg.id); if (!p) continue
      this.pending.delete(msg.id)
      if (msg.error) { p.reject(new Error(`RPC: ${JSON.stringify(msg.error)}`)); continue }
      const r = msg.result
      const text = r?.content?.[0]?.text
      if (typeof text === 'string') {
        try { const inner = JSON.parse(text); inner.status === 'ok' ? p.resolve(inner.data) : p.reject(new Error(`${inner.error?.code}: ${inner.error?.message}`)) } catch (e) { p.reject(e) }
      } else p.resolve(r)
    }
  }

  call(method: string, params: any = {}): Promise<any> {
    const id = this.nextId++
    const line = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      if (this.connected && this.sock) this.sock.write(line); else this.outbox.push(line)
      setTimeout(() => { if (this.pending.has(id)) { this.pending.delete(id); reject(new Error(`timeout: ${method}`)) } }, 10000)
    })
  }

  kill() { this.dead = true; try { this.sock?.destroy() } catch {}; if (this.ownsEngine) { try { this.proc?.kill() } catch {} } }
}

/**
 * Build the two-voice morph demo session: two stateless `MorphOsc` voices (each
 * crossfading saw↔sin) at `freqs`, both summed to the DAC. The morph is driven by
 * a slow in-patch LFO — `morph = (lfo.sine + 1)/2` at `morphHz` — so the timbre
 * sweeps saw↔sin over time entirely inside the patch (no param, fully stateless).
 * Because morph is itself a function of the sample index, `render_window` shows
 * the exact morphed waveform at any point in time. Returns the two output slots.
 */
export async function setupMorphScope(
  c: EngineClient,
  freqs: [number, number] = [220, 330],
  morphHz = 0.15,
): Promise<string[]> {
  await c.call('add_instance', { program: 'FixedSinOsc', instance_name: 'lfo' })
  await c.call('add_instance', { program: 'MorphOsc', instance_name: 'osc1' })
  await c.call('add_instance', { program: 'MorphOsc', instance_name: 'osc2' })
  const morphExpr = { op: 'mul', args: [{ op: 'add', args: [{ op: 'ref', instance: 'lfo', output: 'sine' }, 1] }, 0.5] }
  await c.call('wire', { set: [
    { instance: 'lfo', input: 'freq', expr: morphHz },
    { instance: 'osc1', input: 'freq', expr: freqs[0] },
    { instance: 'osc2', input: 'freq', expr: freqs[1] },
    { instance: 'osc1', input: 'morph', expr: morphExpr },
    { instance: 'osc2', input: 'morph', expr: morphExpr },
    { instance: 'dac', input: 'out', expr: { op: 'ref', instance: 'osc1', output: 'out' } },
    { instance: 'dac', input: 'out', expr: { op: 'ref', instance: 'osc2', output: 'out' } },
  ] })
  return ['osc1.out', 'osc2.out']
}
