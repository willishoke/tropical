// JSON-RPC client over the engine's single-socket endpoint (`frontend --serve`).
//
// The engine owns one Unix-domain socket and routes by method name: set_param /
// get_telemetry are handled in C++ on the data plane (a plain `result`);
// everything else is forwarded to Lean on the control plane (the MCP-style
// `result.content[0].text` wrapping an inner {status,data}). This client speaks
// the same newline JSON-RPC either way and unwraps both shapes.
//
// By default it spawns its own engine on a private socket (one-command demo);
// set TROPICAL_SOCK=/path/to.sock to attach to an already-running --serve engine
// instead (the multi-client path). TROPICAL_ENGINE overrides the binary path.
import { spawn, type ChildProcess } from 'node:child_process'
import { connect, type Socket } from 'node:net'
import { resolve, join } from 'node:path'
import { tmpdir } from 'node:os'
import { unlinkSync } from 'node:fs'

type Pending = { resolve: (v: any) => void; reject: (e: any) => void }

export class EngineClient {
  private proc: ChildProcess | null = null
  private sock: Socket | null = null
  private sockPath: string
  private ownsEngine: boolean
  private engineErr = ''
  private buf = ''
  private pending = new Map<number, Pending>()
  private outbox: string[] = [] // requests issued before the socket is up
  private nextId = 1
  private connected = false
  private dead = false

  constructor(repoRoot: string) {
    const attach = process.env.TROPICAL_SOCK
    if (attach) {
      this.sockPath = attach
      this.ownsEngine = false
    } else {
      this.sockPath = join(tmpdir(), `tropical-tui-${process.pid}.sock`)
      this.ownsEngine = true
      const bin = process.env.TROPICAL_ENGINE ?? './lean/.lake/build/bin/frontend'
      this.proc = spawn(bin, ['--serve', this.sockPath], {
        cwd: repoRoot,
        stdio: ['ignore', 'ignore', 'pipe'], // engine diagnostics → buffer (shown only on failure)
      })
      this.proc.stderr!.on('data', (c: Buffer) => { this.engineErr += c.toString() })
      this.proc.on('exit', () => this.failAll(new Error('engine exited\n' + this.engineErr.trim())))
    }
    this.tryConnect(0)
  }

  private tryConnect(attempt: number) {
    if (this.dead) return
    const s = connect({ path: this.sockPath })
    s.on('connect', () => {
      this.sock = s
      this.connected = true
      for (const line of this.outbox) s.write(line)
      this.outbox = []
    })
    s.on('data', (c: Buffer) => this.onData(c.toString()))
    s.on('error', () => {
      try { s.destroy() } catch {}
      if (this.connected || this.dead) return
      // engine is still booting (stdlib load) or the path isn't bound yet — retry
      if (attempt < 120) setTimeout(() => this.tryConnect(attempt + 1), 150)
      else this.failAll(new Error(`could not connect to ${this.sockPath}\n` + this.engineErr.trim()))
    })
    s.on('close', () => { if (this.connected && !this.dead) this.failAll(new Error('engine connection closed')) })
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
        // control plane: inner {status:"ok",data} | {status:"error",error}
        try {
          const inner = JSON.parse(text)
          if (inner.status === 'ok') p.resolve(inner.data)
          else p.reject(new Error(`${inner.error?.code}: ${inner.error?.message}`))
        } catch (e) { p.reject(e) }
      } else {
        // data plane: plain result (set_param → {name,value}, get_telemetry → {...})
        p.resolve(r)
      }
    }
  }

  call(method: string, params: any = {}): Promise<any> {
    const id = this.nextId++
    const line = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    return new Promise<any>((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      if (this.connected && this.sock) this.sock.write(line)
      else this.outbox.push(line)
      setTimeout(() => {
        if (this.pending.has(id)) { this.pending.delete(id); reject(new Error(`timeout: ${method}`)) }
      }, 10000)
    })
  }

  kill() {
    this.dead = true
    try { this.sock?.destroy() } catch {}
    if (this.ownsEngine) {
      try { this.proc?.kill() } catch {}
      try { unlinkSync(this.sockPath) } catch {}
    }
  }
}

export const repoRoot = resolve(import.meta.dir, '../..')
