// Electron main process.
//
// Owns the native `frontend` binary (the Lean compiler + session + runtime FFI)
// over its newline-delimited JSON-RPC surface (`--rpc`), and bridges it to the
// renderer over IPC. Audio plays out of the HOST's native device (RtAudio); the
// renderer is purely a control surface.
//
// Transport mirrors tui/scope/client.ts and mcp/wire_dac.test.ts: one JSON
// object per line on stdin; replies arrive on stdout, matched by `id`.
// Control-plane replies wrap an inner {status,data} in result.content[0].text;
// data-plane replies (set_param) are a plain `result`. We unwrap both.

const { app, BrowserWindow, ipcMain } = require('electron')
const { spawn } = require('node:child_process')
const path = require('node:path')

const REPO_ROOT = path.join(__dirname, '..')
const ENGINE_BIN = path.join(REPO_ROOT, 'lean/.lake/build/bin/frontend')

class Engine {
  constructor(onStatus) {
    this.onStatus = onStatus
    this.buf = ''
    this.pending = new Map()
    this.nextId = 1
    // The arrow patch-graph path does not route through syncCompile, but we set
    // the var anyway so any session-path compile in this process also takes the
    // session → resolved-root (arrow) path. Inert if the engine ignores it.
    const env = { ...process.env, TROPICAL_ARROW: '1' }
    this.proc = spawn(ENGINE_BIN, ['--rpc'], { cwd: REPO_ROOT, env, stdio: ['pipe', 'pipe', 'pipe'] })
    this.proc.stdout.on('data', (c) => this.onData(c.toString()))
    this.proc.stderr.on('data', (c) => this.onStatus({ kind: 'stderr', text: c.toString() }))
    this.proc.on('exit', (code) => {
      this.onStatus({ kind: 'exit', text: `engine exited (code ${code})` })
      for (const { reject } of this.pending.values()) reject(new Error('engine exited'))
      this.pending.clear()
    })
    this.onStatus({ kind: 'up', text: 'engine starting' })
  }

  onData(s) {
    this.buf += s
    let i
    while ((i = this.buf.indexOf('\n')) >= 0) {
      const line = this.buf.slice(0, i).trim()
      this.buf = this.buf.slice(i + 1)
      if (!line) continue
      let msg
      try { msg = JSON.parse(line) } catch { continue }
      if (typeof msg.id !== 'number') continue
      const p = this.pending.get(msg.id)
      if (!p) continue
      this.pending.delete(msg.id)
      if (msg.error) { p.reject(new Error(`RPC ${JSON.stringify(msg.error)}`)); continue }
      const r = msg.result
      const text = r && r.content && r.content[0] && r.content[0].text
      if (typeof text === 'string') {
        try {
          const inner = JSON.parse(text)
          if (inner.status === 'ok') p.resolve(inner.data)
          else p.reject(new Error(`${inner.error?.code}: ${inner.error?.message}`))
        } catch (e) { p.reject(e) }
      } else {
        p.resolve(r)
      }
    }
  }

  call(method, params = {}) {
    const id = this.nextId++
    const line = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
    // The compile+JIT is synchronous on the engine's one control thread, and a heavy
    // kernel (a modal reverb) can take tens of seconds. A short timeout gives up while
    // the engine is still grinding (the thread stays blocked, the late reply is
    // dropped), so heavy compiles get a long leash while every other call still fails
    // fast. NOTE: this is a diagnostic band-aid — the real fix is compiling off the
    // control thread so this never blocks at all.
    const timeoutMs = method === 'load_patch_graph' ? 180000 : 30000
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      try { this.proc.stdin.write(line) } catch (e) { this.pending.delete(id); reject(e); return }
      setTimeout(() => {
        if (this.pending.has(id)) { this.pending.delete(id); reject(new Error(`timeout: ${method}`)) }
      }, timeoutMs)
    })
  }

  kill() { try { this.proc.kill() } catch {} }
}

let engine = null
let win = null

function createWindow() {
  win = new BrowserWindow({
    width: 1400,
    height: 900,
    backgroundColor: '#16181d',
    title: 'tropical playground',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  })
  win.loadFile(path.join(__dirname, 'renderer', 'index.html'))

  const status = (s) => { if (win && !win.isDestroyed()) win.webContents.send('engine-status', s) }
  engine = new Engine(status)
}

ipcMain.handle('rpc', async (_e, { method, params }) => {
  if (!engine) throw new Error('engine not started')
  try {
    return { ok: true, data: await engine.call(method, params) }
  } catch (e) {
    return { ok: false, error: String(e.message || e) }
  }
})

app.whenReady().then(createWindow)
app.on('window-all-closed', () => { engine?.kill(); app.quit() })
app.on('before-quit', () => engine?.kill())
