// tropical demo — Electron main. Spawns ONE engine (`frontend --serve <sock>`,
// Metal backend + off-RT epoch rendering: this app doubles as a GPU production
// soak. It speaks newline JSON-RPC over the Unix socket (the same single-socket
// control/data-plane split the TUIs used), and bridges it to the renderer over
// IPC. The renderer is the whole instrument; this file is plumbing.
const { app, BrowserWindow, ipcMain } = require('electron')
const { spawn } = require('node:child_process')
const { connect } = require('node:net')
const { join } = require('node:path')
const { tmpdir } = require('node:os')

const REPO = join(__dirname, '..')
const SOCK = join(tmpdir(), `tropical-demo-${process.pid}.sock`)
const BIN = process.env.TROPICAL_ENGINE ?? join(REPO, 'lean/.lake/build/bin/frontend')

let engine = null
let sock = null
let buf = ''
let nextId = 1
const pending = new Map()
const outbox = []

function startEngine() {
  engine = spawn(BIN, ['--serve', SOCK], {
    cwd: REPO,
    env: {
      ...process.env,
      // The demo runs on the GPU: every patch edit dual-loads (JIT stays the
      // scope/render_window reference), while a worker prepares Metal tiles.
      TROPICAL_BACKEND: process.env.TROPICAL_DEMO_JIT ? '' : 'metal',
      TROPICAL_METAL_RENDER_TILE_FRAMES:
        process.env.TROPICAL_DEMO_JIT ? '' : '512',
    },
    stdio: ['ignore', 'ignore', 'pipe'],
  })
  engine.stderr.on('data', (d) => process.stderr.write(`[engine] ${d}`))
  engine.on('exit', (code) => {
    engine = null
    for (const { reject } of pending.values()) reject(new Error('engine exited'))
    pending.clear()
    if (!app.isQuitting) console.error(`engine exited (${code})`)
  })
}

// Kill the engine child. SIGTERM (the engine reaps it cleanly), with a SIGKILL
// fallback in case it's wedged mid-compile. Idempotent: `engine` is nulled on
// exit, so repeated calls are no-ops.
function stopEngine() {
  const child = engine
  if (!child) return
  try { child.kill('SIGTERM') } catch {}
  const t = setTimeout(() => { try { child.kill('SIGKILL') } catch {} }, 2000)
  child.on('exit', () => clearTimeout(t))
}

function tryConnect(attempt = 0) {
  const s = connect({ path: SOCK })
  s.on('connect', () => {
    sock = s
    for (const l of outbox) s.write(l)
    outbox.length = 0
  })
  s.on('data', (c) => {
    buf += c.toString()
    let i
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i).trim()
      buf = buf.slice(i + 1)
      if (!line) continue
      let msg
      try { msg = JSON.parse(line) } catch { continue }
      const p = pending.get(msg.id)
      if (!p) continue
      pending.delete(msg.id)
      if (msg.error) { p.reject(new Error(JSON.stringify(msg.error))); continue }
      const r = msg.result
      const text = r?.content?.[0]?.text
      if (typeof text === 'string') {
        // control plane: MCP envelope wrapping {status, data|error}
        try {
          const inner = JSON.parse(text)
          inner.status === 'ok'
            ? p.resolve(inner.data)
            : p.reject(new Error(`${inner.error?.code}: ${inner.error?.message}`))
        } catch (e) { p.reject(e) }
      } else p.resolve(r)   // data plane: plain result (render_window, playback_position)
    }
  })
  s.on('error', () => {
    try { s.destroy() } catch {}
    if (sock) return
    if (attempt < 200) setTimeout(() => tryConnect(attempt + 1), 100)
  })
}

function call(method, params = {}) {
  const id = nextId++
  const line = JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n'
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    sock ? sock.write(line) : outbox.push(line)
    // load_patch_graph compiles the whole circuit into one kernel — the modal
    // reverb's first (cache-cold) LLVM compile runs minutes-scale; be patient.
    const ms = method === 'load_patch_graph' ? 300000 : 15000
    setTimeout(() => {
      if (pending.has(id)) { pending.delete(id); reject(new Error(`timeout: ${method}`)) }
    }, ms)
  })
}

ipcMain.handle('rpc', (_e, method, params) => call(method, params))

app.whenReady().then(() => {
  startEngine()
  tryConnect()
  const win = new BrowserWindow({
    width: 1440,
    height: 900,
    backgroundColor: '#f2f0e8',
    title: 'tropical',
    webPreferences: { preload: join(__dirname, 'preload.js') },
  })
  win.loadFile(join(__dirname, 'renderer/index.html'))
})

// Engine teardown lives in `before-quit`, NOT `window-all-closed`: the latter
// is documented as *not emitted* when the app quits via Cmd+Q or app.quit(), so
// cleanup hung there leaks the engine on the normal macOS quit path (it gets
// reparented to launchd and orphaned). before-quit fires on every quit path.
app.on('before-quit', () => {
  app.isQuitting = true
  stopEngine()
})

// Closing the window quits the app (single-window instrument); this in turn
// fires before-quit, which does the actual engine kill.
app.on('window-all-closed', () => {
  app.quit()
})

// If the main process itself is signalled (e.g. killed from a terminal), Electron
// won't run before-quit — reap the child ourselves so it never outlives us.
for (const sig of ['SIGINT', 'SIGTERM', 'SIGHUP']) {
  process.on(sig, () => { stopEngine(); app.quit() })
}
