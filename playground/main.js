// tropical demo — Electron main. Spawns one engine (`frontend --serve <sock>`)
// and bridges its newline JSON-RPC socket to the single-scene renderer.
const { app, BrowserWindow, ipcMain } = require('electron')
const { spawn } = require('node:child_process')
const { join } = require('node:path')
const { tmpdir } = require('node:os')
const { JsonLineRpcClient } = require('./rpc-client')

const REPO = join(__dirname, '..')
const SOCK = join(tmpdir(), `tropical-demo-${process.pid}.sock`)
const BIN = process.env.TROPICAL_ENGINE ?? join(REPO, 'lean/.lake/build/bin/frontend')

let engine = null
const controlRpc = new JsonLineRpcClient({ path: SOCK })
const scopeRpc = new JsonLineRpcClient({ path: SOCK })

function startEngine() {
  const useMetal = process.env.TROPICAL_DEMO_JIT !== '1'
  engine = spawn(BIN, ['--serve', SOCK], {
    cwd: REPO,
    env: {
      ...process.env,
      // Metal owns live audio for this dense modal scene. The engine keeps its
      // dual-loaded JIT artifact for the scopes' random-access reads.
      TROPICAL_BACKEND: useMetal ? 'metal' : '',
      // The fixed release candidate keeps the device callback short while the
      // worker renders deeper 512-frame tiles. Qualification may override the
      // two quanta independently; Rgpu must remain a multiple of Bdev.
      TROPICAL_BUFFER_LENGTH: useMetal
        ? (process.env.TROPICAL_DEMO_DEVICE_QUANTUM
          ?? process.env.TROPICAL_DEMO_QUANTUM
          ?? '128')
        : (process.env.TROPICAL_BUFFER_LENGTH ?? ''),
      TROPICAL_METAL_RENDER_TILE_FRAMES:
        useMetal
          ? (process.env.TROPICAL_DEMO_RENDER_QUANTUM
            ?? process.env.TROPICAL_METAL_RENDER_TILE_FRAMES
            ?? '512')
          : '',
    },
    stdio: ['ignore', 'ignore', 'pipe'],
  })
  engine.stderr.on('data', (d) => process.stderr.write(`[engine] ${d}`))
  engine.on('exit', (code) => {
    engine = null
    const error = new Error('engine exited')
    controlRpc.close(error)
    scopeRpc.close(error)
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

function call(method, params = {}) {
  return (method === 'render_window' ? scopeRpc : controlRpc)
    .call(method, params)
}

ipcMain.handle('rpc', (_e, method, params) => call(method, params))

app.whenReady().then(() => {
  startEngine()
  controlRpc.start()
  scopeRpc.start()
  const win = new BrowserWindow({
    width: 1240,
    height: 790,
    minWidth: 840,
    minHeight: 620,
    backgroundColor: '#0b0f12',
    title: 'tropical / night scene 01',
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
