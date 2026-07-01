// Bridge: expose a minimal, audited surface to the renderer. No Node in the
// renderer; everything goes through `window.tropical`.

const { contextBridge, ipcRenderer } = require('electron')

contextBridge.exposeInMainWorld('tropical', {
  // Returns { ok: true, data } or { ok: false, error }.
  rpc: (method, params) => ipcRenderer.invoke('rpc', { method, params }),
  onStatus: (cb) => ipcRenderer.on('engine-status', (_e, s) => cb(s)),
})
