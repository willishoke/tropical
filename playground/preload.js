const { contextBridge, ipcRenderer } = require('electron')

// The entire bridge: one verb. The instrument lives in the renderer; the
// engine lives behind the socket; everything between is this line.
contextBridge.exposeInMainWorld('tropical', {
  call: (method, params) => ipcRenderer.invoke('rpc', method, params),
})
