import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('overlayAPI', {
  onScreenshot: (cb) => ipcRenderer.on('screenshot', (_e, dataUrl) => cb(dataUrl)),
  offScreenshot: () => ipcRenderer.removeAllListeners('screenshot'),
  complete: (croppedDataUrl) => ipcRenderer.send('overlay-complete', croppedDataUrl),
  cancel: () => ipcRenderer.send('overlay-cancel'),
  ready: () => ipcRenderer.send('overlay-ready')
})
