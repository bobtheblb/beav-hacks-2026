import { contextBridge, ipcRenderer } from 'electron'

contextBridge.exposeInMainWorld('api', {
  openExternal: (url) => ipcRenderer.send('open-external', url),
  fetchImageUrl: (url) => ipcRenderer.invoke('fetch-image-url', url),
  captureFullScreen: () => ipcRenderer.invoke('capture-full-screen'),
  startSnip: () => ipcRenderer.invoke('start-snip'),
  analyzeImage: (dataUrl) => ipcRenderer.invoke('analyze-image', dataUrl),
  onImageCaptured: (cb) => ipcRenderer.on('image-captured', (_e, dataUrl) => cb(dataUrl)),
  onCaptureError: (cb) => ipcRenderer.on('capture-error', (_e, msg) => cb(msg)),
  onAnalysisStream: (cb) => ipcRenderer.on('analysis-stream', (_e, chunk) => cb(chunk)),
  removeListeners: () => {
    ipcRenderer.removeAllListeners('image-captured')
    ipcRenderer.removeAllListeners('capture-error')
    ipcRenderer.removeAllListeners('analysis-stream')
  }
})
