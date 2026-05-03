"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("api", {
  openExternal: (url) => electron.ipcRenderer.send("open-external", url),
  fetchImageUrl: (url) => electron.ipcRenderer.invoke("fetch-image-url", url),
  captureFullScreen: () => electron.ipcRenderer.invoke("capture-full-screen"),
  startSnip: () => electron.ipcRenderer.invoke("start-snip"),
  analyzeImage: (dataUrl) => electron.ipcRenderer.invoke("analyze-image", dataUrl),
  onImageCaptured: (cb) => electron.ipcRenderer.on("image-captured", (_e, dataUrl) => cb(dataUrl)),
  onCaptureError: (cb) => electron.ipcRenderer.on("capture-error", (_e, msg) => cb(msg)),
  onAnalysisStream: (cb) => electron.ipcRenderer.on("analysis-stream", (_e, chunk) => cb(chunk)),
  removeListeners: () => {
    electron.ipcRenderer.removeAllListeners("image-captured");
    electron.ipcRenderer.removeAllListeners("capture-error");
    electron.ipcRenderer.removeAllListeners("analysis-stream");
  }
});
