"use strict";
const electron = require("electron");
electron.contextBridge.exposeInMainWorld("overlayAPI", {
  onScreenshot: (cb) => electron.ipcRenderer.on("screenshot", (_e, dataUrl) => cb(dataUrl)),
  offScreenshot: () => electron.ipcRenderer.removeAllListeners("screenshot"),
  complete: (croppedDataUrl) => electron.ipcRenderer.send("overlay-complete", croppedDataUrl),
  cancel: () => electron.ipcRenderer.send("overlay-cancel"),
  ready: () => electron.ipcRenderer.send("overlay-ready")
});
