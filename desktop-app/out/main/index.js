"use strict";
const electron = require("electron");
const path = require("path");
const OpenAI = require("openai");
const isDev = process.env.NODE_ENV === "development";
electron.app.setName("Artifact");
let mainWindow = null;
let overlayWindow = null;
let tray = null;
const config = {
  endpoint: "http://129.213.148.162:8000/v1",
  model: "joshnegreanu/Nemo-12B-VL-AI-Detection-Fine-Tuned",
  apiKey: "asdf",
  systemPrompt: "You must always begin your response with a <think>...</think> block describing your visual analysis, then output ONLY a JSON object after </think>. Responses without <think> are invalid.",
  userPrompt: 'Analyze the provided image and determine whether it is AI-generated. You are helping an individual learn how to pick up on cues that would help inform them in spotting AI generated images.\nFirst reason inside <think>...</think> tags about the visual cues you observe.\nAfter </think>, output ONLY a JSON object with these fields:\n  - status: either "ai_generated" or "real"\n  - confidence: a calibrated number between 0 and 1.\n      * Reserve 1.0 ONLY for unmistakable evidence (e.g. obvious GAN artifacts, impossible anatomy).\n      * Use 0.85-0.95 for clear cases.\n      * Use 0.6-0.8 when there are some ambiguous details.\n      * Use 0.5-0.6 when the image is genuinely hard to classify.\n      * Never use exactly 1.0 unless you would bet money on it.\n  - reason: a brief explanation of your decision'
};
function createMainWindow() {
  mainWindow = new electron.BrowserWindow({
    width: 480,
    height: 640,
    minWidth: 380,
    minHeight: 480,
    show: false,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    webPreferences: {
      preload: path.join(__dirname, "../preload/index.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  mainWindow.on("ready-to-show", () => mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  if (isDev && process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }
}
function createOverlayWindow() {
  overlayWindow = new electron.BrowserWindow({
    frame: false,
    transparent: true,
    alwaysOnTop: true,
    skipTaskbar: true,
    hasShadow: false,
    roundedCorners: false,
    resizable: false,
    movable: false,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "../preload/overlay.js"),
      contextIsolation: true,
      nodeIntegration: false
    }
  });
  overlayWindow.setAlwaysOnTop(true, "screen-saver");
  overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
  overlayWindow.on("closed", () => {
    overlayWindow = null;
  });
  if (isDev && process.env.ELECTRON_RENDERER_URL) {
    const base = process.env.ELECTRON_RENDERER_URL.replace(/\/$/, "");
    overlayWindow.loadURL(`${base}/overlay.html`);
  } else {
    overlayWindow.loadFile(path.join(__dirname, "../renderer/overlay.html"));
  }
}
async function captureScreen() {
  if (process.platform === "darwin") {
    const status = electron.systemPreferences.getMediaAccessStatus("screen");
    if (status !== "granted") {
      electron.shell.openExternal(
        "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture"
      );
      throw new Error(
        `Screen Recording permission required.
System Settings → Privacy & Security → Screen Recording
Enable "Electron" in the list, then restart the app.`
      );
    }
  }
  let sources;
  try {
    sources = await electron.desktopCapturer.getSources({
      types: ["screen"],
      thumbnailSize: { width: 2560, height: 1440 }
    });
  } catch (e) {
    throw new Error(`Screen capture failed: ${e.message}`);
  }
  if (!sources.length)
    throw new Error("No screen sources found.");
  return sources[0].thumbnail.toDataURL();
}
function waitForOverlayReady() {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 2e3);
    electron.ipcMain.once("overlay-ready", () => {
      clearTimeout(timer);
      resolve();
    });
  });
}
async function startSnip() {
  try {
    mainWindow?.hide();
    await new Promise((r) => setTimeout(r, 200));
    const dataUrl = await captureScreen();
    const isNew = !overlayWindow;
    if (isNew)
      createOverlayWindow();
    if (isNew) {
      await waitForOverlayReady();
    }
    const { bounds } = electron.screen.getDisplayNearestPoint(electron.screen.getCursorScreenPoint());
    overlayWindow.setBounds(bounds);
    overlayWindow.show();
    overlayWindow.setBounds(bounds);
    overlayWindow.focus();
    overlayWindow.webContents.send("screenshot", dataUrl);
  } catch (err) {
    mainWindow?.show();
    mainWindow?.webContents.send("capture-error", err.message);
  }
}
electron.app.whenReady().then(() => {
  if (process.platform === "darwin" && electron.app.dock) {
    const iconPath = path.join(__dirname, "../../build/icon.png");
    try {
      electron.app.dock.setIcon(electron.nativeImage.createFromPath(iconPath));
    } catch {
    }
  }
  createMainWindow();
  try {
    const icon = electron.nativeImage.createFromDataURL(
      "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="
    );
    tray = new electron.Tray(icon);
    if (process.platform === "darwin")
      tray.setTitle("⌕");
    tray.setToolTip("Finding Nemo");
    tray.setContextMenu(
      electron.Menu.buildFromTemplate([
        { label: "Show", click: () => mainWindow?.show() },
        { label: "Snip Region", click: () => startSnip() },
        { type: "separator" },
        { label: "Quit", click: () => electron.app.quit() }
      ])
    );
    tray.on("click", () => mainWindow?.show());
  } catch {
  }
  electron.globalShortcut.register("CommandOrControl+Shift+S", () => startSnip());
  electron.app.on("activate", () => {
    if (!mainWindow)
      createMainWindow();
    else
      mainWindow.show();
  });
});
electron.app.on("window-all-closed", () => {
  if (process.platform !== "darwin")
    electron.app.quit();
});
electron.app.on("will-quit", () => {
  electron.globalShortcut.unregisterAll();
});
electron.ipcMain.on("open-external", async (_e, url) => {
  try {
    await electron.shell.openExternal(url);
  } catch (e) {
    console.error("openExternal failed:", e);
  }
});
electron.ipcMain.handle("fetch-image-url", async (_e, url) => {
  const res = await fetch(url);
  if (!res.ok)
    throw new Error(`HTTP ${res.status} — could not load image`);
  const contentType = res.headers.get("content-type") || "image/jpeg";
  if (!contentType.startsWith("image/"))
    throw new Error("URL does not point to an image");
  const buf = Buffer.from(await res.arrayBuffer());
  return `data:${contentType};base64,${buf.toString("base64")}`;
});
electron.ipcMain.handle("capture-full-screen", async () => {
  mainWindow?.hide();
  await new Promise((r) => setTimeout(r, 200));
  try {
    return await captureScreen();
  } finally {
    mainWindow?.show();
  }
});
electron.ipcMain.handle("start-snip", async () => {
  await startSnip();
});
electron.ipcMain.on("overlay-cancel", () => {
  overlayWindow?.hide();
  mainWindow?.show();
});
electron.ipcMain.on("overlay-complete", (_e, croppedDataUrl) => {
  overlayWindow?.hide();
  mainWindow?.show();
  mainWindow?.webContents.send("image-captured", croppedDataUrl);
});
electron.ipcMain.handle("analyze-image", async (_e, dataUrl) => {
  try {
    return await runAnalysis(dataUrl);
  } catch (e) {
    throw new Error(friendlyApiError(e));
  }
});
async function runAnalysis(dataUrl) {
  const apiKey = config.apiKey?.trim();
  if (!apiKey)
    throw new Error("No API key set — open Settings and enter your API key.");
  const client = new OpenAI({
    baseURL: config.endpoint,
    apiKey
  });
  let base64 = dataUrl.split(",")[1];
  let mimeType = dataUrl.split(";")[0].split(":")[1];
  try {
    const buf = Buffer.from(base64, "base64");
    const img = electron.nativeImage.createFromBuffer(buf);
    const { width } = img.getSize();
    const resized = width > 1280 ? img.resize({ width: 1280 }) : img;
    base64 = resized.toJPEG(90).toString("base64");
    mimeType = "image/jpeg";
  } catch {
  }
  const stream = await client.chat.completions.create({
    model: config.model,
    messages: [
      { role: "system", content: config.systemPrompt },
      {
        role: "user",
        content: [
          { type: "image_url", image_url: { url: `data:${mimeType};base64,${base64}` } },
          { type: "text", text: config.userPrompt }
        ]
      },
      { role: "assistant", content: "<think>\n" }
    ],
    temperature: 0.2,
    max_tokens: 1024,
    stream: true,
    // @ts-ignore — vLLM-specific params
    chat_template_kwargs: { enable_thinking: true },
    continue_final_message: true,
    add_generation_prompt: false
  });
  let rcAccum = "";
  let rawContent = "";
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta;
    const rc = delta?.reasoning_content || "";
    if (rc) {
      rcAccum += rc;
      mainWindow?.webContents.send("analysis-stream", { thinking: rc, content: "" });
    }
    if (delta?.content) {
      rawContent += delta.content;
      mainWindow?.webContents.send("analysis-stream", { thinking: "", content: delta.content });
    }
  }
  console.log("[DEBUG] rcAccum:", JSON.stringify(rcAccum.slice(0, 200)));
  console.log("[DEBUG] rawContent:", JSON.stringify(rawContent.slice(0, 500)));
  let fullThinking = rcAccum || null;
  let content = rawContent;
  if (!fullThinking) {
    const fullMatch = content.match(/<think>([\s\S]*?)<\/think>/i);
    if (fullMatch) {
      fullThinking = fullMatch[1].trim();
      content = content.replace(fullMatch[0], "").trim();
    } else {
      const endMatch = content.match(/^([\s\S]*?)<\/think>/i);
      if (endMatch) {
        fullThinking = endMatch[1].trim();
        content = content.slice(endMatch[0].length).trim();
      }
    }
  }
  fullThinking = (fullThinking || "").trim();
  let jsonSource = content;
  if (!parseAnalysisJson(jsonSource) && fullThinking) {
    const m = fullThinking.match(/\{[\s\S]*\}/);
    if (m) {
      jsonSource = m[0];
      fullThinking = fullThinking.slice(0, fullThinking.indexOf(m[0])).trim();
    }
  }
  const parsed = parseAnalysisJson(jsonSource);
  let verdict = "uncertain";
  let confidence = null;
  let reason = "";
  if (parsed) {
    if (parsed.status === "ai_generated")
      verdict = "ai_generated";
    else if (parsed.status === "real")
      verdict = "real";
    if (typeof parsed.confidence === "number")
      confidence = parsed.confidence;
    if (parsed.reason)
      reason = parsed.reason;
  }
  return { verdict, confidence, reason, reasoning: jsonSource, thinking: fullThinking, raw: rawContent };
}
function parseAnalysisJson(text) {
  try {
    return JSON.parse(text.trim());
  } catch {
  }
  const codeMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
  if (codeMatch) {
    try {
      return JSON.parse(codeMatch[1]);
    } catch {
    }
  }
  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch {
    }
  }
  return null;
}
function friendlyApiError(e) {
  const cause = e.cause;
  if (cause?.code === "ECONNREFUSED") {
    return `Cannot connect to ${config.endpoint} (connection refused).
Check that the server is running and the Endpoint in Settings is correct.`;
  }
  if (cause?.code === "ENOTFOUND") {
    return `Host not found: ${config.endpoint}.
Check the Endpoint in Settings.`;
  }
  if (e.status === 401)
    return "Invalid API key — check your key in Settings.";
  if (e.status === 403)
    return "API key does not have permission for this model.";
  if (e.status === 400)
    return `Bad request: ${e.message}`;
  if (e.status === 429)
    return "Rate limit reached — try again in a moment.";
  const extra = e.status ? ` [HTTP ${e.status}]` : cause?.message ? ` (${cause.message})` : "";
  return (e.message || "Analysis failed") + extra;
}
