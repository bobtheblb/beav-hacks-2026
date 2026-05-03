import {
  app,
  BrowserWindow,
  globalShortcut,
  ipcMain,
  desktopCapturer,
  Tray,
  Menu,
  nativeImage,
  shell,
  systemPreferences,
  screen
} from 'electron'
import { join } from 'path'
import OpenAI from 'openai'

const isDev = process.env.NODE_ENV === 'development'

app.setName('Artifact')

let mainWindow = null
let overlayWindow = null
let tray = null

const config = {
  endpoint: 'http://129.213.148.162:8000/v1',
  model: 'joshnegreanu/Nemo-12B-VL-AI-Detection-Fine-Tuned',
  apiKey: 'asdf',
  systemPrompt:
    'You must always begin your response with a <think>...</think> ' +
    'block describing your visual analysis, then output ONLY a JSON ' +
    'object after </think>. Responses without <think> are invalid.',
  userPrompt:
    'Analyze the provided image and determine whether it is AI-generated. You are helping an individual learn how to pick up on cues that would help inform them in spotting AI generated images.\n' +
    'First reason inside <think>...</think> tags about the visual cues you observe.\n' +
    'After </think>, output ONLY a JSON object with these fields:\n' +
    '  - status: either "ai_generated" or "real"\n' +
    '  - confidence: a calibrated number between 0 and 1.\n' +
    '      * Reserve 1.0 ONLY for unmistakable evidence (e.g. obvious GAN artifacts, impossible anatomy).\n' +
    '      * Use 0.85-0.95 for clear cases.\n' +
    '      * Use 0.6-0.8 when there are some ambiguous details.\n' +
    '      * Use 0.5-0.6 when the image is genuinely hard to classify.\n' +
    '      * Never use exactly 1.0 unless you would bet money on it.\n' +
    '  - reason: a brief explanation of your decision'
}

function createMainWindow() {
  mainWindow = new BrowserWindow({
    width: 480,
    height: 640,
    minWidth: 380,
    minHeight: 480,
    show: false,
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  mainWindow.on('ready-to-show', () => mainWindow.show())
  mainWindow.on('closed', () => { mainWindow = null })

  if (isDev && process.env.ELECTRON_RENDERER_URL) {
    mainWindow.loadURL(process.env.ELECTRON_RENDERER_URL)
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

function createOverlayWindow() {
  overlayWindow = new BrowserWindow({
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
      preload: join(__dirname, '../preload/overlay.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  })

  overlayWindow.setAlwaysOnTop(true, 'screen-saver')
  overlayWindow.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true })
  overlayWindow.on('closed', () => { overlayWindow = null })

  if (isDev && process.env.ELECTRON_RENDERER_URL) {
    const base = process.env.ELECTRON_RENDERER_URL.replace(/\/$/, '')
    overlayWindow.loadURL(`${base}/overlay.html`)
  } else {
    overlayWindow.loadFile(join(__dirname, '../renderer/overlay.html'))
  }
}

async function captureScreen() {
  if (process.platform === 'darwin') {
    const status = systemPreferences.getMediaAccessStatus('screen')
    if (status !== 'granted') {
      shell.openExternal(
        'x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture'
      )
      throw new Error(
        `Screen Recording permission required.\n` +
        `System Settings → Privacy & Security → Screen Recording\n` +
        `Enable "Electron" in the list, then restart the app.`
      )
    }
  }

  let sources
  try {
    sources = await desktopCapturer.getSources({
      types: ['screen'],
      thumbnailSize: { width: 2560, height: 1440 }
    })
  } catch (e) {
    throw new Error(`Screen capture failed: ${e.message}`)
  }
  if (!sources.length) throw new Error('No screen sources found.')
  return sources[0].thumbnail.toDataURL()
}

// Resolves when the overlay renderer signals its screenshot listener is registered.
// Falls back after 2 s in case the signal is missed (e.g. window already loaded).
function waitForOverlayReady() {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, 2000)
    ipcMain.once('overlay-ready', () => { clearTimeout(timer); resolve() })
  })
}

async function startSnip() {
  try {
    mainWindow?.hide()
    await new Promise((r) => setTimeout(r, 200))

    const dataUrl = await captureScreen()

    const isNew = !overlayWindow
    if (isNew) createOverlayWindow()

    // For a pre-loaded overlay the renderer is already listening; send immediately.
    // For a newly created one, wait for the 'overlay-ready' signal from the renderer.
    if (isNew) {
      await waitForOverlayReady()
    }

    // Use the display the cursor is on; call setBounds before AND after show()
    // so macOS work-area constraints from window creation are overridden.
    const { bounds } = screen.getDisplayNearestPoint(screen.getCursorScreenPoint())
    overlayWindow.setBounds(bounds)
    overlayWindow.show()
    overlayWindow.setBounds(bounds)
    overlayWindow.focus()
    overlayWindow.webContents.send('screenshot', dataUrl)
  } catch (err) {
    mainWindow?.show()
    mainWindow?.webContents.send('capture-error', err.message)
  }
}

app.whenReady().then(() => {
  if (process.platform === 'darwin' && app.dock) {
    const iconPath = join(__dirname, '../../build/icon.png')
    try { app.dock.setIcon(nativeImage.createFromPath(iconPath)) } catch {}
  }
  createMainWindow()

  // System tray — transparent 1x1 pixel icon; title used on macOS
  try {
    const icon = nativeImage.createFromDataURL(
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII='
    )
    tray = new Tray(icon)
    if (process.platform === 'darwin') tray.setTitle('⌕')
    tray.setToolTip('Finding Nemo')
    tray.setContextMenu(
      Menu.buildFromTemplate([
        { label: 'Show', click: () => mainWindow?.show() },
        { label: 'Snip Region', click: () => startSnip() },
        { type: 'separator' },
        { label: 'Quit', click: () => app.quit() }
      ])
    )
    tray.on('click', () => mainWindow?.show())
  } catch {}

  globalShortcut.register('CommandOrControl+Shift+S', () => startSnip())

  app.on('activate', () => {
    if (!mainWindow) createMainWindow()
    else mainWindow.show()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('will-quit', () => {
  globalShortcut.unregisterAll()
})

// ── IPC handlers ────────────────────────────────────────────────────────────

ipcMain.on('open-external', async (_e, url) => {
  try { await shell.openExternal(url) } catch (e) { console.error('openExternal failed:', e) }
})

ipcMain.handle('fetch-image-url', async (_e, url) => {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status} — could not load image`)
  const contentType = res.headers.get('content-type') || 'image/jpeg'
  if (!contentType.startsWith('image/')) throw new Error('URL does not point to an image')
  const buf = Buffer.from(await res.arrayBuffer())
  return `data:${contentType};base64,${buf.toString('base64')}`
})

ipcMain.handle('capture-full-screen', async () => {
  mainWindow?.hide()
  await new Promise((r) => setTimeout(r, 200))
  try {
    return await captureScreen()
  } finally {
    mainWindow?.show()
  }
})

ipcMain.handle('start-snip', async () => {
  await startSnip()
})

ipcMain.on('overlay-cancel', () => {
  overlayWindow?.hide()
  mainWindow?.show()
})

ipcMain.on('overlay-complete', (_e, croppedDataUrl) => {
  overlayWindow?.hide()
  mainWindow?.show()
  mainWindow?.webContents.send('image-captured', croppedDataUrl)
})

ipcMain.handle('analyze-image', async (_e, dataUrl) => {
  try {
    return await runAnalysis(dataUrl)
  } catch (e) {
    throw new Error(friendlyApiError(e))
  }
})

async function runAnalysis(dataUrl) {
  const apiKey = config.apiKey?.trim()
  if (!apiKey) throw new Error('No API key set — open Settings and enter your API key.')

  const client = new OpenAI({
    baseURL: config.endpoint,
    apiKey
  })

  // Resize to ≤1280 px wide and re-encode as JPEG to keep the payload under ~500 KB.
  let base64 = dataUrl.split(',')[1]
  let mimeType = dataUrl.split(';')[0].split(':')[1]
  try {
    const buf = Buffer.from(base64, 'base64')
    const img = nativeImage.createFromBuffer(buf)
    const { width } = img.getSize()
    const resized = width > 1280 ? img.resize({ width: 1280 }) : img
    base64 = resized.toJPEG(90).toString('base64')
    mimeType = 'image/jpeg'
  } catch {
    // fall through and use the original if nativeImage fails
  }

  const stream = await client.chat.completions.create({
    model: config.model || 'my-adapter',
    messages: [
      { role: 'system', content: config.systemPrompt },
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: `data:${mimeType};base64,${base64}` } },
          { type: 'text', text: config.userPrompt }
        ]
      },
      { role: 'assistant', content: '<think>\n' }
    ],
    temperature: 0.2,
    max_tokens: 1024,
    stream: true,
    // @ts-ignore — vLLM-specific params
    chat_template_kwargs: { enable_thinking: true },
    continue_final_message: true,
    add_generation_prompt: false
  })

  let rcAccum = ''   // reasoning_content field tokens
  let rawContent = '' // all delta.content tokens verbatim

  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta
    const rc = delta?.reasoning_content || ''
    if (rc) {
      rcAccum += rc
      mainWindow?.webContents.send('analysis-stream', { thinking: rc, content: '' })
    }
    if (delta?.content) {
      rawContent += delta.content
      mainWindow?.webContents.send('analysis-stream', { thinking: '', content: delta.content })
    }
  }

  console.log('[DEBUG] rcAccum:', JSON.stringify(rcAccum.slice(0, 200)))
  console.log('[DEBUG] rawContent:', JSON.stringify(rawContent.slice(0, 500)))

  // ── Post-stream parsing (mirrors the Python pattern) ──────────────────────
  // Priority 1: vLLM dedicated reasoning_content field
  let fullThinking = rcAccum || null
  let content = rawContent

  if (!fullThinking) {
    // Case 1: full <think>...</think> block present in content
    const fullMatch = content.match(/<think>([\s\S]*?)<\/think>/i)
    if (fullMatch) {
      fullThinking = fullMatch[1].trim()
      content = content.replace(fullMatch[0], '').trim()
    } else {
      // Case 2: prefill stripped the opening <think>; only </think> remains
      const endMatch = content.match(/^([\s\S]*?)<\/think>/i)
      if (endMatch) {
        fullThinking = endMatch[1].trim()
        content = content.slice(endMatch[0].length).trim()
      }
    }
  }

  fullThinking = (fullThinking || '').trim()

  let jsonSource = content
  if (!parseAnalysisJson(jsonSource) && fullThinking) {
    const m = fullThinking.match(/\{[\s\S]*\}/)
    if (m) {
      jsonSource = m[0]
      fullThinking = fullThinking.slice(0, fullThinking.indexOf(m[0])).trim()
    }
  }

  const parsed = parseAnalysisJson(jsonSource)
  let verdict = 'uncertain'
  let confidence = null
  let reason = ''
  if (parsed) {
    if (parsed.status === 'ai_generated') verdict = 'ai_generated'
    else if (parsed.status === 'real') verdict = 'real'
    if (typeof parsed.confidence === 'number') confidence = parsed.confidence
    if (parsed.reason) reason = parsed.reason
  }

  return { verdict, confidence, reason, reasoning: jsonSource, thinking: fullThinking, raw: rawContent }
}

function parseAnalysisJson(text) {
  try { return JSON.parse(text.trim()) } catch {}
  const codeMatch = text.match(/```(?:json)?\s*([\s\S]*?)\s*```/)
  if (codeMatch) { try { return JSON.parse(codeMatch[1]) } catch {} }
  const jsonMatch = text.match(/\{[\s\S]*\}/)
  if (jsonMatch) { try { return JSON.parse(jsonMatch[0]) } catch {} }
  return null
}

function friendlyApiError(e) {
  const cause = e.cause
  if (cause?.code === 'ECONNREFUSED') {
    return `Cannot connect to ${config.endpoint} (connection refused).\nCheck that the server is running and the Endpoint in Settings is correct.`
  }
  if (cause?.code === 'ENOTFOUND') {
    return `Host not found: ${config.endpoint}.\nCheck the Endpoint in Settings.`
  }
  if (e.status === 401) return 'Invalid API key — check your key in Settings.'
  if (e.status === 403) return 'API key does not have permission for this model.'
  if (e.status === 400) return `Bad request: ${e.message}`
  if (e.status === 429) return 'Rate limit reached — try again in a moment.'
  const extra = e.status ? ` [HTTP ${e.status}]` : cause?.message ? ` (${cause.message})` : ''
  return (e.message || 'Analysis failed') + extra
}
