import { useState, useEffect, useRef, useCallback } from 'react'
import artifactLogo from './artifact-logo.jpg'

function ConfidenceRing({ confidence, color }) {
  const [filled, setFilled] = useState(false)
  useEffect(() => {
    const t = setTimeout(() => setFilled(true), 16)
    return () => clearTimeout(t)
  }, [])
  const SIZE = 114, SW = 7
  const r = (SIZE - SW) / 2
  const circ = 2 * Math.PI * r
  const offset = filled ? circ * (1 - (confidence ?? 0)) : circ
  return (
    <svg width={SIZE} height={SIZE} style={{ transform: 'rotate(-90deg)', display: 'block' }}>
      <circle cx={SIZE/2} cy={SIZE/2} r={r} fill="none" stroke="var(--border)" strokeWidth={SW} />
      <circle
        cx={SIZE/2} cy={SIZE/2} r={r}
        fill="none" stroke={color} strokeWidth={SW}
        strokeDasharray={circ} strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.9s cubic-bezier(0.4,0,0.2,1)', opacity: 0.85 }}
      />
    </svg>
  )
}

function ParticleField() {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.offsetWidth
    const h = canvas.offsetHeight
    canvas.width  = w * dpr
    canvas.height = h * dpr
    ctx.scale(dpr, dpr)

    const cx = w / 2, cy = h / 2
    const maxR = Math.sqrt(cx * cx + cy * cy)

    const N = 55
    const pts = Array.from({ length: N }, () => {
      const maxDist = maxR * (0.45 + Math.random() * 0.55)
      return {
        angle:   Math.random() * Math.PI * 2,
        dist:    Math.random() * maxDist,
        speed:   0.12 + Math.random() * 0.22,
        maxDist,
        r:       Math.random() * 1.6 + 0.5,
        phase:   Math.random() * Math.PI * 2,
        freq:    0.006 + Math.random() * 0.008,
      }
    })

    let raf
    const draw = () => {
      ctx.clearRect(0, 0, w, h)
      for (const p of pts) {
        p.dist  += p.speed
        p.phase += p.freq
        if (p.dist > p.maxDist) {
          p.dist    = 0
          p.angle   = Math.random() * Math.PI * 2
          p.speed   = 0.12 + Math.random() * 0.22
          p.maxDist = maxR * (0.45 + Math.random() * 0.55)
        }
        const x = cx + Math.cos(p.angle) * p.dist
        const y = cy + Math.sin(p.angle) * p.dist
        const progress = p.dist / p.maxDist
        const fade = progress < 0.15 ? progress / 0.15
                   : progress > 0.65 ? (1 - progress) / 0.35
                   : 1
        const a = fade * (0.12 + 0.1 * (0.5 + 0.5 * Math.sin(p.phase)))
        ctx.beginPath()
        ctx.arc(x, y, p.r, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(92,92,138,${a})`
        ctx.fill()
      }
      raf = requestAnimationFrame(draw)
    }
    draw()
    return () => cancelAnimationFrame(raf)
  }, [])
  return <canvas ref={canvasRef} className="particle-canvas" />
}

const VERDICT_META = {
  'ai_generated': { label: 'GEN AI', color: '#c0392b' },
  'real':         { label: 'REAL',          color: '#1a8a4a' },
  'uncertain':    { label: 'UNCERTAIN',     color: '#c07a18' }
}

export default function App() {
  const [view, setView] = useState('home')
  const [image, setImage] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [streamThinking, setStreamThinking] = useState('')
  const [thinkingOpen, setThinkingOpen] = useState(false)
  const [zoomed, setZoomed] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [urlInput, setUrlInput] = useState('')
  const [urlLoading, setUrlLoading] = useState(false)
  const [morphText, setMorphText] = useState('FACT')
  const viewRef = useRef(view)
  useEffect(() => { viewRef.current = view }, [view])
  const prevViewRef = useRef('home')
  function openAbout() { prevViewRef.current = view; setView('about') }
  function closeAbout() { setView(prevViewRef.current) }

  const handleImage = useCallback(async (dataUrl) => {
    setImage(dataUrl)
    setView('analyzing')
    setResult(null)
    setError(null)
    setStreamThinking('')
    setThinkingOpen(false)
    setZoomed(false)
    try {
      const r = await window.api.analyzeImage(dataUrl)
      setResult(r)
    } catch (e) {
      setError(e.message)
    }
    setView('results')
  }, [])

  useEffect(() => {
    window.api.onImageCaptured(handleImage)
    window.api.onCaptureError((msg) => { setError(msg); setView('results') })
    window.api.onAnalysisStream(({ thinking }) => {
      if (thinking) setStreamThinking((p) => p + thinking)
    })
    const onPaste = (e) => {
      if (viewRef.current === 'analyzing') return
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of items) {
        if (item.type.startsWith('image/')) {
          const file = item.getAsFile()
          if (!file) continue
          const reader = new FileReader()
          reader.onload = (ev) => handleImage(ev.target.result)
          reader.readAsDataURL(file)
          break
        }
      }
    }
    window.addEventListener('paste', onPaste)
    return () => { window.api.removeListeners(); window.removeEventListener('paste', onPaste) }
  }, [handleImage])

  useEffect(() => {
    if (view !== 'analyzing') { setMorphText('FACT'); return }
    const WORDS = ['FACT', 'FICTION']
    const TYPE_MS = 42, DELETE_MS = 28, PAUSE_MS = 800
    let cur = 'FACT', wordIdx = 0, deleting = false, timer
    setMorphText(cur)
    const tick = () => {
      const target = WORDS[wordIdx]
      if (!deleting) {
        if (cur.length < target.length) {
          cur = target.slice(0, cur.length + 1)
          setMorphText(cur)
          timer = setTimeout(tick, TYPE_MS)
        } else {
          timer = setTimeout(() => { deleting = true; tick() }, PAUSE_MS)
        }
      } else {
        if (cur.length > 0) {
          cur = cur.slice(0, -1)
          setMorphText(cur)
          timer = setTimeout(tick, DELETE_MS)
        } else {
          wordIdx = (wordIdx + 1) % WORDS.length
          deleting = false
          timer = setTimeout(tick, TYPE_MS)
        }
      }
    }
    timer = setTimeout(tick, PAUSE_MS)
    return () => clearTimeout(timer)
  }, [view])

  useEffect(() => {
    if (!zoomed) return
    const onKey = (e) => { if (e.key === 'Escape') setZoomed(false) }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [zoomed])

  function clear() {
    setImage(null); setResult(null); setError(null)
    setStreamThinking(''); setThinkingOpen(false)
    setZoomed(false); setView('home')
  }

  async function handleUrlSubmit(e) {
    e.preventDefault()
    const url = urlInput.trim()
    if (!url) return
    setUrlLoading(true)
    try {
      const dataUrl = await window.api.fetchImageUrl(url)
      setUrlInput('')
      handleImage(dataUrl)
    } catch (err) {
      setError(err.message)
      setView('results')
    } finally {
      setUrlLoading(false)
    }
  }

  function onDragOver(e) { e.preventDefault(); setDragOver(true) }
  function onDragLeave(e) { if (!e.currentTarget.contains(e.relatedTarget)) setDragOver(false) }
  function onDrop(e) {
    e.preventDefault(); setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = (ev) => handleImage(ev.target.result)
    reader.readAsDataURL(file)
  }

  const meta = result ? VERDICT_META[result.verdict] ?? VERDICT_META['uncertain'] : null
  const thinkingText = result?.thinking || streamThinking
  const thinkContent = (() => {
    if (thinkingText) return thinkingText
    const m = result?.reasoning?.match(/<think>([\s\S]*?)<\/think>/i)
    return m ? m[1].trim() : null
  })()

  return (
    <div className="app">
      <div className="top-bar">
        <span className="header-wordmark"></span>
        {view !== 'analyzing' && (
          <button className="btn-dots" onClick={view === 'about' ? closeAbout : openAbout} title="About">
            •••
          </button>
        )}
      </div>

      {view === 'home' && (
        <main
          className={`main home-view${dragOver ? ' drag-over' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >

          <div className="home-center">
            <div className="home-drop-icon">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
            </div>
            <p className="home-prompt">Drop an image to analyze</p>
            <p className="home-formats">PNG · JPG · WEBP</p>
            <form className="url-form" onSubmit={handleUrlSubmit}>
              <input
                className="url-input"
                type="url"
                placeholder="or paste an image URL…"
                value={urlInput}
                onChange={(e) => setUrlInput(e.target.value)}
                disabled={urlLoading}
              />
            </form>
          </div>

          <div className="home-bottom">
            <p className="shortcut-hint"><kbd>⌘⇧S</kbd> to snip · <kbd>⌘V</kbd> to paste</p>
          </div>
        </main>
      )}

      {view === 'analyzing' && (
        <main className="main analyzing-view">
          <ParticleField />
          <div className="typewriter-wrap">
            <span className="typewriter-text">ARTI[{morphText}]</span>
          </div>
        </main>
      )}

      {view === 'results' && (
        <main
          className={`main results-view${dragOver ? ' drag-over' : ''}`}
          onDragOver={onDragOver}
          onDragLeave={onDragLeave}
          onDrop={onDrop}
        >
          {image && (
            <div className="result-image-wrap" onClick={() => setZoomed(true)}>
              <img src={image} alt="Analyzed" className="result-image" />
            </div>
          )}

          <div className="result-content">
            {error && <div className="error-banner">⚠ {error}</div>}

            {result && meta && (
              <>
                <div className="result-top">
                  <div className="result-left">
                    <div className="ring-wrap">
                      <ConfidenceRing confidence={result.confidence} color={meta.color} />
                      <div className="ring-center">
                        <span className="ring-pct" style={{ color: meta.color }}>
                          {result.confidence != null ? `${Math.round(result.confidence * 100)}` : '—'}
                        </span>
                      </div>
                    </div>
                    <p className="verdict-label" style={{ color: meta.color }}>{meta.label}</p>
                  </div>
                  <div className="result-right">
                    <p className="reason-text">{result.reason || result.reasoning}</p>
                  </div>
                </div>

                {thinkContent && (
                  <div className="result-section">
                    <button className="thinking-toggle" onClick={() => setThinkingOpen((o) => !o)}>
                      <span className="row-label">Extended Reasoning</span>
                      <span className={`chevron${thinkingOpen ? ' open' : ''}`}>›</span>
                    </button>
                    {thinkingOpen && <p className="thinking-content">{thinkContent}</p>}
                  </div>
                )}


              </>
            )}

          </div>
            
          <div className="results-hint">
            <p className="shortcut-hint"><kbd>⌘⇧S</kbd> to snip · <kbd>⌘V</kbd> to paste</p>
          </div>
        </main>
      )}

      {view === 'about' && (
        <main className="main about-view">
          <div className="about-content">
            <div className="about-header">
              <p className="about-appname">ARTI[FACT]</p>
              <p className="about-tagline">AI-generated media detector</p>
            </div>

            <div className="about-section">
              <span className="about-label">Model</span>
              <a className="about-link" href="#" onClick={(e) => { e.preventDefault(); window.api.openExternal('https://huggingface.co/joshnegreanu/Nemo-12B-VL-AI-Detection-Fine-Tuned') }}>
                Finetuned Detection Model ↗
              </a>
            </div>

            <div className="about-section">
              <span className="about-label">Dataset</span>
              <a className="about-link" href="#" onClick={(e) => { e.preventDefault(); window.api.openExternal('https://huggingface.co/datasets/AnnaGao/MMFR-Dataset') }}>
                Finetuning Dataset ↗
              </a>
            </div>

            <div className="about-section">
              <span className="about-label">Learn More</span>
              <div className="about-links">
                {[
                  { label: 'Content Authenticity Initiative', url: 'https://contentauthenticity.org' },
                  { label: 'MIT Media Lab — Detect Fakes', url: 'https://detectfakes.media.mit.edu' },
                  { label: 'Partnership on AI — Synthetic Media', url: 'https://partnershiponai.org/synthetic-media' },
                  { label: 'WITNESS Media Lab', url: 'https://lab.witness.org' },
                  { label: 'Hany Farid — Digital Forensics', url: 'https://farid.berkeley.edu' },
                ].map(({ label, url }) => (
                  <a key={url} className="about-link" href="#" onClick={(e) => { e.preventDefault(); window.api.openExternal(url) }}>
                    {label} ↗
                  </a>
                ))}
              </div>
            </div>

            {/* <button className="btn-new" onClick={closeAbout}>← back</button> */}
          </div>
        </main>
      )}

      {zoomed && (
        <div className="lightbox" onClick={() => setZoomed(false)}>
          <img src={image} alt="Zoomed" className="lightbox-img" />
        </div>
      )}
    </div>
  )
}
