import React, { useEffect, useRef, useCallback } from 'react'
import ReactDOM from 'react-dom/client'

function drawFrame(canvas, img, sel) {
  const ctx = canvas.getContext('2d')
  ctx.clearRect(0, 0, canvas.width, canvas.height)
  ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

  ctx.fillStyle = 'rgba(0, 0, 0, 0.45)'
  ctx.fillRect(0, 0, canvas.width, canvas.height)

  if (!sel) return

  const x = sel.w < 0 ? sel.x + sel.w : sel.x
  const y = sel.h < 0 ? sel.y + sel.h : sel.y
  const w = Math.abs(sel.w)
  const h = Math.abs(sel.h)

  if (w < 2 || h < 2) return

  // Restore the selected region at original brightness
  ctx.clearRect(x, y, w, h)
  ctx.drawImage(
    img,
    (x / canvas.width) * img.naturalWidth,
    (y / canvas.height) * img.naturalHeight,
    (w / canvas.width) * img.naturalWidth,
    (h / canvas.height) * img.naturalHeight,
    x, y, w, h
  )

  ctx.strokeStyle = '#4f8ef7'
  ctx.lineWidth = 2
  ctx.strokeRect(x, y, w, h)

  // Corner handles
  const cs = 8
  ctx.fillStyle = '#4f8ef7'
  for (const [cx, cy] of [[x, y], [x + w, y], [x, y + h], [x + w, y + h]]) {
    ctx.fillRect(cx - cs / 2, cy - cs / 2, cs, cs)
  }

  // Dimensions label
  const label = `${Math.round(w)} × ${Math.round(h)}`
  ctx.font = 'bold 12px -apple-system, BlinkMacSystemFont, sans-serif'
  const tw = ctx.measureText(label).width
  const lx = x
  const ly = y > 24 ? y - 6 : y + h + 18
  ctx.fillStyle = 'rgba(79, 142, 247, 0.92)'
  ctx.fillRect(lx - 2, ly - 14, tw + 10, 18)
  ctx.fillStyle = '#fff'
  ctx.fillText(label, lx + 3, ly)
}

function cropImage(img, canvas, sel) {
  const x = sel.w < 0 ? sel.x + sel.w : sel.x
  const y = sel.h < 0 ? sel.y + sel.h : sel.y
  const w = Math.abs(sel.w)
  const h = Math.abs(sel.h)

  const sx = img.naturalWidth / canvas.width
  const sy = img.naturalHeight / canvas.height

  const crop = document.createElement('canvas')
  crop.width  = Math.round(w * sx)
  crop.height = Math.round(h * sy)
  crop.getContext('2d').drawImage(
    img,
    Math.round(x * sx), Math.round(y * sy), crop.width, crop.height,
    0, 0, crop.width, crop.height
  )
  return crop.toDataURL('image/png')
}

export default function Overlay() {
  const canvasRef = useRef(null)
  const imgRef    = useRef(null)
  const drawing   = useRef(false)
  const start     = useRef({ x: 0, y: 0 })

  useEffect(() => {
    window.overlayAPI.onScreenshot((dataUrl) => {
      const img = new Image()
      img.onload = () => {
        imgRef.current = img
        const canvas = canvasRef.current
        canvas.width  = window.innerWidth
        canvas.height = window.innerHeight
        drawFrame(canvas, img, null)
      }
      img.src = dataUrl
    })

    // Tell main process the listener is registered and we're ready to receive
    window.overlayAPI.ready()

    const onKey = (e) => { if (e.key === 'Escape') window.overlayAPI.cancel() }
    window.addEventListener('keydown', onKey)
    return () => {
      window.overlayAPI.offScreenshot()
      window.removeEventListener('keydown', onKey)
    }
  }, [])

  const onMouseDown = useCallback((e) => {
    drawing.current = true
    start.current = { x: e.clientX, y: e.clientY }
  }, [])

  const onMouseMove = useCallback((e) => {
    if (!drawing.current || !imgRef.current) return
    drawFrame(canvasRef.current, imgRef.current, {
      x: start.current.x, y: start.current.y,
      w: e.clientX - start.current.x,
      h: e.clientY - start.current.y
    })
  }, [])

  const onMouseUp = useCallback((e) => {
    if (!drawing.current) return
    drawing.current = false

    const sel = {
      x: start.current.x, y: start.current.y,
      w: e.clientX - start.current.x,
      h: e.clientY - start.current.y
    }

    if (Math.abs(sel.w) < 10 || Math.abs(sel.h) < 10) {
      window.overlayAPI.cancel()
      return
    }

    window.overlayAPI.complete(cropImage(imgRef.current, canvasRef.current, sel))
  }, [])

  return (
    <>
      <canvas
        ref={canvasRef}
        style={{ position: 'fixed', top: 0, left: 0, width: '100vw', height: '100vh', cursor: 'crosshair', display: 'block' }}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={onMouseUp}
      />
      <div style={{
        position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
        background: 'rgba(0,0,0,0.72)', color: '#fff',
        padding: '6px 14px', borderRadius: 20,
        fontSize: 12, fontFamily: '-apple-system, BlinkMacSystemFont, sans-serif',
        pointerEvents: 'none', whiteSpace: 'nowrap', letterSpacing: 0.2,
        backdropFilter: 'blur(6px)',
      }}>
        Drag to select &nbsp;·&nbsp; <kbd style={{ opacity: 0.75 }}>Esc</kbd> to cancel
      </div>
    </>
  )
}

ReactDOM.createRoot(document.getElementById('overlay-root')).render(
  <React.StrictMode>
    <Overlay />
  </React.StrictMode>
)
