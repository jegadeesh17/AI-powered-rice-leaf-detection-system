"""FastAPI for rice leaf disease prediction."""

from __future__ import annotations

import io
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import model_path, predict_image

app = FastAPI(title="Rice Leaf Disease API", version="1.0.0")
_model = None


def _load_model():
    global _model
    if _model is None:
        path = model_path(ROOT)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}. Run python src/train.py")
        _model = keras.models.load_model(path, compile=False)
    return _model


@app.get("/health")
def health() -> dict:
    path = model_path(ROOT)
    return {"status": "ok", "model_loaded": os.path.exists(path)}


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def web_ui() -> str:
    """Premium browser UI — gives the REST API a polished visual front-end."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="Rice Leaf Disease Detection — AI-powered plant health analysis using deep learning on Cloud Run." />
  <title>Rice Leaf Disease Detection</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #080e10;
      --surface: rgba(16, 30, 20, 0.72);
      --border: rgba(52, 211, 130, 0.12);
      --border-hover: rgba(52, 211, 130, 0.30);
      --accent: #10b981;
      --accent2: #34d399;
      --accent-glow: rgba(16, 185, 129, 0.25);
      --text: #e8f5ee;
      --muted: #6b9e82;
      --input-bg: rgba(10, 20, 14, 0.6);
      --error: #f87171;
      --radius: 16px;
    }

    html, body { height: 100%; }

    body {
      font-family: 'Inter', system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      align-items: flex-start;
      justify-content: center;
      padding: 2.5rem 1rem 4rem;
      position: relative;
      overflow-x: hidden;
    }

    /* Aurora background glows */
    body::before, body::after {
      content: '';
      position: fixed;
      border-radius: 50%;
      filter: blur(90px);
      pointer-events: none;
      z-index: 0;
    }
    body::before {
      width: 500px; height: 500px;
      background: radial-gradient(circle, rgba(16,185,129,0.18) 0%, transparent 70%);
      top: -100px; left: -100px;
    }
    body::after {
      width: 420px; height: 420px;
      background: radial-gradient(circle, rgba(5,150,105,0.14) 0%, transparent 70%);
      bottom: -80px; right: -80px;
    }

    .container {
      position: relative;
      z-index: 1;
      width: 100%;
      max-width: 660px;
    }

    /* ── Header ── */
    .header {
      text-align: center;
      margin-bottom: 2rem;
    }
    .header-icon {
      font-size: 2.8rem;
      display: block;
      margin-bottom: 0.5rem;
      filter: drop-shadow(0 0 18px rgba(16,185,129,0.5));
      animation: float 4s ease-in-out infinite;
    }
    @keyframes float {
      0%, 100% { transform: translateY(0px); }
      50%       { transform: translateY(-6px); }
    }
    .header h1 {
      font-size: 1.85rem;
      font-weight: 700;
      letter-spacing: -0.03em;
      background: linear-gradient(135deg, #34d399 0%, #10b981 50%, #6ee7b7 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    .header p {
      margin-top: 0.4rem;
      font-size: 0.9rem;
      color: var(--muted);
    }
    .header p code {
      font-family: monospace;
      background: rgba(52,211,154,0.12);
      color: var(--accent2);
      padding: 0.1em 0.4em;
      border-radius: 4px;
      font-size: 0.85em;
    }

    /* ── Main Card ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 2rem;
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      box-shadow: 0 4px 40px rgba(0,0,0,0.5), 0 0 0 1px rgba(52,211,130,0.05);
      margin-bottom: 1.25rem;
      transition: border-color 0.3s;
    }
    .card:hover { border-color: var(--border-hover); }

    /* ── Drop Zone ── */
    .drop-zone {
      border: 2px dashed rgba(52,211,130,0.25);
      border-radius: 12px;
      padding: 2.5rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: border-color 0.25s, background 0.25s;
      position: relative;
      overflow: hidden;
    }
    .drop-zone:hover, .drop-zone.drag-over {
      border-color: var(--accent);
      background: rgba(16,185,129,0.06);
    }
    .drop-zone input[type=file] {
      position: absolute; inset: 0;
      opacity: 0; cursor: pointer; width: 100%; height: 100%;
    }
    .drop-icon {
      font-size: 2.2rem;
      display: block;
      margin-bottom: 0.6rem;
      opacity: 0.7;
    }
    .drop-label {
      font-size: 0.95rem;
      font-weight: 500;
      color: var(--text);
    }
    .drop-sub {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 0.3rem;
    }

    /* ── Preview ── */
    #preview-wrap {
      display: none;
      margin-top: 1.25rem;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid var(--border);
      position: relative;
    }
    #preview {
      width: 100%;
      max-height: 280px;
      object-fit: cover;
      display: block;
    }
    .preview-overlay {
      position: absolute; inset: 0;
      background: linear-gradient(to top, rgba(8,14,16,0.6) 0%, transparent 50%);
    }
    #file-name {
      position: absolute;
      bottom: 0.6rem; left: 0.8rem;
      font-size: 0.78rem;
      color: #cde;
      font-weight: 500;
    }

    /* ── Analyze Button ── */
    #btn {
      display: block;
      width: 100%;
      margin-top: 1.4rem;
      padding: 0.85rem 1.5rem;
      border: none;
      border-radius: 10px;
      font-size: 1rem;
      font-weight: 600;
      font-family: 'Inter', sans-serif;
      cursor: pointer;
      background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
      color: #fff;
      letter-spacing: 0.02em;
      box-shadow: 0 0 24px rgba(16,185,129,0.35);
      transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s;
      position: relative;
      overflow: hidden;
    }
    #btn::after {
      content: '';
      position: absolute; inset: 0;
      background: linear-gradient(135deg, rgba(255,255,255,0.08) 0%, transparent 60%);
    }
    #btn:not(:disabled):hover {
      opacity: 0.92;
      transform: translateY(-1px);
      box-shadow: 0 0 32px rgba(16,185,129,0.5);
    }
    #btn:disabled {
      opacity: 0.38;
      cursor: not-allowed;
      box-shadow: none;
    }

    /* ── Spinner ── */
    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.75s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .hint {
      font-size: 0.8rem;
      color: var(--muted);
      margin-top: 0.9rem;
      text-align: center;
    }

    /* ── Results ── */
    #result { margin-top: 1.5rem; }

    .result-diagnosis {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      padding: 1rem 1.25rem;
      background: rgba(16,185,129,0.1);
      border: 1px solid rgba(16,185,129,0.25);
      border-radius: 10px;
      margin-bottom: 1.25rem;
    }
    .diag-icon { font-size: 1.8rem; }
    .diag-name {
      font-size: 1.25rem;
      font-weight: 700;
      color: #6ee7b7;
    }
    .diag-confidence {
      font-size: 0.85rem;
      color: var(--muted);
      margin-top: 2px;
    }

    .prob-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      color: var(--muted);
      margin-bottom: 4px;
    }
    .prob-label .cls-pct { color: var(--accent2); font-weight: 600; }
    .bar {
      height: 7px;
      background: rgba(52,211,130,0.1);
      border-radius: 99px;
      margin-bottom: 0.75rem;
      overflow: hidden;
    }
    .fill {
      height: 100%;
      background: linear-gradient(90deg, #059669, #34d399);
      border-radius: 99px;
      transition: width 0.8s cubic-bezier(0.34,1.56,0.64,1);
      width: 0%;
    }

    .raw-json {
      margin-top: 1rem;
    }
    .raw-json summary {
      font-size: 0.78rem;
      color: var(--muted);
      cursor: pointer;
      user-select: none;
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.3rem 0;
    }
    .raw-json summary::marker { color: var(--accent); }
    .raw-json pre {
      margin-top: 0.5rem;
      background: rgba(10,20,14,0.7);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.9rem;
      font-size: 0.75rem;
      color: #9dd;
      overflow-x: auto;
      line-height: 1.5;
    }

    .err-box {
      display: flex;
      align-items: flex-start;
      gap: 0.6rem;
      background: rgba(239,68,68,0.1);
      border: 1px solid rgba(239,68,68,0.3);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      color: var(--error);
      font-size: 0.9rem;
    }

    /* ── Links Card ── */
    .links-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.25rem 2rem;
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
    }
    .links-title {
      font-size: 0.78rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 0.75rem;
    }
    .links-row {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .link-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.35rem 0.75rem;
      border-radius: 99px;
      border: 1px solid var(--border);
      font-size: 0.8rem;
      color: var(--accent2);
      text-decoration: none;
      transition: border-color 0.2s, background 0.2s;
    }
    .link-pill:hover {
      border-color: var(--accent);
      background: rgba(16,185,129,0.08);
    }
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <div class="header">
      <span class="header-icon">🌾</span>
      <h1>Rice Leaf Disease Detection</h1>
      <p>AI-powered analysis via Cloud Run — same backend as <code>POST /predict</code></p>
    </div>

    <!-- Main Upload Card -->
    <div class="card">
      <div class="drop-zone" id="drop-zone">
        <input type="file" id="file" accept="image/jpeg,image/png,image/jpg" />
        <span class="drop-icon">🍃</span>
        <div class="drop-label">Drop a leaf image here, or click to browse</div>
        <div class="drop-sub">Supports JPEG &amp; PNG</div>
      </div>

      <div id="preview-wrap">
        <img id="preview" alt="Leaf preview" />
        <div class="preview-overlay"></div>
        <span id="file-name"></span>
      </div>

      <button id="btn" disabled>Analyze via API</button>

      <p class="hint">⏱ First request after idle may take 1–2 min (cold start + model load)</p>

      <div id="result"></div>
    </div>

    <!-- Links Card -->
    <div class="links-card">
      <div class="links-title">API Endpoints</div>
      <div class="links-row">
        <a class="link-pill" href="/docs">📄 Swagger UI</a>
        <a class="link-pill" href="/health">💚 Health</a>
        <a class="link-pill" href="/redoc">📑 ReDoc</a>
      </div>
    </div>
  </div>

  <script>
    const fileInput = document.getElementById('file');
    const preview   = document.getElementById('preview');
    const previewWrap = document.getElementById('preview-wrap');
    const btn       = document.getElementById('btn');
    const result    = document.getElementById('result');
    const dropZone  = document.getElementById('drop-zone');
    const fileNameEl = document.getElementById('file-name');
    let selectedFile = null;

    function setFile(f) {
      selectedFile = f;
      btn.disabled = !f;
      result.innerHTML = '';
      if (f) {
        preview.src = URL.createObjectURL(f);
        previewWrap.style.display = 'block';
        fileNameEl.textContent = f.name;
      } else {
        previewWrap.style.display = 'none';
      }
    }

    fileInput.addEventListener('change', () => setFile(fileInput.files[0] || null));

    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f && (f.type === 'image/jpeg' || f.type === 'image/png')) setFile(f);
    });

    btn.addEventListener('click', async () => {
      if (!selectedFile) return;
      btn.disabled = true;
      btn.innerHTML = '<span class="spinner"></span>Analyzing…';
      result.innerHTML = '';

      const form = new FormData();
      form.append('file', selectedFile);
      try {
        const res  = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);

        const probs = data.probabilities || {};
        let barsHtml = '';
        for (const [cls, p] of Object.entries(probs)) {
          barsHtml += `
            <div class="prob-label">
              <span>${cls}</span>
              <span class="cls-pct">${(p*100).toFixed(1)}%</span>
            </div>
            <div class="bar"><div class="fill" data-pct="${p*100}"></div></div>`;
        }

        result.innerHTML = `
          <div class="result-diagnosis">
            <span class="diag-icon">🔬</span>
            <div>
              <div class="diag-name">${data.display_name || data.disease_class}</div>
              <div class="diag-confidence">Confidence: ${(data.confidence*100).toFixed(1)}%</div>
            </div>
          </div>
          ${barsHtml}
          <details class="raw-json">
            <summary>Raw JSON response</summary>
            <pre>${JSON.stringify(data, null, 2)}</pre>
          </details>`;

        // Animate bars after paint
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            document.querySelectorAll('.fill[data-pct]').forEach(el => {
              el.style.width = el.dataset.pct + '%';
            });
          });
        });
      } catch (e) {
        result.innerHTML = `<div class="err-box">⚠️ ${e.message}</div>`;
      }
      btn.disabled = false;
      btn.innerHTML = 'Analyze via API';
    });
  </script>
</body>
</html>"""


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=422, detail="Upload a JPEG or PNG image")
    try:
        image = Image.open(io.BytesIO(await file.read()))
        model = _load_model()
        result = predict_image(model, image)
        result["grad_cam_available"] = True
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
