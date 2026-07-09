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
    """Minimal browser UI — gives the REST API a visual front-end."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rice Leaf Disease API</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: system-ui, -apple-system, sans-serif;
      max-width: 720px; margin: 2rem auto; padding: 0 1rem;
      background: #f8faf5; color: #1a2e1a;
    }
    h1 { color: #1e3c72; margin-bottom: 0.25rem; }
    .sub { color: #555; margin-bottom: 1.5rem; }
    .card {
      background: #fff; border-radius: 12px; padding: 1.5rem;
      box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 1rem;
    }
    input[type=file] { margin: 1rem 0; width: 100%; }
    button {
      background: #2a5298; color: #fff; border: none; padding: 0.75rem 1.5rem;
      border-radius: 8px; font-size: 1rem; cursor: pointer;
    }
    button:disabled { opacity: 0.6; cursor: wait; }
    #preview { max-width: 100%; border-radius: 8px; margin-top: 1rem; display: none; }
    #result { margin-top: 1rem; }
    .diag { font-size: 1.5rem; font-weight: 700; color: #1e3c72; }
    .bar { height: 8px; background: #e2e8f0; border-radius: 4px; margin: 4px 0 12px; }
    .fill { height: 100%; background: #2a5298; border-radius: 4px; }
    .links { margin-top: 1.5rem; font-size: 0.9rem; }
    .links a { color: #2a5298; }
    .err { color: #b91c1c; }
    .hint { font-size: 0.85rem; color: #666; }
  </style>
</head>
<body>
  <h1>🌾 Rice Leaf Disease Detection</h1>
  <p class="sub">Browser UI for the Cloud Run inference API — same backend as <code>POST /predict</code>.</p>

  <div class="card">
    <label for="file"><strong>Upload leaf image</strong> (JPEG or PNG)</label>
    <input type="file" id="file" accept="image/jpeg,image/png,image/jpg" />
    <img id="preview" alt="Preview" />
    <br />
    <button id="btn" disabled>Analyze via API</button>
    <p class="hint">First request after idle may take 1–2 minutes (cold start + model load).</p>
    <div id="result"></div>
  </div>

  <div class="links card">
    <strong>API endpoints</strong><br />
    <a href="/docs">Swagger UI (/docs)</a> ·
    <a href="/health">Health (/health)</a> ·
    <a href="/redoc">ReDoc (/redoc)</a>
  </div>

  <script>
    const fileInput = document.getElementById('file');
    const preview = document.getElementById('preview');
    const btn = document.getElementById('btn');
    const result = document.getElementById('result');
    let selectedFile = null;

    fileInput.addEventListener('change', () => {
      selectedFile = fileInput.files[0] || null;
      btn.disabled = !selectedFile;
      result.innerHTML = '';
      if (selectedFile) {
        preview.src = URL.createObjectURL(selectedFile);
        preview.style.display = 'block';
      } else {
        preview.style.display = 'none';
      }
    });

    btn.addEventListener('click', async () => {
      if (!selectedFile) return;
      btn.disabled = true;
      btn.textContent = 'Analyzing…';
      result.innerHTML = '<p class="hint">Calling POST /predict…</p>';
      const form = new FormData();
      form.append('file', selectedFile);
      try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);
        let bars = '';
        const probs = data.probabilities || {};
        for (const [cls, p] of Object.entries(probs)) {
          bars += `<div>${cls} (${(p*100).toFixed(1)}%)</div>
            <div class="bar"><div class="fill" style="width:${p*100}%"></div></div>`;
        }
        result.innerHTML = `
          <p class="diag">${data.display_name || data.disease_class}</p>
          <p>Confidence: <strong>${(data.confidence*100).toFixed(1)}%</strong></p>
          ${bars}
          <details><summary>Raw JSON</summary><pre>${JSON.stringify(data, null, 2)}</pre></details>`;
      } catch (e) {
        result.innerHTML = `<p class="err">Error: ${e.message}</p>`;
      }
      btn.disabled = false;
      btn.textContent = 'Analyze via API';
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
