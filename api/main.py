"""FastAPI for rice leaf disease prediction."""

from __future__ import annotations

import io
import os
import sys
import time
import sqlite3
import base64
import json
import numpy as np
import cv2
import matplotlib.cm as cm

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.inference import CLASSES, DISPLAY_NAMES, IMG_SIZE, model_path, predict_image, preprocess_image
from src.interpretability import make_gradcam_heatmap

app = FastAPI(title="Rice Leaf Disease API", version="1.0.0")
_model = None
DB_PATH = os.path.join(ROOT, "diagnostic_logs.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS inference_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            source_mode TEXT,
            diagnosed_class TEXT,
            confidence REAL,
            latency_ms REAL
        )
    """)
    conn.commit()
    conn.close()

def log_inference_run(source_mode, diagnosed_class, confidence, latency_ms):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO inference_runs (source_mode, diagnosed_class, confidence, latency_ms)
            VALUES (?, ?, ?, ?)
        """, (source_mode, diagnosed_class, float(confidence), float(latency_ms)))
        conn.commit()
        conn.close()
    except Exception:
        pass

init_db()

def _load_model():
    global _model
    if _model is None:
        path = model_path(ROOT)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing model: {path}. Run python src/train.py")
        _model = keras.models.load_model(path, compile=False)
    return _model

def overlay_heatmap(image, heatmap, alpha=0.45):
    img = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
    heatmap_arr = np.array(heatmap)
    if heatmap_arr.ndim == 3:
        heatmap_arr = heatmap_arr[..., 0]
    heatmap_arr = np.clip(heatmap_arr, 0.0, 1.0)
    heatmap_scaled = np.uint8(255 * heatmap_arr)

    jet = cm.get_cmap("jet") if hasattr(cm, "get_cmap") else cm.jet
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_scaled]
    jet_heatmap = cv2.resize(jet_heatmap, (img.shape[1], img.shape[0]))
    jet_heatmap = np.uint8(255 * jet_heatmap)

    if jet_heatmap.ndim == 3 and jet_heatmap.shape[2] == 4:
        jet_heatmap = jet_heatmap[:, :, :3]

    superimposed_img = jet_heatmap * alpha + img
    return superimposed_img.astype("uint8")

@app.get("/health")
def health() -> dict:
    path = model_path(ROOT)
    return {"status": "ok", "model_loaded": os.path.exists(path)}

@app.get("/logs")
def get_logs() -> JSONResponse:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM inference_runs ORDER BY timestamp DESC LIMIT 100")
        rows = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return JSONResponse(content=rows)
    except Exception:
        return JSONResponse(content=[])

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
      pointer-events: none;
    }
    #file-name {
      position: absolute;
      bottom: 0.6rem; left: 0.8rem;
      font-size: 0.78rem;
      color: #cde;
      font-weight: 500;
      pointer-events: none;
    }
    .gradcam-toggle {
      position: absolute; top: 0.5rem; right: 0.5rem;
      background: rgba(0,0,0,0.6); border: 1px solid rgba(255,255,255,0.2);
      color: white; border-radius: 4px; padding: 0.3rem 0.6rem; font-size: 0.75rem;
      cursor: pointer; backdrop-filter: blur(4px);
    }
    .gradcam-toggle:hover { background: rgba(16,185,129,0.8); border-color: var(--accent); }


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

    .treatment-box {
      margin-top: 1rem; margin-bottom: 1rem;
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 1rem;
      font-size: 0.85rem;
      color: #9dd;
      white-space: pre-wrap;
    }
    .treatment-title {
      font-weight: 600;
      color: var(--accent2);
      margin-bottom: 0.5rem;
      font-size: 0.9rem;
    }

    .report-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      background: rgba(10, 20, 14, 0.6);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.5rem 1rem;
      color: var(--text);
      font-size: 0.85rem;
      cursor: pointer;
      margin-bottom: 0.5rem;
      transition: all 0.2s;
    }
    .report-btn:hover { background: rgba(16, 185, 129, 0.15); border-color: var(--accent); }

    .raw-json {
      margin-top: 0.5rem;
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
      cursor: pointer;
      background: none;
    }
    .link-pill:hover {
      border-color: var(--accent);
      background: rgba(16,185,129,0.08);
    }

    /* Telemetry Modal */
    .telemetry-modal {
      display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.8);
      z-index: 100; align-items: center; justify-content: center; backdrop-filter: blur(4px);
    }
    .telemetry-content {
      background: var(--bg); border: 1px solid var(--border); border-radius: var(--radius);
      padding: 2rem; width: 90%; max-width: 800px; max-height: 80vh; overflow-y: auto;
    }
    .telemetry-content h2 { margin-bottom: 1rem; font-size: 1.5rem; color: var(--accent2); }
    .close-modal { float: right; cursor: pointer; color: var(--muted); font-size: 1.2rem; }
    .close-modal:hover { color: var(--text); }
    .table-wrapper { overflow-x: auto; margin-top: 1rem; }
    table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
    th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); }
    th { color: var(--accent2); font-weight: 600; }
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
        <button class="gradcam-toggle" id="gc-toggle" style="display:none;">Toggle Grad-CAM</button>
      </div>

      <button id="btn" disabled>Analyze via API</button>

      <p class="hint">⏱ First request after idle may take 1–2 min (cold start + model load)</p>

      <div id="result"></div>
    </div>

    <!-- Links Card -->
    <div class="links-card">
      <div class="links-title">System Links</div>
      <div class="links-row">
        <a class="link-pill" href="/docs">📄 Swagger UI</a>
        <a class="link-pill" href="/health">💚 Health</a>
        <button class="link-pill" id="open-modal">📊 View Telemetry</button>
      </div>
    </div>
  </div>

  <!-- Telemetry Modal -->
  <div class="telemetry-modal" id="tel-modal">
    <div class="telemetry-content">
      <span class="close-modal" id="close-modal">✖</span>
      <h2>📊 Execution Logs</h2>
      <div class="table-wrapper">
        <table id="tel-table">
          <thead>
            <tr><th>Timestamp</th><th>Condition</th><th>Confidence</th><th>Latency (ms)</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    const treatments = {
      "Bacterialblight": "* Immediate Application: Spray appropriate copper-based bactericides (e.g., Copper Oxychloride) paired with low-dose Streptomycin formulations.\\n* Soil Regimes: Halt immediate Nitrogen top-dressing to suppress aggressive succulence. Apply balanced Potassium inputs to reinforce cell walls.\\n* Cultural Controls: Drain stagnating floodwaters temporarily to stop mobile waterborne spread across neighboring terraced rows.",
      "Blast": "* Targeted Fungicides: Deploy immediate foliar sprays utilizing standard highly efficient penetrants such as Tricyclazole, Azoxystrobin, or Isoprothiolane.\\n* Irrigation Stabilization: Avoid chronic dry spells; blast spreads faster under water-stressed leaves. Restore reliable field flooding depth.\\n* Silicon Dressing: Incorporate soluble Silicon root enhancers to block direct fungal germ-tube penetration.",
      "Brownspot": "* Nutrient Top-up: Conduct rapid soil testing. Top-up trace elements including Manganese, Zinc, and general complete NPK fertilizers.\\n* Chemical Prevention: Spray broad-spectrum protectants such as Mancozeb or Propiconazole if spot density scales past 15% leaf surface area.\\n* Seed Preparation: For subsequent planting runs, implement standard seed dressing protocol using Carbendazim.",
      "Tungro": "* Vector Eradication: Stop the transmission engine immediately by targeting Green Leafhoppers using Imidacloprid, Thiamethoxam, or Buprofezin.\\n* Infected Disposal: Clear heavily stunted yellowish clumps to minimize source-inoculum reservoirs.\\n* Synchronous Planting: Enforce strict regional resting windows between crop plantings to break the local hopper lifecycle."
    };

    const fileInput = document.getElementById('file');
    const preview   = document.getElementById('preview');
    const previewWrap = document.getElementById('preview-wrap');
    const btn       = document.getElementById('btn');
    const result    = document.getElementById('result');
    const dropZone  = document.getElementById('drop-zone');
    const fileNameEl = document.getElementById('file-name');
    const gcToggle  = document.getElementById('gc-toggle');
    
    let selectedFile = null;
    let originalImgSrc = '';
    let gradcamImgSrc = '';
    let isGradcam = false;
    let currentData = null;

    function setFile(f) {
      selectedFile = f;
      btn.disabled = !f;
      result.innerHTML = '';
      if (f) {
        originalImgSrc = URL.createObjectURL(f);
        preview.src = originalImgSrc;
        previewWrap.style.display = 'block';
        fileNameEl.textContent = f.name;
        gcToggle.style.display = 'none';
        isGradcam = false;
        gcToggle.textContent = "Show Grad-CAM";
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

    gcToggle.addEventListener('click', () => {
      isGradcam = !isGradcam;
      preview.src = isGradcam ? gradcamImgSrc : originalImgSrc;
      gcToggle.textContent = isGradcam ? "Show Original" : "Show Grad-CAM";
    });

    window.downloadReport = function() {
      if (!currentData) return;
      const d = new Date().toLocaleString();
      const condName = currentData.display_name || currentData.disease_class;
      const protocol = treatments[currentData.disease_class] || "N/A";
      const content = `🌾 Diagnostic Report\nDate: ${d}\n\nCondition : ${condName}\nConfidence: ${(currentData.confidence*100).toFixed(1)}%\nLatency   : ${Math.round(currentData.latency_ms || 0)} ms\n\nRecommended Actions:\n${protocol}\n\n---\nGenerated via AI Diagnostics Pipeline`;
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Diagnostic_Report_${currentData.disease_class}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    };

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
        
        currentData = data;
        
        if (data.grad_cam_base64) {
          gradcamImgSrc = "data:image/png;base64," + data.grad_cam_base64;
          gcToggle.style.display = 'block';
        }

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

        const treatText = treatments[data.disease_class] || "No specific treatment protocol found.";

        result.innerHTML = `
          <div class="result-diagnosis">
            <span class="diag-icon">🔬</span>
            <div>
              <div class="diag-name">${data.display_name || data.disease_class}</div>
              <div class="diag-confidence">Confidence: ${(data.confidence*100).toFixed(1)}% • Latency: ${Math.round(data.latency_ms || 0)}ms</div>
            </div>
          </div>
          ${barsHtml}
          
          <div class="treatment-box">
            <div class="treatment-title">💊 Treatment Recommendations</div>
            ${treatText}
          </div>
          <button class="report-btn" onclick="downloadReport()">📥 Download Summary Report</button>
          
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
    
    // Telemetry Modal Logic
    document.getElementById('open-modal').addEventListener('click', async () => {
      document.getElementById('tel-modal').style.display = 'flex';
      try {
        const res = await fetch('/logs');
        const logs = await res.json();
        const tbody = document.querySelector('#tel-table tbody');
        tbody.innerHTML = logs.map(l => `
          <tr>
            <td style="color:var(--muted)">${l.timestamp}</td>
            <td style="font-weight: 500; color: var(--text);">${l.diagnosed_class}</td>
            <td>${(l.confidence*100).toFixed(1)}%</td>
            <td>${Math.round(l.latency_ms)} ms</td>
          </tr>
        `).join('');
      } catch(e) {
        console.error("Failed to fetch logs", e);
      }
    });
    document.getElementById('close-modal').addEventListener('click', () => {
      document.getElementById('tel-modal').style.display = 'none';
    });
  </script>
</body>
</html>"""

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=422, detail="Upload a JPEG or PNG image")
    
    start_t = time.time()
    try:
        image = Image.open(io.BytesIO(await file.read()))
        model = _load_model()
        
        # Inference
        result = predict_image(model, image)
        
        # Grad-CAM
        try:
            img_tensor = preprocess_image(image)
            heatmap = make_gradcam_heatmap(img_tensor, model)
            overlay = overlay_heatmap(image, heatmap)
            
            # Convert to base64
            is_success, buffer = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            if is_success:
                base64_img = base64.b64encode(buffer).decode("utf-8")
                result["grad_cam_base64"] = base64_img
                result["grad_cam_available"] = True
            else:
                result["grad_cam_available"] = False
        except Exception as ge:
            print("GradCAM error:", ge)
            result["grad_cam_available"] = False
            
        latency_ms = (time.time() - start_t) * 1000
        result["latency_ms"] = latency_ms
        
        # Log telemetry
        log_inference_run("API Upload", result["disease_class"], result["confidence"], latency_ms)
        
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
