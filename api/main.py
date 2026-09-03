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
    return r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="description" content="Rice Leaf Disease Detection — AI-powered plant health analysis using deep learning on Cloud Run." />
  <title>Rice Leaf Disease Detection | AI Diagnostic Platform</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #070d0f;
      --surface: rgba(14, 26, 20, 0.75);
      --surface-card: rgba(18, 34, 26, 0.85);
      --surface-hover: rgba(24, 46, 35, 0.95);
      --border: rgba(52, 211, 153, 0.15);
      --border-hover: rgba(52, 211, 153, 0.4);
      --accent: #10b981;
      --accent2: #34d399;
      --accent-glow: rgba(16, 185, 129, 0.25);
      --text: #f0fdf4;
      --muted: #7ca891;
      --input-bg: rgba(10, 20, 15, 0.7);
      --critical: #ef4444;
      --critical-bg: rgba(239, 68, 68, 0.12);
      --critical-border: rgba(239, 68, 68, 0.35);
      --warning: #f59e0b;
      --warning-bg: rgba(245, 158, 11, 0.12);
      --warning-border: rgba(245, 158, 11, 0.35);
      --safe: #10b981;
      --safe-bg: rgba(16, 185, 129, 0.12);
      --radius: 16px;
      --radius-sm: 10px;
    }

    html, body {
      min-height: 100%;
      background: var(--bg);
      color: var(--text);
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }

    body {
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 2rem 1rem 3.5rem;
      position: relative;
      overflow-x: hidden;
    }

    /* Aurora background ambient glows */
    body::before, body::after {
      content: '';
      position: fixed;
      border-radius: 50%;
      filter: blur(100px);
      pointer-events: none;
      z-index: 0;
    }
    body::before {
      width: 550px; height: 550px;
      background: radial-gradient(circle, rgba(16, 185, 129, 0.16) 0%, transparent 70%);
      top: -120px; left: -100px;
    }
    body::after {
      width: 480px; height: 480px;
      background: radial-gradient(circle, rgba(5, 150, 105, 0.14) 0%, transparent 70%);
      bottom: -100px; right: -80px;
    }

    .container {
      position: relative;
      z-index: 1;
      width: 100%;
      max-width: 680px;
      transition: max-width 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .container.expanded {
      max-width: 880px;
    }

    /* ── Header ── */
    .header {
      text-align: center;
      margin-bottom: 1.75rem;
    }
    .header-icon-wrap {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      width: 68px; height: 68px;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid var(--border);
      border-radius: 20px;
      margin-bottom: 0.85rem;
      box-shadow: 0 0 25px rgba(16, 185, 129, 0.2);
    }
    .header-icon {
      font-size: 2.2rem;
      display: block;
      filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.6));
    }
    .header h1 {
      font-size: 1.95rem;
      font-weight: 800;
      letter-spacing: -0.035em;
      background: linear-gradient(135deg, #6ee7b7 0%, #10b981 50%, #34d399 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin-bottom: 0.35rem;
    }
    .header p {
      font-size: 0.92rem;
      color: var(--muted);
      max-width: 520px;
      margin: 0 auto;
    }
    .header-badges {
      display: flex;
      justify-content: center;
      flex-wrap: wrap;
      gap: 0.5rem;
      margin-top: 0.85rem;
    }
    .h-badge {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      font-size: 0.75rem;
      font-weight: 500;
      color: var(--accent2);
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(52, 211, 153, 0.2);
      padding: 0.25rem 0.65rem;
      border-radius: 99px;
    }

    /* ── Main Card ── */
    .card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.75rem;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 0 8px 36px rgba(0, 0, 0, 0.45), 0 0 0 1px rgba(52, 211, 130, 0.05);
      margin-bottom: 1.25rem;
      transition: border-color 0.3s, box-shadow 0.3s;
    }
    .card:hover {
      border-color: var(--border-hover);
      box-shadow: 0 10px 42px rgba(0, 0, 0, 0.55), 0 0 30px rgba(16, 185, 129, 0.08);
    }

    /* ── Drop Zone ── */
    .drop-zone {
      border: 2px dashed rgba(52, 211, 153, 0.28);
      border-radius: var(--radius-sm);
      padding: 2.2rem 1.5rem;
      text-align: center;
      cursor: pointer;
      transition: all 0.25s ease;
      position: relative;
      background: rgba(10, 20, 15, 0.3);
    }
    .drop-zone:hover, .drop-zone.drag-over {
      border-color: var(--accent2);
      background: rgba(16, 185, 129, 0.08);
      transform: scale(1.005);
    }
    .drop-zone input[type=file] {
      position: absolute; inset: 0;
      opacity: 0; cursor: pointer; width: 100%; height: 100%;
      z-index: 5;
    }
    .drop-icon-wrap {
      width: 52px; height: 52px;
      margin: 0 auto 0.75rem;
      display: flex; align-items: center; justify-content: center;
      background: rgba(16, 185, 129, 0.1);
      border-radius: 14px;
      border: 1px solid rgba(52, 211, 153, 0.2);
    }
    .drop-icon { font-size: 1.8rem; }
    .drop-title {
      font-size: 1rem;
      font-weight: 600;
      color: var(--text);
      margin-bottom: 0.2rem;
    }
    .drop-sub {
      font-size: 0.82rem;
      color: var(--muted);
    }

    /* Input actions (Camera & Browse) */
    .input-actions-bar {
      display: flex;
      justify-content: center;
      gap: 0.6rem;
      margin-top: 1rem;
      z-index: 10;
      position: relative;
    }
    .action-chip-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      padding: 0.45rem 0.85rem;
      background: rgba(16, 185, 129, 0.1);
      border: 1px solid rgba(52, 211, 153, 0.25);
      border-radius: 8px;
      color: var(--accent2);
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .action-chip-btn:hover {
      background: rgba(16, 185, 129, 0.2);
      border-color: var(--accent);
      transform: translateY(-1px);
    }

    /* Sample Leaf Chips */
    .samples-wrap {
      margin-top: 1.25rem;
      padding-top: 1.1rem;
      border-top: 1px solid rgba(255, 255, 255, 0.06);
    }
    .samples-title {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.6rem;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .samples-grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.5rem;
    }
    @media (min-width: 520px) {
      .samples-grid { grid-template-columns: repeat(4, 1fr); }
    }
    .sample-btn {
      display: flex;
      flex-direction: column;
      align-items: flex-start;
      padding: 0.6rem 0.75rem;
      background: rgba(14, 28, 20, 0.6);
      border: 1px solid rgba(52, 211, 153, 0.15);
      border-radius: 8px;
      color: var(--text);
      cursor: pointer;
      transition: all 0.2s;
      text-align: left;
    }
    .sample-btn:hover {
      background: rgba(16, 185, 129, 0.12);
      border-color: var(--accent2);
      transform: translateY(-1px);
    }
    .sample-btn.active {
      background: rgba(16, 185, 129, 0.18);
      border-color: var(--accent);
      box-shadow: 0 0 14px rgba(16, 185, 129, 0.25);
    }
    .sample-btn-name {
      font-size: 0.8rem;
      font-weight: 600;
      color: var(--text);
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      width: 100%;
    }
    .sample-btn-type {
      font-size: 0.68rem;
      color: var(--muted);
      margin-top: 2px;
    }

    /* ── Preview & Visualizer ── */
    #preview-wrap {
      display: none;
      margin-top: 1.25rem;
      background: rgba(9, 18, 14, 0.8);
      border: 1px solid var(--border);
      border-radius: var(--radius-sm);
      padding: 0.85rem;
      position: relative;
    }
    .preview-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.75rem;
      padding-bottom: 0.5rem;
      border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .preview-meta {
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .file-badge {
      font-size: 0.78rem;
      font-weight: 600;
      color: var(--accent2);
      background: rgba(16, 185, 129, 0.12);
      padding: 0.2rem 0.55rem;
      border-radius: 6px;
      max-width: 240px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .clear-btn {
      background: none;
      border: 1px solid rgba(239, 68, 68, 0.3);
      color: #f87171;
      padding: 0.25rem 0.65rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 0.3rem;
      transition: all 0.2s;
    }
    .clear-btn:hover {
      background: rgba(239, 68, 68, 0.15);
      border-color: #ef4444;
    }

    /* Display Views */
    .view-modes-bar {
      display: none;
      align-items: center;
      gap: 0.4rem;
      margin-bottom: 0.85rem;
      flex-wrap: wrap;
    }
    .mode-pill {
      background: rgba(14, 28, 20, 0.7);
      border: 1px solid rgba(52, 211, 153, 0.2);
      color: var(--muted);
      padding: 0.3rem 0.7rem;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
    }
    .mode-pill.active {
      background: var(--accent);
      color: #fff;
      border-color: var(--accent);
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
    }

    .visual-stage {
      position: relative;
      border-radius: 8px;
      overflow: hidden;
      background: #000;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 220px;
    }
    .visual-stage img {
      max-width: 100%;
      max-height: 340px;
      object-fit: contain;
      display: block;
      border-radius: 6px;
    }
    .side-by-side-grid {
      display: none;
      grid-template-columns: 1fr 1fr;
      gap: 0.75rem;
      width: 100%;
    }
    .specimen-pane {
      background: rgba(5, 12, 9, 0.7);
      border-radius: 8px;
      padding: 0.5rem;
      text-align: center;
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .specimen-pane img {
      width: 100%;
      height: 180px;
      object-fit: cover;
      border-radius: 6px;
      display: block;
    }
    .specimen-tag {
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--muted);
      margin-top: 0.4rem;
      display: block;
    }

    /* Slider Container */
    .slider-wrap {
      display: none;
      margin-top: 0.75rem;
      padding: 0.6rem 0.85rem;
      background: rgba(14, 28, 20, 0.6);
      border-radius: 8px;
      border: 1px solid rgba(52, 211, 153, 0.15);
    }
    .slider-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.76rem;
      font-weight: 600;
      color: var(--accent2);
      margin-bottom: 0.35rem;
    }
    .slider-input {
      width: 100%;
      accent-color: var(--accent);
      cursor: pointer;
    }

    /* ── Analyze Button ── */
    #btn {
      display: block;
      width: 100%;
      margin-top: 1.25rem;
      padding: 0.95rem 1.5rem;
      border: none;
      border-radius: 10px;
      font-size: 1.02rem;
      font-weight: 700;
      font-family: inherit;
      cursor: pointer;
      background: linear-gradient(135deg, #059669 0%, #10b981 50%, #34d399 100%);
      color: #fff;
      letter-spacing: 0.01em;
      box-shadow: 0 4px 20px rgba(16, 185, 129, 0.35);
      transition: all 0.2s ease;
      position: relative;
      overflow: hidden;
    }
    #btn:not(:disabled):hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 28px rgba(16, 185, 129, 0.5);
    }
    #btn:disabled {
      opacity: 0.45;
      cursor: not-allowed;
      transform: none;
      box-shadow: none;
    }

    /* Multi-step progress tracker */
    .progress-tracker {
      display: none;
      margin-top: 0.85rem;
      padding: 0.75rem 1rem;
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(52, 211, 153, 0.2);
      border-radius: 8px;
    }
    .progress-step-text {
      font-size: 0.82rem;
      color: var(--accent2);
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 0.5rem;
    }
    .pulse-bar {
      height: 4px;
      background: rgba(52, 211, 153, 0.2);
      border-radius: 99px;
      margin-top: 0.5rem;
      overflow: hidden;
      position: relative;
    }
    .pulse-fill {
      height: 100%;
      width: 40%;
      background: var(--accent);
      border-radius: 99px;
      position: absolute;
      animation: pulse-slide 1.4s ease-in-out infinite;
    }
    @keyframes pulse-slide {
      0% { left: -40%; }
      100% { left: 100%; }
    }

    .hint {
      font-size: 0.78rem;
      color: var(--muted);
      margin-top: 0.75rem;
      text-align: center;
    }

    /* ── Results Panel ── */
    #result {
      margin-top: 1.5rem;
      animation: fadeIn 0.4s ease-out;
    }
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(8px); }
      to   { opacity: 1; transform: translateY(0); }
    }

    /* Hero Diagnosis Banner */
    .diagnosis-hero {
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      background: var(--surface-card);
      border: 1px solid var(--border);
      margin-bottom: 1.25rem;
      position: relative;
      overflow: hidden;
    }
    .diagnosis-hero.critical {
      border-left: 6px solid var(--critical);
      background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(18, 34, 26, 0.9) 100%);
      border-color: var(--critical-border);
    }
    .diagnosis-hero.warning {
      border-left: 6px solid var(--warning);
      background: linear-gradient(135deg, rgba(245, 158, 11, 0.12) 0%, rgba(18, 34, 26, 0.9) 100%);
      border-color: var(--warning-border);
    }
    .diag-top-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 0.5rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }
    .severity-pill {
      font-size: 0.72rem;
      font-weight: 700;
      letter-spacing: 0.05em;
      text-transform: uppercase;
      padding: 0.2rem 0.6rem;
      border-radius: 99px;
    }
    .severity-pill.critical {
      background: var(--critical-bg);
      color: #fca5a5;
      border: 1px solid var(--critical-border);
    }
    .severity-pill.warning {
      background: var(--warning-bg);
      color: #fde68a;
      border: 1px solid var(--warning-border);
    }
    .diag-title {
      font-size: 1.45rem;
      font-weight: 800;
      color: #fff;
      letter-spacing: -0.02em;
    }
    .diag-metrics-row {
      display: flex;
      gap: 1rem;
      margin-top: 0.65rem;
      flex-wrap: wrap;
    }
    .diag-metric-item {
      font-size: 0.84rem;
      color: var(--muted);
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
    }
    .diag-metric-val {
      font-weight: 700;
      color: var(--text);
    }

    .low-conf-alert {
      margin-top: 0.75rem;
      padding: 0.65rem 0.85rem;
      background: var(--warning-bg);
      border: 1px solid var(--warning-border);
      border-radius: 8px;
      font-size: 0.8rem;
      color: #fde68a;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }

    /* Probabilities Section */
    .prob-section {
      background: rgba(12, 24, 18, 0.7);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1.15rem 1.25rem;
      margin-bottom: 1.25rem;
    }
    .section-heading {
      font-size: 0.82rem;
      font-weight: 700;
      color: var(--accent2);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.85rem;
      display: flex;
      align-items: center;
      gap: 0.4rem;
    }
    .prob-item { margin-bottom: 0.65rem; }
    .prob-item:last-child { margin-bottom: 0; }
    .prob-info {
      display: flex;
      justify-content: space-between;
      font-size: 0.82rem;
      font-weight: 500;
      margin-bottom: 0.25rem;
      color: var(--text);
    }
    .prob-pct {
      font-weight: 700;
      color: var(--muted);
    }
    .prob-item.is-top .prob-pct {
      color: var(--accent2);
    }
    .bar-bg {
      height: 8px;
      background: rgba(52, 211, 153, 0.1);
      border-radius: 99px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      background: rgba(52, 211, 153, 0.35);
      border-radius: 99px;
      transition: width 0.85s cubic-bezier(0.16, 1, 0.3, 1);
      width: 0%;
    }
    .prob-item.is-top .bar-fill {
      background: linear-gradient(90deg, #059669, #34d399);
      box-shadow: 0 0 10px rgba(16, 185, 129, 0.5);
    }

    /* Structured Treatment Cards */
    .treatment-container {
      margin-bottom: 1.25rem;
    }
    .treatment-grid {
      display: grid;
      gap: 0.65rem;
      margin-top: 0.5rem;
    }
    .action-card {
      background: rgba(14, 28, 20, 0.6);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.85rem 1rem;
      transition: border-color 0.2s;
    }
    .action-card:hover {
      border-color: var(--border-hover);
    }
    .action-card-title {
      font-size: 0.8rem;
      font-weight: 700;
      color: var(--accent2);
      display: flex;
      align-items: center;
      gap: 0.4rem;
      margin-bottom: 0.35rem;
    }
    .action-card-text {
      font-size: 0.83rem;
      color: #d1fae5;
      line-height: 1.45;
    }

    /* Action Buttons Bar */
    .result-actions-bar {
      display: flex;
      flex-wrap: wrap;
      gap: 0.6rem;
      margin-top: 1.25rem;
    }
    .result-action-btn {
      display: inline-flex;
      align-items: center;
      gap: 0.4rem;
      background: rgba(16, 185, 129, 0.12);
      border: 1px solid rgba(52, 211, 153, 0.3);
      border-radius: 8px;
      padding: 0.55rem 1rem;
      color: var(--text);
      font-size: 0.82rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s;
      font-family: inherit;
    }
    .result-action-btn:hover {
      background: rgba(16, 185, 129, 0.22);
      border-color: var(--accent);
      transform: translateY(-1px);
    }
    .result-action-btn.secondary {
      background: rgba(255, 255, 255, 0.05);
      border-color: rgba(255, 255, 255, 0.15);
      color: var(--muted);
    }
    .result-action-btn.secondary:hover {
      background: rgba(255, 255, 255, 0.1);
      color: var(--text);
    }

    /* Raw JSON details */
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
      gap: 0.3rem;
    }
    .raw-json pre {
      margin-top: 0.5rem;
      background: rgba(5, 12, 9, 0.8);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.85rem;
      font-size: 0.74rem;
      color: #a7f3d0;
      overflow-x: auto;
      line-height: 1.45;
    }

    /* ── Links Card ── */
    .links-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.15rem 1.75rem;
      backdrop-filter: blur(20px);
      -webkit-backdrop-filter: blur(20px);
      box-shadow: 0 4px 25px rgba(0,0,0,0.3);
    }
    .links-title {
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.07em;
      color: var(--muted);
      margin-bottom: 0.65rem;
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
      padding: 0.35rem 0.8rem;
      border-radius: 99px;
      border: 1px solid var(--border);
      font-size: 0.8rem;
      font-weight: 500;
      color: var(--accent2);
      text-decoration: none;
      transition: all 0.2s;
      cursor: pointer;
      background: rgba(16, 185, 129, 0.05);
      font-family: inherit;
    }
    .link-pill:hover {
      border-color: var(--accent);
      background: rgba(16, 185, 129, 0.15);
      transform: translateY(-1px);
    }

    /* Telemetry Modal */
    .telemetry-modal {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.82);
      z-index: 100;
      align-items: center;
      justify-content: center;
      backdrop-filter: blur(6px);
      padding: 1rem;
    }
    .telemetry-content {
      background: #0b1512;
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 1.75rem;
      width: 100%;
      max-width: 820px;
      max-height: 85vh;
      overflow-y: auto;
      box-shadow: 0 20px 50px rgba(0,0,0,0.8);
      position: relative;
    }
    .telemetry-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 1.25rem;
    }
    .telemetry-header h2 {
      font-size: 1.35rem;
      font-weight: 700;
      color: #fff;
      display: flex;
      align-items: center;
      gap: 0.45rem;
    }
    .close-modal {
      background: rgba(255,255,255,0.08);
      border: none;
      color: var(--muted);
      width: 32px; height: 32px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 1.1rem;
      display: flex; align-items: center; justify-content: center;
      transition: all 0.2s;
    }
    .close-modal:hover {
      background: rgba(239, 68, 68, 0.2);
      color: #f87171;
    }
    .kpi-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 0.75rem;
      margin-bottom: 1.25rem;
    }
    .kpi-card {
      background: rgba(16, 185, 129, 0.08);
      border: 1px solid rgba(52, 211, 153, 0.2);
      border-radius: 10px;
      padding: 0.85rem;
      text-align: center;
    }
    .kpi-label {
      font-size: 0.72rem;
      color: var(--muted);
      text-transform: uppercase;
      font-weight: 600;
      margin-bottom: 0.2rem;
    }
    .kpi-val {
      font-size: 1.25rem;
      font-weight: 800;
      color: var(--accent2);
    }
    .table-wrapper {
      overflow-x: auto;
      border: 1px solid rgba(255, 255, 255, 0.06);
      border-radius: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.82rem;
    }
    th, td {
      padding: 0.65rem 0.85rem;
      text-align: left;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    th {
      background: rgba(14, 28, 20, 0.8);
      color: var(--accent2);
      font-weight: 600;
    }
    tr:hover td {
      background: rgba(255, 255, 255, 0.02);
    }

    /* Toast notification */
    #toast {
      position: fixed;
      bottom: 2rem;
      left: 50%;
      transform: translateX(-50%) translateY(100px);
      background: #10b981;
      color: #fff;
      padding: 0.65rem 1.25rem;
      border-radius: 99px;
      font-size: 0.85rem;
      font-weight: 600;
      box-shadow: 0 8px 24px rgba(16, 185, 129, 0.4);
      z-index: 200;
      pointer-events: none;
      transition: transform 0.3s cubic-bezier(0.16, 1, 0.3, 1), opacity 0.3s;
      opacity: 0;
    }
    #toast.show {
      transform: translateX(-50%) translateY(0);
      opacity: 1;
    }

    /* Footer */
    .footer {
      text-align: center;
      margin-top: 1.5rem;
      font-size: 0.78rem;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <div class="container" id="main-container">
    <!-- Header -->
    <div class="header">
      <div class="header-icon-wrap">
        <span class="header-icon">🌾</span>
      </div>
      <h1>Rice Leaf Disease Detection</h1>
      <p>AI-powered agricultural diagnostic intelligence — Instant classification with Grad-CAM spatial explainability.</p>
      <div class="header-badges">
        <span class="h-badge">🌿 4 Disease Classes</span>
        <span class="h-badge">🧠 EfficientNetB0</span>
        <span class="h-badge">🔬 Grad-CAM Attention</span>
        <span class="h-badge">⚡ Real-Time API</span>
      </div>
    </div>

    <!-- Main Card -->
    <div class="card" id="main-card">
      <!-- Drop Zone -->
      <div class="drop-zone" id="drop-zone">
        <input type="file" id="file" accept="image/jpeg,image/png,image/jpg" />
        <div class="drop-icon-wrap">
          <span class="drop-icon">🍃</span>
        </div>
        <div class="drop-title">Drop a rice leaf image here, or click to browse</div>
        <div class="drop-sub">Supports JPEG &amp; PNG • High resolution leaf specimen recommended</div>
      </div>

      <!-- Quick Action Buttons -->
      <div class="input-actions-bar">
        <button class="action-chip-btn" id="browse-btn">
          📁 Browse Files
        </button>
        <button class="action-chip-btn" id="camera-btn">
          📷 Take Photo
        </button>
        <input type="file" id="camera-file" accept="image/*" capture="environment" style="display:none;" />
      </div>

      <!-- Quick Test Samples Selector -->
      <div class="samples-wrap" id="samples-wrap">
        <div class="samples-title">
          <span>🌿 Or Try a Verified Specimen:</span>
          <span style="font-size:0.7rem; color:var(--muted); font-weight:400;">1-Click Demo</span>
        </div>
        <div class="samples-grid">
          <button class="sample-btn" data-class="Bacterialblight">
            <span class="sample-btn-name">🍂 Bacterial Blight</span>
            <span class="sample-btn-type">Bacterial • Streaks</span>
          </button>
          <button class="sample-btn" data-class="Blast">
            <span class="sample-btn-name">🌾 Rice Blast</span>
            <span class="sample-btn-type">Fungal • Spindles</span>
          </button>
          <button class="sample-btn" data-class="Brownspot">
            <span class="sample-btn-name">🍁 Brown Spot</span>
            <span class="sample-btn-type">Fungal • Lesions</span>
          </button>
          <button class="sample-btn" data-class="Tungro">
            <span class="sample-btn-name">🍂 Tungro Disease</span>
            <span class="sample-btn-type">Viral • Yellowing</span>
          </button>
        </div>
      </div>

      <!-- Specimen Visualizer / Preview -->
      <div id="preview-wrap">
        <div class="preview-header">
          <div class="preview-meta">
            <span class="file-badge" id="file-name">Specimen</span>
            <span style="font-size:0.75rem; color:var(--muted);" id="file-size"></span>
          </div>
          <button class="clear-btn" id="clear-btn">✕ Clear</button>
        </div>

        <!-- Post-inference mode switchers -->
        <div class="view-modes-bar" id="view-modes-bar">
          <button class="mode-pill active" data-mode="side">🖼️ Side-by-Side</button>
          <button class="mode-pill" data-mode="slider">🔀 Opacity Slider</button>
          <button class="mode-pill" data-mode="original">🍃 Specimen Only</button>
          <button class="mode-pill" data-mode="gradcam">🧠 Heatmap Only</button>
        </div>

        <!-- Visual Stage -->
        <div class="visual-stage" id="visual-stage">
          <img id="preview" alt="Leaf Specimen" />
        </div>

        <!-- Side-by-side view -->
        <div class="side-by-side-grid" id="side-by-side-grid">
          <div class="specimen-pane">
            <img id="sbs-original" alt="Original Specimen" />
            <span class="specimen-tag">Captured Leaf Specimen</span>
          </div>
          <div class="specimen-pane">
            <img id="sbs-gradcam" alt="Grad-CAM Activation" />
            <span class="specimen-tag">Grad-CAM Activation Focus</span>
          </div>
        </div>

        <!-- Opacity Slider -->
        <div class="slider-wrap" id="slider-wrap">
          <div class="slider-label">
            <span>Attention Overlay Opacity</span>
            <span id="slider-val">45%</span>
          </div>
          <input type="range" class="slider-input" id="opacity-slider" min="0" max="100" value="45" />
        </div>
      </div>

      <!-- Primary Action Button -->
      <button id="btn" disabled>⚡ Analyze Specimen via API</button>

      <!-- Multi-step animated progress tracker -->
      <div class="progress-tracker" id="progress-tracker">
        <div class="progress-step-text" id="progress-step-text">
          <span style="display:inline-block; animation:spin 1s linear infinite;">⏳</span> Ingesting &amp; preprocessing specimen...
        </div>
        <div class="pulse-bar">
          <div class="pulse-fill"></div>
        </div>
      </div>

      <p class="hint" id="idle-hint">⏱ Cold starts on Cloud Run may take 1–2 min when waking from idle.</p>

      <!-- Dynamic Diagnostic Results -->
      <div id="result"></div>
    </div>

    <!-- System Links Card -->
    <div class="links-card">
      <div class="links-title">System Endpoints &amp; Telemetry</div>
      <div class="links-row">
        <a class="link-pill" href="/docs">📄 Swagger UI</a>
        <a class="link-pill" href="/health">💚 API Health</a>
        <button class="link-pill" id="open-modal">📊 View Telemetry</button>
      </div>
    </div>

    <!-- Footer -->
    <div class="footer">
      🌾 Institutional Rice Leaf Disease Diagnostic Pipeline • Engineered with Keras 3 &amp; PyTorch Engine
    </div>
  </div>

  <!-- Telemetry Slide-over Modal -->
  <div class="telemetry-modal" id="tel-modal">
    <div class="telemetry-content">
      <div class="telemetry-header">
        <h2>📊 Diagnostic Telemetry &amp; Execution Logs</h2>
        <button class="close-modal" id="close-modal">✕</button>
      </div>

      <div class="kpi-row">
        <div class="kpi-card">
          <div class="kpi-label">Total Analyzed</div>
          <div class="kpi-val" id="kpi-total">—</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Average Latency</div>
          <div class="kpi-val" id="kpi-latency">—</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Top Pathogen</div>
          <div class="kpi-val" id="kpi-top" style="font-size:1rem; padding-top:4px;">—</div>
        </div>
      </div>

      <div class="table-wrapper">
        <table id="tel-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Diagnosed Condition</th>
              <th>Confidence</th>
              <th>Latency (ms)</th>
            </tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Toast Notification -->
  <div id="toast">Copied summary to clipboard!</div>

  <script>
    window.SAMPLE_DATA = {"Brownspot":"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","Tungro":"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","Bacterialblight":"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","Blast":"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"};

    // Structured Agronomic Countermeasure Protocols
    const treatments = {
      "Bacterialblight": {
        immediate: "Drain stagnating floodwaters temporarily to halt waterborne bacterial transmission across neighboring rows.",
        chemical: "Deploy copper-based bactericides (e.g., Copper Oxychloride at 2.5g/L) paired with low-dose Streptomycin formulations.",
        cultural: "Halt Nitrogen top-dressing immediately to avoid succulent leaf growth. Apply balanced Potassium to reinforce leaf cell walls."
      },
      "Blast": {
        immediate: "Avoid drought stress; blast spreads aggressively on moisture-deficient plants. Maintain recommended field water depth.",
        chemical: "Apply systemic foliar penetrants such as Tricyclazole 75% WP (0.6g/L), Azoxystrobin, or Isoprothiolane at onset.",
        cultural: "Incorporate soluble Silicon root enhancers to fortify epidermal barriers against direct fungal germ-tube penetration."
      },
      "Brownspot": {
        immediate: "Conduct rapid soil health assessment to detect micronutrient exhaustion (particularly Potassium, Manganese, or Zinc).",
        chemical: "Apply broad-spectrum protectants such as Mancozeb (2g/L) or Propiconazole if spot coverage exceeds 15% leaf surface area.",
        cultural: "Top-up balanced NPK fertilizer regimes. Dress subsequent seasonal seed stocks with Carbendazim (2g/kg seed)."
      },
      "Tungro": {
        immediate: "Interrupt the viral vector transmission engine by eradicating Green Leafhoppers (Nephotettix virescens) immediately.",
        chemical: "Spray systemic vector insecticides: Imidacloprid 17.8% SL (0.3ml/L), Thiamethoxam, or Buprofezin.",
        cultural: "Rogue out and destroy stunted yellowish clumps. Enforce regional synchronous planting breaks to disrupt the hopper lifecycle."
      }
    };

    const displayNames = {
      "Bacterialblight": "Bacterial Leaf Blight",
      "Blast": "Rice Blast",
      "Brownspot": "Brown Spot",
      "Tungro": "Rice Tungro Disease"
    };

    // DOM Elements
    const fileInput     = document.getElementById('file');
    const cameraInput   = document.getElementById('camera-file');
    const browseBtn     = document.getElementById('browse-btn');
    const cameraBtn     = document.getElementById('camera-btn');
    const dropZone      = document.getElementById('drop-zone');
    const previewWrap   = document.getElementById('preview-wrap');
    const preview       = document.getElementById('preview');
    const fileNameEl    = document.getElementById('file-name');
    const fileSizeEl    = document.getElementById('file-size');
    const clearBtn      = document.getElementById('clear-btn');
    const btn           = document.getElementById('btn');
    const result        = document.getElementById('result');
    const progressTracker = document.getElementById('progress-tracker');
    const progressStepText = document.getElementById('progress-step-text');
    const idleHint      = document.getElementById('idle-hint');
    const mainContainer = document.getElementById('main-container');

    // Visualizer Controls
    const viewModesBar    = document.getElementById('view-modes-bar');
    const visualStage     = document.getElementById('visual-stage');
    const sideBySideGrid  = document.getElementById('side-by-side-grid');
    const sbsOriginal     = document.getElementById('sbs-original');
    const sbsGradcam      = document.getElementById('sbs-gradcam');
    const sliderWrap      = document.getElementById('slider-wrap');
    const opacitySlider   = document.getElementById('opacity-slider');
    const sliderVal       = document.getElementById('slider-val');

    let selectedFile    = null;
    let originalImgSrc  = '';
    let gradcamImgSrc   = '';
    let currentData     = null;
    let blendCanvas     = null;
    let origImageObj    = null;
    let gradcamImageObj = null;

    // Helper: Format File Size
    function formatBytes(bytes) {
      if (!bytes || bytes === 0) return '';
      const k = 1024;
      const sizes = ['B', 'KB', 'MB'];
      const i = Math.floor(Math.log(bytes) / Math.log(k));
      return (bytes / Math.pow(k, i)).toFixed(1) + ' ' + sizes[i];
    }

    // Helper: Convert Base64 dataURL to Blob
    function dataURLtoBlob(dataurl) {
      const arr = dataurl.split(','), mime = arr[0].match(/:(.*?);/)[1],
            bstr = atob(arr[1]);
      let n = bstr.length, u8arr = new Uint8Array(n);
      while(n--) u8arr[n] = bstr.charCodeAt(n);
      return new Blob([u8arr], {type: mime});
    }

    // Trigger file inputs
    browseBtn.addEventListener('click', (e) => { e.preventDefault(); fileInput.click(); });
    cameraBtn.addEventListener('click', (e) => { e.preventDefault(); cameraInput.click(); });

    // Set Active Specimen File
    function setFile(f, customName = null) {
      selectedFile = f;
      btn.disabled = !f;
      result.innerHTML = '';
      resetVisualizer();

      if (f) {
        originalImgSrc = URL.createObjectURL(f);
        preview.src = originalImgSrc;
        previewWrap.style.display = 'block';
        fileNameEl.textContent = customName || f.name;
        fileSizeEl.textContent = formatBytes(f.size);
        btn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        previewWrap.style.display = 'none';
        fileNameEl.textContent = '';
        fileSizeEl.textContent = '';
        mainContainer.classList.remove('expanded');
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      }
    }

    // File Input Listeners
    fileInput.addEventListener('change', () => {
      document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      setFile(fileInput.files[0] || null);
    });
    cameraInput.addEventListener('change', () => {
      document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
      setFile(cameraInput.files[0] || null);
    });

    // Drag and Drop
    dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', e => {
      e.preventDefault();
      dropZone.classList.remove('drag-over');
      const f = e.dataTransfer.files[0];
      if (f && (f.type === 'image/jpeg' || f.type === 'image/png')) {
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
        setFile(f);
      }
    });

    // Clear Button
    clearBtn.addEventListener('click', () => {
      fileInput.value = '';
      cameraInput.value = '';
      setFile(null);
    });

    // Sample Specimen Selection
    document.querySelectorAll('.sample-btn').forEach(btnEl => {
      btnEl.addEventListener('click', () => {
        const cls = btnEl.dataset.class;
        document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('active'));
        btnEl.classList.add('active');

        if (window.SAMPLE_DATA && window.SAMPLE_DATA[cls]) {
          const b64 = window.SAMPLE_DATA[cls];
          const blob = dataURLtoBlob('data:image/jpeg;base64,' + b64);
          blob.name = `Sample_${cls}.jpg`;
          setFile(blob, `Sample: ${displayNames[cls] || cls}`);
        }
      });
    });

    // Visualizer Mode Reset
    function resetVisualizer() {
      viewModesBar.style.display = 'none';
      sideBySideGrid.style.display = 'none';
      sliderWrap.style.display = 'none';
      visualStage.style.display = 'flex';
      preview.style.display = 'block';
      if (blendCanvas) { blendCanvas.remove(); blendCanvas = null; }
    }

    // Set Visualizer Mode
    function setVisualMode(mode) {
      document.querySelectorAll('.mode-pill').forEach(p => {
        p.classList.toggle('active', p.dataset.mode === mode);
      });

      if (mode === 'side') {
        visualStage.style.display = 'none';
        sliderWrap.style.display = 'none';
        sideBySideGrid.style.display = 'grid';
        sbsOriginal.src = originalImgSrc;
        sbsGradcam.src = gradcamImgSrc;
      } else if (mode === 'slider') {
        sideBySideGrid.style.display = 'none';
        visualStage.style.display = 'flex';
        sliderWrap.style.display = 'block';
        preview.style.display = 'none';
        initBlendCanvas();
      } else if (mode === 'original') {
        sideBySideGrid.style.display = 'none';
        sliderWrap.style.display = 'none';
        visualStage.style.display = 'flex';
        if (blendCanvas) blendCanvas.style.display = 'none';
        preview.style.display = 'block';
        preview.src = originalImgSrc;
      } else if (mode === 'gradcam') {
        sideBySideGrid.style.display = 'none';
        sliderWrap.style.display = 'none';
        visualStage.style.display = 'flex';
        if (blendCanvas) blendCanvas.style.display = 'none';
        preview.style.display = 'block';
        preview.src = gradcamImgSrc;
      }
    }

    document.querySelectorAll('.mode-pill').forEach(pill => {
      pill.addEventListener('click', () => setVisualMode(pill.dataset.mode));
    });

    // Interactive Heatmap Alpha Canvas
    function initBlendCanvas() {
      if (!blendCanvas) {
        blendCanvas = document.createElement('canvas');
        blendCanvas.style.maxWidth = '100%';
        blendCanvas.style.maxHeight = '340px';
        blendCanvas.style.borderRadius = '6px';
        visualStage.appendChild(blendCanvas);

        origImageObj = new Image();
        origImageObj.src = originalImgSrc;

        gradcamImageObj = new Image();
        gradcamImageObj.src = gradcamImgSrc;

        Promise.all([
          new Promise(res => { origImageObj.onload = res; }),
          new Promise(res => { gradcamImageObj.onload = res; })
        ]).then(() => {
          blendCanvas.width = origImageObj.naturalWidth || 224;
          blendCanvas.height = origImageObj.naturalHeight || 224;
          drawBlend(opacitySlider.value / 100);
        });
      } else {
        blendCanvas.style.display = 'block';
        drawBlend(opacitySlider.value / 100);
      }
    }

    function drawBlend(alpha) {
      if (!blendCanvas || !origImageObj || !gradcamImageObj) return;
      const ctx = blendCanvas.getContext('2d');
      ctx.clearRect(0, 0, blendCanvas.width, blendCanvas.height);
      ctx.globalAlpha = 1.0;
      ctx.drawImage(origImageObj, 0, 0, blendCanvas.width, blendCanvas.height);
      ctx.globalAlpha = alpha;
      ctx.drawImage(gradcamImageObj, 0, 0, blendCanvas.width, blendCanvas.height);
    }

    opacitySlider.addEventListener('input', (e) => {
      const val = e.target.value;
      sliderVal.textContent = val + '%';
      drawBlend(val / 100);
    });

    // Clipboard Copy Toast
    function showToast(msg = 'Copied summary to clipboard!') {
      const toast = document.getElementById('toast');
      toast.textContent = msg;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 2400);
    }

    window.copySummary = function() {
      if (!currentData) return;
      const cls = currentData.disease_class;
      const name = currentData.display_name || cls;
      const conf = (currentData.confidence * 100).toFixed(1);
      const prot = treatments[cls] || {};
      const summary = `🌾 Rice AI Diagnostic Report\nDisease: ${name}\nConfidence: ${conf}%\nLatency: ${Math.round(currentData.latency_ms || 0)}ms\n\nImmediate Action: ${prot.immediate || 'N/A'}\nChemical Protocol: ${prot.chemical || 'N/A'}\nCultural Practice: ${prot.cultural || 'N/A'}`;
      navigator.clipboard.writeText(summary).then(() => showToast());
    };

    window.downloadReport = function() {
      if (!currentData) return;
      const cls = currentData.disease_class;
      const name = currentData.display_name || cls;
      const d = new Date().toLocaleString();
      const prot = treatments[cls] || {};
      const content = `🌾 RICE LEAF DISEASE DIAGNOSTIC REPORT
Generated via Intelligent Computer Vision Pipeline
Date: ${d}
=====================================================
DIAGNOSTIC SUMMARY:
Condition Diagnosed : ${name} (${cls})
Certainty Level     : ${(currentData.confidence * 100).toFixed(1)}%
Inference Latency   : ${Math.round(currentData.latency_ms || 0)} ms

AGRONOMIC ACTION PLAN:
[⚡ Immediate Intervention]
${prot.immediate || 'N/A'}

[🧪 Targeted Chemical / Spray Regime]
${prot.chemical || 'N/A'}

[🌱 Cultural & Soil Practices]
${prot.cultural || 'N/A'}

=====================================================
Neural Architecture: EfficientNetB0 Transfer Learning
Explainability Engine: Spatial Grad-CAM Layer Mapping
`;
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `Diagnostic_Report_${cls}_${Date.now()}.txt`;
      a.click();
      URL.revokeObjectURL(url);
    };

    window.resetAnalysis = function() {
      setFile(null);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    };

    // Analyze Click Handler
    btn.addEventListener('click', async () => {
      if (!selectedFile) return;
      btn.disabled = true;
      btn.innerHTML = '⚡ Running Deep Neural Analysis…';
      progressTracker.style.display = 'block';
      idleHint.style.display = 'block';
      result.innerHTML = '';

      // Progress animation steps
      const steps = [
        "Ingesting & preprocessing specimen...",
        "Neural inference via EfficientNetB0...",
        "Resolving Grad-CAM spatial activation vectors..."
      ];
      let stepIdx = 0;
      const stepTimer = setInterval(() => {
        stepIdx = (stepIdx + 1) % steps.length;
        progressStepText.innerHTML = `<span style="display:inline-block; animation:spin 1s linear infinite;">⏳</span> ${steps[stepIdx]}`;
      }, 1500);

      const form = new FormData();
      form.append('file', selectedFile);

      try {
        const res = await fetch('/predict', { method: 'POST', body: form });
        clearInterval(stepTimer);
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.statusText);

        currentData = data;
        mainContainer.classList.add('expanded');

        // Grad-CAM Visualizer Setup
        if (data.grad_cam_base64) {
          gradcamImgSrc = "data:image/png;base64," + data.grad_cam_base64;
          viewModesBar.style.display = 'flex';
          setVisualMode('side');
        }

        const cls = data.disease_class;
        const dispName = data.display_name || displayNames[cls] || cls;
        const confPct = (data.confidence * 100).toFixed(1);
        const isCritical = ['Bacterialblight', 'Blast', 'Tungro'].includes(cls);
        const severityTag = isCritical ? 'CRITICAL ALERT' : 'MODERATE WARNING';
        const severityClass = isCritical ? 'critical' : 'warning';
        const prot = treatments[cls] || {
          immediate: "Quarantine infected zone and monitor neighboring crop rows.",
          chemical: "Consult regional agricultural extension office for certified protectants.",
          cultural: "Ensure balanced nutrient top-dressing and sanitize harvesting blades."
        };

        // Class Probability Bars
        const probs = data.probabilities || {};
        let barsHtml = '';
        for (const [pCls, pVal] of Object.entries(probs)) {
          const isTop = (pCls === cls);
          const pDisp = displayNames[pCls] || pCls;
          barsHtml += `
            <div class="prob-item ${isTop ? 'is-top' : ''}">
              <div class="prob-info">
                <span>${isTop ? '✓ ' : ''}${pDisp}</span>
                <span class="prob-pct">${(pVal * 100).toFixed(1)}%</span>
              </div>
              <div class="bar-bg">
                <div class="bar-fill" data-pct="${pVal * 100}"></div>
              </div>
            </div>`;
        }

        // Low confidence warning
        let lowConfHtml = '';
        if (data.confidence < 0.70) {
          lowConfHtml = `
            <div class="low-conf-alert">
              <span>⚠️</span>
              <div>Diagnostic Certainty is below 70%. Ensure the leaf is well-lit and focused, or re-examine for secondary pathogens.</div>
            </div>`;
        }

        result.innerHTML = `
          <!-- Hero Diagnosis Banner -->
          <div class="diagnosis-hero ${severityClass}">
            <div class="diag-top-row">
              <span class="severity-pill ${severityClass}">${severityTag}</span>
              <span style="font-size:0.75rem; color:var(--muted)">Verified via EfficientNetB0</span>
            </div>
            <div class="diag-title">${dispName}</div>
            <div class="diag-metrics-row">
              <span class="diag-metric-item">Certainty: <span class="diag-metric-val">${confPct}%</span></span>
              <span class="diag-metric-item">Latency: <span class="diag-metric-val">${Math.round(data.latency_ms || 0)} ms</span></span>
              <span class="diag-metric-item">Specimen: <span class="diag-metric-val">${data.grad_cam_available ? 'Grad-CAM Mapped' : 'Standard'}</span></span>
            </div>
            ${lowConfHtml}
          </div>

          <!-- Probability Distribution -->
          <div class="prob-section">
            <div class="section-heading">📊 Multiclass Confidence Distribution</div>
            ${barsHtml}
          </div>

          <!-- Structured Agronomic Treatment Cards -->
          <div class="treatment-container">
            <div class="section-heading">🌾 Prescriptive Field Countermeasures</div>
            <div class="treatment-grid">
              <div class="action-card">
                <div class="action-card-title">⚡ Immediate Intervention</div>
                <div class="action-card-text">${prot.immediate}</div>
              </div>
              <div class="action-card">
                <div class="action-card-title">🧪 Targeted Chemical Protocol</div>
                <div class="action-card-text">${prot.chemical}</div>
              </div>
              <div class="action-card">
                <div class="action-card-title">🌱 Soil &amp; Cultural Regimes</div>
                <div class="action-card-text">${prot.cultural}</div>
              </div>
            </div>
          </div>

          <!-- Action Buttons Bar -->
          <div class="result-actions-bar">
            <button class="result-action-btn" onclick="downloadReport()">
              📥 Download Printable Report (.txt)
            </button>
            <button class="result-action-btn secondary" onclick="copySummary()">
              📋 Copy Summary
            </button>
            <button class="result-action-btn secondary" onclick="resetAnalysis()">
              🔄 Analyze Another Leaf
            </button>
          </div>

          <!-- Collapsible Raw JSON -->
          <details class="raw-json">
            <summary>View raw JSON payload</summary>
            <pre>${JSON.stringify(data, null, 2)}</pre>
          </details>
        `;

        // Animate Probability Bars
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            document.querySelectorAll('.bar-fill[data-pct]').forEach(el => {
              el.style.width = el.dataset.pct + '%';
            });
          });
        });

        // Scroll into view
        result.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

      } catch (err) {
        clearInterval(stepTimer);
        result.innerHTML = `
          <div style="padding:1rem; border-radius:10px; background:var(--critical-bg); border:1px solid var(--critical-border); color:#fca5a5; font-size:0.88rem;">
            ⚠️ <b>Analysis Error:</b> ${err.message}
          </div>`;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '⚡ Analyze Specimen via API';
        progressTracker.style.display = 'none';
      }
    });

    // Telemetry Modal & KPI Calculation
    document.getElementById('open-modal').addEventListener('click', async () => {
      document.getElementById('tel-modal').style.display = 'flex';
      try {
        const res = await fetch('/logs');
        const logs = await res.json();
        const tbody = document.querySelector('#tel-table tbody');

        if (logs.length > 0) {
          // Compute KPIs
          document.getElementById('kpi-total').textContent = logs.length;
          const avgMs = logs.reduce((acc, r) => acc + (parseFloat(r.latency_ms) || 0), 0) / logs.length;
          document.getElementById('kpi-latency').textContent = Math.round(avgMs) + ' ms';

          const counts = {};
          logs.forEach(r => { counts[r.diagnosed_class] = (counts[r.diagnosed_class] || 0) + 1; });
          const topCls = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
          document.getElementById('kpi-top').textContent = displayNames[topCls] || topCls;

          tbody.innerHTML = logs.map(l => {
            const isCrit = ['Bacterialblight', 'Blast', 'Tungro'].includes(l.diagnosed_class);
            const color = isCrit ? '#f87171' : '#fbbf24';
            return `
              <tr>
                <td style="color:var(--muted)">${l.timestamp}</td>
                <td style="font-weight:600; color:${color}">${displayNames[l.diagnosed_class] || l.diagnosed_class}</td>
                <td>${(parseFloat(l.confidence) * 100).toFixed(1)}%</td>
                <td>${Math.round(parseFloat(l.latency_ms) || 0)} ms</td>
              </tr>`;
          }).join('');
        } else {
          tbody.innerHTML = `<tr><td colspan="4" style="text-align:center; color:var(--muted); padding:1.5rem;">No execution logs found yet.</td></tr>`;
        }
      } catch (err) {
        console.error('Failed to load telemetry:', err);
      }
    });

    document.getElementById('close-modal').addEventListener('click', () => {
      document.getElementById('tel-modal').style.display = 'none';
    });
    document.getElementById('tel-modal').addEventListener('click', (e) => {
      if (e.target.id === 'tel-modal') document.getElementById('tel-modal').style.display = 'none';
    });
  </script>
</body>
</html>
"""

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
