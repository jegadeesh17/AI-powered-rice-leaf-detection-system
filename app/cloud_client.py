"""Streamlit UI that calls the Cloud Run FastAPI — no local model required."""

from __future__ import annotations

import io
import os
import time

import httpx
import streamlit as st
from PIL import Image

DISPLAY_NAMES = {
    "Bacterialblight": "Bacterial Leaf Blight",
    "Blast": "Rice Blast",
    "Brownspot": "Brown Spot",
    "Tungro": "Rice Tungro Disease",
}
CLASS_ORDER = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]
DEFAULT_API_URL = "https://rice-leaf-api-5obmkzpuaa-el.a.run.app"


def get_api_base_url() -> str:
    try:
        url = st.secrets.get("API_BASE_URL", "")
    except (FileNotFoundError, AttributeError):
        url = ""
    if not url:
        url = os.getenv("API_BASE_URL", DEFAULT_API_URL)
    return url.rstrip("/")


def check_health(client: httpx.Client, base_url: str) -> dict | None:
    try:
        response = client.get(f"{base_url}/health", timeout=120.0)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.sidebar.error(f"API health check failed: {exc}")
        return None


def predict_via_api(client: httpx.Client, base_url: str, image: Image.Image) -> dict:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    response = client.post(
        f"{base_url}/predict",
        files={"file": ("leaf.png", buffer.getvalue(), "image/png")},
        timeout=180.0,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Rice AI — API Client",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
      }
      .banner h1 { color: white; margin: 0; font-size: 1.8rem; }
      .banner p { margin: 0.4rem 0 0; opacity: 0.9; }
      .api-pill {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.85rem;
        margin-top: 0.5rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

api_url = get_api_base_url()

st.markdown(
    f"""
    <div class="banner">
      <h1>🌾 Rice Leaf Diagnostics — API Client</h1>
      <p>Uploads images to the Cloud Run inference API (no local model loaded).</p>
      <span class="api-pill">Backend: {api_url}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Architecture")
    st.markdown(
        """
        **This app (head)** → HTTP → **FastAPI on Cloud Run (body)**

        - UI only: Streamlit Cloud
        - Inference: remote `/predict`
        - Local full app (`app.py`) still has Grad-CAM + SQLite
        """
    )
    st.markdown("### API status")
    with httpx.Client() as client:
        health = check_health(client, api_url)
    if health:
        loaded = health.get("model_loaded", False)
        st.success("API reachable")
        st.write(f"Model loaded: **{'yes' if loaded else 'no'}**")
    st.caption("First request after idle may take 1–2 min (cold start).")

uploaded = st.file_uploader("Upload rice leaf image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded is None:
    st.info("Upload a leaf image to run diagnosis via the REST API.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
col_img, col_res = st.columns([1, 1.2], gap="large")

with col_img:
    st.markdown("#### Input specimen")
    st.image(image, use_container_width=True)

with col_res:
    st.markdown("#### API response")
    with st.spinner("Calling Cloud Run inference API…"):
        start = time.time()
        try:
            with httpx.Client() as client:
                result = predict_via_api(client, api_url, image)
            elapsed_ms = (time.time() - start) * 1000
        except httpx.HTTPStatusError as exc:
            st.error(f"API error {exc.response.status_code}: {exc.response.text}")
            st.stop()
        except Exception as exc:
            st.error(f"Request failed: {exc}")
            st.caption("Cold start? Wait 60s and try again.")
            st.stop()

    disease = result.get("disease_class", "Unknown")
    display = result.get("display_name") or DISPLAY_NAMES.get(disease, disease)
    confidence = float(result.get("confidence", 0))
    probs = result.get("probabilities", {})

    m1, m2 = st.columns(2)
    with m1:
        st.metric("Diagnosis", display)
    with m2:
        st.metric("Confidence", f"{confidence * 100:.1f}%")

    st.metric("Round-trip latency", f"{elapsed_ms:.0f} ms")
    st.caption("Includes network + Cloud Run inference (first call may be slower).")

    st.markdown("#### Class probabilities")
    for cls in CLASS_ORDER:
        prob = float(probs.get(cls, 0))
        st.markdown(f"**{DISPLAY_NAMES.get(cls, cls)}**")
        st.progress(prob, text=f"{prob * 100:.1f}%")

    with st.expander("Raw JSON response"):
        st.json(result)
