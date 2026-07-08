import os
import importlib.util

# Dynamically select backend based on environment capabilities
if importlib.util.find_spec("torch") is not None:
    os.environ["KERAS_BACKEND"] = "torch"
else:
    os.environ["KERAS_BACKEND"] = "tensorflow"
import streamlit as st
import sys
import time
import sqlite3
import json
import pandas as pd

# Ensure project root is in path for absolute importing
project_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, project_root)

import keras
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
import importlib
import src.interpretability
importlib.reload(src.interpretability)
from src.interpretability import make_gradcam_heatmap
from src.inference import CLASSES, DISPLAY_NAMES, IMG_SIZE, model_path, predict_image, preprocess_image

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Rice AI Diagnostics Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# ENTERPRISE DESIGN & CSS INJECTION
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Premium Header Banner */
    .premium-banner {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(42, 82, 152, 0.15);
        position: relative;
    }
    .premium-banner h1 {
        font-weight: 700;
        font-size: 2.4rem;
        margin-bottom: 0.5rem;
        color: #ffffff;
    }
    .premium-banner p {
        font-size: 1.15rem;
        opacity: 0.9;
        margin-bottom: 0;
        font-weight: 300;
    }
    
    /* Modern Card Layouts */
    .status-card {
        padding: 1.5rem;
        border-radius: 12px;
        background-color: #ffffff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        border: 1px solid #f1f5f9;
        margin-bottom: 1.5rem;
        transition: all 0.2s ease-in-out;
    }
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0,0,0,0.08);
    }
    
    .status-card.critical { border-left: 6px solid #ef4444; }
    .status-card.warning  { border-left: 6px solid #f59e0b; }
    .status-card.safe     { border-left: 6px solid #10b981; }
    .status-card.info     { border-left: 6px solid #3b82f6; }
    
    .status-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 0.3rem;
        font-weight: 600;
    }
    .status-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    
    /* Custom Navigation Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 1.05rem;
        font-weight: 600;
        color: #64748b;
        padding: 0 20px;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1e3c72;
        border-bottom: 3px solid #1e3c72;
        background-color: #f8fafc;
        border-radius: 6px 6px 0 0;
    }

    /* Remove the left sidebar completely per product UX */
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Render Global Banner
st.markdown("""
    <div class="premium-banner">
        <h1>🌾 Intelligent Rice Leaf Disease Diagnostics</h1>
        <p>Advanced Computer Vision & Grad-CAM Interpretability Pipeline</p>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# ZERO-DOWNTIME DATABASE LOGGING INFRASTRUCTURE
# ==========================================
DB_PATH = os.path.join(os.path.dirname(__file__), "../diagnostic_logs.db")

def init_db():
    """Initializes embedded local SQLite database. Requires ZERO external server downtime."""
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
    """Silently persists run metadata directly to local storage file."""
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
        # Prevent file lock or access permissions from halting real-time diagnostics
        pass

def fetch_run_history():
    """Retrieves localized SQLite execution telemetry logs."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("SELECT * FROM inference_runs ORDER BY timestamp DESC", conn)
        conn.close()
        return df
    except Exception:
        return pd.DataFrame()


def load_latest_test_accuracy():
    """Read latest exported test accuracy from reports/metrics.json."""
    metrics_path = os.path.join(os.path.dirname(__file__), "../reports/metrics.json")
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        score = payload.get("test_accuracy")
        if score is None:
            return "N/A"
        return f"{float(score) * 100:.1f}%"
    except Exception:
        return "N/A"


def load_latency_snapshot(df_logs: pd.DataFrame):
    """Return latest latency and p95 latency display strings."""
    if df_logs.empty or "latency_ms" not in df_logs.columns:
        return "N/A", "N/A"
    latencies = pd.to_numeric(df_logs["latency_ms"], errors="coerce").dropna()
    if latencies.empty:
        return "N/A", "N/A"
    latest_ms = float(latencies.iloc[0])
    p95_ms = float(latencies.quantile(0.95))
    return f"{latest_ms:.0f} ms", f"{p95_ms:.0f} ms"

# Initialize localized SQL infrastructure globally
init_db()

# ==========================================
# GLOBAL CONSTANTS & HELPERS
# ==========================================
classes = CLASSES
class_display_names = DISPLAY_NAMES

# Treatment Guides Storage mapped cleanly
treatment_protocols = {
    "Bacterialblight": """* Immediate Application: Spray appropriate copper-based bactericides (e.g., Copper Oxychloride) paired with low-dose Streptomycin formulations.
* Soil Regimes: Halt immediate Nitrogen top-dressing to suppress aggressive succulence. Apply balanced Potassium inputs to reinforce cell walls.
* Cultural Controls: Drain stagnating floodwaters temporarily to stop mobile waterborne spread across neighboring terraced rows.""",
    "Blast": """* Targeted Fungicides: Deploy immediate foliar sprays utilizing standard highly efficient penetrants such as Tricyclazole, Azoxystrobin, or Isoprothiolane.
* Irrigation Stabilization: Avoid chronic dry spells; blast spreads faster under water-stressed leaves. Restore reliable field flooding depth.
* Silicon Dressing: Incorporate soluble Silicon root enhancers to block direct fungal germ-tube penetration.""",
    "Brownspot": """* Nutrient Top-up: Conduct rapid soil testing. Top-up trace elements including Manganese, Zinc, and general complete NPK fertilizers.
* Chemical Prevention: Spray broad-spectrum protectants such as Mancozeb or Propiconazole if spot density scales past 15% leaf surface area.
* Seed Preparation: For subsequent planting runs, implement standard seed dressing protocol using Carbendazim.""",
    "Tungro": """* Vector Eradication: Stop the transmission engine immediately by targeting Green Leafhoppers using Imidacloprid, Thiamethoxam, or Buprofezin.
* Infected Disposal: Clear heavily stunted yellowish clumps to minimize source-inoculum reservoirs.
* Synchronous Planting: Enforce strict regional resting windows between crop plantings to break the local hopper lifecycle."""
}

def overlay_heatmap(image, heatmap, alpha=0.4):
    # Force RGB for robust blending (some image sources can be RGBA).
    img = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))

    # Grad-CAM helpers may return 2D, RGB, or RGBA arrays; reduce to a single map.
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

    # Safety guard in case an RGBA colormap slips through.
    if jet_heatmap.ndim == 3 and jet_heatmap.shape[2] == 4:
        jet_heatmap = jet_heatmap[:, :, :3]

    superimposed_img = jet_heatmap * alpha + img
    return superimposed_img.astype("uint8")

def compile_report_buffer(diag_class, conf_score, latency, protocol):
    """Compiles concise diagnostic summary report buffer."""
    content = f"""🌾 Diagnostic Report
Date: {time.strftime('%Y-%m-%d %H:%M:%S')}

Condition : {class_display_names[diag_class]}
Confidence: {conf_score*100:.1f}%
Latency   : {latency:.1f} ms

Recommended Actions:
{protocol}

---
Generated via AI Diagnostics Pipeline
"""
    return content.encode('utf-8')

# ==========================================
# ROBUST CACHED MODEL LOADER
# ==========================================
@st.cache_resource(show_spinner=False)
def load_rice_model():
    target_model = model_path(project_root)
    if not os.path.exists(target_model):
        raise FileNotFoundError(
            f"Missing source-of-truth model artifact: {target_model}. "
            "Regenerate from notebook/train pipeline before launching the app."
        )
    return keras.models.load_model(target_model, compile=False)

# ==========================================
# ENGINE THRESHOLDS (FIXED; NO SIDEBAR UI)
# ==========================================
conf_threshold = 0.70
heatmap_alpha = 0.45
input_mode = "Upload Local Leaf Image"
uploaded_file = None

# Load inference model securely
try:
    with st.spinner("Initializing Neural Network Parameter Graphs..."):
        model = load_rice_model()
except Exception as e:
    st.error(f"Critical System Fault: Unable to load model parameter graphs. Details: {e}")
    st.stop()

# Resolve source image object
active_image = None

history_snapshot = fetch_run_history()
latest_latency_text, latency_p95_text = load_latency_snapshot(history_snapshot)
latest_accuracy_text = load_latest_test_accuracy()
storage_driver_text = f"Embedded SQLite "

# ==========================================
# MAIN TABBED WORKFLOW ORCHESTRATION
# ==========================================
tab_diag, tab_treat, tab_xai, tab_history, tab_ref = st.tabs([
    "🔍 Real-time Diagnostics", 
    "💊 Actionable Treatments", 
    "🧠 AI Explainability (XAI)", 
    "📊 Database Telemetry",
    "📖 Reference Library"
])

# Initialize prediction placeholders safely
top_class = None
top_conf = 0.0
inf_time = 0.0

# ------------------------------------------
# TAB 1: REAL-TIME DIAGNOSTICS
# ------------------------------------------
with tab_diag:
    st.markdown("### 📥 Source Ingestion Mode")
    uploaded_file = st.file_uploader("Upload clear leaf capture", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        active_image = Image.open(uploaded_file)

    if active_image is None:
        st.info("👆 Please upload a crop image above to begin evaluation.")
        
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.markdown(
                f"<div class='status-card info'><div class='status-label'>Latest Latency (P95)</div><div class='status-value'>{latency_p95_text}</div></div>",
                unsafe_allow_html=True,
            )
        with col_p2:
            st.markdown(
                f"<div class='status-card safe'><div class='status-label'>Latest Test Accuracy</div><div class='status-value'>{latest_accuracy_text}</div></div>",
                unsafe_allow_html=True,
            )
        with col_p3:
            st.markdown(
                f"<div class='status-card warning'><div class='status-label'>Storage Driver</div><div class='status-value'>{storage_driver_text}</div></div>",
                unsafe_allow_html=True,
            )
            
    else:
        col_img, col_res = st.columns([1, 1.3], gap="large")
        
        with col_img:
            st.markdown("#### 🌿 Input Specimen")
            st.image(active_image, caption="Analyzed Field Specimen Capture", use_container_width=True)
            
        with col_res:
            st.markdown("#### 🔬 Diagnostic Engine Report")
            
            # Execute inference
            start_t = time.time()
            prediction = predict_image(model, active_image)
            preds = np.array([prediction["probabilities"][cls] for cls in classes], dtype=np.float32)
            inf_time = (time.time() - start_t) * 1000
            
            top_class = prediction["disease_class"]
            top_conf = float(prediction["confidence"])
            
            # Trigger Database Storage persistence silently
            log_inference_run(input_mode, top_class, top_conf, inf_time)
            
            severity_class = "critical" if top_class in ["Bacterialblight", "Blast", "Tungro"] else "warning"
            
            st.markdown(f"""
                <div class="status-card {severity_class}">
                    <div class="status-label">Diagnosed Condition</div>
                    <div class="status-value">{class_display_names[top_class]}</div>
                </div>
            """, unsafe_allow_html=True)
            
            m1, m2 = st.columns(2)
            with m1:
                conf_theme = "safe" if top_conf >= conf_threshold else "warning"
                st.markdown(f"""
                    <div class="status-card {conf_theme}">
                        <div class="status-label">Classification Confidence</div>
                        <div class="status-value">{top_conf*100:.1f}%</div>
                    </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                    <div class="status-card info">
                        <div class="status-label">Inference Overhead</div>
                        <div class="status-value">{inf_time:.0f} ms</div>
                    </div>
                """, unsafe_allow_html=True)
                
            if top_conf < conf_threshold:
                st.warning(f"⚠️ **Low Confidence Warning**: The diagnostic certainty falls below your configured threshold ({conf_threshold*100:.0f}%).")
                
            # Document Generation Action Trigger
            report_data = compile_report_buffer(top_class, top_conf, inf_time, treatment_protocols[top_class])
            st.download_button(
                label="📥 Generate & Download Printable Summary Report",
                data=report_data,
                file_name=f"Diagnostic_Report_{top_class}.txt",
                mime="text/plain",
            )
            st.caption("💡 Report generation operates entirely in-memory using buffer allocation streaming directly to your web browser.")

# ------------------------------------------
# TAB 2: ACTIONABLE TREATMENTS
# ------------------------------------------
with tab_treat:
    st.markdown("### 📋 Prescriptive Treatment Systems")
    if top_class is None:
        st.info("Run an active diagnosis on Tab 1 to dynamically unlock custom countermeasure mapping.")
    else:
        st.success(f"Tailoring protocols for active condition: **{class_display_names[top_class]}**")
        st.markdown(f"```text\n{treatment_protocols[top_class]}\n```")
        st.markdown("""
        #### 👨‍🌾 General Prevention Guidance
        * Regularly sanitize harvesting knives and thresher setups to suppress physical mechanical transmission.
        * Practice balanced NPK top-dressing schedules to keep natural crop protective barrier thickness highly active.
        """)

# ------------------------------------------
# TAB 3: AI EXPLAINABILITY (XAI)
# ------------------------------------------
with tab_xai:
    st.markdown("### 🧠 Spatial Feature Verification (Grad-CAM)")
    if top_class is None or active_image is None:
        st.info("Run an active diagnosis on Tab 1 to render real-time layer activation attention maps.")
    else:
        xc1, xc2 = st.columns([1.2, 1], gap="large")
        with xc1:
            with st.spinner("Resolving spatial activation vectors..."):
                try:
                    img_tensor = preprocess_image(active_image)
                    heatmap = make_gradcam_heatmap(img_tensor, model)
                    overlay = overlay_heatmap(active_image, heatmap, alpha=heatmap_alpha)
                    st.image(overlay, caption=f"Grad-CAM Attention Focus (Alpha: {heatmap_alpha})", use_container_width=True)
                except Exception as ge:
                    st.warning(f"Grad-CAM integration fallback triggered. Details: {ge}")
                    st.image(active_image, caption="Base Image Fallback", use_container_width=True)
                    
        with xc2:
            st.markdown("#### 📊 Probability Density Distribution")
            for cls_name, prob in zip(classes, preds):
                st.markdown(f"**{class_display_names[cls_name]}**")
                st.progress(float(prob), text=f"{prob*100:.1f}%")

# ------------------------------------------
# TAB 4: DATABASE TELEMETRY & LOGS
# ------------------------------------------
with tab_history:
    st.markdown("### 🗄️ Embedded Database Execution Logs")
    st.write("This log table updates automatically via zero-downtime **SQLite local persistence** stored securely in `diagnostic_logs.db`.")
    
    df_logs = fetch_run_history()
    
    if df_logs.empty:
        st.info("Database records are currently blank. Submit inference samples on Tab 1 to populate global execution tracking rows.")
    else:
        # Dynamic telemetry summaries
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Executed Submissions", len(df_logs))
        with col_s2:
            avg_lat = df_logs["latency_ms"].mean()
            st.metric("Average System Latency", f"{avg_lat:.1f} ms")
        with col_s3:
            top_pathogen = df_logs["diagnosed_class"].mode()[0]
            st.metric("Most Frequent Pathogen", class_display_names.get(top_pathogen, top_pathogen))
            
        st.divider()
        st.dataframe(df_logs, use_container_width=True, hide_index=True)
        
        st.caption("📌 Local storage files completely eliminate complex maintenance loops. Your data remains perfectly intact even upon terminating application servers.")

# ------------------------------------------
# TAB 5: REFERENCE LIBRARY
# ------------------------------------------
with tab_ref:
    st.markdown("### 🌾 Visual Pathogen Encyclopedia")
    r1, r2 = st.columns(2, gap="medium")
    with r1:
        st.markdown("""
        <div class="status-card critical">
            <h4>Bacterial Leaf Blight</h4>
            <p><b>Visual Signature:</b> Long, water-soaked yellowish streaks developing downwards from leaf tips and edges.</p>
        </div>
        <div class="status-card warning">
            <h4>Brown Spot</h4>
            <p><b>Visual Signature:</b> Abundant small, round to oval dark-brown lesions with gray centers.</p>
        </div>
        """, unsafe_allow_html=True)
    with r2:
        st.markdown("""
        <div class="status-card critical">
            <h4>Rice Blast</h4>
            <p><b>Visual Signature:</b> Diamond or spindle-shaped spots displaying prominent whitish-gray inner centers accompanied by sharp reddish-brown borders.</p>
        </div>
        <div class="status-card critical">
            <h4>Rice Tungro Disease</h4>
            <p><b>Visual Signature:</b> Pronounced stunting alongside distinct leaf discoloration transforming standard lush green tips into striking orange.</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("🚀 Institutional Rice Leaf Disease Diagnostic Pipeline | Engineered with Keras 3 & PyTorch Multi-Backend Engine")