# Rice Leaf Disease Detection — Technical Specification

---

## Document Control

| Field | Value |
|-------|-------|
| **Document** | PROJECT_SPEC.md |
| **Version** | 1.0 |
| **Status** | Active |
| **Last updated** | 2026-07-08 |
| **Repository** | [github.com/jegadeesh17/AI-powered-rice-leaf-detection-system](https://github.com/jegadeesh17/AI-powered-rice-leaf-detection-system) |
| **Related docs** | [README.md](../README.md), [DEMO.md](./DEMO.md), [reports/evaluation.md](../reports/evaluation.md) |

---

## 1. Executive Summary

Rice Leaf Disease Detection is a **computer vision classification system** for precision agriculture. It identifies four rice leaf diseases using **EfficientNetB0 transfer learning**, serves predictions via **FastAPI** image upload, and provides **Grad-CAM explainability** in Streamlit so users see which leaf regions drove the diagnosis.

**Interview pitch:**

> *"I built a 4-class rice disease classifier with EfficientNetB0, one-command training, Grad-CAM explainability, FastAPI inference, and pytest coverage — optimized for 4GB GPU laptops with CPU fallback."*

---

## 2. Scope

### 2.1 In Scope

| # | Capability |
|---|------------|
| 1 | Image preprocessing and augmentation |
| 2 | EfficientNetB0 transfer learning (freeze → fine-tune) |
| 3 | One-command training script with `--demo` mode |
| 4 | Confusion matrix and evaluation report export |
| 5 | Grad-CAM heatmap generation |
| 6 | FastAPI `POST /predict` image endpoint |
| 7 | Streamlit diagnostics UI |
| 8 | pytest API tests |

### 2.2 Out of Scope

- Mobile on-device deployment (TFLite)
- Multi-crop generalization beyond rice
- Real-time video/stream classification
- Field-condition robustness validation at scale

---

## 3. Requirements

### 3.1 Functional Requirements

| ID | Requirement | Module | Status |
|----|-------------|--------|--------|
| FR-01 | Load and split image dataset | `src/train.py` | ✅ |
| FR-02 | Train EfficientNetB0 classifier | `src/train.py` | ✅ |
| FR-03 | Save Keras model artifact | `models/` | ✅ |
| FR-04 | Image inference | `src/inference.py` | ✅ |
| FR-05 | Grad-CAM visualization | `src/interpretability.py` | ✅ |
| FR-06 | REST image API | `api/main.py` | ✅ |
| FR-07 | Streamlit upload UI | `app/app.py` | ✅ |

### 3.2 Non-Functional Requirements

| ID | Requirement | Target |
|----|-------------|--------|
| NFR-01 | Train on 4GB VRAM | batch size ≤16 |
| NFR-02 | CPU inference supported | No GPU required at serve time |
| NFR-03 | Demo mode without full dataset | `--demo` flag |
| NFR-04 | Health check endpoint | `GET /health` |

---

## 4. Architecture

```text
Mendeley rice leaf images
        │
        ▼
train.py (augmentation, EfficientNetB0)
        │
        ▼
models/ai_system_rice_leaf_final.keras
        │
        ├── inference.py ──▶ api/main.py (POST /predict)
        └── interpretability.py ──▶ app/app.py (heatmap overlay)
```

---

## 5. Data Specification

| Field | Detail |
|-------|--------|
| Source | Mendeley Data — doi: 10.17632/fwcj7stb8r.1 |
| Layout | `data/processed/rice_leaf_split/{train,val,test}/{class}/` |
| Classes | Bacterialblight, Blast, Brownspot, Tungro |
| Demo data | Minimal seed subset for `--demo` training |

---

## 6. Models & Metrics

| Metric | Current run | Notes |
|--------|----------|-------|
| Test accuracy | 0.9866 (98.66%) | Final model + full test split |
| Per-class F1 | 0.9718–0.9975 | Strongest: Tungro, Brownspot |
| Artifact | `visualizations/confusion_matrix.png` | Generated during evaluation |

Regenerate: `python src/train.py` → `reports/evaluation.md`

---

## 7. API Specification

### `GET /health`

Returns model load status.

### `POST /predict`

**Input:** Multipart image file (`UploadFile`).  
**Output:**
```json
{
  "disease_class": "Blast",
  "confidence": 0.94,
  "probabilities": { ... }
}
```

---

## 8. Grad-CAM Explainability

`src/interpretability.py` computes gradient-weighted class activation maps over the final conv layer. Streamlit overlays the heatmap on the uploaded leaf image for analyst trust.

---

## 9. Deployment

```powershell
pip install -r requirements.txt
pip install -r requirements-api.txt
python src/train.py --demo
pytest tests/ -q
uvicorn api.main:app --port 8000
streamlit run app/app.py
```

**Production training:** `pip install py7zr` → `python scripts/download_and_split_dataset.py --replace` → `python src/train.py`  
See [data/DATA_SETUP.md](../data/DATA_SETUP.md).

---

## 10. Testing

`tests/test_api.py` — health check and mocked/stubbed prediction path.

---

## 11. Module Index

| Path | Purpose |
|------|---------|
| `src/train.py` | Training CLI with `--demo` |
| `src/inference.py` | Load model + predict |
| `src/interpretability.py` | Explainability heatmaps |
| `api/main.py` | FastAPI service |
| `app/app.py` | Streamlit diagnostics |
| `notebooks/AI system for rice leaf.ipynb` | Notebook source of truth |

---

## 12. Future Improvements

- TFLite mobile export for field use
- Test-time augmentation for robustness
- Agronomist-validated holdout set from field images
- Multi-GPU training script for full dataset
