# Rice Leaf Disease Detection — Demo Script

5-minute interview walkthrough. CPU inference is supported.

## Prerequisites

```bash
pip install -r requirements.txt
pip install py7zr
python scripts/download_and_split_dataset.py --replace
python src/train.py
pytest tests/ -q
```

See [data/DATA_SETUP.md](../data/DATA_SETUP.md). For a fast synthetic smoke test only: `python src/train.py --demo`.

If you already downloaded the Mendeley `.7z` manually, use `python scripts/download_and_split_dataset.py --from-archive path\to\file.7z --replace`.

## 1. Evaluation (1 min)

```bash
type reports\evaluation.md
```

Highlight per-class precision/recall and `visualizations/confusion_matrix.png`.

## 2. FastAPI (2 min)

```bash
uvicorn api.main:app --reload --port 8000
```

Upload a leaf image via Swagger UI at `http://127.0.0.1:8000/docs` or:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@path/to/leaf.png"
```

## 3. Streamlit + Grad-CAM (2 min)

```bash
streamlit run app/app.py
```

- Upload a leaf image or use sample asset
- Show diagnosis confidence on Tab 1
- Open Tab 3 for Grad-CAM heatmap overlay

## Checklist

- [ ] `models/ai_system_rice_leaf_final.keras` exists after training
- [ ] `pytest tests/ -q` passes
- [ ] API `/health` returns `model_loaded: true`
- [ ] Grad-CAM renders on XAI tab
