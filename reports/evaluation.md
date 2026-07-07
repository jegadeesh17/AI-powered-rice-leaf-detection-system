# Rice Leaf Disease Detection — Evaluation Report

## Test Accuracy: 1.0

## Per-Class Metrics
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Bacterialblight | 1.0 | 1.0 | 1.0 |
| Blast | 1.0 | 1.0 | 1.0 |
| Brownspot | 1.0 | 1.0 | 1.0 |
| Tungro | 1.0 | 1.0 | 1.0 |

## Artifacts
- Confusion matrix: `docs/confusion_matrix.png`
- Model: `models/ai_system_rice_leaf_final.keras`

## Notes
- EfficientNetB0 transfer learning with frozen-then-fine-tuned backbone.
- CPU inference supported; use batch size ≤16 on 4GB VRAM during training.
- Regenerate: `python src/train.py`