# Rice Leaf Disease Detection — Evaluation Report

## Test Accuracy: 0.9866 (98.66%)

## Per-Class Metrics
| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Bacterialblight | 0.9793388429752066 | 0.9916317991631799 | 0.9854469854469855 |
| Blast | 0.9904306220095693 | 0.9539170506912442 | 0.971830985915493 |
| Brownspot | 0.9836065573770492 | 1.0 | 0.9917355371900827 |
| Tungro | 0.9949494949494949 | 1.0 | 0.9974683544303797 |

## Artifacts
- Confusion matrix: `visualizations/confusion_matrix.png`
- Model: `models/ai_system_rice_leaf_final.keras`

## Notes
- EfficientNetB0 transfer learning with frozen-then-fine-tuned backbone.
- CPU inference supported; use batch size ≤16 on 4GB VRAM during training.
- Regenerate: `python src/train.py`