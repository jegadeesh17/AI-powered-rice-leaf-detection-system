"""Export evaluation metrics to reports/evaluation.md."""

from __future__ import annotations

import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
REPORT = os.path.join(ROOT, "reports", "evaluation.md")
METRICS = os.path.join(ROOT, "reports", "metrics.json")


def main(metrics: dict | None = None) -> None:
    if metrics is None:
        if not os.path.exists(METRICS):
            raise FileNotFoundError("Train models first: python src/train.py")
        with open(METRICS, encoding="utf-8") as f:
            metrics = json.load(f)

    lines = [
        "# Rice Leaf Disease Detection — Evaluation Report",
        "",
        f"## Test Accuracy: {metrics['test_accuracy']:.4f} ({metrics['test_accuracy']*100:.2f}%)",
        "",
        "## Per-Class Metrics",
        "| Class | Precision | Recall | F1 |",
        "|-------|-----------|--------|-----|",
    ]
    for name, scores in metrics["per_class"].items():
        lines.append(
            f"| {name} | {scores['precision']} | {scores['recall']} | {scores['f1']} |"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "- Confusion matrix: `visualizations/confusion_matrix.png`",
            f"- Model: `models/ai_system_rice_leaf_final.keras`",
            "",
            "## Notes",
            "- EfficientNetB0 transfer learning with frozen-then-fine-tuned backbone.",
            "- CPU inference supported; use batch size ≤16 on 4GB VRAM during training.",
            "- Regenerate: `python src/train.py`",
        ]
    )
    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
