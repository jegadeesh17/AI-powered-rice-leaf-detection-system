import json

import pytest

from scripts import export_evaluation as exporter


def _sample_metrics():
    return {
        "test_accuracy": 0.95,
        "per_class": {
            "Blast": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
            "Brownspot": {"precision": 0.8, "recall": 0.9, "f1": 0.85},
        },
    }


def test_main_raises_when_metrics_file_missing(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "evaluation.md"
    metrics_path = tmp_path / "reports" / "metrics.json"
    monkeypatch.setattr(exporter, "REPORT", str(report_path))
    monkeypatch.setattr(exporter, "METRICS", str(metrics_path))

    with pytest.raises(FileNotFoundError):
        exporter.main()


def test_main_writes_report_from_provided_metrics(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "evaluation.md"
    monkeypatch.setattr(exporter, "REPORT", str(report_path))

    exporter.main(metrics=_sample_metrics())

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Rice Leaf Disease Detection" in content
    assert "Test Accuracy: 0.9500 (95.00%)" in content
    assert "| Blast | 0.9 | 0.8 | 0.85 |" in content


def test_main_reads_metrics_from_file(tmp_path, monkeypatch):
    report_path = tmp_path / "reports" / "evaluation.md"
    metrics_path = tmp_path / "reports" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(_sample_metrics()), encoding="utf-8")

    monkeypatch.setattr(exporter, "REPORT", str(report_path))
    monkeypatch.setattr(exporter, "METRICS", str(metrics_path))

    exporter.main()

    content = report_path.read_text(encoding="utf-8")
    assert "Per-Class Metrics" in content
    assert "Confusion matrix: `visualizations/confusion_matrix.png`" in content
