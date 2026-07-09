"""Tests for Hugging Face model resolution helpers."""

import os
import sys
from unittest.mock import patch

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def test_ensure_model_file_returns_existing_path(tmp_path):
    from src.model_assets import ensure_model_file

    model_file = tmp_path / "ai_system_rice_leaf_final.keras"
    model_file.write_bytes(b"fake-model")
    result = ensure_model_file(str(model_file))
    assert result == str(model_file)


def test_ensure_model_file_skips_download_without_repo(tmp_path):
    from src.model_assets import ensure_model_file

    missing = tmp_path / "ai_system_rice_leaf_final.keras"
    with patch.dict(os.environ, {}, clear=True):
        result = ensure_model_file(str(missing))
    assert result == str(missing)
    assert not missing.exists()


def test_ensure_model_file_downloads_when_repo_configured(tmp_path):
    from src.model_assets import ensure_model_file

    missing = tmp_path / "ai_system_rice_leaf_final.keras"
    with patch.dict(os.environ, {"HF_MODEL_REPO": "demo/rice-model"}, clear=False):
        with patch("huggingface_hub.hf_hub_download", return_value=str(missing)) as mocked:
            result = ensure_model_file(str(missing))
    assert result == str(missing)
    mocked.assert_called_once()
