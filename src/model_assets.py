"""Resolve model artifacts from local disk or Hugging Face Hub."""

from __future__ import annotations

import os
from pathlib import Path

def hf_repo_id() -> str | None:
    return os.getenv("HF_MODEL_REPO")


def ensure_model_file(local_path: str) -> str:
    """Return path to the model file, downloading from HF Hub when configured."""
    if os.path.exists(local_path):
        return local_path

    repo = hf_repo_id()
    if not repo:
        return local_path

    from huggingface_hub import hf_hub_download

    target_dir = Path(local_path).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    return hf_hub_download(
        repo_id=repo,
        filename=Path(local_path).name,
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,
    )
