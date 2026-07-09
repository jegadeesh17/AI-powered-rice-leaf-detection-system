#!/usr/bin/env python3
"""Upload the trained Keras model to Hugging Face Hub (one-time setup)."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from huggingface_hub import HfApi, create_repo

from src.inference import MODEL_FILENAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload rice leaf model to Hugging Face Hub")
    parser.add_argument("--repo-id", required=True, help="e.g. your-username/rice-leaf-disease-model")
    parser.add_argument("--private", action="store_true", help="Create a private model repo")
    args = parser.parse_args()

    source = os.path.join(ROOT, "models", MODEL_FILENAME)
    if not os.path.exists(source):
        raise SystemExit(f"Model not found at {source}. Run: python src/train.py")

    api = HfApi()
    create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=source,
        path_in_repo=MODEL_FILENAME,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded {MODEL_FILENAME} to https://huggingface.co/{args.repo_id}")
    print(f"Set HF_MODEL_REPO={args.repo_id} in Cloud Run / Streamlit secrets.")


if __name__ == "__main__":
    main()
