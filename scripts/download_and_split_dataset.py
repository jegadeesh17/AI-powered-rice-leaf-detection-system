"""Download the Mendeley rice-leaf dataset and create train/val/test splits.

Source: https://data.mendeley.com/datasets/fwcj7stb8r
DOI: 10.17632/fwcj7stb8r.1 (archive file is Rice Leaf Disease Images.7z)

Usage:
  pip install py7zr
  python scripts/download_and_split_dataset.py
  python scripts/download_and_split_dataset.py --replace   # overwrite existing split
  python scripts/download_and_split_dataset.py --from-archive path/to/file.7z
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
EXTRACT_DIR = RAW_DIR / "mendeley_extract"
ARCHIVE_PATH = RAW_DIR / "Rice_Leaf_Disease_Images.7z"
SPLIT_ROOT = ROOT / "data" / "processed" / "rice_leaf_split"
MANIFEST_PATH = ROOT / "data" / "processed" / "split_manifest.json"

MENDELEY_DATASET_API = "https://data.mendeley.com/public-api/datasets/fwcj7stb8r"
FALLBACK_DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/fwcj7stb8r/"
    "files/fd8717c4-0d65-4c80-a76c-3b61cb68e80d/file_downloaded"
)

CLASSES = ("Bacterialblight", "Blast", "Brownspot", "Tungro")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Map messy folder / label names onto the project's canonical class folders.
CLASS_ALIASES = {
    "bacterialblight": "Bacterialblight",
    "bacterial blight": "Bacterialblight",
    "bacterial_leaf_blight": "Bacterialblight",
    "bacterialleafblight": "Bacterialblight",
    "blast": "Blast",
    "leaf blast": "Blast",
    "rice blast": "Blast",
    "brownspot": "Brownspot",
    "brown spot": "Brownspot",
    "brown_spot": "Brownspot",
    "tungro": "Tungro",
    "rice tungro": "Tungro",
}


def _normalize_class(name: str) -> str | None:
    key = name.strip().lower().replace("-", " ").replace("_", " ")
    key = " ".join(key.split())
    compact = key.replace(" ", "")
    return CLASS_ALIASES.get(key) or CLASS_ALIASES.get(compact)


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


def _open_url(url: str, timeout: int = 120):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "*/*",
            "Referer": "https://data.mendeley.com/datasets/fwcj7stb8r/1",
        },
    )
    return urllib.request.urlopen(req, timeout=timeout)


def resolve_download_url() -> str:
    try:
        with _open_url(MENDELEY_DATASET_API, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        files = payload.get("files") or []
        for item in files:
            details = item.get("content_details") or {}
            url = details.get("download_url")
            if url and str(item.get("filename", "")).lower().endswith(".7z"):
                print(f"Resolved download URL for: {item.get('filename')}")
                return url
        for item in files:
            details = item.get("content_details") or {}
            if details.get("download_url"):
                return details["download_url"]
    except Exception as exc:  # noqa: BLE001 — network metadata is best-effort
        print(f"Could not resolve API download URL ({exc}); using fallback.")
    return FALLBACK_DOWNLOAD_URL


def download_archive(dest: Path, force: bool = False) -> Path:
    if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
        print(f"Archive already present: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    url = resolve_download_url()
    print(f"Downloading Mendeley archive (~179 MB)...\n  {url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")

    with _open_url(url, timeout=600) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        chunk = 1024 * 256
        with open(tmp, "wb") as out:
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                out.write(block)
                done += len(block)
                if total > 0:
                    pct = 100.0 * done / total
                    print(f"\r  {pct:5.1f}% ({done / 1e6:.1f}/{total / 1e6:.1f} MB)", end="", flush=True)
                elif done % (10 * 1024 * 1024) < chunk:
                    print(f"\r  Downloaded {done / 1e6:.1f} MB", end="", flush=True)
    print()
    if done < 1_000_000:
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            "Download failed or returned a tiny file (likely blocked).\n"
            "Manual fallback:\n"
            "  1. Open https://data.mendeley.com/datasets/fwcj7stb8r/1\n"
            "  2. Download 'Rice Leaf Disease Images.7z'\n"
            f"  3. python scripts/download_and_split_dataset.py --from-archive <path-to-7z> --replace"
        )
    tmp.replace(dest)
    print(f"Saved archive to {dest}")
    return dest


def extract_archive(archive: Path, out_dir: Path, force: bool = False) -> Path:
    if out_dir.exists() and any(out_dir.rglob("*")) and not force:
        print(f"Extract directory already populated: {out_dir}")
        return out_dir

    if out_dir.exists() and force:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        import py7zr
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: py7zr\n"
            "Install with:  pip install py7zr\n"
            "Then re-run this script."
        ) from exc

    print(f"Extracting {archive.name} -> {out_dir}")
    with py7zr.SevenZipFile(archive, mode="r") as zf:
        zf.extractall(path=out_dir)
    print("Extraction complete.")
    return out_dir


def discover_class_images(extract_root: Path) -> dict[str, list[Path]]:
    buckets: dict[str, list[Path]] = {c: [] for c in CLASSES}

    for path in extract_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        # Prefer nearest parent directory name as the class label.
        label = None
        for parent in path.parents:
            if parent == extract_root:
                break
            mapped = _normalize_class(parent.name)
            if mapped:
                label = mapped
                break
        if label is None:
            mapped = _normalize_class(path.stem)
            if mapped:
                label = mapped
        if label is None:
            continue
        buckets[label].append(path)

    return buckets


def stratified_split(
    paths: list[Path],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    ordered = list(paths)
    rng.shuffle(ordered)
    n = len(ordered)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Remainder goes to test so counts always sum to n.
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def _looks_like_demo_split(split_root: Path) -> bool:
    """Heuristic: seed_demo_data.py names files Class_split_N.png and keeps tiny counts."""
    sample = list(split_root.rglob("*_train_0.png")) + list(split_root.rglob("*_val_0.png"))
    if sample:
        return True
    total = sum(1 for p in split_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return 0 < total < 500


def archive_existing_split(split_root: Path) -> Path | None:
    if not split_root.exists() or not any(split_root.iterdir()):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    trash = ROOT / ".trash" / stamp / "rice_leaf_split"
    trash.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(split_root), str(trash))
    print(f"Moved previous split -> {trash}")
    return trash


def write_splits(
    class_images: dict[str, list[Path]],
    split_root: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
    replace: bool,
) -> dict:
    missing = [c for c, imgs in class_images.items() if not imgs]
    if missing:
        raise SystemExit(
            "Could not find images for class(es): "
            + ", ".join(missing)
            + f"\nInspect extract folder: {EXTRACT_DIR}"
        )

    totals = {c: len(v) for c, v in class_images.items()}
    total_all = sum(totals.values())
    print("Discovered images:")
    for c in CLASSES:
        print(f"  {c}: {totals[c]}")
    print(f"  TOTAL: {total_all}")

    if total_all < 1000:
        print(
            "WARNING: count is far below the published ~5932 images. "
            "Extraction or class discovery may be incomplete."
        )

    if split_root.exists() and any(split_root.iterdir()):
        if not replace:
            raise SystemExit(
                f"Split already exists at {split_root}\n"
                "Re-run with --replace to archive the old folder and rebuild."
            )
        if _looks_like_demo_split(split_root):
            print("Detected demo/fake seed images in the current split; archiving them.")
        archive_existing_split(split_root)

    split_root.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = defaultdict(dict)

    for class_name, paths in class_images.items():
        parts = stratified_split(paths, train_ratio, val_ratio, seed + hash(class_name) % 10_000)
        for split_name, split_paths in parts.items():
            dest_dir = split_root / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                dest = dest_dir / src.name
                # Avoid collisions if duplicate filenames appear across nested folders.
                if dest.exists():
                    dest = dest_dir / f"{src.stem}_{abs(hash(str(src))) % 10_000_000}{src.suffix}"
                shutil.copy2(src, dest)
            counts[split_name][class_name] = len(split_paths)

    manifest = {
        "source": "Mendeley Data — doi:10.17632/fwcj7stb8r",
        "created": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": round(1 - train_ratio - val_ratio, 4)},
        "counts": {split: dict(per_class) for split, per_class in counts.items()},
        "totals_by_class": totals,
        "total_images": total_all,
        "split_root": str(split_root),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nWrote stratified split:")
    for split_name in ("train", "val", "test"):
        split_total = sum(counts[split_name].values())
        print(f"  {split_name}: {split_total}")
    print(f"Manifest: {MANIFEST_PATH}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download + split Mendeley rice leaf dataset")
    parser.add_argument("--from-archive", type=Path, help="Skip download; use an existing .7z/.zip")
    parser.add_argument("--skip-download", action="store_true", help="Reuse data/raw archive if present")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if archive exists")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract even if folder exists")
    parser.add_argument("--replace", action="store_true", help="Archive existing split and rebuild")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise SystemExit("train_ratio + val_ratio must be < 1.0 (remainder is test)")

    if args.from_archive:
        archive = args.from_archive.expanduser().resolve()
        if not archive.exists():
            raise SystemExit(f"Archive not found: {archive}")
    else:
        archive = download_archive(ARCHIVE_PATH, force=args.force_download) if not args.skip_download else ARCHIVE_PATH
        if not archive.exists():
            raise SystemExit(f"No archive at {archive}. Drop the .7z there or omit --skip-download.")

    extract_archive(archive, EXTRACT_DIR, force=args.force_extract)
    class_images = discover_class_images(EXTRACT_DIR)
    write_splits(
        class_images,
        SPLIT_ROOT,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        replace=args.replace,
    )
    print("\nNext: python src/train.py")


if __name__ == "__main__":
    main()
