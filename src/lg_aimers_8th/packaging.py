from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import List


REQUIRED_MODEL_FILES_ANY = [
    "config.json",
    "tokenizer_config.json",
]


def list_zip_top_levels(zip_path: str | Path) -> List[str]:
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path, "r") as zf:
        top_levels = sorted(
            {
                name.split("/")[0]
                for name in zf.namelist()
                if name.strip() and not name.startswith("__MACOSX/")
            }
        )

    return top_levels


def validate_model_dir(model_dir: str | Path) -> None:
    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"model_dir does not exist: {model_dir}")

    if not model_dir.is_dir():
        raise NotADirectoryError(f"model_dir is not a directory: {model_dir}")

    missing = [
        filename
        for filename in REQUIRED_MODEL_FILES_ANY
        if not (model_dir / filename).exists()
    ]

    if missing:
        raise FileNotFoundError(f"Required model files missing: {missing}")

    has_weight = any(
        path.suffix in [".safetensors", ".bin"]
        for path in model_dir.rglob("*")
        if path.is_file()
    )

    if not has_weight:
        raise FileNotFoundError("No model weight file found. Expected .safetensors or .bin.")


def make_submit_zip(
    model_dir: str | Path,
    zip_path: str | Path,
    validate_model_files: bool = True,
) -> Path:
    model_dir = Path(model_dir)
    zip_path = Path(zip_path)

    if validate_model_files:
        validate_model_dir(model_dir)

    if zip_path.exists():
        zip_path.unlink()

    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(model_dir):
            for filename in files:
                full = Path(root) / filename
                rel = full.relative_to(model_dir)
                zf.write(full, arcname=str(Path("model") / rel))

    top_levels = list_zip_top_levels(zip_path)

    if top_levels != ["model"]:
        raise RuntimeError(f"Invalid submit.zip structure: {top_levels}. Expected only ['model'].")

    return zip_path


def zip_size_gb(zip_path: str | Path) -> float:
    return round(Path(zip_path).stat().st_size / 1024**3, 4)
