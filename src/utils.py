"""Shared helpers for the RDD2022 preparation pipeline."""

from pathlib import Path
import shutil
from typing import Iterable

from config import (
    ANNOTATIONS_DIR,
    CLASS_IDS,
    CLASS_NAMES,
    DATASET_DIR,
    PREPARED_IMAGES_DIR,
    PREPARED_LABELS_DIR,
    RAW_ANNOTATIONS_DIR,
    RAW_IMAGES_DIR,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def reset_directory(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def find_image(image_name: str) -> Path | None:
    matches = [
        path
        for path in RAW_IMAGES_DIR.rglob("*")
        if path.is_file() and path.name.lower() == image_name.lower()
    ]
    return matches[0] if matches else None


def find_annotation(image_stem: str) -> Path | None:
    candidates = list(RAW_ANNOTATIONS_DIR.rglob(f"{image_stem}.xml"))
    if not candidates:
        candidates = list(ANNOTATIONS_DIR.rglob(f"{image_stem}.xml"))
    return candidates[0] if candidates else None


def write_data_yaml() -> Path:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    yaml_lines = [
        f"path: {DATASET_DIR.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_IDS)}",
        "names:",
    ]
    yaml_lines.extend(
        f"  {index}: {CLASS_NAMES[class_id]}"
        for index, class_id in enumerate(CLASS_IDS)
    )
    data_yaml = DATASET_DIR / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    return data_yaml


def iter_source_images() -> Iterable[Path]:
    return (
        path
        for path in RAW_IMAGES_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
