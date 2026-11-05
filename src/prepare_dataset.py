"""Validate pairs, split images 70/20/10, and create data.yaml."""

from pathlib import Path
import argparse
import random
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    ANNOTATIONS_DIR,
    PREPARED_IMAGES_DIR,
    PREPARED_LABELS_DIR,
    RAW_IMAGES_DIR,
    TEST_SPLIT,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from config import ensure_project_directories
from src.utils import iter_source_images, write_data_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    ensure_project_directories()
    pairs = []
    for image in iter_source_images():
        relative = image.relative_to(RAW_IMAGES_DIR)
        label = ANNOTATIONS_DIR / relative.with_suffix(".txt")
        if label.exists():
            pairs.append((image, label, relative))
    if not pairs:
        raise RuntimeError("No image-label pairs found. Run download and conversion first.")

    random.Random(args.seed).shuffle(pairs)
    train_end = int(len(pairs) * TRAIN_SPLIT)
    val_end = train_end + int(len(pairs) * VAL_SPLIT)
    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:],
    }
    for split, split_pairs in splits.items():
        for image, label, relative in split_pairs:
            image_target = PREPARED_IMAGES_DIR / split / relative
            label_target = PREPARED_LABELS_DIR / split / relative.with_suffix(".txt")
            image_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_target)
            shutil.copy2(label, label_target)
    data_yaml = write_data_yaml()
    print({split: len(items) for split, items in splits.items()})
    print(f"Generated {data_yaml}")


if __name__ == "__main__":
    main()
