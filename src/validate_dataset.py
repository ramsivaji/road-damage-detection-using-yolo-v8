"""Validate prepared YOLO images and labels and report class statistics."""

from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CLASS_IDS, PREPARED_IMAGES_DIR, PREPARED_LABELS_DIR


def main() -> None:
    counts = Counter()
    image_count = 0
    invalid = []
    for split in ("train", "val", "test"):
        for image in (PREPARED_IMAGES_DIR / split).rglob("*"):
            if not image.is_file():
                continue
            image_count += 1
            label = PREPARED_LABELS_DIR / split / image.relative_to(PREPARED_IMAGES_DIR / split).with_suffix(".txt")
            if not label.exists():
                invalid.append(f"missing label: {image}")
                continue
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    invalid.append(f"invalid field count: {label}:{line_number}")
                    continue
                class_index = int(fields[0])
                values = [float(value) for value in fields[1:]]
                if class_index < 0 or class_index >= len(CLASS_IDS) or any(value < 0 or value > 1 for value in values):
                    invalid.append(f"invalid box: {label}:{line_number}")
                else:
                    counts[CLASS_IDS[class_index]] += 1
    print(f"Images: {image_count}")
    print(f"Objects by class: {dict(counts)}")
    if invalid:
        print(f"Invalid records: {len(invalid)}")
        raise RuntimeError("\n".join(invalid[:20]))
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
