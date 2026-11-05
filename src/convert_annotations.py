"""Convert RDD2022 Pascal VOC XML annotations to YOLO label files."""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET

from config import ANNOTATIONS_DIR, CLASS_IDS
from config import ensure_project_directories
from utils import find_annotation, find_image


def convert_box(left: float, top: float, right: float, bottom: float, width: float, height: float) -> tuple[float, float, float, float]:
    box_width = max(0.0, right - left)
    box_height = max(0.0, bottom - top)
    return (
        ((left + right) / 2) / width,
        ((top + bottom) / 2) / height,
        box_width / width,
        box_height / height,
    )


def convert_annotation(xml_path: Path, output_path: Path) -> int:
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size: {xml_path}")
    width = float(size.findtext("width", "0"))
    height = float(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {xml_path}")

    rows = []
    for object_node in root.findall("object"):
        class_id = object_node.findtext("name", "").strip()
        if class_id not in CLASS_IDS:
            continue
        box = object_node.find("bndbox")
        if box is None:
            continue
        left = float(box.findtext("xmin", "0"))
        top = float(box.findtext("ymin", "0"))
        right = float(box.findtext("xmax", "0"))
        bottom = float(box.findtext("ymax", "0"))
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            continue
        values = convert_box(left, top, right, bottom, width, height)
        rows.append(f"{CLASS_IDS.index(class_id)} " + " ".join(f"{value:.6f}" for value in values))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    ensure_project_directories()
    xml_files = list(Path("dataset/raw/annotations").rglob("*.xml"))
    converted = objects = 0
    for xml_path in xml_files[: args.limit or None]:
        image = find_image(xml_path.stem + ".jpg")
        if image is None:
            image = find_image(xml_path.stem + ".png")
        if image is None:
            continue
        relative = image.relative_to(Path("dataset/raw/images"))
        output = ANNOTATIONS_DIR / relative.with_suffix(".txt")
        objects += convert_annotation(xml_path, output)
        converted += 1
    print(f"Converted {converted} annotations containing {objects} objects.")


if __name__ == "__main__":
    main()
