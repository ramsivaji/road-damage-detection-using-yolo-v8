"""Download and extract RDD2022 source data."""

from pathlib import Path
import argparse
import shutil
import sys
import zipfile

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import DOWNLOADS_DIR, RDD2022_DOWNLOAD_URL, RAW_ANNOTATIONS_DIR, RAW_IMAGES_DIR
from config import ensure_project_directories


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with partial.open("wb") as output:
            with tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=destination.name,
            ) as progress:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        progress.update(len(chunk))
    partial.replace(destination)


def extract_archive(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=RDD2022_DOWNLOAD_URL)
    parser.add_argument("--country", default="India", help="Country folder to copy, or all")
    args = parser.parse_args()

    ensure_project_directories()
    archive = DOWNLOADS_DIR / "RDD2022.zip"
    extracted = DOWNLOADS_DIR / "extracted"
    if not archive.exists() or not zipfile.is_zipfile(archive):
        if archive.exists():
            archive.unlink()
        print(f"Downloading RDD2022 from {args.url}")
        download_file(args.url, archive)
    else:
        print(f"Using existing archive: {archive}")

    if not extracted.exists() or not any(extracted.iterdir()):
        print(f"Extracting {archive}")
        extracted.mkdir(parents=True, exist_ok=True)
        extract_archive(archive, extracted)

    image_count = 0
    annotation_count = 0
    for source in extracted.rglob("*"):
        if not source.is_file():
            continue
        suffix = source.suffix.lower()
        country_match = args.country.lower() == "all" or any(
            args.country.lower() in part.lower() for part in source.parts
        )
        if not country_match:
            continue
        if suffix in {".jpg", ".jpeg", ".png", ".bmp"}:
            target = RAW_IMAGES_DIR / source.relative_to(extracted)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            image_count += 1
        elif suffix == ".xml":
            target = RAW_ANNOTATIONS_DIR / source.relative_to(extracted)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            annotation_count += 1
    print(f"Copied {image_count} images and {annotation_count} XML annotations.")


if __name__ == "__main__":
    main()
