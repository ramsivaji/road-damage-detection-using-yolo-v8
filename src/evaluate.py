"""Evaluate the project-produced YOLO model on the held-out test split."""

from pathlib import Path
import argparse
import json

from ultralytics import YOLO

from config import BEST_MODEL_PATH, DATA_YAML_PATH, EVALUATION_OUTPUTS_DIR


def evaluate(device: str = "0") -> dict:
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Train the project model first: {BEST_MODEL_PATH}")
    model = YOLO(str(BEST_MODEL_PATH))
    metrics = model.val(
        data=str(DATA_YAML_PATH),
        split="test",
        device=device,
        project=str(EVALUATION_OUTPUTS_DIR),
        name="latest",
        exist_ok=True,
    )
    result = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    EVALUATION_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (EVALUATION_OUTPUTS_DIR / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    evaluate(parser.parse_args().device)
