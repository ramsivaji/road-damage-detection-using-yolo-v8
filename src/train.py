"""Train YOLOv8 on the prepared RDD2022 dataset."""

from pathlib import Path
import argparse
import shutil

from ultralytics import YOLO

from config import (
    BEST_MODEL_PATH,
    DATA_YAML_PATH,
    LAST_MODEL_PATH,
    MODEL_VERSIONS_DIR,
    TRAINING_OUTPUTS_DIR,
    YOLO_MODEL_CONFIG,
    YOLO_PRETRAINED_WEIGHTS,
)


def train(mode: str, epochs: int, batch: int, imgsz: int, lr0: float, device: str):
    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML_PATH}")
    if mode == "scratch":
        model = YOLO(YOLO_MODEL_CONFIG)
        pretrained = False
    elif mode == "transfer":
        model = YOLO(YOLO_PRETRAINED_WEIGHTS)
        pretrained = True
    else:
        raise ValueError("mode must be scratch or transfer")

    model.train(
        data=str(DATA_YAML_PATH),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        device=device,
        pretrained=pretrained,
        project=str(TRAINING_OUTPUTS_DIR),
        name=f"{mode}_training",
        exist_ok=True,
    )
    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    last = run_dir / "weights" / "last.pt"
    if not best.exists() or not last.exists():
        raise RuntimeError(f"Training completed without expected checkpoints in {run_dir}")
    BEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, BEST_MODEL_PATH)
    shutil.copy2(last, LAST_MODEL_PATH)
    shutil.copy2(best, MODEL_VERSIONS_DIR / f"best_{mode}.pt")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("scratch", "transfer"), default="scratch")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    train(args.mode, args.epochs, args.batch, args.imgsz, args.lr0, args.device)


if __name__ == "__main__":
    main()
