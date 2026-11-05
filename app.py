"""Streamlit UI for the complete road-damage workflow."""

import json
import subprocess
import sys
from pathlib import Path

import streamlit as st

from config import (
    BEST_MODEL_PATH,
    CLASS_IDS,
    CLASS_NAMES,
    DATASET_DIR,
    DATA_YAML_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_LEARNING_RATE,
    ensure_project_directories,
)


ROOT = Path(__file__).resolve().parent
ensure_project_directories()
st.set_page_config(page_title="Road Damage Detection", page_icon="🛣️", layout="wide")


def run_script(script: str, *args: str) -> str:
    command = [sys.executable, str(ROOT / "src" / script), *args]
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout)
    return completed.stdout


def dataset_counts() -> dict[str, int]:
    return {
        split: sum(
            1
            for path in (DATASET_DIR / "images" / split).rglob("*")
            if path.is_file()
        )
        for split in ("train", "val", "test")
    }


page = st.sidebar.radio(
    "Pages",
    [
        "Home Dashboard",
        "Download Dataset",
        "Prepare Dataset",
        "Train Model",
        "Evaluate Model",
        "Live Camera Detection",
        "Image Detection",
        "Video Detection",
        "About Project",
    ],
)

if page == "Home Dashboard":
    st.title("Automated Road Damage Detection")
    counts = dataset_counts()
    total = sum(counts.values())
    columns = st.columns(5)
    for column, label, value in zip(
        columns,
        ("Dataset images", "Training", "Validation", "Testing", "Model"),
        (total, counts["train"], counts["val"], counts["test"], "Ready" if BEST_MODEL_PATH.exists() else "Not trained"),
    ):
        column.metric(label, value)
    st.subheader("Classes")
    st.dataframe(
        {"ID": CLASS_IDS, "Name": [CLASS_NAMES[item] for item in CLASS_IDS]},
        hide_index=True,
        use_container_width=True,
    )
    metrics_path = Path("outputs/evaluation/metrics.json")
    if metrics_path.exists():
        st.subheader("Latest evaluation")
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
    else:
        st.info("Evaluation metrics will appear after the project model is evaluated.")

elif page == "Download Dataset":
    st.title("Download RDD2022")
    st.write("Downloads the official RDD2022 archive, extracts images/XML, and prioritizes India in later preparation.")
    if st.button("Download and extract RDD2022", type="primary"):
        with st.spinner("Downloading and extracting the official archive..."):
            try:
                output = run_script("download_dataset.py")
                st.success("Dataset download and extraction completed.")
                st.code(output)
            except Exception as error:
                st.error(str(error))

elif page == "Prepare Dataset":
    st.title("Prepare and validate dataset")
    if st.button("Convert XML and prepare 70:20:10 splits", type="primary"):
        with st.spinner("Converting annotations and preparing splits..."):
            try:
                st.code(run_script("convert_annotations.py"))
                st.code(run_script("prepare_dataset.py"))
                st.code(run_script("validate_dataset.py"))
                st.success(f"Prepared dataset at {DATASET_DIR}")
            except Exception as error:
                st.error(str(error))
    st.write(dataset_counts())
    if DATA_YAML_PATH.exists():
        st.code(DATA_YAML_PATH.read_text(encoding="utf-8"), language="yaml")

elif page == "Train Model":
    st.title("Train YOLOv8 model")
    mode = st.selectbox("Training mode", ["scratch", "transfer"])
    epochs = st.number_input("Epochs", min_value=1, value=DEFAULT_EPOCHS)
    batch = st.number_input("Batch size", min_value=1, value=DEFAULT_BATCH_SIZE)
    imgsz = st.number_input("Image size", min_value=32, value=DEFAULT_IMAGE_SIZE, step=32)
    lr0 = st.number_input("Learning rate", min_value=0.000001, value=DEFAULT_LEARNING_RATE)
    device = st.selectbox("Device", ["0", "cpu"])
    if st.button("Start training", type="primary"):
        with st.spinner("Training is running. Do not close this page..."):
            try:
                st.code(
                    run_script(
                        "train.py", "--mode", mode, "--epochs", str(epochs),
                        "--batch", str(batch), "--imgsz", str(imgsz),
                        "--lr0", str(lr0), "--device", device,
                    )
                )
                st.success(f"Training completed. Best model: {BEST_MODEL_PATH}")
            except Exception as error:
                st.error(str(error))

elif page == "Evaluate Model":
    st.title("Evaluate model")
    device = st.selectbox("Evaluation device", ["0", "cpu"])
    if st.button("Evaluate on test split", type="primary"):
        try:
            st.code(run_script("evaluate.py", "--device", device))
            st.success("Evaluation completed.")
        except Exception as error:
            st.error(str(error))
    metrics_path = Path("outputs/evaluation/metrics.json")
    if metrics_path.exists():
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")))

elif page == "Live Camera Detection":
    st.title("Live camera detection")
    st.warning("This page uses OpenCV camera index 0 and requires a project-trained models/best.pt.")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25)
    frames = st.number_input("Frames per run", min_value=1, value=300)
    if st.button("Start Camera", type="primary"):
        try:
            from src.live_detection import run_camera
            placeholder = st.empty()
            with st.spinner("Camera running..."):
                for frame, result, fps in run_camera(0, confidence, int(frames)):
                    placeholder.image(frame, channels="BGR", caption=f"FPS: {fps:.1f}")
            st.success("Camera session ended.")
        except Exception as error:
            st.error(str(error))

elif page == "Image Detection":
    st.title("Image detection")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25)
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded:
        from io import BytesIO
        from PIL import Image
        from src.image_detection import detect_image
        image = Image.open(uploaded).convert("RGB")
        try:
            annotated, result = detect_image(image, confidence)
            original, detected = st.columns(2)
            original.image(image, caption="Original")
            detected.image(annotated, caption="Detected")
            buffer = BytesIO()
            annotated.save(buffer, format="PNG")
            st.download_button("Download detected image", buffer.getvalue(), "detected.png", "image/png")
            st.write(f"Detected objects: {len(result.boxes)}")
        except Exception as error:
            st.error(str(error))

elif page == "Video Detection":
    st.title("Video detection")
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25)
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded and st.button("Process video", type="primary"):
        temporary = ROOT / "outputs" / uploaded.name
        temporary.write_bytes(uploaded.getvalue())
        try:
            from src.video_detection import detect_video
            output = detect_video(temporary, confidence)
            st.video(str(output))
            st.download_button("Download processed video", output.read_bytes(), output.name, "video/mp4")
        except Exception as error:
            st.error(str(error))
        finally:
            temporary.unlink(missing_ok=True)

elif page == "About Project":
    st.title("About this project")
    st.markdown((ROOT / "README.md").read_text(encoding="utf-8"))
