"""Initial Streamlit entry point for the road damage detection project."""

import streamlit as st

from config import (
    CLASS_IDS,
    CLASS_NAMES,
    DEFAULT_COUNTRY,
    RDD2022_REPOSITORY_URL,
    ensure_project_directories,
)


st.set_page_config(
    page_title="Automated Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
)

ensure_project_directories()

st.title("Real-Time Automated Road Damage Detection")
st.info(
    "Phase 1 setup is complete. The next phases will download RDD2022, convert its "
    "Pascal VOC annotations, train a project-owned model, and enable live webcam "
    "detection."
)

st.subheader("Configured damage categories")
st.dataframe(
    {
        "Class ID": CLASS_IDS,
        "Class name": [CLASS_NAMES[class_id] for class_id in CLASS_IDS],
    },
    hide_index=True,
    use_container_width=True,
)

st.subheader("Project workflow")
st.write(
    "1. Download RDD2022 → 2. Extract and convert XML annotations → "
    "3. Validate and split the dataset → 4. Train from scratch or fine-tune → "
    "5. Evaluate → 6. Detect with images, video, and the webcam."
)

st.warning(
    "No model is bundled with this project. Detection will remain unavailable until "
    "a model is trained by this project from the prepared RDD2022 data."
)

st.caption(f"Dataset source: {RDD2022_REPOSITORY_URL} | Default country: {DEFAULT_COUNTRY}")
