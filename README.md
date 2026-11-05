# Automated Road Damage Detection Using Images

This project will automatically download the publicly available RDD2022 Road
Damage Detection dataset, convert its Pascal VOC XML annotations to YOLO format,
train a YOLOv8 detector, and provide image, video, and live webcam detection
through Streamlit.

The project supports two model-training modes:

- **Train from scratch:** creates the YOLOv8 architecture from `yolov8n.yaml`
  with `pretrained=False`.
- **Transfer learning:** initializes from `yolov8n.pt` and fine-tunes on the
  prepared RDD2022 data.

## Phase 1 status

Phase 1 creates the project layout and initial configuration only. The Streamlit
application currently displays the RDD2022 classes and clearly indicates that
later workflows are not available yet. No dataset, labels, model weights, or
fake training results are included.

Configured classes:

| ID | Class |
|---:|---|
| D00 | Longitudinal Crack |
| D10 | Transverse Crack |
| D20 | Alligator Crack |
| D40 | Pothole |

The class list and RDD2022 source configuration are centralized in
[`config.py`](config.py) so future countries and classes can be added without
scattering configuration through the application.

## Windows setup

From PowerShell in this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation for the current user, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

PyTorch installation may need a platform-specific command for a CUDA-enabled
machine. Use the official PyTorch installation selector, then install the rest
of `requirements.txt`.

## Project layout

```text
road-damage-detection/
├── app.py
├── config.py
├── requirements.txt
├── README.md
├── assets/
├── dataset/
│   ├── downloads/
│   ├── raw/{images,annotations}/
│   ├── images/{train,val,test}/
│   ├── labels/{train,val,test}/
│   └── annotations/
├── models/
│   └── versions/
├── outputs/
│   ├── detections/
│   ├── evaluation/
│   ├── training/
│   └── videos/
└── src/
```

The empty directories are created during setup and will be populated by later
phases. Downloaded archives and extracted RDD2022 source files will remain under
`dataset/downloads/` and `dataset/raw/`; prepared split images and YOLO labels
will be generated separately.

## Important model policy

The eventual training workflow will support both explicit modes described above.
Live detection will load only the model produced by this project. It will not
silently substitute a missing or unrelated model.

## Next phase

After confirming this setup works locally, Phase 2 will add automatic RDD2022
download with country selection and progress reporting. Extraction, XML
conversion, dataset preparation, training, evaluation, and detection will
remain separate phases.
