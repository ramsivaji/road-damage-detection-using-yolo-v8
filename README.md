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
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks activation for the current user, run:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

The requirements file pins CUDA 12.8 PyTorch wheels. They are compatible with
the NVIDIA driver runtime reported by `nvidia-smi`; PyTorch uses its bundled
CUDA runtime and does not require the CUDA toolkit to be installed separately.

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

## Run the complete workflow

```powershell
python -m src.download_dataset --country India
python src\convert_annotations.py
python src\prepare_dataset.py
python src\validate_dataset.py
python src\train.py --mode scratch --epochs 100 --batch 2 --imgsz 640 --device 0
python src\evaluate.py --device 0
python -m streamlit run app.py
```

The official RDD2022 archive is approximately 13 GB. The download script keeps
the original archive and extracted source under `dataset\downloads\`, then
copies only the selected country into `dataset\raw\`. Test images in RDD2022 do
not have XML labels, so only verified train image/XML pairs are included in the
YOLO training splits.
