# Road Sign Detection — YOLOv8 Ensemble

A real-time road sign detection system built with a two-model YOLOv8 ensemble, capable of identifying **47 road sign classes** from both static images and dashcam video footage.

---

## Demo

Upload an image or video to the live Gradio app to see detections instantly.

> Run `road_sign_detection_mvp.ipynb` in Google Colab and click the public URL generated in the final cell.

---

## Results

| Model | Architecture | Classes | mAP@50 | mAP@50-95 | Precision | Recall | Inference |
|---|---|---|---|---|---|---|---|
| Speed Signs | YOLOv8s | 15 | 93.99% | 78.69% | 89.71% | 91.38% | ~4ms/frame |
| LISA Road Signs | YOLOv8n | 47 | — | — | — | — | ~4ms/frame |

Both models run on GPU (T4). Total ensemble pipeline: **~8ms per frame**.

---

## ML Pipeline

### 1. Dataset
- **Speed Signs**: Custom balanced dataset of 15 speed-related sign classes (~638 test images)
- **LISA Road Signs**: [LISA Traffic Sign Dataset](https://universe.roboflow.com/dakota-smith/lisa-road-signs) via Roboflow — 47 classes, ~23,000 annotated images in YOLOv8 format
- Both datasets use YOLO-format bounding box annotations `[class x_center y_center width height]`

### 2. Training
Each model was trained independently on its respective dataset:

```
Epochs        : 30
Image size    : 640 × 640
Batch size    : 16
Optimizer     : SGD  (lr=0.01, weight_decay=0.001)
LR scheduler  : Cosine decay (lrf=0.1)
Early stopping: patience=10
Hardware      : Google Colab T4 GPU
```

- **Speed model**: YOLOv8s (small) — larger backbone for better accuracy on fine-grained speed limit numbers
- **LISA model**: YOLOv8n (nano) — lightweight, fast, handles the broader 47-class set

### 3. Ensemble: Weighted Boxes Fusion (WBF)
Rather than using a single model, both models run on every image and their predictions are merged using **Weighted Boxes Fusion**:

1. Each model outputs bounding boxes in pixel coordinates
2. Boxes are **normalized to [0, 1]** before fusion (critical — raw pixel coordinates break WBF)
3. WBF merges overlapping boxes from both models using a confidence-weighted average
4. Boxes are **denormalized back to pixel coordinates** for drawing

```python
fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
    boxes_list, scores_list, labels_list,
    weights=[1.0, 1.0],   # equal trust in both models
    iou_thr=0.5,           # merge boxes with >50% overlap
    skip_box_thr=0.4       # discard detections below 40% confidence
)
```

This approach reduces false positives — a detection only survives if at least one model is confident about it, and boxes agreed upon by both models get a boosted score.

### 4. Evaluation
Both models are evaluated on held-out test sets with the following metrics:
- **mAP@50** — mean Average Precision at IoU threshold 0.50 (primary metric)
- **mAP@50-95** — mAP averaged across IoU thresholds 0.50 to 0.95 (stricter)
- **Precision** — of all predicted boxes, how many are correct
- **Recall** — of all ground truth signs, how many were detected
- **Confusion matrix** — generated automatically via `plots=True` in `.val()`

---

## Project Structure

```
Yolo-Object/
├── road_sign_detection_mvp.ipynb   # Main notebook — run this in Colab
└── README.md
```

Model weights are stored in Google Drive (not committed to the repo due to file size):
```
My Drive/runs/train/
├── new_dataset_yolo8s_30epochs5/weights/best.pt   # Speed sign model
└── Lisa_yolo8n_30epochs/weights/best.pt            # LISA model
```

---

## How to Run

### Requirements
- Google Colab (free tier works; T4 GPU required)
- Google Drive with trained model weights at the paths above
- Roboflow API key (stored as a Colab Secret named `ROBOFLOW_KEY`)

### Steps

1. Open `road_sign_detection_mvp.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**: `Runtime → Change runtime type → T4 GPU`
3. Add your Roboflow API key: click the 🔑 icon → add secret `ROBOFLOW_KEY`
4. Run all cells top to bottom (`Runtime → Run all`)
5. The final cell launches the Gradio app — click the public link to open the demo

### Gradio App Features
| Tab | Input | Output |
|---|---|---|
| Image | Upload any road sign photo | Annotated image with bounding boxes and confidence scores |
| Dashcam Video | Upload .mp4 / .avi / .mov clip | Fully annotated video with detections on every frame |

---

## Key Design Decisions

**Why two models instead of one?**
The speed sign dataset was curated and balanced specifically for fine-grained number recognition (e.g. telling apart Speed Limit 30 vs 35 vs 40). Training a separate specialist model on it and combining with the broader LISA model via WBF gives better coverage than a single model trained on both datasets combined.

**Why WBF over NMS for fusion?**
Non-Maximum Suppression (NMS) picks one box and discards the rest. WBF averages the coordinates of overlapping boxes weighted by their confidence scores, producing tighter bounding boxes and fewer false positives when models agree.

**Why normalize boxes before WBF?**
WBF expects coordinates in [0, 1] range. Passing raw pixel coordinates (e.g. 640×480) causes the fusion algorithm to compute incorrect IoU overlaps, silently producing wrong detections. All boxes are divided by image width/height before fusion and multiplied back after.

---

## Dataset Citation

```
LISA Traffic Sign Dataset
@misc{lisa-road-signs_dataset,
  title        = { LISA Road Signs Dataset },
  author       = { Dakota Smith },
  year         = { 2023 },
  url          = { https://universe.roboflow.com/dakota-smith/lisa-road-signs },
  note         = { visited on 2025-04-01 },
  organization = { Roboflow }
}
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection models |
| [ensemble-boxes](https://github.com/ZFTurbo/Weighted-Boxes-Fusion) | Weighted Boxes Fusion |
| [Roboflow](https://roboflow.com) | Dataset management and download |
| [Gradio](https://gradio.app) | Web demo UI |
| OpenCV | Image and video processing |
| Google Colab + T4 GPU | Training and inference |
