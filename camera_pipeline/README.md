# Camera Pipeline
A modular perception pipeline combining:
- Object Detection (YOLO)
- Semantic Segmentation (SegFormer)
- Lane Detection (Ultrafast Lane Detection)

Designed for processing image sequences and generating visual outputs + intermediate data for downstream tasks (e.g., tracking, planning, sensor fusion).

---

## Components
### 1. Object Detection (YOLO)

- Multi-model detection:
  - Two custom YOLO models
  - One COCO pretrained model
- Extracts:
  - Bounding boxes
  - Vehicle center points (car, bus, truck, motorcycle)
- Outputs:
  - Annotated images
  - `detections_xy.json`

---

### 2. Semantic Segmentation (SegFormer)

- Model: `segformer-b2-cityscapes`
- Produces:
  - Pixel-wise classification
- Key classes:
  - Road, sidewalk, car, person, building
- Outputs:
  - Colorized segmentation images

---

### 3. Lane Detection (UFLD)

- Ultrafast Lane Detection
- Detects lane markings in real time
- Optional:
  - Binary lane mask extraction
- Outputs:
  - Lane overlay images

---


##  Usage (Basic Flow)

### 1. Object Detection

```python
detector.process_folder("input_images", "output/yolo")
```

---

### 2. Segmentation

```python
segmenter.process_folder("input_images", "output/segmentation")
```

---

### 3. Lane Detection

```python
lane_detector.process_folder("input_images", "output/lanes")
```

---

## Outputs

| Module        | Output Type              |
|--------------|--------------------------|
| YOLO         | Boxes + vehicle centers  |
| SegFormer    | Pixel-wise segmentation  |
| Lane         | Lane overlays + masks    |

---

## How They Work Together

- **Segmentation** → understands scene (road, sidewalk)
- **Lane detection** → extracts drivable structure
- **YOLO** → detects dynamic objects
