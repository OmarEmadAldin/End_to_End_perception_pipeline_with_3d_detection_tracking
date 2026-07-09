# Multi-Object Tracking Evaluation (MOT) Module

## Overview
This module provides a complete evaluation pipeline for **multi-object tracking (MOT)** systems.It consists of two main components:

- `Evaluator` → Handles dataset alignment, timestamp synchronization, and prediction interpolation
- `MOTMetrics` → Computes standard tracking metrics (MOTA, MOTP, Precision, Recall, etc.)

The design is tailored for **sensor-fusion tracking outputs** (e.g., LiDAR + Radar) and compares them against ground truth annotations.

---

## 1. Evaluation Pipeline

The evaluation follows this sequence:

1. Load ground truth data (samples, annotations, instances, categories)
2. Build ground truth objects per frame
3. Align tracker predictions with timestamps
4. Interpolate predictions when timestamps do not match exactly
5. Perform frame-by-frame matching
6. Accumulate metrics

---

## 2. Ground Truth Handling (`Evaluator`)

### Data Sources
The evaluator loads:

- `sample.json` → timestamps
- `sample_annotation.json` → object positions
- `instance.json` → object identity
- `category.json` → object classes

### Ground Truth Structure

Each object is represented as:

```python
{
    'id': instance_token,
    'x': position_x,
    'y': position_y,
    'class': category_name
}
```

Ground truth is indexed by:

```python
sample_token → list of objects
```

---

## 3. Timestamp Alignment

Tracking outputs and ground truth are rarely synchronized.

### Solution:
- Ground truth uses **discrete timestamps**
- Tracker outputs are **continuous histories**
- We interpolate tracker states at GT timestamps

---

## 4. Interpolation Logic

Given a track history:

```python
(ts_list, xy_list)
```

We compute:

α = (t_query - t0) / (t1 - t0)

Interpolated position:

x = (1 - α)x0 + αx1

### Constraints:
- No extrapolation beyond ±0.5 seconds
- Handles edge cases (start/end of track)

---

## 5. Track Filtering

Only valid tracking states are used:

```python
sensor ∈ {lidar, radar, birth(lidar)}
```

Tracks with fewer than 2 points are discarded.

---

## 6. Matching (Hungarian Algorithm)

For each frame:

1. Compute distance matrix:

d(i,j) = √((x_gt - x_pred)² + (y_gt - y_pred)²)

2. Apply threshold:

- Match only if distance < 2 meters

3. Solve assignment using:

```python
scipy.optimize.linear_sum_assignment
```

---

## 7. Metrics (`MOTMetrics`)

### True Positives (TP)
Matched GT–prediction pairs

### False Positives (FP)
Predictions with no matching GT

### False Negatives (FN)
GT objects not detected

### ID Switches (IDS)
Occurs when a GT object is matched to a different track ID than previous frame

---

## 8. Metric Formulas

### MOTA (Multi-Object Tracking Accuracy)

MOTA = 1 - (FN + FP + IDS) / GT

---

### MOTP (Precision)

MOTP = total distance error / total matches

---

### Precision

Precision = TP / (TP + FP)

---

### Recall

Recall = TP / (TP + FN)

---

### F1 Score

F1 = 2 × (Precision × Recall) / (Precision + Recall)

---

## 9. Output Example

```python
{
    'MOTA'     : 0.8421,
    'MOTP_m'   : 0.7321,
    'IDS'      : 5,
    'TP'       : 120,
    'FP'       : 15,
    'FN'       : 20,
    'Recall'   : 0.8571,
    'Precision': 0.8889,
    'F1'       : 0.8727,
    'Total_GT' : 140
}
```

