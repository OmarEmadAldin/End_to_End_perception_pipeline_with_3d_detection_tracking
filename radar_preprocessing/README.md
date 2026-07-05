# Radar Processing & Visualization Pipeline

A radar data processing pipeline for **nuScenes RADAR sweeps**, generating:

- Polar visualizations
- Angular velocity plots
- Structured JSON outputs for downstream tracking (e.g., Kalman Filter, sensor fusion)

I can have the vx and vy directly and build linear kalman filter but i want to apply EKF and learn its normalization method
---

## Radar Data Representation

Each radar point includes:

| Feature   | Description |
|----------|-------------|
| x, y     | Cartesian coordinates |
| vx, vy   | Velocity components |
| rho      | Range (distance) |
| phi      | Angle |
| phi_dot  | Angular velocity |



---

##  Dependencies

Install required Python packages:

```bash
pip install numpy pandas matplotlib tqdm
pip install nuscenes-devkit
```

---

##  Usage

### 1. Configure Paths

Update paths in the script:

```python
INPUT_DIR = "path/to/RADAR_FRONT"
OUTPUT_DIR = "path/to/output"
```

---

### 2. Run Pipeline

```bash
python radar_pipeline.py
```

---

## Outputs

### 1. Polar Plot

- Radar points in polar coordinates
- Useful for:
  - Range-angle analysis
  - Sensor behavior visualization

---

### 2. Velocity Plot

- Scatter plot of:
  - `phi` (angle)
  - `phi_dot` (angular velocity)
- Useful for:
  - Motion pattern analysis
  - Tracking initialization

---

### 3. JSON Output

File:

```
radar_all_frames.json
```

Structure:

```json
{
  "frames": [
    {
      "frame_id": 0,
      "num_points": 120,
      "radar_points": [
        {
          "rho": 12.3,
          "phi": 0.45,
          "phi_dot": -0.02,
          "x": 10.2,
          "y": 5.1,
          "vx": 1.2,
          "vy": -0.3
        }
      ]
    }
  ]
}
```

---

### 4. GIF Visualizations

- `polar_plot.gif`
- `velocity_plot.gif`

---

## Key Processing Steps

1. Load `.pcd` radar sweep
2. Extract raw radar features
3. Filter **moving targets only**
4. Convert to polar space
5. Compute angular velocity
6. Save visualizations and JSON

---

## Notes

- Only **moving objects** are used:
  ```python
  dyn_prop == 0
  ```
- Handles empty radar frames safely

---

## Result
<p align="center">
  <img src="output_p.gif" width="45%" />
  <img src="output.gif" width="45%" />
</p>
