# 3D Object Detection & BEV Visualization using PointPillars

This project performs **3D object detection on LiDAR point cloud data** using the **PointPillars model** from MMDetection3D. 
---

## Model & Framework
- Framework: MMDetection3D (OpenMMLab)
- Model: PointPillars (nuScenes pretrained)


---

## Required Files

You must download:

### 1. Config file (Will be added with the code)
configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py

### 2. Checkpoint file (Will be added with the code)
hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d.pth

---

## Installation

```bash
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d

pip install -r requirements.txt
pip install open3d opencv-python tqdm

pip install -v -e .
```
you could use openPCDet but i have issue with my laptop
---

##  Usage

Update paths in the script:

```python
config_file = "path/to/config.py"
checkpoint_file = "path/to/checkpoint.pth"

input_folder = "path/to/lidar/bin/files"
output_3d_folder = "path/to/output/3d"
output_bev_folder = "path/to/output/bev"
```

Run:
```bash
python 3d_det_pointpillar.py
```

---

##  Outputs

### 3D Visualization
- Rendered using Open3D
- Includes bounding boxes with orientation

###  BEV Visualization
- Grid-based top-down map
- Ego vehicle shown at center
- Bounding boxes + heading direction (which is probably wrong)

### JSON Output
Each frame contains:
```json
{
  "frame.bin": [
    {
      "class": "car",
      "x": 10.2,
      "y": -3.5,
      "score": 0.95
    }
  ]
}
```

We could have done all of that using PCL instead of pointpillars but pointpillars more advanced and there's another project in which i use PCL over a KITTI dataset sample
---

## supported Classes
Filtered vehicle classes:
- car
- truck
- bus


---
## Results
<p align="center">
  <img src="output/out.gif" width="45%" />
  <img src="output/output__.gif" width="25%" />
</p>