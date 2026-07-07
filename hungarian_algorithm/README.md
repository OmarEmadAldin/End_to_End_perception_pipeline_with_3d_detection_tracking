# The Hungarian Algorithm

## Overview

This module performs **multi-sensor fusion** by associating:

-   Camera detections (2D)
-   LiDAR points (3D position)
-   Radar points (3D position + velocity)

The fusion is done by: 
1. Projecting LiDAR and Radar points into the
camera image plane 
2. Matching them with camera detections using
distance-based assignment 
3. Producing a unified representation per
object

------------------------------------------------------------------------

## Pipeline

### 1. Input Data

-   **Camera Event**
    -   Intrinsics matrix `K`
    -   Calibration (sensor-to-ego transform)
    -   Ego pose (global position/orientation)
    -   2D detections
-   **LiDAR Points**
    -   Global coordinates `(x, y)`
-   **Radar Points**
    -   Global coordinates `(x, y)`
    -   Velocity `(vx, vy)`

------------------------------------------------------------------------

### 2. Coordinate Transformation

Each point is transformed:

Global Frame → Ego Frame → Camera Frame

``` python
p_cam = global_to_sensor(p_global, calib, ego)
```

------------------------------------------------------------------------

### 3. Projection to Image

3D points are projected into the image plane:

``` python
uv = project_to_image(p_cam, K)
```

Pinhole camera model:

u = fx \* X / Z + cx\
v = fy \* Y / Z + cy

------------------------------------------------------------------------
The association between camera detections and projected sensor points is
based on **Euclidean distance in image (pixel) space**.

For each pair of:
- detection **i** with pixel center `(u_i, v_i)`
- projected sensor point **j** with pixel location `(u_j, v_j)`

the cost is defined as:

```
C(i, j) = || p_det(i) - p_proj(j) ||_2
```

Expanded:

```
C(i, j) = sqrt( (u_i - u_j)^2 + (v_i - v_j)^2 )
```

Where:

| Symbol      | Meaning                                              |
|-------------|------------------------------------------------------|
| `u_i, v_i`  | Pixel center of camera detection i (from detector)   |
| `u_j, v_j`  | Projected pixel of sensor point j (lidar or radar)   |
| `C(i, j)`   | Cost (pixel distance) between detection i and point j|

---

## 4.1 Full Cost Matrix

For **N** detections and **M** projected sensor points, we build an
N × M cost matrix:

```
         point_0   point_1   point_2  ...  point_M
det_0  [  C(0,0)   C(0,1)   C(0,2)  ...  C(0,M) ]
det_1  [  C(1,0)   C(1,1)   C(1,2)  ...  C(1,M) ]
det_2  [  C(2,0)   C(2,1)   C(2,2)  ...  C(2,M) ]
  .    [    .         .        .              .   ]
det_N  [  C(N,0)   C(N,1)   C(N,2)  ...  C(N,M) ]
```


------------------------------------------------------------------------

### 5. Hungarian Matching

min Σ C(i, j)

Using: linear_sum_assignment()

------------------------------------------------------------------------

### 6. Output

Each fused object contains:

{ id, class, confidence, pixel_center,

lidar_x, lidar_y, lidar_score, lidar_px_dist,

radar_x, radar_y, radar_vx, radar_vy, radar_px_dist }

