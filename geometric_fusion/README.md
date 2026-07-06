# Geometric Fusion

## Overview
These two python codes deals with the data preprocessed from the camera,radar and lidar. and  do the transformations needed for frames alignment using the calibration data provided with the nuscenes sample.
And also it makes fixed duration between the timestamps and add the available data in each one. it make a json files with a proportion between all the readings.

## Pipeline

1. Load raw detections from:
   - LiDAR (x, y)
   - Radar (range, angle, velocity)
   - Camera (pixel detections)

2. Transform all detections:
   sensor → ego → global coordinate frame

3. Temporal alignment:
   - Normalize timestamps
   - Group detections into fixed time windows

4. Fusion:
   - Aggregate detections across sensors
   - Store unified frame representation

5. Output:
   - JSON file with aligned multi-sensor detections per frame
   
## Mathematical Foundations

### Quaternion to Rotation Matrix
R(q) =
[1 - 2(y^2 + z^2)   2(xy - zw)     2(xz + yw)]
[2(xy + zw)         1 - 2(x^2 + z^2) 2(yz - xw)]
[2(xz - yw)         2(yz + xw)     1 - 2(x^2 + y^2)]

### Coordinate Transformation
(R is rotation matrix and t is translational matrix
ego is the vechile

p_ego = R_se * p_sensor + t_se   )
p_global = R_eg * p_ego + t_eg

### Velocity Transformation
v_global = R_eg * (R_se * v_sensor)

### Radar Conversio
x = rho * cos(phi)
y = rho * sin(phi)

### Camera Projection
[u v w]^T = K [X Y Z]^T
u = u/w, v = v/w
