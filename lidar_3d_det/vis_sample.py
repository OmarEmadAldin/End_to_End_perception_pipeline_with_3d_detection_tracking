import numpy as np
import open3d as o3d

# Load NuScenes point cloud (.bin with 5 values)
file_path = "/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/sweeps/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151605198172.pcd.bin"
points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 5)

# Keep only x, y, z (ignore intensity & ring index for visualization)
xyz = points[:, :3]
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)

z_vals = xyz[:, 2]
z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min())
colors = np.stack([z_norm, 1 - z_norm, 0.5 * np.ones_like(z_norm)], axis=1)
pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([pcd])