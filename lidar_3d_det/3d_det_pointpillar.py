import os
import json
import numpy as np
import open3d as o3d
import cv2
from tqdm import tqdm
from mmdet3d.apis import init_model, inference_detector
from config import *

class PointPillars3DRenderer:
    def __init__(self, config, checkpoint, device='cuda:0', score_thr=0.3):

        self.model = init_model(config, checkpoint, device=device)
        self.score_thr = score_thr
        self.num_point_features = 5

        # ---------------- BEV CONFIG ----------------
        self.xmin, self.xmax = -51.2, 51.2
        self.ymin, self.ymax = -51.2, 51.2
        self.bev_res = 0.1

        self.img_w = int((self.xmax - self.xmin) / self.bev_res)
        self.img_h = int((self.ymax - self.ymin) / self.bev_res)

    def process_folder(self, input_folder, output_3d_folder, output_bev_folder):

        os.makedirs(output_3d_folder, exist_ok=True)
        os.makedirs(output_bev_folder, exist_ok=True)

        files = sorted([f for f in os.listdir(input_folder) if f.endswith('.bin')])

        all_detections = {}

        for f in tqdm(files):
            path = os.path.join(input_folder, f)

            save_3d = os.path.join(output_3d_folder, f.replace('.bin', '.png'))
            save_bev = os.path.join(output_bev_folder, f.replace('.bin', '.png'))

            detections = self._process_single(path, save_3d, save_bev)
            all_detections[f] = detections

        # Save JSON (GLOBAL)
        json_path = os.path.join(output_3d_folder, "detections_xy.json")
        with open(json_path, "w") as fp:
            json.dump(all_detections, fp, indent=2)

        print(f"[INFO] JSON saved at: {json_path}")
        return all_detections

    def _process_single(self, pcd_path, save_3d_path, save_bev_path):

        result, _ = inference_detector(self.model, pcd_path)

        bboxes = result.pred_instances_3d.bboxes_3d
        scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        labels = result.pred_instances_3d.labels_3d.cpu().numpy()

        # SAFE extraction
        centers = bboxes.gravity_center.cpu().numpy()
        dims = bboxes.dims.cpu().numpy()
        yaws = bboxes.yaw.cpu().numpy()

        boxes = np.concatenate([centers, dims, yaws[:, None]], axis=1)

        # FILTER
        mask = scores >= self.score_thr
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        detections_xy = []
        for box, label, score in zip(boxes, labels, scores):

            class_name = NUSCENES_CLASSES[int(label)]

            if class_name in VEHICLE_CLASSES:
                cx, cy = box[0], box[1]

                detections_xy.append({
                    "class": class_name,
                    "x": float(cx),
                    "y": float(cy),
                    "score": float(score)
                })

        # ---------------- 3D RENDER ----------------
        self._render_and_save(pcd_path, boxes, labels, scores, save_3d_path)
        # ---------------- LOAD POINTS FOR BEV ----------------
        pts = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, self.num_point_features)
        points_xy = pts[:, :2]
        # ---------------- BEV RENDER ----------------
        self._save_bev(points_xy, boxes, labels, scores, save_bev_path)

        return detections_xy

    def _render_and_save(self, pcd_path, boxes, labels, scores, save_path):

        pts = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, self.num_point_features)
        xyz = pts[:, :3]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        colors = np.ones_like(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = pcd.voxel_down_sample(0.1)

        geometries = [pcd]

        for box, label, score in zip(boxes, labels, scores):

            cx, cy, cz, dx, dy, dz, yaw = box
            class_name = NUSCENES_CLASSES[int(label)]
            color = COLOR_MAP_3D.get(class_name, (0,1,1))

            R = o3d.geometry.get_rotation_matrix_from_axis_angle([0, 0, yaw])

            obb = o3d.geometry.OrientedBoundingBox(
                center=(cx, cy, cz),
                R=R,
                extent=(dx, dy, dz)
            )
            obb.color = color

            geometries.append(obb)

        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)

        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0, 0])
        opt.point_size = 1.0

        for g in geometries:
            vis.add_geometry(g)

        ctr = vis.get_view_control()
        ctr.set_front([0, -1, -0.5])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, 0, 1])
        ctr.set_zoom(0.3)

        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(save_path)
        vis.destroy_window()

    def _save_bev(self, points_xy, boxes, labels, scores, save_path):

        img = np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        # GRID
        for x in np.arange(self.xmin, self.xmax, 10):
            px1, py1 = self._world_to_pixel(x, self.ymin)
            px2, py2 = self._world_to_pixel(x, self.ymax)
            cv2.line(img, (px1, py1), (px2, py2), (40, 40, 40), 1)

        for y in np.arange(self.ymin, self.ymax, 10):
            px1, py1 = self._world_to_pixel(self.xmin, y)
            px2, py2 = self._world_to_pixel(self.xmax, y)
            cv2.line(img, (px1, py1), (px2, py2), (40, 40, 40), 1)

        # POINT CLOUD
        for x, y in points_xy:
            if self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax:
                px, py = self._world_to_pixel(x, y)
                img[py, px] = (80, 80, 80)

        # EGO VEHICLE
        ego_corners = self._box_corners_bev(0, 0, 4.5, 2.0, 0.0)
        ego_px = np.array([self._world_to_pixel(x, y) for x, y in ego_corners], dtype=np.int32)
        cv2.fillPoly(img, [ego_px], (255, 255, 255))

        # BOXES
        for box, label, score in zip(boxes, labels, scores):

            cx, cy, cz, dx, dy, dz, yaw = box
            class_name = NUSCENES_CLASSES[int(label)]
            color = CLASS_COLORS.get(class_name, (255, 255, 255))

            corners = self._box_corners_bev(cx, cy, dx, dy, yaw)
            corners_px = np.array([self._world_to_pixel(x, y) for x, y in corners], dtype=np.int32)

            cv2.polylines(img, [corners_px], True, color, 2)

            # heading
            fx = cx + (dx/2)*np.cos(yaw)
            fy = cy + (dx/2)*np.sin(yaw)

            px1, py1 = self._world_to_pixel(cx, cy)
            px2, py2 = self._world_to_pixel(fx, fy)

            cv2.line(img, (px1, py1), (px2, py2), color, 2)

            cv2.putText(img, class_name, (px1+3, py1-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # AXIS
        px0, py0 = self._world_to_pixel(0, 0)
        px1, py1 = self._world_to_pixel(10, 0)
        cv2.arrowedLine(img, (px0, py0), (px1, py1), (255,255,255), 2)

        cv2.imwrite(save_path, img)

    def _world_to_pixel(self, x, y):
        px = int((x - self.xmin) / self.bev_res)
        py = int((self.ymax - y) / self.bev_res)
        return px, py

    def _box_corners_bev(self, cx, cy, dx, dy, yaw):

        half_dx, half_dy = dx/2, dy/2

        corners = np.array([
            [ half_dx,  half_dy],
            [ half_dx, -half_dy],
            [-half_dx, -half_dy],
            [-half_dx,  half_dy],
        ])

        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])

        corners = corners @ rot.T
        corners[:, 0] += cx
        corners[:, 1] += cy

        return corners


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":

    config_file = "/home/omar_ben_emad/3d_object_detection_and_tracking/mmdetection3d/configs/pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py"
    checkpoint_file = "/home/omar_ben_emad/3d_object_detection_and_tracking/lidar_3d_detector/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth"

    input_folder = "/home/omar_ben_emad/3d_object_detection_and_tracking/data/v1.0-mini/sweeps/LIDAR_TOP"

    output_3d_folder = "/home/omar_ben_emad/3d_object_detection_and_tracking/output/3d_detections"
    output_bev_folder = "/home/omar_ben_emad/3d_object_detection_and_tracking/output/bev_detections"


    detector = PointPillars3DRenderer(config_file, checkpoint_file)

    detector.process_folder(
        input_folder,
        output_3d_folder,
        output_bev_folder
    )
