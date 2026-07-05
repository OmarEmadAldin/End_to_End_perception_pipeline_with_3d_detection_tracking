import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import glob
import subprocess
import json

class MultiYOLODetector:

    def __init__(self,
                 model1_path,
                 model2_path,
                 coco_model="yolov8n.pt",
                 device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model1 = YOLO(model1_path)
        self.model2 = YOLO(model2_path)
        self.model3 = YOLO(coco_model)

        self.names1 = self.model1.names
        self.names2 = self.model2.names
        self.names3 = self.model3.names

        self.vehicle_classes = {"car", "truck", "bus", "motorcycle"}

    def _extract_boxes(self, result, names, source):
        detections = []
        vehicle_centers = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = names[cls]

            det = {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "conf": conf,
                "class": label,
                "source": source
            }

            detections.append(det)

            if source == "coco" and label in self.vehicle_classes:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                vehicle_centers.append({
                    "class": label,
                    "center": (cx, cy),
                    "conf": conf
                })

        return detections, vehicle_centers

    def _process_single(self, image_path, save_dir):

        frame = cv2.imread(image_path)
        r1 = self.model1(frame)[0]
        r2 = self.model2(frame)[0]
        r3 = self.model3(frame)[0]

        detections = []
        vehicle_centers = []

        d1, _ = self._extract_boxes(r1, self.names1, "custom1")
        d2, _ = self._extract_boxes(r2, self.names2, "custom2")
        d3, vc = self._extract_boxes(r3, self.names3, "coco")

        detections += d1 + d2 + d3
        vehicle_centers += vc

        # Draw results
        output_img = self._draw(frame, detections, vehicle_centers)
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"det_{filename}")
        cv2.imwrite(save_path, output_img)

        return save_path, vehicle_centers


    def _draw(self, frame, detections, vehicle_centers):
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['class']} {det['conf']:.2f}"

            color = (0, 255, 0) if det["source"] == "coco" else (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw centers for vehicles
        for v in vehicle_centers:
            cx, cy = v["center"]
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{v['class']}",
                        (cx + 5, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)

        return frame

    def process_folder(self, input_folder, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [
            f for f in sorted(os.listdir(input_folder))
            if f.lower().endswith(valid_exts)]

        results = []
        detections = []
        for filename in tqdm(image_files, desc="Processing Images", unit="img"):
            image_path = os.path.join(input_folder, filename)

            try:
                save_path, centers = self._process_single(image_path, save_dir)
                results.append((image_path, save_path, centers))
                detections.append((centers , image_path))
            except Exception as e:
                tqdm.write(f"[ERROR] Failed on {filename}: {e}")
                
        json_path = os.path.join(save_dir, "detections_xy.json")
        with open(json_path, "w") as fp:
            json.dump(detections, fp, indent=2)
        return results

    # -----------------------------
    def imgs_to_gif(self, folder, output_gif="output.gif", fps=30):

        images = sorted(glob.glob(os.path.join(folder, "det_*")))
        if not images:
            raise ValueError(f"No images found in {folder} matching 'det_*'")

        list_file = os.path.join(folder, "images.txt")
        with open(list_file, "w") as f:
            for img in images:
                f.write(f"file '{img}'\n")

        palette_path = os.path.join(folder, "palette.png")
        gif_path = os.path.join(folder, output_gif)

        subprocess.run([
            "ffmpeg", "-y",
            "-r", str(fps),
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-vf", "palettegen",
            palette_path
        ], check=True)

        subprocess.run([
            "ffmpeg", "-y",
            "-r", str(fps),
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-i", palette_path,
            "-lavfi", "paletteuse",
            gif_path
        ], check=True)

        print(f"GIF saved at: {gif_path}")