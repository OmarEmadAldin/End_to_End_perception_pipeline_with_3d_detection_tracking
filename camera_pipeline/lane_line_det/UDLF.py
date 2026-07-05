import os
import cv2
import glob
import subprocess
import numpy as np
from tqdm import tqdm
from .ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector, ModelType
class UltrafastLaneSegmenter__:
    def __init__(self, model_path, model_type=ModelType.CULANE,use_gpu=False):


        self.model_path = model_path
        self.model_type = model_type
        self.use_gpu = use_gpu

        self.detector = UltrafastLaneDetector(
            self.model_path,
            self.model_type
        )

    def _process_single(self, image_path, save_dir):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        output_img = self.detector.detect_lanes(img)

        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"lane_{filename}")

        cv2.imwrite(save_path, output_img)

        return save_path

    def process_single_image(self, image_path,
                            save_path=None, show=False):

        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        output_img = self.detector.detect_lanes(img)

        if show:
            cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)
            cv2.imshow("Detected lanes", output_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save_path:
            cv2.imwrite(save_path, output_img)

        return output_img

    def process_folder(self, input_folder, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")

        image_files = [
            f for f in sorted(os.listdir(input_folder))
            if f.lower().endswith(valid_exts)
        ]

        results = []

        for filename in tqdm(image_files, desc="Processing Images", unit="img"):
            image_path = os.path.join(input_folder, filename)

            try:
                save_path = self._process_single(image_path, save_dir)
                results.append((image_path, save_path))

            except Exception as e:
                tqdm.write(f"[ERROR] Failed on {filename}: {e}")

        # print(f"\n[LaneDetector] Processed {len(results)} images")
        return results

    def extract_lane_mask(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        output = self.detector.detect_lanes(img)

        hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
        yellow_mask = cv2.inRange(hsv, (15, 80, 80), (35, 255, 255))

        mask = cv2.bitwise_or(white_mask, yellow_mask)

        return mask

    def imgs_to_gif(self, folder,
                    output_gif="output.gif", fps=30):

        images = sorted(glob.glob(os.path.join(folder, "lane_*")))

        if not images:
            raise ValueError(f"No images found in {folder} matching 'lane_*'")

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

