import os
import torch
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from tqdm import tqdm
import subprocess
import glob

class SegFormerSegmenter:

    def __init__(self,
             model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024",
             device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Cityscapes color palette (simplified but meaningful)
        self.color_map = {
            0: (0, 0, 0),          # background
            1: (255, 255, 255),    # road
            2: (244, 35, 232),    # sidewalk
            5: (153, 153, 153),   # pole
            7: (250, 170, 30),    # traffic light
            10: (70, 70, 70),     # building
            11: (220, 20, 60),    # person
            13: (0, 0, 142),      # car
        }

    def _colorize(self, seg_map):
        h, w = seg_map.shape
        color_img = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in self.color_map.items():
            color_img[seg_map == class_id] = color

        return color_img

    def _process_single(self, image_path, save_dir):
        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False
        )

        pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Colorized segmentation
        color_seg = self._colorize(pred_seg)
        seg_image = Image.fromarray(color_seg)

        filename = os.path.basename(image_path)
        save_path = os.path.join(save_dir, f"seg_{filename}")
        seg_image.save(save_path)
        # print(f"[SegFormer] Saved: {save_path}")

        return save_path

    def process_folder(self, input_folder, save_dir):
        os.makedirs(save_dir, exist_ok=True)

        valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
        image_files = [ f for f in sorted(os.listdir(input_folder)) if f.lower().endswith(valid_exts) ]
        results = []
        for filename in tqdm(image_files, desc="Processing Images", unit="img"): 
            image_path = os.path.join(input_folder, filename) 
            try: 
                save_path = self._process_single(image_path, save_dir) 
                results.append((image_path, save_path)) 
            except Exception as e: 
                tqdm.write(f"[ERROR] Failed on {filename}: {e}") 
                # print(f"\n[SegFormer] Processed {len(results)} images") 
        return results

    def imgs_to_gif(self, folder, output_gif="output.gif", fps=30):

        images = sorted(glob.glob(os.path.join(folder, "seg_*")))

        if not images:
            raise ValueError(f"No images found in {folder} matching 'seg_*'")

        # Build a temporary text file list for ffmpeg (most reliable way)
        list_file = os.path.join(folder, "images.txt")
        with open(list_file, "w") as f:
            for img in images:
                f.write(f"file '{img}'\n")

        palette_path = os.path.join(folder, "palette.png")
        gif_path = os.path.join(folder, output_gif)

        # Generate palette
        subprocess.run([
            "ffmpeg", "-y",
            "-r", str(fps),
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-vf", "palettegen",
            palette_path
        ], check=True)

        # Generate GIF
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