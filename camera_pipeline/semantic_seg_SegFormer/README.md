# SegFormer Semantic Segmentation Pipeline

A high-performance semantic segmentation pipeline using SegFormer (NVIDIA) via Hugging Face Transformers. Designed for processing image folders, generating colorized segmentation outputs, and exporting GIF visualizations.

## Features
- Semantic segmentation using SegFormer (B2 Cityscapes)
- GPU acceleration (automatic if CUDA is available)
- Batch processing of image folders
- Colorized segmentation maps (Cityscapes-style palette)
	Export:
		Segmented images
		Animated GIF visualization

## Model
Default model:
```
nvidia/segformer-b2-finetuned-cityscapes-1024-1024
- Pretrained on Cityscapes dataset
- Optimized for urban scenes (roads, cars, pedestrians, buildings)
```

## Dependencies
```
pip install torch torchvision torchaudio 
pip install transformers 
pip install pillow 
pip install numpy 
pip install tqdm
```

## Usage
**Initialize Segmenter**
```
segmenter = SegFormerSegmenter( model_name="nvidia/segformer-b2-finetuned-cityscapes-1024-1024" )
```

**Process Folder**
```
segmenter.process_folder( input_folder="input_images", save_dir="output" )
```

**Generate GIF**
```
segmenter.imgs_to_gif( folder="output", output_gif="segmentation.gif", fps=30 )
```

## Visualization coloring
The segmentation output is colorized using a simplified Cityscapes palette:

Class	         ID	        Color (RGB)
Background	  0	          (0, 0, 0)
Road	          1	          (255, 255, 255)
Sidewalk	  2	          (244, 35, 232)
Pole	          5	          (153, 153, 153)
Traffic Light	  7	          (250, 170, 30)
Building	  10	          (70, 70, 70)
Person	          11	          (220, 20, 60)
Car	          13	          (0, 0, 142)

## Result

<p align="center">
  <img src="output/output.gif" width="400" alt="SegFormer Result">
</p>
