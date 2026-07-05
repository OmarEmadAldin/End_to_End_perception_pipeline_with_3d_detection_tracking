# Multi YOLO Detector

A multi-model object detection pipeline built on Ultralytics YOLO, combining multiple custom models with a COCO-pretrained model for enhanced detection and vehicle center extraction.

## Features
Run multiple YOLO models simultaneously:
Two custom-trained models
One COCO-pretrained YOLOv8 model
Merge detections from all models
Extract vehicle center points (cars, trucks, buses, motorcycles)
Save:
Annotated images
JSON file with detection centers
Generate animated GIF from output frames

## Models Used
- model1 → Custom YOLO model for traffic light detection then we do color filter to know which color
- model2 → Custom YOLO model traffic signs based on it we change the car behaviour
- model3 → COCO pretrained (yolov8n.pt by default) mainly concerned here with vehicle detection

## Usage
**Initialize Detector**
```
detector = MultiYOLODetector(
    model1_path="models/model1.pt",
    model2_path="models/model2.pt",
    coco_model="yolov8n.pt"
)
```
**Process Images**
```
detector.process_folder( input_folder="input_images", save_dir="output" )
```

** **
```
detector.imgs_to_gif( folder="output", output_gif="result.gif", fps=30 )
```
In which we use **FFmpeg** command

