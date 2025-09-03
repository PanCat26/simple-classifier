# Face Recognition & Object Classification App

This project uses **OpenCV** and **YOLOv8n** (from Ultralytics) to recognize faces and classify webcam images as **Owner**, **Other person**, **Pet**, or **Nobody**.

---

## Setup Instructions

### 1. Create and activate Conda environment
```bash
conda create -n facerec python=3.10 -y
conda activate facerec
```

### 2. Install dependencies
Install core dependencies via Conda and pip:

```bash
# Conda packages
conda install -c conda-forge cmake dlib ffmpeg numpy

# Pip packages
pip install opencv-python face_recognition ultralytics
```

### 3. Close other applications using webcam
Ensure no other program (Zoom, Teams etc.) is running and using the webcam.

---

## Usage

### Capture a picture
Run the picture taker script:
```bash
python picture_taker.py
```
- The webcam will open.  
- When satisfied with the picture, press **`s`** to save.

### Run the classifier
Run the classifier script:
```bash
python classifier.py
```
- The model will classify the input image into one of four categories:
  - **Owner**
  - **Other person**
  - **Pet**
  - **Nobody**

---

## Model

- **YOLOv8n** (nano model from [Ultralytics](https://github.com/ultralytics/ultralytics)) is used for object detection and classification.  

---
