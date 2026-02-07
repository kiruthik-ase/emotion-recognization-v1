# Emotion Recognition System 🎭

A complete emotion recognition project that uses **YOLO** for face detection and **FER** (Facial Expression Recognition) for classifying emotions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)

## ✨ Features

- **Real-time Face Detection** using YOLOv8 and Haar Cascade
- **7 Emotion Classes**: Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral
- **Multiple Input Modes**: Webcam, Images, Videos
- **Modern Web Interface** with Flask
- **Easy-to-use Python API**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd emotion_recognition
pip install -r requirements.txt
```

### 2. Run Webcam Demo

```bash
python scripts/webcam_demo.py
```

### 3. Run Web Application

```bash
python app/web_app.py
```

Open http://localhost:5000 in your browser.

## 📁 Project Structure

```
emotion_recognition/
├── src/                      # Core modules
│   ├── face_detector.py      # YOLO face detection
│   ├── emotion_classifier.py # FER emotion classification
│   ├── emotion_pipeline.py   # Combined pipeline
│   └── utils.py              # Utility functions
├── scripts/                  # Demo scripts
│   ├── webcam_demo.py        # Real-time webcam demo
│   ├── process_image.py      # Process single image
│   └── process_video.py      # Process video file
├── app/                      # Web application
│   ├── web_app.py            # Flask server
│   ├── templates/            # HTML templates
│   └── static/               # CSS styles
├── tests/                    # Unit tests
├── requirements.txt          # Dependencies
└── README.md                 # This file
```

## 🖥️ Usage

### Command Line Scripts

**Process an Image:**
```bash
python scripts/process_image.py --input photo.jpg --output result.jpg
```

**Process a Video:**
```bash
python scripts/process_video.py --input video.mp4 --output result.mp4 --preview
```

**Webcam Demo Controls:**
- `Q` - Quit
- `S` - Save screenshot
- `B` - Toggle emotion bar
- `F` - Toggle FPS display

### Python API

```python
from src.emotion_pipeline import create_pipeline
import cv2

# Create pipeline
pipeline = create_pipeline()

# Load image
image = cv2.imread("photo.jpg")

# Process
results = pipeline.process_image(image)

# Print results
for r in results:
    print(f"Emotion: {r.emotion} ({r.confidence:.1%})")

# Annotate and save
annotated = pipeline.annotate_image(image, results)
cv2.imwrite("result.jpg", annotated)
```

## 🎨 Detected Emotions

| Emotion | Emoji | Color |
|---------|-------|-------|
| Happy | 😊 | Yellow |
| Sad | 😢 | Blue |
| Angry | 😠 | Red |
| Fear | 😨 | Purple |
| Surprise | 😲 | Orange |
| Disgust | 🤢 | Green |
| Neutral | 😐 | Gray |

## ⚙️ Configuration

### Pipeline Options

```python
pipeline = create_pipeline(
    face_confidence=0.5,      # Min face detection confidence
    emotion_confidence=0.3,   # Min emotion confidence
    use_fer_detection=True    # Use FER's MTCNN (more accurate)
)
```

## 📝 Requirements

- Python 3.8+
- Webcam (for real-time detection)
- ~200MB disk space for models

## 🛠️ Tech Stack

- **Face Detection**: YOLOv8, OpenCV Haar Cascade
- **Emotion Recognition**: FER (Facial Expression Recognition)
- **Deep Learning**: TensorFlow/Keras
- **Web Framework**: Flask
- **Computer Vision**: OpenCV

## 📄 License

MIT License - feel free to use for personal and commercial projects.

## 🙏 Acknowledgments

- [FER Library](https://github.com/justinshenk/fer) for emotion recognition
- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- OpenCV community

---

Made with ❤️ for emotion AI research
