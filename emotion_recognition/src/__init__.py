"""Emotion Recognition Package - YOLO Face Detection + FER Emotion Classification"""

from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier
from .emotion_pipeline import EmotionRecognitionPipeline

__version__ = "1.0.0"
__all__ = ["FaceDetector", "EmotionClassifier", "EmotionRecognitionPipeline"]
