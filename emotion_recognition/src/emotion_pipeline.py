"""
Emotion Recognition Pipeline
Combines YOLO face detection with DeepFace emotion classification.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .face_detector import FaceDetector
from .emotion_classifier import EmotionClassifier


@dataclass
class EmotionResult:
    """Result from emotion recognition pipeline."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    emotion: str
    confidence: float
    all_scores: Dict[str, float]
    face_confidence: float


class EmotionRecognitionPipeline:
    """
    Complete emotion recognition pipeline.
    Combines face detection and emotion classification.
    """
    
    def __init__(
        self,
        face_confidence: float = 0.5,
        emotion_confidence: float = 0.3,
        use_fer_detection: bool = True
    ):
        """
        Initialize the emotion recognition pipeline.
        
        Args:
            face_confidence: Minimum confidence for face detection.
            emotion_confidence: Minimum confidence for emotion classification.
            use_fer_detection: Use DeepFace's built-in face detection (more reliable).
        """
        self.face_confidence = face_confidence
        self.emotion_confidence = emotion_confidence
        self.use_fer_detection = use_fer_detection
        
        # Initialize components
        if not use_fer_detection:
            self.face_detector = FaceDetector(confidence_threshold=face_confidence)
        self.emotion_classifier = EmotionClassifier(mtcnn=use_fer_detection)
        
        print("[Pipeline] Emotion Recognition Pipeline initialized")
    
    def process_image(self, image: np.ndarray) -> List[EmotionResult]:
        """
        Process a single image through the pipeline.
        
        Args:
            image: Input image (BGR format).
            
        Returns:
            List of EmotionResult objects for each detected face.
        """
        if image is None:
            return []
        
        results = []
        
        if self.use_fer_detection:
            # Use DeepFace's built-in detection
            fer_results = self.emotion_classifier.analyze_full_image(image)
            
            for r in fer_results:
                if r['confidence'] >= self.emotion_confidence:
                    results.append(EmotionResult(
                        bbox=r['bbox'],
                        emotion=r['emotion'],
                        confidence=r['confidence'],
                        all_scores=r['scores'],
                        face_confidence=1.0
                    ))
        else:
            # Use separate YOLO detection + emotion classification
            faces = self.face_detector.detect_and_crop(image)
            
            for face_img, det in faces:
                emotion_result = self.emotion_classifier.predict(face_img)
                
                if emotion_result and emotion_result['confidence'] >= self.emotion_confidence:
                    results.append(EmotionResult(
                        bbox=det['bbox'],
                        emotion=emotion_result['emotion'],
                        confidence=emotion_result['confidence'],
                        all_scores=emotion_result['scores'],
                        face_confidence=det['confidence']
                    ))
        
        return results
    
    def process_frame(self, frame: np.ndarray) -> List[EmotionResult]:
        """
        Process a video frame (optimized for real-time).
        
        Args:
            frame: Video frame (BGR format).
            
        Returns:
            List of EmotionResult objects.
        """
        # For now, same as process_image
        # Can be optimized with tracking, skip frames, etc.
        return self.process_image(frame)
    
    def annotate_image(
        self,
        image: np.ndarray,
        results: List[EmotionResult],
        show_scores: bool = True,
        show_bar: bool = True
    ) -> np.ndarray:
        """
        Annotate image with emotion detection results.
        
        Args:
            image: Input image.
            results: List of EmotionResult objects.
            show_scores: Show confidence scores.
            show_bar: Show emotion probability bar.
            
        Returns:
            Annotated image.
        """
        annotated = image.copy()
        
        for i, result in enumerate(results):
            x1, y1, x2, y2 = result.bbox
            color = self.emotion_classifier.get_emotion_color(result.emotion)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            if show_scores:
                label = f"{result.emotion.upper()} {result.confidence:.0%}"
            else:
                label = result.emotion.upper()
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            cv2.rectangle(
                annotated,
                (x1, y1 - text_h - 10),
                (x1 + text_w + 10, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 5, y1 - 5),
                font,
                font_scale,
                (255, 255, 255),
                thickness
            )
            
            # Draw emotion bar for first face
            if show_bar and i == 0:
                annotated = self.emotion_classifier.draw_emotion_bar(
                    annotated,
                    result.all_scores,
                    position=(10, 10)
                )
        
        return annotated
    
    def get_dominant_emotion(self, results: List[EmotionResult]) -> Optional[str]:
        """
        Get the dominant emotion from results.
        
        Args:
            results: List of EmotionResult objects.
            
        Returns:
            Most confident emotion across all faces, or None.
        """
        if not results:
            return None
        
        # Find result with highest confidence
        best_result = max(results, key=lambda r: r.confidence)
        return best_result.emotion
    
    def get_emotion_summary(self, results: List[EmotionResult]) -> Dict[str, int]:
        """
        Get summary of emotions across all faces.
        
        Args:
            results: List of EmotionResult objects.
            
        Returns:
            Dictionary with emotion counts.
        """
        summary = {emotion: 0 for emotion in EmotionClassifier.EMOTIONS}
        
        for result in results:
            if result.emotion in summary:
                summary[result.emotion] += 1
        
        return summary


def create_pipeline(
    face_confidence: float = 0.5,
    emotion_confidence: float = 0.3,
    use_fer_detection: bool = True
) -> EmotionRecognitionPipeline:
    """
    Factory function to create emotion recognition pipeline.
    
    Args:
        face_confidence: Minimum face detection confidence.
        emotion_confidence: Minimum emotion classification confidence.
        use_fer_detection: Use DeepFace's built-in face detection.
        
    Returns:
        Configured EmotionRecognitionPipeline.
    """
    return EmotionRecognitionPipeline(
        face_confidence=face_confidence,
        emotion_confidence=emotion_confidence,
        use_fer_detection=use_fer_detection
    )


if __name__ == "__main__":
    # Test the pipeline
    import sys
    
    pipeline = create_pipeline()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to save screenshot")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every frame
        results = pipeline.process_frame(frame)
        
        # Annotate
        annotated = pipeline.annotate_image(frame, results)
        
        # Add FPS info
        cv2.putText(
            annotated,
            f"Faces: {len(results)}",
            (annotated.shape[1] - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        cv2.imshow("Emotion Recognition Pipeline", annotated)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"screenshot_{frame_count}.jpg", annotated)
            print(f"Saved screenshot_{frame_count}.jpg")
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
