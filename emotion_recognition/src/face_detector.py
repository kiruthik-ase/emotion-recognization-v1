"""
Face Detection Module using YOLOv8
Provides fast and accurate face detection for emotion recognition pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from ultralytics import YOLO


class FaceDetector:
    """YOLO-based face detector for emotion recognition."""
    
    # Colors for visualization (BGR format)
    BOX_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255)  # White
    TEXT_BG_COLOR = (0, 255, 0)  # Green background
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        confidence_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize the face detector.
        
        Args:
            model_path: Path to YOLO model weights. Uses yolov8n-face if None.
            confidence_threshold: Minimum confidence for detections (0-1).
            device: Device to run inference on ('cpu', 'cuda', or 'auto').
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        # Load YOLO model
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
        else:
            # Use YOLOv8 nano model - will be fine-tuned for face detection
            # For pure face detection, we use the standard model and filter for person/face
            self.model = YOLO("yolov8n.pt")
        
        print(f"[FaceDetector] Model loaded successfully")
    
    def detect(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format).
            
        Returns:
            List of detection dictionaries with keys:
                - 'bbox': (x1, y1, x2, y2) bounding box coordinates
                - 'confidence': Detection confidence score
                - 'center': (cx, cy) center point of the face
        """
        if image is None:
            return []
        
        # Run YOLO inference
        results = self.model(
            image, 
            conf=self.confidence_threshold,
            verbose=False,
            device=self.device if self.device != "auto" else None
        )
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for i, box in enumerate(boxes):
                # Get class - we want person class (0) for face region
                cls = int(box.cls[0])
                
                # Filter for person class (we'll crop upper body for face)
                if cls == 0:  # person class
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # Estimate face region (upper 1/3 of person bounding box)
                    height = y2 - y1
                    face_y2 = y1 + height * 0.35
                    
                    # Adjust width to be more square for face
                    width = x2 - x1
                    center_x = (x1 + x2) / 2
                    face_width = min(width * 0.8, face_y2 - y1)
                    face_x1 = center_x - face_width / 2
                    face_x2 = center_x + face_width / 2
                    
                    detections.append({
                        'bbox': (int(face_x1), int(y1), int(face_x2), int(face_y2)),
                        'confidence': conf,
                        'center': (int(center_x), int((y1 + face_y2) / 2))
                    })
        
        return detections
    
    def detect_with_cascade(self, image: np.ndarray) -> List[dict]:
        """
        Alternative face detection using OpenCV Haar Cascade.
        More reliable for pure face detection.
        
        Args:
            image: Input image as numpy array (BGR format).
            
        Returns:
            List of detection dictionaries.
        """
        if image is None:
            return []
        
        # Load cascade classifier
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                'bbox': (x, y, x + w, y + h),
                'confidence': 1.0,  # Cascade doesn't provide confidence
                'center': (x + w // 2, y + h // 2)
            })
        
        return detections
    
    def detect_and_crop(
        self, 
        image: np.ndarray, 
        padding: float = 0.2,
        use_cascade: bool = True
    ) -> List[Tuple[np.ndarray, dict]]:
        """
        Detect faces and return cropped face images.
        
        Args:
            image: Input image as numpy array (BGR format).
            padding: Padding around face as fraction of face size.
            use_cascade: Use Haar Cascade for more reliable face detection.
            
        Returns:
            List of tuples (cropped_face_image, detection_info).
        """
        if use_cascade:
            detections = self.detect_with_cascade(image)
        else:
            detections = self.detect(image)
        
        h, w = image.shape[:2]
        cropped_faces = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Add padding
            face_w = x2 - x1
            face_h = y2 - y1
            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)
            
            # Ensure within image bounds
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            # Crop face
            face_img = image[y1:y2, x1:x2].copy()
            
            if face_img.size > 0:
                # Update bbox in detection info
                det['bbox'] = (x1, y1, x2, y2)
                cropped_faces.append((face_img, det))
        
        return cropped_faces
    
    def draw_detections(
        self, 
        image: np.ndarray, 
        detections: List[dict],
        labels: Optional[List[str]] = None,
        colors: Optional[List[Tuple[int, int, int]]] = None
    ) -> np.ndarray:
        """
        Draw detection boxes on image.
        
        Args:
            image: Input image to draw on.
            detections: List of detection dictionaries.
            labels: Optional labels for each detection.
            colors: Optional colors for each detection (BGR).
            
        Returns:
            Image with drawn detections.
        """
        result = image.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            
            # Get color
            color = colors[i] if colors and i < len(colors) else self.BOX_COLOR
            
            # Draw bounding box
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label if provided
            if labels and i < len(labels):
                label = labels[i]
                
                # Get text size
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )
                
                # Draw background rectangle
                cv2.rectangle(
                    result,
                    (x1, y1 - text_h - 10),
                    (x1 + text_w + 10, y1),
                    color,
                    -1
                )
                
                # Draw text
                cv2.putText(
                    result,
                    label,
                    (x1 + 5, y1 - 5),
                    font,
                    font_scale,
                    self.TEXT_COLOR,
                    thickness
                )
        
        return result


if __name__ == "__main__":
    # Test the face detector
    import sys
    
    detector = FaceDetector()
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        detections = detector.detect_with_cascade(frame)
        
        # Draw detections
        result = detector.draw_detections(
            frame, 
            detections,
            labels=[f"Face {i+1}" for i in range(len(detections))]
        )
        
        cv2.imshow("Face Detection", result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
