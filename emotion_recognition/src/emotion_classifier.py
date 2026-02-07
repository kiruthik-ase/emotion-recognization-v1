"""
Emotion Classification Module using DeepFace
Classifies facial expressions into 7 emotion categories.
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional


class EmotionClassifier:
    """DeepFace-based emotion classifier for facial expression recognition."""
    
    # Emotion categories
    EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    
    # Emoji representations
    EMOTION_EMOJIS = {
        'angry': '😠',
        'disgust': '🤢',
        'fear': '😨',
        'happy': '😊',
        'sad': '😢',
        'surprise': '😲',
        'neutral': '😐'
    }
    
    # Colors for each emotion (BGR format for OpenCV)
    EMOTION_COLORS = {
        'angry': (0, 0, 255),      # Red
        'disgust': (0, 128, 0),    # Dark Green
        'fear': (128, 0, 128),     # Purple
        'happy': (0, 255, 255),    # Yellow
        'sad': (255, 0, 0),        # Blue
        'surprise': (0, 165, 255), # Orange
        'neutral': (128, 128, 128) # Gray
    }
    
    def __init__(self, mtcnn: bool = False):
        """
        Initialize the emotion classifier.
        
        Args:
            mtcnn: Whether to use MTCNN for face detection (ignored in DeepFace mode).
        """
        # DeepFace will be imported on first use to avoid slow startup
        self._deepface = None
        self._face_cascade = None
        print("[EmotionClassifier] Initialized (DeepFace backend)")
    
    def _get_deepface(self):
        """Lazy load DeepFace."""
        if self._deepface is None:
            from deepface import DeepFace
            self._deepface = DeepFace
        return self._deepface
    
    def _get_face_cascade(self):
        """Get OpenCV face cascade for detection."""
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade
    
    def predict(self, face_image: np.ndarray) -> Optional[Dict]:
        """
        Predict emotion for a face image.
        
        Args:
            face_image: Cropped face image (BGR format).
            
        Returns:
            Dictionary with:
                - 'emotion': Top predicted emotion
                - 'confidence': Confidence score (0-1)
                - 'scores': Dict of all emotion scores
            Returns None if no face/emotion detected.
        """
        if face_image is None or face_image.size == 0:
            return None
        
        try:
            DeepFace = self._get_deepface()
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                face_image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(result, list):
                result = result[0]
            
            emotions = result.get('emotion', {})
            
            if not emotions:
                return None
            
            # Normalize scores to 0-1 range
            total = sum(emotions.values())
            if total > 0:
                emotions = {k: v / 100 for k, v in emotions.items()}
            
            # Find top emotion
            top_emotion = max(emotions, key=emotions.get)
            top_confidence = emotions[top_emotion]
            
            return {
                'emotion': top_emotion,
                'confidence': top_confidence,
                'scores': emotions
            }
            
        except Exception as e:
            print(f"[EmotionClassifier] Error: {e}")
            return None
    
    def predict_top_k(
        self, 
        face_image: np.ndarray, 
        k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Get top-k emotions with scores.
        
        Args:
            face_image: Cropped face image (BGR format).
            k: Number of top emotions to return.
            
        Returns:
            List of (emotion, score) tuples, sorted by score descending.
        """
        result = self.predict(face_image)
        
        if result is None:
            return []
        
        scores = result['scores']
        sorted_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_emotions[:k]
    
    def get_emotion_color(self, emotion: str) -> Tuple[int, int, int]:
        """
        Get BGR color for an emotion.
        
        Args:
            emotion: Emotion name.
            
        Returns:
            BGR color tuple.
        """
        return self.EMOTION_COLORS.get(emotion.lower(), (128, 128, 128))
    
    def get_emotion_emoji(self, emotion: str) -> str:
        """
        Get emoji representation for an emotion.
        
        Args:
            emotion: Emotion name.
            
        Returns:
            Emoji string.
        """
        return self.EMOTION_EMOJIS.get(emotion.lower(), '❓')
    
    def analyze_full_image(self, image: np.ndarray) -> List[Dict]:
        """
        Analyze full image and detect all faces with emotions.
        
        Args:
            image: Full image (BGR format).
            
        Returns:
            List of dictionaries with face locations and emotions.
        """
        if image is None:
            return []
        
        try:
            DeepFace = self._get_deepface()
            
            # Use DeepFace to analyze the entire image
            results = DeepFace.analyze(
                image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            
            if not isinstance(results, list):
                results = [results]
            
            processed_results = []
            for r in results:
                # Get face region
                region = r.get('region', {})
                x = region.get('x', 0)
                y = region.get('y', 0)
                w = region.get('w', 100)
                h = region.get('h', 100)
                
                emotions = r.get('emotion', {})
                
                if not emotions:
                    continue
                
                # Normalize to 0-1
                emotions = {k: v / 100 for k, v in emotions.items()}
                
                top_emotion = max(emotions, key=emotions.get)
                
                processed_results.append({
                    'bbox': (x, y, x + w, y + h),
                    'emotion': top_emotion,
                    'confidence': emotions[top_emotion],
                    'scores': emotions
                })
            
            return processed_results
            
        except Exception as e:
            print(f"[EmotionClassifier] Error in full image analysis: {e}")
            # Fallback: use Haar cascade for face detection
            return self._analyze_with_cascade(image)
    
    def _analyze_with_cascade(self, image: np.ndarray) -> List[Dict]:
        """Fallback analysis using Haar Cascade."""
        face_cascade = self._get_face_cascade()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        results = []
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            emotion_result = self.predict(face_img)
            
            if emotion_result:
                results.append({
                    'bbox': (x, y, x + w, y + h),
                    'emotion': emotion_result['emotion'],
                    'confidence': emotion_result['confidence'],
                    'scores': emotion_result['scores']
                })
        
        return results
    
    def draw_emotion_bar(
        self, 
        image: np.ndarray, 
        emotions: Dict[str, float],
        position: Tuple[int, int] = (10, 10),
        bar_width: int = 150,
        bar_height: int = 15
    ) -> np.ndarray:
        """
        Draw emotion probability bar chart on image.
        
        Args:
            image: Image to draw on.
            emotions: Dictionary of emotion scores.
            position: Top-left position for the chart.
            bar_width: Maximum width of bars.
            bar_height: Height of each bar.
            
        Returns:
            Image with emotion bars drawn.
        """
        result = image.copy()
        x, y = position
        
        # Sort emotions by score
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        for i, (emotion, score) in enumerate(sorted_emotions):
            bar_y = y + i * (bar_height + 5)
            
            # Draw background bar
            cv2.rectangle(
                result,
                (x, bar_y),
                (x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Draw filled bar
            fill_width = int(bar_width * score)
            color = self.get_emotion_color(emotion)
            cv2.rectangle(
                result,
                (x, bar_y),
                (x + fill_width, bar_y + bar_height),
                color,
                -1
            )
            
            # Draw label
            label = f"{emotion}: {score:.2f}"
            cv2.putText(
                result,
                label,
                (x + bar_width + 10, bar_y + bar_height - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )
        
        return result


if __name__ == "__main__":
    # Test the emotion classifier
    import sys
    
    classifier = EmotionClassifier(mtcnn=True)
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Analyze frame
        results = classifier.analyze_full_image(frame)
        
        # Draw results
        for r in results:
            x1, y1, x2, y2 = r['bbox']
            emotion = r['emotion']
            confidence = r['confidence']
            color = classifier.get_emotion_color(emotion)
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{emotion} {confidence:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        
        # Draw emotion bar for first face
        if results:
            frame = classifier.draw_emotion_bar(frame, results[0]['scores'])
        
        cv2.imshow("Emotion Classification", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
