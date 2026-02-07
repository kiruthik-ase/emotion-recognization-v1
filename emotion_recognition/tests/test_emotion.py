"""Tests for Emotion Classifier and Pipeline modules."""

import sys
from pathlib import Path
import pytest
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEmotionClassifier:
    """Test cases for EmotionClassifier class."""
    
    @pytest.fixture
    def classifier(self):
        """Create an emotion classifier instance."""
        from src.emotion_classifier import EmotionClassifier
        return EmotionClassifier(mtcnn=False)
    
    @pytest.fixture
    def sample_face(self):
        """Create a sample face-like image."""
        # Create a 48x48 grayscale-ish image (typical FER input size)
        return np.random.randint(100, 200, (48, 48, 3), dtype=np.uint8)
    
    def test_init(self, classifier):
        """Test classifier initialization."""
        assert classifier is not None
        assert hasattr(classifier, 'EMOTIONS')
        assert len(classifier.EMOTIONS) == 7
    
    def test_emotions_list(self, classifier):
        """Test emotion categories."""
        expected = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        assert classifier.EMOTIONS == expected
    
    def test_get_emotion_color(self, classifier):
        """Test emotion color mapping."""
        for emotion in classifier.EMOTIONS:
            color = classifier.get_emotion_color(emotion)
            assert isinstance(color, tuple)
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)
    
    def test_get_emotion_emoji(self, classifier):
        """Test emotion emoji mapping."""
        for emotion in classifier.EMOTIONS:
            emoji = classifier.get_emotion_emoji(emotion)
            assert isinstance(emoji, str)
            assert len(emoji) > 0
    
    def test_predict_empty(self, classifier):
        """Test prediction on empty image."""
        result = classifier.predict(np.array([]))
        assert result is None
    
    def test_predict_none(self, classifier):
        """Test prediction on None."""
        result = classifier.predict(None)
        assert result is None


class TestEmotionPipeline:
    """Test cases for EmotionRecognitionPipeline."""
    
    @pytest.fixture
    def pipeline(self):
        """Create a pipeline instance."""
        from src.emotion_pipeline import create_pipeline
        return create_pipeline(
            face_confidence=0.5,
            emotion_confidence=0.3,
            use_fer_detection=True
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image."""
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    def test_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert hasattr(pipeline, 'emotion_classifier')
    
    def test_process_none(self, pipeline):
        """Test processing None image."""
        results = pipeline.process_image(None)
        assert results == []
    
    def test_process_returns_list(self, pipeline, sample_image):
        """Test process returns list."""
        results = pipeline.process_image(sample_image)
        assert isinstance(results, list)
    
    def test_annotate_image(self, pipeline, sample_image):
        """Test image annotation."""
        results = []  # Empty results
        annotated = pipeline.annotate_image(sample_image, results)
        
        assert annotated is not None
        assert annotated.shape == sample_image.shape
    
    def test_get_dominant_emotion_empty(self, pipeline):
        """Test dominant emotion with empty results."""
        result = pipeline.get_dominant_emotion([])
        assert result is None
    
    def test_get_emotion_summary(self, pipeline):
        """Test emotion summary."""
        summary = pipeline.get_emotion_summary([])
        
        assert isinstance(summary, dict)
        assert len(summary) == 7  # 7 emotion classes
        assert all(v == 0 for v in summary.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
