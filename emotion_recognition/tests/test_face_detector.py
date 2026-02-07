"""Tests for Face Detector module."""

import sys
from pathlib import Path
import pytest
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.face_detector import FaceDetector


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    @pytest.fixture
    def detector(self):
        """Create a face detector instance."""
        return FaceDetector(confidence_threshold=0.5)
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample blank image."""
        # Create a 640x480 blank image
        return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def test_init(self, detector):
        """Test detector initialization."""
        assert detector is not None
        assert detector.confidence_threshold == 0.5
    
    def test_detect_empty_image(self, detector, sample_image):
        """Test detection on blank image returns no faces."""
        detections = detector.detect_with_cascade(sample_image)
        assert isinstance(detections, list)
        # Blank image should have no faces
        assert len(detections) == 0
    
    def test_detect_none_image(self, detector):
        """Test detection handles None image gracefully."""
        detections = detector.detect_with_cascade(None)
        assert detections == []
    
    def test_detection_format(self, detector):
        """Test detection dictionary format."""
        # Create image with potential face-like region
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        detections = detector.detect_with_cascade(img)
        
        # Even if no faces, format should be correct
        assert isinstance(detections, list)
        
        for det in detections:
            assert 'bbox' in det
            assert 'confidence' in det
            assert 'center' in det
            assert len(det['bbox']) == 4
            assert len(det['center']) == 2
    
    def test_draw_detections(self, detector, sample_image):
        """Test drawing detections on image."""
        # Create fake detections
        detections = [
            {'bbox': (100, 100, 200, 200), 'confidence': 0.9, 'center': (150, 150)}
        ]
        
        result = detector.draw_detections(sample_image, detections)
        
        assert result is not None
        assert result.shape == sample_image.shape
        # Result should be different from original (box drawn)
        assert not np.array_equal(result, sample_image)
    
    def test_draw_with_labels(self, detector, sample_image):
        """Test drawing detections with labels."""
        detections = [
            {'bbox': (100, 100, 200, 200), 'confidence': 0.9, 'center': (150, 150)}
        ]
        labels = ["Face 1"]
        
        result = detector.draw_detections(sample_image, detections, labels=labels)
        
        assert result is not None
        assert result.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
