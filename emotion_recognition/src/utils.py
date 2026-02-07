"""
Utility functions for emotion recognition project.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import base64


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load an image from file.
    
    Args:
        path: Path to image file.
        
    Returns:
        Image as numpy array (BGR) or None if failed.
    """
    try:
        image = cv2.imread(path)
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def save_image(image: np.ndarray, path: str) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image as numpy array.
        path: Output path.
        
    Returns:
        True if successful.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(path, image)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def resize_image(
    image: np.ndarray,
    max_size: int = 1280,
    keep_aspect: bool = True
) -> np.ndarray:
    """
    Resize image to maximum dimension.
    
    Args:
        image: Input image.
        max_size: Maximum dimension (width or height).
        keep_aspect: Maintain aspect ratio.
        
    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    
    if max(h, w) <= max_size:
        return image
    
    if keep_aspect:
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)
    else:
        new_w = new_h = max_size
    
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def image_to_base64(image: np.ndarray) -> str:
    """
    Convert image to base64 string for web display.
    
    Args:
        image: Image as numpy array (BGR).
        
    Returns:
        Base64 encoded string.
    """
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def base64_to_image(base64_str: str) -> Optional[np.ndarray]:
    """
    Convert base64 string to image.
    
    Args:
        base64_str: Base64 encoded image.
        
    Returns:
        Image as numpy array.
    """
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None


def get_video_info(path: str) -> Optional[dict]:
    """
    Get video file information.
    
    Args:
        path: Path to video file.
        
    Returns:
        Dictionary with video info or None.
    """
    try:
        cap = cv2.VideoCapture(path)
        
        info = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / cap.get(cv2.CAP_PROP_FPS)
        }
        
        cap.release()
        return info
    except Exception as e:
        print(f"Error getting video info: {e}")
        return None


def create_video_writer(
    path: str,
    width: int,
    height: int,
    fps: float = 30.0,
    codec: str = 'mp4v'
) -> cv2.VideoWriter:
    """
    Create video writer for output.
    
    Args:
        path: Output path.
        width: Video width.
        height: Video height.
        fps: Frames per second.
        codec: Video codec (mp4v, avc1, etc.).
        
    Returns:
        VideoWriter object.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*codec)
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def draw_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.7,
    font_color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = 2,
    padding: int = 5
) -> np.ndarray:
    """
    Draw text with background rectangle.
    
    Args:
        image: Image to draw on.
        text: Text string.
        position: (x, y) position for text.
        font_scale: Font scale.
        font_color: Text color (BGR).
        bg_color: Background color (BGR).
        thickness: Text thickness.
        padding: Padding around text.
        
    Returns:
        Image with text drawn.
    """
    result = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = position
    
    # Draw background
    cv2.rectangle(
        result,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + padding),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(result, text, (x, y), font, font_scale, font_color, thickness)
    
    return result


def calculate_fps(start_time: float, frame_count: int) -> float:
    """
    Calculate FPS from start time and frame count.
    
    Args:
        start_time: Start time in seconds.
        frame_count: Number of frames processed.
        
    Returns:
        Frames per second.
    """
    import time
    elapsed = time.time() - start_time
    return frame_count / elapsed if elapsed > 0 else 0


class FPSCounter:
    """Helper class for FPS calculation."""
    
    def __init__(self, avg_frames: int = 30):
        """
        Initialize FPS counter.
        
        Args:
            avg_frames: Number of frames to average.
        """
        import time
        self.avg_frames = avg_frames
        self.times: List[float] = []
        self.start_time = time.time()
    
    def update(self) -> float:
        """
        Update counter and return current FPS.
        
        Returns:
            Current FPS.
        """
        import time
        current_time = time.time()
        self.times.append(current_time)
        
        # Keep only recent frames
        if len(self.times) > self.avg_frames:
            self.times = self.times[-self.avg_frames:]
        
        if len(self.times) < 2:
            return 0.0
        
        elapsed = self.times[-1] - self.times[0]
        return (len(self.times) - 1) / elapsed if elapsed > 0 else 0.0
