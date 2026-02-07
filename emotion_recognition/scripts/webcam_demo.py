#!/usr/bin/env python3
"""
Real-time Webcam Emotion Detection Demo
Detects faces and recognizes emotions using your webcam.
"""

import sys
import os
import cv2
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.emotion_pipeline import create_pipeline
from src.utils import FPSCounter, draw_text_with_background


def main():
    """Run webcam emotion detection demo."""
    print("=" * 50)
    print("  EMOTION RECOGNITION - WEBCAM DEMO")
    print("=" * 50)
    print("\nInitializing models... (this may take a moment)")
    
    # Create pipeline
    pipeline = create_pipeline(
        face_confidence=0.5,
        emotion_confidence=0.3,
        use_fer_detection=True  # Use FER's MTCNN for better accuracy
    )
    
    print("\nModels loaded successfully!")
    print("\nControls:")
    print("  q - Quit")
    print("  s - Save screenshot")
    print("  b - Toggle emotion bar")
    print("  f - Toggle FPS display")
    print("-" * 50)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Please check if webcam is connected and not in use.")
        return 1
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS counter
    fps_counter = FPSCounter(avg_frames=30)
    
    # Display options
    show_bar = True
    show_fps = True
    screenshot_count = 0
    
    # Create output directory for screenshots
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\nWebcam opened. Starting detection...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam")
            break
        
        # Mirror the frame for more intuitive interaction
        frame = cv2.flip(frame, 1)
        
        # Process frame
        results = pipeline.process_frame(frame)
        
        # Annotate
        annotated = pipeline.annotate_image(frame, results, show_bar=show_bar)
        
        # Update FPS
        fps = fps_counter.update()
        
        # Draw info overlay
        info_y = annotated.shape[0] - 10
        
        if show_fps:
            annotated = draw_text_with_background(
                annotated,
                f"FPS: {fps:.1f}",
                (10, info_y),
                font_scale=0.6,
                bg_color=(0, 100, 0)
            )
        
        # Draw face count
        annotated = draw_text_with_background(
            annotated,
            f"Faces: {len(results)}",
            (annotated.shape[1] - 100, 30),
            font_scale=0.6,
            bg_color=(100, 0, 0)
        )
        
        # Show dominant emotion
        if results:
            dominant = pipeline.get_dominant_emotion(results)
            emoji = pipeline.emotion_classifier.get_emotion_emoji(dominant)
            annotated = draw_text_with_background(
                annotated,
                f"Dominant: {dominant.upper()}",
                (annotated.shape[1] - 180, 60),
                font_scale=0.6,
                bg_color=pipeline.emotion_classifier.get_emotion_color(dominant)
            )
        
        # Display
        cv2.imshow("Emotion Recognition - Press Q to Quit", annotated)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == ord('Q'):
            print("\nQuitting...")
            break
        
        elif key == ord('s') or key == ord('S'):
            # Save screenshot
            screenshot_path = output_dir / f"screenshot_{screenshot_count:04d}.jpg"
            cv2.imwrite(str(screenshot_path), annotated)
            print(f"Screenshot saved: {screenshot_path}")
            screenshot_count += 1
        
        elif key == ord('b') or key == ord('B'):
            show_bar = not show_bar
            print(f"Emotion bar: {'ON' if show_bar else 'OFF'}")
        
        elif key == ord('f') or key == ord('F'):
            show_fps = not show_fps
            print(f"FPS display: {'ON' if show_fps else 'OFF'}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print("\nDemo ended. Thank you!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
