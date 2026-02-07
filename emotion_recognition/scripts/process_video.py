#!/usr/bin/env python3
"""
Process a video file for emotion recognition.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Process video file."""
    parser = argparse.ArgumentParser(
        description="Detect emotions in a video file"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input video path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output video path (default: input_emotions.mp4)"
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1 = all frames)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum emotion confidence (0-1)"
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show preview while processing"
    )
    
    args = parser.parse_args()
    
    # Check input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    # Set output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_emotions.mp4"
    
    print("=" * 50)
    print("  EMOTION RECOGNITION - VIDEO PROCESSING")
    print("=" * 50)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Skip frames: {args.skip_frames}")
    print("\nLoading models...")
    
    # Import after argument parsing
    import cv2
    from src.emotion_pipeline import create_pipeline
    from src.utils import get_video_info, create_video_writer, FPSCounter
    
    # Create pipeline
    pipeline = create_pipeline(
        emotion_confidence=args.confidence,
        use_fer_detection=True
    )
    
    # Get video info
    video_info = get_video_info(str(input_path))
    if video_info is None:
        print(f"ERROR: Could not read video: {input_path}")
        return 1
    
    print(f"\nVideo info:")
    print(f"  Resolution: {video_info['width']}x{video_info['height']}")
    print(f"  FPS: {video_info['fps']:.2f}")
    print(f"  Duration: {video_info['duration']:.1f}s")
    print(f"  Frames: {video_info['frame_count']}")
    
    # Open input video
    cap = cv2.VideoCapture(str(input_path))
    
    # Create output video writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = create_video_writer(
        str(output_path),
        video_info['width'],
        video_info['height'],
        video_info['fps']
    )
    
    # Process frames
    print("\nProcessing...")
    
    frame_count = 0
    processed_count = 0
    last_results = []
    fps_counter = FPSCounter()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every Nth frame
            if frame_count % args.skip_frames == 0:
                results = pipeline.process_frame(frame)
                last_results = results
                processed_count += 1
            
            # Annotate with last results
            annotated = pipeline.annotate_image(frame, last_results)
            
            # Add progress info
            progress = frame_count / video_info['frame_count'] * 100
            fps = fps_counter.update()
            
            cv2.putText(
                annotated,
                f"Frame: {frame_count}/{video_info['frame_count']} ({progress:.1f}%)",
                (10, annotated.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            
            # Write frame
            writer.write(annotated)
            
            # Show preview
            if args.preview:
                cv2.imshow("Processing Preview (Q to quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing cancelled by user")
                    break
            
            # Print progress
            if frame_count % 30 == 0:
                print(f"  Progress: {progress:.1f}% ({frame_count}/{video_info['frame_count']}) - FPS: {fps:.1f}")
    
    finally:
        cap.release()
        writer.release()
        if args.preview:
            cv2.destroyAllWindows()
    
    print("\n" + "-" * 50)
    print(f"Processing complete!")
    print(f"  Frames processed: {processed_count}")
    print(f"  Output saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
