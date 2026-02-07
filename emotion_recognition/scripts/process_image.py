#!/usr/bin/env python3
"""
Process a single image for emotion recognition.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    """Process single image."""
    parser = argparse.ArgumentParser(
        description="Detect emotions in an image"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output image path (default: input_emotions.jpg)"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display the result"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum emotion confidence (0-1)"
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
        output_path = input_path.parent / f"{input_path.stem}_emotions{input_path.suffix}"
    
    print("=" * 50)
    print("  EMOTION RECOGNITION - IMAGE PROCESSING")
    print("=" * 50)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print("\nLoading models...")
    
    # Import after argument parsing for faster --help
    import cv2
    from src.emotion_pipeline import create_pipeline
    from src.utils import load_image, save_image
    
    # Create pipeline
    pipeline = create_pipeline(
        emotion_confidence=args.confidence,
        use_fer_detection=True
    )
    
    # Load image
    print("Processing image...")
    image = load_image(str(input_path))
    
    if image is None:
        print(f"ERROR: Could not load image: {input_path}")
        return 1
    
    # Process
    results = pipeline.process_image(image)
    
    # Annotate
    annotated = pipeline.annotate_image(image, results)
    
    # Print results
    print("\n" + "-" * 50)
    print(f"Detected {len(results)} face(s)")
    
    for i, result in enumerate(results):
        emoji = pipeline.emotion_classifier.get_emotion_emoji(result.emotion)
        print(f"\nFace {i + 1}:")
        print(f"  Emotion: {result.emotion.upper()} {emoji}")
        print(f"  Confidence: {result.confidence:.1%}")
        print("  All scores:")
        for emotion, score in sorted(result.all_scores.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 20)
            print(f"    {emotion:10s} {bar} {score:.2f}")
    
    print("-" * 50)
    
    # Save result
    save_image(annotated, str(output_path))
    print(f"\nResult saved to: {output_path}")
    
    # Display if requested
    if not args.no_display:
        print("\nPress any key to close...")
        cv2.imshow("Emotion Recognition Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
