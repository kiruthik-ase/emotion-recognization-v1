"""
Flask Web Application for Emotion Recognition
Provides web interface for image upload and webcam emotion detection.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
import base64

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from flask import Flask, render_template, request, jsonify, Response
import cv2
import numpy as np

from src.emotion_pipeline import create_pipeline
from src.utils import base64_to_image, image_to_base64

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Initialize pipeline (lazy loading)
_pipeline = None


def get_pipeline():
    """Get or create emotion recognition pipeline."""
    global _pipeline
    if _pipeline is None:
        print("Loading emotion recognition models...")
        _pipeline = create_pipeline(
            face_confidence=0.5,
            emotion_confidence=0.3,
            use_fer_detection=True
        )
        print("Models loaded successfully!")
    return _pipeline


# Video streaming generator
def generate_frames():
    """Generate video frames for streaming."""
    camera = cv2.VideoCapture(0)
    pipeline = get_pipeline()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Mirror frame
        frame = cv2.flip(frame, 1)
        
        # Process frame
        results = pipeline.process_frame(frame)
        
        # Annotate
        annotated = pipeline.annotate_image(frame, results, show_bar=True)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', annotated)
        if not ret:
            continue
        
        # Yield frame
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    camera.release()


@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_image():
    """Analyze uploaded image for emotions."""
    try:
        # Get image data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image = base64_to_image(data['image'])
        
        if image is None:
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Process image
        pipeline = get_pipeline()
        results = pipeline.process_image(image)
        
        # Annotate image
        annotated = pipeline.annotate_image(image, results, show_bar=True)
        
        # Convert annotated image to base64
        annotated_b64 = image_to_base64(annotated)
        
        # Prepare response
        emotions_data = []
        for i, result in enumerate(results):
            emotions_data.append({
                'face_id': i + 1,
                'emotion': result.emotion,
                'confidence': round(result.confidence * 100, 1),
                'emoji': pipeline.emotion_classifier.get_emotion_emoji(result.emotion),
                'all_scores': {k: round(v * 100, 1) for k, v in result.all_scores.items()}
            })
        
        return jsonify({
            'success': True,
            'annotated_image': f'data:image/jpeg;base64,{annotated_b64}',
            'faces_count': len(results),
            'emotions': emotions_data
        })
    
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


def main():
    """Run the Flask application."""
    print("=" * 50)
    print("  EMOTION RECOGNITION - WEB APPLICATION")
    print("=" * 50)
    print("\nStarting server...")
    print("Open http://localhost:5000 in your browser")
    print("-" * 50)
    
    # Pre-load models
    get_pipeline()
    
    # Run app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == '__main__':
    main()
