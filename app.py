"""
Wildlife Injury Detection and Rescue Alert System
Main Flask Application
"""

import os
import sys
import json
import logging
import datetime
import uuid
import base64
from pathlib import Path
from flask import Flask, render_template, request, jsonify, Response, redirect, url_for
from werkzeug.utils import secure_filename
import requests


def running_on_vercel():
    return os.getenv("VERCEL") == "1" or bool(os.getenv("VERCEL_ENV"))


VERCEL_MODE = running_on_vercel()
INFERENCE_URL = os.getenv("INFERENCE_URL", "").strip()
INFERENCE_TOKEN = os.getenv("INFERENCE_TOKEN", "").strip()
DISABLE_VIDEO = os.getenv("DISABLE_VIDEO", "").strip() in {"1", "true", "True", "yes", "YES"}
DISABLE_LIVE_CAMERA = os.getenv("DISABLE_LIVE_CAMERA", "").strip() in {"1", "true", "True", "yes", "YES"}

# Optional heavy deps (not used on Vercel)
if not VERCEL_MODE:
    import numpy as np
    import cv2
    from PIL import Image
else:
    np = None
    cv2 = None
    Image = None

# Import custom modules
from modules.alert_generator import AlertGenerator
from utils.logger import setup_logger
from utils.helpers import allowed_file, create_directories

if not VERCEL_MODE:
    from modules.yolo_detector import YOLODetector
    from modules.slowfast_analyzer import SlowFastAnalyzer
    from modules.video_processor import VideoProcessor
    from modules.live_camera import LiveCameraManager
else:
    YOLODetector = None
    SlowFastAnalyzer = None
    VideoProcessor = None
    LiveCameraManager = None

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'wildlife-injury-detection-secret-key-2024'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Configure upload folders
BASE_DIR = Path(__file__).parent

if VERCEL_MODE:
    TMP_BASE = Path(os.getenv("VERCEL_TMP_DIR") or os.getenv("TMPDIR") or os.getenv("TEMP") or "/tmp")
    DATA_DIR = TMP_BASE / "wildlife_injury_detection"
    UPLOAD_FOLDER = DATA_DIR / "uploads"
    ANNOTATED_FOLDER = DATA_DIR / "annotated"
    EVIDENCE_FOLDER = DATA_DIR / "evidence"
    LOG_FOLDER = DATA_DIR / "logs"
else:
    UPLOAD_FOLDER = BASE_DIR / 'static' / 'uploads'
    ANNOTATED_FOLDER = BASE_DIR / 'static' / 'annotated'
    EVIDENCE_FOLDER = BASE_DIR / 'static' / 'evidence'
    LOG_FOLDER = BASE_DIR / 'logs'

# Create directories
create_directories([UPLOAD_FOLDER, ANNOTATED_FOLDER, EVIDENCE_FOLDER, LOG_FOLDER])

app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['ANNOTATED_FOLDER'] = str(ANNOTATED_FOLDER)
app.config['EVIDENCE_FOLDER'] = str(EVIDENCE_FOLDER)
app.config['LOG_FOLDER'] = str(LOG_FOLDER)

# Allowed extensions
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'flv'}

# Initialize logger
logger = setup_logger('WildlifeInjuryDetection', LOG_FOLDER)

# Initialize detectors and analyzers
if not VERCEL_MODE:
    try:
        yolo_detector = YOLODetector()
        logger.info("YOLOv8 detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize YOLOv8 detector: {e}")
        yolo_detector = None

    try:
        slowfast_analyzer = SlowFastAnalyzer()
        logger.info("SlowFast analyzer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize SlowFast analyzer: {e}")
        slowfast_analyzer = None
else:
    logger.info("VERCEL_MODE enabled: ML inference dependencies are disabled.")
    yolo_detector = None
    slowfast_analyzer = None

# Initialize alert generator
alert_generator = AlertGenerator(LOG_FOLDER)

# Initialize video processor
video_processor = None if VERCEL_MODE else VideoProcessor(
    yolo_detector, slowfast_analyzer, alert_generator, ANNOTATED_FOLDER, EVIDENCE_FOLDER
)

# Initialize live camera manager
live_camera_manager = None if VERCEL_MODE else LiveCameraManager(
    yolo_detector, slowfast_analyzer, alert_generator, EVIDENCE_FOLDER
)


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():
    """Dashboard with system status"""
    stats = {
        'total_alerts': alert_generator.get_alert_count(),
        'active_cameras': live_camera_manager.get_active_camera_count() if live_camera_manager else 0,
        'processed_images': 0,
        'processed_videos': 0
    }
    return render_template('dashboard.html', stats=stats)


@app.route('/image-upload', methods=['GET', 'POST'])
def image_upload():
    """Handle image upload and processing"""
    if request.method == 'POST':
        if VERCEL_MODE:
            if not INFERENCE_URL:
                return jsonify({
                    'error': 'Image inference is disabled on Vercel for this repo. Set INFERENCE_URL to a separate inference server (running this app with requirements-full.txt) to enable /image-upload.'
                }), 501

            if 'file' not in request.files:
                return jsonify({'error': 'No file provided'}), 400

            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400

            headers = {}
            if INFERENCE_TOKEN:
                headers['Authorization'] = f"Bearer {INFERENCE_TOKEN}"

            try:
                resp = requests.post(
                    INFERENCE_URL.rstrip('/') + '/image-upload',
                    files={'file': (file.filename, file.stream, file.mimetype or 'application/octet-stream')},
                    headers=headers,
                    timeout=120,
                )
                content_type = resp.headers.get('content-type', '')
                if 'application/json' in content_type:
                    return jsonify(resp.json()), resp.status_code
                return (resp.text, resp.status_code, {'Content-Type': content_type or 'text/plain'})
            except requests.RequestException as e:
                logger.error(f"INFERENCE_URL proxy error: {e}")
                return jsonify({'error': f'Inference proxy error: {str(e)}'}), 502

        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            filename_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            saved_filename = f"{filename_without_ext}_{unique_id}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(filepath)
            
            logger.info(f"Image uploaded: {saved_filename}")
            
            # Process image
            try:
                result = process_image(filepath, unique_id)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error processing image: {e}")
                return jsonify({'error': str(e)}), 500
    
    return render_template('image_upload.html')


def process_image(image_path, unique_id):
    """Process uploaded image for wildlife injury detection"""
    logger.info(f"Processing image: {image_path}")

    if cv2 is None or np is None:
        raise RuntimeError("Image processing dependencies are not installed.")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to load image")
    
    # Check if detector is available
    if yolo_detector is None:
        raise RuntimeError("YOLO detector not initialized")
    
    # Run YOLOv8 detection
    detections = yolo_detector.detect(image)
    logger.info(f"YOLOv8 detections: {detections}")
    
    # Filter animal detections using the detector's built-in filter
    animal_detections = yolo_detector.filter_animals(detections, min_confidence=0.60)
    
    results = {
        'image_id': unique_id,
        'detections': [],
        'annotated_image': None,
        'annotated_image_data_url': None,
        'alerts': []
    }
    
    # Process each detection
    for detection in animal_detections:
        x1, y1, x2, y2 = detection['bbox']
        
        # Extract animal ROI
        roi = image[y1:y2, x1:x2]
        
        # Analyze for injury (static analysis)
        injury_result = analyze_injury_static(roi, detection)
        
        detection_result = {
            'class': detection['class_name'],
            'confidence': float(detection['confidence']),
            'bbox': detection['bbox'],
            'injury_probability': injury_result['injury_probability'],
            'injury_indicators': injury_result['indicators']
        }
        
        results['detections'].append(detection_result)
        
        # Generate alert if injury probability >= 0.75
        if injury_result['injury_probability'] >= 0.75:
            alert = alert_generator.generate_alert(
                input_mode='image',
                camera_id=unique_id,
                animal_class=detection['class_name'],
                yolo_confidence=detection['confidence'],
                injury_probability=injury_result['injury_probability'],
                bbox=detection['bbox'],
                evidence_path=None
            )
            if alert:
                results['alerts'].append(alert)
            logger.warning(f"Injury detected! Probability: {injury_result['injury_probability']}")
        
        # Draw bounding box on image (colors: dark green for normal, black for injury)
        if injury_result['injury_probability'] >= 0.75:
            color = (0, 0, 0)  # Black
            label = f"INJURY: {detection['class_name']} {detection['confidence']:.2f} ({injury_result['injury_probability']:.2f})"
        else:
            color = (0, 100, 0)  # Dark green
            label = f"{detection['class_name']} {detection['confidence']:.2f}"
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save annotated image locally when possible, and always return an inline preview too.
    if not VERCEL_MODE:
        annotated_filename = f"annotated_{unique_id}.jpg"
        annotated_path = os.path.join(app.config['ANNOTATED_FOLDER'], annotated_filename)
        cv2.imwrite(annotated_path, image)
        results['annotated_image'] = annotated_filename

    ok, buffer = cv2.imencode(".jpg", image)
    if ok:
        results['annotated_image_data_url'] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
    
    return results


def analyze_injury_static(roi, detection):
    """Analyze static image for injury indicators"""
    injury_probability = 0.0
    indicators = []
    
    # Get ROI dimensions
    h, w = roi.shape[:2]
    
    if h == 0 or w == 0:
        return {'injury_probability': 0.0, 'indicators': []}
    
    # Analyze aspect ratio
    aspect_ratio = w / h if h > 0 else 1
    
    # Check for abnormal posture (collapsed body)
    if aspect_ratio > 2.0 or aspect_ratio < 0.5:
        injury_probability += 0.3
        indicators.append('Abnormal posture detected')
    
    # Check for small ROI (animal might be down)
    if h < 50 or w < 50:
        injury_probability += 0.2
        indicators.append('Possible body collapse')
    
    # Analyze color distribution for visible wounds
    try:
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Look for red/orange tones (possible wounds)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask_red = cv2.inRange(hsv_roi, lower_red, upper_red)
        red_ratio = np.sum(mask_red > 0) / (h * w)
        
        if red_ratio > 0.1:
            injury_probability += 0.25
            indicators.append('Possible visible wound')
        
        # Look for dark spots (possible injuries)
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_ratio = np.sum(gray_roi < 50) / (h * w)
        
        if dark_ratio > 0.15:
            injury_probability += 0.2
            indicators.append('Possible injury marks')
        
    except Exception as e:
        logger.warning(f"Error in color analysis: {e}")
    
    # Analyze symmetry
    try:
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Split ROI into left and right halves
        mid_x = w // 2
        left_half = gray_roi[:, :mid_x]
        right_half = cv2.flip(gray_roi[:, mid_x:], 1)
        
        # Compare histograms
        hist_left = cv2.calcHist([left_half], [0], None, [256], [0, 256])
        hist_right = cv2.calcHist([right_half], [0], None, [256], [0, 256])
        
        # Calculate similarity
        hist_diff = cv2.compareHist(hist_left, hist_right, cv2.HISTCMP_CORREL)
        
        if hist_diff < 0.7:
            injury_probability += 0.15
            indicators.append('Asymmetrical stance')
        
    except Exception as e:
        logger.warning(f"Error in symmetry analysis: {e}")
    
    # Cap probability at 1.0
    injury_probability = min(injury_probability, 1.0)
    
    return {
        'injury_probability': injury_probability,
        'indicators': indicators
    }


@app.route('/video-upload', methods=['GET', 'POST'])
def video_upload():
    """Handle video upload and processing"""
    if (VERCEL_MODE or DISABLE_VIDEO) and request.method == 'POST':
        return jsonify({
            'error': 'Video processing is disabled for this deployment.'
        }), 501

    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_id = str(uuid.uuid4())
            filename_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            saved_filename = f"{filename_without_ext}_{unique_id}{ext}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], saved_filename)
            file.save(filepath)
            
            logger.info(f"Video uploaded: {saved_filename}")
            
            # Process video in background
            try:
                result = video_processor.process_video(filepath, unique_id)
                return jsonify(result)
            except Exception as e:
                logger.error(f"Error processing video: {e}")
                return jsonify({'error': str(e)}), 500
    
    return render_template('video_upload.html')


@app.route('/live-camera', methods=['GET', 'POST'])
def live_camera():
    """Handle live camera stream processing"""
    if (VERCEL_MODE or DISABLE_LIVE_CAMERA) and request.method == 'POST':
        return jsonify({
            'error': 'Live camera processing is disabled for this deployment.'
        }), 501

    if request.method == 'POST':
        data = request.get_json()
        
        if 'action' in data:
            if data['action'] == 'start':
                camera_id = data.get('camera_id', f"camera_{uuid.uuid4().hex[:8]}")
                rtsp_url = data.get('rtsp_url', '')
                
                try:
                    result = live_camera_manager.start_camera(camera_id, rtsp_url)
                    return jsonify(result)
                except Exception as e:
                    logger.error(f"Error starting camera: {e}")
                    return jsonify({'error': str(e)}), 500
            
            elif data['action'] == 'stop':
                camera_id = data.get('camera_id')
                
                try:
                    result = live_camera_manager.stop_camera(camera_id)
                    return jsonify(result)
                except Exception as e:
                    logger.error(f"Error stopping camera: {e}")
                    return jsonify({'error': str(e)}), 500
            
            elif data['action'] == 'status':
                cameras = live_camera_manager.get_cameras()
                return jsonify({'cameras': cameras})
    
    return render_template('live_camera.html')


@app.route('/video-feed/<camera_id>')
def video_feed(camera_id):
    """Stream video feed from live camera"""
    if VERCEL_MODE or DISABLE_LIVE_CAMERA:
        return jsonify({'error': 'Video streaming is disabled for this deployment.'}), 501

    return Response(live_camera_manager.generate_frames(camera_id),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/alerts')
def alerts():
    """View all generated alerts"""
    alerts_list = alert_generator.get_all_alerts()
    return render_template('alerts.html', alerts=alerts_list)


@app.route('/api/alerts')
def api_alerts():
    """API endpoint for alerts"""
    alerts_list = alert_generator.get_all_alerts()
    return jsonify(alerts_list)


@app.route('/system-status')
def system_status():
    """Get system status"""
    status = {
        'vercel_mode': VERCEL_MODE,
        'disable_video': DISABLE_VIDEO,
        'disable_live_camera': DISABLE_LIVE_CAMERA,
        'inference_proxy_enabled': bool(INFERENCE_URL) if VERCEL_MODE else True,
        'yolo_model_loaded': yolo_detector is not None and yolo_detector.loaded,
        'slowfast_model_loaded': slowfast_analyzer is not None,
        'active_cameras': live_camera_manager.get_active_camera_count() if live_camera_manager else 0,
        'total_alerts': alert_generator.get_alert_count(),
        'uptime': datetime.datetime.now().isoformat()
    }
    return jsonify(status)


@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors"""
    return render_template('error.html', error='Page not found'), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    return render_template('error.html', error='Internal server error'), 500


if __name__ == '__main__':
    logger.info("Starting Wildlife Injury Detection System...")
    app.run(debug=True, host='0.0.0.0', port=5000)
