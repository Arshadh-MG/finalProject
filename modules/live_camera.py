"""
Live Camera Manager Module for Wildlife Injury Detection
Handles live camera streams and real-time processing
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
import os
import glob
from pathlib import Path
from utils.condition_assessment import assess_condition


class LiveCameraManager:
    """Manages live camera streams for real-time wildlife injury detection"""
    
    def __init__(self, yolo_detector, slowfast_analyzer, alert_generator, evidence_folder):
        """
        Initialize live camera manager
        
        Args:
            yolo_detector: YOLOv8 detector instance
            slowfast_analyzer: SlowFast analyzer instance
            alert_generator: Alert generator instance
            evidence_folder: Folder for evidence files
        """
        self.yolo_detector = yolo_detector
        self.slowfast_analyzer = slowfast_analyzer
        self.alert_generator = alert_generator
        self.evidence_folder = Path(evidence_folder)
        
        # Create folder
        self.evidence_folder.mkdir(parents=True, exist_ok=True)
        
        # Camera management
        self.cameras = {}  # camera_id -> CameraInfo
        self.camera_lock = threading.Lock()
        
        # Processing parameters
        self.frame_skip = 3
        self.buffer_size = 64
        
        # Alert suppression
        self.alert_timestamps = {}
        self.alert_cooldown = 300  # seconds
        
        # Logger
        self.logger = logging.getLogger('LiveCameraManager')
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    
    def start_camera(self, camera_id, rtsp_url=''):
        """Start a live camera stream"""
        with self.camera_lock:
            if camera_id in self.cameras:
                return {'status': 'error', 'message': f'Camera {camera_id} already running'}
            
            # Try to open camera
            cap = None
            if rtsp_url:
                self.logger.info(f"Opening RTSP stream: {rtsp_url}")
                cap = cv2.VideoCapture(rtsp_url)
            else:
                # Avoid probing local camera devices when none exist (common in cloud containers).
                if os.name != "nt":
                    device_candidates = glob.glob("/dev/video*")
                    if not device_candidates:
                        return {
                            'status': 'error',
                            'message': 'No local camera devices found. Use an RTSP URL (cloud containers do not provide /dev/video* webcams).'
                        }

                # Try multiple camera indices if default fails
                for idx in range(2):  # try 0 and 1
                    self.logger.info(f"Trying camera index {idx}")
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        self.logger.info(f"Successfully opened camera index {idx}")
                        break
                    cap.release()
                    cap = None
            
            if cap is None or not cap.isOpened():
                return {'status': 'error', 'message': 'Failed to open any camera'}
            
            camera_info = {
                'camera_id': camera_id,
                'rtsp_url': rtsp_url,
                'capture': cap,
                'running': True,
                'thread': None,
                'frame_queue': queue.Queue(maxsize=10),
                'animal_buffers': {},
                'stats': {
                    'frames_processed': 0,
                    'animals_detected': 0,
                    'alerts_generated': 0
                }
            }
            
            thread = threading.Thread(target=self._process_camera, args=(camera_id,))
            thread.daemon = True
            thread.start()
            
            camera_info['thread'] = thread
            self.cameras[camera_id] = camera_info
            
            self.logger.info(f"Camera {camera_id} started")
            
            return {'status': 'success', 'message': f'Camera {camera_id} started', 'camera_id': camera_id}
    
    def stop_camera(self, camera_id):
        """Stop a live camera stream"""
        with self.camera_lock:
            if camera_id not in self.cameras:
                return {'status': 'error', 'message': f'Camera {camera_id} not found'}
            
            camera_info = self.cameras[camera_id]
            camera_info['running'] = False
            
            if camera_info['thread']:
                camera_info['thread'].join(timeout=5)
            
            if camera_info['capture']:
                camera_info['capture'].release()
            
            del self.cameras[camera_id]
            
            self.logger.info(f"Camera {camera_id} stopped")
            
            return {'status': 'success', 'message': f'Camera {camera_id} stopped'}
    
    def _process_camera(self, camera_id):
        """Process camera frames in background thread for alert generation"""
        camera_info = self.cameras.get(camera_id)
        if not camera_info:
            self.logger.error(f"Camera {camera_id} info not found")
            return
        
        cap = camera_info['capture']
        frame_count = 0
        
        while camera_info['running']:
            try:
                ret, frame = cap.read()
                
                if not ret:
                    self.logger.warning(f"Camera {camera_id} lost frame, attempting reconnect...")
                    time.sleep(1)
                    # Attempt to reopen
                    if camera_info['rtsp_url']:
                        new_cap = cv2.VideoCapture(camera_info['rtsp_url'])
                    else:
                        new_cap = cv2.VideoCapture(0)
                    if new_cap.isOpened():
                        cap.release()
                        camera_info['capture'] = new_cap
                        cap = new_cap
                        self.logger.info(f"Camera {camera_id} reconnected")
                    continue
                
                frame_count += 1
                
                # Skip frames for efficiency
                if frame_count % self.frame_skip != 0:
                    continue
                
                # Run YOLOv8 detection if available
                if self.yolo_detector and self.yolo_detector.loaded:
                    detections = self.yolo_detector.detect(frame)
                    animal_detections = self.yolo_detector.filter_animals(detections, min_confidence=0.60)
                else:
                    animal_detections = []
                
                camera_info['stats']['frames_processed'] += 1
                camera_info['stats']['animals_detected'] = max(
                    camera_info['stats']['animals_detected'],
                    len(animal_detections)
                )
                
                # Process each detected animal for possible alert
                for det in animal_detections:
                    animal_id = f"{camera_id}_{det['class_name']}_{det['bbox']}"
                    
                    if animal_id not in camera_info['animal_buffers']:
                        camera_info['animal_buffers'][animal_id] = {
                            'frames': [],
                            'class_name': det['class_name'],
                            'last_bbox': det['bbox'],
                            'last_damage': 0.0
                        }
                    
                    buffer = camera_info['animal_buffers'][animal_id]
                    buffer['frames'].append(frame.copy())
                    buffer['last_bbox'] = det['bbox']
                    
                    if len(buffer['frames']) > self.buffer_size:
                        buffer['frames'] = buffer['frames'][-self.buffer_size:]
                    
                    # Analyze if enough frames and slowfast is available
                    if len(buffer['frames']) >= 16 and self.slowfast_analyzer:
                        try:
                            analysis = self.slowfast_analyzer.analyze_clip(buffer['frames'])
                            buffer['last_damage'] = analysis['injury_probability']
                            
                            if analysis['injury_probability'] >= 0.75:
                                if self._can_generate_alert(camera_id):
                                    evidence_path = self._save_evidence(buffer['frames'], camera_id, animal_id)
                                    
                                    alert = self.alert_generator.generate_alert(
                                        input_mode='live_camera',
                                        camera_id=camera_id,
                                        animal_class=det['class_name'],
                                        yolo_confidence=det['confidence'],
                                        injury_probability=analysis['injury_probability'],
                                        bbox=det['bbox'],
                                        evidence_path=evidence_path
                                    )
                                    
                                    if alert:
                                        camera_info['stats']['alerts_generated'] += 1
                                        self._update_alert_timestamp(camera_id)
                        except Exception as e:
                            self.logger.error(f"Error analyzing clip: {e}")
                
                self._cleanup_buffers(camera_info)
                time.sleep(0.01)
                
            except Exception as e:
                self.logger.error(f"Error in _process_camera: {e}")
                time.sleep(1)
    
    def _can_generate_alert(self, camera_id):
        """Check if alert can be generated (cooldown)"""
        if camera_id not in self.alert_timestamps:
            return True
        
        last_alert = self.alert_timestamps[camera_id]
        time_diff = time.time() - last_alert
        
        return time_diff >= self.alert_cooldown
    
    def _update_alert_timestamp(self, camera_id):
        """Update alert timestamp for camera"""
        self.alert_timestamps[camera_id] = time.time()
    
    def _save_evidence(self, frames, camera_id, animal_id):
        """Save evidence frames as video"""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.evidence_folder / f"live_{camera_id}_{animal_id}_{timestamp}.mp4"
            
            if len(frames) == 0:
                return None
            
            height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10, (width, height))
            
            for frame in frames[-64:]:
                out.write(frame)
            
            out.release()
            self.logger.info(f"Evidence saved: {output_path}")
            return str(output_path)
        
        except Exception as e:
            self.logger.error(f"Error saving evidence: {e}")
            return None
    
    def _cleanup_buffers(self, camera_info):
        """Clean up old animal buffers"""
        current_time = time.time()
        
        to_remove = []
        for animal_id, buffer in camera_info['animal_buffers'].items():
            if 'last_update' in buffer:
                if current_time - buffer['last_update'] > 10:
                    to_remove.append(animal_id)
            else:
                buffer['last_update'] = current_time
        
        for animal_id in to_remove:
            del camera_info['animal_buffers'][animal_id]
    
    def generate_frames(self, camera_id):
        """
        Generator for video streaming (MJPEG).
        Yields annotated frames with damage percentages.
        """
        with self.camera_lock:
            if camera_id not in self.cameras:
                self.logger.error(f"Camera {camera_id} not found for streaming")
                return
            camera_info = self.cameras[camera_id]
        
        while True:
            try:
                with self.camera_lock:
                    if camera_id not in self.cameras:
                        self.logger.info(f"Camera {camera_id} stopped, ending stream")
                        break
                    camera_info = self.cameras[camera_id]
                    if not camera_info['running']:
                        self.logger.info(f"Camera {camera_id} not running, ending stream")
                        break
                    cap = camera_info['capture']
                
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Run detection if detector available
                if self.yolo_detector and self.yolo_detector.loaded:
                    detections = self.yolo_detector.detect(frame)
                    animal_detections = self.yolo_detector.filter_animals(detections)
                else:
                    animal_detections = []
                
                # Compute damage percentages for each detection
                damage_percentages = []
                for det in animal_detections:
                    roi = self.yolo_detector.get_animal_roi(frame, det['bbox'])
                    damage, _ = assess_condition(roi)
                    damage_percentages.append(damage)
                
                # Draw detections with damage percentages
                annotated_frame = self.yolo_detector.draw_detections(frame, animal_detections, damage_percentages=damage_percentages)
                
                # Encode as JPEG
                ret, jpeg = cv2.imencode('.jpg', annotated_frame)
                if not ret:
                    self.logger.error("Failed to encode frame")
                    continue
                
                frame_bytes = jpeg.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except Exception as e:
                self.logger.error(f"Error in generate_frames for {camera_id}: {e}")
                time.sleep(0.1)
                continue
    
    def get_cameras(self):
        """Get list of active cameras"""
        cameras = []
        with self.camera_lock:
            for camera_id, info in self.cameras.items():
                cameras.append({
                    'camera_id': camera_id,
                    'rtsp_url': info['rtsp_url'],
                    'running': info['running'],
                    'stats': info['stats'].copy()
                })
        return cameras
    
    def get_active_camera_count(self):
        """Get number of active cameras"""
        with self.camera_lock:
            return sum(1 for info in self.cameras.values() if info['running'])
    
    def get_camera_stats(self, camera_id):
        """Get statistics for specific camera"""
        with self.camera_lock:
            if camera_id not in self.cameras:
                return None
            return self.cameras[camera_id]['stats'].copy()


if __name__ == "__main__":
    print("Live camera manager module ready for use")
