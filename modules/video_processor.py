"""
Video Processor Module for Wildlife Injury Detection
Handles video upload and processing with YOLOv8 and SlowFast
"""

import os
import cv2
import numpy as np
from pathlib import Path
import logging


class VideoProcessor:
    """Processes uploaded videos for wildlife injury detection"""
    
    def __init__(self, yolo_detector, slowfast_analyzer, alert_generator, 
                 annotated_folder, evidence_folder):
        """
        Initialize video processor
        
        Args:
            yolo_detector: YOLOv8 detector instance
            slowfast_analyzer: SlowFast analyzer instance
            alert_generator: Alert generator instance
            annotated_folder: Folder for annotated videos
            evidence_folder: Folder for evidence clips
        """
        self.yolo_detector = yolo_detector
        self.slowfast_analyzer = slowfast_analyzer
        self.alert_generator = alert_generator
        self.annotated_folder = Path(annotated_folder)
        self.evidence_folder = Path(evidence_folder)
        
        # Create folders
        self.annotated_folder.mkdir(parents=True, exist_ok=True)
        self.evidence_folder.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.frame_skip = 5  # Process every Nth frame for efficiency
        self.clip_length = 32  # Frames per clip for SlowFast
        self.min_frames_for_analysis = 16
        
        self.logger = logging.getLogger('VideoProcessor')
    
    def process_video(self, video_path, video_id):
        """
        Process uploaded video for wildlife injury detection
        
        Args:
            video_path: Path to uploaded video
            video_id: Unique identifier for video
        
        Returns:
            Processing results dictionary
        """
        self.logger.info(f"Starting video processing: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
        
        # Initialize tracking
        animal_tracks = {}  # animal_id -> {'frames': [], 'positions': []}
        
        # Process frames
        frame_count = 0
        detection_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames for efficiency
            if frame_count % self.frame_skip != 0:
                frame_count += 1
                continue
            
            # Run YOLOv8 detection
            detections = self.yolo_detector.detect(frame)
            
            # Filter animal detections
            animal_detections = self.yolo_detector.filter_animals(detections, min_confidence=0.60)
            
            for det in animal_detections:
                animal_id = f"{det['class_name']}_{det['bbox']}"
                
                # Initialize track if new
                if animal_id not in animal_tracks:
                    animal_tracks[animal_id] = {
                        'class_name': det['class_name'],
                        'frames': [],
                        'bboxes': [],
                        'confidences': []
                    }
                
                # Add to track
                animal_tracks[animal_id]['frames'].append(frame.copy())
                animal_tracks[animal_id]['bboxes'].append(det['bbox'])
                animal_tracks[animal_id]['confidences'].append(det['confidence'])
                
                # Keep only recent frames
                max_frames = self.clip_length * 2
                if len(animal_tracks[animal_id]['frames']) > max_frames:
                    animal_tracks[animal_id]['frames'] = animal_tracks[animal_id]['frames'][-max_frames:]
                    animal_tracks[animal_id]['bboxes'] = animal_tracks[animal_id]['bboxes'][-max_frames:]
                    animal_tracks[animal_id]['confidences'] = animal_tracks[animal_id]['confidences'][-max_frames:]
            
            # Draw detections on frame (no injury info for now)
            annotated_frame = self.yolo_detector.draw_detections(frame, animal_detections)
            
            detection_results.append({
                'frame': frame_count,
                'detections': animal_detections,
                'annotated_frame': annotated_frame
            })
            
            frame_count += 1
        
        cap.release()
        
        # Analyze each tracked animal
        alerts = []
        analysis_results = []
        
        for animal_id, track in animal_tracks.items():
            if len(track['frames']) < self.min_frames_for_analysis:
                continue
            
            # Create clips for SlowFast analysis
            clips = self._create_clips(track['frames'], self.clip_length)
            
            for clip in clips:
                # Analyze with SlowFast
                analysis = self.slowfast_analyzer.analyze_clip(clip)
                
                # Get average bbox for this track
                avg_bbox = self._get_average_bbox(track['bboxes'][-len(clip):])
                avg_confidence = np.mean(track['confidences'][-len(clip):])
                
                result = {
                    'animal_id': animal_id,
                    'animal_class': track['class_name'],
                    'yolo_confidence': float(avg_confidence),
                    'injury_probability': analysis['injury_probability'],
                    'classification': analysis['classification'],
                    'indicators': analysis['indicators'],
                    'bbox': avg_bbox
                }
                
                analysis_results.append(result)
                
                # Generate alert if injury detected
                if analysis['injury_probability'] >= 0.75:
                    # Save evidence clip
                    evidence_path = self._save_evidence_clip(clip, video_id, animal_id)
                    
                    # Generate alert
                    alert = self.alert_generator.generate_alert(
                        input_mode='video',
                        camera_id=video_id,
                        animal_class=track['class_name'],
                        yolo_confidence=avg_confidence,
                        injury_probability=analysis['injury_probability'],
                        bbox=avg_bbox,
                        evidence_path=evidence_path
                    )
                    
                    if alert:
                        alerts.append(alert)
        
        # Save annotated video
        annotated_video_path = self._save_annotated_video(detection_results, video_id, fps, (width, height))
        
        # Create detection log
        detection_log = self._create_detection_log(video_id, detection_results, analysis_results)
        
        return {
            'video_id': video_id,
            'total_frames_processed': frame_count,
            'unique_animals': len(animal_tracks),
            'analysis_results': analysis_results,
            'alerts': alerts,
            'annotated_video': annotated_video_path,
            'detection_log': detection_log
        }
    
    def _create_clips(self, frames, clip_length):
        """Create overlapping clips from frames"""
        clips = []
        num_frames = len(frames)
        
        if num_frames < clip_length:
            clips.append(frames)
        else:
            # Create clips with overlap
            step = clip_length // 2
            for i in range(0, num_frames - clip_length + 1, step):
                clip = frames[i:i + clip_length]
                clips.append(clip)
        
        return clips
    
    def _get_average_bbox(self, bboxes):
        """Calculate average bounding box"""
        if not bboxes:
            return [0, 0, 0, 0]
        
        avg_x1 = int(np.mean([b[0] for b in bboxes]))
        avg_y1 = int(np.mean([b[1] for b in bboxes]))
        avg_x2 = int(np.mean([b[2] for b in bboxes]))
        avg_y2 = int(np.mean([b[3] for b in bboxes]))
        
        return [avg_x1, avg_y1, avg_x2, avg_y2]
    
    def _save_evidence_clip(self, frames, video_id, animal_id):
        """Save evidence clip"""
        try:
            output_path = self.evidence_folder / f"evidence_{video_id}_{animal_id}.mp4"
            
            if len(frames) == 0:
                return None
            
            height, width = frames[0].shape[:2]
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, 10, (width, height))
            
            for frame in frames:
                out.write(frame)
            
            out.release()
            
            return str(output_path)
        
        except Exception as e:
            self.logger.error(f"Error saving evidence clip: {e}")
            return None
    
    def _save_annotated_video(self, detection_results, video_id, fps, size):
        """Save annotated video"""
        try:
            output_path = self.annotated_folder / f"annotated_{video_id}.mp4"
            
            width, height = size
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            for result in detection_results:
                if 'annotated_frame' in result:
                    out.write(result['annotated_frame'])
            
            out.release()
            
            return str(output_path)
        
        except Exception as e:
            self.logger.error(f"Error saving annotated video: {e}")
            return None
    
    def _create_detection_log(self, video_id, detection_results, analysis_results):
        """Create detection log"""
        import json
        import datetime
        
        log_data = {
            'video_id': video_id,
            'timestamp': datetime.datetime.now().isoformat(),
            'total_frames': len(detection_results),
            'analysis_results': analysis_results
        }
        
        log_path = self.annotated_folder / f"log_{video_id}.json"
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        return str(log_path)


if __name__ == "__main__":
    # Test the video processor
    print("Video processor module ready for use")