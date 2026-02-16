"""
SlowFast Network Analyzer Module for Wildlife Injury Detection
Analyzes animal behavior and motion patterns to detect injuries
"""

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
from collections import deque
from pathlib import Path


class SlowFastAnalyzer:
    """SlowFast Network-based behavior and injury analyzer"""
    
    def __init__(self, num_classes=2):
        """
        Initialize SlowFast analyzer
        
        Args:
            num_classes: Number of output classes (2 for Normal/Injury)
        """
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Frame buffer for temporal analysis
        self.frame_buffers = {}
        self.buffer_size = 64  # Number of frames to keep
        
        # Injury indicators
        self.injury_indicators = [
            'limping',
            'uneven_gait',
            'leg_dragging',
            'abnormal_posture',
            'sudden_collapse',
            'prolonged_immobility',
            'distress_movement',
            'visible_injury'
        ]
        
        # Try to load SlowFast model
        try:
            self.model = self._create_slowfast_model()
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
            print(f"SlowFast model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Could not load SlowFast model: {e}")
            print("Using heuristic-based injury analysis instead")
            self.model = None
            self.loaded = False
    
    def _create_slowfast_model(self):
        """
        Create a simplified SlowFast-like model for demonstration
        In production, use pre-trained SlowFast model
        """
        # Simplified two-pathway model
        class SlowFastLite(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                
                # Slow pathway (spatial features)
                self.slow_path = nn.Sequential(
                    nn.Conv3d(3, 64, kernel_size=3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.Conv3d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm3d(128),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.AdaptiveAvgPool3d(1)
                )
                
                # Fast pathway (motion features)
                self.fast_path = nn.Sequential(
                    nn.Conv3d(3, 32, kernel_size=3, padding=1),
                    nn.BatchNorm3d(32),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.Conv3d(32, 64, kernel_size=3, padding=1),
                    nn.BatchNorm3d(64),
                    nn.ReLU(),
                    nn.MaxPool3d(2),
                    nn.AdaptiveAvgPool3d(1)
                )
                
                # Fusion and classification
                self.classifier = nn.Sequential(
                    nn.Linear(128 + 64, 128),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.Linear(128, num_classes)
                )
            
            def forward(self, x):
                # Slow pathway
                slow_features = self.slow_path(x)
                slow_features = slow_features.view(slow_features.size(0), -1)
                
                # Fast pathway
                fast_features = self.fast_path(x)
                fast_features = fast_features.view(fast_features.size(0), -1)
                
                # Fusion
                combined = torch.cat([slow_features, fast_features], dim=1)
                output = self.classifier(combined)
                
                return output
        
        return SlowFastLite(num_classes=self.num_classes)
    
    def initialize_buffer(self, animal_id):
        """Initialize frame buffer for tracking an animal"""
        self.frame_buffers[animal_id] = {
            'frames': deque(maxlen=self.buffer_size),
            'positions': deque(maxlen=self.buffer_size),
            'last_detection': None,
            'injury_history': []
        }
    
    def add_frame(self, animal_id, frame, bbox):
        """
        Add frame to buffer for temporal analysis
        
        Args:
            animal_id: Unique identifier for the animal
            frame: Current frame
            bbox: Bounding box [x1, y1, x2, y2]
        """
        if animal_id not in self.frame_buffers:
            self.initialize_buffer(animal_id)
        
        buffer = self.frame_buffers[animal_id]
        
        # Store frame
        buffer['frames'].append(frame)
        
        # Store position (center of bbox)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        buffer['positions'].append((center_x, center_y))
        
        # Update last detection
        buffer['last_detection'] = {
            'bbox': bbox,
            'frame_idx': len(buffer['frames']) - 1
        }
    
    def analyze_motion(self, animal_id):
        """
        Analyze motion patterns for injury indicators
        
        Args:
            animal_id: Unique identifier for the animal
        
        Returns:
            Dictionary with motion analysis results
        """
        if animal_id not in self.frame_buffers:
            return {'error': 'No frames in buffer'}
        
        buffer = self.frame_buffers[animal_id]
        positions = list(buffer['positions'])
        
        if len(positions) < 10:
            return {'error': 'Insufficient frames for analysis'}
        
        # Calculate motion metrics
        motion_analysis = {
            'velocity': [],
            'acceleration': [],
            'direction_changes': 0,
            'stationary_frames': 0,
            'irregularity_score': 0.0
        }
        
        # Calculate velocities
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            velocity = np.sqrt(dx**2 + dy**2)
            motion_analysis['velocity'].append(velocity)
        
        # Calculate accelerations
        velocities = motion_analysis['velocity']
        for i in range(1, len(velocities)):
            acc = velocities[i] - velocities[i-1]
            motion_analysis['acceleration'].append(acc)
        
        # Count direction changes
        for i in range(2, len(positions)):
            v1 = (positions[i-1][0] - positions[i-2][0], positions[i-1][1] - positions[i-2][1])
            v2 = (positions[i][0] - positions[i-1][0], positions[i][1] - positions[i-1][1])
            
            if v1[0] * v2[0] + v1[1] * v2[1] < 0:  # Dot product negative
                motion_analysis['direction_changes'] += 1
        
        # Count stationary frames
        for vel in velocities:
            if vel < 2:  # Threshold for stationary
                motion_analysis['stationary_frames'] += 1
        
        # Calculate irregularity score
        if len(velocities) > 0:
            vel_array = np.array(velocities)
            vel_std = np.std(vel_array)
            vel_mean = np.mean(vel_array)
            
            if vel_mean > 0:
                motion_analysis['irregularity_score'] = vel_std / vel_mean
        
        return motion_analysis
    
    def analyze_injury_from_buffer(self, animal_id):
        """
        Comprehensive injury analysis from buffered frames
        
        Args:
            animal_id: Unique identifier for the animal
        
        Returns:
            Dictionary with injury probability and indicators
        """
        if animal_id not in self.frame_buffers:
            return {
                'injury_probability': 0.0,
                'indicators': [],
                'classification': 'unknown'
            }
        
        buffer = self.frame_buffers[animal_id]
        
        # Check if we have enough frames
        if len(buffer['frames']) < 8:
            return {
                'injury_probability': 0.0,
                'indicators': ['Insufficient frames'],
                'classification': 'insufficient_data'
            }
        
        indicators = []
        injury_probability = 0.0
        
        # Analyze motion
        motion_analysis = self.analyze_motion(animal_id)
        
        if 'error' not in motion_analysis:
            # Check for limping (irregular velocity patterns)
            if motion_analysis['irregularity_score'] > 0.5:
                injury_probability += 0.25
                indicators.append('limping')
            
            # Check for leg dragging (low velocity with direction)
            if motion_analysis['stationary_frames'] > len(buffer['positions']) * 0.3:
                injury_probability += 0.2
                indicators.append('prolonged_immobility')
            
            # Check for distress movement (frequent direction changes)
            if motion_analysis['direction_changes'] > 5:
                injury_probability += 0.2
                indicators.append('distress_movement')
            
            # Check for sudden collapse (abrupt velocity changes)
            if len(motion_analysis['acceleration']) > 0:
                max_acc = max(abs(a) for a in motion_analysis['acceleration'])
                if max_acc > 50:  # Large acceleration
                    injury_probability += 0.3
                    indicators.append('sudden_collapse')
        
        # Analyze visual features from frames
        visual_analysis = self.analyze_visual_features(buffer['frames'])
        injury_probability += visual_analysis['injury_probability']
        indicators.extend(visual_analysis['indicators'])
        
        # Cap probability
        injury_probability = min(injury_probability, 1.0)
        
        # Determine classification
        if injury_probability >= 0.75:
            classification = 'injured'
        elif injury_probability >= 0.4:
            classification = 'possible_injury'
        else:
            classification = 'normal'
        
        return {
            'injury_probability': injury_probability,
            'indicators': indicators,
            'classification': classification,
            'motion_analysis': motion_analysis if 'error' not in motion_analysis else None
        }
    
    def analyze_visual_features(self, frames):
        """
        Analyze visual features from frames for injury signs
        
        Args:
            frames: List of frames
        
        Returns:
            Dictionary with visual analysis results
        """
        indicators = []
        injury_probability = 0.0
        
        if len(frames) == 0:
            return {'injury_probability': 0.0, 'indicators': []}
        
        # Convert frames to analysis format
        # Analyze color distribution for wounds/abnormalities
        for frame in frames[-5:]:  # Analyze last 5 frames
            try:
                if len(frame.shape) == 3:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                    # Look for red tones (possible wounds)
                    lower_red = np.array([0, 50, 50])
                    upper_red = np.array([10, 255, 255])
                    mask = cv2.inRange(hsv, lower_red, upper_red)
                    
                    red_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
                    
                    if red_ratio > 0.1:
                        injury_probability += 0.1
                        if 'visible_injury' not in indicators:
                            indicators.append('visible_injury')
                
            except Exception:
                pass
        
        # Check for size changes (animal might be collapsing)
        if len(frames) >= 2:
            first_frame = frames[0]
            last_frame = frames[-1]
            
            if first_frame.shape == last_frame.shape:
                diff = cv2.absdiff(first_frame, last_frame)
                change_ratio = np.sum(diff > 30) / (diff.shape[0] * diff.shape[1])
                
                if change_ratio > 0.3:
                    injury_probability += 0.15
                    if 'abnormal_posture' not in indicators:
                        indicators.append('abnormal_posture')
        
        return {
            'injury_probability': injury_probability,
            'indicators': indicators
        }
    
    def analyze_clip(self, clip_frames):
        """
        Analyze a clip of frames using SlowFast-style analysis
        
        Args:
            clip_frames: List of frames (32-64 frames)
        
        Returns:
            Injury analysis result
        """
        if len(clip_frames) < 8:
            return {
                'injury_probability': 0.0,
                'indicators': ['Insufficient frames'],
                'classification': 'insufficient_data'
            }
        
        # For now, use heuristic-based analysis
        # In production, this would use actual SlowFast inference
        indicators = []
        injury_probability = 0.0
        
        # Stack frames for batch processing
        try:
            # Analyze motion patterns
            motion_features = self._extract_motion_features(clip_frames)
            
            # Check for various injury indicators
            if motion_features['velocity_variance'] > 30:
                injury_probability += 0.25
                indicators.append('limping')
            
            if motion_features['direction_changes'] > len(clip_frames) * 0.1:
                injury_probability += 0.2
                indicators.append('uneven_gait')
            
            if motion_features['avg_velocity'] < 2:
                injury_probability += 0.3
                indicators.append('prolonged_immobility')
            
            # Analyze visual features
            visual_features = self._extract_visual_features(clip_frames)
            
            if visual_features['has_wounds']:
                injury_probability += 0.3
                indicators.append('visible_injury')
            
            if visual_features['posture_abnormal']:
                injury_probability += 0.2
                indicators.append('abnormal_posture')
            
        except Exception as e:
            print(f"Error in clip analysis: {e}")
        
        # Cap probability
        injury_probability = min(injury_probability, 1.0)
        
        # Determine classification
        if injury_probability >= 0.75:
            classification = 'injured'
        elif injury_probability >= 0.4:
            classification = 'possible_injury'
        else:
            classification = 'normal'
        
        return {
            'injury_probability': injury_probability,
            'indicators': list(set(indicators)),
            'classification': classification
        }
    
    def _extract_motion_features(self, frames):
        """Extract motion features from frames"""
        features = {
            'velocity_variance': 0.0,
            'direction_changes': 0,
            'avg_velocity': 0.0
        }
        
        if len(frames) < 2:
            return features
        
        # Simple motion detection using frame differencing
        velocities = []
        
        for i in range(1, len(frames)):
            diff = cv2.absdiff(frames[i], frames[i-1])
            motion = np.sum(diff > 30)
            velocities.append(motion)
        
        if velocities:
            features['avg_velocity'] = np.mean(velocities)
            features['velocity_variance'] = np.var(velocities)
            
            # Count direction changes (significant motion changes)
            for i in range(1, len(velocities)):
                if abs(velocities[i] - velocities[i-1]) > np.mean(velocities):
                    features['direction_changes'] += 1
        
        return features
    
    def _extract_visual_features(self, frames):
        """Extract visual features from frames"""
        features = {
            'has_wounds': False,
            'posture_abnormal': False
        }
        
        if len(frames) == 0:
            return features
        
        # Analyze last frame
        frame = frames[-1]
        
        try:
            # Check for wounds (red/orange areas)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
            
            if np.sum(mask > 0) / (frame.shape[0] * frame.shape[1]) > 0.1:
                features['has_wounds'] = True
            
            # Check posture (aspect ratio changes)
            # This is a simplified check
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h > 0 else 1
                
                # Abnormal if very elongated or very flat
                if aspect_ratio > 3.0 or aspect_ratio < 0.33:
                    features['posture_abnormal'] = True
        
        except Exception:
            pass
        
        return features
    
    def clear_buffer(self, animal_id):
        """Clear frame buffer for an animal"""
        if animal_id in self.frame_buffers:
            del self.frame_buffers[animal_id]
    
    def get_buffer_status(self, animal_id):
        """Get current buffer status for an animal"""
        if animal_id not in self.frame_buffers:
            return {'status': 'no_buffer', 'frame_count': 0}
        
        buffer = self.frame_buffers[animal_id]
        return {
            'status': 'active',
            'frame_count': len(buffer['frames']),
            'buffer_capacity': self.buffer_size
        }


# Standalone function for quick analysis
def analyze_behavior(frames, animal_id='default'):
    """
    Standalone function to analyze animal behavior
    
    Args:
        frames: List of frames
        animal_id: Unique identifier for the animal
    
    Returns:
        Injury analysis result
    """
    analyzer = SlowFastAnalyzer()
    
    # Add frames to buffer
    for i, frame in enumerate(frames[:analyzer.buffer_size]):
        bbox = [0, 0, frame.shape[1], frame.shape[0]]  # Full frame as bbox
        analyzer.add_frame(animal_id, frame, bbox)
    
    # Analyze
    result = analyzer.analyze_injury_from_buffer(animal_id)
    
    return result


if __name__ == "__main__":
    # Test the analyzer
    analyzer = SlowFastAnalyzer()
    
    # Create dummy frames for testing
    test_frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(32)]
    
    result = analyzer.analyze_clip(test_frames)
    print(f"Injury probability: {result['injury_probability']:.2f}")
    print(f"Classification: {result['classification']}")
    print(f"Indicators: {result['indicators']}")
