"""
YOLOv8 Detector Module for Wildlife Injury Detection
Handles animal detection using YOLOv8 model
"""

import os
import numpy as np
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO


class YOLODetector:
    """YOLOv8-based animal detector"""
    
    def __init__(self, model_path=None, conf_threshold=0.25):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model (uses default custom model if None)
            conf_threshold: Confidence threshold for detections
        """
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.loaded = False
        
        # (Optional) keep class lists for reference, but no longer used for filtering
        self.animal_classes = {
            15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow',
            20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
            24: 'backpack', 25: 'umbrella',
        }
        
        self.target_classes = [
            'dog', 'cat', 'horse', 'sheep', 'cow', 'elephant', 'bear',
            'zebra', 'giraffe', 'bird', 'deer', 'fox', 'rabbit', 'squirrel',
            'mouse', 'rat', 'pig', 'goat', 'chicken', 'duck', 'goose',
            'turkey', 'person'
        ]
        
        # Try to load YOLOv8 model
        try:
            if model_path:
                # Use the provided model path
                if os.path.exists(model_path):
                    self.model = YOLO(model_path)
                else:
                    raise FileNotFoundError(f"Model not found: {model_path}")
            else:
                # Default to custom trained model in 'models/best.pt'
                default_model = Path(__file__).parent.parent / 'models' / 'best.pt'
                if default_model.exists():
                    print(f"Loading custom model: {default_model}")
                    self.model = YOLO(str(default_model))
                else:
                    # Fallback to standard YOLOv8n if custom model missing
                    print("Custom model not found, loading default yolov8n.pt")
                    self.model = YOLO('yolov8n.pt')
            
            self.model.to(self.device)
            print(f"Model class names: {self.model.names}")
            print(f"YOLOv8 model loaded successfully on {self.device}")
            self.loaded = True
            
        except Exception as e:
            print(f"Error loading YOLOv8 model: {e}")
            self.model = None
            self.loaded = False
    
    def detect(self, image, conf_threshold=None):
        """
        Detect objects in image using YOLOv8. Now returns ALL detections
        (no class filtering) with confidence above threshold.
        
        Args:
            image: Input image (numpy array)
            conf_threshold: Confidence threshold (uses default if None)
        
        Returns:
            List of detections with bbox, class, confidence
        """
        if not self.loaded or self.model is None:
            return []
        
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        try:
            # Run inference
            results = self.model(image, conf=conf_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    
                    # Get class name
                    class_name = result.names[class_id]
                    
                    # --- REMOVED CLASS FILTER ---
                    # Now every detection is added, regardless of class.
                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': confidence,
                        'class_id': class_id,
                        'class_name': class_name
                    }
                    detections.append(detection)
            
            # Apply Non-Maximum Suppression (NMS)
            detections = self.apply_nms(detections)
            
            return detections
            
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
            return []
    
    def apply_nms(self, detections, iou_threshold=0.45):
        """
        Apply Non-Maximum Suppression to remove overlapping detections
        
        Args:
            detections: List of detections
            iou_threshold: IoU threshold for NMS
        
        Returns:
            Filtered list of detections
        """
        if len(detections) == 0:
            return []
        
        # Extract boxes and scores
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            iou_threshold
        )
        
        if len(indices) == 0:
            return []
        
        # Return filtered detections
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        
        return [detections[i] for i in indices]
    
    def get_animal_roi(self, image, bbox):
        """
        Extract region of interest (ROI) for detected animal
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
        
        Returns:
            Cropped image ROI
        """
        x1, y1, x2, y2 = bbox
        
        # Add padding
        h, w = image.shape[:2]
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return image[y1:y2, x1:x2]
    
    def draw_detections(self, image, detections, damage_percentages=None):
        """
        Draw bounding boxes and labels on image with damage percentage.
        
        Args:
            image: Input image
            detections: List of detections
            damage_percentages: Optional list of damage percentages (same length as detections)
        
        Returns:
            Annotated image
        """
        annotated = image.copy()
        
        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Get damage percentage if provided
            damage = damage_percentages[i] if damage_percentages and i < len(damage_percentages) else 0.0
            
            # Set color based on damage severity
            if damage >= 0.75:
                color = (0, 0, 0)          # Black for severe
            elif damage >= 0.4:
                color = (0, 0, 139)        # Dark red for moderate
            else:
                color = (0, 100, 0)        # Dark green for minor/normal
            
            label = f"{class_name} {confidence:.2f} | Damage: {damage:.1%}"
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def filter_animals(self, detections, min_confidence=0.25):
        """
        Filter detections to keep only those with sufficient confidence.
        (Class restriction removed – all detections above confidence are returned.)
        
        Args:
            detections: List of all detections
            min_confidence: Minimum confidence threshold
        
        Returns:
            Filtered list of detections meeting the confidence threshold.
        """
        return [det for det in detections if det['confidence'] >= min_confidence]


# Standalone function for quick detection
def detect_wildlife(image_path, model_path=None, conf_threshold=0.25):
    """
    Standalone function to detect wildlife in image
    
    Args:
        image_path: Path to input image
        model_path: Path to YOLOv8 model
        conf_threshold: Confidence threshold
    
    Returns:
        List of detections
    """
    detector = YOLODetector(model_path, conf_threshold)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    
    # Detect
    detections = detector.detect(image)
    
    return detections


if __name__ == "__main__":
    # Test the detector
    detector = YOLODetector()
    
    # Test with a sample image path (replace with actual image)
    test_image_path = "test.jpg"
    if os.path.exists(test_image_path):
        detections = detect_wildlife(test_image_path)
        print(f"Found {len(detections)} detections:")
        for det in detections:
            print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    else:
        print("No test image found. YOLOv8 detector ready for use.")