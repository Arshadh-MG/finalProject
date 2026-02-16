"""
Helper Utilities for Wildlife Injury Detection
Provides common utility functions
"""

import os
import hashlib
import uuid
from pathlib import Path
from datetime import datetime


def allowed_file(filename, allowed_extensions):
    """
    Check if file extension is allowed
    
    Args:
        filename: Name of the file
        allowed_extensions: Set of allowed extensions
    
    Returns:
        True if allowed, False otherwise
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def create_directories(directories):
    """
    Create directories if they don't exist
    
    Args:
        directories: List of directory paths (Path objects or strings)
    """
    for directory in directories:
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)


def generate_unique_id():
    """
    Generate a unique identifier
    
    Returns:
        Unique ID string
    """
    return str(uuid.uuid4())


def get_file_hash(filepath):
    """
    Get MD5 hash of a file
    
    Args:
        filepath: Path to the file
    
    Returns:
        MD5 hash string
    """
    hash_md5 = hashlib.md5()
    
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    return hash_md5.hexdigest()


def format_timestamp(timestamp=None):
    """
    Format timestamp as string
    
    Args:
        timestamp: datetime object (uses current time if None)
    
    Returns:
        Formatted timestamp string
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def format_bbox(bbox):
    """
    Format bounding box for display
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        Formatted string
    """
    return f"({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})"


def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) for two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU score
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0
    
    return intersection / union


def get_file_size(filepath):
    """
    Get file size in human-readable format
    
    Args:
        filepath: Path to the file
    
    Returns:
        Size string
    """
    size = os.path.getsize(filepath)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    
    return f"{size:.2f} TB"


def sanitize_filename(filename):
    """
    Sanitize filename to remove special characters
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    # Remove special characters
    valid_chars = f"-_.() {filename}"
    sanitized = ''.join(c for c in filename if c.isalnum() or c in valid_chars)
    
    return sanitized


def get_video_info(video_path):
    """
    Get video information
    
    Args:
        video_path: Path to video file
    
    Returns:
        Dictionary with video info
    """
    import cv2
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    info = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    }
    
    cap.release()
    
    return info


class VideoWriterHelper:
    """Helper class for writing video files"""
    
    def __init__(self, output_path, fps=30, frame_size=(640, 480), codec='mp4v'):
        """
        Initialize video writer
        
        Args:
            output_path: Path to output video
            fps: Frames per second
            frame_size: (width, height)
            codec: Video codec
        """
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
    
    def write(self, frame):
        """Write a frame to video"""
        self.writer.write(frame)
    
    def release(self):
        """Release the video writer"""
        self.writer.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


if __name__ == "__main__":
    # Test helper functions
    print("Testing helper functions...")
    
    # Test allowed_file
    assert allowed_file("test.jpg", {'jpg', 'png'}) == True
    assert allowed_file("test.exe", {'jpg', 'png'}) == False
    
    # Test calculate_iou
    box1 = [0, 0, 10, 10]
    box2 = [5, 5, 15, 15]
    iou = calculate_iou(box1, box2)
    print(f"IoU: {iou}")
    
    print("Helper functions ready")
