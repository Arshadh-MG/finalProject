"""
Alert Generator Module for Wildlife Injury Detection
Generates structured alerts when injury is detected
"""

import os
import json
import datetime
import uuid
from pathlib import Path


class AlertGenerator:
    """Generates and manages wildlife injury alerts"""
    
    def __init__(self, log_folder=None):
        """
        Initialize alert generator
        
        Args:
            log_folder: Path to folder for storing alert logs
        """
        self.log_folder = Path(log_folder) if log_folder else Path('logs')
        self.log_folder.mkdir(parents=True, exist_ok=True)
        
        # Alert storage
        self.alerts = []
        self.alert_file = self.log_folder / 'alerts.json'
        
        # Load existing alerts
        self._load_alerts()
        
        # Duplicate alert prevention
        self.recent_alerts = {}  # camera_id -> last_alert_time
        self.duplicate_prevention_window = 300  # 5 minutes in seconds
        
        # Alert statistics
        self.stats = {
            'total_alerts': len(self.alerts),
            'alerts_by_type': {},
            'alerts_by_mode': {}
        }
    
    def _load_alerts(self):
        """Load alerts from file"""
        if self.alert_file.exists():
            try:
                with open(self.alert_file, 'r') as f:
                    self.alerts = json.load(f)
            except Exception as e:
                print(f"Error loading alerts: {e}")
                self.alerts = []
    
    def _save_alerts(self):
        """Save alerts to file"""
        try:
            with open(self.alert_file, 'w') as f:
                json.dump(self.alerts, f, indent=2)
        except Exception as e:
            print(f"Error saving alerts: {e}")
    
    def _check_duplicate(self, camera_id):
        """Check if recent alert exists for this camera"""
        if camera_id in self.recent_alerts:
            last_alert_time = self.recent_alerts[camera_id]
            time_diff = (datetime.datetime.now() - last_alert_time).total_seconds()
            
            if time_diff < self.duplicate_prevention_window:
                return True
        
        return False
    
    def _update_recent_alert(self, camera_id):
        """Update recent alert timestamp for camera"""
        self.recent_alerts[camera_id] = datetime.datetime.now()
    
    def generate_alert(self, input_mode, camera_id, animal_class, yolo_confidence, 
                       injury_probability, bbox, evidence_path=None):
        """
        Generate structured alert for wildlife injury
        
        Args:
            input_mode: 'image', 'video', or 'live_camera'
            camera_id: Unique identifier for camera or upload
            animal_class: Detected animal class
            yolo_confidence: YOLOv8 detection confidence
            injury_probability: Calculated injury probability
            bbox: Bounding box [x1, y1, x2, y2]
            evidence_path: Path to saved evidence file
        
        Returns:
            Alert dictionary
        """
        # Check for duplicate alert
        if self._check_duplicate(camera_id):
            print(f"Duplicate alert prevented for camera {camera_id}")
            return None
        
        # Create alert
        alert = {
            "alert_id": str(uuid.uuid4()),
            "alert_type": "Possible Wildlife Injury",
            "input_mode": input_mode,
            "camera_id": camera_id,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "animal_class": animal_class,
            "yolo_confidence": round(yolo_confidence, 4),
            "injury_probability": round(injury_probability, 4),
            "bounding_box": [int(x) for x in bbox],
            "evidence_path": evidence_path,
            "severity": self._calculate_severity(injury_probability),
            "status": "active"
        }
        
        # Add to alerts list
        self.alerts.append(alert)
        
        # Update recent alert
        self._update_recent_alert(camera_id)
        
        # Save to file
        self._save_alerts()
        
        # Update statistics
        self._update_stats(alert)
        
        # Log alert
        self._log_alert(alert)
        
        print(f"ALERT GENERATED: {animal_class} - Injury Probability: {injury_probability:.2f}")
        
        return alert
    
    def _calculate_severity(self, injury_probability):
        """Calculate alert severity based on injury probability"""
        if injury_probability >= 0.90:
            return "critical"
        elif injury_probability >= 0.80:
            return "high"
        elif injury_probability >= 0.75:
            return "medium"
        else:
            return "low"
    
    def _update_stats(self, alert):
        """Update alert statistics"""
        self.stats['total_alerts'] = len(self.alerts)
        
        # By type
        alert_type = alert['alert_type']
        if alert_type not in self.stats['alerts_by_type']:
            self.stats['alerts_by_type'][alert_type] = 0
        self.stats['alerts_by_type'][alert_type] += 1
        
        # By mode
        mode = alert['input_mode']
        if mode not in self.stats['alerts_by_mode']:
            self.stats['alerts_by_mode'][mode] = 0
        self.stats['alerts_by_mode'][mode] += 1
    
    def _log_alert(self, alert):
        """Log alert to file"""
        log_file = self.log_folder / 'alert_log.txt'
        
        try:
            with open(log_file, 'a') as f:
                f.write(f"[{alert['timestamp']}] {alert['alert_type']} - ")
                f.write(f"Animal: {alert['animal_class']}, ")
                f.write(f"Injury Prob: {alert['injury_probability']:.2f}, ")
                f.write(f"Mode: {alert['input_mode']}\n")
        except Exception as e:
            print(f"Error logging alert: {e}")
    
    def get_all_alerts(self, limit=None):
        """Get all alerts"""
        if limit:
            return self.alerts[-limit:]
        return self.alerts
    
    def get_alert_count(self):
        """Get total number of alerts"""
        return len(self.alerts)
    
    def get_alerts_by_camera(self, camera_id):
        """Get alerts for specific camera"""
        return [a for a in self.alerts if a['camera_id'] == camera_id]
    
    def get_alerts_by_severity(self, severity):
        """Get alerts by severity level"""
        return [a for a in self.alerts if a.get('severity') == severity]
    
    def get_alerts_by_date(self, date_str):
        """Get alerts for specific date (YYYY-MM-DD)"""
        return [a for a in self.alerts if a['timestamp'].startswith(date_str)]
    
    def get_statistics(self):
        """Get alert statistics"""
        return self.stats
    
    def mark_alert_resolved(self, alert_id):
        """Mark an alert as resolved"""
        for alert in self.alerts:
            if alert.get('alert_id') == alert_id:
                alert['status'] = 'resolved'
                alert['resolved_at'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self._save_alerts()
                return True
        return False
    
    def clear_alerts(self):
        """Clear all alerts"""
        self.alerts = []
        self._save_alerts()
        self.stats = {
            'total_alerts': 0,
            'alerts_by_type': {},
            'alerts_by_mode': {}
        }
    
    def export_alerts(self, format='json', filepath=None):
        """Export alerts to file"""
        if filepath is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.log_folder / f"alerts_export_{timestamp}.{format}"
        
        if format == 'json':
            with open(filepath, 'w') as f:
                json.dump(self.alerts, f, indent=2)
        
        elif format == 'csv':
            import csv
            if self.alerts:
                keys = self.alerts[0].keys()
                with open(filepath, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=keys)
                    writer.writeheader()
                    writer.writerows(self.alerts)
        
        elif format == 'txt':
            with open(filepath, 'w') as f:
                for alert in self.alerts:
                    f.write(f"Alert ID: {alert.get('alert_id')}\n")
                    f.write(f"Type: {alert['alert_type']}\n")
                    f.write(f"Time: {alert['timestamp']}\n")
                    f.write(f"Animal: {alert['animal_class']}\n")
                    f.write(f"Injury Probability: {alert['injury_probability']:.2f}\n")
                    f.write(f"Severity: {alert.get('severity', 'N/A')}\n")
                    f.write("-" * 40 + "\n")
        
        return str(filepath)


if __name__ == "__main__":
    # Test the alert generator
    generator = AlertGenerator()
    
    alert = generator.generate_alert(
        input_mode='image',
        camera_id='test_camera_001',
        animal_class='dog',
        yolo_confidence=0.92,
        injury_probability=0.82,
        bbox=[100, 100, 300, 300],
        evidence_path='/path/to/evidence.jpg'
    )
    
    print(f"Generated alert: {alert['alert_id']}")
    print(f"Total alerts: {generator.get_alert_count()}")
