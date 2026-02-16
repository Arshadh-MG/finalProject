"""
Condition Assessment Module for Wildlife Injury Detection
Provides functions to assess damage/injury percentage from an image ROI.
"""

import cv2
import numpy as np


def assess_condition(roi):
    """
    Assess the condition of an object in a region of interest (ROI).
    
    Args:
        roi: numpy array (image crop) of the detected object.
    
    Returns:
        damage_percentage: float between 0 and 1 indicating estimated damage.
        indicators: list of strings describing observed issues.
    """
    if roi is None or roi.size == 0:
        return 0.0, []
    
    h, w = roi.shape[:2]
    if h == 0 or w == 0:
        return 0.0, []
    
    damage = 0.0
    indicators = []
    total_pixels = h * w
    
    # 1. Color analysis for wounds/bleeding (red/orange)
    try:
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_ratio = np.sum(red_mask > 0) / total_pixels
        # Cap contribution at 0.5
        red_contrib = min(red_ratio * 2.0, 0.5)
        damage += red_contrib
        if red_ratio > 0.05:
            indicators.append('possible bleeding')
        print(f"[DEBUG] red_ratio={red_ratio:.3f}, contrib={red_contrib:.3f}")
    except Exception as e:
        print(f"[DEBUG] red analysis error: {e}")
    
    # 2. Dark spots (deep injuries, holes)
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        dark_mask = gray < 30
        dark_ratio = np.sum(dark_mask) / total_pixels
        dark_contrib = min(dark_ratio * 1.5, 0.4)
        damage += dark_contrib
        if dark_ratio > 0.05:
            indicators.append('dark spots')
        print(f"[DEBUG] dark_ratio={dark_ratio:.3f}, contrib={dark_contrib:.3f}")
    except Exception as e:
        print(f"[DEBUG] dark analysis error: {e}")
    
    # 3. Asymmetry (left/right halves)
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mid = w // 2
        left = gray[:, :mid]
        right = gray[:, mid:]
        if left.shape[1] > 0 and right.shape[1] > 0:
            right_flipped = cv2.flip(right, 1)
            # Resize to same dimensions if necessary
            if left.shape != right_flipped.shape:
                min_w = min(left.shape[1], right_flipped.shape[1])
                left = left[:, :min_w]
                right_flipped = right_flipped[:, :min_w]
            if left.shape == right_flipped.shape:
                diff = cv2.absdiff(left, right_flipped)
                asym_ratio = np.sum(diff > 30) / (left.shape[0] * left.shape[1])
                asym_contrib = min(asym_ratio * 1.0, 0.3)
                damage += asym_contrib
                if asym_ratio > 0.2:
                    indicators.append('asymmetry')
                print(f"[DEBUG] asym_ratio={asym_ratio:.3f}, contrib={asym_contrib:.3f}")
    except Exception as e:
        print(f"[DEBUG] asymmetry error: {e}")
    
    # 4. Surface integrity (edge density / texture changes)
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        edge_contrib = min(edge_density * 0.5, 0.2)
        damage += edge_contrib
        if edge_density > 0.2:
            indicators.append('surface damage')
        print(f"[DEBUG] edge_density={edge_density:.3f}, contrib={edge_contrib:.3f}")
    except Exception as e:
        print(f"[DEBUG] edge analysis error: {e}")
    
    # Final cap
    damage = min(damage, 1.0)
    print(f"[DEBUG] TOTAL DAMAGE = {damage:.3f}")
    
    return damage, indicators