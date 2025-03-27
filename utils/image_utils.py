#!/usr/bin/env python
import os
from PIL import Image

def save_result(result_image, output_path):
    """Save a PIL Image to the specified output path"""
    try:
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Save the image
        result_image.save(output_path)
        print(f"Saved extracted object to: {os.path.abspath(output_path)}")
        return True
    except Exception as e:
        print(f"Error saving result: {e}")
        return False

def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) between two bounding boxes"""
    # Determine coordinates of intersection rectangle
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    # Check if there is an intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    # Compute intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute areas of both bounding boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Compute IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    
    return iou 