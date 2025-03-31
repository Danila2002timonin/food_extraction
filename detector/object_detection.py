#!/usr/bin/env python
import torch
import numpy as np
import cv2
import os
import json
from utils.image_utils import calculate_iou

# Load data from JSON files
def load_json_data(filename):
    """Load data from a JSON file"""
    # Get the path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    file_path = os.path.join(parent_dir, 'ai_services', 'data', filename)
    
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception:
        # Return appropriate default based on expected structure
        return [] if filename.endswith('items.json') else {}

# Load multi-piece dish items from JSON
MULTI_PIECE_DISH_ITEMS = load_json_data('multi_piece_dish_items.json')

def merge_similar_objects(detected_objects, distance_threshold=250, max_object_ratio=0.5):
    """Merge similar objects of the same type that are close to each other"""
    if not detected_objects or len(detected_objects) <= 1:
        return detected_objects
    
    # Get image dimensions from the first object's box
    img_width = 0
    img_height = 0
    for obj in detected_objects:
        box = obj["box"]
        img_width = max(img_width, box[2])
        img_height = max(img_height, box[3])
    
    # If we couldn't determine image dimensions, use default values
    if img_width == 0 or img_height == 0:
        img_width = 2000
        img_height = 2000
    
    # Check if majority of detections are the same class
    label_counts = {}
    for obj in detected_objects:
        label = obj["label"]
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Identify the most common label
    most_common_label = max(label_counts, key=label_counts.get)
    most_common_count = label_counts[most_common_label]
    
    # If 70% or more of detections are the same label, this is likely a dish with multiple pieces
    is_multi_piece_dish = most_common_count >= 0.7 * len(detected_objects)
    
    # Adjust distance threshold based on image size for large images
    adaptive_distance = min(distance_threshold, max(img_width, img_height) * 0.1)
    
    # For multi-piece dishes like salads, try a more aggressive approach
    if is_multi_piece_dish and most_common_label.lower() in MULTI_PIECE_DISH_ITEMS:
        # Find objects of the most common type
        common_objects = [obj for obj in detected_objects if obj["label"] == most_common_label]
        other_objects = [obj for obj in detected_objects if obj["label"] != most_common_label]
        
        # Create a single large bounding box that encompasses all objects of the most common type
        if common_objects:
            min_x = min(obj["box"][0] for obj in common_objects)
            min_y = min(obj["box"][1] for obj in common_objects)
            max_x = max(obj["box"][2] for obj in common_objects)
            max_y = max(obj["box"][3] for obj in common_objects)
            
            # Calculate average confidence
            avg_confidence = sum(obj["score"] for obj in common_objects) / len(common_objects)
            
            # Create a new merged object for the entire dish
            dish_obj = {
                "box": [min_x, min_y, max_x, max_y],
                "label": f"{most_common_label}_dish",
                "score": avg_confidence,
                "merged_count": len(common_objects)
            }
            
            # Check if the merged object is small compared to the image
            box_width = max_x - min_x
            box_height = max_y - min_y
            box_ratio = (box_width * box_height) / (img_width * img_height)
            
            if box_ratio >= 0.2:
                return [dish_obj] + other_objects
            
            # Otherwise, expand the box slightly to try to capture more of the dish
            expanded_min_x = max(0, min_x - box_width * 0.2)
            expanded_min_y = max(0, min_y - box_height * 0.2)
            expanded_max_x = min(img_width, max_x + box_width * 0.2)
            expanded_max_y = min(img_height, max_y + box_height * 0.2)
            
            dish_obj["box"] = [expanded_min_x, expanded_min_y, expanded_max_x, expanded_max_y]
            dish_obj["expanded"] = True
            
            return [dish_obj] + other_objects
    
    # Continue with normal clustering approach if the aggressive approach didn't work
    # Group objects by label
    objects_by_label = {}
    for obj in detected_objects:
        label = obj["label"]
        if label not in objects_by_label:
            objects_by_label[label] = []
        objects_by_label[label].append(obj)
    
    # Process each group of objects
    merged_objects = []
    
    for label, objects in objects_by_label.items():
        # If there's only one object of this type, no need to merge
        if len(objects) <= 1:
            merged_objects.extend(objects)
            continue
        
        # List to track which objects have been merged
        merged_indices = set()
        
        # For food items where we expect multiple pieces (like broccoli, salad, etc.)
        if label.lower() in MULTI_PIECE_DISH_ITEMS:
            # Find objects that are close to each other
            for i in range(len(objects)):
                if i in merged_indices:
                    continue
                    
                current_group = [i]
                box_i = objects[i]["box"]
                
                for j in range(i + 1, len(objects)):
                    if j in merged_indices:
                        continue
                        
                    box_j = objects[j]["box"]
                    
                    # Calculate center points
                    center_i = ((box_i[0] + box_i[2]) / 2, (box_i[1] + box_i[3]) / 2)
                    center_j = ((box_j[0] + box_j[2]) / 2, (box_j[1] + box_j[3]) / 2)
                    
                    # Calculate Euclidean distance between centers
                    distance = np.sqrt((center_i[0] - center_j[0])**2 + (center_i[1] - center_j[1])**2)
                    
                    # If objects are close, add to the current group
                    if distance < adaptive_distance:
                        current_group.append(j)
                        merged_indices.add(j)
                
                if len(current_group) > 1:
                    # We found objects to merge
                    merged_indices.add(i)
                    
                    # Calculate the bounding box that encompasses all objects in the group
                    min_x = min(objects[idx]["box"][0] for idx in current_group)
                    min_y = min(objects[idx]["box"][1] for idx in current_group)
                    max_x = max(objects[idx]["box"][2] for idx in current_group)
                    max_y = max(objects[idx]["box"][3] for idx in current_group)
                    
                    # Calculate average confidence
                    avg_confidence = sum(objects[idx]["score"] for idx in current_group) / len(current_group)
                    
                    # Create a new merged object
                    merged_obj = {
                        "box": [min_x, min_y, max_x, max_y],
                        "label": f"{label}_cluster",
                        "score": avg_confidence,
                        "merged_count": len(current_group)
                    }
                    
                    merged_objects.append(merged_obj)
                elif i not in merged_indices:
                    # This object wasn't merged with others
                    merged_objects.append(objects[i])
            
            # Add any objects that weren't merged
            for i, obj in enumerate(objects):
                if i not in merged_indices:
                    merged_objects.append(obj)
        else:
            # For other types of objects, don't merge
            merged_objects.extend(objects)
    
    # Sort by confidence score
    merged_objects.sort(key=lambda x: x["score"], reverse=True)
    
    # Try to merge clusters of the same type if there are multiple
    merged_clusters = {}
    non_cluster_objects = []
    
    for obj in merged_objects:
        if "_cluster" in obj["label"]:
            base_label = obj["label"].split("_cluster")[0]
            if base_label not in merged_clusters:
                merged_clusters[base_label] = []
            merged_clusters[base_label].append(obj)
        else:
            non_cluster_objects.append(obj)
    
    # Final objects list
    final_objects = non_cluster_objects
    
    # For each type of cluster, check if they can be merged
    for base_label, clusters in merged_clusters.items():
        if len(clusters) <= 1:
            final_objects.extend(clusters)
            continue
        
        # Check if we can merge these clusters
        min_x = min(cluster["box"][0] for cluster in clusters)
        min_y = min(cluster["box"][1] for cluster in clusters)
        max_x = max(cluster["box"][2] for cluster in clusters)
        max_y = max(cluster["box"][3] for cluster in clusters)
        
        # Calculate total merged count and average confidence
        total_merged = sum(cluster.get("merged_count", 1) for cluster in clusters)
        avg_confidence = sum(cluster["score"] for cluster in clusters) / len(clusters)
        
        # Check if the super-cluster is reasonable (not too large)
        box_width = max_x - min_x
        box_height = max_y - min_y
        box_ratio = (box_width * box_height) / (img_width * img_height)
        
        if box_ratio <= max_object_ratio:
            # Create a super-cluster
            super_cluster = {
                "box": [min_x, min_y, max_x, max_y],
                "label": f"{base_label}_dish",
                "score": avg_confidence,
                "merged_count": total_merged
            }
            final_objects.append(super_cluster)
        else:
            # Too large, keep original clusters
            final_objects.extend(clusters)
    
    # Sort by confidence score
    final_objects.sort(key=lambda x: x["score"], reverse=True)
    
    return final_objects

def detect_objects(detector, image):
    """Detect objects in an image using Hugging Face models"""
    try:
        # Normalize image format - ensure we're always working with RGB (no alpha channel)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Prepare image for the model
        inputs = detector.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(detector.device) for k, v in inputs.items()}
        
        # Perform inference
        with torch.no_grad():
            outputs = detector.model(**inputs)
        
        # Convert outputs based on model type
        detected_objects = []
        
        # Different processing based on model architecture
        if detector.model_type in ["detr", "dino"]:
            # Process DETR/DINO model outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(detector.device)
            results = detector.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=detector.detection_threshold
            )[0]
            
            # Extract results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Convert all box coordinates to regular Python ints (not numpy int64)
                box = [int(round(i)) for i in box.tolist()]
                
                # Get the label name from the model's config
                label_name = detector.model.config.id2label[label.item()]
                
                # Skip background class (usually "dining table" or similar)
                if label_name.lower() in ["dining table", "table", "background"]:
                    continue
                
                detected_objects.append({
                    "label": label_name,
                    "score": float(score.item()),
                    "box": box  # [x1, y1, x2, y2]
                })
            
        elif detector.model_type in ["faster-rcnn", "yolos"]:
            # Process Faster R-CNN or YOLOS outputs
            target_sizes = torch.tensor([image.size[::-1]]).to(detector.device)
            results = detector.processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=detector.detection_threshold
            )[0]
            
            # Extract results
            for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                # Convert all box coordinates to regular Python ints (not numpy int64)
                box = [int(round(i)) for i in box.tolist()]
                
                # Get the label name from the model's config
                label_name = detector.model.config.id2label[label.item()]
                
                # Skip background class
                if label_name.lower() in ["dining table", "table", "background"]:
                    continue
                    
                detected_objects.append({
                    "label": label_name,
                    "score": float(score.item()),
                    "box": box  # [x1, y1, x2, y2]
                })
        
        # If we didn't detect any objects with the current threshold, try a lower threshold as fallback
        if not detected_objects and detector.detection_threshold > 0.05:
            # Temporarily lower the threshold
            original_threshold = detector.detection_threshold
            detector.detection_threshold = 0.05
            
            # Retry detection
            detected_objects = detect_objects(detector, image)
            
            # Restore original threshold
            detector.detection_threshold = original_threshold
            
        # Attempt to find cups/glasses that might have been missed
        detected_objects = find_missing_tableware(detector, image, detected_objects)
        
        # Merge similar objects
        detected_objects = merge_similar_objects(detected_objects)
        
        return detected_objects
        
    except Exception as e:
        if detector.debug_mode:
            traceback.print_exc()
        return []

def find_missing_tableware(detector, image, detected_objects):
    """Find commonly missed tableware items with heuristic methods"""
    # Convert to numpy array
    img_np = np.array(image)
    
    # Get existing object boxes to avoid duplicates
    existing_boxes = [obj['box'] for obj in detected_objects]
    
    # Look for circular shapes that could be cups or plates
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Detect circles using Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=100,
            param1=100,
            param2=30,
            minRadius=30,
            maxRadius=200
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            # Limit the number of detected circles to avoid overwhelming
            max_circles = 10
            if len(circles) > max_circles:
                circles = circles[:max_circles]
            
            for (x, y, r) in circles:
                # Create a box from the circle
                x1, y1 = max(0, x - r), max(0, y - r)
                x2, y2 = min(image.width, x + r), min(image.height, y + r)
                
                # Convert all numpy int64 to regular Python ints for JSON serialization
                new_box = [int(x1), int(y1), int(x2), int(y2)]
                
                # Check if this circle overlaps significantly with any existing box
                is_new = True
                for box in existing_boxes:
                    iou = calculate_iou(new_box, box)
                    if iou > 0.3:  # If significant overlap
                        is_new = False
                        break
                
                if is_new:
                    detected_objects.append({
                        "label": "cup",  # Default to cup, GPT can refine this based on context
                        "score": 0.7,  # Moderate confidence
                        "box": new_box,
                        "from_heuristic": True
                    })
                    existing_boxes.append(new_box)
    except Exception:
        pass
    
    return detected_objects 