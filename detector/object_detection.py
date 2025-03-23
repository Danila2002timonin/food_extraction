#!/usr/bin/env python
import torch
import numpy as np

def detect_objects(detector, image):
    """
    Detect objects in an image using Hugging Face models.
    
    Args:
        detector: HuggingFaceDetector instance
        image: PIL Image to process
        
    Returns:
        list: List of detected objects with bounding boxes, labels, and scores
    """
    # Prepare the image for the model
    inputs = detector.processor(images=image, return_tensors="pt")
    inputs = {k: v.to(detector.device) for k, v in inputs.items()}
    
    # Run the model prediction
    with torch.no_grad():
        outputs = detector.model(**inputs)
    
    # Convert outputs to usable format
    target_sizes = torch.tensor([image.size[::-1]]).to(detector.device)
    results = detector.processor.post_process_object_detection(
        outputs, 
        target_sizes=target_sizes,
        threshold=detector.detection_threshold
    )[0]
    
    # Extract and prepare the results
    detected_objects = []
    
    # Track confidence scores by label for later comparison
    label_confidence = {}
    
    # Extract detected objects
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label_name = detector.model.config.id2label[label.item()]
        confidence = score.item()
        
        # Process the bounding box
        x1, y1, x2, y2 = box.tolist()
        
        # Track highest confidence for each label
        if label_name not in label_confidence or confidence > label_confidence[label_name]:
            label_confidence[label_name] = confidence
        
        # Create a detection object
        detection = {
            "box": [x1, y1, x2, y2],
            "label": label_name,
            "score": confidence
        }
        detected_objects.append(detection)
    
    # Find additional tableware objects that might be missed
    tableware_objects = find_missing_tableware(detector, image, detected_objects)
    if tableware_objects:
        detected_objects.extend(tableware_objects)
    
    # If no objects detected, return an empty list
    if not detected_objects:
        print("No objects detected in the image")
        return []
    
    # Sort by confidence score (highest first)
    detected_objects.sort(key=lambda x: x["score"], reverse=True)
    
    return detected_objects

def find_missing_tableware(detector, image, detected_objects):
    """
    Attempt to find typical tableware items (plates, bowls) using heuristic methods
    when the model fails to detect them.
    
    Args:
        detector: HuggingFaceDetector instance
        image: PIL Image object
        detected_objects: List of already detected objects
        
    Returns:
        list: Additional detected tableware objects
    """
    from food_extraction.utils.image_utils import calculate_iou
    
    # Convert PIL image to OpenCV format for processing
    cv_image = np.array(image)
    cv_image = cv_image[:, :, ::-1].copy()  # RGB to BGR
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    
    # Get image dimensions
    height, width = gray.shape
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Use Canny edge detector to find edges
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours in the edge map
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    additional_objects = []
    
    # Only consider the top N contours to avoid processing noise
    top_n = 10
    for i, contour in enumerate(contours[:top_n]):
        # Skip very small contours
        area = cv2.contourArea(contour)
        if area < (width * height * 0.01):  # Skip if less than 1% of image
            continue
            
        # Get the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Skip if the aspect ratio is too extreme (not likely to be a plate)
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio > 3 or aspect_ratio < 0.33:
            continue
            
        # Check if this contour significantly overlaps with any existing detection
        box = [float(x), float(y), float(x + w), float(y + h)]
        
        is_overlapping = False
        for obj in detected_objects:
            iou = calculate_iou(box, obj["box"])
            if iou > 0.3:  # If IoU > 30%, consider it an overlap
                is_overlapping = True
                break
                
        if is_overlapping:
            continue
            
        # Analyze contour shape to identify circular/oval objects (likely plates/bowls)
        shape_match = False
        
        # Calculate circularity: 4*pi*area/perimeter^2 (1.0 is a perfect circle)
        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            
            # Plates/bowls are somewhat circular (but not perfect circles)
            if 0.5 < circularity < 0.9:
                shape_match = True
                
        if shape_match:
            # Add as a potential plate/bowl
            additional_objects.append({
                "box": box,
                "label": "plate",  # Default label for heuristic detections
                "score": 0.5,      # Moderate confidence for heuristic detection
                "from_heuristic": True  # Mark as coming from heuristic detection
            })
    
    return additional_objects 