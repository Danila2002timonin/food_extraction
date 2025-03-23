#!/usr/bin/env python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_detections(image, detected_objects, output_path):
    """
    Visualize detected objects on the image with bounding boxes and labels.
    
    Args:
        image: PIL Image object
        detected_objects: List of detected objects with bounding boxes, labels, and scores
        output_path: Path to save the visualization
    """
    # Convert PIL image to numpy array for visualization
    img_viz = np.array(image)
    
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(12, 9))
    
    # Display the image
    ax.imshow(img_viz)
    
    # Generate unique colors for different object classes
    unique_labels = set(obj["label"] for obj in detected_objects)
    colors = plt.cm.hsv(np.linspace(0, 0.9, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))
    
    # Add bounding boxes, labels, and confidence scores
    for obj in detected_objects:
        x1, y1, x2, y2 = obj["box"]
        label = obj["label"]
        score = obj["score"]
        is_heuristic = obj.get("from_heuristic", False)
        
        # Get color for this class
        color = label_to_color[label]
        if is_heuristic:
            # Use a different style for heuristic detections
            edgecolor = tuple(color[:3])
            linewidth = 1
            linestyle = '--'
        else:
            edgecolor = tuple(color[:3])
            linewidth = 2
            linestyle = '-'
        
        # Add a rectangle patch
        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1, 
            linewidth=linewidth, 
            edgecolor=edgecolor, 
            linestyle=linestyle,
            facecolor='none')
        ax.add_patch(rect)
        
        # Add text with class name and confidence score
        ax.text(
            x1, y1 - 5, 
            f"{label} ({score:.2f})", 
            bbox=dict(facecolor=edgecolor, alpha=0.7),
            fontsize=10, color='white')
    
    # Remove axes
    plt.axis('off')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=100)
    plt.close(fig) 