#!/usr/bin/env python
import numpy as np
import json
import os
from openai import OpenAI

# Load data from JSON files
def load_json_data(filename):
    """Load data from a JSON file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'data', filename)
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return {} if filename.endswith('mapping.json') or filename.endswith('matches.json') else []

# Load dictionaries from JSON files
FOOD_CONTAINER_MAPPING = load_json_data('food_container_mapping.json')
FOOD_ITEMS = load_json_data('food_items.json')
TABLEWARE_ITEMS = load_json_data('tableware_items.json')
FORBIDDEN_MATCHES = load_json_data('forbidden_matches.json')

def get_container_from_mapping(dish_type):
    """Get possible containers for a dish type from the mapping dictionary"""
    # Convert dish_type to lowercase for case-insensitive matching
    dish_type = dish_type.lower()
    
    # Check for direct match
    if dish_type in FOOD_CONTAINER_MAPPING:
        return FOOD_CONTAINER_MAPPING[dish_type]
    
    # Check for partial match
    for key in FOOD_CONTAINER_MAPPING:
        if key in dish_type or dish_type in key:
            return FOOD_CONTAINER_MAPPING[key]
    
    # Default to the most common containers if no match
    return ["plate", "bowl"]

def generate_embeddings(openai_client, texts):
    """Generate embeddings for a list of texts using OpenAI's embedding model"""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        
        embeddings = [data.embedding for data in response.data]
        return embeddings
        
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return None

def is_food_item(label):
    """Check if the given label is a food item"""
    label = label.lower()
    
    # Check against our food items list
    for food in FOOD_ITEMS:
        if food in label or label in food:
            return True
    
    # Check for common food suffixes/patterns
    food_patterns = ["_dish", "_cluster", "food", "dish", "meal"]
    for pattern in food_patterns:
        if pattern in label:
            return True
    
    return False

def is_tableware(label):
    """Check if the given label is tableware"""
    label = label.lower()
    
    # Check against our tableware items list
    for item in TABLEWARE_ITEMS:
        if item == label or f"{item}s" == label:  # Handle plural forms
            return True
    
    return False

def is_forbidden_match(dish_type, label):
    """Check if this label is forbidden for this dish type"""
    dish_type = dish_type.lower()
    label = label.lower()
    
    # Check if there are any forbidden matches for this dish
    for dish_key in FORBIDDEN_MATCHES:
        if dish_key in dish_type or dish_type in dish_key:
            forbidden_items = FORBIDDEN_MATCHES[dish_key]
            for item in forbidden_items:
                if label == item or label.startswith(item):
                    return True
    
    return False

def get_largest_object(objects):
    """Get the largest object by bounding box area"""
    if not objects:
        return None
    
    return max(objects, key=lambda x: (x["box"][2] - x["box"][0]) * (x["box"][3] - x["box"][1]))

def semantic_object_matching(openai_client, dish_info, detected_objects):
    """Match dish description with detected objects using semantic similarity"""
    if not dish_info or not detected_objects:
        return None
    
    dish_type = dish_info.get('dish', '').lower()
    description = dish_info.get('description', '').lower()
    print(f"Matching dish: {dish_type}, description: {description}")
    
    # CRITICAL CHECK: If looking for stuffed cabbage/pepper, NEVER select cups
    if "stuffed" in dish_type and ("cabbage" in dish_type or "pepper" in dish_type):
        print("STRICT RULE: Looking for stuffed cabbage/pepper, will never select cups")
        # Remove all cups from consideration
        objects_without_cups = [obj for obj in detected_objects if not obj["label"].lower() == "cup"]
        
        # If we have any objects left, work with them
        if objects_without_cups:
            print(f"Removed {len(detected_objects) - len(objects_without_cups)} cups from consideration")
            detected_objects = objects_without_cups
        else:
            print("Warning: All detected objects were cups. Will try to find alternatives.")
            # Get the largest non-cup object
            non_cups = [obj for obj in detected_objects if obj["label"].lower() != "cup"]
            if non_cups:
                largest_non_cup = get_largest_object(non_cups)
                print(f"Selecting largest non-cup object: {largest_non_cup['label']}")
                return largest_non_cup
    
    # Filter out forbidden matches for the given dish
    if dish_type:
        filtered_objects = []
        for obj in detected_objects:
            if is_forbidden_match(dish_type, obj["label"]):
                print(f"Excluding {obj['label']} as it's forbidden for {dish_type}")
            else:
                filtered_objects.append(obj)
        
        if filtered_objects:
            detected_objects = filtered_objects
            print(f"Using {len(detected_objects)} objects after filtering forbidden matches")
    
    # First, check for dish objects which represent entire dishes
    dish_objects = [obj for obj in detected_objects if "_dish" in obj["label"]]
    if dish_objects:
        print("Found dish objects, giving them highest priority")
        
        # Extract the base label without "_dish" suffix
        for obj in dish_objects:
            base_label = obj["label"].split("_dish")[0]
            
            # Check if this base label is related to our dish
            if base_label.lower() in description or \
               base_label.lower() in dish_type:
                print(f"Found direct match with dish of {base_label} (merged count: {obj.get('merged_count', 0)})")
                return obj
        
        # Even if no direct match found, prioritize the largest dish
        largest_dish = get_largest_object(dish_objects)
        print(f"Using largest dish of {largest_dish['label'].split('_dish')[0]} (merged count: {largest_dish.get('merged_count', 0)})")
        return largest_dish
    
    # Then, check for cluster objects which represent merged similar objects
    cluster_objects = [obj for obj in detected_objects if "_cluster" in obj["label"]]
    if cluster_objects:
        print("Found merged object clusters, giving them priority")
        
        # Extract the base label without "_cluster" suffix
        for obj in cluster_objects:
            base_label = obj["label"].split("_cluster")[0]
            
            # Check if this base label is related to our dish
            if base_label.lower() in description or \
               base_label.lower() in dish_type:
                print(f"Found direct match with cluster of {base_label} objects (merged count: {obj.get('merged_count', 0)})")
                return obj
        
        # Sort clusters by size (area) and merged count
        for cluster in cluster_objects:
            box = cluster["box"]
            cluster["area"] = (box[2] - box[0]) * (box[3] - box[1])
        
        # Prioritize larger clusters with more merged objects
        sorted_clusters = sorted(
            cluster_objects, 
            key=lambda x: (x.get("merged_count", 0), x.get("area", 0)), 
            reverse=True
        )
        
        largest_cluster = sorted_clusters[0]
        print(f"Using largest cluster of {largest_cluster['label'].split('_cluster')[0]} (merged count: {largest_cluster.get('merged_count', 0)})")
        return largest_cluster
    
    # Separate food items from tableware
    food_objects = []
    tableware_objects = []
    other_objects = []
    
    for obj in detected_objects:
        if is_food_item(obj["label"]):
            food_objects.append(obj)
        elif is_tableware(obj["label"]):
            tableware_objects.append(obj)
        else:
            other_objects.append(obj)
    
    # Print categorization for debugging
    print(f"Categorized objects: {len(food_objects)} food items, {len(tableware_objects)} tableware, {len(other_objects)} other")
    
    # If we have food items and the description is about food, prioritize those
    if food_objects and dish_type:
        print("Prioritizing food items for semantic matching")
        
        # Prepare texts for embedding
        texts = [description]
        for obj in food_objects:
            texts.append(obj["label"].lower())
            
        # Generate embeddings
        embeddings = generate_embeddings(openai_client, texts)
        
        if embeddings:
            # Dish description embedding is the first one
            dish_embedding = embeddings[0]
            
            # Calculate cosine similarity for each object
            for i, obj in enumerate(food_objects):
                obj_embedding = embeddings[i+1]  # +1 because dish description is first
                
                # Cosine similarity between dish description and object label
                similarity = np.dot(dish_embedding, obj_embedding) / (
                    np.linalg.norm(dish_embedding) * np.linalg.norm(obj_embedding)
                )
                
                # Combined score: (object confidence + semantic similarity) / 2
                obj["match_score"] = (obj["score"] + similarity) / 2
                print(f"Food item: {obj['label']}, similarity: {similarity:.2f}, match score: {obj['match_score']:.2f}")
            
            # Sort food objects by match score
            food_objects.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            
            # If we have a good food match (match score > 0.3), use it
            if food_objects and food_objects[0].get("match_score", 0) > 0.3:
                best_match = food_objects[0]
                print(f"Using food item match: {best_match['label']} (match score: {best_match['match_score']:.2f})")
                return best_match
    
    # If no good food match found, look for the specified container
    if 'container' in dish_info and dish_info['container']:
        container_name = dish_info['container'].lower()
        print(f"Looking for specified container: {container_name}")
        
        # First, look for direct matches
        container_matches = [obj for obj in detected_objects if obj["label"].lower() == container_name]
        
        if container_matches:
            # If multiple matches, use the one with the highest confidence
            container_matches.sort(key=lambda x: x["score"], reverse=True)
            best_container = container_matches[0]
            print(f"Using exact container match: {best_container['label']} (confidence: {best_container['score']:.2f})")
            return best_container
    
    # If we're looking for a food item and the only tableware is cup, prefer bowl or sandwich
    if "stuffed" in dish_type and all(obj["label"].lower() == "cup" for obj in tableware_objects):
        print("PREFERENCE: All tableware objects are cups, looking for better alternatives")
        
        # Check if we have a bowl
        bowls = [obj for obj in detected_objects if obj["label"].lower() == "bowl"]
        if bowls:
            bowl = max(bowls, key=lambda x: x["score"])
            print(f"Found a bowl, using it instead of cup: {bowl['label']} (score: {bowl['score']:.2f})")
            return bowl
        
        # Check if we have a sandwich or other food item
        if food_objects:
            food = max(food_objects, key=lambda x: x["score"])
            print(f"Using food item instead of cup: {food['label']} (score: {food['score']:.2f})")
            return food
    
    # Get possible containers from our mapping
    possible_containers = get_container_from_mapping(dish_type)
    print(f"Possible containers from mapping: {possible_containers}")
    
    # Add the container specified by GPT-4o
    if "container" in dish_info and dish_info["container"]:
        possible_containers.append(dish_info["container"].lower())
    
    # As a fallback, try semantic matching with all objects
    print("Trying semantic matching with all detected objects as a fallback")
    
    # For certain dish types, explicitly avoid cups
    avoid_cups = any(kw in dish_type for kw in ["stuffed", "soup", "pasta", "steak", "sandwich"])
    
    # Prepare texts for embedding
    texts = [description]
    eligible_objects = []
    
    for obj in detected_objects:
        # Skip cups for certain dish types
        if avoid_cups and obj["label"].lower() == "cup":
            print(f"Explicitly skipping cup for {dish_type}")
            continue
        
        eligible_objects.append(obj)
        texts.append(obj["label"].lower())
    
    if not eligible_objects:
        print("No eligible objects after filtering cups. Falling back to non-cup objects.")
        non_cups = [obj for obj in detected_objects if obj["label"].lower() != "cup"]
        if non_cups:
            best_non_cup = max(non_cups, key=lambda x: x["score"])
            print(f"Selected best non-cup object: {best_non_cup['label']} (score: {best_non_cup['score']:.2f})")
            return best_non_cup
        
        # If all options are cups but we need something, use the sandwich (with lowest confidence)
        # or the largest cup as a last resort
        if "sandwich" in [obj["label"].lower() for obj in detected_objects]:
            sandwich = [obj for obj in detected_objects if obj["label"].lower() == "sandwich"][0]
            print(f"Desperate fallback: using sandwich despite low confidence: {sandwich['score']:.2f}")
            return sandwich
        
        largest_obj = get_largest_object(detected_objects)
        print(f"Last resort when all options are cups: using largest object: {largest_obj['label']}")
        return largest_obj
    
    # Generate embeddings
    embeddings = generate_embeddings(openai_client, texts)
    
    if embeddings:
        # Dish description embedding is the first one
        dish_embedding = embeddings[0]
        
        # Calculate cosine similarity for each object
        for i, obj in enumerate(eligible_objects):
            obj_embedding = embeddings[i+1]  # +1 because dish description is first
            
            # Cosine similarity between dish description and object label
            similarity = np.dot(dish_embedding, obj_embedding) / (
                np.linalg.norm(dish_embedding) * np.linalg.norm(obj_embedding)
            )
            
            # Adjust match score based on object type
            if is_food_item(obj["label"]):
                # Boost food items
                obj["match_score"] = (obj["score"] + similarity * 1.5) / 2
                print(f"Food item boost: {obj['label']}, adjusted score: {obj['match_score']:.2f}")
            elif is_tableware(obj["label"]):
                # Only use tableware if they're very confident detections or good container matches
                if obj["label"].lower() in possible_containers:
                    obj["match_score"] = (obj["score"] + similarity) / 2
                    print(f"Container match: {obj['label']}, score: {obj['match_score']:.2f}")
                else:
                    # Heavily penalize cups for food items
                    if obj["label"].lower() == "cup" and "stuffed" in dish_type:
                        obj["match_score"] = (obj["score"] * 0.3 + similarity * 0.2) / 2
                        print(f"Cup heavily penalized for stuffed dish: {obj['match_score']:.2f}")
                    else:
                        # Penalize tableware that doesn't match the expected container
                        obj["match_score"] = (obj["score"] * 0.7 + similarity * 0.3) / 2
                        print(f"Non-matching tableware: {obj['label']}, penalized score: {obj['match_score']:.2f}")
            else:
                # Regular objects
                obj["match_score"] = (obj["score"] + similarity) / 2
        
        # Sort eligible objects by match score
        eligible_objects.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        if eligible_objects:
            best_match = eligible_objects[0]
            match_score = best_match.get("match_score", 0)
            
            # Only accept matches with reasonable scores
            if match_score > 0.3:
                print(f"Best semantic match: {best_match['label']} (confidence: {best_match['score']:.2f}, match score: {match_score:.2f})")
                return best_match
            else:
                print(f"Best match score too low ({match_score:.2f}), looking for alternatives")
    
    # If all else fails, look for the largest food-like object
    food_candidates = [obj for obj in detected_objects if is_food_item(obj["label"])]
    if food_candidates:
        # Get the largest food item by area
        largest_food = get_largest_object(food_candidates)
        print(f"Using largest food item: {largest_food['label']}")
        return largest_food
    
    # If still no match, use the largest plate or bowl if available
    plates_and_bowls = [obj for obj in detected_objects if obj["label"].lower() in ["plate", "bowl"]]
    if plates_and_bowls:
        largest_container = get_largest_object(plates_and_bowls)
        print(f"Using largest container: {largest_container['label']}")
        return largest_container
    
    # If all else fails, use the object with highest confidence that isn't a cup
    non_cups = [obj for obj in detected_objects if obj["label"].lower() != "cup"]
    if non_cups:
        highest_conf = max(non_cups, key=lambda x: x["score"])
        print(f"As a last resort, using highest confidence non-cup object: {highest_conf['label']}")
        return highest_conf
    
    # If all else has failed and we only have cups, and there's a sandwhich, select it anyway
    sandwich_objects = [obj for obj in detected_objects if obj["label"].lower() == "sandwich"]
    if sandwich_objects:
        sandwich = sandwich_objects[0]
        print(f"Final attempt: using sandwich despite everything: {sandwich['label']}")
        return sandwich
    
    # Absolute last resort - just use any object with confidence > 0.3
    confident_objects = [obj for obj in detected_objects if obj["score"] > 0.3]
    if confident_objects:
        obj = max(confident_objects, key=lambda x: x["score"])
        print(f"Last resort: using object with confidence > 0.3: {obj['label']}")
        return obj
    
    # If we still have nothing, use the largest object
    if detected_objects:
        largest_obj = get_largest_object(detected_objects)
        print(f"Absolute last resort: using largest object: {largest_obj['label']}")
        return largest_obj
    
    # No match found
    return None 