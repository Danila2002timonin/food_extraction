#!/usr/bin/env python
import os
import sys
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import base64
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from openai import OpenAI
from dotenv import load_dotenv
import warnings
import requests
import time
warnings.filterwarnings("ignore")

class HuggingFaceDetector:
    def __init__(self, api_key=None, model_name=None, detection_threshold=0.1):
        """
        Initialize the detector with Hugging Face model and OpenAI GPT-4o
        
        Args:
            api_key (str, optional): OpenAI API key
            model_name (str, optional): Hugging Face model name for object detection
            detection_threshold (float, optional): Confidence threshold for detection
        """
        # Load from .env file first
        load_dotenv()
        
        # Get OpenAI API key
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            print("Warning: No OpenAI API key provided. Please set the OPENAI_API_KEY environment variable or in .env file.")
            raise ValueError("OpenAI API key is required. Please provide it as an argument or set it in your environment variables.")
        
        # Initialize OpenAI client
        print("Initializing OpenAI client...")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        print("OpenAI client initialized successfully")
        
        # Set confidence threshold for detection
        self.detection_threshold = detection_threshold
        print(f"Using detection threshold: {self.detection_threshold}")
        
        # Set the model name (with default fallback)
        self.model_name = model_name or "facebook/detr-resnet-50"
        print(f"Using model: {self.model_name}")
        
        # Initialize the Hugging Face model
        self.load_model()
        
        self.debug_mode = False
        
    def load_model(self):
        """Load the specified object detection model"""
        print(f"Loading Hugging Face model: {self.model_name}...")
        
        # Different model loading based on architecture type
        if "detr" in self.model_name.lower():
            # DETR model family
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "detr"
        elif "faster-rcnn" in self.model_name.lower():
            # Faster R-CNN model family
            from transformers import AutoFeatureExtractor, AutoModelForObjectDetection
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "faster-rcnn"
        elif "yolos" in self.model_name.lower():
            # YOLOS model family
            from transformers import YolosImageProcessor, YolosForObjectDetection
            self.processor = YolosImageProcessor.from_pretrained(self.model_name)
            self.model = YolosForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "yolos"
        elif "dino" in self.model_name.lower():
            # DINO (DETR improved) model family
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "dino"
        else:
            # Default to DETR model
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "detr"
        
        # Set device (GPU if available, else CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
        print(f"Model loaded successfully. Using device: {self.device}")
        
    def process_image(self, image_path, text_prompt=None, debug=False, auto_mode=False):
        """
        Process the image to detect objects, then use GPT-4o to select the 
        object that best matches the prompt.
        
        Args:
            image_path (str): Path to the input image
            text_prompt (str, optional): Text description of what to find (not needed in auto mode)
            debug (bool): Whether to save debug visualizations
            auto_mode (bool): Whether to automatically identify the main dish
            
        Returns:
            PIL.Image: Cropped image of the detected object
        """
        self.debug_mode = debug
        print(f"Processing image: {image_path}")
        
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
                
            print(f"Opening image file: {os.path.abspath(image_path)}")
            # Try to open the image with robust error handling
            try:
                image = Image.open(image_path)
                
                # Validate the image was loaded properly
                if not hasattr(image, 'mode'):
                    raise ValueError("Invalid image format")
                    
                # Force load the image data (checking for corruption)
                image.load()
                
                # Ensure the image is in RGB format for consistent processing
                if image.mode != 'RGB':
                    print(f"Converting image from {image.mode} to RGB format")
                    image = image.convert('RGB')
            except Exception as img_error:
                print(f"Error loading image: {img_error}")
                print("Attempting to repair/convert the image...")
                # Try to repair/reload the image
                try:
                    # Alternative loading approach
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                    import io
                    image = Image.open(io.BytesIO(img_data)).convert('RGB')
                    print("Successfully repaired image format")
                except Exception as repair_error:
                    print(f"Failed to repair image: {repair_error}")
                    raise ValueError(f"Could not process image at {image_path}. The image may be corrupted or in an unsupported format.")
            
            # In auto mode, use GPT-4o to identify the main dish
            if auto_mode:
                print("Auto mode: Using GPT-4o to identify the main dish...")
                dish_description = self.identify_main_dish(image_path)
                if dish_description:
                    print(f"Using GPT-4o identified dish as prompt: \"{dish_description}\"")
                    text_prompt = dish_description
                else:
                    print("Failed to identify main dish. Please provide a text prompt.")
                    return None
            elif not text_prompt:
                # If not in auto mode and no prompt provided
                print("No text prompt provided. Please specify a prompt or use auto mode.")
                return None
            
            # Step 1: Detect all objects in the image
            detected_objects = self.detect_objects(image)
            
            if not detected_objects:
                print("No objects detected in the image")
                return None
            
            # Print detected objects
            print(f"\nDetected {len(detected_objects)} objects:")
            for i, obj in enumerate(detected_objects):
                source = "(heuristic)" if obj.get("from_heuristic", False) else ""
                print(f"{i}: {obj['label']} (confidence: {obj['score']:.2f}) {source}")
            
            # Step 2: Save detections for debugging
            if self.debug_mode:
                self.visualize_detections(image, detected_objects, "debug_detections.png")
                print("Saved detection visualization as debug_detections.png")
            
            # Step 3: Use GPT-4o to select the object that best matches the prompt
            selected_object = self.select_object_with_gpt(text_prompt, detected_objects, is_auto_mode=auto_mode)
            
            if selected_object is None:
                print("\nCould not match any detected object with the prompt")
                return None
                
            print(f"\nSelected object: {selected_object['label']} at {selected_object['box']}")
            
            # Step 4: Crop the image based on the selected object
            x1, y1, x2, y2 = selected_object['box']
            
            # Make the crop even tighter by moving edges inward to crop borders of the dish
            width, height = x2 - x1, y2 - y1
            
            # Calculate inward crop percentage (adjust as needed)
            inward_percent = 0.05  # 5% inward from each edge
            
            # Apply tighter cropping by moving edges inward
            x1_tight = x1 + int(width * inward_percent)
            y1_tight = y1 + int(height * inward_percent)
            x2_tight = x2 - int(width * inward_percent)
            y2_tight = y2 - int(height * inward_percent)
            
            # Ensure we have a valid crop area (minimum 10x10 pixels)
            if (x2_tight - x1_tight) < 10 or (y2_tight - y1_tight) < 10:
                print("Warning: Tighter crop area too small, using original bounds")
                x1_tight, y1_tight, x2_tight, y2_tight = x1, y1, x2, y2
            
            # Ensure we stay within image boundaries
            x1_tight = max(0, x1_tight)
            y1_tight = max(0, y1_tight)
            x2_tight = min(image.width, x2_tight)
            y2_tight = min(image.height, y2_tight)
            
            print(f"Original crop: {x1},{y1},{x2},{y2}")
            print(f"Tighter crop: {x1_tight},{y1_tight},{x2_tight},{y2_tight}")
            
            cropped_image = image.crop((x1_tight, y1_tight, x2_tight, y2_tight))
            
            # Save cropped image for debugging
            if self.debug_mode:
                cropped_image.save("debug_cropped.png")
                print("Saved cropped image as debug_cropped.png")
            
            return cropped_image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def detect_objects(self, image):
        """
        Detect objects in the image using the loaded model
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            list: List of detected objects with label, score, and box coordinates
        """
        try:
            # Normalize image format - ensure we're always working with RGB (no alpha channel)
            if image.mode != 'RGB':
                print(f"Converting image from {image.mode} to RGB format")
                image = image.convert('RGB')
                
            # Prepare image for the model
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Perform inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Convert outputs based on model type
            detected_objects = []
            
            # Different processing based on model architecture
            if self.model_type in ["detr", "dino"]:
                # Process DETR/DINO model outputs
                target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
                results = self.processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=self.detection_threshold
                )[0]
                
                # Extract results
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    # Convert all box coordinates to regular Python ints (not numpy int64)
                    box = [int(round(i)) for i in box.tolist()]
                    
                    # Get the label name from the model's config
                    label_name = self.model.config.id2label[label.item()]
                    
                    # Skip background class (usually "dining table" or similar)
                    if label_name.lower() in ["dining table", "table", "background"]:
                        continue
                    
                    detected_objects.append({
                        "label": label_name,
                        "score": float(score.item()),  # Convert to Python float
                        "box": box  # [x1, y1, x2, y2]
                    })
                
            elif self.model_type in ["faster-rcnn", "yolos"]:
                # Process Faster R-CNN or YOLOS outputs
                target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
                results = self.processor.post_process_object_detection(
                    outputs, target_sizes=target_sizes, threshold=self.detection_threshold
                )[0]
                
                # Extract results
                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    # Convert all box coordinates to regular Python ints (not numpy int64)
                    box = [int(round(i)) for i in box.tolist()]
                    
                    # Get the label name from the model's config
                    label_name = self.model.config.id2label[label.item()]
                    
                    # Skip background class
                    if label_name.lower() in ["dining table", "table", "background"]:
                        continue
                        
                    detected_objects.append({
                        "label": label_name,
                        "score": float(score.item()),  # Convert to Python float
                        "box": box  # [x1, y1, x2, y2]
                    })
            
            # If we didn't detect any objects with the current threshold, try a lower threshold as fallback
            if not detected_objects and self.detection_threshold > 0.05:
                print(f"No objects detected with threshold {self.detection_threshold}. Trying with a lower threshold (0.05)...")
                
                # Temporarily lower the threshold
                original_threshold = self.detection_threshold
                self.detection_threshold = 0.05
                
                # Retry detection
                detected_objects = self.detect_objects(image)
                
                # Restore original threshold
                self.detection_threshold = original_threshold
            
            # Apply additional heuristics to find missing objects
            if self.debug_mode:
                print(f"Initial detection found {len(detected_objects)} objects")
                
            # Attempt to find cups/glasses that might have been missed
            detected_objects = self.find_missing_tableware(image, detected_objects)
            
            return detected_objects
            
        except Exception as e:
            print(f"Error detecting objects: {e}")
            # Return empty list on error to allow the process to continue
            return []
    
    def find_missing_tableware(self, image, detected_objects):
        """
        Additional heuristics to find commonly missed tableware items
        
        Args:
            image (PIL.Image): Input image
            detected_objects (list): Already detected objects
            
        Returns:
            list: Updated list of detected objects
        """
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
                    print(f"Limiting heuristic detections to {max_circles} items (from {len(circles)})")
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
                        iou = self.calculate_iou(new_box, box)
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
        except Exception as e:
            print(f"Error in heuristic detection: {e}")
        
        return detected_objects
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou
    
    def visualize_detections(self, image, detected_objects, output_path):
        """Visualize detected objects with bounding boxes"""
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)
        
        # Add bounding boxes and labels
        for obj in detected_objects:
            x1, y1, x2, y2 = obj['box']
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            # Add label with confidence score
            label_text = f"{obj['label']} ({obj['score']:.2f})"
            ax.text(x1, y1-10, label_text, color='white', fontsize=12, 
                    bbox=dict(facecolor='red', alpha=0.8))
        
        plt.savefig(output_path)
        plt.close()
    
    def select_object_with_gpt(self, text_prompt, detected_objects, is_auto_mode=False):
        """
        Use GPT-4o to select the object that best matches the prompt
        
        Args:
            text_prompt (str): Text description of what to find
            detected_objects (list): List of detected objects
            is_auto_mode (bool): Whether we're in auto mode (affects prompt)
            
        Returns:
            dict or None: Selected object or None if no match
        """
        # Limit the number of objects to avoid overwhelming GPT
        # First, sort by score (descending)
        sorted_objects = sorted(detected_objects, key=lambda x: x.get("score", 0), reverse=True)
        
        # Only keep top 20 objects
        max_objects = 20
        if len(sorted_objects) > max_objects:
            print(f"Limiting objects sent to GPT to top {max_objects} (from {len(sorted_objects)})")
            objects_for_gpt = sorted_objects[:max_objects]
        else:
            objects_for_gpt = sorted_objects
        
        # Clean the objects for JSON serialization
        clean_objects = []
        for obj in objects_for_gpt:
            clean_obj = {
                "label": str(obj.get("label", "")),
                "score": float(obj.get("score", 0.0)),
                "box": [int(coord) for coord in obj.get("box", [0, 0, 0, 0])],
                "from_heuristic": bool(obj.get("from_heuristic", False))
            }
            clean_objects.append(clean_obj)
        
        # Prepare the prompt for GPT-4o - with improved instructions for semantic matching
        auto_instruction = ""
        if is_auto_mode:
            auto_instruction = """
            IMPORTANT: Since I'm in auto mode where I've already analyzed the image and provided 
            a description of the main dish, you should prioritize objects that match this description,
            even if they're not a perfect match to labeled objects. Be extremely generous in matching,
            as the object detection labels are limited and may not perfectly match the actual dish.
            """
            
        prompt = f"""
        I have detected the following objects in an image:
        
        {json.dumps(clean_objects, indent=2)}
        
        Based on the user's request: "{text_prompt}", which object should I select?
        
        {auto_instruction}
        
        IMPORTANT: Be flexible with matching. For example:
        - If the user asks for a 'dish' or 'plate with food', objects like 'bowl', 'sandwich', etc. could match
        - If the user asks for a 'drink', objects like 'cup', 'bottle', etc. could match
        - If no perfect match exists, select the object that seems most semantically related to the request
        
        However, if there is truly no object that could reasonably match the user's request, indicate that with -1.
        
        Please provide your answer as a JSON object with the following format:
        {{
          "selected_index": (index of the selected object, or -1 if none could possibly match),
          "reasoning": "detailed explanation for why this object was selected or why no match was possible",
          "confidence": (a number from 0-1 indicating how confident you are in this match)
        }}
        
        The word "json" must be included in your response for proper processing.
        """
        
        try:
            # Call the OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that helps with image analysis. Your goal is to find the most relevant object that matches the user's request, even if it's not a perfect match."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Extract the response
            response_text = response.choices[0].message.content
            print(f"GPT-4o response: {response_text}")
            
            # Parse the JSON response
            try:
                gpt_response = json.loads(response_text)
                selected_index = gpt_response.get("selected_index", -1)
                confidence = gpt_response.get("confidence", 0.0)
                
                # Use a lower confidence threshold in auto mode
                min_confidence = 0.15 if is_auto_mode else 0.3
                
                if selected_index >= 0 and selected_index < len(objects_for_gpt) and confidence > min_confidence:
                    print(f"Match found with confidence: {confidence}")
                    # Return the original object from the full list
                    return objects_for_gpt[selected_index]
                else:
                    # No matching object found or confidence too low
                    print(f"No sufficient match found (confidence: {confidence if selected_index >= 0 else 0.0})")
                    return None
                    
            except json.JSONDecodeError:
                print("Failed to parse GPT-4o response as JSON")
                return None
                
        except Exception as e:
            print(f"Error with GPT-4o selection: {e}")
            return None
    
    def save_result(self, result_image, output_path):
        """Save the resulting image to the specified path"""
        result_image.save(output_path)
        print(f"Result saved to {output_path}")
        print(f"Output saved to: {os.path.abspath(output_path)}")

    def remove_background_with_stability(self, image_path, output_path="extracted_object_nobg.png"):
        """
        Remove background from an image using Stability AI API
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the output image
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Load .env file to ensure we have the latest API key
        from dotenv import load_dotenv
        load_dotenv(override=True)
        
        # Load API key from environment
        api_key = os.getenv("STABLE_DIFFUSION_API_KEY")
        
        if not api_key:
            print("ERROR: No Stability AI API key found. Please set STABLE_DIFFUSION_API_KEY in .env file")
            return False
            
        try:
            print(f"Removing background using Stability AI...")
            
            # Check image dimensions before sending
            from PIL import Image
            img = Image.open(image_path)
            width, height = img.size
            total_pixels = width * height
            
            print(f"Input image size: {width}x{height} = {total_pixels} pixels")
            
            # Stability AI has a limit of 4,194,304 pixels (2048x2048)
            MAX_PIXELS = 4_194_304
            
            # If image is too large, resize it
            if total_pixels > MAX_PIXELS:
                print(f"Image too large for Stability AI API (limit {MAX_PIXELS} pixels). Resizing...")
                
                # Calculate the scaling factor to fit within the limit
                scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                
                # Resize the image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save to a temporary file
                temp_path = os.path.splitext(image_path)[0] + "_resized_temp.png"
                img.save(temp_path)
                print(f"Resized image to {new_width}x{new_height} = {new_width * new_height} pixels")
                
                # Use the resized image for background removal
                use_path = temp_path
            else:
                use_path = image_path
            
            # Print API key for debugging (first/last 5 chars only)
            print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
            
            response = requests.post(
                f"https://api.stability.ai/v2beta/stable-image/edit/remove-background",
                headers={
                    "authorization": f"Bearer {api_key}",
                    "accept": "image/*"
                },
                files={
                    "image": open(use_path, "rb")
                },
                data={
                    "output_format": "png"
                },
            )
            
            print(f"API response status code: {response.status_code}")
            
            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    file.write(response.content)
                print(f"Background removed successfully. Saved to {output_path}")
                
                # Clean up temporary file if it was created
                if total_pixels > MAX_PIXELS:
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
                return True
            else:
                print(f"Error removing background: {response.status_code}")
                print(response.text)
                
                # Try alternative approach - Stability AI has multiple API endpoints
                print("Attempting alternative API endpoint...")
                try:
                    response = requests.post(
                        f"https://api.stability.ai/v1/generation/stable-image/remove-background",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Accept": "image/*"
                        },
                        files={
                            "image": open(use_path, "rb")
                        },
                        data={
                            "output_format": "png"
                        },
                    )
                    
                    print(f"Alternative API response status code: {response.status_code}")
                    
                    if response.status_code == 200:
                        with open(output_path, 'wb') as file:
                            file.write(response.content)
                        print(f"Background removed successfully with alternative endpoint. Saved to {output_path}")
                        
                        # Clean up temporary file if it was created
                        if total_pixels > MAX_PIXELS:
                            try:
                                os.remove(temp_path)
                            except:
                                pass
                                
                        return True
                except Exception as alt_error:
                    print(f"Alternative endpoint also failed: {alt_error}")
                
                return False
                
        except Exception as e:
            print(f"Error using Stability AI for background removal: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def extend_image_with_stability(self, image_path, output_path="extended_image.png", left=0, right=0, up=0, down=0):
        """
        Extend an image using Stability AI's outpainting
        
        Args:
            image_path (str): Path to the input image
            output_path (str): Path to save the output image
            left (int): Pixels to extend on left
            right (int): Pixels to extend on right
            up (int): Pixels to extend on top
            down (int): Pixels to extend on bottom
            
        Returns:
            bool: True if successful, False otherwise
        """
        import requests
        import os
        from PIL import Image
        from dotenv import load_dotenv
        
        # Load .env file to ensure we have the latest API key
        load_dotenv(override=True)
        
        # Load API key from environment
        api_key = os.getenv("STABLE_DIFFUSION_API_KEY")
        
        if not api_key:
            print("ERROR: No Stability AI API key found. Please set STABLE_DIFFUSION_API_KEY in .env file")
            return False
            
        try:
            print(f"Extending image using Stability AI outpainting...")
            
            # Add clear log about which image is being used for outpainting
            print(f"INPUT IMAGE FOR OUTPAINTING: {os.path.basename(image_path)}")
            if "_nobg" in image_path:
                print("✓ Using background-removed image for outpainting (recommended)")
            else:
                print("⚠ Using image with background for outpainting (consider using --remove-bg first)")
                
            print(f"Extensions: left={left}, right={right}, up={up}, down={down}")
            
            # Check image size before sending
            img = Image.open(image_path)
            img_width, img_height = img.size
            total_pixels = img_width * img_height
            print(f"Input image size: {img_width}x{img_height} = {total_pixels} pixels")
            
            # Stability AI has a limit of around 4 million pixels
            MAX_PIXELS = 4_194_304  # 2048x2048
            
            # If image is too large, resize it
            if total_pixels > MAX_PIXELS:
                print(f"Image too large for Stability AI API (limit {MAX_PIXELS} pixels). Resizing...")
                
                # Calculate the scaling factor to fit within the limit
                scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
                new_width = int(img_width * scale_factor)
                new_height = int(img_height * scale_factor)
                
                # Resize the image
                img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Save to a temporary file
                temp_path = os.path.splitext(image_path)[0] + "_resized_temp.png"
                img.save(temp_path)
                print(f"Resized image to {new_width}x{new_height} = {new_width * new_height} pixels")
                
                # Use the resized image for outpainting
                use_path = temp_path
            else:
                use_path = image_path
            
            # Prepare the data payload - only include non-zero extensions
            data = {"output_format": "png"}  # Use PNG instead of WEBP for better compatibility
            
            if left > 0:
                data["left"] = left
            if right > 0:
                data["right"] = right
            if up > 0:
                data["up"] = up
            if down > 0:
                data["down"] = down
                
            # Make the API request exactly as documented
            print(f"Sending outpainting request to Stability AI API...")
            
            response = requests.post(
                "https://api.stability.ai/v2beta/stable-image/edit/outpaint",
                headers={
                    "authorization": f"Bearer {api_key}",
                    "accept": "image/*"
                },
                files={
                    "image": open(use_path, "rb")
                },
                data=data
            )
            
            print(f"API response status code: {response.status_code}")
            
            if response.status_code == 200:
                with open(output_path, 'wb') as file:
                    file.write(response.content)
                print(f"Image extended successfully. Saved to {output_path}")
                
                # Clean up temporary file if it was created
                if total_pixels > MAX_PIXELS:
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                        
                return True
            else:
                error_info = response.json() if response.content else {"error": "Unknown error"}
                print(f"Error extending image: {response.status_code}")
                print(error_info)
                
                if response.status_code == 404 and "internal_not_found" in str(error_info):
                    print("\nSUGGESTION: The outpainting feature may not be available with your current Stability AI API plan.")
                    print("The background removal feature is working, but outpainting requires a different access level.")
                    print("Please check your API key permissions or contact Stability AI for more information.")
                    
                return False
                
        except Exception as e:
            print(f"Error using Stability AI for image extension: {e}")
            import traceback
            traceback.print_exc()
            return False

    def identify_main_dish(self, image_path):
        """
        Use GPT-4o Vision to identify and describe the main dish in the image
        
        Args:
            image_path (str): Path to the input image
            
        Returns:
            str: Description of the main dish in the image that aligns with common object detection labels
        """
        print("Using GPT-4o Vision to identify the main dish in the image...")
        
        try:
            # Prepare the image for OpenAI - with robust error handling
            try:
                with open(image_path, "rb") as image_file:
                    image_data = image_file.read()
                
                # Verify image data is valid before sending to OpenAI
                try:
                    from PIL import Image
                    import io
                    Image.open(io.BytesIO(image_data)).verify()
                except Exception as verify_error:
                    print(f"Warning: Image verification failed: {verify_error}")
                    print("Attempting to fix the image format...")
                    # Try to repair the image by loading and resaving
                    img = Image.open(image_path).convert('RGB')
                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG")
                    image_data = buffer.getvalue()
                    print("Image repair completed")
                
                base64_image = base64.b64encode(image_data).decode('utf-8')
            except Exception as e:
                print(f"Error preparing image for OpenAI: {e}")
                return None
            
            # Create a prompt that asks GPT-4o to identify the main dish while using common object detection terms
            prompt = """
            Look at this image carefully and identify the MAIN food dish or plate visible.
            
            IMPORTANT: Please describe the main dish using simple, common terms that would match these potential object detection labels: 
            plate, bowl, cup, sandwich, pizza, cake, hot dog, donut, carrot, broccoli, orange, banana, apple, hamburger.
            
            In your description:
            1. Start with the most general category that fits (e.g., "plate of food", "bowl with soup", "sandwich")
            2. Then include 1-2 key identifying features (color, sauce, main ingredients)
            3. Keep your description simple and aligned with common object categories
            
            Focus ONLY on the main food item/dish - ignore drinks, cutlery, or other elements.
            
            Your description should be concise (1-2 sentences) and use terminology that would help an object detection system identify this dish.
            """
            
            # Call the OpenAI API with the image
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Extract the response
            dish_description = response.choices[0].message.content.strip()
            print(f"GPT-4o identified: {dish_description}")
            
            return dish_description
            
        except Exception as e:
            print(f"Error identifying main dish: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description='Extract objects from images using Hugging Face models and GPT-4o')
    parser.add_argument('--image', required=True, help='Path to input image')
    parser.add_argument('--prompt', help='Text description of the object to extract (not required in auto mode)')
    parser.add_argument('--auto', action='store_true', help='Automatically identify and extract the main dish without requiring a prompt')
    parser.add_argument('--output', default='extracted_object.png', help='Path to save the output image')
    parser.add_argument('--api_key', default=None, help='OpenAI API key (if not set in environment)')
    parser.add_argument('--model', default='facebook/detr-resnet-101', 
                      help='Hugging Face model name for object detection (e.g., facebook/detr-resnet-101, microsoft/table-transformer-detection, facebook/detr-resnet-101)')
    parser.add_argument('--threshold', type=float, default=0.1, 
                      help='Detection confidence threshold (0.0-1.0)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with visualizations')
    
    # New Stability AI features
    parser.add_argument('--remove-bg', action='store_true', help='Use Stability AI to remove background after extraction')
    parser.add_argument('--extend', action='store_true', help='Use Stability AI to extend image after extraction')
    parser.add_argument('--extend-left', type=int, default=0, help='Pixels to extend on the left side')
    parser.add_argument('--extend-right', type=int, default=0, help='Pixels to extend on the right side')
    parser.add_argument('--extend-up', type=int, default=0, help='Pixels to extend on the top')
    parser.add_argument('--extend-down', type=int, default=0, help='Pixels to extend on the bottom')
    parser.add_argument('--stability-api-key', help='Stability AI API key (if not set in environment)')
    
    args = parser.parse_args()
    
    # Print feature information
    print("\n=== Food Extractor with Hugging Face and Stability AI ===")
    print("Features:")
    print("  - Object detection using Hugging Face DETR models")
    print("  - Semantic object selection with GPT-4o")
    print("  - Background removal with Stability AI")
    print("  - Image extension (outpainting) with Stability AI (may require upgraded API access)")
    print("==================================================\n")
    
    # Verify that either a prompt is provided or auto mode is enabled
    if not args.prompt and not args.auto:
        print("Error: You must either provide a prompt with --prompt or use --auto mode")
        return
        
    # Set Stability API key if provided
    if args.stability_api_key:
        os.environ["STABLE_DIFFUSION_API_KEY"] = args.stability_api_key
    
    try:
        if args.auto:
            print(f"Auto mode: Processing main dish from image: {args.image}")
        else:
            print(f"Processing object '{args.prompt}' from image: {args.image}")
            
        detector = HuggingFaceDetector(
            api_key=args.api_key,
            model_name=args.model,
            detection_threshold=args.threshold
        )
        
        # Extract the object from the image
        result = detector.process_image(
            image_path=args.image,
            text_prompt=args.prompt,
            debug=args.debug,
            auto_mode=args.auto
        )
        
        if result:
            # Save the extracted object
            extracted_path = args.output
            detector.save_result(result, extracted_path)
            print(f"Object extraction completed successfully!")
            print(f"Image processing flow: ORIGINAL → EXTRACTED ({os.path.basename(extracted_path)})")
            
            # Apply Stability AI background removal if requested
            if args.remove_bg:
                # Define output path for the no-background image
                nobg_path = os.path.splitext(extracted_path)[0] + "_nobg.png"
                print(f"\n[STEP: BACKGROUND REMOVAL]")
                print(f"Processing image: {os.path.basename(extracted_path)} → {os.path.basename(nobg_path)}")
                success = detector.remove_background_with_stability(extracted_path, nobg_path)
                if success:
                    # Update extracted_path for potential further processing
                    extracted_path = nobg_path
                    print(f"Image processing flow: ORIGINAL → EXTRACTED → NO BACKGROUND ({os.path.basename(extracted_path)})")
                else:
                    print("Background removal failed, continuing with original extraction")
            
            # Apply Stability AI image extension if requested
            if args.extend:
                # Check if any extension values are provided
                if args.extend_left > 0 or args.extend_right > 0 or args.extend_up > 0 or args.extend_down > 0:
                    # Define output path for the extended image
                    extended_path = os.path.splitext(extracted_path)[0] + "_extended.png"
                    print(f"\n[STEP: IMAGE EXTENSION/OUTPAINTING]")
                    
                    # IMPORTANT: Always check if we have a background-removed version available
                    # This ensures we always use the no-bg version for outpainting if it exists
                    nobg_path = os.path.splitext(args.output)[0] + "_nobg.png"
                    source_for_outpainting = extracted_path
                    
                    if os.path.exists(nobg_path) and args.remove_bg:
                        print(f"Background-removed image found, using it for outpainting")
                        source_for_outpainting = nobg_path
                    
                    print(f"Processing image: {os.path.basename(source_for_outpainting)} → {os.path.basename(extended_path)}")
                    success = detector.extend_image_with_stability(
                        source_for_outpainting, 
                        extended_path,
                        left=args.extend_left,
                        right=args.extend_right,
                        up=args.extend_up,
                        down=args.extend_down
                    )
                    if success:
                        print(f"Final output saved to: {os.path.abspath(extended_path)}")
                        print(f"Complete image processing flow: ORIGINAL → EXTRACTED" + 
                              (f" → NO BACKGROUND" if args.remove_bg else "") + 
                              f" → EXTENDED ({os.path.basename(extended_path)})")
                    else:
                        print(f"Image extension failed, final output is: {os.path.abspath(extracted_path)}")
                else:
                    print("No extension values provided, skipping image extension")
                    print(f"Final output saved to: {os.path.abspath(extracted_path)}")
            else:
                print(f"Final output saved to: {os.path.abspath(extracted_path)}")
            
        else:
            print("Failed to process the image.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 