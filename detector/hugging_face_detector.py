#!/usr/bin/env python
import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
from transformers import DetrImageProcessor, DetrForObjectDetection
import warnings
from dotenv import load_dotenv
from openai import OpenAI

# Импорт из других модулей
from food_extraction.ai_services.gpt_service import select_object_with_gpt, identify_main_dish
from food_extraction.ai_services.stability_ai import remove_background_with_stability, extend_image_with_stability
from food_extraction.detector.object_detection import detect_objects, find_missing_tableware
from food_extraction.detector.visualization import visualize_detections
from food_extraction.utils.image_utils import save_result as utils_save_result, calculate_iou

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
                dish_description = identify_main_dish(self.openai_client, image_path)
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
            detected_objects = detect_objects(self, image)
            
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
                visualize_detections(image, detected_objects, "debug_detections.png")
                print("Saved detection visualization as debug_detections.png")
            
            # Step 3: Use GPT-4o to select the object that best matches the prompt
            selected_object = select_object_with_gpt(self.openai_client, text_prompt, detected_objects, is_auto_mode=auto_mode)
            
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
            x1_tight = x1 + (width * inward_percent)
            y1_tight = y1 + (height * inward_percent)
            x2_tight = x2 - (width * inward_percent)
            y2_tight = y2 - (height * inward_percent)
            
            # Ensure we don't crop too tight (minimum size check)
            min_size = 50  # minimum size in pixels
            if x2_tight - x1_tight < min_size or y2_tight - y1_tight < min_size:
                print("Crop would be too small with inward adjustment, using original crop")
                crop_box = (x1, y1, x2, y2)
            else:
                crop_box = (x1_tight, y1_tight, x2_tight, y2_tight)
            
            # Round values to integers for cropping
            crop_box = tuple(map(int, crop_box))
            
            # Save crop box for debugging if requested
            if self.debug_mode:
                debug_img = image.copy()
                debug_draw = cv2.cvtColor(np.array(debug_img), cv2.COLOR_RGB2BGR)
                cv2.rectangle(debug_draw, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (0, 255, 0), 2)
                cv2.imwrite("debug_cropped.png", debug_draw)
                print("Saved crop box visualization as debug_cropped.png")
            
            # Crop the image
            cropped_image = image.crop(crop_box)
            
            # Return the cropped image
            return cropped_image
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None 

    def save_result(self, result_image, output_path):
        """
        Save a PIL Image to the specified output path.
        
        Args:
            result_image: PIL Image to save
            output_path: Path to save the image
            
        Returns:
            bool: True if successful, False otherwise
        """
        return utils_save_result(result_image, output_path) 