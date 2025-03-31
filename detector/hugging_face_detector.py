#!/usr/bin/env python
import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import io
import traceback
from transformers import (
    DetrImageProcessor,
    DetrForObjectDetection,
    AutoFeatureExtractor,
    AutoModelForObjectDetection,
    YolosImageProcessor,
    YolosForObjectDetection
)
import warnings
from dotenv import load_dotenv
from openai import OpenAI

# Импорт из других модулей
from ai_services.gpt_service import select_object_with_gpt, identify_main_dish
from ai_services.stability_ai import remove_background_with_stability, extend_image_with_stability
from detector.object_detection import detect_objects, find_missing_tableware
from detector.visualization import visualize_detections
from utils.image_utils import save_result as utils_save_result, calculate_iou

warnings.filterwarnings("ignore")

class HuggingFaceDetector:
    def __init__(self, api_key=None, model_name=None, detection_threshold=0.1):
        """Initialize detector with Hugging Face model and OpenAI"""
        load_dotenv()
        
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        self.detection_threshold = detection_threshold
        
        self.model_name = model_name or "facebook/detr-resnet-50"
        
        self.load_model()
        self.debug_mode = False
        
    def load_model(self):
        """Load the specified object detection model"""
        if "detr" in self.model_name.lower():
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "detr"
        elif "faster-rcnn" in self.model_name.lower():
            self.processor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = AutoModelForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "faster-rcnn"
        elif "yolos" in self.model_name.lower():
            self.processor = YolosImageProcessor.from_pretrained(self.model_name)
            self.model = YolosForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "yolos"
        elif "dino" in self.model_name.lower():
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "dino"
        else:
            self.processor = DetrImageProcessor.from_pretrained(self.model_name)
            self.model = DetrForObjectDetection.from_pretrained(self.model_name)
            self.model_type = "detr"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        
    def process_image(self, image_path, text_prompt=None, debug=False, auto_mode=False):
        """Process image to detect objects and select the best match"""
        self.debug_mode = debug
        
        try:
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
                
            try:
                image = Image.open(image_path)
                
                if not hasattr(image, 'mode'):
                    raise ValueError("Invalid image format")
                    
                image.load()
                
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            except Exception as img_error:
                try:
                    with open(image_path, 'rb') as f:
                        img_data = f.read()
                    image = Image.open(io.BytesIO(img_data)).convert('RGB')
                except Exception as repair_error:
                    raise ValueError(f"Could not process image at {image_path}")
            
            detected_objects = detect_objects(self, image)
            
            if not detected_objects:
                return None
            
            if self.debug_mode:
                visualize_detections(image, detected_objects, "debug_detections.png")
            
            selected_object = None
            
            if auto_mode:
                dish_info = identify_main_dish(self.openai_client, image_path)
                
                if dish_info:
                    from ai_services.semantic_matching import semantic_object_matching
                    selected_object = semantic_object_matching(self.openai_client, dish_info, detected_objects)
                    
                    if not selected_object:
                        dish_prompt = f"{dish_info['description']} in {dish_info['container']}"
                        selected_object = select_object_with_gpt(self.openai_client, dish_prompt, detected_objects, is_auto_mode=True)
                else:
                    return None
                    
            elif text_prompt:
                selected_object = select_object_with_gpt(self.openai_client, text_prompt, detected_objects, is_auto_mode=False)
                
                if not selected_object:
                    dish_terms = text_prompt.lower().split()
                    container_terms = ["bowl", "plate", "glass", "cup"]
                    
                    found_container = next((term for term in dish_terms if term in container_terms), None)
                    
                    if found_container:
                        dish_info = {
                            "dish": text_prompt,
                            "container": found_container,
                            "description": text_prompt
                        }
                        
                        from ai_services.semantic_matching import semantic_object_matching
                        selected_object = semantic_object_matching(self.openai_client, dish_info, detected_objects)
            else:
                return None
            
            if selected_object is None:
                return None
            
            x1, y1, x2, y2 = selected_object['box']
            
            img_width, img_height = image.size
            
            is_composite_object = "_dish" in selected_object["label"] or "_cluster" in selected_object["label"]
            
            if is_composite_object:
                if "_dish" in selected_object["label"]:
                    expansion_ratio = 0.10
                else:
                    expansion_ratio = 0.15
                
                width = x2 - x1
                height = y2 - y1
                
                x1_expanded = max(0, x1 - width * expansion_ratio)
                y1_expanded = max(0, y1 - height * expansion_ratio)
                x2_expanded = min(img_width, x2 + width * expansion_ratio)
                y2_expanded = min(img_height, y2 + height * expansion_ratio)
                
                crop_box = (x1_expanded, y1_expanded, x2_expanded, y2_expanded)
            else:
                width, height = x2 - x1, y2 - y1
                
                if selected_object["label"].lower() in ["plate", "bowl"]:
                    inward_percent = 0.03
                    
                    x1_tight = x1 + (width * inward_percent)
                    y1_tight = y1 + (height * inward_percent)
                    x2_tight = x2 - (width * inward_percent)
                    y2_tight = y2 - (height * inward_percent)
                    
                    min_size = 50
                    if x2_tight - x1_tight < min_size or y2_tight - y1_tight < min_size:
                        crop_box = (x1, y1, x2, y2)
                    else:
                        crop_box = (x1_tight, y1_tight, x2_tight, y2_tight)
                else:
                    crop_box = (x1, y1, x2, y2)
            
            crop_box = tuple(map(int, crop_box))
            
            if self.debug_mode:
                debug_img = image.copy()
                debug_draw = cv2.cvtColor(np.array(debug_img), cv2.COLOR_RGB2BGR)
                cv2.rectangle(debug_draw, (crop_box[0], crop_box[1]), (crop_box[2], crop_box[3]), (0, 255, 0), 2)
                cv2.imwrite("debug_cropped.png", debug_draw)
            
            cropped_image = image.crop(crop_box)
            
            return cropped_image
            
        except Exception as e:
            if self.debug_mode:
                traceback.print_exc()
            return None 

    def save_result(self, result_image, output_path):
        """Save image to specified output path"""
        result_image.save(output_path) 