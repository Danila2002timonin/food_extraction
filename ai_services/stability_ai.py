#!/usr/bin/env python
import os
import time
import json
import requests
import base64
from PIL import Image
from io import BytesIO
import sys
import traceback
from dotenv import load_dotenv

def remove_background_with_stability(image_path, output_path="extracted_object_nobg.png", debug_mode=False):
    """Remove the background from an image using Stability AI's API"""
    try:
        # Load .env file to ensure we have the latest API key
        load_dotenv(override=True)
        
        # Get API key from environment
        api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
        if not api_key:
            return False
        
        # Check image dimensions before sending
        img = Image.open(image_path)
        width, height = img.size
        total_pixels = width * height
        
        # Stability AI has a limit of 4,194,304 pixels (2048x2048)
        MAX_PIXELS = 4_194_304
        
        # If image is too large, resize it
        use_path = image_path
        if total_pixels > MAX_PIXELS:
            # Calculate the scaling factor to fit within the limit
            scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to a temporary file
            temp_path = os.path.splitext(image_path)[0] + "_resized_temp.png"
            img.save(temp_path)
            
            # Use the resized image for background removal
            use_path = temp_path
        
        # Use v2beta endpoint as in food_extractor_huggingface.py
        response = requests.post(
            "https://api.stability.ai/v2beta/stable-image/edit/remove-background",
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
        
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                file.write(response.content)
            
            # Clean up temporary file if it was created
            if total_pixels > MAX_PIXELS:
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            return True
        else:
            # Try alternative approach - Stability AI has multiple API endpoints
            try:
                response = requests.post(
                    "https://api.stability.ai/v1/generation/stable-image/remove-background",
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
                
                if response.status_code == 200:
                    with open(output_path, 'wb') as file:
                        file.write(response.content)
                    
                    # Clean up temporary file if it was created
                    if total_pixels > MAX_PIXELS:
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                            
                    return True
            except:
                pass
            
            return False
            
    except Exception as e:
        if debug_mode:
            traceback.print_exc()
        return False
        
def extend_image_with_stability(image_path, output_path="extended_image.png", left=0, right=0, up=0, down=0, debug_mode=False):
    """Extend an image in specified directions using Stability AI's Outpainting API"""
    try:
        # Load .env file to ensure we have the latest API key
        load_dotenv(override=True)
        
        # Get API key from environment
        api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
        if not api_key:
            return False
        
        # Check image size before sending
        img = Image.open(image_path)
        img_width, img_height = img.size
        total_pixels = img_width * img_height
        
        # Stability AI has a limit of around 4 million pixels
        MAX_PIXELS = 4_194_304  # 2048x2048
        
        # If image is too large, resize it
        use_path = image_path
        if total_pixels > MAX_PIXELS:
            # Calculate the scaling factor to fit within the limit
            scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to a temporary file
            temp_path = os.path.splitext(image_path)[0] + "_resized_temp.png"
            img.save(temp_path)
            
            # Use the resized image for outpainting
            use_path = temp_path
        
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
        
        if response.status_code == 200:
            with open(output_path, 'wb') as file:
                file.write(response.content)
            
            # Clean up temporary file if it was created
            if total_pixels > MAX_PIXELS:
                try:
                    os.remove(temp_path)
                except:
                    pass
                    
            return True
        else:
            # Try the v1 endpoint as fallback
            try:
                # Prepare the alternative payload
                alt_payload = {
                    "image": f"data:image/png;base64,{base64.b64encode(open(use_path, 'rb').read()).decode('utf-8')}",
                    "mask": None,  # No mask needed for full outpainting
                    "width": img_width + left + right,
                    "height": img_height + up + down,
                    "padding_left": left,
                    "padding_right": right,
                    "padding_top": up,
                    "padding_bottom": down,
                    "prompt": "food, dish, plate, high quality, detailed, realistic",
                    "outpainting_mode": "PRECISE",
                    "samples": 1,
                    "cfg_scale": 7,
                    "steps": 30
                }
                
                alt_response = requests.post(
                    "https://api.stability.ai/v1/generation/stable-image/outpaint",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    json=alt_payload
                )
                
                if alt_response.status_code == 200:
                    alt_result = alt_response.json()
                    if 'artifacts' in alt_result and len(alt_result['artifacts']) > 0:
                        extended_image_b64 = alt_result['artifacts'][0]['base64']
                        extended_image_data = base64.b64decode(extended_image_b64)
                        
                        with open(output_path, "wb") as f:
                            f.write(extended_image_data)
                            
                        return True
                    
            except:
                pass
                
            return False
            
    except Exception as e:
        if debug_mode:
            traceback.print_exc()
        return False