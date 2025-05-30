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
            return False, "Missing API key"
        
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
                    
            return True, "Success"
        else:
            error_message = f"API error: {response.status_code}"
            try:
                error_details = response.json()
                error_message += f" - {json.dumps(error_details)}"
            except:
                error_message += f" - {response.text[:200]}"
                
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
                            
                    return True, "Success (fallback API)"
                else:
                    fallback_error = f"Fallback API error: {response.status_code}"
                    try:
                        fallback_details = response.json()
                        fallback_error += f" - {json.dumps(fallback_details)}"
                    except:
                        fallback_error += f" - {response.text[:200]}"
                    
                    error_message += f"; {fallback_error}"
            except Exception as e:
                error_message += f"; Fallback exception: {str(e)}"
            
            return False, error_message
            
    except Exception as e:
        if debug_mode:
            traceback.print_exc()
        return False, f"Exception: {str(e)}"
        
def extend_image_with_stability(image_path, output_path="extended_image.png", left=0, right=0, up=0, down=0, prompt=None, debug_mode=False):
    """Extend an image in specified directions using Stability AI's Outpainting API"""
    try:
        # Load .env file to ensure we have the latest API key
        load_dotenv(override=True)
        
        # Get API key from environment
        api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
        if not api_key:
            return False, "Missing API key"
        
        # Check image size before sending
        img = Image.open(image_path)
        img_width, img_height = img.size
        total_pixels = img_width * img_height
        
        # Stability AI has a limit of around 4 million pixels
        MAX_PIXELS = 4_194_304  # 2048x2048
        
        # If image is too large, resize it
        use_path = image_path
        temp_path = None
        if total_pixels > MAX_PIXELS:
            # Calculate the scaling factor to fit within the limit
            scale_factor = (MAX_PIXELS / total_pixels) ** 0.5
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
            
            # Save to a temporary file - preserve full directory path
            temp_dir = os.path.dirname(image_path)
            temp_base = os.path.basename(image_path)
            temp_name = os.path.splitext(temp_base)[0]
            temp_path = os.path.join(temp_dir, f"{temp_name}_resized_temp.png")
            img.save(temp_path)
            
            use_path = temp_path
        
        data = {"output_format": "png"}
        
        if left > 0:
            data["left"] = left
        if right > 0:
            data["right"] = right
        if up > 0:
            data["up"] = up
        if down > 0:
            data["down"] = down
        if prompt:
            data["prompt"] = prompt
        else:
            data["prompt"] = "food dish elegantly plated on white ceramic plate, clean white background, professional food photography, high quality, realistic lighting, restaurant style presentation, no extra objects, minimal shadows, centered composition"

        if debug_mode:
            print(f"Using v2beta endpoint with data: {data}")
            
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
            success = True
            error_message = "Success with v2beta endpoint"
        else:
            error_message = f"API error: {response.status_code}"
            try:
                error_details = response.json()
                error_message += f" - {json.dumps(error_details)}"
            except:
                error_message += f" - {response.text[:200]}"
            success = False
            
            if debug_mode:
                print(f"API error details: {error_message}")
        
        # Clean up temporary file if it was created
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                if debug_mode:
                    print(f"Failed to remove temp file {temp_path}: {str(e)}")
                    
        return success, error_message
            
    except Exception as e:
        if debug_mode:
            traceback.print_exc()
        return False, f"Exception: {str(e)}"