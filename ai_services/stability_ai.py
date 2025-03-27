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

def remove_background_with_stability(image_path, output_path="extracted_object_nobg.png"):
    """Remove the background from an image using Stability AI's API"""
    try:
        # Load .env file to ensure we have the latest API key
        load_dotenv(override=True)
        
        # Get API key from environment
        api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
        if not api_key:
            print("Error: STABLE_DIFFUSION_API_KEY environment variable is not set.")
            return False
            
        print(f"Removing background from image: {image_path}")
        
        # Check image dimensions before sending
        img = Image.open(image_path)
        width, height = img.size
        total_pixels = width * height
        
        print(f"Input image size: {width}x{height} = {total_pixels} pixels")
        
        # Stability AI has a limit of 4,194,304 pixels (2048x2048)
        MAX_PIXELS = 4_194_304
        
        # If image is too large, resize it
        use_path = image_path
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
        
        # Print API key for debugging (first/last 5 chars only)
        print(f"Using API key: {api_key[:5]}...{api_key[-5:]}")
        
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
        print(f"Error removing background: {e}")
        traceback.print_exc()
        return False
        
def extend_image_with_stability(image_path, output_path="extended_image.png", left=0, right=0, up=0, down=0):
    """Extend an image in specified directions using Stability AI's Outpainting API"""
    try:
        # Load .env file to ensure we have the latest API key
        load_dotenv(override=True)
        
        # Get API key from environment
        api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
        if not api_key:
            print("Error: STABLE_DIFFUSION_API_KEY environment variable is not set.")
            return False
        
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
        use_path = image_path
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
            
            # Save a debug copy for troubleshooting
            debug_path = os.path.splitext(output_path)[0] + "_debug.png"
            with open(debug_path, 'wb') as file:
                file.write(response.content)
            
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
                
            # Try the v1 endpoint as fallback
            print("Attempting alternative endpoint...")
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
                            
                        print(f"Image extended successfully with alternative endpoint. Saved to: {output_path}")
                        return True
                    
            except Exception as alt_error:
                print(f"Alternative endpoint also failed: {alt_error}")
                
            return False
            
    except Exception as e:
        print(f"Error extending image: {e}")
        traceback.print_exc()
        return False