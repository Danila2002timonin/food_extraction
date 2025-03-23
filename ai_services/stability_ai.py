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

def remove_background_with_stability(image_path, output_path="extracted_object_nobg.png"):
    """
    Remove the background from an image using Stability AI's API.
    
    Args:
        image_path: Path to the input image
        output_path: Path to save the output image
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get API key from environment
        api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
        if not api_key:
            print("Error: STABLE_DIFFUSION_API_KEY environment variable is not set.")
            return False
            
        print(f"Removing background from image: {image_path}")
        
        # Prepare headers for the API request
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Ensure we have a valid image file
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return False
            
        # Prepare the image for upload
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the payload for the API request
        payload = {
            "image_base64": encoded_image,
        }
        
        # Make the API request to remove background
        print("Calling Stability AI API to remove background...")
        response = requests.post(
            "https://api.stability.ai/v1/generation/image-to-alpha",
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        # Parse the response
        result = response.json()
        
        # Check if the response contains the expected data
        if 'artifacts' not in result or len(result['artifacts']) == 0:
            print("Error: Unexpected API response format")
            print(f"Response: {result}")
            return False
            
        # Get the base64-encoded image with alpha channel
        output_b64 = result['artifacts'][0]['base64']
        
        # Convert the base64 string to an image
        output_data = base64.b64decode(output_b64)
        
        # Create the output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save the image with alpha channel
        with open(output_path, "wb") as output_file:
            output_file.write(output_data)
            
        print(f"Background removed successfully. Saved to: {os.path.abspath(output_path)}")
        return True
        
    except Exception as e:
        print(f"Error removing background: {e}")
        traceback.print_exc()
        return False
        
def extend_image_with_stability(image_path, output_path="extended_image.png", left=0, right=0, up=0, down=0):
    """
    Extend an image in specified directions using Stability AI's Outpainting API.
    
    Args:
        image_path: Path to input image
        output_path: Path to save output image
        left: Pixels to extend on the left
        right: Pixels to extend on the right
        up: Pixels to extend on the top
        down: Pixels to extend on the bottom
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get API key from environment
        api_key = os.environ.get("STABLE_DIFFUSION_API_KEY")
        if not api_key:
            print("Error: STABLE_DIFFUSION_API_KEY environment variable is not set.")
            return False
            
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found at {image_path}")
            return False
            
        # Check if any extensions are requested
        if left == 0 and right == 0 and up == 0 and down == 0:
            print("No extension values provided. Skipping image extension.")
            return False
            
        # Load the image to get its dimensions
        original_image = Image.open(image_path)
        width, height = original_image.size
        
        # Determine the new dimensions
        new_width = width + left + right
        new_height = height + up + down
        
        print(f"Extending image from {width}x{height} to {new_width}x{new_height}")
        print(f"Extensions: left={left}, right={right}, up={up}, down={down}")
        
        # Prepare headers for the API request
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        # Convert image to base64
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Set up the API request for outpainting
        payload = {
            "image": f"data:image/png;base64,{encoded_image}",
            "mask": None,  # No mask needed for full outpainting
            "width": new_width,
            "height": new_height,
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
        
        # Make the API request
        print("Calling Stability AI API for image extension (outpainting)...")
        print("This may take a moment...")
        response = requests.post(
            "https://api.stability.ai/v1/generation/stable-image/outpaint",
            headers=headers,
            json=payload
        )
        
        # Check for non-200 responses
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        # Process the response
        result = response.json()
        
        # Save the temporary file for debugging
        with open("outpainting_response.json", "w") as f:
            json.dump(result, f, indent=2)
        
        if 'artifacts' not in result or len(result['artifacts']) == 0:
            print("Error: Unexpected API response format")
            print(f"Response: {result}")
            return False
            
        # Get the base64-encoded extended image
        extended_image_b64 = result['artifacts'][0]['base64']
        extended_image_data = base64.b64decode(extended_image_b64)
        
        # Save the extended image
        with open(output_path, "wb") as f:
            f.write(extended_image_data)
        
        # Also save a temporary version for debugging
        with open("extracted_object_temp_extend.png", "wb") as f:
            f.write(extended_image_data)
            
        print(f"Image extended successfully. Saved to: {os.path.abspath(output_path)}")
        return True
        
    except Exception as e:
        print(f"Error extending image: {e}")
        traceback.print_exc()
        return False 