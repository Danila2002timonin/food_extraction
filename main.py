#!/usr/bin/env python
import os
import sys
import argparse
import traceback

from detector.hugging_face_detector import HuggingFaceDetector
from ai_services.stability_ai import remove_background_with_stability, extend_image_with_stability

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
    
    # Stability AI features
    parser.add_argument('--remove-bg', action='store_true', help='Use Stability AI to remove background after extraction')
    parser.add_argument('--extend', action='store_true', help='Use Stability AI to extend image after extension')
    parser.add_argument('--extend-left', type=int, default=0, help='Pixels to extend on the left side')
    parser.add_argument('--extend-right', type=int, default=0, help='Pixels to extend on the right side')
    parser.add_argument('--extend-up', type=int, default=0, help='Pixels to extend on the top')
    parser.add_argument('--extend-down', type=int, default=0, help='Pixels to extend on the bottom')
    parser.add_argument('--stability-api-key', help='Stability AI API key (if not set in environment)')
    
    args = parser.parse_args()
    
    # Verify that either a prompt is provided or auto mode is enabled
    if not args.prompt and not args.auto:
        return
        
    # Set Stability API key if provided
    if args.stability_api_key:
        os.environ["STABLE_DIFFUSION_API_KEY"] = args.stability_api_key
    
    try:
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
            
            # Apply Stability AI background removal if requested
            if args.remove_bg:
                # Define output path for the no-background image
                nobg_path = os.path.splitext(extracted_path)[0] + "_nobg.png"
                success = remove_background_with_stability(extracted_path, nobg_path, debug_mode=args.debug)
                if success:
                    # Update extracted_path for potential further processing
                    extracted_path = nobg_path
            
            # Apply Stability AI image extension if requested
            if args.extend:
                # Check if any extension values are provided
                if args.extend_left > 0 or args.extend_right > 0 or args.extend_up > 0 or args.extend_down > 0:
                    # Define output path for the extended image
                    extended_path = os.path.splitext(extracted_path)[0] + "_extended.png"
                    
                    # Try to use the background-removed image if it exists
                    nobg_path = os.path.splitext(args.output)[0] + "_nobg.png"
                    source_for_outpainting = extracted_path
                    
                    if os.path.exists(nobg_path) and args.remove_bg:
                        source_for_outpainting = nobg_path
                    
                    extend_image_with_stability(
                        source_for_outpainting, 
                        extended_path,
                        left=args.extend_left,
                        right=args.extend_right,
                        up=args.extend_up,
                        down=args.extend_down,
                        debug_mode=args.debug
                    )
    except Exception as e:
        if args.debug:
            traceback.print_exc()


if __name__ == "__main__":
    main() 