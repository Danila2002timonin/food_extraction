#!/usr/bin/env python
import os
import sys
import argparse
import traceback

from food_extraction.detector.hugging_face_detector import HuggingFaceDetector

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
            
            # Apply Stability AI background removal if requested
            if args.remove_bg:
                # Define output path for the no-background image
                nobg_path = os.path.splitext(extracted_path)[0] + "_nobg.png"
                from food_extraction.ai_services.stability_ai import remove_background_with_stability
                success = remove_background_with_stability(extracted_path, nobg_path)
                if success:
                    # Update extracted_path for potential further processing
                    extracted_path = nobg_path
                else:
                    print("Background removal failed, continuing with original extraction")
            
            # Apply Stability AI image extension if requested
            if args.extend:
                # Check if any extension values are provided
                if args.extend_left > 0 or args.extend_right > 0 or args.extend_up > 0 or args.extend_down > 0:
                    # Define output path for the extended image
                    extended_path = os.path.splitext(extracted_path)[0] + "_extended.png"
                    from food_extraction.ai_services.stability_ai import extend_image_with_stability
                    success = extend_image_with_stability(
                        extracted_path, 
                        extended_path,
                        left=args.extend_left,
                        right=args.extend_right,
                        up=args.extend_up,
                        down=args.extend_down
                    )
                    if success:
                        print(f"Final output saved to: {os.path.abspath(extended_path)}")
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
        traceback.print_exc()


if __name__ == "__main__":
    main() 