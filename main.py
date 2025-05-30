import os
import sys
import argparse
import traceback
from pathlib import Path
import glob
import datetime
from PIL import Image

from detector.hugging_face_detector import HuggingFaceDetector
from ai_services.stability_ai import remove_background_with_stability, extend_image_with_stability

def process_single_image(detector, image_path, args):
    """Process a single image with the provided detector and arguments."""
    # Generate the output subfolder
    if args.output_dir:
        # Create subfolder based on image name
        base_filename = os.path.basename(image_path)
        image_name = os.path.splitext(base_filename)[0]
        image_output_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
    else:
        # Single file mode - use the output path
        image_output_dir = os.path.dirname(args.output)
        image_name = os.path.splitext(os.path.basename(args.output))[0]
    
    # Print processing information
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Processing started: {timestamp}")
    print(f"Input image: {image_path}")
    print(f"Parameters: prompt='{args.prompt}', auto={args.auto}, model={args.model}, threshold={args.threshold}")
    
    try:
        # Extract the object from the image
        print("Extracting object from image...")
        
        # Process the image with the detector
        result = detector.process_image(
            image_path=image_path,
            text_prompt=args.prompt,
            debug=args.debug,
            auto_mode=args.auto
        )
        
        if not result:
            print(f"No result found for {image_path}")
            return False
        
        # Set output paths
        if args.output_dir:
            # Use original filename in the dedicated subfolder
            extracted_path = os.path.join(image_output_dir, f"{image_name}.png")
        else:
            extracted_path = args.output
        
        # Save the extracted object
        print(f"Saving extracted object to {extracted_path}")
        detector.save_result(result, extracted_path)
        
        # Apply Stability AI background removal if requested
        nobg_path = None
        if args.remove_bg:
            # Define output path for the no-background image
            nobg_path = os.path.join(image_output_dir, f"{image_name}_nobg.png")
            
            print(f"Removing background with Stability AI...")
            
            success, error_details = remove_background_with_stability(
                extracted_path, 
                nobg_path, 
                debug_mode=args.debug
            )
            
            if success:
                # Update extracted_path for potential further processing
                extracted_path = nobg_path
                print(f"Background removed successfully. Saved to {nobg_path}")
            else:
                print(f"Background removal failed: {error_details}")
                # If background removal failed, set nobg_path to None
                nobg_path = None
        
        # Apply Stability AI image extension if requested
        if args.extend:
            # Check if any extension values are provided
            if args.extend_left > 0 or args.extend_right > 0 or args.extend_up > 0 or args.extend_down > 0:
                
                # Only proceed with extension if background was removed successfully
                if not args.remove_bg:
                    print("Background removal not requested. Image extension requires background removal. Skipping extension.")
                elif nobg_path is None:
                    print("Background removal failed, skipping image extension")
                else:
                    # Create a version with solid white background for outpainting
                    # try:
                    #     # Open the transparent PNG
                    #     transparent_img = Image.open(nobg_path)
                        
                    #     # Create a white background image of the same size
                    #     white_bg = Image.new("RGBA", transparent_img.size, (255, 255, 255, 255))
                        
                    #     # Paste the transparent image on the white background
                    #     white_bg.paste(transparent_img, (0, 0), transparent_img)
                        
                    #     # Convert to RGB to ensure compatibility
                    #     white_bg = white_bg.convert("RGB")
                        
                    #     # Save as a new file
                    #     white_bg_path = os.path.join(image_output_dir, f"{image_name}_nobg_white.png")
                    #     white_bg.save(white_bg_path)
                        
                    #     # Use this as source for outpainting
                    #     source_path = white_bg_path
                    #     print(f"Using white background version for extension: {source_path}")
                    # except Exception as e:
                    #     # If this fails, fall back to the transparent version
                    #     source_path = nobg_path
                    #     print(f"Failed to create white background version, using transparent version: {str(e)}")
                    #     print(f"Using transparent image as source for extension: {source_path}")
                    source_path = nobg_path
                    print(f"Using transparent image as source for extension skipping adding white background: {source_path}")
                    
                    # Define output path for the extended image
                    extended_path = os.path.join(image_output_dir, f"{image_name}_extended.png")
                    
                    # Proceed with extension
                    print(f"Extending image with Stability AI... (left={args.extend_left}, right={args.extend_right}, up={args.extend_up}, down={args.extend_down})")

                    extension_success, extension_error = extend_image_with_stability(
                        source_path,
                        extended_path,
                        left=args.extend_left,
                        right=args.extend_right,
                        up=args.extend_up,
                        down=args.extend_down,
                        debug_mode=args.debug
                    )
                    
                    if extension_success:
                        print(f"Image extended successfully. Saved to {extended_path}")
                    else:
                        print(f"Image extension failed: {extension_error}")
        
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        
        if args.debug:
            traceback.print_exc()
        
        return False

def main():
    parser = argparse.ArgumentParser(description='Extract objects from images using Hugging Face models and GPT-4o')
    # Input options - either single image or directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', help='Path to input image')
    input_group.add_argument('--image-dir', help='Directory containing images to process')
    
    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--output', default='extracted_object.png', help='Path to save the output image (for single image)')
    output_group.add_argument('--output-dir', help='Directory to save output images (for directory processing)')
    
    # Filter options for directory processing
    parser.add_argument('--pattern', default='*.jpg,*.jpeg,*.png', help='Comma-separated glob patterns for images to process (for directory mode)')
    
    # Other arguments
    parser.add_argument('--prompt', help='Text description of the object to extract (not required in auto mode)')
    parser.add_argument('--auto', action='store_true', help='Automatically identify and extract the main dish without requiring a prompt')
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
        print("Error: Either --prompt or --auto must be specified")
        parser.print_help()
        return
    
    # Verify output directory exists if processing a directory of images
    if args.image_dir and not args.output_dir:
        print("Error: --output-dir must be specified when using --image-dir")
        parser.print_help()
        return
    
    # Create output directory if it doesn't exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Set Stability API key if provided
    if args.stability_api_key:
        os.environ["STABLE_DIFFUSION_API_KEY"] = args.stability_api_key
    
    try:
        detector = HuggingFaceDetector(
            api_key=args.api_key,
            model_name=args.model,
            detection_threshold=args.threshold
        )
        
        if args.image:
            # Process a single image
            print(f"Processing single image: {args.image}")
            process_single_image(detector, args.image, args)
        else:
            # Process a directory of images
            patterns = args.pattern.split(',')
            image_files = []
            
            # Collect all files matching the patterns
            for pattern in patterns:
                pattern_path = os.path.join(args.image_dir, pattern.strip())
                image_files.extend(glob.glob(pattern_path))
            
            if not image_files:
                print(f"No images found matching patterns {args.pattern} in directory {args.image_dir}")
                return
            
            # Print batch processing summary
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Batch processing started: {timestamp}")
            print(f"Input directory: {args.image_dir}")
            print(f"Total images found: {len(image_files)}")
            
            # Process each image
            successful = 0
            for image_path in image_files:
                print(f"\nProcessing {image_path}...")
                if process_single_image(detector, image_path, args):
                    successful += 1
            
            # Print completion information
            summary_message = f"Processed {successful} of {len(image_files)} images successfully"
            print(f"\n{summary_message}")
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"Batch processing completed: {timestamp}")
            
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 