#!/usr/bin/env python
import os
import sys
import argparse
import glob
import traceback
from pathlib import Path

from detector.hugging_face_detector import HuggingFaceDetector
from ai_services.stability_ai import remove_background_with_stability, extend_image_with_stability

def process_image(image_path, output_dir, detector, extend_args, outpaint_prompt, debug):
    """Process a single image with all steps"""
    
    # Create output directory for this image
    image_name = os.path.basename(image_path).split('.')[0]
    image_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Set output paths
    extracted_path = os.path.join(image_output_dir, f"{image_name}_extracted.png")
    nobg_path = os.path.join(image_output_dir, f"{image_name}_nobg.png")
    extended_path = os.path.join(image_output_dir, f"{image_name}_extended.png")
    
    try:
        # Step 1: Extract the object
        result = detector.process_image(
            image_path=image_path,
            text_prompt=None,
            debug=debug,
            auto_mode=True
        )
        
        if not result:
            return False
            
        # Save the extracted object
        detector.save_result(result, extracted_path)
        
        # Step 2: Remove background
        success = remove_background_with_stability(extracted_path, nobg_path, debug_mode=debug)
        if not success:
            return False
            
        # Step 3: Extend image
        extend_image_with_stability(
            nobg_path,
            extended_path,
            left=extend_args.get('left', 0),
            right=extend_args.get('right', 0),
            up=extend_args.get('up', 0),
            down=extend_args.get('down', 0),
            prompt=outpaint_prompt,
            debug_mode=debug
        )
        
        return True
    except Exception as e:
        if debug:
            traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch process images with extraction, background removal and outpainting')
    parser.add_argument('--image-dir', required=True, help='Directory containing images to process')
    parser.add_argument('--output-dir', default='testing_results', help='Directory to save outputs')
    parser.add_argument('--api-key', default=None, help='OpenAI API key (if not set in environment)')
    parser.add_argument('--model', default='facebook/detr-resnet-101', help='HuggingFace model name')
    parser.add_argument('--threshold', type=float, default=0.1, help='Detection confidence threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Stability AI features
    parser.add_argument('--stability-api-key', help='Stability AI API key (if not set in environment)')
    parser.add_argument('--extend-left', type=int, default=200, help='Pixels to extend on the left side')
    parser.add_argument('--extend-right', type=int, default=200, help='Pixels to extend on the right side')
    parser.add_argument('--extend-up', type=int, default=200, help='Pixels to extend on the top')
    parser.add_argument('--extend-down', type=int, default=200, help='Pixels to extend on the bottom')
    parser.add_argument('--outpaint-prompt', type=str, 
                      default="Food on a beautiful white plate, centered composition, professional food photography, white background, studio lighting, high resolution, detailed texture",
                      help='Prompt for Stability AI outpainting')
    
    args = parser.parse_args()
    
    # Set Stability API key if provided
    if args.stability_api_key:
        os.environ["STABLE_DIFFUSION_API_KEY"] = args.stability_api_key
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize detector
    detector = HuggingFaceDetector(
        api_key=args.api_key,
        model_name=args.model,
        detection_threshold=args.threshold
    )
    
    # Get list of images
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext.upper())))
    
    if not image_paths:
        print(f"No images found in {args.image_dir}")
        return
    
    # Prepare extension arguments
    extend_args = {
        'left': args.extend_left,
        'right': args.extend_right,
        'up': args.extend_up,
        'down': args.extend_down
    }
    
    # Process each image
    total = len(image_paths)
    success_count = 0
    
    for i, image_path in enumerate(image_paths):
        print(f"[{i+1}/{total}] Processing {os.path.basename(image_path)}...")
        success = process_image(image_path, args.output_dir, detector, extend_args, args.outpaint_prompt, args.debug)
        if success:
            success_count += 1
            print(f"  ✓ Success")
        else:
            print(f"  ✗ Failed")
    
    print(f"\nCompleted processing {success_count}/{total} images")
    print(f"Results saved to {os.path.abspath(args.output_dir)}")

if __name__ == "__main__":
    main() 