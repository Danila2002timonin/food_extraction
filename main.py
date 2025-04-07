#!/usr/bin/env python
import os
import sys
import argparse
import traceback
from pathlib import Path
import glob
import io
import datetime

from detector.hugging_face_detector import HuggingFaceDetector
from ai_services.stability_ai import remove_background_with_stability, extend_image_with_stability

def process_single_image(detector, image_path, args):
    """Process a single image with the provided detector and arguments."""
    # Create a log capture for this image
    log_output = io.StringIO()
    
    # Generate the output subfolder
    if args.output_dir:
        # Create subfolder based on image name
        base_filename = os.path.basename(image_path)
        image_name = os.path.splitext(base_filename)[0]
        image_output_dir = os.path.join(args.output_dir, image_name)
        os.makedirs(image_output_dir, exist_ok=True)
        
        # Set up log file
        log_file_path = os.path.join(image_output_dir, "processing_log.txt")
    else:
        # Single file mode - use the output path
        image_output_dir = os.path.dirname(args.output)
        image_name = os.path.splitext(os.path.basename(args.output))[0]
        log_file_path = os.path.splitext(args.output)[0] + "_log.txt"
    
    # Start log with timestamp and input information
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"Processing started: {timestamp}\n"
    log_message += f"Input image: {image_path}\n"
    log_message += f"Parameters: prompt='{args.prompt}', auto={args.auto}, model={args.model}, threshold={args.threshold}\n"
    
    print(log_message)
    log_output.write(log_message)
    
    try:
        # Extract the object from the image
        result_message = "Extracting object from image..."
        print(result_message)
        log_output.write(result_message + "\n")
        
        # Capture stdout during model execution to get detailed logs
        original_stdout = sys.stdout
        stdout_capture = io.StringIO()
        
        try:
            # Redirect stdout to our capture
            sys.stdout = stdout_capture
            
            # Process the image with the detector
            result = detector.process_image(
                image_path=image_path,
                text_prompt=args.prompt,
                debug=args.debug,
                auto_mode=args.auto
            )
            
            # Get the captured output
            detailed_logs = stdout_capture.getvalue()
        finally:
            # Restore stdout no matter what
            sys.stdout = original_stdout
        
        # Print the detailed logs to console and add to our log
        if detailed_logs:
            print(detailed_logs)
            log_output.write(detailed_logs)
        
        if not result:
            error_msg = f"No result found for {image_path}"
            print(error_msg)
            log_output.write(error_msg + "\n")
            
            # Save log even if processing failed
            with open(log_file_path, 'w') as log_file:
                log_file.write(log_output.getvalue())
            
            return False
        
        # Set output paths
        if args.output_dir:
            # Use original filename in the dedicated subfolder
            extracted_path = os.path.join(image_output_dir, f"{image_name}.png")
        else:
            extracted_path = args.output
        
        # Save the extracted object
        save_msg = f"Saving extracted object to {extracted_path}"
        print(save_msg)
        log_output.write(save_msg + "\n")
        
        detector.save_result(result, extracted_path)
        
        # Apply Stability AI background removal if requested
        nobg_path = None
        if args.remove_bg:
            # Define output path for the no-background image
            nobg_path = os.path.join(image_output_dir, f"{image_name}_nobg.png")
            
            bg_msg = f"Removing background with Stability AI..."
            print(bg_msg)
            log_output.write(bg_msg + "\n")
            
            # Capture stdout for background removal as well
            stdout_capture = io.StringIO()
            
            try:
                sys.stdout = stdout_capture
                success = remove_background_with_stability(extracted_path, nobg_path, debug_mode=args.debug)
                bg_detailed_logs = stdout_capture.getvalue()
            finally:
                sys.stdout = original_stdout
            
            if bg_detailed_logs:
                print(bg_detailed_logs)
                log_output.write(bg_detailed_logs)
            
            if success:
                # Update extracted_path for potential further processing
                extracted_path = nobg_path
                success_msg = f"Background removed successfully. Saved to {nobg_path}"
                print(success_msg)
                log_output.write(success_msg + "\n")
            else:
                fail_msg = "Background removal failed"
                print(fail_msg)
                log_output.write(fail_msg + "\n")
                # If background removal failed, set nobg_path to None
                nobg_path = None
        
        # Apply Stability AI image extension if requested
        if args.extend:
            # Check if any extension values are provided
            if args.extend_left > 0 or args.extend_right > 0 or args.extend_up > 0 or args.extend_down > 0:
                # Define output path for the extended image
                extended_path = os.path.join(image_output_dir, f"{image_name}_extended.png")
                
                # Use the background-removed image if it was successfully created
                source_for_outpainting = extracted_path
                
                # Since nobg_path is already known from earlier processing,
                # we can use it directly without checking file existence
                if nobg_path is not None and args.remove_bg:
                    source_msg = f"Using no-background image as source for extension: {nobg_path}"
                    print(source_msg)
                    log_output.write(source_msg + "\n")
                    source_for_outpainting = nobg_path
                else:
                    source_msg = f"Using original extracted image as source for extension: {extracted_path}"
                    print(source_msg)
                    log_output.write(source_msg + "\n")
                
                extend_msg = f"Extending image with Stability AI... (left={args.extend_left}, right={args.extend_right}, up={args.extend_up}, down={args.extend_down})"
                print(extend_msg)
                log_output.write(extend_msg + "\n")
                
                # Capture stdout for extend operation as well
                stdout_capture = io.StringIO()
                
                try:
                    sys.stdout = stdout_capture
                    extend_image_with_stability(
                        source_for_outpainting, 
                        extended_path,
                        left=args.extend_left,
                        right=args.extend_right,
                        up=args.extend_up,
                        down=args.extend_down,
                        prompt=args.outpaint_prompt,
                        debug_mode=args.debug
                    )
                    extend_detailed_logs = stdout_capture.getvalue()
                finally:
                    sys.stdout = original_stdout
                
                if extend_detailed_logs:
                    print(extend_detailed_logs)
                    log_output.write(extend_detailed_logs)
                
                success_ext_msg = f"Image extended successfully. Saved to {extended_path}"
                print(success_ext_msg)
                log_output.write(success_ext_msg + "\n")
        
        # Finalize log
        success_final_msg = f"Processing completed successfully for {image_path}"
        print(success_final_msg)
        log_output.write(success_final_msg + "\n")
        
        # Save the log to file
        with open(log_file_path, 'w') as log_file:
            log_file.write(log_output.getvalue())
        
        return True
    except Exception as e:
        error_msg = f"Error processing {image_path}: {str(e)}"
        print(error_msg)
        log_output.write(error_msg + "\n")
        
        if args.debug:
            trace = traceback.format_exc()
            print(trace)
            log_output.write(trace + "\n")
        
        # Save log even if processing failed
        with open(log_file_path, 'w') as log_file:
            log_file.write(log_output.getvalue())
        
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
    parser.add_argument('--outpaint-prompt', type=str, 
                      default="Food on a beautiful white plate, centered composition, professional food photography, white background, studio lighting, high resolution, detailed texture",
                      help='Prompt for Stability AI outpainting')
    
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
            
            # Create summary log
            summary_log_path = os.path.join(args.output_dir, "processing_summary.txt")
            with open(summary_log_path, "w") as summary_log:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary_log.write(f"Batch processing started: {timestamp}\n")
                summary_log.write(f"Input directory: {args.image_dir}\n")
                summary_log.write(f"Total images found: {len(image_files)}\n\n")
            
            # Process each image
            successful = 0
            for image_path in image_files:
                print(f"\nProcessing {image_path}...")
                if process_single_image(detector, image_path, args):
                    successful += 1
            
            # Update summary log with completion information
            summary_message = f"Processed {successful} of {len(image_files)} images successfully"
            print(f"\n{summary_message}")
            
            with open(summary_log_path, "a") as summary_log:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                summary_log.write(f"\nBatch processing completed: {timestamp}\n")
                summary_log.write(f"{summary_message}\n")
            
    except Exception as e:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    main() 