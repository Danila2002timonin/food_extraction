# Image Object Detection and Extraction

A Python utility for detecting and extracting objects from images based on text descriptions using Hugging Face's DETR model and OpenAI's GPT-4o.

## Overview

This project combines modern object detection with natural language understanding to extract objects from images based on user text prompts. It uses:

1. **Hugging Face DETR Models** for object detection
2. **OpenAI GPT-4o** for semantic matching between detected objects and user prompts
3. **Additional heuristics** to find objects that might be missed by the neural network

## How It Works

1. The script analyzes an image using a DETR (DEtection TRansformer) model to detect all visible objects
2. It uses GPT-4o to match the user's text prompt with the detected objects
3. The best matching object is cropped from the image
4. The result is saved as a new image file

## Installation

1. Clone this repository:
```
git clone https://github.com/yourusername/image-detection-extraction.git
cd image-detection-extraction
```

2. Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
STABLE_DIFFUSION_API_KEY=your_stability_ai_key_here  # Required for background removal and image extension
```

## Usage

Extract an object from an image based on a text description:

```bash
python food_extractor_huggingface.py --image "path/to/image.jpg" --prompt "dish with red sauce"
```

Automatically identify and extract the main dish (no prompt needed):
```bash
python food_extractor_huggingface.py --image "path/to/image.jpg" --auto
```

### Advanced Features with Stability AI

#### Background Removal

Remove the background from the extracted object:
```bash
python food_extractor_huggingface.py --image "path/to/image.jpg" --prompt "dish with red sauce" --remove-bg
```

#### Image Extension (Outpainting)

Extend the extracted image in one or more directions:
```bash
python food_extractor_huggingface.py --image "path/to/image.jpg" --prompt "dish with red sauce" --extend --extend-left 200 --extend-right 200
```

You can combine these options:
```bash
# Extract, remove background, and extend
python food_extractor_huggingface.py --image "path/to/image.jpg" --auto --remove-bg --extend --extend-up 100 --extend-down 100
```

Additional options:
```bash
# With debug visualizations
python food_extractor_huggingface.py --image "path/to/image.jpg" --prompt "dish with red sauce" --debug

# With a specific model
python food_extractor_huggingface.py --image "path/to/image.jpg" --prompt "dish with red sauce" --model facebook/detr-resnet-101

# With a custom detection threshold
python food_extractor_huggingface.py --image "path/to/image.jpg" --prompt "dish with red sauce" --threshold 0.05
```

## Output Files

- `extracted_object.png`: The cropped object based on your text prompt
- `extracted_object_nobg.png`: (when using --remove-bg) The extracted object with background removed
- `extracted_object_nobg_extended.png`: (when using --extend) The extended image after processing
- `debug_detections.png`: (when debug is enabled) Visualization of all detected objects
- `debug_cropped.png`: (when debug is enabled) Intermediate cropped image

## Dependencies

- OpenCV
- PyTorch
- Transformers (Hugging Face)
- OpenAI
- Stability AI APIs (for background removal and image extension)
- Matplotlib
- Pillow
- NumPy

## License

This project is licensed under the MIT License - see the LICENSE file for details. 