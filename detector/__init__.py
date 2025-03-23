#!/usr/bin/env python
# Инициализация пакета detector
from food_extraction.detector.hugging_face_detector import HuggingFaceDetector
from food_extraction.detector.object_detection import detect_objects, find_missing_tableware
from food_extraction.detector.visualization import visualize_detections
