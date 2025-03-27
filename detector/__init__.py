#!/usr/bin/env python
# Инициализация пакета detector
from detector.hugging_face_detector import HuggingFaceDetector
from detector.object_detection import detect_objects, find_missing_tableware
from detector.visualization import visualize_detections
