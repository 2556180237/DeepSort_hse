from .base_detector import BaseDetector, DetectionResult
from .registry import create_detector, list_detectors, register_detector, DETECTOR_REGISTRY

# Auto-import detector implementations to trigger @register_decorator
from . import yolo_detector  # noqa: F401, E402

__all__ = ["BaseDetector", "DetectionResult", "create_detector", "list_detectors",
           "register_detector", "DETECTOR_REGISTRY"]
