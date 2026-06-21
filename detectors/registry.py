from .base_detector import BaseDetector, DetectionResult


DETECTOR_REGISTRY = {}


def register_detector(name):
    """Decorator to register a detector class in the registry."""
    def decorator(cls):
        DETECTOR_REGISTRY[name] = cls
        return cls
    return decorator


def create_detector(name, **kwargs):
    """Create a detector instance by name.

    Parameters
    ----------
    name : str
        Registered detector name (e.g. "yolo11s", "yolov8n", "rtdetr-l").
    **kwargs
        Additional arguments passed to the detector constructor.

    Returns
    -------
    BaseDetector
        Detector instance.
    """
    if name not in DETECTOR_REGISTRY:
        available = ", ".join(DETECTOR_REGISTRY.keys())
        raise ValueError(f"Unknown detector '{name}'. Available: {available}")
    return DETECTOR_REGISTRY[name](**kwargs)


def list_detectors():
    """List all registered detector names."""
    return list(DETECTOR_REGISTRY.keys())
