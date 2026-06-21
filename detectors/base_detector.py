from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """Abstract base class for person detectors.

    All detector implementations must inherit from this class and implement
    the `detect` method. This provides a unified interface so that detectors
    can be swapped before execution without changing the tracking pipeline.

    Parameters
    ----------
    model_name : str
        Name/identifier of the model (e.g. "yolo11s", "rtdetr-l").
    device : str
        Device to run inference on (e.g. "cpu", "cuda").
    conf_threshold : float
        Confidence threshold for detections.
    img_size : int
        Input image size for the detector.
    classes : list or None
        List of class IDs to keep. If None, keeps only "person" (class 0 in COCO).
    """

    def __init__(self, model_name="base", device="cpu", conf_threshold=0.3,
                 img_size=640, classes=None):
        self.model_name = model_name
        self.device = device
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.classes = classes if classes is not None else [0]
        self._loaded = False

    @abstractmethod
    def load_model(self):
        """Load the model weights and initialize the detector."""
        self._loaded = True

    @abstractmethod
    def detect(self, frame):
        """Run detection on a single frame.

        Parameters
        ----------
        frame : ndarray
            BGR image of shape (H, W, 3), as read by cv2.imread.

        Returns
        -------
        list[Detection]
            List of Detection objects with bounding boxes in (x, y, w, h) format,
            confidence scores, and feature=None (features are computed separately
            by the REID model).
        """
        pass

    def detect_batch(self, frames):
        """Run detection on a batch of frames.

        Default implementation iterates over frames. Subclasses can override
        for batch-optimized inference.

        Parameters
        ----------
        frames : list[ndarray]
            List of BGR images.

        Returns
        -------
        list[list[Detection]]
            List of detection lists, one per frame.
        """
        return [self.detect(frame) for frame in frames]

    def is_loaded(self):
        return self._loaded

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device}, conf={self.conf_threshold})"


class DetectionResult:
    """Lightweight detection result container.

    Stores raw detection output (bbox, confidence, class) before conversion
    to deep_sort Detection objects.
    """

    def __init__(self, bbox, confidence, class_id=0, mask=None):
        self.bbox = np.asarray(bbox, dtype=np.float64)  # (x, y, w, h)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.mask = mask  # Optional segmentation mask

    def to_tlwh(self):
        return self.bbox.copy()

    def to_tlbr(self):
        ret = self.bbox.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.bbox.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def __repr__(self):
        return f"DetectionResult(bbox={self.bbox}, conf={self.confidence:.3f}, cls={self.class_id})"
