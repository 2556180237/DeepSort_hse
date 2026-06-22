"""YOLO11s-seg detector with segmentation mask support.

Uses Ultralytics YOLO11-seg models for person detection + instance segmentation.
Provides both bounding boxes and segmentation masks for body REID.
"""

import numpy as np
import cv2

from detectors.base_detector import BaseDetector, DetectionResult
from detectors.registry import register_detector


@register_detector("yolo11s-seg")
@register_detector("yolo11n-seg")
class YOLOSegDetector(BaseDetector):
    """YOLO11-seg detector using Ultralytics API.

    Supports YOLO11-seg models (nano/small) for detection + segmentation.
    Models are auto-downloaded from Ultralytics hub on first use.

    Parameters
    ----------
    model_name : str
        Model identifier: "yolo11s-seg", "yolo11n-seg".
    device : str
        "cpu" or "cuda".
    conf_threshold : float
        Confidence threshold.
    img_size : int
        Input image size.
    classes : list or None
        Class IDs to keep (default: [0] = person).
    """

    def __init__(self, model_name="yolo11s-seg", device="cpu",
                 conf_threshold=0.3, img_size=640, classes=None):
        super().__init__(model_name=model_name, device=device,
                         conf_threshold=conf_threshold, img_size=img_size,
                         classes=classes)
        self.model = None

    def load_model(self):
        from ultralytics import YOLO
        weight_name = f"{self.model_name}.pt"
        self.model = YOLO(weight_name)
        self._loaded = True

    def detect(self, frame):
        if not self._loaded:
            self.load_model()

        results = self.model(frame, conf=self.conf_threshold,
                             imgsz=self.img_size, classes=self.classes,
                             verbose=False)

        detections = []
        for r in results:
            boxes = r.boxes
            masks = r.masks if r.masks is not None else [None] * len(boxes)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                w = x2 - x1
                h = y2 - y1

                mask = None
                if masks is not None and i < len(masks) and masks[i] is not None:
                    mask_data = masks[i].data.cpu().numpy()
                    if mask_data.ndim == 3:
                        mask_data = mask_data[0]
                    mask = mask_data.astype(np.uint8)

                detections.append(DetectionResult(
                    bbox=[x1, y1, w, h], confidence=conf,
                    class_id=cls_id, mask=mask))
        return detections

    def detect_batch(self, frames):
        if not self._loaded:
            self.load_model()

        results = self.model(frames, conf=self.conf_threshold,
                             imgsz=self.img_size, classes=self.classes,
                             verbose=False)

        all_detections = []
        for r in results:
            frame_dets = []
            boxes = r.boxes
            masks = r.masks if r.masks is not None else [None] * len(boxes)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                w = x2 - x1
                h = y2 - y1

                mask = None
                if masks is not None and i < len(masks) and masks[i] is not None:
                    mask_data = masks[i].data.cpu().numpy()
                    if mask_data.ndim == 3:
                        mask_data = mask_data[0]
                    mask = mask_data.astype(np.uint8)

                frame_dets.append(DetectionResult(
                    bbox=[x1, y1, w, h], confidence=conf,
                    class_id=cls_id, mask=mask))
            all_detections.append(frame_dets)
        return all_detections
