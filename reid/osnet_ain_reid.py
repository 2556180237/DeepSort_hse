import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

from .base_reid import BaseReID
from .registry import register_reid


@register_reid("osnet_ain_x1_0")
class OSNetAINReID(BaseReID):
    """OSNet-AIN REID model using torchvision-based implementation.

    Uses OSNet-AIN (Omni-Scale Network with Instance Normalization),
    pre-trained on MSMT17 via torchreid. This is a lighter alternative
    that uses the torchreid FeatureExtractor with the AIN variant.

    Feature dimension: 512.

    Parameters
    ----------
    model_name : str
        "osnet_ain_x1_0".
    device : str
        "cpu" or "cuda".
    """

    WEIGHT_FILE = "osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"

    def __init__(self, model_name="osnet_ain_x1_0", device="cpu", **kwargs):
        super().__init__(model_name=model_name, device=device,
                         input_size=(256, 128))
        self._extractor = None

    def load_model(self):
        from torchreid.reid.utils import FeatureExtractor
        from huggingface_hub import hf_hub_download

        repo_id = "kaiyangzhou/osnet"
        self._weight_path = hf_hub_download(repo_id=repo_id, filename=self.WEIGHT_FILE)

        self._extractor = FeatureExtractor(
            model_name="osnet_ain_x1_0",
            model_path=self._weight_path,
            device=self.device,
        )
        self._loaded = True

    @property
    def feature_dim(self):
        return 512

    def _crop_and_resize(self, image, box):
        x, y, w, h = box
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image.shape[1], int(x + w))
        y2 = min(image.shape[0], int(y + h))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        crop = cv2.resize(crop, (self.input_size[1], self.input_size[0]))
        return crop

    def extract_features(self, image, boxes):
        if not self._loaded:
            self.load_model()

        if len(boxes) == 0:
            return []

        crops = [self._crop_and_resize(image, box) for box in boxes]
        crops_rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]

        features = self._extractor(crops_rgb)
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        else:
            features = np.array(features)

        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        return [f.astype(np.float32) for f in features]
