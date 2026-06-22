import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from .base_reid import BaseReID
from .registry import register_reid


@register_reid("osnet_x1_0")
@register_reid("osnet_x0_25")
class OSNetReID(BaseReID):
    """OSNet REID model using torchreid FeatureExtractor.

    Models pre-trained on MSMT17, downloaded from HuggingFace.
    Feature dimension: 512.

    Parameters
    ----------
    model_name : str
        "osnet_x1_0" or "osnet_x0_25".
    device : str
        "cpu" or "cuda".
    """

    WEIGHT_MAP = {
        "osnet_x1_0": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "osnet_x0_25": "osnet_x0_25_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
    }

    def __init__(self, model_name="osnet_x1_0", device="cpu", **kwargs):
        super().__init__(model_name=model_name, device=device,
                         input_size=(256, 128))
        self._extractor = None
        self._weight_path = None

    def load_model(self):
        from torchreid.reid.utils import FeatureExtractor
        from huggingface_hub import hf_hub_download

        repo_id = "kaiyangzhou/osnet"
        filename = self.WEIGHT_MAP.get(self.model_name, self.WEIGHT_MAP["osnet_x1_0"])
        self._weight_path = hf_hub_download(repo_id=repo_id, filename=filename)

        self._extractor = FeatureExtractor(
            model_name=self.model_name,
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
        crops_bgr = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops]

        features = self._extractor(crops_bgr)
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()
        else:
            features = np.array(features)

        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        return [f.astype(np.float32) for f in features]
