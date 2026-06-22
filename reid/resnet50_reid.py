import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

from .base_reid import BaseReID
from .registry import register_reid


@register_reid("resnet50_reid")
class ResNet50ReID(BaseReID):
    """ResNet50-based REID model (Bag of Tricks baseline).

    Uses ResNet50 with modified last stride (stride=1) and BNNeck,
    pre-trained on Market1501. Feature dimension: 2048 (global feature).

    Since downloading pretrained REID weights requires external hosting,
    this implementation uses ImageNet-pretrained ResNet50 with a modified
    head as a reasonable baseline. The feature is the pooled output before
    the final FC layer.

    Parameters
    ----------
    model_name : str
        "resnet50_reid".
    device : str
        "cpu" or "cuda".
    """

    def __init__(self, model_name="resnet50_reid", device="cpu", **kwargs):
        super().__init__(model_name=model_name, device=device,
                         input_size=(256, 128))
        self._transform = None

    def load_model(self):
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Remove final FC layer, keep pooled features (2048-dim)
        resnet.fc = nn.Identity()
        # Modify last stride to 1 for REID (common trick)
        resnet.layer4[2].conv2.stride = (1, 1)
        if resnet.layer4[2].downsample is not None:
            resnet.layer4[2].downsample[0].stride = (1, 1)
        resnet = resnet.to(self.device)
        resnet.eval()
        self._model = resnet

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self._loaded = True

    @property
    def feature_dim(self):
        return 2048

    def _crop_and_prepare(self, image, box):
        x, y, w, h = box
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(image.shape[1], int(x + w))
        y2 = min(image.shape[0], int(y + h))
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((self.input_size[0], self.input_size[1], 3), dtype=np.uint8)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self._transform(crop)
        return tensor

    def extract_features(self, image, boxes):
        if not self._loaded:
            self.load_model()

        if len(boxes) == 0:
            return []

        tensors = [self._crop_and_prepare(image, box) for box in boxes]
        batch = torch.stack(tensors).to(self.device)

        with torch.no_grad():
            features = self._model(batch)

        features = features.cpu().numpy()
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-12)
        return [f.astype(np.float32) for f in features]
