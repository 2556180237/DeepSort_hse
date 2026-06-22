from abc import ABC, abstractmethod
import numpy as np


class BaseReID(ABC):
    """Abstract base class for person re-identification models.

    All REID implementations must inherit from this class and implement
    `load_model` and `extract_features`. This provides a unified interface
    so that REID models can be swapped before execution.

    Parameters
    ----------
    model_name : str
        Name/identifier of the model (e.g. "osnet_x1_0", "resnet50_reid").
    device : str
        Device to run inference on (e.g. "cpu", "cuda").
    input_size : tuple
        Input image size (height, width) for the model.
    """

    def __init__(self, model_name="base", device="cpu", input_size=(256, 128)):
        self.model_name = model_name
        self.device = device
        self.input_size = input_size
        self._loaded = False
        self._model = None

    @abstractmethod
    def load_model(self):
        """Load model weights and initialize."""
        self._loaded = True

    @abstractmethod
    def extract_features(self, image, boxes):
        """Extract REID feature vectors for detected persons.

        Parameters
        ----------
        image : ndarray
            BGR image of shape (H, W, 3), as read by cv2.imread.
        boxes : list of (x, y, w, h)
            Bounding boxes in top-left width-height format.

        Returns
        -------
        list[ndarray]
            List of feature vectors, one per box.
        """
        pass

    def is_loaded(self):
        return self._loaded

    @property
    def feature_dim(self):
        """Return feature dimension. Override if known."""
        return 512

    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name}, device={self.device})"
