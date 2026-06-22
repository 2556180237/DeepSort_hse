from .base_reid import BaseReID
from .registry import create_reid, list_reid_models, register_reid, REID_REGISTRY

# Auto-import REID implementations to trigger @register_reid
from . import osnet_reid  # noqa: F401, E402
from . import resnet50_reid  # noqa: F401, E402
from . import osnet_ain_reid  # noqa: F401, E402

__all__ = ["BaseReID", "create_reid", "list_reid_models", "register_reid", "REID_REGISTRY"]
