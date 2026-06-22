"""Body REID package: autonomous person re-identification.

Components:
- identity_database.py: Identity database with kNN search
- identity_manager.py: Track identity management with temporal voting
- segmentation_detector.py: YOLO11s-seg for detection + segmentation
"""

from .identity_database import IdentityDatabase, Identity
from .identity_manager import IdentityManager, TrackIdentityHistory

__all__ = ["IdentityDatabase", "Identity", "IdentityManager",
           "TrackIdentityHistory"]
