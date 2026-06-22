"""Identity database for autonomous body REID.

Stores person descriptors (REID features) and metadata. Supports kNN search
for matching query descriptors against known identities, creation of new
identities, and centroid-based lookups.

Pipeline:
1. Extract REID descriptor for each detection
2. Query database with descriptor → best matching identity or None (new)
3. Record identification in track history with frame timestamp
4. Resolve identity per track via majority voting over temporal window T
5. Resolve conflicts between active tracks (same identity → reset)
"""

import time
import numpy as np
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors


class Identity:
    """A single person identity in the database.

    Attributes
    ----------
    identity_id : int
        Unique identifier.
    descriptors : list[np.ndarray]
        Stored REID feature vectors.
    centroid : np.ndarray
        Running mean of descriptors (updated on add).
    frame_ids : list[int]
        Frame IDs where this identity was seen.
    """

    def __init__(self, identity_id, descriptor, frame_id=0):
        self.identity_id = identity_id
        self.descriptors = [descriptor.copy()]
        self.centroid = descriptor.copy()
        self.frame_ids = [frame_id]

    def add_descriptor(self, descriptor, frame_id=0):
        self.descriptors.append(descriptor.copy())
        n = len(self.descriptors)
        self.centroid = self.centroid * (n - 1) / n + descriptor / n
        self.frame_ids.append(frame_id)

    def __repr__(self):
        return f"Identity(id={self.identity_id}, n_desc={len(self.descriptors)})"


class IdentityDatabase:
    """Database of person identities with kNN search.

    Parameters
    ----------
    distance_threshold : float
        Maximum cosine distance for a match. If nearest neighbor distance
        exceeds this, a new identity is created.
    min_knn_samples : int
        Minimum number of descriptors in an identity before it can be matched
        via centroid (instead of raw descriptors).
    use_centroid : bool
        If True, match against centroids. If False, match against all
        raw descriptors.
    max_descriptors : int
        Maximum descriptors stored per identity (FIFO eviction).
    """

    def __init__(self, distance_threshold=0.3, use_centroid=True,
                 max_descriptors=50):
        self.distance_threshold = distance_threshold
        self.use_centroid = use_centroid
        self.max_descriptors = max_descriptors
        self._identities = {}
        self._next_id = 1
        self._index_dirty = True
        self._knn = None
        self._knn_ids = []

    @property
    def num_identities(self):
        return len(self._identities)

    def _rebuild_index(self):
        """Rebuild kNN index from stored descriptors or centroids."""
        if not self._identities:
            self._knn = None
            self._knn_ids = []
            self._index_dirty = False
            return

        if self.use_centroid:
            data = np.array([self._identities[i].centroid
                             for i in self._knn_ids])
        else:
            data = []
            ids = []
            for iid in self._knn_ids:
                identity = self._identities[iid]
                for d in identity.descriptors:
                    data.append(d)
                    ids.append(iid)
            data = np.array(data)
            self._knn_ids = ids

        if len(data) == 0:
            self._knn = None
        else:
            self._knn = NearestNeighbors(n_neighbors=1, metric="cosine",
                                         algorithm="brute")
            self._knn.fit(data)
        self._index_dirty = False

    def query(self, descriptor, frame_id=0):
        """Find best matching identity for a descriptor.

        Returns
        -------
        tuple (identity_id or None, distance)
            If distance < threshold, returns (identity_id, distance).
            Otherwise returns (None, distance) indicating a new identity.
        """
        if self._index_dirty:
            self._rebuild_index()

        if self._knn is None or self.num_identities == 0:
            return None, 1.0

        desc = descriptor.reshape(1, -1)
        distances, indices = self._knn.kneighbors(desc)
        dist = distances[0, 0]
        idx = indices[0, 0]

        if self.use_centroid:
            matched_id = self._knn_ids[idx]
        else:
            matched_id = self._knn_ids[idx]

        if dist <= self.distance_threshold:
            return matched_id, dist
        return None, dist

    def add_identity(self, descriptor, frame_id=0):
        """Create a new identity and return its ID."""
        iid = self._next_id
        self._next_id += 1
        identity = Identity(iid, descriptor, frame_id)
        self._identities[iid] = identity
        self._knn_ids.append(iid)
        self._index_dirty = True
        return iid

    def update_identity(self, identity_id, descriptor, frame_id=0):
        """Add a descriptor to an existing identity."""
        if identity_id not in self._identities:
            return False
        identity = self._identities[identity_id]
        if len(identity.descriptors) >= self.max_descriptors:
            identity.descriptors.pop(0)
        identity.add_descriptor(descriptor, frame_id)
        self._index_dirty = True
        return True

    def add_or_update(self, identity_id, descriptor, frame_id=0):
        """Update existing identity or create new one."""
        if identity_id is not None and identity_id in self._identities:
            self.update_identity(identity_id, descriptor, frame_id)
            return identity_id
        return self.add_identity(descriptor, frame_id)

    def get_all_descriptors_and_labels(self):
        """Return all descriptors with their identity labels.

        Used for clustering evaluation.
        """
        descriptors = []
        labels = []
        for iid, identity in self._identities.items():
            for d in identity.descriptors:
                descriptors.append(d)
                labels.append(iid)
        return np.array(descriptors), np.array(labels)

    def get_centroids_and_labels(self):
        """Return centroids with their identity labels."""
        centroids = []
        labels = []
        for iid, identity in self._identities.items():
            centroids.append(identity.centroid)
            labels.append(iid)
        return np.array(centroids), np.array(labels)

    def clear(self):
        self._identities = {}
        self._next_id = 1
        self._knn = None
        self._knn_ids = []
        self._index_dirty = True
