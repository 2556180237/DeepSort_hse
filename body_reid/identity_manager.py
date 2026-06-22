"""Track identity manager for autonomous body REID.

Manages identity assignment to active tracks through:
1. Per-frame identification via kNN against the identity database
2. Temporal window voting (majority vote over last T frames)
3. Conflict resolution between active tracks with same identity
"""

import numpy as np
from collections import defaultdict, deque
from .identity_database import IdentityDatabase


class TrackIdentityHistory:
    """Per-track identification history.

    Stores (frame_id, identity_id, distance) entries for a track.
    """

    def __init__(self, track_id):
        self.track_id = track_id
        self.history = deque()  # (frame_id, identity_id, distance)
        self.resolved_identity = None  # current resolved identity
        self.last_update_frame = -1

    def add_observation(self, frame_id, identity_id, distance):
        self.history.append((frame_id, identity_id, distance))
        self.last_update_frame = frame_id

    def resolve(self, temporal_window, min_votes=1):
        """Resolve identity via majority voting over temporal window.

        Parameters
        ----------
        temporal_window : int
            Only consider observations within last `temporal_window` frames.
        min_votes : int
            Minimum votes for a valid identity. If no identity reaches
            min_votes, resolved_identity is set to None.

        Returns
        -------
        int or None
            Resolved identity ID.
        """
        if not self.history:
            self.resolved_identity = None
            return None

        current_frame = self.last_update_frame
        votes = defaultdict(int)

        for frame_id, identity_id, dist in self.history:
            if current_frame - frame_id > temporal_window:
                continue
            if identity_id is not None:
                votes[identity_id] += 1

        if not votes:
            self.resolved_identity = None
            return None

        best_id = max(votes, key=votes.get)
        if votes[best_id] >= min_votes:
            self.resolved_identity = best_id
        else:
            self.resolved_identity = None

        return self.resolved_identity

    def reset(self):
        self.resolved_identity = None


class IdentityManager:
    """Manages identity assignment and conflict resolution for active tracks.

    Parameters
    ----------
    distance_threshold : float
        Cosine distance threshold for identity matching.
    temporal_window : int
        Window size (in frames) for majority voting.
    min_votes : int
        Minimum votes for identity resolution.
    use_centroid : bool
        Whether to use centroids in kNN search.
    max_descriptors : int
        Max descriptors per identity.
    conflict_strategy : str
        "reset" — reset all conflicting tracks to None.
        "keep_best" — keep track with most votes, reset others.
    """

    def __init__(self, distance_threshold=0.3, temporal_window=30,
                 min_votes=1, use_centroid=True, max_descriptors=50,
                 conflict_strategy="reset"):
        self.database = IdentityDatabase(
            distance_threshold=distance_threshold,
            use_centroid=use_centroid,
            max_descriptors=max_descriptors)
        self.temporal_window = temporal_window
        self.min_votes = min_votes
        self.conflict_strategy = conflict_strategy
        self._track_histories = {}  # track_id -> TrackIdentityHistory
        self._active_tracks = set()

    def _get_history(self, track_id):
        if track_id not in self._track_histories:
            self._track_histories[track_id] = TrackIdentityHistory(track_id)
        return self._track_histories[track_id]

    def identify_detection(self, descriptor, frame_id):
        """Query the database for a single detection descriptor.

        Returns
        -------
        tuple (identity_id or None, distance)
        """
        return self.database.query(descriptor, frame_id)

    def update(self, frame_id, track_descriptors):
        """Process one frame: identify, update database, resolve, fix conflicts.

        Parameters
        ----------
        frame_id : int
            Current frame index.
        track_descriptors : dict
            {track_id: descriptor} for all active tracks on this frame.

        Returns
        -------
        dict
            {track_id: resolved_identity_id} for all active tracks.
        """
        self._active_tracks = set(track_descriptors.keys())

        # Step 1: Identify each track's descriptor against database
        frame_matches = {}
        for track_id, descriptor in track_descriptors.items():
            identity_id, dist = self.identify_detection(descriptor, frame_id)
            frame_matches[track_id] = (identity_id, dist, descriptor)

        # Step 2: Add/update database entries
        for track_id, (identity_id, dist, descriptor) in frame_matches.items():
            if identity_id is not None:
                self.database.update_identity(identity_id, descriptor, frame_id)
            # Don't create new identity yet — wait for resolution

        # Step 3: Record observations in track history
        for track_id, (identity_id, dist, descriptor) in frame_matches.items():
            history = self._get_history(track_id)
            history.add_observation(frame_id, identity_id, dist)

        # Step 4: Create new identities for unmatched tracks
        for track_id, (identity_id, dist, descriptor) in frame_matches.items():
            if identity_id is None:
                new_id = self.database.add_identity(descriptor, frame_id)
                history = self._get_history(track_id)
                # Replace the None observation with the new identity
                if history.history:
                    f, _, d = history.history.pop()
                    history.history.append((f, new_id, d))

        # Step 5: Resolve identities via majority voting
        resolved = {}
        for track_id in self._active_tracks:
            history = self._track_histories[track_id]
            resolved[track_id] = history.resolve(
                self.temporal_window, self.min_votes)

        # Step 6: Conflict resolution
        resolved = self._resolve_conflicts(resolved)

        return resolved

    def _resolve_conflicts(self, resolved):
        """Resolve conflicts where multiple tracks share the same identity.

        Parameters
        ----------
        resolved : dict
            {track_id: identity_id} for active tracks.

        Returns
        -------
        dict
            Updated {track_id: identity_id} after conflict resolution.
        """
        # Group tracks by identity
        identity_to_tracks = defaultdict(list)
        for track_id, identity_id in resolved.items():
            if identity_id is not None:
                identity_to_tracks[identity_id].append(track_id)

        # Find conflicts
        for identity_id, tracks in identity_to_tracks.items():
            if len(tracks) <= 1:
                continue

            if self.conflict_strategy == "reset":
                # Reset all conflicting tracks
                for track_id in tracks:
                    resolved[track_id] = None
                    self._track_histories[track_id].reset()
            elif self.conflict_strategy == "keep_best":
                # Keep track with most history entries, reset others
                best_track = max(
                    tracks,
                    key=lambda t: len(self._track_histories[t].history))
                for track_id in tracks:
                    if track_id != best_track:
                        resolved[track_id] = None
                        self._track_histories[track_id].reset()

        return resolved

    def get_identity_assignments(self):
        """Return current identity assignments for all known tracks.

        Returns
        -------
        dict
            {track_id: resolved_identity_id}
        """
        return {
            tid: hist.resolved_identity
            for tid, hist in self._track_histories.items()
        }

    def get_all_descriptors_and_labels(self):
        """Return all descriptors and identity labels from the database."""
        return self.database.get_all_descriptors_and_labels()

    def get_statistics(self):
        """Return database statistics."""
        return {
            "num_identities": self.database.num_identities,
            "num_tracks": len(self._track_histories),
            "active_tracks": len(self._active_tracks),
        }

    def clear(self):
        self.database.clear()
        self._track_histories = {}
        self._active_tracks = set()
