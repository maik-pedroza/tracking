# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
from datetime import datetime
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track

class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    today: Optional[datetime.date]
            Provide today's date, for naming of tracks

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.
    gating_only_position : Optional[bool]
        Used during gating, comparing KF predicted and measured states. If True, only the x, y position of the state distribution is considered during gating. Defaults to False, where x,y, aspect ratio and height will be considered.
    """

    def __init__(
        self,
        metric,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        override_track_class=None,
        today=None,
        gating_only_position=False,
    ):
        self.today = today
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init
        self.gating_only_position = gating_only_position

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self.del_tracks_ids = []
        self._next_id = 1
        self._used_ids = set()  # Track all IDs that have been used
        if override_track_class:
            self.track_class = override_track_class
        else:
            self.track_class = Track

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections, today=None):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.
        today: Optional[datetime.date]
            Provide today's date, for naming of tracks
        """
        if self.today:
            if today is None:
                today = datetime.now().date()
            # Check if its a new day, then refresh idx
            if today != self.today:
                self.today = today
                self._next_id = 1
                self._used_ids = set()  # Reset used IDs for a new day

        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections, similarity_dict = self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            similarity = similarity_dict.get((track_idx, detection_idx), None)
            self.tracks[track_idx].update(self.kf, detections[detection_idx], similarity=similarity)
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        new_tracks = []
        self.del_tracks_ids = []
        for t in self.tracks:
            if not t.is_deleted():
                new_tracks.append(t)
            else:
                self.del_tracks_ids.append(t.track_id)
                
        self.tracks = new_tracks
        # self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            # Use persistent features instead of regular features that get cleared
            features += track.persistent_features
            targets += [track.track_id for _ in track.persistent_features]
            # We still clear the temporary features buffer to maintain compatibility
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets
        )

    def _match(self, detections):
        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices, detection_indices,
                only_position=self.gating_only_position
            )
            return cost_matrix

        # Separa los tracks confirmados y no confirmados.
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Emparejamiento basado en apariencia.
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
            gated_metric,
            self.metric.matching_threshold,
            self.max_age,
            self.tracks,
            detections,
            confirmed_tracks,
        )

        # Emparejamiento IOU para el resto.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update == 1
        ]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1
        ]
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
            iou_matching.iou_cost,
            self.max_iou_distance,
            self.tracks,
            detections,
            iou_track_candidates,
            unmatched_detections,
        )

        all_matches = matches_a + matches_b
        unmatched_tracks = set(unmatched_tracks_a + unmatched_tracks_b)
        unmatched_detections = set(unmatched_detections)

        # Umbral de similitud: se requiere que la similitud coseno sea al menos:
        similarity_threshold = 1.0 - self.metric.matching_threshold

        similarity_dict = {}
        valid_matches = []

        for track_idx, detection_idx in all_matches:
            track_feature = self.tracks[track_idx].latest_feature
            detection_feature = detections[detection_idx].feature
            if track_feature is None or detection_feature is None:
                similarity = None
            else:
                norm_track = track_feature / np.linalg.norm(track_feature) if np.linalg.norm(track_feature) != 0 else track_feature
                norm_detection = detection_feature / np.linalg.norm(detection_feature) if np.linalg.norm(detection_feature) != 0 else detection_feature
                similarity = np.dot(norm_track, norm_detection)
            
            # Solo aceptamos el emparejamiento si hay similitud válida y es suficiente.
            if similarity is not None and similarity >= similarity_threshold:
                valid_matches.append((track_idx, detection_idx))
                similarity_dict[(track_idx, detection_idx)] = similarity
                # Eliminamos aquellos emparejamientos aceptados de las listas "sin match".
                unmatched_tracks.discard(track_idx)
                unmatched_detections.discard(detection_idx)
            else:
                # Si no se cumple, se añaden ambos a los sin match.
                unmatched_tracks.add(track_idx)
                unmatched_detections.add(detection_idx)

        return valid_matches, list(unmatched_tracks), list(unmatched_detections), similarity_dict

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())

            
        if self.today:
            track_id = "{}_{}".format(self.today, self._next_id)
        else:
            track_id = "{}".format(self._next_id)
        
        # Record this ID as used
        self._used_ids.add(str(self._next_id))
        
        self.tracks.append(
            self.track_class(
                mean,
                covariance,
                track_id,
                self.n_init,
                self.max_age,
                # mean, covariance, self._next_id, self.n_init, self.max_age,
                feature=detection.feature,
                original_ltwh=detection.get_ltwh(),
                det_class=detection.class_name,
                det_conf=detection.confidence,
                instance_mask=detection.instance_mask,
                others=detection.others,
            )
        )

        self._next_id += 1

    def delete_all_tracks(self):
        self.tracks = []
