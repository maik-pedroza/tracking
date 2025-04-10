# vim: expandtab:ts=4:sw=4
import numpy as np

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.
    original_ltwh : Optional List
        Bounding box associated with matched detection
    det_class : Optional str
        Classname of matched detection
    det_conf : Optional float
        Confidence associated with matched detection
    instance_mask : Optional 
        Instance mask associated with matched detection
    others : Optional any
        Any supplementary fields related to matched detection

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurrence.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    persistent_features : List[ndarray]
        A persistent cache of features that is not cleared during tracker update.
        This helps improve reidentification by maintaining multiple views of the tracked object.

    """

    def __init__(
        self,
        mean,
        covariance,
        track_id,
        n_init,
        max_age,
        feature=None,
        original_ltwh=None,
        det_class=None,
        det_conf=None,
        instance_mask=None,
        others=None,
    ):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        self.persistent_features = []  # Persistent features that don't get cleared
        self.latest_feature = None
        if feature is not None:
            self.features.append(feature)
            self.persistent_features.append(feature)  # Also add to persistent features
            self.latest_feature = feature


        self._n_init = n_init
        self._max_age = max_age

        self.original_ltwh = original_ltwh
        self.det_class = det_class
        self.det_conf = det_conf
        self.instance_mask = instance_mask
        self.others = others

    def to_tlwh(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`. This function is POORLY NAMED. But we are keeping the way it works the way it works in order not to break any older libraries that depend on this.

        Returns
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.
        """
        return self.to_ltwh(orig=orig, orig_strict=orig_strict)

    def to_ltwh(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Params
        ------
        orig : bool
            To use original detection (True) or KF predicted (False). Only works for original dets that are horizontal BBs.
        orig_strict: bool 
            Only relevant when orig is True. If orig_strict is True, it ONLY outputs original bbs and will not output kalman mean even if original bb is not available. 

        Returns
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.

        """
        if orig:
            if self.original_ltwh is None:
                if orig_strict:
                    return None
                # else if not orig_strict, return kalman means below
            else:
                return self.original_ltwh.copy()

        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`. This original function is POORLY NAMED. But we are keeping the way it works the way it works in order not to break any older projects that depend on this.
        USE THIS AT YOUR OWN RISK. LIESSSSSSSSSS!
        Returns LIES
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.
        """
        return self.to_ltrb(orig=orig, orig_strict=orig_strict)

    def to_ltrb(self, orig=False, orig_strict=False):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Params
        ------
        orig : bool
            To use original detection (True) or KF predicted (False). Only works for original dets that are horizontal BBs.

        Returns
        -------
        ndarray
            The KF-predicted bounding box by default.
            If `orig` is True and track is matched to a detection this round, then the original det is returned.
        """
        ret = self.to_ltwh(orig=orig, orig_strict=orig_strict)
        if ret is not None:
            ret[2:] = ret[:2] + ret[2:]
        return ret

    def get_det_conf(self):
        """
        `det_conf` will be None is there are no associated detection this round
        """
        return self.det_conf

    def get_det_class(self):
        """
        Only `det_class` will be persisted in the track even if there are no associated detection this round.
        """
        return self.det_class

    def get_instance_mask(self):
        '''
        Get instance mask associated with detection. Will be None is there are no associated detection this round
        '''
        return self.instance_mask
    
    def get_det_supplementary(self):
        """
        Get supplementary info associated with the detection. Will be None is there are no associated detection this round.
        """
        return self.others

    def get_feature(self):
        '''
        Get latest appearance feature
        '''
        return self.latest_feature

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1
        self.original_ltwh = None
        self.det_conf = None
        self.instance_mask = None
        self.others = None

    def update(self, kf, detection):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """
        self.original_ltwh = detection.get_ltwh()
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah()
        )
        self.features.append(detection.feature)
        self.persistent_features.append(detection.feature)  # Add to persistent features
        
        # Limit the number of persistent features but keep more for better appearance matching
        max_persistent_features = 2000  # Increased from 1000 for better appearance history
        if len(self.persistent_features) > max_persistent_features:
            # Keep the most recent features
            self.persistent_features = self.persistent_features[-max_persistent_features:]
            
        self.latest_feature = detection.feature
        self.det_conf = detection.confidence
        self.det_class = detection.class_name
        self.instance_mask = detection.instance_mask
        self.others = detection.others

        self.hits += 1

        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step)."""
        previous_state = self.state
        
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted


class GSITrack(Track):
    """
    Extension of the Track class that implements Gaussian-smoothed Interpolation (GSI)
    for improved trajectory prediction during occlusions.
    
    This implementation maintains a history of positions and can generate interpolated
    positions when the track is temporarily lost, using Gaussian weighting to smooth
    the predictions.
    """
    
    def __init__(
        self,
        mean,
        covariance,
        track_id,
        n_init,
        max_age,
        feature=None,
        original_ltwh=None,
        det_class=None,
        det_conf=None,
        instance_mask=None,
        others=None,
        position_history_size=30,
        gsi_sigma=2.0,
        interpolation_max_frames=5
    ):
        """
        Parameters
        ----------
        position_history_size : int
            Maximum number of positions to keep in history for interpolation
        gsi_sigma : float
            Sigma parameter for Gaussian weighting (controls smoothness)
        interpolation_max_frames : int
            Maximum number of frames to perform interpolation before falling back to KF
        
        Additional parameters are the same as in the Track class.
        """
        super().__init__(
            mean, covariance, track_id, n_init, max_age, 
            feature, original_ltwh, det_class, det_conf, instance_mask, others
        )
        
        # GSI specific parameters
        self.position_history = []  # List of (frame_idx, position) tuples
        self.position_history_size = position_history_size
        self.gsi_sigma = gsi_sigma
        self.interpolation_max_frames = interpolation_max_frames
        self.frame_idx = 0
        self.last_detection_frame = 0
        self.is_interpolating = False
    
    def predict(self, kf):
        """
        Override predict method to use GSI when appropriate.
        """
        self.frame_idx += 1
        
        # Determine if we should use interpolation
        frames_since_detection = self.frame_idx - self.last_detection_frame
        if (self.time_since_update > 0 and 
            self.time_since_update <= self.interpolation_max_frames and
            len(self.position_history) >= 3):
            
            # Use GSI for prediction
            self.is_interpolating = True
            interpolated_position = self._interpolate_position()
            
            # Update mean with interpolated position, but keep covariance from KF prediction
            mean, covariance = kf.predict(self.mean, self.covariance)
            
            # Only replace the position part (x, y) of the state
            mean[0] = interpolated_position[0]  # x
            mean[1] = interpolated_position[1]  # y
            
            self.mean, self.covariance = mean, covariance
        else:
            # Use regular Kalman prediction
            self.is_interpolating = False
            self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        
        self.age += 1
        self.time_since_update += 1
        self.original_ltwh = None
        self.det_conf = None
        self.instance_mask = None
        self.others = None
    
    def update(self, kf, detection):
        """
        Override update method to record position history for interpolation.
        """
        # Call the parent update method first
        super().update(kf, detection)
        
        # Update position history
        position = (self.mean[0], self.mean[1])  # x, y
        self.position_history.append((self.frame_idx, position))
        
        # Limit history size
        if len(self.position_history) > self.position_history_size:
            self.position_history = self.position_history[-self.position_history_size:]
        
        self.last_detection_frame = self.frame_idx
        self.is_interpolating = False
    
    def _gaussian_weight(self, distance, sigma):
        """
        Calculate Gaussian weight based on distance.
        
        Parameters
        ----------
        distance : float
            Distance value (typically time difference)
        sigma : float
            Sigma parameter controlling the spread of the Gaussian
            
        Returns
        -------
        float
            Gaussian weight value
        """
        return np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    def _interpolate_position(self):
        """
        Perform Gaussian-smoothed interpolation based on position history.
        
        Returns
        -------
        tuple
            Interpolated (x, y) position
        """
        if not self.position_history:
            # Fallback to current position if no history
            return (self.mean[0], self.mean[1])
        
        # Sort history by frame index
        sorted_history = sorted(self.position_history, key=lambda x: x[0])
        
        # Extract times and positions
        times = np.array([h[0] for h in sorted_history])
        positions = np.array([h[1] for h in sorted_history])
        
        # Current time for interpolation
        current_time = self.frame_idx
        
        # Calculate time differences
        time_diffs = np.abs(times - current_time)
        
        # Calculate weights using Gaussian function
        weights = np.array([self._gaussian_weight(diff, self.gsi_sigma) for diff in time_diffs])
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Calculate weighted position
        interpolated_x = np.sum(weights * positions[:, 0])
        interpolated_y = np.sum(weights * positions[:, 1])
        
        return (interpolated_x, interpolated_y)
