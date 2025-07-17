# vim: expandtab:ts=4:sw=4
"""
imm_track.py
------------

IMMTrack: Track implementation that uses an Interacting-Multiple-Model (IMM)
filter internally instead of a single constant-velocity Kalman model.  This is
useful for targets whose motion can switch between different regimes (e.g.
constant-velocity *vs* coordinated-turn) where a single model is insufficient.

The class conforms to the same external interface as the original `Track`
class so that it can be plugged into the existing `Tracker` by passing
`override_track_class=IMMTrack` when constructing the `Tracker`.

Only linear Kalman sub-models are provided for now (all share the same state
space as the original one).  They differ in their process noise parameters,
which allows e.g. one model to capture smoother motion (small process noise)
and another one to react faster to manoeuvres (larger process noise).

The implementation follows the standard four-step IMM cycle (interaction,
prediction, update, fusion).  See e.g. Bar-Shalom et al. "Multitarget-
Multisensor Tracking" for details.

NOTE: This is a *first, minimal* implementation meant to demonstrate the
concept.  It can be extended with non-linear EKF/UKF sub-models or different
state dimensions.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from .kalman_filter import KalmanFilter
from .track import Track, TrackState


class IMMTrack(Track):
    """A *Track* that maintains several motion models simultaneously using the
    Interacting-Multiple-Model (IMM) algorithm.

    The public API is kept identical to *Track* so that existing pipelines keep
    working.  Only *predict* and *update* are overridden to embed the IMM
    machinery.  The other helper methods (.to_ltwh(), etc.) rely on the fused
    state that the IMM computes, therefore they work transparently.
    """

    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id,
        n_init,
        max_age,
        feature=None,
        original_ltwh=None,
        det_class=None,
        det_conf=None,
        instance_mask=None,
        others=None,
        *,
        # IMM-specific arguments
        transition_matrix: np.ndarray | None = None,
        model_process_noise_scalers: Sequence[float] | None = None,
        init_model_probabilities: Sequence[float] | None = None,
        sub_models: Sequence[object] | None = None,
    ):
        """Create an IMM track.

        Parameters
        ----------
        transition_matrix : Optional[np.ndarray]
            *M x M* model switch probability matrix.  If *None*, a default
            high-self-probability matrix is built (0.90 on the diagonal,
            remaining mass evenly split).
        model_process_noise_scalers : Optional[Sequence[float]]
            Scaling factors (>0) applied to the default process noise of the
            *KalmanFilter* for each sub-model.  By default two models are
            created: one with scale 1.0 (smooth motion) and one with scale 10.0
            (aggressive manoeuvre model).
        init_model_probabilities : Optional[Sequence[float]]
            Initial probability for each model.  Will be normalised if it does
            not sum to one.  Defaults to uniform.
        """
        super().__init__(
            mean,
            covariance,
            track_id,
            n_init,
            max_age,
            feature,
            original_ltwh,
            det_class,
            det_conf,
            instance_mask,
            others,
        )

        # ------------------------------------------------------------------
        # Selección de sub-modelos -----------------------------------------
        # ------------------------------------------------------------------
        if sub_models is not None:
            # El usuario proporciona directamente los filtros que quiere usar
            self.models = list(sub_models)
        else:
            # Si no se indican, generamos modelos por defecto.
            # Idealmente: CV (KF), CA (KF aceleración), CT (EKF).
            from .kalman_filter import KalmanFilter  # import local para evitar ciclos
            try:
                from .kinematic_filters import CAKalmanFilter, CTRVExtendedKalmanFilter
            except ImportError:
                # En caso de fallo, degradar a múltiples KalmanFilter simples
                CAKalmanFilter, CTRVExtendedKalmanFilter = KalmanFilter, KalmanFilter  # type: ignore

            self.models = [KalmanFilter(), CAKalmanFilter(), CTRVExtendedKalmanFilter()]

            # Si se solicita simples escalados de ruido se respeta también
            if model_process_noise_scalers is not None:
                # Ajustar ruido del primer modelo según la lista suministrada
                for kf_obj, scale in zip(self.models, model_process_noise_scalers):
                    # Solo para filtros lineales simples
                    if hasattr(kf_obj, "_std_weight_position"):
                        kf_obj._std_weight_position *= scale  # type: ignore[attr-defined]
                        kf_obj._std_weight_velocity *= scale  # type: ignore[attr-defined]

        self.num_models: int = len(self.models)

        # Model probabilities (mu)
        if init_model_probabilities is None:
            init_model_probabilities = np.full(self.num_models, 1.0 / self.num_models)
        else:
            init_model_probabilities = np.asarray(init_model_probabilities, dtype=float)
            init_model_probabilities /= init_model_probabilities.sum()
        self.mu: np.ndarray = init_model_probabilities.copy()

        # Transition matrix Pi (rows i -> cols j: P(j | i))
        if transition_matrix is None:
            self.transition_matrix = self._default_transition_matrix(self.num_models)
        else:
            self.transition_matrix = np.asarray(transition_matrix, dtype=float)
            assert (
                self.transition_matrix.shape[0]
                == self.transition_matrix.shape[1]
                == self.num_models
            ), "Transition matrix must be square with size equal to number of models"
            # Normalise rows
            self.transition_matrix = (
                self.transition_matrix
                / self.transition_matrix.sum(axis=1, keepdims=True)
            )

        # For each model we keep its own state (mean, covariance)
        self.model_means: List[np.ndarray] = [mean.copy() for _ in range(self.num_models)]
        self.model_covs: List[np.ndarray] = [covariance.copy() for _ in range(self.num_models)]

        # Initialise fused state to provided mean/cov (already done by Track)

    # ------------------------------------------------------------------
    # Helper utilities -------------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _default_transition_matrix(m: int, stay_prob: float = 0.90) -> np.ndarray:
        """Return a default *m x m* transition matrix with *stay_prob* on the
        diagonal and the remaining mass spread uniformly across the other
        models.
        """
        off_diag = (1.0 - stay_prob) / (m - 1)
        Pi = np.full((m, m), off_diag)
        np.fill_diagonal(Pi, stay_prob)
        return Pi

    @staticmethod
    def _gaussian_likelihood(d: np.ndarray, S: np.ndarray) -> float:
        """Compute the likelihood *N(d; 0, S)* (i.e. centred at zero) for vector
        *d* with covariance *S*.

        Uses the log-space computation to avoid numerical underflow and then
        exponentiates the result.
        """
        k = d.shape[0]
        try:
            # Use Cholesky for numerical stability
            L = np.linalg.cholesky(S)
            log_det_S = 2.0 * np.log(np.diag(L)).sum()
            v = np.linalg.solve(L, d)
            maha = np.dot(v, v)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if S is not PD
            S_inv = np.linalg.pinv(S)
            log_det_S = np.log(np.linalg.det(S) + 1e-9)
            maha = float(d.T @ S_inv @ d)
        log_likelihood = -0.5 * (k * np.log(2.0 * np.pi) + log_det_S + maha)
        return float(np.exp(log_likelihood))

    # ------------------------------------------------------------------
    # Core IMM cycle ---------------------------------------------------
    # ------------------------------------------------------------------
    def _interaction(self):
        """Perform the mixing (interaction) step and return mixed states."""
        Pi = self.transition_matrix
        mu_prev = self.mu

        # Normalising constants for each destination model j
        c_j = Pi.T @ mu_prev  # shape (M,)

        mixed_means: List[np.ndarray] = []
        mixed_covs: List[np.ndarray] = []
        for j in range(self.num_models):
            if c_j[j] == 0:
                # Avoid division by zero – fallback to previous mean/cov for that model
                mixed_means.append(self.model_means[j])
                mixed_covs.append(self.model_covs[j])
                continue

            # Conditional mixing probabilities mu_{i|j}
            mu_ij = (Pi[:, j] * mu_prev) / c_j[j]

            # Mixed mean
            m_mean = sum(mu_ij[i] * self.model_means[i] for i in range(self.num_models))

            # Mixed covariance
            m_cov = np.zeros_like(self.model_covs[0])
            for i in range(self.num_models):
                diff = self.model_means[i] - m_mean
                m_cov += mu_ij[i] * (self.model_covs[i] + np.outer(diff, diff))

            mixed_means.append(m_mean)
            mixed_covs.append(m_cov)

        return mixed_means, mixed_covs, c_j

    def _fusion(self):
        """Fuse model-specific means/covs into a single state using *self.mu*."""
        fused_mean = sum(self.mu[i] * self.model_means[i] for i in range(self.num_models))

        fused_cov = np.zeros_like(self.model_covs[0])
        for i in range(self.num_models):
            diff = self.model_means[i] - fused_mean
            fused_cov += self.mu[i] * (self.model_covs[i] + np.outer(diff, diff))

        self.mean = fused_mean
        self.covariance = fused_cov

    # ------------------------------------------------------------------
    # Overridden public methods ---------------------------------------
    # ------------------------------------------------------------------
    def predict(self, _kf_unused=None):  # noqa: D401 – keep signature for compatibility
        """Override *Track.predict* – run IMM predict cycle."""
        # Interaction (mixing)
        mixed_means, mixed_covs, _ = self._interaction()

        # Model-specific prediction
        for i in range(self.num_models):
            self.model_means[i], self.model_covs[i] = self.models[i].predict(
                mixed_means[i], mixed_covs[i]
            )

        # Age bookkeeping (same as original Track.predict)
        self.age += 1
        self.time_since_update += 1
        self.original_ltwh = None
        self.det_conf = None
        self.instance_mask = None
        self.others = None
        self.cosine_similarity = None

        # No measurement yet – fuse predicted states to update public mean/cov
        self._fusion()

    def update(self, _kf_unused, detection, similarity=None):  # noqa: D401
        """Override *Track.update* to run IMM measurement update."""
        z = detection.to_xyah()

        # ------------------------------------------------------------------
        # Compute likelihoods for each model and run its update step
        # ------------------------------------------------------------------
        likelihoods = np.zeros(self.num_models, dtype=float)

        for i in range(self.num_models):
            # Project to measurement space to obtain innovation statistics
            z_mean, S = self.models[i].project(self.model_means[i], self.model_covs[i])
            residual = z - z_mean

            likelihoods[i] = self._gaussian_likelihood(residual, S)

            # Perform the standard KF correction
            self.model_means[i], self.model_covs[i] = self.models[i].update(
                self.model_means[i], self.model_covs[i], z
            )

        # ------------------------------------------------------------------
        # Update model probabilities (mu)
        # ------------------------------------------------------------------
        # mu_bar_j = sum_i Pi_{ij} * mu_prev_i
        mu_bar = self.transition_matrix.T @ self.mu
        mu_likelihood = mu_bar * likelihoods
        if mu_likelihood.sum() == 0:
            # Degenerate case – reset to uniform
            self.mu = np.full(self.num_models, 1.0 / self.num_models)
        else:
            self.mu = mu_likelihood / mu_likelihood.sum()

        # ------------------------------------------------------------------
        # Fuse into global state
        # ------------------------------------------------------------------
        self._fusion()

        # ------------------------------------------------------------------
        # Copy over the appearance / meta-data housekeeping from the original
        # implementation.
        # ------------------------------------------------------------------
        self.original_ltwh = detection.get_ltwh()
        self.features.append(detection.feature)
        self.persistent_features.append(detection.feature)

        # Limit persistent features to a reasonable budget (keep last 2000)
        max_persistent_features = 2000
        if len(self.persistent_features) > max_persistent_features:
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