# vim: expandtab:ts=4:sw=4
"""deep_sort_realtime.deep_sort.kinematic_filters

Filtros cinemáticos adicionales para el IMM:
* CAKalmanFilter           – modelo lineal de aceleración constante.
* CTRVExtendedKalmanFilter – stub EKF (próxima implementación completa).

Ambos comparten la misma interfaz pública que `KalmanFilter` original para
ser intercambiables en `IMMTrack`.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg

# Los filtros se inspiran en la implementación de `KalmanFilter` existente
# (deep_sort_realtime.deep_sort.kalman_filter).  Copiamos su estructura y
# ampliamos la dinámica.


class CAKalmanFilter:
    """Filtro de Kalman lineal con aceleración constante (CA).

    El estado abarca posición, velocidad y aceleración en cada componente:
        x, y, a, h, vx, vy, va, vh, ax, ay, aa, ah   (12 dimensiones)
    donde (x, y) es el centro, *a* el aspect-ratio y *h* la altura del bbox.

    La matriz de transición utiliza ∆t = 1 (frame) por simplicidad.
    """

    ndim: int = 4  # nº de componentes de la medición (x,y,a,h)
    dt: float = 1.0

    def __init__(self):
        ndim, dt = self.ndim, self.dt
        order = 2  # aceleración constante ⇒ 2ª derivada

        # Construir la matriz de transición de movimiento (F)
        # Para cada dimensión: [pos, vel, acc]
        F_block = np.array(
            [
                [1, dt, 0.5 * dt * dt],
                [0, 1, dt],
                [0, 0, 1],
            ]
        )
        # Repetir para las 4 variables (x, y, a, h)
        self._motion_mat = np.block(
            [
                [np.kron(np.eye(ndim), F_block)],
            ]
        )

        # Matriz de observación: medimos solo posición (x, y, a, h)
        H_block = np.array([[1, 0, 0]])  # toma solo la posición
        self._update_mat = np.kron(np.eye(ndim), H_block)

        # Parámetros heurísticos de ruido (se ajustan como en el KF original)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._std_weight_acceleration = 1.0 / 3200

    # ------------------------------------------------------------------
    # API pública (idéntica al KalmanFilter base) -----------------------
    # ------------------------------------------------------------------
    def initiate(self, measurement: np.ndarray):
        """Crear un nuevo track a partir de la medición (x, y, a, h)."""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean_acc = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel, mean_acc]

        std = [
            2 * self._std_weight_position * measurement[3],  # x
            2 * self._std_weight_position * measurement[3],  # y
            1e-2,  # a
            2 * self._std_weight_position * measurement[3],  # h
            10 * self._std_weight_velocity * measurement[3],  # vx
            10 * self._std_weight_velocity * measurement[3],  # vy
            1e-5,  # va
            10 * self._std_weight_velocity * measurement[3],  # vh
            10 * self._std_weight_acceleration * measurement[3],  # ax
            10 * self._std_weight_acceleration * measurement[3],  # ay
            1e-5,  # aa
            10 * self._std_weight_acceleration * measurement[3],  # ah
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """Paso de predicción CA."""
        # Ruido de proceso – escalado con h
        std_pos = [self._std_weight_position * mean[3]] * 4
        std_vel = [self._std_weight_velocity * mean[3]] * 4
        std_acc = [self._std_weight_acceleration * mean[3]] * 4
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel, std_acc]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T))
            + motion_cov
        )
        return mean, covariance

    def project(self, mean, covariance):
        """Proyectar estado a espacio de medición (x, y, a, h)."""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Corrección con nueva medición."""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_cov

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Distancia de puerta de Mahalanobis (idéntica lógica al KF base)."""
        from .kalman_filter import chi2inv95  # import local para evitar ciclos

        projected_mean, projected_cov = self.project(mean, covariance)
        if only_position:
            projected_mean, projected_cov = projected_mean[:2], projected_cov[:2, :2]
            measurements = measurements[:, :2]
            gating_dim = 2
        else:
            gating_dim = 4

        # cholesky y distancia cuadrática
        cholesky_factor = np.linalg.cholesky(projected_cov)
        d = measurements - projected_mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha


class CTRVExtendedKalmanFilter:
    """Filtro de Kalman *extendido* para el modelo CTRV (Coordinated Turn Rate & Velocity).

    Estado (7×1):
        x, y        – centro del bounding box
        a, h        – aspect-ratio y altura (se asumen constantes)
        v           – velocidad (módulo)
        ψ (yaw)     – rumbo/heading
        ω           – razón de giro (yaw rate)

    Las mediciones siguen siendo (x, y, a, h).
    """

    dt: float = 1.0  # tamaño de paso (frames)

    def __init__(self):
        # Pesos heurísticos compatibles con KalmanFilter base
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
        self._std_weight_turnrate = 1.0 / 720

    # ------------------------------------------------------------------
    # Utilidades internas ----------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normaliza ángulo a rango [-π, π]."""
        return (angle + np.pi) % (2 * np.pi) - np.pi

    # ------------------------------------------------------------------
    # API público -------------------------------------------------------
    # ------------------------------------------------------------------
    def initiate(self, measurement: np.ndarray):
        """Inicializa el estado a partir de la medición (x, y, a, h)."""
        x, y, a, h = measurement
        mean = np.array([x, y, a, h, 0.0, 0.0, 0.0])  # v=0, yaw=0, yaw_rate=0

        std = [
            2 * self._std_weight_position * h,  # x
            2 * self._std_weight_position * h,  # y
            1e-2,  # a
            2 * self._std_weight_position * h,  # h
            10 * self._std_weight_velocity * h,  # v
            np.pi / 4,  # yaw incertidumbre (~45º)
            10 * self._std_weight_turnrate * h,  # yaw_rate
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def _state_transition(self, mean: np.ndarray):
        """Propaga el estado aplicando CTRV durante dt."""
        px, py, a, h, v, yaw, yaw_rate = mean
        dt = self.dt

        if abs(yaw_rate) > 1e-4:
            px_p = px + v / yaw_rate * (np.sin(yaw + yaw_rate * dt) - np.sin(yaw))
            py_p = py + v / yaw_rate * (-np.cos(yaw + yaw_rate * dt) + np.cos(yaw))
        else:
            px_p = px + v * dt * np.cos(yaw)
            py_p = py + v * dt * np.sin(yaw)

        a_p = a  # aspect-ratio assume constant
        h_p = h  # altura assume constant
        v_p = v  # velocidad constante
        yaw_p = self._normalize_angle(yaw + yaw_rate * dt)
        yaw_rate_p = yaw_rate  # yaw_rate constante

        return np.array([px_p, py_p, a_p, h_p, v_p, yaw_p, yaw_rate_p])

    def _jacobian_F(self, mean: np.ndarray):
        """Calcula la matriz Jacobiana F = ∂f/∂x del modelo CTRV."""
        px, py, a, h, v, yaw, yaw_rate = mean
        dt = self.dt

        F = np.eye(7)

        if abs(yaw_rate) > 1e-4:
            v_by_w = v / yaw_rate
            sin_yaw = np.sin(yaw)
            cos_yaw = np.cos(yaw)
            sin_yaw_wdt = np.sin(yaw + yaw_rate * dt)
            cos_yaw_wdt = np.cos(yaw + yaw_rate * dt)

            F[0, 4] = 1.0 / yaw_rate * (sin_yaw_wdt - sin_yaw)
            F[0, 5] = v_by_w * (cos_yaw_wdt - cos_yaw)
            F[0, 6] = (
                v * (cos_yaw_wdt * dt * yaw_rate - (sin_yaw_wdt - sin_yaw)) / (yaw_rate ** 2)
            )

            F[1, 4] = 1.0 / yaw_rate * (-cos_yaw_wdt + cos_yaw)
            F[1, 5] = v_by_w * (sin_yaw_wdt - sin_yaw)
            F[1, 6] = (
                v * (sin_yaw_wdt * dt * yaw_rate - (-cos_yaw_wdt + cos_yaw)) / (yaw_rate ** 2)
            )
        else:
            F[0, 4] = dt * np.cos(yaw)
            F[0, 5] = -v * dt * np.sin(yaw)
            F[1, 4] = dt * np.sin(yaw)
            F[1, 5] = v * dt * np.cos(yaw)

        # yaw derivatives
        F[5, 6] = dt

        return F

    def predict(self, mean: np.ndarray, covariance: np.ndarray):
        """Paso de predicción EKF."""
        mean_pred = self._state_transition(mean)
        F = self._jacobian_F(mean)

        # Ruido de proceso Q (heurístico similar a KF base)
        std_pos = [self._std_weight_position * mean[3]] * 2  # x,y
        std_a_h = [1e-2, self._std_weight_position * mean[3]]  # a, h
        std_v = self._std_weight_velocity * mean[3]
        std_yaw = np.deg2rad(5)  # ~5º
        std_yaw_rate = self._std_weight_turnrate * mean[3]

        q = np.square(
            np.r_[std_pos, std_a_h, std_v, std_yaw, std_yaw_rate]
        )
        Q = np.diag(q)

        covariance_pred = F @ covariance @ F.T + Q
        return mean_pred, covariance_pred

    # ------------------- medición ------------------------------------
    def project(self, mean: np.ndarray, covariance: np.ndarray):
        """Proyecta el estado al espacio de medición (x, y, a, h)."""
        # h(x) = [x, y, a, h]
        H = np.zeros((4, 7))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # a
        H[3, 3] = 1  # h

        mean_z = H @ mean

        # Ruido de medición R
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        R = np.diag(np.square(std))

        S = H @ covariance @ H.T + R
        return mean_z, S

    def update(self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray):
        """Actualiza con la medición mediante EKF."""
        mean_z, S = self.project(mean, covariance)

        # Ganancia de Kalman
        H = np.zeros((4, 7))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1
        H[3, 3] = 1

        K = covariance @ H.T @ np.linalg.inv(S)

        innovation = measurement - mean_z

        mean_upd = mean + K @ innovation
        mean_upd[5] = self._normalize_angle(mean_upd[5])  # normaliza yaw

        covariance_upd = (np.eye(7) - K @ H) @ covariance
        return mean_upd, covariance_upd

    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """Distancia de Mahalanobis para graficar/gateo."""
        from .kalman_filter import chi2inv95

        mean_z, S = self.project(mean, covariance)

        if only_position:
            mean_z = mean_z[:2]
            S = S[:2, :2]
            measurements = measurements[:, :2]
            gating_dim = 2
        else:
            gating_dim = 4

        chol_factor = np.linalg.cholesky(S)
        d = measurements - mean_z
        z = scipy.linalg.solve_triangular(
            chol_factor, d.T, lower=True, check_finite=False, overwrite_b=True
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha 