# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
import math

# from sklearn.utils.linear_assignment_ import linear_assignment
from scipy.optimize import linear_sum_assignment
from . import kalman_filter


INFTY_COST = 1e5


def min_cost_matching(
    distance_metric,
    max_distance,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
    direction_penalty=True  # Nuevo parámetro para activar/desactivar penalización por dirección
):
    """Solve linear assignment problem.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    direction_penalty : bool
        If True, apply a direction penalty to the cost matrix based on movement patterns.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    if len(detection_indices) == 0 or len(track_indices) == 0:
        return [], track_indices, detection_indices  # Nothing to match.

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)
    
    # Aplicar penalización por diferencia de trayectoria si está habilitado
    if direction_penalty:
        cost_matrix = apply_trajectory_penalty(cost_matrix, tracks, detections, track_indices, detection_indices)
    
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    # indices = linear_assignment(cost_matrix)
    indices = np.vstack(linear_sum_assignment(cost_matrix)).T

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(detection_indices):
        if col not in indices[:, 1]:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(track_indices):
        if row not in indices[:, 0]:
            unmatched_tracks.append(track_idx)
    for row, col in indices:
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))
    return matches, unmatched_tracks, unmatched_detections


def matching_cascade(
    distance_metric,
    max_distance,
    cascade_depth,
    tracks,
    detections,
    track_indices=None,
    detection_indices=None,
):
    """Run matching cascade.

    Parameters
    ----------
    distance_metric : Callable[List[Track], List[Detection], List[int], List[int]) -> ndarray
        The distance metric is given a list of tracks and detections as well as
        a list of N track indices and M detection indices. The metric should
        return the NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection indices.
    max_distance : float
        Gating threshold. Associations with cost larger than this value are
        disregarded.
    cascade_depth: int
        The cascade depth, should be se to the maximum track age.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : Optional[List[int]]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above). Defaults to all tracks.
    detection_indices : Optional[List[int]]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above). Defaults to all
        detections.

    Returns
    -------
    (List[(int, int)], List[int], List[int])
        Returns a tuple with the following three entries:
        * A list of matched track and detection indices.
        * A list of unmatched track indices.
        * A list of unmatched detection indices.

    """
    if track_indices is None:
        track_indices = list(range(len(tracks)))
    if detection_indices is None:
        detection_indices = list(range(len(detections)))

    unmatched_detections = detection_indices
    matches = []
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:  # No detections left
            break

        track_indices_l = [
            k for k in track_indices if tracks[k].time_since_update == 1 + level
        ]
        if len(track_indices_l) == 0:  # Nothing to match at this level
            continue

        matches_l, _, unmatched_detections = min_cost_matching(
            distance_metric,
            max_distance,
            tracks,
            detections,
            track_indices_l,
            unmatched_detections,
            direction_penalty=True  # Activar penalización por dirección
        )
        matches += matches_l
    unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))
    return matches, unmatched_tracks, unmatched_detections


def apply_trajectory_penalty(cost_matrix, tracks, detections, track_indices, detection_indices):
    """
    Aplica una penalización a la matriz de costos basada en la diferencia de dirección
    y velocidad entre los tracks y las nuevas detecciones.
    
    Parameters
    ----------
    cost_matrix : ndarray
        Matriz de costos original
    tracks : List[track.Track]
        Lista de tracks existentes
    detections : List[detection.Detection]
        Lista de nuevas detecciones
    track_indices : List[int]
        Índices de los tracks a considerar
    detection_indices : List[int]
        Índices de las detecciones a considerar
        
    Returns
    -------
    ndarray
        Matriz de costos con penalizaciones aplicadas
    """
    modified_cost_matrix = cost_matrix.copy()
    
    # Penalizaciones para diferentes condiciones
    DIRECTION_DIFF_PENALTY = 0.75  # Penalización por diferencia significativa en dirección
    LONG_ABSENCE_PENALTY = 0.5     # Penalización para tracks que han estado ausentes por mucho tiempo
    VELOCITY_DIFF_PENALTY = 0.5    # Penalización por diferencia significativa en velocidad
    
    # Umbrales para considerar diferencias significativas
    DIRECTION_THRESHOLD = math.pi/2  # 90 grados de diferencia en dirección
    LONG_ABSENCE_THRESHOLD = 20      # Frames sin actualización considerados como ausencia larga
    VELOCITY_DIFF_THRESHOLD = 5.0    # Diferencia significativa en velocidad
    
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        
        # Obtener dirección y velocidad promedio del track
        track_direction = track.get_movement_direction()
        track_velocity = track.get_average_velocity()
        
        if track_direction is None or track_velocity is None:
            continue  # No tenemos suficiente información para este track
        
        # La magnitud de la velocidad es importante para determinar si un objeto está en movimiento
        track_speed = np.linalg.norm(track_velocity)
        
        # Solo aplicar penalizaciones de dirección si el objeto está en movimiento significativo
        if track_speed > 1.0:
            for col, detection_idx in enumerate(detection_indices):
                detection = detections[detection_idx]
                
                # Si el track estuvo ausente por mucho tiempo, aumentar costo
                if track.time_since_update > LONG_ABSENCE_THRESHOLD:
                    modified_cost_matrix[row, col] += LONG_ABSENCE_PENALTY
                
                # Calcular posición relativa de la detección respecto al último punto del track
                if len(track.position_history) > 0:
                    last_pos = track.position_history[-1]
                    det_pos = np.array([detection.to_xyah()[0], detection.to_xyah()[1]])
                    
                    # Vector desde la última posición del track hacia la detección
                    direction_vector = det_pos - last_pos
                    
                    # Solo calcular si el vector tiene longitud significativa
                    if np.linalg.norm(direction_vector) > 1.0:
                        # Calcular dirección del movimiento de la detección relativa al track
                        det_direction = np.arctan2(direction_vector[1], direction_vector[0])
                        
                        # Calcular diferencia de dirección (considerando que es circular)
                        direction_diff = min(
                            abs(det_direction - track_direction),
                            2 * math.pi - abs(det_direction - track_direction)
                        )
                        
                        # Aplicar penalización si la diferencia de dirección es significativa
                        if direction_diff > DIRECTION_THRESHOLD:
                            # La penalización es proporcional a la diferencia de dirección
                            penalty = DIRECTION_DIFF_PENALTY * (direction_diff / math.pi)
                            modified_cost_matrix[row, col] += penalty
    
    return modified_cost_matrix


def gate_cost_matrix(
    kf,
    cost_matrix,
    tracks,
    detections,
    track_indices,
    detection_indices,
    gated_cost=INFTY_COST,
    only_position=False,
):
    """Invalidate infeasible entries in cost matrix based on the state
    distributions obtained by Kalman filtering.

    Parameters
    ----------
    kf : The Kalman filter.
    cost_matrix : ndarray
        The NxM dimensional cost matrix, where N is the number of track indices
        and M is the number of detection indices, such that entry (i, j) is the
        association cost between `tracks[track_indices[i]]` and
        `detections[detection_indices[j]]`.
    tracks : List[track.Track]
        A list of predicted tracks at the current time step.
    detections : List[detection.Detection]
        A list of detections at the current time step.
    track_indices : List[int]
        List of track indices that maps rows in `cost_matrix` to tracks in
        `tracks` (see description above).
    detection_indices : List[int]
        List of detection indices that maps columns in `cost_matrix` to
        detections in `detections` (see description above).
    gated_cost : Optional[float]
        Entries in the cost matrix corresponding to infeasible associations are
        set this value. Defaults to a very large value.
    only_position : Optional[bool]
        If True, only the x, y position of the state distribution is considered
        during gating. Defaults to False.

    Returns
    -------
    ndarray
        Returns the modified cost matrix.

    """
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])
    for row, track_idx in enumerate(track_indices):
        track = tracks[track_idx]
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost
    return cost_matrix
