from typing import Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from src.config import KALMAN_DT, KALMAN_PROCESS_NOISE_VAR, KALMAN_MEASUREMENT_NOISE_VAR
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ConstantVelocityKalmanFilter:
    def __init__(
        self,
        dt: float = KALMAN_DT,
        q_var: float = KALMAN_PROCESS_NOISE_VAR,
        r_var: float = KALMAN_MEASUREMENT_NOISE_VAR,
    ):
        self.dt = float(dt)
        self.q_var = float(q_var)
        self.r_var = float(r_var)

        self.F = np.eye(6, dtype=float)
        self.F[0, 3] = self.dt
        self.F[1, 4] = self.dt
        self.F[2, 5] = self.dt

        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.Q = np.eye(6, dtype=float) * self.q_var
        self.R = np.eye(3, dtype=float) * self.r_var

        self.state = np.zeros(6, dtype=float)
        self.P = np.eye(6, dtype=float)
        self.is_initialized = False

    def initialize(
        self,
        initial_position: Union[Sequence[float], np.ndarray],
        initial_velocity: Optional[Union[Sequence[float], np.ndarray]] = None,
    ) -> "ConstantVelocityKalmanFilter":
        pos = np.asarray(initial_position, dtype=float).ravel()
        if len(pos) < 3:
            raise ValueError(f"initial_position must have at least 3 elements, got {len(pos)}")

        self.state = np.zeros(6, dtype=float)
        self.state[:3] = pos[:3]

        if initial_velocity is not None:
            vel = np.asarray(initial_velocity, dtype=float).ravel()
            self.state[3:] = vel[:3]

        self.P = np.eye(6, dtype=float)
        self.is_initialized = True
        return self

    def fit(
        self,
        positions: np.ndarray,
        dt: Optional[float] = None,
    ) -> "ConstantVelocityKalmanFilter":
        pos = np.asarray(positions, dtype=float)
        if len(pos) == 0:
            raise ValueError("Cannot fit Kalman filter on empty position array.")

        if dt is not None and dt > 0:
            self.dt = float(dt)
            self.F[0, 3] = self.dt
            self.F[1, 4] = self.dt
            self.F[2, 5] = self.dt

        if len(pos) >= 2:
            initial_vel = (pos[1] - pos[0]) / self.dt
        else:
            initial_vel = np.zeros(3)

        self.initialize(pos[0], initial_velocity=initial_vel)
        logger.info(f"Kalman filter initialized at {self.state[:3]} with velocity {self.state[3:]}")
        return self

    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_initialized:
            raise RuntimeError("Filter must be initialized before predict().")

        predicted_state = self.F @ self.state
        predicted_cov = self.F @ self.P @ self.F.T + self.Q
        return predicted_state, predicted_cov

    def predict_position(self) -> Tuple[np.ndarray, float]:
        pred_state, pred_cov = self.predict()
        pos_var = float(np.trace(pred_cov[:3, :3]))
        return pred_state[:3], float(np.sqrt(pos_var))

    def update(
        self,
        measurement: Union[Sequence[float], np.ndarray],
        predicted_state: Optional[np.ndarray] = None,
        predicted_cov: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if predicted_state is None or predicted_cov is None:
            predicted_state, predicted_cov = self.predict()

        meas = np.asarray(measurement, dtype=float).ravel()[:3]
        y = meas - (self.H @ predicted_state)
        S = (self.H @ predicted_cov @ self.H.T) + self.R
        K = predicted_cov @ self.H.T @ np.linalg.inv(S)

        self.state = predicted_state + (K @ y)
        I_KH = np.eye(6) - (K @ self.H)
        self.P = (I_KH @ predicted_cov @ I_KH.T) + (K @ self.R @ K.T)

        return self.state, self.P

    def step(
        self,
        measurement: Union[Sequence[float], np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        pred_state, pred_cov = self.predict()
        pred_pos = pred_state[:3].copy()
        uncertainty = float(np.sqrt(np.trace(pred_cov[:3, :3])))

        self.update(measurement, predicted_state=pred_state, predicted_cov=pred_cov)
        return pred_pos, self.state[:3].copy(), uncertainty

    def filter_sequence(
        self,
        positions: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pos = np.asarray(positions, dtype=float)
        n = len(pos)
        if n < 2:
            raise ValueError("Need at least 2 position measurements for Kalman filtering.")

        self.initialize(pos[0])
        predicted_positions = []
        errors_3d = []

        for i in range(1, n):
            pred_pos, _, _ = self.step(pos[i])
            err = float(np.linalg.norm(pos[i] - pred_pos))
            predicted_positions.append(pred_pos)
            errors_3d.append(err)

        return np.asarray(predicted_positions), np.asarray(errors_3d)

    def predict_sequence(
        self,
        positions: np.ndarray,
        frames: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        pos = np.asarray(positions, dtype=float)
        n = len(pos)
        if n < 2:
            raise ValueError("Need at least 2 position measurements for sequence prediction.")

        frame_indices = np.asarray(frames) if frames is not None else np.arange(n)
        self.initialize(pos[0])
        records = []

        for i in range(1, n):
            actual = pos[i]
            pred_pos, _, uncertainty = self.step(actual)
            err_vec = actual - pred_pos
            err_3d = float(np.linalg.norm(err_vec))

            records.append({
                "frame": frame_indices[i],
                "actual_x_mm": actual[0],
                "actual_y_mm": actual[1],
                "actual_z_mm": actual[2],
                "predicted_x_mm": pred_pos[0],
                "predicted_y_mm": pred_pos[1],
                "predicted_z_mm": pred_pos[2],
                "error_x_mm": abs(err_vec[0]),
                "error_y_mm": abs(err_vec[1]),
                "error_z_mm": abs(err_vec[2]),
                "error_3d_mm": err_3d,
                "uncertainty_3d_mm": uncertainty,
            })

        return pd.DataFrame(records)
