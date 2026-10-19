from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.config import N_LAGS, ALPHA, MODELS_DIR
from src.utils.io import save_joblib, load_joblib
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RespiratoryRidgePredictor:
    def __init__(self, alpha: float = ALPHA, n_lags: int = N_LAGS):
        self.alpha = float(alpha)
        self.n_lags = int(n_lags)
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=self.alpha)
        self.is_fitted = False
        self._X_train_raw: Optional[np.ndarray] = None
        self._y_train_raw: Optional[np.ndarray] = None

    def create_features(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str] = ("rpm", "x_mm", "y_mm", "z_mm"),
        target_cols: Sequence[str] = ("x_mm", "y_mm", "z_mm"),
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        df_sorted = df.sort_values("frame").copy().reset_index(drop=True)
        feature_names: List[str] = []

        for lag in range(1, self.n_lags + 1):
            for col in feature_cols:
                name = f"{col}_lag{lag}"
                df_sorted[name] = df_sorted[col].shift(lag)
                feature_names.append(name)

        df_clean = df_sorted.dropna(subset=feature_names + list(target_cols)).reset_index(drop=True)

        X = df_clean[feature_names].values.astype(float)
        y = df_clean[list(target_cols)].values.astype(float)
        frames = df_clean["frame"].values

        return X, y, frames, feature_names

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RespiratoryRidgePredictor":
        X_arr = np.asarray(X_train, dtype=float)
        y_arr = np.asarray(y_train, dtype=float)

        if len(X_arr) == 0 or len(y_arr) == 0:
            raise ValueError("Training data cannot be empty.")
        if len(X_arr) != len(y_arr):
            raise ValueError(f"Length mismatch: X has {len(X_arr)} rows, y has {len(y_arr)} rows.")

        X_scaled = self.scaler.fit_transform(X_arr)
        self.model.fit(X_scaled, y_arr)
        self.is_fitted = True
        self._X_train_raw = X_arr.copy()
        self._y_train_raw = y_arr.copy()

        logger.info(f"Ridge predictor fitted: alpha={self.alpha}, n_lags={self.n_lags}, samples={len(X_arr)}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict() is called.")

        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(1, -1)

        X_scaled = self.scaler.transform(X_arr)
        return self.model.predict(X_scaled)

    def update(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        refit: bool = True,
    ) -> "RespiratoryRidgePredictor":
        X_add = np.asarray(X_new, dtype=float)
        y_add = np.asarray(y_new, dtype=float)

        if X_add.ndim == 1:
            X_add = X_add.reshape(1, -1)
        if y_add.ndim == 1:
            y_add = y_add.reshape(1, -1)

        if self._X_train_raw is not None and self._y_train_raw is not None:
            combined_X = np.vstack([self._X_train_raw, X_add])
            combined_y = np.vstack([self._y_train_raw, y_add])
        else:
            combined_X = X_add
            combined_y = y_add

        if refit:
            self.fit(combined_X, combined_y)
        else:
            self._X_train_raw = combined_X
            self._y_train_raw = combined_y

        return self

    def predict_sequence(
        self,
        X: np.ndarray,
        y_actual: np.ndarray,
        frames: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        y_pred = self.predict(X)
        y_true = np.asarray(y_actual, dtype=float)

        err_vec = np.abs(y_true - y_pred)
        err_3d = np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))
        frame_indices = np.asarray(frames) if frames is not None else np.arange(len(y_pred))

        return pd.DataFrame({
            "frame": frame_indices,
            "actual_x_mm": y_true[:, 0],
            "actual_y_mm": y_true[:, 1],
            "actual_z_mm": y_true[:, 2],
            "predicted_x_mm": y_pred[:, 0],
            "predicted_y_mm": y_pred[:, 1],
            "predicted_z_mm": y_pred[:, 2],
            "error_x_mm": err_vec[:, 0],
            "error_y_mm": err_vec[:, 1],
            "error_z_mm": err_vec[:, 2],
            "error_3d_mm": err_3d,
        })

    def save(
        self,
        model_path: Optional[Union[str, Path]] = None,
        scaler_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[Path, Path]:
        m_path = Path(model_path or (MODELS_DIR / "respiratory_ridge_model.joblib"))
        s_path = Path(scaler_path or (MODELS_DIR / "respiratory_ridge_scaler.joblib"))

        save_joblib(self.model, m_path)
        save_joblib(self.scaler, s_path)
        logger.info(f"Saved Ridge model to {m_path} and scaler to {s_path}")
        return m_path, s_path

    def load(
        self,
        model_path: Union[str, Path],
        scaler_path: Union[str, Path],
    ) -> "RespiratoryRidgePredictor":
        self.model = load_joblib(Path(model_path))
        self.scaler = load_joblib(Path(scaler_path))
        self.is_fitted = True
        logger.info(f"Loaded Ridge model from {model_path} and scaler from {scaler_path}")
        return self
