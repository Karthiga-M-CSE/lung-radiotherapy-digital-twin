from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import joblib

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_csv(file_path: Union[str, Path], required_columns: Optional[List[str]] = None) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"Required CSV file does not exist: {path}")

    df = pd.read_csv(path)
    logger.debug(f"Loaded {len(df)} rows from {path.name}")

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing required columns in {path.name}: {missing}")
            raise ValueError(f"Missing columns in {path.name}: {missing}")

    return df


def save_csv(df: pd.DataFrame, file_path: Union[str, Path], index: bool = False) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    logger.info(f"Saved {len(df)} rows to: {path}")
    return path


def load_rpm_signal(file_path: Union[str, Path]) -> np.ndarray:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"RPM signal file not found: {path}")

    try:
        data = np.loadtxt(path)
    except Exception:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().replace(",", "\n").splitlines()
        values = []
        for line in raw:
            line = line.strip()
            if line:
                try:
                    values.append(float(line))
                except ValueError:
                    continue
        data = np.asarray(values, dtype=float)

    logger.debug(f"Loaded RPM signal with {len(data)} samples from {path.name}")
    return data


def save_joblib(obj: Any, file_path: Union[str, Path]) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved model artifact: {path}")
    return path


def load_joblib(file_path: Union[str, Path]) -> Any:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)
