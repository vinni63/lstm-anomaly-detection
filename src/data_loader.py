"""Safe loading and cleaning for local CPU-monitoring CSV files."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

try:
    from src.config import LABEL_COLUMN, METRIC_COLUMN, TIMESTAMP_COLUMN
except ModuleNotFoundError:
    from config import LABEL_COLUMN, METRIC_COLUMN, TIMESTAMP_COLUMN


def load_cpu_data(path: str | Path) -> pd.DataFrame:
    """Parse timestamps, remove duplicates, validate CPU values and fill small gaps."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    frame = pd.read_csv(path)
    required = {TIMESTAMP_COLUMN, METRIC_COLUMN}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"CSV must contain {sorted(required)}; missing {sorted(missing)}")
    frame[TIMESTAMP_COLUMN] = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame[METRIC_COLUMN] = pd.to_numeric(frame[METRIC_COLUMN], errors="coerce")
    frame = frame.dropna(subset=[TIMESTAMP_COLUMN, METRIC_COLUMN])
    frame = frame.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
    frame = frame.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    # Interpolation supports occasional missing metric observations in user data.
    frame[METRIC_COLUMN] = frame[METRIC_COLUMN].interpolate().bfill().ffill()
    if frame.empty:
        raise ValueError("No valid timestamp/cpu_usage rows remain after cleaning.")
    if (frame[METRIC_COLUMN] < 0).any() or (frame[METRIC_COLUMN] > 100).any():
        raise ValueError("cpu_usage must be in the range 0 to 100.")
    if LABEL_COLUMN in frame:
        frame[LABEL_COLUMN] = pd.to_numeric(frame[LABEL_COLUMN], errors="coerce").fillna(0).astype(int).clip(0, 1)
    return frame

