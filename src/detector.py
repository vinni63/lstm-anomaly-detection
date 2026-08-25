"""Prediction-error anomaly scoring and validation-derived thresholding."""
from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from src.config import LABEL_COLUMN, METRIC_COLUMN, TIMESTAMP_COLUMN
    from src.preprocessing import inverse_values
    from src.sequence_builder import create_sequences
except ModuleNotFoundError:
    from config import LABEL_COLUMN, METRIC_COLUMN, TIMESTAMP_COLUMN
    from preprocessing import inverse_values
    from sequence_builder import create_sequences


def calculate_threshold(validation_errors, percentile: float = 99.0) -> float:
    """Use normal validation errors: threshold reflects expected normal model error."""
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100].")
    errors = np.asarray(validation_errors, dtype=float)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("validation_errors must be a non-empty finite array.")
    return float(np.percentile(errors, percentile))


def classify_errors(errors, threshold: float) -> np.ndarray:
    return (np.asarray(errors) > threshold).astype(int)


def detect_frame(model, scaled_values, frame: pd.DataFrame, scaler, window_size: int, threshold: float) -> pd.DataFrame:
    """Predict each arriving target from preceding window, then score observed error."""
    x, y_scaled = create_sequences(scaled_values, window_size)
    predicted_scaled = model.predict(x, verbose=0).ravel()
    actual = inverse_values(y_scaled, scaler)
    predicted = inverse_values(predicted_scaled, scaler)
    errors = np.abs(actual - predicted)
    # Target i corresponds to frame row i + window_size.
    result = pd.DataFrame({
        TIMESTAMP_COLUMN: frame[TIMESTAMP_COLUMN].iloc[window_size:].to_numpy(),
        "actual": actual,
        "predicted": predicted,
        "absolute_error": errors,
        "threshold": threshold,
        "detected_anomaly": classify_errors(errors, threshold),
    })
    if LABEL_COLUMN in frame:
        result["true_anomaly"] = frame[LABEL_COLUMN].iloc[window_size:].to_numpy()
    return result

