"""Chronological split and scaling. No random shuffling is used for time series."""
from __future__ import annotations
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

try:
    from src.config import METRIC_COLUMN, SCALER_PATH, TRAIN_FRACTION, VALIDATION_FRACTION
except ModuleNotFoundError:
    from config import METRIC_COLUMN, SCALER_PATH, TRAIN_FRACTION, VALIDATION_FRACTION


def chronological_split(frame: pd.DataFrame, train_fraction=TRAIN_FRACTION, validation_fraction=VALIDATION_FRACTION):
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must be positive and sum to less than 1.")
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * (train_fraction + validation_fraction))
    return frame.iloc[:train_end].copy(), frame.iloc[train_end:validation_end].copy(), frame.iloc[validation_end:].copy()


def fit_scaler(train_frame: pd.DataFrame) -> MinMaxScaler:
    scaler = MinMaxScaler()
    scaler.fit(train_frame[[METRIC_COLUMN]])  # Fit ONLY to training data: avoids leakage.
    return scaler


def transform_values(frame: pd.DataFrame, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.transform(frame[[METRIC_COLUMN]]).astype("float32")


def inverse_values(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def save_scaler(scaler: MinMaxScaler, path=SCALER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)


def load_scaler(path=SCALER_PATH) -> MinMaxScaler:
    return joblib.load(path)

