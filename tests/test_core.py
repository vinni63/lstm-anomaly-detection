import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.detector import calculate_threshold, classify_errors
from src.preprocessing import chronological_split, fit_scaler, transform_values
from src.realtime import SimulatedRealtimeDetector
from src.sequence_builder import create_sequences


def test_sequence_generation_shape_and_target():
    x, y = create_sequences(np.array([1, 2, 3, 4, 5]), 3)
    assert x.shape == (2, 3, 1)
    assert x[0, :, 0].tolist() == [1, 2, 3]
    assert y[:, 0].tolist() == [4, 5]


def test_preprocessing_is_chronological_and_scaler_fits_train_only():
    frame = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=10, freq="min"), "cpu_usage": range(10)})
    train, validation, test = chronological_split(frame, .6, .2)
    scaler = fit_scaler(train)
    assert len(train) == 6 and len(validation) == 2 and len(test) == 2
    assert transform_values(test, scaler).max() > 1  # Later values are not leaked into scaler fit.


def test_threshold_and_classification():
    threshold = calculate_threshold([1, 2, 3, 4], 75)
    assert threshold == 3.25
    assert classify_errors([3.25, 3.26], threshold).tolist() == [0, 1]


class DummyModel:
    def predict(self, x, verbose=0):
        return np.array([[x[0, -1, 0]]])


def test_realtime_scores_after_observation_arrives():
    scaler = MinMaxScaler().fit(np.array([[0.], [100.]]))
    stream = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=1, freq="min"), "cpu_usage": [80.]})
    detector = SimulatedRealtimeDetector(DummyModel(), scaler, scaler.transform([[10.], [20.], [30.]]), stream, 3, threshold=10)
    record = detector.step()
    assert record["predicted"] == 30.0 and record["detected_anomaly"] == 1

