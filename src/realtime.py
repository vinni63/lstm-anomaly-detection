"""Stateful simulated stream. Swap `step` input with an API/MQTT/Kafka source later."""
from __future__ import annotations
from collections import deque
import numpy as np
from src.preprocessing import inverse_values


class SimulatedRealtimeDetector:
    """Predict before the observation arrives, then score it after it arrives."""
    def __init__(self, model, scaler, seed_values, stream_frame, window_size, threshold):
        if len(seed_values) < window_size:
            raise ValueError("seed_values needs at least window_size observations.")
        self.model, self.scaler = model, scaler
        self.stream_frame = stream_frame.reset_index(drop=True)
        self.window = deque(np.asarray(seed_values)[-window_size:].reshape(-1), maxlen=window_size)
        self.window_size, self.threshold, self.position, self.history = window_size, threshold, 0, []

    @property
    def finished(self):
        return self.position >= len(self.stream_frame)

    def step(self):
        if self.finished:
            return None
        prediction_scaled = float(self.model.predict(np.array(self.window, dtype="float32").reshape(1, self.window_size, 1), verbose=0)[0, 0])
        row = self.stream_frame.iloc[self.position]
        actual = float(row["cpu_usage"])
        predicted = float(inverse_values(np.array([prediction_scaled]), self.scaler)[0])
        error = abs(actual - predicted)
        record = {"timestamp": row["timestamp"], "actual": actual, "predicted": predicted,
                  "absolute_error": error, "threshold": self.threshold, "detected_anomaly": int(error > self.threshold)}
        if "is_anomaly" in row.index:
            record["true_anomaly"] = int(row["is_anomaly"])
        # The actual may join the window only after comparison for the next event.
        self.window.append(float(self.scaler.transform(np.array([[actual]]))[0, 0]))
        self.position += 1
        self.history.append(record)
        return record
