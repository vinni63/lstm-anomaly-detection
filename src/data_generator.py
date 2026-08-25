"""Generate an interview-friendly CPU time series with labelled test anomalies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from src.config import LABEL_COLUMN, METRIC_COLUMN, RAW_DATA_PATH, RANDOM_SEED, TIMESTAMP_COLUMN
except ModuleNotFoundError:  # Supports: python src/data_generator.py
    from config import LABEL_COLUMN, METRIC_COLUMN, RAW_DATA_PATH, RANDOM_SEED, TIMESTAMP_COLUMN


def generate_synthetic_cpu_data(n_points: int = 1_000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Create gradual seasonal CPU behavior; inject anomalies only after training period.

    Labels are retained exclusively to measure detection quality. The LSTM receives
    only CPU values and learns normal next-value prediction.
    """
    if n_points < 200:
        raise ValueError("n_points must be at least 200.")
    rng = np.random.default_rng(seed)
    index = np.arange(n_points)
    baseline = 45 + 10 * np.sin(2 * np.pi * index / 144) + 4 * np.sin(2 * np.pi * index / 48)
    trend = 0.006 * index
    cpu = baseline + trend + rng.normal(0, 1.3, n_points)
    labels = np.zeros(n_points, dtype=int)

    # Keep the first 80% normal, so train/validation contain clean behavior.
    anomaly_positions = [820, 835, 855, 875, 890, 915, 930, 970]
    for position_index, pos in enumerate(anomaly_positions):
        # Alternating sharp spikes and drops makes both incident styles visible.
        cpu[pos] += 30 if position_index % 2 == 0 else -28
        labels[pos] = 1
    # Sustained overload region: still generated after the normal validation data.
    cpu[945:953] += 22
    labels[945:953] = 1
    cpu = np.clip(cpu, 1, 100)
    timestamps = pd.date_range("2025-01-01", periods=n_points, freq="min")
    return pd.DataFrame({TIMESTAMP_COLUMN: timestamps, METRIC_COLUMN: cpu.round(2), LABEL_COLUMN: labels})


def save_synthetic_data(output_path: Path = RAW_DATA_PATH, n_points: int = 1_000) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = generate_synthetic_cpu_data(n_points=n_points)
    data.to_csv(output_path, index=False)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CPU monitoring data.")
    parser.add_argument("--output", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--points", type=int, default=1_000)
    args = parser.parse_args()
    path = save_synthetic_data(args.output, args.points)
    print(f"Saved synthetic dataset to {path}")
