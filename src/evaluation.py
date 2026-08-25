"""Evaluate predictions; labels are optional and never used for detection."""
from __future__ import annotations
import argparse
import json
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    from src.config import METADATA_PATH, MODEL_PATH, RAW_DATA_PATH, SCALER_PATH
    from src.data_loader import load_cpu_data
    from src.detector import calculate_threshold, detect_frame
    from src.model import load_model
    from src.preprocessing import chronological_split, load_scaler, transform_values
except ModuleNotFoundError:
    from config import METADATA_PATH, MODEL_PATH, RAW_DATA_PATH, SCALER_PATH
    from data_loader import load_cpu_data
    from detector import calculate_threshold, detect_frame
    from model import load_model
    from preprocessing import chronological_split, load_scaler, transform_values


def evaluate(dataset_path=RAW_DATA_PATH, percentile=None):
    metadata = json.loads(METADATA_PATH.read_text())
    window_size = metadata["window_size"]
    percentile = percentile or metadata["threshold_percentile"]
    frame = load_cpu_data(dataset_path)
    _, validation_frame, test_frame = chronological_split(frame)
    model, scaler = load_model(MODEL_PATH), load_scaler(SCALER_PATH)
    val_results = detect_frame(model, transform_values(validation_frame, scaler), validation_frame, scaler, window_size, 0)
    threshold = calculate_threshold(val_results.absolute_error, percentile)
    results = detect_frame(model, transform_values(test_frame, scaler), test_frame, scaler, window_size, threshold)
    summary = {"threshold": threshold, "detected_anomalies": int(results.detected_anomaly.sum()),
               "anomaly_percentage": float(results.detected_anomaly.mean() * 100)}
    if "true_anomaly" in results:
        y_true, y_pred = results.true_anomaly, results.detected_anomaly
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        summary.update({"precision": float(precision), "recall": float(recall), "f1_score": float(f1),
                        "accuracy": float(accuracy_score(y_true, y_pred)), "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()})
    print(json.dumps(summary, indent=2))
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    parser.add_argument("--percentile", type=float, default=None)
    args = parser.parse_args()
    evaluate(args.data, args.percentile)

