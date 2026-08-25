"""Train on chronological normal data, derive threshold on normal validation data."""
from __future__ import annotations
import argparse
import json
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras

try:
    from src.config import (BATCH_SIZE, EPOCHS, METADATA_PATH, MODEL_PATH, RANDOM_SEED, RAW_DATA_PATH,
                            THRESHOLD_PERCENTILE, WINDOW_SIZE)
    from src.data_loader import load_cpu_data
    from src.detector import calculate_threshold, detect_frame
    from src.model import build_lstm_model
    from src.preprocessing import chronological_split, fit_scaler, save_scaler, transform_values
    from src.sequence_builder import create_sequences
except ModuleNotFoundError:
    from config import (BATCH_SIZE, EPOCHS, METADATA_PATH, MODEL_PATH, RANDOM_SEED, RAW_DATA_PATH,
                        THRESHOLD_PERCENTILE, WINDOW_SIZE)
    from data_loader import load_cpu_data
    from detector import calculate_threshold, detect_frame
    from model import build_lstm_model
    from preprocessing import chronological_split, fit_scaler, save_scaler, transform_values
    from sequence_builder import create_sequences


def train(dataset_path=RAW_DATA_PATH, window_size=WINDOW_SIZE, percentile=THRESHOLD_PERCENTILE, epochs=EPOCHS):
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    frame = load_cpu_data(dataset_path)
    train_frame, validation_frame, test_frame = chronological_split(frame)
    scaler = fit_scaler(train_frame)
    train_scaled, validation_scaled = transform_values(train_frame, scaler), transform_values(validation_frame, scaler)
    x_train, y_train = create_sequences(train_scaled, window_size)
    x_val, y_val = create_sequences(validation_scaled, window_size)
    model = build_lstm_model(window_size)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ]
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                        batch_size=BATCH_SIZE, verbose=1, callbacks=callbacks)
    save_scaler(scaler)
    # Validation is normal in the synthetic data; in real use provide a known-clean period.
    validation_results = detect_frame(model, validation_scaled, validation_frame, scaler, window_size, threshold=0)
    threshold = calculate_threshold(validation_results["absolute_error"], percentile)
    metadata = {"window_size": window_size, "threshold_percentile": percentile, "threshold": threshold,
                "train_rows": len(train_frame), "validation_rows": len(validation_frame), "test_rows": len(test_frame),
                "final_val_loss": float(history.history["val_loss"][-1])}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"Training complete. Model: {MODEL_PATH}")
    print(f"Validation-derived {percentile}th percentile threshold: {threshold:.3f} CPU points")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(RAW_DATA_PATH))
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--percentile", type=float, default=THRESHOLD_PERCENTILE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    train(args.data, args.window, args.percentile, args.epochs)

