"""Small configurable Keras LSTM for next-value prediction."""
from __future__ import annotations
import tensorflow as tf
from tensorflow import keras

try:
    from src.config import DENSE_UNITS, DROPOUT_RATE, LSTM_UNITS_1, LSTM_UNITS_2
except ModuleNotFoundError:
    from config import DENSE_UNITS, DROPOUT_RATE, LSTM_UNITS_1, LSTM_UNITS_2


def build_lstm_model(window_size: int) -> keras.Model:
    model = keras.Sequential([
        keras.layers.Input(shape=(window_size, 1), name="cpu_window"),
        keras.layers.LSTM(LSTM_UNITS_1, return_sequences=True),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.LSTM(LSTM_UNITS_2),
        keras.layers.Dropout(DROPOUT_RATE),
        keras.layers.Dense(DENSE_UNITS, activation="relu"),
        keras.layers.Dense(1, name="next_cpu_prediction"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(), loss="mse", metrics=["mae"])
    return model


def load_model(path):
    return keras.models.load_model(path)

