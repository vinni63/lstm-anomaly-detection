"""Central, easy-to-explain configuration for the demo."""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "synthetic_cpu.csv"
MODEL_DIR = ROOT_DIR / "models"
MODEL_PATH = MODEL_DIR / "lstm_model.keras"
SCALER_PATH = MODEL_DIR / "scaler.pkl"
METADATA_PATH = MODEL_DIR / "metadata.json"

TIMESTAMP_COLUMN = "timestamp"
METRIC_COLUMN = "cpu_usage"
LABEL_COLUMN = "is_anomaly"  # Used only for evaluation; never fed to the model.

WINDOW_SIZE = 30
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20
THRESHOLD_PERCENTILE = 99.0
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DROPOUT_RATE = 0.20
DENSE_UNITS = 16
EPOCHS = 20
BATCH_SIZE = 32
RANDOM_SEED = 42

