import tensorflow as tf
from src.model import build_lstm_model, load_model


def test_model_can_train_save_and_load(tmp_path):
    model = build_lstm_model(3)
    x = tf.random.uniform((8, 3, 1), seed=1)
    y = tf.random.uniform((8, 1), seed=2)
    model.fit(x, y, epochs=1, verbose=0)
    path = tmp_path / "smoke.keras"
    model.save(path)
    assert load_model(path).input_shape == (None, 3, 1)

