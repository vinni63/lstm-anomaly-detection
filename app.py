"""Streamlit dashboard for inspecting detection and demonstrating streaming inference."""
from __future__ import annotations
import json
import time
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import METADATA_PATH, MODEL_PATH, RAW_DATA_PATH, SCALER_PATH
from src.data_generator import save_synthetic_data
from src.data_loader import load_cpu_data
from src.detector import calculate_threshold, detect_frame
from src.model import load_model
from src.preprocessing import chronological_split, load_scaler, transform_values
from src.realtime import SimulatedRealtimeDetector

st.set_page_config(page_title="LSTM Real-Time Anomaly Detection", layout="wide")

@st.cache_resource
def load_artifacts():
    return load_model(MODEL_PATH), load_scaler(SCALER_PATH), json.loads(METADATA_PATH.read_text())

def chart(results, title, columns):
    fig = go.Figure()
    for column in columns:
        fig.add_trace(go.Scatter(x=results.timestamp, y=results[column], name=column.replace("_", " ")))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=45, b=10))
    return fig

def main():
    st.title("LSTM Real-Time Anomaly Detection")
    st.caption("The model predicts from prior CPU readings, then scores the actual reading when it arrives.")
    with st.sidebar:
        st.header("Controls")
        source = st.radio("Dataset", ["Synthetic demo", "Local CSV"], horizontal=True)
        upload = st.file_uploader("Local CSV (timestamp,cpu_usage,is_anomaly)", type="csv") if source == "Local CSV" else None
        delay = st.slider("Simulation delay (seconds)", 0.0, 1.0, 0.2, 0.05)
        percentile = st.slider("Threshold percentile", 90.0, 100.0, 99.0, 0.5)
        start = st.button("Start / continue simulation", type="primary")
        reset = st.button("Reset simulation")
    if not RAW_DATA_PATH.exists():
        save_synthetic_data()

    try:
        if upload is not None:
            # Read uploaded Streamlit file directly
            frame = pd.read_csv(upload)

            frame["timestamp"] = pd.to_datetime(
                frame["timestamp"]
            )

            frame["cpu_usage"] = pd.to_numeric(
                frame["cpu_usage"]
            )

            required_columns = [
                "timestamp",
                "cpu_usage"
            ]

            if "is_anomaly" in frame.columns:
                required_columns.append(
                    "is_anomaly"
                )

            frame = frame[required_columns]

        else:
            # Normal local file
            frame = load_cpu_data(
                RAW_DATA_PATH
            )

        model, scaler, metadata = load_artifacts()

    except Exception as exc:
        st.error(
            f"Artifacts/data unavailable: {exc}. "
            "Generate data and train first."
        )

        st.code(
            "python -m src.data_generator\n"
            "python -m src.train"
        )

        return

    window_size = metadata["window_size"]
    # Input shape is fixed when the saved LSTM is trained. Retrain with --window to change it.
    st.sidebar.number_input("Sequence length (saved model)", min_value=1, value=int(window_size), disabled=True)
    _, validation_frame, test_frame = chronological_split(frame)
    validation = detect_frame(model, transform_values(validation_frame, scaler), validation_frame, scaler, window_size, 0)
    threshold = calculate_threshold(validation.absolute_error, percentile)
    if reset or "simulator" not in st.session_state:
        preceding = frame.iloc[:len(frame) - len(test_frame)]
        st.session_state.simulator = SimulatedRealtimeDetector(model, scaler, transform_values(preceding, scaler), test_frame, window_size, threshold)
    simulator = st.session_state.simulator
    simulator.threshold = threshold
    if start and not simulator.finished:
        simulator.step(); time.sleep(delay); st.rerun()
    results = pd.DataFrame(simulator.history)
    if results.empty:
        st.info("Click **Start / continue simulation**. Each refresh processes exactly one observation.")
        return
    current = results.iloc[-1]
    labels = ["Current Value", "Predicted Value", "Anomaly Score", "Threshold", "Total Anomalies", "Current Status"]
    values = [f"{current.actual:.1f}%", f"{current.predicted:.1f}%", f"{current.absolute_error:.2f}", f"{threshold:.2f}", int(results.detected_anomaly.sum()), "ANOMALY" if current.detected_anomaly else "NORMAL"]
    for column, label, value in zip(st.columns(6), labels, values): column.metric(label, value)
    if current.detected_anomaly: st.error("⚠️ Anomaly detected: error exceeds normal-validation threshold.")
    st.plotly_chart(chart(results, "Actual vs predicted CPU", ["actual", "predicted"]), use_container_width=True)
    st.plotly_chart(chart(results, "Prediction error and threshold", ["absolute_error", "threshold"]), use_container_width=True)
    anomalies = results[results.detected_anomaly == 1]
    if not anomalies.empty:
        fig = go.Figure(go.Scatter(x=results.timestamp, y=results.actual, name="CPU usage"))
        fig.add_trace(go.Scatter(x=anomalies.timestamp, y=anomalies.actual, mode="markers", marker=dict(color="red", size=10), name="Detected anomaly"))
        fig.update_layout(title="CPU usage with detected anomalies", height=320)
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Recent observations")
    st.dataframe(results.tail(15).sort_values("timestamp", ascending=False), use_container_width=True)
    if "true_anomaly" in results:
        # Synthetic labels are shown only as an evaluation aid, never used by detection.
        tn = int(((results.true_anomaly == 0) & (results.detected_anomaly == 0)).sum())
        fp = int(((results.true_anomaly == 0) & (results.detected_anomaly == 1)).sum())
        fn = int(((results.true_anomaly == 1) & (results.detected_anomaly == 0)).sum())
        tp = int(((results.true_anomaly == 1) & (results.detected_anomaly == 1)).sum())
        matrix = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Predicted normal", "Predicted anomaly"],
                                      y=["Actual normal", "Actual anomaly"], text=[[tn, fp], [fn, tp]], texttemplate="%{text}"))
        matrix.update_layout(title="Running confusion matrix (synthetic labels only)", height=280)
        st.plotly_chart(matrix, use_container_width=True)
    st.caption(f"Window: {window_size}. Trained on {metadata['train_rows']} normal chronological readings. Threshold: {percentile}th percentile of normal validation error.")

if __name__ == "__main__":
    main()
