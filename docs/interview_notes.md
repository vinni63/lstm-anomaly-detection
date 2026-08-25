# Interview notes

## Two-minute explanation

I built a prediction-based anomaly detector for server CPU utilization. Rather than learning labels, it learns normal temporal behavior. I clean and sort timestamped CPU readings, split them chronologically, fit a MinMax scaler only on the normal training section, and make sliding windows of 30 readings. Each window is an LSTM input shaped `(samples, 30, 1)` and its target is the next CPU reading. The LSTM predicts that next reading, then I inverse-transform it and compare it to the actual reading when it arrives. Absolute prediction error is the anomaly score. I calculate the threshold from the 99th percentile of errors on normal validation data, so it is based on the model’s expected normal error rather than a guessed value. In the Streamlit demo, one test observation arrives at a time; it is predicted from prior observations, scored, displayed, and then added to the next window. Synthetic labels are only for precision, recall, F1, and confusion-matrix evaluation. This design could later receive data from an API, IoT device, Kafka, or MQTT source.

## 15 likely questions and answers

1. **What is anomaly detection?** Finding observations that deviate meaningfully from expected behavior.
2. **What is time-series data?** Ordered measurements where time and recent history matter.
3. **What is an RNN?** A neural network that carries information through sequence steps.
4. **What is an LSTM?** A gated RNN that keeps a cell state for longer useful context.
5. **Why LSTM over basic RNN?** Its forget, input, and output gates mitigate vanishing-gradient memory problems.
6. **What is a window?** The 30 prior CPU values used to predict the next one.
7. **Why normalize?** CPU scale is consistent and training gradients become more stable; I save the exact scaler for inference.
8. **Why sliding windows?** They turn one ordered series into many supervised next-step examples without losing temporal order.
9. **What is the input shape?** `(number_of_samples, 30_timesteps, 1_feature)`.
10. **What is prediction error?** `abs(actual CPU - predicted CPU)`, calculated in original CPU units.
11. **Why a percentile threshold?** It calibrates alerts to the upper tail of normal validation errors; it is explainable and tunable.
12. **Why train on normal data?** The model should learn the baseline, not reproduce anomalous behavior as normal.
13. **What happens in real-time?** Predict from prior window, wait for actual, score/classify, then append actual to the window.
14. **Why MSE and what metrics matter?** MSE emphasizes larger prediction misses; precision, recall, and F1 are more informative than accuracy on rare anomalies.
15. **Limitations and production plan?** Drift and contaminated training can hurt results. Monitor error distributions, retrain using reviewed-clean data, add multivariate features, per-service thresholds, alert controls, and an authenticated source adapter.

## Extra concise answers

**Gates:** Forget decides what old context to discard; input decides what to write; the cell state is long-term memory; output exposes relevant state. **If anomalies appear during training:** exclude or label-review them because they can widen normal behavior. **Data drift:** watch validation/error distributions and retrain only after review. **Sensor/API connection:** replace the simulator’s next incoming row with an event from a REST poll, MQTT callback, or Kafka consumer and keep the same window/predict/compare sequence.
