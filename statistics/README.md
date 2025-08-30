# Statistics Module

This directory contains utilities and notebooks for collecting, logging, analyzing, and visualizing metrics from federated learning (FL) experiments in Human Activity Recognition (HAR). The Statistics Collector module as named on the thesis, is a key server-side component (as described in the thesis) that automates two essential tasks after FL completion: (1) recording metrics (e.g., model performance, training time, energy consumption, network quality) into a CSV file, and (2) generating visualizations and summary statistics for quick evaluation without manual analysis. This automation is crucial for handling the large number of hyperparameter combinations in experiments.

Metrics focus on model performance (confusion matrix, accuracy, precision, recall, F1-score), training time (per-round and total, with/without compilation overhead), energy consumption (using BatteryStats API for charge/voltage drops), and network quality (RSSI, latency, download/upload speeds, RX/TX data). On the experimentation controls like disabling background apps, enabling Airplane mode (with Wi-Fi on), and minimizing screen brightness are applied for accurate measurements.

## Key Features

- **Model Performance Metrics**: Computes confusion matrix, accuracy (proportion of correct predictions), precision (correct positives out of predicted positives), recall (correct positives out of actual positives), and F1-score (harmonic mean of precision and recall). Useful for identifying class confusion in HAR datasets.
- **Training Time Measurement**: Tracks per-round and cumulative training time, distinguishing between actual training and model compilation overhead in TFLite.
- **Energy Consumption Estimation**: Uses BatteryStats API to calculate energy per round/device via charge drop × average voltage (in mWh), summing to total energy. Supports metrics like battery voltage, charge (mAh), current (mA), and temperature.
- **Network Quality Monitoring**: Captures RSSI (Wi-Fi signal strength in dBm), latency (via ping), download/upload speeds (via auxiliary Flask server with 10MB dummy file), and RX/TX data volumes.
- **Visualization and Logging**: Generates plots (e.g., accuracy/loss, performance scores, training time, charge/voltage drops) and summary files (e.g., `useful_stats.txt` with rounds, clients, test samples, participation).

## Key Files

- **stats_utils.py**: Core module with functions for logging and visualizing metrics (e.g., `parse_experiments_statistics_to_df()`, `prepare_experiment_directory()`, `generate_performance_scores_plot()`, `get_network_stats_df()`).
- **measure_personal_laptop_wifi_signal.py**: Script for measuring Wi-Fi signal strength (RSSI) on a personal laptop, useful for network quality baselines.
- **read_statistics/**: Directory with notebooks for building final visualizations:
  - `1_performance_under_dataset_variation.ipynb`: Visualizes performance across dataset variations.
  - `2_performance_under_label_skew.ipynb`: Examines label imbalance effects.
  - `3_performance_under_quantity_skew.ipynb`: Studies quantity skew impacts.
  - `4_performance_under_mix_skew.ipynb`: Analyzes mixed skew scenarios.
  - `5_performance_scaling.ipynb`: Evaluates scaling performance.
  - `6_local_epoch_rounds_tradeoff.ipynb`: Explores trade-offs between local epochs and rounds.
  - `7_model_size_energy_trade_off.ipynb`: Investigates model size vs. energy consumption.
  - `8_mid_training_drop_out_test.ipynb`: Tests mid-training client dropouts.
  - `9_client_availability_test.ipynb`: Assesses client availability impacts.
  - `10_signal_quality_test.ipynb`: Analyzes signal quality metrics.
  - `manual_data_extraction.ipynb`: For manual extraction of data.
  - `read_metrics_from_csv.ipynb`: Reads and processes metrics from CSV logs.

## Notes

* The module automates metric collection (e.g., RSSI, latency, speeds, RX/TX) and generates logs/plots for quick insights, reducing manual analysis for numerous experiments.
* An example of the automatically generated plots and statistics are stored in the `experiment_outputs` directory.
* `BatteryStats API` is practical but less precise than hardware monitors. But we are good for now since we are focusing on repeatability over absolute accuracy.

