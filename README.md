# Federated Learning for Human Activity Recognition on Android Devices

This repository contains the implementation and experimental framework for the diploma thesis:  **“On-Device Federated Learning for Human Activity Recognition”** (National Technical University of Athens, 2025).

⚠️ **Note:** This is research-driven code. It evolved iteratively during experimentation, so expect suboptimal implementations, quick fixes, and design decisions motivated by research needs rather than good programming practices.

## Thesis Materials
The complete thesis document and defense presentation can be found in the [`thesis/`](./thesis) directory.

## Experimental Results
All experiment plots, and summary statistics generated during the thesis are collected in the [`experiments/`](./experiments) directory.

## Overview

The project investigates Federated Learning (FL) which is a privacy-preserving approach where a shared model is trained across many devices without sending raw data to a server (only model updates are exchanged) on resource-constrained Android devices for Human Activity Recognition (HAR). Unlike most simulation-based work, this system runs on real hardware: a Flower-based server coordinates multiple Android clients, each performing on-device training and evaluation with TensorFlow Lite (TFLite).

The pipeline includes:
- Dataset preparation and partitioning
- Model construction and TFLite export
- Server orchestration of FL rounds
- Android client training/evaluation
- Automated logging and statistical analysis

Research experiments focused on three axes:
- **Data heterogeneity** (label imbalance, quantity skew, non-IID partitions)
- **Energy efficiency** (local epochs vs. communication trade-offs)
- **Network reliability** (dropouts, unstable Wi-Fi)

## Key Components

### Federated Learning Server ([`server/`](./server))
- Coordinates the FL process:
  1. Sends partitioned datasets & initial global model to clients.
  2. Collects local updates after training.
  3. Aggregates weights → recompiles a new global model.
  4. Redistributes the model for the next round.
- **Design goals:** scalability, rapid experimentation, extensibility.  
- **Parameters:** local epochs, number of rounds, min/target clients per round.  
- **Note:** Logging uses CSV (future extension: PostgreSQL).

### Android Client ([`client/`](./client))
- Java-based app using Flower and TFLite.
- Performs local training and evaluation.
- Communicates with server via gRPC.
- Main Components:
  - MainActivity.java: UI & FL connection.
  - DataLoader.java: Batching interface (PyTorch-like).
  - TransferLearningModelWrapper.java: Model training/evaluation wrapper.
  
⚠️ **Limitation:** TFLite requires full model recompilation per round.

### Data Factory ([`data_factory/`](./data_factory))
- Prepares HAR datasets (HARSense, UCI HAR, PAMAP2, MHealth, PhysioNet, MotionSense).  
- Converts raw files → unified `.csv` with sensor features + activity labels.  
- Supports:
  - Train-test splits (e.g., 70/30) with scaling.
  - Client partitioning with reproducibility via random seeds.
  - Class imbalance simulation for non-IID experiments.  

### Model Factory ([`model_factory/`](./model_factory))
- Automates Deep Neural Network (DNN) creation.  
- Supports:
  - Custom hidden layers and sizes.
  - Adam optimizer with configurable learning rate.
  - Random or custom weight initialization (for FL aggregation).
  - Export to TFLite (in order to be used on Android or any other low-resources device).  


### Statistics Module ([`statistics/`](./statistics))
- Automates experiment logging and visualization.
- Collects:
  - Model metrics: accuracy, precision, recall, F1, confusion matrix.
  - Training time: with/without TFLite compilation.
  - Energy: via BatteryStats API (voltage × charge drop).
  - Network: RSSI, latency, upload/download speeds.
  - Includes Jupyter notebooks for analysis (label skew, scaling, dropout, energy trade-offs).
  - Outputs plots (e.g., accuracy_loss.png, performance_scores.png) + summary text files.

**Note:** Each module has its own **README.md** with detailed instructions, usage examples, and design notes. Please refer to them for deeper explanations.

### Citation

If you use this code, please cite:
```
@thesis{matsoukas2025federatedHAR,
  title={On-Device Federated Learning for Human Activity Recognition},
  author={Dimitrios Matsoukas},
  school={National Technical University of Athens},
  year={2025}
}
```
