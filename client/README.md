# Android Application

This directory contains the **Android application** that acts as a client in the **Federated Learning (FL) system for Human Activity Recognition (HAR)**. The client app is implemented in **Java**, built on top of the [Flower](https://github.com/adap/flower/tree/main/examples/android) framework Android example, and uses **TensorFlow Lite (TFLite)** for on-device training and evaluation.  

## Overview  

The Android client enables a group of smartphones to collaboratively train a shared machine learning model under the coordination of a central FL server.  

### Key Design Goals
- **Scalability** -> Support of heterogeneous clients (different devices, hardware specs).  
- **Rapid experimentation** -> Fast switching of datasets, models, and hyperparameters.  
- **Cross-platform support** -> Although this thesis uses Android/TFLite, the design remains compatible with other ML frameworks (e.g., PyTorch, custom ML frameworks).  


⚠️ **Limitation:** TFLite does not support programmatic weight updates.  
→ The server must recompile and send the **entire `.tflite` model** each round, instead of only updated weights.  

## Usage  

1. Open the app on your Android device.  
2. Enter the **server’s IP address** and **port number**.  
3. Press **Connect** and then **Train**.  
4. Once enough clients are connected, the server initiates FL training.  
5. The client trains locally and reports metrics back to the server.  

## Main Components  

### MainActivity.java 
- Handles the app lifecycle and UI.  
- Connects to the FL server via **gRPC**.  
- Orchestrates training and evaluation on the device.  

### DataLoader.java  
- Loads and manages datasets received from the server.  
- Provides **batched access** to training/test data (PyTorch-like interface).  

Example usage:  
```java
trainLoader = new DataLoader(X_train, y_train, batchSize, numFeatures);
BatchOfData trainBatch = trainLoader.next();
float[][] X_batch = trainBatch.getX();
float[] y_batch = trainBatch.getY();
```
### TransferLearningModelWrapper.java

- Reconstructs models using metadata + weights from the server.

- Compiles models into a TFLite-ready format.

- Provides fit() and evaluate() methods for training and testing.

Example usage:
``` java
TransferLearningModelWrapper model = new TransferLearningModelWrapper(modelFile, input, hidden, output);

// Training
model.fit(trainLoader, n_epochs);

// Evaluation
Metrics metrics = model.evaluate(testLoader);
float acc = metrics.getAccuracy();
```
### Notes

- Designed for research/experimental purposes.

- Requires a running FL server (see [`server/`](../server) directory).

- Tested with TensorFlow Lite and Flower 1.x.
