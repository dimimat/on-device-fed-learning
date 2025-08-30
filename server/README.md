# Federated Learning Server  

This directory contains the **central server** for the Federated Learning (FL) system.  
It is implemented in **Python** using the [Flower](https://flower.ai/) framework and coordinates the training of distributed **Android clients** performing **Human Activity Recognition (HAR)**.  

## Overview  

The server orchestrates the FL process:  
1. Send the partitioned datasets to the clients.  
2. Sends the initial global model to all connected Android clients.  
3. Collects local updates from clients after training.  
4. Aggregates weights and recompiles a new global model.  
5. Distributes the updated model for the next training round.  

This enables collaborative training **without sharing raw data**.  

## Key Design Goals  
- **Scalability** → Handles heterogeneous devices (smartphones, Raspberry Pi, etc.) thanks Flower’s modular design.  
- **Rapid experimentation** → Easy to modify datasets, hyperparameters, and model architectures.  
- **Extensibility** → Supports TensorFlow, PyTorch, and custom ML backends (experiments focused on TFLite).  

## Training Parameters Configuration 
The server defines and controls both ML and system-level parameters:  
- **Number of Local Epochs** → training epochs per client before sending updates.  
- **Number of Rounds** → total federated training rounds.  
- **Min Clients per Round** → minimum clients required to start a round.  
- **Target Clients per Round** → preferred number of clients per round.  

Example: with min=3 and target=5, training starts once 5 clients connect but can proceed if only 3 remain active.  

## Notes
- Current logging mechanism stores results as CSV files (future work: PostgreSQL).
