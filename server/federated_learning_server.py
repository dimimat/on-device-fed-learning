#!/usr/bin/env python3 
from typing import Any, Callable, Dict, List, Optional, Tuple
import flwr as fl
from custom_flower_server_fn import CustomServer
from flwr.server import SimpleClientManager
import sys
from datetime import datetime
import os
import pandas as pd
import json
import ast
#from flask import Flask, send_file, request

statistics_path = os.path.abspath("../statistics")
sys.path.append(statistics_path)
import stats_utils
#from stats_utils import generate_general_stats, generate_performance_scores_plot, parse_experiments_statistics_to_df, prepare_experiment_directory

def start_server(n_clients, client_shift, n_rounds, model_config, data_filename, path_to_exp_directory, csv_filename):
    # Create strategy
    strategy = fl.server.strategy.FedAvgAndroid(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        #min_fit_clients=1,
        #min_evaluate_clients=1,
        #min_available_clients=1,
        evaluate_fn=None,
        on_fit_config_fn=fit_config,
        initial_parameters=None,
    )

    # init custom server
    client_manager = SimpleClientManager()
    custom_server = CustomServer(client_manager=client_manager, client_shift=client_shift, strategy=strategy, model_config=model_config, filename=data_filename, path_to_exp_directory=path_to_exp_directory,csv_filename=csv_filename)

    # Start Flower server for X rounds of federated learning
    _ = fl.server.start_server(
        server=custom_server,
        server_address="192.168.1.1:8080",
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
    )

def fit_config(server_round: int):
    #config = {"n_epochs": 500.0,}
    config = {"n_epochs": 3.0}
    return config

def generate_statistics(exp_name, path_to_exp_statistics,data_filename):
    """
    Generates various experiment statistics and saves them to the experiment directory.

    Args:
        exp_name (str): The name of the experiment.
        path_to_exp_statistics (str): The base directory where statistics will be saved.
    """
    #print(f"Generating statistics for experiment: {exp_name}")
    files_to_remove = ["performance_scores.png", "training_time_plot.png", \
                    "general_stats.txt", "accuracy_loss_plot.png", "voltage_drop_plot.png", \
                    "charge_drop_plot.png", "label_names.txt"]

    # Prepare the directory for storing statistics
    experiment_dir = stats_utils.prepare_experiment_directory(path_to_exp_statistics, exp_name, files_to_remove)

    # Parse experiment logs into a DataFrame
    df = stats_utils.parse_experiments_statistics_to_df(path_to_exp_statistics, exp_name, csv_filename="logs.csv")

    # Generate training time plot
    stats_utils.generate_training_time_plot(df, experiment_dir, filename="training_time_plot.png")
    #print("Training time plot saved.")

    # Generate general statistics
    stats_utils.generate_general_stats(df, experiment_dir, filename="general_stats.txt")
    #print("General statistics saved.")

    # Generate final model performance statistics
    first_client_name = ast.literal_eval(df['devices_names'][0])[0]
    y_true = json.loads(df['y_true'].iloc[-1])[first_client_name]
    y_pred = json.loads(df['y_pred'].iloc[-1])[first_client_name]
    label_names = load_labels(data_filename)

    stats_utils.generate_performance_scores_plot(y_true, y_pred, label_names, os.path.join(path_to_exp_statistics, exp_name), "performance_scores.png")
    print("Performance scores plot saved.")
   
    # save label names
    stats_utils.save_label_names(label_names, path_to_exp_statistics, exp_name, filename="label_names.txt")
    print("Label names saved.")

    # generate accuracy loss plots
    stats_utils.generate_accuracy_loss_plot(df, first_client_name, os.path.join(path_to_exp_statistics, exp_name), filename="accuracy_loss_plot.png")
    print("Accuracy-Loss plot saved.")

    # generate energy related statistics
    energy_df = stats_utils.extract_energy_data(df)
    stats_utils.generate_voltage_drop_plot(energy_df, os.path.join(path_to_exp_statistics, exp_name), filename="voltage_drop_plot.png")
    stats_utils.generate_charge_drop_plot(energy_df, os.path.join(path_to_exp_statistics, exp_name), filename="charge_drop_plot.png")

    print(f"All statistics saved in: {experiment_dir}")

def load_labels(filename):
    """
    Loads class labels from the 'labels.txt' file.

    Args:
        filename (str): The name of the directory containing 'labels.txt'.

    Returns:
        list: A list of class labels, or an empty list if the file does not exist.
    """

    # Get path to working data from config file
    #path_to_config = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "config.txt")
    path_to_config = os.path.join(os.path.dirname(os.getcwd()), "config.txt")
    with open(path_to_config, 'r') as file:
        path_to_working_data = file.readline().split(" ")[0]

    labels_path = os.path.join(path_to_working_data, filename, 'labels.txt')

    # Check if file exists
    if not os.path.exists(labels_path):
        print(f"Warning: Labels file not found at {labels_path}. Returning an empty list.")
        return []

    # Read labels
    with open(labels_path, 'r', encoding='utf-8') as f:
        labels = [line.strip() for line in f.readlines()]

    return labels
if __name__ == "__main__":
    # start flower server
    n_clients = 1
    client_shift = 0
    n_rounds = 1
    fileIdx = None
    print("\nStarting Federated Learning Server...\n")

    try:
        if len(sys.argv) == 5:
            n_clients = int(sys.argv[1])
            client_shift = int(sys.argv[2])
            n_rounds = int(sys.argv[3])
            fileIdx = sys.argv[4]
            print(f"Received input arguments: n_clients={n_clients}, client_shift={client_shift}, n_rounds={n_rounds}, fileIdx={fileIdx}")
        elif len(sys.argv) == 2:
            fileIdx = sys.argv[1]
            print(f"Received file index: {fileIdx}")
        elif len(sys.argv) == 1:
            print(f"Default arguments: n_clients={n_clients}, client_shift={client_shift}, n_rounds={n_rounds}, fileIdx={fileIdx}")
        else:
            print("Invalid input! Please provide arguments in the following format:\n")
            print("python server.py <n_clients> <client_shift> <n_rounds> <fileIdx>")
            print(" or")
            print("python server.py <fileIdx>")
            sys.exit(2)
    except ValueError:
        print("Error: Invalid argument type! Please ensure numerical values for clients, shift, and rounds.")
        sys.exit(2)

    model_config = [12,10,6]
    data_filename = "exp_5_iid_case"

    # get experiment name
    if fileIdx is not None:
        exp_name = fileIdx
    else:
        exp_name = data_filename + "_" + datetime.now().strftime("%Y:%m:%d_%H:%M:%S")

    print(f"Experiment Name: {exp_name}")

    path_to_exp_statistics = "/Users/admin/Desktop/thesis/dataset/metrics/"
    csv_filename = "logs.csv" 
    path_to_exp_directory = os.path.join(path_to_exp_statistics, exp_name)
    start_server(n_clients=n_clients, client_shift=client_shift, n_rounds=n_rounds, model_config=model_config, data_filename=data_filename, path_to_exp_directory=path_to_exp_directory, csv_filename=csv_filename)
    
    # generate statistics
    generate_statistics(exp_name, path_to_exp_statistics, data_filename)

