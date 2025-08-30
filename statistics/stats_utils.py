import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import ast

### Usefull functions to log and visualize the statistics of the experiments ###
def parse_experiments_statistics_to_df(path_to_exp_statistics, exp_name, csv_filename="logs.csv"):
    path_to_csv_file = os.path.join(path_to_exp_statistics, exp_name, csv_filename)
    return pd.read_csv(path_to_csv_file)

def prepare_experiment_directory(base_path, experiment_name, files_to_remove):
    """
    Prepares the experiment directory by creating it if necessary and 
    deleting old log files if they exist.
    
    Args:
        base_path (str): The path where the experiment directory should be created.
        experiment_name (str): The name of the experiment.
    
    Returns:
        str: The full path to the experiment directory.
    """
    # Create the full experiment directory path
    experiment_dir = os.path.join(base_path, experiment_name)
    
    # Create the directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)

    # Define the list of files to check and remove if they exist
    #files_to_remove = ["performance_scores.png", "training_time_plot.png", "general_stats.txt"]

    for filename in files_to_remove:
        file_path = os.path.join(experiment_dir, filename)
        if os.path.exists(file_path):
            os.remove(file_path)  # Delete the file

    print(f"Experiment directory prepared: {experiment_dir}")
    return experiment_dir


def generate_performance_scores_plot(y_true, y_pred, label_names, path_to_exp_logs, filename="performance_scores.png"):
    # Compute confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    class_report = classification_report(y_true, y_pred, output_dict=True)

    # Convert classification report to DataFrame and round values
    metrics_df = pd.DataFrame(class_report).T.round(2)

    # Extract global metrics
    accuracy = class_report["accuracy"]
    macro_avg = metrics_df.loc["macro avg"]
    weighted_avg = metrics_df.loc["weighted avg"]

    # Create a combined figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    ### Left: Confusion Matrix with Label Names ###
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax[0], xticklabels=label_names, yticklabels=label_names)
    ax[0].set_title("Confusion Matrix")
    ax[0].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")

    ### Right: Classification Report Text ###
    report_text = ""
    for label in metrics_df.index[:-3]:  # Exclude last 3 rows (macro avg, weighted avg, accuracy)
        try:
            class_name = label_names[int(float(label))]  # Convert to float first, then int
        except (ValueError, IndexError):
            class_name = label  # Use default label if conversion fails

        report_text += (
            f"{class_name} → "
            f"Precision: {metrics_df.loc[label, 'precision']:.2f} | "
            f"Recall: {metrics_df.loc[label, 'recall']:.2f} | "
            f"F1-score: {metrics_df.loc[label, 'f1-score']:.2f} | "
            f"Support: {int(metrics_df.loc[label, 'support'])}\n"
        )

    report_text += f"\nOverall Accuracy: {accuracy:.2%}"
    report_text += f"\nMacro Avg    → Precision: {macro_avg['precision']:.2f} | Recall: {macro_avg['recall']:.2f} | F1-score: {macro_avg['f1-score']:.2f} | Support: {int(macro_avg['support'])}\n"
    report_text += f"Weighted Avg → Precision: {weighted_avg['precision']:.2f} | Recall: {weighted_avg['recall']:.2f} | F1-score: {weighted_avg['f1-score']:.2f} | Support: {int(weighted_avg['support'])}\n"

    # Display classification report text
    ax[1].text(0, 0.5, report_text, fontsize=12, va="center", ha="left", family="monospace", wrap=True)
    ax[1].set_axis_off()

    # Save the combined image
    path_to_file = os.path.join(path_to_exp_logs, filename)
    plt.tight_layout()
    plt.savefig(path_to_file, dpi=300)
    plt.close()

    print(f"Performance scores plot saved at: {path_to_file}")



def generate_general_stats(df, path_to_exp_logs, filename="general_stats.txt"):
    df_copy = df.copy()
    total_global_rounds = len(df_copy)

    # Convert string representation of lists into actual lists
    def clean_devices_names(devices):
        try:
            # If devices is a string that looks like a list, convert it
            if isinstance(devices, str):
                parsed_devices = ast.literal_eval(devices)
                if isinstance(parsed_devices, list):
                    return parsed_devices
            return devices  # Return as-is if already a list
        except (ValueError, SyntaxError):
            return []  # Return empty list if parsing fails

    df_copy['devices_names'] = df_copy['devices_names'].apply(clean_devices_names)

    total_clients = len(set(df_copy['devices_names'].explode()))
    total_test_samples = df_copy['n_samples'].sum()

    # Calculate the total training time (with and without compilation)
    total_training_time = df_copy['training_time'].sum()
    total_training_time_without_compiling = df_copy['training_without_compile_times'].sum()

    # Convert to minutes
    total_training_time_minutes = total_training_time / 60
    total_training_time_without_compiling_minutes = total_training_time_without_compiling / 60

    # Format clients per round in a clear way
    clients_per_round = "\n".join([
        f"  - Round {row['n_round']}: {', '.join(map(str, row['devices_names']))}"
        for _, row in df_copy.iterrows()
    ])

    # Prepare content for the stats text
    stats_text = (
        f"Federated Learning Experiment Summary\n"
        f"======================================\n"
        f"Total Global Rounds: {total_global_rounds}\n"
        f"Total Clients Involved: {total_clients}\n"
        f"Total Test Samples Processed: {total_test_samples}\n\n"
        f"Total Training Time (including compilation): {total_training_time:.2f} seconds ({total_training_time_minutes:.2f} minutes)\n"
        f"Total Training Time (excluding compilation): {total_training_time_without_compiling:.2f} seconds ({total_training_time_without_compiling_minutes:.2f} minutes)\n\n"
        f"Clients per Global Round:\n"
        f"{clients_per_round}"
    )

    # Save as .txt file
    path_to_file = os.path.join(path_to_exp_logs, filename)
    with open(path_to_file, "w", encoding="utf-8") as f:
        f.write(stats_text + "\n")

    # log network related stuff
    network_df = get_network_stats_df(df_copy)
    with open(path_to_file, "a", encoding="utf-8") as f:
        for _, row in network_df.iterrows():
            log_line = (
                f"Client: {row['Client']} | "
                f"Avg RSSI: {row['Avg RSSI']:.2f} dBm | "
                f"Avg Latency: {row['Avg Latency (ms)']:.2f} ms | "
                f"Avg Download: {row['Avg Download (Mbps)']:.2f} Mbps | "
                f"Avg Upload: {row['Avg Upload (Mbps)']:.2f} Mbps | "
                f"Total RX: {row['Total Received (MB)']:.2f} MB | "
                f"Total TX: {row['Total Sent (MB)']:.2f} MB\n"
            )
            f.write(log_line)

    print(f"General statistics saved at: {path_to_file}")

def generate_training_time_plot(df, path_to_exp_logs,filename="training_time_plot.png"):
    """
    Generates a line plot showing training time per global round.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df['n_round'], df['training_time'], label="Training Time", marker="o", linestyle="-")
    plt.plot(df['n_round'], df['training_without_compile_times'], label="Training (No Compilation)", marker="x", linestyle="--")

    total_time = df['training_time'].sum()
    total_time_no_compile = df['training_without_compile_times'].sum()

    plt.xlabel("Global Round")
    plt.ylabel("Time (Seconds)")
    plt.title(f"Training Time per Global Round\n(Total: {total_time:.2f} sec | No Compilation: {total_time_no_compile:.2f} sec)")
    plt.legend()
    plt.grid(True)

    path_to_file = os.path.join(path_to_exp_logs, filename)
    plt.savefig(path_to_file, dpi=300)
    plt.close()
    print(f"Training time plot saved at: {path_to_file}")

def generate_accuracy_loss_plot(df, client_name, path_to_exp_logs,filename="accuracy_loss_plot.png"):
    rounds = df["n_round"].unique()
    
    accuracies = []
    losses = []
    
    for rnd in rounds:
        row = df[df["n_round"] == rnd].iloc[0]

        try:
            y_true_dict = json.loads(row["y_true"])
            y_pred_dict = json.loads(row["y_pred"])
        except json.JSONDecodeError:
            print(f"Skipping round {rnd} due to invalid JSON format")
            continue

        if client_name in y_true_dict and client_name in y_pred_dict:
            y_true = np.array(y_true_dict[client_name])
            y_pred = np.array(y_pred_dict[client_name])

            # Compute Accuracy and Loss
            accuracy = np.mean(y_true == y_pred)  # Example accuracy calculation
            loss = np.mean((y_true - y_pred) ** 2)  # Example loss calculation (MSE)
            #loss = log_loss(y_true, y_pred)

            accuracies.append(accuracy)
            losses.append(loss)
        else:
            print(f"Client {client_name} not found in round {rnd}")
    
    # Plot Accuracy and Loss
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(rounds[:len(accuracies)], accuracies, marker="o", linestyle="-", color="b", label="Accuracy")
    ax[0].set_title("Model Accuracy per Round")
    ax[0].set_xlabel("Round")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].plot(rounds[:len(losses)], losses, marker="s", linestyle="-", color="r", label="Loss")
    ax[1].set_title("Model Loss per Round")
    ax[1].set_xlabel("Round")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    plt.tight_layout()
    #plt.show()
    path_to_file = os.path.join(path_to_exp_logs, filename)
    plt.savefig(path_to_file, dpi=300)
    plt.close()
    print(f"Accuracy Loss plot saved at: {path_to_file}")
    
def get_accuracy_loss_values_for_dfs(df, client_name):    
    rounds = df["n_round"].unique()
    accuracies = []
    losses = []
    
    for rnd in rounds:
        row = df[df["n_round"] == rnd].iloc[0]

        try:
            y_true_dict = json.loads(row["y_true"])
            y_pred_dict = json.loads(row["y_pred"])
        except json.JSONDecodeError:
            print(f"Skipping round {rnd} due to invalid JSON format")
            continue

        if client_name in y_true_dict and client_name in y_pred_dict:
            y_true = np.array(y_true_dict[client_name])
            y_pred = np.array(y_pred_dict[client_name])

            # Compute Accuracy and Loss
            accuracy = np.mean(y_true == y_pred)  # Example accuracy calculation
            loss = np.mean((y_true - y_pred) ** 2)  # Example loss calculation (MSE)
            #loss = log_loss(y_true, y_pred)

            accuracies.append(accuracy)
            losses.append(loss)
        else:
            print(f"Client {client_name} not found in round {rnd}")
    
    return accuracies, losses

# --- energy related statistics -----
def fix_json_string(s):
    """Convert a non-standard JSON string into a proper JSON format."""
    s = s.strip('{} ')
    pairs = s.split(', ')
    json_pairs = []
    for pair in pairs:
        key, value = pair.split(':')
        json_pairs.append(f'"{key.strip()}": "{value.strip()}"')
    return '{' + ', '.join(json_pairs) + '}'

def extract_energy_data(df):
    """Extract all battery attributes before/after for each device."""
    data_list = []

    for _, row in df.iterrows():
        n_round = row['n_round']
        device_names = ast.literal_eval(row['devices_names'])  # Get all devices

        # Parse before and after energy measurements for each device
        before_data_all = json.loads(row['energy_before']) if pd.notna(row['energy_before']) else {}
        after_data_all = json.loads(row['energy_after']) if pd.notna(row['energy_after']) else {}

        for device_name in device_names:
            # Get device-specific data with fallback to empty string
            before_data = before_data_all.get(device_name, '').replace(" = ", ":") if device_name in before_data_all else ''
            after_data = after_data_all.get(device_name, '').replace(" = ", ":") if device_name in after_data_all else ''

            try:
                before = json.loads(fix_json_string(before_data)) if before_data else {}
                after = json.loads(fix_json_string(after_data)) if after_data else {}

                # Helper function to safely extract values
                def safe_extract(data, key, unit, cast_type):
                    value = data.get(key, np.nan)
                    if pd.isna(value) or value == '':
                        return np.nan
                    try:
                        return cast_type(value.replace(unit, ""))
                    except (ValueError, AttributeError):
                        return np.nan

                # Append all attributes with NaN handling
                data_list.append({
                    "n_round": n_round,
                    "device": device_name,
                    "temperature_before": safe_extract(before, 'Temperature', " C", float),
                    "temperature_after": safe_extract(after, 'Temperature', " C", float),
                    "current_now_before": safe_extract(before, 'CurrentNow', " uA", float),
                    "current_now_after": safe_extract(after, 'CurrentNow', " uA", float),
                    "current_avg_before": safe_extract(before, 'CurrentAverage', " uA", float),
                    "current_avg_after": safe_extract(after, 'CurrentAverage', " uA", float),
                    "capacity_before": safe_extract(before, 'Capacity', " %", int),
                    "capacity_after": safe_extract(after, 'Capacity', " %", int),
                    "voltage_before": safe_extract(before, 'Voltage', " mV", float),
                    "voltage_after": safe_extract(after, 'Voltage', " mV", float),
                    "charge_before": safe_extract(before, 'ChargeCounter', " uAH", float),
                    "charge_after": safe_extract(after, 'ChargeCounter', " uAH", float)
                })
            except json.JSONDecodeError:
                # If JSON parsing fails, append with NaN values
                data_list.append({
                    "n_round": n_round,
                    "device": device_name,
                    "temperature_before": np.nan,
                    "temperature_after": np.nan,
                    "current_now_before": np.nan,
                    "current_now_after": np.nan,
                    "current_avg_before": np.nan,
                    "current_avg_after": np.nan,
                    "capacity_before": np.nan,
                    "capacity_after": np.nan,
                    "voltage_before": np.nan,
                    "voltage_after": np.nan,
                    "charge_before": np.nan,
                    "charge_after": np.nan
                })

    return pd.DataFrame(data_list)

def generate_voltage_drop_plot(df, path_to_exp_logs,filename="voltage_drop_plot.png"):
    """Plots the voltage drop for each device without modifying the original DataFrame."""

    # Calculate voltage drop (without modifying df)
    voltage_drops = df.copy()
    voltage_drops["voltage_drop"] = voltage_drops["voltage_before"] - voltage_drops["voltage_after"]

    plt.figure(figsize=(15, 7))

    # Plot each device's voltage drop separately and add cumulative information to the legend
    for device in df["device"].unique():
        subset = voltage_drops[voltage_drops["device"] == device]

        # Calculate cumulative energy consumption (first before and last after voltage)
        first_before = subset["voltage_before"].iloc[0]
        last_after = subset["voltage_after"].iloc[-1]
        total_drop = first_before - last_after

        # Add the information as a comment/legend entry
        label = f"{device} (Total Drop: {total_drop} mV | From: {first_before} mV To: {last_after} mV)"

        # Plot the voltage drop
        plt.plot(subset.index, subset["voltage_drop"], marker='o', linestyle='-', label=label)

    plt.xlabel("Measurement Index (Row Number)")
    plt.ylabel("Voltage Drop (mV)")
    plt.title("Voltage Drop Per Device")
    plt.legend(title="Device")
    path_to_file = os.path.join(path_to_exp_logs, filename)
    plt.tight_layout()
    plt.savefig(path_to_file, dpi=300)
    plt.close()
    #plt.grid()
    #plt.show()

def generate_charge_drop_plot(df, path_to_exp_logs,filename="charge_drop_plot.png"):
    """Plots the charge drop for each device without modifying the original DataFrame."""

    # Calculate charge drop (without modifying df)
    charge_drops = df.copy()
    charge_drops["charge_drop"] = charge_drops["charge_before"] - charge_drops["charge_after"]

    plt.figure(figsize=(10, 5))

    # Plot each device's charge drop separately and add cumulative information to the legend
    for device in df["device"].unique():
        subset = charge_drops[charge_drops["device"] == device]

        # Calculate cumulative charge consumption (first charge_before and last charge_after)
        first_charge_before = subset["charge_before"].iloc[0] / 1000
        last_charge_after = subset["charge_after"].iloc[-1] / 1000
        total_charge_drop = first_charge_before - last_charge_after

        # Add the information as a comment/legend entry
        label = (
            f"{device} "
            f"Total Drop: {total_charge_drop:.3f} mAH "
            f"From: {first_charge_before:.3f} mAH "
            f"To: {last_charge_after:.3f} mAH"
        )
        # Plot the charge drop
        plt.plot(subset.index, subset["charge_drop"], marker='o', linestyle='-', label=label)

    plt.xlabel("Measurement Index (Row Number)")
    plt.ylabel("Charge Drop (mAH)")
    plt.title("Charge Drop Per Device")
    plt.legend(title="Device")
    path_to_file = os.path.join(path_to_exp_logs, filename)
    plt.tight_layout()
    plt.savefig(path_to_file, dpi=300)
    plt.close()
    #plt.grid()
    #plt.show()
    
#def generate_charge_drop_plot(df):
#    """Plots the charge drop for each device without modifying the original DataFrame."""
#    
#    # Create a copy and calculate charge drop in mAH
#    charge_drops = df.copy()
#    charge_drops["charge_drop"] = (charge_drops["charge_before"] - charge_drops["charge_after"]) / 1000
#    
#    # Debugging: Print unique devices and their data
#    print("Unique devices found:", charge_drops["device"].unique())
#    print("Number of devices:", len(charge_drops["device"].unique()))
#    for device in charge_drops["device"].unique():
#        print(f"\nData for {device}:")
#        print(charge_drops[charge_drops["device"] == device][["n_round", "charge_before", "charge_after", "charge_drop"]])
#    
#    plt.figure(figsize=(12, 6))
#    
#    # Plot each device's charge drop
#    for device in charge_drops["device"].unique():
#        subset = charge_drops[charge_drops["device"] == device]
#        
#        # Calculate cumulative charge info
#        first_charge_before = subset["charge_before"].iloc[0] / 1000 if pd.notna(subset["charge_before"].iloc[0]) else np.nan
#        last_charge_after = subset["charge_after"].iloc[-1] / 1000 if pd.notna(subset["charge_after"].iloc[-1]) else np.nan
#        total_charge_drop = first_charge_before - last_charge_after if pd.notna(first_charge_before) and pd.notna(last_charge_after) else np.nan
#        
#        # Create label
#        label = f"{device} (Drop: {total_charge_drop if pd.notna(total_charge_drop) else 'NaN'} mAH)"
#        
#        # Use n_round if it varies, otherwise use index
#        x_data = subset["n_round"] if len(subset["n_round"].unique()) > 1 else subset.index
#        y_data = subset["charge_drop"]
#        
#        # Plot only non-NaN values
#        valid_mask = ~y_data.isna()
#        if valid_mask.any():
#            plt.plot(
#                x_data[valid_mask],
#                y_data[valid_mask],
#                marker='o',
#                linestyle='-',
#                label=label
#            )
#        else:
#            # Ensure device appears in legend even with no valid data
#            plt.plot([], [], marker='o', linestyle='-', label=label)
#    
#    # Customize plot
#    plt.xlabel("Round Number" if len(charge_drops["n_round"].unique()) > 1 else "Measurement Index")
#    plt.ylabel("Charge Drop (mAH)")
#    plt.title("Charge Drop Per Device")
#    plt.legend(title="Devices", bbox_to_anchor=(1.05, 1), loc='upper left')
#    plt.grid(True)
#    plt.tight_layout()
#    plt.savefig(path_to_file, dpi=300)
#    plt.close()
    
def get_network_stats_df(df):
    client_stats = {}
    
    # Iterate over rounds
    df_copy = df.copy()
    for idx, row in df_copy.iterrows():
        clients = row['devices_names'] if isinstance(row['devices_names'], list) else ast.literal_eval(row['devices_names'])
        #clients = ast.literal_eval(row['devices_names'])
    
        # Parse other JSON fields
        rssi = json.loads(row['avg_rssi'])
        latency = json.loads(row['latency'])
        download = json.loads(row['download_speed'])
        upload = json.loads(row['upload_speed'])
        rx = json.loads(row['rx_data'])
        tx = json.loads(row['tx_data'])
        
        for i, client in enumerate(clients):
            if client not in client_stats:
                client_stats[client] = {
                    'rssi': [], 'latency': [], 'download': [], 'upload': [],
                    'rx_total': 0, 'tx_total': 0
                }
                
            client_stats[client]['rssi'].append(float(rssi[client]))
            client_stats[client]['latency'].append(float(latency[client]))
            client_stats[client]['download'].append(float(download[client]))
            client_stats[client]['upload'].append(float(upload[client]))        
            client_stats[client]['rx_total'] += int(rx[client])
            client_stats[client]['tx_total'] += int(tx[client])
    
    # Prepare summary table
    summary = {
        'Client': [],
        'Avg RSSI': [],
        'Avg Latency (ms)': [],
        'Avg Download (Mbps)': [],
        'Avg Upload (Mbps)': [],
        'Total Received (MB)': [],
        'Total Sent (MB)': []
    }
    
    for client, stats in client_stats.items():
        summary['Client'].append(client)
        summary['Avg RSSI'].append(sum(stats['rssi']) / len(stats['rssi']))
        summary['Avg Latency (ms)'].append(sum(stats['latency']) / len(stats['latency']))
        summary['Avg Download (Mbps)'].append(sum(stats['download']) / len(stats['download']))
        summary['Avg Upload (Mbps)'].append(sum(stats['upload']) / len(stats['upload']))
        summary['Total Received (MB)'].append(stats['rx_total'] / (1024 ** 2))
        summary['Total Sent (MB)'].append(stats['tx_total'] / (1024 ** 2))
    
    summary_df = pd.DataFrame(summary)
    return summary_df

def save_label_names(label_names, path_to_exp_logs, experiment_name, filename="label_names.txt"):
    # Create the full experiment directory path
    experiment_dir = os.path.join(path_to_exp_logs, experiment_name)
    path_to_file = os.path.join(experiment_dir, filename)
    with open(path_to_file, "w") as f:
        f.write(",".join(label_names))
        
def load_label_names(path_to_exp_logs, experiment_name, filename="label_names.txt"):
    # Create the full experiment directory path
    experiment_dir = os.path.join(path_to_exp_logs, experiment_name)
    path_to_file = os.path.join(experiment_dir, filename)
    with open(path_to_file, "r") as f:
        return [label.strip() for label in f.read().split(",")]        
        