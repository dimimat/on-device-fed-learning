import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def read_android_logger_sensor_data(path_to_sensor_data):
    def safe_read(filename):
        file_path = os.path.join(path_to_sensor_data, filename)
        if os.path.exists(file_path):
            return pd.read_csv(file_path)
        else:
            print(f"Warning: {filename} not found.")
            return None

    acc_df = safe_read("Accelerometer.csv")
    ag_df = safe_read("gravity.csv")
    rot_rate_df = safe_read("Gyroscope.csv")
    orientation_df = safe_read("Orientation.csv")

    return acc_df, ag_df, rot_rate_df, orientation_df



#def pretty_plot(df, is_orientation=False):
#    fig = go.Figure()
#    frame = ['x', 'y', 'z'] if not is_orientation else ['roll', 'pitch', 'yaw'] 
#    for axis in frame:
#        fig.add_trace(go.Scatter(x = df.index, y = df[axis], name = axis))
#    fig.show()
    
def pretty_plot(df, is_orientation=False, use_df_index=True):
    fig = go.Figure()
    frame = ['x', 'y', 'z'] if not is_orientation else ['roll', 'pitch', 'yaw'] 
    for axis in frame:
        if (use_df_index):
            fig.add_trace(go.Scatter(x = df.index, y = df[axis], name = axis))
        else:
            fig.add_trace(go.Scatter(x = np.arange(df.shape[0]), y = df[axis], name = axis))
    fig.show()
    
def check_sampling_rate(df):
    consecutive_deltas = df['seconds_elapsed'].diff()
    avg_sampling_rate = 1 / np.mean(consecutive_deltas)
    shortest_gap = np.min(consecutive_deltas)
    maximum_gap = np.max(consecutive_deltas)
    total_num_of_samples = len(df.index)

    print(f"Total Samples: {total_num_of_samples}")
    print(f"Average Sampling Rate: {avg_sampling_rate:.2f} Hz")
    print(f"Shortest Sampling Gap: {shortest_gap:.5f} s")
    print(f"Maximum Sampling Gap: {maximum_gap:.5f} s")

    plt.figure(figsize=(10, 5))
    plt.plot(consecutive_deltas, marker="o", linestyle="-", alpha=0.6)
    plt.xlabel("Sample Index")
    plt.ylabel("Time Between Samples (seconds)")
    plt.title("Variation in Sampling Gaps")
    plt.grid(True)
    plt.show()
    
def sliding_window_moving_average(data, window_size):
    step_size = window_size // 2  # 50% overlap
    num_windows = (len(data) - window_size) // step_size + 1  # Number of valid windows
    
    return np.array([np.mean(data[i : i + window_size]) for i in range(0, num_windows * step_size, step_size)])

def apply_low_pass_filter(df, fs, cutoff=20, order=3):
    sos = signal.butter(order, cutoff, btype='low', output='sos', fs=fs)
    filtered_data = {col: signal.sosfilt(sos, df[col].values) for col in df.columns}
    return pd.DataFrame(filtered_data)

def process_dataframe(df, fs, window_size):
    filtered_df = df
#     filtered_df = apply_low_pass_filter(df, fs)  # Step 1: LPF
    smoothed_df = pd.DataFrame({col: sliding_window_moving_average(filtered_df[col], window_size) for col in df.columns})  # Step 2: Moving Average
    return smoothed_df

# mapping {AG, acc, Gravity, RR, to_euler(RV), cos} => {_, acc_df, ag_df, rot_rate_df, orientation_df, _}
def to_data1_df(acc_df, ag_df, rot_rate_df, orientation_df, fs=None, window_size=None):
    # first try it without rotational vector
    #dfs = [acc_df[['x', 'y', 'z']],ag_df[['x', 'y', 'z']], rot_rate_df[['x', 'y', 'z']]]
    # it seems that ours AG(x,y,z) is dataset's 1 AG(x,z,-y)
    ag_df_dataset_1 = ag_df.copy()
    ag_df_dataset_1['y'] = ag_df['y']
    ag_df_dataset_1['z'] = ag_df['z']
   
    if fs is not None and window_size is not None:
        acc_df_ = process_dataframe(acc_df, fs, window_size) 
        ag_df_ = process_dataframe(ag_df, fs, window_size) 
        rot_rate_df_ = process_dataframe(rot_rate_df, fs, window_size) 
        dfs = [acc_df_[['x', 'y', 'z']],ag_df_[['x', 'y', 'z']], rot_rate_df_[['x', 'y', 'z']]]
    else:
        dfs = [acc_df[['x', 'y', 'z']],ag_df_dataset_1[['x', 'y', 'z']], rot_rate_df[['x', 'y', 'z']]]

    # Find the minimum number of samples
    min_samples = min(len(df) for df in dfs)

    # Trim each DataFrame to the minimum length
    balanced_df_list = [df.iloc[:min_samples] for df in dfs] 
    balanced_df = pd.concat(balanced_df_list, axis = 1)
    
    return balanced_df 

def take_df_subset(df, s, e):
    n = len(df)
    start = int(n * s)
    end = int(n * (1 - e))    
    return df.iloc[start:end]