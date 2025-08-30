# Data Factory Module

This directory provides utilities for loading, preprocessing, partitioning, and formatting Human Activity Recognition (HAR) datasets for federated learning experiments. It supports multiple datasets([HARSense/Dataset 1](https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset), [UCI HAR/Dataset 2](https://paperswithcode.com/dataset/har), [Pamap2/Dataset 3](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring), [MHealth/Dataset 4](https://archive.ics.uci.edu/dataset/319/mhealth+dataset), [Acc Using Smartphones/Dataset 5](https://physionet.org/content/accelerometry-walk-climb-drive/1.0.0/), [MotionSense/Dataset 6](https://paperswithcode.com/dataset/motionsense)) by collecting all sensor features into a single .csv file and adding a dedicated "activity" column for labels. It handles dataset unification, train-test splitting, client partitioning, and class imbalance simulation, ensuring reproducibility with random seeds. The Data Factory is a key component of the thesis infrastructure (Section 4.4: Dataset Processing and Partitioning Infrastructure), responsible for preparing data in a format compatible with the server.

## Key Features
* Dataset Loading: Converts raw datasets (e.g., .txt files per subject/activity) to a standardized .csv format.
* Train-Test Splitting: Splits datasets (e.g., 70/30 ratio) with optional scaling (StandardScaler, MinMaxScaler) and label mapping.
* Client Partitioning: Uses `to_client()` to divide data among N clients, with options for shared or individual test sets and uneven data fractions.
* Class Imbalance Simulation: Normalizes datasets via up/downsampling, then applies user-defined ratios (e.g., reducing one class to 20% per client in a round-robin fashion) using `class_imbalance()`.
* Data Downsampling: Functions like `downsample_train_test_split()` and `take_first_n_fraction()` to reduce dataset size for experiments.
* Saving Client Data: Stores partitioned data in a server-expected directory structure via `save_client_data()`.
* Visualization and Analysis: Includes plot_features(), `correlation_matrix()`, and `print_balance()` for data inspection.
* Notebooks: Exploratory Jupyter notebooks for dataset statistics, real data evaluation, and experiment preparation

## Key Files
* `dataloading.py`: Core module with loading and preprocessing functions.
* `dataloaders_for_every_har_dataset.ipynb`: Creates dataloaders and partitions for all HAR datasets.
* `create_har_dataset_for_performance_experiments.ipynb`: Prepares datasets for performance tests under variations like IID/non-IID.
* `preliminary_experiments_results.ipynb`: Analyzes preliminary experiment results with class imbalance.
* `test_real_data_on_dataset_1.ipynb`: Tests real Android sensor data against Dataset 1.
* `evaluate_real_sensor_data_on_dataset1_model_6.ipynb`: Evaluates real data against a model trained on Dataset 6.
* `real_test_data_1.ipynb`: Compares real data with Dataset 1 samples visually.
* `Other notebooks`: For data statistics (`data_statistics.ipynb`), personal dataset creation (`create_you_own_dataset.ipynb`), etc.

## Usage

### Loading a Dataset
```{python}
from dataloading import load_data6  # For MotionSense (Dataset 6)
df = load_data6()  # Returns a unified Pandas DataFrame
```
### Train-Test Splitting with Scaling
```{python}
X_train, y_train, X_test, y_test, labels = train_test_split(
    df, test_size=0.3, scaler_type="standard", should_map_labels=True, random_seed=42
)
```

### Partitioning to Clients
```{python}
client_data = to_client(
    data=(X_train, y_train, X_test, y_test), max_clients=5
)
```

### Simulating Class Imbalance
Define imbalance ratios (e.g., reduce one class to 20% per client, in round-robin format):
```
class_ratio_list = [
    [0.2, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.2, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.2, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.2, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.2, 1.0]
]

client_data_im = []
for idx, (X_train, y_train, X_test, y_test) in enumerate(client_data):
    class_ratio = class_ratio_list[idx]
    X_train_im, y_train_im, X_test_im, y_test_im = class_imbalance(
        (X_train, y_train, X_test, y_test), class_ratio, balance="downsampling"
    )
    client_data_im.append((X_train_im, y_train_im, X_test_im, y_test_im))
```

### Saving Client Data
```{python}
save_client_data(client_data_im, "test", labels)
```

### Example: Full Pipeline for MotionSense (as in Thesis)
```
# Load MotionSense dataset
df6 = load_data6()

# Apply train-test split
X_train, y_train, X_test, y_test, labels = train_test_split(
    df6, test_size=0.3, scaler_type="standard", should_map_labels=True, random_seed=42
)

# Partition data to 5 clients
client_data = to_client(data=(X_train, y_train, X_test, y_test), max_clients=5)

# Define class ratio list
class_ratio_list = [
    [0.2, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 0.2, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.2, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.2, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 0.2, 1.0]
]

# Apply the class ratio list
client_data_im = []
for idx, (X_train, y_train, X_test, y_test) in enumerate(client_data):
    class_ratio = class_ratio_list[idx]
    X_train_im, y_train_im, X_test_im, y_test_im = class_imbalance(
        (X_train, y_train, X_test, y_test), class_ratio, balance="downsampling"
    )
    client_data_im.append((X_train_im, y_train_im, X_test_im, y_test_im))

# Save client data
save_client_data(client_data_im, "test", labels)
```
