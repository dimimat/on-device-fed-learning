"""
Helper module for loading and preprocessing Human Activity Recognition (HAR) datasets used in model evaluation
and federated learning experiments. Supports HARSense (Dataset 1), UCI HAR (Dataset 2), and Pamap2 (Dataset 3),
with utilities for train-test splitting, client partitioning, class imbalance handling, and data visualization.
Key functions include:

- load_data1(), load_data2(), load_data3(), load_data4(), load_data5(), load_data6()
      : Load HARSense, UCI HAR, Pamap2, MHealth, HAR using smartphones and MotionSense  datasets.
- train_test_split(): Split data into training and testing sets with optional scaling.
- to_client(): Partition data into client subsets for federated learning.
- class_imbalance(), class_imbalance_2(): Manage class imbalance via over/undersampling.
- downsample_train_test_split(), take_first_n_fraction(): Downsample datasets.
- plot_features(), correlation_matrix(), print_balance(): Visualize data distributions.

The whole purpose of this module is to standardizes the data preparation for HAR experiments
ensuring compatibility across datasets.
"""

import os
import numpy as np
import pandas as pd
import torch 
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn import preprocessing
from sklearn import model_selection
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from collections import Counter

#workflow: sample data with replacement -> [change_volume] -> [class_imbalance] -> output
#class_imbalance -> [make it balanced : oversampling || undersampling] -> [debalanced it] -> output
#to_client: sample with replacement if we select max clients = 1 otherwise sample without replacement

# =================================== dataloading ==================================
# dataset format:  concat(X_train,y_train), labels column 'activity' -> to DataFrame
# NOTE: if we have labeled data from multiple subjects we will keep subject id in the 'subject' column : a.k.a Federated Data

# ---> Data1: HARsense
# -> usefull links: https://ieee-dataport.org/open-access/harsense-statistical-human-activity-recognition-dataset
# -> info: sensors = {accelerometers,gyroscope}, devices = {Samsumg Galaxy A32, Poco X2} (on waists and pockets)
#          labels: {Walking, Standing, Upstaires, Downstaires, Running, Sitting}
#-> TODO: add subject_id

def load_data1():
    # load dataset
    data1_path = "/Users/admin/Desktop/thesis/dataset/Data_1_HARSense_Statistical_Human_Activity_Recognition/HARSense/All Users Combined.csv" # <- (PATH HERE)
    df = pd.read_csv(data1_path)
    #labels2idx = {"Walking":0, "Standing":1 ,"Upstaires":2, "Downstaires":3, "Running":4, "Sitting":5}
    #df["activity"].replace(labels2idx,inplace=True)

    return df

# ---> Data2: HAR using smartphones
# -> usefull links: https://www.kaggle.com/datasets/uciml/human-activity-recognition-with-smartphones/code
#                   https://paperswithcode.com/dataset/har
# -> info: sensors = {accelerometers,estimated body acceleration}, devices = {Samsung Galaxy S2} (on waist)
#          labels = {"walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"}
#          561 feature vector with time and frequency domain attributes
#-> TODO: add subject_id, experiment with PCA or something (to many features)


def load_data2():
    # load data
    data2_path = "/Users/admin/Desktop/thesis/dataset/Data_2_human_activity_recognition_using_smartphones/"
    X_train_df = pd.read_fwf(data2_path + "UCI HAR Dataset/train/" + "X_train.txt",header=None)
    y_train_df = pd.read_fwf(data2_path + "UCI HAR Dataset/train/" + "y_train.txt",header=None)
    X_test_df = pd.read_fwf(data2_path + "UCI HAR Dataset/test/" + "X_test.txt",header=None)
    y_test_df = pd.read_fwf(data2_path + "UCI HAR Dataset/test/" + "y_test.txt",header=None)

    # hardwire labels
    labels = {1:"walking", 2:"walking_upstairs", 3:"walking_downstairs", 4:"sitting", 5:"standing", 6:"laying"}

    # label conversion: indices to text
    y_train_df = y_train_df.replace(labels).astype(str)
    y_test_df = y_test_df.replace(labels).astype(str)

    # concat results
    Xy_train_df = pd.concat((X_train_df,y_train_df),axis=1)
    Xy_train_df.columns = [*Xy_train_df.columns[:-1], "activity"] # set labels name to activity
    Xy_test_df = pd.concat((X_test_df,y_test_df),axis=1)
    Xy_test_df.columns = [*Xy_test_df.columns[:-1], "activity"] # set labels name to activity

    df = pd.concat((Xy_train_df,Xy_test_df),axis=0)
    return df

# ---> Data3: Pamap2
# -> usefull links: https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring
#                   https://www.kaggle.com/code/avrahamcalev/time-series-models-pamap2-dataset
#                   https://paperswithcode.com/dataset/pamap2
# -> info: sensors = {IMU,HR-monitor}, devices = {3 colibri IMU(writs,chest,ankle), HR-monitor(chest mounted)}
#          labels = {"lying", "sitting", "standing", "walking", "running", "cycling",
                     #"nordic_walking", "watching_TV", "computer_work", "car_driving",
#                    "ascending_stairs", "descending_stairs", "vacuum_cleaning",
#                    "vacumm_cleaning", "ironing", "folding_laundry", "house_cleaning",
#                    "playing_soccer", "rope_jumping"}
#-> TODO: add subject_id, we may want to filter the sheer volume of data, maybe drop last 4 columns (no extra information)


def load_data3(n_subjects=4): # max = 9
    data3_path = "/Users/admin/Desktop/thesis/dataset/Data_3_pamap2_physical_activity_monitoring/PAMAP2_Dataset/"
    data_type = "/Protocol/"

    subject_list = ["subject10" + str(idx) + ".txt" for idx in range(1,n_subjects + 1,1)]

    # read subjects data
    lines = []
    for subject in subject_list:
        with open(data3_path + data_type + subject)as f:
            lines.append(f.readlines())

    flat_lines = [item for sublist in lines for item in sublist]

    data = []
    # convert str to float
    for sample in flat_lines:
        sample_vec = sample.strip('\n').split()
        sample_vec = [float(x) for x in sample_vec] # str to float
        data.append(sample_vec)


    # hardwire labels
    labels = {1:"lying", 2:"sitting", 3:"standing", 4:"walking", 5:"running", 6:"cycling", \
          7:"nordic_walking", 9:"watching_TV", 10:"computer_work", 11:"car_driving", \
          12:"ascending_stairs", 13:"descending_stairs", 16:"vacuum_cleaning", \
          17:"ironing", 18:"folding_laundry", 19:"house_cleaning", \
          20:"playing_soccer", 24:"rope_jumping"}

    # convert to DataFrame
    df = pd.DataFrame(data)
    df.drop([0,2],axis=1,inplace=True) # drop timestamp and heart rate attribute
    df.columns = ["activity",*range(51)] # ["activity",*df.columns[:-1]] # dont change activity order (to avoid unecessary copy)
    ind = np.where(df['activity'].to_numpy() == 0.0)[0] # find zero lable indices
    df.drop(ind,inplace=True) # delete zero labels (transient activities)
    df["activity"].replace(labels,inplace=True)

    # remove samples that contain Nan values
    # we could augment data.But unnecessary
    to_remove = np.zeros(df.shape[0],dtype=bool)
    for idx in range(df.shape[1] - 1):
        find_nans = df[idx].isna().to_numpy()
        to_remove = np.logical_or(to_remove,find_nans)
    ind = np.where(to_remove == True)[0]

    df.reset_index(drop=True,inplace=True) # reset indexing after shuffling
    df.drop(ind,axis=0,inplace=True)
    df.reset_index(drop=True,inplace=True) # just in case

    return df

# ---> Data4: MHealth 
# -> usefull links: https://archive.ics.uci.edu/dataset/319/mhealth+dataset
#                   https://www.kaggle.com/datasets/gaurav2022/mobile-health 
# -> info: sensors = {accelerometers,2-LEAD ECG}, devices = {Shimmer 2} ( accelerometers(right wrist,left ankle, chest), 2-LEAD(chest) )
#          labels = {"null", "standing_still", "sitting_and_relaxing", "lying_down", "walking", "climbing_stairs",
#                   "waist_bend_forward", "frontal_elevation_of_arms", "knees_bending", "cycling",
#                   "jogging", "running", "jummp_front_and_back"}
#-> TODO: add subject_id


def load_data4():
    data4_path = "/Users/admin/Desktop/thesis/dataset/Data_4_MHEALTHDATASET/"
    id_list = [*range(11)][1:]
    lines = []

    # scan every subject
    for idx in id_list:
        with open(data4_path + "mHealth_subject" + str(idx) + ".txt") as f:
            data = f.readlines()
            lines.append(data)

    # skip subject id for the time being : convert list of list to list
    flat_lines = [item for sublist in lines for item in sublist]

    data = []
    # convert str to float
    for sample in flat_lines:
        sample_vec = sample.strip('\n').split('\t')
        sample_vec = [float(x) for x in sample_vec] # str to float
        data.append(sample_vec)

    # convert to df
    df = pd.DataFrame(data)



    df.columns = [*df.columns[:-1], "activity"] # set labels name to activity

    # filter null indices: we may want to keep this if we want to classify null activities
    #ind = np.where(df['activity'].to_numpy() == 'null')[0]
    ind = np.where(df['activity'].to_numpy() == 0)[0]
    df = df.drop(ind)
    
    # hardwire labels
    labels = {0:"null", 1:"standing_still", 2:"sitting_and_relaxing", 3:"lying_down", 4:"walking", 5:"climbing_stairs", \
             6:"waist_bend_forward", 7:"frontal_elevation_of_arms", 8:"knees_bending", 9:"cycling", \
             10:"jogging", 11:"running", 12:"jummp_front_and_back"}
    df['activity'].replace(labels,inplace=True)
    
    return df

# ---> Data5: acc data (raw)
# -> usefull links: https://physionet.org/content/accelerometry-walk-climb-drive/1.0.0/
# -> info: sensors = {accelerometers}, devices = {3 axial ActiGraph GT3X+} (left wrist,left-right ankle, left hip)
#          labels = {"walking", "descending_stairs", "ascending_stairs", "driving", "clapping", "non_study_activity"};
#-> TODO: add subject_id

def load_data5():
    data5_path = "/Users/admin/Desktop/thesis/dataset/Data_5_labeled-raw-accelerometry-data-captured-during-walking-stair-climbing-and-driving-1.0.0/"
    subject_path_list = os.listdir(data5_path + "raw_accelerometry_data") # concenrate every file in directory

    # correlate subjects id : (for Fed data creation)
    subject_id = {}
    df_list = []
    for idx,subject_path in enumerate(subject_path_list):
        subject_id[idx] = subject_path
        current_df = pd.read_csv(data5_path + "raw_accelerometry_data/" + subject_path)
        # append subject column
        subject_id_data = np.ones(current_df.shape[0],dtype=np.int32) * idx
        current_df.insert(1,"subject",subject_id_data)
        df_list.append(current_df)

    df = pd.concat(df_list,axis=0) # concat dfs

    labels = {1:"walking", 2:"descending_stairs", 3:"ascending_stairs", 4:"driving", 77:"clapping", 99:"non_study_activity"};
    df.drop("time_s",axis=1,inplace=True) # drop timestamp
    df.reset_index(drop=True,inplace=True)
    # remove label 99 : null activity
    ind = np.where(df['activity'].to_numpy() == 99)[0]
    df.drop(ind,inplace=True)
    df['activity'].replace(labels,inplace=True) # replace label tags with names

    return df

# ---> Data6: MotionSense
# -> usefull links: https://www.kaggle.com/datasets/malekzadeh/motionsense-dataset
#                   https://paperswithcode.com/dataset/motionsense
# -> info: sensors = {accelerometer, gravitometer}, devices = {iPhone6} 
#          labels = {'downstairs', 'upstairs', 'walking', 'jogging', 'standing', 'sitting'}
#-> TODO: add subject_id

def set_data_types(data_types=["userAcceleration"]):
    """
    Select the sensors and the mode to shape the final dataset.
    
    Args:
        data_types: A list of sensor data type from this list: [attitude, gravity, rotationRate, userAcceleration] 

    Returns:
        It returns a list of columns to use for creating time-series from files.
    """
    dt_list = []
    for t in data_types:
        if t != "attitude":
            dt_list.append([t+".x",t+".y",t+".z"])
        else:
            dt_list.append([t+".roll", t+".pitch", t+".yaw"])
    print(dt_list)
    return dt_list

def get_ds_infos():
    """
    Read the file includes data subject information.
    
    Data Columns:
    0: code [1-24]
    1: weight [kg]
    2: height [cm]
    3: age [years]
    4: gender [0:Female, 1:Male]
    
    Returns:
        A pandas DataFrame that contains inforamtion about data subjects' attributes 
    """ 
    data6_path = "/Users/admin/Desktop/thesis/dataset/Data_6_MotionSense/" # <- path here
    dss = pd.read_csv(data6_path + "data_subjects_info.csv")
    print("[INFO] -- Data subjects' information is imported.")
    
    return dss

def creat_time_series(folder_name, dt_list, act_labels, trial_codes, mode="mag", labeled=True):
    """
    Args:
        folder_name: one of 'A_DeviceMotion_data', 'B_Accelerometer_data', or C_Gyroscope_data
        dt_list: A list of columns that shows the type of data we want.
        act_labels: list of activites
        trial_codes: list of trials
        mode: It can be 'raw' which means you want raw data
        for every dimention of each data type,
        [attitude(roll, pitch, yaw); gravity(x, y, z); rotationRate(x, y, z); userAcceleration(x,y,z)].
        or it can be 'mag' which means you only want the magnitude for each data type: (x^2+y^2+z^2)^(1/2)
        labeled: True, if we want a labeld dataset. False, if we only want sensor values.

    Returns:
        It returns a time-series of sensor data.
    
    """
    num_data_cols = len(dt_list) if mode == "mag" else len(dt_list*3)

    if labeled:
        dataset = np.zeros((0,num_data_cols+7)) # "7" --> [act, code, weight, height, age, gender, trial] 
    else:
        dataset = np.zeros((0,num_data_cols))
        
    ds_list = get_ds_infos()
    
    print("[INFO] -- Creating Time-Series")
    for sub_id in ds_list["code"]:
        for act_id, act in enumerate(act_labels):
            for trial in trial_codes[act_id]:
                fname = folder_name+'/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                raw_data = pd.read_csv(fname)
                raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                vals = np.zeros((len(raw_data), num_data_cols))
                for x_id, axes in enumerate(dt_list):
                    if mode == "mag":
                        vals[:,x_id] = (raw_data[axes]**2).sum(axis=1)**0.5        
                    else:
                        vals[:,x_id*3:(x_id+1)*3] = raw_data[axes].values
                    vals = vals[:,:num_data_cols]
                if labeled:
                    lbls = np.array([[act_id,
                            sub_id-1,
                            ds_list["weight"][sub_id-1],
                            ds_list["height"][sub_id-1],
                            ds_list["age"][sub_id-1],
                            ds_list["gender"][sub_id-1],
                            trial          
                           ]]*len(raw_data), dtype=int)
                    vals = np.concatenate((vals, lbls), axis=1)
                dataset = np.append(dataset,vals, axis=0)
    cols = []
    for axes in dt_list:
        if mode == "raw":
            cols += axes
        else:
            cols += [str(axes[0][:-2])]
            
    if labeled:
        cols += ["act", "id", "weight", "height", "age", "gender", "trial"]
    
    dataset = pd.DataFrame(data=dataset, columns=cols)
    return dataset

def load_data6():
    # wrap function for creat_time_series
    ACT_LABELS = ["dws","ups", "wlk", "jog", "std", "sit"]
    TRIAL_CODES = {
        ACT_LABELS[0]:[1,2,11],
        ACT_LABELS[1]:[3,4,12],
        ACT_LABELS[2]:[7,8,15],
        ACT_LABELS[3]:[9,16],
        ACT_LABELS[4]:[6,14],
        ACT_LABELS[5]:[5,13]
    }
    
    data6_path = "/Users/admin/Desktop/thesis/dataset/Data_6_MotionSense/"
    folder_name = data6_path + "A_DeviceMotion_data/A_DeviceMotion_data" # <- path here
    sdt = ["attitude", "gravity", "rotationRate", "userAcceleration"]
    dt_list = set_data_types(sdt)
    act_labels = ACT_LABELS [0:6]
    trial_codes = [TRIAL_CODES[act] for act in act_labels]
    
    df = creat_time_series(folder_name, dt_list, act_labels, trial_codes, mode="raw", labeled=True)
    
    # modify existing format
    # 1. act -> activity
    # 2. id -> subject
    # 3. match activit_id with har name
    # 4. remove weight,height,age,gender,trial
    
    df.drop(["weight","height","age","gender","trial"],axis=1,inplace=True) # 4. drop timestamp
    # 1,2
    columns = [*df.columns]
    columns[-2] = 'activity'
    columns[-1] = 'subject'
    df.columns = columns
    # 3
    labels = {0:'downstairs', 1:'upstairs', 2:'walking', 3:'jogging', 4:'standing', 5:'sitting'}
    df['activity'].replace(labels,inplace=True) # replace label tags with names # -> find labels

    return df

#===================================================================================
def train_test_split(data_df,test_size=0.2, scaler_type="standard", should_map_labels=True, random_seed=42): # data_df = (n_samples + header) x (n_features + labels)
    # seperate features,labels
    label_tag = 'activity'
    X_df = data_df.drop(label_tag, axis=1)
    y_df = data_df[label_tag]
    #labels = y_df.unique() # get labels
        
    # label encoding
    labels = None
    if should_map_labels:
        le = preprocessing.LabelEncoder()        
        y = le.fit_transform(y_df.values)
        labels = le.classes_
    else:
        y = y_df.values
        
    # train test split
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X_df,y,test_size=test_size,random_state=np.random.RandomState(random_seed))
    
    # feature preprocessing
    if scaler_type == "minmax":
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        scaler = StandardScaler()
        
    scaler.fit(X_train)

    X_train = scaler.transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)
    
    return X_train,y_train.astype(np.float32),X_test,y_test.astype(np.float32),labels

# sampling data with replacement
def to_client(data, max_clients, only_one_test_data=True, data_frac=None, random_seed=42): # n_frac : data volume knob
    X_train,y_train,X_test,y_test = data # decompile data to train test split format
    
    if data_frac != None:
        train_size = [round(X_train.shape[0]*frac) for frac in data_frac]
        test_size = [round(X_test.shape[0]*frac) for frac in data_frac]
    else:
        train_size = [int(X_train.shape[0] / max_clients) for _ in range(max_clients)]
        test_size = [int(X_test.shape[0] / max_clients) for _ in range(max_clients)]
    
    # dichotomize data to N clients
    train_idx_list = [torch.randint(0,X_train.shape[0],(train_size[i],), generator=torch.Generator().manual_seed(random_seed + i)).tolist() for i in range(max_clients)]
    test_idx_list = [torch.randint(0,X_test.shape[0],(test_size[i],), generator=torch.Generator().manual_seed(random_seed + i)).tolist() for i in range(max_clients)]
    
    # to client data : [(X_train,y_train,X_test,y_test) ,() , .... ()]
    if only_one_test_data:
        client_data = [( X_train[train_idx_list[i]], y_train[train_idx_list[i]], X_test, y_test) for i in range(max_clients)]
    else:
        client_data = [( X_train[train_idx_list[i]], y_train[train_idx_list[i]], X_test[test_idx_list[i]], y_test[test_idx_list[i]] ) for i in range(max_clients)]

    return client_data

def save_client_data(client_data,filename, labels=None):
    # get path to working data from config file
    path_to_config = os.path.join(os.path.dirname(os.getcwd()),"config.txt")
    with open(path_to_config, 'r') as file:
        path_to_working_data = file.readline().split(" ")[0]

    # create mandadory directories
    os.makedirs(os.path.join(path_to_working_data,filename),exist_ok=True)
    os.makedirs(os.path.join(path_to_working_data,filename,'X_train'), exist_ok=True) # X_train
    os.makedirs(os.path.join(path_to_working_data,filename,'y_train'), exist_ok=True) # y_train
    os.makedirs(os.path.join(path_to_working_data,filename,'X_test'), exist_ok=True) # X_test
    os.makedirs(os.path.join(path_to_working_data,filename,'y_test'), exist_ok=True) # y_test
    # Save labels if provided
    if labels is not None:
        labels_path = os.path.join(path_to_working_data, filename, 'labels.txt')
        with open(labels_path, 'w', encoding='utf-8') as f:
            for label in labels:
                f.write(f"{label}\n")
        print(f"Labels saved in {labels_path}")
        
    for idx,(X_train,y_train,X_test,y_test) in enumerate(client_data):
        #print(f"saving partition_{str(idx)}.txt")
        # X_train
        tag = 'X_train'
        #print(tag)
        #with open(path_to_working_data + filename + tag + "partition_" + str(idx) + ".txt",'w') as f:
        with open(os.path.join(path_to_working_data, filename, tag, "partition_" + str(idx) + ".txt"),'w') as f:
            for line in X_train:
                line2list = [str(value) for value in line]
                line2str = ' '.join(line2list)
                f.write(line2str)
                f.write('\n')

        # y_train
        tag = 'y_train'
        #print(tag)
        #with open(path_to_working_data + filename + tag + "partition_" + str(idx) + ".txt",'w') as f:
        with open(os.path.join(path_to_working_data, filename, tag, "partition_" + str(idx) + ".txt"),'w') as f:
            for value in y_train:
                f.write(str(value))
                f.write('\n')

        # X_test
        tag = 'X_test'
        #print(tag)
        #with open(path_to_working_data + filename + tag + "partition_" + str(idx) + ".txt",'w') as f:
        with open(os.path.join(path_to_working_data, filename, tag, "partition_" + str(idx) + ".txt"),'w') as f:
            for line in X_test:
                line2list = [str(value) for value in line]
                line2str = ' '.join(line2list)
                f.write(line2str)
                f.write('\n')

        # y_test
        tag = 'y_test'
        #print(tag)
        #with open(path_to_working_data + filename + tag + "partition_" + str(idx) + ".txt",'w') as f:
        with open(os.path.join(path_to_working_data, filename, tag, "partition_" + str(idx) + ".txt"),'w') as f:
            for value in y_test:
                f.write(str(value))
                f.write('\n')

def class_imbalance(data,class_ratio,balance=True):
    X_train, y_train, X_test, y_test = data
    # step 1: balance data
    if balance == True:
        X_train,y_train,X_test,y_test = balance_fn(X_train,y_train,X_test,y_test,True) # balance dataset
    
    # step 2: debalance it based on class_ratio
    # train
    X_idx = []
    y_idx = []
    for idx,ratio in enumerate(class_ratio):
        ind = np.where(y_train == idx)[0] # find indices at class == idx
        _ = np.random.shuffle(ind) # shuffle indices  pass by reference
        ind = ind[0:round(ind.shape[0] * ratio)] # filter indices to match ratio
        X = X_train[ind]
        y = y_train[ind]
        X_idx.append(X)
        y_idx.append(y)
    X_train = np.concatenate(X_idx,axis=0) # concate labels
    y_train = np.concatenate(y_idx,axis=0)
    X_train,y_train = shuffle(X_train,y_train) # shuffle data
    
    # test
    #X_idx = []
    #y_idx = []
    #for idx,ratio in enumerate(class_ratio):
    #    ind = np.where(y_test == idx)[0] # find indices at class == idx
    #    _ = np.random.shuffle(ind) # shuffle indices  pass by reference
    #    ind = ind[0:round(ind.shape[0] * ratio)] # filter indices to match ratio
    #    X = X_test[ind]
    #    y = y_test[ind]
    #    X_idx.append(X)
    #    y_idx.append(y)
    #X_test = np.concatenate(X_idx,axis=0)
    #y_test = np.concatenate(y_idx,axis=0)
    #X_test,y_test = shuffle(X_test,y_test) # shuffle data
    
    return (X_train,y_train,X_test,y_test)

def balance_fn(X_train,y_train,X_test,y_test,over=False, random_seed=42):
    if over == True:
        # oversampling
        sampler = RandomOverSampler(random_state=random_seed)
    else:
        # undersamping
        sampler = RandomUnderSampler(random_state=random_seed)
    
    X_train_b,y_train_b = sampler.fit_resample(X_train, y_train)
    X_test_b,y_test_b = sampler.fit_resample(X_test, y_test)
    
    return X_train_b,y_train_b,X_test_b,y_test_b

def print_balance(y_train,y_test,bins=6):
    counts,_ = np.histogram(y_train,bins)
    ratio = counts / counts.sum()
    print(f"y_train counts: {counts} ratio: {ratio}")
    counts,_ = np.histogram(y_test,bins)
    ratio = counts / counts.sum()
    print(f"y_test counts: {counts} ratio: {ratio}")
    #print("balanced_ratio: " + str(1/6))

def print_balance_2(y_train, y_test):
    train_counts = Counter(y_train)
    test_counts = Counter(y_test)

    train_info =  "Train label distribution:"
    for label in sorted(train_counts):
        train_info += f" {label}: {train_counts[label]} "
       
    print(train_info)
    
    test_info = "Test label distribution:"
    for label in sorted(test_counts):
        test_info += f" {label}: {test_counts[label]} "
    print(test_info)
    
def plot_features(df):
    columns = list(df.columns)
    for column in columns:
        plt.figure()
        plt.title(column)
        plt.plot(df[column])

def correlation_matrix(df,has_label=False):
    if has_label == True:
        # insert dataframe: drop activity column and compute corr matrix
        df.drop('activity',axis=1)
    
    f = plt.figure(figsize=(12, 10))
    plt.matshow(df.corr(), fignum=f.number)
    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    #plt.title('Correlation Matrix', fontsize=16)

def downsample_train_test_split(X_train, y_train, X_test, y_test, frac=0.5, random_seed=42):
    """
    Downsamples both train and test sets to a given fraction using random indexing.
    
    Parameters:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing labels
        frac (float): Fraction of data to keep
        seed (int): Random seed for reproducibility

    Returns:
        Tuple of downsampled: (X_train_ds, y_train_ds, X_test_ds, y_test_ds)
    """
    np.random.seed(random_seed)

    # Downsample train set
    train_indices = np.random.permutation(len(X_train))[:int(frac * len(X_train))]
    X_train_ds = X_train[train_indices]
    y_train_ds = y_train[train_indices]

    # Downsample test set
    test_indices = np.random.permutation(len(X_test))[:int(frac * len(X_test))]
    X_test_ds = X_test[test_indices]
    y_test_ds = y_test[test_indices]

    return X_train_ds, y_train_ds, X_test_ds, y_test_ds 

def take_first_n_fraction(X_train, y_train, X_test, y_test, frac=0.5):
    """
    Takes the first N samples from train and test sets based on the given fraction.

    Parameters:
        X_train, y_train, X_test, y_test: np.ndarray
            Already split datasets.
        frac: float
            Fraction of samples to keep (e.g., 0.5 keeps 50%).

    Returns:
        Subsampled datasets.
    """
    n_train = int(len(X_train) * frac)
    n_test = int(len(X_test) * frac)

    return (
        X_train[:n_train], y_train[:n_train],
        X_test[:n_test], y_test[:n_test]
    )

#if __name__ == "__main__":
#    # load dataset
#    harsense_datapath = "/Users/admin/Desktop/thesis/dataset/HARSense_Statistical_Human_Activity_Recognition/HARSense/All Users Combined.csv"
#    df = pd.read_csv(harsense_datapath)
#
#    # train_test_split
#    X_train,y_train,X_test,y_test,labels = train_test_split(df,test_size=0.2)
#
#    ## dichotomize data to clients
#    #data_frac = [0.5,0.25,0.125,0.125]
#    #client_data = to_client(data=(X_train,y_train,X_test,y_test),max_clients=4,data_frac=data_frac)
#    #
#    ## get data statistics
#    #for idx,(X_train,y_train,X_test,y_test) in enumerate(client_data):
#    #    print('-----------')
#    #    counts,_ = np.histogram(y_train,bins=len(labels))
#    #    ratio = counts / counts.sum()
#    #    print(f"y_train counts: {counts} ratio: {ratio}")
#    #    counts,_ = np.histogram(y_test,bins=len(labels))
#    #    ratio = counts / counts.sum()
#    #    print(f"y_test counts: {counts} ratio: {ratio}")
#    #
#    ## save client
#    #save_client_data(client_data,filename='test1')
#
#    # prepare volume_test : sampling with replacement 
#    #data_frac_list = []
#    #value = 0.5
#    #for i in range(10):
#    #    data_frac = [round(value,2),round(1 - value,2)]
#    #    data_frac_list.append(data_frac)
#    #    value += 0.05 
#    #
#    #print(data_frac_list)
#    #client_data = to_client(data=(X_train,y_train,X_test,y_test),max_clients=2,data_frac=data_frac)
#
#    # test1 -> volume test : volume = 0.05:0.05:1.0 : sampling with replacement
#    step = 0.05
#    volumes = np.arange(0,1+step,step)
#    client_data_volume = []
#    for volume in volumes:
#        volume = round(volume,2)
#        client_data = to_client(data=(X_train,y_train,X_test,y_test),max_clients=1,data_frac=[volume])[0] # single client data : (X_train,y_train,X_test,y_test)
#        client_data_volume.append(client_data)
#
#    save_client_data(client_data_volume,filename='volume_test_1')
#
#    # test2 -> class imbalance test:
#    #class_ratio = list(np.ones(6) * 0.5)
#    #X_train2,y_train2,X_test2,y_test2 = class_imbalance_2(X_train,y_train,X_test,y_test,class_ratio,balance=True) 
