import os
import pickle
import random
import numpy as np
import pandas as pd
import scipy.io as sio
import wfdb
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from config import databset_folder, csv_file, processed_folder
from config import ECGNotExistError, PCGNotExistError, SignalLengthMismatchError
from config import segmentation_method, beat_num, n_components, kf_n_splits,random_state
from utils import save_data_to_pickle, read_dat_wav, linear_interpolation, logger, load_data_from_pickle
from segmentation import get_segmented_samples
from emd_decomposition import emd_decomposition, analyze_imfs_correlations, adaptive_reconstruct_signal, reconstruct_with_correlations

def single_z_score(records:tuple):
    """
    Perform z-score normalization on a single record (ecg, pcg)
    Parameters:
    records: tuple - Tuple containing ECG and PCG signals, or tuple containing multiple signals
    Returns:
    tuple - Normalized signals
    """
    def _z_score(data):
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            raise ValueError("Standard deviation is zero, cannot perform z-score normalization.")
        return (data - mean) / std
    ecg_data, pcg_data = records
    z_scored_ecg = _z_score(ecg_data)
    z_scored_pcg = _z_score(pcg_data)
    return (z_scored_ecg, z_scored_pcg)

def load_all_files(csv_file, databset_folder):
    '''
    ecg_data: Dictionary containing ECG signals, with record name as key and ECG signal as value
    pcg_data: Dictionary containing PCG signals, with record name as key and PCG signal as value
    labels: Dictionary containing record names and labels, with record name as key and label as value
    fs: Sampling frequency
    noisy_record_names: List containing noisy records
    '''
    def _check_synchronized_ecg_pcg(record):
        sig_names = record.sig_name
        # Check if ECG and PCG exist simultaneously
        # and if they have the same length
        if "ECG" not in sig_names:
            raise ECGNotExistError
        if "PCG" not in sig_names:
            raise PCGNotExistError
        pcg_data = record.p_signal[:, record.sig_name.index('PCG')]
        ecg_data = record.p_signal[:, record.sig_name.index('ECG')]
        if len(pcg_data) != len(ecg_data):
            raise SignalLengthMismatchError
        return (ecg_data, pcg_data)
    
    def _scale_data(data, x):
        """
        Function used for scaling the data from -x to x
        """
        max_val = np.max(np.abs(data))
        scaled_data = (x * data) / max_val
        return scaled_data

    def _load_single_pcg_ecg(path,sub_id):
        record = wfdb.rdrecord(os.path.join(path, sub_id))
        fs = record.fs
        ecg_data, pcg_data = _check_synchronized_ecg_pcg(record)
        # Linear interpolation
        ecg_data = linear_interpolation(ecg_data)
        pcg_data = linear_interpolation(pcg_data)
        pcg_data = _scale_data(pcg_data,1)
        ecg_data = _scale_data(ecg_data,1)
        return pcg_data, ecg_data, fs

    ecg_data_list, pcg_data_list, labels = [], [], []
    noisy_record_names = []
    df = pd.read_csv(csv_file, header=None)

    for row in df.itertuples(index=False):
        record_name, label, noise_label = row
        if noise_label:
            try:
                pcg_data, ecg_data, fs = _load_single_pcg_ecg(databset_folder, record_name)
            except ECGNotExistError: 
                print(f"****Something error happend: ECG file not included in record < {record_name} >****")
                continue
            except PCGNotExistError:
                print(f"****Something error happend: PCG file not included in record < {record_name} >****")
                continue
            except SignalLengthMismatchError:
                print(f"****Something error happend: ECG and PCG doesn't have same length. Record name: < {record_name} >****")
                continue
            pcg_data, ecg_data = single_z_score((pcg_data, ecg_data))
            ecg_data_list.append({record_name: ecg_data})
            pcg_data_list.append({record_name: pcg_data})

            # Convert label -1 to 0
            label = 0 if label == -1 else label
            labels.append({record_name: label})
        else:
            noisy_record_names.append(record_name)
            continue
            
    return ecg_data_list, pcg_data_list, labels, fs, noisy_record_names

def shuffle_data_using_kf(ecg_data_list, pcg_data_list, labels):
    '''
    Shuffle data using KFold and save as fixed 5-fold pickle files
    '''
    def _get_label_by_key(label_list, target_key):
        """
        Get value list through specified key
        """
        value = [d.get(target_key) for d in label_list if target_key in d][0]
        return value
    def _split_x_y(data, labels):
        x, y = [], []
        record_name = list(data.keys())[0]
        segments = list(data.values())[0]
        label = _get_label_by_key(labels, record_name)
        for segment in segments:
            x.append(segment)
            y.append(label)
        return x, y

    ecg_normal_segmented_list, ecg_abnormal_segmented_list = ecg_data_list
    pcg_normal_segmented_list, pcg_abnormal_segmented_list = pcg_data_list
    ecg_X, pcg_X, ecg_y, pcg_y = [], [], [], []
    # kf = KFold(n_splits=kf_n_splits, shuffle=True, random_state=random_state)
    kf = StratifiedKFold(n_splits=kf_n_splits, shuffle=True, random_state=random_state)
    # After shuffling, the index list is still arranged from small to large, but values are randomly taken in ascending order.
    
    for item in ecg_normal_segmented_list:
        temp_x, temp_y = _split_x_y(item, labels)
        ecg_X.extend(temp_x)
        ecg_y.extend(temp_y)
    for item in ecg_abnormal_segmented_list:
        temp_x, temp_y = _split_x_y(item, labels)
        ecg_X.extend(temp_x)
        ecg_y.extend(temp_y)
    for item in pcg_normal_segmented_list:
        temp_x, temp_y = _split_x_y(item, labels)
        pcg_X.extend(temp_x)
        pcg_y.extend(temp_y)
    for item in pcg_abnormal_segmented_list:
        temp_x, temp_y = _split_x_y(item, labels)
        pcg_X.extend(temp_x)
        pcg_y.extend(temp_y)
    
    ecg_n_ab_X , pcg_n_ab_X, y = [], [], []
    for ecg, pcg, ecg_label, pcg_label in zip(ecg_X, pcg_X, ecg_y, pcg_y):
        if ecg_label == pcg_label:
            ecg_n_ab_X.append(ecg)
            pcg_n_ab_X.append(pcg)
            y.append(ecg_label)

    ecg_n_ab_X , pcg_n_ab_X, y = np.array(ecg_n_ab_X), np.array(pcg_n_ab_X), np.array(y)
    for i, (train_index, test_index) in enumerate(kf.split(ecg_n_ab_X, y)):
        ecg_X_train, ecg_X_test = ecg_n_ab_X[train_index], ecg_n_ab_X[test_index]
        pcg_X_train, pcg_X_test = pcg_n_ab_X[train_index], pcg_n_ab_X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        save_data_to_pickle(processed_folder, f'{segmentation_method}_data_fold_{i+1}', (ecg_X_train, ecg_X_test, pcg_X_train, pcg_X_test, y_train, y_test))

        
def turn_data_2_0(data_list, name):
    print(f"{name} is being turning to 0!")
    return_data_list = []
    for item in data_list:
        record_name = list(item.keys())[0]
        signal = list(item.values())[0]
        signal_0 = np.zeros_like(signal)
        return_data_list.append({record_name:signal_0})
    return return_data_list
   
ecg_data_list, pcg_data_list, labels, fs, noisy_record_names = load_all_files(csv_file, databset_folder)
ecg_normal_list, pcg_normal_list, ecg_abnormal_list, pcg_abnormal_list = [], [], [], []
for ecg, pcg, label in zip(ecg_data_list, pcg_data_list, labels):
    lables = list(label.values())[0]
    record_name = list(label.keys())[0]
    ecg = list(ecg.values())[0]
    pcg = list(pcg.values())[0]

    if lables == 0:
        ecg_normal_list.append({record_name: ecg})
        pcg_normal_list.append({record_name: pcg})
    else:
        ecg_abnormal_list.append({record_name: ecg})
        pcg_abnormal_list.append({record_name: pcg})
# ecg_normal: 116, ecg_abnormal: 272, pcg_normal: 116, pcg_abnormal: 272 

# # Only use ECG for training, turn PCG data to 0, input to multi-modal model
# ecg_normal_list = turn_data_2_0(ecg_normal_list, 'ecg_normal_list')
# ecg_abnormal_list = turn_data_2_0(ecg_abnormal_list, 'ecg_abnormal_list')
# pcg_normal_list = turn_data_2_0(pcg_normal_list, 'pcg_normal_list')
# pcg_abnormal_list = turn_data_2_0(pcg_abnormal_list, 'pcg_abnormal_list')

# segment the new signals
ecg_normal_segmented_list = get_segmented_samples(ecg_normal_list, segmentation_method, beat_num)
ecg_abnormal_segmented_list = get_segmented_samples(ecg_abnormal_list, segmentation_method, beat_num)
pcg_normal_segmented_list = get_segmented_samples(pcg_normal_list, segmentation_method, beat_num)
pcg_abnormal_segmented_list = get_segmented_samples(pcg_abnormal_list, segmentation_method, beat_num)
save_data_to_pickle(processed_folder, f'{segmentation_method}_segmented_signals_list_{n_components}', 
    (ecg_normal_segmented_list, ecg_abnormal_segmented_list, pcg_normal_segmented_list, pcg_abnormal_segmented_list, labels))

# Load segmented data
ecg_normal_segmented_list, ecg_abnormal_segmented_list, pcg_normal_segmented_list, pcg_abnormal_segmented_list, labels= load_data_from_pickle(processed_folder, 'b2b_segmented_signals_list_adaptive')
# When using the regular zip() function, the length of the result equals the length of the shortest input list. This is because zip() stops when the shortest input sequence is exhausted.

# shuffle data using KFold 5-fold cross-validation
shuffle_data_using_kf((ecg_normal_segmented_list, ecg_abnormal_segmented_list), 
                    (pcg_normal_segmented_list, pcg_abnormal_segmented_list), labels)