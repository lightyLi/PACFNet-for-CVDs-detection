import os
import scipy.io as sio
import numpy as np
from scipy import interpolate
from config import fs
from config import annotation_folder
from config import segmentation_method, beat_num
from plot_fig import plot_signals_comparison

# load annotation data
def get_annotation_dict(sub_id):
    mat_file_name = os.path.join(annotation_folder, f"{sub_id}_StateAns.mat")
    mat_dict=sio.loadmat(mat_file_name)
    
    state_ans = mat_dict['state_ans']   # Return the state_ans item in the mat file
    state_dict = {key[0,0]:str(value).split('\'')[1] for (key,value) in state_ans }
    # state_dict key-value pairs are: {point: state_name}
    return state_dict

# Time scale normalization
def time_scale_normalize_signal(signals, fs, target_time=1):
    """
    Normalize ECG signals of different lengths to specified length
    
    Parameters:
    signals:    List containing ecg and pcg signals
    fs:        int - Sampling frequency
    target_length: int - Target length (number of points)
    Returns:
    List[np.ndarray] - List of normalized signals
    """
    def _normalize_signal_sample(signal):
        """
        Normalize a single signal to specified length
        """
        x_original = np.linspace(0, target_time, len(signal))
        x_target = np.linspace(0, target_time, target_length)
        f = interpolate.interp1d(x_original, signal, kind='cubic')
        normalized_signal = f(x_target)
        return normalized_signal
    
    target_length = int(target_time * fs)
    normalized_signals = []
    
    for data in signals:
        normalized_data = _normalize_signal_sample(data)
        normalized_signals.append((normalized_data))

    return normalized_signals

def get_segmented_samples_single_record(sub_id, data, segmentation_method='b2b', beat_num=1, window_length=3, overlap_size=2, pad_end=False):
    '''
    Get segmented samples according to different segmentation methods
    '''
    # Get b2b segmentation positions
    b2b_annotation_dict = get_annotation_dict(sub_id)

    # Use two methods to get segmentation positions
    def _sliding_window_segmentation(data, fs, window_length, overlap_size, pad_end=False):
        """
        Sliding window segmentation
        When segmenting, need to handle the last segment separately, padding with zeros if length is insufficient
        window_length, overlap_size are both in seconds
        Function returns a list of tuples
        """
        assert (overlap_size < window_length and overlap_size >= 0), 'Overlap size must be between 0 and window length!'
            
        segments_points = [] # tuple list to store the begin and the end point of a segment

        # Calculate total recording time
        window_samples = int(window_length * fs)
        step_samples = window_samples - int(overlap_size * fs)

        # Calculate number of complete segments
        num_complete_segments = (len(data) - window_samples) // step_samples + 1
        
        # Segment complete sections
        for i in range(num_complete_segments):
            start = i * step_samples
            end = start + window_samples
            segments_points.append((start, end))
        # Handle the last segment
        last_start = num_complete_segments * step_samples
        if last_start < len(data):
            remain_samples = len(data) - last_start
            if pad_end and remain_samples < window_samples and len(data) % step_samples!=0:
                # Zero padding
                segments_points.append((last_start, last_start+window_samples))
            # If not padding and length is insufficient, discard
        return segments_points
    def _b2b_segmentation(state_annotations, beat_num=1):
        '''
        Mainly reads the mat files provided in the dataset to get the start time of each state in each file
        According to the dataset description, these annotations have been manually calibrated and should be directly usable.
        S1 in PCG generally follows the R peak, with a delay of several tens to hundreds of ms
        Can directly use S1-S1 segmentation, or use S1 to determine the position of R peaks, using R-R segmentation

        In the return value, segmented_beats is a list composed of tuples

        Note: All of the following annotations are the start times of each state (sampling points corresponding to the start time)
        '''
        all_annotations = list(state_annotations.keys())
        s_1 = [key for (key,value) in state_annotations.items() if value == "S1"]
        s_2 = [key for (key,value) in state_annotations.items() if value == "S2"]
        systole = [key for (key,value) in state_annotations.items() if value == "systole"]
        diastole = [key for (key,value) in state_annotations.items() if value == "diastole"]
        segmented_beats = s_1
        # return beats according to beat_num
        if beat_num:
            segmented_beats_list = [(segmented_beats[i],segmented_beats[i+beat_num])  for i in range(0,len(segmented_beats)-beat_num, beat_num)]
        return (all_annotations, s_1, s_2, systole, diastole, segmented_beats_list)

    # Perform segmentation
    def _segmentation(data, segmented_points, pad_end=False):
        segmented_samples = []  # Use tuple type to store segmented sections, first element is ecg, second is PCG
        if pad_end:
            _, final_end_point = segmented_points[-1]
            last_segmentation_length = final_end_point - len(data)
            data = np.concatenate((data, np.zeros(last_segmentation_length)))
        for start, end in segmented_points:
            segmented_samples.append(data[start:end])
        return segmented_samples
    
    if segmentation_method == "b2b":
        *_, segmented_points = _b2b_segmentation(b2b_annotation_dict, beat_num=beat_num)
        
    else:
        segmented_points = _sliding_window_segmentation(data, fs, window_length, overlap_size, pad_end=pad_end)
    
    segmented_samples = _segmentation(data, segmented_points, pad_end=pad_end)
    return segmented_samples, segmented_points

def get_segmented_samples(data_list, segmentation_method=segmentation_method, beat_num=beat_num, window_length=3, overlap_size=2, pad_end=False):
    '''
    Get segmented samples from the input records
    data_list: List of records, each record is a dictionary with key-value pairs: {record_name: data}
    '''
    segmented_samples_list = []
    for item in data_list:
        record_name = list(item.keys())[0]
        data = list(item.values())[0]
        segmented_samples, _ = get_segmented_samples_single_record(record_name, data, segmentation_method, beat_num, window_length, overlap_size, pad_end)
        # Time scale normalization
        if segmentation_method == 'b2b':
            segmented_samples = time_scale_normalize_signal(segmented_samples, fs)

        segmented_samples_list.append({record_name: segmented_samples})

    return segmented_samples_list