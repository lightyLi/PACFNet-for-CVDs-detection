import os
import pickle
import time
import numpy as np
from scipy.io import wavfile
from functools import wraps
import logging
from datetime import datetime
from config import log_folder

def save_data_to_pickle(path, file_name, data):
    # Used to save data to pickle file
    # Create save directory (if it doesn't exist)
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, f'{file_name}.pkl'), 'wb') as f:
        pickle.dump(data, f)
    print(f"****Segments saved to {path}/{file_name}.pkl****")

def load_data_from_pickle(path, file_name):
    # Used to read data from pickle file
    with open(os.path.join(path, f'{file_name}.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data

def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"{func.__name__} execution time: {execution_time:.2f} seconds")
        return result
    return wrapper

# Create logs directory (if it doesn't exist)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)

# Generate log filename
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = os.path.join(log_folder, f'training_{current_time}.log')

def setup_logger():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Create global logger instance
logger = setup_logger()

def read_dat_wav(path, sub_id):
    with open(os.path.join(path, f'{sub_id}.dat'), 'rb') as f:
        pcg_data = np.fromfile(f, dtype=np.int64)  # Adjust dtype according to actual data type
    fs, ecg_data = wavfile.read(os.path.join(path, f'{sub_id}.wav'))
    return ecg_data, pcg_data, fs

def linear_interpolation(data):
    if np.isnan(data).any():
        mask = np.isnan(data)
        x = np.arange(len(data))
        filled_data = np.interp(x, x[~mask], data[~mask])
        return filled_data
    else:
        return data