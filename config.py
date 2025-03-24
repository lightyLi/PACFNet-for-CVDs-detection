import os
from datetime import datetime
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')

class ECGNotExistError(Exception):
    pass
class PCGNotExistError(Exception):
    pass
class SignalLengthMismatchError(Exception):
    pass

fs = 2000


# Segmentation parameters
segmentation_method = 'b2b'
beat_num = 1

# Cross validation parameters
kf_n_splits = 5
random_state = 42

# Model and training parameters
kernel_size = 7
layer_n = 64
dropout_rate = 0.5

train_batch_size = 32
test_batch_size = 128
epochs = 100
learning_rate = 0.001

databset_folder = "/root/dataset/physionet_2016_training-a"
processed_folder = "/root/autodl-fs/processed"
csv_file = "/root/dataset/annotations_physionet2016/updated/training-a/REFERENCE_withSQI.csv"
annotation_folder = '/root/dataset/annotations_physionet2016/hand_corrected/training-a_StateAns'
log_folder = "/root/logs"
tensorboard_log_folder = '/root/tf-logs'
models_folder = '/root/models'
training_result_folder = os.path.join('/root/autodl-fs/training_result', f'{segmentation_method}_{current_time}_testing')