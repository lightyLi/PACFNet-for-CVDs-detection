import os
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
import multiprocessing
from tensorflow.keras.callbacks import Callback
import keras.backend as K

from utils import load_data_from_pickle, save_data_to_pickle
from plot_fig import plot_training_curves
from config import processed_folder, tensorboard_log_folder, models_folder, segmentation_method, training_result_folder
from config import kf_n_splits, train_batch_size, test_batch_size, learning_rate, epochs, current_time
from model import dual_stream_model
from evaluation_metrix import (evaluate_metrics, get_final_evaluation_metrics, print_final_result,
                               save_evaluation_metrics, plot_loss_accuracy_lr_epoch_fig)

def generate_train_test_data(GPU_num, val_split=0.1):
    def _calculate_class_weight(y_train):
        n_pos = np.sum(y_train[:, 1])  # Number of positive samples
        n_neg = np.sum(y_train[:, 0])  # Number of negative samples
        total = n_pos + n_neg
        class_weight = {
            0: total / (2 * n_neg),
            1: total / (2 * n_pos)
        }
        return class_weight

    def _create_dataset(ecg, pcg, y, batch_size, is_training=False):
        # Create dataset
        dataset = tf.data.Dataset.from_tensor_slices(((ecg, pcg), y))
        # Set sharding strategy
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        dataset = dataset.with_options(options)

        if is_training:
            # Shuffle training dataset
            dataset = dataset.shuffle(buffer_size=13000)
            print("running")

        # Batch processing
        dataset = dataset.batch(batch_size*GPU_num)
        # Prefetch data
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
        
    for i in range(1, kf_n_splits+1):
        ecg_X_train_data, ecg_X_test_data, pcg_X_train_data, pcg_X_test_data, y_train_data, y_test_data = load_data_from_pickle(processed_folder, f'{segmentation_method}_data_fold_{i}')
        y_train_data = tf.keras.utils.to_categorical(y_train_data, num_classes=2)
        y_test_data = tf.keras.utils.to_categorical(y_test_data, num_classes=2)
        class_weight = _calculate_class_weight(ecg_X_train_data)
       yield _create_dataset(ecg_X_train_data, pcg_X_train_data, y_train_data, train_batch_size, is_training=True), _create_dataset(ecg_X_test_data, pcg_X_test_data, y_test_data, test_batch_size), class_weight


def multi_category_focal_loss(class_weight, gamma=2.0):
    """
    Improved multi-category Focal Loss
    - Enhanced numerical stability
    - Added batch dimension normalization
    """
    def focal_loss(y_true, y_pred):
        # Convert weights to tensor
        weight_matrix = tf.zeros_like(y_true)
        for cls_id, weight in class_weight.items():
            weight_matrix = weight_matrix + y_true[:, cls_id:cls_id+1] * weight  
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate cross entropy
        cross_entropy = -y_true * K.log(y_pred)
        # Focal term
        probs = tf.where(y_true > 0, y_pred, 1 - y_pred)
        focal_weight = K.pow(1 - probs, gamma)
        # Apply weights
        loss = weight_matrix * focal_weight * cross_entropy
        # Normalization
        normalizer = K.sum(y_true, axis=-1)
        normalizer = K.maximum(normalizer, 1)  # Prevent division by zero
        loss = K.sum(loss, axis=-1) / normalizer
        # Average across batch dimension
        return K.mean(loss)
    return focal_loss
        
def train_model(GPU_num, file_name_startwith):
    histories = []
    confusion_matrix_list = []
    metrics_list = []
    lr_list = []
    for fold, (train_set, test_set, class_weight) in enumerate(generate_train_test_data(GPU_num), 1):
        print(f"Training fold {fold}/{kf_n_splits}...")

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',  # Monitor training accuracy
            factor=0.1,         # Reduce learning rate by half
            patience=5,         # Reduce learning rate if no improvement for 5 epochs
            verbose=1,          # Print learning rate change information
            min_lr=1e-6        # Minimum learning rate
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,        # Stop training if no improvement for 20 epochs
            restore_best_weights=True,
            verbose=1
        )
        tensorboard_callback = TensorBoard(
            log_dir=tensorboard_log_folder,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch',
            profile_batch=0,  # Disable profiler
        )
        # checkpoint_callback = ModelCheckpoint(
        #     filepath=os.path.join(models_folder, f'{segmentation_method}_periodic_model_fold{fold}.h5'),
        #     monitor='val_loss',
        #     mode='min',
        #     save_best_only=True,
        #     save_weights_only=False,
        #     verbose=0
        # )
        class LearningRateLogger(Callback):
            def __init__(self):
                super(LearningRateLogger, self).__init__()
                self.learning_rates = []
            def on_epoch_end(self, epoch, logs=None):
                lr = self.model.optimizer.learning_rate.numpy()  # Get current learning rate
                self.learning_rates.append(lr)
        lr_logger = LearningRateLogger()
        
        # Initialize model
        print('Using categorical_crossentropy')
        with strategy.scope():
            model = dual_stream_model()
            model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=5.0),
                        loss='categorical_crossentropy', 
                        # loss=multi_category_focal_loss(class_weight, gamma=2.0),
                        metrics=['accuracy'])
        # Train current fold
        train_history = model.fit(train_set,
                validation_data=test_set,
                epochs=epochs,
                class_weight=class_weight,
                callbacks=[early_stopping, reduce_lr, tensorboard_callback, lr_logger],
                verbose=1)
        learning_rates = lr_logger.learning_rates
        lr_list.append(learning_rates)
        
        # Evaluate current fold
        testing_metric_result = evaluate_metrics(model, test_set, fold, file_name_startwith)
        
        confusion_matrix_list.append(testing_metric_result['confusion_matrix'])
        metrics_list.append(testing_metric_result['metrics'])
        histories.append(train_history)
    
        plot_loss_accuracy_lr_epoch_fig(train_history, learning_rates, fold, file_name_startwith)
        
        # Save model and related data after training
        model.save(os.path.join(training_result_folder, f"{file_name_startwith}_fold_{fold}_model.h5"))
        save_data_to_pickle(training_result_folder, f"{file_name_startwith}_fold_{fold}_learning_rate", lr_list)
        save_evaluation_metrics(testing_metric_result['confusion_matrix'], testing_metric_result['metrics'], f'fold_{fold}', file_name_startwith)  # Save evaluation results after each training round

    return histories, confusion_matrix_list, metrics_list

if __name__ == '__main__':
    # # Ensure the following folders exist
    # if not os.path.exists(processed_folder):
    #     os.makedirs(processed_folder)
    # if not os.path.exists(tensorboard_log_folder):
    #     os.makedirs(tensorboard_log_folder)
    # if not os.path.exists(models_folder):
    #     os.makedirs(models_folder)
    # if not os.path.exists(training_result_folder):
    #     os.makedirs(training_result_folder)
    
    # Determine file name prefix for saving
    file_name_startwith = f'{segmentation_method}_{current_time}'

    # Multi-GPU parallel training
    multiprocessing.set_start_method('spawn', force=True)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Available GPUs: {len(gpus)}")
    # Create distributed strategy
    strategy = tf.distribute.MirroredStrategy()
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    GPU_num = tf.distribute.MirroredStrategy().num_replicas_in_sync

    print(f"Training: {file_name_startwith}, Initial LR is {learning_rate}.")
    histories, confusion_matrix_list, metrics_list = train_model(GPU_num, file_name_startwith)

    save_data_to_pickle(training_result_folder, f'{file_name_startwith}_train_history', histories)
    final_metric_result = get_final_evaluation_metrics(confusion_matrix_list, metrics_list, file_name_startwith)
    final_confusion_matrix = final_metric_result['confusion_matrix']
    final_metrics = final_metric_result['metrics']
    save_evaluation_metrics(final_confusion_matrix, final_metrics, 'final_result' ,file_name_startwith)
    
    # Print final results
    print_final_result(metrics_list, final_metrics)