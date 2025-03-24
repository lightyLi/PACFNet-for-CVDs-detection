import os
import numpy as np
import matplotlib.pyplot as plt
from config import processed_folder, fs

def plot_training_curves(index, history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    ax1.plot(history.history['loss'], label='Training Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy curves
    ax2.plot(history.history['accuracy'], label='Training Accuracy')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f'Training_{index}.jpg')
    return fig

def plot_signals_comparison(signal1, signal2, tmin=0, tmax=-1, labels=None, title=None, sample_rate=fs):
    """
    Compare and display two signals
    
    Parameters:
    signal1: First signal
    signal2: Second signal
    labels: tuple, Signal labels (label1, label2)
    title: str, Chart title
    sample_rate: int, Sampling rate for generating time axis
    """
    if labels is None:
        labels = ('Original Signal', 'Reconstructed Signal')
    if title is None:
        title = 'Signal Comparison'
    
    # Get the range to display
    point_index_min = tmin * fs
    if tmax != -1:
        point_index_max = tmax * fs
    else:
        point_index_max = len(signal1)
    signal1 = signal1[point_index_min:point_index_max]
    signal2 = signal2[point_index_min:point_index_max]
    
    # Generate time axis (unit: seconds)
    t = np.arange(len(signal1)) / sample_rate
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot signals
    plt.plot(t, signal1, 'b-', label=labels[0], linewidth=1.5, alpha=0.8)
    plt.plot(t, signal2, 'r--', label=labels[1], linewidth=1.5, alpha=0.8)
    
    # Set figure properties
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(processed_folder, f'{title}.jpg'))
    plt.show()