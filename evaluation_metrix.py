import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from config import segmentation_method, training_result_folder
from utils import save_data_to_pickle

def evaluate_metrics(model, test_set, fold, file_name_startwith):
    # Get model predictions
    y_pred_proba = model.predict(test_set)
    # Convert probabilities to binary predictions (0 or 1)
    y_pred = np.argmax(y_pred_proba, axis=1)  # Get the class index for each sample
    
    y_test_onehot = np.concatenate([y.numpy() for _, y in test_set])
    y_test = np.argmax(y_test_onehot, axis=1)
    # Save model prediction results to file
    save_data_to_pickle(training_result_folder, f"{file_name_startwith}_fold_{fold}_y_pred_proba", ({'y_test_onehot':y_test_onehot, 'y_test':y_test, 'y_pred_proba':y_pred_proba, 'y_pred':y_pred}))

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Also called recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    auc = plot_roc_curve(y_test, y_pred_proba[:, 1], fold = fold, file_name_startwith=file_name_startwith)
    
    # Print results
    print(f"\n=== Fold {fold} Test Set Evaluation Results ===")
    print(f"Confusion Matrix:")
    print(f"True Negative (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive (TP): {tp}")
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity/Recall: {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC: {auc:.4f}")
    
    confusion_matrix_dict = {
            'tn': tn, 'fp': fp,
            'fn': fn, 'tp': tp,
    }
    plot_fusion_matrix(confusion_matrix_dict, fold = fold,file_name_startwith=file_name_startwith)
    
    # Return metrics dictionary
    return {
        'confusion_matrix': confusion_matrix_dict,
        'metrics': {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'auc': auc,
        }
    }

def plot_fusion_matrix(confusion_matrix_dict, fold=None, file_name_startwith=None):
    tn = confusion_matrix_dict['tn']
    fp = confusion_matrix_dict['fp']
    fn = confusion_matrix_dict['fn']
    tp = confusion_matrix_dict['tp']
    # Generate confusion matrix
    new_confusion_matrix = np.array([[tn, fp], [fn, tp]])
    # Print confusion matrix
    print(f"Confusion Matrix:\n{new_confusion_matrix}")
    # Save confusion matrix to file
    plt.figure(figsize=(10, 8))
    sns.heatmap(new_confusion_matrix, annot=True, fmt="g", cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.savefig(os.path.join(training_result_folder, f"{file_name_startwith}_fold_{fold}_confusion_matrix.png"))
    plt.close()

# Plot ROC curve
def plot_roc_curve(y_true, y_scores, fold=None, file_name_startwith=None):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label='ROC Curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(training_result_folder, f"{file_name_startwith}_fold_{fold}_roc_curve.png"))  # Save image
    plt.close()  # Close image to free memory
    return roc_auc

def get_final_evaluation_metrics(confusion_matrix_dict_list, metrics_list, file_name_startwith):
    # Calculate sample standard deviation
    accuracy_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    f1_score_list = []
    auc_list  = []
    for item in metrics_list:
        accuracy_list.append(item['accuracy'])
        sensitivity_list.append(item['sensitivity'])
        specificity_list.append(item['specificity'])
        precision_list.append(item['precision'])
        f1_score_list.append(item['f1_score'])
        auc_list.append(item['auc'])
    accuracy_std = np.std(accuracy_list, ddof=1)  # Use sample standard deviation
    sensitivity_std = np.std(sensitivity_list, ddof=1)
    specificity_std = np.std(specificity_list, ddof=1)
    precision_std = np.std(precision_list, ddof=1)
    f1_score_std = np.std(f1_score_list, ddof=1)
    auc_std = np.std(auc_list, ddof=1)
    
    tn = 0
    fp = 0
    fn = 0
    tp = 0
    for confusion_matrix_dict in confusion_matrix_dict_list:
        tn += confusion_matrix_dict['tn']
        fp += confusion_matrix_dict['fp']
        fn += confusion_matrix_dict['fn']
        tp += confusion_matrix_dict['tp']

    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Also called recall
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    auc = np.mean(auc_list)
    
    # Print results
    print("\n=== Final Test Set Evaluation Results ===")
    print(f"Confusion Matrix:")
    print(f"True Negative (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")
    print(f"True Positive (TP): {tp}")
    print("\nPerformance Metrics:")
    print(f"Accuracy: {accuracy:.4f} ± {accuracy_std:.4f}")
    print(f"Sensitivity/Recall: {sensitivity:.4f} ± {sensitivity_std:.4f}")
    print(f"Specificity: {specificity:.4f} ± {specificity_std:.4f}")
    print(f"Precision: {precision:.4f} ± {precision_std:.4f}")
    print(f"F1-Score: {f1:.4f} ± {f1_score_std:.4f}")
    print(f"AUC: {auc:.4f} ± {auc_std:.4f}")
    
    final_confusion_matrix = {
            'tn': tn, 'fp': fp,
            'fn': fn, 'tp': tp
    }
    plot_fusion_matrix(final_confusion_matrix, fold= "final", file_name_startwith=file_name_startwith)
    return {
        'confusion_matrix': final_confusion_matrix,
        'metrics': {
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'auc': auc,
            'std': {
                'accuracy': accuracy_std,
                'sensitivity': sensitivity_std,
                'specificity': specificity_std,
                'precision': precision_std,
                'f1_score': f1_score_std,
                'auc': auc_std,
            },
        }
    }

def save_evaluation_metrics(evaluation_metrics_dict, metrics, fold_name, file_name_startwith):
    combined_metrics = {}
    file_path = os.path.join(training_result_folder, f"{file_name_startwith}_evaluation_metrics.json")
    
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            combined_metrics = json.load(file)

    combined_metrics[fold_name] = {
        'evaluation_metrics': evaluation_metrics_dict,
        'metrics': metrics
    }

    with open(file_path, "w") as file:
        json.dump(combined_metrics, file,  default=lambda x: int(x) if isinstance(x, np.integer) else x)
    print(f"Evaluation results saved to file: {file_name_startwith}_evaluation_metrics.json")

# Plot loss and accuracy graph
def plot_training_loss_epoch(history, save_path):
    # Plot loss
    plt.figure(figsize=(6, 4))
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title('Training Loss - Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)  # Save image
    plt.close()  # Close image to free memory
    
def plot_training_accuracy_epoch(history, save_path):
    # Plot loss
    plt.figure(figsize=(6, 4))
    
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.title('Training Accuracy - Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)  # Save image
    plt.close()  # Close image to free memory

def plot_val_loss_epoch(history, save_path):
    # Plot loss
    plt.figure(figsize=(6, 4))

    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Validation Loss - Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)  # Save image
    plt.close()  # Close image to free memory

def plot_val_accuracy_epoch(history, save_path):
    # Plot loss
    plt.figure(figsize=(6, 4))
    
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy - Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)  # Save image
    plt.close()  # Close image to free memory

def plot_loss__epoch(history, save_path):
    # Plot loss
    plt.figure(figsize=(6, 4))
    
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss - Training vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(save_path)  # Save image
    plt.close()  # Close image to free memory

def plot_accuracy_epoch(history, save_path):
    # Plot loss
    plt.figure(figsize=(6, 4))

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy - Training vs Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(save_path)  # Save image
    plt.close()  # Close image to free memory

# Plot learning rate graph
def plot_learning_rate(learning_rates, save_path):
    plt.figure(figsize=(6, 4))
    plt.plot(learning_rates, label='Learning Rate')
    plt.title('Learning Rate - Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.savefig(save_path)  # Save image
    plt.close()  # Close image to free memory

def plot_loss_accuracy_lr_epoch_fig(history, learning_rates, fold, file_name_startwith):
    save_path = os.path.join(training_result_folder, f"{file_name_startwith}_fold_{fold}_")
    plot_training_loss_epoch(history, save_path + "training_loss_epoch.png")
    plot_training_accuracy_epoch(history, save_path + "training_accuracy_epoch.png")
    plot_val_loss_epoch(history, save_path + "val_loss_epoch.png")
    plot_val_accuracy_epoch(history, save_path + "val_accuracy_epoch.png")
    plot_loss__epoch(history, save_path + "loss_epoch.png")
    plot_accuracy_epoch(history, save_path + "accuracy_epoch.png")
    plot_learning_rate(learning_rates, save_path + "learning_rate.png")

def print_final_result(metrics_list, final_metrics):
    # Print all results
    print('**** Final Training Results ****')
    for index, metrics in enumerate(metrics_list):
        accuracy = metrics['accuracy']
        sensitivity = metrics['sensitivity']
        specificity = metrics['specificity']
        precision = metrics['precision']
        f1 = metrics['f1_score']
        auc = metrics['auc']
        
        print(f'Round {index+1} Training:')
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Sensitivity/Recall: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print('==================\n')
 
    final_accuracy = final_metrics['accuracy']
    final_sensitivity = final_metrics['sensitivity']
    final_specificity = final_metrics['specificity']
    final_precision = final_metrics['precision']
    final_f1 = final_metrics['f1_score']
    final_auc = final_metrics['auc']
    accuracy_std = final_metrics['std']['accuracy']
    sensitivity_std = final_metrics['std']['sensitivity']
    specificity_std = final_metrics['std']['specificity']
    precision_std = final_metrics['std']['precision']
    f1_score_std = final_metrics['std']['f1_score']
    auc_std = final_metrics['std']['auc']
    print('\n\n\n================================')
    print('================================')
    print('================================')
    print('================================')
    print(f'**** 5-Fold Cross-Validation Final Results: ****')
    print(f"Accuracy: {final_accuracy:.4f} ± {accuracy_std:.4f}")
    print(f"Sensitivity/Recall: {final_sensitivity:.4f} ± {sensitivity_std:.4f}")
    print(f"Specificity: {final_specificity:.4f} ± {specificity_std:.4f}")
    print(f"Precision: {final_precision:.4f} ± {precision_std:.4f}")
    print(f"F1-Score: {final_f1:.4f} ± {f1_score_std:.4f}")
    print(f"AUC: {final_auc:.4f} ± {auc_std:.4f}")
    print('================================')