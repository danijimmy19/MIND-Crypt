"""
This script contains utility function required for model evaluation.
"""

import os

from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, auc, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import itertools

import matplotlib.pyplot as plt
from modelZooOLd import AdaptivePaddingAdd


def plot_convergence_curves(model_path, figure_path):
    """
    This function is used for plotting the convergence curves for training and validation.
    :param model_path: path to the directory containing history.csv file
    :param figure_path: path to save the figure.
    """

    # convergence curves
    history_df = os.path.join(model_path, "history.csv")
    history_df = pd.read_csv(history_df)

    # Create subplots
    plt.figure(figsize=(12, 6))

    epochs = range(1, len(history_df) + 1)

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_df['loss'], label='Training Loss')
    plt.plot(epochs, history_df['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.ylim(0, 1)
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_df['acc'], label='Training Accuracy')
    plt.plot(epochs, history_df['val_acc'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()

    # Display the plots
    plt.tight_layout()
    plt.savefig(figure_path, dpi=100)
    plt.close()


def plot_roc_curve(y_test, y_pred_prob, results_path):
    # Calculate False Positive Rate and True Positive Rate
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")

    # Specify the file path for saving the ROC curve plot
    roc_auc_fig_path = os.path.join(results_path)
    plt.savefig(roc_auc_fig_path, dpi=200)
    plt.close()  # Close the plot to avoid displaying it inline if running in a notebook


def plot_confusion_matrix(conf_matrix, target_names, title, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.get_cmap('Wistia'))
    plt.title(title)
    # plt.colorbar()

    tick_marks = range(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)

    thresh = conf_matrix.max() / 2.0
    for i, j in itertools.product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], 'd'), horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(save_path)
    plt.show()


def generate_evaluation_statistics(x, y, model_path, label, results_path):
    """
    This function is used for evaluating a model for multiclass classification.
    :param x_test: Features of the testing dataset
    :param y_test: One-hot encoded labels corresponding to the input features
    :param model_path: Path to load the trained model
    :param label: To distinguish training, validation, and testing file names
    :param results_path: Path to save the evaluation results
    :return: None
    """
    print("Loading the model for evaluation ...")
    model = models.load_model(model_path)

    # Perform predictions on the test data
    y_pred_prob = model.predict(x)  # Use flatten() for binary classification output
    y_pred = (y_pred_prob > 0.5).astype(int)

    class_names = ["Random (class-0)", "Read (class-1)"]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    classification_rep = classification_report(y, y_pred, target_names=['Class 0', 'Class 1'],
                                               output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)
    precision = precision_score(y, y_pred, average='binary')
    recall = recall_score(y, y_pred, average='binary')
    f1 = f1_score(y, y_pred, average='binary')
    roc_auc = roc_auc_score(y, y_pred_prob)  # Use y_pred_prob for ROC AUC as it requires probabilities
    tn, fp, fn, tp = conf_matrix.ravel()

    # Calculate additional metrics
    tpr = tp / (tp + fn)  # True Positive Rate
    tnr = tn / (tn + fp)  # True Negative Rate
    fpr = fp / (tn + fp)  # False Positive Rate
    fnr = fn / (tp + fn)  # False Negative Rate

    # path to the results
    results_path = os.path.join(results_path, label + "-statistics")
    os.makedirs(results_path, exist_ok=True)

    # Convert the classification report to a pandas DataFrame and save it
    classification_rep_df = pd.DataFrame.from_dict(classification_rep)
    classification_rep_path = os.path.join(results_path, "classification-report.csv")
    classification_rep_df.to_csv(classification_rep_path, index=False)

    # Log and save the evaluation results
    print("Accuracy: {}".format(accuracy))
    print("Classification Report saved to {}".format(classification_rep_path))
    print("Confusion Matrix:\n", conf_matrix)

    # Optionally, plot and save the confusion matrix and ROC curve
    # Implement or call your plotting functions if needed, for example:
    plot_confusion_matrix(conf_matrix, target_names=['Class 0', 'Class 1'], title='Confusion matrix',
                          save_path=os.path.join(results_path, "conf_matrix.png"))
    plot_roc_curve(y, y_pred_prob, os.path.join(results_path, "roc_curve.png"))

    # Save evaluation metrics to a CSV file
    metrics = {
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1-score': [f1],
        'roc-auc': [roc_auc],
        'tp': [tp],
        'fp': [fp],
        'tn': [tn],
        'fn': [fn],
        'tpr': [tpr],
        'tnr': [tnr],
        'fpr': [fpr],
        'fnr': [fnr]
        }
    metrics_df = pd.DataFrame(metrics)
    csv_file_path = os.path.join(results_path, "stats.csv")
    metrics_df.to_csv(csv_file_path, index=False)

    print(f"Statistics of the testing dataset are saved at location {csv_file_path}.")

    return metrics_df
