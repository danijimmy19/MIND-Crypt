"""
This script contains the utilities for training and evaluating the models.
"""

import optuna
from optuna.integration import TFKerasPruningCallback

import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "../utilities"))

import tensorflow as tf
tf.random.set_seed(979)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

import json

import pandas as pd
from dataLoaderUtilities import *
from modelZoo import *

from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback, ModelCheckpoint, LearningRateScheduler, TensorBoard

import datetime
import warnings

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix


def evaluate_model_metrics(x_test, y_test, model_file):
    print("Loading and evaluating the model...")
    model = models.load_model(model_file)
    y_pred_prob = model.predict(x_test)  # Get probability predictions
    y_pred = np.argmax(y_pred_prob, axis=1)  # Convert probabilities to class predictions
    y_true = np.argmax(y_test, axis=1)  # Assuming y_test is also one-hot encoded

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if cm.shape == (2, 2):  # binary classification
        tn, fp, fn, tp = cm.ravel()

        # Derived rates
        tpr = tp / (tp + fn) if (tp + fn) != 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) != 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) != 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) != 0 else 0
    else:
        tn = fp = fn = tp = None  # Not applicable for multi-class

    # Compute metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob, multi_class='ovr')  # For binary classification OVR approach is suitable

    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": roc_auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TPR": tpr,
        "TNR": tnr,
        "FPR": fpr,
        "FNR": fnr
    }
    return metrics


def plot_and_save_roc_curve(x_test, y_test, model_file, filepath):
    model = models.load_model(model_file)
    y_pred_prob = model.predict(x_test)  # Get probability predictions
    y_true = np.argmax(y_test, axis=1)  # Assuming y_test is also one-hot encoded

    # Compute ROC curve and ROC area for the positive class
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob[:, 1])
    roc_auc = roc_auc_score(y_true, y_pred_prob[:, 1])

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(filepath))  # Save the figure to a file
    plt.show()
    print(f"ROC curve saved as {filepath}")

def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    """Plots and saves the confusion matrix as a heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()


# Plot Convergence Curves
def plot_convergence_curves(history, save_dir):
    """Plots training and validation accuracy/loss curves."""

    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists

    # Create figure for Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(False)

    # Save Accuracy Plot
    accuracy_plot_path = os.path.join(save_dir, "convergence_accuracy.png")
    plt.savefig(accuracy_plot_path)
    print(f"Accuracy convergence plot saved at: {accuracy_plot_path}")
    plt.close()

    # Create figure for Loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(False)

    # Save Loss Plot
    loss_plot_path = os.path.join(save_dir, "convergence_loss.png")
    plt.savefig(loss_plot_path)
    print(f"Loss convergence plot saved at: {loss_plot_path}")
    plt.close()

    # Create combined figure for both Accuracy & Loss
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy Curve
    axes[0].plot(history.history["accuracy"], label="Train Accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation Accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].set_ylim(0, 1)  # Fix y-axis range
    axes[0].grid(False)

    # Loss Curve
    axes[1].plot(history.history["loss"], label="Train Loss")
    axes[1].plot(history.history["val_loss"], label="Validation Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(False)

    # Save combined plot
    combined_plot_path = os.path.join(save_dir, "convergence_curves.png")
    fig.savefig(combined_plot_path)
    print(f"Combined convergence curves saved at: {combined_plot_path}")

    # plt.show()
    plt.close()


def load_params(json_path):
    """ Load model parameters from a JSON file. """
    with open(json_path, "r") as f:
        return json.load(f)


def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return res


class ReportIntermediates(Callback):
    def __init__(self, trial):
        super().__init__()
        self.trial = trial

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs.get('val_accuracy')
        if val_accuracy is not None:
            self.trial.report(val_accuracy, step=epoch)
            if self.trial.should_prune():
                raise optuna.TrialPruned()

def cyclic_lr(num_epochs, high_lr, low_lr):
    return lambda i: low_lr + ((num_epochs-1) - i % num_epochs) / (num_epochs-1) * (high_lr - low_lr)


class ModelTrainer:
    def __init__(self, params, input_shape, nb_classes, name):
        self.params = params
        self.input_shape = input_shape
        self.nb_classes = nb_classes
        self.name = name

    def train_model(self, x_train, y_train, x_valid, y_valid, nb_classes, model_path):
        if self.name == "bi-lstm-binary":
            model = biLSTMBinary(self.params, self.input_shape, nb_classes, self.name)
        if self.name == "cnn-1d-binary":
            model = cnn1dBinary(self.params, self.input_shape, nb_classes, self.name)
        if self.name == "lstm-binary":
            model = lstmBinary(self.params, self.input_shape, nb_classes, self.name)

        # name_time_stamp = str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f"))
        name_time_stamp = ""
        model_path_ckpt = os.path.join(model_path + "" + name_time_stamp, "checkpoint")
        os.makedirs(model_path_ckpt, exist_ok=True)

        model_checkpoint_file = os.path.join(model_path_ckpt, str(self.name) + "-checkpoint.ckpt.keras")
        checkpoint = ModelCheckpoint(filepath=model_checkpoint_file,
                                     save_weights_only=False,
                                     monitor="val_accuracy",
                                     verbose=2,
                                     save_best_only=True,
                                     mode="max")

        call_backs = [checkpoint]

        learning_rate_schedular = LearningRateScheduler(cyclic_lr(10,0.00001, 0.00979))
        call_backs.append(learning_rate_schedular)

        call_backs.append(EarlyStopping(monitor="val_accuracy", mode="max", patience=5))
        # call_backs.append(ReportIntermediates(trial))
        # call_backs.append(TFKerasPruningCallback(trial, 'val_accuracy'))

        # TensorBoard callback
        log_dir = os.path.join(model_path + "" + name_time_stamp, "logs", "fit")
        os.makedirs(log_dir, exist_ok=True)
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=0)
        call_backs.append(tensorboard_callback)

        print("training model ...")
        history = model.fit(x_train, y_train,
                            batch_size=self.params["batch_size"],
                            epochs=self.params["epochs"],
                            validation_data=(x_valid, y_valid),
                            callbacks=call_backs,
                            shuffle=True)

        trial_model_path = os.path.join(model_path + "" + name_time_stamp, "trial-model")
        os.makedirs(trial_model_path, exist_ok=True)

        model_file = os.path.join(trial_model_path, str(self.name) + "-model.keras")
        model.save(model_file)

        # save the training history of the model to csv file
        history_path = os.path.join(trial_model_path, "history.csv")
        pd.DataFrame.from_dict(history.history).to_csv(history_path, index=False)

        return model_file, history

    def categorical_test_model(self, x_test, y_test, model_path):
        print("evaluating the model on unseen (testing) dataset ...")
        print("loading model from %s" % model_path)
        model = models.load_model(model_path)
        # print("summary of the model")
        # model.summary()
        results = model.evaluate(x_test, y_test, verbose=2, return_dict=True)
        print("Evaluation results:", results)

        # Check if the file exists before attempting to remove it
        trial_model_dir = os.path.join(model_path, "trial-model")
        if os.path.exists(trial_model_dir):
            os.remove(trial_model_dir)
            print(f"{trial_model_dir} has been deleted.")
        else:
            print(f"{trial_model_dir} does not exist.")

        return results["accuracy"], results["loss"]
