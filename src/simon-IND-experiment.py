"""
Experiment A: Enc(P1) vs Enc(P2).
The two messages P1 and P2 are encrypted using SIMON32/64-CBC mode of encryption.

- Class 0: Enc(IV1i XOR P1)
- Class 1: Enc(IV2i XOR P2)

The two messages selected for encryption are as follows:
   1. P1 = 0 (32-bit)
   2. P2 = 1 (32-bit)

python train_model.py \
    --cipher_name speck-32-64 \
    --exp_name dl-models-binary-level \
    --model_name_1d bi-lstm-binary \
    --dataset_name exp-b-speck-32-64-cbc-enc-0-vs-enc-1 \
    --train_data_path ../data-speck-32-64-enc-0-vs-enc-1/train-data.npz \
    --valid_data_path ../data-speck-32-64-enc-0-vs-enc-1/valid-data.npz \
    --test_data_path ../data-speck-32-64-enc-0-vs-enc-1/test-data.npz

"""

import os, sys, re
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "utilities"))

import argparse
import pandas as pd

from modelTrainerUtilities import *
from tensorflow.keras.utils import to_categorical


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment settings for cipher classification")
    parser.add_argument('--cipher_name', type=str, default="simon-32-64", choices=['simon-32-64'],
                        help='Cipher type to use (simon-32-64)')
    parser.add_argument('--rounds', type=int, default=32,
                        help='Number of cipher rounds')
    parser.add_argument('--dataset_name', type=str, default="exp-a-simon-32-64-cbc-enc-0-vs-enc-1",
                        help='Name of the dataset directory')
    parser.add_argument('--exp_name', type=str, default="dl-models-binary-level",
                        help='Experiment name, e.g. dl-models-binary-level')
    parser.add_argument('--models_root', type=str, default="../trained-models/",
                        help="Root path for saving trained models")
    parser.add_argument('--model_name_1d', type=str, default="cnn-1d-binary",
                        choices=['bi-lstm-binary', 'cnn-bi-lstm-binary', 'bi-lstm-cnn-binary', 'cnn-1d-binary', 'lstm-binary'],
                        help='Name of the 1D model architecture')
    parser.add_argument('--train_data_path', type=str,
                        default="../data-simon-32-64-enc-0-vs-enc-1/train-data.npz",
                        help='Path to training data')
    parser.add_argument('--valid_data_path', type=str,
                        default="../data-simon-32-64-enc-0-vs-enc-1/valid-data.npz",
                        help='Path to validation data')
    parser.add_argument('--test_data_path', type=str,
                        default="../data-simon-32-64-enc-0-vs-enc-1/test-data.npz",
                        help='Path to test data')

    args = parser.parse_args()

    def append_rounds(path: str, rounds: int) -> str:
        base, ext = os.path.splitext(path)
        if ext != ".npz":
            return path
        # avoid double-suffix, e.g., *-32.npz
        if re.search(rf"-{rounds}\.npz$", path):
            return path
        return f"{base}-{rounds}{ext}"

    args.train_data_path = append_rounds(args.train_data_path, args.rounds)
    args.valid_data_path = append_rounds(args.valid_data_path, args.rounds)
    args.test_data_path = append_rounds(args.test_data_path, args.rounds)

    return args

if __name__ == "__main__":

    args = parse_args()

    cipher_name = args.cipher_name
    rounds = args.rounds
    dataset_name = args.dataset_name
    # dl-model-char-level: This experiments uses text representation of the data for classification
    exp_name = args.exp_name
    # choices: bi-lstm-binary, cnn-bi-lstm-binary, bi-lstm-cnn-binary, cnn-1d-binary, lstm-binary
    model_name_1d = args.model_name_1d

    # load the dataset
    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path
    test_data_path = args.test_data_path

    # path to the model hyperparameters
    params_path = os.path.join("../dl-models-binary-level/optuna-db/", exp_name, cipher_name, model_name_1d, "best_trial.json")
    print(f"loading parameters from {params_path}")
    # load the params
    params = load_params(params_path)["Params"]
    print(f"parameters are: {params}")

    # path to save the trained models
    models_path = os.path.join(args.models_root, dataset_name, model_name_1d)
    os.makedirs(models_path, exist_ok=True)

    print(f"Experiment: {exp_name} | Cipher: {cipher_name} | Rounds: {rounds} | Model: {model_name_1d}")
    print(f"Train: {train_data_path}\nValid: {valid_data_path}\nTest:  {test_data_path}\nOut:   {models_path}")

    print("loading training dataset ...")
    x_train, y_train, nb_classes = load_data_simon(train_data_path, num_samples_per_class=None,
                                                   num_features=None)
    print("loading validation dataset ...")
    x_valid, y_valid, nb_classes = load_data_simon(valid_data_path, num_samples_per_class=None,
                                                   num_features=None)
    print("loading testing dataset ...")
    x_test, y_test, nb_classes = load_data_simon(test_data_path, num_samples_per_class=None, num_features=None)

    # Print shapes of data
    print(f"Original training shape: {x_train.shape}")
    print(f"Original validation shape: {x_valid.shape}")
    print(f"Original test shape: {x_test.shape}")

    # Reshape not needed as text sequences are already padded properly
    input_length = x_train.shape[1]  # Should be same as max_seq_len
    input_shape = (input_length, 1)

    print(f"Input vector length = {input_length}")

    # One-hot encode labels
    y_train = to_categorical(y_train, num_classes=nb_classes)
    y_valid = to_categorical(y_valid, num_classes=nb_classes)
    y_test = to_categorical(y_test, num_classes=nb_classes)

    # Print final label shapes
    print(f"Shape of training labels = {y_train.shape}")
    print(f"Shape of validation labels = {y_valid.shape}")
    print(f"Shape of testing labels = {y_test.shape}")

    # initialize the model tuning class
    ONE_D_MODELS = ModelTrainer(params, input_shape, nb_classes, model_name_1d)

    # training a model with selected parameters
    print(f"training a {model_name_1d} model ...")
    trained_model_path, history = ONE_D_MODELS.train_model(x_train, y_train, x_valid, y_valid, nb_classes, models_path)

    # Get the directory where the model is stored
    model_save_dir = os.path.dirname(trained_model_path)
    # Call the function to plot curves
    plot_convergence_curves(history, model_save_dir)

    print(f"Generating statistics for the training dataset ...")
    training_stats = evaluate_model_metrics(x_train, y_train, trained_model_path)
    print(f"Generating statistics for the validation dataset ...")
    validation_stats = evaluate_model_metrics(x_valid, y_valid, trained_model_path)
    print(f"Generating statistics for the testing dataset ...")
    testing_stats = evaluate_model_metrics(x_test, y_test, trained_model_path)

    # Create a DataFrame
    stats_df = pd.DataFrame([training_stats, validation_stats, testing_stats],
                            index=["Training", "Validation", "Testing"])
    stats_df.columns = training_stats.keys()

    # Save DataFrame to CSV
    stats_csv_path = os.path.join(model_save_dir, "stats-training-validation-testing.csv")
    stats_df.to_csv(stats_csv_path)

    # testing a model
    accuracy, loss = ONE_D_MODELS.categorical_test_model(x_test, y_test, trained_model_path)
    print(f"Test Accuracy: {accuracy:.6f} | Test Loss: {loss:.6f}")
    print("Done!")