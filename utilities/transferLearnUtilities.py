"""
This script contains functions required for applying transfer learning.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras import models
from tensorflow.keras.models import Model


def print_breaker(string):
    ts = string
    print(f"{ts:*^79}")
    print()


def load_feature_extractor(path):
    """
    This function is used for loading the feature extractor.
    :param path: path of the trained feature extractor
    :return: feature extractor model
    """
    print(f"loading the feature extractor from {path} ...")
    # load the pre-trained model for feature extraction
    print(f"loading the trained model ...")
    feature_extractor_model = models.load_model(path)
    print("summary of the model")
    feature_extractor_model.summary()
    print_breaker("loading the feature extractor ...")

    flatten_layer_output = feature_extractor_model.get_layer('flatten').output
    # Create a new model that will output the activations from the Flatten layer
    feature_extraction_model = Model(inputs=feature_extractor_model.input, outputs=flatten_layer_output)
    print("summary of the feature extractor ...")
    feature_extraction_model.summary()

    return feature_extraction_model


def extract_features(feature_extractor, data_np, labels_np):
    """
    This function is used for extracting the features of the data present in the dataframe data_df
    :param feature_extractor: feature extractor object
    :param data_np: numpy array containing the data
    :return: numpy array containing extracted features
    """
    features = feature_extractor.predict(data_np)
    print(f"shape of features = {features.shape}")
    print(f"shape of labels = {labels_np.shape}")
    return features, labels_np


def prepare_tr_learn_data(np_file_path):
    """
    This function is used for preparing the data for transfer learning.
    :param np_file_path: path to the numpy file containing the dataset
    :return: x_*, y_*, (y_* + x_*) where * = {train, valid, test}
    """
    np_data = np.load(np_file_path, allow_pickle=True)
    x_df = pd.DataFrame(np_data["x"])
    print(f"shape of the features df = {x_df.shape}")
    y_df = pd.DataFrame(np_data["y"], columns=["label"])
    print(f"shape of the labeled df = {y_df.shape}")
    df = pd.concat([y_df, x_df], axis=1)
    print(f"shape of the merged df = {df.shape}")

    return x_df, y_df, df