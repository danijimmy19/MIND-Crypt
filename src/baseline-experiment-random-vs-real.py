"""
Baseline Experiment: Random-vs-Real. The goal of this experiment is to train DL model to distinguish random string from ciphertext string.

- Class 0: random class
- Class 1: real class

The two messages selected for encryption are selected such that the difference between the two messages is delta.

- P1 = N, Here, N is any random value or 16 bit integer
- P2 = N XOR diff

Further, the encryption of these messages is performed using SPECK32/64 cipher scheme, and the ciphertext generated is represented as binary for ML training.

The size of the input vector to ML classifier is 64 bits where the ciphertext of each message is represented using 2 words. Each word is 16-bits long that represents first and second half of Pi considered.

Input Vector = [w1, w2, w3, w4]; Here, w1, w2 represents P1, and w3, w4 represents P2
Labels = [0, 1]; 0 - random class, 1 - real class

To execute this script, use following command:

python baseline-experiment-random-vs-real.py \
  --num_rounds 15 \
  --depth 10 \
  --num_epochs 200 \
  --bs 5000 \
  --models_dir_path /path/to/save/trained/model \
  --dataset_dir_path /path/to/gohr/random-vs-cipher-dataset \
  --statistics_dir_path /path/to/save/results

"""

import os
import sys
import datetime
import argparse
sys.path.append("../utilities/")

from speckUtilities import *
from modelEvalUtilities import *
import numpy as np
import pandas as pd
from pickle import dump

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (Dense, Conv1D, Input, Reshape, Permute, Add,
                                     Flatten, BatchNormalization, Activation)
from tensorflow.keras.regularizers import l2

import matplotlib.pyplot as plt
from IPython.display import Image

def cyclic_lr(num_epochs, high_lr, low_lr):
    res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
    return res


def make_checkpoint(datei):
    res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
    return res


# make residual tower of convolutional blocks
def make_resnet(num_blocks=2, num_filters=32, num_outputs=1, d1=64, d2=64, word_size=16, ks=3,depth=5,
                reg_param=0.0001, final_activation='sigmoid'):
    # input and preprocessing layers
    inp = Input(shape=(num_blocks * word_size * 2,))
    rs = Reshape((2 * num_blocks, word_size))(inp)
    perm = Permute((2, 1))(rs)

    # add a single residual layer that will expand the data to num_filters channels this is a bit-sliced layer
    conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
    conv0 = BatchNormalization()(conv0)
    conv0 = Activation('relu')(conv0)

    # add residual blocks
    shortcut = conv0

    for i in range(depth):
        conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation('relu')(conv1)
        conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)
        shortcut = Add()([shortcut, conv2])

    # add prediction head
    flat1 = Flatten()(shortcut)
    dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1)
    dense1 = BatchNormalization()(dense1)
    dense1 = Activation('relu')(dense1)
    dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Activation('relu')(dense2)
    out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense2)
    model = Model(inputs=inp, outputs=out)
    return (model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPECK32/64 ML experiment - Random vs Real")
    parser.add_argument('--num_rounds', type=int, required=True)
    parser.add_argument('--depth', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--bs', type=int, default=5000)
    parser.add_argument('--models_dir_path', type=str, required=True)
    parser.add_argument('--dataset_dir_path', type=str, required=True)
    parser.add_argument('--statistics_dir_path', type=str, required=True)
    args = parser.parse_args()

    # create the network
    net = make_resnet(depth=args.depth, reg_param=10 ** -5)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    # generate training and validation data
    x_train, y_train = make_train_data(10 ** 7, args.num_rounds)
    x_valid, y_valid = make_train_data(10 ** 6, args.num_rounds)
    x_test, y_test = make_train_data(10 ** 6, args.num_rounds)

    # saving the generated training, validation, and testing datasets
    np.savez(os.path.join(args.dataset_dir_path, "training-dataset.npz"), x=x_train, y=y_train)
    np.savez(os.path.join(args.dataset_dir_path, "validation-dataset.npz"), x=x_valid, y=y_valid)
    np.savez(os.path.join(args.dataset_dir_path, "testing-dataset.npz"), x=x_test, y=y_test)

    print(f"training, validatio, and testing datasets saved to: \n {args.dataset_dir_path}")

    # set up model checkpoint
    checkpoint_path = os.path.join(args.models_dir_path,
                                   "best-" + str(args.num_rounds) + "-depth" + str(args.depth) + "-checkpoint.h5")
    check = make_checkpoint(checkpoint_path)

    log_dir = os.path.join(args.models_dir_path, "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # create learn rate schedule
    lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))

    print(f"training the model for {args.num_epochs} epochs ...")
    # train and evaluate
    history = net.fit(x_train, y_train,
                      epochs=args.num_epochs,
                      batch_size=args.bs,
                      validation_data=(x_valid, y_valid),
                      callbacks=[lr, check, tensorboard_callback])

    # save the training history of the model to csv file
    history_path = os.path.join(args.models_dir_path, "history.csv")
    pd.DataFrame.from_dict(history.history).to_csv(history_path, index=False)

    # path to save the convergence curves
    figure_path = os.path.join(args.models_dir_path, "convergence-curves.png")
    plot_convergence_curves(args.models_dir_path, figure_path)

    testing_results_path = os.path.join(args.statistics_dir_path)
    # perform evaluation on the testing dataset with the trained model
    generate_evaluation_statistics(x_test, y_test, args.model_file_path, "testing", testing_results_path)

    print('Done!')