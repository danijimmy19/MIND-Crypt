"""
This script contains the architectures of the model.
"""
from tensorflow.python.keras.engine.input_layer import InputLayer

"""
This python script contains different model architectures that are used for implementing the attack.
"""

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv1D, AveragePooling1D, Input, MaxPooling1D, BatchNormalization
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import optimizers, layers
from tensorflow.keras.layers import Flatten, Dense, Activation, Dropout, ELU, GlobalAveragePooling1D
from tensorflow.keras.layers import Bidirectional, LSTM, Embedding, Attention, Add, Layer, Reshape, Concatenate
from tensorflow.keras.layers import Input, MultiHeadAttention, LayerNormalization, GlobalMaxPooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.layers import ZeroPadding1D
from tensorflow.keras.layers import InputLayer, Reshape, Permute

from tensorflow.keras import models
from tensorflow.keras.models import Model


def print_breaker(string):
    ts = string
    print(f"{ts:*^79}")
    print()


def lstmBinary(params, input_shape, nb_classes, name):
    """
        This function builds and compiles the LSTM model architecture.
        :param params: dictionary containing parameters of the model.
        :param input_shape: input shape of the model. (max. password length)
        :param name: name of the classification model.
        :param nb_classes: number of classes in the data.
        :return: compiled model for training.
        """
    print(f"Building and compiling {name} model architecture.")
    model = Sequential(name=name)
    model.add(InputLayer(input_shape=input_shape, name="input-layer"))
    model.add(Reshape((2 * 1, 16)))  # Adjust if needed based on your data
    model.add(Permute((2, 1)))

    for i in range(params["n_lstm_layers"]):
        model.add(LSTM(params["lstm_cells"], return_sequences=True, name=f"lstm-layer-{i + 1}"))

    model.add(LSTM(params["lstm_cells"], return_sequences=False, name=f"lstm-layer-{params['n_lstm_layers'] + 1}"))
    model.add(Dropout(params["drop_rate"], name="dropout-layer-lstm"))

    for i in range(params["n_dense_layers"]):
        model.add(Dense(params["fc_neurons"], activation=params["fc_act"], name=f"dense-layer-{i + 1}"))
        model.add(Dropout(params["drop_rate_fc"], name=f"dropout-layer-dense-{i + 1}"))

    model.add(Dense(nb_classes, activation='softmax', name="output-layer"))

    # Optimizer selection
    optimizer_name = params.get('optimizer', 'Adam')
    learning_rate = params.get('learning_rate', 0.001)

    if optimizer_name == "Adamax":
        optimizer = optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_name == "Adagrad":
        optimizer = optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_name == "Nadam":
        optimizer = optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_name == "SGD":
        optimizer = optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    metrics = [
        "accuracy",
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def biLSTMBinary(params, input_shape, nb_classes, name):
    """
    This function builds and compiles the biLSTM model architecture.
    :param params: dictionary containing parameters of the model.
    :param input_shape: input shape of the model. (max. password length)
    :param name: name of the classification model.
    :param nb_classes: number of classes in the data.
    :return: compiled model for training.
    """
    print(f"building and compiling {name} model architecture.")
    model = Sequential(name=name)
    model.add(InputLayer(input_shape=input_shape, name="input-layer"))
    model.add(Reshape((2 * 1, 16)))  # 2* num_blocks, word_size
    model.add(Permute((2, 1)))

    for i in range(params["n_lstm_layers"]):
        model.add(Bidirectional(LSTM(params["lstm_cells"], return_sequences=True, name=f"bilstm-layer-{i + 1}")))

    model.add(Bidirectional(
        LSTM(params["lstm_cells"], return_sequences=False, name=f"bilstm-layer-{params['n_lstm_layers'] + 1}")))
    model.add(Dropout(params["drop_rate"], name="dropout-layer-lstm"))

    for i in range(params["n_dense_layers"]):
        model.add(Dense(params["fc_neurons"], activation=params["fc_act"], name=f"dense-layer-{i + 1}"))
        model.add(Dropout(params["drop_rate_fc"], name=f"dropout-layer-dense-{i + 1}"))

    model.add(Dense(nb_classes, activation='softmax', name="output-layer"))

    if params['optimizer'] == "Adamax":
        optimizer = optimizers.Adamax(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999,
                                      epsilon=1e-07)
    if params['optimizer'] == "Adam":
        optimizer = optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                    amsgrad=False)
    if params['optimizer'] == "RMSprop":
        optimizer = optimizers.RMSprop(learning_rate=params['learning_rate'], rho=0.9, momentum=0.0, epsilon=1e-07,
                                       centered=False)

    if params['optimizer'] == "Adagrad":
        optimizer = optimizers.Adagrad(learning_rate=params['learning_rate'], initial_accumulator_value=0.1,
                                       epsilon=1e-07)
    if params['optimizer'] == "Nadam":
        optimizer = optimizers.Nadam(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    if params['optimizer'] == "SGD":
        optimizer = optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.0, nesterov=False)

    metrics = [
        "accuracy",
        tf.keras.metrics.FalseNegatives(name="fn"),
        tf.keras.metrics.FalsePositives(name="fp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=metrics)

    return model


def cnn1dBinary(params, input_shape, nb_classes, name):
    """
    This function builds and compiles the CNN model with DF architecture.
    :param params: parameters of the model.
    :param name: name of the model.
    :param nb_classes: number of classes in the data.
    :return: compiled model for training.
    """
    print("building and compiling  CNN model architecture.")

    model = Sequential(name=name)
    model.add(InputLayer(input_shape=input_shape, name="input-layer"))
    model.add(Reshape((2 * 1, 16)))  # 2* num_blocks, word_size
    model.add(Permute((2, 1)))

    # Input
    model.add(Conv1D(filters=params['filter_nums'], kernel_size=params['kernel_size'],
                     strides=params['conv_stride_size'], padding='same',
                     name='block1_conv1'))
    model.add(BatchNormalization(axis=-1))
    model.add(ELU(alpha=1.0, name='block1_adv_act1'))
    model.add(Conv1D(filters=params['filter_nums'], kernel_size=params['kernel_size'],
                     strides=params['conv_stride_size'], padding='same',
                     name='block1_conv2'))
    model.add(BatchNormalization(axis=-1))
    model.add(ELU(alpha=1.0, name='block1_adv_act2'))
    model.add(MaxPooling1D(pool_size=params['pool_size'], strides=params['conv_stride_size'],
                           padding='same', name='block1_pool'))
    model.add(Dropout(rate=params['drop_rate'], name='block1_dropout'))

    # Add convolutional blocks dynamically
    for i in range(params['n_conv_layers']):
        model.add(Conv1D(filters=params['filter_nums'], kernel_size=params['kernel_size'],
                         strides=params['conv_stride_size'], padding='same',
                         name=f'conv_{i + 1}'))
        model.add(BatchNormalization())
        model.add(Activation(params['activation_fn']))
        model.add(MaxPooling1D(pool_size=params['pool_size'], strides=params['pool_stride_size'],
                               padding='same', name=f'pool_{i + 1}'))
        model.add(Dropout(rate=params['drop_rate']))

    model.add(Flatten(name='flatten'))

    # Add dense layers dynamically
    for i in range(params['n_dense_layers']):
        model.add(Dense(params['fc_neurons'], kernel_initializer=glorot_uniform(seed=0), name=f'fc_{i + 1}'))
        model.add(BatchNormalization())
        model.add(Activation(params[f'fc_act']))
        model.add(Dropout(rate=params[f'drop_rate_fc']))

    # Output layer
    model.add(Dense(nb_classes, kernel_initializer=glorot_uniform(seed=0), name='output'))
    model.add(Activation("softmax", name="pred_layer"))

    if params['optimizer'] == "Adamax":
        optimizer = optimizers.Adamax(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999,
                                      epsilon=1e-07)
    if params['optimizer'] == "Adam":
        optimizer = optimizers.Adam(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                    amsgrad=False)
    if params['optimizer'] == "RMSprop":
        optimizer = optimizers.RMSprop(learning_rate=params['learning_rate'], rho=0.9, momentum=0.0, epsilon=1e-07,
                                       centered=False)

    if params['optimizer'] == "Adagrad":
        optimizer = optimizers.Adagrad(learning_rate=params['learning_rate'], initial_accumulator_value=0.1,
                                       epsilon=1e-07)
    if params['optimizer'] == "Nadam":
        optimizer = optimizers.Nadam(learning_rate=params['learning_rate'], beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    if params['optimizer'] == "SGD":
        optimizer = optimizers.SGD(learning_rate=params['learning_rate'], momentum=0.0, nesterov=False)

    print('compiling model...')
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
                  optimizer=optimizer,
                  metrics=["accuracy"])

    return model