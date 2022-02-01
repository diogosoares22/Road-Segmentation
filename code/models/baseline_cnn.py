"""
This file contains the code for the baseline CNN provided with the project, restructured and refactored for
tensorflow version 2.7.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Input
from tensorflow.keras.initializers import TruncatedNormal, Zeros, Constant
from tensorflow.keras.regularizers import L2
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.metrics import BinaryAccuracy
from code.models.model_superclass import ModelSuperclass
from code.helper_functions import f1_m


class BaselineCNN(ModelSuperclass):

    def __init__(self, params, load=False, load_params=None):
        """
        Initialize architecture and pass hyperparameters.
        @param params: Dictionary of parameters to provide to the model.
        Required keys: "seed", "patch_size", "num_channels", "num_labels"
        @param load: Boolean indicating whether to load the model from a h5 file.
        @param load_params: Parameters required for loading the model. Required keys: "path", "version"
        """

        super().__init__(params, load, load_params)

        # Set attributes of the model
        self.name = "Baseline_CNN_provided"
        self.seed = params["seed"]
        self.patch_size = params["patch_size"]
        self.num_channels = params["num_channels"]
        self.num_labels = params["num_labels"]

        if load:
            self.load_from_file(load_params["path"], load_params["version"])
        else:
            self.init_model()

    def init_model(self):
        """
        Initializes the baseline cnn architecture
        """

        # Setting up the layers
        input = Input((self.patch_size, self.patch_size, self.num_channels))  # Input layer
        conv1 = Conv2D(  # First convolution
            filters=32,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            use_bias=True,
            kernel_initializer=TruncatedNormal(stddev=0.1, seed=self.seed),
            bias_initializer=Zeros(),
            data_format='channels_last'
        )(input)
        pool1 = MaxPool2D(  # Max pooling
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        )(conv1)
        conv2 = Conv2D(  # Second convolution
            filters=64,
            kernel_size=5,
            strides=1,
            padding="same",
            activation="relu",
            use_bias=True,
            kernel_initializer=TruncatedNormal(stddev=0.1, seed=self.seed),
            bias_initializer=Zeros(),
            data_format='channels_last'
        )(pool1)
        pool2 = MaxPool2D(  # Max pooling
            pool_size=(2, 2),
            strides=(2, 2),
            padding="same"
        )(conv2)
        flat = Flatten(data_format='channels_last')(pool2)  # Flattening
        dense = Dense(  # Fully connected layer
            units=512,
            activation="relu",
            use_bias=True,
            kernel_initializer=TruncatedNormal(stddev=0.1, seed=self.seed),
            bias_initializer=Constant(value=0.1),
            kernel_regularizer=L2(5e-4),
            bias_regularizer=L2(5e-4)
        )(flat)
        out = Dense(  # Output layer
            units=self.num_labels,
            use_bias=True,
            kernel_initializer=TruncatedNormal(stddev=0.1, seed=self.seed),
            bias_initializer=Constant(value=0.1),
            kernel_regularizer=L2(5e-4),
            bias_regularizer=L2(5e-4),
            activation="sigmoid"
        )(dense)

        # Initialize model
        self.model = Model(inputs=input, outputs=out, name=self.name)

    def train(self, train_data, batch_size=16, learning_rate=0.001, epochs=20, val_data=None):
        """
        Trains the model with the data provided
        @param train_data: Data and labels given through a data loader.
        @param batch_size: Batch size.
        @param learning_rate: Initial learning rate.
        @param epochs: Number of epochs for training.
        @param val_data: If not none, validation data given through a data loader.
        @return: history of the model including metrics
        """

        # Compile model
        lr = ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=len(train_data),
            decay_rate=0.95,
            staircase=True
        )
        optimizer = Adam(
            learning_rate=lr)  # SGD was original optimizer in given code, but for comparison with other models, we use Adam everywhere
        loss_function = BinaryCrossentropy()
        metric = [BinaryAccuracy(), f1_m]

        self.model.compile(optimizer=optimizer, loss=loss_function, metrics=metric)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # Train model
        history = self.model.fit(x=train_data, batch_size=batch_size, epochs=epochs, validation_data=val_data)

        return history

    def evaluate(self, data, labels, batch_size=16):
        """
        Evaluate the model with the data provided and corresponding labels.
        @return: metrics of the evaluation
        """
        return self.model.evaluate(data, labels, batch_size=batch_size)

    def predict(self, data, batch_size=16):
        """
        Predicts an output given data.
        @param data: Correctly sized data
        @param batch_size: batch size of the prediction.
        @return: the predicted labels for the given data.
        """

        return self.model.predict(data, batch_size=batch_size)

