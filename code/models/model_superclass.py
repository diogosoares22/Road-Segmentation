import os
import tensorflow as tf
from code.helper_functions import f1_m
"""
This file contains the superclass for all our models which ensure a consistent pattern in our code.
It raises a NotImplementedError for all methods, as it is not intended to be used on its own, and only provides
indications on the methods and parameters required. In each model implementation, these methods must therefore all be
overridden.
The init method provides basic assertion that all parameters required for initialisation are passed and should always
be called in the subclass initialisation.
"""


class ModelSuperclass:

    name = None
    is_categorical = True  # Set to false is model operates pixel-wise

    def __init__(self, params, load, load_params):
        # Assert all parameters required for initialisation were passed
        keys = params.keys()
        req_params = ["seed", "patch_size", "num_channels", "num_labels"]
        for p in req_params:
            assert p in keys, "Parameters must include the key " + p

        # Parameters have the correct type and somewhat valid values
        assert params["seed"] is None or isinstance(params["seed"], int), "Parameter seed passed must be None or of type int"

        intParams = ["patch_size", "num_channels", "num_labels"]
        for intParam in intParams:
            assert isinstance(params[intParam], int), "Parameter " + intParam + " must be of type int"
            assert params[intParam] > 0, "Parameter " + intParam + " must be positive"

        # Assert all parameters required for loading were passed
        if load:
            keys = load_params.keys()
            req_params = ["path", "version"]
            for p in req_params:
                assert p in keys, "Loading parameters must include the key " + p
        # Make sure GPU is set up whenever we instanciate a model
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)

    def init_model(self):
        raise NotImplementedError

    def load_from_file(self, path, version):
        """
        Load model from a h5 file and stores it in self.model
        @param path: path to where all models are saved.
        @param version: version of the model
        @return:
        """
        filename = os.path.join(path, self.name, self.name)
        if version is not None:
            filename += "_" + str(version)
        filename += ".h5"

        self.model = tf.keras.models.load_model(filename, custom_objects={"f1_m" : f1_m})

    def train(self, train_data, batch_size, learning_rate, epochs, val_data=None):
        raise NotImplementedError

    def evaluate(self, data, labels):
        raise NotImplementedError

    def predict(self, data):
        raise NotImplementedError

    def save_to_file(self, path, version=None):
        """
        Saves the model into a h5 file.
        @param path: path to where all models are saved.
        @param version: version of the model
        """
        filename = os.path.join(path, self.name)
        if version is not None:
            filename += "_" + str(version)
        filename += ".h5"

        self.model.save(filename)
