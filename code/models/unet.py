import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import matplotlib.pyplot as plt

from code.helper_functions import f1_m
from code.models.model_superclass import ModelSuperclass


# Heavily inspired by https://medium.com/geekculture/u-net-implementation-from-scratch-using-tensorflow-b4342266e406


class Unet(ModelSuperclass):

    def __init__(self, params, load=False, load_params=None, name="Unet"):
        """
        Initialize architecture and pass hyperparameters.
        @param params: Dictionary of parameters to provide to the model.
        Required keys: "seed", "patch_size", "num_channels", "num_labels"
        @param load: Boolean indicating whether to load the model from a h5 file.
        @param load_params: Parameters required for loading the model. Required keys: "path", "version"
        """
        super().__init__(params, load, load_params)
        self.name = name
        self.is_categorical = False
        self.seed = params["seed"]
        self.patch_size = params["patch_size"]
        self.num_channels = params["num_channels"]
        self.num_labels = params["num_labels"]

        if load:
            self.load_from_file(load_params["path"], load_params["version"])
        else:
            self.init_model()

    def encoder_mini_block(self, input, n_filters=32, dropout_prob=0.3, max_pooling=True):
        """
        Encoder block consisting of two convolutions with specified number of filters followed by a dropout and max
        pooling layer if indicated .

        @param input: The previous layer of the model.
        @param n_filters: Number of filters referring to the dimensionality of the current layers
        @param dropout_prob: Rate of the dropout layer.
        @param max_pooling: Whether we apply a max pooling layer or not.
        @return: the final layer corresponding to the end of the block, same layer without max pooling
        """
        conv = tf.keras.layers.Conv2D(n_filters,
                                      kernel_size=3,  # filter size
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='HeNormal')(input)
        conv = tf.keras.layers.Conv2D(n_filters,
                                      kernel_size=3,  # filter size
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='HeNormal')(conv)

        conv = tf.keras.layers.BatchNormalization()(conv, training=False)
        if dropout_prob > 0:
            conv = tf.keras.layers.Dropout(dropout_prob)(conv)
        if max_pooling:
            next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv)
        else:
            next_layer = conv
        skip_connection = conv
        return next_layer, skip_connection

    def decoder_mini_block(self, prev_layer_input, skip_layer_input, n_filters=32):
        """
        Decoder block consisting of a transposed convolution, followed by a concatenation with the skip layer and two
        convolutions with specified number of filters.

        @param prev_layer_input: The previous layer of the model.
        @param skip_layer_input: corresponding layer from the encoder path.
        @param n_filters: Number of filters referring to the dimensionality of the current layers
        @return: the final layer corresponding to the end of the block, same layer without max pooling
        """
        up = tf.keras.layers.Conv2DTranspose(n_filters,
                                             kernel_size=(3, 3),
                                             strides=(2, 2),
                                             padding='same')(prev_layer_input)
        merge = tf.keras.layers.Concatenate(axis=3)([up, skip_layer_input])
        conv = tf.keras.layers.Conv2D(n_filters,
                                      kernel_size=3,
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='HeNormal')(merge)
        conv = tf.keras.layers.Conv2D(n_filters,
                                      kernel_size=3,
                                      activation='relu',
                                      padding='same',
                                      kernel_initializer='HeNormal')(conv)
        return conv

    def init_model(self, n_filters=16):
        """
        Initializes the Unet architecture, corresponding to Unet defined in the original paper: https://rdcu.be/cDIWc
        """
        inputs = tf.keras.layers.Input((self.patch_size, self.patch_size, self.num_channels))
        cblock1 = self.encoder_mini_block(inputs, n_filters, dropout_prob=0, max_pooling=True)
        cblock2 = self.encoder_mini_block(cblock1[0], n_filters * 2, dropout_prob=0, max_pooling=True)
        cblock3 = self.encoder_mini_block(cblock2[0], n_filters * 4, dropout_prob=0, max_pooling=True)
        cblock4 = self.encoder_mini_block(cblock3[0], n_filters * 8, dropout_prob=0.3, max_pooling=True)
        cblock5 = self.encoder_mini_block(cblock4[0], n_filters * 16, dropout_prob=0.3, max_pooling=False)

        ublock6 = self.decoder_mini_block(cblock5[0], cblock4[1], n_filters * 8)
        ublock7 = self.decoder_mini_block(ublock6, cblock3[1], n_filters * 4)
        ublock8 = self.decoder_mini_block(ublock7, cblock2[1], n_filters * 2)
        ublock9 = self.decoder_mini_block(ublock8, cblock1[1], n_filters)

        conv9 = tf.keras.layers.Conv2D(n_filters,
                                       kernel_size=3,
                                       activation='relu',
                                       padding='same',
                                       kernel_initializer='he_normal')(ublock9)
        conv10 = tf.keras.layers.Conv2D(1, 1, padding='same')(conv9)

        self.model = tf.keras.Model(inputs=inputs, outputs=conv10, name=self.name)

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
        lr = ExponentialDecay(
            initial_learning_rate=learning_rate,  # was 0.0005 needs to be moved as hyperparameter
            decay_steps=len(train_data),
            decay_rate=0.85,
            staircase=True
        )
        metric = [tf.keras.metrics.BinaryAccuracy(), f1_m]

        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                           loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=metric)
        # seed not working as intended
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)

        history = self.model.fit(x=train_data, batch_size=batch_size, epochs=epochs, validation_data=val_data)

        return history

    def evaluate(self, data, labels, batch_size=16):
        """
        Evaluate the model with the data provided and corresponding labels.
        @return: metrics of the evaluation
        """

        return self.model.evaluate(data, labels, batch_size=batch_size)

    def predict(self, img, label=None, graphs=False):
        """
        Predicts an output given an image.
        @param img: Correctly sized image
        @param label: Label corresponding to the image. Used only for visualization purposes with training data.
        @param graphs: Whether to show an image of the image, label, prediction.
        @return: the predicted labels for the image.
        """
        pred = self.model.predict(img[np.newaxis, ...])
        if graphs:
            fig, arr = plt.subplots(2, 2, figsize=(15, 15))
            arr[0, 0].imshow(img)
            arr[0, 0].set_title('Processed Image')
            arr[1, 0].imshow(pred[0, :, :], cmap=plt.cm.gray)
            arr[1, 0].set_title('Predicted Masked Image ')
            if label is not None:
                arr[0, 1].imshow(label, cmap=plt.cm.gray)
                arr[0, 1].set_title('Actual Masked Image ')
            plt.show()
        return pred
