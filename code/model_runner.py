"""
This file contains the class for running all our models which ensure a consistent pattern in our code.
"""

# Python libraries
import numpy as np
from os.path import join
from os import listdir
from random import shuffle

# Our models
from code.models.baseline_cnn import BaselineCNN
from code.models.drn import DRN
from code.models.unet import Unet
from code.models.transunet import TransUNet

# Data loader
from dataset_loader import DatasetLoader

# Helper functions  # See later how much of that we still need + only import used functions, not all of them
from mask_to_submission import masks_to_submission
from preprocessing import create_patch_images
from visualize_results import label_to_img, recreate_full_image_and_get_patches, save_images
from post_processing import run_post_processing_for_file, connectivity_post_processing
from helper_functions import normalize_array


class ModelRunner:
    """
    Model runner takes all the directories as initialisation attributes, and provides a framework to tune,
    train and predict with any model we pass to it.
    """
    NUM_CHANNELS = 3
    NUM_LABELS = 1
    SEED = 66478
    IMG_SIZE = 400
    TEST_IMG_SIZE = 608

    def __init__(self):
        self.dataset_dir = "../data"  # relative path to dataset directory
        self.training_images_dir = join(self.dataset_dir, "training/images/")  # folder of training images
        self.training_labels_dir = join(self.dataset_dir, "training/groundtruth/")  # folder of ground truth images
        self.testing_data_dir = join(self.dataset_dir, "test_set_images/")  # folder of testing images
        self.predictions_dir = "../predictions/"  # relative path to predictions directory

    def __k_fold_cross_validation(self, k, params, model_name, init_params):
        """
        Performs k-fold cross-validation for given parameters on indicated model.
        @param k: Number of folds.
        @param params: Training parameters
        @param model_name: Name of the model to tune.
        @param init_params: Initialisation parameters for the model to tune.
        @return:
        """
        val_split = 1 / k
        val_losses = []
        val_f1_scores = []

        # Get image names
        data = listdir(self.training_images_dir)
        shuffle(data)

        for i in range(k):
            print(f"split {i}")

            # Initialise the right model
            if model_name == "BaselineCNN":
                model = BaselineCNN(init_params)
            elif model_name == "DRN":
                model = DRN(init_params)
            elif model_name == "Unet":
                model = Unet(init_params)
            else:
                model = TransUNet(init_params)

            # Get data split
            split = int(len(data) * val_split)
            train_data = data[:i * split] + data[(i + 1) * split:]
            val_data = data[i * split:(i + 1) * split]

            # Setup our data iterators
            train_data = DatasetLoader(batch_size=params["batch_size"], patch_size=model.patch_size,
                                       image_size=self.IMG_SIZE, data_dir=self.training_images_dir,
                                       images_list=train_data, categorical=model.is_categorical,
                                       labels_dir=self.training_labels_dir)
            val_data = DatasetLoader(batch_size=params["batch_size"], patch_size=model.patch_size,
                                     image_size=self.IMG_SIZE, data_dir=self.training_images_dir,
                                     images_list=val_data, categorical=model.is_categorical,
                                     labels_dir=self.training_labels_dir)

            # Train model with specified parameters
            history = model.train(train_data, batch_size=params["batch_size"], learning_rate=params["lr"],
                                  epochs=params["epochs"], val_data=val_data)

            # Save loss and f1_score
            val_losses.append(history.history["val_loss"][-1])
            val_f1_scores.append(history.history["f1_m"][-1])

        return sum(val_losses) / len(val_losses), sum(val_f1_scores) / len(val_f1_scores)

    def tune_hyperparameters(self, params, fixed, placeholders, k, model_name, init_params, batch_size=(4, 8, 16)):
        """
        Tune hyperparameters with k-fold cross-validation for given model. Possible values for the respective parameters
        are set inside this function, such that we tune all models with the exact same possible values, i.e. we explore
        the same space for all models we tune here. The only parameter for which we test different values depending
        on the model is the batch_size, due to the fact that some models predict on patches, while others perform
        pixel-wise segmentation, and therefore need different orders of magnitudes of batch sizes.
        If several parameters are passed for tuning, the parameters will be tuned one by one due to timing constraints.
        @param params: List of names of parameters to tune, in the order they should be tuned.
        @param fixed: Dictionary of values fixed for parameters which should not be tuned. Must contain an entry for
                      each parameter that can be tuned in theory but which you do not plan on tuning.
                      E.g. if you won't tune the learning rate, pass {'lr': 0.01} as fixed value for the learning rate.
                      Pass an empty dictionary if you don't have any.
        @param placeholders: Dictionary of placeholder values for parameters which have not been tuned yet. Must contain
                             an entry for each parameter tuned except the first one.
                             E.g. If you plan on tuning ['lr', 'epochs'], pass {'epochs': 20} as placeholder value for
                             the epochs number while tuning the learning rate.
                             Pass an empty dictionary if you don't have any.
        @param k: k value for k-fold cross-validation.
        @param model_name: Name of the model for which to tune hyperparameters.
        @param init_params: Parameters for the model initialisation.
        @param batch_size: Values of batch sizes to test. Pass an empty iterable if you don't want to tune this parameter.
        """

        tuning_values = {
            "lr": np.logspace(-1, -4, 10),  # all 4 models
            "patch_size": [256, 320, 400],  # unet + transunet
            "batch_size": batch_size,  # 4 models
            "epochs": [20, 25, 30],  # 4 models
        }

        # Check parameters passed
        list_of_possible_parameters = set(tuning_values.keys())
        assert all(
            p in list_of_possible_parameters for p in params), "Your list of parameters contains invalid entries!"
        assert all(p in list_of_possible_parameters for p in
                   fixed.keys()), "Your list of fixed parameters contains invalid entries!"
        assert all(p in list_of_possible_parameters for p in
                   placeholders.keys()), "Your list of placeholder parameters contains invalid entries!"
        assert set(params).union(
            fixed.keys()) == list_of_possible_parameters, "Your list of parameters and/or list of fixed entries are missing entries!"
        assert set(params).intersection(
            fixed.keys()) == set(), "You have specified the same parameter(s) inside 'params' and 'fixed'!"
        assert set(params[1:]) == set(
            placeholders.keys()), "Your placeholder parameters do not match the parameters you want to tune!"
        assert model_name in {"BaselineCNN", "Unet", "DRN", "TransUNet"}, "Invalid model name!"

        # Initialize dictionary for storing the best parameter values obtained
        best_parameters = dict()

        for i, p in enumerate(params):

            tune_vals = tuning_values[p]
            f1_scores = np.zeros(len(tune_vals))

            for j, val in enumerate(tune_vals):
                print(f"testing parameter {p}: {val}")
                init_params_tune = init_params
                if p in init_params.keys():
                    init_params_tune[p] = val

                fold_params = {  # set value for parameter p for this run
                    p: val,
                }
                fold_params.update(fixed)  # Add fixed values
                fold_params.update(best_parameters)  # Add previously tuned values
                untuned_params = params[i + 1:]  # Add placeholders for remaining untrained parameters
                for u in untuned_params:
                    fold_params[u] = placeholders[u]

                _, validation_f1_score = self.__k_fold_cross_validation(k, fold_params, model_name, init_params_tune)
                f1_scores[j] = validation_f1_score
                print(f"f1 score for {val}: {validation_f1_score}")
            print(f1_scores)
            # Identify and store the best parameter
            best_parameters[p] = tune_vals[f1_scores.argmax()]
            print(best_parameters[p])

        return best_parameters

    def train(self, params, model, val_split=0.0, shuffle_data=True, extended_dataset=True, split=1, rotations=False):
        """
        Train and save given model
        @param params: Dictionary of parameters for the training.
        @param model: Initialised model.
        @param val_split: Float in [0, 1) indicating the validation split.
        @param shuffle_data: Boolean, shuffle data before splitting if True.
        @param extended_dataset: True if you want to use the entire dataset.
        @param split: Split of the dataset to be used for training and validation. Float in (0, 1].
        @param rotations: True if you want to use original dataset + 90 degree rotations only
        """

        assert (val_split >= 0) and (val_split < 1), "'val_split' is a value between 0 and 1!"
        assert isinstance(shuffle_data, bool), "'shuffle_data' is a boolean value!"
        assert set(params.keys()) == {"batch_size", "epochs",
                                      "lr"}, "Your parameters are not complete or you pass invalid parameters!"

        if extended_dataset:  # Train with entire dataset
            data = listdir(self.training_images_dir)
        elif rotations:  # Initial dataset + 90 degree rotations
            data = ["satImage_%.3d" % i for i in range(1, 101)]
            data += [name + "_" + str(i) for i in range(90, 371, 90) for name in data]
            data = [name + ".png" for name in data]
        else:  # Train with initial dataset
            data = ["satImage_%.3d.png" % i for i in range(1, 101)]
        shuffle(data)
        data = data[:int(len(data)*split)]

        # Split images into training and validation sets
        split = int(len(data) * val_split)
        train_data = data[split:]
        val_data = data[:split]

        # Setup our data iterators
        train_data = DatasetLoader(batch_size=params["batch_size"], patch_size=model.patch_size,
                                   image_size=self.IMG_SIZE, data_dir=self.training_images_dir,
                                   images_list=train_data, categorical=model.is_categorical,
                                   labels_dir=self.training_labels_dir)
        if len(val_data) > 0:
            val_data = DatasetLoader(batch_size=params["batch_size"], patch_size=model.patch_size,
                                     image_size=self.IMG_SIZE, data_dir=self.training_images_dir,
                                     images_list=val_data, categorical=model.is_categorical,
                                     labels_dir=self.training_labels_dir)
        else:
            val_data = None  # If no validation data

        # Train model with given parameters
        history = model.train(train_data, batch_size=params["batch_size"], learning_rate=params["lr"],
                              epochs=params["epochs"], val_data=val_data)
        model.save_to_file("../models/{}".format(model.name))

        return model, history

    def predict(self, model, post_processing=False):
        """
        Predict with given model
        @param: model: Initialised model.
        @param: post_processing: Application of post processing
        """

        test_filenames = listdir(self.testing_data_dir)

        test_size = len(test_filenames)

        # Load testing set (DatasetLoader or not?)

        images = create_patch_images(self.testing_data_dir, test_filenames, model.patch_size)

        # Predict with given model
        if model.name in ("DRN", "Baseline_CNN_provided"):
            predictions = [model.predict(img) for img in images]
            prediction_images = [
                label_to_img(test_size, self.TEST_IMG_SIZE, self.TEST_IMG_SIZE, model.patch_size, model.patch_size,
                             prediction) for prediction in predictions]
        else:
            predictions = [model.predict(patch) for img in images for patch in img]
            prediction_images = recreate_full_image_and_get_patches(np.asarray(predictions), test_size,
                                                                    self.TEST_IMG_SIZE, model.patch_size,
                                                                    16)

        # Save predictions to predictions folder

        predictions_path = join(self.predictions_dir, model.name + "/")

        save_images(predictions_path, prediction_images, test_filenames, "prediction_")

        predictions_files = [predictions_path + "prediction_test_{}.png".format(i) for i in range(1, test_size + 1)]

        if post_processing:
            print("Applying post process")
            for pred_file in predictions_files:
                run_post_processing_for_file(pred_file,
                                             connectivity_post_processing, (0, 4))

        masks_to_submission(predictions_path + "/predictions.csv", *predictions_files)

        return model

    def predict_ensemble(self, model_1, model_2, post_processing=False):
        """
        Predict ensemble of two given models
        @param: model1: Initialised model.
        @param: model2: Initialised model.      
        @param: post_processing: Application of post processing
        """
        test_filenames = listdir(self.testing_data_dir)

        test_size = len(test_filenames)

        # Load testing set (DatasetLoader or not?)

        images_1 = create_patch_images(self.testing_data_dir, test_filenames, model_1.patch_size)
        images_2 = create_patch_images(self.testing_data_dir, test_filenames, model_2.patch_size)

        # Predict with given model
        predictions_1 = [model_1.predict(patch) for img in images_1 for patch in img]
        predictions_2 = [model_2.predict(patch) for img in images_2 for patch in img]

        prediction_images_1 = recreate_full_image_and_get_patches(np.asarray(predictions_1), test_size,
                                                                  self.TEST_IMG_SIZE, model_1.patch_size,
                                                                  16, raw=True)
        prediction_images_2 = recreate_full_image_and_get_patches(np.asarray(predictions_2), test_size,
                                                                  self.TEST_IMG_SIZE, model_2.patch_size,
                                                                  16, raw=True)

        prediction_images_1 = normalize_array(prediction_images_1)

        prediction_images_2 = normalize_array(prediction_images_2)

        prediction_images_raw = (prediction_images_1 + prediction_images_2) / 2

        prediction_images = label_to_img(test_size, self.TEST_IMG_SIZE, self.TEST_IMG_SIZE, 16, 16,
                                         prediction_images_raw, threshold=0.618,
                                         pixelwise=True)

        # Save predictions to predictions folder

        predictions_path = join(self.predictions_dir, "ensemble" + "/")

        save_images(predictions_path, prediction_images, test_filenames, "prediction_")

        predictions_files = [predictions_path + "prediction_test_{}.png".format(i) for i in range(1, test_size + 1)]

        if post_processing:
            print("Applying post process")
            for pred_file in predictions_files:
                run_post_processing_for_file(pred_file,
                                             connectivity_post_processing, (0, 4))

        masks_to_submission(predictions_path + "/predictions.csv", *predictions_files)
