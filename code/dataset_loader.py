import os
from preprocessing import extract_data, extract_labels, balance_data
from tensorflow.keras.utils import Sequence
import numpy as np


class DatasetLoader(Sequence):

    def __init__(self, batch_size, patch_size, image_size, data_dir, images_list, categorical=None, labels_dir=None):
        """
        Instantiates the dataset loader.
        @param batch_size: Batch size. Assumes batch_size is a valid input, no checks are performed here.
        @param patch_size: Size of patches. Assumes patch_size is a valid input, no checks are performed here.
        @param image_size: Height of the images (square images)
        @param data_dir: Directory of images to load
        @param images_list: List of names of images to load from the data_dir (and corresponding labels from labels_dir if any)
        @param categorical: True if labels are categorical, False is labels are pixelwise, None if no labels are provided
        @param labels_dir: Directory of labels to load, None if no labels are provided
        """

        assert os.path.isdir(data_dir), "Data path provided does not exist!"
        assert (categorical is None and labels_dir is None) or (categorical is not None and labels_dir is not None), \
            "Either provide both parameters 'categorical' and 'labels_dir', or none of them!"
        if categorical is not None:
            assert isinstance(categorical, bool), "'categorical' must be of boolean type or None!"
            assert os.path.isdir(labels_dir), "Labels path provided does not exist!"

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.image_size = image_size
        self.patches_per_image = int(np.ceil(self.image_size / self.patch_size) ** 2)
        self.categorical = categorical
        self.data_dir = data_dir
        self.labels_dir = labels_dir
        self.images_list = images_list

    def __len__(self):
        """
        Number of batches in the sequence
        """
        return int(np.ceil(len(self.images_list) * self.patches_per_image / self.batch_size))

    def __geti(self, index):
        """
        Get current image index
        @param index: batch index
        """
        return index * self.batch_size / self.patches_per_image

    def __getitem__(self, index):
        """
        Returns a batch of data
        @param index: batch index
        """
        selected_images = self.images_list[int(np.floor(self.__geti(index))):int(np.ceil(self.__geti(index + 1)))]
        batch_data = extract_data(self.data_dir, selected_images, self.patch_size, self.image_size)
        start_index = int(((self.__geti(index)) % 1) * self.patches_per_image)
        end_index = int(len(batch_data) - ((self.__geti(index + 1)) % 1) * self.patches_per_image)
        batch_data = batch_data[start_index:end_index]
        if self.labels_dir is not None:
            batch_labels = extract_labels(self.labels_dir, selected_images, self.patch_size,
                                          pixelwise=(not self.categorical), image_size=self.image_size)
            batch_labels = batch_labels[start_index:end_index]
            if self.categorical:
                batch_data, batch_labels = balance_data(batch_data, batch_labels)
            return batch_data, batch_labels
        else:
            return batch_data
