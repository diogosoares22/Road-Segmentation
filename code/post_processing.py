import cv2
import numpy as np
import imageio
import skimage.util as ski


def connectivity_post_processing(image_prediction, min_connectivity_for_pos, max_connectivity_for_neg):
    """ 
    Function flips positive nodes with less connectivity than min_connectivity, 
    and flips negative nodes with more connectivity than max_connectivity 
    @param image_prediction: Respective image prediction
    @param min_connectivity_for_pos: Minimum connectivity for positive patches
    @param max_connectivity_for_neg: Maximum connectivity for negative patches
    """
    image_correction = image_prediction.copy()
    shape = image_prediction.shape
    flag = True
    while (flag):
        flag = False
        for i in range(shape[0]):
            for j in range(shape[1]):
                connectivity = get_connectivity(image_correction, i, j)
                if image_correction[i, j] == 1 and connectivity <= min_connectivity_for_pos:
                    flag = True
                    image_correction[i, j] = 0
                if image_correction[i, j] == 0 and connectivity >= max_connectivity_for_neg:
                    flag = True
                    image_correction[i, j] = 1
    return image_correction


def get_connectivity(image_prediction, i, j):
    """ 
    Return the connectivity for a certain coordinate 
    @param image_prediction: Respective image prediction
    @param i : ith row
    @param j : jth column 
    """
    initial_pos_i = 0
    initial_pos_j = 0
    final_pos_i = 0
    final_pos_j = 0
    if (i - 1) >= 0:
        initial_pos_i = image_prediction[i - 1, j]
    if (j - 1) >= 0:
        initial_pos_j = image_prediction[i, j - 1]
    if (j + 1) <= image_prediction.shape[1] - 1:
        final_pos_j = image_prediction[i, j + 1]
    if (i + 1) <= image_prediction.shape[0] - 1:
        final_pos_i = image_prediction[i + 1, j]
    return initial_pos_j + initial_pos_i + final_pos_i + final_pos_j


def run_post_processing_for_file(image_filename, post_processing_func, args, patch_size=16):
    """ 
    Function that runs post processing direclty into a file and saves the result
    @param image_filename: Respective image filename
    @param post_processing_func: Function to apply in processing
    @param args: Function arguments
    @param patch_size: Patch size to apply function
    """
    image = cv2.imread(image_filename, 0) / 255

    image = ski.view_as_blocks(image, (patch_size, patch_size)).mean(axis=(2, 3))

    result = post_processing_func(image, *args)

    result = np.repeat(np.repeat(result, 16, axis=1), 16, axis=0)

    imageio.imwrite(image_filename, result)
