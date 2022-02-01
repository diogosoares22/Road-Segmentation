import os
import imageio
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt


def recreate_full_image_and_get_patches(predictions, test_size, test_img_size, og_patch_size, dest_patch_size, raw=False):
    """
    Recreates full images and get respective patches
    @param predictions: Predictions
    @param test_size: Test size
    @param test_img_size: Test Image size
    @param og_patch_size: Patch Size
    @param dest_patch_size: Destination Patch Size
    @param raw: if True returns the data without recreating patches.
    """
    pred_reshaped = np.reshape(predictions,
                                      (test_size, len(predictions) // test_size, og_patch_size, og_patch_size))
    to_substract = (np.ceil(test_img_size / og_patch_size) * og_patch_size - test_img_size) / np.floor(
        test_img_size / og_patch_size)

    if to_substract % 1 != 0:
        raise ValueError("patch size does not divide image equally")

    row = np.zeros((test_size, test_img_size, test_img_size))
    row[:] = np.nan
    coord = og_patch_size - int(to_substract)
    nb_patches_per_row = int(np.ceil(test_img_size / og_patch_size))
    for i in range(nb_patches_per_row):
        for j in range(nb_patches_per_row):
            row[:, j * coord:j * coord + og_patch_size,
                i * coord:i * coord + og_patch_size] = np.nanmean(
                    [pred_reshaped[:, nb_patches_per_row * i + j], row[:, j * coord:j * coord + (og_patch_size), i * coord:i * coord + (og_patch_size)]], axis=0)
    if raw:
        return row
    for i in range(40):
        plt.imshow(row[i])
        plt.show()
    labels = label_to_img(test_size, test_img_size, test_img_size, dest_patch_size, dest_patch_size, row,
                          pixelwise=True)

    return labels


def label_to_img(test_size, imgwidth, imgheight, w, h, labels, threshold=-3, pixelwise=False):
    """
    Convert array of labels to an image
    @param test_size: Test size
    @param imgwidth: Image Width
    @param imgheight: Image Height
    @param w: Width
    @param h: Height
    @param labels: Respective labels
    @param threshold: threshold to convert the mean of a patch to 1 or 0.
    @param pixelwise: Whether the model predicted in pixelwise.
    """
    if pixelwise:
        array_labels = np.zeros([test_size, imgwidth, imgheight])
    else:
        array_labels = np.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if pixelwise:
                array_labels[:, j:j + w, i:i + h] = np.where(np.mean(labels[:, j:j + w, i:i + h], axis=(1, 2)) > threshold, 1,
                                                             0)[:, np.newaxis, np.newaxis]
            else:
                if labels[idx][0] > 0.5:
                    l = 1
                else:  # bgrd
                    l = 0
                array_labels[j:j + w, i:i + h] = l
                idx = idx + 1
    return array_labels


def save_images(directory, images, original_names, prefix):
    """
    Saves images
    @param directory: Directory
    @param images: Images
    @param original_names: Original file name
    @param prefix: Prefix
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)
    for i, img in enumerate(images):
        imageio.imwrite(directory + prefix + original_names[i], img)


def make_img_overlay(path, img_name, predicted_img, pixel_depth=255):
    """
    Makes prediction overlay with picture
    @param path: Image path
    @param img_name: Image Name
    @param predicted_img: Predicted Image
    @param pixel_depth: Respective pixel depth
    """
    img = cv2.imread(path + img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img).astype(np.uint8)
    w, h = img.shape[0], img.shape[1]
    color_mask = np.zeros((w, h, 3), dtype=np.uint8)
    color_mask[:, :, 0] = predicted_img * pixel_depth

    background = Image.fromarray(img, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img


def save_history_to_csv(history, directory, filename):
    """
    Saves training history to file.
    @param history: History object returned by the training.
    @param directory: path to directory to which the file should be saved.
    @param filename: name of the file without extension.
    @return:
    """
    history_df = pd.DataFrame(history.history)
    os.makedirs(directory, exist_ok=True)
    hist_file = os.path.join(directory, filename + ".json")
    with open(hist_file, mode='w') as f:
        history_df.to_json(f)
        f.close()
