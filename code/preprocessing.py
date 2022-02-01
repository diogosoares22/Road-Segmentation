import os
import cv2
import numpy as np
import random


def img_crop(im, w, h):
    """ 
    Extract patches from a given image 
    @param im: Respective image
    @param w: Width
    @param h: Height
    """
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    to_substract_h = (np.ceil(imgheight / h) * h - imgheight) / np.floor(imgheight / h)
    to_substract_w = (np.ceil(imgwidth / w) * w - imgwidth) / np.floor(imgwidth / w)

    if to_substract_h % 1 != 0 or to_substract_w % 1 != 0:
        raise ValueError("patch size does not divide image equally")
    for i in range(0, imgheight - int(to_substract_h), h - int(to_substract_h)):
        for j in range(0, imgwidth - int(to_substract_w), w - int(to_substract_w)):
            if is_2d:
                im_patch = im[j:j + w, i:i + h]
            else:
                im_patch = im[j:j + w, i:i + h, :]
            list_patches.append(im_patch)
    return list_patches


def load_images(directory, filenames, isLabel=False, image_size=0):
    """
    Loads images from a specific directory with filenames
    @param directory: File directory
    @param filenames: Filenames
    @param isLabel: if it's to load label or not
    @param image_size: Original image size, used to resize rotated image to the correct shape
    """
    imgs = []
    for img_file in filenames:
        image_filename = directory + img_file
        if os.path.isfile(image_filename):
            img = cv2.imread(image_filename)
            if not isLabel:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if image_size > 0:
                img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
            imgs.append(img / 255)
        else:
            print('File ' + image_filename + ' does not exist')
    return imgs


def create_patch_images(directory, filenames, patch_size, image_size=0):
    """
    Extract the images into a 4D tensor [image index, y, x, channels].
    Values are rescaled from [0, 255] down to [-0.5, 0.5].
    @param directory: File directory
    @param filenames: Filenames
    @param patch_size: Patch Size
    @param image_size: Original image size, used to resize rotated image to the correct shape
    """
    imgs = load_images(directory, filenames, image_size=image_size)

    num_images = len(imgs)
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(num_images)]

    return np.asarray(img_patches)


def extract_data(directory, filenames, patch_size, image_size=0):
    """ 
    Extract data from directory, with filenames
    @param directory: File directory
    @param filenames: Filenames
    @param patch_size: Patch Size
    @param image_size: Original image size, used to resize rotated image to the correct shape
    """
    img_patches = create_patch_images(directory, filenames, patch_size, image_size)
    data = [img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))]

    return np.asarray(data, dtype=np.float64)


def value_to_class(v):
    """
    Assign a label to a patch.
    @param v: Patch Matrix
    """
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = np.sum(v)
    if df > foreground_threshold:  # road
        return 1
    else:  # bgrd
        return 0


def extract_labels(directory, filenames, patch_size, pixelwise=False, image_size=0):
    """
    Extract the labels into a 1-hot matrix [image index, label index].
    @param directory: File directory
    @param filenames: Filenames
    @param patch_size: Patch Size
    @param pixelwise: If it's pixelwise or not
    @param image_size: Original image size, used to resize rotated image to the correct shape
    """
    gt_imgs = load_images(directory, filenames, isLabel=True, image_size=image_size)

    num_images = len(gt_imgs)
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(num_images)]
    data = np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    if pixelwise:
        labels = np.where(data[:, :, :, 0] > 0.25, 1, 0)
    else:
        labels = np.where(np.mean(data, axis=(1, 2, 3)) > 0.25, 1, 0)
    # Convert to dense 1-hot representation.
    return labels.astype(np.float32)


def extract_random_patches_data_and_labels(image_directory, image_filenames, label_directory, label_filenames,
                                           patch_size, random_patches_per_image):
    """
    Extract random data from directory, labels into a 1-hot matrix [image index, label index].
    @param image_directory: Image directory
    @param image_filenames: Image Filenames
    @param label_directory: Label directory
    @param label_filenames: Label Filenames
    @param patch_size: Patch Size
    @param random_patches_per_image: Number of patches per image
    """
    images = load_images(image_directory, image_filenames)
    labels = load_images(label_directory, label_filenames)

    imgwidth = images[0].shape[0]
    imgheight = images[0].shape[1]

    img_patches = []
    gt_patches = []

    for t in range(len(images)):

        current_patches = {(i, j) for i in range(0, imgheight, patch_size) for j in range(0, imgwidth, patch_size)}

        counter = 0

        while random_patches_per_image > counter:

            i = random.randrange(0, imgwidth - patch_size - 1)
            j = random.randrange(0, imgwidth - patch_size - 1)

            if (i, j) not in current_patches:
                current_patches.add((i, j))

                img_patches.append(images[t][i: i + patch_size, j: j + patch_size, :])
                gt_patches.append(labels[t][i: i + patch_size, j: j + patch_size])

                counter += 1

    labels = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])

    return np.asarray(img_patches), labels.astype(np.float32)


def balance_data(data, labels):
    """
    Balance data to have the same frequency
    @param data: Image data
    @param labels: Image labels
    """

    c1 = np.count_nonzero(labels)
    c0 = len(labels) - c1

    min_c = min(c0, c1)
    idx0 = [i for i, j in enumerate(labels) if j == 0]
    idx1 = [i for i, j in enumerate(labels) if j == 1]
    new_indices = idx0[0:min_c] + idx1[0:min_c]
    random.shuffle(new_indices)
    new_data = data[new_indices, :, :, :]
    new_labels = labels[new_indices]

    return new_data, new_labels
